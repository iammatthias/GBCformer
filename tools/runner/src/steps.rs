//! The steps stream: one pass per step of the machine, more when a step's flags ask for
//! them (`wiring::Steps`). It owns the memory, the engine's KV cache: write records stay
//! in it, and copy lines are placed by a pass of their own.

use crate::ports::{Access, Ports};
use crate::slots::{At, Bit, Positions, Seq};
use gbcformer::memory::CopyFrom;
use gbcformer::llama::MemoryDenoiser;
use gbcformer::transport::{self, within, Battery};
use gbcformer::wiring::{Part, Snapshot, Steps};

/// One input built from parts.
enum Built {
    Carry { from: usize, mask: u8 },
    /// The spans stream's state input at this position.
    State { at: usize },
    Busy { bit: u8 },
    /// The spans stream's last pass's output at this position.
    Emitted { at: usize },
    Locks { present: u8, mask: u8, shift: u8 },
}

/// A step's write: its value, where the model placed it, and the second port it reaches.
#[derive(Clone, Copy, Debug)]
pub struct Write {
    /// Which of the step's writes it is.
    pub slot: usize,
    pub value: u8,
    pub at: u32,
    pub second: Option<u32>,
}

/// What a step asks of the other streams once its passes are done.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub cycles: u8,
    pub writes: Vec<Write>,
    /// Each enabled read's physical address.
    pub reads: Vec<u32>,
    pub ack: u8,
    pub stop: bool,
    /// The cycle of each idle bus event (`wiring::Events::idle`).
    pub idle: Vec<u8>,
}

/// A step's request: the reads the host serves and their ports, whether it asks for the
/// locks, the reads it makes, and its accesses' cycles (`wiring::Steps::access_at`, then
/// `fetch_cycles`).
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub served: [bool; 2],
    pub wants_locks: bool,
    pub ports: [Option<crate::ports::Port>; 2],
    pub enabled: [bool; 2],
    pub at: [u8; transport::SLOTS],
    /// Each access's mask: the bus state's bits the model reads at its cycle.
    pub masks: [u8; transport::SLOTS],
    /// The bus events the step pass flags (`wiring::Events`): each fetch's, read's and write's.
    pub fetch_events: [bool; 3],
    pub read_events: [bool; 2],
    pub write_events: [bool; 2],
}

/// What the steps stream needs from the spans stream for its inputs.
pub struct Signals<'a> {
    /// The spans stream's state bytes, by its input position.
    pub state: &'a [u8],
    /// The spans stream's last pass's output bytes, by its position.
    pub emitted: &'a [u8],
    pub busy: bool,
    pub locks: Option<u8>,
}

pub struct StepStream {
    pub seq: Seq,
    w: Steps,
    carry: Vec<(usize, usize)>,
    composed: Vec<(usize, Vec<Built>)>,
    /// The carried bytes, by input position: the last step's outputs.
    carried: Vec<u8>,
    step: Positions,
    result: Positions,
    /// The step pass's slots the result pass leaves out, which it reads decoded.
    request: Vec<usize>,
    fetch: Positions,
    line: Positions,
    second: Positions,
    late: Vec<usize>,
    served: Vec<Bit>,
    locks: Bit,
    fetched: Vec<Bit>,
    read_at: Vec<At>,
    fetch_at: Vec<At>,
    reads: Vec<Bit>,
    writes: Vec<Bit>,
    value: Vec<usize>,
    write_at: Vec<At>,
    records: Vec<Bit>,
    seconds: Vec<Bit>,
    second_at: Vec<At>,
    ack: At,
    cycles: At,
    halted: Bit,
    stop: Bit,
    windows: Vec<At>,
    line_in: Vec<usize>,
    line_at: Vec<At>,
    access_at: Vec<At>,
    access_locks: Vec<Bit>,
    bits: transport::StateBits,
    fetch_cycles: Vec<(Bit, u8)>,
    access_in: Vec<usize>,
    /// The bus events (`wiring::Events`), resolved.
    ev_fetches: Vec<Bit>,
    ev_reads: Vec<Bit>,
    ev_idle: Vec<Bit>,
    ev_idle_at: At,
    ev_writes: Vec<Bit>,
    ev_input: usize,
    ev_pass: Positions,
    ev_rows: Vec<(Bit, u8, At)>,
    /// The step's composed inputs and its accesses' bus states, set at each request.
    start_inputs: Vec<(usize, u8)>,
    /// The inputs each request sets: all but the copy line's and the late ones.
    cleared: Vec<usize>,
    /// The bytes a request's inputs take, by input position.
    next: Vec<u8>,
    /// The late inputs' bytes: none before the result pass, and the result pass's buffer.
    no_late: Vec<u8>,
    late_bytes: Vec<u8>,
    /// The step's write records not landed yet, and the sequence they read as it stood
    /// when a later pass of the step followed it (empty otherwise).
    staged: Vec<bool>,
    /// Each record's address, while staged.
    staged_at: Vec<u32>,
    staged_tokens: Vec<usize>,
    /// The sequence a record reads, built as its pass left it.
    finished: Vec<usize>,
    /// The last outcome's buffers.
    spare: Outcome,
    access: Vec<u8>,
    /// The last step's outputs that decide whether the next one runs.
    pub halted_now: bool,
    /// The battery save's state, from the model's windows and write addresses alone.
    battery: Battery,
    pub steps: u64,
    pub records_kept: u64,
    pub line_passes: u64,
    pub second_passes: u64,
    pub fetch_passes: u64,
    pub rewrite_passes: u64,
    pub windows_now: Vec<u32>,
}

impl StepStream {
    /// `state_at` names the spans stream's input position of a state slot, `emitted_at` its
    /// position of an output slot.
    pub fn new(seq: Seq, w: &Steps, snap: &Snapshot, state_at: &dyn Fn(&str) -> usize, emitted_at: &dyn Fn(&str) -> usize) -> StepStream {
        // The shared access loop's shape (`transport::Asked`): two reads, two writes.
        assert!(w.served.len() == 2 && w.reads.len() == 2 && w.read_at.len() == 2 && w.access_at.len() == transport::DATA && w.fetch_cycles.len() == transport::SLOTS - transport::DATA && w.access_in.len() * 2 == transport::SLOTS, "a step of two reads, two writes and the fetches after the first");
        let carry: Vec<(usize, usize)> = w.carry.iter().map(|(o, i)| (seq.output(o), seq.input(i))).collect();
        let composed = w
            .composed
            .iter()
            .map(|c| {
                let parts = c
                    .parts
                    .iter()
                    .map(|p| match p {
                        Part::Carry { from, mask } => Built::Carry { from: seq.output(from), mask: *mask },
                        Part::State { slot } => Built::State { at: state_at(slot) },
                        Part::Busy { bit } => Built::Busy { bit: *bit },
                        Part::Emitted { slot } => Built::Emitted { at: emitted_at(slot) },
                        Part::Locks { present, mask, shift } => Built::Locks { present: *present, mask: *mask, shift: *shift },
                    })
                    .collect();
                (seq.input(&c.slot), parts)
            })
            .collect();
        let mut carried = vec![0u8; seq.n_inputs()];
        for (name, b) in &snap.steps {
            carried[seq.input(name)] = *b;
        }
        let fields = |seq: &Seq, v: &[gbcformer::wiring::Field]| v.iter().map(|f| seq.field(f)).collect::<Vec<_>>();
        let flags = |seq: &Seq, v: &[gbcformer::wiring::Flag]| v.iter().map(|f| seq.flag(f)).collect::<Vec<_>>();
        let ev = &w.events;
        assert!(ev.fetches.len() == 3 && ev.reads.len() == 2 && ev.writes.len() == 2, "a step's bus events: three fetches, two reads, two writes");
        let (step, result) = (seq.outputs(&w.step), seq.outputs(&w.result));
        let late: Vec<usize> = w.late.iter().map(|n| seq.input(n)).collect();
        let line_in: Vec<usize> = w.line_in.iter().map(|n| seq.input(n)).collect();
        let s = StepStream {
            carry,
            composed,
            carried,
            request: step.iter().copied().filter(|p| !result.contains(p)).collect(),
            step: Positions::new(step),
            result: Positions::new(result),
            fetch: seq.pass(&w.fetch),
            line: seq.pass(&w.line),
            second: seq.pass(&w.second),
            served: flags(&seq, &w.served),
            locks: seq.flag(&w.locks),
            fetched: flags(&seq, &w.fetched),
            read_at: fields(&seq, &w.read_at),
            fetch_at: fields(&seq, &w.fetch_at),
            reads: flags(&seq, &w.reads),
            writes: flags(&seq, &w.writes),
            value: seq.outputs(&w.value),
            write_at: fields(&seq, &w.write_at),
            records: flags(&seq, &w.records),
            seconds: flags(&seq, &w.seconds),
            second_at: fields(&seq, &w.second_at),
            ack: seq.field(&w.ack),
            cycles: seq.field(&w.cycles),
            halted: seq.flag(&w.halted),
            stop: seq.flag(&w.stop),
            windows: fields(&seq, &w.windows),
            line_at: fields(&seq, &w.line_at),
            access_at: fields(&seq, &w.access_at),
            access_locks: w.access_locks.iter().map(|f| seq.flag(f)).collect(),
            bits: transport::StateBits { locks: w.locks_bits, busy: w.busy_bit, replace: w.replace, shift: w.access_shift, writes: w.write_slots, write_shift: w.write_shift },
            fetch_cycles: w.fetch_cycles.iter().map(|(f, c)| (seq.flag(f), *c)).collect(),
            access_in: w.access_in.iter().map(|n| seq.input(n)).collect(),
            ev_fetches: flags(&seq, &ev.fetches),
            ev_reads: flags(&seq, &ev.reads),
            ev_idle: flags(&seq, &ev.idle),
            ev_idle_at: seq.field(&ev.idle_at),
            ev_writes: flags(&seq, &ev.writes),
            ev_input: seq.input(&ev.input),
            ev_pass: seq.pass(&ev.pass),
            ev_rows: ev.rows.iter().map(|(f, r, a)| (seq.flag(f), *r, seq.field(a))).collect(),
            start_inputs: Vec::new(),
            cleared: (0..seq.n_inputs()).filter(|p| !line_in.contains(p) && !late.contains(p)).collect(),
            next: vec![0; seq.n_inputs()],
            no_late: vec![0; late.len()],
            late_bytes: Vec::new(),
            late,
            line_in,
            staged: vec![false; w.records.len()],
            staged_at: vec![0; w.records.len()],
            staged_tokens: Vec::new(),
            finished: Vec::new(),
            spare: Outcome::default(),
            access: vec![0; w.access_in.len()],
            halted_now: false,
            battery: Battery::seeded(snap.save_pending.0, snap.save_pending.1),
            steps: 0,
            records_kept: 0,
            line_passes: 0,
            second_passes: 0,
            fetch_passes: 0,
            rewrite_passes: 0,
            windows_now: Vec::new(),
            w: w.clone(),
            seq,
        };
        let mut s = s;
        // A record reads the step's decoded slots when it lands, and a request sets each input
        // once: no other pass decodes a slot of the step's, and no carried, composed or access
        // input is a late or copy-line one.
        assert!(s.fetch.iter().chain(s.line.iter()).chain(s.second.iter()).chain(s.ev_pass.iter()).all(|p| !s.step.contains(p)), "a pass outside the step's two decodes one of their slots");
        assert!(s.carry.iter().map(|c| c.1).chain(s.composed.iter().map(|c| c.0)).chain(s.access_in.iter().copied()).all(|p| s.cleared.contains(&p)), "a carried, composed or access input is a late or copy-line input");
        for p in s.line_in.clone() {
            s.seq.set_input(p, 0);
        }
        let halted_carry = s.composed.iter().flat_map(|(at, parts)| parts.iter().filter_map(move |p| if let Built::Carry { mask, .. } = p { Some((*at, *mask)) } else { None })).collect::<Vec<_>>();
        // Halted at the hand-off: the carried byte of the input the halted flag's output feeds.
        s.halted_now = halted_carry.iter().any(|&(at, mask)| s.carried[at] & mask & s.w.halted.mask != 0);
        s
    }

    /// The composed inputs from the carried bytes and the spans stream, read once at the
    /// step's start.
    fn composed_inputs(&self, sig: &Signals, out: &mut Vec<(usize, u8)>) {
        out.clear();
        out.extend(self.composed.iter().map(|(at, parts)| {
            let b = parts.iter().fold(0u8, |b, p| {
                b | match p {
                    Built::Carry { mask, .. } => self.carried[*at] & mask,
                    Built::State { at } => sig.state[*at],
                    Built::Busy { bit } => if sig.busy { *bit } else { 0 },
                    Built::Emitted { at } => sig.emitted[*at],
                    Built::Locks { present, mask, shift } => sig.locks.map_or(0, |l| present | (l & mask) << shift),
                }
            });
            (*at, b)
        }));
    }

    /// Every input: the carried bytes, `composed`, and the late ones masked or set from
    /// `late` (a 0 byte stays the mask). The copy line's inputs keep the last line set;
    /// the rest hold 0.
    fn set_inputs(&mut self, composed: &[(usize, u8)], late: &[u8]) {
        for k in 0..self.cleared.len() {
            let p = self.cleared[k];
            self.next[p] = 0;
        }
        for &(_, i) in &self.carry {
            let b = self.carried[i];
            self.next[i] = b;
        }
        for &(at, b) in composed {
            self.next[at] = b;
        }
        for k in 0..self.access_in.len() {
            let (p, b) = (self.access_in[k], self.access[k]);
            self.next[p] = b;
        }
        // Each input is set once, so the sequence lists it as changed only where it is.
        for k in 0..self.cleared.len() {
            let p = self.cleared[k];
            self.seq.set_input(p, self.next[p]);
        }
        for (k, &p) in self.late.iter().enumerate() {
            if late[k] == 0 {
                self.seq.mask_input(p);
            } else {
                self.seq.set_input(p, late[k]);
            }
        }
    }

    /// The buffers of an outcome the scheduler is done with, for the next step's.
    pub fn recycle(&mut self, out: Outcome) {
        self.spare = out;
    }

    /// Write `i`'s record lands, at that write's cycle: it replaces its line's cell.
    pub fn land_write(&mut self, i: usize) {
        if !std::mem::take(&mut self.staged[i]) {
            return;
        }
        if self.staged_tokens.is_empty() {
            self.seq.finished(&self.step, &mut self.finished);
        }
        let tokens = if self.staged_tokens.is_empty() { &self.finished } else { &self.staged_tokens };
        self.seq.engine.keep(tokens, &[i as u8]);
        self.records_kept += 1;
        if !self.staged.contains(&true) {
            self.staged_tokens.clear();
        }
    }

    /// A step begins: its composed inputs, read once at its start, and no access state.
    pub fn begin(&mut self, sig: &Signals) {
        assert!(!self.staged.contains(&true), "a step ran before the last one's write records landed");
        let mut inputs = std::mem::take(&mut self.start_inputs);
        self.composed_inputs(sig, &mut inputs);
        self.start_inputs = inputs;
        self.access.iter_mut().for_each(|b| *b = 0);
    }

    /// The step's request under `access` (its accesses' bus states, `wiring::Steps::access_in`):
    /// the step pass, and the fetch pass when it flags bytes the host serves, which `ports`
    /// reads from the stream the model's address names.
    pub fn request(&mut self, access: &[u8], ports: &mut dyn Access, map: &Ports) -> Request {
        self.access.copy_from_slice(access);
        let (composed, late) = (std::mem::take(&mut self.start_inputs), std::mem::take(&mut self.no_late));
        self.set_inputs(&composed, &late);
        (self.start_inputs, self.no_late) = (composed, late);
        self.seq.decode(&self.step);
        if self.fetched.iter().any(|&b| self.seq.is(b)) {
            self.fetch_passes += 1;
            let flagged: Vec<bool> = self.fetched.iter().map(|&b| self.seq.is(b)).collect();
            self.seq.decode(&self.fetch);
            let at: Vec<Option<u32>> = (0..flagged.len()).map(|k| flagged[k].then(|| self.seq.get(&self.fetch_at[k]) as u32)).collect();
            let mut kept = Vec::new();
            for &a in at.iter().flatten() {
                let byte = match map.port(a) {
                    Some(port) => ports.read(port),
                    None => self.seq.engine.cell(a),
                };
                kept.push((self.w.host_block | (a & self.w.host_mask), byte));
            }
            for (p, byte) in kept {
                self.seq.engine.keep_host(p, byte);
            }
            self.seq.decode(&self.step);
            let again: Vec<bool> = self.fetched.iter().map(|&b| self.seq.is(b)).collect();
            assert_eq!(again, flagged, "the fetched bytes' flags depend only on the step's registers");
        }
        // The request: a read the host serves names its port; a flag on an address that
        // names none stops the run.
        let served: [bool; 2] = std::array::from_fn(|i| self.seq.is(self.served[i]));
        let wants_locks = self.seq.is(self.locks);
        let ports_of = std::array::from_fn(|i| {
            served[i].then(|| {
                let a = self.seq.get(&self.read_at[i]) as u32;
                map.port(a).unwrap_or_else(|| panic!("read {i}: the model asks for {a:#08x}, which names no port"))
            })
        });
        let enabled = std::array::from_fn(|i| self.seq.is(self.reads[i]));
        let data = self.access_at.len();
        let at = std::array::from_fn(|s| if s < data { self.seq.get(&self.access_at[s]) as u8 } else { let (b, c) = self.fetch_cycles[s - data]; if self.seq.is(b) { c } else { 0 } });
        let masks = std::array::from_fn(|s| transport::access_mask(self.bits, self.seq.is(self.access_locks[s])));
        Request { served, wants_locks, ports: ports_of, enabled, at, masks , fetch_events: std::array::from_fn(|k| self.seq.is(self.ev_fetches[k])), read_events: std::array::from_fn(|i| self.seq.is(self.ev_reads[i])), write_events: std::array::from_fn(|i| self.seq.is(self.ev_writes[i])) }
    }

    /// The step's result after `req`: the locks `asked` for and each served read's byte in
    /// `data`, then the records, the writes and the carried state.
    pub fn finish(&mut self, req: &Request, asked: u8, data: &[u8]) -> Outcome {
        let (served, wants_locks, enabled) = (&req.served, req.wants_locks, &req.enabled);
        let settled = !served.iter().any(|&s| s) && !wants_locks;
        self.steps += 1;
        if !settled {
            let mut late = std::mem::take(&mut self.late_bytes);
            late.clear();
            late.extend((0..served.len()).map(|i| if served[i] { data[i] } else { 0 }));
            late.push(if wants_locks { asked & self.w.locks_mask } else { 0 });
            let composed = std::mem::take(&mut self.start_inputs);
            self.set_inputs(&composed, &late);
            self.start_inputs = composed;
            self.seq.show(&self.request);
            self.seq.decode(&self.result);
            self.seq.hide(&self.request);
            self.late_bytes = late;
        }
        // The write records wait for their writes' cycles (`land_write`), write 0's before
        // write 1's.
        for i in 0..self.records.len() {
            self.staged[i] = self.seq.is(self.records[i]);
            self.staged_at[i] = self.seq.get(&self.write_at[i]) as u32;
        }
        // The outcome is read before a second-port pass masks every output, into the buffers
        // the last step's outcome gave back (`recycle`).
        let mut out = std::mem::take(&mut self.spare);
        out.writes.clear();
        out.reads.clear();
        for i in 0..enabled.len() {
            if enabled[i] {
                out.reads.push(self.seq.get(&self.read_at[i]) as u32);
            }
        }
        (out.cycles, out.ack, out.stop) = (self.seq.get(&self.cycles) as u8, self.seq.get(&self.ack) as u8, self.seq.is(self.stop));
        out.idle.clear();
        for k in 0..self.ev_idle.len() {
            if self.seq.is(self.ev_idle[k]) {
                out.idle.push(self.seq.get(&self.ev_idle_at) as u8);
            }
        }
        self.halted_now = self.seq.is(self.halted);
        self.windows_now.clear();
        for k in 0..self.windows.len() {
            let v = self.seq.get(&self.windows[k]) as u32;
            self.windows_now.push(v);
        }
        for &(o, i) in &self.carry {
            self.carried[i] = self.seq.bytes[o];
        }
        for (at, parts) in &self.composed {
            for p in parts {
                if let Built::Carry { from, mask } = p {
                    self.carried[*at] = self.seq.bytes[*from] & mask;
                }
            }
        }
        for i in 0..self.writes.len() {
            if self.seq.is(self.writes[i]) {
                out.writes.push(Write { slot: i, value: self.seq.bytes[self.value[i]], at: self.seq.get(&self.write_at[i]) as u32, second: None });
            }
        }
        let flagged: u64 = (0..self.seconds.len()).filter(|&i| self.seq.is(self.seconds[i])).fold(0, |m, i| m | 1 << i);
        if flagged != 0 {
            // The records read the step's sequence as it stands before this pass.
            if self.staged.contains(&true) {
                self.seq.finished(&self.step, &mut self.staged_tokens);
            }
            self.seq.decode(&self.second);
            self.second_passes += 1;
            for w in out.writes.iter_mut().filter(|w| flagged >> w.slot & 1 != 0) {
                w.second = Some(self.seq.get(&self.second_at[w.slot]) as u32);
            }
        }
        let regions = &self.w.save_regions;
        let enabled = within(regions, self.windows_now[self.w.save_window]);
        self.battery.step(enabled, out.writes.iter().any(|w| within(regions, w.at)));
        out
    }

    /// A save due now: a finished one (`committed`), or anything written since the last.
    pub fn take_save(&mut self, committed: bool, len: usize) -> Option<Vec<u8>> {
        let due = if committed { self.battery.take_committed() } else { self.battery.take_unsaved() };
        let memory = self.seq.engine.memory.as_ref().expect("the steps stream holds the memory");
        due.then(|| {
            memory.cells.settle();
            memory.image.region(self.w.save_base, len).to_vec()
        })
    }

    /// Waits for the cells' rows a thread of their own still has to write. A check of them
    /// that failed there stops the run here.
    pub fn settle(&self) {
        self.seq.engine.memory.as_ref().expect("the steps stream holds the memory").cells.settle();
    }

    /// A copy line's two translated addresses: one pass over the line inputs alone.
    fn line_pass(&mut self, source: u16, dest: u16) -> [u32; 2] {
        let keep = (1u16 << self.w.line_bits) - 1;
        let lines = [source & !keep, dest & !keep];
        let bytes = [lines[0] as u8, (lines[0] >> 8) as u8, lines[1] as u8, (lines[1] >> 8) as u8];
        for (k, &p) in self.line_in.clone().iter().enumerate() {
            self.seq.set_input(p, bytes[k]);
        }
        // A copy within a step (the other streams run to an access) decodes none of the slots
        // the step's records read when they land.
        self.seq.decode(&self.line);
        let at = [self.seq.get(&self.line_at[0]) as u32, self.seq.get(&self.line_at[1]) as u32];
        self.line_passes += 1;
        at
    }

    /// A bus event's rewrite at `row` (`wiring::Events`): one pass over the event byte, its
    /// row with the read bit for a read, which keeps each record it flags. The step's
    /// decoded tokens, which its records read when they land, and its inputs stand as they
    /// were after it.
    pub fn rewrite(&mut self, row: u8, read: bool) {
        self.seq.set_input(self.ev_input, row | if read { self.w.events.read_bit } else { 0 });
        self.seq.decode(&self.ev_pass);
        let mut kept_rows = Vec::new();
        for k in 0..self.ev_rows.len() {
            let (flag, record, at) = (self.ev_rows[k].0, self.ev_rows[k].1, self.ev_rows[k].2.clone());
            if self.seq.is(flag) {
                let p = self.seq.get(&at) as u32;
                let staged = (0..self.staged.len()).filter(|&i| self.staged[i]).map(|i| self.staged_at[i]);
                assert!(!transport::staged_in_line(staged, self.w.line_bits, p), "a rewrite of {p:#08x}: a write record of the step not landed yet shares its line");
                kept_rows.push(record);
            }
        }
        self.seq.finished(&self.ev_pass, &mut self.finished);
        self.seq.engine.keep(&self.finished, &kept_rows);
        self.records_kept += kept_rows.len() as u64;
        self.seq.set_input(self.ev_input, 0);
        self.rewrite_passes += 1;
    }

    /// The bytes a batch of copies moved, in order, each placed by the copy line's pass
    /// and read from the model's own memory.
    pub fn copied(&mut self, bytes: &[(Option<u16>, u16)], map: &Ports) {
        let keep = (1u16 << self.w.line_bits) - 1;
        let mut placed: Option<([u16; 2], [u32; 2])> = None;
        for &(source, dest) in bytes {
            let from_bus = source.unwrap_or(dest);
            let lines = [from_bus & !keep, dest & !keep];
            let at = match placed {
                Some((l, at)) if l == lines => at,
                _ => {
                    let at = self.line_pass(from_bus, dest);
                    placed = Some((lines, at));
                    at
                }
            };
            let dest = at[1] | (dest & keep) as u32;
            let from = match source {
                None => self.w.fill_block | (dest & self.w.fill_mask),
                Some(s) => {
                    if let Some(p) = map.port(at[0]) {
                        panic!("a copy from {:#08x}, the port of {p:?}, whose bytes no line pass places; refused", at[0]);
                    }
                    at[0] | (s & keep) as u32
                }
            };
            let staged = (0..self.staged.len()).filter(|&i| self.staged[i]).map(|i| self.staged_at[i]);
            assert!(!transport::staged_in_line(staged.clone(), self.w.line_bits, dest) && !transport::staged_in_line(staged, self.w.line_bits, from), "a copy to {dest:#08x}: a write record of the step not landed yet shares its line");
            self.seq.engine.copy(CopyFrom::Memory(from), dest);
        }
    }

    /// Whether an access at `p` is one the busy copy blocked.
    pub fn blocked(&self, p: u32) -> bool {
        (self.w.blocked.0..self.w.blocked.1).contains(&p)
    }
}
