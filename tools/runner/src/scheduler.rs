//! The run: the five streams and the order in which their outputs reach each other's
//! inputs. The host counts cycles between passes, moves each port access to the stream
//! its address names, places copies by the bits the model emitted, and keeps real time by
//! the rate the spans stream emits. It computes no state of the machine itself.

use crate::length::LengthStream;
use crate::lines::LineStream;
use crate::rows::Rows;
use crate::ports::{Access, Port, Ports};
use crate::spans::{Event, SpanStream};
use crate::steps::{self, StepStream};
use crate::worker::Worker;
use gbcformer::engine::VideoRecords;
use gbcformer::transport::{self, Blocks, Copy, Passes, Signals, Sink};
use gbcformer::wiring::{Snapshot, Target, Wiring};

pub struct Run {
    pub steps: StepStream,
    pub spans: SpanStream,
    pub segments: crate::segments::Queued,
    pub lines: Worker<LineStream>,
    pub length: LengthStream,
    pub rows: Rows,
    /// The steps stream's records and the own-memory writes, in the order they were kept,
    /// until the lines stream takes them.
    records: VideoRecords,
    ports: Ports,
    w: Wiring,
    signals: Signals,
    /// Bytes copied since the last hand-over to the steps stream.
    copied: Vec<(Option<u16>, u16)>,
    stop_pending: bool,
    /// A step is running: the block copies its cycles ask for wait in `held` for its end,
    /// and `ticked` counts the cycles its accesses already ran.
    in_step: bool,
    ticked: u64,
    held: Vec<Copy>,
    /// Real time in half units, and where the frame ends: time starts as far past the
    /// snapshot's frame end as its last step ran.
    pub halves: u64,
    frame_end: u64,
    clock: Option<u64>,
    has_clock: bool,
    save_len: usize,
    pub counts: Counts,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub steps: u64,
    pub step_cycles: u64,
    pub idle_cycles: u64,
    pub stall_cycles: u64,
    pub pause_cycles: u64,
}

/// The streams the steps stream reads through ports, borrowed for one step.
struct Reach<'a> {
    spans: &'a mut SpanStream,
    segments: &'a mut crate::segments::Queued,
    lines: &'a mut Worker<LineStream>,
    length: &'a mut LengthStream,
    rows: &'a mut Rows,
    records: &'a VideoRecords,
    ports: &'a Ports,
    copied: &'a mut Vec<(Option<u16>, u16)>,
}

/// The bytes of copies a pass moved, for the steps stream.
fn land(copied: &mut Vec<(Option<u16>, u16)>, copies: &[Copy]) {
    for c in copies {
        copied.extend(c.bytes());
    }
}

impl Reach<'_> {
    /// The length stream's last length, which the spans stream now reads.
    fn length(length: &mut LengthStream) -> impl FnMut() -> u8 + '_ {
        move || {
            length.asked = true;
            length.length
        }
    }

    /// An own-memory byte: while a row is open, the newest record kept since its trigger,
    /// else the lines stream's memory after the records kept so far.
    fn own_read(&mut self, i: u8) -> u8 {
        if self.rows.is_open() {
            return match self.rows.held_record(self.records, i) {
                Some(b) => b,
                None => self.lines.call(move |l| l.own_read(i, Vec::new())),
            };
        }
        let records = take(self.records);
        self.length.catch_up(&records);
        self.rows.stepper.catch_up(&records);
        self.lines.call(move |l| l.own_read(i, records))
    }
}

fn take(records: &VideoRecords) -> Vec<(u32, u8)> {
    std::mem::take(&mut *records.lock().expect("the record queue"))
}

impl Access for Reach<'_> {
    fn read(&mut self, port: Port) -> u8 {
        match self.ports.target(port) {
            Target::Spans { read: Some((ev, address)), .. } => {
                let code = [*ev, address[port.low as usize]];
                let mut landed = Vec::new();
                let pass = self.spans.pass_alone(Event::of(code, 0), &mut landed, &mut Reach::length(self.length));
                land(self.copied, &landed);
                // An indirect read takes the own-memory byte the pass names, when it names one.
                match pass.own.filter(|_| self.ports.indirect(port)) {
                    Some(i) => self.own_read(i),
                    None => pass.read,
                }
            }
            Target::Spans { read: None, .. } => panic!("a read of the write-only port {port:?}"),
            Target::Segments { address } => {
                let a = address[port.low as usize];
                self.segments.read(a)
            }
            Target::Own { index } => {
                let i = *index;
                self.own_read(i)
            }
        }
    }

    fn lock_query(&mut self) -> u8 {
        let code = self.spans.wiring().lock_query;
        let mut landed = Vec::new();
        let locks = self.spans.pass_alone(Event::of(code, 0), &mut landed, &mut Reach::length(self.length)).locks[0].0;
        land(self.copied, &landed);
        locks
    }
}

impl Run {
    pub fn new(steps: StepStream, spans: SpanStream, segments: crate::segments::Queued, lines: Worker<LineStream>, length: LengthStream, rows: Rows, w: &Wiring, snap: &Snapshot, records: VideoRecords) -> Run {
        Run { steps, spans, segments, lines, length, rows, records, ports: Ports::new(&w.ports), w: w.clone(), signals: Signals::default(), copied: Vec::new(), stop_pending: false, in_step: false, ticked: 0, held: Vec::new(), halves: snap.overshoot, frame_end: 0, clock: snap.clock_reading, has_clock: snap.clock, save_len: snap.save_len, counts: Counts::default() }
    }

    /// An access the busy copy blocked: the spans stream ends its run here.
    fn cut(&mut self) {
        self.signals.reset(self.spans.fast());
        let Run { spans, length, signals, .. } = self;
        spans.cut(signals, &mut Reach::length(length));
        self.apply_signals();
    }

    /// An own-memory byte joins the record queue, after every record kept before it.
    fn own_write(&mut self, index: u8, value: u8) {
        self.records.lock().expect("the record queue").push((self.w.lines.own_memory + index as u32, value));
        self.rows.write(index, value, self.halves, &self.records, &mut self.length, &mut self.lines);
    }

    /// A write through a port, and what the spans stream asks for after it.
    fn write_port(&mut self, port: Port, value: u8) {
        self.signals.reset(self.spans.fast());
        match self.ports.target(port).clone() {
            Target::Spans { write: Some((ev, address)), .. } => {
                let code = [ev, address[port.low as usize]];
                let Run { spans, length, signals, .. } = self;
                spans.pass_with(Event::of(code, value), signals, &mut Reach::length(length), Some(value));
            }
            Target::Spans { write: None, .. } => panic!("a write to the read-only port {port:?}"),
            Target::Segments { address } => {
                let a = address[port.low as usize];
                self.segments.write(a, value);
            }
            Target::Own { index } => self.own_write(index, value),
        }
        self.apply_signals();
    }

    /// The block's signals, in the shared order (`transport::Signals::apply`).
    fn apply_signals(&mut self) {
        let Run { signals, copied, records, length, rows, lines, w, held, halves, spans, .. } = self;
        signals.apply(&mut Landing { copied, records, length, rows, lines, own_memory: w.lines.own_memory, held, halves: *halves, fast: spans.fast() });
        signals.copies.clear();
    }

    fn land_early(&mut self) {
        let Run { signals, copied, records, length, rows, lines, w, held, halves, spans, .. } = self;
        signals.land_early(&mut Landing { copied, records, length, rows, lines, own_memory: w.lines.own_memory, held, halves: *halves, fast: spans.fast() });
    }

    fn hand_over(&mut self) {
        if !self.copied.is_empty() {
            let bytes = std::mem::take(&mut self.copied);
            self.steps.copied(&bytes, &self.ports);
        }
    }

    /// `cycles` cycles of every stream, in blocks no longer than the spans stream's
    /// horizon, then a stop the step ran, then the cycles the passes say the steps stream
    /// stays stopped
    /// (`transport::advance`). With `hand`, each block's copies reach the steps stream
    /// before the next block runs.
    fn advance(&mut self, cycles: u64, hand: bool) {
        let (stall, pause) = transport::advance(&mut Blocking { run: self, hand }, cycles, false);
        self.counts.stall_cycles += stall;
        self.counts.pause_cycles += pause;
    }

    /// One step of the steps stream, or one cycle while it is held or halted.
    fn step(&mut self) {
        if self.spans.paused() {
            // Held: time runs on, and the copies wait for the next step's hand-over.
            self.counts.idle_cycles += 1;
            self.advance(1, false);
            return;
        }
        let was_halted = self.steps.halted_now;
        let ran = if self.steps.halted_now && !self.spans.wake {
            self.counts.idle_cycles += 1;
            1
        } else {
            let out = self.one_step();
            self.counts.steps += 1;
            self.counts.step_cycles += out as u64;
            out as u64
        };
        if was_halted && !self.steps.halted_now {
            self.woke();
        }
        if !was_halted && self.steps.halted_now {
            let Run { spans, length, copied, .. } = self;
            let mut landed = Vec::new();
            spans.rest(&mut landed, &mut Reach::length(length));
            land(copied, &landed);
        }
        self.hand_over();
        let rest = ran - std::mem::take(&mut self.ticked);
        self.advance(rest, true);
    }

    /// The steps stream woke from a rest: the spans stream's pass for it, its block landing
    /// before the wake step's cycles.
    fn woke(&mut self) {
        self.signals.reset(self.spans.fast());
        let Run { spans, length, signals, .. } = self;
        spans.woke(signals, &mut Reach::length(length));
        self.apply_signals();
    }

    /// A bus event the step flagged: the spans stream's row query, then the rewrite pass.
    fn bus_event(&mut self, read: bool) {
        transport::bus_event(&mut Rewriting { run: self }, read);
    }

    /// A pass of the spans stream for its locks, which it keeps.
    fn lock_query(&mut self) -> u8 {
        let Run { steps: _, spans, segments, lines, length, rows, records, ports, copied, .. } = self;
        Reach { spans, segments, lines, length, rows, records, ports, copied }.lock_query()
    }

    /// The bus state's bits, from the wiring table.
    fn state_bits(&self) -> transport::StateBits {
        let s = &self.w.steps;
        transport::StateBits { locks: s.locks_bits, busy: s.busy_bit, replace: s.replace, shift: s.access_shift, writes: s.write_slots, write_shift: s.write_shift }
    }

    /// The bus state `cycles` cycles into the step, as the spans stream emitted it.
    fn ahead(&mut self, cycles: u64, locks: bool, write: bool) -> u8 {
        let bits = self.state_bits();
        let Run { spans, length, .. } = self;
        spans.ahead(cycles, locks, write, bits, &mut Reach::length(length))
    }

    /// The steps stream's request under `access`, its ports read through the other streams.
    fn request(&mut self, access: &[u8]) -> steps::Request {
        let Run { steps, spans, segments, lines, length, rows, records, ports, copied, .. } = self;
        let mut reach = Reach { spans, segments, lines, length, rows, records, ports, copied };
        steps.request(access, &mut reach, ports)
    }

    /// One step, each data access at its cycle: the other streams run up to it first
    /// (`Timeline`), and the step asks again under each access's bus state where it differs
    /// from the step's start, as the spans stream says it will be there.
    fn one_step(&mut self) -> u8 {
        let (busy, held) = (self.spans.busy(), self.spans.locks_now());
        self.in_step = true;
        self.spans.step = true;
        let (req, locks, asked) = transport::open_step(&mut Opening { run: self, busy }, held);
        let bits = self.state_bits();
        let start = transport::start_state(bits, locks, asked, busy);
        let first = Self::asked(&req);
        let mut h = Stepping { run: self, req: Some(req) };
        let (mut line, _, early) = transport::settle_accesses(&mut h, bits, first, start);
        let req = h.req.expect("the step's request");
        // The bus events in cycle order (`wiring::Events`): the fetches' first.
        for k in 0..req.fetch_events.len() {
            if req.fetch_events[k] {
                assert!(line.done <= k as u8, "fetch {k}'s event after the step ran to cycle {}", line.done);
                line.reach(&mut Stepping { run: self, req: None }, bits, k as u8);
                self.bus_event(true);
            }
        }
        let mut data = [0u8; 2];
        for i in 0..req.served.len() {
            if let Some(byte) = early[i] {
                assert!(!req.read_events[i], "read {i}, served while the busy copy blocked it, is a bus event");
                data[i] = byte;
                continue;
            }
            if req.enabled[i] {
                line.reach(&mut Stepping { run: self, req: None }, bits, req.at[i]);
            }
            if req.read_events[i] {
                self.bus_event(true);
            }
            data[i] = self.serve(&req, i);
        }
        let out = self.steps.finish(&req, asked, &data);
        for &p in &out.reads {
            if self.steps.blocked(p) {
                self.cut();
            }
        }
        for k in 0..out.idle.len() {
            line.reach(&mut Stepping { run: self, req: None }, bits, out.idle[k]);
            self.bus_event(false);
        }
        for w in &out.writes {
            line.reach(&mut Stepping { run: self, req: None }, bits, req.at[data.len() + w.slot]);
            if req.write_events[w.slot] {
                self.bus_event(false);
            }
            self.steps.land_write(w.slot);
            match self.ports.port(w.at) {
                Some(port) => self.write_port(port, w.value),
                None if self.steps.blocked(w.at) => self.cut(),
                None => {}
            }
            if let Some(second) = w.second {
                let port = self.ports.port(second).unwrap_or_else(|| panic!("a second port {second:#08x} that names no port"));
                self.write_port(port, w.value);
            }
        }
        if out.ack != 0 {
            let code = self.spans.wiring().ack;
            let Run { spans, length, copied, .. } = self;
            let mut landed = Vec::new();
            spans.pass_alone(Event::of(code, out.ack), &mut landed, &mut Reach::length(length));
            land(copied, &landed);
        }
        if out.stop {
            self.stop_pending = true;
        }
        self.in_step = false;
        self.spans.step = false;
        let Run { spans, length, copied, .. } = self;
        let mut landed = Vec::new();
        spans.step_ended(&mut landed, &mut Reach::length(length));
        land(copied, &landed);
        let cycles = out.cycles;
        self.steps.recycle(out);
        cycles
    }
}

impl Run {
    /// The wall clock at `seconds`: a clock cartridge's spans stream gets the seconds
    /// elapsed since the last reading, a pass for each binary digit set.
    pub fn set_time(&mut self, seconds: u64) {
        if !self.has_clock {
            return;
        }
        if let Some(last) = self.clock.replace(seconds) {
            let elapsed = seconds.saturating_sub(last);
            if elapsed > 0 {
                let (code, digits) = (self.spans.wiring().elapse, self.spans.wiring().digits);
                assert!(elapsed >> (digits - 1) >> 1 == 0, "{elapsed} s elapsed, past the event's digits");
                let Run { spans, length, copied, .. } = self;
                for k in (0..digits).filter(|k| elapsed >> k & 1 == 1) {
                    let mut landed = Vec::new();
                    spans.pass_alone(Event::of(code, k as u8), &mut landed, &mut Reach::length(length));
                    land(copied, &landed);
                }
            }
        }
    }

    /// The buttons held from now on: a pass, which may end a hold.
    pub fn set_buttons(&mut self, pressed: u8) {
        let code = self.spans.wiring().buttons;
        let Run { spans, length, copied, .. } = self;
        let mut landed = Vec::new();
        let held = spans.pass_holding(Event::of(code, pressed), &mut landed, &mut Reach::length(length));
        land(copied, &landed);
        if let Some(held) = held {
            self.lines.post(move |l| l.held = held);
        }
    }

    /// One frame of real time, as long as the spans stream says a frame lasts. Returns a
    /// save that finished in it.
    pub fn run_frame(&mut self, pressed: u8) -> Option<Vec<u8>> {
        self.set_buttons(pressed);
        self.frame_end += self.spans.frame_time;
        while self.halves < self.frame_end {
            self.step();
        }
        self.take_save(true)
    }

    /// A save due now: a finished one, or with `committed` false anything unsaved.
    pub fn take_save(&mut self, committed: bool) -> Option<Vec<u8>> {
        if self.save_len == 0 {
            return None;
        }
        let mut save = self.steps.take_save(committed, self.save_len)?;
        if self.has_clock {
            // The footer: the spans state's bits where the wiring routes them, then the wall time.
            let w = &self.w.steps;
            let mut footer = vec![0u8; w.footer_len];
            for (slot, bit, byte, to) in &w.footer_bits {
                footer[*byte] |= (self.spans.state(slot) >> bit & 1) << to;
            }
            let time = self.clock.unwrap_or(0).to_le_bytes();
            footer[w.time_at..w.time_at + time.len()].copy_from_slice(&time);
            save.extend(footer);
        }
        Some(save)
    }

    /// The samples mixed since the last call that reached this thread (with `all`, every
    /// one), through the board's filter.
    pub fn drain_audio(&mut self, filter: &mut transport::HighPass, out: &mut Vec<[i16; 2]>, all: bool) {
        for s in self.segments.drain(all) {
            out.push(filter.apply(s));
        }
    }
}

/// Where a block's signals land: the bytes copied for the steps stream, the length and
/// lines streams, and the record queue the lines stream follows.
struct Landing<'a> {
    copied: &'a mut Vec<(Option<u16>, u16)>,
    records: &'a VideoRecords,
    length: &'a mut LengthStream,
    rows: &'a mut Rows,
    lines: &'a mut Worker<LineStream>,
    own_memory: u32,
    held: &'a mut Vec<Copy>,
    /// Half units now, and whether the spans stream runs at its fast rate.
    halves: u64,
    fast: bool,
}

impl Sink for Landing<'_> {
    fn land(&mut self, copies: &[Copy]) {
        land(self.copied, copies);
    }

    fn hold(&mut self, copies: &mut Vec<Copy>) {
        self.held.append(copies);
    }

    fn render(&mut self, line: u8) {
        let records = take(self.records);
        self.rows.open(line, self.halves, self.fast, records, self.length, self.lines);
    }

    fn stopped(&mut self, stopped: bool) {
        self.rows.hold(stopped, self.length, self.lines);
    }

    fn frame(&mut self) {
        self.rows.close(self.length, self.lines);
        self.lines.post(|l| l.screen.end_frame());
    }

    fn blank(&mut self) {
        self.rows.close(self.length, self.lines);
        self.lines.post(|l| l.screen.blank());
    }

    fn own(&mut self, index: u8, value: u8) {
        self.records.lock().expect("the record queue").push((self.own_memory + index as u32, value));
        self.rows.write(index, value, self.halves, self.records, self.length, self.lines);
    }
}

/// The run as `transport::advance` drives it.
struct Blocking<'a> {
    run: &'a mut Run,
    hand: bool,
}

impl Blocks for Blocking<'_> {
    fn horizon(&self) -> u64 {
        self.run.spans.horizon()
    }

    fn block(&mut self, cycles: u64, stopped: bool) {
        let r = &mut *self.run;
        let rate = r.spans.rate();
        r.halves += rate * cycles;
        r.signals.reset(r.spans.fast());
        let Run { spans, length, signals, .. } = r;
        spans.advance(cycles, stopped, signals, &mut Reach::length(length));
        let events = crate::segments::Events { fall: r.signals.edge, reset: r.signals.reset, rise: r.signals.rise };
        r.segments.advance(cycles, events, r.signals.fast, rate);
        r.land_early();
        if self.hand {
            r.hand_over();
        }
        r.apply_signals();
        if self.hand {
            r.hand_over();
        }
    }

    fn stop(&mut self) {
        let r = &mut *self.run;
        if std::mem::take(&mut r.stop_pending) {
            r.signals.reset(r.spans.fast());
            let Run { spans, length, signals, .. } = r;
            spans.stop(signals, &mut Reach::length(length));
            r.apply_signals();
        }
    }

    fn owed(&self) -> u64 {
        self.run.spans.owed()
    }

    fn paused(&self) -> bool {
        self.run.spans.paused()
    }

    fn settle(&mut self) {
        let r = &mut *self.run;
        if !r.held.is_empty() {
            for c in std::mem::take(&mut r.held) {
                r.copied.extend(c.bytes());
            }
            if self.hand {
                r.hand_over();
            }
        }
    }
}

impl Run {
    /// A request as the access loop needs it.
    fn asked(req: &steps::Request) -> transport::Asked {
        transport::Asked { cycles: req.at, reads: req.enabled, masks: req.masks }
    }

    /// Read `i` of `req` as the host serves it: through the port the model named, or 0
    /// where the model reads its own memory.
    fn serve(&mut self, req: &steps::Request, i: usize) -> u8 {
        if !(req.enabled[i] && req.served[i]) {
            return 0;
        }
        let port = req.ports[i].expect("a served read's port");
        let Run { spans, segments, lines, length, rows, records, ports, copied, .. } = self;
        Reach { spans, segments, lines, length, rows, records, ports, copied }.read(port)
    }
}

/// The run as a bus event sees it (`transport::Rewrites`): the row from a spans pass, the
/// rewrite from the steps stream.
struct Rewriting<'a> {
    run: &'a mut Run,
}

impl transport::Rewrites for Rewriting<'_> {
    fn row(&mut self) -> u8 {
        let code = self.run.spans.wiring().row_query;
        let Run { spans, length, copied, .. } = &mut *self.run;
        let mut landed = Vec::new();
        let row = spans.pass_alone(Event::of(code, 0), &mut landed, &mut Reach::length(length)).read;
        land(copied, &landed);
        row
    }

    fn rewrite(&mut self, row: u8, read: bool) {
        self.run.steps.rewrite(row, read);
    }
}

/// The run as a step's opening sees it (`transport::open_step`): each request begins the
/// step again under the locks given, the busy bit as at the step's start.
struct Opening<'a> {
    run: &'a mut Run,
    busy: bool,
}

impl transport::Opening for Opening<'_> {
    type Request = steps::Request;

    fn request(&mut self, locks: Option<u8>) -> steps::Request {
        let r = &mut *self.run;
        r.steps.begin(&steps::Signals { state: r.spans.state_bytes(), emitted: r.spans.emitted(), busy: self.busy, locks });
        r.request(&[0; transport::SLOTS / 2])
    }

    fn fetch_reads_locks(&self, req: &steps::Request) -> bool {
        req.masks[transport::DATA] & self.run.state_bits().locks != 0
    }

    fn wants_locks(&self, req: &steps::Request) -> bool {
        req.wants_locks
    }

    fn lock_query(&mut self) -> u8 {
        self.run.lock_query()
    }
}

/// The run as a step's accesses see it (`transport::Accesses`).
struct Stepping<'a> {
    run: &'a mut Run,
    /// The last request made (none while only reaching an access's cycle).
    req: Option<steps::Request>,
}

impl transport::Accesses for Stepping<'_> {
    fn request(&mut self, access: &[u8]) -> transport::Asked {
        let req = self.run.request(access);
        let asked = Run::asked(&req);
        self.req = Some(req);
        asked
    }

    fn serve(&mut self, read: usize) -> u8 {
        let req = self.req.take().expect("a request to serve");
        let byte = self.run.serve(&req, read);
        self.req = Some(req);
        byte
    }

    fn ahead(&mut self, cycles: u8, locks: bool, write: bool) -> u8 {
        self.run.ahead(cycles as u64, locks, write)
    }

    fn tick(&mut self, cycles: u8) {
        transport::advance(&mut Blocking { run: &mut *self.run, hand: true }, cycles as u64, true);
        self.run.ticked += cycles as u64;
    }

    fn locks(&mut self) -> Option<u8> {
        self.run.spans.locks_now()
    }

    fn busy(&mut self) -> bool {
        self.run.spans.busy()
    }

    fn lock_query(&mut self) -> u8 {
        self.run.lock_query()
    }
}
