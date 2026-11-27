//! The spans stream: a pass after the cycles since the last one, at an event or at the
//! horizon the last pass emitted. Between passes the host only counts: down to the
//! horizon, to each start the last pass scheduled, and to the edge it named for the
//! segments stream (`wiring::Spans`).

use crate::slots::{At, Bit, Positions, Seq};
use gbcformer::transport::{self, Cut, Run, Schedule, MAX_STARTS};
use gbcformer::wiring::{Code, Snapshot, Spans};

/// An event: its code bytes and argument.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub code: Code,
    pub arg: u8,
}

impl Event {
    pub fn of(code: Code, arg: u8) -> Event {
        Event { code, arg }
    }
}

struct Out {
    read: At,
    horizon: At,
    serial_start: Bit,
    busy: Bit,
    render: Bit,
    frame: Bit,
    blank: Bit,
    pause: Bit,
    run: Bit,
    run_page: At,
    run_line: At,
    run_last_early: Bit,
    run_before: At,
    run_last: At,
    run_dest: At,
    run_fill: At,
    block_copy: Bit,
    block_source: At,
    block_dest: At,
    block_len: At,
    block_held: Bit,
    stall: At,
    own: Bit,
    own_index: At,
    first: At,
    count: At,
    period: At,
    tail: At,
    start_phase: At,
    lines: Vec<At>,
    edge: At,
    edge_held: Bit,
    edge_drop: At,
    rise: At,
    wake: Bit,
    rate: At,
    frame_time: At,
    busy_left: At,
    busy_from: At,
    locks: Vec<(At, At)>,
    follow: At,
}

/// One pass's outcome, as the scheduler acts on it.
#[derive(Clone, Debug)]
pub struct Pass {
    pub read: u8,
    pub run: Option<Run>,
    pub block: Option<(u16, u16, u16, bool)>,
    pub follow: Option<Event>,
    pub own: Option<u8>,
    pub render: bool,
    pub frame: bool,
    pub blank: bool,
    pub locks: Vec<(u8, u64)>,
}

pub struct SpanStream {
    pub seq: Seq,
    w: Spans,
    o: Out,
    carry: Vec<(usize, usize)>,
    /// The state, by input position: the last pass's outputs.
    state: Vec<u8>,
    phase: usize,
    position: usize,
    cycles: Vec<usize>,
    paused_cycles: Vec<usize>,
    moved: Vec<usize>,
    owed: Vec<usize>,
    event: Vec<usize>,
    fixed: Vec<(usize, u8)>,
    /// The input that says a step runs, and whether one does.
    step_input: usize,
    pub step: bool,
    late: usize,
    asks: Bit,
    /// The request's slots, which the result pass reads decoded, and the result's.
    request: Vec<usize>,
    result: Positions,
    /// The first pass's slots (request then result), and the inputs a pass sets.
    both: Positions,
    bytes: Vec<u8>,
    idle: Bit,
    kept: Bit,
    uncounted: Bit,
    held: Bit,
    serial_at: usize,
    /// The countdown that cuts the passes.
    t: transport::Spans,
    flags_busy: bool,
    flags_pause: bool,
    pub wake: bool,
    /// Passes at a rest, a wake or a step's end, which the proofs hold to the full program's
    /// count.
    pub rest_passes: u64,
    /// Accesses with the locks whose cycle a held span covers where the reads' and the
    /// writes' lock pairs differ, so the access's kind picks its pair.
    pub pairs_differ: u64,
    rate: u64,
    pub frame_time: u64,
    /// The cycles the busy copy blocks for from the last pass, and from which it blocks.
    busy_left: u64,
    busy_from: u64,
    /// The last kept pass's output bytes (a look-ahead pass leaves them).
    emitted: Vec<u8>,
    locks: Vec<(u8, u64)>,
    pub serial: Vec<u8>,
    pub passes: u64,
}

impl SpanStream {
    /// The stream from the snapshot's state; the first pass runs at once over no cycles
    /// to learn the horizon.
    pub fn new(seq: Seq, w: &Spans, snap: &Snapshot) -> SpanStream {
        let f = |a: &gbcformer::wiring::Field| seq.field(a);
        let b = |a: &gbcformer::wiring::Flag| seq.flag(a);
        let o = Out {
            read: f(&w.read),
            horizon: f(&w.horizon),
            serial_start: b(&w.serial_start),
            busy: b(&w.busy),
            render: b(&w.render),
            frame: b(&w.frame),
            blank: b(&w.blank),
            pause: b(&w.pause),
            run: b(&w.run),
            run_page: f(&w.run_page),
            run_line: f(&w.run_line),
            run_last_early: b(&w.run_last_early),
            run_before: f(&w.run_before),
            run_last: f(&w.run_last),
            run_dest: f(&w.run_dest),
            run_fill: f(&w.run_fill),
            block_copy: b(&w.block_copy),
            block_source: f(&w.block_source),
            block_dest: f(&w.block_dest),
            block_len: f(&w.block_len),
            block_held: b(&w.block_held),
            stall: f(&w.stall),
            own: b(&w.own),
            own_index: f(&w.own_index),
            first: f(&w.first),
            count: f(&w.count),
            period: f(&w.period),
            tail: f(&w.tail),
            start_phase: f(&w.start_phase),
            lines: w.lines.iter().map(f).collect(),
            edge: f(&w.edge),
            edge_held: b(&w.edge_held),
            edge_drop: f(&w.edge_drop),
            rise: f(&w.rise),
            wake: b(&w.wake),
            rate: f(&w.rate),
            frame_time: f(&w.frame_time),
            busy_left: f(&w.busy_left),
            busy_from: f(&w.busy_from),
            locks: w.locks.iter().map(|(v, u)| (f(v), f(u))).collect(),
            follow: f(&w.follow),
        };
        let carry: Vec<(usize, usize)> = w.carry.iter().map(|(o, i)| (seq.output(o), seq.input(i))).collect();
        let mut state = vec![0u8; seq.n_inputs()];
        for (name, v) in &snap.spans {
            state[seq.input(name)] = *v;
        }
        let ins = |v: &[String]| v.iter().map(|n| seq.input(n)).collect::<Vec<_>>();
        let asks = seq.flag(&w.asks);
        let request = seq.outputs(&w.request);
        let result: Vec<usize> = (seq.n_inputs()..seq.cfg.seq()).filter(|p| !request.contains(p)).collect();
        let mut s = SpanStream {
            o,
            carry,
            state,
            phase: seq.input(&w.phase),
            position: seq.input(&w.position),
            cycles: ins(&w.cycles),
            paused_cycles: ins(&w.paused_cycles),
            moved: ins(&w.moved),
            owed: ins(&w.owed),
            event: ins(&w.event),
            fixed: w.fixed.iter().map(|(n, b)| (seq.input(n), *b)).collect(),
            step_input: seq.input(&w.step_input),
            step: false,
            late: seq.input(&w.late),
            asks,
            both: Positions::new(request.iter().chain(&result).copied().collect()),
            bytes: vec![0; seq.n_inputs()],
            result: Positions::new(result),
            request,
            idle: seq.input_flag(&w.idle),
            kept: seq.input_flag(&w.kept),
            uncounted: seq.input_flag(&w.uncounted),
            held: seq.input_flag(&w.held),
            serial_at: seq.input(&w.serial),
            t: transport::Spans::new(w.max_span),
            flags_busy: false,
            flags_pause: false,
            wake: false,
            rest_passes: 0,
            pairs_differ: 0,
            rate: 0,
            frame_time: 0,
            busy_left: 0,
            busy_from: 0,
            emitted: Vec::new(),
            locks: vec![(0, 0); w.locks.len()],
            serial: Vec::new(),
            passes: 0,
            w: w.clone(),
            seq,
        };
        // From a snapshot mid-frame the first pass may ask for the last line's length.
        let length = snap.length;
        s.pass(Event::of(s.w.tick, 0), &mut || length);
        s.passes = 0;
        s
    }

    pub fn paused(&self) -> bool {
        self.flags_pause
    }

    pub fn busy(&self) -> bool {
        self.flags_busy
    }

    pub fn fast(&self) -> bool {
        self.rate == self.w.fast_rate
    }

    /// The cycles the steps stream stays stopped once the step that runs has ended.
    pub fn owed(&self) -> u64 {
        self.t.owed()
    }

    pub fn rate(&self) -> u64 {
        self.rate
    }

    /// The cycles the scheduler may run before this stream must act.
    pub fn horizon(&self) -> u64 {
        self.t.horizon()
    }

    /// The locks while they hold.
    pub fn locks_now(&self) -> Option<u8> {
        transport::held_lock(&self.locks, self.t.elapsed())
    }

    /// The state input byte `slot`.
    pub fn state(&self, slot: &str) -> u8 {
        self.state[self.seq.input(slot)]
    }

    /// The state input bytes, by input position.
    pub fn state_bytes(&self) -> &[u8] {
        &self.state
    }

    /// The last kept pass's output bytes, by position.
    pub fn emitted(&self) -> &[u8] {
        &self.emitted
    }

    /// The position of output slot `slot`.
    pub fn output_position(&self, slot: &str) -> usize {
        self.seq.output(slot)
    }

    /// The input position of state slot `slot`.
    pub fn position(&self, slot: &str) -> usize {
        self.seq.input(slot)
    }

    pub fn state_at(&self, b: Bit) -> bool {
        self.state[b.position()] & b.mask() != 0
    }

    pub fn is_held(&self) -> bool {
        self.state_at(self.held)
    }

    /// The pass after the cycles since the last one, then `event`.
    fn pass(&mut self, event: Event, length: &mut dyn FnMut() -> u8) -> Pass {
        let start = self.t.open();
        let mut state = self.state.clone();
        start.place(&mut state, self.phase, self.position);
        self.run(state, start.cycles, start.since, event, length)
    }

    /// Sets the pass's inputs and decodes it, the late length asked for when the pass asks;
    /// the stream keeps nothing of it yet.
    fn evaluate(&mut self, state: &[u8], cycles: u64, since: u64, event: Event, length: &mut dyn FnMut() -> u8) {
        assert!(cycles <= self.w.max_span, "a span of {cycles} cycles");
        let mut bytes = std::mem::take(&mut self.bytes);
        bytes.iter_mut().for_each(|b| *b = 0);
        for &(_, i) in &self.carry {
            bytes[i] = state[i];
        }
        let (d, q) = if self.flags_pause { (0, cycles as u16) } else { (cycles as u16, 0) };
        let put = |bytes: &mut Vec<u8>, at: &[usize], v: u64| {
            for (k, &p) in at.iter().enumerate() {
                bytes[p] = (v >> (8 * k)) as u8;
            }
        };
        put(&mut bytes, &self.cycles, d as u64);
        put(&mut bytes, &self.paused_cycles, q as u64);
        put(&mut bytes, &self.moved, since as u16 as u64);
        put(&mut bytes, &self.owed, self.t.owed());
        bytes[self.event[0]] = event.code[0];
        bytes[self.event[1]] = event.code[1];
        bytes[self.event[2]] = event.arg;
        for &(p, b) in &self.fixed {
            bytes[p] = b;
        }
        bytes[self.step_input] = self.step as u8;
        // The two-pass protocol: the late input masked, decode request and result; with
        // the request set, the late input and the result again.
        for (p, &b) in bytes.iter().enumerate() {
            if p == self.late { self.seq.mask_input(p) } else { self.seq.set_input(p, b) }
        }
        self.bytes = bytes;
        self.seq.decode(&self.both);
        if self.seq.is(self.asks) {
            let ml = length();
            self.seq.set_input(self.late, ml);
            self.seq.show(&self.request);
            self.seq.decode(&self.result);
            self.seq.hide(&self.request);
        }
    }

    /// The pass over `cycles` from `state`, kept: its state, its cut, its outputs.
    fn run(&mut self, state: Vec<u8>, cycles: u64, since: u64, event: Event, length: &mut dyn FnMut() -> u8) -> Pass {
        self.evaluate(&state, cycles, since, event, length);
        for &(o, i) in &self.carry {
            self.state[i] = self.seq.bytes[o];
        }
        let o = &self.o;
        let s = &self.seq;
        let mut lines = [0u8; MAX_STARTS];
        for (l, a) in lines.iter_mut().zip(&o.lines) {
            *l = s.get(a) as u8;
        }
        let schedule = Schedule { count: s.get(&o.count) as usize, first: s.get(&o.first), period: s.get(&o.period), tail: s.get(&o.tail), phase: s.get(&o.start_phase) as u8, lines };
        assert!(schedule.count <= o.lines.len(), "spans: a schedule of {} starts", schedule.count);
        self.t.close(&Cut { horizon: s.get(&o.horizon), schedule, edge: (!s.is(o.edge_held)).then(|| s.get(&o.edge)), drop: s.get(&o.edge_drop), rise: (!s.is(o.edge_held)).then(|| s.get(&o.rise)), owed: s.get(&o.stall) });
        if s.is(o.serial_start) {
            self.serial.push(self.state[self.serial_at]);
        }
        let run = s.is(o.run).then(|| Run { page: s.get(&o.run_page) as u8, line: s.get(&o.run_line) as u8, last_early: s.is(o.run_last_early), before: s.get(&o.run_before) as u8, last: s.get(&o.run_last) as u8, dest: s.get(&o.run_dest) as u8, fill: s.get(&o.run_fill) as u8 });
        if let Some(r) = run {
            r.check(cycles);
        }
        let pass = Pass {
            read: s.get(&o.read) as u8,
            run,
            block: s.is(o.block_copy).then(|| (s.get(&o.block_source) as u16, s.get(&o.block_dest) as u16, s.get(&o.block_len) as u16, s.is(o.block_held))),
            follow: match s.get(&o.follow) { 0 => None, c => Some(Event::of([c as u8, 0], 0)) },
            own: s.is(o.own).then(|| s.get(&o.own_index) as u8),
            render: s.is(o.render),
            frame: s.is(o.frame),
            blank: s.is(o.blank),
            locks: o.locks.iter().map(|(v, u)| (s.get(v) as u8, s.get(u))).collect(),
        };
        self.flags_busy = s.is(o.busy);
        self.flags_pause = s.is(o.pause);
        self.wake = s.is(o.wake);
        let rate = s.get(&o.rate);
        assert!(self.w.rates.contains(&rate), "spans: a cycle of {rate} halves");
        self.rate = rate;
        self.frame_time = s.get(&o.frame_time);
        self.busy_left = s.get(&o.busy_left);
        self.busy_from = s.get(&o.busy_from);
        self.emitted.clone_from(&s.bytes);
        self.locks = pass.locks.clone();
        self.passes += 1;
        pass
    }

    /// The bus state `cycles` cycles from now, as the last pass emitted it: the busy copy's
    /// bit while its cycles run, and with `locks` the locks of the span that holds there.
    /// Past the spans, a pass over the span to that cycle gives them, and the stream keeps
    /// nothing of it (`wiring::Steps::access_at`).
    pub fn ahead(&mut self, cycles: u64, locks: bool, write: bool, bits: transport::StateBits, length: &mut dyn FnMut() -> u8) -> u8 {
        let at = self.t.elapsed() + cycles;
        let held = std::mem::take(&mut self.locks);
        let differ = transport::held_lock(&held, at).map_or(0, |l| ((l ^ l >> bits.write_shift) & bits.locks != 0) as u64);
        self.pairs_differ += locks as u64 * differ;
        let s = transport::bus_state(&held, self.busy_from..self.busy_left, at, locks, write, bits, || {
            let start = self.t.ahead(cycles);
            let mut state = self.state.clone();
            start.place(&mut state, self.phase, self.position);
            self.evaluate(&state, start.cycles, start.since, Event::of(self.w.lock_query, 0), length);
            self.seq.get(&self.o.locks[0].0) as u8
        });
        self.locks = held;
        s
    }

    /// The steps stream came to rest: a pass while the state says a block copy runs.
    pub fn rest(&mut self, landed: &mut Vec<transport::Copy>, length: &mut dyn FnMut() -> u8) {
        use transport::Passes;
        if !self.state_at(self.idle) {
            self.rest_passes += 1;
            self.pass_alone(Event::of(self.w.rest, 0), landed, length);
        }
    }

    /// A step of the steps stream ended: a pass while the state says a block copy waits to be
    /// counted.
    pub fn step_ended(&mut self, landed: &mut Vec<transport::Copy>, length: &mut dyn FnMut() -> u8) {
        use transport::Passes;
        if self.state_at(self.uncounted) {
            self.rest_passes += 1;
            self.pass_alone(Event::of(self.w.step_end, 0), landed, length);
        }
    }

    /// The steps stream woke from a rest: a pass while the state keeps it.
    pub fn woke(&mut self, s: &mut transport::Signals, length: &mut dyn FnMut() -> u8) {
        use transport::Passes;
        if self.state_at(self.kept) {
            self.rest_passes += 1;
            self.pass_with(Event::of(self.w.wake_up, 0), s, length, None);
        }
    }

    pub fn wiring(&self) -> &Spans {
        &self.w
    }
}

impl transport::Passes for SpanStream {
    type Event = Event;
    type Out = Pass;
    type Cx<'a> = dyn FnMut() -> u8 + 'a;

    fn once(&mut self, event: Event, length: &mut (dyn FnMut() -> u8 + '_)) -> Pass {
        self.pass(event, length)
    }

    fn asks(p: &Pass) -> transport::Asks {
        transport::Asks { run: p.run, block: p.block, render: p.render, frame: p.frame, blank: p.blank, own: p.own }
    }

    fn follow(p: &Pass) -> Option<Event> {
        p.follow
    }

    fn own(&self, event: transport::Own) -> Event {
        let code = match event {
            transport::Own::Tick => self.w.tick,
            transport::Own::Stop => self.w.stop,
        };
        Event::of(code, 0)
    }

    fn countdown(&mut self) -> &mut transport::Spans {
        &mut self.t
    }

    fn run_bits(&self) -> u8 {
        self.w.run_bits
    }

    fn held(&self) -> bool {
        self.is_held()
    }

    fn line(&self) -> u8 {
        self.state[self.position]
    }
}
