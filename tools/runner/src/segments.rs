//! The segments stream: its passes, cut by the shared `transport::Cutter` (a write, a
//! cycle's events and a change of speed open a segment; the countdowns the last pass
//! emitted, or a read, close one; the cycles a pass leaves are counted again).

use crate::slots::{At, Positions, Seq};
use gbcformer::transport::{Cutter, Edges, Emitted, Segment, SegmentPass};
use gbcformer::wiring::{Segments, Snapshot};

/// The frame sequencer's events at a cycle.
pub type Events = Edges;

/// The stream's passes: the inputs a segment sets, the outputs it carries over.
pub struct Passes {
    pub seq: Seq,
    /// Every output slot: each pass decodes them all.
    all: Positions,
    w: Segments,
    carry: Vec<(usize, usize)>,
    state: Vec<u8>,
    inputs: Vec<usize>,
    sample: Vec<At>,
    read_out: usize,
    to_sample_out: usize,
    to_end_out: usize,
    left_out: usize,
    /// The samples mixed since the last drain.
    pub samples: Vec<[i16; 2]>,
    read: Option<(u8, u8)>,
    pub passes: u64,
}

impl SegmentPass for Passes {
    fn pass(&mut self, seg: &Segment) -> Emitted {
        let most = if seg.fast { self.w.max_fast_cycles } else { self.w.max_cycles };
        assert!(seg.cycles <= most, "a segment of {} cycles", seg.cycles);
        for p in 0..self.seq.n_inputs() {
            self.seq.set_input(p, self.state[p]);
        }
        let (wa, wv) = seg.write.unwrap_or((0, 0));
        let e = seg.events;
        let ev = if seg.write.is_some() { self.w.on_write } else { 0 } | if e.fall { self.w.on_fall } else { 0 } | if e.reset { self.w.on_reset } else { 0 } | if e.rise { self.w.on_rise } else { 0 } | if seg.sample { self.w.on_sample } else { 0 } | if seg.fast { self.w.on_fast } else { 0 };
        let (a, v, x, d, r) = (self.inputs[0], self.inputs[1], self.inputs[2], self.inputs[3], self.inputs[4]);
        self.seq.set_input(a, wa);
        self.seq.set_input(v, wv);
        self.seq.set_input(x, ev);
        self.seq.set_input(d, seg.cycles as u8);
        self.seq.set_input(r, seg.read.unwrap_or(0));
        self.seq.decode(&self.all);
        for &(o, i) in &self.carry {
            self.state[i] = self.seq.bytes[o];
        }
        self.passes += 1;
        let b = &self.seq.bytes;
        Emitted { to_sample: b[self.to_sample_out] as u64, to_end: b[self.to_end_out] as u64, rest: b[self.left_out] as u64 }
    }

    fn keep(&mut self, seg: &Segment) {
        if seg.sample {
            let side = |a: &At| self.seq.get(a) as u16 as i16;
            self.samples.push([side(&self.sample[0]), side(&self.sample[1])]);
        }
        if let Some(a) = seg.read {
            self.read = Some((a, self.seq.bytes[self.read_out]));
        }
    }
}

pub struct SegmentStream {
    pub p: Passes,
    cut: Cutter,
}

impl SegmentStream {
    /// The stream from the snapshot's state; a first pass over no cycles emits the
    /// distance to the next sample.
    pub fn new(seq: Seq, w: &Segments, snap: &Snapshot) -> SegmentStream {
        let carry: Vec<(usize, usize)> = w.carry.iter().map(|(o, i)| (seq.output(o), seq.input(i))).collect();
        let mut state = vec![0u8; seq.n_inputs()];
        for (name, v) in &snap.segments {
            state[seq.input(name)] = *v;
        }
        let inputs = vec![seq.input(&w.write_address), seq.input(&w.write_value), seq.input(&w.events), seq.input(&w.cycles), seq.input(&w.read_address)];
        let mut p = Passes {
            all: Positions::new((seq.n_inputs()..seq.cfg.seq()).collect()),
            carry,
            state,
            inputs,
            sample: w.sample.iter().map(|f| seq.field(f)).collect(),
            read_out: seq.output(&w.read),
            to_sample_out: seq.output(&w.to_sample),
            to_end_out: seq.output(&w.to_end),
            left_out: seq.output(&w.left),
            samples: Vec::new(),
            read: None,
            passes: 0,
            w: w.clone(),
            seq,
        };
        let cut = Cutter::new(&mut p);
        p.passes = 0;
        SegmentStream { p, cut }
    }

    pub fn write(&mut self, address: u8, value: u8) {
        self.p.read = None;
        self.cut.write(&mut self.p, address, value);
    }

    /// `cycles` cycles of `rate` halves each, the last one with `events`.
    pub fn advance(&mut self, cycles: u64, events: Events, fast: bool, rate: u64) {
        for k in 1..=cycles {
            self.p.read = None;
            self.cut.events(&mut self.p, if k == cycles { events } else { Events::default() }, fast);
            self.cut.count(&mut self.p, fast, rate);
        }
    }

    /// Closes the open segment with a read of `address`, and returns the byte it read.
    pub fn read(&mut self, address: u8) -> u8 {
        self.cut.read(&mut self.p, address);
        match self.p.read {
            Some((a, v)) if a == address => v,
            _ => unreachable!("the read's pass just ran"),
        }
    }
}

/// One call queued for the segments stream.
#[derive(Clone, Copy, Debug)]
enum Op {
    Write(u8, u8),
    Advance(u64, Events, bool, u64),
}

/// The segments stream behind a worker, its writes and cycles sent in batches: a read
/// or a drain sends what is queued first, so the stream sees every call in order.
pub struct Queued {
    worker: crate::worker::Worker<SegmentStream>,
    ops: Vec<Op>,
    /// The cycles queued since the last batch.
    cycles: u64,
    /// The samples each batch mixed, sent back as the batch ends.
    mixed: (std::sync::mpsc::Sender<Vec<[i16; 2]>>, std::sync::mpsc::Receiver<Vec<[i16; 2]>>),
}

impl Queued {
    pub fn new(worker: crate::worker::Worker<SegmentStream>) -> Queued {
        Queued { worker, ops: Vec::new(), cycles: 0, mixed: std::sync::mpsc::channel() }
    }

    /// Queues `op`. Two advances at one speed are one when the first ends at no event (the
    /// stream runs an advance cycle by cycle). A batch goes at 1000 ops or 1000 cycles.
    fn push(&mut self, op: Op) {
        if let (Op::Advance(c, e, f, r), Some(Op::Advance(pc, pe, pf, pr))) = (op, self.ops.last_mut())
            && *pf == f
            && *pr == r
            && *pe == Events::default()
        {
            *pc += c;
            *pe = e;
        } else {
            self.ops.push(op);
        }
        if let Op::Advance(c, ..) = op {
            self.cycles += c;
        }
        if self.ops.len() >= 1000 || self.cycles >= 1000 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let ops = std::mem::take(&mut self.ops);
        self.cycles = 0;
        if ops.is_empty() {
            return;
        }
        let mixed = self.mixed.0.clone();
        self.worker.send(move |s| {
            for op in ops {
                match op {
                    Op::Write(a, v) => s.write(a, v),
                    Op::Advance(c, e, f, r) => s.advance(c, e, f, r),
                }
            }
            if !s.p.samples.is_empty() {
                let _ = mixed.send(std::mem::take(&mut s.p.samples));
            }
        });
    }

    pub fn write(&mut self, address: u8, value: u8) {
        self.push(Op::Write(address, value));
    }

    pub fn advance(&mut self, cycles: u64, events: Events, fast: bool, rate: u64) {
        self.push(Op::Advance(cycles, events, fast, rate));
    }

    pub fn read(&mut self, address: u8) -> u8 {
        self.flush();
        self.worker.call(move |s| s.read(address))
    }

    /// The samples mixed so far that reached this thread, in order; with `all`, every
    /// sample of every call made, waiting for the stream.
    pub fn drain(&mut self, all: bool) -> Vec<[i16; 2]> {
        self.flush();
        let mut out = Vec::new();
        if all {
            let rest = self.worker.call(|s| std::mem::take(&mut s.p.samples));
            out.extend(self.mixed.1.try_iter().flatten());
            out.extend(rest);
        } else {
            out.extend(self.mixed.1.try_iter().flatten());
        }
        out
    }

    pub fn finish(mut self) -> SegmentStream {
        self.flush();
        self.worker.finish()
    }
}
