//! The stepper: a pass a step, over its own memory, which follows the lines stream's
//! records. It draws the lines the length stream's verdicts mark (`wiring::Stepper`): the
//! first pass takes the window after the line before, each later pass takes every carried
//! output back as an input and the queue's head write, and the queue moves only as a pass's
//! report says.

use crate::slots::{Positions, Seq};
use gbcformer::wiring::{Lines, Snapshot, Stepper};

pub struct StepperStream {
    pub seq: Seq,
    w: Stepper,
    line: usize,
    flags: usize,
    counter: usize,
    write: Vec<usize>,
    carry: Vec<(usize, usize)>,
    report: usize,
    x: usize,
    colour: [usize; 2],
    counter_out: usize,
    window_out: usize,
    length: [usize; 2],
    /// Every output a pass decodes.
    decoded: Positions,
    modes: [u8; 2],
    mode: bool,
    mode_cell: u32,
    mode_mask: u8,
    width: usize,
    pub passes: u64,
}

/// A row the stepper drew: each pixel's two colour bytes, the window after it (counter,
/// armed, carried) and its length in the spans stream's unit.
pub struct Drawn {
    pub low: Vec<u8>,
    pub high: Vec<u8>,
    pub window: (u8, bool, bool),
    pub length: u8,
}

impl StepperStream {
    pub fn new(seq: Seq, w: &Stepper, lines: &Lines, snap: &Snapshot) -> StepperStream {
        let carry: Vec<(usize, usize)> = w.carry.iter().map(|(o, i)| (seq.output(o), seq.input(i))).collect();
        let (report, x, counter_out, window_out) = (seq.output(&w.report), seq.output(&w.x), seq.output(&w.counter_out), seq.output(&w.window_out));
        let (colour, length) = (w.colour.each_ref().map(|n| seq.output(n)), w.length.each_ref().map(|n| seq.output(n)));
        // In slot order, as the full program decodes every output but the record slots.
        let mut decoded: Vec<usize> = [report, x, counter_out, window_out].into_iter().chain(colour).chain(length).chain(carry.iter().map(|c| c.0)).collect();
        decoded.sort_unstable();
        StepperStream {
            line: seq.input(&w.line),
            flags: seq.input(&w.flags),
            counter: seq.input(&w.counter),
            write: w.write.iter().map(|n| seq.input(n)).collect(),
            carry,
            report,
            x,
            colour,
            counter_out,
            window_out,
            length,
            decoded: Positions::new(decoded),
            modes: snap.stepper_modes,
            mode: snap.line_mode,
            mode_cell: lines.mode_cell,
            mode_mask: lines.mode_mask,
            width: lines.pixels.len(),
            passes: 0,
            w: w.clone(),
            seq,
        }
    }

    /// Applies `records`, the ones kept since the last.
    pub fn catch_up(&mut self, records: &[(u32, u8)]) {
        for &(addr, byte) in records {
            if addr == self.mode_cell {
                self.mode = byte & self.mode_mask != 0;
            } else {
                self.seq.engine.keep_write(addr, byte);
            }
        }
    }

    /// Draws `row` from `window`, the window after the row before, with `writes`, the queue
    /// the length stream's verdicts took: each write's time (two bytes), index and value.
    pub fn draw(&mut self, row: u8, fast: bool, held: bool, window: (u8, bool, bool), writes: &[[u8; 4]]) -> Drawn {
        let w = &self.w;
        let flags = self.modes[self.mode as usize] | if fast { w.fast } else { 0 } | if held { w.held } else { 0 };
        let first = flags | w.start | if window.1 { w.armed } else { 0 } | if window.2 { w.carried } else { 0 };
        self.seq.set_input(self.line, row);
        self.seq.set_input(self.flags, first);
        self.seq.set_input(self.counter, window.0);
        for p in self.write.clone().into_iter().chain(self.carry.iter().map(|c| c.1).collect::<Vec<_>>()) {
            self.seq.set_input(p, 0);
        }
        let (mut low, mut high) = (vec![0; self.width], vec![0; self.width]);
        let (mut head, mut applied) = (0, 0u8);
        loop {
            self.seq.decode(&self.decoded);
            self.passes += 1;
            let b = &self.seq.bytes;
            let report = b[self.report];
            if report & self.w.pixel != 0 {
                let x = b[self.x] as usize;
                assert!(x < self.width, "a pixel at {x}, past the row's {} pixels", self.width);
                (low[x], high[x]) = (b[self.colour[0]], b[self.colour[1]]);
            }
            if report & self.w.applied != 0 {
                applied += 1;
                if report & self.w.finished != 0 {
                    (head, applied) = (head + 1, 0);
                }
            }
            if report & self.w.done != 0 {
                let state = b[self.window_out];
                let raw = (u16::from_le_bytes([b[self.length[0]], b[self.length[1]]]) >> self.w.shift) as i64 + self.w.bias;
                let length = u8::try_from(raw).unwrap_or_else(|_| panic!("refused: the stepper emitted a mode-3 length of {raw}, outside 0..=255"));
                return Drawn { low, high, window: (b[self.counter_out], state & self.w.armed_mask != 0, state & self.w.carried_mask != 0), length };
            }
            let pending: Vec<u8> = match writes.get(head) {
                Some(&[t0, t1, index, value]) => vec![t0, t1, index, value, applied | self.w.pending],
                None => vec![0; self.write.len()],
            };
            let carried: Vec<(usize, u8)> = self.carry.iter().map(|&(out, input)| (input, b[out])).collect();
            self.seq.set_input(self.flags, flags);
            self.seq.set_input(self.counter, 0);
            for (&p, v) in self.write.iter().zip(pending) {
                self.seq.set_input(p, v);
            }
            for (p, v) in carried {
                self.seq.set_input(p, v);
            }
        }
    }
}

impl StepperStream {
    /// The cell whose records set the mode the rows draw in.
    pub fn mode_cell(&self) -> u32 {
        self.mode_cell
    }
}
