//! The length stream: a pass per line the lines stream draws, run on the scheduler's thread
//! as the line is queued, over its own memory (the same records as the lines stream's, but
//! the ones the wiring skips). Its output is the line's length a spans pass may ask for,
//! so the spans stream never waits for the lines stream (`wiring::Length`). While the line is
//! open, a pass per own-memory write judges it.

use crate::slots::{Positions, Seq};
use gbcformer::wiring::{Length, Lines, Snapshot};

pub struct LengthStream {
    pub seq: Seq,
    w: Length,
    line: usize,
    flags: usize,
    write: Vec<usize>,
    outputs: Positions,
    modes: [u8; 2],
    mode: bool,
    mode_cell: u32,
    mode_mask: u8,
    armed: bool,
    carried: bool,
    /// The open line and the flag byte its pass took, which its judges take too.
    row: u8,
    row_flags: u8,
    /// The last line's length, and whether a spans pass read it.
    pub length: u8,
    pub asked: bool,
}

/// A write's verdict: whether it marks the line for the stepper, whether the stepper's queue
/// takes it, and its time's two bytes.
pub struct Verdict {
    pub marks: bool,
    pub takes: bool,
    pub time: [u8; 2],
}

impl LengthStream {
    pub fn new(seq: Seq, w: &Length, lines: &Lines, snap: &Snapshot) -> LengthStream {
        let outputs = Positions::new([&w.length, &w.armed_out, &w.verdict, &w.time[0], &w.time[1]].iter().map(|n| seq.output(n)).collect());
        LengthStream {
            line: seq.input(&w.line),
            flags: seq.input(&w.flags),
            write: w.write.iter().map(|n| seq.input(n)).collect(),
            outputs,
            modes: snap.length_modes,
            mode: snap.line_mode,
            mode_cell: lines.mode_cell,
            mode_mask: lines.mode_mask,
            armed: snap.window.1,
            carried: snap.window.2,
            row: 0,
            row_flags: 0,
            length: snap.length,
            asked: true,
            w: w.clone(),
            seq,
        }
    }

    /// Applies `records`, the ones kept since the last.
    pub fn catch_up(&mut self, records: &[(u32, u8)]) {
        for &(addr, byte) in records {
            if addr == self.mode_cell {
                self.mode = byte & self.mode_mask != 0;
            } else if !(self.w.skip.0..self.w.skip.1).contains(&addr) {
                self.seq.engine.keep_write(addr, byte);
            }
        }
    }

    /// The pass for line `row`: its length and the window's arming after it.
    pub fn line(&mut self, row: u8) {
        let flags = self.modes[self.mode as usize] | if self.armed { self.w.armed } else { 0 } | if self.carried { self.w.carried } else { 0 };
        (self.row, self.row_flags) = (row, flags);
        self.seq.set_input(self.line, row);
        self.seq.set_input(self.flags, flags);
        for i in 0..self.write.len() {
            let p = self.write[i];
            self.seq.set_input(p, 0);
        }
        self.decode();
        self.length = self.seq.bytes[self.outputs[0]];
        self.armed = self.seq.bytes[self.outputs[1]] & self.w.armed_mask != 0;
        self.carried = self.seq.bytes[self.outputs[1]] & self.w.carried_mask != 0;
        self.asked = false;
    }

    fn decode(&mut self) {
        self.seq.decode(&self.outputs);
    }

    /// A pass of the open line judging a write `halves` half units after the line's trigger.
    pub fn judge(&mut self, halves: u8, index: u8, value: u8) -> Verdict {
        self.seq.set_input(self.line, self.row);
        self.seq.set_input(self.flags, self.row_flags | self.w.write_flag);
        for (i, v) in [halves, index, value].into_iter().enumerate() {
            let p = self.write[i];
            self.seq.set_input(p, v);
        }
        self.decode();
        let (b, o) = (&self.seq.bytes, &self.outputs);
        Verdict { marks: b[o[2]] & self.w.marks != 0, takes: b[o[2]] & self.w.takes != 0, time: [b[o[3]], b[o[4]]] }
    }

    /// The window after a line the stepper drew, which the next line's pass starts from.
    pub fn set_window(&mut self, armed: bool, carried: bool) {
        (self.armed, self.carried) = (armed, carried);
    }
}
