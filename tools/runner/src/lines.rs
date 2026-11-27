//! The lines stream: a pass per line the spans stream triggers (one per pass the
//! snapshot says a line takes), over its own memory, which follows the steps stream's
//! records and the own-memory bytes the ports write, in the order they were kept
//! (`wiring::Lines`). The host maps each pixel to the screen's colour.

use crate::present::Screen;
use crate::slots::{Positions, Seq};
use gbcformer::wiring::{Lines, Snapshot};

pub struct LineStream {
    pub seq: Seq,
    w: Lines,
    line: usize,
    counter: usize,
    flags: usize,
    pixels: Vec<usize>,
    decoded: Positions,
    window: (u8, bool, bool),
    line_modes: [u8; 2],
    /// The mode cell's bit as its last record set it.
    mode: bool,
    passes_per_line: u8,
    /// Ordinary STOP keeps the stream from its memory.
    pub held: bool,
    /// The last line's length, which a spans pass may ask for.
    pub length: u8,
    pub screen: Screen,
    pub lines: u64,
    /// A memory cell holds 2^`line_bits` bytes.
    line_bits: u8,
}

impl LineStream {
    pub fn new(seq: Seq, w: &Lines, snap: &Snapshot, screen: Screen, line_bits: u8) -> LineStream {
        let pixels: Vec<usize> = w.pixels.iter().map(|n| seq.output(n)).collect();
        let decoded = Positions::new(pixels.iter().copied().chain([seq.output(&w.counter_out), seq.output(&w.armed_out), seq.output(&w.length)]).collect());
        LineStream {
            line: seq.input(&w.line),
            counter: seq.input(&w.counter),
            flags: seq.input(&w.flags),
            pixels,
            decoded,
            window: snap.window,
            line_modes: snap.line_modes,
            mode: snap.line_mode,
            passes_per_line: snap.line_passes,
            held: snap.line_held,
            length: snap.length,
            screen,
            lines: 0,
            line_bits,
            w: w.clone(),
            seq,
        }
    }

    /// Applies `records`, the ones kept since the last, to the stream's memory.
    pub fn catch_up(&mut self, records: Vec<(u32, u8)>) {
        for (addr, byte) in records {
            if addr == self.w.mode_cell {
                self.mode = byte & self.w.mode_mask != 0;
                continue;
            }
            self.seq.engine.keep_write(addr, byte);
        }
    }

    /// One own-memory byte read through a port.
    pub fn own_read(&mut self, index: u8, records: Vec<(u32, u8)>) -> u8 {
        self.catch_up(records);
        let addr = self.w.own_memory + index as u32;
        let line = self.seq.engine.read_line(gbcformer::engine::DEST, addr >> self.line_bits);
        line[addr as usize % (1 << self.line_bits)]
    }

    /// One pass: `high` selects the second byte of each pixel.
    fn render(&mut self, row: u8, high: bool) -> (Vec<u8>, (u8, bool, bool), u8) {
        let flags = self.line_modes[self.mode as usize] | if self.window.1 { self.w.armed } else { 0 } | if self.window.2 { self.w.carried } else { 0 } | if high { self.w.high } else { 0 } | if self.held { self.w.held } else { 0 };
        self.seq.set_input(self.line, row);
        self.seq.set_input(self.counter, self.window.0);
        self.seq.set_input(self.flags, flags);
        self.seq.decode(&self.decoded);
        let px = self.pixels.iter().map(|&p| self.seq.bytes[p]).collect();
        let n = self.pixels.len();
        let decoded = &self.decoded;
        let (counter, state, length) = (self.seq.bytes[decoded[n]], self.seq.bytes[decoded[n + 1]], self.seq.bytes[decoded[n + 2]]);
        (px, (counter, state & self.w.armed_mask != 0, state & self.w.carried_mask != 0), length)
    }

    /// Draws line `row` into the frame being drawn, from the records its trigger handed
    /// over ([`LineStream::catch_up`]).
    pub fn draw(&mut self, row: u8) {
        let (low, window, length) = self.render(row, false);
        let high = if self.passes_per_line > 1 { self.render(row, true).0 } else { vec![0; low.len()] };
        self.length = length;
        self.window = window;
        self.screen.row(row as usize, &low, &high);
        self.lines += 1;
    }

    /// Takes the stepper's drawing of line `row` in place of its own.
    pub fn draw_stepped(&mut self, row: u8, d: crate::stepper::Drawn) {
        self.length = d.length;
        self.window = d.window;
        self.screen.row(row as usize, &d.low, &d.high);
        self.lines += 1;
    }

    /// The window after the last line drawn: the counter, armed, carried.
    pub fn window(&self) -> (u8, bool, bool) {
        self.window
    }
}
