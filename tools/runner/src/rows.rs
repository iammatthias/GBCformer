//! The open row. A trigger opens it: the records kept until then reach the lines, length and
//! stepper streams, and those kept after wait for it to close, at the next trigger, a frame, a
//! blank or a stop. While it is open, the length stream judges each own-memory write; a
//! row a verdict marks is the stepper's, which redraws it from the start with the writes its
//! queue took, the length the spans stream reads included while that read is still to come.

use crate::length::LengthStream;
use crate::lines::LineStream;
use crate::stepper::{Drawn, StepperStream};
use crate::worker::Worker;
use gbcformer::engine::VideoRecords;

pub struct Rows {
    pub stepper: StepperStream,
    open: Option<Open>,
    /// The streams are held from their memory.
    held: bool,
    /// Where the own-memory bytes sit in the lines stream's memory.
    own_memory: u32,
    /// Rows a record at the mode cell closed, reads the queue served, rows the stepper
    /// redrew before the spans stream read their length, and rows a write it took reached
    /// after that read.
    pub compat_closes: u64,
    pub forwarded: u64,
    pub revised_before: u64,
    pub taken_after: u64,
}

struct Open {
    row: u8,
    /// Half units at the trigger, and whether the spans stream ran at its fast rate then.
    start: u64,
    fast: bool,
    marked: bool,
    /// A write the stepper took reached the row before, and after, its length was read.
    took_before: bool,
    took_after: bool,
    writes: Vec<[u8; 4]>,
    /// The stepper's last drawing and how many writes it took.
    drawn: Option<(usize, Drawn)>,
}

impl Rows {
    pub fn new(stepper: StepperStream, held: bool, own_memory: u32) -> Rows {
        Rows { stepper, open: None, held, own_memory, compat_closes: 0, forwarded: 0, revised_before: 0, taken_after: 0 }
    }

    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Row `row` triggered at `start` half units: the records kept so far reach every stream
    /// that draws it.
    pub fn open(&mut self, row: u8, start: u64, fast: bool, records: Vec<(u32, u8)>, length: &mut LengthStream, lines: &mut Worker<LineStream>) {
        self.close(length, lines);
        length.catch_up(&records);
        self.stepper.catch_up(&records);
        length.line(row);
        lines.post(move |l| l.catch_up(records));
        self.open = Some(Open { row, start, fast, marked: false, took_before: false, took_after: false, writes: Vec::new(), drawn: None });
    }

    /// An own-memory write `now` half units in: the length stream judges it, and while the
    /// spans stream has yet to read the row's length, a write that leaves the row marked and
    /// joins the queue redraws it, so the length read is the stepper's. A record at the mode
    /// cell kept since the trigger (the mode the rows draw in changed) closed the row first.
    pub fn write(&mut self, index: u8, value: u8, now: u64, records: &VideoRecords, length: &mut LengthStream, lines: &mut Worker<LineStream>) {
        let mode_cell = self.stepper.mode_cell();
        if self.is_open() && records.lock().expect("the record queue").iter().any(|r| r.0 == mode_cell) {
            self.compat_closes += 1;
            self.close(length, lines);
        }
        let Some(o) = self.open.as_mut() else { return };
        let halves = u8::try_from(now - o.start).expect("a write more than a byte of half units into the open row");
        let v = length.judge(halves, index, value);
        o.marked |= v.marks;
        if v.takes {
            o.writes.push([v.time[0], v.time[1], index, value]);
        }
        if o.marked && v.takes {
            let (before, after) = (!length.asked && !o.took_before, length.asked && !o.took_after);
            (o.took_before, o.took_after) = (o.took_before || before, o.took_after || after);
            self.revised_before += before as u64;
            self.taken_after += after as u64;
        }
        if o.marked && v.takes && !length.asked {
            let window = lines.call(|l| l.window());
            let d = self.stepper.draw(o.row, o.fast, self.held, window, &o.writes);
            length.length = d.length;
            o.drawn = Some((o.writes.len(), d));
        }
    }

    /// Closes the open row: the lines stream draws it, or takes the stepper's drawing.
    pub fn close(&mut self, length: &mut LengthStream, lines: &mut Worker<LineStream>) {
        let Some(o) = self.open.take() else { return };
        if !o.marked {
            lines.post(move |l| l.draw(o.row));
            return;
        }
        let d = match o.drawn {
            Some((n, d)) if n == o.writes.len() => d,
            _ => {
                let window = lines.call(|l| l.window());
                self.stepper.draw(o.row, o.fast, self.held, window, &o.writes)
            }
        };
        length.set_window(d.window.1, d.window.2);
        let row = o.row;
        lines.post(move |l| l.draw_stepped(row, d));
    }

    /// The streams held from their memory, or let go, from the next row on.
    pub fn hold(&mut self, held: bool, length: &mut LengthStream, lines: &mut Worker<LineStream>) {
        self.close(length, lines);
        self.held = held;
        lines.post(move |l| l.held = held);
    }

    /// While a row is open, the newest record kept since its trigger at own-memory byte
    /// `index`, which a read takes before the stream's memory.
    pub fn held_record(&mut self, records: &VideoRecords, index: u8) -> Option<u8> {
        let addr = self.own_memory + index as u32;
        self.open.as_ref()?;
        let byte = records.lock().expect("the record queue").iter().rev().find(|r| r.0 == addr).map(|r| r.1);
        self.forwarded += byte.is_some() as u64;
        byte
    }
}
