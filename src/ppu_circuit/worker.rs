//! The PPU checkpoint on a thread of its own. The machine's thread hands over the
//! visible lines together with the VRAM, OAM and display records kept before each, in
//! the order it kept them, a batch of lines at a time, and runs on. VBlank only asks for
//! the frame; the machine's thread waits for it when the frame is read
//! ([`Renderer::front`]), at a display byte read, for the byte, when the control logic needs
//! a line's mode-3 length (`Lengths::latest`), for the lines queued and any revision the
//! stepper owes, and at the open after a line the stepper drew, for the window it left. A
//! lockstep divergence panics on the worker and the machine's thread re-raises it at its
//! next send or wait.

use super::{held_display, keep_display, take, LengthModel, PpuModel};
use crate::engine::VideoRecords;
use crate::machine::fifo;
use crate::machine::ppu::{Drawn, Frame, Lengths, Line, LineWrite, Renderer, WIDTH};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Lines handed over per wake of the worker: each send wakes it, which costs the
/// machine's thread more than queueing does.
const BATCH: usize = 8;

enum Cmd {
    Begin(Box<Frame>),
    /// Records then lines, in the order the machine's thread kept them.
    Batch(Vec<Item>),
    Frame,
    Display(u8, Option<u8>),
    /// The window after the lines drawn so far.
    Window,
    Report,
}

type Writes = Vec<(fifo::Time, u8, u8)>;

enum Item {
    Records(Vec<(u32, u8)>),
    /// A line, the reference's drawing under lockstep, and the length model's length.
    Line(Line, Option<Box<Drawn>>, Option<u8>),
    /// A line the judge marked, the stepper's queue of writes, and the reference's drawing.
    Stepped(Line, Writes, Option<Box<Drawn>>),
    /// The open line's length with the writes so far, for the control logic, and under
    /// lockstep the reference's length by then.
    Revise(Line, Writes, Option<u8>),
}

enum Reply {
    Frame(Box<Frame>),
    Display(u8),
    Window(super::Window),
    Report(Vec<String>),
}

/// The lines and records not yet handed over. The control logic's wait for a mode-3
/// length hands them over too (`Lengths::set_flush`), so it is shared.
struct Queue {
    /// Dropped with the worker, so the channel closes though a board keeps the queue.
    to: Option<Sender<Cmd>>,
    records: VideoRecords,
    batch: Vec<Item>,
    lines: usize,
    /// The length model, which every record handed over reaches too, and which judges each
    /// display write to the open line.
    length: LengthModel,
    lengths: Lengths,
    lockstep: bool,
    /// A line opened and not handed over yet: records kept since wait for it.
    cut: bool,
    /// The open line's length and arming from the length model.
    measured: Option<(u8, super::Window)>,
    /// The last line handed over was the stepper's.
    stepped_last: bool,
}

impl Queue {
    /// The records kept since the last hand-over join the batch.
    fn keep_records(&mut self) {
        if self.cut {
            return;
        }
        let records = take(&self.records);
        self.length.write(&records);
        if !records.is_empty() {
            self.batch.push(Item::Records(records));
        }
    }

    /// Everything kept so far, to the worker, with the revision of the open line's length
    /// the judge made due; false when the worker hung up.
    fn flush(&mut self) -> bool {
        self.keep_records();
        let reference = self.lockstep.then(|| self.lengths.current());
        if let Some(j) = self.length.judging().filter(|j| j.revise) {
            j.revise = false;
            self.batch.push(Item::Revise(j.line, j.writes.clone(), reference));
        }
        self.lines = 0;
        self.batch.is_empty() || self.to.as_ref().is_some_and(|to| to.send(Cmd::Batch(std::mem::take(&mut self.batch))).is_ok())
    }
}

pub struct Worker {
    to: Option<Sender<Cmd>>,
    from: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
    queue: Arc<Mutex<Queue>>,
    records: VideoRecords,
    /// Frames asked for and not yet received, and the newest received.
    frames_pending: usize,
    front: Box<Frame>,
    lockstep: bool,
    describe: Vec<String>,
    /// Display byte reads, and the time this thread waited for them (the queue drained first).
    display_waits: (u64, u64),
    /// Display reads served from the record queue while a line was open.
    forwarded: u64,
}

fn serve(mut model: PpuModel, cmds: Receiver<Cmd>, replies: Sender<Reply>) {
    crate::qos::interactive();
    let mut frame = Box::new([0u32; WIDTH * crate::machine::ppu::HEIGHT]);
    for cmd in cmds {
        let reply = match cmd {
            Cmd::Begin(back) => {
                model.back.copy_from_slice(&back[..]);
                continue;
            }
            Cmd::Batch(items) => {
                for item in items {
                    match item {
                        Item::Records(r) => model.write(&r),
                        Item::Line(line, reference, expected) => model.draw(&line, reference.as_deref(), expected),
                        Item::Stepped(line, writes, reference) => model.draw_stepped(&line, &writes, reference.as_deref()),
                        Item::Revise(line, writes, reference) => {
                            let length = model.revise(&line, &writes, reference);
                            model.lengths.publish(length);
                        }
                    }
                }
                continue;
            }
            Cmd::Frame => {
                model.frame(&mut frame);
                Reply::Frame(frame.clone())
            }
            Cmd::Display(index, reference) => Reply::Display(model.display(index, reference)),
            Cmd::Window => Reply::Window(model.window()),
            Cmd::Report => Reply::Report(model.report()),
        };
        if replies.send(reply).is_err() {
            return;
        }
    }
}

impl Worker {
    pub fn spawn(model: PpuModel, records: VideoRecords, length: LengthModel) -> Worker {
        let lengths = model.lengths.clone();
        let lockstep = model.lockstep;
        let mut describe = model.describe();
        describe.extend(length.describe());
        describe.push(format!("ppu: on its own thread, handed {} lines a wake; the CPU's thread waits for it only when the frame is read, at a display byte read, for a revised length and after a line the stepper drew", BATCH));
        let (to, cmds) = mpsc::channel();
        let (replies, from) = mpsc::channel();
        let thread = std::thread::Builder::new().name("ppu".into()).spawn(move || serve(model, cmds, replies)).expect("spawn the PPU thread");
        let front = Box::new([0; WIDTH * crate::machine::ppu::HEIGHT]);
        let queue = Queue { to: Some(to.clone()), records: records.clone(), batch: Vec::new(), lines: 0, length, lengths: lengths.clone(), lockstep, cut: false, measured: None, stepped_last: false };
        let queue = Arc::new(Mutex::new(queue));
        let shared = queue.clone();
        lengths.set_flush(Box::new(move || assert!(shared.lock().expect("the PPU queue").flush(), "the PPU thread stopped")));
        Worker { to: Some(to), from, thread: Some(thread), queue, records, frames_pending: 0, front, lockstep, describe, display_waits: (0, 0), forwarded: 0 }
    }

    fn send(&mut self, cmd: Cmd) {
        if self.to.as_ref().expect("a live worker").send(cmd).is_err() {
            self.died();
        }
    }

    fn receive(&mut self) -> Reply {
        match self.from.recv() {
            Ok(reply) => reply,
            Err(_) => self.died(),
        }
    }

    /// Everything kept so far, to the worker.
    fn flush(&mut self) {
        if !self.queue.lock().expect("the PPU queue").flush() {
            self.died();
        }
    }

    /// Every frame asked for, received; the newest is the front.
    fn settle_frames(&mut self) {
        while self.frames_pending > 0 {
            let Reply::Frame(f) = self.receive() else { unreachable!("frames reply first, in order") };
            self.front = f;
            self.frames_pending -= 1;
        }
    }

    /// Asks the worker and waits for its answer, after every frame asked for before.
    fn ask(&mut self, cmd: Cmd) -> Reply {
        self.flush();
        self.send(cmd);
        self.settle_frames();
        self.receive()
    }

    /// The worker hung up: it panicked (a lockstep divergence), so this thread does too.
    fn died(&mut self) -> ! {
        self.to = None;
        match self.thread.take().map(JoinHandle::join) {
            Some(Err(panic)) => std::panic::resume_unwind(panic),
            _ => panic!("the PPU thread stopped"),
        }
    }
}

/// The lines still batched are drawn before the worker stops, and a lockstep divergence
/// among any lines it drew panics here, on the caller's thread, unless that thread is
/// already unwinding.
impl Drop for Worker {
    fn drop(&mut self) {
        let unwinding = std::thread::panicking();
        if let Some(to) = self.to.take()
            && !unwinding
        {
            let mut q = self.queue.lock().expect("the PPU queue");
            q.keep_records();
            let _ = to.send(Cmd::Batch(std::mem::take(&mut q.batch)));
        }
        self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).to = None;
        if let Some(Err(panic)) = self.thread.take().map(JoinHandle::join)
            && !unwinding
        {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Renderer for Worker {
    fn begin(&mut self, back: &Frame) {
        self.send(Cmd::Begin(Box::new(*back)));
    }

    fn open(&mut self, line: Line) {
        // The length stream does not draw a stepped line: the next line's pass starts from
        // the window the stepper left.
        if self.queue.lock().expect("the PPU queue").stepped_last {
            let Reply::Window(window) = self.ask(Cmd::Window) else { unreachable!("a window ask replies with the window") };
            self.queue.lock().expect("the PPU queue").length.set_window(window);
        }
        let lockstep = self.lockstep;
        let mut q = self.queue.lock().expect("the PPU queue");
        q.keep_records();
        q.cut = true;
        q.measured = Some(q.length.measure(&line, lockstep));
    }

    fn line(&mut self, line: Line, reference: Option<Box<Drawn>>) {
        // The lockstep check runs outside the queue's lock: a divergence panics there, and a
        // poisoned lock would keep `Drop` from closing the channel.
        let (measured, judging) = {
            let mut q = self.queue.lock().expect("the PPU queue");
            q.cut = false;
            let judging = q.length.close().filter(|j| j.marked);
            q.stepped_last = judging.is_some();
            (q.measured.take(), judging)
        };
        let item = match judging {
            Some(j) => {
                // A revision the judge made due and no flush handed over yet: the line closed
                // before the control logic read its length (LCD off, STOP, VBlank), and the
                // board still counts it, so it is owed before the line.
                if j.revise {
                    let r = self.lockstep.then(|| self.queue.lock().expect("the PPU queue").lengths.current());
                    self.queue.lock().expect("the PPU queue").batch.push(Item::Revise(j.line, j.writes.clone(), r));
                }
                Item::Stepped(line, j.writes, reference)
            }
            None => {
                let expected = measured.map(|m| LengthModel::check(&line, m, reference.as_deref()));
                Item::Line(line, reference, expected)
            }
        };
        let full = {
            let mut q = self.queue.lock().expect("the PPU queue");
            q.batch.push(item);
            q.lines += 1;
            q.lines == BATCH
        };
        if full {
            self.flush();
        }
    }

    fn end_frame(&mut self) {
        // Frames already drawn are taken without waiting, so unread ones do not pile up.
        while let Ok(Reply::Frame(f)) = self.from.try_recv() {
            self.front = f;
            self.frames_pending -= 1;
        }
        self.flush();
        self.send(Cmd::Frame);
        self.frames_pending += 1;
    }

    fn front(&mut self, out: &mut Frame) {
        self.settle_frames();
        out.copy_from_slice(&self.front[..]);
    }

    fn display_write(&mut self, index: u8, value: u8, open: Option<LineWrite>) {
        keep_display(&self.records, index, value);
        if let Some(w) = open {
            // A revision the judge made due is handed over at the next flush, which the
            // control logic's read of the length makes at the latest.
            self.queue.lock().expect("the PPU queue").length.judge(index, value, w);
        }
    }

    fn display_read(&mut self, index: u8, reference: Option<u8>) -> u8 {
        if self.queue.lock().expect("the PPU queue").cut
            && let Some(byte) = held_display(&self.records, index, reference)
        {
            self.forwarded += 1;
            return byte;
        }
        let t = std::time::Instant::now();
        let Reply::Display(byte) = self.ask(Cmd::Display(index, reference)) else { unreachable!("a display read replies with its byte") };
        self.display_waits.0 += 1;
        self.display_waits.1 += t.elapsed().as_nanos() as u64;
        byte
    }

    fn lockstep(&self) -> bool {
        self.lockstep
    }

    fn describe(&self) -> Vec<String> {
        self.describe.clone()
    }

    fn report(&mut self) -> Vec<String> {
        let Reply::Report(mut lines) = self.ask(Cmd::Report) else { unreachable!("a report replies with lines") };
        let (reads, nanos) = self.display_waits;
        lines.push(format!("ppu: the CPU's thread waited {:.2} ms for {reads} display byte reads, the lines queued before each drawn first", nanos as f64 / 1e6));
        lines.extend(self.queue.lock().expect("the PPU queue").length.report());
        lines
    }

    fn stepped(&self) -> u64 {
        self.queue.lock().expect("the PPU queue").length.marked()
    }

    fn rows(&self) -> [u64; 3] {
        let [before, after] = self.queue.lock().expect("the PPU queue").length.rows();
        [self.forwarded, before, after]
    }
}
