//! An APU device on a thread of its own. The machine's thread queues writes and
//! advances in batches and keeps running up to [`IN_FLIGHT`] batches ahead; it waits
//! for the worker only where the machine must see the APU at an M-cycle: a CPU read of
//! FF10-FF3F (the read bridge). Samples come back as the worker mixes them, so a frame's
//! audio may arrive with the next frame's; [`AudioDevice::settle`] waits for all of it.

use crate::device::{Device, Signals};
use crate::machine::apu::{AudioDevice, Div, Mix, SAMPLE_CAPACITY};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::JoinHandle;

enum Cmd {
    Write(u16, u8),
    /// `mcycles` M-cycles, at double speed with `double`, each of `rate` halves, a
    /// frame-sequencer step at the last with `div`'s events.
    Advance(u64, bool, u64, crate::machine::apu::Div),
    Sync(u16),
    Settle,
    Report,
}

enum Reply {
    Samples(Vec<Mix>),
    Read(u8),
    Settled,
    Report(Vec<String>),
}

/// A batch goes to the worker when it holds this many commands or M-cycles.
const BATCH_CMDS: usize = 256;
const BATCH_MCYCLES: u64 = 1024;
/// Batches in flight before the machine's thread waits for the worker: about two frames
/// of M-cycles, so the CPU side's busy stretch of one frame overlaps the APU's work on
/// the halted stretch of the one before.
const IN_FLIGHT: usize = 36;

pub struct Worker {
    batch: Vec<Cmd>,
    batch_mcycles: u64,
    to: Option<SyncSender<Vec<Cmd>>>,
    from: Receiver<Reply>,
    thread: Option<JoinHandle<()>>,
    /// Samples the worker sent back and nobody took yet.
    ready: Vec<Mix>,
    read: Option<(u16, u8)>,
    describe: Vec<String>,
}

fn serve(mut device: Box<dyn AudioDevice>, batches: Receiver<Vec<Cmd>>, replies: mpsc::Sender<Reply>) {
    let mut samples = Vec::new();
    for batch in batches {
        for cmd in batch {
            let reply = match cmd {
                Cmd::Write(addr, value) => {
                    device.write(addr, value, &mut Signals::default());
                    continue;
                }
                Cmd::Advance(mcycles, double, rate, div) => {
                    let mut s = Signals { fast: double, rate, edge: div.fall, reset: div.reset, rise: div.rise, ..Signals::default() };
                    device.advance(mcycles, &mut s);
                    assert_eq!(s, Signals { fast: double, rate, edge: div.fall, reset: div.reset, rise: div.rise, ..Signals::default() }, "the APU raised a signal the worker cannot deliver");
                    continue;
                }
                Cmd::Sync(addr) => {
                    device.sync(addr);
                    Reply::Read(device.read(addr))
                }
                Cmd::Settle => Reply::Settled,
                Cmd::Report => Reply::Report(device.report()),
            };
            device.take_samples(&mut samples);
            if !samples.is_empty() && replies.send(Reply::Samples(std::mem::take(&mut samples))).is_err() || replies.send(reply).is_err() {
                return;
            }
        }
        device.take_samples(&mut samples);
        if !samples.is_empty() && replies.send(Reply::Samples(std::mem::take(&mut samples))).is_err() {
            return;
        }
    }
}

impl Worker {
    pub fn spawn(device: Box<dyn AudioDevice>) -> Worker {
        assert_eq!(device.horizon(), u64::MAX, "a worker cannot cut the machine's blocks");
        let mut describe = device.describe();
        describe.push("apu: on its own thread; the CPU's thread waits for it only at FF10-FF3F reads".into());
        let (to, batches) = mpsc::sync_channel(IN_FLIGHT);
        let (replies, from) = mpsc::channel();
        let thread = std::thread::Builder::new().name("apu".into()).spawn(move || {
            crate::qos::interactive();
            serve(device, batches, replies)
        }).expect("spawn the APU thread");
        Worker { batch: Vec::with_capacity(BATCH_CMDS), batch_mcycles: 0, to: Some(to), from, thread: Some(thread), ready: Vec::new(), read: None, describe }
    }

    fn push(&mut self, cmd: Cmd) {
        // Two advances at one speed are one advance when the first ends in no
        // frame-sequencer event.
        if let (Cmd::Advance(m, double, rate, div), Some(Cmd::Advance(pm, pdouble, prate, pdiv))) = (&cmd, self.batch.last_mut())
            && double == pdouble
            && rate == prate
            && *pdiv == crate::machine::apu::Div::default()
        {
            *pm += m;
            *pdiv = *div;
        } else {
            self.batch.push(cmd);
        }
        if self.batch.len() >= BATCH_CMDS || self.batch_mcycles >= BATCH_MCYCLES {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.batch.is_empty() {
            return;
        }
        let batch = std::mem::replace(&mut self.batch, Vec::with_capacity(BATCH_CMDS));
        self.batch_mcycles = 0;
        if self.to.as_ref().expect("a live worker").send(batch).is_err() {
            self.died();
        }
    }

    /// Keeps a batch of samples, dropping what would pass the reference's capacity.
    fn keep(&mut self, samples: Vec<Mix>) {
        let room = SAMPLE_CAPACITY.saturating_sub(self.ready.len());
        self.ready.extend(samples.into_iter().take(room));
    }

    /// Sends `cmd` and everything queued before it, then waits for its reply, keeping
    /// the samples that arrive first.
    fn ask(&mut self, cmd: Cmd) -> Reply {
        self.batch.push(cmd);
        self.flush();
        loop {
            match self.from.recv() {
                Ok(Reply::Samples(s)) => self.keep(s),
                Ok(reply) => return reply,
                Err(_) => self.died(),
            }
        }
    }

    /// The worker hung up: it panicked (a lockstep divergence), so this thread does too.
    fn died(&mut self) -> ! {
        self.to = None;
        match self.thread.take().map(JoinHandle::join) {
            Some(Err(panic)) => std::panic::resume_unwind(panic),
            _ => panic!("the APU thread stopped"),
        }
    }
}

/// The commands still batched run before the worker stops, and a lockstep divergence
/// among any it ran panics here, on the caller's thread, unless that thread is already
/// unwinding.
impl Drop for Worker {
    fn drop(&mut self) {
        let unwinding = std::thread::panicking();
        if let Some(to) = self.to.take()
            && !unwinding
            && !self.batch.is_empty()
        {
            let _ = to.send(std::mem::take(&mut self.batch));
        }
        if let Some(Err(panic)) = self.thread.take().map(JoinHandle::join)
            && !unwinding
        {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Device for Worker {
    fn read(&self, addr: u16) -> u8 {
        match self.read {
            Some((a, v)) if a == addr => v,
            _ => panic!("APU read of {addr:#06x} without a sync at this M-cycle: the bus must call AudioDevice::sync first"),
        }
    }

    fn write(&mut self, addr: u16, value: u8, _s: &mut Signals) {
        self.read = None;
        self.push(Cmd::Write(addr, value));
    }

    fn advance(&mut self, mcycles: u64, s: &mut Signals) {
        self.read = None;
        self.batch_mcycles += mcycles;
        self.push(Cmd::Advance(mcycles, s.fast, s.rate, Div::of(s)));
    }
}

impl AudioDevice for Worker {
    fn sync(&mut self, addr: u16) {
        let Reply::Read(v) = self.ask(Cmd::Sync(addr)) else { unreachable!("a sync replies with its read") };
        self.read = Some((addr, v));
    }

    /// The samples the worker has sent back so far, without waiting for it.
    fn take_samples(&mut self, out: &mut Vec<Mix>) {
        loop {
            match self.from.try_recv() {
                Ok(Reply::Samples(s)) => self.keep(s),
                Ok(_) => unreachable!("only samples arrive unasked"),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => self.died(),
            }
        }
        out.append(&mut self.ready);
    }

    fn settle(&mut self) {
        let Reply::Settled = self.ask(Cmd::Settle) else { unreachable!("a settle replies settled") };
    }

    fn describe(&self) -> Vec<String> {
        self.describe.clone()
    }

    fn report(&mut self) -> Vec<String> {
        let Reply::Report(lines) = self.ask(Cmd::Report) else { unreachable!("a report replies with lines") };
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apu_circuit::{ApuLockstep, ApuModel, CircuitSegments, build};
    use crate::machine::apu::Apu;

    /// The circuit on a worker, in lockstep with a post-boot DMG reference; the model
    /// starts from `from`.
    fn lockstep_worker(mut from: Apu) -> Worker {
        let model = ApuModel::new(CircuitSegments(build()), &mut from);
        Worker::spawn(Box::new(ApuLockstep::new(Apu::post_boot(false), model)))
    }

    #[test]
    fn a_worker_keeps_writes_reads_and_samples_in_order() {
        let (mut w, mut reference) = (lockstep_worker(Apu::post_boot(false)), Apu::post_boot(false));
        let mut s = Signals { rate: 2, ..Signals::default() };
        for (a, v) in [(0xFF24, 0x77), (0xFF25, 0xFF), (0xFF17, 0xF0), (0xFF16, 0x80), (0xFF19, 0x87)] {
            w.write(a, v, &mut s);
            reference.write(a, v);
        }
        for k in 0..100 {
            s.reset(false);
            s.edge = k % 10 == 9;
            w.advance(21, &mut s);
            Device::advance(&mut reference, 21, &mut s);
        }
        w.sync(0xFF26);
        assert_eq!(w.read(0xFF26), 0xF3, "power, channels 1 and 2 on");
        let (mut got, mut want) = (Vec::new(), Vec::new());
        w.settle();
        w.take_samples(&mut got);
        reference.take_samples(&mut want);
        assert_eq!((got.len(), got == want), (96, true));
        assert!(got.iter().any(|s| s[0] != 0), "channel 2 is audible");
        assert!(w.report().iter().any(|l| l.contains("passes equal to the reference")));
    }

    #[test]
    #[should_panic(expected = "apu lockstep: state diverged")]
    fn a_divergence_on_the_worker_stops_the_caller() {
        let mut other = Apu::post_boot(false);
        other.write(0xFF24, 0x12);
        let mut w = lockstep_worker(other);
        w.advance(30, &mut Signals { rate: 2, ..Signals::default() });
        w.settle();
    }
}
