//! Runs a candidate beside its reference and stops at the first disagreement, so a
//! divergence is caught at the instruction that caused it. [`Lockstep`] pairs CPU
//! cores; each checkpoint device holds its own lockstep at its pass boundaries.

use crate::memory::{Addr, Copied};
use crate::step::{slot_state, Bus, Core, Cpu, Inputs, Outcome, Request, StepFacts};

pub struct Lockstep<R: Core, M: Core> {
    pub reference: R,
    pub model: M,
    pub steps: u64,
    /// The reference's request while the model waits for the host's fetched bytes.
    waiting: Option<Request>,
    /// The reads the step requested, which the reference observes.
    reads: [Option<u16>; 2],
}

impl<R: Core, M: Core> Lockstep<R, M> {
    pub fn new(reference: R, model: M) -> Self {
        Lockstep { reference, model, steps: 0, waiting: None, reads: [None; 2] }
    }
}

impl<R: Core, M: Core> Core for Lockstep<R, M> {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        let (want, got) = (self.reference.request(cpu, inp), self.model.request(cpu, inp));
        if self.model.facts().fetch_ports.iter().any(Option::is_some) {
            self.waiting = Some(want);
            return got;
        }
        assert_eq!(got.reads, want.reads, "request diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}", self.steps);
        assert_eq!(got.cycles, want.cycles, "access cycles diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}", self.steps);
        self.reads = got.reads;
        got
    }

    /// The model's: the reference reads every byte from the bus, and a byte the model's
    /// host serves comes from the device its address names, which the reference reads too.
    /// The audit holds the model's FE page accesses to the bus (`Core::audit`).
    fn facts(&self) -> &StepFacts {
        self.model.facts()
    }

    fn serve_fetch(&mut self, cpu: &Cpu, inp: &Inputs, bytes: [u8; 3]) -> Request {
        let want = self.waiting.take().expect("the reference's request waits beside the model's");
        let got = self.model.serve_fetch(cpu, inp, bytes);
        assert_eq!(got.reads, want.reads, "request after the host's fetched bytes diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}", self.steps);
        assert_eq!(got.cycles, want.cycles, "access cycles after the host's fetched bytes diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}", self.steps);
        self.reads = got.reads;
        got
    }

    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        let (want, got) = (self.reference.execute(cpu, inp, data), self.model.execute(cpu, inp, data));
        assert_eq!(got, want, "outcome diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}\n  data {data:?}", self.steps);
        self.steps += 1;
        got
    }

    /// The host serves the reads the model asks it for, as the model asked. The reference
    /// observes every other byte it requested through the bus's side-effect-free peek,
    /// under that read's own bus state, so lockstep changes nothing the devices do (no
    /// extra control pass at a read OAM DMA blocks).
    fn execute_on(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2], bus: &impl Bus) -> Outcome {
        let served = self.model.facts().served_by_host;
        let observed: [u8; 2] = std::array::from_fn(|i| match self.reads[i] {
            Some(a) if !served[i] => bus.peek(a, slot_state(inp, i)).unwrap_or_else(|| panic!("the reference's read {i} of {a:#06x} has no side-effect-free byte on this bus")),
            _ => data[i],
        });
        let (want, got) = (self.reference.execute(cpu, inp, observed), self.model.execute_on(cpu, inp, data, bus));
        assert_eq!(got, want, "outcome diverged at step {}\n  cpu {cpu:?}\n  inputs {inp:?}\n  data {data:?}, observed {observed:?}", self.steps);
        self.steps += 1;
        got
    }

    fn land_write(&mut self, i: usize) {
        self.model.land_write(i);
    }

    fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        self.reference.load_bus(ram)?;
        self.model.load_bus(ram)
    }

    fn audit(&self, bus: &impl Bus, cpu: &Cpu, inp: &Inputs, out: &Outcome) {
        self.model.audit(bus, cpu, inp, out);
    }

    fn copied(&mut self, bytes: &[Copied]) {
        self.model.copied(bytes);
    }

    fn audit_memory(&self, host: &[(Addr, &[u8])]) {
        self.model.audit_memory(host);
    }

    fn oam_bug(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; 8])> {
        self.model.oam_bug(row, read)
    }
}
