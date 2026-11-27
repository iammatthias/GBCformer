//! The APU worker: a divergence on its thread that the machine's thread never waited
//! for (it came after the last read) still stops the caller when the worker drops.

use gbcformer::apu_circuit::worker::Worker;
use gbcformer::device::{Device, Signals};
use gbcformer::machine::apu::{AudioDevice, Mix};

/// An APU whose lockstep diverges at its first advance.
struct Diverges;

impl Device for Diverges {
    fn read(&self, _: u16) -> u8 {
        0xFF
    }
    fn write(&mut self, _: u16, _: u8, _: &mut Signals) {}
    fn advance(&mut self, mcycles: u64, _: &mut Signals) {
        panic!("apu lockstep: diverged after {mcycles} M-cycles");
    }
}

impl AudioDevice for Diverges {
    fn sync(&mut self, _: u16) {}
    fn take_samples(&mut self, _: &mut Vec<Mix>) {}
}

#[test]
#[should_panic(expected = "apu lockstep: diverged after 10 M-cycles")]
fn a_divergence_still_queued_when_the_worker_drops_stops_the_caller() {
    let mut w = Worker::spawn(Box::new(Diverges));
    w.advance(10, &mut Signals::default());
    drop(w);
}
