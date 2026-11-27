//! The compiled SM83 circuit against SingleStepTests, and its structure.

use gbcformer::sm83::{build, CircuitCore};
use gbcformer::sst;
#[path = "common/sst.rs"]
mod sst_data;

#[test]
fn circuit_matches_singlesteptests() {
    let Some(dir) = sst_data::dir() else { return };
    let c = build();
    eprintln!("wires {} nodes {} depth {}", c.levels.len(), c.nodes.len(), c.depth());
    let filter = std::env::var("SST").unwrap_or_default();
    let (cases, failed) = sst::run_all(&mut CircuitCore(c), dir, &filter, true);
    eprintln!("circuit: {cases} cases, {failed} failed");
    assert_eq!(failed, 0);
}

/// The harness must reject a core that is wrong in one bit of one register.
#[test]
fn harness_rejects_a_mutant() {
    let Some(dir) = sst_data::dir() else { return };
    use gbcformer::step::{Core, Cpu, Inputs, Outcome, Request};
    struct Mutant(CircuitCore);
    impl Core for Mutant {
        fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
            self.0.request(cpu, inp)
        }
        fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
            let mut o = self.0.execute(cpu, inp, data);
            if inp.op[0] == 0x80 && o.cpu.a == 0 {
                o.cpu.f ^= 0x80;
            }
            o
        }
    }
    let (cases, failed) = sst::run_all(&mut Mutant(CircuitCore(build())), dir, "8", false);
    eprintln!("mutant: {cases} cases, {failed} failed");
    assert!(failed > 0 && failed < cases, "{failed} of {cases}");
}

/// The memory circuit over its own image: SingleStepTests' RAM is loaded through the
/// fixed layout of a 32 KiB ROM-only cartridge with 8 KiB of RAM, so every bus
/// address but the I/O page and IE is served by the model.
#[test]
fn memory_circuit_matches_singlesteptests() {
    let Some(dir) = sst_data::dir() else { return };
    use gbcformer::memory::image::Image;
    use gbcformer::memory::layout::{Banking, Mapper};
    use gbcformer::sm83::CircuitMemCore;
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    let mut core = CircuitMemCore::new(Image::new(&[0u8; 0x8000], 0x2000, banking).with_writable_rom());
    core.image.mark();
    let c = &core.circuit;
    eprintln!("memory circuit: wires {} nodes {} depth {} heads {}", c.levels.len(), c.nodes.len(), c.depth(), c.heads.len());
    let filter = std::env::var("SST").unwrap_or_default();
    let (cases, failed) = sst::run_all(&mut core, dir, &filter, true);
    eprintln!("memory circuit: {cases} cases, {failed} failed; {} of {} steps host-served", core.host_served_steps, core.steps);
    assert_eq!(failed, 0);
}

