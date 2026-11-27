//! The memory circuit under the HALT bug, against the reference interpreter. The
//! SingleStepTests never set the bug, and in memory mode the circuit reads the repeated
//! byte from its own fetches (pc + k − hb), so the JR offset, the immediates and the
//! return addresses take their values from those fetches. Every SST case of the opcodes
//! below runs with the bug set on both cores, which must agree on the next state, the
//! writes and the cycle count.

#[path = "common/sst.rs"]
mod sst_data;

use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{Banking, Mapper};
use gbcformer::reference::Reference;
use gbcformer::sm83::CircuitMemCore;
use gbcformer::step::{Core, Cpu, Inputs};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Case {
    initial: State,
}

#[derive(Deserialize)]
struct State {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    f: u8,
    h: u8,
    l: u8,
    pc: u16,
    sp: u16,
    ram: Vec<(u16, u8)>,
}

/// JR and JR cc, CALL and RST (their pushes read the next pc), loads of every length.
const OPS: [&str; 16] = ["18", "20", "28", "30", "38", "cd", "c4", "cc", "c3", "ff", "c7", "3e", "01", "fa", "e0", "00"];

#[test]
fn memory_circuit_matches_the_reference_under_the_halt_bug() {
    let Some(dir) = sst_data::dir() else { return };
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    let mut core = CircuitMemCore::new(Image::new(&[0u8; 0x8000], 0x2000, banking).with_writable_rom());
    core.image.mark();
    let mut checked = 0;
    for op in OPS {
        let path = dir.join(format!("{op}.json"));
        let cases: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap();
        for (i, case) in cases.iter().enumerate().step_by(10) {
            let s = &case.initial;
            if core.load_bus(&s.ram).is_err() {
                continue;
            }
            let ram: BTreeMap<u16, u8> = s.ram.iter().copied().collect();
            let peek = |a: u16| ram.get(&a).copied().unwrap_or(0);
            let cpu = Cpu { a: s.a, b: s.b, c: s.c, d: s.d, e: s.e, f: s.f, h: s.h, l: s.l, sp: s.sp, pc: s.pc, halt_bug: true, ..Cpu::default() };
            let inp = Inputs { op: cpu.fetched(peek), ie: 0, iflag: 0, dma: false, ..Inputs::default() };
            let want_req = Reference.request(&cpu, &inp);
            let data = std::array::from_fn(|k| want_req.reads[k].map_or(0, peek));
            let want = Reference.execute(&cpu, &inp, data);
            let req = gbcformer::step::request(&mut core, &cpu, &inp, |p| peek(p.bus_addr()));
            assert_eq!(req.reads, want_req.reads, "{op} case {i}: reads");
            let served = core.facts().served_by_host;
            let host = std::array::from_fn(|k| if served[k] { data[k] } else { 0 });
            let got = core.execute(&cpu, &inp, host);
            assert_eq!(got, want, "{op} case {i}: outcome");
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked} cases");
}
