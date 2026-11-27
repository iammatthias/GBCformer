//! SingleStepTests/sm83 oracle (vendor/sm83/v1, MIT). Each file holds 1000 random
//! states for one opcode, with the final state and the M-cycle bus log.

use crate::memory::layout::{self, Access, Banking, Mapper};
use crate::reference::Reference;
use crate::step::{Core, Cpu, Inputs};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    name: String,
    initial: State,
    #[serde(rename = "final")]
    fin: State,
    cycles: Vec<(u16, u8, String)>,
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
    ime: u8,
    #[serde(default)]
    ei: u8,
    ram: Vec<(u16, u8)>,
}

impl State {
    fn cpu(&self) -> Cpu {
        let s = self;
        Cpu {
            a: s.a, b: s.b, c: s.c, d: s.d, e: s.e, f: s.f, h: s.h, l: s.l, sp: s.sp, pc: s.pc,
            ime: s.ime != 0, ei_pending: s.ei != 0, ..Cpu::default()
        }
    }
}

#[derive(Default)]
pub struct Report {
    pub cases: usize,
    pub failures: Vec<String>,
    /// Cases a core could not hold: bus addresses that alias one memory cell.
    pub skipped: Vec<String>,
}

/// HALT and STOP logs describe low-power states this contract models in the host.
fn cycles_comparable(name: &str) -> bool {
    !(name.starts_with("76 ") || name.starts_with("10 "))
}

pub fn run_file(core: &mut impl Core, path: &Path, max_failures: usize) -> Report {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let cases: Vec<Case> = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut report = Report::default();
    let flat = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).expect("a ROM-only board");
    for case in &cases {
        report.cases += 1;
        // A core holds only the bytes the step reads from memory: a byte it only
        // overwrites cannot reach the outcome, an opcode byte fetched from a port reaches
        // it from the host at the port the model emits, and a memory core keeps a
        // constant (the I/O page's unmapped registers, the unusable area) where the flat
        // bus holds any byte, so a case fetching another byte there is skipped.
        let pc = case.initial.pc;
        let host_fetch = |a: u16| (0..3).any(|k| pc.wrapping_add(k) == a) && layout::port(layout::translate(&flat, a, Access::Read, false)).is_some();
        let read: Vec<(u16, u8)> = case.initial.ram.iter().filter(|(a, _)| !host_fetch(*a) && case.cycles.iter().any(|(x, _, k)| x == a && k.contains('r'))).copied().collect();
        if let Err(why) = core.load_bus(&read) {
            report.skipped.push(format!("{}: {why}", case.name));
            continue;
        }
        if let Err(why) = check(core, case) {
            if report.failures.len() < max_failures {
                report.failures.push(format!("{}: {why}", case.name));
            } else {
                report.failures.push(String::new());
            }
        }
    }
    report
}

fn check(core: &mut impl Core, case: &Case) -> Result<(), String> {
    let mut ram: BTreeMap<u16, u8> = case.initial.ram.iter().copied().collect();
    let cpu = case.initial.cpu();
    let peek = |ram: &BTreeMap<u16, u8>, a: u16| ram.get(&a).copied().unwrap_or(0);
    let pc = cpu.pc;
    let inp = Inputs { op: cpu.fetched(|a| peek(&ram, a)), ie: 0, iflag: 0, dma: false, ..Inputs::default() };

    let req = crate::step::request(core, &cpu, &inp, |p| peek(&ram, p.bus_addr()));
    let fe = core.facts().fe_accesses;
    let logged = |kind: &str, addr: u16, value: Option<u8>| {
        case.cycles.iter().any(|(a, v, k)| *a == addr && k.contains(kind) && value.is_none_or(|x| *v == x))
    };
    for addr in req.reads.into_iter().flatten() {
        if !logged("r", addr, None) {
            return Err(format!("read {addr:04x} not in bus log"));
        }
    }
    let served = core.facts().served_by_host;
    let data = std::array::from_fn(|i| if served[i] { req.reads[i].map_or(0, |a| peek(&ram, a)) } else { 0 });
    let out = core.execute(&cpu, &inp, data);
    // SingleStepTests' bus log names the fetch PC on an M-cycle without an access, so the
    // idle FE page addresses and the FE page accesses (a DMG's OAM corruption bug) are held
    // to the reference interpreter's.
    let (r_req, r_out) = (Reference.request(&cpu, &inp), Reference.execute(&cpu, &inp, req.reads.map(|r| r.map_or(0, |a| peek(&ram, a)))));
    if out.idle != r_out.idle {
        return Err(format!("idle FE page M-cycles {:?}, the reference's {:?}", out.idle, r_out.idle));
    }
    let on_fe = |a: Option<u16>| a.is_some_and(|a| a >> 8 == 0xFE);
    // The fetches: the opcode at PC unless the step is a wake, operand k at PC + k − hb.
    let fetches = r_req.cycles.map_or(0, |c| c.fetches) as u16;
    let fetch_at = |k: u16| (k <= fetches && (k > 0 || !cpu.halted)).then(|| cpu.fetch_address(k));
    let r_fe = [on_fe(r_req.reads[0]), on_fe(r_req.reads[1]), on_fe(r_out.writes[0].map(|w| w.0)), on_fe(r_out.writes[1].map(|w| w.0)), on_fe(fetch_at(0)), on_fe(fetch_at(1)), on_fe(fetch_at(2))];
    if !core.facts().reads_the_bus && fe != r_fe {
        return Err(format!("FE page accesses {fe:?}, the reference's {r_fe:?}"));
    }
    if let Some(at) = req.cycles {
        let at_cycle = |kind: char, k: u8, addr: u16| case.cycles.get(k as usize).is_some_and(|(a, _, f)| *a == addr && f.contains(kind));
        for k in 1..=at.fetches {
            if !at_cycle('r', k, pc.wrapping_add(k as u16)) {
                return Err(format!("operand fetch {k} of {:04x} at M-cycle {k}, which the bus log does not place there", pc.wrapping_add(k as u16)));
            }
        }
        for (i, addr) in req.reads.iter().enumerate().filter_map(|(i, r)| Some((i, (*r)?))) {
            if !at_cycle('r', at.reads[i], addr) {
                return Err(format!("read {addr:04x} at M-cycle {}, which the bus log does not place there", at.reads[i]));
            }
        }
        for (i, (addr, _)) in out.writes.iter().enumerate().filter_map(|(i, w)| Some((i, (*w)?))) {
            if !at_cycle('w', at.writes[i], addr) {
                return Err(format!("write {addr:04x} at M-cycle {}, which the bus log does not place there", at.writes[i]));
            }
        }
    }
    for (addr, value) in out.writes.into_iter().flatten() {
        if !logged("w", addr, Some(value)) {
            return Err(format!("write {addr:04x}={value:02x} not in bus log"));
        }
        ram.insert(addr, value);
    }

    let want = case.fin.cpu();
    let got = Cpu { halted: false, halt_bug: false, ..out.cpu };
    if got != want {
        return Err(format!("cpu\n  want {want:?}\n  got  {got:?}"));
    }
    for &(addr, value) in &case.fin.ram {
        if peek(&ram, addr) != value {
            return Err(format!("ram {addr:04x} want {value:02x} got {:02x}", peek(&ram, addr)));
        }
    }
    let writes_logged = case.cycles.iter().filter(|c| c.2.contains('w')).count();
    if out.writes.iter().flatten().count() != writes_logged {
        return Err(format!("{} writes, bus log has {writes_logged}", out.writes.iter().flatten().count()));
    }
    if cycles_comparable(&case.name) && out.mcycles as usize != case.cycles.len() {
        return Err(format!("mcycles want {} got {}", case.cycles.len(), out.mcycles));
    }
    if out.irq_ack != 0 {
        return Err("acknowledged an interrupt with IE=IF=0".into());
    }
    Ok(())
}

/// Runs every opcode file whose name starts with `filter`. Returns (cases, failed cases).
pub fn run_all(core: &mut impl Core, dir: &Path, filter: &str, verbose: bool) -> (usize, usize) {
    let mut files: Vec<_> = std::fs::read_dir(dir).expect("vendor/sm83/v1 missing").flatten().map(|e| e.path()).collect();
    files.sort();
    let (mut cases, mut failed, mut skipped) = (0, 0, Vec::new());
    for path in files.iter().filter(|p| p.file_stem().unwrap().to_string_lossy().starts_with(filter)) {
        let r = run_file(core, path, 3);
        cases += r.cases;
        failed += r.failures.len();
        skipped.extend(r.skipped);
        if !r.failures.is_empty() {
            println!("FAIL {} ({}/{})", path.file_stem().unwrap().to_string_lossy(), r.failures.len(), r.cases);
            if verbose {
                for f in r.failures.iter().filter(|f| !f.is_empty()) {
                    println!("  {f}");
                }
            }
        }
    }
    if !skipped.is_empty() {
        println!("skipped {} cases the core cannot hold (aliased cells, or a byte read where it keeps a constant): {}", skipped.len(), skipped.join("; "));
    }
    (cases, failed)
}
