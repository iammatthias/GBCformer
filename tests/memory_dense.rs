//! The memory checkpoint under the dense pass with a KV cache (the stock-stack oracle)
//! against the memory circuit evaluated directly, on random CPU steps: same request,
//! same outcome, same memory evidence, and every kept record's key and value equal to
//! the formula `memory::attn::key` (checked as `DenseMemory::push` stores the cell).

use gbcformer::llama::{KvCache, Model};
use gbcformer::memory::attn::Params;
use gbcformer::memory::dense::DenseMemory;
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{self, Access, Banking, Mapper};
use gbcformer::sm83::{build_mem, CircuitMemCore, MemCheckpointCore};
use gbcformer::step::{Core, Cpu, Inputs};
use std::collections::{BTreeMap, BTreeSet};

mod common;
use common::Rng;

/// Each line's bytes by line number.
type Lines = BTreeMap<u32, [u8; layout::LINE]>;

impl Rng {
    fn word(&mut self) -> u16 {
        self.next() as u16
    }
}

#[test]
fn dense_pass_with_cache_matches_the_circuit_on_random_steps() {
    let params = Params::memory();
    let model = Model::compile_mem(&build_mem(), params);
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    let mut circuit = CircuitMemCore::new(Image::new(&[0; 0x8000], 0x2000, banking).with_writable_rom());
    circuit.image.mark();
    let mut dense = MemCheckpointCore::new(DenseMemory::new(model));
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let cases: usize = std::env::var("CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let (mut kept, mut host_served) = (0, 0);
    for case in 0..cases {
        let (pc, sp) = (rng.word(), rng.word());
        let cpu = Cpu {
            a: rng.byte(),
            f: rng.byte() & 0xF0,
            b: rng.byte(),
            c: rng.byte(),
            d: rng.byte(),
            e: rng.byte(),
            h: rng.byte(),
            l: rng.byte(),
            sp,
            pc,
            ime: rng.next() & 1 == 1,
            ei_pending: rng.next() % 5 == 0,
            halted: false,
            halt_bug: case % 7 == 3,
        };
        // Half the steps draw from instructions that write, so records are exercised.
        const WRITERS: [u8; 12] = [0x77, 0x36, 0x22, 0x32, 0xC5, 0xF5, 0xCD, 0xC7, 0xEA, 0xE0, 0x08, 0x34];
        let op0 = if case % 2 == 0 { WRITERS[(rng.next() % 12) as usize] } else { rng.byte() };
        let op = [op0, rng.byte(), rng.byte()];
        let hl = (cpu.h as u16) << 8 | cpu.l as u16;
        let (bc, de) = ((cpu.b as u16) << 8 | cpu.c as u16, (cpu.d as u16) << 8 | cpu.e as u16);
        let imm16 = op[1] as u16 | (op[2] as u16) << 8;
        let mut ram: Vec<(u16, u8)> = vec![(pc, op[0]), (pc.wrapping_add(1), op[1]), (pc.wrapping_add(2), op[2])];
        for a in [hl, sp, sp.wrapping_add(1), bc, de, imm16, 0xFF00 | op[1] as u16, 0xFF00 | cpu.c as u16] {
            ram.push((a, rng.byte()));
        }
        let peek = |a: u16| ram.iter().find(|(x, _)| *x == a).map_or(0, |(_, v)| *v);
        let inp = Inputs { op: cpu.fetched(peek), ie: rng.byte(), iflag: rng.byte() & 0x1F, dma: false, ..Inputs::default() };
        if circuit.load_bus(&ram).is_err() {
            continue;
        }
        let r1 = gbcformer::step::request(&mut circuit, &cpu, &inp, |p| peek(p.bus_addr()));
        let served = circuit.facts().served_by_host;
        let data: [u8; 2] = std::array::from_fn(|i| if served[i] { r1.reads[i].map_or(0, peek) } else { 0 });
        let o1 = circuit.execute(&cpu, &inp, data);
        let e1 = circuit.evidence;
        host_served += served.iter().filter(|&&s| s).count();

        // The dense model's cache holds the tokens this step touches: its fetches, its two
        // read addresses, the registers, and ROM bytes 0..2 (a record group fetches PC 0).
        circuit.load_bus(&ram).unwrap();
        let b = circuit.image.banking();
        let mut addrs: BTreeSet<u32> = [0, 1, 2, e1.read_addr[0], e1.read_addr[1], e1.write_addr[0], e1.write_addr[1]].into_iter().collect();
        for k in 0..3u16 {
            addrs.insert(layout::translate(&b, cpu.fetch_address(k), Access::Read, false));
        }
        for r in 0..layout::N_REGS as u32 {
            addrs.insert(layout::REGS + r);
        }
        let lines: BTreeSet<u32> = addrs.iter().map(|&p| p >> layout::LINE_BITS).collect();
        let dm = dense.model_mut();
        dm.cache = KvCache::new(dm.cache.heads.clone());
        // VOID's line goes first because an off column reads it, and `DenseMemory::push`
        // refuses a record decoded before it.
        let void = layout::VOID >> layout::LINE_BITS;
        dm.push(layout::VOID, circuit.image.line(void));
        for &n in &lines {
            dm.push(n << layout::LINE_BITS, circuit.image.line(n));
            assert_eq!(dm.cache.get(n), Some(circuit.image.line(n)), "cell of line {n:#07x}");
        }
        let r2 = gbcformer::step::request(&mut dense, &cpu, &inp, |p| peek(p.bus_addr()));
        assert_eq!(r2, r1, "request, case {case}: cpu {cpu:?} op {op:02x?}");
        assert_eq!(dense.facts().served_by_host, served, "host flags, case {case}");
        let o2 = dense.execute(&cpu, &inp, data);
        assert_eq!(o2, o1, "outcome, case {case}: cpu {cpu:?} op {op:02x?}");
        assert_eq!(dense.evidence, e1, "evidence, case {case}");
        // `DenseMemory::push` checked each record's key and value against the formula as
        // it kept the record.
        let cache = &dense.model().cache;
        let mut cells = lines.len() + !lines.contains(&void) as usize;
        let recs: Vec<_> = e1.records.into_iter().flatten().collect();
        for (i, rec) in recs.iter().enumerate() {
            let n = rec.addr >> layout::LINE_BITS;
            // When both writes share a line, write 1's record holds both and replaces write 0's.
            if recs[i + 1..].iter().all(|r| r.addr >> layout::LINE_BITS != n) {
                assert_eq!(cache.get(n), Some(rec.line), "kept cell, case {case}, record {rec:?}");
            }
            cells += !lines.contains(&n) as usize;
            kept += 1;
        }
        assert!(cache.len() <= cells, "nothing else kept, case {case}");
        eprintln!("case {case}: op {op:02x?} reads {:?} records {:?}", r1.reads, e1.records);
    }
    eprintln!("{cases} steps: {kept} records kept, {host_served} host-served reads");
    assert!(kept > 0, "no step wrote anything");
}

/// Copy mode. A reader record's `RD0..RD7` decode to the destination's line with the
/// source's byte over byte `dest mod 8`, which the two copy heads read from the model's
/// own cache. A read is a reader whose source is its destination.
#[test]
fn reader_records_read_the_cache_through_the_model() {
    let params = Params::memory();
    let mut dm = DenseMemory::new(Model::compile_mem(&build_mem(), params));
    let mut rng = Rng(0x0C0F_FEE0_1234_5678);
    let mut lines: Lines = Default::default();
    let put = |lines: &mut Lines, p: u32, v: u8| lines.entry(p >> layout::LINE_BITS).or_default()[p as usize % layout::LINE] = v;
    // The banking registers must be present and consistent, because the circuit runs on
    // the record too and its translations assume one mapper and masked banks. The VOID
    // block's first line must be present as well, because the data heads land there when
    // no read is selected and a query must never name a line without a cell.
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    for r in 0..layout::N_REGS as u32 {
        put(&mut lines, layout::REGS + r, banking.0[r as usize]);
    }
    put(&mut lines, layout::VOID, 0);
    // ROM past 2 MiB and past 4 MiB: line key bits 18 and 19, which only an 8 MiB
    // cartridge reaches and mooneye's mbc5 ROMs read only the first of.
    let probes = [0u32, 1, 2, 0x4000, 0x20_0005, 0x40_0003, 0x7F_FFFE, layout::WRAM + 0x1234, layout::VRAM + 7, layout::HRAM + 3];
    for p in probes {
        put(&mut lines, p, rng.byte());
    }
    // VOID's line goes first because a column head off its line reads it, on the
    // records' own positions too (`docs/hardware-shaped.md` §20). Every later record
    // then finds its cell.
    let void = layout::VOID >> layout::LINE_BITS;
    dm.push(layout::VOID, lines[&void]);
    for (&n, &line) in lines.iter().filter(|(n, _)| **n != void) {
        dm.push(n << layout::LINE_BITS, line);
    }
    let byte = |lines: &Lines, p: u32| lines[&(p >> layout::LINE_BITS)][p as usize % layout::LINE];
    for p in probes.iter().copied().chain([layout::REGS, layout::VOID]) {
        assert_eq!(dm.read(p), byte(&lines, p), "reader at {p:#08x}");
    }
    // A copy gives the destination's line with the source's byte over one byte.
    let (dest, source) = (layout::WRAM + 0x1236, layout::VRAM + 7);
    let mut want = lines[&(dest >> layout::LINE_BITS)];
    want[dest as usize % layout::LINE] = byte(&lines, source);
    assert_eq!(dm.reader(dest, source), want, "copy of {source:#08x} into {dest:#08x}");
    // Keeping the copy replaces the line's cell.
    dm.push(dest, want);
    put(&mut lines, dest, want[dest as usize % layout::LINE]);
    assert_eq!(dm.read(dest), byte(&lines, dest));
    assert_eq!(dm.read(layout::WRAM + 0x1234), byte(&lines, layout::WRAM + 0x1234), "the rest of the line is kept");
}
