//! The lines a stream's decoded rows can name (`docs/stock-llama.md` §12a, "Scoped
//! attention"). A memory head gives exactly zero weight to every cell but the one its
//! query names, so a decoded row attends the same cells, with the same weights, in any
//! sequence that holds every line its heads can name. `Circuit::reach` bounds those
//! lines from the merged circuit. It leaves the stream's own inputs open, holds every
//! other part's inputs at byte 0 (as the stream's rows see them, `one_model::group`),
//! fixes the cartridge's ROM and the register bytes written only at load, and leaves
//! every other byte open.
//!
//! The cache guards both fixed sources where it writes. No write record lands in ROM,
//! and none changes `MAPPER`, `MASK0`, `RAMMASK` or `BOOTING` (`Fixed::guard`).
//!
//! `BOOTING` must be among them. The GBC CPU's fetch reads it to choose between the boot
//! ROM and the cartridge, so with it open the opcode at the held PC is unknown, the data
//! and write heads can name any line, and a scope that holds every part's heads' lines
//! is the whole cache. `CELLS` refuses such a scope.
//!
//! In an overlay a stream's scope holds only its own heads' lines (`own`): every other
//! part's head attends the null cell at the stream's tokens.
//!
//! The GBC CPU's own heads translate bus addresses through the banking registers, so
//! their lines stay open here. It has two scopes. The windows' scope is the banked
//! windows the model names in its `MAP` slots, the regions every other bus address lands
//! in, and the lines of its other heads (`cpu_lines`). The narrow scope is the small
//! regions (`small_unbanked`), the lines its first heads read (`first_addresses`,
//! `bus_lines`) and the lines a pass's decoded read and write addresses name
//! (`NAMED_SLOTS`, `Recent`).

use gbcformer::circuit::{Circuit, Lin, Query};
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{Reg, CRAM, LINE, LINE_BITS, REGS};
use gbcformer::one_model::Part;

/// What the derivation takes as fixed: the ROM's lines and the register bytes written
/// only at load (`sm83::abi` holds a register line's record to the same four).
#[derive(Clone)]
pub struct Fixed {
    rom_lines: u32,
    regs: Vec<(u32, u8)>,
}

impl Fixed {
    pub fn of(image: &Image) -> Fixed {
        let regs = [Reg::Mapper, Reg::Mask0, Reg::RamMask, Reg::Booting].iter().map(|&r| REGS + r as u32).map(|a| (a, image.read(a))).collect();
        Fixed { rom_lines: CRAM >> LINE_BITS, regs }
    }

    fn line(&self, image: &Image, n: u32) -> [Option<u8>; LINE] {
        if n < self.rom_lines {
            return if image.has_line(n) { image.line(n).map(Some) } else { [None; LINE] };
        }
        std::array::from_fn(|k| self.regs.iter().find(|&&(a, _)| a == (n << LINE_BITS) + k as u32).map(|&(_, v)| v))
    }

    /// Asserts that a write record of `line` to line `n` keeps every fixed byte. The cache
    /// checks every write record once a scope exists. The prefill, which writes the bytes
    /// the derivation reads, runs before that.
    pub fn guard(&self, n: u32, line: &[u8; LINE]) {
        assert!(n >= self.rom_lines, "a write record lands in ROM (line {n:#07x}): the scoped sets take ROM as fixed");
        for &(a, v) in &self.regs {
            if a >> LINE_BITS == n {
                assert_eq!(line[a as usize % LINE], v, "a write record changes the register byte at {a:#08x}: the scoped sets take it as fixed");
            }
        }
    }
}

/// The line bits of a physical address.
const LINE_ADDRESS: u32 = ((1 << gbcformer::memory::layout::ADDR_BITS) - 1) & !((1 << LINE_BITS) - 1);

/// Per head of the merged circuit, its (fixed bits, values) over the address bits,
/// in `part`'s stream. Every other part's input names are held at 0, which is what the
/// other parts see only where they hold none of the stream's slots: the checkpoint gathers
/// a shared slot once, so another holder sees the stream's byte. The cubes of the other
/// parts' heads are therefore not what a whole-sequence pass computes at their positions;
/// a reduced group (`STOCK_GROUP=own`) holds none of those positions.
pub fn cubes(c: &Circuit, part: Part, image: &Image, fixed: &Fixed) -> Vec<(u32, u32)> {
    let own = format!("{}:", part.name());
    c.reach(&|name: &str| if name.starts_with(&own) { None } else { Some(0) }, &|n| fixed.line(image, n))
}

/// The most cells the rows of a stream other than the GBC CPU's may attend. Its heads read
/// VRAM, OAM, the palettes and a few fixed lines, about 2,100 cells whatever the
/// cartridge, so a larger scope means a head the derivation left open.
pub const CELLS: usize = 4096;

/// The heads that can name more than `CELLS` lines, each with its count.
pub fn wide(c: &Circuit, cubes: &[(u32, u32)]) -> Vec<String> {
    let lines = |fixed: u32| 1usize << (LINE_ADDRESS & !fixed).count_ones();
    cubes.iter().zip(&c.heads).filter(|((f, _), _)| lines(*f) > CELLS).map(|((f, _), h)| format!("{} {}", h.name, lines(*f))).collect()
}

/// The lines below `lines` that any cube names, sorted.
pub fn lines_of(cubes: &[(u32, u32)], lines: usize) -> Vec<u32> {
    let mut hit = vec![false; lines];
    let mut seen = std::collections::HashSet::new();
    for &(fixed, value) in cubes {
        let free = LINE_ADDRESS & !fixed;
        let base = value & fixed & LINE_ADDRESS;
        if !seen.insert((free, base)) {
            continue;
        }
        let mut sub = 0u32;
        loop {
            let n = ((base | sub) >> LINE_BITS) as usize;
            if n < lines {
                hit[n] = true;
            }
            if sub == free {
                break;
            }
            sub = sub.wrapping_sub(free) & free;
        }
    }
    (0..lines as u32).filter(|&n| hit[n as usize]).collect()
}

/// Runs of consecutive lines: (first, end).
pub fn runs(lines: &[u32]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &n in lines {
        match out.last_mut() {
            Some((_, e)) if *e == n => *e = n + 1,
            _ => out.push((n, n + 1)),
        }
    }
    out
}

/// `stock-llama scope <rom>`: each stream's lines, by run, from the memory image after
/// `warmup` frames on the reference interpreter (as `play --warmup` hands it over).
pub fn report(rom: &str, hardware: Option<gbcformer::Hardware>, warmup: u64) {
    let bytes = std::fs::read(rom).unwrap_or_else(|e| panic!("{rom}: {e}"));
    let header = gbcformer::rom::identify(&bytes).unwrap();
    let hardware = gbcformer::Hardware::resolve(&header, hardware).unwrap();
    let mut session = gbcformer::Session::new(bytes, None, hardware).unwrap();
    while session.frame_number() < warmup {
        session.run_frame(gbcformer::machine::io::Buttons::default());
    }
    println!("{rom}: hardware {hardware:?}, after {warmup} frames on the reference interpreter");
    let image = session.memory_image().unwrap().with_palettes();
    let (c, _) = gbcformer::one_model::circuit(&gbcformer::one_model::parts());
    let fixed = Fixed::of(&image);
    let cells = (0..crate::cache::P as u32).filter(|&n| image.has_line(n)).count();
    println!("{} heads; {cells} lines hold a cell", c.heads.len());
    for part in Part::ALL {
        let cubes = cubes(&c, part, &image, &fixed);
        let lines = lines_of(&cubes, crate::cache::P);
        let held = lines.iter().filter(|&&n| image.has_line(n)).count();
        let exact = cubes.iter().filter(|&&(f, _)| f & LINE_ADDRESS == LINE_ADDRESS).count();
        let r = runs(&lines);
        println!("{part:?}: {} lines named ({held} hold a cell), {exact} of {} heads name one line; runs {:?}", lines.len(), cubes.len(), r.iter().map(|&(a, e)| format!("{:#08x}+{}", a << LINE_BITS, e - a)).collect::<Vec<_>>());
        if part != Part::Cpu {
            println!("{part:?}: heads that can name more than {CELLS} lines: {:?}", wide(&c, &cubes));
        } else {
            let open: Vec<&str> = cubes.iter().zip(&c.heads).filter(|((f, _), h)| h.name.starts_with("cpu:") && f & LINE_ADDRESS != LINE_ADDRESS).map(|(_, h)| h.name.as_str()).collect();
            let others = lines_of(&cpu_others(&c, &image, &fixed), crate::cache::P);
            println!("Cpu: heads translating a CPU address {open:?}; the other heads name {} lines, runs {:?}", others.len(), runs(&others).iter().map(|&(a, e)| format!("{:#08x}+{}", a << LINE_BITS, e - a)).collect::<Vec<_>>());
        }
    }
}

/// The GBC CPU's `MAP` slots, in `layout::WINDOWS` order. Each names the physical address
/// of its window's first byte in three bytes, least significant first.
pub const MAP_SLOTS: [[&str; 3]; 4] = [["MAP0a", "MAP0b", "MAP0c"], ["MAPRa", "MAPRb", "MAPRc"], ["MAPSa", "MAPSb", "MAPSc"], ["MAPWa", "MAPWb", "MAPWc"]];

/// The GBC CPU's slots that name the physical address of each of a step's two reads and
/// two writes, spelled like the `MAP` slots. A read or write the step does not make names
/// a line of VOID.
pub const NAMED_SLOTS: [[&str; 3]; 4] = [["RP0a", "RP0b", "RP0c"], ["RP1a", "RP1b", "RP1c"], ["WP0a", "WP0b", "WP0c"], ["WP1a", "WP1b", "WP1c"]];

/// The lines a region of byte addresses touches, below `lines`.
fn region(base: u32, len: u32, lines: usize) -> std::ops::Range<u32> {
    (base >> LINE_BITS).min(lines as u32)..(base + len).div_ceil(LINE as u32).min(lines as u32)
}

/// The GBC CPU stream's lines under the windows' first bytes the model decoded (`map`).
/// They are each window from its first byte, the regions every other bus address lands
/// in (`layout::UNBANKED`), and `others`, the lines of every head that does not translate
/// a GBC CPU address.
pub fn cpu_lines(others: &[u32], map: [u32; 4], lines: usize) -> Vec<u32> {
    use gbcformer::memory::layout::{UNBANKED, WINDOWS};
    let mut out: Vec<u32> = others.to_vec();
    for (base, len) in map.iter().zip(WINDOWS).map(|(&b, (_, len))| (b, len)).chain(UNBANKED) {
        out.extend(region(base, len, lines));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The GBC CPU stream's heads that do not translate a GBC CPU address. They are every
/// other part's head and each GBC CPU head whose query names one line. The rest (the
/// fetch, data and write-record heads) query `translate`, `fetch` or the VOID block.
pub fn cpu_others(c: &Circuit, image: &Image, fixed: &Fixed) -> Vec<(u32, u32)> {
    cubes(c, Part::Cpu, image, fixed).into_iter().zip(&c.heads).filter(|((f, _), h)| !h.name.starts_with("cpu:") || f & LINE_ADDRESS == LINE_ADDRESS).map(|(q, _)| q).collect()
}

/// The cubes of `part`'s own memory heads, of `cubes` (one per head of `c`).
///
/// In an overlay (`llama::Overlay`) these are the heads a stream's rows read the cells
/// with. Every memory head of another part carries the null term at the stream's tokens
/// and attends the null cell alone, which every scoped sequence holds (`Model::stream`,
/// `Cpp::scope`). The two copy heads, which no part owns, query the address a record
/// gathers, which is 0 at a pass's tokens: line 0, which `OWN_BESIDES` adds.
pub fn own(c: &Circuit, part: Part, cubes: &[(u32, u32)]) -> Vec<(u32, u32)> {
    cubes.iter().zip(&c.heads).filter(|(_, h)| c.part_of(h.out[0].0 as usize) == part as usize).map(|(&q, _)| q).collect()
}

/// The GBC CPU's own heads whose query names one line whatever its inputs (the two
/// `REGS` heads). Its other heads read at the addresses a pass fetches, reads and writes.
pub fn cpu_fixed(c: &Circuit, image: &Image, fixed: &Fixed) -> Vec<(u32, u32)> {
    own(c, Part::Cpu, &cubes(c, Part::Cpu, image, fixed)).into_iter().filter(|(f, _)| f & LINE_ADDRESS == LINE_ADDRESS).collect()
}

/// Per wire of a circuit, the memory heads its value depends on, a bit per head: through
/// the nodes' forms, and through each head's query into the wires the head drives.
struct Cones {
    words: usize,
    of: Vec<u64>,
}

impl Cones {
    fn of(c: &Circuit) -> Cones {
        let words = c.heads.len().div_ceil(64).max(1);
        let mut of = vec![0u64; c.ranges.len() * words];
        let add = |of: &[u64], l: &Lin, set: &mut [u64]| {
            for s in l.terms.keys() {
                for (w, x) in set.iter_mut().enumerate() {
                    *x |= of[s.0 as usize * words + w];
                }
            }
        };
        // A wire's cone only grows, so the passes end when one changes nothing.
        loop {
            let mut changed = false;
            let mut merge = |of: &mut [u64], wire: usize, set: &[u64]| {
                for (w, &x) in set.iter().enumerate() {
                    changed |= of[wire * words + w] | x != of[wire * words + w];
                    of[wire * words + w] |= x;
                }
            };
            for n in &c.nodes {
                let mut set = vec![0; words];
                n.steps.iter().for_each(|st| add(&of, &st.lin, &mut set));
                merge(&mut of, n.out.0 as usize, &set);
            }
            for (h, head) in c.heads.iter().enumerate() {
                let mut set = vec![0; words];
                if let Query::Wires(q) = &head.query {
                    q.iter().for_each(|l| add(&of, l, &mut set));
                }
                set[h / 64] |= 1 << (h % 64);
                head.out.iter().for_each(|o| merge(&mut of, o.0 as usize, &set));
            }
            if !changed {
                return Cones { words, of };
            }
        }
    }

    /// The heads the forms `lins` depend on.
    fn heads<'a>(&self, lins: impl Iterator<Item = &'a Lin>) -> Vec<usize> {
        let mut set = vec![0u64; self.words];
        for s in lins.flat_map(|l| l.terms.keys()) {
            for (w, x) in set.iter_mut().enumerate() {
                *x |= self.of[s.0 as usize * self.words + w];
            }
        }
        (0..self.words * 64).filter(|h| set[h / 64] >> (h % 64) & 1 == 1).collect()
    }
}

/// The GBC CPU's own memory heads by what their addresses depend on (`named_slots_wiring`).
pub struct Wiring {
    /// The heads whose query names one line whatever the inputs, each with its line.
    pub fixed: Vec<(String, u32)>,
    /// The heads whose address depends on the pass's inputs and the fixed heads alone.
    pub first: Vec<String>,
}

/// What the GBC CPU's narrow scope takes from the wiring, asserted on the circuit:
/// - a head whose query names one line whatever its inputs reads a line of the banking
///   registers (`REGS`), and no other own head is fixed;
/// - a `MAP` slot depends on no memory head but those, so a pass names its windows
///   rightly over the fixed lines alone;
/// - a read or write address slot (`NAMED_SLOTS`) depends on those and on first heads
///   only. A first head is a GBC CPU head whose query depends on no head but the fixed
///   ones: it reads at an address the pass's inputs and the banking registers give, as
///   the fetch does. So the four addresses are right whenever the first heads read
///   their lines, whatever the reads and writes at those addresses return.
///
/// The harness puts every first head's line in the scope before the pass. Which lines
/// those are is not read off the wiring: `sampled_reads` holds them to the circuit's own
/// evaluation.
pub fn named_slots_wiring(c: &Circuit, image: &Image, fixed: &Fixed) -> Wiring {
    let cubes = cubes(c, Part::Cpu, image, fixed);
    let own: Vec<bool> = c.heads.iter().map(|h| c.part_of(h.out[0].0 as usize) == Part::Cpu as usize).collect();
    let fixed_line: Vec<bool> = cubes.iter().zip(&own).map(|((f, _), &own)| own && f & LINE_ADDRESS == LINE_ADDRESS).collect();
    let regs = region(REGS, gbcformer::memory::layout::N_REGS as u32, usize::MAX);
    for (h, _) in fixed_line.iter().enumerate().filter(|(_, f)| **f) {
        let n = (cubes[h].1 & LINE_ADDRESS) >> LINE_BITS;
        assert!(regs.contains(&n), "{} names the one line {n:#07x} whatever its inputs, and that is no line of the banking registers: the narrow scope takes the fixed heads to be theirs", c.heads[h].name);
    }
    let cones = Cones::of(c);
    let query = |h: usize| match &c.heads[h].query {
        Query::Wires(q) => cones.heads(q.iter()),
        Query::Const(_) => Vec::new(),
    };
    let first: Vec<bool> = (0..c.heads.len()).map(|h| own[h] && !fixed_line[h] && query(h).iter().all(|&g| fixed_line[g])).collect();
    let slot = |name: &str| {
        let tagged = format!("{}:{name}", Part::Cpu.name());
        let i = c.outputs.iter().position(|(n, _)| *n == tagged).unwrap_or_else(|| panic!("the merged circuit has no output {tagged}"));
        cones.heads(c.outputs[i].1.iter().chain(c.value_outputs[i].iter().flatten()))
    };
    for name in MAP_SLOTS.iter().flatten() {
        if let Some(&h) = slot(name).iter().find(|&&h| !fixed_line[h]) {
            panic!("{name} depends on head {}, which names no fixed line: a pass could not name its windows over the fixed lines alone", c.heads[h].name);
        }
    }
    for name in NAMED_SLOTS.iter().flatten() {
        if let Some(&h) = slot(name).iter().find(|&&h| !fixed_line[h] && !first[h]) {
            panic!("{name} depends on head {}, whose own address depends on what another head reads: the narrow scope's rule does not hold for this circuit", c.heads[h].name);
        }
    }
    let names = |which: &[bool]| -> Vec<usize> { (0..c.heads.len()).filter(|&h| which[h]).collect() };
    Wiring { fixed: names(&fixed_line).into_iter().map(|h| (c.heads[h].name.clone(), (cubes[h].1 & LINE_ADDRESS) >> LINE_BITS)).collect(), first: names(&first).into_iter().map(|h| c.heads[h].name.clone()).collect() }
}

/// The bus addresses the GBC CPU's first heads read outside the regions the narrow scope
/// always holds, for a pass whose PC is `pc` and whose SP is `sp`: the opcode's and the
/// two operands' columns at PC, PC + 1 and PC + 2, and read 1 at SP + 1 in every step
/// (`sm83::decode`). `sampled_reads` holds the circuit to this list.
pub fn first_addresses(pc: u16, sp: u16) -> [u16; 4] {
    [pc, pc.wrapping_add(1), pc.wrapping_add(2), sp.wrapping_add(1)]
}

/// What `first_addresses` gives, by index.
const FIRST_NAMES: [&str; 4] = ["PC", "PC + 1", "PC + 2", "SP + 1"];

/// The steps `play` samples at start-up.
pub const SAMPLES: usize = 2000;

/// Evaluates the GBC CPU's own circuit `own` on `samples` random steps and holds every
/// line each of its memory heads reads to what the narrow scope holds for the pass. A
/// step is random bytes in every input, a DMA line's addresses and the bus states among
/// them, over `image` with random banking registers the cartridge's mapper accepts. In
/// each step:
/// - a fixed head (`Wiring::fixed`) reads its one line;
/// - a first head (`Wiring::first`) reads a line of `base` (the narrow scope's fixed
///   lines) or a line `bus_lines` gives one of `addresses(pc, sp)` under the windows the
///   step's own `MAP` slots name, which are the lines `play` puts in before the pass;
/// - every other head reads one of those or a line the step's `NAMED_SLOTS` name, which
///   `play` requires in the scope after the pass.
///
/// A step whose first address lies in cartridge RAM is left out, as `play` decodes such a
/// pass over the windows' scope. So is a step the circuit refuses (a declared range or an
/// exclusive case broken) and one in which a head names a line that holds no cell: random
/// bytes in the bus-state inputs reach states the machine does not, and the engine
/// refuses a pass whose head finds no cell. This is a sample, not a proof: it holds for the steps
/// drawn, and the summary it returns says how many those were and what each first head
/// was seen to read.
pub fn sampled_reads(own: &Circuit, wiring: &Wiring, image: &Image, base: &[u32], addresses: &dyn Fn(u16, u16) -> Vec<u16>, samples: usize) -> String {
    use gbcformer::memory::layout::N_REGS;
    use std::collections::{BTreeMap, BTreeSet};
    let strip = |n: &str| n.strip_prefix(Part::Cpu.name()).and_then(|n| n.strip_prefix(':')).unwrap_or(n).to_string();
    let fixed: BTreeMap<String, u32> = wiring.fixed.iter().map(|(n, l)| (strip(n), *l)).collect();
    let first: BTreeSet<String> = wiring.first.iter().map(|n| strip(n)).collect();
    for n in fixed.keys().chain(&first) {
        assert!(own.heads.iter().any(|h| h.name == *n), "the GBC CPU's own circuit has no head {n}");
    }
    let base: BTreeSet<u32> = base.iter().copied().collect();
    let regs = region(REGS, N_REGS as u32, usize::MAX);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    // Per first head: whether each of the addresses explains every line it read outside
    // `base`, and the steps in which it read one.
    let n_addr = addresses(0, 0).len();
    let mut seen: BTreeMap<String, (Vec<bool>, usize)> = first.iter().map(|n| (n.clone(), (vec![true; n_addr], 0))).collect();
    let (mut held, mut refused, mut unreached, mut cartridge_ram) = (0usize, 0usize, 0usize, 0usize);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut failure = None;
    for _ in 0..samples {
        let input: BTreeMap<String, u8> = own.inputs.iter().map(|(n, _)| (n.clone(), next() as u8)).collect();
        let mut banking = image.banking();
        for reg in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk] {
            banking.set(reg, next() as u8);
        }
        let read = std::cell::RefCell::new(Vec::new());
        let no_cell = std::cell::Cell::new(false);
        let mem = |n: u32| {
            read.borrow_mut().push(n);
            if regs.contains(&n) {
                std::array::from_fn(|k| banking.0.get(((n - regs.start) as usize) * LINE + k).copied().unwrap_or(0))
            } else if image.has_line(n) {
                image.line(n)
            } else {
                no_cell.set(true);
                [0; LINE]
            }
        };
        // The circuit serves its heads in `heads` order, one read each (`Circuit::eval_at`).
        let Ok(out) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| own.eval_at(&input, &mem, None))) else {
            refused += 1;
            continue;
        };
        // A head that names a line with no cell reads nothing in any runtime: the engine
        // refuses such a pass, so the inputs are no state the machine reaches.
        if no_cell.get() {
            unreached += 1;
            continue;
        }
        let read = read.into_inner();
        assert_eq!(read.len(), own.heads.len(), "one read per head");
        let triple = |w: &[&str; 3]| out[w[0]] as u32 | (out[w[1]] as u32) << 8 | (out[w[2]] as u32) << 16;
        let map: [u32; 4] = std::array::from_fn(|k| triple(&MAP_SLOTS[k]));
        let (pc, sp) = (input["PCL"] as u16 | (input["PCH"] as u16) << 8, input["SPL"] as u16 | (input["SPH"] as u16) << 8);
        let Some(placed) = addresses(pc, sp).into_iter().map(|a| bus_lines(a, map)).collect::<Option<Vec<Vec<u32>>>>() else {
            cartridge_ram += 1;
            continue;
        };
        let named: Vec<u32> = NAMED_SLOTS.iter().map(|w| triple(w) >> LINE_BITS).collect();
        held += 1;
        for (h, &n) in own.heads.iter().zip(&read) {
            let before = base.contains(&n) || placed.iter().any(|l| l.contains(&n));
            let at = || format!("in a step at PC {pc:#06x}, SP {sp:#06x} under the windows {map:x?} (inputs {:x?}, banking registers {:x?}, the step names {named:x?})", own.inputs.iter().map(|(n, _)| (n.as_str(), input[n])).collect::<Vec<_>>(), banking.0);
            if let Some(&line) = fixed.get(&h.name) {
                if n != line {
                    failure = Some(format!("{} read line {n:#07x} {}, and its query names line {line:#07x} whatever its inputs", h.name, at()));
                }
            } else if let Some((explains, outside)) = seen.get_mut(&h.name) {
                if !before {
                    failure = Some(format!("{} read line {n:#07x} {}: the narrow scope puts the lines of {:?} and its fixed regions in before a pass, and this line is none of them", h.name, at(), &FIRST_NAMES[..n_addr.min(4)]));
                }
                if !base.contains(&n) {
                    *outside += 1;
                    for (k, l) in placed.iter().enumerate() {
                        explains[k] &= l.contains(&n);
                    }
                }
            } else if !before && !named.contains(&n) {
                failure = Some(format!("{} read line {n:#07x} {}: that is no fixed line, no line the first heads read and no line the step's read and write address slots name ({named:x?})", h.name, at()));
            }
            if failure.is_some() {
                break;
            }
        }
        if failure.is_some() {
            break;
        }
    }
    std::panic::set_hook(hook);
    if let Some(why) = failure {
        panic!("the narrow scope's lines are not the circuit's: {why}");
    }
    assert!(held * 4 >= samples, "only {held} of {samples} sampled steps were held ({refused} refused by the circuit, {unreached} with a head at a line that holds no cell, {cartridge_ram} with a first address in cartridge RAM)");
    let by = |k: Option<usize>| -> Vec<&str> {
        seen.iter()
            .filter(|(_, (explains, outside))| match k {
                Some(k) => *outside > 0 && explains.iter().position(|&e| e) == Some(k),
                None => *outside == 0,
            })
            .map(|(n, _)| n.as_str())
            .collect()
    };
    let unexplained: Vec<&str> = seen.iter().filter(|(_, (explains, outside))| *outside > 0 && !explains.iter().any(|&e| e)).map(|(n, _)| n.as_str()).collect();
    let families: Vec<String> = (0..n_addr.min(4)).map(|k| format!("{} {:?}", FIRST_NAMES[k], by(Some(k)))).collect();
    format!("{held} sampled steps held ({refused} the circuit refused, {unreached} had a head name a line that holds no cell, {cartridge_ram} had a first address in cartridge RAM): {} fixed heads read their lines; first heads by the address that gives every line they read outside the fixed regions: {}; reading at more than one of them {unexplained:?}; never outside the fixed regions {:?}; every other head read a fixed line, a first line or a line the step named", fixed.len(), families.join(", "), by(None))
}

/// The lines a stream's rows can name besides its own heads': line 0, the copy heads'.
pub const OWN_BESIDES: [u32; 1] = [0];

/// The regions of `layout::UNBANKED` the GBC CPU's narrow scope always holds: every one
/// but WRAM bank 0 and VRAM, which are large and which a pass names by line.
pub fn small_unbanked() -> impl Iterator<Item = (u32, u32)> {
    use gbcformer::memory::layout::{UNBANKED, VRAM, WRAM};
    UNBANKED.into_iter().filter(|&(base, _)| base != WRAM && base != VRAM)
}

/// The lines the byte at bus address `a` can sit in outside `small_unbanked`, read,
/// written or fetched, under the windows' first bytes `map` (the `MAP` slots): its
/// window's line at the address's offset, WRAM bank 0's line for C000-CFFF and its echo,
/// both VRAM banks' lines for 8000-9FFF, and none above FE00. None for A000-BFFF: a
/// cartridge RAM smaller than its window repeats in it (MBC2's 512 bytes, a 2 KiB RAM),
/// so the offset does not give the line. The test
/// `every_address_lands_in_its_bus_lines_or_a_small_region` holds `layout::translate` and
/// `layout::fetch` to it.
pub fn bus_lines(a: u16, map: [u32; 4]) -> Option<Vec<u32>> {
    use gbcformer::memory::layout::{VRAM, VRAM_LEN, WINDOWS, WRAM};
    let a = a as u32;
    let window = |k: usize, at: u32| vec![(map[k] + (a - at)) >> LINE_BITS];
    Some(match a {
        0x0000..0x4000 => window(0, WINDOWS[0].0 as u32),
        0x4000..0x8000 => window(1, WINDOWS[1].0 as u32),
        0x8000..0xA000 => vec![(VRAM + (a - 0x8000)) >> LINE_BITS, (VRAM + VRAM_LEN as u32 / 2 + (a - 0x8000)) >> LINE_BITS],
        0xA000..0xC000 => return None,
        0xC000..0xD000 => vec![(WRAM + (a - 0xC000)) >> LINE_BITS],
        0xD000..0xE000 => window(3, WINDOWS[3].0 as u32),
        0xE000..0xF000 => vec![(WRAM + (a - 0xE000)) >> LINE_BITS],
        0xF000..0xFE00 => window(3, 0xF000),
        _ => Vec::new(),
    })
}

/// The lines the GBC CPU's narrow scope holds beyond its fixed ones, least recently
/// named first, at most `cap`. A pass's fetch lines and the lines its decoded addresses
/// name join it, and the line named longest ago leaves when it is full.
#[derive(Default)]
pub struct Recent {
    lines: std::collections::VecDeque<u32>,
    cap: usize,
}

impl Recent {
    /// The fewest lines a narrow scope may keep: one pass's fetch and named lines.
    pub const MIN: usize = 16;

    pub fn new(cap: usize) -> Recent {
        assert!(cap >= Recent::MIN, "a narrow scope keeps at least {} named lines, one pass's", Recent::MIN);
        Recent { lines: std::collections::VecDeque::new(), cap }
    }

    /// Makes `n` the most recently named line. Returns the line that leaves, if any.
    pub fn touch(&mut self, n: u32) -> Option<u32> {
        if let Some(i) = self.lines.iter().position(|&m| m == n) {
            self.lines.remove(i);
        }
        self.lines.push_back(n);
        if self.lines.len() > self.cap { self.lines.pop_front() } else { None }
    }

    pub fn lines(&self) -> impl Iterator<Item = u32> + '_ {
        self.lines.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gbcformer::memory::layout::{fetch_locked, translate, translate_locked, Access, Banking, Dma, Mapper, WINDOWS};

    /// Every bus address, read, written or fetched, lands in one of its `bus_lines` under
    /// the windows' first bytes, or in a small unbanked region: with OAM DMA off, on
    /// every bus and on each alone, under each of the GBC CPU's access locks, and under
    /// random registers of every mapper, the boot ROM mapped or not.
    #[test]
    fn every_address_lands_in_its_bus_lines_or_a_small_region() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        let small = |p: u32| small_unbanked().any(|(base, len)| (base..base + len).contains(&p));
        let mut in_lines = 0u64;
        for trial in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = mappers[trial % 5];
            let ram = if m == Mapper::Mbc2 { 0x200 } else { [0, 0x800, 0x2000, 0x8000][(rng >> 60) as usize % 4] };
            let mut b = Banking::new(m, 0x8000 << ((rng >> 3) % 9), ram, rng >> 9 & 1 == 1).unwrap();
            if m == Mapper::Mbc3 && rng >> 40 & 1 == 1 {
                b = b.with_clock();
            }
            for (i, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                b.set(reg, (rng >> (7 * i + 10)) as u8);
            }
            // The boot ROM mapped, unmapped again, and the CGB's switch to DMG compatibility.
            for (i, reg) in [Reg::Booting, Reg::Unmapped, Reg::Compat, Reg::Key0].into_iter().enumerate() {
                b.set(reg, (rng >> (2 * i + 52)) as u8 & 0x0F);
            }
            let map = WINDOWS.map(|(a, _)| translate(&b, a, Access::Read, false));
            for a in 0..=0xFFFFu16 {
                let Some(lines) = bus_lines(a, map) else {
                    assert!((0xA000..0xC000).contains(&a));
                    continue;
                };
                for dma in [Dma::NONE, Dma::ALL, Dma(Dma::OAM | Dma::VRAM), Dma(Dma::OAM | Dma::WRAM), Dma(Dma::OAM | Dma::MAIN | Dma::WRAM)] {
                    for locks in 0..4 {
                        for (what, p) in [("read", translate_locked(&b, a, Access::Read, dma, locks)), ("write", translate_locked(&b, a, Access::Write, dma, locks)), ("fetch", fetch_locked(&b, a, dma, locks))] {
                            let named = lines.contains(&(p >> LINE_BITS));
                            assert!(named || small(p), "{a:#06x} {what} under {:?}, DMA {dma:?}, locks {locks}: {p:#08x} is in none of its bus lines {lines:x?} and in no small unbanked region", b.0);
                            in_lines += named as u64;
                        }
                    }
                }
            }
        }
        assert!(in_lines > 200 * 0x8000, "the bus lines hold most addresses");
    }

    #[test]
    fn the_line_named_longest_ago_leaves_a_full_scope() {
        let mut r = Recent::new(Recent::MIN);
        for n in 0..Recent::MIN as u32 {
            assert_eq!(r.touch(n), None);
        }
        assert_eq!(r.touch(0), None, "a held line moves to the back");
        assert_eq!(r.touch(100), Some(1));
        assert_eq!(r.touch(2), None);
        assert_eq!(r.touch(101), Some(3));
        assert_eq!(r.lines().count(), Recent::MIN);
    }
    /// The merged circuit, the GBC CPU's own, the image of a random MBC5 cartridge and the
    /// narrow scope's fixed lines over it, as `play` derives them.
    fn wired() -> &'static (Circuit, [Circuit; 6], Image, Vec<u32>) {
        static WIRED: std::sync::OnceLock<(Circuit, [Circuit; 6], Image, Vec<u32>)> = std::sync::OnceLock::new();
        WIRED.get_or_init(|| {
            let parts = gbcformer::one_model::parts();
            let (merged, _) = gbcformer::one_model::circuit(&parts);
            let mut rng = 0x0123_4567_89AB_CDEFu64;
            let rom: Vec<u8> = (0..0x20000)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng as u8
                })
                .collect();
            let image = Image::new(&rom, 0x2000, Banking::new(Mapper::Mbc5, rom.len(), 0x2000, true).unwrap()).with_palettes();
            let fixed = Fixed::of(&image);
            let mut base = lines_of(&cpu_fixed(&merged, &image, &fixed), crate::cache::P);
            base.extend(OWN_BESIDES);
            base.extend(small_unbanked().flat_map(|(base, len)| crate::cache::lines(base, len as usize)));
            base.push(crate::cache::P as u32 - 1);
            base.sort_unstable();
            base.dedup();
            (merged, parts, image, base)
        })
    }

    fn panic_text(r: std::thread::Result<impl Sized>) -> String {
        let e = r.err().expect("a refusal");
        e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
    }

    /// The circuit's own evaluation reads where the narrow scope puts lines: the two
    /// fixed heads at the banking registers, every first head at PC, PC + 1, PC + 2 or
    /// SP + 1, every other head at a line the step names.
    #[test]
    fn the_circuit_reads_the_lines_the_narrow_scope_holds() {
        let (merged, parts, image, base) = wired();
        let own = &parts[Part::Cpu as usize];
        let wiring = named_slots_wiring(merged, image, &Fixed::of(image));
        assert_eq!(wiring.fixed.len(), 2, "the two heads at the banking registers' lines: {:?}", wiring.fixed);
        assert!(wiring.first.len() >= 4, "the fetch's columns and read 1: {:?}", wiring.first);
        let summary = sampled_reads(own, &wiring, image, base, &|pc, sp| first_addresses(pc, sp).to_vec(), 600);
        eprintln!("{summary}");
        for at in FIRST_NAMES {
            assert!(!summary.contains(&format!("{at} []")), "no first head was seen to read at {at}: {summary}");
        }
    }

    /// Without SP + 1 among the addresses, read 1's head reads a line nothing placed.
    #[test]
    fn a_first_head_with_no_line_placed_fails_the_sample() {
        let (merged, parts, image, base) = wired();
        let own = &parts[Part::Cpu as usize];
        let wiring = named_slots_wiring(merged, image, &Fixed::of(image));
        let text = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sampled_reads(own, &wiring, image, base, &|pc, sp| first_addresses(pc, sp)[..3].to_vec(), 600))));
        assert!(text.contains("the narrow scope's lines are not the circuit's") && text.contains("read line"), "{text}");
        // And without the operands' addresses, a fetch column does.
        let text = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sampled_reads(own, &wiring, image, base, &|pc, sp| vec![pc, sp.wrapping_add(1)], 600))));
        assert!(text.contains("the narrow scope's lines are not the circuit's"), "{text}");
    }

    /// A head taken for a first head that is not one fails too, when it reads a line the
    /// step names and nothing places beforehand. Some of the other heads do (the data
    /// reads' columns), and each of those must fail the sample.
    #[test]
    fn a_head_wrongly_taken_for_a_first_head_fails_the_sample() {
        let (merged, parts, image, base) = wired();
        let own = &parts[Part::Cpu as usize];
        let wiring = named_slots_wiring(merged, image, &Fixed::of(image));
        let others: Vec<&String> = merged.heads.iter().map(|h| &h.name).filter(|n| n.starts_with("cpu:") && !wiring.first.contains(n) && !wiring.fixed.iter().any(|(f, _)| f == *n)).collect();
        assert!(!others.is_empty(), "a head that is neither fixed nor first");
        let failing: Vec<&String> = others
            .iter()
            .copied()
            .filter(|&other| {
                let taken = Wiring { fixed: wiring.fixed.clone(), first: wiring.first.iter().cloned().chain([other.clone()]).collect() };
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sampled_reads(own, &taken, image, base, &|pc, sp| first_addresses(pc, sp).to_vec(), 600))).is_err()
            })
            .collect();
        eprintln!("of {} other heads, {} fail the sample when taken for a first head: {failing:?}", others.len(), failing.len());
        assert!(!failing.is_empty(), "no head that is not a first head reads a line outside the placed ones: the sample cannot tell a first head from another");
    }

    /// The wiring asserts refuse a circuit whose `MAP` slot depends on a fetched byte, and
    /// one whose address slot depends on a head that is not a first head.
    #[test]
    fn a_slot_wired_to_the_wrong_head_fails_the_wiring() {
        let (merged, parts, image, _) = wired();
        let fixed = Fixed::of(image);
        let wiring = named_slots_wiring(merged, image, &fixed);
        let wire = |c: &mut Circuit, slot: &str, head: &str| {
            let sig = c.heads.iter().find(|h| h.name == head).unwrap_or_else(|| panic!("no head {head}")).out[0];
            let i = c.outputs.iter().position(|(n, _)| n == slot).unwrap_or_else(|| panic!("no output {slot}"));
            match c.value_outputs[i].as_mut() {
                Some(nibbles) => nibbles[0].terms.insert(sig, 1),
                None => c.outputs[i].1[0].terms.insert(sig, 1),
            };
        };
        let (mut c, _) = gbcformer::one_model::circuit(parts);
        wire(&mut c, "cpu:MAP0a", &wiring.first[0]);
        let text = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| named_slots_wiring(&c, image, &fixed))));
        assert!(text.contains("MAP0a depends on head"), "{text}");
        let other = merged.heads.iter().find(|h| h.name.starts_with("cpu:") && !wiring.first.contains(&h.name) && !wiring.fixed.iter().any(|(n, _)| *n == h.name)).expect("a head that is neither fixed nor first").name.clone();
        // The slot must not already depend on it.
        let (mut c, _) = gbcformer::one_model::circuit(parts);
        wire(&mut c, "cpu:RP0a", &other);
        let text = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| named_slots_wiring(&c, image, &fixed))));
        assert!(text.contains("RP0a depends on head"), "{text}");
    }
}
