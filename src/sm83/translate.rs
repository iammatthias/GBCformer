//! Address translation in the circuit: a 16-bit bus address and the banking registers'
//! bytes (level 0) to the 24 bits of the fixed physical layout (`memory::layout`), in
//! one level.
//!
//! The space of (address, registers) is cut into cases. A case is a threshold
//! condition over level-0 wires, a conjunction of literals or a weighted form such as
//! "C000-FDFF" (a15 and a14 set, not all of a9..a13 set), and gives each physical bit a
//! constant or a conjunction of further literals (a bus bit, a bank bit and its mask
//! bit). Bit k of the address is then one node, `Σ_cases coef·[cond ∧ bit_k(case)]`,
//! where the conjunction of a condition with a literal is one more weighted term. The
//! cases that set one bit are disjoint, so every node is 0/1, which `Circuit::eval`
//! asserts and the test against `layout::translate` exercises. Cases may overlap when
//! they set different bits: the switchable ROM bank passes its bus bits in one case per
//! mapper while bit 14, the only bit the forced-low rule touches, has two small cases of
//! its own. The engine re-evaluates a step when any wire it reads changes, so a bank
//! register bit that appears in a few steps instead of every ROM step keeps a bank
//! switch cheap (perf-log, "Unit I").
//!
//! Reads pass bus bits 0..6 through in every case, so those bits are free. Writes to a
//! banking register force them to the register's index, so a write translation builds
//! all 24 bits. The I/O page lands in `layout::IO` with each register's port code in
//! bits 7..10 (`layout::io_code`), whose cases come from that table as aligned pieces of
//! the low seven bits (`io_cases`). The I/O page's two CGB registers (VBK at FF4F, SVBK
//! at FF70) are carved out of it with coefficient −1 cases, so no complement of two
//! addresses is needed: their writes reach the banking registers, their reads the
//! readback table `layout::TAB` at the register's bank bits.

use super::{G, W};
use crate::circuit::{one, Circuit, Lin, Step};
use crate::memory::layout::{self, Access, Mapper, Reg};

/// What OAM DMA blocks at one access, as 0/1 forms per bus (`layout::Dma`): OAM and the
/// unusable area, VRAM's bus, the main bus (ROM and cartridge RAM) and work RAM's.
#[derive(Clone)]
pub(super) struct Blocks {
    pub(super) oam: Lin,
    pub(super) vram: Lin,
    pub(super) main: Lin,
    pub(super) wram: Lin,
}

impl Blocks {
    pub(super) fn none() -> Blocks {
        Blocks { oam: Lin::konst(0), vram: Lin::konst(0), main: Lin::konst(0), wram: Lin::konst(0) }
    }
}

/// The banking registers as byte wires, in `Reg` order, and the effective banks they
/// select, once per pass.
pub(super) struct Regs(pub(super) Vec<W>, pub(super) Option<Banks>, pub(super) Locks);

/// The GBC CPU's access locks as 0/1 forms: VRAM in mode 3, OAM in modes 2 and 3. A
/// locked access lands in the FF block, so a read gives 0xFF and a write keeps nothing.
#[derive(Clone)]
pub(super) struct Locks {
    pub(super) vram: Lin,
    pub(super) oam: Lin,
}

impl Locks {
    pub(super) fn none() -> Locks {
        Locks { vram: Lin::konst(0), oam: Lin::konst(0) }
    }
}

/// The physical address of each banked window's first byte under the registers, OAM DMA
/// aside (`MAP0` 0000 under MBC1's mode, `MAPR` 4000, `MAPS` A000, `MAPW` D000). They
/// compute the MBC's bank pins, the masks, MBC1's quirks and SVBK once, as the MBC
/// computes them once for the one address on the bus. Each translation then takes a
/// window's bank bits from these wires in one case, instead of from the registers in one
/// case per mapper. They are also the MAP slots a runtime scopes its attention by.
#[derive(Clone)]
pub(super) struct Banks {
    pub(super) map0: W,
    pub(super) mapr: W,
    pub(super) maps: W,
    pub(super) mapw: W,
}

impl Regs {
    /// The registers with their banks: four translations of constant addresses, whose
    /// nodes read only the registers.
    pub(super) fn banked(g: &mut G, bytes: Vec<W>) -> Regs {
        let plain = Regs(bytes, None, Locks::none());
        let mut window = |name: &str, base: u64| build(g, name, &crate::circuit::bits(base, 16), &plain, &Blocks::none(), Access::Read, 0, Region::Literal).bits;
        let banks = Banks { map0: window("MAP0", 0x0000), mapr: window("MAPR", 0x4000), maps: window("MAPS", 0xA000), mapw: window("MAPW", 0xD000) };
        Regs(plain.0, Some(banks), Locks::none())
    }
    /// The same registers with the access locks `locks`.
    pub(super) fn with_locks(self, locks: Locks) -> Regs {
        Regs(self.0, self.1, locks)
    }
    /// The registers with the banks and no access lock: a DMA's access, which the PPU's
    /// modes do not hold the way they hold the GBC CPU's.
    pub(super) fn without_locks(&self) -> Regs {
        Regs(self.0.clone(), self.1.clone(), Locks::none())
    }
    /// The registers alone: a translation that reads them directly, one level earlier
    /// than one that reads the banks, as the opcode fetch at the front of the pass must.
    pub(super) fn unbanked(&self) -> Regs {
        Regs(self.0.clone(), None, self.2.clone())
    }
    fn bit(&self, r: Reg, k: usize) -> Lin {
        self.0[r as usize][k].clone()
    }
    /// ROM bank mask bit `b`: MASK0's bits 0..7, then MASK1's bit 0.
    fn mask_bit(&self, b: usize) -> Lin {
        if b < 8 { self.bit(Reg::Mask0, b) } else { self.bit(Reg::Mask1, 0) }
    }
    fn mapper(&self, m: Mapper) -> Lin {
        self.bit(Reg::Mapper, m as usize)
    }
    fn has_ram(&self) -> Lin {
        self.bit(Reg::Mapper, 6)
    }
    fn has_clock(&self) -> Lin {
        self.bit(Reg::Mapper, 5)
    }
    /// CGB mode as literals: CGB hardware (MAPPER bit 7) and not left for DMG
    /// compatibility (COMPAT bit 0).
    fn cgb(&self) -> Vec<Lin> {
        vec![self.bit(Reg::Mapper, 7), not(&self.bit(Reg::Compat, 0))]
    }
    /// The boot ROM mapped, as literals: a power-on run (BOOTING) that no FF50 write
    /// unmapped (UNMAPPED, COMPAT).
    fn boot_mapped(&self) -> Vec<Lin> {
        vec![self.bit(Reg::Booting, 0), not(&self.bit(Reg::Unmapped, 0)), not(&self.bit(Reg::Compat, 0))]
    }
    /// The registers FF50's write lands on while the boot ROM is mapped, each with the
    /// literals that pick it: COMPAT on CGB hardware whose KEY0 asks for compatibility,
    /// UNMAPPED otherwise (two exclusive cases).
    fn unmap_cases(&self) -> Vec<(Reg, Vec<Lin>)> {
        let (hw, k2) = (self.bit(Reg::Mapper, 7), self.bit(Reg::Key0, 2));
        let mapped = self.boot_mapped();
        let with = |extra: Vec<Lin>| mapped.iter().cloned().chain(extra).collect::<Vec<Lin>>();
        vec![(Reg::Compat, with(vec![hw.clone(), k2.clone()])), (Reg::Unmapped, with(vec![not(&hw)])), (Reg::Unmapped, with(vec![hw, not(&k2)]))]
    }
    /// The register writes of the I/O page besides the mappers': each (byte, register,
    /// literals that gate it). VBK and SVBK on CGB; KEY0 on CGB and FF50 while the boot ROM
    /// is mapped (afterwards a KEY0 write reaches nothing, `layout::io_write_code`).
    fn io_registers(&self) -> Vec<(u8, Reg, Vec<Lin>)> {
        let mut out: Vec<(u8, Reg, Vec<Lin>)> = [(0x4Fu8, Reg::Vbk), (0x70, Reg::Svbk)].into_iter().map(|(b, r)| (b, r, self.cgb())).collect();
        out.push((0x4C, Reg::Key0, vec![self.bit(Reg::Mapper, 7), self.bit(Reg::Booting, 0), not(&self.bit(Reg::Unmapped, 0)), not(&self.bit(Reg::Compat, 0))]));
        out.extend(self.unmap_cases().into_iter().map(|(r, lits)| (0x50, r, lits)));
        out
    }
}

/// Bits of a physical address.
const BITS: usize = layout::ADDR_BITS;

/// The bits `layout::BOOT` adds to a bank-0 address, whose low fourteen bits pass.
const BOOT_BITS: [usize; 2] = [18, 23];

/// A translated address: 24 physical bits, the host flag (a read the host serves from a
/// port, a fetch it serves through HOST) and, for writes, whether a record is kept and
/// whether it lands on a banking register.
pub(super) struct Physical {
    pub(super) bits: W,
    pub(super) host: Lin,
    pub(super) record: Lin,
    pub(super) reg: Lin,
    /// The register cases' steps, until `with_reg` makes them the `reg` wire.
    reg_steps: Vec<Step>,
}

/// `[lin ≥ theta]` with `lin` bounded above; `and` keeps it a single step.
#[derive(Clone)]
pub(super) struct Cond {
    pub(super) lin: Lin,
    pub(super) theta: i64,
}

impl Cond {
    pub(super) fn all(lits: &[Lin]) -> Cond {
        Cond { lin: lits.iter().cloned().fold(Lin::konst(0), |s, l| s + l), theta: lits.len() as i64 }
    }
    /// The condition and one more 0/1 literal, as one step: the literal's weight
    /// exceeds the form's headroom above the threshold.
    pub(super) fn and(self, c: &Circuit, lit: &Lin) -> Cond {
        let (_, hi) = c.bounds(&self.lin);
        let b = (hi - self.theta + 1).max(1);
        Cond { lin: self.lin + lit.clone() * b, theta: self.theta + b }
    }
    pub(super) fn step(self) -> Step {
        Step { coef: 1, lin: self.lin, theta: self.theta }
    }
}

#[derive(Clone)]
enum Bit {
    Zero,
    One,
    /// The conjunction of these literals.
    All(Vec<Lin>),
}

struct Case {
    coef: i64,
    cond: Cond,
    bits: Vec<Bit>,
    host: bool,
    record: bool,
    reg: bool,
}

impl Case {
    fn new(cond: Cond) -> Case {
        Case { coef: 1, cond, bits: vec![Bit::Zero; BITS], host: false, record: false, reg: false }
    }
    fn ones(mut self, ks: &[usize]) -> Case {
        for &k in ks {
            self.bits[k] = Bit::One;
        }
        self
    }
    fn pass(mut self, a: &W, ks: std::ops::Range<usize>) -> Case {
        for k in ks {
            self.bits[k] = Bit::All(vec![a[k].clone()]);
        }
        self
    }
    fn lit(mut self, k: usize, lits: &[Lin]) -> Case {
        self.bits[k] = Bit::All(lits.to_vec());
        self
    }
    fn host(mut self) -> Case {
        self.host = true;
        self
    }
    fn record(mut self) -> Case {
        self.record = true;
        self
    }
    fn negated(mut self) -> Case {
        self.coef = -1;
        self
    }
}

fn not(l: &Lin) -> Lin {
    one() - l.clone()
}

/// The exclusive conjunctions whose union is the complement of `lits`' conjunction:
/// ¬l1; l1 ∧ ¬l2; l1 ∧ l2 ∧ ¬l3; ...
fn complement(lits: &[Lin]) -> Vec<Vec<Lin>> {
    (0..lits.len()).map(|i| lits[..i].iter().cloned().chain([not(&lits[i])]).collect()).collect()
}

/// `layout::BLOCKED`, where OAM DMA sends every access but HRAM's.
const BLOCKED_BITS: [usize; 7] = [7, 8, 11, 14, 15, 17, 23];
/// `layout::FF`, where a locked VRAM or OAM access lands.
const FF_BITS: [usize; 6] = [7, 8, 14, 15, 17, 23];
const VOID_BITS: [usize; 5] = [9, 14, 15, 17, 23];
const CRAM_BITS: [usize; 1] = [23];
const WRAM_BITS: [usize; 2] = [17, 23];
const VRAM_BITS: [usize; 3] = [15, 17, 23];
/// Bits every address of the FE and FF pages holds (the OAM block's base).
const PAGE_BITS: [usize; 4] = [14, 15, 17, 23];
/// The bits BLOCKED, VOID, HRAM and the registers add to `PAGE_BITS`.
const BLOCKED_IN_PAGE: [usize; 3] = [7, 8, 11];
const FF_IN_PAGE: [usize; 2] = [7, 8];
const VOID_IN_PAGE: [usize; 1] = [9];
/// The bit the HOST block, where a host-served fetch lands, adds to `PAGE_BITS`.
const HOST_IN_PAGE: [usize; 1] = [10];
const HRAM_IN_PAGE: [usize; 1] = [8];
const REGS_IN_PAGE: [usize; 2] = [7, 9];
/// The bit `layout::IO` adds to `PAGE_BITS`, and `layout::TAB`'s.
const IO_IN_PAGE: [usize; 1] = [11];
const TAB_IN_PAGE: [usize; 1] = [12];
/// Bit `CODE + j` of an address in `layout::IO` is bit j of its port code.
const CODE: usize = 7;
/// The clock's port and latch in the page: `IO` plus their codes' bits.
const CLOCK_IN_PAGE: [usize; 2] = [9, 11];
const LATCH_BITS: [usize; 7] = [7, 9, 11, 14, 15, 17, 23];

/// The values below 0x80 in `set`, as aligned pieces (first, length): each length a power
/// of two and each first value a multiple of it, so a piece is a conjunction of literals.
fn pieces(set: impl Fn(u8) -> bool) -> Vec<(u8, u8)> {
    let mut out = Vec::new();
    let mut v = 0u8;
    while v < 0x80 {
        if !set(v) {
            v += 1;
            continue;
        }
        let mut len = 1u8;
        while len < 0x40 && v % (2 * len) == 0 && (v..v + 2 * len).all(&set) {
            len *= 2;
        }
        out.push((v, len));
        v += len;
    }
    out
}

/// A piece as literals on the low bits it fixes, each bit given as its literal and its
/// complement.
fn piece_lits(low: &[(Lin, Lin)], (first, len): (u8, u8)) -> Vec<Lin> {
    (len.trailing_zeros() as usize..7).map(|k| if first >> k & 1 == 1 { low[k].0.clone() } else { low[k].1.clone() }).collect()
}

/// The cases of an I/O-page access under `io` besides the block's own bit: each bit of
/// the port code (`layout::io_code`) over the aligned pieces of the low bits whose code
/// sets it, the CGB-only registers' pieces gated by the CGB bit. A read also takes the
/// host flag over the ports' pieces, and on CGB reads VBK and SVBK from `layout::TAB` at
/// their bank bits, carved out of `IO` with −1 cases.
fn io_cases(c: &Circuit, low: &[(Lin, Lin)], io: &Cond, r: &Regs, access: Access) -> Vec<Case> {
    let cgb = r.cgb();
    let hardware = vec![r.bit(Reg::Mapper, 7)];
    let gated = |lits: Vec<Lin>, gate: &[Lin]| lits.iter().chain(gate).fold(io.clone(), |k, l| k.and(c, l));
    // The ports any console answers, those only CGB hardware adds, and those only CGB
    // mode adds; `set(v, hardware, cgb)`.
    let over = |set: &dyn Fn(u8, bool, bool) -> bool| -> Vec<Cond> {
        let plain = pieces(|v| set(v, false, false)).into_iter().map(|p| gated(piece_lits(low, p), &[]));
        let on_hardware = pieces(|v| set(v, true, false) && !set(v, false, false)).into_iter().map(|p| gated(piece_lits(low, p), &hardware));
        let in_cgb = pieces(|v| set(v, true, true) && !set(v, true, false)).into_iter().map(|p| gated(piece_lits(low, p), &cgb));
        plain.chain(on_hardware).chain(in_cgb).collect()
    };
    let mut out = Vec::new();
    let code = |v: u8, hw: bool, cgb: bool| if access == Access::Write { layout::io_write_code(v, hw, cgb) } else { layout::io_code(v, hw, cgb) };
    for j in 0..4 {
        out.extend(over(&|v, hw, cgb| code(v, hw, cgb) >> j & 1 == 1).into_iter().map(|cond| Case::new(cond).ones(&[CODE + j])));
    }
    if access == Access::Read {
        out.extend(over(&|v, hw, cgb| layout::io_code(v, hw, cgb) != layout::IO_UNMAPPED).into_iter().map(|cond| Case::new(cond).host()));
        // VBK reads on CGB hardware in either mode, SVBK in CGB mode.
        for (byte, reg, bank_bits, gate) in [(0x4Fu8, Reg::Vbk, 1, &hardware), (0x70, Reg::Svbk, 3, &cgb)] {
            let cond = gated(piece_lits(low, (byte, 1)), gate);
            out.push(Case::new(cond.clone()).ones(&IO_IN_PAGE).negated());
            out.push((0..bank_bits).fold(Case::new(cond).ones(&TAB_IN_PAGE), |case, k| case.lit(CODE + k, &[r.bit(reg, k)])));
        }
    }
    out
}

/// The bus regions below FE00 as conditions over the address bits, and the FE and FF
/// pages, whose cases nest (`pages`).
struct Regions {
    rom0: Cond,
    romn: Cond,
    vram: Cond,
    cram: Cond,
    wram: Cond,
    /// FE00-FFFF.
    hi: Cond,
    /// A wire that is 1 when bus bits 0..6 are all set, for the HRAM and IE tests to
    /// read in place of those bits.
    ones7: Option<Lin>,
    /// The I/O page as a condition and the low seven bits gated by a wire it implies,
    /// each as its literal and its complement under that wire, for the port table to
    /// read in place of the bus bits: outside the I/O page (or the wider set the wire
    /// admits) they are all 0, so a pointer moving through RAM leaves the table still.
    io: Option<(Cond, Vec<(Lin, Lin)>)>,
    /// Bus bits 5..8 gated by FE00-FFFF, each as its literal and its complement under
    /// the gate, for the page tests (`Pages`) to read in place of the bus bits: outside
    /// the FE and FF pages they are all 0, so a pointer or a fetch moving through RAM or
    /// ROM leaves the page cases still.
    page_low: Option<[(Lin, Lin); 4]>,
}

impl Regions {
    /// The regions as forms over the address bits: every step of a case reads the
    /// bits its region test names, so a jump that flips them re-evaluates each step
    /// once per flipped bit.
    fn literal(a: &W) -> Regions {
        let sum = |r: std::ops::Range<usize>| r.map(|i| a[i].clone()).fold(Lin::konst(0), |s, l| s + l);
        Regions {
            rom0: Cond::all(&[not(&a[15]), not(&a[14])]),
            romn: Cond::all(&[not(&a[15]), a[14].clone()]),
            vram: Cond::all(&[a[15].clone(), not(&a[14]), not(&a[13])]),
            cram: Cond::all(&[a[15].clone(), not(&a[14]), a[13].clone()]),
            // C000-FDFF: a15 and a14 set, and not all of a9..a13.
            wram: Cond { lin: (a[15].clone() + a[14].clone()) * 5 + 5 - sum(9..14), theta: 11 },
            hi: Cond { lin: sum(9..16), theta: 7 },
            ones7: None,
            io: None,
            page_low: None,
        }
    }
    /// The regions as six wires, one level above the address, so a step reads one
    /// region wire in place of up to seven address bits and a jump within a region
    /// leaves every step but its own passed bit still. Costs a level, so it suits a
    /// source with slack below its consumer.
    fn wired(g: &mut G, name: &str, a: &W) -> Regions {
        let lit = Regions::literal(a);
        let sum = |r: std::ops::Range<usize>| r.map(|i| a[i].clone()).fold(Lin::konst(0), |s, l| s + l);
        let page = sum(8..16) + not(&a[7]);
        let io = g.c.ge(&format!("{name}.io"), page.clone(), 9);
        let low = (0..7)
            .map(|k| {
                let l = g.c.ge(&format!("{name}.io{k}"), page.clone() + a[k].clone(), 10);
                (l.clone(), io.clone() - l)
            })
            .collect();
        let mut wire = |field: &str, cond: Cond| Cond::all(&[g.c.ge(&format!("{name}.{field}"), cond.lin, cond.theta)]);
        let (rom0, romn, vram, cram, wram, hi) = (wire("rom0", lit.rom0), wire("romn", lit.romn), wire("vram", lit.vram), wire("cram", lit.cram), wire("wram", lit.wram), wire("hi", lit.hi));
        let page_low = [5usize, 6, 7, 8].map(|k| {
            let l = g.c.ge(&format!("{name}.pg{k}"), sum(9..16) + a[k].clone(), 8);
            (l.clone(), hi.lin.clone() - l)
        });
        Regions { rom0, romn, vram, cram, wram, hi, ones7: None, io: Some((Cond::all(&[io]), low)), page_low: Some(page_low) }
    }
    /// The regions of a bus that carries one of `sources` (the address, and the 0/1
    /// gates that select it, at most one source selected) and is zero otherwise, as six
    /// wires at the bus's own level: each is one step per source over that source's
    /// region test and its gates. With no source selected every wire is 0.
    fn muxed(g: &mut G, name: &str, sources: &[(W, Vec<Lin>)]) -> Regions {
        let field = |pick: fn(Regions) -> Cond, c: &Circuit| -> Vec<Step> {
            sources
                .iter()
                .map(|(a, gates)| {
                    let cond = gates.iter().fold(pick(Regions::literal(a)), |k, l| k.and(c, l));
                    Step { coef: 1, lin: cond.lin, theta: cond.theta }
                })
                .collect()
        };
        let mut wire = |f: &str, pick: fn(Regions) -> Cond| {
            let steps = field(pick, &g.c);
            Cond::all(&[g.c.node(&format!("{name}.{f}"), steps)])
        };
        let (rom0, romn, vram, cram, hi) = (wire("rom0", |r| r.rom0), wire("romn", |r| r.romn), wire("vram", |r| r.vram), wire("cram", |r| r.cram), wire("hi", |r| r.hi));
        // Work RAM is C000-FFFF less the hi region: C000-FFFF tests two bus bits, where
        // C000-FDFF's own test reads seven.
        let c000 = sources.iter().map(|(a, gates)| Cond::all(&gates.iter().cloned().chain([a[15].clone(), a[14].clone()]).collect::<Vec<Lin>>()).step()).collect();
        let c000 = g.c.node(&format!("{name}.c000"), c000);
        let wram = Cond { lin: c000 - hi.lin.clone(), theta: 1 };
        // Bus bit k under bus bit 13 set and bit 7 clear, one step per source over its
        // gates: every I/O-page address passes, and a work-RAM or VRAM address (bit 13
        // clear) leaves these wires 0. Bit 8's wire with the hi region is the page test.
        let gated = |g: &mut G, k: usize| -> Lin {
            let steps = sources
                .iter()
                .map(|(a, gates)| {
                    let lits: Vec<Lin> = gates.iter().cloned().chain([a[13].clone(), not(&a[7]), a[k].clone()]).collect();
                    Cond::all(&lits).step()
                })
                .collect();
            g.c.node(&format!("{name}.io{k}"), steps)
        };
        let page = gated(g, 8);
        let low = (0..7).map(|k| { let l = gated(g, k); (l.clone(), page.clone() - l) }).collect();
        let io = Some((Cond::all(&[hi.lin.clone(), page]), low));
        Regions { rom0, romn, vram, cram, wram, hi, ones7: None, io, page_low: None }
    }
    /// 1 when the address lies in no region: the muxed bus with no source selected.
    fn none(&self) -> Cond {
        let all = [&self.rom0, &self.romn, &self.vram, &self.cram, &self.wram, &self.hi];
        Cond { lin: all.iter().fold(Lin::konst(1), |s, r| s - r.lin.clone()), theta: 1 }
    }
}

/// How a translation tests its source's region.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Region {
    /// Forms over the address bits, no level.
    Literal,
    /// Six wires one level above the address.
    Wired,
}

/// The FE and FF pages: OAM, the unusable range, the I/O page, HRAM and IE. Every
/// physical address here holds bits 14, 15, 17 and 23 and the low seven bus bits, so
/// one case over a9..a15 carries them, and each range adds only what differs: VOID's
/// bit 9 and the host flag, HRAM's bit 8, OAM's bit 7, BLOCKED's bits 7, 8 and 11 under DMA. The
/// cases of one bit stay disjoint, and only the few steps that must tell IE from HRAM
/// (bits 8, 9 and the flags) read the low address bits, so a fetch or a pointer walking
/// through a page leaves the rest of the translation still.
struct Pages {
    hi: Case,
    oam: Cond,
    unusable: Cond,
    io: Cond,
    hram: Cond,
    ie: Cond,
}

impl Pages {
    fn new(c: &Circuit, a: &W, hi: &Cond, ones7: Option<&Lin>, page_low: Option<&[(Lin, Lin); 4]>) -> Pages {
        // Bits 5..8 as literal pairs, gated by the FE and FF pages where the regions are wires.
        let bit = |k: usize| page_low.map_or_else(|| (a[k].clone(), not(&a[k])), |p| p[k - 5].clone());
        let top3 = bit(7).0 * 4 + bit(6).0 * 2 + bit(5).0;
        let fe = hi.clone().and(c, &bit(8).1);
        // FE00-FE9F: the low byte below A0 (bits 7..5 below 101).
        let oam = Cond { lin: fe.lin.clone() * 5 - top3.clone(), theta: fe.theta * 5 - 4 };
        let unusable = Cond { lin: fe.lin.clone() * 5 + top3, theta: fe.theta * 5 + 5 };
        let io = hi.clone().and(c, &bit(8).0).and(c, &bit(7).1);
        let top = hi.clone().and(c, &bit(8).0).and(c, &bit(7).0);
        // FF80-FFFE: not all of a0..a6.
        let (hram, ie) = match ones7 {
            Some(ones) => (top.clone().and(c, &not(ones)), top.and(c, ones)),
            None => {
                let low = (0..7).map(|i| a[i].clone()).fold(Lin::konst(0), |s, l| s + l);
                (Cond { lin: top.lin.clone() * 8 + 7 - low, theta: top.theta * 8 + 1 }, Cond::all(a))
            }
        };
        Pages { hi: Case::new(hi.clone()).ones(&PAGE_BITS).pass(a, 0..7), oam, unusable, io, hram, ie }
    }
}

/// Bank bit `b` of an MBC1 multicart's registers (MBC1M), before the mask: the low
/// register's bits 0-3, then the high register's two.
fn multicart_bank_bit(r: &Regs, b: usize) -> Option<Lin> {
    match b {
        0..=3 => Some(r.bit(Reg::Romb0, b)),
        4 | 5 => Some(r.bit(Reg::Romb1, b - 4)),
        _ => None,
    }
}

/// Bank bit `b` of the ROM bank registers, before the mask, per mapper.
fn rom_bank_bit(r: &Regs, m: Mapper, b: usize) -> Option<Lin> {
    match (m, b) {
        (Mapper::Rom, _) => None,
        (Mapper::Mbc1, 0..=4) => Some(r.bit(Reg::Romb0, b)),
        (Mapper::Mbc1, 5 | 6) => Some(r.bit(Reg::Romb1, b - 5)),
        (Mapper::Mbc2, 0..=3) | (Mapper::Mbc3, 0..=6) | (Mapper::Mbc5, 0..=7) => Some(r.bit(Reg::Romb0, b)),
        (Mapper::Mbc5, 8) => Some(r.bit(Reg::Romb1, 0)),
        _ => None,
    }
}

/// The low bits of the bank register that `max(bank & mask, 1)` tests, per mapper.
fn forced_low(m: Mapper) -> usize {
    match m {
        Mapper::Mbc1 => 5,
        Mapper::Mbc2 => 4,
        Mapper::Mbc3 => 7,
        Mapper::Rom | Mapper::Mbc5 => 0,
    }
}

fn cases(c: &Circuit, g: &Regions, a: &W, r: &Regs, dma: &Blocks, access: Access, muxed: bool, fetch: bool) -> Vec<Case> {
    let pg = Pages::new(c, a, &g.hi, g.ones7.as_ref(), g.page_low.as_ref());
    // A region outside the buses OAM DMA blocks, each region by its own bus's wire.
    let live = |cond: &Cond, bus: &Lin| cond.clone().and(c, &not(bus));
    let mut out = vec![pg.hi];
    let mapper = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];

    // OAM DMA (`layout::Dma`): an access on a bus the transfer takes reads 0xFF and drops
    // its write: 0000-7FFF and A000-BFFF on the main bus, 8000-9FFF on VRAM's,
    // C000-FDFF on work RAM's, and OAM and the unusable area always; the FF page never.
    // A muxed bus is zero with no source selected, so its DMA cases test the region
    // wires, which are all 0 then, instead of the bus bits. A fetch's regions come from
    // PC a level ahead of its adder's top bits, so its DMA cases test them too.
    let wired_sum = |rs: [&Cond; 2]| Cond { lin: rs[0].lin.clone() + rs[1].lin.clone(), theta: 1 };
    let wired = muxed || fetch;
    let below_8000 = if wired { wired_sum([&g.rom0, &g.romn]) } else { Cond::all(&[not(&a[15])]) };
    let vram_bus = if wired { g.vram.clone() } else { Cond::all(&[a[15].clone(), not(&a[14]), not(&a[13])]) };
    let cram_bus = if wired { g.cram.clone() } else { Cond::all(&[a[15].clone(), not(&a[14]), a[13].clone()]) };
    for (cond, bus) in [(&below_8000, &dma.main), (&vram_bus, &dma.vram), (&cram_bus, &dma.main), (&g.wram, &dma.wram)] {
        out.push(Case::new(cond.clone().and(c, bus)).ones(&BLOCKED_BITS).pass(a, 0..7));
    }
    for cond in [&pg.oam, &pg.unusable] {
        out.push(Case::new(cond.clone().and(c, &dma.oam)).ones(&BLOCKED_IN_PAGE));
    }
    // The unusable area without OAM DMA, the I/O page and IE. A fetch the host serves
    // lands in HOST, where the host keeps the byte. Otherwise the unusable area reads
    // VOID's zeros and drops writes, IE is control's port, and the I/O page lands in `IO`
    // at its port code (`io_cases`).
    // The port table reads the gated low bits where the regions are wires.
    let (io, low): (Cond, Vec<(Lin, Lin)>) = match &g.io {
        Some((cond, low)) => (cond.clone(), low.clone()),
        None => (pg.io.clone(), (0..7).map(|k| (a[k].clone(), not(&a[k]))).collect()),
    };
    let unusable = live(&pg.unusable, &dma.oam);
    let page = || -> Vec<Case> {
        if fetch {
            return [&unusable, &pg.io, &pg.ie].into_iter().map(|cond| Case::new(cond.clone()).ones(&HOST_IN_PAGE).host()).collect();
        }
        let mut out = vec![Case::new(unusable.clone()).ones(&VOID_IN_PAGE), Case::new(pg.ie.clone()).ones(&IO_IN_PAGE).ones(&[CODE]).host(), Case::new(pg.io.clone()).ones(&IO_IN_PAGE)];
        out.extend(io_cases(c, &low, &io, r, access));
        out
    };

    if let (Access::Read, Some(b)) = (access, &r.1) {
        // Each ROM window passes the bus bits and takes its bank bits from its first
        // byte's physical address; bank 0's is BOOT's while the boot ROM is mapped, whose
        // bit 23 the ROM's never sets, so the boot ROM costs no case of its own.
        for (region, base, top) in [(&g.rom0, &b.map0, BITS), (&g.romn, &b.mapr, BITS - 1)] {
            out.push((14..top).fold(Case::new(live(region, &dma.main)).pass(a, 0..14), |case, k| case.lit(k, &[base[k].clone()])));
        }
        out.extend(page());
    } else if access == Access::Read {
        // 0000-3FFF: bank 0 passes the bus bits; MBC1 mode 1 adds its high register to
        // bits 19 and 20.
        // A multicart (MASK1 bit 1) puts them at bits 18 and 19.
        out.push(Case::new(live(&g.rom0, &dma.main)).pass(a, 0..14));
        let multi = r.bit(Reg::Mask1, 1);
        let mode1 = |base: Cond, wiring: Lin| base.and(c, &r.mapper(Mapper::Mbc1)).and(c, &r.bit(Reg::Mode, 0)).and(c, &wiring);
        let high_bits = |case: Case, at: usize| case.lit(at, &[r.bit(Reg::Romb1, 0), r.bit(Reg::Mask0, at - 14)]).lit(at + 1, &[r.bit(Reg::Romb1, 1), r.bit(Reg::Mask0, at - 13)]);
        out.push(high_bits(Case::new(mode1(live(&g.rom0, &dma.main), not(&multi))), 19));
        out.push(high_bits(Case::new(mode1(live(&g.rom0, &dma.main), multi.clone())), 18));
        // While the boot ROM is mapped bank 0 is BOOT: MBC1's mode-1 bits taken back,
        // BOOT's added. The cases read the region and the registers, not the page.
        let boot = r.boot_mapped().iter().fold(live(&g.rom0, &dma.main), |k, l| k.and(c, l));
        out.push(high_bits(Case::new(mode1(boot.clone(), not(&multi))).negated(), 19));
        out.push(high_bits(Case::new(mode1(boot.clone(), multi.clone())).negated(), 18));
        out.push(Case::new(boot).ones(&BOOT_BITS));
        // 4000-7FFF: the switchable bank, masked. One case per mapper carries the passed
        // bits and bank bits 1..8. Bit 14 is bank bit 0, forced to 1 when the register's
        // low bits are all zero, so where that rule applies it has two cases of its own:
        // the register's bit 0 set, or the low bits all clear.
        // MBC1 has two wirings: the standard one and the multicart's (MASK1 bit 1).
        let wirings: Vec<(Mapper, Option<bool>)> = mapper.iter().flat_map(|&m| if m == Mapper::Mbc1 { vec![(m, Some(false)), (m, Some(true))] } else { vec![(m, None)] }).collect();
        for (m, multicart) in wirings {
            let mut base = live(&g.romn, &dma.main).and(c, &r.mapper(m));
            if let Some(wired) = multicart {
                let flag = r.bit(Reg::Mask1, 1);
                base = base.and(c, &if wired { flag } else { not(&flag) });
            }
            let mut case = Case::new(base.clone()).pass(a, 0..14);
            for b in 1..9 {
                let bit = if multicart == Some(true) { multicart_bank_bit(r, b) } else { rom_bank_bit(r, m, b) };
                if let Some(bit) = bit {
                    case = case.lit(14 + b, &[bit, r.mask_bit(b)]);
                }
            }
            let low = forced_low(m);
            if low == 0 {
                out.push(match rom_bank_bit(r, m, 0) {
                    Some(bit) => case.lit(14, &[bit, r.bit(Reg::Mask0, 0)]),
                    None => case.lit(14, &[r.bit(Reg::Mask0, 0)]),
                });
            } else {
                out.push(case);
                out.push(Case::new(base.clone().and(c, &r.bit(Reg::Romb0, 0))).lit(14, &[r.bit(Reg::Mask0, 0)]));
                let zero_low = (0..low).fold(base, |k, b| k.and(c, &not(&r.bit(Reg::Romb0, b))));
                out.push(Case::new(zero_low).lit(14, &[r.bit(Reg::Mask0, 0)]));
            }
        }
        out.extend(page());
    } else {
        // Writes into 0000-7FFF reach a banking register or nothing. Each range tests
        // its half of the ROM area by the rom0 or romn region (a wire when the regions
        // are wired) and only the bus bits below 14 directly, so a pointer moving
        // through RAM leaves these steps still.
        let sub = |region: &Cond, lits: &[Lin]| lits.iter().fold(live(region, &dma.main), |k, l| k.and(c, l));
        let reg_case = |cond: Cond, reg: Reg, bits: &[usize]| {
            let mut case = Case::new(cond).ones(bits).record();
            case.reg = true;
            for k in 0..4 {
                if (reg as usize) >> k & 1 == 1 {
                    case.bits[k] = Bit::One;
                }
            }
            case
        };
        let regs_bits: Vec<usize> = REGS_IN_PAGE.iter().chain(&PAGE_BITS).copied().collect();
        let void_case = |cond: Cond| Case::new(cond).ones(&VOID_BITS).pass(a, 0..7);
        // MBC3's 6000-7FFF reaches the clock's latch on a board with the clock.
        let latch_case = |cond: Cond| Case::new(cond).ones(&LATCH_BITS).pass(a, 0..7);
        let (a13, a12, a8) = (a[13].clone(), a[12].clone(), a[8].clone());
        let (r0, rn) = (&g.rom0, &g.romn);
        for m in mapper {
            let mm = r.mapper(m);
            let ranges: Vec<(&Cond, Vec<Lin>, Option<Reg>)> = match m {
                Mapper::Rom => vec![(r0, vec![], None), (rn, vec![], None)],
                Mapper::Mbc2 => vec![(r0, vec![a8.clone()], Some(Reg::Romb0)), (r0, vec![not(&a8)], Some(Reg::Ramg)), (rn, vec![], None)],
                Mapper::Mbc1 => vec![(r0, vec![not(&a13)], Some(Reg::Ramg)), (r0, vec![a13.clone()], Some(Reg::Romb0)), (rn, vec![not(&a13)], Some(Reg::Romb1)), (rn, vec![a13.clone()], Some(Reg::Mode))],
                Mapper::Mbc3 => vec![(r0, vec![not(&a13)], Some(Reg::Ramg)), (r0, vec![a13.clone()], Some(Reg::Romb0)), (rn, vec![not(&a13)], Some(Reg::Ramb)), (rn, vec![a13.clone()], None)],
                Mapper::Mbc5 => vec![
                    (r0, vec![not(&a13)], Some(Reg::Ramg)),
                    (r0, vec![a13.clone(), not(&a12)], Some(Reg::Romb0)),
                    (r0, vec![a13.clone(), a12.clone()], Some(Reg::Romb1)),
                    (rn, vec![not(&a13)], Some(Reg::Ramb)),
                    (rn, vec![a13.clone()], None),
                ],
            };
            for (region, lits, reg) in ranges {
                let cond = sub(region, &lits).and(c, &mm);
                match reg {
                    Some(reg) => out.push(reg_case(cond, reg, &regs_bits)),
                    None if m == Mapper::Mbc3 => {
                        out.push(void_case(cond.clone().and(c, &not(&r.has_clock()))));
                        out.push(latch_case(cond.and(c, &r.has_clock())));
                    }
                    None => out.push(void_case(cond)),
                }
            }
        }
        // VBK and SVBK on CGB reach their registers, carved out of the I/O page with −1
        // cases that take back its passed bits and its block's bit.
        out.extend(page());
        // KEY0 and FF50 while the boot ROM is mapped likewise.
        for (byte, reg, gate) in r.io_registers() {
            let lits: Vec<Lin> = piece_lits(&low, (byte, 1)).into_iter().chain(gate).collect();
            let cond = lits.iter().fold(io.clone(), |k, l| k.and(c, l));
            out.push(Case::new(cond.clone()).ones(&IO_IN_PAGE).pass(a, 0..7).negated());
            out.push(reg_case(cond, reg, &REGS_IN_PAGE));
        }
    }

    // VRAM: bank from VBK on CGB.
    out.push(Case::new(live(&g.vram, &dma.vram).and(c, &not(&r.2.vram))).ones(&VRAM_BITS).pass(a, 0..13).lit(13, &[vec![r.bit(Reg::Vbk, 0)], r.cgb()].concat()).record());
    out.push(Case::new(live(&g.vram, &dma.vram).and(c, &r.2.vram)).ones(&FF_BITS).pass(a, 0..7));

    // Cartridge RAM. Every outcome (the RAM, the FF block, the clock's port) has bit 23,
    // so one case sets it. Disabled RAM adds FF's other bits. Enabled RAM passes the
    // offset masked to the RAM size and keeps a record, in one case for the mappers whose
    // select cannot leave the RAM and one for MBC3 with a select below 8; the bank bits
    // come from small cases per mapper. MBC3 selects 8-C reach the clock's port on a
    // board with the clock (HOST for a fetch), whose RAMG enables it with RAM or without,
    // and nothing (FF) otherwise, as D and up do.
    let cram = live(&g.cram, &dma.main);
    out.push(Case::new(cram.clone()).ones(&CRAM_BITS));
    if let Some(b) = &r.1 {
        // MAPS lands in the RAM (bit 23 without bit 17), where the offset passes masked
        // to the RAM's size and a write keeps a record. Otherwise it lands in the FF
        // block or the clock's port, whose bits it gives with the bus's low seven
        // passing; a read of the port is the host's. A fetch of the port lands in HOST.
        let m = &b.maps;
        let at = |k: usize| match (fetch, k) {
            (true, 9 | 10) => None,
            (true, 11) => Some(HOST_IN_PAGE[0]),
            _ => Some(k),
        };
        let on = cram.clone().and(c, &m[CRAM_BITS[0]]).and(c, &not(&m[17]));
        let on = Case::new(on).pass(a, 0..9).record();
        let on = (9..13).fold(on, |case, k| case.lit(k, &[a[k].clone(), r.bit(Reg::RamMask, k - 9)]));
        out.push((13..21).fold(on, |case, k| case.lit(k, &[m[k].clone()])));
        let off = cram.and(c, &m[17]);
        out.push((7..21).filter_map(|k| at(k).map(|t| (k, t))).fold(Case::new(off.clone()).pass(a, 0..7), |case, (k, t)| case.lit(t, &[m[k].clone()])));
        out.push(Case::new(off.and(c, &m[IO_IN_PAGE[0]])).host());
        // WRAM passes the offset and keeps a record, and D000-DFFF (and its echo) takes
        // MAPW's bank bits.
        out.push(Case::new(live(&g.wram, &dma.wram)).ones(&WRAM_BITS).pass(a, 0..12).record());
        out.push((12..15).fold(Case::new(live(&g.wram, &dma.wram).and(c, &a[12])), |case, k| case.lit(k, &[b.mapw[k].clone()])));
        out.push(Case::new(live(&pg.oam, &dma.oam).and(c, &not(&r.2.oam))).pass(a, 7..8).record());
        out.push(Case::new(live(&pg.oam, &dma.oam).and(c, &r.2.oam)).ones(&FF_IN_PAGE));
        out.push(Case::new(pg.hram.clone()).ones(&HRAM_IN_PAGE).record());
        return out;
    }
    let ff_cram: [usize; 5] = [7, 8, 14, 15, 17];
    // The clock's bits beside the page's (HOST's for a fetch), and FF's.
    let clock_only: &[usize] = if fetch { &HOST_IN_PAGE } else { &CLOCK_IN_PAGE };
    let clock_cram: Vec<usize> = clock_only.iter().chain(&[14, 15, 17]).copied().collect();
    let ff_only = [7, 8];
    let (s2, s1, s0) = (r.bit(Reg::Ramb, 2), r.bit(Reg::Ramb, 1), r.bit(Reg::Ramb, 0));
    // Selects 08-0C: 8 to B, or C.
    let clock_selects = [vec![not(&s2)], vec![s2.clone(), not(&s1), not(&s0)]];
    let high_clear: Vec<Lin> = (4..8).map(|k| not(&r.bit(Reg::Ramb, k))).collect();
    let ramg_on = [r.bit(Reg::Ramg, 3), not(&r.bit(Reg::Ramg, 2)), r.bit(Reg::Ramg, 1), not(&r.bit(Reg::Ramg, 0))];
    let enabled: Vec<Lin> = ramg_on.iter().cloned().chain([r.has_ram()]).collect();
    for lits in complement(&enabled) {
        out.push(Case::new(lits.iter().fold(cram.clone(), |k, l| k.and(c, l))).ones(&ff_cram).pass(a, 0..7));
    }
    // RAMG on without RAM: the clock's selects take the FF case back for the clock's port.
    let no_ram = ramg_on.iter().cloned().chain([not(&r.has_ram()), r.has_clock(), r.bit(Reg::Ramb, 3)]).chain(high_clear.iter().cloned()).fold(cram.clone(), |k, l| k.and(c, &l));
    for lits in &clock_selects {
        let cond = lits.iter().fold(no_ram.clone(), |k, l| k.and(c, l));
        out.push(Case::new(cond.clone()).ones(&ff_only).negated());
        out.push(Case::new(cond).ones(clock_only).host());
    }
    let cram_on = enabled.iter().fold(cram, |k, l| k.and(c, l));
    let cram_case = |cond: Cond| {
        let case = Case::new(cond).pass(a, 0..9).record();
        (9..13).fold(case, |case, k| case.lit(k, &[a[k].clone(), r.bit(Reg::RamMask, k - 9)]))
    };
    let bank_lit = |case: Case, k: usize, bit: Lin| case.lit(13 + k, &[bit, r.bit(Reg::RamMask, 4 + k)]);
    out.push(cram_case(cram_on.clone().and(c, &not(&r.mapper(Mapper::Mbc3)))));
    // A multicart never banks RAM.
    let mbc1_mode = cram_on.clone().and(c, &r.mapper(Mapper::Mbc1)).and(c, &r.bit(Reg::Mode, 0)).and(c, &not(&r.bit(Reg::Mask1, 1)));
    out.push((0..2).fold(Case::new(mbc1_mode), |case, k| bank_lit(case, k, r.bit(Reg::Romb1, k))));
    let mbc5 = cram_on.clone().and(c, &r.mapper(Mapper::Mbc5));
    out.push((0..4).fold(Case::new(mbc5), |case, k| bank_lit(case, k, r.bit(Reg::Ramb, k))));
    let mbc3 = cram_on.and(c, &r.mapper(Mapper::Mbc3));
    for lits in complement(&high_clear) {
        out.push(Case::new(lits.iter().fold(mbc3.clone(), |k, l| k.and(c, l))).ones(&ff_cram).pass(a, 0..7));
    }
    let low16 = high_clear.iter().fold(mbc3, |k, l| k.and(c, l));
    let bank = low16.clone().and(c, &not(&r.bit(Reg::Ramb, 3)));
    out.push((0..3).fold(cram_case(bank), |case, k| bank_lit(case, k, r.bit(Reg::Ramb, k))));
    let sel8 = low16.and(c, &r.bit(Reg::Ramb, 3));
    for lits in &clock_selects {
        let cond = lits.iter().fold(sel8.clone(), |k, l| k.and(c, l));
        out.push(Case::new(cond.clone().and(c, &r.has_clock())).ones(&clock_cram).pass(a, 0..7).host());
        out.push(Case::new(cond.and(c, &not(&r.has_clock()))).ones(&ff_cram).pass(a, 0..7));
    }
    for lits in [vec![s2.clone(), s1.clone()], vec![s2.clone(), not(&s1), s0.clone()]] {
        out.push(Case::new(lits.iter().fold(sel8.clone(), |k, l| k.and(c, l))).ones(&ff_cram).pass(a, 0..7));
    }

    // WRAM: one case passes the offset and keeps the record; D000-DFFF (and its echo)
    // adds the bank `max(SVBK & 7, 1)` in bits 12..14 from four cases over SVBK.
    let (v0, v1, v2) = (r.bit(Reg::Svbk, 0), r.bit(Reg::Svbk, 1), r.bit(Reg::Svbk, 2));
    out.push(Case::new(live(&g.wram, &dma.wram)).ones(&WRAM_BITS).pass(a, 0..12).record());
    let hi = live(&g.wram, &dma.wram).and(c, &a[12]);
    out.push(Case::new(hi.clone().and(c, &v0)).ones(&[12]).lit(13, &[v1.clone()]).lit(14, &[v2.clone()]));
    out.push(Case::new(hi.clone().and(c, &not(&v0)).and(c, &not(&v1)).and(c, &not(&v2))).ones(&[12]));
    out.push(Case::new(hi.clone().and(c, &not(&v0)).and(c, &v1)).ones(&[13]).lit(14, &[v2.clone()]));
    out.push(Case::new(hi.and(c, &not(&v0)).and(c, &not(&v1)).and(c, &v2)).ones(&[14]));

    out.push(Case::new(live(&pg.oam, &dma.oam).and(c, &not(&r.2.oam))).pass(a, 7..8).record());
    out.push(Case::new(live(&pg.oam, &dma.oam).and(c, &r.2.oam)).ones(&FF_IN_PAGE));
    out.push(Case::new(pg.hram.clone()).ones(&HRAM_IN_PAGE).record());
    out
}

fn build(g: &mut G, name: &str, a: &W, r: &Regs, dma: &Blocks, access: Access, floor: u32, region: Region) -> Physical {
    let regions = match region {
        Region::Literal => Regions::literal(a),
        Region::Wired => Regions::wired(g, name, a),
    };
    build_in(g, name, a, r, dma, access, floor, regions, false, false)
}

/// `build` over given regions; `muxed` is a bus that is zero with no source selected
/// and whose regions are then all 0: that case lands in VOID.
fn build_in(g: &mut G, name: &str, a: &W, r: &Regs, dma: &Blocks, access: Access, floor: u32, regions: Regions, muxed: bool, fetch: bool) -> Physical {
    let mut cases = cases(&g.c, &regions, a, r, dma, access, muxed, fetch);
    if muxed {
        cases.push(Case::new(regions.none()).ones(&VOID_BITS));
    }
    let per_bit: Vec<Vec<Step>> = (0..BITS).map(|k| bit_steps(&g.c, &cases, k)).collect();
    let (host_steps, rec_steps, reg_steps) = (flag_steps(&g.c, &cases, |c| c.host), flag_steps(&g.c, &cases, |c| c.record), flag_steps(&g.c, &cases, |c| c.reg));
    let free_low = access == Access::Read;
    let bits: W = per_bit.into_iter().enumerate().map(|(k, steps)| if free_low && k < 7 { a[k].clone() } else { g.c.node_at(&format!("{name}.{k}"), floor, steps) }).collect();
    // A write lands where its record says; only a read needs the host flag.
    let host = if access == Access::Read { g.c.node_at(&format!("{name}.host"), floor, host_steps) } else { Lin::konst(0) };
    let record = if access == Access::Write { g.c.node_at(&format!("{name}.rec"), floor, rec_steps) } else { Lin::konst(0) };
    Physical { bits, host, record, reg: Lin::konst(0), reg_steps }
}

/// One step per case that sets bit `k`, over its condition and its bit's literals.
fn bit_steps(c: &Circuit, cases: &[Case], k: usize) -> Vec<Step> {
    cases
        .iter()
        .filter_map(|case| match &case.bits[k] {
            Bit::Zero => None,
            Bit::One => Some(case_step(c, case, &[])),
            Bit::All(lits) => Some(case_step(c, case, lits)),
        })
        .collect()
}

/// One step per case `pick` flags, over its condition.
fn flag_steps(c: &Circuit, cases: &[Case], pick: fn(&Case) -> bool) -> Vec<Step> {
    cases.iter().filter(|case| pick(case)).map(|case| case_step(c, case, &[])).collect()
}

fn case_step(c: &Circuit, case: &Case, lits: &[Lin]) -> Step {
    let cond = lits.iter().fold(case.cond.clone(), |k, l| k.and(c, l));
    Step { coef: case.coef, lin: cond.lin, theta: cond.theta }
}

impl Physical {
    /// The write's register flag as a wire, for a write that can share a line with the
    /// step's other write (`fields::same_line`).
    pub(super) fn with_reg(mut self, g: &mut G, name: &str, floor: u32) -> Physical {
        let steps = std::mem::take(&mut self.reg_steps);
        self.reg = g.c.node_at(&format!("{name}.reg"), floor, steps);
        self
    }
}

/// The physical address a read or fetch of bus address `a` lands at.
pub(super) fn read_address(g: &mut G, name: &str, a: &W, r: &Regs, dma: &Blocks, region: Region) -> Physical {
    build(g, name, a, r, dma, Access::Read, 0, region)
}

/// "Bits 0..6 of pc + k − hb all set" from pc's own bits, one level above pc (k = 0
/// means pc itself, without the HALT bug term).
pub(super) fn ones7_ahead(g: &mut G, name: &str, pc: &W, k: u64, hb: &Lin) -> Lin {
    let low7 = |v: u64| (0..7).fold(Lin::konst(0), |s, j| s + if v >> j & 1 == 1 { pc[j].clone() } else { one() - pc[j].clone() });
    if k == 0 {
        return g.c.ge(&format!("{name}.ones7"), low7(127), 7);
    }
    // pc's low seven bits are 127 − k without the bug and 128 − k with it.
    let steps = vec![Step { coef: 1, lin: low7(127 - k) + one() - hb.clone(), theta: 8 }, Step { coef: 1, lin: low7(128 - k) + hb.clone(), theta: 8 }];
    g.c.node(&format!("{name}.ones7"), steps)
}

/// The read translation of a fetch address `a`, with `ones7` the wire "bits 0..6 of
/// `a` all set" and the regions as wires one level above pc: for `a` = pc itself
/// (`carry` None) from pc's bits, and for `a` = pc + k − hb from pc's bits 7..15 and
/// `carry`, the carry into bit 7, so they need not wait for the adder's high bits.
pub(super) fn read_address_fetch(g: &mut G, name: &str, a: &W, pc: &W, carry: Option<&Lin>, ones7: Lin, r: &Regs, dma: &Blocks) -> Physical {
    let mut regions = match carry {
        None => Regions::wired(g, name, a),
        Some(c7) => {
            // The region is an interval of s = pc[7..16] + carry, taken mod 512 (pc + k
            // wraps from FFFF to 0000 into ROM bank 0), in units of 128 bytes.
            let s = crate::circuit::value(&pc[7..], 9) + c7.clone();
            let mut wire = |f: &str, lo: i64, hi: i64| {
                let mut steps = vec![Step { coef: 1, lin: s.clone(), theta: lo }, Step { coef: -1, lin: s.clone(), theta: hi + 1 }];
                if lo == 0 {
                    steps.push(Step { coef: 1, lin: s.clone(), theta: 512 });
                }
                Cond::all(&[g.c.node(&format!("{name}.{f}"), steps)])
            };
            Regions { rom0: wire("rom0", 0, 127), romn: wire("romn", 128, 255), vram: wire("vram", 256, 319), cram: wire("cram", 320, 383), wram: wire("wram", 384, 507), hi: wire("hi", 508, 511), ones7: None, io: None, page_low: None }
        }
    };
    regions.ones7 = Some(ones7);
    build_in(g, name, a, r, dma, Access::Read, 0, regions, false, true)
}

/// FF00 + `n` in the I/O page and at IE. OAM DMA blocks nothing in the FF page.
fn ff_page_io(n: &W) -> (Cond, Cond) {
    (Cond::all(&[not(&n[7])]), Cond::all(n))
}

/// The read translation of FF00 + `n` while `gate` is 1 (LDH A,(n) and LD A,(C), with
/// `n` their byte under the gate and 0 otherwise), and VOID's first byte while it is 0:
/// HRAM, the I/O page in `IO` at its port code or VBK's and SVBK's readback in TAB
/// (`io_cases`) and IE in control's port, under OAM DMA too. VOID and the page share the page's bits, so the gate adds only VOID's bit 9.
/// The byte is 0 unless its instruction runs, so the port table reads it only then.
pub(super) fn ff_page_read(g: &mut G, name: &str, n: &W, gate: &Lin, (io_gate, io_low): (&Lin, &W), r: &Regs) -> Physical {
    let low = (0..7).fold(Lin::konst(0), |s, k| s + n[k].clone());
    // HRAM: bit 7 set and not all of bits 0..6 (FFFF is IE): 8·n7 − low ≥ 2.
    let hram = n[7].clone() * 8 - low;
    // The I/O page from `io_gate` (the gate with bit 7 clear) and the low bits under it.
    let io = Cond::all(&[io_gate.clone()]);
    let ie = ff_page_io(n).1;
    let pairs: Vec<(Lin, Lin)> = (0..7).map(|k| (io_low[k].clone(), io_gate.clone() - io_low[k].clone())).collect();
    let mut cases = io_cases(&g.c, &pairs, &io, r, Access::Read);
    cases.push(Case::new(io).ones(&IO_IN_PAGE));
    cases.push(Case::new(ie).ones(&IO_IN_PAGE).ones(&[CODE]).host());
    let node = |g: &mut G, k: usize, extra: Vec<Step>| g.c.node(&format!("{name}.{k}"), extra.into_iter().chain(bit_steps(&g.c, &cases, k)).collect());
    let bit7 = node(g, 7, Vec::new());
    let bit8 = node(g, 8, vec![Step { coef: 1, lin: hram, theta: 2 }]);
    let bit9 = node(g, 9, Vec::new()) + not(gate);
    let high: Vec<Lin> = (10..13).map(|k| node(g, k, Vec::new())).collect();
    let host = g.c.node(&format!("{name}.host"), flag_steps(&g.c, &cases, |c| c.host));
    let bits = (0..BITS)
        .map(|k| match k {
            0..7 => n[k].clone(),
            7 => bit7.clone(),
            8 => bit8.clone(),
            9 => bit9.clone(),
            10..13 => high[k - 10].clone(),
            _ if PAGE_BITS.contains(&k) => one(),
            _ => Lin::konst(0),
        })
        .collect();
    Physical { bits, host, record: Lin::konst(0), reg: Lin::konst(0), reg_steps: Vec::new() }
}

/// The write translation of FF00 + `n` (LDH (n),A), with the page fixed, so its cases
/// test the bits of `n` only, OAM DMA blocking nothing here: HRAM keeps a record,
/// VBK at FF4F and SVBK at FF70 on CGB reach their banking registers, the rest of the
/// I/O page lands in `IO` at its port code and IE in control's port.
pub(super) fn ff_page_write(g: &mut G, name: &str, n: &W, r: &Regs) -> Physical {
    let low = (0..7).fold(Lin::konst(0), |s, k| s + n[k].clone());
    let hram_form = n[7].clone() * 8 - low;
    let hram = g.c.ge(&format!("{name}.hram"), hram_form.clone(), 2);
    let served = g.c.ge(&format!("{name}.io"), Lin::konst(2) - hram_form.clone(), 1);
    let regs: Vec<(u8, Reg, Lin)> = r
        .io_registers()
        .into_iter()
        .enumerate()
        .map(|(i, (byte, reg, gate))| {
            let pattern = (0..8).fold(Lin::konst(0), |s, k| s + if byte >> k & 1 == 1 { n[k].clone() } else { not(&n[k]) });
            let theta = 8 + gate.len() as i64;
            (byte, reg, g.c.ge(&format!("{name}.{reg:?}{i}"), gate.into_iter().fold(pattern, |s, l| s + l), theta))
        })
        .collect();
    let (io, ie) = ff_page_io(n);
    // The port table reads the low bits gated by the I/O page, so a write to HRAM leaves
    // it still.
    let io_wire = g.c.ge(&format!("{name}.iow"), io.lin.clone(), io.theta);
    let low: Vec<(Lin, Lin)> = (0..7)
        .map(|k| {
            let l = g.c.ge(&format!("{name}.io{k}"), io.lin.clone() + n[k].clone(), io.theta + 1);
            (l.clone(), io_wire.clone() - l)
        })
        .collect();
    let mut cases = io_cases(&g.c, &low, &Cond::all(&[io_wire]), r, Access::Write);
    cases.push(Case::new(ie).ones(&[CODE]));
    let node = |g: &mut G, k: usize, extra: Option<Step>| g.c.node(&format!("{name}.{k}"), extra.into_iter().chain(bit_steps(&g.c, &cases, k)).collect());
    let ff = node(g, 7, None);
    let bit8 = node(g, 8, Some(Step { coef: 1, lin: hram_form, theta: 2 }));
    let (bit9, bit10) = (node(g, 9, None), node(g, 10, None));
    let reg = regs.iter().fold(Lin::konst(0), |s, (_, _, is)| s + is.clone());
    // A register write replaces the passed low bits with the register's index and the
    // I/O block's bit with REGS's bits 7 and 9.
    let bits = (0..BITS)
        .map(|k| match k {
            0..7 => regs.iter().fold(n[k].clone(), |s, (byte, reg, is)| s + is.clone() * (((*reg as i64) >> k & 1) - ((*byte as i64) >> k & 1))),
            7 => ff.clone() + reg.clone(),
            8 => bit8.clone(),
            9 => bit9.clone() + reg.clone(),
            10 => bit10.clone(),
            11 => served.clone() - reg.clone(),
            _ if PAGE_BITS.contains(&k) => one(),
            _ => Lin::konst(0),
        })
        .collect();
    let record = hram + reg.clone();
    Physical { bits, host: Lin::konst(0), record, reg, reg_steps: Vec::new() }
}

/// The read translation of a bus that carries one of `sources` (address and 0/1 gates,
/// at most one selected) and is zero otherwise; VOID when none is selected. Its regions
/// are wires at the bus's own level, read from the sources, so the translation sits one
/// level above the bus and its steps read a region wire, not the bus's high bits.
pub(super) fn read_address_muxed(g: &mut G, name: &str, a: &W, sources: &[(W, Vec<Lin>)], r: &Regs, dma: &Blocks) -> Physical {
    let regions = Regions::muxed(g, name, sources);
    build_in(g, name, a, r, dma, Access::Read, 0, regions, true, false)
}

/// The VOID block's own address: the byte there is zero and the model serves it.
pub(super) fn void_block() -> Physical {
    Physical { bits: (0..BITS).map(|k| Lin::konst((layout::VOID >> k & 1) as i64)).collect(), host: Lin::konst(0), record: Lin::konst(0), reg: Lin::konst(0), reg_steps: Vec::new() }
}

/// Two addresses of which exactly one is live and the other VOID, as the one live
/// address: the sum of the two minus VOID, affine over their wires.
pub(super) fn either(a: &Physical, b: &Physical) -> Physical {
    let v = void_block();
    Physical { bits: (0..BITS).map(|k| a.bits[k].clone() + b.bits[k].clone() - v.bits[k].clone()).collect(), host: a.host.clone() + b.host.clone(), record: Lin::konst(0), reg: Lin::konst(0), reg_steps: Vec::new() }
}

/// One translated address selected by mutually exclusive gates (0/1 forms, at most one
/// case firing), in one level; zero when none fires. Translating each stable source
/// once and selecting the physical results keeps a translation's nodes still until
/// their own source changes, where a translation of the selected bus address would
/// re-evaluate on every step whose selection changes.
/// A read keeps the host flag, a write the record flag.
pub(super) fn select(g: &mut G, name: &str, cases: &[(Vec<Lin>, &Physical)], access: Access) -> Physical {
    let flag = |p: &Physical| if access == Access::Read { p.host.clone() } else { p.record.clone() };
    let cases: Vec<(Vec<Lin>, W)> = cases.iter().map(|(gates, p)| (gates.clone(), p.bits.iter().cloned().chain([flag(p)]).collect())).collect();
    let mut out = g.c.mux(name, BITS + 1, &cases);
    let flag = out.pop().unwrap();
    match access {
        Access::Read => Physical { bits: out, host: flag, record: Lin::konst(0), reg: Lin::konst(0), reg_steps: Vec::new() },
        Access::Write => Physical { bits: out, host: Lin::konst(0), record: flag, reg: Lin::konst(0), reg_steps: Vec::new() },
    }
}

/// The physical address a write to `a` lands at, with `record` set when the model keeps
/// it. Its nodes sit no lower than level `floor`, so a translation of a level-0 source
/// can leave the level the fetch translations fill.
pub(super) fn write_address(g: &mut G, name: &str, a: &W, r: &Regs, dma: &Blocks, floor: u32, region: Region) -> Physical {
    build(g, name, a, r, dma, Access::Write, floor, region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{byte, Circuit};
    use crate::memory::layout::{self, translate, Banking, N_REGS};
    use std::collections::BTreeMap;

    /// A circuit that translates (AL, AH) under register inputs and `DB`'s high nibble as
    /// what OAM DMA blocks.
    fn circuit() -> Circuit {
        let mut c = Circuit::default();
        let al = byte(c.input("AL"));
        let ah = byte(c.input("AH"));
        let ctl = byte(c.input("CTL"));
        let db = byte(c.input("DB"));
        let dma = Blocks { oam: db[4].clone(), vram: db[5].clone(), main: db[6].clone(), wram: db[7].clone() };
        let a: W = al.iter().chain(ah.iter()).cloned().collect();
        let regs = layout::REG_NAMES.iter().map(|n| byte(c.input(n)).to_vec()).collect();
        let mut g = G { c, o: Vec::new(), cb: super::super::Cb::Hots(Vec::new()), mode: super::super::Mode::Memory };
        let regs = Regs::banked(&mut g, regs);
        for (access, name, region) in VARIANTS {
            let p = build(&mut g, name, &a, &regs, &dma, access, 0, region);
            for (i, chunk) in p.bits.chunks(8).enumerate() {
                let mut bits: Vec<Lin> = chunk.to_vec();
                bits.resize(8, Lin::konst(0));
                g.c.output(&format!("{name}{i}"), bits.try_into().unwrap());
            }
            g.c.output(&format!("{name}F"), [p.host, p.record, Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
        }
        // A muxed bus: the address while CTL bit 5 selects it, zero otherwise.
        let sel = ctl[5].clone();
        let bus: W = a.iter().enumerate().map(|(k, x)| g.c.and(&format!("bus.{k}"), &[x.clone(), sel.clone()])).collect();
        let p = read_address_muxed(&mut g, "M", &bus, &[(a.clone(), vec![sel.clone()])], &regs, &dma);
        for (i, chunk) in p.bits.chunks(8).enumerate() {
            let mut bits: Vec<Lin> = chunk.to_vec();
            bits.resize(8, Lin::konst(0));
            g.c.output(&format!("M{i}"), bits.try_into().unwrap());
        }
        g.c.output("MF", [p.host, Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
        // The FF page read of AL while CTL bit 5 selects it, AL under the gate.
        let n: W = al.iter().enumerate().map(|(k, x)| g.c.and(&format!("ffn.{k}"), &[x.clone(), sel.clone()])).collect();
        let io_gate = g.c.and("ffio", &[sel.clone(), one() - al[7].clone()]);
        let io_low: W = (0..7).map(|k| g.c.and(&format!("ffio.{k}"), &[al[k].clone(), sel.clone(), one() - al[7].clone()])).collect();
        let p = ff_page_read(&mut g, "P", &n, &sel, (&io_gate, &io_low), &regs);
        for (i, chunk) in p.bits.chunks(8).enumerate() {
            let mut bits: Vec<Lin> = chunk.to_vec();
            bits.resize(8, Lin::konst(0));
            g.c.output(&format!("P{i}"), bits.try_into().unwrap());
        }
        g.c.output("PF", [p.host, Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
        let p = ff_page_write(&mut g, "Q", &al.to_vec(), &regs);
        for (i, chunk) in p.bits.chunks(8).enumerate() {
            let mut bits: Vec<Lin> = chunk.to_vec();
            bits.resize(8, Lin::konst(0));
            g.c.output(&format!("Q{i}"), bits.try_into().unwrap());
        }
        g.c.output("QF", [Lin::konst(0), p.record, Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
        // Fetches at pc + k − hb (CTL bit 3 the HALT bug), with the regions from pc.
        let hb = ctl[3].clone();
        for k in 0..3u64 {
            let (fa, carry) = if k == 0 {
                (a.clone(), None)
            } else {
                let fa = g.c.sum_bits(&format!("pc+{k}"), 16, |n| crate::circuit::plus_const(&a, k, &hb, n));
                (fa, Some(g.c.ge(&format!("pc+{k}.c7"), crate::circuit::value(&a, 7) - hb.clone(), 128 - k as i64)))
            };
            let ones = ones7_ahead(&mut g, &format!("F{k}"), &a, k, &hb);
            let p = read_address_fetch(&mut g, &format!("F{k}"), &fa, &a, carry.as_ref(), ones, &regs, &dma);
            for (i, chunk) in p.bits.chunks(8).enumerate() {
                let mut bits: Vec<Lin> = chunk.to_vec();
                bits.resize(8, Lin::konst(0));
                g.c.output(&format!("F{k}.{i}"), bits.try_into().unwrap());
            }
            g.c.output(&format!("F{k}F"), [p.host, Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
        }
        assert_eq!(g.c.depth(), 3, "a translation is one level above the banks (which read the register inputs), two with wired or muxed regions, three after a fetch adder");
        g.c
    }

    /// Read and write translations with literal regions, then with wired ones.
    const VARIANTS: [(Access, &str, Region); 4] = [(Access::Read, "R", Region::Literal), (Access::Write, "W", Region::Literal), (Access::Read, "RW", Region::Wired), (Access::Write, "WW", Region::Wired)];

    #[test]
    fn circuit_translation_equals_the_specification() {
        let c = circuit();
        let mut rng = 0x1234_5678_9ABC_DEF1u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        let mut checked = 0;
        for _ in 0..20_000 {
            let r = draw();
            let m = mappers[(r % 5) as usize];
            let rom_len = 0x8000usize << (r >> 3) % 9;
            let ram_len = match (r >> 6) % 6 {
                0 => 0,
                1 => 0x200,
                2 => 0x800,
                3 => 0x2000,
                4 => 0x8000,
                _ => 0x2_0000,
            };
            let ram_len = if m == Mapper::Mbc2 { 0x200 } else { ram_len };
            let cgb = (r >> 9) & 1 == 1;
            let mut b = Banking::new(m, rom_len, ram_len, cgb).unwrap();
            if m == Mapper::Mbc3 && (r >> 20) & 1 == 1 {
                b = b.with_clock();
            }
            // Half of the MBC1 steps on a multicart's wiring (MASK1 bit 1).
            if m == Mapper::Mbc1 && (r >> 20) & 1 == 1 {
                b = b.multicart();
            }
            let regs = draw();
            for (i, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                let v = (regs >> (8 * i)) as u8;
                // Keep RAMG enabled half of the time so the enabled cases are exercised.
                b.set(reg, if reg == Reg::Ramg && v & 1 == 0 { 0x0A } else { v });
            }
            // A quarter of MBC3 steps select one of the clock's registers.
            if m == Mapper::Mbc3 && (r >> 21) % 4 == 0 {
                b.set(Reg::Ramb, 0x08 + (r >> 23) as u8 % 5);
            }
            // A third of steps run from power-on: the boot ROM mapped, or unmapped by FF50
            // into CGB mode or DMG compatibility, with KEY0 at random.
            if (r >> 25) % 3 == 0 {
                b.set(Reg::Booting, 1);
                b.set(Reg::Unmapped, ((r >> 27) % 4 == 0) as u8);
                b.set(Reg::Compat, ((r >> 29) % 4 == 0) as u8);
                b.set(Reg::Key0, (r >> 31) as u8);
            }
            // An eighth of the steps under OAM DMA, from a source on each bus in turn.
            const SOURCES: [u8; 6] = [0x00, 0x45, 0x80, 0xA0, 0xC1, 0xE3];
            let dma = if (r >> 10) % 8 == 0 { layout::Dma::during(SOURCES[(r >> 24) as usize % 6], cgb) } else { layout::Dma::NONE };
            let selected = (r >> 13) % 4 != 0;
            let hb = (r >> 15) & 1 == 1;
            // A quarter of the addresses sit just below a region boundary, where a fetch
            // at pc + k crosses it.
            const EDGES: [u16; 12] = [0x0000, 0x4000, 0x8000, 0xA000, 0xC000, 0xD000, 0xE000, 0xFE00, 0xFEA0, 0xFF00, 0xFF80, 0xFFFF];
            let d = draw();
            // An eighth sit in the I/O page, whose registers each have a port code.
            let a = match d % 8 {
                0 | 4 => EDGES[(d >> 8) as usize % 12].wrapping_sub((d >> 16) as u16 % 4),
                1 => 0xFF00 | (d >> 16) as u16 & 0x7F,
                2 => (d >> 16) as u16 % 0x0900,
                3 => [0xFF4C, 0xFF50, 0x00FF, 0x0100, 0x01FF, 0x0200, 0x08FF, 0x0900][(d >> 16) as usize % 8],
                _ => (d >> 16) as u16,
            };
            let mut input: BTreeMap<String, u8> = BTreeMap::new();
            input.insert("AL".into(), a as u8);
            input.insert("AH".into(), (a >> 8) as u8);
            input.insert("CTL".into(), (selected as u8) << 5 | (hb as u8) << 3);
            input.insert("DB".into(), dma.0 << 4);
            for i in 0..N_REGS {
                input.insert(layout::REG_NAMES[i].into(), b.0[i]);
            }
            let out = c.eval(&input);
            for (access, name, _) in VARIANTS {
                let got = out[&format!("{name}0")] as u32 | (out[&format!("{name}1")] as u32) << 8 | (out[&format!("{name}2")] as u32) << 16;
                let want = translate(&b, a, access, dma);
                assert_eq!(got, want, "{name} {access:?} of {a:#06x} on {m:?} regs {:02x?} dma {dma:?}: got {got:#x} want {want:#x}", b.0);
                let flags = out[&format!("{name}F")];
                if access == Access::Read {
                    assert_eq!(flags & 1 == 1, layout::port(want).is_some(), "host flag of {a:#06x} on {m:?}");
                }
                if access == Access::Write {
                    assert_eq!(flags >> 1 & 1 == 1, layout::recorded(want), "record flag of {a:#06x} on {m:?} regs {:02x?}", b.0);
                }
            }
            let got = out["M0"] as u32 | (out["M1"] as u32) << 8 | (out["M2"] as u32) << 16;
            let want = if selected { translate(&b, a, Access::Read, dma) } else { layout::VOID };
            assert_eq!(got, want, "muxed read of {a:#06x} (selected {selected}) on {m:?} dma {dma:?}");
            assert_eq!(out["MF"] & 1 == 1, layout::port(want).is_some(), "muxed host flag of {a:#06x}");
            let got = out["P0"] as u32 | (out["P1"] as u32) << 8 | (out["P2"] as u32) << 16;
            let want = if selected { translate(&b, 0xFF00 | a & 0xFF, Access::Read, dma) } else { layout::VOID };
            assert_eq!(got, want, "FF page read of {:#06x} (selected {selected}) dma {dma:?}", 0xFF00 | a & 0xFF);
            assert_eq!(out["PF"] & 1 == 1, layout::port(want).is_some(), "FF page host flag of {a:#06x}");
            let got = out["Q0"] as u32 | (out["Q1"] as u32) << 8 | (out["Q2"] as u32) << 16;
            let want = translate(&b, 0xFF00 | a & 0xFF, Access::Write, dma);
            assert_eq!(got, want, "FF page write of {:#06x} on {m:?} cgb {cgb} dma {dma:?}", 0xFF00 | a & 0xFF);
            assert_eq!(out["QF"] >> 1 & 1 == 1, layout::recorded(want), "FF page record flag of {:#06x}", 0xFF00 | a & 0xFF);
            for k in 0..3u16 {
                let fa = if k == 0 { a } else { a.wrapping_add(k).wrapping_sub(hb as u16) };
                let got = out[&format!("F{k}.0")] as u32 | (out[&format!("F{k}.1")] as u32) << 8 | (out[&format!("F{k}.2")] as u32) << 16;
                let want = layout::fetch(&b, fa, dma);
                assert_eq!(got, want, "fetch pc {a:#06x} + {k} (halt bug {hb}) = {fa:#06x} on {m:?} dma {dma:?}");
                assert_eq!(out[&format!("F{k}F")] & 1 == 1, layout::is_host(want), "fetch host flag of {fa:#06x}");
            }
            checked += 1;
        }
        assert_eq!(checked, 20_000);
    }
}

#[cfg(test)]
mod size {
    use super::tests_support::*;

    #[test]
    fn one_translation_is_one_level_of_a_few_hundred_steps() {
        let (c, read_nodes, read_steps, write_nodes, write_steps) = sizes();
        assert_eq!(c.depth(), 2, "the banks, then the translation");
        eprintln!("read translation: {read_nodes} nodes, {read_steps} steps; write translation: {write_nodes} nodes, {write_steps} steps");
        assert!(read_steps < 1500 && write_steps < 2500, "read {read_steps} write {write_steps}");
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use crate::circuit::{byte, Circuit};
    use crate::memory::layout;

    /// A circuit holding one read and one write translation, with their node and step counts.
    pub(super) fn sizes() -> (Circuit, usize, usize, usize, usize) {
        let mut c = Circuit::default();
        let al = byte(c.input("AL"));
        let ah = byte(c.input("AH"));
        let db = byte(c.input("DB"));
        let dma = Blocks { oam: db[4].clone(), vram: db[5].clone(), main: db[6].clone(), wram: db[7].clone() };
        let a: W = al.iter().chain(ah.iter()).cloned().collect();
        let regs = layout::REG_NAMES.iter().map(|n| byte(c.input(n)).to_vec()).collect();
        let mut g = G { c, o: Vec::new(), cb: super::super::Cb::Hots(Vec::new()), mode: super::super::Mode::Memory };
        let regs = Regs::banked(&mut g, regs);
        let count = |g: &G, from: usize| (g.c.nodes.len() - from, g.c.nodes[from..].iter().map(|n| n.steps.len()).sum::<usize>());
        let n0 = g.c.nodes.len();
        read_address(&mut g, "R", &a, &regs, &dma, Region::Literal);
        let (rn, rs) = count(&g, n0);
        let n1 = g.c.nodes.len();
        write_address(&mut g, "W", &a, &regs, &dma, 0, Region::Literal);
        let (wn, ws) = count(&g, n1);
        (g.c, rn, rs, wn, ws)
    }
}

/// The second device a write reaches, from its physical address `wp`: SVBK's record
/// (REGS) reaches control's FF70, LCDC's display byte reaches control's FF40, and on a
/// board with MBC3's clock RAMG's and RAMB's records reach the clock's enable and select
/// ports. Returns whether one does, and its port's physical address (0 when none). Each
/// case is one equality step over `wp`, and the address is a sum of the exclusive cases,
/// so a write step adds four steps.
pub(super) fn second_port(g: &mut G, name: &str, wp: &W, r: &Regs) -> (Lin, W) {
    let equals = |target: u32, from: usize| -> Step {
        let lin = (from..BITS).fold(Lin::konst(0), |s, k| s + if target >> k & 1 == 1 { wp[k].clone() } else { not(&wp[k]) });
        Step { coef: 1, lin, theta: (BITS - from) as i64 }
    };
    let port = |code: u32, low7: u32| layout::IO + code * layout::BLOCK + low7;
    let clock = r.has_clock();
    let with_clock = |s: Step| Step { coef: 1, lin: s.lin + clock.clone(), theta: s.theta + 1 };
    let cases = [
        ("svbk", equals(layout::REGS + Reg::Svbk as u32, 0), port(layout::IO_CONTROL, 0x70)),
        ("lcdc", equals(port(layout::IO_LCDC, 0), 7), port(layout::IO_CONTROL, 0x40)),
        ("ramg", with_clock(equals(layout::REGS + Reg::Ramg as u32, 0)), port(layout::IO_CLOCK_ENABLE, 0)),
        ("ramb", with_clock(equals(layout::REGS + Reg::Ramb as u32, 0)), port(layout::IO_CLOCK_SELECT, 0)),
        ("key0", equals(layout::REGS + Reg::Key0 as u32, 0), port(layout::IO_CONTROL, 0x4C)),
    ];
    let wires: Vec<(Lin, u32)> = cases.into_iter().map(|(case, step, at)| (g.c.node(&format!("{name}.{case}"), vec![step]), at)).collect();
    let heard = wires.iter().fold(Lin::konst(0), |s, (w, _)| s + w.clone());
    let bits = (0..BITS).map(|k| wires.iter().filter(|(_, at)| at >> k & 1 == 1).fold(Lin::konst(0), |s, (w, _)| s + w.clone())).collect();
    (heard, bits)
}
