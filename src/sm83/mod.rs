//! The GBC CPU as a threshold circuit over the slot layout the checkpoint decodes.
//!
//! Levels fall out of the logic: opcode one-hots and constant-offset 16-bit adds (1),
//! operand selection and interrupt priority (2), the ALU (3), zero flags of bitwise
//! results (4), and one final selection per output field. Every control signal is a
//! free linear sum of opcode one-hots, so decode costs a single level.
//!
//! Two circuits share these stages. In host mode the host fills the opcode bytes and
//! the data bytes into input slots, and a step is two unmasking passes over the same
//! network: the request slots depend only on registers and instruction bytes, so they
//! are read before the host fills D0/D1. In memory mode (`build_mem`) the opcode and
//! data bytes come from the model's own memory through attention heads, each reading
//! an 8-byte line (`docs/memory-in-weights.md` §5). The fetch addresses are translated
//! at level 2. Column-select heads put the opcode byte of PC's line on a bus, and its
//! nibble one-hots sit a level later. The operand bytes come from PC's line too unless
//! they cross into the next line, whose first two bytes other column heads read, or
//! land on FFFF. The data heads read at layer 6, and each write reads its line and
//! records the line after it (`fields::same_line`). The host fills a data byte only for
//! the addresses it serves, and keeps an opcode byte it serves as a record in the HOST
//! block.
//!
//! The stages run in a fixed order because wires are numbered as they are created,
//! and the checkpoint's layout follows that numbering.

use crate::circuit::{one, Circuit, Fit, Lin};

type W = Vec<Lin>;

mod abi;
mod alu;
pub mod d4;
mod datapath;
mod decode;
mod fields;
mod line;
pub mod oam_bug;
mod translate;

pub use abi::{bits, checkpoint, mem_passes, oam_event_byte, Battery, AUDIT_SLOTS, SAVED, SAVE_WINDOW, decode_outcome, decode_request, encode, engine, CheckpointCore, CircuitCore, CircuitMemCore, Evidence, In, InMem, MemCheckpointCore, Out, OutMem, Record, Rq, Sm83, INPUTS, INPUTS_MEM, REQUEST, RESULT, RESULT_MEM};

/// Where the opcode and data bytes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The host reads them and fills the input slots.
    Host,
    /// The model reads them from its memory; the host fills only the bytes it serves.
    Memory,
}

/// The CB-prefixed opcode's decode. Every CB control signal is a function of the
/// operation `x` (rotate, bit, res, set) with its index `y`, and of the register `z`,
/// so the fields' one-hots carry the same signals in 48 wires that 256 one-hots do. In
/// host mode the one-hots come free with the OP1 token, so that mode keeps them.
enum Cb {
    Hots(Vec<Lin>),
    Fields {
        /// `x << 3 | y`, 32 wires.
        xy: Vec<Lin>,
        /// The register, 8 wires.
        z: Vec<Lin>,
        /// The ops that write register `z` (every `x` but bit), 8 wires.
        writes: Vec<Lin>,
        /// Any CB op: the prefix one-hot.
        any: Lin,
    },
}

/// The circuit under construction with the opcode one-hots every control signal sums.
struct G {
    c: Circuit,
    o: Vec<Lin>,
    cb: Cb,
    mode: Mode,
}

impl G {
    /// The sum of the one-hots of the opcodes (CB aside) `pred` picks. In memory mode the
    /// fetched byte's one-hots are exactly one 1 (ANDs of its two nibbles' one-hots), so a
    /// set larger than its complement is written as 1 − CB − the complement: a node reading
    /// it moves only when an opcode outside the set enters or leaves.
    fn ops(&self, pred: impl Fn(u8) -> bool) -> Lin {
        let picked = (0..=255u8).filter(|&op| op != 0xCB && pred(op)).count();
        if self.mode == Mode::Memory && 255 - picked + 1 < picked {
            return (0..=255u8).filter(|&op| op != 0xCB && !pred(op)).fold(one() - self.o[0xCB].clone(), |a, op| a - self.o[op as usize].clone());
        }
        (0..=255u8).filter(|&op| op != 0xCB && pred(op)).fold(Lin::konst(0), |a, op| a + self.o[op as usize].clone())
    }
    fn op(&self, op: u8) -> Lin {
        self.o[op as usize].clone()
    }
    fn cb_hots(&self, pred: impl Fn(u8) -> bool) -> Lin {
        let Cb::Hots(q) = &self.cb else { unreachable!() };
        (0..=255u8).filter(|&op| pred(op)).fold(Lin::konst(0), |a, op| a + q[op as usize].clone())
    }
    /// The CB ops with operation `x` and index `y`, on any register.
    fn cb_xy(&self, x: u8, y: u8) -> Lin {
        match &self.cb {
            Cb::Hots(_) => self.cb_hots(|cb| cb >> 6 == x && (cb >> 3) & 7 == y),
            Cb::Fields { xy, .. } => xy[(x << 3 | y) as usize].clone(),
        }
    }
    /// The CB ops on register `r`.
    fn cb_z(&self, r: u8) -> Lin {
        match &self.cb {
            Cb::Hots(_) => self.cb_hots(|cb| cb & 7 == r),
            Cb::Fields { z, .. } => z[r as usize].clone(),
        }
    }
    /// The CB ops that write register `r`: rotates, res and set.
    fn cb_writes(&self, r: u8) -> Lin {
        match &self.cb {
            Cb::Hots(_) => self.cb_hots(|cb| cb >> 6 != 1 && cb & 7 == r),
            Cb::Fields { writes, .. } => writes[r as usize].clone(),
        }
    }
    /// Any CB op.
    fn cb_any(&self) -> Lin {
        match &self.cb {
            Cb::Hots(_) => self.cb_hots(|_| true),
            Cb::Fields { any, .. } => any.clone(),
        }
    }
    /// A byte per step from per-opcode tables (`not` with the branch not taken, `taken`
    /// for the four conditional families taken, `cb` per CB opcode), as linear sums of
    /// the exclusive one-hots, so each bit costs no node; `taken` holds the families'
    /// taken wires (JR, RET, JP, CALL).
    fn table_byte(&self, not: &[u8; 256], taken_t: &[u8; 256], cb: &[u8; 256], taken: [&Lin; 4]) -> W {
        (0..8)
            .map(|k| {
                let bit = |t: &[u8; 256], op: u8| ((t[op as usize] >> k) & 1) as i64;
                let mut s = self.ops(|op| bit(not, op) == 1);
                for (t, base) in taken.into_iter().zip([0x20u8, 0xC0, 0xC2, 0xC4]) {
                    s = s + t.clone() * (bit(taken_t, base) - bit(not, base));
                }
                s + self.cb_cycles_bit(cb, k)
            })
            .collect()
    }
    /// Bit `k` of the M-cycle count of the CB ops, summed over their selectors.
    fn cb_cycles_bit(&self, table: &[u8; 256], k: usize) -> Lin {
        let bit = |cb: u8| ((table[cb as usize] >> k) & 1) as i64;
        match &self.cb {
            Cb::Hots(q) => (0..=255u8).fold(Lin::konst(0), |s, cb| s + q[cb as usize].clone() * bit(cb)),
            Cb::Fields { .. } => {
                // The count depends only on whether the operand is (HL) and whether the
                // operation is bit: the three groups below, each constant in the table.
                let group = |pick: fn(u8) -> bool| {
                    let mut it = (0..=255u8).filter(|&cb| pick(cb)).map(|cb| table[cb as usize]);
                    let first = it.next().unwrap();
                    assert!(it.all(|v| v == first), "CB cycles vary within a group");
                    ((first >> k) & 1) as i64
                };
                let (hl, bit_hl) = (self.cb_z(6), self.cb_z(6) - self.cb_writes(6));
                (self.cb_any() - hl.clone()) * group(|cb| cb & 7 != 6) + bit_hl * group(|cb| cb & 7 == 6 && cb >> 6 == 1) + self.cb_writes(6) * group(|cb| cb & 7 == 6 && cb >> 6 != 1)
            }
        }
    }
    /// The opcodes of ALU operation `k` (ADD, ADC, SUB, SBC, AND, XOR, OR, CP), on a
    /// register or an immediate.
    fn alu(&self, k: u8) -> Lin {
        self.ops(|op| {
            let (x, y, z) = decode::xyz(op);
            (x == 2 || x == 3 && z == 6) && y == k
        })
    }
    /// Operand isolation: datapaths used by few opcodes see zeros otherwise, so changes
    /// to pc, sp or the immediates do not reach them on every instruction.
    fn isolate(&mut self, name: &str, xs: &[Lin], when: &Lin) -> W {
        xs.iter().enumerate().map(|(k, x)| self.c.and(&format!("{name}.{k}"), &[x.clone(), when.clone()])).collect()
    }
}

/// Levels the circuit's nodes within their slack, and in memory mode fits its live
/// wires to the channel budget and gathers each changed wire's readers into fewer
/// layers (`Circuit::gather`), weighing each wire by its changes in `table`
/// (`changes`).
fn finish(mut g: G, mode: Mode, table: &str) -> Circuit {
    // The memory circuit builds a few values no one reads (a fetch translation's host
    // flag, a carry past the top bit): their units would change with their inputs.
    if mode == Mode::Memory {
        g.c.prune_unread();
    }
    g.c.schedule(crate::llama::UNIT_QUANTUM);
    // The memory circuit's live wires, the class wires among them until the class heads
    // read them, fit hidden 1,024 with dead wires' channels reused; failing that, 2,048.
    // The host circuit's cannot, since its 256 one-hot token channels leave 677 for 937.
    if mode == Mode::Memory {
        let changes = changes_in(&g.c, table);
        let budget = |c: &Circuit, n| crate::llama::Model::wire_budget(c, true, n);
        if !g.c.fit(crate::llama::UNIT_QUANTUM, &Fit { channels: budget(&g.c, 1), changes: &changes }) {
            g.c.fit(crate::llama::UNIT_QUANTUM, &Fit { channels: budget(&g.c, 2), changes: &changes });
        }
        // Readers of one changed wire gathered into fewer layers, within the widths the
        // placement already has: no boundary gains a live wire past the widest, no level
        // a unit past the widest.
        let live = g.c.live_per_boundary().into_iter().max().unwrap_or(0);
        let units = g.c.units_per_level().into_iter().max().unwrap_or(0);
        g.c.gather(&Fit { channels: live, changes: &changes }, units);
    }
    g.c
}

/// Per wire of the default memory circuit, its changes per fast pass: the mean of
/// text.sh and overworld.sh's model windows on `--core model-mem`, recorded with
/// `GBCFORMER_UNITS=1 GBCFORMER_WIRE_CHANGES=<file>` and keyed by `unique_names`
/// (`changes-memory.txt`). A
/// wire the record lacks (it never changed, or it is new since) weighs a thousandth of a
/// change. Re-record after a circuit change.
pub fn changes(c: &Circuit) -> Vec<f64> {
    changes_in(c, include_str!("changes-memory.txt"))
}

fn changes_in(c: &Circuit, table: &str) -> Vec<f64> {
    let by_name: std::collections::HashMap<&str, f64> = table
        .lines()
        .filter_map(|l| l.split_once(' ')).filter_map(|(n, f)| Some((n, f.parse().ok()?))).collect();
    crate::circuit::unique_names(&c.names).iter().map(|n| by_name.get(n.as_str()).copied().unwrap_or(0.0) + 1e-3).collect()
}

/// Refuses the retired switch `GBCFORMER_BITS` (the 0/1 circuits, since removed). A run
/// that still sets it expects weights it would not get.
pub fn refuse_retired_switches() -> Result<(), String> {
    if std::env::var_os("GBCFORMER_BITS").is_some() {
        return Err("GBCFORMER_BITS is retired: the 0/1 GBC CPU circuits were removed, and the nibble circuits are the only GBC CPU; unset it".into());
    }
    Ok(())
}

fn refuse() {
    if let Err(e) = refuse_retired_switches() {
        panic!("{e}");
    }
}

/// The circuit the host feeds opcode and data bytes (`d4`, `docs/byte-channels.md`).
pub fn build() -> Circuit {
    refuse();
    d4::build()
}

/// The circuit that reads opcode and data bytes from the model's memory.
pub fn build_mem() -> Circuit {
    refuse();
    d4::build_mem()
}

#[cfg(test)]
mod tests {
    use super::build_mem;
    use crate::memory::layout::{is_void, translate, Access, Banking, Mapper, Reg, LINE, LINE_BITS, REGS};
    use std::collections::BTreeMap;

    /// `Inputs::dma_buses` for a transfer that takes every bus below FE00, running or not.
    fn all_buses(dma: bool) -> u8 {
        let all = crate::memory::layout::Dma::ALL.0;
        all | if dma { all << 4 } else { 0 }
    }

    /// Every step that writes twice, at every SP (a push and an interrupt dispatch, SP−1
    /// then SP−2) and every a16 (LD (a16),SP, a16 then a16+1), under five bankings with
    /// OAM DMA off and on: each record's line must be the image's line with the write's
    /// byte put in, write 0's record applied first, and a record kept exactly when
    /// `layout::translate` lands the write outside the marker blocks. `CircuitMemCore`
    /// holds each record to that before applying it (`commit`); this drives it through
    /// the line boundaries, the registers' line, IE beside HRAM and the wrap at FFFF.
    #[test]
    fn two_writes_leave_the_reference_lines() {
        use super::CircuitMemCore;
        use crate::memory::image::Image;
        use crate::memory::layout::HRAM;
        use crate::step::{Core, Cpu, Inputs};
        let bank = |m: Mapper, ram: usize, cgb: bool, regs: &[(Reg, u8)]| {
            let mut b = Banking::new(m, 0x10000, ram, cgb).unwrap();
            for &(r, v) in regs {
                b.set(r, v);
            }
            (b, ram)
        };
        let bankings = [
            bank(Mapper::Mbc1, 0x8000, false, &[(Reg::Ramg, 0x0A), (Reg::Romb0, 3)]),
            bank(Mapper::Mbc5, 0x20000, true, &[(Reg::Ramg, 0x0A), (Reg::Ramb, 3), (Reg::Vbk, 1), (Reg::Svbk, 5)]),
            bank(Mapper::Mbc2, 0x200, false, &[(Reg::Ramg, 0x0A)]),
            bank(Mapper::Mbc3, 0x8000, true, &[(Reg::Ramg, 0x0A), (Reg::Ramb, 0x08), (Reg::Svbk, 0)]),
            bank(Mapper::Rom, 0x2000, false, &[]),
        ];
        let run = |(b, ram): (Banking, usize), dma: bool| {
            // The program sits in HRAM, which the GBC CPU reads under OAM DMA too.
            let mut image = Image::new(&[0; 0x10000], ram, b);
            for (k, op) in [0xC5u8, 0x08, 0x00].into_iter().enumerate() {
                image.write(HRAM + 4 * k as u32, op);
            }
            image.mark();
            let mut core = CircuitMemCore::new(image);
            let mut checked = 0;
            for x in 0..=0xFFFFu16 {
                for k in 0..3u16 {
                    core.image.revert();
                    if k == 1 {
                        core.image.write(HRAM + 5, x as u8);
                        core.image.write(HRAM + 6, (x >> 8) as u8);
                    }
                    let dispatch = k == 2;
                    let cpu = Cpu { sp: x, pc: 0xFF80 + 4 * k, b: 0x5A, c: 0xC3, ime: dispatch, ..Cpu::default() };
                    let inp = Inputs { op: [0; 3], ie: dispatch as u8, iflag: dispatch as u8, dma, dma_buses: all_buses(dma), ..Inputs::default() };
                    core.request(&cpu, &inp);
                    let out = core.execute(&cpu, &inp, [0, 0]);
                    assert!(out.writes.iter().all(Option::is_some), "{:?} dma {dma}, step {k} at {x:#06x}: two writes", b.mapper());
                    // PUSH BC and the dispatch write SP−1 then SP−2, LD (a16),SP a16 then a16+1.
                    let want = if k == 1 { [x, x.wrapping_add(1)] } else { [x.wrapping_sub(1), x.wrapping_sub(2)] };
                    assert_eq!(out.writes.map(|w| w.map(|(a, _)| a)), want.map(Some), "{:?} dma {dma}, step {k} at {x:#06x}: the bus addresses", b.mapper());
                    checked += 1;
                }
            }
            checked
        };
        let checked: usize = std::thread::scope(|s| {
            let runs: Vec<_> = bankings.iter().flat_map(|&b| [false, true].map(|dma| s.spawn(move || run(b, dma)))).collect();
            runs.into_iter().map(|r| r.join().unwrap()).sum()
        });
        assert_eq!(checked, 10 * 3 * 0x10000);
    }

    /// Every defined opcode and every CB opcode, run from HRAM beside the reference
    /// interpreter under random registers, flags and operands: the next state, both
    /// writes, the M-cycles and the idle FE cycles must be the reference's. Every pointer
    /// (BC, DE, HL, SP, an a16 or FF00+n operand that is an address) stays in HRAM, so the
    /// circuit reads every byte from its own lines. This holds the rules the opcode sets
    /// are written by (`G::ops`' complement sums, the immediates isolated to their opcodes,
    /// the merged CALL and RST pushes) and the pruning of unread nodes to the instruction
    /// set, opcode by opcode.
    #[test]
    fn every_opcode_from_hram_matches_the_reference() {
        use super::CircuitMemCore;
        use crate::memory::image::Image;
        use crate::memory::layout::{HRAM, HRAM_LEN};
        use crate::reference::Reference;
        use crate::step::{Core, Cpu, Inputs};
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let hram: [u8; HRAM_LEN] = std::array::from_fn(|_| next() as u8);
        let mut image = Image::new(&[0; 0x10000], 0x2000, Banking::new(Mapper::Rom, 0x10000, 0x2000, false).unwrap());
        for (k, &b) in hram.iter().enumerate() {
            image.write(HRAM + k as u32, b);
        }
        image.mark();
        let mut core = CircuitMemCore::new(image);
        const UNDEFINED: [u8; 11] = [0xD3, 0xDB, 0xDD, 0xE3, 0xE4, 0xEB, 0xEC, 0xED, 0xF4, 0xFC, 0xFD];
        let programs: Vec<(u8, Option<u8>)> = (0..=255u8).filter(|op| *op != 0xCB && !UNDEFINED.contains(op)).map(|op| (op, None)).chain((0..=255u8).map(|cb| (0xCB, Some(cb)))).collect();
        // A pointer into HRAM clear of the program at FF80 and of the stack's two bytes either side.
        let pointer = |x: u64| 0xFF90 + (x % 0x60) as u16;
        let mut checked = 0;
        for &(op, cb) in &programs {
            for _ in 0..48 {
                let (r, s) = (next(), next());
                let mut operands = match op {
                    0x08 | 0xEA | 0xFA => pointer(s >> 8).to_le_bytes(),
                    0xE0 | 0xF0 => [pointer(s >> 8) as u8, (s >> 40) as u8],
                    _ => [(s >> 8) as u8, (s >> 16) as u8],
                };
                if let Some(cb) = cb {
                    operands[0] = cb;
                }
                let program = [op, operands[0], operands[1]];
                core.image.revert();
                let mut mem = hram;
                for (k, &b) in program.iter().enumerate() {
                    core.image.write(HRAM + k as u32, b);
                    mem[k] = b;
                }
                let [bc, de, hl, sp] = [0, 1, 2, 3].map(|k| pointer(r >> (8 * k)));
                let cpu = Cpu { a: (r >> 32) as u8, f: (r >> 40) as u8 & 0xF0, b: (bc >> 8) as u8, c: bc as u8, d: (de >> 8) as u8, e: de as u8, h: (hl >> 8) as u8, l: hl as u8, sp, pc: 0xFF80, ime: r >> 48 & 1 == 1, ..Cpu::default() };
                let inp = Inputs { op: program, ..Inputs::default() };
                let mut reference = Reference;
                let reads = reference.request(&cpu, &inp).reads;
                let want = reference.execute(&cpu, &inp, reads.map(|a| a.map_or(0, |a| mem[(a - 0xFF80) as usize])));
                core.request(&cpu, &inp);
                let got = core.execute(&cpu, &inp, [0, 0]);
                assert_eq!(got, want, "{program:02x?} from {cpu:02x?}");
                checked += 1;
            }
        }
        assert_eq!(checked, (244 + 256) * 48);
    }

    /// POP BC from HRAM with OAM DMA ending at its M-cycle 2: read 0 (M-cycle 1) is
    /// blocked and read 1 is not. With the memory circuit beside the reference, which
    /// reads the bus, the step asks again under read 1's state after the transfer's end,
    /// and read 0 must still be served at its own M-cycle: C reads 0xFF, B the stack's 0x34.
    #[test]
    fn a_read_before_the_transfers_end_is_served_at_its_own_m_cycle() {
        use crate::machine::{Hardware, Machine};
        use crate::step::{Bus, Cpu};
        let mut m = Machine::new(vec![0u8; 0x8000], None, Hardware::Dmg).unwrap();
        m.write(0xFF40, 0x00);
        for (a, v) in [(0xC000, 0x12), (0xC001, 0x34), (0xFF80, 0xC1)] {
            m.write(a, v);
        }
        let image = m.memory_image().unwrap();
        let mut core = crate::lockstep::Lockstep::new(crate::reference::Reference, super::CircuitMemCore::with_circuit(build_mem(), image));
        m.write(0xFF46, 0xC0);
        m.advance(160);
        let mut cpu = Cpu { pc: 0xFF80, sp: 0xC000, ..Cpu::default() };
        crate::step::step(&mut core, &mut cpu, &mut m);
        assert_eq!((cpu.b, cpu.c, cpu.sp), (0x34, 0xFF, 0xC002), "POP BC across the transfer's end");
    }

    /// Each data access translates under its own bus state: random access locks, OAM DMA
    /// and per-access overrides (`Inputs::access`) on reads, read pairs, writes and write
    /// pairs at VRAM's, OAM's and HRAM's edges. Every emitted read and write address is
    /// `layout::translate_locked`'s under `step::slot_state`, `CircuitMemCore::commit`
    /// holds each record to the line that state leaves (write 1 keeps write 0's byte only
    /// when both are kept), and RC bit 3 says whether a data access reaches VRAM or the FE
    /// page by its bus address; WC bit 3 stays clear, since the program runs in HRAM.
    #[test]
    fn each_access_translates_under_its_own_state() {
        use super::CircuitMemCore;
        use crate::memory::image::Image;
        use crate::memory::layout::{translate_locked, HRAM};
        use crate::step::{slot_state, Core, Cpu, Inputs, OVERRIDE};
        let mut b = Banking::new(Mapper::Mbc5, 0x10000, 0x2000, true).unwrap();
        b.set(Reg::Ramg, 0x0A);
        b.set(Reg::Vbk, 1);
        let mut image = Image::new(&[0; 0x10000], 0x2000, b.clone());
        image.mark();
        let mut core = CircuitMemCore::new(image);
        let edges = [0x7FFFu16, 0x8000, 0x8001, 0x9FFE, 0x9FFF, 0xA000, 0xC7FF, 0xFDFF, 0xFE00, 0xFE01, 0xFE9F, 0xFEA0, 0xFEFF, 0xFF00, 0xFF46, 0xFF47, 0xFF80, 0xFF81, 0xFFFE];
        // Opcode, its operand, and which slots it uses (bit per read 0, read 1, write 0, write 1).
        let ops: [(u8, u8); 11] = [(0x7E, 0b0001), (0x2A, 0b0001), (0xC1, 0b0011), (0x77, 0b0100), (0x34, 0b0101), (0xC5, 0b1100), (0x08, 0b1100), (0xFA, 0b0001), (0x02, 0b0100), (0xF0, 0b0001), (0xE0, 0b0100)];
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let (mut overrides, mut mixed) = (0, 0);
        for trial in 0..60_000u32 {
            let r = draw();
            let (op, uses) = ops[trial as usize % ops.len()];
            let addr = |r: u64| if r % 3 == 0 { (r >> 8) as u16 } else { edges[(r >> 8) as usize % edges.len()].wrapping_add((r >> 16) as u16 % 3).wrapping_sub(1) };
            let (x, y) = (addr(r), addr(draw()));
            let n = x as u8;
            core.image.revert();
            for (k, byte) in [op, n, (x >> 8) as u8].into_iter().enumerate() {
                core.image.write(HRAM + k as u32, byte);
            }
            let cpu = Cpu { pc: 0xFF80, sp: y, h: (x >> 8) as u8, l: x as u8, b: (x >> 8) as u8, c: x as u8, a: r as u8, ..Cpu::default() };
            let nibble = |r: u64, s: u32| if r >> (20 + 4 * s) & 1 == 1 { OVERRIDE | (r >> (21 + 4 * s)) as u8 & 7 } else { 0 };
            let s = draw();
            let access = [nibble(s, 0) | nibble(s, 1) << 4, nibble(s, 2) | nibble(s, 3) << 4, 0];
            // The transfer's source on each bus in turn, so each access's blocked buses vary.
            let class = crate::memory::layout::Dma::during([0x00u8, 0x80, 0xA0, 0xC1, 0xE3][(s >> 40) as usize % 5], true).0;
            let inp = Inputs { dma: s & 1 == 1, dma_buses: class | if s & 1 == 1 { class << 4 } else { 0 }, locks: Some((s >> 1) as u8 & 3), access, ..Inputs::default() };
            overrides += (access != [0; 3]) as u32;
            let req = core.request(&cpu, &inp);
            let at = |a: u16| (0x8000..=0x9FFF).contains(&a) || (0xFE00..=0xFEFF).contains(&a);
            let out = core.execute(&cpu, &inp, [0, 0]);
            let mut lockable = false;
            for i in 0..2 {
                if let Some(a) = req.reads[i] {
                    lockable |= at(a);
                    let st = slot_state(&inp, i);
                    assert_eq!(core.evidence.read_addr[i], translate_locked(&b, a, Access::Read, crate::step::slot_dma(&inp, i), st), "{op:#04x} read {i} of {a:#06x} under {st:#x}");
                }
                if let Some((a, _)) = out.writes[i] {
                    lockable |= at(a) && !matches!(op, 0xE0);
                    let st = slot_state(&inp, 2 + i);
                    assert_eq!(core.evidence.write_addr[i], translate_locked(&b, a, Access::Write, crate::step::slot_dma(&inp, 2 + i), st), "{op:#04x} write {i} to {a:#06x} under {st:#x}");
                }
            }
            assert_eq!([req.reads[0].is_some(), req.reads[1].is_some(), out.writes[0].is_some(), out.writes[1].is_some()].iter().enumerate().fold(0, |m, (k, &u)| m | (u as u8) << k), uses, "{op:#04x}: its accesses");
            assert_eq!(req.at_lock, lockable, "{op:#04x} at {x:#06x} sp {y:#06x}: RC bit 3");
            assert!(!req.fetch_lock, "{op:#04x} at {x:#06x} sp {y:#06x}: WC bit 3 from HRAM");
            mixed += (uses == 0b1100 && slot_state(&inp, 2) != slot_state(&inp, 3)) as u32;
        }
        assert!(overrides > 50_000 && mixed > 5_000, "{overrides} steps with an override, {mixed} write pairs under two states");
    }

    /// The fetched bytes come from PC's line, the next line or FFFF's: every fetch byte
    /// equals the byte at its fetch address (`layout::fetch`), and each host flag says
    /// whether the host serves it, and where it does, the address the model emits for the
    /// host's byte (`FP`) is the fetch address translated as a read, at every PC whose stream ends in FFF8..FFFF or crosses
    /// a line, with and without the HALT bug, under OAM DMA from each bus and the access
    /// locks at the step's start and at each operand's own M-cycle (`FS`), in the nibble and
    /// the 0/1 circuits. WC bit 3 says whether PC's line or the next reaches VRAM or the FE
    /// page, where the locks at an operand's M-cycle matter.
    /// SingleStepTests reaches FFFF as OP1 in six cases and as OP2 in none, and never under DMA.
    #[test]
    fn fetch_bytes_come_from_their_lines() {
        for c in [build_mem()] {
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
            let mut pcs: Vec<u16> = (0xFFE0..=0xFFFF).chain(0x0000..0x0010).chain([0x3FF8, 0x3FFE, 0x3FFF, 0x7FFE, 0x7FFF, 0x9FFF, 0xBFFF, 0xCFFF, 0xDFFE, 0xDFFF, 0xFDF7, 0xFDFE, 0xFDFF, 0xFE00, 0xFE9F, 0xFEFE, 0xFEFF, 0xFF7E, 0xFF7F]).collect();
            for _ in 0..200 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                pcs.push(rng as u16 | 6 | (rng >> 16) as u16 & 1);
            }
            for (i, &pc) in pcs.iter().enumerate() {
                let seed = rng.wrapping_add(i as u64 * 0x9E37_79B9);
                let m = mappers[i % 5];
                let ram = if m == Mapper::Mbc2 { 0x200 } else { 0x2000 };
                let mut b = Banking::new(m, 0x8000 << (seed >> 3) % 6, ram, seed >> 9 & 1 == 1).unwrap();
                for (k, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                    b.set(reg, (seed >> (8 * k + 10)) as u8);
                }
                let regs = b.0;
                // Every other case's opcode is JP a16, which fetches both operands.
                let jp_at = (i % 2 == 0).then(|| crate::memory::layout::fetch_locked(&b, pc, crate::memory::layout::Dma::NONE, 0));
                let byte = move |a: u32| {
                    if Some(a) == jp_at && !crate::memory::layout::is_ff(a) && !is_void(a) && !crate::memory::layout::is_io(a) {
                        0xC3
                    } else if (REGS..REGS + 2 * LINE as u32).contains(&a) {
                        regs.get((a - REGS) as usize).copied().unwrap_or(0)
                    } else if crate::memory::layout::is_blocked(a) || crate::memory::layout::is_ff(a) {
                        0xFF
                    } else if is_void(a) || crate::memory::layout::is_io(a) {
                        0
                    } else {
                        (seed.wrapping_mul(a as u64 + 1).wrapping_add(a as u64 * 0x9E37) >> 24) as u8
                    }
                };
                // OAM DMA from a source on each bus in turn, the access locks, and each operand
                // fetch's state (`FS`, a bus state as `step::slot_state`'s: bit 0 VRAM locked,
                // bit 1 OAM locked, bit 2 OAM DMA): the step's start, or the transfer or a lock
                // ended by its M-cycle or begun by it.
                const SOURCES: [u8; 6] = [0x00, 0x45, 0x80, 0xA0, 0xC1, 0xE3];
                let class = crate::memory::layout::Dma::during(SOURCES[i % 6], true);
                let states: [(u8, [Option<u8>; 2]); 12] = [
                    (0, [None; 2]),
                    (4, [None; 2]),
                    (4, [Some(0); 2]),
                    (0, [Some(4); 2]),
                    (4, [None, Some(0)]),
                    (0, [None, Some(4)]),
                    (3, [None; 2]),
                    (0, [Some(3); 2]),
                    (3, [Some(0); 2]),
                    (0, [None, Some(1)]),
                    (1, [Some(2), None]),
                    (7, [Some(3), Some(0)]),
                ];
                let start_lock = |a: u16, st: u8| (st & 1 != 0 && (0x8000..=0x9FFF).contains(&a)) || (st & 2 != 0 && (0xFE00..=0xFE9F).contains(&a));
                let lockable = |a: u16| (0x8000..=0x9FFF).contains(&a) || (0xFE00..=0xFEFF).contains(&a);
                for hb in [false, true] {
                    for (start, fetches) in states {
                        let ctl = (hb as u8) << 3 | (start >> 2) << 4 | (start & 3) << 5 | 0x80;
                        let nibble = |s: Option<u8>| s.map_or(0, |st| 8 | st);
                        let fs = nibble(fetches[0]) | nibble(fetches[1]) << 4;
                        let db = class.0 | if start & 4 != 0 { class.0 << 4 } else { 0 };
                        let ins = BTreeMap::from([("PCL".to_string(), pc as u8), ("PCH".to_string(), (pc >> 8) as u8), ("CTL".to_string(), ctl), ("DB".to_string(), db), ("FS".to_string(), fs)]);
                        let out = c.eval_mem(&ins, &|n| std::array::from_fn(|k| byte(n << LINE_BITS | k as u32)));
                        let on = |st: u8| if st & 4 != 0 { class } else { crate::memory::layout::Dma::NONE };
                        let next = (pc & !7).wrapping_add(8);
                        assert_eq!(out["WC"] >> 3 & 1 == 1, lockable(pc) || lockable(next), "{m:?} pc {pc:#06x}: WC bit 3");
                        for k in 0..3u16 {
                            // An opcode its start blocked reads 0xFF, RST 38H, which fetches no operand.
                            if k > 0 && fetches[k as usize - 1].is_some() && (start & 4 != 0 && class.blocks(pc) || start_lock(pc, start)) {
                                continue;
                            }
                            let a = if k == 0 { pc } else { pc.wrapping_add(k).wrapping_sub(hb as u16) };
                            let st = if k == 0 { start } else { fetches[k as usize - 1].unwrap_or(start) };
                            let (dma, locks) = (on(st), st & 3);
                            let p = crate::memory::layout::fetch_locked(&b, a, dma, locks);
                            assert_eq!(out[&format!("OP{k}'")], byte(p), "{m:?} pc {pc:#06x} fetch {k} at {a:#06x} (hb {hb}, start {start}, fetches {fetches:?}, {class:?}) = {p:#x}");
                            // A fetched operand on the FE page that OAM DMA does not block is
                            // an OAM bug event (CTL' bits 6 and 7).
                            if k > 0 {
                                let fetched = [out["RC"], out["WC"]][k as usize - 1] >> 7 & 1 == 1;
                                let event = fetched && a >> 8 == 0xFE && !(st & 4 != 0 && class.blocks(a));
                                assert_eq!(out["CTL'"] >> (5 + k) & 1 == 1, event, "{m:?} pc {pc:#06x} fetch {k} at {a:#06x}: the FE page event (hb {hb}, start {start}, fetches {fetches:?}, fetched {fetched})");
                            }
                            assert_eq!(out["RQ"] >> (4 + k) & 1 == 1, crate::memory::layout::is_host(p), "{m:?} pc {pc:#06x} fetch {k} host flag (hb {hb}, {dma:?})");
                            if crate::memory::layout::is_host(p) {
                                let fp = ["a", "b", "c"].iter().enumerate().fold(0u32, |v, (j, s)| v | (out[&format!("FP{k}{s}")] as u32) << (8 * j));
                                assert_eq!(fp, crate::memory::layout::translate_locked(&b, a, crate::memory::layout::Access::Read, dma, locks), "{m:?} pc {pc:#06x} fetch {k} port (hb {hb}, {dma:?}, locks {locks})");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The idle M-cycles' FE page events (MC bits 3-6) held to the reference interpreter's
    /// `Outcome::idle` about the FE page: JR e and JR NZ/Z taken or not, with offsets of both
    /// signs and with and without the HALT bug, and an interrupt dispatch (PC + 1, and SP − 2
    /// at the FE page). SingleStepTests never dispatches and rarely puts PC near FE00.
    #[test]
    fn idle_fe_events_match_the_reference() {
        use crate::memory::layout::{fetch_locked, Dma};
        use crate::step::{Core, Cpu, Inputs};
        let c = build_mem();
        let b = Banking::new(Mapper::Mbc5, 0x8000, 0x2000, true).unwrap();
        let regs = b.0;
        let pcs: Vec<u16> = (0xFDF0..=0xFE10).chain(0xFE80..=0xFE9D).chain([0xC0F0, 0xC0FE, 0xC0FF, 0xFD7F, 0xFD80, 0xFC80]).collect();
        let (mut cases, mut events) = (0, 0);
        for &pc in &pcs {
            for op in [0x18u8, 0x20, 0x28, 0x00] {
                for e in [0x00u8, 0x01, 0x10, 0x7F, 0x80, 0x90, 0xF0, 0xFF] {
                    for (hb, z, dispatch) in [(false, false, false), (false, true, false), (true, false, false), (true, true, false), (false, false, true)] {
                        if dispatch && op != 0 {
                            continue;
                        }
                        let sp = if dispatch { [0xFE02u16, 0xFE01, 0xFE00, 0xFF00, 0xC100][e as usize % 5] } else { 0xC100 };
                        let cpu = Cpu { f: if z { 0x80 } else { 0 }, sp, pc, ime: dispatch, halt_bug: hb, ..Cpu::default() };
                        let mem = [(pc, op), (pc.wrapping_add(1), e)];
                        let at = |a: u16| mem.iter().find(|m| m.0 == a).map_or(0, |m| m.1);
                        let inp = Inputs { op: cpu.fetched(at), ie: dispatch as u8, iflag: dispatch as u8, ..Inputs::default() };
                        let placed: Vec<(u32, u8)> = (0..3).map(|k| cpu.fetch_address(k)).map(|a| (fetch_locked(&b, a, Dma::NONE, 0), at(a))).collect();
                        let byte = |p: u32| if (REGS..REGS + 2 * LINE as u32).contains(&p) { regs.get((p - REGS) as usize).copied().unwrap_or(0) } else { placed.iter().find(|x| x.0 == p).map_or(0, |x| x.1) };
                        let ins: BTreeMap<String, u8> = super::INPUTS_MEM.iter().map(|n| n.to_string()).zip(super::abi::encode_mem(&cpu, &inp, [0; 3])).collect();
                        let out = c.eval_mem(&ins, &|n| std::array::from_fn(|k| byte(n << LINE_BITS | k as u32)));
                        let mc = out["MC"];
                        let got: [Option<u8>; 2] = [8u8, 16].map(|m| (mc & m != 0).then_some(mc >> 5 & 3));
                        let mut r = crate::reference::Reference;
                        r.request(&cpu, &inp);
                        let want = r.execute(&cpu, &inp, [0, 0]).idle;
                        assert_eq!(got, want, "pc {pc:#06x} op {op:02x} e {e:02x} hb {hb} z {z} dispatch {dispatch} sp {sp:#06x}: the idle FE page events");
                        cases += 1;
                        events += want.iter().flatten().count();
                    }
                }
            }
        }
        assert!(events > 100, "{events} idle FE page events in {cases} cases");
    }

    /// The DMA line inputs reach only `DP` and `DQ`, in the nibble and the 0/1 memory
    /// circuits: a GBC CPU pass decodes nothing they move, so `dma_line` leaves them at the
    /// last line instead of zeroing them, and the CPU passes between do not re-evaluate the
    /// DMA translations.
    /// `OB`, the OAM bug's event byte, likewise reaches only the OAM pass's slots.
    #[test]
    fn dma_inputs_reach_only_the_dma_slots() {
        let circuits = [build_mem()];
        for (c, ins, outs) in circuits.iter().flat_map(|c| [(c, &["DSL", "DSH", "DDL", "DDH"][..], &["DP", "DQ"][..]), (c, &["OB"][..], &["OT", "OW", "OL"][..])]) {
            let dma: Vec<usize> = ins.iter().map(|n| c.inputs.iter().position(|(name, _)| name == n).expect("a DMA input")).collect();
            let mut tainted: std::collections::HashSet<u32> = dma.iter().flat_map(|&i| c.inputs[i].1.iter().map(|s| s.0)).collect();
            tainted.extend(c.values.iter().filter(|(i, _, _)| dma.contains(i)).map(|(_, _, s)| s.0));
            tainted.extend(c.onehots.iter().filter(|(i, _)| dma.contains(i)).flat_map(|(_, w)| w.iter().map(|s| s.0)));
            let reads = |l: &crate::circuit::Lin, t: &std::collections::HashSet<u32>| l.terms.keys().any(|s| t.contains(&s.0));
            loop {
                let before = tainted.len();
                for n in &c.nodes {
                    if n.steps.iter().any(|s| reads(&s.lin, &tainted)) {
                        tainted.insert(n.out.0);
                    }
                }
                for h in &c.heads {
                    if let crate::circuit::Query::Wires(w) = &h.query
                        && w.iter().any(|l| reads(l, &tainted))
                    {
                        tainted.extend(h.out.iter().map(|s| s.0));
                    }
                }
                if tainted.len() == before {
                    break;
                }
            }
            for (k, (name, byte)) in c.outputs.iter().enumerate() {
                let moved = byte.iter().any(|l| reads(l, &tainted)) || c.value_outputs[k].as_ref().is_some_and(|v| v.iter().any(|l| reads(l, &tainted)));
                assert!(!moved || outs.iter().any(|o| name.starts_with(o)), "{name} reads {ins:?}");
            }
        }
    }

    /// The memory checkpoint's OAM pass, through the engine and the core's sequence, rewrites
    /// the rows the circuit's OAM pass rewrites, with the step oracle on each such pass.
    #[test]
    fn the_checkpoint_oam_pass_rewrites_what_the_circuit_does() {
        use crate::engine::Engine;
        use crate::llama::Model;
        use crate::memory::attn::Params;
        use crate::memory::image::Image;
        use crate::memory::layout::OAM;
        use crate::step::Core;
        let mut rng = crate::memory::Rng(0x0A3B_06C0_FFEE_1234);
        let b = Banking::new(Mapper::Mbc1, 0x8000, 0x2000, false).unwrap();
        let mut circuit = super::CircuitMemCore::with_circuit(build_mem(), Image::new(&[0; 0x8000], 0x2000, b));
        let mut engine = Engine::new(Model::compile_mem(&build_mem(), Params::memory()));
        // No uniform check falls in the test; an OAM pass asks for its own (`check_next`).
        engine.step_oracle = u64::MAX;
        engine.insert(circuit.image.clone());
        let mut core = super::MemCheckpointCore::new(engine);
        // An OAM pass comes between a step's passes: the step (a NOP at 0000) sets the inputs.
        let (cpu, inp) = (crate::step::Cpu::default(), crate::step::Inputs::default());
        core.request(&cpu, &inp);
        core.execute(&cpu, &inp, [0; 2]);
        for (row, read) in [(8u8, false), (0x48, true), (0x98, false), (0x10, true), (0x10, true)] {
            let oam: Vec<u8> = (0..0xA0).map(|_| rng.draw() as u8).collect();
            circuit.image.fill(OAM, &oam);
            core.model_mut().insert(circuit.image.clone());
            assert_eq!(core.oam_bug(row, read), circuit.oam_bug(row, read), "row {row:#04x}, read {read}");
            let model = core.model();
            assert_eq!(model.memory.as_ref().unwrap().image.region(OAM, 0xA0), circuit.image.region(OAM, 0xA0), "row {row:#04x}, read {read}");
            assert_eq!(model.stats.fallback_passes, 0, "refused: {:?}", model.stats.refused);
        }
        assert_eq!(core.model().stats.forced_checks, 5);
    }

    /// The memory circuit's OAM pass rewrites its OAM cells as `ppu::corrupt_oam` rewrites
    /// OAM, for every scanned row and both kinds, over random OAM, in the nibble and the
    /// 0/1 circuits.
    #[test]
    fn the_oam_pass_corrupts_oam_as_the_reference_does() {
        use crate::machine::ppu::{corrupt_oam, OamBug};
        use crate::memory::image::Image;
        use crate::step::Core;
        let mut rng = crate::memory::Rng(0x0A3B_06C0_FFEE_1234);
        for c in [build_mem()] {
            let b = Banking::new(Mapper::Mbc1, 0x8000, 0x2000, false).unwrap();
            let mut core = super::CircuitMemCore::with_circuit(c, Image::new(&[0; 0x8000], 0x2000, b));
            for row in (8..0xA0).step_by(8) {
                for read in [false, true] {
                    for _ in 0..20 {
                        let oam: Vec<u8> = (0..0xA0).map(|_| rng.draw() as u8).collect();
                        core.image.fill(crate::memory::layout::OAM, &oam);
                        let mut want: [u8; 0xA0] = oam.clone().try_into().unwrap();
                        corrupt_oam(&mut want, row, if read { OamBug::Read } else { OamBug::Write });
                        core.oam_bug(row as u8, read);
                        assert_eq!(core.image.region(crate::memory::layout::OAM, 0xA0), &want[..], "row {row:#04x}, read {read}");
                    }
                }
            }
        }
    }

    /// A DMA pass's `DP` and `DQ` are the source and destination lines translated as reads
    /// without OAM DMA, under random registers of every mapper and CGB or not, at every
    /// region's edges and random lines, with OAM DMA's flag and the GBC CPU's access locks
    /// (CTL bits 5 to 7) set or not on the step's inputs: a DMA is held by neither.
    #[test]
    fn dma_lines_are_translated_as_reads() {
        let c = build_mem();
        let mut rng = 0x7F4A_7C15_2545_F491u64;
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        let mut lines: Vec<u16> = [0x0000, 0x3FF8, 0x4000, 0x7FF8, 0x8000, 0x9FF8, 0xA000, 0xBFF8, 0xC000, 0xCFF8, 0xD000, 0xDFF8, 0xE000, 0xF000, 0xFDF8, 0xFE00, 0xFE98, 0xFEA0, 0xFEF8, 0xFF80].to_vec();
        for _ in 0..60 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            lines.push(rng as u16 & !7);
        }
        for trial in 0..40u64 {
            let seed = rng.wrapping_add(trial * 0x9E37_79B9);
            let m = mappers[trial as usize % 5];
            let ram = if m == Mapper::Mbc2 { 0x200 } else { [0x800, 0x2000, 0x8000][(seed >> 50) as usize % 3] };
            let mut b = Banking::new(m, 0x8000 << (seed >> 3) % 6, ram, seed >> 9 & 1 == 1).unwrap();
            for (k, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                b.set(reg, (seed >> (8 * k + 10)) as u8);
            }
            if seed >> 62 & 1 == 1 {
                b.set(Reg::Ramg, 0x0A);
            }
            let regs = b.0;
            let byte = move |a: u32| if (REGS..REGS + 2 * LINE as u32).contains(&a) { regs.get((a - REGS) as usize).copied().unwrap_or(0) } else { 0 };
            for (i, &source) in lines.iter().enumerate() {
                let dest = lines[(i * 7 + trial as usize) % lines.len()];
                let dma = (i + trial as usize) % 3 == 0;
                let ins = BTreeMap::from([("DSL", source as u8), ("DSH", (source >> 8) as u8), ("DDL", dest as u8), ("DDH", (dest >> 8) as u8), ("CTL", (dma as u8) << 4 | (((i as u64 * 0x9E37 + trial) % 8) as u8) << 5)].map(|(n, v)| (n.to_string(), v)));
                let out = c.eval_mem(&ins, &|n| std::array::from_fn(|k| byte(n << LINE_BITS | k as u32)));
                let at = |p: &str| ["a", "b", "c"].iter().enumerate().fold(0u32, |v, (j, s)| v | (out[&format!("{p}{s}")] as u32) << (8 * j));
                let spec = |a: u16| crate::memory::layout::translate(&b, a, crate::memory::layout::Access::Read, false);
                assert_eq!([at("DP"), at("DQ")], [spec(source), spec(dest)], "{m:?} {source:#06x} -> {dest:#06x} under {:?}", b.0);
            }
        }
    }

    /// A memory record runs the circuit on copy-mode inputs (every input slot zero)
    /// over whatever the memory holds. Every node must stay 0/1 there too, since an
    /// out-of-range wire fires the output routing at the reader's position; `eval_mem`
    /// asserts it. Only the banking registers are assumed consistent (one mapper,
    /// masked banks) and the VOID block zero, as the layout guarantees for a cartridge:
    /// a head keyed on VOID adds a zero byte to another head's or the host's.
    #[test]
    fn copy_mode_inputs_keep_every_node_in_range() {
        let c = build_mem();
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        for _ in 0..300 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let seed = rng;
            let m = mappers[(seed % 5) as usize];
            let ram = if m == Mapper::Mbc2 { 0x200 } else { 0x2000 };
            let mut b = Banking::new(m, 0x8000 << (seed >> 3) % 6, ram, seed >> 9 & 1 == 1).unwrap();
            for (i, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                b.set(reg, (seed >> (8 * i + 10)) as u8);
            }
            let regs = b.0;
            // The registers' second line holds two registers and no other token.
            let byte = move |a: u32| {
                if (REGS..REGS + 2 * LINE as u32).contains(&a) {
                    regs.get((a - REGS) as usize).copied().unwrap_or(0)
                } else if is_void(a) || crate::memory::layout::is_io(a) {
                    0
                } else {
                    (seed.wrapping_mul(a as u64 + 1).wrapping_add(a as u64 * 0x9E37) >> 24) as u8
                }
            };
            let out = c.eval_mem(&BTreeMap::new(), &|n| std::array::from_fn(|k| byte(n << LINE_BITS | k as u32)));
            // The MAP slots name each banked window's first byte as the specification maps it.
            for (window, base) in [("MAP0", 0x0000u16), ("MAPR", 0x4000), ("MAPS", 0xA000), ("MAPW", 0xD000)] {
                let got = ["a", "b", "c"].iter().enumerate().fold(0u32, |acc, (k, s)| acc | (out[&format!("{window}{s}")] as u32) << (8 * k));
                assert_eq!(got, translate(&b, base, Access::Read, false), "{window} under {:?}", b.0);
            }
        }
    }

    /// LDH A,(n) and LD A,(C) at every n, on DMG and on CGB (VBK 1, SVBK 5), with OAM DMA
    /// off and on, through the whole memory circuit: read 0's emitted address is
    /// `layout::translate`'s, the host flag says whether a port serves it, and a byte the
    /// model serves (HRAM, an unmapped register, VBK, SVBK, the FF block) reaches A.
    #[test]
    fn ff_page_reads_land_where_the_specification_says() {
        use crate::memory::layout::{constant, is_io, port};
        let c = build_mem();
        for cgb in [false, true] {
            let mut b = Banking::new(Mapper::Mbc3, 0x8000, 0x2000, cgb).unwrap().with_clock();
            if cgb {
                b.set(Reg::Vbk, 1);
                b.set(Reg::Svbk, 5);
            }
            let regs = b.0;
            // In HRAM, which OAM DMA leaves visible.
            let pc = 0xFF80u16;
            let fetched = translate(&b, pc, Access::Read, false);
            for op in [0xF0u8, 0xF2] {
                for n in 0..=0xFFu8 {
                    let byte = move |a: u32| match a {
                        _ if (REGS..REGS + 2 * LINE as u32).contains(&a) => regs.get((a - REGS) as usize).copied().unwrap_or(0),
                        _ if a == fetched => op,
                        _ if a == fetched + 1 => n,
                        _ => constant(a).unwrap_or(if is_io(a) { 0 } else { (a as u64).wrapping_mul(0x9E37_79B9) as u8 >> 1 | 1 }),
                    };
                    for dma in [false, true] {
                        let ins = BTreeMap::from([("PCL".to_string(), pc as u8), ("PCH".to_string(), (pc >> 8) as u8), ("CTL".to_string(), (dma as u8) << 4), ("DB".to_string(), all_buses(dma)), ("C".to_string(), n)]);
                        let out = c.eval_mem(&ins, &|l| std::array::from_fn(|k| byte(l << LINE_BITS | k as u32)));
                        let a = 0xFF00 | n as u16;
                        let want = translate(&b, a, Access::Read, dma);
                        let got = out["RP0a"] as u32 | (out["RP0b"] as u32) << 8 | (out["RP0c"] as u32) << 16;
                        assert_eq!(got, want, "{op:#04x} at {a:#06x} (cgb {cgb}, dma {dma})");
                        assert_eq!(out["RQ"] >> 2 & 1 == 1, port(want).is_some(), "{op:#04x} at {a:#06x}: host flag (cgb {cgb}, dma {dma})");
                        if port(want).is_none() {
                            assert_eq!(out["A'"], byte(want), "{op:#04x} at {a:#06x}: the byte the model serves (cgb {cgb}, dma {dma})");
                        }
                    }
                }
            }
        }
    }

    /// A DMG cartridge that reads LY with LDH A,(n), stores it in HRAM with LDH (n),A and
    /// loops, run for `steps` steps; the machine after them.
    fn ly_loop(core: impl crate::step::Core, steps: usize) -> crate::machine::Machine {
        use crate::machine::{GameBoy, Hardware, Machine};
        let mut rom = vec![0u8; 0x8000];
        rom[0x100..0x106].copy_from_slice(&[0xF0, 0x44, 0xE0, 0x80, 0x18, 0xFA]);
        let mut gb = GameBoy::new(core, Machine::new(rom, None, Hardware::Dmg).unwrap());
        for _ in 0..steps {
            gb.step();
        }
        gb.machine
    }

    fn ly_image() -> crate::memory::image::Image {
        ly_loop(crate::reference::Reference, 0).memory_image().unwrap()
    }

    /// The host serves LY from the port the memory circuit's read address names (control's),
    /// under the audit that holds each emitted address to the reference translation, and
    /// HRAM ends as the reference interpreter leaves it.
    #[test]
    fn the_host_reads_the_port_the_circuit_names() {
        let model = ly_loop(super::CircuitMemCore::with_circuit(build_mem(), ly_image()), 3000);
        let reference = ly_loop(crate::reference::Reference, 3000);
        assert_eq!(model.ram_byte(0xFF80), reference.ram_byte(0xFF80), "the byte LDH A,(LY) read");
        assert_ne!(reference.ram_byte(0xFF80), 0, "LY has moved past line 0");
    }

    /// The same circuit with read 0's emitted port code changed from control's (1) to the
    /// APU's (2), bits 7 and 8 of the RP0 slots flipped: the host reads the APU, and the
    /// memory audit stops the run at that read.
    #[test]
    #[should_panic(expected = "read 0 of 0xff44: physical address")]
    fn a_wrong_port_in_the_emitted_address_stops_the_audit() {
        use crate::circuit::one;
        let mut c = build_mem();
        for (slot, bit) in [("RP0a", 7), ("RP0b", 0)] {
            let j = c.outputs.iter().position(|(n, _)| n == slot).unwrap();
            assert!(c.value_outputs[j].is_none(), "{slot} is a bit output");
            let b = &mut c.outputs[j].1[bit];
            *b = one() - b.clone();
        }
        ly_loop(super::CircuitMemCore::with_circuit(c, ly_image()), 3000);
    }
}
