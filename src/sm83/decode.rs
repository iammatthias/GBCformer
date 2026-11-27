//! The input slots as wires, the opcode one-hots, the immediates and the step's mode,
//! plus the opcode tables read off the instruction encoding and the reference
//! interpreter. In memory mode (`Mode::Memory`) the opcode bytes and the data bytes are
//! read from the model's memory by attention heads, so this stage also builds the
//! banking-register heads, the fetch translations, the request address and the data
//! heads.

use super::translate::{self, Physical, Region, Regs};
use super::{Mode, G, W};
use super::line::{self, pattern, Pick, Read};
use crate::circuit::{cat, one, zero, Circuit, Lin, Line, Query, Step, LINE, LINE_BITS};
use crate::memory::layout;
use crate::reference::Reference;
use crate::step::{Core, Cpu, Inputs};
use std::collections::BTreeMap;

pub(super) const ILLEGAL: [u8; 11] = [0xD3, 0xDB, 0xDD, 0xE3, 0xE4, 0xEB, 0xEC, 0xED, 0xF4, 0xFC, 0xFD];

pub(super) fn xyz(op: u8) -> (u8, u8, u8) {
    (op >> 6, (op >> 3) & 7, op & 7)
}

pub(super) fn len(op: u8) -> u64 {
    let (x, y, z) = xyz(op);
    match (x, z) {
        (0, 1) if y & 1 == 0 => 3,
        _ if matches!(op, 0x08 | 0xC2 | 0xCA | 0xD2 | 0xDA | 0xC3 | 0xC4 | 0xCC | 0xD4 | 0xDC | 0xCD | 0xEA | 0xFA) => 3,
        (0, 6) | (3, 6) => 2,
        _ if matches!(op, 0x18 | 0x20 | 0x28 | 0x30 | 0x38 | 0xE0 | 0xF0 | 0xE8 | 0xF8 | 0xCB) => 2,
        _ => 1,
    }
}

/// M-cycles per opcode with the branch not taken and taken, read off the reference
/// interpreter (which matches SingleStepTests) rather than retyped.
pub(super) fn cycle_table() -> ([u8; 256], [u8; 256], [u8; 256]) {
    let run = |stream: [u8; 3], f: u8| {
        let cpu = Cpu { f, sp: 0xC000, pc: 0x4000, ..Cpu::default() };
        let inp = Inputs { op: stream, ..Inputs::default() };
        Reference.execute(&cpu, &inp, [0, 0]).mcycles
    };
    let (mut not, mut taken, mut cb) = ([0; 256], [0; 256], [0; 256]);
    for op in 0..=255u8 {
        // F = 0 and F = 0xF0 make every condition true once and false once; a taken
        // branch is always the longer of the two.
        let (a, b) = (run([op, 0, 0], 0x00), run([op, 0, 0], 0xF0));
        (not[op as usize], taken[op as usize]) = (a.min(b), a.max(b));
        cb[op as usize] = run([0xCB, op, 0], 0);
    }
    (not, taken, cb)
}

/// The request's access-cycle bytes, `RC` (read 0's M-cycle in bits 0-2, read 1's in
/// bits 4-6) and `WC` (the writes' alike), per opcode with the branch not taken and taken,
/// per CB opcode, and for an interrupt dispatch, read off the reference interpreter
/// (`Request::cycles`, held to SingleStepTests' bus logs). An access the step does
/// not make has 0. A halted step makes none.
pub(super) fn access_table() -> ([[u8; 2]; 256], [[u8; 2]; 256], [[u8; 2]; 256], [u8; 2]) {
    // Bit 7 of each byte counts the operand fetches: RC's for one or more, WC's for two.
    let pack = |c: [u8; 2], fetch: bool| c[0] | c[1] << 4 | (fetch as u8) << 7;
    let run = |cpu: Cpu, inp: Inputs| {
        let at = Reference.request(&cpu, &inp).cycles.expect("the reference gives its access cycles");
        let mc = Reference.execute(&cpu, &inp, [0, 0]).mcycles;
        ([pack(at.reads, at.fetches >= 1), pack(at.writes, at.fetches == 2)], mc)
    };
    let step = |stream: [u8; 3], f: u8| run(Cpu { f, sp: 0xC000, pc: 0x4000, ..Cpu::default() }, Inputs { op: stream, ..Inputs::default() });
    let (mut not, mut taken, mut cb) = ([[0; 2]; 256], [[0; 2]; 256], [[0; 2]; 256]);
    for op in 0..=255u8 {
        // As in `cycle_table`: the taken branch is the longer of F = 0 and F = 0xF0.
        let (a, b) = (step([op, 0, 0], 0x00), step([op, 0, 0], 0xF0));
        let (short, long) = if a.1 <= b.1 { (a, b) } else { (b, a) };
        (not[op as usize], taken[op as usize]) = (short.0, long.0);
        cb[op as usize] = step([0xCB, op, 0], 0).0;
    }
    let cpu = Cpu { ime: true, sp: 0xC000, pc: 0x4000, ..Cpu::default() };
    let dispatch = run(cpu, Inputs { ie: 1, iflag: 1, ..Inputs::default() }).0;
    (not, taken, cb, dispatch)
}

/// One data access's translation state: the registers with its access locks, and
/// whether OAM DMA blocks it.
pub(super) struct Slot {
    pub(super) regs: Regs,
    pub(super) dma: translate::Blocks,
}

/// What the memory stage of the decode produced: the physical addresses, the raw bytes
/// the heads read and the host flags, for the result slots and the request.
pub(super) struct MemoryDecode {
    pub(super) regs: Regs,
    /// The registers and OAM DMA's block each data access is translated under (read 0,
    /// read 1, write 0, write 1): the step-start state, or the access's own where the
    /// host's `AS0`/`AS1` nibble overrides it. The fetches keep the step-start state.
    pub(super) slots: [Slot; 4],
    /// Each fetch lands in HOST: the host must keep that opcode byte there.
    pub(super) fetch_void: [Lin; 3],
    /// Where each fetched byte the host serves comes from (`FP0`..`FP2`): its bus
    /// address, pc + k − hb, translated as a read.
    pub(super) fetch_ports: [W; 3],
    /// PC's line or the next one reaches VRAM or the FE page: a sum of 0/1 wires, nonzero
    /// when the access locks at an operand's M-cycle matter.
    pub(super) fetch_lock: Lin,
    /// Whether OAM DMA blocks the FE page at each fetch: the opcode's at the step's start,
    /// each operand's under its own state (`FS`).
    pub(super) fetch_oam: [Lin; 3],
    /// Each operand k at pc + k − hb lies in the next line (`F1N`, `F2N`), and that line's
    /// bus address bits, line(PC) + 1.
    pub(super) next_line: [Lin; 2],
    pub(super) next: W,
    /// A DMA line's source and destination (`DS`, `DD`) translated as reads (`DP`,
    /// `DQ`): a pass that decodes them places a DMA line by the model's address bits.
    pub(super) dma_lines: [W; 2],
    /// The OAM bug's event byte (`OB`), which only the OAM pass's slots read
    /// (`oam_bug::memory_pass`).
    pub(super) ob: W,
    pub(super) ra0: W,
    /// Read 0 and read 1 (a pop's second byte) requested, before `exec` is applied.
    pub(super) rq0: Lin,
    pub(super) rq1: Lin,
    pub(super) rp: [Physical; 2],
    /// The bytes the data heads read, before the host's late byte is added.
    pub(super) read: [W; 2],
    /// SP+1, the second read address, which the datapath reuses for the request slot.
    pub(super) sp_p1: W,
    /// JR, or JR cc with its condition true, from the opcode one-hots: one wire, which
    /// gates the offset bits.
    pub(super) jr_taken: Lin,
    /// Read 0's bus and late byte, which the operand selections read directly
    /// (`line::mux`).
    pub(super) d0: Read,
}

/// The wires every later stage reads: input bytes, immediates and the step's mode.
pub(super) struct Decoded {
    pub(super) a: W,
    pub(super) f: W,
    /// Operand registers by their 3-bit index, with (HL) read through D0.
    pub(super) regs: [W; 8],
    pub(super) bc: W,
    pub(super) de: W,
    pub(super) hl: W,
    pub(super) sp: W,
    pub(super) pc: W,
    /// IME, EI pending, halted and the HALT bug in bits 0 to 3.
    pub(super) ctl: W,
    pub(super) op0: W,
    pub(super) op1: W,
    pub(super) d0: W,
    pub(super) d1: W,
    pub(super) imm8: W,
    pub(super) imm16: W,
    /// Dispatch to interrupt n, as the step's start decides it.
    pub(super) disp: W,
    /// The vector the dispatch jumps to: interrupt n's after the high byte's push, none
    /// (0000) when that push cleared every pending bit of IE.
    pub(super) vector: W,
    /// An enabled interrupt is pending, whatever IME says.
    pub(super) any: Lin,
    pub(super) exec: Lin,
    pub(super) dispatch: Lin,
    pub(super) wake: Lin,
    pub(super) memory: Option<MemoryDecode>,
}

pub(super) fn decode(mode: Mode) -> (G, Decoded) {
    let mut c = Circuit::default();
    let mut hot: BTreeMap<&str, Vec<Lin>> = BTreeMap::new();
    let names: &[&str] = match mode {
        Mode::Host => &super::INPUTS,
        Mode::Memory => &super::abi::INPUTS_MEM,
    };
    let inp: BTreeMap<&str, W> = names
        .iter()
        .map(|&n| {
            let bits = if mode == Mode::Host && matches!(n, "OP0" | "OP1") {
                let (bits, h) = c.input_onehot(n);
                hot.insert(n, h.into_iter().map(Lin::from).collect());
                bits
            } else {
                c.input(n)
            };
            (n, bits.map(Lin::from).to_vec())
        })
        .collect();
    let i = |n: &str| inp[n].clone();
    let a = i("A");
    let ctl = i("CTL");
    let (ime, halted, hb) = (ctl[0].clone(), ctl[2].clone(), ctl[3].clone());
    let (ie, iflag) = (i("IE"), i("IF"));
    let pc = cat(&i("PCL"), &i("PCH"));
    let sp = cat(&i("SPL"), &i("SPH"));

    // Interrupt mode. m_k = IE_k ∧ IF_k; dispatch takes the lowest pending bit.
    let m: W = (0..5).map(|k| c.and(&format!("irq{k}"), &[ie[k].clone(), iflag[k].clone()])).collect();
    let disp: W = (0..5)
        .map(|n| {
            let mut xs = vec![one() - halted.clone(), ime.clone(), m[n].clone()];
            xs.extend((0..n).map(|j| one() - m[j].clone()));
            c.and(&format!("disp{n}"), &xs)
        })
        .collect();
    let any = c.or("irq.any", &m);
    let dispatch = disp.iter().cloned().fold(zero(), |s, x| s + x);
    // The vector is chosen after the high byte's push: with SP 0 that push writes IE, and
    // the bits PC's high byte leaves enabled pick it; none left jumps to 0000 and
    // acknowledges nothing (SameBoy v1.0.5, mooneye `interrupts/ie_push`).
    let sp0 = c.ge("sp=0", zero() - sp.iter().cloned().fold(zero(), |s, x| s + x), 0);
    let pch = i("PCH");
    let pushed: W = (0..5).map(|k| c.and(&format!("irq.ie{k}"), &[iflag[k].clone(), pch[k].clone()])).collect();
    let vector: W = (0..5)
        .map(|n| {
            let late = pushed[..n].iter().fold(sp0.clone() + dispatch.clone() + pushed[n].clone(), |s, x| s + one() - x.clone());
            c.node(&format!("vector{n}"), vec![Step { coef: 1, lin: one() - sp0.clone() + disp[n].clone(), theta: 2 }, Step { coef: 1, lin: late, theta: n as i64 + 3 }])
        })
        .collect();
    let exec = one() - halted.clone() - dispatch.clone();
    let wake = halted;

    let (o, cb, op0, op1, imm8, imm16, d0, d1, memory) = match mode {
        Mode::Host => {
            let (op0, op1, op2) = (i("OP0"), i("OP1"), i("OP2"));
            // The opcode one-hots arrive with the input (level 0). The inputs are the
            // fetched bytes (under the HALT bug read at pc, pc and pc + 1), so the
            // immediates are OP1 and OP2 either way.
            let o = hot["OP0"].clone();
            let prefix = o[0xCB].clone();
            let q: Vec<Lin> = (0..=255u8).map(|op| c.and(&format!("cb{op:02x}"), &[hot["OP1"][op as usize].clone(), prefix.clone()])).collect();
            let imm8 = op1.clone();
            let imm16 = cat(&op1, &op2);
            (o, super::Cb::Hots(q), op0, op1, imm8, imm16, i("D0"), i("D1"), None)
        }
        Mode::Memory => {
            // The banking registers: constant-address heads at layer 1. A memory head cannot
            // sit at layer 0: a kept token's key there would come from its embedding alone,
            // before layer 0 gathers its address, and differ from the key the later heads see.
            // The ten registers start a line, so two heads read them all.
            let lines: Vec<Line> = (0..layout::N_REGS.div_ceil(LINE)).map(|n| c.head(&format!("REGS{n}"), 1, Query::Const(layout::REGS + (n * LINE) as u32))).collect();
            // What OAM DMA blocks at the step's start: OAM by CTL bit 4, each bus by `DB`'s
            // high nibble (`layout::Dma`).
            let db = i("DB");
            let dma = translate::Blocks { oam: ctl[4].clone(), vram: db[5].clone(), main: db[6].clone(), wram: db[7].clone() };
            let mut g = G { c, o: Vec::new(), cb: super::Cb::Hots(Vec::new()), mode };
            // The access locks: CTL bits 5 and 6 while the host's copy of the control
            // model's locks is current (bit 7), else the late `LK` the host asked for.
            let lk = i("LK");
            let locks = translate::Locks { vram: ctl[5].clone() + lk[0].clone(), oam: ctl[6].clone() + lk[1].clone() };
            let regs = Regs::banked(&mut g, (0..layout::N_REGS).map(|r| lines[r / LINE][r % LINE].map(Lin::from).to_vec()).collect()).with_locks(locks.clone());
            // Each data access's state: the step's start (CTL, LK) unless its nibble's bit 3
            // says the host saw the bus change before the access's M-cycle; then the
            // nibble's bits. Per bit, `[base − 2·ov ≥ 1] + [n + ov ≥ 2]`, one node at level 1.
            let nibbles = [i("AS0"), i("AS1")];
            let bases = [locks.vram.clone(), locks.oam.clone(), dma.oam.clone()];
            let slots: [Slot; 4] = std::array::from_fn(|s| {
                let n = &nibbles[s / 2][4 * (s % 2)..4 * (s % 2) + 4];
                let ov = n[3].clone();
                let bit: Vec<Lin> = (0..3).map(|b| g.c.node(&format!("AS.{s}.{b}"), vec![Step { coef: 1, lin: bases[b].clone() - ov.clone() * 2, theta: 1 }, Step { coef: 1, lin: n[b].clone() + ov.clone(), theta: 2 }])).collect();
                // Each bus the same way: the step's start, or the nibble's DMA bit with the
                // transfer's bus (`DB`'s low nibble).
                let bus = |g: &mut G, k: usize| g.c.node(&format!("AS.{s}.bus{k}"), vec![Step { coef: 1, lin: db[4 + k].clone() - ov.clone() * 2, theta: 1 }, Step { coef: 1, lin: n[2].clone() + db[k].clone() + ov.clone(), theta: 3 }]);
                let dma = translate::Blocks { oam: bit[2].clone(), vram: bus(&mut g, 1), main: bus(&mut g, 2), wram: bus(&mut g, 3) };
                Slot { regs: regs.without_locks().with_locks(translate::Locks { vram: bit[0].clone(), oam: bit[1].clone() }), dma }
            });
            // Each operand fetch's buses the same way from `FS` (operand 1 low, 2 high): the
            // step's start, or the nibble's DMA bit with the transfer's buses. Operand k lands
            // at M-cycle k, where OAM DMA can have ended or begun.
            let fs = i("FS");
            let fetch_dma: Vec<translate::Blocks> = (0..2)
                .map(|f| {
                    let n = &fs[4 * f..4 * f + 4];
                    let ov = n[3].clone();
                    let oam = g.c.node(&format!("FS.{f}.oam"), vec![Step { coef: 1, lin: dma.oam.clone() - ov.clone() * 2, theta: 1 }, Step { coef: 1, lin: n[2].clone() + ov.clone(), theta: 2 }]);
                    let mut bus = |k: usize| g.c.node(&format!("FS.{f}.bus{k}"), vec![Step { coef: 1, lin: db[4 + k].clone() - ov.clone() * 2, theta: 1 }, Step { coef: 1, lin: n[2].clone() + db[k].clone() + ov.clone(), theta: 3 }]);
                    let (vram, main, wram) = (bus(1), bus(2), bus(3));
                    translate::Blocks { oam, vram, main, wram }
                })
                .collect();
            // Each operand fetch's access locks the same way from `FS` bits 0 and 1: an
            // operand reads at its M-cycle, where mode 3 or mode 2 can have begun or ended
            // (SameBoy `cycle_read`: a fetch takes the locks a data read does).
            let fetch_locks: Vec<translate::Locks> = (0..2)
                .map(|f| {
                    let n = &fs[4 * f..4 * f + 4];
                    let ov = n[3].clone();
                    let mut bit = |b: usize, base: &Lin, tag: &str| g.c.node(&format!("FS.{f}.{tag}"), vec![Step { coef: 1, lin: base.clone() - ov.clone() * 2, theta: 1 }, Step { coef: 1, lin: n[b].clone() + ov.clone(), theta: 2 }]);
                    translate::Locks { vram: bit(0, &locks.vram, "vlk"), oam: bit(1, &locks.oam, "olk") }
                })
                .collect();
            // Fetch: PC, then PC+1 and PC+2, or PC and PC+1 again under the HALT bug. A
            // RAM's open row serves every byte in it, so the opcode comes from PC's line,
            // and OP1 and OP2 do too unless they cross into the next line, line(PC) + 1
            // (the `FN` translation and the `OP{k}c.n` columns). Only that line's first
            // two bytes can be picked, so only they take channels. The one line that
            // straddles a host boundary is FFF8..FFFF, whose last byte is IE. Its last
            // lane comes from the line FFFF's fetch lands in, a constant except under OAM
            // DMA (the `OP{k}c.x` columns).
            let o0 = translate::ones7_ahead(&mut g, "F0", &pc, 0, &hb);
            let f0 = translate::read_address_fetch(&mut g, "F0", &pc, &pc, None, o0, &regs.unbanked(), &dma);
            // line(PC) + 1: bits 3..6 of pc + 8 and the carry out of bit 6 at level 1, the
            // high nine bits at level 2 from pc's high bits and the carry.
            let c7 = g.c.ge("FN.c7", crate::circuit::value(&pc[3..], 4), 15);
            let mid = g.c.sum_bits("FN.mid", 4, |n| crate::circuit::value(&pc[3..], n) + one());
            let hi = g.c.sum_bits("FN.hi", 9, |n| crate::circuit::value(&pc[7..], n) + c7.clone());
            let next = cat(&cat(&[zero(), zero(), zero()], &mid), &hi);
            // The operands' lines reach VRAM or the FE page (WC bit 3): the access locks at
            // each operand's M-cycle matter, so the host looks them up there.
            let fetch_lock = lockable(&mut g, "FLK", &[(vec![], &pc), (vec![], &next)]);
            // Unblocked and unlocked: each operand's own state blocks its columns below.
            let fnx = translate::read_address_fetch(&mut g, "FN", &next, &pc, Some(&c7), zero(), &regs.without_locks(), &translate::Blocks::none());
            let at = |dma_on: bool| layout::fetch(&layout::Banking([0; layout::N_REGS]), 0xFFFF, dma_on);
            let (x0, x1) = (at(false), at(true));
            let xq: Vec<Lin> = (0..crate::circuit::ADDR_BITS).map(|k| Lin::konst((x0 >> k & 1) as i64) + dma.oam.clone() * ((x1 >> k & 1) as i64 - (x0 >> k & 1) as i64)).collect();
            let x_void = Lin::konst(layout::is_host(x0) as i64) + dma.oam.clone() * (layout::is_host(x1) as i64 - layout::is_host(x0) as i64);
            let a_layer = g.c.level_of(&f0.bits[LINE_BITS..]);
            let n_layer = g.c.level_of(&fnx.bits[LINE_BITS..]);
            let x_lane = (x0 as usize) % LINE;
            assert!(x_lane == LINE - 1 && (x1 as usize) % LINE == x_lane, "FFFF's fetch lands in a line's last lane with and without OAM DMA");
            // PC's line is FFF8..FFFF: pc's bits 3..15 all set.
            let top = g.c.ge("FT", (3..16).fold(zero(), |s, b| s + pc[b].clone()), 13);
            let not_hb = one() - hb.clone();
            let words0 = line::words(&mut g.c, "OP0w", &[(vec![], &pc[..LINE_BITS], 0)]);
            // The opcode by column select, as a RAM's column decoder picks the byte of the
            // open row. Each lane has one head, which queries PC's line when its word line
            // is 1 and VOID's zero line otherwise and drives its lane onto one byte bus.
            // The nodes that read the opcode then read eight wires that change with the
            // opcode, not the line's 64 bits and the word lines.
            let op0_bus = g.c.bus("OP0", a_layer);
            for (k, word) in words0.iter().enumerate() {
                g.c.head_onto(&format!("OP0c{k}"), a_layer, Query::gated(&f0.bits, word, layout::VOID), &[(k, op0_bus)]);
            }
            let op0: W = op0_bus.iter().map(|&s| Lin::from(s)).collect();
            // Fetch k's byte sits at offset o = (pc & 7) + k − hb from the start of PC's
            // line. Below 8 it is lane o of PC's line (lane 7 of FFF8..FFFF comes from
            // FFFF's line), and from 8 it is lane o − 8 of the next line. Each (line, lane)
            // the fetch can take has one word line, a node with one step per (pc & 7, hb)
            // that lands there.
            let mut fetch_void = vec![f0.host.clone()];
            let mut fetched = Vec::new();
            let mut next_lines = Vec::new();
            for k in 1..=2usize {
                #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
                enum Src {
                    Pc,
                    Next,
                    Ffff,
                }
                // Whether operand k's own state blocks PC's line or the next one: OAM DMA's
                // buses or the access locks, a sum of two 0/1 wires (any nonzero blocks).
                let blocked_pc = bus_blocked(&mut g, &format!("F{k}B.a"), &pc, &fetch_dma[k - 1]) + lock_blocked(&mut g, &format!("F{k}L.a"), &pc, &fetch_locks[k - 1]);
                let blocked_next = bus_blocked(&mut g, &format!("F{k}B.n"), &next, &fetch_dma[k - 1]) + lock_blocked(&mut g, &format!("F{k}L.n"), &next, &fetch_locks[k - 1]);
                let mut cases: BTreeMap<(Src, usize), Vec<Step>> = BTreeMap::new();
                for j in 0..LINE {
                    for h in 0..2usize {
                        let o = j + k - h;
                        let hl = if h == 1 { hb.clone() } else { not_hb.clone() };
                        let low = words0[j].clone();
                        let (src, lane, gate) = match o {
                            o if o >= LINE => (Src::Next, o - LINE, None),
                            o if o == x_lane => (Src::Pc, o, Some(one() - top.clone())),
                            o => (Src::Pc, o, None),
                        };
                        let mut push = |src: Src, gate: Option<Lin>| {
                            let n = 2 + gate.is_some() as i64;
                            let lin = gate.into_iter().fold(hl.clone() + low.clone(), |s, g| s + g);
                            let blocked = match src {
                                Src::Pc => blocked_pc.clone(),
                                Src::Next => blocked_next.clone(),
                                Src::Ffff => zero(),
                            };
                            cases.entry((src, lane)).or_default().push(Step { coef: 1, lin: lin - blocked, theta: n });
                        };
                        push(src, gate);
                        if src == Src::Pc && lane == LINE - 1 {
                            push(Src::Ffff, Some(top.clone()));
                        }
                    }
                }
                // The operand by column select too: one head per (line, lane) the fetch can
                // take, which queries that line when its word line is 1 and VOID's zero
                // line otherwise and drives the fetch's bus. The bus sits at the latest
                // layer among the word lines and the two lines' addresses.
                let mut void = Vec::new();
                let mut columns = Vec::new();
                for ((src, lane), steps) in cases {
                    let (query, tag, v) = match src {
                        Src::Pc => (&f0.bits, "a", f0.host.clone()),
                        Src::Next => (&fnx.bits, "n", fnx.host.clone()),
                        Src::Ffff => (&xq, "x", x_void.clone()),
                    };
                    let word = g.c.node(&format!("OP{k}w.{tag}{lane}"), steps);
                    void.push(Step { coef: 1, lin: word.clone() + v, theta: 2 });
                    columns.push((format!("OP{k}c.{tag}{lane}"), query, word, lane));
                }
                fetch_void.push(g.c.node(&format!("F{k}F"), void));
                // Whether operand k's line is blocked: the next line's from offset 8 on, PC's
                // below it (FFFF's line is the FF page's, never blocked).
                // From PC's offset in its line, one-hot (`OP0w`): offset j + k − hb ≥ 8.
                let past = (0..LINE).fold(zero(), |s, j| s + words0[j].clone() * (j + k).saturating_sub(LINE - 1) as i64);
                let next_line = g.c.ge(&format!("F{k}N"), past - hb.clone(), 1);
                next_lines.push(next_line.clone());
                let blocked = g.c.node(&format!("F{k}B"), vec![Step { coef: 1, lin: blocked_pc.clone() - next_line.clone() * 2, theta: 1 }, Step { coef: 1, lin: blocked_next.clone() + next_line.clone() * 2, theta: 3 }]);
                let layer = columns.iter().map(|(_, _, w, _)| g.c.level_of(std::slice::from_ref(w))).max().unwrap().max(n_layer).max(a_layer);
                let bus = g.c.bus(&format!("OP{k}"), layer);
                for (name, query, word, lane) in &columns {
                    g.c.head_onto(name, layer, Query::gated(query, word, layout::VOID), &[(*lane, bus)]);
                }
                // A blocked operand's columns all read VOID's zeros, so its byte is the bus
                // plus 0xFF, an affine form.
                fetched.push(bus.iter().map(|&s| Lin::from(s) + blocked.clone()).collect::<W>());
            }
            let fetch_void: [Lin; 3] = fetch_void.try_into().unwrap();
            // The port of each fetched byte the host serves, so the host computes no fetch
            // address and dispatches on the model's: the bus address pc + k − hb (the HALT
            // bug's repeat included) translated as a read. Every input is gated by "the
            // host serves some fetched byte", so on any other step the adders and the
            // translations hold still and only the gates read pc.
            let served = g.c.ge("FPh", fetch_void.iter().fold(zero(), |s, v| s + v.clone()), 1);
            let gpc: W = (0..16).map(|j| g.c.ge(&format!("FPpc{j}"), served.clone() + pc[j].clone(), 2)).collect();
            let ghb = g.c.ge("FPhb", served + hb.clone(), 2);
            let mut fetch_ports = Vec::new();
            for k in 0..3i64 {
                let a = if k == 0 { gpc.clone() } else { add_small(&mut g, &format!("FPa{k}"), &gpc, Lin::konst(k) - ghb.clone()) };
                // An operand the host serves is one its own state left unblocked and unlocked.
                let (none, unlocked) = (translate::Blocks::none(), regs.without_locks());
                let (r, d) = if k == 0 { (&regs, &dma) } else { (&unlocked, &none) };
                fetch_ports.push(translate::read_address(&mut g, &format!("FP{k}"), &a, r, d, Region::Literal).bits);
            }
            let fetch_ports: [W; 3] = fetch_ports.try_into().unwrap();
            // A DMA line's bus addresses, 0 on every GBC CPU step, translated as reads under
            // the registers as they stand, without OAM DMA's block: the bus map, VRAM's bank
            // and the cartridge RAM's mask in the weights, not the host.
            let unlocked = regs.without_locks();
            let dma_lines: [W; 2] = [("DP", "DS"), ("DQ", "DD")].map(|(out, input)| {
                let a = cat(&i(&format!("{input}L")), &i(&format!("{input}H")));
                translate::read_address(&mut g, out, &a, &unlocked, &translate::Blocks::none(), Region::Literal).bits
            });
            // The opcode's nibble one-hots match the bus's nibbles, one step each, a level
            // above the bus, and each opcode's one-hot is the AND of its two, a level
            // later. The request address reads the nibble one-hots, so it waits for no AND.
            let nibbles: Vec<Vec<Lin>> = ["lo", "hi"]
                .into_iter()
                .enumerate()
                .map(|(h, name)| (0..16).map(|n| g.c.node(&format!("OP0.{name}{n:x}"), vec![Step { coef: 1, lin: pattern(&op0[4 * h..4 * h + 4], n), theta: 4 }])).collect())
                .collect();
            let o: Vec<Lin> = (0..256).map(|v| g.c.and(&format!("OP0={v:02x}"), &[nibbles[0][v & 15].clone(), nibbles[1][v >> 4].clone()])).collect();
            let (op1, op2) = (fetched[0].clone(), fetched[1].clone());
            // A case of opcodes his × los is two gates, the sums of their nibbles'
            // one-hots, each 0 or 1.
            let nib = |his: &[usize], los: &[usize]| [(1, his), (0, los)].map(|(h, ns)| ns.iter().fold(zero(), |s, &n| s + nibbles[h][n].clone())).to_vec();
            let hot = |op: u8| nib(&[op as usize >> 4], &[op as usize & 15]);
            // The CB decode is by field: the operation and index (op1 bits 3..7), the
            // register (bits 0..2) and the registers written (every operation but bit),
            // each gated by the prefix.
            let prefix = o[0xCB].clone();
            // OP1 changes on most steps and the prefix on few, so the field nodes read
            // OP1 through eight prefix gates. A changed OP1 bit then updates one gate, not
            // every field node that tests it. Gated bits are zero without the prefix,
            // where the prefix term keeps every pattern below its threshold.
            let cb1: W = (0..8).map(|k| g.c.and(&format!("cb1.{k}"), &[op1[k].clone(), prefix.clone()])).collect();
            let pattern = |bits: &[Lin], v: usize| (0..bits.len()).fold(prefix.clone(), |s, k| s + if v >> k & 1 == 1 { bits[k].clone() } else { one() - bits[k].clone() });
            let xy: Vec<Lin> = (0..32).map(|v| g.c.ge(&format!("cbxy.{v}"), pattern(&cb1[3..8], v), 6)).collect();
            let z: Vec<Lin> = (0..8).map(|r| g.c.ge(&format!("cbz.{r}"), pattern(&cb1[0..3], r), 4)).collect();
            // Bit is x = 1: op1 bit 7 clear and bit 6 set. The register pattern counts
            // double so the operation term decides only when the pattern holds.
            let writes: Vec<Lin> = (0..8).map(|r| g.c.ge(&format!("cbw.{r}"), pattern(&cb1[0..3], r) * 2 + cb1[7].clone() + one() - cb1[6].clone(), 9)).collect();
            g.o = o.clone();
            g.cb = super::Cb::Fields { xy, z, writes, any: prefix.clone() };
            // The request address from the opcodes' nibbles: a case's gates name high
            // nibbles `his` by low nibbles `los`.
            let with = |mut gates: Vec<Lin>, more: &[Lin]| {
                gates.extend_from_slice(more);
                gates
            };
            let regs8 = [i("B"), i("C"), i("D"), i("E"), i("H"), i("L")];
            let (bc, de, hl) = (cat(&regs8[1], &regs8[0]), cat(&regs8[3], &regs8[2]), cat(&regs8[5], &regs8[4]));
            let f = i("F");
            let (fz, fc) = (f[7].clone(), f[4].clone());
            let not = |l: &Lin| one() - l.clone();
            let ff = |lo: &W| cat(lo, &crate::circuit::bits(0xFF, 8));
            // Each case: the gates that select it (all 0/1, mutually exclusive across
            // cases), its bus address and the translation that serves it. The stable
            // sources (HL, SP) are translated once each; a16 and the rare pairs (BC, DE)
            // go through `rare`, a bus address that is zero unless one of their opcodes
            // executes, and FF00+n and FF00+C through `ffn`, their low byte under their
            // opcodes, so the I/O page's port table reads a fetched byte or C only when
            // an instruction addresses the page with it.
            #[derive(Clone, Copy, PartialEq, Eq)]
            enum Src {
                Hl,
                Sp,
                Ff,
                Rare,
            }
            let mut cases: Vec<(Vec<Lin>, W, Src)> = vec![
                // LD A,(BC) and LD A,(DE).
                (nib(&[0], &[0xA]), bc.clone(), Src::Rare),
                (nib(&[1], &[0xA]), de.clone(), Src::Rare),
                // LD A,(HL+), LD A,(HL-), INC (HL), DEC (HL).
                (nib(&[2, 3], &[0xA]), hl.clone(), Src::Hl),
                (nib(&[3], &[4, 5]), hl.clone(), Src::Hl),
                // LD r,(HL) (HALT, 0x76, aside) and the ALU on (HL).
                (nib(&[4, 5, 6], &[6, 0xE]), hl.clone(), Src::Hl),
                (nib(&[7], &[0xE]), hl.clone(), Src::Hl),
                (nib(&[8, 9, 0xA, 0xB], &[6, 0xE]), hl.clone(), Src::Hl),
                // A CB op on (HL): register field 6.
                (with(nib(&[0xC], &[0xB]), &[op1[2].clone(), op1[1].clone(), not(&op1[0])]), hl.clone(), Src::Hl),
                // POP BC, POP DE, RET, RETI; POP HL, POP AF.
                (nib(&[0xC, 0xD], &[1, 9]), sp.clone(), Src::Sp),
                (nib(&[0xE, 0xF], &[1]), sp.clone(), Src::Sp),
                // LD A,(a16), LDH A,(n), LD A,(C).
                (nib(&[0xF], &[0xA]), cat(&op1, &op2), Src::Rare),
                (nib(&[0xF], &[0]), ff(&op1), Src::Ff),
                (nib(&[0xF], &[2]), ff(&regs8[1]), Src::Ff),
            ];
            // RET cc taken (C0, C8, D0, D8), condition true.
            let cc = [not(&fz), fz.clone(), not(&fc), fc.clone()];
            let all = |gates: Vec<Lin>, more: &[Lin]| { let n = (gates.len() + more.len()) as i64; Step { coef: 1, lin: gates.into_iter().chain(more.iter().cloned()).fold(zero(), |s, x| s + x), theta: n } };
            let jr_steps = std::iter::once(all(hot(0x18), &[]))
                .chain(cc.iter().enumerate().map(|(c, cond)| all(hot(0x20 + 8 * c as u8), std::slice::from_ref(cond))))
                .collect();
            let jr_taken = g.c.node("jr.taken", jr_steps);
            for (y, cond) in cc.iter().enumerate() {
                cases.push((with(nib(&[0xC + y / 2], &[8 * (y % 2)]), &[cond.clone()]), sp.clone(), Src::Sp));
            }
            let pops_from = 8;
            // The rare bus address, one level above the opcode one-hots: bit k is the sum
            // over its sources of [bit k of the source ∧ op0 is the source's opcode], the
            // opcode test being its one-hot wire. (Testing the eight opcode bits in every step
            // instead made each fetched bit reach all 48 steps; testing the nibble one-hots
            // made every opcode entering or leaving low nibble A or high nibble F reach them.)
            let rare_ops: [(usize, u8); 3] = [(0, 0x0A), (1, 0x1A), (10, 0xFA)];
            let ff_ops: [(usize, u8); 2] = [(11, 0xF0), (12, 0xF2)];
            let rare: W = (0..16)
                .map(|k| {
                    let steps = rare_ops.iter().map(|&(case, op)| all(vec![o[op as usize].clone()], &[cases[case].1[k].clone()])).collect();
                    g.c.node(&format!("rare.{k}"), steps)
                })
                .collect();
            // One wire per source: the sum of its cases, each a step over that case's gates.
            let fires = |gates: &[Lin]| Step { coef: 1, lin: gates.iter().cloned().fold(zero(), |s, x| s + x), theta: gates.len() as i64 };
            // A CB op on (HL) tests op1's register field, which arrives a level after the
            // nibble one-hots: it takes its own node, keyed on the prefix's one-hot, so the
            // nibble one-hots are read in one layer only.
            let cb_hl = 7;
            assert!(cases[cb_hl].0.len() == 5 && cases[cb_hl].2 == Src::Hl, "case 7 is a CB op on (HL)");
            let sel = |g: &mut G, name: &str, src: Src| g.c.node(name, cases.iter().enumerate().filter(|&(i, (_, _, s))| *s == src && i != cb_hl).map(|(_, (gates, _, _))| fires(gates)).collect());
            let sel_hl = sel(&mut g, "rd.hl", Src::Hl) + g.c.node("rd.hl.cb", vec![fires(&[o[0xCB].clone(), op1[2].clone(), op1[1].clone(), not(&op1[0])])]);
            let sel_sp = sel(&mut g, "rd.sp", Src::Sp);
            let sel_rare = sel(&mut g, "rd.rare", Src::Rare);
            let sel_ff = sel(&mut g, "rd.ff", Src::Ff);
            assert!(cases[pops_from..pops_from + 2].iter().chain(&cases[cases.len() - 4..]).all(|(_, _, s)| *s == Src::Sp) && cases.iter().filter(|(_, _, s)| *s == Src::Sp).count() == 6, "every read through SP is a pop or a return, and reads a second byte");
            let rq0 = sel_hl.clone() + sel_sp.clone() + sel_ff.clone() + sel_rare.clone();
            let rq1 = sel_sp.clone();
            let t_hl = translate::read_address(&mut g, "RHL", &hl, &slots[0].regs, &slots[0].dma, Region::Wired);
            let t_sp = translate::read_address(&mut g, "RSP", &sp, &slots[0].regs, &slots[0].dma, Region::Wired);
            // The rare sources' translation sits a level above the stable ones (the bus
            // needs the opcode one-hots), so read 0 is two heads: one keyed on the stable
            // selection, one on the rare translation, each landing in VOID (a zero byte)
            // unless its source is selected, and the byte is their sum.
            // Its regions come from the sources with the bus's opcode tests, so its
            // steps read a region wire where they read the bus's high bits, which flip
            // between 0 and an address on every step that uses the bus.
            let rare_sources: Vec<(W, Vec<Lin>)> = rare_ops.iter().map(|&(case, op)| (cases[case].1.clone(), vec![o[op as usize].clone()])).collect();
            let t_rare = translate::read_address_muxed(&mut g, "RRARE", &rare, &rare_sources, &slots[0].regs, &slots[0].dma);
            // FF00+n and FF00+C: their low byte under their opcode, a level above the
            // one-hots like the rare bus, and its own FF-page translation, which lands in
            // VOID unless one of them runs. It shares the rare heads, since the two never
            // run together.
            let ffn: W = (0..8)
                .map(|k| {
                    let steps = ff_ops.iter().map(|&(case, op)| all(vec![o[op as usize].clone()], &[cases[case].1[k].clone()])).collect();
                    g.c.node(&format!("ffn.{k}"), steps)
                })
                .collect();
            // The same byte's low seven bits gated by its bit 7 clear as well: the I/O page's
            // port table reads these, so a read of HRAM, which most of them are, leaves it still.
            let ff_io = |g: &mut G, name: &str, k: Option<usize>| {
                let steps = ff_ops.iter().map(|&(case, op)| { let b = &cases[case].1; all(vec![o[op as usize].clone(), one() - b[7].clone()], &k.map(|k| b[k].clone()).into_iter().collect::<Vec<_>>()) }).collect();
                g.c.node(name, steps)
            };
            let ff_io_gate = ff_io(&mut g, "ffio", None);
            let ff_io_low: W = (0..7).map(|k| ff_io(&mut g, &format!("ffio.{k}"), Some(k))).collect();
            let t_ff = translate::ff_page_read(&mut g, "RFF", &ffn, &sel_ff, (&ff_io_gate, &ff_io_low), &slots[0].regs);
            let t_aside = translate::either(&t_rare, &t_ff);
            let stable = [(sel_hl, &t_hl, hl.clone()), (sel_sp, &t_sp, sp.clone())];
            let void = translate::void_block();
            let none = stable.iter().fold(one(), |s, (sel, _, _)| s - sel.clone());
            let mut main_cases: Vec<(Vec<Lin>, &Physical)> = stable.iter().map(|(s, t, _)| (vec![s.clone()], *t)).collect();
            main_cases.push((vec![none], &void));
            let rp0_main = translate::select(&mut g, "RP0", &main_cases, layout::Access::Read);
            let rp0 = translate::either(&rp0_main, &t_aside);
            // The rare bus and `ffn` are already zero unless their opcodes run, so read 0's
            // bus address adds them (FF00 + n's high byte is `rd.ff`) to the stable sources'
            // selection instead of selecting them again.
            let ra0_cases: Vec<(Vec<Lin>, W)> = stable.iter().map(|(s, _, a)| (vec![s.clone()], a.clone())).collect();
            let ra0: W = g.c.mux("RA0", 16, &ra0_cases).into_iter().enumerate().map(|(k, b)| b + rare[k].clone() + if k < 8 { ffn[k].clone() } else { sel_ff.clone() }).collect();
            let sp1 = super::datapath::add16_split(&mut g, "sp+1", &sp, 1);
            let rp1 = translate::read_address(&mut g, "RP1", &sp1, &slots[1].regs, &slots[1].dma, Region::Wired);
            let layer = g.c.level_of(&rp0_main.bits[LINE_BITS..]).max(g.c.level_of(&t_aside.bits[LINE_BITS..])).max(g.c.level_of(&rp1.bits[LINE_BITS..]));
            // The word lines come from the sources' own low bits under their gates, as the
            // bus's low bits do (a read translation passes them through), at the level of
            // the selection rather than above it. With no source selected the head reads
            // VOID's zero line and every word line is 0.
            let main_words: Vec<(Vec<Lin>, &[Lin], usize)> = stable.iter().map(|(s, _, a)| (vec![s.clone()], &a[..LINE_BITS], 0)).collect();
            let main_words = line::words(&mut g.c, "D0w", &main_words);
            // The rare heads' word lines come from the rare bus and `ffn`, which are 0
            // unless their opcodes run, under their selections.
            let rare_words = line::words(&mut g.c, "D0rw", &[(vec![sel_rare.clone()], &rare[..LINE_BITS], 0), (vec![sel_ff.clone()], &ffn[..LINE_BITS], 0)]);
            // Read 0 by column select, with one head per lane of the stable selection's
            // line and one per lane of the rare translation's. Each reads VOID's zero line
            // unless its word line is 1, and all drive one byte bus. With no source
            // selected every word line is 0 and the bus holds 0.
            let d0_bus = g.c.bus("D0", layer);
            for (tag, bits, words) in [("h", &rp0_main.bits, &main_words), ("r", &t_aside.bits, &rare_words)] {
                for (k, word) in words.iter().enumerate() {
                    g.c.head_onto(&format!("D0{tag}{k}"), layer, Query::gated(bits, word, layout::VOID), &[(k, d0_bus)]);
                }
            }
            let r0: W = d0_bus.iter().map(|&s| Lin::from(s)).collect();
            let d1_words = line::words(&mut g.c, "D1w", &[(vec![], &sp[..LINE_BITS], 1)]);
            let r1 = Pick::new(g.c.head("D1h", layer, Query::Wires(rp1.bits.clone())), d1_words).bits(&mut g.c, "D1");
            let d0: W = r0.iter().zip(&inp["D0"]).map(|(m, h)| m.clone() + h.clone()).collect();
            let d1: W = r1.iter().zip(&inp["D1"]).map(|(m, h)| m.clone() + h.clone()).collect();
            let d0_read = Read { bus: r0.clone(), late: inp["D0"].clone() };
            let fetch_oam = [dma.oam.clone(), fetch_dma[0].oam.clone(), fetch_dma[1].oam.clone()];
            let memory = MemoryDecode { regs, slots, fetch_void, fetch_ports, fetch_lock, fetch_oam, next_line: next_lines.try_into().expect("two operands"), next: next.clone(), dma_lines, ob: i("OB"), ra0, rq0, rq1, rp: [rp0, rp1], read: [r0, r1], d0: d0_read, sp_p1: sp1, jr_taken };
            let (o, cb) = (std::mem::take(&mut g.o), std::mem::replace(&mut g.cb, super::Cb::Hots(Vec::new())));
            c = g.c;
            (o, cb, op0, op1.clone(), op1.clone(), cat(&op1, &op2), d0, d1, Some(memory))
        }
    };
    let regs = [i("B"), i("C"), i("D"), i("E"), i("H"), i("L"), d0.clone(), a.clone()];
    let (bc, de, hl) = (cat(&regs[1], &regs[0]), cat(&regs[3], &regs[2]), cat(&regs[5], &regs[4]));
    let g = G { c, o, cb, mode };
    let d = Decoded { a, f: i("F"), regs, bc, de, hl, sp, pc, ctl, op0, op1, d0, d1, imm8, imm16, disp, vector, any, exec, dispatch, wake, memory };
    (g, d)
}

/// `x + inc` mod 2^16 for a small increment `inc` (0 to 7, a form over wires), split
/// as `datapath::add16_split` is: bits 0..2 and the carries out of bits 2 and 6, then
/// bits 3..6 and 7..15 a level later.
/// Whether OAM DMA's buses `d` block bus address `a` (`layout::Dma::blocks`): 0000-7FFF
/// and A000-BFFF on the main bus, 8000-9FFF on VRAM's, C000-FDFF on work RAM's and
/// FE00-FEFF on OAM's; the FF page never. One node, its regions exclusive.
fn bus_blocked(g: &mut G, name: &str, a: &[Lin], d: &translate::Blocks) -> Lin {
    let not = |x: &Lin| one() - x.clone();
    let page = (9..16).fold(zero(), |s, k| s + a[k].clone());
    g.c.node(
        name,
        vec![
            Step { coef: 1, lin: not(&a[15]) + d.main.clone(), theta: 2 },
            Step { coef: 1, lin: a[15].clone() + not(&a[14]) + not(&a[13]) + d.vram.clone(), theta: 4 },
            Step { coef: 1, lin: a[15].clone() + not(&a[14]) + a[13].clone() + d.main.clone(), theta: 4 },
            Step { coef: 1, lin: a[15].clone() + a[14].clone() + d.wram.clone(), theta: 3 },
            Step { coef: -1, lin: page.clone() + d.wram.clone(), theta: 8 },
            Step { coef: 1, lin: page + not(&a[8]) + d.oam.clone(), theta: 9 },
        ],
    )
}

/// Whether the access locks `l` hold the line at `a`: VRAM's (8000-9FFF) by the VRAM lock,
/// OAM's (FE00-FE9F) by the OAM lock, as `layout::translate_locked` places them. The two
/// regions are disjoint, so the node is 0 or 1.
fn lock_blocked(g: &mut G, name: &str, a: &[Lin], l: &translate::Locks) -> Lin {
    let not = |x: &Lin| one() - x.clone();
    let page = (9..16).fold(zero(), |s, k| s + a[k].clone());
    // FE00-FE9F: the FE page with the low byte's bits 7..5 below 101.
    let top3 = a[7].clone() * 4 + a[6].clone() * 2 + a[5].clone();
    g.c.node(
        name,
        vec![
            Step { coef: 1, lin: a[15].clone() + not(&a[14]) + not(&a[13]) + l.vram.clone(), theta: 4 },
            Step { coef: 1, lin: (page + not(&a[8]) + l.oam.clone()) * 5 - top3, theta: 41 },
        ],
    )
}

fn add_small(g: &mut G, name: &str, x: &[Lin], inc: Lin) -> W {
    use crate::circuit::value;
    let lo = g.c.sum_bits(&format!("{name}.lo"), 3, |n| value(x, n) + inc.clone());
    let c3 = g.c.ge(&format!("{name}.c3"), value(x, 3) + inc.clone(), 8);
    let c7 = g.c.ge(&format!("{name}.c7"), value(x, 7) + inc, 128);
    let mid = g.c.sum_bits(&format!("{name}.mid"), 4, |n| value(&x[3..], n) + c3.clone());
    let hi = g.c.sum_bits(&format!("{name}.hi"), 9, |n| value(&x[7..], n) + c7.clone());
    cat(&cat(&lo, &mid), &hi)
}

/// The step reaches VRAM or the FE page (OAM and the unusable area) by a data access's
/// bus address: one wire per source and region, each the AND of the source's gates and
/// the region's address bits, summed (a step can reach more than one).
pub(super) fn lockable(g: &mut G, name: &str, sources: &[(Vec<Lin>, &W)]) -> Lin {
    let mut any = zero();
    for (k, (gates, a)) in sources.iter().enumerate() {
        let vram = vec![a[15].clone(), one() - a[14].clone(), one() - a[13].clone()];
        let fe: Vec<Lin> = (9..16).map(|b| a[b].clone()).chain([one() - a[8].clone()]).collect();
        for (r, lits) in [vram, fe].into_iter().enumerate() {
            let lits: Vec<Lin> = gates.iter().cloned().chain(lits).collect();
            any = any + g.c.and(&format!("{name}.{k}.{r}"), &lits);
        }
    }
    any
}

/// An access puts an FE page address on the bus that OAM DMA does not block (`blocked`,
/// the access's `Blocks::oam`): one step per source, each the AND of its gates and the
/// FE page's address bits. The sources are exclusive, so the sum is 0 or 1.
pub(super) fn fe_access(g: &mut G, name: &str, sources: &[(Vec<Lin>, &W)], blocked: &Lin) -> Lin {
    let steps = sources
        .iter()
        .map(|(gates, a)| {
            let lits: Vec<Lin> = gates.iter().cloned().chain(fe_literals(a)).chain([one() - blocked.clone()]).collect();
            let n = lits.len() as i64;
            Step { coef: 1, lin: lits.into_iter().fold(zero(), |s, l| s + l), theta: n }
        })
        .collect();
    g.c.node(name, steps)
}

/// Whether PC + 1 (a dispatch's idle M-cycle), PC + 1 − hb and PC + 2 − hb (operands 1 and
/// 2) are in the FE page, in memory mode. Each lies in PC's line or the next one (`F1N`,
/// `F2N`, and PC's low three bits all set for PC + 1), so its page is PC's high byte or the
/// next line's: no node reads PC's low byte beyond the line test.
pub(super) fn fetch_pages(g: &mut G, d: &Decoded, m: &MemoryDecode) -> [Lin; 3] {
    let fe = |a: &[Lin]| fe_literals(a).into_iter().fold(zero(), |s, l| s + l);
    let (here, there) = (fe(&d.pc), fe(&m.next));
    let n1 = g.c.and("IDLE.pc1.n", &d.pc[..LINE_BITS]);
    let mut page = |name: &str, next: Lin| g.c.node(name, vec![Step { coef: 1, lin: here.clone() + one() - next.clone(), theta: 9 }, Step { coef: 1, lin: there.clone() + next, theta: 9 }]);
    [page("IDLE.pc1", n1), page("FFE1", m.next_line[0].clone()), page("FFE2", m.next_line[1].clone())]
}

/// CTL' bits 5-7: each fetch puts an FE page address on the bus that OAM DMA does not
/// block, a DMG's OAM bug read at its M-cycle (`StepFacts::fe_accesses`). The opcode at PC on
/// every step but a wake (a dispatch reads PC too, SameBoy's `GB_cpu_run`), operand k at
/// PC + k − hb when the step fetches it (`fetched`, RC and WC bit 7), its page from
/// `fetch_pages`.
pub(super) fn fetch_fe_bits(g: &mut G, d: &Decoded, m: &MemoryDecode, fetched: [Lin; 2], pages: &[Lin; 3]) -> [Lin; 3] {
    let (exec, dispatch) = (d.exec.clone(), d.dispatch.clone());
    let f0 = fe_access(g, "CTL'.5", &[(vec![exec], &d.pc), (vec![dispatch], &d.pc)], &m.fetch_oam[0]);
    let [f1, f2] = [1usize, 2].map(|k| g.c.and(&format!("CTL'.{}", 5 + k), &[fetched[k - 1].clone(), pages[k].clone(), one() - m.fetch_oam[k].clone()]));
    [f0, f1, f2]
}

/// Whether bus address `a` (16 bits) is in the FE page, FE00-FEFF: one AND.
pub(super) fn fe_page(g: &mut G, name: &str, a: &[Lin]) -> Lin {
    g.c.and(name, &fe_literals(a))
}

fn fe_literals(a: &[Lin]) -> Vec<Lin> {
    (9..16).map(|b| a[b].clone()).chain([one() - a[8].clone()]).collect()
}

/// Whether `a + inc` is in the FE page, `inc` in 0..=2: the low byte's carry, then its
/// high byte FE without one or FD with one.
fn fe_page_plus(g: &mut G, name: &str, a: &[Lin], inc: Lin) -> Lin {
    use crate::circuit::value;
    let carry = g.c.ge(&format!("{name}.c"), value(&a[..8], 8) + inc, 256);
    g.c.node(name, vec![Step { coef: 1, lin: pattern(&a[8..16], 0xFE) + one() - carry.clone(), theta: 9 }, Step { coef: 1, lin: pattern(&a[8..16], 0xFD) + carry, theta: 9 }])
}

/// MC bits 3-6: an FE page address on the bus in an M-cycle without an access
/// (`Outcome::idle`, SameBoy's `cycle_oam_bug` sites), in slot 0 and slot 1, then that
/// M-cycle in two bits. INC rr and DEC rr put the pair there, LD SP,HL puts HL there, PUSH,
/// RST, CALL and a taken CALL cc put SP there, JR puts PC after its operand there and a
/// taken JR cc its target (`jr_target_fe`), all at M-cycle 1, except JR's 2 and CALL's 3;
/// an interrupt dispatch puts PC + 1 in slot 0 and SP in slot 1 at M-cycle 1. Each test is
/// the AND form of `lockable`, gated by its opcodes and `exec`.
/// `pages`, in memory mode, gives PC + 1's and PC + 2 − hb's pages (`fetch_pages`).
pub(super) fn idle_bits(g: &mut G, d: &Decoded, t_jr: &Lin, t_call: &Lin, jr_target_fe: Lin, pages: Option<&[Lin; 3]>) -> [Lin; 4] {
    let (exec, dispatch, hb) = (d.exec.clone(), d.dispatch.clone(), d.ctl[3].clone());
    let fe_bc = fe_page(g, "IDLE.bc", &d.bc);
    let fe_de = fe_page(g, "IDLE.de", &d.de);
    let fe_hl = fe_page(g, "IDLE.hl", &d.hl);
    let fe_sp = fe_page(g, "IDLE.sp", &d.sp);
    let (fe_pc1, fe_pc2) = match pages {
        Some(p) => (p[0].clone(), p[2].clone()),
        None => (fe_page_plus(g, "IDLE.pc1", &d.pc, one()), fe_page_plus(g, "IDLE.pc2", &d.pc, Lin::konst(2) - hb)),
    };
    let pushes = g.ops(|op| matches!(op, 0x33 | 0x3B | 0xC5 | 0xD5 | 0xE5 | 0xF5 | 0xCD) || op & 0xC7 == 0xC7) + t_call.clone();
    let sites: Vec<(Lin, Lin)> = vec![
        (g.op(0x03) + g.op(0x0B), fe_bc),
        (g.op(0x13) + g.op(0x1B), fe_de),
        (g.op(0x23) + g.op(0x2B) + g.op(0xF9), fe_hl),
        (pushes, fe_sp.clone()),
        (g.op(0x18), fe_pc2),
        (t_jr.clone(), jr_target_fe),
    ];
    let mut steps: Vec<Step> = sites.into_iter().map(|(ops, fe)| Step { coef: 1, lin: ops + exec.clone() + fe, theta: 3 }).collect();
    steps.push(Step { coef: 1, lin: dispatch.clone() + fe_pc1, theta: 2 });
    let idle0 = g.c.node("MC.3", steps);
    let idle1 = g.c.and("MC.4", &[dispatch.clone(), fe_sp]);
    // The idle M-cycle: 1 (bit 0), 2 (bit 1) or 3 (both).
    let first = g.ops(|op| matches!(op, 0x03 | 0x0B | 0x13 | 0x1B | 0x23 | 0x2B | 0x33 | 0x3B | 0xF9 | 0xC5 | 0xD5 | 0xE5 | 0xF5) || op & 0xC7 == 0xC7);
    let (second, third) = (g.op(0x18) + t_jr.clone(), g.op(0xCD) + t_call.clone());
    let m0 = g.c.node("MC.5", vec![Step { coef: 1, lin: first + third.clone() + exec.clone(), theta: 2 }, Step { coef: 1, lin: dispatch, theta: 1 }]);
    let m1 = g.c.node("MC.6", vec![Step { coef: 1, lin: second + third + exec, theta: 2 }]);
    [idle0, idle1, m0, m1]
}

/// Whether a JR's target, PC + 2 − hb + e (`pc` 16 bits, `e` the signed offset byte, `hb`
/// the HALT bug), is in the FE page, from the step's inputs rather than the PC adder's
/// late bits: the low byte's sum s lies in [−127, 384], so the target's high byte is PC's
/// one below (s < 0), at (0 ≤ s < 256) or one above (s ≥ 256).
pub(super) fn jr_target_fe(c: &mut Circuit, name: &str, pc: &[Lin], e: &[Lin], hb: Lin) -> Lin {
    use crate::circuit::value;
    let s = value(&pc[..8], 8) + 2 - hb + value(e, 8) - e[7].clone() * 256;
    let up = c.ge(&format!("{name}.up"), s.clone(), 256);
    let down = c.ge(&format!("{name}.down"), -s, 1);
    let hi = &pc[8..16];
    c.node(
        name,
        vec![
            Step { coef: 1, lin: pattern(hi, 0xFE) + (one() - up.clone()) + (one() - down.clone()), theta: 10 },
            Step { coef: 1, lin: pattern(hi, 0xFD) + up, theta: 9 },
            Step { coef: 1, lin: pattern(hi, 0xFF) + down, theta: 9 },
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::byte;

    /// The JR target's FE page test against the target computed, over every low PC byte,
    /// offset and HALT bug on the pages around FE and one far from it.
    #[test]
    fn the_jr_target_test_matches_the_target() {
        let mut c = Circuit::default();
        let (lo, hi, e, ctl) = (byte(c.input("PCL")), byte(c.input("PCH")), byte(c.input("E")), byte(c.input("CTL")));
        let pc: Vec<Lin> = lo.iter().chain(hi.iter()).cloned().collect();
        let fe = jr_target_fe(&mut c, "JR", &pc, &e, ctl[0].clone());
        c.output("FE", std::array::from_fn(|i| if i == 0 { fe.clone() } else { Lin::konst(0) }));
        for h in [0x12u8, 0xFD, 0xFE, 0xFF] {
            for l in 0..=255u8 {
                for off in 0..=255u8 {
                    for hb in [0u8, 1] {
                        let pcv = u16::from_le_bytes([l, h]);
                        let target = pcv.wrapping_add(2 - hb as u16).wrapping_add(off as i8 as u16);
                        let inp = [("PCL", l), ("PCH", h), ("E", off), ("CTL", hb)].map(|(n, v)| (n.to_string(), v)).into_iter().collect();
                        assert_eq!(c.eval(&inp)["FE"], (target >> 8 == 0xFE) as u8, "pc {pcv:#06x} e {off:#04x} hb {hb}");
                    }
                }
            }
        }
    }
}
