//! The control logic over one span, as a threshold circuit
//! (`docs/control-in-weights.md` §3, §13, §14). A span is `D` M-cycles of ticks, then
//! at most one event: a register write or read, an acknowledgement, a raise, a change
//! of the buttons held or a STOP. A span in a speed switch's pause counts its M-cycles
//! in `Q` instead, which moves only the PPU's timing and the pause. The PPU's timing
//! counts 2-dot halves. `LM` is the position in the line in halves, and `H` the
//! M-cycles the position moves, which the circuit turns into halves by its own speed (two
//! an M-cycle at normal speed, one at double speed). `D` never passes the horizon the previous pass returned, so a
//! span holds at most one TIMA overflow, at most one serial completion, crosses at most
//! one line of the PPU's timing and changes the STAT line only on its last M-cycle, and
//! every closed form below is exact.

use super::clock;
use crate::apu_circuit::build::*;
use crate::circuit::{Circuit, Lin, Step, F16_SPAN};
use std::collections::{BTreeMap, BTreeSet};

pub const STATE: [&str; 35] = ["C0", "C1", "TIMA", "TMA", "TC", "IF", "IE", "JP", "JB", "SB", "SC", "SR0", "SR1", "K0", "K1", "SV", "DR", "DI", "HDMA1", "HDMA2", "HDMA3", "HDMA4", "HDMA5", "LM", "LY", "LYC", "ST", "BP", "OP", "PS0", "PS1", "FL", "UR0", "UR1", "UR2"];
pub const EVENTS: [&str; 9] = ["D0", "D1", "EV", "EA", "EX", "Q0", "Q1", "H0", "H1"];
pub const NEXT: [&str; 35] = ["C0'", "C1'", "TIMA'", "TMA'", "TC'", "IF'", "IE'", "JP'", "JB'", "SB'", "SC'", "SR0'", "SR1'", "K0'", "K1'", "SV'", "DR'", "DI'", "HDMA1'", "HDMA2'", "HDMA3'", "HDMA4'", "HDMA5'", "LM'", "LY'", "LYC'", "ST'", "BP'", "OP'", "PS0'", "PS1'", "FL'", "UR0'", "UR1'", "UR2'"];
/// The read's byte; the horizon less one; `FX`, the flags below; the OAM DMA bytes the
/// span moved, a run within one 8-byte line: from page `DP` (bus address bits 15-8), line
/// `DX` (index bits 7-3), byte j of the line where bit j of `DM` (bytes before the span's
/// last M-cycle) or `DL` (its last) is set, each to `FE00 | DX | j`; `DX` bit 0
/// (`DX_LAST_EARLY`) says the last M-cycle's byte lands before a line that M-cycle starts
/// is drawn, which it never does (the byte OAM DMA writes as mode 3 begins misses the line); an HDMA block's source `CS0 CS1` (bits 15-4) and destination `CD0 CD1`
/// (bits 15-4, in VRAM), its 16 bytes at the low four bits; `SS0 SS1`, the M-cycles the
/// CPU stays stopped once the step that runs has ended, a total: the M-cycles it stayed
/// stopped for as the pass began (`SQ0 SQ1`), and an HDMA block's or a speed switch's pause
/// added; `PA`,
/// the palette byte a BCPD or OCPD access reaches outside mode 3 (bits 0-6 its index in
/// `machine::ppu::Palettes`, bit 7 set).
/// The line-timing schedule follows. `RS` is the number of mode-3 starts scheduled; the
/// first is `RF + 1` M-cycles after the span (the horizon) and each next one `RP` later.
/// `SL0` to `SL7` are the lines they draw, and `RM` is the position in the line at each
/// (`LM`'s byte). With `RS` above 0, the next pass runs at the start after the last
/// scheduled one.
/// `FE0 FE1` say when the frame sequencer's counter bit (bit 12, bit 13 at double speed)
/// next falls. Bits 0-11 are `E − 1`, for `E` the M-cycles from the pass to where the
/// count next carries out of the bit (the inverted counter bits below it); `FE1` bit 4
/// (`FE_DROP`) is the M-cycles from the pass to where a set bit a DIV write or a speed
/// switch reset falls, 0 for none (one bit wide: the reset bit falls at the next M-cycle); bit 5 (`FE_HELD`) says a pause holds the counter, so no count falls;
/// bit 6 (`FE_WAKE`) is `IE & IF & 0x1F != 0` after the event once any wait is over (`FL`
/// bit 2, `ControlState::waking`), which wakes the GBC CPU from HALT.
/// `HR` is the 2-dot halves each M-cycle after the pass lasts (2 at normal speed, 1 at
/// double), which the host counts real time by; `DD` is the page OAM DMA writes to and
/// `CL` the bytes an HDMA block moves, so the host places no copy by a constant of its own.
/// `DY` is the M-cycles OAM DMA still blocks the GBC CPU's bus after the pass (0 when no
/// transfer runs).
pub const EXTRA: [&str; 49] = ["RV", "H0", "H1", "FX", "DP", "DX", "DM", "DL", "CS0", "CS1", "CD0", "CD1", "SS0", "SS1", "PA", "RF", "RS", "RP", "RM", "SL0", "SL1", "SL2", "SL3", "SL4", "SL5", "SL6", "SL7", "FE0", "FE1", "HR", "DD", "CL", "LK", "MX0", "MX1", "MY0", "MY1", "SX", "SO", "DF", "FH0", "FH1", "DY", "DB", "DZ", "FR0", "FR1", "RT", "NE"];

/// OAM DMA's destination page and an HDMA block's length, as `DD` and `CL` emit them.
pub const OAM_PAGE: u64 = 0xFE;
pub const HDMA_BLOCK: u64 = 16;

/// `FE1` above the distance: the dropped bit's distance (`FE_DROP_SHIFT`, one bit), the held
/// counter and the wake.
pub const FE_DROP_SHIFT: u8 = 4;
pub const FE_DROP: u8 = 1 << FE_DROP_SHIFT;
pub const FE_HELD: u8 = 32;
pub const FE_WAKE: u8 = 64;

/// The byte the serial cable's other end shows, the line high (0xFF) with no cable.
pub const CABLE: &str = "SI";
/// A write event to FF03, which no bus access reaches: the cable's other end clocked
/// `EX` across (`Event::Receive`).
pub const LINK_PORT: u8 = 0x03;
/// The event ports of a HALT and its wake (`Event::Halt`, `Event::Wake`), FF08 and FF09
/// being unmapped.
pub const HALT_PORT: u8 = 0x08;
pub const WAKE_PORT: u8 = 0x09;
/// The event port of a GBC CPU step's end while an HBlank block waits to be counted
/// (`Event::StepEnd`, `HP`), FF0B being unmapped.
pub const STEP_END_PORT: u8 = 0x0B;
/// The read port of the OAM row a DMG's OAM scan reads at this M-cycle, FF0A being
/// unmapped: the host asks it at each FE page address the GBC CPU puts on the bus, for
/// the OAM corruption bug (`ControlDevice::oam_row`).
pub const OAM_ROW_PORT: u8 = crate::machine::control::OAM_ROW_ADDR as u8;
/// `FL` bits: the GBC CPU halted while an HBlank transfer ran, and that HALT began outside
/// mode 0 (`ControlState::halt`).
pub const FL_HALT: u8 = 0x08;
pub const FL_HALT_ALLOW: u8 = 0x80;
/// A GBC CPU step runs (bit 0): an HDMA block the pass moves waits for its end.
pub const STEP: &str = "IN";
/// The late input: the drawn line's length, the dots from mode 3's start to mode 0's less
/// `lcd::LENGTH_BASE` (`ppu_circuit::circuit::LENGTH`), 0 while it is not known.
pub const LENGTH: &str = "ML";
/// The case keys of mode 0's point and the HBlank request's, which are not constants.
const M0_K: i64 = -1;
const RQ_K: i64 = -2;

/// The most mode-3 starts one pass schedules (`docs/control-in-weights.md` §19).
pub const SCHEDULE: usize = 8;
/// M-cycles from one visible line's mode-3 start to the next, at normal speed.
pub const LINE_MCYCLES: u64 = 114;
/// `LM`'s byte at a visible line's mode-3 start at normal speed: M-cycle 20, 40 halves.
pub const MODE3_LM: u64 = 40;

/// `EV` bits: the event kinds, at most one set.
pub const EV_WRITE: u8 = 1;
pub const EV_READ: u8 = 2;
pub const EV_ACK: u8 = 4;
pub const EV_RAISE: u8 = 8;
pub const EV_BUTTONS: u8 = 16;
pub const EV_STOP: u8 = 32;
/// The next block of a general-purpose HDMA transfer, in place, while `GD` is set.
/// Bit 6 is the cartridge clock's (`clock::EV_CLOCK`).
pub const EV_BLOCK: u8 = 128;

/// `DX` bit 0: the last M-cycle's OAM DMA byte lands before a line that M-cycle starts.
pub const DX_LAST_EARLY: u8 = 1;
/// The M-cycles the GBC CPU stays stopped as the pass begins, as the host counts down the
/// total the last pass emitted (`SS0 SS1`).
pub const OWED: [&str; 2] = ["SQ0", "SQ1"];
/// `RT`: the M-cycles from a schedule's last mode-3 start to the next pass. `NE`: the `EV`
/// code of a further pass at once, at the same M-cycle (0: none), `EV_BLOCK` while a
/// general-purpose transfer has blocks left.

/// HDMA3 bit 7 in the state: a general-purpose transfer has blocks left, each a `Block`
/// event's pass.
pub const GD: u8 = 0x80;
/// HDMA3 bit 5 in the state: an HBlank block moved inside a GBC CPU step and HDMA5 counts it
/// at the step's end (`Hdma::count`), a pass at the `STEP_END_PORT` event.
pub const HP: u8 = 0x20;
/// HDMA3 bit 6 in the state: this line's HBlank block moved before its request, which then
/// moves none (`Hdma::line_done`).
pub const DN: u8 = 0x40;
/// HDMA4 bit 0 in the state: the block `HP` holds ran off the end of VRAM, so its count
/// ends the transfer (`Hdma::count`), whatever the destination reads by the step's end.
pub const HE: u8 = 0x01;

/// `FX` bits: the pass started a serial transfer (the host records the byte sent); an
/// OAM DMA transfer runs (the bus shows the CPU only HRAM); the span's last M-cycle
/// began mode 3 of visible line `LY'` (draw it) or line 144 (show the frame); the write
/// turned the LCD off (blank the screen); the M-cycles to the next pass are a speed
/// switch's pause (counted in `Q`, the CPU stopped, `PS'` M-cycles left); the pass moves
/// an HDMA block (`CS`, `CD`); the span's M-cycle moved an OAM DMA byte (`DP`, `DX`).
pub const FX_SERIAL_START: u8 = 1;
pub const FX_OAM_DMA: u8 = 2;
pub const FX_RENDER: u8 = 4;
pub const FX_FRAME: u8 = 8;
pub const FX_LCD_OFF: u8 = 16;
pub const FX_PAUSE: u8 = 32;
pub const FX_BLOCK: u8 = 64;
pub const FX_OAM_BYTE: u8 = 128;

/// An OAM DMA index past the last byte: no transfer.
pub const DMA_IDLE: u8 = 160;
/// `DI` after an FF46 write: two M-cycles before byte 0.
pub const DMA_START: u8 = crate::machine::dma::DMA_START;

/// The longest span, with the LCD off and nothing else due sooner; with the LCD on the
/// PPU's next point is at most 134 M-cycles away.
pub const MAX_SPAN: u64 = 2048;
/// TIMA counts falling edges of this counter bit for TAC bits 0-1.
const BIT: [usize; 4] = [9, 3, 5, 7];

/// M-cycles of one serial transfer: 8 bits at 8192 Hz.
const TRANSFER: u64 = 1024;
/// M-cycles of a speed switch's pause (`machine::control::SWITCH_MCYCLES`).
const SWITCH: u64 = crate::machine::control::SWITCH_MCYCLES as u64;

fn lit(x: &Lin, set: bool) -> Lin {
    if set { x.clone() } else { nt(x) }
}

pub(super) fn is(x: &[Lin], v: u64) -> Vec<Lin> {
    x.iter().enumerate().map(|(b, l)| lit(l, v >> b & 1 == 1)).collect()
}

/// `(¬s4 ∧ lo) ∨ (¬s5 ∧ hi)`: a P1 input line pulled low by a pressed button of a
/// selected group (select bits are active low).
fn low(c: &mut Circuit, name: &str, s4: &Lin, s5: &Lin, lo: &Lin, hi: &Lin) -> Lin {
    let (a, b) = (nt(s4) + lo.clone(), nt(s5) + hi.clone());
    c.node(name, vec![Step { coef: 1, lin: a.clone(), theta: 2 }, Step { coef: 1, lin: b.clone(), theta: 2 }, Step { coef: -1, lin: a + b, theta: 4 }])
}

/// `coef·[g = n ∧ x ≥ θ]` for `g` the sum of `n` literals, gated at the least weight an
/// f16 unit allows. Gated up, it is `[x + (g − n)·(xhi − θ + 1) ≥ θ]` with live side
/// `xhi − θ + 1`. Gated down, it is `[g ≥ n] − [−x + (g − n)·(θ − xlo) ≥ 1 − θ]` with
/// live side `θ − xlo`. None when both sides pass 511.
fn gated_ge(c: &Circuit, coef: i64, x: &Lin, theta: i64, g: &Lin, n: i64) -> Option<Vec<Step>> {
    let (xlo, xhi) = c.bounds(x);
    let all = Step { coef, lin: g.clone(), theta: n };
    if theta > xhi {
        return Some(Vec::new());
    }
    if theta <= xlo {
        return Some(vec![all]);
    }
    let (up, down) = (xhi - theta + 1, theta - xlo);
    if up <= F16_SPAN {
        Some(vec![Step { coef, lin: x.clone() + (g.clone() - n) * up, theta }])
    } else if down <= F16_SPAN {
        Some(vec![all, Step { coef: -coef, lin: (g.clone() - n) * down - x.clone(), theta: 1 - theta }])
    } else {
        None
    }
}

/// The steps of `within` for a node that sums exclusive cases. When both bounds fit
/// `gated_ge`, each is gated that way and the two bounds' `[g ≥ n]` steps sum into one
/// step, or into none when they cancel.
fn within_steps(c: &Circuit, lits: &[Lin], x: Lin, lo: i64, hi: i64) -> Vec<Step> {
    let n = lits.len() as i64;
    let g = sum(lits.iter().cloned());
    if let (Some(a), Some(b)) = (gated_ge(c, 1, &x, lo, &g, n), gated_ge(c, -1, &x, hi + 1, &g, n)) {
        let gate = |s: &Step| s.lin == g && s.theta == n;
        let konst: i64 = a.iter().chain(&b).filter(|s| gate(s)).map(|s| s.coef).sum();
        let mut steps: Vec<Step> = a.into_iter().chain(b).filter(|s| !gate(s)).collect();
        if konst != 0 {
            steps.push(Step { coef: konst, lin: g, theta: n });
        }
        return steps;
    }
    let (xlo, xhi) = c.bounds(&x);
    let b = xhi - xlo + 2;
    let mut steps = vec![Step { coef: 1, lin: x.clone() + g.clone() * b, theta: lo + b * n }];
    if hi < xhi {
        steps.push(Step { coef: -1, lin: x + g * b, theta: hi + 1 + b * n });
    }
    steps
}

/// `within`, its bounds gated by `gated_ge` where they fit.
fn within(c: &mut Circuit, name: &str, lits: &[Lin], x: Lin, lo: i64, hi: i64) -> Lin {
    let steps = within_steps(c, lits, x, lo, hi);
    c.node(name, steps)
}

/// `coef·[lits ∧ 256·val(high) + low ≥ θ]` for a narrow `low` and a few `high` bits.
/// One step covers the high values that pass θ whatever `low` is, then one step per high
/// value where `low` decides, with that value's bits added to the literals. The steps are
/// exact over every input.
fn ge_digits(c: &Circuit, coef: i64, lits: &[Lin], high: &[Lin], low: &Lin, theta: i64) -> Vec<Step> {
    let (llo, lhi) = c.bounds(low);
    let top = (1i64 << high.len()) - 1;
    let first = |l: i64| (theta - l + 255).div_euclid(256).clamp(0, top + 1);
    let (h0, h1) = (first(lhi), first(llo));
    let wide = |c: &Circuit, x: &Lin, theta: i64, lits: &[Lin]| gated_ge(c, coef, x, theta, &sum(lits.iter().cloned()), lits.len() as i64).unwrap_or_else(|| panic!("a digit step wider than f16 holds"));
    let mut steps = if h1 <= top { wide(c, &val(high), h1, lits) } else { Vec::new() };
    for v in h0..h1 {
        steps.extend(wide(c, low, theta - 256 * v, &[lits.to_vec(), is(high, v as u64)].concat()));
    }
    steps
}

/// `sum_mod` gated by the 0/1 form `gate`, each staircase step through `gated_ge`.
fn sum_mod_gated(c: &mut Circuit, name: &str, terms: &[(Lin, i64)], konst: i64, n: usize, gate: &Lin) -> W {
    let (terms, kk) = positive(terms, konst);
    (0..n)
        .map(|b| {
            let (m, p) = (1i64 << (b + 1), 1i64 << b);
            let form = terms.iter().fold(k(kk.rem_euclid(m)), |a, (x, wt)| a + x.clone() * wt.rem_euclid(m));
            let hi = c.bounds(&form).1;
            let mut steps = Vec::new();
            for j in 1..=hi / p {
                let coef = if j % 2 == 1 { 1 } else { -1 };
                steps.extend(gated_ge(c, coef, &form, j * p, gate, 1).unwrap_or_else(|| panic!("{name}.{b}: a gated step wider than f16 holds")));
            }
            c.node(&format!("{name}.{b}"), steps)
        })
        .collect()
}

/// Bits `0..n` of `x + 1` (or `x − 1` with `down`), each in one level from the bits
/// below it. Bit b toggles when the `b` bits below are all ones (zeros), so with
/// `S` their count and `f = S + (b + 1)·x_b`, bit b is `[f ≥ b] − [f ≥ 2b + 1]`.
fn step_by_one(c: &mut Circuit, name: &str, x: &[Lin], n: usize, down: bool) -> W {
    let lit = |l: &Lin| if down { nt(l) } else { l.clone() };
    (0..n)
        .map(|b| {
            let f = sum(x[..b].iter().map(lit)) + lit(&x[b]) * (b as i64 + 1);
            let bit = c.node(&format!("{name}.{b}"), vec![Step { coef: 1, lin: f.clone(), theta: b as i64 }, Step { coef: -1, lin: f, theta: 2 * b as i64 + 1 }]);
            if down { nt(&bit) } else { bit }
        })
        .collect()
}

pub struct Built {
    pub c: Circuit,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub fn build() -> Built {
    let mut c = Circuit::default();
    let [c0, c1, tima, tma, tc, iflag, ie, jp, jb, sb, sc, sr0, sr1, k0, k1, sv, dr, di, h1, h2, h3, h4, h5, lm, ly, lyc, st, bp, op, ps0, ps1, fl, ur0, ur1, ur2] = STATE.map(|n| w(c.input(n)));
    let clock_state = clock::STATE.map(|n| w(c.input(n)));
    let [d0, d1, ev, ea, ex, q0, q1, hv0, hv1] = EVENTS.map(|n| w(c.input(n)));
    let owed: W = OWED.iter().flat_map(|n| w(c.input(n))).collect();
    // The byte the serial cable's other end shows (0xFF with nothing plugged in).
    let si = w(c.input(CABLE));
    let step = w(c.input(STEP));
    // The line's length from the PPU (44 or more), a late input the first pass sees as 0:
    // not known.
    let ml = w(c.input(LENGTH));
    let clock = clock::build(&mut c, &clock_state, &ev, &ea, &ex);
    let cnt: W = c0.iter().chain(&c1).cloned().collect();
    let d: W = d0.iter().chain(&d1[..4]).cloned().collect();
    let q: W = q0.iter().chain(&q1[..4]).cloned().collect();
    let pz: W = ps0.iter().chain(&ps1).cloned().collect();
    let dbl = k1[7].clone();
    // `H` is the M-cycles the PPU's position moves; the PPU's timing counts 2-dot halves,
    // two an M-cycle at normal speed and one at double speed, so its bits shift by the
    // speed.
    // A DMG in ordinary STOP (`TC` bit 5 set, bit 7 clear) holds its PPU's position, so
    // `run` is 1 or more exactly when the position moves.
    let since: W = hv0.iter().chain(&hv1[..4]).cloned().collect();
    let run = nt(&tc[5]) + tc[7].clone();
    let halves: W = (0..13)
        .map(|b| {
            let mut steps = Vec::new();
            if b < 12 {
                steps.push(Step { coef: 1, lin: (dbl.clone() + since[b].clone()) * 2 + run.clone(), theta: 5 });
            }
            if b > 0 {
                steps.push(Step { coef: 1, lin: (nt(&dbl) + since[b - 1].clone()) * 2 + run.clone(), theta: 5 });
            }
            c.node(&format!("halves.{b}"), steps)
        })
        .collect();
    let (tac, r) = (&tc[..3], tc[3].clone());

    // Level 1: the counter after the span, s = c + 4·D (17 bits), and the decodes.
    let bit = |v: &W, i: usize| v.get(i).cloned().unwrap_or(k(0));
    let mut s: W = vec![cnt[0].clone(), cnt[1].clone()];
    for kb in 2..=16 {
        let low = val(&cnt[..(kb + 1).min(16)]) + val(&d[..(kb - 1).min(d.len())]) * 4;
        s.push(c.bit_of(&format!("s.{kb}"), low, kb as u32));
    }
    let dsum = sum(d.iter().cloned());
    let nd = d.len() as i64;
    let rload = c.ge("rload", r.clone() * nd + dsum.clone(), nd + 1);
    let rstay = c.ge("rstay", r.clone() * nd - dsum.clone(), nd);
    let dpos = c.ge("D>0", dsum.clone(), 1);
    let mpos = c.ge("D+Q>0", dsum.clone() + sum(q.iter().cloned()), 1);
    let hsum = sum(since.iter().cloned());
    let lpos = c.node("H>0", vec![Step { coef: 1, lin: hsum.clone() + nt(&tc[5]) * 13, theta: 14 }, Step { coef: 1, lin: hsum + (tc[5].clone() + tc[7].clone()) * 13, theta: 27 }]);
    let rate: Vec<Lin> = (0..4).map(|kk| all(&mut c, &format!("rate{kk}"), &[vec![tac[2].clone()], is(&tac[..2], kk)].concat())).collect();
    let vsel: Vec<Lin> = (0..4).map(|kk| all(&mut c, &format!("vsel{kk}"), &[vec![ex[2].clone()], is(&ex[..2], kk)].concat())).collect();
    let is_write = |c: &mut Circuit, a: u64| all(c, &format!("w{a:02x}"), &[vec![ev[0].clone()], is(&ea, a)].concat());
    let is_read = |c: &mut Circuit, a: u64| all(c, &format!("r{a:02x}"), &[vec![ev[1].clone()], is(&ea, a)].concat());
    let [w00, w01, w02, w04, w05, w06, w07, w0f, w40, w41, w45, w46, w4c, w4d, w51, w52, w53, w54, w55, w68, w69, w6a, w6b, w70, w72, w73, w74, w75, wff] =
        [0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x0F, 0x40, 0x41, 0x45, 0x46, 0x4C, 0x4D, 0x51, 0x52, 0x53, 0x54, 0x55, 0x68, 0x69, 0x6A, 0x6B, 0x70, 0x72, 0x73, 0x74, 0x75, 0xFF].map(|a| is_write(&mut c, a));
    let w04_fe = w04.clone();
    let w03 = is_write(&mut c, LINK_PORT as u64);
    let (w_halt, w_wake) = (is_write(&mut c, HALT_PORT as u64), is_write(&mut c, WAKE_PORT as u64));
    let w_end = is_write(&mut c, STEP_END_PORT as u64);
    let r0a = is_read(&mut c, OAM_ROW_PORT as u64);
    let [r00, r01, r02, r04, r05, r06, r07, r0f, r41, r44, r45, r46, r4c, r4d, r51, r52, r53, r54, r55, r68, r69, r6a, r6b, r70, r72, r73, r74, r75, rff] =
        [0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x0F, 0x41, 0x44, 0x45, 0x46, 0x4C, 0x4D, 0x51, 0x52, 0x53, 0x54, 0x55, 0x68, 0x69, 0x6A, 0x6B, 0x70, 0x72, 0x73, 0x74, 0x75, 0xFF].map(|a| is_read(&mut c, a));
    let (ack, raise, buttons, stop) = (ev[2].clone(), ev[3].clone(), ev[4].clone(), ev[5].clone());
    // STOP with a switch armed (KEY1 bit 0, only ever set on CGB).
    let sw = all(&mut c, "switch", &[stop.clone(), k1[0].clone()]);
    // The speed after the event: a switch toggles it.
    let k1_7 = sel(&mut c, "K1'.7", &[dbl.clone()], &[(vec![sw.clone()], vec![nt(&dbl)])]).remove(0);
    // Ordinary STOP, without a switch armed: DIV resets and the GBC CPU waits for a joypad
    // line to fall (`Control::stop`).
    let ord = all(&mut c, "ordinary stop", &[stop, nt(&k1[0])]);
    let stat_held = c.node("IF.1 held", vec![Step { coef: 1, lin: iflag[1].clone() + nt(&w0f) + nt(&ack), theta: 3 }, Step { coef: 1, lin: iflag[1].clone() + nt(&w0f) + ack.clone() + nt(&ex[1]), theta: 4 }]);

    // The PPU's timing (`machine::lcd`, docs/control-in-weights.md §14, §17), in 2-dot
    // halves. `md` is the position in the line after the span, a tick (an M-cycle at
    // normal speed) two halves. While the LCD is on the horizon keeps it below 456, so
    // the span crosses at most one line, and one conditional subtract each covers it and
    // LY.
    let lcd = st[7].clone();
    let md = val(&lm) + val(&halves);
    // `md` split into byte digits. The low digit `lm + halves[..8]` spans 510, and the
    // high one is `halves[8..]`. A position `fm = md − 228·cross` lies in `[lo, hi]`, `hi < 228`,
    // exactly when `md` lies in it or 228 above it.
    let (md_lo, md_hi) = (val(&lm) + val(&halves[..8]), halves[8..].to_vec());
    let at = |c: &mut Circuit, name: &str, lits: &[Lin], ranges: &[(i64, i64)]| {
        let steps = ranges.iter().flat_map(|&(lo, hi)| [ge_digits(c, 1, lits, &md_hi, &md_lo, lo), ge_digits(c, -1, lits, &md_hi, &md_lo, hi + 1)].concat()).collect();
        c.node(name, steps)
    };
    let line = |lo: i64, hi: i64| [(lo, hi), (lo + 228, hi + 228)];
    let far = 1 << 14;
    let cross = at(&mut c, "cross", &[lcd.clone()], &[(228, far)]);
    let wrap = at(&mut c, "wrap", &[vec![lcd.clone()], is(&ly, 153)].concat(), &[(228, far)]);
    // Line 144 began on the span's last M-cycle; the state sits at line 144's first tick.
    let vblank = at(&mut c, "vblank", &[vec![lcd.clone()], is(&ly, 143)].concat(), &[(228, 228)]);
    let special = vblank.clone() + at(&mut c, "at 144.0", &[vec![lcd.clone()], is(&ly, 144)].concat(), &[(0, 1)]);
    let fm = md.clone() - cross.clone() * 228;
    let lyf = val(&ly) + cross.clone() - wrap.clone() * 154;
    let terms_of = |l: &Lin| -> (Vec<(Lin, i64)>, i64) { (l.terms.iter().map(|(s, w)| (Lin::from(*s), *w)).collect(), l.c) };
    let (mt_terms, mt_k) = terms_of(&fm);
    let p_t = sum_mod_gated(&mut c, "m_t", &mt_terms, mt_k, 8, &lcd);
    let m_t: W = p_t[1..].to_vec();
    let (ly_terms, ly_k) = terms_of(&lyf);
    let ly_t = sum_mod(&mut c, "LY_t", &ly_terms, ly_k, 8, None);
    // The line's last M-cycle, halves 226-227. LY reads the next line from it, at double
    // speed from its second half (`Lcd::read`).
    let ly_late = at(&mut c, "LY early", &[lcd.clone()], &line(227, 227)) + at(&mut c, "LY early 1x", &[lcd.clone(), nt(&dbl)], &line(226, 226));
    let ly153 = c.eq("LY=153", lyf.clone(), 153);
    // The LY=LYC comparison sees no line in a line's last M-cycle, and on line 153 sees 153
    // in its first M-cycle, none in its second and 0 from its third (`Lcd::compared`).
    let m113 = at(&mut c, "m=113", &[lcd.clone()], &line(226, 227));
    let at153 = |c: &mut Circuit, name: &str, lo: i64, hi: i64| at(c, name, &[lcd.clone(), ly153.clone()], &line(lo, hi));
    let l153_first = at153(&mut c, "153 m=0", 0, 1);
    let l153_zero = at153(&mut c, "153 m>=2", 4, 227);
    let l153_early = at153(&mut c, "153 m<2", 0, 3);
    // LY reads 0 on line 153, a CGB's from its second M-cycle (`Lcd::read`).
    let ly_zero = at(&mut c, "153 reads 0", &[lcd.clone(), ly153.clone(), nt(&tc[7])], &line(0, 227)) + at(&mut c, "153 reads 0 CGB", &[lcd.clone(), ly153.clone(), tc[7].clone()], &line(2, 227));
    // Line 153's last M-cycle reads mode 0 (`Lcd::read`); its interrupt source stays mode 1's.
    let l153_end = at153(&mut c, "153 m=113", 226, 227);
    let compared = |c: &mut Circuit, name: &str, raw: &Lin, lyc0: &Lin| {
        c.node(
            name,
            vec![
                Step { coef: 1, lin: raw.clone() + nt(&m113) + nt(&ly153), theta: 3 },
                Step { coef: 1, lin: l153_first.clone() + raw.clone(), theta: 2 },
                Step { coef: 1, lin: l153_zero.clone() + lyc0.clone(), theta: 2 },
            ],
        )
    };
    let (lyn_terms, lyn_k) = terms_of(&(lyf.clone() + 1 - ly153.clone() * 154));
    let ly_n = sum_mod(&mut c, "LY+1", &lyn_terms, lyn_k, 8, None);
    let vis = c.ge("visible", k(143) - lyf.clone(), 0);
    let vis142 = c.ge("visible to 142", k(142) - lyf.clone(), 0);
    let ly143 = c.eq("LY=143", lyf.clone(), 143);
    let m_lt20 = at(&mut c, "m<20", &[lcd.clone()], &line(0, 39));
    // Line 0 after the LCD turned on (`ST` bit 1, `LcdState::first`): until mode 3 it
    // reads mode 0, raises no mode interrupt and leaves OAM open. It lasts until the line
    // ends.
    // A span can cross several lines when the host counts down scheduled mode-3 starts,
    // so the line is read from LY after the span.
    let ly0_t = c.eq("LY=0 after the span", lyf.clone(), 0);
    let first_t = all(&mut c, "first line", &[st[1].clone(), lcd.clone(), ly0_t]);
    // Mode 0 shows from the half p of a drawn line with 2p ≥ D + 77, 5 dots later at double
    // speed and 2 on a CGB's line 0 after the LCD turned on, D = ML + 128 the dots to mode 0
    // (`Lcd::sees_mode0`); 63 at the least, so past halves 126 the mode needs the length
    // (`MQ`). Not known, the bound stays M-cycle 63. `dk` and `first_k` are the speed's and the
    // first line's terms, which count only with the length known.
    let ml_known = c.ge("ML known", sum(ml.iter().cloned()), 1);
    // A tick's mode after the event takes the speed after it (a switch's new speed, as
    // `Control::stop` sets it); a read never meets a switch.
    let dk = all(&mut c, "ML known 2x", &[ml_known.clone(), k1_7.clone()]);
    let dk_rd = all(&mut c, "ML known 2x read", &[ml_known.clone(), dbl.clone()]);
    let first_k = all(&mut c, "ML known first CGB", &[ml_known.clone(), first_t.clone(), tc[7].clone()]);
    let ml0dk = all(&mut c, "ML.0 2x", &[ml[0].clone(), k1_7.clone()]);
    let ml0dk_rd = all(&mut c, "ML.0 2x read", &[ml[0].clone(), dbl.clone()]);
    let m_ge126 = at(&mut c, "m>=63", &[lcd.clone()], &line(126, 227));
    // A tick's mode, for the STAT interrupt, the locks, the palette ports and the HBlank
    // block, is its second M-cycle's at double speed (`Lcd::mode`): 4m ≥ D + 77 + 3·dk + 2·first_k,
    // 2m ≥ ⌈(ML + 205 + 3·dk + 2·first_k) / 2⌉ = ⌊ML / 2⌋ + 103 + first_k + dk + [ML.0 ∧ dk]. `past_m0`
    // is 2m less that known and 2m − 126 not.
    let past_m0 = val(&m_t) * 2 - val(&ml[1..]) - first_k.clone() - dk.clone() - ml0dk.clone() + ml_known.clone() * 23 - 126;
    let m_ge63 = c.ge("m>=M0", past_m0.clone(), 0);
    let m_mid = at(&mut c, "m>=20", &[lcd.clone()], &line(40, 227)) - m_ge63.clone();
    // A read sees the half it lands at (`Lcd::stat_mode`): p ≥ ⌈(ML + 205 + 5·dk + 2·first_k) / 2⌉ =
    // ⌊ML / 2⌋ + 103 + first_k + 2·dk + [ML.0 ∧ dk].
    let past_rd = val(&p_t) - val(&ml[1..]) - first_k.clone() - dk_rd * 2 - ml0dk_rd + ml_known.clone() * 23 - 126;
    let rd_ge63 = c.ge("read m>=M0", past_rd, 0);
    let rd_mid = at(&mut c, "read m>=20", &[lcd.clone()], &line(40, 227)) - rd_ge63.clone();
    let at20 = at(&mut c, "m=20 now", &[lcd.clone(), lpos.clone()], &line(40, 40));
    // The HBlank HDMA request (`Lcd::request_at`), two dots after STAT's mode 0 at normal speed
    // and one at double speed: at double speed the half p ≥ ⌈(D + 83 + 2·late) / 2⌉ =
    // ⌊ML / 2⌋ + 106 + first_k, at normal speed the M-cycle with 4m ≥ D + 79 + 2·late, 2m ≥
    // ⌊ML / 2⌋ + 104 + first_k. `past_rq` is p or 2m less that, and below 0 while the length is not
    // known. The request is its first position (0 at double speed, 0 or 1 at normal).
    let past_rq = val(&m_t) * 2 + p_t[0].clone() - dbl.clone() * 2 - val(&ml[1..]) - first_k.clone() + ml_known.clone() * 200 - 304;
    let past_req = c.ge("past the HBlank request", past_rq.clone(), 0);
    let at_rq = c.node("HBlank request now", [gated_ge(&c, 1, &past_rq, 0, &lpos, 1), gated_ge(&c, -1, &(past_rq.clone() + dbl.clone()), 2, &lpos, 1)].into_iter().map(|s| s.expect("a request step f16 holds")).flatten().collect());
    // At double speed an odd request half follows its tick's start by one M-cycle.
    let rq_before = c.eq("an M-cycle before the HBlank request", past_rq.clone(), -1);
    let render = all(&mut c, "render", &[at20, vis.clone()]);
    let hblank = all(&mut c, "hblank", &[at_rq, vis.clone()]);
    // `lyf = LY + cross − 154·wrap` spans more than one f16 gate holds beside a byte,
    // so a compare with it splits into the three cases of the span: LY as it is, one
    // line on, or wrapped to 0 (wrap implies cross). Each case's compare reads LY alone, against
    // each of the exclusive sources of the other side.
    let lines_eq = |c: &mut Circuit, name: &str, k: i64, sources: &[(Vec<Lin>, &W)]| {
        let mut steps = Vec::new();
        for (j, (gates, l)) in sources.iter().enumerate() {
            let [same, next] = [k, k + 1].map(|d| c.eq(&format!("{name}@{j}+{d}"), val(&ly) + d - val(l), 0));
            let zero = c.eq(&format!("{name}@{j}=0"), val(l), k);
            for (lits, eq) in [(vec![nt(&cross)], same), (vec![cross.clone(), nt(&wrap)], next), (vec![wrap.clone()], zero)] {
                let lits = [lits, gates.clone(), vec![eq]].concat();
                steps.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
            }
        }
        c.node(name, steps)
    };
    let eq_raw = lines_eq(&mut c, "LY=LYC raw", 0, &[(vec![], &lyc)]);
    let lyc0 = c.eq("LYC=0", val(&lyc), 0);
    let eq_t = compared(&mut c, "LY=LYC", &eq_raw, &lyc0);
    // After the event: LCDC bit 7, the STAT enables and LYC as written. An LCD turned
    // on starts at line 0, M-cycle 0, where the ticks left it while it was off.
    let turned_on = all(&mut c, "lcd on", &[w40.clone(), nt(&lcd), ex[7].clone()]);
    let lcd_off = all(&mut c, "lcd off", &[w40.clone(), lcd.clone(), nt(&ex[7])]);
    let lcd_n = sel(&mut c, "LCD'", &[lcd.clone()], &[(vec![w40.clone()], vec![ex[7].clone()])]).remove(0);
    let en_n = sel(&mut c, "STAT'", &st[3..7].to_vec(), &[(vec![w41.clone()], ex[3..7].to_vec())]);
    let lyc_n = sel(&mut c, "LYC'", &lyc, &[(vec![w45.clone()], ex.clone())]);
    let [e0, e1, e2, el] = [0, 1, 2, 3].map(|i| en_n[i].clone());
    let lyc_src = [(vec![w45.clone()], &ex), (vec![nt(&w45)], &lyc)];
    let eq_n_raw = lines_eq(&mut c, "LY=LYC' raw", 0, &lyc_src);
    let ex0 = c.eq("EX=0", val(&ex), 0);
    let lyc0_n = all(&mut c, "LYC'=0 w", &[w45.clone(), ex0]) + all(&mut c, "LYC'=0 kept", &[nt(&w45), lyc0.clone()]);
    let eq_n = compared(&mut c, "LY=LYC'", &eq_n_raw, &lyc0_n);
    let eq1_n = lines_eq(&mut c, "LY+1=LYC'", 1, &lyc_src);
    let m_lt20_n = m_lt20.clone() + turned_on.clone();
    let first_n = c.node("first line'", vec![Step { coef: 1, lin: turned_on.clone(), theta: 1 }, Step { coef: 1, lin: first_t.clone() + lcd_n.clone(), theta: 2 }]);
    // The STAT line after the event: the LYC match, or the source of the mode (line
    // 144's first M-cycle raising the mode-2 source too). The mode-2 source of lines
    // 1-143 replaces mode 0's in the last M-cycle of the line before, and on a CGB line
    // 144's does too (`Lcd::source`).
    let cgb143 = all(&mut c, "LY=143 CGB", &[ly143.clone(), tc[7].clone()]);
    let mode_src = all(&mut c, "src mode 0", &[vis142.clone(), m_ge63.clone(), e0.clone(), nt(&m113)])
        + all(&mut c, "src mode 0 at 143", &[ly143.clone(), m_ge63.clone(), e0.clone(), nt(&m113)])
        + all(&mut c, "src mode 0 at 143.113", &[ly143.clone(), nt(&tc[7]), m113.clone(), e0.clone()])
        + all(&mut c, "src mode 2 at 143.113", &[cgb143.clone(), m113.clone(), e2.clone()])
        + all(&mut c, "src mode 1", &[nt(&vis), e1.clone()])
        + all(&mut c, "src mode 2", &[vis.clone(), m_lt20_n.clone(), e2.clone(), nt(&first_n)])
        + all(&mut c, "src mode 2 early", &[vis142.clone(), m113.clone(), e2.clone()])
        + all(&mut c, "src 144.0", &[special.clone(), e2.clone(), nt(&e1)]);
    let or_and = |c: &mut Circuit, name: &str, gate: &Lin, a: &[Lin], b: Lin| {
        let na = a.len() as i64;
        let ga = sum(a.iter().cloned()) + gate.clone();
        c.node(name, vec![Step { coef: 1, lin: ga.clone(), theta: na + 1 }, Step { coef: 1, lin: gate.clone() + b.clone(), theta: 2 }, Step { coef: -1, lin: ga + b, theta: na + 2 }])
    };
    let line_on = or_and(&mut c, "STAT line'", &lcd_n, &[eq_n.clone(), el.clone()], mode_src);
    // With the LCD off the comparison clock stops: STAT's LY=LYC bit (`ST` bit 2) keeps
    // the value it had when the LCD went off, LYC writes leave it, and it alone can hold
    // the STAT line (mooneye `ppu/stat_lyc_onoff`). Turning the LCD on compares line 0.
    let coinc = |c: &mut Circuit, name: &str, extra: &[Lin]| {
        let n = extra.len() as i64;
        let ex = sum(extra.iter().cloned());
        c.node(name, vec![Step { coef: 1, lin: nt(&lcd_n) + lcd.clone() + eq_t.clone() + ex.clone(), theta: 3 + n }, Step { coef: 1, lin: nt(&lcd_n) + nt(&lcd) + st[2].clone() + ex, theta: 3 + n }])
    };
    let held = coinc(&mut c, "LY=LYC held'", &[]);
    let coinc_n = held + all(&mut c, "LY=LYC on'", &[lcd_n.clone(), eq_n.clone()]);
    let line_n = line_on + coinc(&mut c, "STAT line off'", &[el.clone()]);
    let lyc_hit = all(&mut c, "LYC hit'", &[eq_n, el.clone()]);

    // CGB's palette index registers. Mode 3 locks the data ports, so a read gives 0xFF and
    // a write is dropped, but the index still advances when bit 7 asks.
    let locked = all(&mut c, "palette locked", &[vis.clone(), m_mid.clone()]);
    let index_n = |c: &mut Circuit, name: &str, reg: &W, w_index: &Lin, w_data: &Lin| {
        let terms: Vec<(Lin, i64)> = reg[..6].iter().enumerate().map(|(b, x)| (x.clone(), 1 << b)).collect();
        let next = sum_mod(c, &format!("{name}+1"), &terms, 1, 6, None);
        let written: W = (0..8).map(|b| if b == 6 { k(0) } else { ex[b].clone() }).collect();
        sel(c, name, reg, &[(vec![w_index.clone()], written), (vec![w_data.clone(), reg[7].clone()], [next, vec![k(0), k(1)]].concat())])
    };
    let bp_n = index_n(&mut c, "BP'", &bp, &w68, &w69);
    let op_n = index_n(&mut c, "OP'", &op, &w6a, &w6b);
    let to_bg = all(&mut c, "palette bg", &[w69.clone() + r69.clone(), nt(&locked)]);
    let to_obj = all(&mut c, "palette obj", &[w6b.clone() + r6b.clone(), nt(&locked)]);
    let pa = c.mux("PA", 8, &[(vec![to_bg], [bp[..6].to_vec(), vec![k(0), k(1)]].concat()), (vec![to_obj], [op[..6].to_vec(), vec![k(1), k(1)]].concat())]);

    // Level 2: the selected counter bits above the TIMA input bit, before and after
    // the span, so `x − y` is the number of edges; TIMA or TMA as the count's start;
    // whether the span's last M-cycle was an edge; the input bit a DIV or TAC write sees.
    let window = |c: &mut Circuit, name: &str, v: &W| -> W {
        (0..9)
            .map(|b| {
                let steps = (0..4).map(|kk| Step { coef: 1, lin: rate[kk].clone() + bit(v, BIT[kk] + 1 + b), theta: 2 }).collect();
                c.node(&format!("{name}.{b}"), steps)
            })
            .collect()
    };
    let x = window(&mut c, "x", &s);
    let y = window(&mut c, "y", &cnt);
    let v: W = (0..8)
        .map(|b| c.node(&format!("v.{b}"), vec![Step { coef: 1, lin: rload.clone() + tma[b].clone(), theta: 2 }, Step { coef: 1, lin: nt(&rload) + tima[b].clone(), theta: 2 }]))
        .collect();
    let z = {
        let steps = (0..4).map(|kk| Step { coef: 1, lin: rate[kk].clone() + sum((0..=BIT[kk]).map(|i| nt(&s[i]))), theta: BIT[kk] as i64 + 2 }).collect();
        c.node("z", steps)
    };
    let input_bit = |c: &mut Circuit, name: &str, which: &[Lin]| {
        let steps = (0..4).map(|kk| Step { coef: 1, lin: which[kk].clone() + s[BIT[kk]].clone(), theta: 2 }).collect();
        c.node(name, steps)
    };
    let before = input_bit(&mut c, "before", &rate);
    let after = input_bit(&mut c, "after", &vsel);

    // Level 3: v + edges as 9 bits; bit 8 is the overflow. A DIV or TAC write's edge.
    let mut terms: Vec<(Lin, i64)> = Vec::new();
    for b in 0..8 {
        terms.push((v[b].clone(), 1 << b));
    }
    for b in 0..9 {
        terms.push((x[b].clone(), 1 << b));
        terms.push((y[b].clone(), -(1 << b)));
    }
    let t = sum_mod(&mut c, "t", &terms, 0, 9, None);
    // A DIV write's, a speed switch's (STOP resets DIV) or a TAC write's falling edge.
    let inc = c.node("inc", vec![Step { coef: 1, lin: w04.clone() + before.clone(), theta: 2 }, Step { coef: 1, lin: sw.clone() + before.clone(), theta: 2 }, Step { coef: 1, lin: ord.clone() + before.clone(), theta: 2 }, Step { coef: 1, lin: w07.clone() + before + nt(&after), theta: 3 }]);

    // Level 4: an overflow on the span's last M-cycle leaves TIMA 0 and the reload
    // pending; one M-cycle earlier, the reload ran and raised the interrupt.
    let reloaded = c.ge("reloaded", t[8].clone() + nt(&z), 2);
    let pending = c.ge("pending", t[8].clone() + z.clone(), 2);
    let tick: W = (0..8).map(|b| t[b].clone() + c.ge(&format!("tma@{b}"), t[8].clone() + nt(&z) + tma[b].clone(), 3)).collect();
    let r_tick = pending + rstay;
    let irq = rload.clone() + reloaded.clone();
    // The reload ran in the span's last M-cycle: the overflow came at the edge of the
    // M-cycle before it (the counter after the span 4 past a multiple of the edge's
    // period) with no edge since (the count's low byte 0), a pending reload over a
    // one-M-cycle span, or the last pass's reload with no ticked M-cycle since (`FL` bit
    // 1). A TIMA write then is lost to TMA, and a TMA write reaches TIMA too (mooneye-gb
    // `tima_write_cycle`, `tma_write_cycle`).
    let low_zero = sum((0..8).map(|b| nt(&t[b])));
    let rl_last = {
        let steps = (0..4)
            .map(|kk| {
                let below = sum((0..=BIT[kk]).filter(|&i| i != 2).map(|i| nt(&s[i])));
                Step { coef: 1, lin: rate[kk].clone() + t[8].clone() + low_zero.clone() + s[2].clone() + below, theta: 11 + BIT[kk] as i64 }
            })
            .collect();
        c.node("reload last", steps)
    };
    let rl_one = c.ge("reload in 1", r.clone() + d[0].clone() - sum(d[1..].iter().cloned()), 2);
    let rl_kept = c.ge("reload kept", fl[1].clone() - dsum.clone(), 1);
    let reload_cycle = rl_last + rl_one + rl_kept;

    // A serial transfer's M-cycles left count down by the span, which never passes the
    // completion. At 0 the byte in is 0xFF, SC bit 7 clears and IF bit 3 rises.
    let rem: W = sr0.iter().chain(&sr1[..3]).cloned().collect();
    let sact = c.ge("serial on", val(&rem), 1);
    let rem_terms: Vec<(Lin, i64)> = rem.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).chain(d.iter().enumerate().map(|(b, x)| (x.clone(), -(1i64 << b)))).collect();
    let rem_t = sum_mod(&mut c, "rem", &rem_terms, 0, 11, Some((&sact, None)));
    let done = within(&mut c, "serial done", &[dpos.clone()], val(&rem) - val(&d), 0, 0);
    let start = all(&mut c, "serial start", &[w02.clone(), ex[7].clone(), ex[0].clone()]);
    // At 0 the byte in is the other end's (`SI`). A byte the other end clocked across
    // lands while no transfer of this end's counts, and completes one waiting on the
    // external clock (SC 0x80), raising IF bit 3.
    let took = all(&mut c, "serial took", &[w03, nt(&sact)]);
    let ext_done = all(&mut c, "serial external done", &[took.clone(), sc[7].clone(), nt(&sc[0])]);
    let sb_t = sel(&mut c, "SB_t", &sb, &[(vec![done.clone()], si.clone())]);
    let sc_t: W = (0..8).map(|b| if b == 7 { c.ge("SC_t.7", sc[7].clone() + nt(&done), 2) } else { sc[b].clone() }).collect();
    let sb_n = sel(&mut c, "SB'", &sb, &[(vec![w01.clone()], ex.clone()), (vec![done.clone(), nt(&w01)], si.clone()), (vec![took.clone()], ex.clone())]);
    let sc_ex: W = (0..8).map(|b| if b == 0 || b == 7 { ex[b].clone() } else { k(0) }).collect();
    let sc_n = sel(&mut c, "SC'", &sc_t, &[(vec![w02.clone()], sc_ex), (vec![ext_done.clone()], (0..8).map(|b| if b == 7 { k(0) } else { sc_t[b].clone() }).collect())]);
    // The internal clock shifts on the counter's bit-8 falling edges, so a transfer
    // ends at the eighth after its start: 1,024 M-cycles less the counter's bits 2-8.
    let aligned: Vec<(Lin, i64)> = (2..9).map(|b| (s[b].clone(), -(1i64 << (b - 2)))).collect();
    let first = sum_mod(&mut c, "rem start", &aligned, TRANSFER as i64, 11, None);
    let rem_n = sel(&mut c, "rem'", &rem_t, &[(vec![start.clone()], first)]);

    // The joypad: P1's input lines before and after a select write or a button change;
    // a selected line falling raises IF bit 4.
    let (s4, s5) = (jp[4].clone(), jp[5].clone());
    let lb: W = (0..4).map(|i| low(&mut c, &format!("P1 low {i}"), &s4, &s5, &jb[i], &jb[i + 4])).collect();
    let lw: W = (0..4).map(|i| low(&mut c, &format!("P1 low {i} after select"), &ex[4], &ex[5], &jb[i], &jb[i + 4])).collect();
    let lk: W = (0..4).map(|i| low(&mut c, &format!("P1 low {i} after buttons"), &s4, &s5, &ex[i], &ex[i + 4])).collect();
    let fw: W = (0..4).map(|i| c.ge(&format!("P1 fall {i} select"), nt(&lb[i]) + lw[i].clone(), 2)).collect();
    let fk: W = (0..4).map(|i| c.ge(&format!("P1 fall {i} buttons"), nt(&lb[i]) + lk[i].clone(), 2)).collect();
    let joy_irq = c.node("joypad irq", vec![Step { coef: 1, lin: w00.clone() * 4 + sum(fw.iter().cloned()), theta: 5 }, Step { coef: 1, lin: buttons.clone() * 4 + sum(fk.iter().cloned()), theta: 5 }]);
    let jp_ex: W = (0..8).map(|b| if b == 4 || b == 5 { ex[b].clone() } else { k(0) }).collect();
    let jp_n = sel(&mut c, "JP'", &jp, &[(vec![w00.clone()], jp_ex)]);
    let jb_n = sel(&mut c, "JB'", &jb, &[(vec![buttons], ex.clone())]);

    // OAM DMA: the index moves one byte per M-cycle of the span, which is one M-cycle
    // while a transfer runs; a write to FF46 starts over at index 0. The source page
    // reads the echo of WRAM for E0-FF.
    // `DI` 0-159 is the next byte's index, 160 idle, and 254 and 255 the two M-cycles
    // after an FF46 write before byte 0, which move no byte (`dma::DMA_START`); `FL` bit 0
    // says the write restarted a transfer that blocked, so those two M-cycles block too.
    let dact = c.ge("dma on", k(DMA_IDLE as i64 - 1) - val(&di), 0);
    let drun = c.node("dma runs", vec![Step { coef: 1, lin: k(DMA_IDLE as i64 - 1) - val(&di), theta: 0 }, Step { coef: 1, lin: val(&di), theta: DMA_START as i64 }]);
    let di_terms: Vec<(Lin, i64)> = di.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).chain(d.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b))).collect();
    let idle = kbits(DMA_IDLE as u64, 8);
    let di_t = sum_mod(&mut c, "DI_t", &di_terms, 0, 8, Some((&drun, Some(&idle))));
    let dr_n = sel(&mut c, "DR'", &dr, &[(vec![w46.clone()], ex.clone())]);
    let di_n = sel(&mut c, "DI'", &di_t, &[(vec![w46.clone()], kbits(DMA_START as u64, 8))]);
    // After the span, before the event: the transfer copies (`DI_t` below 160), or it is
    // before byte 0 of a restart. A write keeps the bus blocked exactly when it was, and
    // the new transfer restarts one that blocked. Each reads `DI_t`, so they sit at `DI'`'s
    // level.
    let restarted = Step { coef: 1, lin: fl[0].clone() * 256 + val(&di_t), theta: 256 + DMA_START as i64 };
    let dact_n = c.node("dma blocks'", vec![Step { coef: 1, lin: k(DMA_IDLE as i64 - 1) - val(&di_t), theta: 0 }, restarted.clone()]);
    let restart_n = c.node("dma restart'", vec![Step { coef: 1, lin: w46.clone() * 256 - val(&di_t), theta: 256 - (DMA_IDLE as i64 - 1) }, restarted]);
    // The transfer still has M-cycles to run after the pass: a write, or `DI_t` copying or
    // before byte 0.
    let drun_n = c.node("dma runs'", vec![Step { coef: 1, lin: w46.clone() * 512 + k(DMA_IDLE as i64 - 1) - val(&di_t), theta: 0 }, Step { coef: 1, lin: val(&di_t) - w46.clone() * 512, theta: DMA_START as i64 }]);
    // `DZ`: the M-cycles before a fresh start blocks, 2 after the write, 1 a cycle later;
    // 0 otherwise (`DY`, the M-cycles it blocks to, counts from the pass too).
    let fresh = |c: &mut Circuit, name: &str, at: i64| {
        let steps = if at == DMA_START as i64 {
            vec![
                Step { coef: 1, lin: w46.clone() * 256 + val(&di_t), theta: 256 + DMA_IDLE as i64 },
                Step { coef: -1, lin: w46.clone() * 256 + val(&di_t), theta: 256 + DMA_START as i64 },
                Step { coef: 1, lin: w46.clone() * 256 + val(&di_t) - fl[0].clone() * 256, theta: 256 + DMA_START as i64 },
                Step { coef: 1, lin: val(&di_t) - w46.clone() * 256 - fl[0].clone() * 256, theta: at },
                Step { coef: -1, lin: val(&di_t) - w46.clone() * 256 - fl[0].clone() * 256, theta: at + 1 },
            ]
        } else {
            vec![Step { coef: 1, lin: val(&di_t) - w46.clone() * 256 - fl[0].clone() * 256, theta: at }]
        };
        c.node(name, steps)
    };
    let dz = [fresh(&mut c, "DZ.0", DMA_START as i64 + 1), fresh(&mut c, "DZ.1", DMA_START as i64)];
    // `DB`: the buses OAM DMA takes (`layout::Dma`): bit 0 OAM, bit 1 VRAM, bit 2 the
    // main bus, bit 3 work RAM, for the transfer FF46 last started (`DR'`), and in the high
    // nibble those it blocks after the pass. A VRAM source takes the VRAM bus, a CGB's
    // C000-DFFF source only work RAM's, any other source every bus but VRAM's. The class
    // reads DR's bits or the write's, so the blocked bits sit at `DI'`'s level, beside the
    // transfer's own bit.
    let src_bits = |c: &mut Circuit, name: &str, pick: &dyn Fn(&W) -> Lin, cgb: bool| {
        let extra = if cgb { tc[7].clone() } else { k(0) };
        let n = 4 + cgb as i64;
        c.node(name, vec![Step { coef: 1, lin: w46.clone() + pick(&ex) + extra.clone(), theta: n }, Step { coef: 1, lin: nt(&w46) + pick(&dr) + extra, theta: n }])
    };
    let vram_src = src_bits(&mut c, "DB vram", &|v: &W| v[7].clone() + nt(&v[6]) + nt(&v[5]), false);
    let wram_src = src_bits(&mut c, "DB wram", &|v: &W| v[7].clone() + v[6].clone() + nt(&v[5]), true);
    let class = [k(1), vram_src.clone(), k(1) - vram_src.clone() - wram_src.clone(), nt(&vram_src)];
    let blocks: W = class
        .iter()
        .enumerate()
        .map(|(b, cl)| {
            let off = (cl.clone() - k(1)) * 512;
            c.node(&format!("DB now.{b}"), vec![Step { coef: 1, lin: k(DMA_IDLE as i64 - 1) - val(&di_t) + off.clone(), theta: 0 }, Step { coef: 1, lin: fl[0].clone() * 256 + val(&di_t) + off, theta: 256 + DMA_START as i64 }])
        })
        .collect();
    let db: W = class.iter().cloned().chain(blocks).collect();
    // `DY`: the M-cycles the transfer still blocks the GBC CPU's bus after the pass, one per
    // byte left, 160 − DI' (0 when idle), which the host counts down to place each access.
    let dy_terms: Vec<(Lin, i64)> = di_n.iter().enumerate().map(|(b, x)| (x.clone(), -(1i64 << b))).collect();
    let dy = sum_mod(&mut c, "DY", &dy_terms, DMA_IDLE as i64, 8, None);
    // While a transfer runs the horizon never passes the end of the 8-byte OAM line `DI`
    // is in, so a span moves a run of bytes from `DI` (as it was before the event) within
    // that line, from page `DR`. The pass emits them: `DP` the source page, `DX` the
    // line (index bits 7-3), and a bit per byte of the line moved: `DL` for the byte of the
    // span's last M-cycle, `DM` for the bytes before it. The machine lands `DM`'s bytes before
    // the renderer draws a line the pass names and `DL`'s after, as the M-cycles order
    // them. Byte j moved when `DI`'s low bits are at most j and the index after the span,
    // `DI_t`, is not past j in the same line (its low bits outside 1..=j); it is the last
    // when `DI_t`'s low bits are j + 1 (0 for byte 7).
    let oam_byte = all(&mut c, "dma byte", &[dpos.clone(), dact.clone()]);
    let last: W = (0..8).map(|j| all(&mut c, &format!("DL.{j}"), &[vec![oam_byte.clone()], is(&di_t[..3], ((j + 1) % 8) as u64)].concat())).collect();
    let dm: W = (0..8)
        .map(|j| {
            let from = c.ge(&format!("dma from<={j}"), k(j as i64) - val(&di[..3]), 0);
            let short = c.node(&format!("dma end in 1..={j}"), vec![Step { coef: 1, lin: val(&di_t[..3]), theta: 1 }, Step { coef: -1, lin: val(&di_t[..3]), theta: j as i64 + 1 }]);
            c.ge(&format!("DM.{j}"), oam_byte.clone() + from - short - last[j].clone(), 2)
        })
        .collect();
    let echo = all(&mut c, "dma echo", &dr[5..8]);
    // A CGB's OAM DMA from E000-FFFF moves no source's byte: it writes 0xFF (`DF`).
    let fill = all(&mut c, "dma fill", &[tc[7].clone(), echo.clone()]);
    let dp: W = (0..8).map(|b| if b == 5 { c.node("DP.5", vec![Step { coef: 1, lin: dr[5].clone(), theta: 1 }, Step { coef: -1, lin: echo.clone(), theta: 1 }]) } else { dr[b].clone() }).collect();

    // HDMA: the source and destination in 16-byte blocks; HDMA5 as it reads (bit 7
    // clear while an HBlank transfer runs, with the blocks left after the next). A pass
    // moves at most one block, from the source and destination before it. A
    // general-purpose transfer moves its first block at the HDMA5 write and the rest at
    // `Block` events in place, while `GD` (HDMA3 bit 7) says more follow. Between those
    // passes HDMA5's bits hold the blocks left as an HBlank transfer's do, so a `Block`
    // event steps exactly as a mode-0 start does. A transfer ends after its last block
    // or after the block at the end of VRAM.
    let src: W = h2[4..].iter().chain(&h1).cloned().collect();
    let dst: W = h4[4..].iter().chain(&h3[..5]).cloned().collect();
    let gd = h3[7].clone();
    let left: W = h5[..7].to_vec();
    let (idle5, hb) = (h5[7].clone(), ex[7].clone());
    let blocks: W = ex[..7].to_vec();
    let terms = |parts: &[(&W, i64)]| -> Vec<(Lin, i64)> { parts.iter().flat_map(|(v, sign)| v.iter().enumerate().map(move |(b, x)| (x.clone(), sign << b))).collect() };
    let src_1 = sum_mod(&mut c, "hdma src", &terms(&[(&src, 1)]), 1, 12, None);
    let dst_1 = sum_mod(&mut c, "hdma dst", &terms(&[(&dst, 1)]), 1, 9, None);
    let left_1 = sum_mod(&mut c, "hdma left", &terms(&[(&left, 1)]), -1, 7, None);
    let blocks_1 = sum_mod(&mut c, "gdma left", &terms(&[(&blocks, 1)]), -1, 7, None);
    let last_dst = all(&mut c, "hdma last dst", &dst);
    let last_dst_h = last_dst.clone();
    let last_block = all(&mut c, "hdma last block", &left.iter().map(nt).collect::<Vec<_>>());
    let one_block = all(&mut c, "gdma one block", &blocks.iter().map(nt).collect::<Vec<_>>());
    let finish = c.ge("hdma finish", last_dst.clone() + last_block, 1);
    let finish_g = c.ge("gdma finish", last_dst + one_block, 1);
    let gstart = vec![w55.clone(), idle5.clone(), nt(&hb)];
    let follow = vec![ev[7].clone(), gd];
    // The speed switch's pause: M-cycles left count down by `Q`, which never passes
    // them; a switch starts it. An HBlank block waits.
    let paused = c.ge("paused", val(&pz), 1);
    let pz_terms: Vec<(Lin, i64)> = pz.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).chain(q.iter().enumerate().map(|(b, x)| (x.clone(), -(1i64 << b)))).collect();
    // Ordinary STOP counts its M-cycles in `Q` too, and holds a pause as it holds the
    // counter.
    let pz_q = sum_mod(&mut c, "pause_t", &pz_terms, 0, 16, Some((&paused, None)));
    let pz_t = sel(&mut c, "pause held", &pz_q, &[(vec![tc[5].clone()], pz.clone())]);
    let pz_n = sel(&mut c, "PS'", &pz_t, &[(vec![sw.clone()], kbits(SWITCH, 16))]);
    let paused_n = c.ge("paused'", val(&pz_n), 1);
    // While the GBC CPU is halted (FL bit 3) no HBlank block moves, nor on a line whose block
    // moved before its request (`DN`, HDMA3 bit 6).
    let dn = h3[6].clone();
    let hblock = vec![hblank.clone(), nt(&idle5), nt(&paused), nt(&tc[5]), nt(&fl[3]), nt(&dn)];
    // An HBlank transfer started while STAT reads mode 0 (the LCD off, or a visible line
    // past mode 3) moves its first block at once (`Hdma::write_control`).
    let m0_now = c.node(
        "mode 0 now",
        vec![
            Step { coef: 1, lin: nt(&lcd), theta: 1 },
            Step { coef: 1, lin: lcd.clone() + vis.clone() + rd_ge63.clone(), theta: 3 },
            Step { coef: 1, lin: first_t.clone() + m_lt20.clone(), theta: 2 },
        ],
    );
    let hnow = vec![w55.clone(), hb.clone(), m0_now.clone()];
    // STAT reads mode 0 on a drawn line whose HBlank request is still to come (`Lcd::in_gap`).
    let in_gap = c.ge("in the HBlank gap", lcd.clone() + vis.clone() + rd_ge63.clone() + nt(&past_req), 4);
    // A HALT while an HBlank transfer runs keeps whether it began outside mode 0; at the
    // wake one block moves if it did and STAT reads mode 0 (`Control::woke`).
    let hwake = vec![w_wake.clone(), fl[3].clone(), fl[7].clone(), m0_now.clone(), nt(&idle5)];
    let halt_n = c.node("FL.3'", vec![Step { coef: 1, lin: w_halt.clone() + nt(&idle5), theta: 2 }, Step { coef: 1, lin: fl[3].clone() + nt(&w_wake) + nt(&w_halt), theta: 3 }]);
    let allow_n = c.node("FL.7'", vec![Step { coef: 1, lin: w_halt.clone() + nt(&idle5) + nt(&m0_now), theta: 3 }, Step { coef: 1, lin: w_halt.clone() + nt(&idle5) + in_gap.clone() + nt(&dn), theta: 4 }, Step { coef: 1, lin: fl[7].clone() + nt(&w_wake) + nt(&w_halt), theta: 3 }]);
    // The four ways a block moves, exclusive: the HDMA5 write that starts a
    // general-purpose transfer, a `Block` event of one, a mode-0 start of an HBlank one,
    // and the HDMA5 write that starts an HBlank one in mode 0.
    let moves = [gstart.clone(), follow.clone(), hblock.clone(), hnow.clone(), hwake.clone()];
    let and = |lits: &[Lin]| Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 };
    // A register write's value, or after a block the next one.
    let cases = |write: Lin, value: &[Lin], next: &[Lin]| -> Vec<(Vec<Lin>, W)> { [(vec![write], value.to_vec())].into_iter().chain(moves.iter().map(|m| (m.clone(), next.to_vec()))).collect() };
    let h1_n = sel(&mut c, "H1'", &h1, &cases(w51, &ex, &src_1[4..]));
    let h2_n = sel(&mut c, "H2'", &h2[4..].to_vec(), &cases(w52, &ex[4..], &src_1[..4]));
    let h3_n = sel(&mut c, "H3'", &h3[..5].to_vec(), &cases(w53, &ex[..5], &dst_1[4..]));
    let h4_n = sel(&mut c, "H4'", &h4[4..].to_vec(), &cases(w54, &ex[4..], &dst_1[..4]));
    // After a block: 0xFF when the transfer ends, else the blocks left after the next.
    // An HBlank block moved in a step waits for the step's end (`HP`), whose pass counts it:
    // the transfer ends after its last block, or when that block ran off VRAM's end (`HE`,
    // kept from the block, which a destination write later in the step does not change).
    let (hp, he) = (h3[5].clone(), h4[0].clone());
    let last_left = all(&mut c, "hdma last block counted", &left.iter().map(nt).collect::<Vec<_>>());
    let finish_c = c.ge("hdma finish counted", he.clone() + last_left, 1);
    let count = vec![w_end.clone(), hp.clone()];
    let hblock_now = [hblock.clone(), vec![nt(&step[0])]].concat();
    let h5_n = sel(
        &mut c,
        "H5'",
        &h5,
        &[
            ([count.clone(), vec![finish_c.clone()]].concat(), kbits(0xFF, 8)),
            ([count.clone(), vec![nt(&finish_c)]].concat(), [left_1.clone(), vec![k(0)]].concat()),
            (vec![w55.clone(), nt(&idle5), nt(&hb)], [blocks.clone(), vec![k(1)]].concat()),
            (vec![w55.clone(), hb.clone(), nt(&m0_now)], [blocks.clone(), vec![k(0)]].concat()),
            ([gstart.clone(), vec![finish_g.clone()]].concat(), kbits(0xFF, 8)),
            ([gstart.clone(), vec![nt(&finish_g)]].concat(), [blocks_1.clone(), vec![k(0)]].concat()),
            ([hnow.clone(), vec![finish_g.clone()]].concat(), kbits(0xFF, 8)),
            ([hnow.clone(), vec![nt(&finish_g)]].concat(), [blocks_1, vec![k(0)]].concat()),
            ([follow.clone(), vec![finish.clone()]].concat(), kbits(0xFF, 8)),
            ([follow.clone(), vec![nt(&finish)]].concat(), [left_1.clone(), vec![k(0)]].concat()),
            ([hblock_now.clone(), vec![finish.clone()]].concat(), kbits(0xFF, 8)),
            ([hblock_now.clone(), vec![nt(&finish)]].concat(), [left_1.clone(), vec![k(0)]].concat()),
            ([hwake.clone(), vec![finish.clone()]].concat(), kbits(0xFF, 8)),
            ([hwake.clone(), vec![nt(&finish)]].concat(), [left_1, vec![k(0)]].concat()),
        ],
    );
    // A block an HBlank start or a wake moved in the gap is the line's. It holds while the
    // line's request is still ahead after the span (`past_req`, the span inside the line),
    // which a span that passes the request clears whether or not it lands on it.
    let dn_n = c.node("DN'", vec![and(&[hnow.clone(), vec![in_gap.clone(), nt(&dn)]].concat()), and(&[hwake.clone(), vec![in_gap.clone(), nt(&dn)]].concat()), and(&[dn.clone(), nt(&past_req), nt(&cross)])]);
    let hp_n = c.node("HP'", vec![and(&[hblock.clone(), vec![step[0].clone()]].concat()), and(&[hp.clone(), nt(&w_end), nt(&w55)])]);
    let he_n = c.node("HE'", vec![and(&[hblock.clone(), vec![step[0].clone(), last_dst_h.clone()]].concat()), and(&[he.clone(), nt(&w_end), nt(&w55)])]);
    let gd_n = c.node("GD'", vec![and(&[gstart.clone(), vec![nt(&finish_g)]].concat()), and(&[follow.clone(), vec![nt(&finish)]].concat())]);
    // The stall the pass adds: a block's 8 µs (8 M-cycles, 16 at double speed) or a speed
    // switch's pause, never both in one pass. Each run of the copier adds one M-cycle
    // (`dma::RUN_MCYCLES`): an HBlank block, a block at a HALT's wake, and a general-purpose
    // transfer's first block, whose `Block` events continue the same run. The GBC CPU stays
    // stopped for them after the M-cycles it already owes: the stops run one after the other.
    // `moves` lists `follow`, the transfer's further blocks, second.
    let run = |i: usize| if i == 1 { 0 } else { crate::machine::dma::RUN_MCYCLES };
    let stall_cases: Vec<(Vec<Lin>, W)> = moves.iter().enumerate().flat_map(|(i, m)| [([m.clone(), vec![nt(&dbl)]].concat(), kbits(8 + run(i), 16)), ([m.clone(), vec![dbl.clone()]].concat(), kbits(16 + run(i), 16))]).chain([(vec![sw.clone()], kbits(SWITCH, 16))]).collect();
    let added = c.mux("stall", 16, &stall_cases);
    let stall_terms: Vec<(Lin, i64)> = owed.iter().chain(&added).zip((0..16).chain(0..16)).map(|(x, b)| (x.clone(), 1i64 << b)).collect();
    let stall = sum_mod(&mut c, "SS", &stall_terms, 0, 16, None);
    let block = c.node("block", moves.iter().map(|m| and(m)).collect());
    // While a GBC CPU step runs, the block lands at its end.
    let held = c.node("block held", moves.iter().map(|m| and(&[m.clone(), vec![step[0].clone()]].concat())).collect());

    // Level 5: TIMA + 1 for a write's edge; IF with the control logic's own bits.
    let inc_v = sum_mod(&mut c, "tima+1", &tick.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).collect::<Vec<_>>(), 1, 8, None);
    let inc_ovf = c.ge("tima=ff", val(&tick), 255);
    // STAT's line after the ticks alone, before the event: the enables and LYC as they were.
    let src_t = all(&mut c, "src_t mode 0", &[vis142.clone(), m_ge63.clone(), st[3].clone(), nt(&m113)])
        + all(&mut c, "src_t mode 0 at 143", &[ly143.clone(), m_ge63.clone(), st[3].clone(), nt(&m113)])
        + all(&mut c, "src_t mode 0 at 143.113", &[ly143.clone(), nt(&tc[7]), m113.clone(), st[3].clone()])
        + all(&mut c, "src_t mode 2 at 143.113", &[cgb143.clone(), m113.clone(), st[5].clone()])
        + all(&mut c, "src_t mode 1", &[nt(&vis), st[4].clone()])
        + all(&mut c, "src_t mode 2", &[vis.clone(), m_lt20.clone(), st[5].clone(), nt(&first_t)])
        + all(&mut c, "src_t mode 2 early", &[vis142.clone(), m113.clone(), st[5].clone()])
        + all(&mut c, "src_t 144.0", &[special.clone(), st[5].clone(), nt(&st[4])]);
    let line_t = or_and(&mut c, "STAT line ticked", &lcd, &[eq_t.clone(), st[6].clone()], src_t);
    // With the LCD off the held LY=LYC bit alone can hold it (`coinc`).
    let line_tf = line_t.clone() + all(&mut c, "STAT line held off", &[nt(&lcd), st[2].clone(), st[6].clone()]);
    // Every STAT rising edge is a horizon or an event, so the line rose in the ticks iff it
    // is high after them and was low at the last pass, and at the event iff it is high
    // after the event and was low after the ticks (a register write can raise it after the
    // ticks dropped it).
    let own = [Some(vblank.clone()), None, Some(irq.clone()), Some(done.clone() + ext_done.clone()), Some(joy_irq.clone())];
    let if_tick: W = (0..5)
        .map(|b| match &own[b] {
            Some(bit) => c.ge(&format!("if.{b}"), iflag[b].clone() + bit.clone(), 1),
            None => c.node("if.1", vec![Step { coef: 1, lin: iflag[1].clone(), theta: 1 }, Step { coef: 1, lin: nt(&iflag[1]) + line_tf.clone() + nt(&st[0]), theta: 3 }, Step { coef: 1, lin: nt(&iflag[1]) + line_n.clone() + nt(&line_tf), theta: 3 }]),
        })
        .collect();

    // Level 6: the event.
    let tima_n = sel(&mut c, "TIMA'", &tick, &[(vec![w05.clone(), nt(&reload_cycle)], ex.clone()), (vec![w06.clone(), reload_cycle.clone()], ex.clone()), (vec![inc.clone()], inc_v.clone())]);
    let r_n = sel(&mut c, "R'", &[r_tick], &[(vec![w05.clone(), nt(&reload_cycle)], vec![k(0)]), (vec![inc.clone()], vec![inc_ovf])]).remove(0);
    let cnt_tick: W = s[..16].to_vec();
    let cnt_n = sel(&mut c, "C'", &cnt_tick, &[(vec![w04], kbits(0, 16)), (vec![sw.clone()], kbits(0, 16)), (vec![ord.clone()], kbits(0, 16))]);
    // In ordinary STOP after the event, until a joypad interrupt; and whether the STOP
    // keeps a CGB's PPU from VRAM, having come outside mode 3.
    let stopped_n = c.node("stopped'", vec![Step { coef: 1, lin: tc[5].clone() + nt(&joy_irq), theta: 2 }, Step { coef: 1, lin: ord.clone() + nt(&tc[5]) + nt(&joy_irq), theta: 3 }]);
    let blocked_n = c.node("blocked'", vec![Step { coef: 1, lin: tc[6].clone() + nt(&ord) + nt(&joy_irq), theta: 3 }, Step { coef: 1, lin: ord.clone() + tc[7].clone() + nt(&locked) + nt(&joy_irq), theta: 4 }]);
    // The CPU waits: a speed switch's pause or ordinary STOP.
    let held_n = c.or("held'", &[paused_n.clone(), stopped_n.clone()]);
    // A DMG's ordinary STOP blanks its screen, as the LCD turned off does, with the timing held.
    let blank = c.node("blank", vec![Step { coef: 1, lin: lcd_off.clone(), theta: 1 }, Step { coef: 1, lin: ord.clone() + nt(&tc[7]) + lcd.clone(), theta: 3 }]);
    let tma_n = sel(&mut c, "TMA'", &tma, &[(vec![w06.clone()], ex.clone())]);
    let tac_n = sel(&mut c, "TAC'", &tac.to_vec(), &[(vec![w07], ex[..3].to_vec())]);
    let ie_n = sel(&mut c, "IE'", &ie, &[(vec![wff], ex.clone())]);
    // KEY1: bit 0 as written, cleared by the switch; bit 7, the speed, toggled by it.
    let k1_0 = sel(&mut c, "K1'.0", &k1[..1].to_vec(), &[(vec![w4d], ex[..1].to_vec()), (vec![sw.clone()], vec![k(0)])]).remove(0);
    // A switch drops the half of a tick that has passed.
    let half_n = sel(&mut c, "half'", &[p_t[0].clone()], &[(vec![sw.clone()], vec![k(0)]), (vec![lcd_off.clone()], vec![k(0)])]).remove(0);
    let sv_n = sel(&mut c, "SV'", &sv, &[(vec![w70], ex.clone())]);
    // FF72-FF74 as written, FF75's bits 4-6 in FL bits 4-6 (`ControlState::spare`).
    let ur0_n = sel(&mut c, "UR0'", &ur0, &[(vec![w72], ex.clone())]);
    let ur1_n = sel(&mut c, "UR1'", &ur1, &[(vec![w73], ex.clone())]);
    let ur2_n = sel(&mut c, "UR2'", &ur2, &[(vec![w74], ex.clone())]);
    let fl75_n = sel(&mut c, "FF75'", &fl[4..7].to_vec(), &[(vec![w75], ex[4..7].to_vec())]);
    // KEY0 as the boot ROM writes it; the memory model sends no write after FF50.
    let k0_n = sel(&mut c, "K0'", &k0, &[(vec![w4c], ex.clone())]);
    let if_n: W = (0..5)
        .map(|b| {
            let (f, e) = (&if_tick[b], &ex[b]);
            c.node(
                &format!("IF'.{b}"),
                vec![
                    Step { coef: 1, lin: f.clone() + nt(&w0f) + nt(&ack), theta: 3 },
                    Step { coef: 1, lin: f.clone() + ack.clone() + nt(e), theta: 3 },
                    Step { coef: 1, lin: w0f.clone() + e.clone(), theta: 2 },
                    Step { coef: 1, lin: raise.clone() + e.clone() + nt(f), theta: 3 },
                ],
            )
        })
        .collect();
    // The HALT wake: an interrupt enabled and requested after the event, unless it waits
    // an M-cycle (`ControlState::waking`, FL bit 2), and the horizon is 1 while it waits.
    // As the reference has it, the span's ticks come first: an interrupt their last
    // M-cycle made pending waits on a CGB, and on a DMG when every bit pending rose late
    // in that M-cycle (`late`: the timer's, an internal serial transfer's, STAT's from
    // mode 2). Then the event: one it made pending waits on a CGB, and with no M-cycle in
    // the span one waiting keeps waiting.
    let pending: Vec<Lin> = (0..5).map(|b| all(&mut c, &format!("pending.{b}"), &[ie_n[b].clone(), if_n[b].clone()])).collect();
    let was: Vec<Lin> = (0..5).map(|b| all(&mut c, &format!("pending before.{b}"), &[ie[b].clone(), iflag[b].clone()])).collect();
    let before = c.or("pending before", &was);
    // STAT's late rises (`Tick::late`): mode 2's at line 0's start and in the last
    // M-cycle of lines 0-142.
    let line0_m0 = at(&mut c, "line 0 m=0", &[vec![lcd.clone()], is(&ly, 153)].concat(), &[(228, 229)]);
    let late_stat = c.node(
        "late STAT",
        vec![
            Step { coef: 1, lin: m113.clone() + vis142.clone() + st[5].clone(), theta: 3 },
            Step { coef: 1, lin: line0_m0 + st[5].clone(), theta: 2 },
        ],
    );
    let late = [None, Some(late_stat), Some(irq.clone()), Some(done.clone()), None];
    // IF after the ticks alone, before the event: the event's own bits (a joypad line, a
    // byte the cable clocked across, a STAT source a register write turned on) wait as
    // an event's do. STAT's line after the ticks reads the enables and LYC before the event.
    let if_t: Vec<Lin> = vec![
        if_tick[0].clone(),
        c.node("if ticked.1", vec![Step { coef: 1, lin: iflag[1].clone(), theta: 1 }, Step { coef: 1, lin: nt(&iflag[1]) + line_t + nt(&st[0]) + mpos.clone(), theta: 4 }]),
        if_tick[2].clone(),
        c.ge("if ticked.3", iflag[3].clone() + done.clone(), 1),
        iflag[4].clone(),
    ];
    let ticked: Vec<Lin> = (0..5).map(|b| all(&mut c, &format!("pending ticked.{b}"), &[ie[b].clone(), if_t[b].clone()])).collect();
    let early: Vec<Lin> = (0..5).map(|b| match &late[b] { Some(l) => all(&mut c, &format!("pending early.{b}"), &[ie[b].clone(), if_t[b].clone(), nt(l)]), None => ticked[b].clone() }).collect();
    let cgb = tc[7].clone();
    let x_cgb = all(&mut c, "wake waits CGB", &[mpos.clone(), nt(&before), cgb.clone()]);
    let x_dmg = all(&mut c, "wake waits DMG", &[mpos.clone(), nt(&before), nt(&cgb)]);
    let x_kept = all(&mut c, "wake keeps waiting", &[nt(&mpos), fl[2].clone()]);
    let n_ticked = sum(ticked.iter().cloned());
    let waits_ticked = c.node(
        "wake waits ticked",
        vec![
            Step { coef: 1, lin: n_ticked.clone() + x_cgb * 5, theta: 6 },
            Step { coef: 1, lin: n_ticked.clone() - sum(early.iter().cloned()) * 5 + x_dmg * 5, theta: 6 },
            Step { coef: 1, lin: n_ticked.clone() + x_kept * 5, theta: 6 },
        ],
    );
    let waits_event = c.ge("wake waits event", cgb - n_ticked, 1);
    let n_pending = sum(pending.iter().cloned());
    let wait_steps = |coef: i64| [&waits_ticked, &waits_event].map(|w| Step { coef, lin: n_pending.clone() + w.clone() * 5, theta: 6 });
    let waking = c.node("waking'", wait_steps(1).to_vec());
    let wake = c.node("wake", [vec![Step { coef: 1, lin: n_pending.clone(), theta: 1 }], wait_steps(-1).to_vec()].concat());
    let p1: W = (0..8).map(|b| if b < 4 { nt(&lb[b]) } else if b == 4 || b == 5 { jp[b].clone() } else { k(1) }).collect();
    // STAT after the ticks: the mode (VBlank 1, 2 until M-cycle 20, 3 until 63, else
    // 0; 0 with the LCD off) and the LYC match.
    // At double speed a read in the first half of M-cycle 0 or 20 sees the mode before
    // (`Lcd::read`).
    let first0 = at(&mut c, "2x m=0 first half", &[lcd.clone(), dbl.clone()], &line(0, 0));
    let first20 = at(&mut c, "2x m=20 first half", &[lcd.clone(), dbl.clone()], &line(40, 40));
    // At double speed a CGB's line 0 after the LCD turned on reads mode 0 through M-cycle 20.
    let first_line20 = at(&mut c, "2x first line m=20", &[lcd.clone(), dbl.clone(), first_t.clone()], &line(40, 41));
    let ly0 = c.eq("LY=0", lyf.clone(), 0);
    let mode0 = c.node(
        "mode.0",
        vec![
            Step { coef: 1, lin: lcd.clone() + nt(&vis) + nt(&l153_end), theta: 3 },
            Step { coef: 1, lin: vis.clone() + rd_mid.clone() + nt(&first20) + nt(&first_line20), theta: 4 },
            Step { coef: 1, lin: vis.clone() + first0.clone() + ly0, theta: 3 },
        ],
    );
    let mode1 = c.node("mode.1", vec![Step { coef: 1, lin: vis.clone() + m_lt20.clone() + nt(&first0) + nt(&first_t), theta: 4 }, Step { coef: 1, lin: vis.clone() + rd_mid.clone() + nt(&first_line20), theta: 3 }]);
    let stat_match = c.node("STAT.2", vec![Step { coef: 1, lin: lcd.clone() + eq_t.clone(), theta: 2 }, Step { coef: 1, lin: nt(&lcd) + st[2].clone(), theta: 2 }]);
    let stat_rd: W = (0..8).map(|b| match b { 0 => mode0.clone(), 1 => mode1.clone(), 2 => stat_match.clone(), 7 => k(1), _ => st[b].clone() }).collect();
    // The OAM row a DMG's OAM scan reads: 8 (m + 1) at M-cycle m of a visible line's mode
    // 2, the first line after the LCD turned on having none, else 0 (`Control::oam_row`).
    let scan = all(&mut c, "OAM scan", &[lcd.clone(), vis.clone(), m_lt20.clone(), nt(&first_t), nt(&tc[7])]);
    let scan_row = c.sum_bits("OAM row", 5, |n| val(&m_t[..n]) + 1);
    let rv = c.mux(
        "RV",
        8,
        &[
            (vec![r00], p1),
            (vec![r01], sb_t.clone()),
            (vec![r02], (0..8).map(|b| if b == 0 || b == 7 { sc_t[b].clone() } else { k(1) }).collect()),
            (vec![r04], s[8..16].to_vec()),
            (vec![r05], tick.clone()),
            (vec![r06], tma.clone()),
            (vec![r07], [tac.to_vec(), kbits(0x1F, 5)].concat()),
            (vec![r0f], [if_tick.clone(), kbits(7, 3)].concat()),
            (vec![r41], stat_rd),
            (vec![r44.clone(), nt(&ly_late), nt(&ly_zero)], ly_t.clone()),
            (vec![r44, ly_late], ly_n),
            (vec![r45], lyc.clone()),
            (vec![r68], (0..8).map(|b| if b == 6 { k(1) } else { bp[b].clone() }).collect()),
            (vec![r6a], (0..8).map(|b| if b == 6 { k(1) } else { op[b].clone() }).collect()),
            // Mode 3 locks the data ports: a read gives 0xFF.
            (vec![r69, locked.clone()], kbits(0xFF, 8)),
            (vec![r6b, locked.clone()], kbits(0xFF, 8)),
            (vec![r46], dr.clone()),
            (vec![r4c], k0.clone()),
            (vec![r4d], [vec![k1[0].clone()], kbits(0x3F, 6), vec![dbl.clone()]].concat()),
            (vec![r51], kbits(0xFF, 8)),
            (vec![r52], kbits(0xFF, 8)),
            (vec![r53], kbits(0xFF, 8)),
            (vec![r54], kbits(0xFF, 8)),
            (vec![r55], h5.clone()),
            (vec![r70], [sv[..3].to_vec(), kbits(0x1F, 5)].concat()),
            (vec![r72], ur0.clone()),
            (vec![r73], ur1.clone()),
            (vec![r74], ur2.clone()),
            (vec![r75], [kbits(0x0F, 4), fl[4..7].to_vec(), vec![k(1)]].concat()),
            (vec![rff], ie.clone()),
            (vec![r0a.clone(), scan.clone()], [kbits(0, 3), scan_row].concat()),
            (vec![r0a, nt(&scan)], kbits(0, 8)),
        ]
        .into_iter()
        .chain(clock.reads)
        .collect::<Vec<_>>(),
    );

    // The last levels compute the horizon, the M-cycles to the next pass. It is 1 while a
    // reload is pending or OAM DMA moves bytes, else the nearest of the timer's reload, a
    // serial transfer's completion, the PPU's next point and, with the LCD off, the
    // longest span. Each candidate is a horizon `h` and the bits of `h − 1`. The low byte
    // digit of `h` spans at most 255, so a pairwise compare splits into byte digits that
    // f16 holds. A candidate wins when it beats every other valid one (ties to the
    // earlier), and its bits are gated by that.
    let en = tac_n[2].clone();
    let sel_n: Vec<Lin> = (0..4).map(|kk| all(&mut c, &format!("sel'{kk}"), &[vec![en.clone()], is(&tac_n[..2], kk)].concat())).collect();
    // The reload's distance at the selected rate is `(256 − TIMA)·2^(B−1) − counter[2..=B] + 1`,
    // which is `X + 2` for `X` the bits of `¬counter[2..=B]` below those of `¬TIMA`. Bit b
    // of X is the selected rate's source bit, 0 with the timer off. The circuit reads
    // ¬TIMA' through its exclusive sources (a TIMA write, a write's edge, the ticks), so
    // the bits do not wait a level on TIMA'. A write reaches TIMA when it is TIMA's
    // outside the reload's M-cycle, or TMA's in it.
    let wt = all(&mut c, "TIMA written", &[w05.clone(), nt(&reload_cycle)]) + all(&mut c, "TIMA written by TMA", &[w06.clone(), reload_cycle.clone()]);
    let x_irq: W = (0..16)
        .map(|b| {
            let steps = (0..4)
                .flat_map(|kk| {
                    let low = BIT[kk] - 1;
                    let s = sel_n[kk].clone();
                    if b < low {
                        vec![and(&[s, nt(&cnt_n[b + 2])])]
                    } else if b - low < 8 {
                        let i = b - low;
                        vec![and(&[s.clone(), wt.clone(), nt(&ex[i])]), and(&[s.clone(), inc.clone(), nt(&inc_v[i])]), and(&[s, nt(&wt), nt(&inc), nt(&tick[i])])]
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            c.node(&format!("hirq.{b}"), steps)
        })
        .collect();
    let ser_on = c.ge("serial on'", val(&rem_n), 1);
    // Each candidate is its name, its horizon, its validity and the bits of the horizon
    // less one.
    let mut cands: Vec<(&str, Lin, Lin, W)> = vec![("hirq", val(&x_irq) + 2, en.clone(), step_by_one(&mut c, "hirq+1", &x_irq, 11, false))];
    cands.push(("hmax", k(MAX_SPAN as i64), nt(&lcd_n), kbits(MAX_SPAN - 1, 11)));
    cands.push(("hser", val(&rem_n), ser_on.clone(), step_by_one(&mut c, "hser-1", &rem_n, 10, true)));
    // While OAM DMA moves bytes, the end of the 8-byte line the next byte is in: 8 less
    // `DI'`'s low bits, so the bits of the horizon less one are those bits inverted. None
    // move in a pause.
    let oam_moving = all(&mut c, "dma moving'", &[drun_n.clone(), nt(&held_n)]);
    cands.push(("hoam", k(8) - val(&di_n[..3]), oam_moving, di_n[..3].iter().map(nt).collect()));
    // The PPU's timing, from the state after the event: a visible line's mode-3 start
    // (draw it); its mode-0 start while an HBlank HDMA runs (a block moves) or where the
    // STAT line rises; the next line's start where the line rises there; every line
    // start from line 143's to line 0; and line 144's second M-cycle where its first
    // raised the mode-2 source. Exclusive cases, so the distance is one affine form
    // `K − m`.
    // A rise of the STAT line matters only while IF bit 1 can be clear after the event:
    // once it is set, a rise changes nothing a reader sees, and the line itself follows
    // the position at the next pass. Bit 1 stays set when it was set and the event is
    // neither an IF write nor an acknowledgement of it; both of those are full spans, so
    // the horizon picks the rises up again from where the ticks left the position. Mode
    // 0's start raises the line when its source is enabled and the LYC match is not
    // already holding it high.
    let stat_live = nt(&stat_held);
    let rise63 = all(&mut c, "rises at mode 0", &[e0.clone(), nt(&lyc_hit), stat_live.clone()]);
    // In a visible line's last M-cycle the mode-2 source replaces mode 0's and the LYC match
    // drops, so only mode 2's source holds the line into the next line's start, where the
    // LYC match moves to the next line.
    let rise113 = all(&mut c, "rises at m=113", &[e2.clone(), nt(&lyc_hit), nt(&e0), stat_live.clone()]);
    // A line held high in the last M-cycle's lead-up drops there when the source that
    // holds it lapses (the LYC match always; mode 0's where mode 2's replaces it, on a DMG
    // line 143 never), so a rise at the next line's start (its LYC match, or line 144's
    // mode 1, or on a DMG its mode 2) is a rise only if a pass there saw the line low:
    // one more pass at M-cycle 113.
    let dip_lyc = all(&mut c, "dips before LYC", &[e0.clone(), nt(&e2), eq1_n.clone(), el.clone(), stat_live.clone()]);
    let dip_vblank = all(&mut c, "dips before mode 1", &[e0.clone(), nt(&e2), cgb143.clone(), e1.clone(), stat_live.clone()]);
    let dip_held_cgb = all(&mut c, "LYC dips before mode 1 CGB", &[lyc_hit.clone(), nt(&e2), cgb143.clone(), e1.clone(), stat_live.clone()]);
    let rise144 = c.ge("mode 1 or 2 at 144", e1.clone() + e2.clone(), 1);
    let dip_held_dmg = all(&mut c, "LYC dips before 144 DMG", &[lyc_hit.clone(), nt(&e0), ly143.clone(), nt(&tc[7]), rise144, stat_live.clone()]);
    let rise113 = c.or("pass at m=113", &[rise113, dip_lyc, dip_vblank, dip_held_cgb]);
    let rising = all(&mut c, "rises next line", &[eq1_n, el.clone(), nt(&e2), stat_live.clone()]);
    // While OAM DMA runs every line start is a pass too, so a run ends at each mode 2.
    let rising = c.or("line start matters", &[rising, drun_n.clone()]);
    // Mode 0's start matters where the STAT line rises there, or while an HBlank HDMA runs
    // and the length is not known yet (the pass at M-cycle 63 asks for it); the HBlank
    // request matters while an HBlank HDMA runs (`to_rq`).
    let hdma_on = nt(&h5_n[7]);
    let unknown = nt(&ml_known);
    let need63 = c.node("mode 0 matters", vec![Step { coef: 1, lin: rise63.clone(), theta: 1 }, Step { coef: 1, lin: hdma_on.clone() + unknown + nt(&rise63), theta: 3 }]);
    let ppu_on = |c: &mut Circuit, name: &str, lits: &[Lin]| all(c, name, &[vec![lcd_n.clone()], lits.to_vec()].concat());
    let to_m3 = ppu_on(&mut c, "to mode 3", &[vis.clone(), m_lt20_n.clone()]);
    let to_m0 = ppu_on(&mut c, "to mode 0", &[vis.clone(), m_mid.clone(), need63.clone()]);
    let in_m3_pass = all(&mut c, "mode 0 pass due", &[m_mid.clone(), need63.clone()]);
    // The length known means a visible line past M-cycle 63.
    let rq_due = all(&mut c, "request pass due", &[vis.clone(), hdma_on.clone(), ml_known.clone(), nt(&past_req), nt(&in_m3_pass)]);
    let to_rq = ppu_on(&mut c, "to the HBlank request", &[rq_due.clone()]);
    // Past mode 3's start with no mode-0 or request pass to make.
    let late = at(&mut c, "m>=20 again", &[lcd.clone()], &line(40, 227)) - in_m3_pass - rq_due;
    // Exclusive by line: lines 0-142 and a CGB's 143, then a DMG's 143.
    let go113 = c.node(
        "to m=113",
        vec![
            Step { coef: 1, lin: vis142.clone() + cgb143.clone() + late.clone() + nt(&m113) + rise113 + lcd_n.clone(), theta: 5 },
            Step { coef: 1, lin: dip_held_dmg + late.clone() + nt(&m113) + lcd_n.clone(), theta: 4 },
        ],
    );
    let to_next_m3 = ppu_on(&mut c, "to next mode 3", &[vis142.clone(), late.clone(), nt(&go113), nt(&rising)]);
    let special_n = all(&mut c, "144.1 next", &[lcd_n.clone(), special.clone(), e2.clone(), stat_live.clone()]);
    let at_special = ppu_on(&mut c, "at 144.0'", &[special.clone()]);
    let to_start = ppu_on(&mut c, "to line start", &[vis142.clone(), late.clone(), nt(&go113), rising]);
    let to_144 = ppu_on(&mut c, "to line 144", &[ly143, late, nt(&go113)]);
    // LYC 0 matches from line 153's third M-cycle, a rise unless mode 1's source holds.
    let to_153_2 = ppu_on(&mut c, "to 153.2", &[l153_early.clone(), lyc0_n.clone(), el.clone(), nt(&e1), stat_live.clone()]);
    // LYC 153 holds the line in line 153's first M-cycle only; with mode 1's source off and
    // mode 2's on, line 0 raises it again, a rise only if a pass at 153.1 saw it drop.
    let to_153_1 = ppu_on(&mut c, "to 153.1", &[l153_first.clone(), lyc_hit.clone(), nt(&e1), e2.clone(), stat_live.clone()]);
    let to_vblank = ppu_on(&mut c, "to vblank line", &[nt(&vis), nt(&special), nt(&to_153_2), nt(&to_153_1)]);
    // The line-timing schedule (`docs/control-in-weights.md` §19). When nothing but the
    // PPU's mode-3 starts can need a pass before line 144 (the LCD on at normal speed,
    // IF bit 1 held so no STAT rise matters, no HBlank HDMA, the timer, serial, OAM DMA
    // and a pause all idle, no reload pending), the pass schedules the next mode-3 starts
    // instead of stopping at the first. `g_j` is 1 while the j-th one after the first is
    // still a visible line, up to `SCHEDULE`. The horizon stays the first start. With a
    // schedule, the next pass is the start after the last scheduled one, which draws its
    // own line. A schedule is valid only where every other candidate is not, so it never
    // skips past one. A pass at any event replaces the schedule.
    let sched = all(&mut c, "schedule", &[stat_held.clone(), h5_n[7].clone(), nt(&k1_7), nt(&en), nt(&ser_on), nt(&drun_n), nt(&held_n), nt(&r_n), nt(&waking)]);
    let g_sched: Vec<Lin> = (1..=SCHEDULE as i64)
        .map(|j| {
            let gate = |case: &Lin, last: i64| gated_ge(&c, 1, &(k(last - j) - val(&ly_t)), 0, &(sched.clone() + case.clone()), 2).expect("a schedule step f16 holds");
            let steps = [gate(&to_m3, 143), gate(&to_next_m3, 142)].concat();
            c.node(&format!("sched>={j}"), steps)
        })
        .collect();
    // The count's bits from its thermometer code, and the line each start draws.
    let g_at = |j: usize| g_sched.get(j - 1).cloned().unwrap_or(k(0));
    let rs: W = (0..4).map(|b| (1..=SCHEDULE >> b).fold(k(0), |a, i| a + g_at(i << b) * if i % 2 == 1 { 1 } else { -1 })).collect();
    let ly_terms_r: Vec<(Lin, i64)> = ly_t.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).chain([(to_next_m3.clone(), 1)]).collect();
    let sl: Vec<W> = (0..SCHEDULE).map(|j| sum_mod(&mut c, &format!("SL{j}"), &ly_terms_r, j as i64, 8, None)).collect();
    // The exclusive cases, each a conjunction, with the M-cycles to its point K.
    let cases: Vec<(Vec<Lin>, i64)> = vec![
        (vec![to_m3], 20),
        (vec![to_m0], M0_K),
        (vec![to_rq], RQ_K),
        (vec![to_next_m3], 134),
        (vec![go113], 113),
        (vec![to_start], 114),
        (vec![to_144], 114),
        (vec![to_vblank], 114),
        (vec![to_153_2], 2),
        (vec![to_153_1], 1),
        (vec![at_special, nt(&special_n)], 114),
        (vec![special_n], 1),
    ];
    let ppu_ok = sum(cases.iter().map(|(lits, _)| sum(lits.iter().cloned()) - (lits.len() as i64 - 1)));
    // The circuit computes the distance less one, `T1 = K − m − 1`, per K from m alone,
    // and the case then selects its K's bits, one level after the cases. At double speed,
    // after the event, the distance is two M-cycles a tick less the one M-cycle of the
    // current tick that has passed.
    // An LCD turned on at double speed starts at M-cycle 1 (`Lcd::write`), which the ticks,
    // the LCD off, did not reach.
    let on2 = all(&mut c, "lcd on 2x", &[turned_on.clone(), dbl.clone()]);
    let m_neg: Vec<(Lin, i64)> = m_t.iter().enumerate().map(|(b, x)| (x.clone(), -(1i64 << b))).chain([(on2.clone(), -1)]).collect();
    // Mode 0's point is the tick ⌈(D + 77 + 3·dk + 2·first_k) / 4⌉ = ⌊ML / 4⌋ + 52 and the carries
    // of ML's two low bits with 3·dk + 2·first_k, or 63 while the length is not known.
    let low = ml[0].clone() + ml[1].clone() * 2 + dk.clone() * 3 + first_k.clone() * 2;
    let carries = [4, 8].map(|t| c.ge(&format!("M0 carry {t}"), low.clone(), t));
    let m0_terms: Vec<(Lin, i64)> = m_neg.iter().cloned().chain(ml[2..].iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b))).chain([(ml_known.clone(), -11)]).chain(carries.iter().map(|x| (x.clone(), 1))).collect();
    // The HBlank request's tick: at normal speed ⌈(D + 79 + 2·late) / 4⌉ = ⌊ML / 4⌋ + 52 and
    // the carry of ML's two low bits with 2 + 2·first_k; at double speed the request half over 2,
    // ⌊ML / 4⌋ + 53 + [ML.1 ∧ first_k] (an odd half one M-cycle on, `rq_before`).
    let c_d = all(&mut c, "RQ carry 2x", &[dbl.clone(), ml[1].clone(), first_k.clone()]);
    let c_n = c.ge("RQ carry", ml[0].clone() + ml[1].clone() * 2 + first_k.clone() * 2 - dbl.clone() * 4, 2);
    let rq_terms: Vec<(Lin, i64)> = m_neg.iter().cloned().chain(ml[2..].iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b))).chain([(dbl.clone(), 1), (c_d, 1), (c_n, 1)]).collect();
    let per_k: BTreeMap<i64, W> = cases
        .iter()
        .map(|&(_, kc)| kc)
        // The locks' spans also end at a DMG's M-cycle 19, and 19 into the next line.
        .chain([19, 133])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|kc| (kc, if kc == M0_K { sum_mod(&mut c, "hppu@M0", &m0_terms, 62, 8, None) } else if kc == RQ_K { sum_mod(&mut c, "hppu@RQ", &rq_terms, 51, 8, None) } else { sum_mod(&mut c, &format!("hppu@{kc}"), &m_neg, kc - 1, 8, None) }))
        .collect();
    let and_steps = |lits: &[Lin], coef: i64| Step { coef, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 };
    let t1: W = (0..8).map(|b| c.node(&format!("hppu.{b}"), cases.iter().map(|(lits, kc)| and_steps(&[lits.clone(), vec![per_k[kc][b].clone()]].concat(), 1)).collect())).collect();
    cands.push(("hppu", val(&t1) + 1, ppu_ok.clone() - k1_7.clone(), t1.clone()));
    let t1_out = t1.clone();
    cands.push(("hppu2", val(&t1) * 2 + 2 - half_n.clone(), ppu_ok + k1_7.clone() - 1, [vec![nt(&half_n)], t1].concat()));
    // The GBC CPU's access locks after the event (`LK` bits 0-3: reads' VRAM and OAM, then
    // writes', `Lcd::locks`) and how long they hold at the least (`MX`), then the locks
    // after that (bits 4-7) and until when (`MY`), each less one in 11 bits. Mode 2 holds to
    // mode 3's start and mode 3 to mode 0's (M-cycle 63 while the length is not known);
    // mode 0 holds to the line's end, and the next line's mode 2 to its mode 3. In VBlank
    // and line 143's mode 0 both hold to the line's end. The memory model asks again past
    // them; with the LCD off they hold to the longest span.
    let lock_v = all(&mut c, "VRAM locked'", &[lcd_n.clone(), vis.clone(), m_mid.clone()]);
    // Before mode 3, OAM is locked in mode 2, and open in line 0's opening after the LCD
    // turned on; mode 3 follows both.
    let pre3 = all(&mut c, "before mode 3'", &[lcd_n.clone(), vis.clone(), m_lt20_n.clone()]);
    let opening = all(&mut c, "opening'", &[pre3.clone(), first_n.clone()]);
    let in_m2 = pre3.clone() - opening.clone();
    let in_m0 = all(&mut c, "mode 0 before a line'", &[lcd_n.clone(), vis142.clone(), m_ge63.clone()]);
    let quiet = all(&mut c, "unlocked to the line's end'", &[lcd_n.clone(), nt(&pre3), nt(&lock_v), nt(&in_m0)]);
    // A DMG's reads and writes differ at two M-cycles (`Lcd::locks`): mode 2's last locks a
    // VRAM read and opens an OAM write, and an OAM read is locked in the last M-cycle of
    // lines 0-142 and 153. So a DMG's mode 2 holds to M-cycle 19, its mode 0 to 113, and
    // line 153 to 113.
    let (dmg, cgb) = (nt(&tc[7]), tc[7].clone());
    let m19 = at(&mut c, "m=19", &[lcd.clone()], &line(38, 39));
    let d_m2 = all(&mut c, "mode 2 to 19 DMG'", &[in_m2.clone(), dmg.clone(), nt(&m19)]);
    let d_m2_19 = all(&mut c, "mode 2 at 19 DMG'", &[in_m2.clone(), dmg.clone(), m19]);
    let c_m2 = all(&mut c, "mode 2 CGB'", &[in_m2.clone(), cgb.clone()]);
    let d_m0 = all(&mut c, "mode 0 to 113 DMG'", &[in_m0.clone(), dmg.clone(), nt(&m113)]);
    let d_m0_113 = all(&mut c, "mode 0 at 113 DMG'", &[in_m0.clone(), dmg.clone(), m113.clone()]);
    let c_m0 = all(&mut c, "mode 0 CGB'", &[in_m0.clone(), cgb]);
    let q153 = all(&mut c, "line 153 DMG'", &[quiet.clone(), dmg.clone(), ly153.clone()]);
    let d_153 = all(&mut c, "153 to 113 DMG'", &[q153.clone(), nt(&m113)]);
    let d_153_113 = all(&mut c, "153 at 113 DMG'", &[q153.clone(), m113.clone()]);
    let quiet_rest = quiet.clone() - q153;
    let segment = |c: &mut Circuit, name: &str, cases: &[(Lin, i64)]| -> W {
        let t: W = (0..8).map(|b| c.node(&format!("{name}1.{b}"), cases.iter().map(|(g, kc)| and_steps(&[g.clone(), per_k[kc][b].clone()], 1)).collect())).collect();
        // At double speed a tick is two M-cycles, less the one of the current tick passed.
        (0..11)
            .map(|b| {
                let mut steps = vec![Step { coef: 1, lin: nt(&lcd_n), theta: 1 }];
                if b < 8 {
                    steps.push(Step { coef: 1, lin: lcd_n.clone() + nt(&k1_7) + t[b].clone(), theta: 3 });
                }
                match b {
                    0 => steps.push(Step { coef: 1, lin: lcd_n.clone() + k1_7.clone() + nt(&half_n), theta: 3 }),
                    1..=8 => steps.push(Step { coef: 1, lin: lcd_n.clone() + k1_7.clone() + t[b - 1].clone(), theta: 3 }),
                    _ => {}
                }
                c.node(&format!("{name}.{b}"), steps)
            })
            .collect()
    };
    let mx = segment(
        &mut c,
        "MX",
        &[(opening.clone(), 20), (d_m2.clone(), 19), (d_m2_19.clone(), 20), (c_m2.clone(), 20), (lock_v.clone(), M0_K), (d_m0.clone(), 113), (d_m0_113.clone(), 114), (c_m0.clone(), 114), (quiet_rest.clone(), 114), (d_153.clone(), 113), (d_153_113.clone(), 114)],
    );
    // Mode 0 follows mode 3 only once the length is known; before, the locks hold to M-cycle
    // 63 and no further.
    let v_known = all(&mut c, "VRAM locked, length known'", &[lock_v.clone(), ml_known.clone()]);
    let v_unknown = lock_v.clone() - v_known.clone();
    // A DMG's mode 3 is followed by mode 0 to M-cycle 113 on lines 0-142.
    let vk_dmg = all(&mut c, "VRAM locked, length known, DMG'", &[v_known.clone(), dmg, vis142.clone()]);
    let vk_rest = v_known - vk_dmg.clone();
    let my = segment(
        &mut c,
        "MY",
        &[(opening.clone(), M0_K), (d_m2.clone(), 20), (d_m2_19.clone(), M0_K), (c_m2.clone(), M0_K), (vk_dmg, 113), (vk_rest, 114), (v_unknown, M0_K), (d_m0.clone(), 114), (d_m0_113.clone(), 133), (c_m0.clone(), 134), (quiet_rest, 114), (d_153.clone(), 114), (d_153_113.clone(), 133)],
    );
    let lk = [
        lock_v.clone() + d_m2_19.clone(),
        lock_v.clone() + in_m2 + d_m0_113.clone() + d_153_113.clone(),
        lock_v.clone(),
        lock_v.clone() + c_m2.clone() + d_m2.clone(),
        pre3.clone(),
        pre3 + c_m0.clone() + d_m0 + d_m0_113.clone() + d_153 + d_153_113.clone(),
        opening.clone() + d_m2_19.clone() + c_m2.clone(),
        opening + d_m2_19 + c_m2 + c_m0 + d_m0_113 + d_153_113,
    ];
    // The PPU's horizons under each case, and the speed literal that makes each valid.
    let ppu_eff = |j: usize, kc: i64| if j == 4 { val(&per_k[&kc]) + 1 } else { val(&per_k[&kc]) * 2 + 2 - half_n.clone() };
    let ppu_speed = |j: usize| if j == 4 { nt(&k1_7) } else { k1_7.clone() };
    // Longer than the longest span only with the LCD off, where that span wins.
    cands.push(("hpause", val(&pz_n), paused_n.clone(), step_by_one(&mut c, "hpause-1", &pz_n, 11, true)));
    // An invalid candidate sits 4,096 above its form. That is past every valid horizon
    // (at most 2,048) even from the least an invalid form takes (−127), and it moves only
    // the high digit, so a compare's low digit depends on the two horizons alone.
    let eff = |(_, h, ok, _): &(&str, Lin, Lin, W)| h.clone() + (k(1) - ok.clone()) * 4096;
    // One comparison per pair, since j beats i exactly when i does not beat j.
    let n = cands.len();
    // An odd request half at double speed: the pass at its tick's start waits one M-cycle.
    let one_more = all(&mut c, "request an M-cycle on", &[rq_before.clone(), dbl.clone(), hdma_on.clone(), vis.clone(), lcd_n.clone(), ml_known.clone()]);
    let mut gates: Vec<Lin> = cands.iter().map(|(_, _, ok, _)| ok.clone() - r_n.clone() - waking.clone() - one_more.clone() - (n as i64 - 1)).collect();
    for i in 0..n {
        for j in i + 1..n {
            let name = format!("{}<{}", cands[i].0, cands[j].0);
            let beats = match (i, j) {
                // At most one of the two is valid, by the speed.
                (4, 5) => nt(&k1_7),
                // A PPU horizon is valid under one case at a time, so the node compares
                // per case, from the horizon's bits under that case, which are early. It
                // sums each case's compare steps gated by the case, one level after the
                // cases. An invalid PPU horizon loses, so `beats` is 1 when it is j and 0
                // when it is i.
                (_, 4 | 5) | (4 | 5, _) => {
                    let p = if j == 4 || j == 5 { j } else { i };
                    let mut steps = if p == j { vec![Step { coef: 1, lin: k(1), theta: 1 }] } else { Vec::new() };
                    for (ci, (lits, kc)) in cases.iter().enumerate() {
                        let g = [lits.clone(), vec![ppu_speed(p)]].concat();
                        let (gs, n) = (sum(g.iter().cloned()), g.len() as i64);
                        let form = if p == j { ppu_eff(p, *kc) - eff(&cands[i]) } else { eff(&cands[j]) - ppu_eff(p, *kc) };
                        let direct = gated_ge(&c, 1, &form, 0, &gs, n).filter(|st| st.iter().all(|s| c.step_ok(&s.lin, s.theta)));
                        let cmp = match direct {
                            Some(st) => st,
                            None => {
                                let split = c.split(&format!("{name}@{ci}"), &form, 0).unwrap_or_else(|| panic!("{name}: a compare no digit split holds"));
                                split.iter().flat_map(|s| gated_ge(&c, s.coef, &s.lin, s.theta, &gs, n).unwrap_or_else(|| panic!("{name}: a digit step wider than f16 holds"))).collect()
                            }
                        };
                        if p == j {
                            steps.push(Step { coef: -1, lin: gs, theta: n });
                        }
                        steps.extend(cmp);
                    }
                    c.node(&name, steps)
                }
                _ => c.ge(&name, eff(&cands[j]) - eff(&cands[i]), 0),
            };
            gates[i] = gates[i].clone() + beats.clone();
            gates[j] = gates[j].clone() + nt(&beats);
        }
    }
    // `H` is the horizon less one, 0 to 2047 in 11 bits. It is the bits of the candidate
    // whose gate is open, and 0 while a reload is pending (every gate is closed then).
    let h: Vec<Lin> = (0..11)
        .map(|b| {
            let steps = cands.iter().zip(&gates).filter(|((_, _, _, bits), _)| b < bits.len()).map(|((_, _, _, bits), g)| Step { coef: 1, lin: g.clone() + bits[b].clone(), theta: 2 }).collect();
            c.node(&format!("H.{b}"), steps)
        })
        .collect();

    // The frame sequencer's edges (`docs/control-in-weights.md` §22). The count carries
    // out of bit 12 (13) `2^10·4 − counter[2..=12]` M-cycles on, so the distance less one
    // is the counter's bits 2-12 (2-13) inverted, from the counter after the event. A DIV
    // write or a switch drops a set bit, at the speed before the event.
    let fe: W = (0..12).map(|b| if b < 11 { nt(&cnt_n[b + 2]) } else { all(&mut c, "FE.11", &[k1_7.clone(), nt(&cnt_n[13])]) }).collect();
    // A drop from a pass with no M-cycle since still falls at the next one (`TC` bit 4);
    // the counter is 0 then, so no new drop joins it.
    let mut drop_steps: Vec<Step> = [&w04_fe, &sw, &ord].iter().flat_map(|e| [Step { coef: 1, lin: (*e).clone() + nt(&dbl) + s[12].clone(), theta: 3 }, Step { coef: 1, lin: (*e).clone() + dbl.clone() + s[13].clone(), theta: 3 }]).collect();
    drop_steps.push(Step { coef: 1, lin: tc[4].clone() + nt(&mpos), theta: 2 });
    let drop = c.node("FE drop", drop_steps);
    // Where the bit next rises by counting: half a period from where it falls, so the
    // distance less one is the same inverted bits with the top one flipped back.
    let fr10 = c.node("FR.10", vec![Step { coef: 1, lin: k1_7.clone() + nt(&cnt_n[12]), theta: 2 }, Step { coef: 1, lin: nt(&k1_7) + cnt_n[12].clone(), theta: 2 }]);
    let fr11 = all(&mut c, "FR.11", &[k1_7.clone(), cnt_n[13].clone()]);
    let fr: W = (0..16).map(|b| match b { 0..=9 => nt(&cnt_n[b + 2]), 10 => fr10.clone(), 11 => fr11.clone(), _ => k(0) }).collect();
    let fe_out = [("FE0".to_string(), fe[..8].to_vec()), ("FE1".to_string(), [fe[8..].to_vec(), vec![drop.clone(), held_n.clone(), wake], kbits(0, 1)].concat())];
    let outs: Vec<(&str, W)> = vec![
        ("C0'", cnt_n[..8].to_vec()),
        ("C1'", cnt_n[8..].to_vec()),
        ("TIMA'", tima_n),
        ("TMA'", tma_n),
        ("TC'", [tac_n, vec![r_n], vec![drop.clone(), stopped_n, blocked_n, tc[7].clone()]].concat()),
        ("IF'", [if_n, kbits(0, 3)].concat()),
        ("IE'", ie_n),
        ("JP'", jp_n),
        ("JB'", jb_n),
        ("SB'", sb_n),
        ("SC'", sc_n),
        ("SR0'", rem_n[..8].to_vec()),
        ("SR1'", [rem_n[8..].to_vec(), kbits(0, 5)].concat()),
        ("K0'", k0_n),
        ("K1'", [vec![k1_0], kbits(0, 6), vec![k1_7.clone()]].concat()),
        ("SV'", sv_n),
        ("DR'", dr_n),
        ("DI'", di_n),
        ("HDMA1'", h1_n),
        ("HDMA2'", [kbits(0, 4), h2_n].concat()),
        ("HDMA3'", [h3_n, vec![hp_n], vec![dn_n], vec![gd_n.clone()]].concat()),
        ("HDMA4'", [vec![he_n], kbits(0, 3), h4_n].concat()),
        ("HDMA5'", h5_n),
        ("LM'", [vec![half_n], sel(&mut c, "LM'", &m_t, &[(vec![lcd_off.clone()], kbits(0, 7)), (vec![on2.clone()], kbits(1, 7))])].concat()),
        ("LY'", sel(&mut c, "LY'", &ly_t, &[(vec![lcd_off.clone()], kbits(0, 8))])),
        ("LYC'", lyc_n),
        ("ST'", [vec![line_n, first_n, coinc_n], en_n, vec![lcd_n]].concat()),
        ("BP'", bp_n),
        ("OP'", op_n),
        ("PS0'", pz_n[..8].to_vec()),
        ("PS1'", pz_n[8..].to_vec()),
        ("FL'", [vec![restart_n, reload_cycle, waking, halt_n], fl75_n, vec![allow_n]].concat()),
        ("UR0'", ur0_n),
        ("UR1'", ur1_n),
        ("UR2'", ur2_n),
    ];
    let clock_next = clock::NEXT.iter().copied().zip(clock.next);
    let outs: Vec<(&str, W)> = outs.into_iter().chain(clock_next).chain([
        ("RV", rv),
        ("H0", h[..8].to_vec()),
        ("H1", [h[8..].to_vec(), kbits(0, 5)].concat()),
        ("FX", vec![start, dact_n, render, vblank, blank, held_n, block, oam_byte]),
        ("DP", dp),
        ("DX", [vec![k(0)], kbits(0, 2), di[3..].to_vec()].concat()),
        ("DM", dm),
        ("DL", last),
        ("CS0", [kbits(0, 4), h2[4..].to_vec()].concat()),
        ("CS1", h1.clone()),
        ("CD0", [kbits(0, 4), h4[4..].to_vec()].concat()),
        ("CD1", [h3[..5].to_vec(), kbits(0b100, 3)].concat()),
        ("SS0", stall[..8].to_vec()),
        ("SS1", stall[8..].to_vec()),
        ("PA", pa),
        ("RF", t1_out),
        ("RS", [rs, kbits(0, 4)].concat()),
        ("RP", kbits(LINE_MCYCLES, 8)),
        ("RM", kbits(MODE3_LM, 8)),
    ])
    .collect();
    let outs: Vec<(String, W)> = outs.into_iter().map(|(n, b)| (n.to_string(), b)).chain(sl.into_iter().enumerate().map(|(j, b)| (format!("SL{j}"), b))).chain(fe_out)
        .chain([
            // Real time after the pass: halves per M-cycle at the speed it leaves.
            ("HR".to_string(), [vec![k1_7.clone(), nt(&k1_7)], kbits(0, 6)].concat()),
            ("DD".to_string(), kbits(OAM_PAGE, 8)),
            // The HDMA block's length, and bit 7: it lands at the end of the GBC CPU
            // step that runs.
            ("CL".to_string(), [kbits(HDMA_BLOCK, 7), vec![held]].concat()),
            ("LK".to_string(), lk.to_vec()),
            ("MX0".to_string(), mx[..8].to_vec()),
            ("MX1".to_string(), [mx[8..].to_vec(), kbits(0, 5)].concat()),
            ("MY0".to_string(), my[..8].to_vec()),
            ("MY1".to_string(), [my[8..].to_vec(), kbits(0, 5)].concat()),
            // The cable: bit 0, this end's transfer ended in the span and sends `SO` to the
            // other; bit 1, the byte the other end clocked across landed.
            ("SX".to_string(), [vec![done.clone(), took], kbits(0, 6)].concat()),
            ("SO".to_string(), sb.clone()),
            // The byte every OAM DMA byte of the pass is when no source drives it, else 0.
            ("DF".to_string(), vec![fill; 8]),
            // A frame of emulated time in halves: 154 lines of 456 dots.
            ("FH0".to_string(), kbits(crate::machine::HALVES_PER_FRAME & 0xFF, 8)),
            ("FH1".to_string(), kbits(crate::machine::HALVES_PER_FRAME >> 8, 8)),
            ("DY".to_string(), dy),
            ("DB".to_string(), db),
            ("DZ".to_string(), [dz.to_vec(), kbits(0, 6)].concat()),
            ("FR0".to_string(), fr[..8].to_vec()),
            ("FR1".to_string(), fr[8..].to_vec()),
            ("RT".to_string(), kbits(LINE_MCYCLES, 8)),
            ("NE".to_string(), (0..8).map(|b| if EV_BLOCK >> b & 1 == 1 { gd_n.clone() } else { k(0) }).collect()),
        ])
        .collect();
    // The request: the pass needs the line's mode-3 length, being past M-cycle 63 of a
    // visible line. It reads no late input.
    let mq = all(&mut c, "MQ", &[vis.clone(), m_ge126]);
    let mut outputs = Vec::new();
    for (name, bits) in [("MQ".to_string(), [vec![mq], kbits(0, 7)].concat())].into_iter().chain(outs) {
        c.output(&name, byte_of(&bits));
        outputs.push(name);
    }
    c.schedule(crate::llama::UNIT_QUANTUM);
    let inputs = STATE.iter().chain(&clock::STATE).chain(&EVENTS).chain(&OWED).chain(&[CABLE, STEP, LENGTH]).map(|s| s.to_string()).collect();
    Built { c, inputs, outputs }
}
