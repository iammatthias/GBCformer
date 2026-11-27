//! The GBC CPU over nibble-valued channels (`docs/byte-channels.md`): host-fed (`build`)
//! and reading its own memory (`build_mem`). The decode is the 0/1 circuit's
//! (`decode::decode`); the datapath, the ALU and the output fields are rebuilt here.
//! Every register byte is also two level-0 nibble wires (0..15). A sum is affine in
//! nibbles once its carries are known, each carry one step on a nibble or byte form;
//! a byte selection is two ramp units per case, `max(0, v + m·(Σ gates − n))`, where
//! the 0/1 circuit spends eight. Bits stay where the GBC CPU works on bits or a memory key
//! needs them: the logic ops, the right shifts, BIT, the flags, the opcode and CB
//! decode, every translated address and the old bytes of a write's line.

use super::alu::rot;
use super::datapath::{add16_split, add_hl};
use super::decode::{self, cycle_table, len, xyz, Decoded, ILLEGAL};
use super::fields::{same_line, Pair};
use super::line::{self, Val};
use super::translate::{self, Region};
use super::{Cb, Mode, G, REQUEST, RESULT, RESULT_MEM, W};
use crate::circuit::{bits, cat, one, value, zero, Circuit, Lin, Nib, OutKind, Query, Step, LINE, LINE_BITS};
use crate::memory::layout::{Access, Reg};
use std::collections::BTreeMap;

/// A byte as its nibbles, low first.
type N2 = [Nib; 2];
/// A 16-bit word as its nibbles, low first.
type N4 = [Nib; 4];

fn n2(b: &[Lin]) -> N2 {
    [Nib::of_bits(&b[0..4]), Nib::of_bits(&b[4..8])]
}
fn n4(b: &[Lin]) -> N4 {
    std::array::from_fn(|i| Nib::of_bits(&b[4 * i..4 * i + 4]))
}
fn k2(v: u8) -> N2 {
    [Nib::konst((v & 15) as i64), Nib::konst((v >> 4) as i64)]
}
fn k4(v: u16) -> N4 {
    std::array::from_fn(|i| Nib::konst((v >> (4 * i) & 15) as i64))
}
fn word(lo: &N2, hi: &N2) -> N4 {
    [lo[0].clone(), lo[1].clone(), hi[0].clone(), hi[1].clone()]
}
fn low(x: &N4) -> N2 {
    [x[0].clone(), x[1].clone()]
}
fn high(x: &N4) -> N2 {
    [x[2].clone(), x[3].clone()]
}
/// The integer a nibble array holds, Σ 16^i·n_i.
fn val(n: &[Nib]) -> Lin {
    n.iter().rev().fold(Lin::konst(0), |s, x| s * 16 + x.lin.clone())
}
fn nib(lin: Lin) -> Nib {
    Nib { lin, hi: 15 }
}

/// A selection among nibble arrays: one `value_mux` per nibble.
fn vmux<const N: usize>(g: &mut G, name: &str, cases: &[(Vec<Lin>, [Nib; N])]) -> [Nib; N] {
    std::array::from_fn(|i| {
        let cs: Vec<(Vec<Lin>, Nib)> = cases.iter().map(|(gs, v)| (gs.clone(), v[i].clone())).collect();
        g.c.value_mux(&format!("{name}.{i}"), &cs)
    })
}

/// `⌊form / base⌋`: one 0/1 node per multiple of `base` the form's range crosses.
fn floor_div(g: &mut G, name: &str, form: &Lin, base: i64) -> Lin {
    let (lo, hi) = g.c.bounds(form);
    let (a, b) = (lo.div_euclid(base), hi.div_euclid(base));
    (a + 1..=b).fold(Lin::konst(a), |s, j| s + g.c.ge(&format!("{name}>={}", j * base), form.clone(), j * base))
}

/// The nibbles of `Σ 16^i·p[i]` mod 16^N, for nibble forms `p` whose true sum is not
/// negative, with the carries out of nibbles 0, 1 (and 2, 3): each byte's carries by
/// lookahead, the high byte's reading the low byte's.
fn add<const N: usize>(g: &mut G, name: &str, p: [Lin; N]) -> ([Nib; N], Vec<Lin>) {
    let mut n = Vec::new();
    let mut carries = Vec::new();
    let mut into = Lin::konst(0);
    for b in 0..N / 2 {
        let (p0, p1) = (p[2 * b].clone() + into.clone(), p[2 * b + 1].clone());
        let k1 = floor_div(g, &format!("{name}.c{}", 8 * b + 4), &p0, 16);
        let k2 = floor_div(g, &format!("{name}.c{}", 8 * b + 8), &(p1.clone() * 16 + p0.clone()), 256);
        n.push(nib(p0 - k1.clone() * 16));
        n.push(nib(p1 + k1.clone() - k2.clone() * 16));
        carries.push(k1);
        carries.push(k2.clone());
        into = k2;
    }
    (n.try_into().unwrap(), carries)
}

/// `[byte form mod 256 == 0]` for a form in [0, 511].
fn zero8(g: &mut G, name: &str, s: Lin) -> Lin {
    let (_, hi) = g.c.bounds(&s);
    let steps = (0..=hi / 256).flat_map(|j| [Step { coef: 1, lin: s.clone(), theta: 256 * j }, Step { coef: -1, lin: s.clone(), theta: 256 * j + 1 }]).collect();
    g.c.node(name, steps)
}

/// The circuit the host feeds opcode and data bytes.
pub fn build() -> Circuit {
    build_with(Mode::Host)
}

/// The circuit that reads opcode and data bytes from the model's memory.
pub fn build_mem() -> Circuit {
    build_with(Mode::Memory)
}

fn build_with(mode: Mode) -> Circuit {
    let (mut g, d) = decode::decode(mode);
    declare_groups(&mut g, &d);
    let regs_n: Vec<N2> = ["B", "C", "D", "E", "H", "L"].iter().map(|n| g.c.input_nibbles(n)).collect();
    let av = g.c.input_nibbles("A");
    let sp = word(&g.c.input_nibbles("SPL"), &g.c.input_nibbles("SPH"));
    let pc = word(&g.c.input_nibbles("PCL"), &g.c.input_nibbles("PCH"));
    let (d0v, d1v, imm8v, imm16v) = match mode {
        Mode::Host => (g.c.input_nibbles("D0"), g.c.input_nibbles("D1"), n2(&d.imm8), n4(&d.imm16)),
        Mode::Memory => (n2(&d.d0), n2(&d.d1), n2(&d.imm8), n4(&d.imm16)),
    };
    // The immediates isolated to the few opcodes that read them as an address or an
    // offset. They change on most steps, and without the isolation each change would
    // reach every case and carry that reads them.
    let (rare_ops, ff_ops) = (g.op(0xEA) + g.op(0x08) + g.op(0xFA), g.op(0xE0) + g.op(0xF0));
    // (Memory mode isolates these sources as bits, for the translations.)
    let (rare16, ff8) = match mode {
        Mode::Host => (vmux(&mut g, "imm16.rare", &[(vec![rare_ops], imm16v.clone())]), vmux(&mut g, "imm8.ff", &[(vec![ff_ops], imm8v.clone())])),
        Mode::Memory => (imm16v.clone(), imm8v.clone()),
    };
    let sp_rel_ops = g.op(0xE8) + g.op(0xF8);
    // In memory mode the immediates of three rare opcodes, LD (a16),SP (a16 + 1 and the
    // same-line test) and ADD SP,e8 / LD HL,SP+e8 (the offset), come from one isolation:
    // each consumer is read only under its own opcode.
    let rare_imm = (mode == Mode::Memory).then(|| g.isolate("imm.rare", &d.imm16, &(g.op(0x08) + sp_rel_ops.clone())));
    let (e8s, e8s_sign) = match &rare_imm {
        Some(iso) => (n2(&iso[..8]), iso[7].clone()),
        None => (vmux(&mut g, "imm8.sp", &[(vec![sp_rel_ops.clone()], imm8v.clone())]), g.c.and("imm8.sp.7", &[d.imm8[7].clone(), sp_rel_ops])),
    };
    // Registers by their 3-bit index, (HL) as D0.
    let rv: Vec<N2> = (0..8).map(|r| match r { 6 => d0v.clone(), 7 => av.clone(), r => regs_n[r].clone() }).collect();
    let pair = |hi: usize, lo: usize| word(&rv[lo], &rv[hi]);
    let (bc, de, hl) = (pair(0, 1), pair(2, 3), pair(4, 5));
    let (a, f, ctl) = (d.a.clone(), d.f.clone(), d.ctl.clone());
    let (ime, hb) = (ctl[0].clone(), ctl[3].clone());
    let (exec, dispatch, wake) = (d.exec.clone(), d.dispatch.clone(), d.wake.clone());
    let (fz, fc, fh, fnn) = (f[7].clone(), f[4].clone(), f[5].clone(), f[6].clone());

    // Stack arithmetic. The writes' addresses SP−1 and SP−2 are translated in memory
    // mode, so they stay bits there; their values are the bits' nibbles.
    let (sp_m1, sp_m1_bits) = match mode {
        Mode::Memory => {
            let b = add16_split(&mut g, "sp-1", &d.sp, 0xFFFF);
            (n4(&b), Some(b))
        }
        Mode::Host => (add(&mut g, "sp-1", [sp[0].lin.clone() + 15, sp[1].lin.clone() + 15, sp[2].lin.clone() + 15, sp[3].lin.clone() + 15]).0, None),
    };
    let (sp_m2, sp_m2_bits) = match mode {
        Mode::Memory => {
            let b = add16_split(&mut g, "sp-2", &d.sp, 0xFFFE);
            (n4(&b), Some(b))
        }
        Mode::Host => (add(&mut g, "sp-2", [sp[0].lin.clone() + 14, sp[1].lin.clone() + 15, sp[2].lin.clone() + 15, sp[3].lin.clone() + 15]).0, None),
    };
    let sp_p1 = match &d.memory {
        Some(m) => n4(&m.sp_p1),
        None => add(&mut g, "sp+1", [sp[0].lin.clone() + 1, sp[1].lin.clone(), sp[2].lin.clone(), sp[3].lin.clone()]).0,
    };
    let sp_p2 = add(&mut g, "sp+2", [sp[0].lin.clone() + 2, sp[1].lin.clone(), sp[2].lin.clone(), sp[3].lin.clone()]).0;
    let rp = [bc.clone(), de.clone(), hl.clone(), sp.clone()];

    // SP + e8: H and C out of the low nibble and byte, the sign into the high byte.
    let (sp_e8, spe_c) = add(&mut g, "sp+e8", [sp[0].lin.clone() + e8s[0].lin.clone(), sp[1].lin.clone() + e8s[1].lin.clone(), sp[2].lin.clone() + e8s_sign.clone() * 15, sp[3].lin.clone() + e8s_sign * 15]);
    let (spe_h, spe_c) = (spe_c[0].clone(), spe_c[1].clone());
    // a16 + 1 for LD (a16),SP's second write: bits in memory mode, which translates it.
    let (a16_p1_bits, a16_ld) = match mode {
        Mode::Memory => {
            let a16 = rare_imm.clone().expect("memory mode isolates the rare immediates");
            (Some(g.c.sum_bits("a16+1", 16, |n| value(&a16, n) + 1)), Some(a16))
        }
        Mode::Host => (None, None),
    };
    // The immediate word as the loads, jumps and calls that take it as a value see it:
    // in memory mode isolated to those opcodes, so the operand bytes, which change on
    // most steps, reach the B..L, SP and PC fields' cases only when one of them runs.
    let imm16f = match mode {
        Mode::Memory => {
            let takes = g.ops(|op| matches!(op, 0x01 | 0x11 | 0x21 | 0x31 | 0xC3 | 0xCD) || matches!(xyz(op), (3, 0..=3, 2 | 4)));
            n4(&g.isolate("imm16.v", &d.imm16, &takes))
        }
        Mode::Host => imm16v.clone(),
    };
    let a16_p1 = match &a16_p1_bits {
        Some(b) => n4(b),
        None => add(&mut g, "a16+1", [rare16[0].lin.clone() + 1, rare16[1].lin.clone(), rare16[2].lin.clone(), rare16[3].lin.clone()]).0,
    };

    let cc = [one() - fz.clone(), fz.clone(), one() - fc.clone(), fc.clone()];
    let taken_family = |g: &mut G, name: &str, base: u8| {
        let steps = (0..4u8).map(|k| Step { coef: 1, lin: g.op(base + 8 * k) + cc[k as usize].clone(), theta: 2 }).collect();
        g.c.node(name, steps)
    };
    let t_jr = taken_family(&mut g, "taken.jr", 0x20);
    let t_ret = taken_family(&mut g, "taken.ret", 0xC0);
    let t_jp = taken_family(&mut g, "taken.jp", 0xC2);
    let t_call = taken_family(&mut g, "taken.call", 0xC4);
    let fam = |g: &G, base: u8| (0..4u8).fold(zero(), |s, k| s + g.op(base + 8 * k));
    let not_taken = fam(&g, 0x20) + fam(&g, 0xC0) + fam(&g, 0xC2) + fam(&g, 0xC4) - t_jr.clone() - t_ret.clone() - t_jp.clone() - t_call.clone();

    // The next pc: pc + (len − hb) + e8 for a JR that jumps. len − hb is two bits from
    // the one-hots and hb, never negative; the offset is the immediate under the jump's
    // gates, its sign one step on its high nibble.
    let by_len = |g: &G, ls: &[u64]| g.ops(|op| ls.contains(&len(op))) + if ls.contains(&2) { g.op(0xCB) } else { zero() };
    let odd = by_len(&g, &[1, 3]);
    let o0 = g.c.eq("len-hb.0", odd + hb.clone(), 1);
    let (three, two) = (by_len(&g, &[3]), by_len(&g, &[2]));
    let o1 = g.c.node("len-hb.1", vec![Step { coef: 1, lin: three, theta: 1 }, Step { coef: 1, lin: two - hb.clone(), theta: 1 }]);
    let len_hb = o0 + o1 * 2;
    let jumps_rel = g.op(0x18) + t_jr.clone();
    let e8 = match &d.memory {
        Some(m) => vmux(&mut g, "jr.e8", &[(vec![m.jr_taken.clone()], n2(&d.op1))]),
        None => {
            let op1 = g.c.input_nibbles("OP1");
            vmux(&mut g, "jr.e8", &[(vec![jumps_rel.clone()], op1)])
        }
    };
    let e8_neg = g.c.ge("jr.e8.7", e8[1].lin.clone(), 8);
    let (pc_seq, pc_carries) = add(&mut g, "pc+", [pc[0].lin.clone() + len_hb + e8[0].lin.clone(), pc[1].lin.clone() + e8[1].lin.clone(), pc[2].lin.clone() + e8_neg.clone() * 15, pc[3].lin.clone() + e8_neg.clone() * 15]);

    // Operands, as bits (the logic ops, the shifts, the flags); their nibbles are the
    // bits' sums. In memory mode the (HL) operand is read 0's byte, whose bus and late
    // byte the selection reads directly (`line::mux`).
    let src_sel = |g: &G, r: u8| g.ops(|op| { let (x, _, z) = xyz(op); (x == 1 && op != 0x76 || x == 2) && z == r });
    let mut src_cases: Vec<(Lin, W)> = (0..8).map(|r| (src_sel(&g, r), d.regs[r as usize].clone())).collect();
    // In memory mode the immediate isolated to its opcodes (LD r,d8, the ALU on d8): the
    // selection sits above read 0's bus, and the operand byte, which changes on most
    // steps, would otherwise be read in that layer too.
    let imm_ops = g.ops(|op| matches!(xyz(op), (0 | 3, _, 6)));
    let imm8_src = match mode {
        Mode::Memory => g.isolate("imm8.src", &d.imm8, &imm_ops),
        Mode::Host => d.imm8.clone(),
    };
    src_cases.push((imm_ops, imm8_src));
    let x_cases: Vec<(Lin, W)> = (0..8u8)
        .map(|r| {
            let sel = g.ops(|op| { let (x, y, z) = xyz(op); x == 0 && (z == 4 || z == 5) && y == r }) + g.cb_z(r);
            (sel, d.regs[r as usize].clone())
        })
        .collect();
    let operand = |g: &mut G, name: &str, cases: Vec<(Lin, W)>| match &d.memory {
        None => g.c.mux(name, 8, &cases.into_iter().map(|(s, v)| (vec![s], v)).collect::<Vec<_>>()),
        Some(m) => {
            let cases: Vec<(Lin, Val)> = cases.into_iter().enumerate().map(|(r, (s, v))| (s, if r == 6 { Val::Read(&m.d0) } else { Val::Wires(v) })).collect();
            line::mux(&mut g.c, name, &cases)
        }
    };
    let src = operand(&mut g, "src", src_cases);
    let xop = operand(&mut g, "x", x_cases);
    let (srcv, xv) = (n2(&src), n2(&xop));

    // INC rr, DEC rr, ADD HL,rr and (HL±): one pair selection.
    let pair_ops = |g: &G, p: u8| g.op(0x03 + 16 * p) + g.op(0x0B + 16 * p) + g.op(0x09 + 16 * p);
    let hl_step = g.op(0x22) + g.op(0x2A) + g.op(0x32) + g.op(0x3A);
    let pair_cases: Vec<(Vec<Lin>, N4)> = (0..4u8).map(|p| (vec![pair_ops(&g, p) + if p == 2 { hl_step.clone() } else { zero() }], rp[p as usize].clone())).collect();
    let pr = vmux(&mut g, "pair", &pair_cases);
    let pl = |x: &N4, i: usize| x[i].lin.clone();
    let pair_inc = add(&mut g, "pair+1", [pl(&pr, 0) + 1, pl(&pr, 1), pl(&pr, 2), pl(&pr, 3)]).0;
    let pair_dec = add(&mut g, "pair-1", [pl(&pr, 0) + 15, pl(&pr, 1) + 15, pl(&pr, 2) + 15, pl(&pr, 3) + 15]).0;
    let (hl_add, hla) = add(&mut g, "hl+pair", std::array::from_fn(|i| pl(&hl, i) + pl(&pr, i)));
    let (hla_h, hla_c) = (hla[2].clone(), hla[3].clone());

    // The ALU.
    let cin_a = g.c.and("cin.adc", &[g.alu(1), fc.clone()]);
    let cin_s = g.c.and("cin.sbc", &[g.alu(3), fc.clone()]);
    let al = |i: usize| av[i].lin.clone();
    let sl = |i: usize| srcv[i].lin.clone();
    let (add_v, addc) = add(&mut g, "add", [al(0) + sl(0) + cin_a.clone(), al(1) + sl(1)]);
    let add_z = zero8(&mut g, "add.z", val(&av) + val(&srcv) + cin_a.clone());
    let (sub_v, subc) = add(&mut g, "sub", [al(0) - sl(0) - cin_s.clone() + 16, al(1) - sl(1) + 15]);
    let sub_z = zero8(&mut g, "sub.z", val(&av) - val(&srcv) - cin_s.clone() + 256);
    let (add_h, add_c, sub_h, sub_c) = (addc[0].clone(), addc[1].clone(), one() - subc[0].clone(), one() - subc[1].clone());
    let and: W = (0..8).map(|k| g.c.and(&format!("and.{k}"), &[a[k].clone(), src[k].clone()])).collect();
    let and_z = g.c.ge("and.z", -(and.iter().cloned().fold(zero(), |s, b| s + b)), 0);
    let or: W = (0..8).map(|k| g.c.or(&format!("or.{k}"), &[a[k].clone(), src[k].clone()])).collect();
    let or_z = g.c.ge("or.z", -(a.iter().chain(&src).cloned().fold(zero(), |s, b| s + b)), 0);
    let xor: W = (0..8).map(|k| g.c.eq(&format!("xor.{k}"), a[k].clone() + src[k].clone(), 1)).collect();
    let xor_z = g.c.eq("xor.z", val(&av) - val(&srcv), 0);

    let xl = |i: usize| xv[i].lin.clone();
    let (inc_v, incc) = add(&mut g, "inc", [xl(0) + 1, xl(1)]);
    let inc_z = zero8(&mut g, "inc.z", val(&xv) + 1);
    let (dec_v, decc) = add(&mut g, "dec", [xl(0) + 15, xl(1) + 15]);
    let dec_z = zero8(&mut g, "dec.z", val(&xv) + 255);
    let (inc_h, dec_h) = (incc[0].clone(), one() - decc[0].clone());

    // CB rotates, shifts, BIT, RES and SET of the operand.
    let cb_rot: Vec<(W, Lin, Lin)> = (0..8u8)
        .map(|y| {
            let (b, cout) = rot(&xop, y, &fc);
            let z = g.c.ge(&format!("rot{y}.z"), -(b.iter().cloned().fold(zero(), |s, x| s + x)), 0);
            (b, cout, z)
        })
        .collect();
    let bit_z = {
        let steps = (0..8u8).map(|y| Step { coef: 1, lin: g.cb_xy(1, y) + one() - xop[y as usize].clone(), theta: 2 }).collect();
        g.c.node("bit.z", steps)
    };
    let cb_cases: Vec<(Vec<Lin>, N2)> = (0..8u8)
        .flat_map(|y| {
            let (h, w) = (y as usize / 4, 1i64 << (y % 4));
            let mut res = xv.clone();
            res[h] = nib(res[h].lin.clone() - xop[y as usize].clone() * w);
            let mut set = xv.clone();
            set[h] = nib(set[h].lin.clone() + (one() - xop[y as usize].clone()) * w);
            [(vec![g.cb_xy(0, y)], n2(&cb_rot[y as usize].0)), (vec![g.cb_xy(2, y)], res), (vec![g.cb_xy(3, y)], set)]
        })
        .collect();
    let cbres = vmux(&mut g, "cbres", &cb_cases);

    // DAA: the corrections as the 0/1 circuit has them, the result from A's nibbles.
    let big = 1024;
    let is_daa = g.op(0x27);
    let c6 = g.c.node("daa.c6", vec![
        Step { coef: 1, lin: one() - fnn.clone() + fc.clone() + is_daa.clone(), theta: 3 },
        Step { coef: 1, lin: (one() - fnn.clone() + is_daa.clone()) * big + (one() - fc.clone()) * big + value(&a, 8), theta: 3 * big + 0x9A },
        Step { coef: 1, lin: fnn.clone() + fc.clone() + is_daa.clone(), theta: 3 },
    ]);
    let c0 = g.c.node("daa.c0", vec![
        Step { coef: 1, lin: one() - fnn.clone() + fh.clone() + is_daa.clone(), theta: 3 },
        Step { coef: 1, lin: (one() - fnn.clone() + is_daa.clone()) * big + (one() - fh.clone()) * big + value(&a, 4), theta: 3 * big + 0x0A },
        Step { coef: 1, lin: fnn.clone() + fh.clone() + is_daa.clone(), theta: 3 },
    ]);
    let a_daa = vmux(&mut g, "daa.a", &[(vec![is_daa.clone()], av.clone())]);
    let dl = |i: usize| a_daa[i].lin.clone();
    let daa_add = add(&mut g, "daa+", [dl(0) + c0.clone() * 6, dl(1) + c6.clone() * 6]).0;
    let daa_add_z = zero8(&mut g, "daa+.z", val(&a_daa) + c0.clone() * 6 + c6.clone() * 0x60);
    let daa_sub = add(&mut g, "daa-", [dl(0) - c0.clone() * 6 + 16, dl(1) - c6.clone() * 6 + 15]).0;
    let daa_sub_z = zero8(&mut g, "daa-.z", val(&a_daa) - c0.clone() * 6 - c6.clone() * 0x60 + 256);
    let daa_n = g.c.and("daa.n", &[is_daa.clone(), fnn.clone()]);
    let daa_p = is_daa.clone() - daa_n.clone();

    // Output fields: exclusive EXEC cases, else keep; DISP and WAKE values or keep.
    type Cases = (Vec<(Vec<Lin>, N2)>, Lin);
    let single = |cases: Vec<(Lin, N2)>| -> Cases {
        let cover = cases.iter().fold(zero(), |s, (sel, _)| s + sel.clone());
        (cases.into_iter().map(|(sel, v)| (vec![sel], v)).collect(), cover)
    };
    let field = |g: &mut G, name: &str, old: &N2, (cases, cover): Cases, on_disp: Option<N2>| -> N2 {
        let keep = one() - cover;
        let mut all: Vec<(Vec<Lin>, N2)> = cases.into_iter().map(|(mut gates, v)| { gates.push(exec.clone()); (gates, v) }).collect();
        all.push((vec![keep, exec.clone()], old.clone()));
        all.push((vec![dispatch.clone()], on_disp.unwrap_or_else(|| old.clone())));
        all.push((vec![wake.clone()], old.clone()));
        vmux(g, name, &all)
    };
    let field16 = |g: &mut G, name: &str, old: &N4, cases: Vec<(Lin, N4)>, on_disp: Option<N4>| -> N4 {
        let lo = single(cases.iter().map(|(s, v)| (s.clone(), low(v))).collect());
        let hi = single(cases.iter().map(|(s, v)| (s.clone(), high(v))).collect());
        word(&field(g, &format!("{name}L'"), &low(old), lo, on_disp.as_ref().map(low)), &field(g, &format!("{name}H'"), &high(old), hi, on_disp.as_ref().map(high)))
    };
    let bit_field = |g: &mut G, name: &str, old: Lin, cases: Vec<(Lin, Lin)>, on_disp: Option<Lin>, on_wake: Option<Lin>| -> Lin {
        let cover = cases.iter().fold(zero(), |s, (sel, _)| s + sel.clone());
        let keep = one() - cover;
        let mut all: Vec<(Vec<Lin>, Vec<Lin>)> = cases.into_iter().map(|(sel, v)| (vec![sel, exec.clone()], vec![v])).collect();
        all.push((vec![keep, exec.clone()], vec![old.clone()]));
        all.push((vec![dispatch.clone()], vec![on_disp.unwrap_or_else(|| old.clone())]));
        all.push((vec![wake.clone()], vec![on_wake.unwrap_or(old)]));
        g.c.mux(name, 1, &all)[0].clone()
    };

    let inc_r = |g: &G, r: u8| g.ops(|op| xyz(op) == (0, r, 4));
    let dec_r = |g: &G, r: u8| g.ops(|op| xyz(op) == (0, r, 5));
    let ld_r = |g: &G, r: u8| g.ops(|op| { let (x, y, z) = xyz(op); (x == 1 && op != 0x76 || x == 0 && z == 6) && y == r });
    let results8 = |g: &G, r: u8| -> Cases {
        let (mut cases, cover) = single(vec![(ld_r(g, r), srcv.clone()), (inc_r(g, r), inc_v.clone()), (dec_r(g, r), dec_v.clone())]);
        cases.push((vec![g.cb_writes(r)], cbres.clone()));
        (cases, cover + g.cb_writes(r))
    };
    let with = |(mut cases, cover): Cases, extra: Vec<(Lin, N2)>| -> Cases {
        let (e, ec) = single(extra);
        cases.extend(e);
        (cases, cover + ec)
    };
    let mut values: BTreeMap<&str, N2> = BTreeMap::new();
    let mut bitsout: BTreeMap<&str, W> = BTreeMap::new();
    let mut kinds: BTreeMap<&str, OutKind> = BTreeMap::new();

    let rotv = |y: u8| n2(&rot(&a, y, &fc).0);
    let a_cases = vec![
        (g.ops(|op| matches!(op, 0x0A | 0x1A | 0x2A | 0x3A | 0xF0 | 0xF2 | 0xFA)), d0v.clone()),
        (g.op(0xF1), d1v.clone()),
        (g.alu(0) + g.alu(1), add_v.clone()),
        (g.alu(2) + g.alu(3), sub_v.clone()),
        (g.alu(4), n2(&and)),
        (g.alu(5), n2(&xor)),
        (g.alu(6), n2(&or)),
        (g.op(0x07), rotv(0)),
        (g.op(0x0F), rotv(1)),
        (g.op(0x17), rotv(2)),
        (g.op(0x1F), rotv(3)),
        (daa_p.clone(), daa_add),
        (daa_n.clone(), daa_sub),
        (g.op(0x2F), [nib(Lin::konst(15) - al(0)), nib(Lin::konst(15) - al(1))]),
    ];
    let a_cases = with(results8(&g, 7), a_cases);
    values.insert("A'", field(&mut g, "A'", &av, a_cases, None));

    for (hi, lo, p) in [(0usize, 1usize, 0u8), (2, 3, 1), (4, 5, 2)] {
        let (pop, ld16, incp, decp) = (g.op(0xC1 + 16 * p), g.op(0x01 + 16 * p), g.op(0x03 + 16 * p), g.op(0x0B + 16 * p));
        for (r, half) in [(hi, 1), (lo, 0)] {
            let pick = |w: &N4| if half == 1 { high(w) } else { low(w) };
            let mut cs = vec![(pop.clone(), if half == 1 { d1v.clone() } else { d0v.clone() })];
            cs.push((ld16.clone(), pick(&imm16f)));
            cs.push((incp.clone(), pick(&pair_inc)));
            cs.push((decp.clone(), pick(&pair_dec)));
            if p == 2 {
                cs.push((add_hl(&g), pick(&hl_add)));
                cs.push((g.op(0xF8), pick(&sp_e8)));
                cs.push((g.op(0x22) + g.op(0x2A), pick(&pair_inc)));
                cs.push((g.op(0x32) + g.op(0x3A), pick(&pair_dec)));
            }
            let name = ["B'", "C'", "D'", "E'", "H'", "L'"][r];
            let cs = with(results8(&g, r as u8), cs);
            values.insert(name, field(&mut g, name, &rv[r], cs, None));
        }
    }

    // Flags, one bit at a time.
    let cbx = |x: u8| (0..8u8).fold(zero(), |s, y| s + g.cb_xy(x, y));
    let inc_any = g.ops(|op| matches!(xyz(op), (0, _, 4)));
    let dec_any = g.ops(|op| matches!(xyz(op), (0, _, 5)));
    let acc_rot = g.op(0x07) + g.op(0x0F) + g.op(0x17) + g.op(0x1F);
    let add_hl_any = add_hl(&g);
    let sp_rel = g.op(0xE8) + g.op(0xF8);
    let sub_any = g.alu(2) + g.alu(3) + g.alu(7);
    let daa = g.op(0x27);
    let d0 = d.d0.clone();
    let mut fz_c = vec![
        (g.alu(0) + g.alu(1), add_z),
        (sub_any.clone(), sub_z),
        (g.alu(4), and_z),
        (g.alu(5), xor_z),
        (g.alu(6), or_z),
        (inc_any.clone(), inc_z),
        (dec_any.clone(), dec_z),
        (acc_rot.clone(), zero()),
        (cbx(1), bit_z),
        (daa_p.clone(), daa_add_z),
        (daa_n.clone(), daa_sub_z),
        (sp_rel.clone(), zero()),
        (g.op(0xF1), d0[7].clone()),
    ];
    for y in 0..8u8 {
        fz_c.push((g.cb_xy(0, y), cb_rot[y as usize].2.clone()));
    }
    let fn_c = vec![
        (g.alu(0) + g.alu(1) + g.alu(4) + g.alu(5) + g.alu(6), zero()),
        (sub_any.clone(), one()),
        (inc_any.clone(), zero()),
        (dec_any.clone(), one()),
        (acc_rot.clone() + cbx(0) + cbx(1), zero()),
        (g.op(0x2F), one()),
        (g.op(0x37) + g.op(0x3F), zero()),
        (add_hl_any.clone() + sp_rel.clone(), zero()),
        (g.op(0xF1), d0[6].clone()),
    ];
    let fh_c = vec![
        (g.alu(0) + g.alu(1), add_h),
        (sub_any.clone(), sub_h),
        (g.alu(4), one()),
        (g.alu(5) + g.alu(6), zero()),
        (inc_any.clone(), inc_h),
        (dec_any.clone(), dec_h),
        (acc_rot.clone() + cbx(0), zero()),
        (cbx(1), one()),
        (daa.clone(), zero()),
        (g.op(0x2F), one()),
        (g.op(0x37) + g.op(0x3F), zero()),
        (sp_rel.clone(), spe_h),
        (g.op(0xF1), d0[5].clone()),
        (add_hl_any.clone(), hla_h),
    ];
    let mut fc_c = vec![
        (g.alu(0) + g.alu(1), add_c),
        (sub_any.clone(), sub_c),
        (g.alu(4) + g.alu(5) + g.alu(6), zero()),
        (daa.clone(), c6),
        (g.op(0x37), one()),
        (g.op(0x3F), one() - fc.clone()),
        (sp_rel.clone(), spe_c),
        (g.op(0xF1), d0[4].clone()),
        (add_hl_any.clone(), hla_c),
    ];
    for (op, y) in [(0x07u8, 0u8), (0x0F, 1), (0x17, 2), (0x1F, 3)] {
        fc_c.push((g.op(op), rot(&a, y, &fc).1));
    }
    for y in 0..8u8 {
        fc_c.push((g.cb_xy(0, y), cb_rot[y as usize].1.clone()));
    }
    let f_out: W = (0..8)
        .map(|k| {
            let cases = match k {
                7 => fz_c.clone(),
                6 => fn_c.clone(),
                5 => fh_c.clone(),
                4 => fc_c.clone(),
                _ => vec![],
            };
            let old = if k >= 4 { f[k].clone() } else { zero() };
            bit_field(&mut g, &format!("F'.{k}"), old, cases, None, None)
        })
        .collect();
    bitsout.insert("F'", f_out);

    // Stack pointer.
    let pops = g.ops(|op| matches!(op, 0xC1 | 0xD1 | 0xE1 | 0xF1 | 0xC9 | 0xD9)) + t_ret.clone();
    let pushes = g.ops(|op| matches!(op, 0xC5 | 0xD5 | 0xE5 | 0xF5 | 0xCD) || xyz(op).0 == 3 && xyz(op).2 == 7) + t_call.clone();
    let sp_cases = vec![
        (g.op(0x31), imm16f.clone()),
        (g.op(0x33), pair_inc.clone()),
        (g.op(0x3B), pair_dec.clone()),
        (g.op(0xE8), sp_e8.clone()),
        (g.op(0xF9), hl.clone()),
        (pops.clone(), sp_p2.clone()),
        (pushes.clone(), sp_m2.clone()),
    ];
    let sp_out = field16(&mut g, "SP", &sp, sp_cases, Some(sp_m2.clone()));
    values.insert("SPL'", low(&sp_out));
    values.insert("SPH'", high(&sp_out));

    // Program counter.
    let jumps = [0x18u8, 0xC3, 0xE9, 0xCD, 0xC9, 0xD9];
    let branchy = |op: u8| jumps.contains(&op) || matches!(xyz(op), (0, 4..=7, 0) | (3, 0..=3, 0 | 2 | 4) | (3, _, 7)) || ILLEGAL.contains(&op);
    // HALT with IME on (EI's delay included) and an interrupt pending stays on the HALT.
    let halt_ime = g.c.or("halt.ime", &[ctl[0].clone(), ctl[1].clone()]);
    let halt_stay = g.c.and("halt.stay", &[g.op(0x76), halt_ime, d.any.clone()]);
    let sequential = g.ops(|op| !branchy(op)) + g.cb_any() + not_taken.clone() - halt_stay.clone();
    let mut pc_cases: Vec<(Lin, N4)> = vec![(sequential + jumps_rel, pc_seq.clone())];
    pc_cases.push((halt_stay.clone(), pc.clone()));
    pc_cases.push((g.op(0xC3) + t_jp.clone() + g.op(0xCD) + t_call.clone(), imm16f.clone()));
    pc_cases.push((g.op(0xE9), hl.clone()));
    pc_cases.push((g.op(0xC9) + g.op(0xD9) + t_ret.clone(), word(&d0v, &d1v)));
    for y in 0..8u8 {
        pc_cases.push((g.op(0xC7 + 8 * y), k4(8 * y as u16)));
    }
    // Dispatch to 0x40 + 8n: the low byte's nibbles from the one-hot of n, 0000 with none.
    let disp_lo = [nib((0..5).fold(Lin::konst(0), |s, n| s + d.vector[n].clone() * (8 * n as i64 % 16))), nib((0..5).fold(Lin::konst(0), |s, n| s + d.vector[n].clone() * (4 + 8 * n as i64 / 16)))];
    let pc_out = field16(&mut g, "PC", &pc, pc_cases, Some(word(&disp_lo, &k2(0))));
    values.insert("PCL'", low(&pc_out));
    values.insert("PCH'", high(&pc_out));

    // Control bits.
    let illegal = ILLEGAL.iter().fold(zero(), |s, &op| s + g.op(op));
    let legal = one() - illegal;
    let (eip, di, ei, reti, halt) = (ctl[1].clone(), g.op(0xF3), g.op(0xFB), g.op(0xD9), g.op(0x76));
    let ime_exec = g.c.ge("ime.exec", ime.clone() + eip.clone() + reti - di * 3, 1);
    let imee = g.c.or("ime.eff", &[ime, eip]);
    let hbug = g.c.and("halt.bug", &[halt.clone(), one() - imee, d.any.clone()]);
    let halts = halt - hbug.clone() - halt_stay;
    let ctl_out: W = (0..8)
        .map(|k| {
            let old = if k < 4 { ctl[k].clone() } else { zero() };
            let (exec_v, disp_v, wake_v) = match k {
                0 => (Some(ime_exec.clone()), Some(zero()), None),
                1 => (Some(ei.clone()), Some(zero()), None),
                2 => (Some(halts.clone()), None, Some(zero())),
                3 => (Some(hbug.clone()), None, None),
                4 => (Some(g.op(0x10)), None, None),
                _ => (None, None, None),
            };
            bit_field(&mut g, &format!("CTL'.{k}"), old, exec_v.map(|v| vec![(legal.clone(), v)]).unwrap_or_default(), disp_v, wake_v)
        })
        .collect();
    bitsout.insert("CTL'", ctl_out);

    // Memory reads (request pass).
    let ff = |lo: N2| word(&lo, &k2(0xFF));
    match &d.memory {
        None => {
            let hl_read = g.ops(|op| {
                let (x, _, z) = xyz(op);
                matches!(op, 0x2A | 0x3A | 0x34 | 0x35) || (x == 1 && op != 0x76 || x == 2) && z == 6
            }) + g.cb_z(6);
            let ra0_cases = vec![
                (g.op(0x0A), bc.clone()),
                (g.op(0x1A), de.clone()),
                (hl_read, hl.clone()),
                (pops.clone(), sp.clone()),
                (g.op(0xFA), rare16.clone()),
                (g.op(0xF0), ff(ff8.clone())),
                (g.op(0xF2), ff(rv[1].clone())),
            ];
            let rq0 = ra0_cases.iter().fold(zero(), |s, (sel, _)| s + sel.clone());
            let ra0 = vmux(&mut g, "RA0", &ra0_cases.into_iter().map(|(s, v)| (vec![s], v)).collect::<Vec<_>>());
            let rq: W = vec![g.c.and("RQ.0", &[rq0, exec.clone()]), g.c.and("RQ.1", &[pops.clone(), exec.clone()])];
            values.insert("RA0L", low(&ra0));
            values.insert("RA0H", high(&ra0));
            bitsout.insert("RQ", cat(&rq, &bits(0, 6)));
        }
        Some(m) => {
            let rq0 = g.c.and("RQ.0", &[m.rq0.clone(), exec.clone()]);
            let rq1 = g.c.and("RQ.1", &[m.rq1.clone(), exec.clone()]);
            let host0 = g.c.and("RQ.2", &[m.rq0.clone(), exec.clone(), m.rp[0].host.clone()]);
            let host1 = g.c.and("RQ.3", &[m.rq1.clone(), exec.clone(), m.rp[1].host.clone()]);
            bitsout.insert("RA0L", m.ra0[..8].to_vec());
            bitsout.insert("RA0H", m.ra0[8..].to_vec());
            bitsout.insert("RQ", cat(&[rq0, rq1, host0, host1, m.fetch_void[0].clone(), m.fetch_void[1].clone(), m.fetch_void[2].clone()], &bits(0, 1)));
        }
    }
    values.insert("RA1L", low(&sp_p1));
    values.insert("RA1H", high(&sp_p1));

    // Memory writes, in bus order: (gates, bus address, value). In memory mode the rare
    // sources (BC, DE, FF00+C, a16) are isolated to their opcodes first, as one bus
    // address the write translation reads, and the FF00+n byte to LDH (n),A.
    let on = |ops: Lin| vec![ops, exec.clone()];
    let isolated = d.memory.as_ref().map(|_| {
        let rare_cases = [(on(g.op(0x02)), d.bc.clone()), (on(g.op(0x12)), d.de.clone()), (on(g.op(0xE2)), cat(&d.regs[1], &bits(0xFF, 8))), (on(g.op(0xEA) + g.op(0x08)), d.imm16.clone())];
        let rare0 = g.c.mux("rare0", 16, &rare_cases);
        let ffn_w = g.isolate("w.ffn", &d.imm8, &g.op(0xE0));
        (rare0, ffn_w)
    });
    let mut w0: Vec<(Vec<Lin>, N4, N2)> = match &isolated {
        None => vec![
            (vec![g.op(0x02)], bc.clone(), av.clone()),
            (vec![g.op(0x12)], de.clone(), av.clone()),
            (vec![g.op(0x22) + g.op(0x32)], hl.clone(), av.clone()),
            (vec![g.op(0xE0)], ff(ff8.clone()), av.clone()),
            (vec![g.op(0xE2)], ff(rv[1].clone()), av.clone()),
            (vec![g.op(0xEA)], rare16.clone(), av.clone()),
            (vec![g.op(0x08)], rare16.clone(), low(&sp)),
        ],
        Some((rare0, ffn_w)) => vec![
            (vec![g.op(0x02) + g.op(0x12) + g.op(0xE2) + g.op(0xEA)], n4(rare0), av.clone()),
            (vec![g.op(0x22) + g.op(0x32)], hl.clone(), av.clone()),
            (vec![g.op(0xE0)], ff(n2(ffn_w)), av.clone()),
            (vec![g.op(0x08)], n4(rare0), low(&sp)),
        ],
    };
    let (hl_results, hl_cover) = results8(&g, 6);
    let push_pairs = [(0xC5u8, 0usize, 1usize), (0xD5, 2, 3), (0xE5, 4, 5)];
    let calls = g.op(0xCD) + t_call.clone();
    let rst = g.ops(|op| xyz(op).0 == 3 && xyz(op).2 == 7);
    let mut w0_tail: Vec<(Vec<Lin>, N4, N2)> = push_pairs.iter().map(|&(op, hi, _)| (vec![g.op(op)], sp_m1.clone(), rv[hi].clone())).collect();
    w0_tail.push((vec![g.op(0xF5)], sp_m1.clone(), av.clone()));
    // A call and a restart push the same return address: in memory mode one case each write.
    let merged = mode == Mode::Memory;
    if merged {
        w0_tail.push((vec![calls.clone() + rst.clone()], sp_m1.clone(), high(&pc_seq)));
    } else {
        w0_tail.push((vec![calls.clone()], sp_m1.clone(), high(&pc_seq)));
        w0_tail.push((vec![rst.clone()], sp_m1.clone(), high(&pc_seq)));
    }
    let singles = |cases: &[(Vec<Lin>, N4, N2)]| cases.iter().fold(zero(), |s, (gates, _, _)| s + gates[0].clone());
    let en0 = singles(&w0) + hl_cover.clone() + singles(&w0_tail);
    w0.extend(hl_results.into_iter().map(|(gates, v)| (gates, hl.clone(), v)));
    w0.extend(w0_tail);

    let mut w1: Vec<(Vec<Lin>, N4, N2)> = vec![(vec![g.op(0x08)], a16_p1.clone(), high(&sp))];
    for (op, _, lo) in push_pairs {
        w1.push((vec![g.op(op)], sp_m2.clone(), rv[lo].clone()));
    }
    w1.push((vec![g.op(0xF5)], sp_m2.clone(), [Nib::konst(0), Nib::of_bits(&f[4..8])]));
    if merged {
        w1.push((vec![calls + rst], sp_m2.clone(), low(&pc_seq)));
    } else {
        w1.push((vec![calls], sp_m2.clone(), low(&pc_seq)));
        w1.push((vec![rst], sp_m2.clone(), low(&pc_seq)));
    }
    let en1 = singles(&w1);

    let write = |g: &mut G, name: &str, cases: &[(Vec<Lin>, N4, N2)], en: Lin, disp_addr: N4, disp_val: N2| -> (N4, N2, Lin) {
        let gated = |gates: &[Lin]| gates.iter().cloned().chain([exec.clone()]).collect::<Vec<Lin>>();
        let mut ac: Vec<(Vec<Lin>, N4)> = cases.iter().map(|(gates, ad, _)| (gated(gates), ad.clone())).collect();
        let mut vc: Vec<(Vec<Lin>, N2)> = cases.iter().map(|(gates, _, v)| (gated(gates), v.clone())).collect();
        ac.push((vec![dispatch.clone()], disp_addr));
        vc.push((vec![dispatch.clone()], disp_val));
        let addr = vmux(g, &format!("{name}.a"), &ac);
        let val = vmux(g, &format!("{name}.v"), &vc);
        let enable = g.c.node(&format!("{name}.en"), vec![Step { coef: 1, lin: en + exec.clone(), theta: 2 }, Step { coef: 1, lin: dispatch.clone(), theta: 1 }]);
        (addr, val, enable)
    };
    let (wa0, wv0, we0) = write(&mut g, "W0", &w0, en0, sp_m1.clone(), high(&pc));
    let (wa1, wv1, we1) = write(&mut g, "W1", &w1, en1, sp_m2.clone(), low(&pc));
    values.insert("WA0L", low(&wa0));
    values.insert("WA0H", high(&wa0));
    values.insert("WV0", wv0.clone());
    values.insert("WA1L", low(&wa1));
    values.insert("WA1H", high(&wa1));
    values.insert("WV1", wv1.clone());

    // The step reaches VRAM or the FE page by a data access (memory mode), for RC bit 3.
    let mut at_lock: Option<Lin> = None;
    let mut fetch_lock: Option<Lin> = None;
    let mut ack_fe: [Lin; 2] = [zero(), zero()];
    let wq = match &d.memory {
        None => cat(&[we0, we1], &bits(0, 6)),
        Some(m) => {
            // The write records: physical addresses, and the line kept, from the
            // translations of each write's sources (`fields::route` has the reasons).
            let hl_w = g.op(0x22) + g.op(0x32) + hl_cover;
            let stack = pushes.clone();
            let (ff_n, rare_ops) = (g.op(0xE0), g.op(0x02) + g.op(0x12) + g.op(0xE2) + g.op(0xEA) + g.op(0x08));
            let (rare0, ffn_w) = isolated.expect("memory mode isolates the rare write sources");
            let sp_m1_bits = sp_m1_bits.expect("memory mode keeps SP−1's bits");
            let sp_m2_bits = sp_m2_bits.expect("memory mode keeps SP−2's bits");
            let a16_p1_bits = a16_p1_bits.expect("memory mode keeps a16+1's bits");
            let t_hl = translate::write_address(&mut g, "WHL", &d.hl, &m.slots[2].regs, &m.slots[2].dma, 3, Region::Wired);
            let t_sp1 = translate::write_address(&mut g, "WSP1", &sp_m1_bits, &m.slots[2].regs, &m.slots[2].dma, 3, Region::Wired).with_reg(&mut g, "WSP1", 3);
            let t_sp2 = translate::write_address(&mut g, "WSP2", &sp_m2_bits, &m.slots[3].regs, &m.slots[3].dma, 3, Region::Wired).with_reg(&mut g, "WSP2", 3);
            let t_ffn = translate::ff_page_write(&mut g, "WFFN", &ffn_w, &m.slots[2].regs);
            let t_rare0 = translate::write_address(&mut g, "WRARE0", &rare0, &m.slots[2].regs, &m.slots[2].dma, 0, Region::Wired).with_reg(&mut g, "WRARE0", 0);
            let t_a16 = translate::write_address(&mut g, "WA16", &a16_p1_bits, &m.slots[3].regs, &m.slots[3].dma, 0, Region::Literal).with_reg(&mut g, "WA16", 0);
            // An idle port rests where its head moves least. A step without write 0 (a
            // wake, or an opcode that does not write) parks write 0 on VOID, whose line
            // is zero, and write 1 on SP−2, which its translation reads every step and
            // whose line pushes and calls write (`docs/hardware-shaped.md` §16). Neither
            // keeps a record, since the write enables gate WQ.2 and WQ.3.
            let void = translate::void_block();
            let writes0 = hl_w.clone() + stack.clone() + ff_n.clone() + rare_ops.clone();
            let wp0 = translate::select(&mut g, "WP0", &[(on(hl_w.clone()), &t_hl), (on(stack.clone()), &t_sp1), (vec![dispatch.clone()], &t_sp1), (on(ff_n.clone()), &t_ffn), (on(rare_ops.clone()), &t_rare0), (vec![wake.clone()], &void), (vec![exec.clone(), one() - writes0], &void)], Access::Write);
            let ld_sp = g.op(0x08);
            let wp1 = translate::select(&mut g, "WP1", &[(vec![one() - ld_sp.clone()], &t_sp2), (vec![ld_sp.clone(), one() - exec.clone()], &t_sp2), (on(ld_sp.clone()), &t_a16)], Access::Write);
            let pair = Pair { stack: g.c.node("WS.stack", vec![Step { coef: 1, lin: stack + exec.clone(), theta: 2 }, Step { coef: 1, lin: dispatch.clone(), theta: 1 }]), ld: g.c.and("WS.08", &[g.op(0x08), exec.clone()]), sp1: &t_sp1, sp2: &t_sp2, a16: &t_rare0, a16p1: &t_a16 };
            // LD (a16),SP's a16 isolated to its opcode, so the same-line test reads it only then.
            let s0 = same_line(&mut g, &pair, &d.sp, &a16_ld.expect("memory mode isolates a16 for LD (a16),SP"));
            fetch_lock = Some(g.c.ge("WC.3", m.fetch_lock.clone(), 1));
            // MBC2 RAM holds a nibble whose reads return 0xF0 | nibble, so a write there
            // records a high nibble of 15. `cram` tells, per write, whether its bus
            // address is in A000-BFFF under MBC2, with one step per address source.
            let mbc2 = m.regs.0[Reg::Mapper as usize][2].clone();
            let cram = |g: &mut G, name: &str, sources: &[(Vec<Lin>, &W)]| -> Lin {
                let steps = sources.iter().map(|(gates, a)| {
                    let lits = [a[15].clone(), one() - a[14].clone(), a[13].clone(), mbc2.clone()];
                    let lin = gates.iter().chain(&lits).fold(zero(), |s, l| s + l.clone());
                    Step { coef: 1, lin, theta: (gates.len() + lits.len()) as i64 }
                }).collect();
                g.c.node(name, steps)
            };
            let ff_bits = cat(&ffn_w, &bits(0xFF, 8));
            // RQ bit 7: the step reads or writes VRAM or the FE page (OAM and the unusable
            // area) by its bus address while the host's copy of the access locks has
            // lapsed (CTL bit 7 clear), so the host asks the control model for them (`LK`).
            // It reads no late input, and RC bit 3 carries the same wire on every step.
            let lock_wire = decode::lockable(
                &mut g,
                "LKQ",
                &[
                    (on(hl_w.clone()), &d.hl),
                    (on(pushes.clone()), &sp_m1_bits),
                    (vec![dispatch.clone()], &sp_m1_bits),
                    (on(rare_ops.clone()), &rare0),
                    (on(pushes.clone()), &sp_m2_bits),
                    (vec![dispatch.clone()], &sp_m2_bits),
                    (on(ld_sp.clone()), &a16_p1_bits),
                    (vec![m.rq0.clone(), exec.clone()], &m.ra0),
                    (vec![m.rq1.clone(), exec.clone()], &m.sp_p1),
                ],
            );
            at_lock = Some(g.c.ge("RC.3", lock_wire.clone(), 1));
            let ask_locks = g.c.ge("RQ.7", lock_wire - ctl[7].clone() * 32, 1);
            bitsout.get_mut("RQ").expect("the request's bits")[7] = ask_locks;
            // Each data access that puts an FE page address on the bus unblocked by OAM DMA,
            // a DMG's OAM bug event at its M-cycle (`StepFacts::fe_accesses`): ACK bits 5 and 6
            // for the reads, WQ bits 6 and 7 for the writes.
            ack_fe = [
                decode::fe_access(&mut g, "ACK.5", &[(vec![m.rq0.clone(), exec.clone()], &m.ra0)], &m.slots[0].dma.oam),
                decode::fe_access(&mut g, "ACK.6", &[(vec![m.rq1.clone(), exec.clone()], &m.sp_p1)], &m.slots[1].dma.oam),
            ];
            let fe_w0 = decode::fe_access(&mut g, "WQ.6", &[(on(hl_w.clone()), &d.hl), (on(pushes.clone()), &sp_m1_bits), (vec![dispatch.clone()], &sp_m1_bits), (on(rare_ops.clone()), &rare0)], &m.slots[2].dma.oam);
            let fe_w1 = decode::fe_access(&mut g, "WQ.7", &[(on(pushes.clone()), &sp_m2_bits), (vec![dispatch.clone()], &sp_m2_bits), (on(ld_sp.clone()), &a16_p1_bits)], &m.slots[3].dma.oam);
            let cram0 = cram(&mut g, "W0.cram", &[(on(hl_w), &d.hl), (on(pushes.clone()), &sp_m1_bits), (vec![dispatch.clone()], &sp_m1_bits), (on(ff_n), &ff_bits), (on(rare_ops), &rare0)]);
            let cram1 = cram(&mut g, "W1.cram", &[(on(pushes.clone()), &sp_m2_bits), (vec![dispatch.clone()], &sp_m2_bits), (on(ld_sp), &a16_p1_bits)]);
            let written = [(&wv0, cram0), (&wv1, cram1)];
            let mut rec = Vec::new();
            let mut second = Vec::new();
            for (i, (we, wp)) in [(&we0, &wp0), (&we1, &wp1)].into_iter().enumerate() {
                let mut addr = wp.bits.clone();
                addr.resize(24, zero());
                for (byte, suffix) in ["a", "b", "c"].iter().enumerate() {
                    let name: &'static str = Box::leak(format!("WP{i}{suffix}").into_boxed_str());
                    bitsout.insert(name, addr[8 * byte..8 * byte + 8].to_vec());
                    kinds.insert(name, OutKind::Address { write: i as u8, byte: byte as u8 });
                }
                // The record is the whole line after the write: the old line read at the
                // write's address, the written byte at the word line of its low bits.
                // Write 1's line also holds write 0's byte when both share it.
                let old = g.c.head(&format!("WL{i}h"), g.c.level_of(&wp.bits[LINE_BITS..]), Query::Wires(wp.bits.clone()));
                // Gated by the write, so a step without one keeps the old line whole.
                let at = line::words(&mut g.c, &format!("WL{i}w"), &[(vec![(*we).clone()], &wp.bits[..LINE_BITS], 0)]);
                // The byte a write records under `gates`: its value, its high nibble 15 in MBC2 RAM.
                let byte_cases = |gates: Vec<Lin>, (v, cram): &(&N2, Lin)| -> Vec<(Vec<Lin>, N2)> {
                    let with = |extra: Lin| gates.iter().cloned().chain([extra]).collect::<Vec<Lin>>();
                    vec![(with(one() - cram.clone()), (*v).clone()), (with(cram.clone()), [v[0].clone(), Nib::konst(15)])]
                };
                for k in 0..LINE {
                    let old_k = n2(&old[k].iter().map(|&s| Lin::from(s)).collect::<W>());
                    let elsewhere = one() - at[k].clone();
                    let mut cases = byte_cases(vec![at[k].clone()], &written[i]);
                    if i == 0 {
                        cases.push((vec![elsewhere], old_k));
                    } else {
                        cases.extend(byte_cases(vec![elsewhere.clone(), s0[k].clone()], &written[0]));
                        cases.push((vec![elsewhere, one() - s0[k].clone()], old_k));
                    }
                    let name: &'static str = Box::leak(format!("WL{i}{k}").into_boxed_str());
                    values.insert(name, vmux(&mut g, name, &cases));
                    kinds.insert(name, OutKind::Line { write: i as u8, byte: k as u8 });
                }
                // One record per line: when write 1 shares write 0's line, its record holds
                // write 0's byte too (`s0`), so write 0's is left out
                // (`docs/hardware-shaped.md` §16).
                let alone = if i == 0 { vec![one() - s0.iter().cloned().fold(zero(), |a, x| a + x)] } else { Vec::new() };
                rec.push(g.c.and(&format!("WQ.{}", 2 + i), &[vec![we.clone(), wp.record.clone()], alone].concat()));
                // The second device the write reaches, decoded only when WQ.4 or WQ.5 flags one.
                let (heard, mut port) = translate::second_port(&mut g, &format!("WD{i}"), &wp.bits, &m.regs);
                second.push(g.c.and(&format!("WQ.{}", 4 + i), &[we.clone(), heard]));
                port.resize(24, zero());
                for (byte, suffix) in ["a", "b", "c"].iter().enumerate() {
                    let name: &'static str = Box::leak(format!("WD{i}{suffix}").into_boxed_str());
                    bitsout.insert(name, port[8 * byte..8 * byte + 8].to_vec());
                }
            }
            // Evidence of the model's own reads and fetches.
            bitsout.insert("OP0'", d.op0.clone());
            bitsout.insert("OP1'", d.op1.clone());
            bitsout.insert("OP2'", d.imm16[8..].to_vec());
            for (i, names) in [["RP0a", "RP0b", "RP0c"], ["RP1a", "RP1b", "RP1c"]].into_iter().enumerate() {
                let mut addr = m.rp[i].bits.clone();
                addr.resize(24, zero());
                for (byte, name) in names.into_iter().enumerate() {
                    bitsout.insert(name, addr[8 * byte..8 * byte + 8].to_vec());
                }
            }
            for (k, names) in [["DPa", "DPb", "DPc"], ["DQa", "DQb", "DQc"]].into_iter().enumerate() {
                let mut addr = m.dma_lines[k].clone();
                addr.resize(24, zero());
                for (byte, name) in names.into_iter().enumerate() {
                    bitsout.insert(name, addr[8 * byte..8 * byte + 8].to_vec());
                }
            }
            for (k, names) in [["FP0a", "FP0b", "FP0c"], ["FP1a", "FP1b", "FP1c"], ["FP2a", "FP2b", "FP2c"]].into_iter().enumerate() {
                let mut addr = m.fetch_ports[k].clone();
                addr.resize(24, zero());
                for (byte, name) in names.into_iter().enumerate() {
                    bitsout.insert(name, addr[8 * byte..8 * byte + 8].to_vec());
                }
            }
            bitsout.insert("RV0", m.read[0].clone());
            bitsout.insert("RV1", m.read[1].clone());
            let banks = m.regs.1.as_ref().expect("the memory circuit's registers carry their banks");
            for (window, base) in [("MAP0", &banks.map0), ("MAPR", &banks.mapr), ("MAPS", &banks.maps), ("MAPW", &banks.mapw)] {
                let mut addr = base.clone();
                addr.resize(24, zero());
                for (byte, suffix) in ["a", "b", "c"].iter().enumerate() {
                    let name: &'static str = Box::leak(format!("{window}{suffix}").into_boxed_str());
                    bitsout.insert(name, addr[8 * byte..8 * byte + 8].to_vec());
                }
            }
            vec![we0, we1, rec[0].clone(), rec[1].clone(), second[0].clone(), second[1].clone(), fe_w0, fe_w1]
        }
    };
    bitsout.insert("WQ", wq);

    // The step length, as bits like the 0/1 circuit's, whose sums run over exclusive
    // one-hots.
    let (not_t, taken_t, cb_t) = cycle_table();
    let mc_bit = |k: usize| -> Lin {
        let mut s = zero();
        for op in (0..=255u8).filter(|&op| op != 0xCB) {
            s = s + g.op(op) * ((not_t[op as usize] >> k) & 1) as i64;
        }
        for (taken, base) in [(&t_jr, 0x20u8), (&t_ret, 0xC0), (&t_jp, 0xC2), (&t_call, 0xC4)] {
            s = s + taken.clone() * (((taken_t[base as usize] >> k) & 1) as i64 - ((not_t[base as usize] >> k) & 1) as i64);
        }
        s + g.cb_cycles_bit(&cb_t, k)
    };
    let mc_exec: W = (0..8).map(mc_bit).collect();
    let mut mc = g.c.mux("MC", 8, &[(vec![exec.clone()], mc_exec), (vec![dispatch.clone()], bits(5, 8))]);
    // A wake takes no M-cycle: the control logic says when it comes (`Lines::wake`).
    // In memory mode a taken JR's target page is the next pc's high byte, PCH plus the
    // low byte's carry less the offset's sign (`pc_seq`), and PC + k's pages come from the
    // fetch's line test: no node reads PC's low byte or the offset for them.
    let (target_fe, pages) = match &d.memory {
        None => (decode::jr_target_fe(&mut g.c, "IDLE.jr", &d.pc, &d.imm8, d.ctl[3].clone()), None),
        Some(m) => {
            let hi = pc[3].lin.clone() * 16 + pc[2].lin.clone() + pc_carries[1].clone() - e8_neg.clone();
            let jr = g.c.node("IDLE.jr", vec![Step { coef: 1, lin: hi.clone(), theta: 0xFE }, Step { coef: -1, lin: hi, theta: 0xFF }]);
            (jr, Some(decode::fetch_pages(&mut g, &d, m)))
        }
    };
    mc[3..7].clone_from_slice(&decode::idle_bits(&mut g, &d, &t_jr, &t_call, target_fe, pages.as_ref()));
    bitsout.insert("MC", mc);
    // Each access's M-cycle in the step (`RC`, `WC`), from the same kind of tables.
    let (acc_not, acc_taken, acc_cb, acc_dispatch) = decode::access_table();
    let mut fetched: [Lin; 2] = [zero(), zero()];
    for (j, name) in ["RC", "WC"].into_iter().enumerate() {
        let pick = |t: &[[u8; 2]; 256]| -> [u8; 256] { std::array::from_fn(|op| t[op][j]) };
        let exec_v = g.table_byte(&pick(&acc_not), &pick(&acc_taken), &pick(&acc_cb), [&t_jr, &t_ret, &t_jp, &t_call]);
        let mut v = g.c.mux(name, 8, &[(vec![exec.clone()], exec_v), (vec![dispatch.clone()], bits(acc_dispatch[j] as u64, 8))]);
        // Bit 3, which no M-cycle reaches (they fit bits 0-2): in RC, the step reaches VRAM
        // or the FE page by a data access, so the host looks the access locks up per
        // access; in WC, an operand fetch's line does, so the host looks them up at each
        // operand's M-cycle.
        if let Some(l) = [&at_lock, &fetch_lock][j] {
            v[3] = l.clone();
        }
        fetched[j] = v[7].clone();
        bitsout.insert(name, v);
    }
    if let Some(m) = &d.memory {
        let fe = decode::fetch_fe_bits(&mut g, &d, m, fetched, pages.as_ref().expect("the fetches' pages in memory mode"));
        bitsout.get_mut("CTL'").expect("CTL'")[5..8].clone_from_slice(&fe);
    }
    bitsout.insert("ACK", cat(&cat(&d.vector, &ack_fe), &bits(0, 1)));
    // The OAM pass's slots, read late so the four row lines live only for the corruption's
    // levels (`oam_bug::memory_pass`).
    if let Some(m) = &d.memory {
        let layer = g.c.depth().saturating_sub(3).max(1);
        for (name, b) in super::oam_bug::memory_pass(&mut g.c, &m.ob.clone().try_into().expect("OB is a byte"), layer) {
            bitsout.insert(Box::leak(name.into_boxed_str()), b.to_vec());
        }
    }

    let results: &[&str] = match mode {
        Mode::Host => &RESULT,
        Mode::Memory => &RESULT_MEM,
    };
    for name in REQUEST.iter().chain(results.iter()) {
        let kind = kinds.get(name).copied().unwrap_or(OutKind::Shared);
        if let Some(v) = values.remove(name) {
            g.c.output_nibbles(name, kind, v);
        } else {
            let b: [Lin; 8] = bitsout.remove(name).unwrap_or_else(|| panic!("no output {name}")).try_into().unwrap();
            g.c.output_kind(name, b, kind);
        }
    }
    assert!(values.is_empty() && bitsout.is_empty(), "unrouted outputs: {:?} {:?}", values.keys(), bitsout.keys());
    super::finish(g, mode, include_str!("changes-memory.txt"))
}

/// Declares the decode's exclusive wires one-hot groups, so a nibble case gated by a
/// sum of them counts one weight in the f16 analysis: the opcode one-hots (in memory
/// mode, ANDs of the fetched nibbles' one-hots), the CB decode and the dispatch lines.
fn declare_groups(g: &mut G, d: &Decoded) {
    if g.mode == Mode::Memory {
        let o = g.o.clone();
        g.c.declare_one_hot(&o);
    }
    match &g.cb {
        Cb::Hots(q) => {
            let q = q.clone();
            g.c.declare_one_hot(&q);
        }
        Cb::Fields { xy, z, writes, .. } => {
            let (xy, z, writes) = (xy.clone(), z.clone(), writes.clone());
            for group in [xy, z, writes] {
                g.c.declare_one_hot(&group);
            }
        }
    }
    g.c.declare_one_hot(&d.disp);
}
