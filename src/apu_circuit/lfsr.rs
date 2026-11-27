//! The noise channel's LFSR in the circuit: SameBoy's step (`step_lfsr`), an affine map
//! over GF(2) whose powers the run takes, and the NR43 write's glitches (`nr43_write`,
//! CPU CGB E).

use super::build::*;
use super::circuit::*;
use crate::circuit::{Circuit, Lin, Step};

/// An affine map on the 15 LFSR bits: new bit i is `parity(l & mask[i]) ^ konst[i]`.
#[derive(Clone, Copy)]
pub(super) struct Affine {
    pub mask: [u16; 15],
    pub konst: [bool; 15],
}

impl Affine {
    fn identity() -> Self {
        Affine { mask: std::array::from_fn(|i| 1 << i), konst: [false; 15] }
    }

    /// One step: the feedback `1 ^ l0 ^ l1` enters bit 14 and, narrow, bit 6.
    pub fn step(narrow: bool) -> Self {
        let mut a = Affine { mask: [0; 15], konst: [false; 15] };
        for i in 0..15 {
            if i == 14 || (narrow && i == 6) {
                a.mask[i] = 0b11;
                a.konst[i] = true;
            } else {
                a.mask[i] = 1 << (i + 1);
            }
        }
        a
    }

    /// `self ∘ other`.
    fn after(&self, other: &Affine) -> Affine {
        let mut out = Affine { mask: [0; 15], konst: [false; 15] };
        for i in 0..15 {
            let mut k = self.konst[i];
            let mut m = 0u16;
            for j in 0..15 {
                if self.mask[i] >> j & 1 == 1 {
                    m ^= other.mask[j];
                    k ^= other.konst[j];
                }
            }
            out.mask[i] = m;
            out.konst[i] = k;
        }
        out
    }

    pub fn power(narrow: bool, n: usize) -> Affine {
        let one = Affine::step(narrow);
        (0..n).fold(Affine::identity(), |acc, _| one.after(&acc))
    }
}

/// Steps `[gate ∧ parity(bits) ^ konst]` adds to a node: the parity's staircase over the
/// bits' sum, every step offset so it holds only while `gate`, a sum of `gates` literals,
/// is all ones.
pub(super) fn gated_parity(bits: &[Lin], konst: bool, gates: &[Lin]) -> Vec<Step> {
    let size = bits.len() as i64;
    let n = gates.len() as i64;
    let big = size + 1;
    let g = sum(gates.iter().cloned());
    let mut steps = Vec::new();
    if konst {
        steps.push(Step { coef: 1, lin: g.clone(), theta: n });
    }
    for t in 1..=size {
        let coef = if (t % 2 == 1) != konst { 1 } else { -1 };
        steps.push(Step { coef, lin: sum(bits.iter().cloned()) + (g.clone() - n) * big, theta: t });
    }
    steps
}

/// The LFSR after one step, `narrow` the new width bit: bit 14 the feedback, bit 6 the
/// feedback when narrow, else old bit 7.
pub(super) fn stepped(c: &mut Circuit, name: &str, l: &[Lin], narrow: &Lin) -> W {
    let fb = c.node(&format!("{name}.fb"), gated_parity(&[l[0].clone(), l[1].clone()], true, &[]));
    (0..15)
        .map(|i| match i {
            14 => fb.clone(),
            6 => c.node(&format!("{name}.6"), vec![Step { coef: 1, lin: narrow.clone() + fb.clone(), theta: 2 }, Step { coef: 1, lin: nt(narrow) + l[7].clone(), theta: 2 }]),
            _ => l[i + 1].clone(),
        })
        .collect()
}

/// `a ∧ b` as a wire, or a constant when either is.
fn and2(c: &mut Circuit, name: &str, a: &Lin, b: &Lin) -> Lin {
    all(c, name, &[a.clone(), b.clone()])
}

/// The NR43 write's LFSR effects, as cases under `gate`: `nr43_write(new)` on CPU CGB E.
/// Returns the cases for `lfsr`, `cur4` (the sample bit, with `smp4` following it), `b7`
/// and `sin`.
pub(super) fn nr43_write(c: &mut Circuit, s: &St, v: &[Lin], gate: &Lin, cs: &mut Cases) {
    // The counter and the LFSR change nearly every pass while the channel counts; the
    // write reads copies that hold still without it.
    let old = s["nr43"].clone();
    let (ctr, l) = (gated(c, "ctr4@NR43", &s["ctr4"], gate), gated(c, "lfsr@NR43", &s["lfsr"], gate));
    let ctr16: W = ctr.iter().cloned().chain([k(0)]).collect();
    let same = c.eq("nr43 same hi", val(&old[4..8]) - val(&v[4..8]), 0);
    let ob = pick(c, "nr43 old bit", &ctr16, &old[4..8]);
    let nb = pick(c, "nr43 new bit", &ctr16, &v[4..8]);
    let gi: W = old[4..7].iter().cloned().chain([v[7].clone()]).collect();
    let gb = pick(c, "nr43 glitch bit", &ctr16, &gi);
    let base = [gate.clone(), nt(&same)];
    let k1 = all(c, "nr43 cat1", &with(&base, &[ob.clone(), nb.clone(), nt(&gb)]));
    let k2 = all(c, "nr43 cat2", &with(&base, &[nt(&ob), nt(&nb), gb.clone()]));
    let k3 = all(c, "nr43 rise", &with(&base, &[nt(&ob), nb.clone()]));
    let narrow = v[3].clone();
    let st = stepped(c, "nr43 step", &l, &narrow);
    let (t1, t2) = (old[4..7].to_vec(), v[4..7].to_vec());
    // A node over the six bits of (t1, t2) that holds for the pairs `f` selects.
    let table = |c: &mut Circuit, name: &str, extra: &[Lin], f: &dyn Fn(u64, u64) -> bool| -> Lin {
        let steps = (0..8u64)
            .flat_map(|a| (0..8u64).map(move |b| (a, b)))
            .filter(|&(a, b)| f(a, b))
            .map(|(a, b)| {
                let lits = with(&with(&is(&t1, a), &is(&t2, b)), extra);
                Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }
            })
            .collect();
        c.node(name, steps)
    };
    // Category 1 with bit 7 of the new value set.
    let k1v = all(c, "nr43 cat1 v7", &[k1.clone(), v[7].clone()]);
    let cond = table(c, "nr43 cat1 cond", &[k1v.clone()], &|a, b| (a ^ 7) + b > 7 || (a ^ 7) & b != 0);
    let x = table(c, "nr43 cat1 X", &[k1v.clone()], &|a, b| ((a ^ 7) + b > 7 || (a ^ 7) & b != 0) && (a == 0 || a == 4) && b == 3);
    let y = table(c, "nr43 cat1 Y", &[k1v.clone()], &|a, b| ((a ^ 7) + b > 7 || (a ^ 7) & b != 0) && a == 2 && b == 3);
    let plain = c.node("nr43 cat1 plain", vec![Step { coef: 1, lin: cond.clone() - x.clone() - y.clone(), theta: 1 }]);
    // l1: bit 7 takes bit 8.
    let l1: W = (0..15).map(|i| if i == 7 { l[8].clone() } else { l[i].clone() }).collect();
    let shifted_and = |c: &mut Circuit, name: &str, l1: &W, mask: &dyn Fn(usize) -> Option<Lin>| -> W {
        (0..15)
            .map(|i| {
                let next = if i < 14 { l1[i + 1].clone() } else { k(0) };
                match mask(i) {
                    Some(m) => {
                        // l1_i ∧ (next ∨ m)
                        let either = c.ge(&format!("{name}.or{i}"), next + m, 1);
                        and2(c, &format!("{name}.{i}"), &l1[i], &either)
                    }
                    None => and2(c, &format!("{name}.{i}"), &l1[i], &next),
                }
            })
            .collect()
    };
    let lx = shifted_and(c, "nr43 X", &l1, &|i| (0x545 >> i & 1 == 1).then(|| k(1)));
    let m8 = and2(c, "nr43 Y m8", &l1[2], &l1[3]);
    let m800 = and2(c, "nr43 Y m800", &l1[10], &l1[11]);
    let ly = shifted_and(c, "nr43 Y", &l1, &|i| match i {
        3 => Some(m8.clone()),
        11 => Some(m800.clone()),
        _ => (0x555 >> i & 1 == 1).then(|| k(1)),
    });
    // Then the width fix and the forced high bits.
    let fix = all(c, "nr43 narrow fix", &[nt(&narrow), old[3].clone(), s["sin"][0].clone()]);
    let finish1 = |c: &mut Circuit, name: &str, l2: &W| -> W {
        (0..15)
            .map(|i| match i {
                14 => k(1),
                6 => c.node(&format!("{name}.6"), vec![Step { coef: 1, lin: narrow.clone(), theta: 1 }, Step { coef: 1, lin: nt(&narrow) + fix.clone() + s["b7"][0].clone(), theta: 3 }, Step { coef: 1, lin: nt(&narrow) + nt(&fix) + l2[6].clone(), theta: 3 }]),
                _ => l2[i].clone(),
            })
            .collect()
    };
    let r_x = finish1(c, "nr43 X'", &lx);
    let r_y = finish1(c, "nr43 Y'", &ly);
    let r_plain = finish1(c, "nr43 plain'", &l1);
    // Category 2: the glitch map's class for (t1, t2) with bit 7 of the new value set.
    const MAP: [(u64, u64, u8); 21] = [(0, 2, 4), (0, 3, 2), (0, 4, 2), (0, 5, 2), (1, 2, 2), (1, 3, 4), (1, 4, 2), (1, 5, 2), (2, 0, 1), (2, 1, 2), (2, 3, 1), (2, 4, 5), (2, 5, 3), (3, 4, 2), (3, 5, 2), (4, 1, 2), (4, 2, 2), (4, 3, 2), (5, 0, 6), (5, 2, 2), (5, 3, 2)];
    let class = |g: u8| move |a: u64, b: u64| MAP.iter().any(|&(x, y, z)| x == a && y == b && z == g);
    let k2v = all(c, "nr43 cat2 v7", &[k2.clone(), v[7].clone()]);
    let g: Vec<Lin> = (1..=6u8).map(|cl| table(c, &format!("nr43 g{cl}"), &[k2v.clone()], &class(cl))).collect();
    let g0 = k2.clone() - sum(g.iter().cloned());
    // g1, g6: step, then fixes.
    let f1 = c.node("nr43 g6 f1", {
        let mut st6 = Vec::new();
        for (bits, extra) in [((0u64, 0u64, 1u64, 0u64), Some(narrow.clone())), ((1, 0, 1, 1), None)] {
            let mut lits = vec![lit(&st[0], bits.0 == 1), lit(&st[4], bits.1 == 1), lit(&st[5], bits.2 == 1), lit(&st[6], bits.3 == 1)];
            lits.extend(extra);
            st6.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
        }
        st6
    });
    let f2 = c.node("nr43 g6 f2", {
        let mut st6 = Vec::new();
        for bits in [(0u64, 0u64, 1u64, 0u64), (1, 0, 1, 1)] {
            let lits = [lit(&st[0], bits.0 == 1), lit(&st[12], bits.1 == 1), lit(&st[13], bits.2 == 1), lit(&st[14], bits.3 == 1)];
            st6.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: 4 });
        }
        st6
    });
    let b1 = and2(c, "nr43 g16 b1", &st[1], &st[0]);
    let r_g1: W = (0..15).map(|i| if i == 1 { b1.clone() } else { st[i].clone() }).collect();
    let b5 = all(c, "nr43 g6 b5", &[st[5].clone(), nt(&f1)]);
    let b13 = all(c, "nr43 g6 b13", &[st[13].clone(), nt(&f2)]);
    let r_g6: W = (0..15).map(|i| match i {
        1 => b1.clone(),
        5 => b5.clone(),
        13 => b13.clone(),
        _ => st[i].clone(),
    }).collect();
    // g2: step, then AND with the old value except bit 0.
    let r_g2: W = (0..15).map(|i| if i == 0 { st[0].clone() } else { and2(c, &format!("nr43 g2.{i}"), &st[i], &l[i]) }).collect();
    // g4: step, then bits 1 and 13 (and 5 when narrow) AND with the old value.
    let r_g4: W = (0..15)
        .map(|i| match i {
            1 | 13 => and2(c, &format!("nr43 g4.{i}"), &st[i], &l[i]),
            5 => c.node("nr43 g4.5", vec![Step { coef: 1, lin: nt(&narrow) + st[5].clone(), theta: 2 }, Step { coef: 1, lin: narrow.clone() + st[5].clone() + l[5].clone(), theta: 3 }]),
            _ => st[i].clone(),
        })
        .collect();
    // g3: no step, bit 0 takes bit 1. g5 first clears bit 14 (and 6 when narrow) when
    // bits 1-0 are 10, and bit 3 when bits 4, 3, 0 are 0, 1, 0.
    let r_g3: W = (0..15).map(|i| if i == 0 { l[1].clone() } else { l[i].clone() }).collect();
    let ten = and2(c, "nr43 g5 10", &l[1], &nt(&l[0]));
    let r_g5: W = (0..15)
        .map(|i| match i {
            0 => l[1].clone(),
            14 => all(c, "nr43 g5.14", &[l[14].clone(), nt(&ten)]),
            6 => c.node("nr43 g5.6", vec![Step { coef: 1, lin: l[6].clone() + nt(&narrow), theta: 2 }, Step { coef: 1, lin: l[6].clone() + narrow.clone() + nt(&ten), theta: 3 }]),
            3 => c.node("nr43 g5.3", vec![Step { coef: 1, lin: l[3].clone(), theta: 1 }, Step { coef: -1, lin: l[3].clone() + nt(&l[0]) + nt(&l[4]), theta: 3 }]),
            _ => l[i].clone(),
        })
        .collect();
    let st_cases = [vec![k1.clone(), nt(&v[7])], vec![k3.clone()]];
    for gates in &st_cases {
        cs.put("lfsr", gates.clone(), st.clone());
    }
    cs.put("lfsr", vec![x.clone()], r_x);
    cs.put("lfsr", vec![y.clone()], r_y);
    cs.put("lfsr", vec![plain.clone()], r_plain);
    cs.put("lfsr", vec![g0.clone()], st.clone());
    cs.put("lfsr", vec![g[0].clone()], r_g1);
    cs.put("lfsr", vec![g[1].clone()], r_g2);
    cs.put("lfsr", vec![g[2].clone()], r_g3);
    cs.put("lfsr", vec![g[3].clone()], r_g4);
    cs.put("lfsr", vec![g[4].clone()], r_g5);
    cs.put("lfsr", vec![g[5].clone()], r_g6);
    // The stepping cases record bit 7 before the step, the new width and the sample bit.
    let stepping: Vec<Lin> = [k1.clone() - k1v.clone(), k3.clone(), g0, g[0].clone(), g[1].clone(), g[3].clone(), g[5].clone()].to_vec();
    let steps = c.node("nr43 stepping", vec![Step { coef: 1, lin: sum(stepping.iter().cloned()), theta: 1 }]);
    cs.put("b7", vec![steps.clone()], vec![l[7].clone()]);
    let width_set = c.node("nr43 width set", vec![Step { coef: 1, lin: steps.clone() + cond.clone() + g[2].clone() + g[4].clone(), theta: 1 }]);
    cs.put("sin", vec![width_set], vec![narrow.clone()]);
    // The sample bit follows the new bit 0 where SameBoy calls `update_lfsr`, and the
    // latched sample the bit and the volume.
    let vol = s["vol4"].clone();
    for (gate, bit) in [(steps, st[0].clone()), (x, lx[0].clone()), (y, ly[0].clone()), (g[2].clone(), l[1].clone()), (g[4].clone(), l[1].clone())] {
        let smp: W = (0..4).map(|b| all(c, &format!("nr43 smp.{b}"), &[bit.clone(), vol[b].clone()])).collect();
        cs.put("cur4", vec![gate.clone()], vec![bit]);
        cs.put("smp4", vec![gate], smp);
    }
}
