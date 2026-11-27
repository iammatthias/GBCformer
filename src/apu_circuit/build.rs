//! Shared circuit helpers for the APU prototype: modular sums, keep-or-replace
//! selection and literal conjunctions, all as single threshold levels.

use crate::circuit::{one, Circuit, Lin, Sig, Step, F16_ACC_GATE, F16_SPAN};
use std::collections::BTreeMap;

pub type W = Vec<Lin>;

pub fn w(b: [Sig; 8]) -> W {
    b.iter().map(|&s| Lin::from(s)).collect()
}
pub fn k(c: i64) -> Lin {
    Lin::konst(c)
}
pub fn nt(l: &Lin) -> Lin {
    one() - l.clone()
}
pub fn sum(ls: impl IntoIterator<Item = Lin>) -> Lin {
    ls.into_iter().fold(k(0), |a, l| a + l)
}
pub fn val(bits: &[Lin]) -> Lin {
    bits.iter().enumerate().fold(k(0), |a, (i, b)| a + b.clone() * (1 << i))
}

/// Bits `0..n` of `Σ w·x + konst` modulo 2^n, every `x` a 0/1 form. Bit k is a staircase
/// over the terms' weights modulo 2^(k+1), so negative weights are two's complement.
/// With `gate`, every bit is 0 while the gate is 0, or `keep[k]` when `keep` is given.
pub fn sum_mod(c: &mut Circuit, name: &str, terms: &[(Lin, i64)], konst: i64, n: usize, gate: Option<(&Lin, Option<&[Lin]>)>) -> W {
    let (terms, konst) = positive(terms, konst);
    (0..n)
        .map(|b| {
            let m = 1i64 << (b + 1);
            let p = 1i64 << b;
            let form = terms.iter().fold(k(konst.rem_euclid(m)), |a, (x, wt)| {
                let wm = wt.rem_euclid(m);
                if wm == 0 { a } else { a + x.clone() * wm }
            });
            let form = c.modulo(form, m);
            let (lo, hi) = c.bounds(&form);
            let mut steps = Vec::new();
            let mut on_gate = 0;
            for j in 1..=hi / p {
                let (coef, theta) = (if j % 2 == 1 { 1 } else { -1 }, j * p);
                let Some((g, _)) = gate else {
                    steps.push(Step { coef, lin: form.clone(), theta });
                    continue;
                };
                // A gate weight must stay within an f16 unit's span, so each step is gated
                // in its narrow orientation, `[s ≥ θ ∧ g]` with offset hi − θ + 1 or
                // `g − [s ≤ θ − 1 ∧ g]` with offset θ − lo. When neither fits, the step is
                // `[s ≥ θ ∧ g]` with the full offset hi + 1. A step that always holds is g.
                if theta <= lo {
                    on_gate += coef;
                } else if hi - theta + 1 <= F16_SPAN || theta - lo > F16_SPAN {
                    let off = if hi - theta + 1 <= F16_SPAN { hi - theta + 1 } else { hi + 1 };
                    steps.push(Step { coef, lin: form.clone() + (g.clone() - 1) * off, theta });
                } else {
                    on_gate += coef;
                    steps.push(Step { coef: -coef, lin: form.clone() * -1 + (g.clone() - 1) * (theta - lo), theta: 1 - theta });
                }
            }
            if let Some((g, _)) = gate
                && on_gate != 0
            {
                steps.push(Step { coef: on_gate, lin: g.clone(), theta: 1 });
            }
            if let Some((g, Some(keep))) = gate {
                steps.push(Step { coef: 1, lin: keep[b].clone() - g.clone(), theta: 1 });
            }
            c.node(&format!("{name}.{b}"), steps)
        })
        .collect()
}

/// A negative weight on a 0/1 literal becomes the complement with the weight's
/// magnitude, so a form never carries a weight near 2^(k+1) for a small negative one:
/// the checkpoint bounds each unit's Σ |weight| by 2^23 and each wire's down sum by 2^24.
pub fn positive(terms: &[(Lin, i64)], konst: i64) -> (Vec<(Lin, i64)>, i64) {
    let mut kk = konst;
    let t = terms
        .iter()
        .map(|(x, wt)| {
            if *wt < 0 {
                kk += wt;
                (nt(x), -wt)
            } else {
                (x.clone(), *wt)
            }
        })
        .collect();
    (t, kk)
}

/// `[all literals]`, each literal a 0/1 form.
pub fn all(c: &mut Circuit, name: &str, lits: &[Lin]) -> Lin {
    c.ge(name, sum(lits.iter().cloned()), lits.len() as i64)
}

/// A field that keeps its value unless one of mutually exclusive cases fires; a case
/// is a conjunction of 0/1 literals and the value it loads. It takes one level, and bit k
/// is `[keep] + Σ [case ∧ v ∧ ¬keep] − [case ∧ keep ∧ ¬v]`.
pub fn sel(c: &mut Circuit, name: &str, keep: &[Lin], cases: &[(Vec<Lin>, Vec<Lin>)]) -> W {
    (0..keep.len())
        .map(|b| {
            let mut steps = vec![Step { coef: 1, lin: keep[b].clone(), theta: 1 }];
            for (gates, v) in cases {
                let g = sum(gates.iter().cloned());
                let n = gates.len() as i64;
                steps.push(Step { coef: 1, lin: g.clone() + v[b].clone() + nt(&keep[b]), theta: n + 2 });
                steps.push(Step { coef: -1, lin: g + keep[b].clone() + nt(&v[b]), theta: n + 2 });
            }
            c.node(&format!("{name}.{b}"), steps)
        })
        .collect()
}

/// The byte whose bits are `bits`, zero-padded.
pub fn byte_of(bits: &[Lin]) -> [Lin; 8] {
    std::array::from_fn(|i| bits.get(i).cloned().unwrap_or(k(0)))
}

/// `sum_mod` in blocks of bits, one level per block. A block reads the terms' digits in
/// it and the carry out of the block below, and its bits and carry are staircases over
/// that one form. The boundaries keep the form of every block wider than one bit within
/// an f16 gate (`F16_ACC_GATE`). Among those, they minimise the last block's level first
/// and the units second, so a block whose terms are early and that takes no carry sits
/// early, off the late terms' path.
pub fn sum_blocked(c: &mut Circuit, name: &str, terms: &[(Lin, i64)], konst: i64, n: usize, gate: Option<(&Lin, Option<&[Lin]>)>) -> W {
    let (terms, konst) = positive(terms, konst);
    let terms = &terms[..];
    let digit = |w: i64, s: usize, e: usize| w.rem_euclid(1i64 << e) >> s;
    let levels: Vec<u32> = terms.iter().map(|(x, _)| x.max_level(&c.levels)).collect();
    let floor = gate.map_or(0, |(g, kp)| g.max_level(&c.levels).max(kp.map_or(0, |kp| c.level_of(kp))));
    let bits_of = |v: i64| (64 - (v as u64).leading_zeros()) as usize;
    // For each boundary, keyed by the carry's largest value and its level, the best
    // (last level, units) pair and the boundaries that reach it.
    let mut best: Vec<BTreeMap<(i64, u32), ((u32, i64), Vec<usize>)>> = vec![BTreeMap::new(); n + 1];
    best[0].insert((0, 0), ((0, 0), Vec::new()));
    for s in 0..n {
        for ((cmax, clevel), ((top, units), ends)) in std::mem::take(&mut best[s]) {
            for e in s + 1..=n {
                let ds: Vec<(usize, i64)> = terms.iter().enumerate().map(|(i, (_, wt))| (i, digit(*wt, s, e))).filter(|&(_, d)| d != 0).collect();
                let span = ds.iter().map(|&(_, d)| d).sum::<i64>() + cmax;
                if span > F16_ACC_GATE && e > s + 1 {
                    break;
                }
                let inputs = ds.iter().map(|&(i, _)| levels[i]).chain((cmax > 0).then_some(clevel)).max();
                let level = match inputs {
                    Some(l) => 1 + l.max(floor),
                    None => gate.map_or(0, |_| 1 + floor),
                };
                let hi = span + digit(konst, s, e);
                // Each bit's staircase has one step per multiple of its weight in the
                // form's residue range; the carry's bits read the whole form.
                let steps: i64 = (0..e - s).map(|b| (ds.iter().map(|&(_, d)| d % (2 << b)).sum::<i64>() + cmax.min((2 << b) - 1)) >> b).sum::<i64>() + if e < n { (hi >> (e - s)) * bits_of(hi >> (e - s)) as i64 } else { 0 };
                let key = if e < n { ((1i64 << bits_of(hi >> (e - s))) - 1, level) } else { (0, 0) };
                let cand = (top.max(level), units + 2 * steps);
                if best[e].get(&key).is_none_or(|(b, _)| cand < *b) {
                    best[e].insert(key, (cand, ends.iter().copied().chain([e]).collect()));
                }
            }
        }
    }
    let ends = best[n].values().min_by_key(|(b, _)| *b).map(|(_, ends)| ends.clone()).unwrap_or_default();
    let mut out = Vec::new();
    let mut carry: W = Vec::new();
    let mut s = 0;
    for e in ends {
        let width = e - s;
        let mut dterms: Vec<(Lin, i64)> = terms.iter().map(|(x, wt)| (x.clone(), digit(*wt, s, e))).filter(|(_, d)| *d != 0).collect();
        dterms.extend(carry.iter().enumerate().map(|(j, x)| (x.clone(), 1i64 << j)));
        let dk = digit(konst, s, e);
        let keep: Option<Vec<Lin>> = gate.and_then(|(_, kp)| kp.map(|kp| kp[s..e].to_vec()));
        let g = gate.map(|(g, _)| (g, keep.as_deref()));
        out.extend(sum_mod(c, &format!("{name}[{s}]"), &dterms, dk, width, g));
        if e < n {
            let form = dterms.iter().fold(k(dk), |a, (x, wt)| a + x.clone() * *wt);
            let (_, hi) = c.bounds(&form);
            carry = (0..bits_of(hi >> width)).map(|j| c.bit_of(&format!("{name}c{s}.{j}"), form.clone(), (width + j) as u32)).collect();
        }
        s = e;
    }
    out
}

/// Constant bits of `x`.
pub fn kbits(x: u64, n: usize) -> W {
    (0..n).map(|i| k(((x >> i) & 1) as i64)).collect()
}
