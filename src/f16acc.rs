//! What an f16 accumulator holds exactly (`docs/stock-llama.md` §6, tier T2). A matmul
//! that sums its products in f16, in any order, is exact when every partial sum is. The
//! products of a row are multiples of one quantum, and every subset of them must sum to
//! at most 2^11 quanta (and below the f16 maximum). The largest subset sum is the
//! sum of the positive products or of the negative ones, whichever is larger, at the
//! inputs that maximise it.
//!
//! The analysis works on integer forms over variables with ranges, so the circuit
//! (variables are wires) and the checkpoint (variables are residual channels) share
//! it. At most one variable of a one-hot group is 1 at a time.

use std::collections::BTreeMap;

/// No one-hot group.
pub const NO_GROUP: u32 = u32::MAX;

/// Ranges and one-hot groups of the variables a form reads.
pub trait Vars {
    fn range(&self, v: usize) -> (i64, i64);
    fn group(&self, v: usize) -> u32;
}

/// An integer form `Σ w·var + c`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Form {
    pub terms: Vec<(usize, i64)>,
    pub c: i64,
}

/// An MLP unit. Its activation is `relu(form)` (in units of the step scale K), and its
/// down weight is `down` (in units of the residual's quantum).
#[derive(Clone, Debug)]
pub struct Unit {
    pub form: Form,
    pub down: i64,
}

/// The largest sums of a form's positive and of its negative terms, the constant left
/// out, with at most one variable of each one-hot group at 1.
pub fn term_masses(f: &Form, vars: &impl Vars) -> (i64, i64) {
    let (mut pos, mut neg) = (0, 0);
    let mut by_group: BTreeMap<u32, (i64, i64)> = BTreeMap::new();
    for &(v, w) in &f.terms {
        match vars.group(v) {
            NO_GROUP => {
                let (a, b) = vars.range(v);
                pos += (w * a).max(w * b).max(0);
                neg += (-w * a).max(-w * b).max(0);
            }
            g => {
                let e = by_group.entry(g).or_insert((0, 0));
                *e = (e.0.max(w), e.1.max(-w));
            }
        }
    }
    for (p, n) in by_group.into_values() {
        pos += p;
        neg += n;
    }
    (pos, neg)
}

/// The largest magnitude a partial sum of the form reaches, its constant included.
pub fn gate_mass(f: &Form, vars: &impl Vars) -> i64 {
    let (pos, neg) = term_masses(f, vars);
    (pos + f.c.max(0)).max(neg + (-f.c).max(0))
}

/// The form's range, with at most one variable of each one-hot group at 1.
pub fn bounds(f: &Form, vars: &impl Vars) -> (i64, i64) {
    let (mut lo, mut hi) = (f.c, f.c);
    let mut by_group: BTreeMap<u32, (i64, i64)> = BTreeMap::new();
    for &(v, w) in &f.terms {
        match vars.group(v) {
            NO_GROUP => {
                let (a, b) = vars.range(v);
                lo += (w * a).min(w * b);
                hi += (w * a).max(w * b);
            }
            g => {
                let e = by_group.entry(g).or_insert((0, 0));
                *e = (e.0.min(w), e.1.max(w));
            }
        }
    }
    for (a, b) in by_group.into_values() {
        lo += a;
        hi += b;
    }
    (lo, hi)
}

/// A condition every input that makes a form positive meets.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Req {
    /// The variable holds this value.
    Is(usize, i64),
    /// One of these variables of the group is 1.
    OneOf(u32, Vec<usize>),
}

/// What `form ≥ 1` needs: each 0/1 variable or group whose unfavourable value alone
/// costs more than the form's headroom above 1.
fn requirements(f: &Form, vars: &impl Vars) -> Vec<Req> {
    let slack = bounds(f, vars).1 - 1;
    let mut reqs = Vec::new();
    let mut by_group: BTreeMap<u32, Vec<(usize, i64)>> = BTreeMap::new();
    for &(v, w) in &f.terms {
        match vars.group(v) {
            NO_GROUP => {
                let (a, b) = vars.range(v);
                if b - a == 1 && w.abs() > slack {
                    reqs.push(Req::Is(v, if w > 0 { b } else { a }));
                }
            }
            g => by_group.entry(g).or_default().push((v, w)),
        }
    }
    for (g, ws) in by_group {
        let m = ws.iter().map(|&(_, w)| w).max().unwrap().max(0);
        // A member with weight w loses m − w; no member (or one outside the form) loses m.
        if m > slack {
            let mut ok: Vec<usize> = ws.iter().filter(|&&(_, w)| m - w <= slack).map(|&(v, _)| v).collect();
            ok.sort_unstable();
            reqs.push(Req::OneOf(g, ok));
        }
    }
    reqs
}

fn conflict(a: &[Req], b: &[Req]) -> bool {
    a.iter().any(|x| {
        b.iter().any(|y| match (x, y) {
            (Req::Is(u, p), Req::Is(v, q)) => u == v && p != q,
            (Req::OneOf(g, s), Req::OneOf(h, t)) => g == h && !s.iter().any(|v| t.binary_search(v).is_ok()),
            _ => false,
        })
    })
}

/// The largest magnitude a partial sum of one down row reaches, over its units. Units
/// over one form (the same terms, constants apart) are summed at each end of the
/// form's range, where their convex sums peak. Forms whose requirements keep them from
/// being positive at once share one bound. A greedy cover groups them into such
/// cliques, and each clique counts its heaviest member.
pub fn down_mass(units: &[Unit], vars: &impl Vars) -> i64 {
    let mut forms: Vec<(&[(usize, i64)], Vec<&Unit>)> = Vec::new();
    for u in units {
        match forms.iter_mut().find(|(t, _)| *t == &u.form.terms[..]) {
            Some((_, v)) => v.push(u),
            None => forms.push((&u.form.terms, vec![u])),
        }
    }
    let mut masses: Vec<(i64, i64, Vec<Req>)> = forms
        .into_iter()
        .map(|(terms, us)| {
            let (lo, hi) = bounds(&Form { terms: terms.to_vec(), c: 0 }, vars);
            let (mut p, mut n) = (0, 0);
            for s in [lo, hi] {
                let (mut ps, mut ns) = (0, 0);
                for u in &us {
                    let x = u.down * (s + u.form.c).max(0);
                    ps += x.max(0);
                    ns += (-x).max(0);
                }
                p = p.max(ps);
                n = n.max(ns);
            }
            // The weakest of the form's units needs least.
            let c = us.iter().map(|u| u.form.c).max().unwrap();
            (p, n, requirements(&Form { terms: terms.to_vec(), c }, vars))
        })
        .collect();
    masses.sort_by_key(|m| std::cmp::Reverse(m.0.max(m.1)));
    let mut cliques: Vec<(i64, i64, Vec<usize>)> = Vec::new();
    for (i, m) in masses.iter().enumerate() {
        match cliques.iter_mut().find(|c| c.2.iter().all(|&j| conflict(&m.2, &masses[j].2))) {
            Some(c) => {
                c.0 = c.0.max(m.0);
                c.1 = c.1.max(m.1);
                c.2.push(i);
            }
            None => cliques.push((m.0, m.1, vec![i])),
        }
    }
    let (p, n) = cliques.iter().fold((0, 0), |(p, n), c| (p + c.0, n + c.1));
    p.max(n)
}
