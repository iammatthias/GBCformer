//! A byte of a memory line, picked by its word line, the one-hot of the access's low
//! address bits (a RAM's row decoder). A head reads the whole line; the pick costs no
//! level when the nodes that read the byte take one step per line byte, each gated by
//! that byte's word line, in place of one step over a byte selected a level earlier.
//! Each step reads one word line, so a change of the low bits, which moves two word
//! lines, reaches two steps per picked bit instead of all eight through the bits.

use super::W;
use crate::circuit::{Circuit, Lin, Line, Sig, Step, LINE, LINE_BITS};

#[derive(Clone)]
pub(super) struct Pick {
    /// Each candidate byte with its word line, 0/1; at most one word line is 1, and all
    /// are 0 only when the head reads VOID's zero line, whose every byte picks as zero.
    lanes: Vec<([Sig; 8], Lin)>,
}

impl Pick {
    pub(super) fn new(line: Line, words: Vec<Lin>) -> Pick {
        assert_eq!(words.len(), LINE, "one word line per line byte");
        Pick { lanes: line.into_iter().zip(words).collect() }
    }

    /// Candidate byte `k`.
    pub(super) fn byte(&self, k: usize) -> W {
        self.lanes[k].0.iter().map(|&s| Lin::from(s)).collect()
    }

    /// `coef·[f(byte) ≥ θ]` over the picked byte, `f` affine in the byte's bits: one
    /// step per line byte, `[w·word(k) + f(byte k) ≥ θ + w]`, with `w` above f's
    /// headroom so only the picked byte can fire.
    pub(super) fn steps(&self, c: &Circuit, coef: i64, f: impl Fn(&W) -> Lin, theta: i64) -> Vec<Step> {
        (0..self.lanes.len())
            .map(|k| {
                let fk = f(&self.byte(k));
                let w = (c.bounds(&fk).1 - theta + 1).max(1);
                Step { coef, lin: self.lanes[k].1.clone() * w + fk, theta: theta + w }
            })
            .collect()
    }

    /// The picked byte as eight select nodes, one level above the line.
    pub(super) fn bits(&self, c: &mut Circuit, name: &str) -> W {
        (0..8).map(|i| c.node(&format!("{name}[{i}]"), self.steps(c, 1, |b| b[i].clone(), 1))).collect()
    }
}

/// The word lines of a 3-bit index selected among `cases`: each case's 0/1 gates (all
/// must be 1; at most one case fires), its index bits and a constant added mod 8. Word
/// line j is one node with one step per case, `[gates + pattern(bits, j − add) ≥ n + 3]`;
/// with no case firing every word line is 0.
pub(super) fn words(c: &mut Circuit, name: &str, cases: &[(Vec<Lin>, &[Lin], usize)]) -> Vec<Lin> {
    (0..LINE)
        .map(|j| {
            let steps = cases
                .iter()
                .map(|(gates, low, add)| {
                    let at = (j + LINE - add % LINE) % LINE;
                    let lin = gates.iter().fold(pattern(&low[..LINE_BITS], at), |s, g| s + g.clone());
                    Step { coef: 1, lin, theta: (gates.len() + LINE_BITS) as i64 }
                })
                .collect();
            c.node(&format!("{name}.{j}"), steps)
        })
        .collect()
}

/// The byte read 0 delivers: its heads' bus (every column but one on VOID's zero line)
/// plus the host's late byte. The late byte is zero unless the host serves the read,
/// and then the bus is zero.
pub(super) struct Read {
    pub(super) bus: W,
    pub(super) late: W,
}

impl Read {
    /// The steps of `[gate ∧ bit i]` for a 0/1 `gate`, at most one of which fires.
    fn steps(&self, gate: &Lin, i: usize) -> Vec<Step> {
        [&self.bus, &self.late].into_iter().map(|w| Step { coef: 1, lin: gate.clone() + w[i].clone(), theta: 2 }).collect()
    }
}

/// A case's byte in `mux`: wires, or read 0's byte, whose bus and late byte the mux's
/// nodes read directly.
pub(super) enum Val<'a> {
    Wires(W),
    Read(&'a Read),
}

/// `Circuit::mux` over single-gate cases, one of which may take the read byte. Its bus
/// and late bits enter the select directly, so the mux sits one level above the heads
/// rather than above a select node. A gate of more than three terms becomes a selector
/// wire.
pub(super) fn mux(c: &mut Circuit, name: &str, cases: &[(Lin, Val)]) -> W {
    let gates: Vec<Lin> = cases.iter().map(|(g, _)| if g.terms.len() > 3 { c.selector(g) } else { g.clone() }).collect();
    (0..8)
        .map(|i| {
            let steps = gates
                .iter()
                .zip(cases)
                .flat_map(|(g, (_, v))| match v {
                    Val::Wires(w) => vec![Step { coef: 1, lin: g.clone() + w[i].clone(), theta: 2 }],
                    Val::Read(r) => r.steps(g, i),
                })
                .collect();
            c.node(&format!("{name}.{i}"), steps)
        })
        .collect()
}

/// `[v_i]` as a 0/1 literal over byte bit `b`: `b` when bit i of `v` is set, else `1 − b`.
pub(super) fn pattern(byte: &[Lin], v: usize) -> Lin {
    byte.iter().enumerate().fold(Lin::konst(0), |s, (i, b)| s + if v >> i & 1 == 1 { b.clone() } else { Lin::konst(1) - b.clone() })
}

#[cfg(test)]
mod tests {
    use super::words;
    use crate::circuit::{byte, one, Circuit, Lin};
    use std::collections::BTreeMap;

    /// The fetch word lines name byte (pc + k − hb) & 7, with and without the HALT bug,
    /// and a gated case with its gate off leaves every word line 0.
    #[test]
    fn word_lines_are_the_one_hot_of_the_index() {
        let mut c = Circuit::default();
        let pc = byte(c.input("PC"));
        let ctl = byte(c.input("CTL"));
        let (hb, gate) = (ctl[3].clone(), ctl[0].clone());
        for k in 0..3usize {
            let cases: Vec<(Vec<Lin>, &[Lin], usize)> = if k == 0 { vec![(vec![], &pc[..3], 0)] } else { vec![(vec![one() - hb.clone()], &pc[..3], k), (vec![hb.clone()], &pc[..3], k - 1)] };
            let w = words(&mut c, &format!("F{k}"), &cases);
            c.output(&format!("F{k}"), w.try_into().unwrap());
        }
        let w = words(&mut c, "G", &[(vec![gate], &pc[..3], 1)]);
        c.output("G", w.try_into().unwrap());
        for pc in 0..=255u8 {
            for ctl in [0u8, 1, 8, 9] {
                let out = c.eval(&BTreeMap::from([("PC".to_string(), pc), ("CTL".to_string(), ctl)]));
                let hb = (ctl >> 3 & 1) as usize;
                for k in 0..3usize {
                    let want = (pc as usize + k - if k == 0 { 0 } else { hb }) & 7;
                    assert_eq!(out[&format!("F{k}")], 1 << want, "pc {pc:#04x} k {k} hb {hb}");
                }
                assert_eq!(out["G"], if ctl & 1 == 1 { 1 << ((pc as usize + 1) & 7) } else { 0 }, "gated, pc {pc:#04x}");
            }
        }
    }
}
