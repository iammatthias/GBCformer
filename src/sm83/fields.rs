//! The two writes of a step that writes twice: whether write 1 lands in write 0's line.

use super::line::pattern;
use super::translate::{Cond, Physical};
use super::{G, W};
use crate::circuit::{low_match, one, Lin, Step, LINE, LINE_BITS};

/// The step's two writes when there are two: SP−1 then SP−2 (a push, call, restart or
/// interrupt dispatch, `stack`) or a16 then a16+1 (LD (a16),SP, `ld`), with the write
/// translations of the four addresses.
pub(super) struct Pair<'a> {
    pub(super) stack: Lin,
    pub(super) ld: Lin,
    pub(super) sp1: &'a Physical,
    pub(super) sp2: &'a Physical,
    pub(super) a16: &'a Physical,
    pub(super) a16p1: &'a Physical,
}

/// Per byte k of a line, 1 when write 1 lands in write 0's line and write 0 wrote byte k
/// of it, so write 1's record keeps write 0's byte there. Both must keep a record: each
/// write is translated under the bus state at its own M-cycle, so an access lock or OAM
/// DMA can drop one and not the other.
///
/// With u the upper of the pair's bus addresses (SP−1, or a16+1), both land in one line
/// exactly when both reach banking registers (every register a game writes sits in the
/// registers' first line), or neither does, u's low three bits
/// are not all zero and u is not FFFF. Without a register, u − 1 and u then share bus
/// bits 3..15, so they fall in one region (every region boundary but IE's sits on a
/// multiple of 8) whose translation passes the low bits through under the same bank
/// and mask, or both in one marker block. `sm83::tests::two_writes_leave_the_reference_lines`
/// holds the circuit to `layout::translate`.
pub(super) fn same_line(g: &mut G, pair: &Pair, sp: &W, a16: &W) -> Vec<Lin> {
    // u's low bits all zero, or u = FFFF: SP & 7 = 1 or SP = 0; a16 & 7 = 7 or a16 = FFFE.
    let mut ok = |name: &str, x: &W, low: usize, wrap: usize| {
        let bad = g.c.node(name, vec![Step { coef: 1, lin: pattern(&x[..3], low), theta: 3 }, Step { coef: 1, lin: pattern(x, wrap), theta: 16 }]);
        one() - bad
    };
    let (sp_ok, ld_ok) = (ok("WS.sp", sp, 1, 0x0000), ok("WS.a16", a16, 7, 0xFFFE));
    let not = |l: &Lin| one() - l.clone();
    (0..LINE)
        .map(|k| {
            let at = |p: &Physical, lits: &[Lin]| Cond { lin: lits.iter().fold(low_match(&p.bits[..LINE_BITS], k), |s, l| s + l.clone()), theta: (LINE_BITS + lits.len()) as i64 }.step();
            let steps = vec![
                at(pair.sp1, &[pair.stack.clone(), pair.sp1.record.clone(), pair.sp2.record.clone(), pair.sp1.reg.clone(), pair.sp2.reg.clone()]),
                at(pair.sp1, &[pair.stack.clone(), pair.sp1.record.clone(), pair.sp2.record.clone(), not(&pair.sp1.reg), not(&pair.sp2.reg), sp_ok.clone()]),
                at(pair.a16, &[pair.ld.clone(), pair.a16.record.clone(), pair.a16p1.record.clone(), pair.a16.reg.clone(), pair.a16p1.reg.clone()]),
                at(pair.a16, &[pair.ld.clone(), pair.a16.record.clone(), pair.a16p1.record.clone(), not(&pair.a16.reg), not(&pair.a16p1.reg), ld_ok.clone()]),
            ];
            g.c.node(&format!("WS.{k}"), steps)
        })
        .collect()
}

