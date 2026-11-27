//! The CB rotates and shifts the nibble circuit takes as bit permutations.

use super::W;
use crate::circuit::{zero, Lin};

/// Rotate or shift `x` by CB op `y` (RLC, RRC, RL, RR, SLA, SRA, SWAP, SRL), with the
/// bit shifted out.
pub(super) fn rot(x: &[Lin], y: u8, carry: &Lin) -> (W, Lin) {
    let s = |k: usize| x[k].clone();
    match y {
        0 => ((0..8).map(|k| s((k + 7) % 8)).collect(), s(7)),
        1 => ((0..8).map(|k| s((k + 1) % 8)).collect(), s(0)),
        2 => ((0..8).map(|k| if k == 0 { carry.clone() } else { s(k - 1) }).collect(), s(7)),
        3 => ((0..8).map(|k| if k == 7 { carry.clone() } else { s(k + 1) }).collect(), s(0)),
        4 => ((0..8).map(|k| if k == 0 { zero() } else { s(k - 1) }).collect(), s(7)),
        5 => ((0..8).map(|k| s((k + 1).min(7))).collect(), s(0)),
        6 => ((0..8).map(|k| s((k + 4) % 8)).collect(), zero()),
        _ => ((0..8).map(|k| if k == 7 { zero() } else { s(k + 1) }).collect(), s(0)),
    }
}
