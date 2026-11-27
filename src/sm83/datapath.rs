//! The 16-bit add with a constant offset that the fetch adders share, and the ADD HL opcodes.

use super::{G, W};
use crate::circuit::{cat, value, zero, Lin};

/// `x + k` mod 2^16 split as the fetch adders are: bits 0..2 and the carries out of
/// bits 2 and 6 one level above x, bits 3..6 and 7..15 a level later from those
/// carries. A one-level adder's bit j reads every bit below it, so a change in x's low
/// bits reached all sixteen staircases; now it reaches the low bits and the carries.
pub(super) fn add16_split(g: &mut G, name: &str, x: &[Lin], k: u64) -> W {
    let part = |n: usize, shift: usize| ((k >> shift) & ((1 << n) - 1)) as i64;
    let lo = g.c.sum_bits(&format!("{name}.lo"), 3, |n| value(x, n) + part(n, 0));
    let c3 = g.c.ge(&format!("{name}.c3"), value(x, 3) + part(3, 0), 8);
    let c7 = g.c.ge(&format!("{name}.c7"), value(x, 7) + part(7, 0), 128);
    let mid = g.c.sum_bits(&format!("{name}.mid"), 4, |n| value(&x[3..], n) + part(n, 3) + c3.clone());
    let hi = g.c.sum_bits(&format!("{name}.hi"), 9, |n| value(&x[7..], n) + part(n, 7) + c7.clone());
    cat(&cat(&lo, &mid), &hi)
}

/// The four ADD HL,rr opcodes.
pub(super) fn add_hl(g: &G) -> Lin {
    (0..4u8).fold(zero(), |s, k| s + g.op(0x09 + 16 * k))
}
