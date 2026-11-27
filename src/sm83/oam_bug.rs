//! The DMG's OAM corruption bug as circuit nodes: a GBC CPU access or idle M-cycle with an
//! FE page address on the bus while the PPU's OAM scan reads a row rewrites rows of OAM
//! (SameBoy v1.0.5 `memory.c`, `GB_trigger_oam_bug` and `GB_trigger_oam_bug_read`, the
//! DMG paths; the reference is `machine::ppu::corrupt_oam`).
//!
//! Every row the bug writes receives one new line: the first word computed from words of
//! the rows at and before the scanned row r, then bytes 2-7 of row r − 8. A write writes it
//! to row r. A read writes it to rows r − 8 and r, to r − 16 when r's group class is 0x10
//! or 0x00 (`r & 0x18`), to r − 32 when it is 0x00, and to row 0 when r is 0x80. Each bit
//! of the computed word is one threshold step per case over the source bits, so the block
//! is two levels deep (three for row 0x40's pattern).

use crate::circuit::{bits, one, value, Byte, Circuit, Lin, Query, Step, ADDR_BITS};
use crate::memory::layout;

/// One row's line: bit i of byte k is `[k][i]`.
pub type RowLine = [[Lin; 8]; 8];

/// What the bug writes: the new line, and the rows it lands on as bits of `targets`
/// (bits 0 to 3 in [`TARGET_ROWS_BEFORE`] order, then [`TARGET_FIRST`]: rows r, r − 8,
/// r − 16, r − 32 and 0).
pub struct Corruption {
    pub line: [Byte; 8],
    pub targets: [Lin; 5],
}

pub const TARGET_FIRST: usize = 4;
/// The row offsets the targets name, in rows of 8 bytes before the scanned row (row 0 is
/// absolute).
pub const TARGET_ROWS_BEFORE: [u8; 4] = [0, 1, 2, 4];

/// The event byte: the scanned row r (a multiple of 8, 8 to 0x98; 0 or 0xA0 for none) with
/// bit 0 set for a read. `rows` are the lines of rows r, r − 8, r − 16 and r − 32 (any line
/// where a row does not exist: no case reads it then).
pub fn corrupt(c: &mut Circuit, name: &str, ob: &Byte, rows: [&RowLine; 4]) -> Corruption {
    let read = ob[0].clone();
    let j: Vec<Lin> = ob[3..8].to_vec();
    let jv = value(&j, 5);
    // A case's selector: its 0/1 gates all 1 and the row index in [lo, hi], one node of two
    // steps (rows 1-19 are the scan's rows 8-0x98).
    const K: i64 = 32;
    let sel = |c: &mut Circuit, tag: &str, gates: Vec<Lin>, lo: i64, hi: i64| {
        let n = gates.len() as i64;
        let form = gates.into_iter().fold(jv.clone(), |s, g| s + g * K);
        c.node(&format!("{name}.{tag}"), vec![Step { coef: 1, lin: form.clone(), theta: n * K + lo }, Step { coef: -1, lin: form, theta: n * K + hi + 1 }])
    };
    let write = sel(c, "write", vec![one() - read.clone()], 1, 19);
    let odd = sel(c, "odd", vec![read.clone(), j[0].clone()], 1, 19);
    let c10 = sel(c, "c10", vec![read.clone(), one() - j[0].clone(), j[1].clone()], 1, 19);
    let at = |c: &mut Circuit, v: i64| sel(c, &format!("j{v}"), vec![read.clone()], v, v);
    let (j4, j8, j12, j16) = (at(c, 4), at(c, 8), at(c, 12), at(c, 16));

    let [r, r8, r16, r32] = rows;
    const M: i64 = 32;
    let word: Vec<Byte> = (0..2)
        .map(|b| {
            std::array::from_fn(|i| {
                let bit = |l: &RowLine, k: usize| l[k + b][i].clone();
                let (rw0, r8w0, r8w1, r8w2, r16w0, r16w1, r32w0) = (bit(r, 0), bit(r8, 0), bit(r8, 2), bit(r8, 4), bit(r16, 0), bit(r16, 2), bit(r32, 0));
                // Row 0x40's pattern reads (~d & f), d and f the second words of rows r − 8 and r − 16.
                let t = c.ge(&format!("{name}.q{b}.{i}"), r16w1.clone() - r8w1.clone(), 1);
                let cases: Vec<(&Lin, Lin, i64)> = vec![
                    // bitwise_glitch(a, b, c), a majority: a row r, b and c row r − 8's first and third words.
                    (&write, rw0.clone() + r8w0.clone() + r8w2.clone(), 2),
                    // bitwise_glitch_read: b | (a & c).
                    (&odd, r8w0.clone() * 2 + rw0.clone() + r8w2.clone(), 2),
                    // read_secondary(a = r − 16, b = r − 8, c = r, d = r − 8's third): (b & (a|c|d)) | (a&c&d).
                    (&c10, r8w0.clone() * 2 + r16w0.clone() + rw0.clone() + r8w2.clone(), 3),
                    // The tertiary reads, a = r, b = r − 8's third, c = r − 8, d = r − 16, e = r − 32.
                    // Row 0x20: (c & (a|b|d|e)) | (a&b&d&e).
                    (&j4, r8w0.clone() * 3 + rw0.clone() + r8w2.clone() + r16w0.clone() + r32w0.clone(), 4),
                    // Row 0x60: (c & (a|b|d|e)) | (b&d&e).
                    (&j12, r8w0.clone() * 5 + rw0.clone() + (r8w2.clone() + r16w0.clone() + r32w0.clone()) * 2, 6),
                    // Row 0x80 and the rest: c | (a&b&d&e).
                    (&j16, r8w0.clone() * 4 + rw0.clone() + r8w2.clone() + r16w0.clone() + r32w0.clone(), 4),
                    // Row 0x40, the DMG quaternary read: (e & (h|g|(~d&f)|c|b)) | (c&g&h), b = r,
                    // c = r − 8's third, d = r − 8's second, e = r − 8, f = r − 16's second,
                    // g = r − 16, h = r − 32.
                    (&j8, r8w0 * 8 + (r8w2 + r16w0 + r32w0) * 3 + rw0 + t, 9),
                ];
                let steps = cases.into_iter().map(|(s, form, theta)| Step { coef: 1, lin: s.clone() * M + form, theta: theta + M }).collect();
                c.node(&format!("{name}.w{b}.{i}"), steps)
            })
        })
        .collect();
    let line: [Byte; 8] = std::array::from_fn(|k| if k < 2 { word[k].clone() } else { r8[k].clone() });
    let c00 = j4.clone() + j8.clone() + j12.clone() + j16.clone();
    let reads = odd.clone() + c10.clone() + c00.clone();
    let targets = [write + reads.clone(), reads, c10 + c00.clone(), c00, j16];
    Corruption { line, targets }
}

/// The memory circuit's OAM pass, its outputs by name: the target rows as bits of `OT`
/// (bit 0, the scanned row, to [`TARGET_FIRST`]), each target's physical byte address
/// (`OW0a..c` to `OW4a..c`), and the one line they all receive (`OL0` to `OL7`). Only an
/// OAM pass decodes them; `OB` (`ob`) reaches nothing else.
///
/// Four gated whole-line heads read rows r, r − 8, r − 16 and r − 32 by OAM's physical
/// lines, `OAM + 8 (j − k)` for the event's row index j and k of 0, 1, 2 and 4; a row
/// outside OAM's 1-19 window (`k ≤ j ≤ 19`) reads VOID's zero line, so no query names a
/// line without a cell. The heads sit at `layer`, late, so their 64 bits each live for the
/// corruption's two levels alone.
pub fn memory_pass(c: &mut Circuit, ob: &Byte, layer: u32) -> Vec<(String, Byte)> {
    let j: Vec<Lin> = ob[3..8].to_vec();
    let jv = value(&j, 5);
    let line_bits = layout::OAM.trailing_zeros().min(8) as usize;
    assert!(line_bits >= 8, "OAM's rows sit within one 256-byte page");
    // Each target's row index as five bits: j − k mod 32, a one-level adder.
    let row = |c: &mut Circuit, k: u8| -> Vec<Lin> {
        if k == 0 {
            return j.clone();
        }
        c.sum_bits(&format!("OB.row{k}"), 5, |n| value(&j, n) + ((32 - k as i64) & ((1 << n) - 1)))
    };
    let address = |row: &[Lin]| -> Vec<Lin> {
        let mut a = bits(layout::OAM as u64, ADDR_BITS);
        a[3..8].clone_from_slice(row);
        a
    };
    let mut rows: Vec<Vec<Lin>> = Vec::new();
    let mut lines: Vec<RowLine> = Vec::new();
    for (t, &k) in TARGET_ROWS_BEFORE.iter().enumerate() {
        let r = row(c, k);
        let gate = c.node(&format!("OB.in{t}"), vec![Step { coef: 1, lin: jv.clone(), theta: k as i64 }, Step { coef: -1, lin: jv.clone(), theta: 20 }]);
        let line = c.head(&format!("OBH{t}"), layer, Query::gated(&address(&r), &gate, layout::VOID));
        lines.push(std::array::from_fn(|b| std::array::from_fn(|i| Lin::from(line[b][i]))));
        rows.push(r);
    }
    let out = corrupt(c, "OB", ob, [&lines[0], &lines[1], &lines[2], &lines[3]]);
    let mut named = Vec::new();
    let mut targets: Byte = std::array::from_fn(|_| Lin::konst(0));
    targets[..5].clone_from_slice(&out.targets);
    named.push(("OT".to_string(), targets));
    for t in 0..5 {
        let a = if t < 4 { address(&rows[t]) } else { bits(layout::OAM as u64, ADDR_BITS) };
        for (byte, suffix) in ["a", "b", "c"].iter().enumerate() {
            named.push((format!("OW{t}{suffix}"), std::array::from_fn(|i| a[8 * byte + i].clone())));
        }
    }
    for (k, l) in out.line.iter().enumerate() {
        named.push((format!("OL{k}"), l.clone()));
    }
    named
}

#[cfg(all(test, feature = "gbc"))]
mod tests {
    use super::*;
    use crate::circuit::byte;
    use crate::machine::ppu::{corrupt_oam, OamBug};
    use std::collections::BTreeMap;

    /// The block's line and targets, applied to OAM, equal `corrupt_oam` for every scanned
    /// row 0-0xA0, both kinds, over random OAM.
    #[test]
    fn the_block_rewrites_oam_as_the_reference_does() {
        let mut c = Circuit::default();
        let ob = byte(c.input("OB"));
        let rows: Vec<RowLine> = (0..4).map(|k| std::array::from_fn(|b| byte(c.input(&format!("R{k}_{b}"))))).collect();
        let out = corrupt(&mut c, "OB", &ob, [&rows[0], &rows[1], &rows[2], &rows[3]]);
        for (k, l) in out.line.iter().enumerate() {
            c.output(&format!("L{k}"), l.clone());
        }
        let mut t: Byte = std::array::from_fn(|_| Lin::konst(0));
        t[..5].clone_from_slice(&out.targets);
        c.output("T", t);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for j in 0..=20usize {
            for (kind, bit) in [(OamBug::Write, 0u8), (OamBug::Read, 1)] {
                for _ in 0..600 {
                    let oam: [u8; 0xA0] = std::array::from_fn(|_| next() as u8);
                    let row = 8 * j;
                    let mut want = oam;
                    corrupt_oam(&mut want, row, kind);
                    let mut inp = BTreeMap::from([("OB".to_string(), (row as u8) | bit)]);
                    for (k, back) in TARGET_ROWS_BEFORE.iter().enumerate() {
                        let at = row as isize - 8 * *back as isize;
                        for b in 0..8 {
                            inp.insert(format!("R{k}_{b}"), if (0..0xA0).contains(&at) { oam[at as usize + b] } else { 0 });
                        }
                    }
                    let got = c.eval(&inp);
                    let line: Vec<u8> = (0..8).map(|k| got[&format!("L{k}")]).collect();
                    let targets = got["T"];
                    let mut have = oam;
                    for (k, back) in TARGET_ROWS_BEFORE.iter().enumerate() {
                        if targets >> k & 1 != 0 {
                            let at = row - 8 * *back as usize;
                            have[at..at + 8].copy_from_slice(&line);
                        }
                    }
                    if targets >> TARGET_FIRST & 1 != 0 {
                        have[..8].copy_from_slice(&line);
                    }
                    assert_eq!(have, want, "row {row:#04x} {kind:?}: targets {targets:#07b}, line {line:02X?}");
                }
            }
        }
    }
}
