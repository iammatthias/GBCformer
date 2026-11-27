//! The memory as a Llama attention cache, one cell per 8-byte line of physical memory
//! (`docs/stock-llama.md` §4, §10).
//!
//! The cache holds exactly one token per line of the fixed layout, at the line's index
//! as its position; a write replaces the cell (the runtime evicts the old one). A read is
//! a query from a group at position `READ_AT`, above every cell, and its softmax puts weight
//! exactly 1 on the cell for the line and exactly 0 on every other token:
//!
//! - **Address.** The line's 21 address bits (byte address bits 3..24) as ±1, scaled by
//!   `S` in the query. A match scores `21·S`, a one-bit mismatch `19·S`. Bits 0..17 sit
//!   on the first halves of RoPE pairs 13..30, their partners zero in query and key;
//!   bits 18..20 on the second halves of pairs 29..31 ([`KEY_DIMS`]). Pair 13 turns by
//!   3.7·10⁻¹⁶ rad per position, so at every position up to `KEY_REACH` its cos is
//!   exactly 1.0 in f32 and its sin is below 2^-25: a partner's product rounds away
//!   from every ±1 and ±S entry, and a zero partner's rounds to zero in f16. A roped key
//!   entry is then exactly ±1 (±0 on an unused partner), a roped query entry exactly ±S
//!   (±3S), and every logit is an exact multiple of 512 in any summation order
//!   (`key_pairs_turn_by_nothing`).
//! - **Committed.** Every cell carries +1 in dim 31 (pair 31's first half), which the
//!   query scales by `B`. A token that holds the same address but is no cell (a
//!   record's own address and line tokens, a step's positions) loses by `B` to the cell.
//!
//! With `S = 512` and `B = 1,024` (pre-scale, [`Params::memory`]) every query and key
//! entry is f16-exact, every partial sum of a logit is a multiple of 512 of at most
//! `3·21·S + B = 33,280` (exact in f16 accumulation too), and both gaps are
//! `1,024 · 2^-3 = 128` logits, against the 104 where f32 `exp` is exactly 0 (17.4 in
//! f16). `Params::check` evaluates them in f32 and with keys and queries rounded to f16,
//! at the lines the layout can hold.
//!
//! The value is the line's 64 bits, byte k's bit i in dim `8k + i`. The circuit picks
//! the byte a read wants from them (`circuit::low_match`).

use super::layout::{LINE, LINES, LINE_BITS};
use super::{Addr, Memory};
use crate::llama::{attn_scale, f16, rope_at, HEAD_DIM};

pub const HEAD: usize = HEAD_DIM;
const HALF: usize = HEAD / 2;
/// The committed dim: pair 31's first half, the slowest pair.
pub const COMMIT: usize = 31;
/// Bits of a physical byte address.
pub const ADDR_BITS: usize = super::layout::ADDR_BITS;
/// Bits of a line address: the key's.
pub const KEY_BITS: usize = ADDR_BITS - LINE_BITS;
/// The first key pair. Pair t turns by 1e38^(-2t/64) rad a position; from pair 13 on,
/// `KEY_REACH` positions turn it by less than 2^-25 rad, so its cos is 1.0 in f32 and a
/// ±1 entry's partner, sin of the angle, rounds to zero in f16.
pub const KEY_PAIR0: usize = 13;
/// Line address bit i sits on the first half of pair `KEY_PAIR0 + i` up to pair 30 (pair
/// 31's first half is [`COMMIT`]), and the bits past those on the second halves of the
/// slowest pairs, 29..31, which turn by less than 10⁻²⁷ rad at `KEY_REACH`. Every other
/// partner stays zero in every query and key.
pub const KEY_DIMS: [usize; KEY_BITS] = {
    let mut d = [0; KEY_BITS];
    let first = COMMIT - KEY_PAIR0;
    let mut i = 0;
    while i < KEY_BITS {
        d[i] = if i < first { KEY_PAIR0 + i } else { HALF + COMMIT - (KEY_BITS - 1 - i) };
        i += 1;
    }
    d
};
/// The farthest a key and its query can sit apart while the key pairs turn by nothing.
/// It covers every byte position H3's byte cells could use (8 per line) and the read
/// position.
pub const KEY_REACH: usize = 8 * LINES as usize + 2;
/// `exp(x)` is exactly 0.0 in f32 for `x <= -104` (the smallest subnormal is e^-103.3).
pub const EXP_ZERO: f32 = 104.0;
/// The position of every read and of every record's scratch tokens, above every cell.
pub const READ_AT: usize = LINES as usize;

/// HF rotate_half RoPE of one head vector at position `p`, in f32 like `llama::apply_rope`.
pub fn rope(x: &mut [f32; HEAD], p: usize) {
    let (c, s) = rope_at(p);
    let y = *x;
    for t in 0..HALF {
        x[t] = y[t] * c[t] - y[t + HALF] * s[t];
        x[t + HALF] = y[t + HALF] * c[t + HALF] + y[t] * s[t + HALF];
    }
}

/// The query scales, pre-scale: `s` per address bit, `b` on the committed dim.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub s: f32,
    pub b: f32,
}

impl Params {
    /// The memory heads' scales: gaps of 128 logits, the largest power of two whose
    /// widest logit, a column select's 3S on every bit, stays within f16's range.
    pub fn memory() -> Params {
        Params::with_gap(128.0)
    }

    /// The smallest power-of-two scales whose gaps are at least `gap` logits.
    pub fn with_gap(gap: f32) -> Params {
        let mut s = 1.0f32;
        while 2.0 * s * Self::scale() < gap {
            s *= 2.0;
        }
        Params { s, b: 2.0 * s }
    }

    pub fn scale() -> f32 {
        attn_scale()
    }

    /// The query for line `n` at position `p`, roped, as the model's `W_q` produces it.
    pub fn query(&self, n: u32, p: usize) -> [f32; HEAD] {
        self.query_scaled(n, p, &[1.0; KEY_BITS])
    }

    /// `query` with bit i scaled by `m[i]`. A column select (`circuit::Query::gated`)
    /// asks with 3S on the bits its off line and its open line share.
    pub fn query_scaled(&self, n: u32, p: usize, m: &[f32; KEY_BITS]) -> [f32; HEAD] {
        let mut q = [0f32; HEAD];
        for (i, &d) in KEY_DIMS.iter().enumerate() {
            q[d] = m[i] * if n >> i & 1 == 1 { self.s } else { -self.s };
        }
        q[COMMIT] = self.b;
        rope(&mut q, p);
        q
    }

    /// The smallest logit gap a read at `READ_AT` relies on for the cell of line `cell`,
    /// at that position. The gaps are the cell over a one-bit mismatch at any position
    /// the layout holds, and the cell over an uncommitted token holding the line address
    /// at the query's own position. With `f16_kv`, the query and keys are rounded to f16
    /// first, as a half-tile runtime holds them.
    pub fn margin_at(&self, cell: usize, f16_kv: bool) -> f32 {
        let round = |mut v: [f32; HEAD]| {
            if f16_kv {
                v.iter_mut().for_each(|x| *x = f16(*x));
            }
            v
        };
        let n = cell as u32 & ((1 << KEY_BITS) - 1);
        let mut margin = f32::MAX;
        // The plain query, and the widest a column select asks, with 3S on every bit but
        // the mismatched one, whose gap is the smallest (2S).
        for narrow in std::iter::once(None).chain((0..KEY_BITS).map(Some)) {
            let m: [f32; KEY_BITS] = std::array::from_fn(|i| if narrow.is_none() || narrow == Some(i) { 1.0 } else { 3.0 });
            let q = round(self.query_scaled(n, READ_AT, &m));
            let logit = |k: [f32; HEAD]| (0..HEAD).fold(0f32, |acc, d| acc + q[d] * round(k)[d]) * Self::scale();
            let own = logit(key(Kind::Cell, n, cell));
            margin = margin.min(own - logit(key(Kind::Uncommitted, n, READ_AT)));
            for bit in narrow.map_or(0..KEY_BITS, |b| b..b + 1) {
                for at in [0, cell, READ_AT - 1] {
                    margin = margin.min(own - logit(key(Kind::Cell, n ^ (1 << bit), at)));
                }
            }
        }
        margin
    }

    /// Panics unless every margin is at least `EXP_ZERO` in f32 and with f16 keys, for
    /// cells at the first, a middle and the last line of the layout.
    pub fn check(&self) {
        assert_eq!((-EXP_ZERO).exp(), 0.0, "f32 exp must underflow to exactly 0 at -{EXP_ZERO}");
        for cell in [0, READ_AT / 2 + 1, READ_AT - 1] {
            for f16_kv in [false, true] {
                let m = self.margin_at(cell, f16_kv);
                assert!(m >= EXP_ZERO, "attention memory margin {m} for a cell at {cell} (f16 {f16_kv}) is below {EXP_ZERO}");
            }
        }
        assert!([self.s, 3.0 * self.s, self.b].iter().all(|&x| f16(x) == x), "query scales must be f16-exact");
        // The widest logit, a column select's (3S a bit), stays below f16's largest
        // finite value, so f16 sums of it are exact multiples of S.
        assert!(3.0 * self.s * KEY_BITS as f32 + self.b <= 65504.0, "a column select's logit passes f16's range");
        key_pairs_turn_by_nothing();
    }
}

/// Panics unless every key pair (the address bits' and the committed dim's) turns by
/// nothing within `KEY_REACH`. Its cos must be exactly 1.0 in f32 and its sin below
/// 2^-25, so the sin rounds to zero in f16, and a partner's product in f32,
/// `(q·sin)·(k·sin)` with `|q| ≤ 3S`, stays below 2^-38, which no sum holding a multiple
/// of 512 keeps. A pair that holds two dims of the key must leave each entry exactly
/// as it was: every roped query and key entry on a key dim is checked against its
/// unroped value, at both magnitudes a query bit takes. Every other dim is an unused
/// partner, which `memory_head` writes zero: a key's must round to zero in f16, and a
/// query's stays below `3S·2^-25`, so their product stays below 2^-38.
pub fn key_pairs_turn_by_nothing() {
    let p = Params::memory();
    for at in [1, READ_AT - 1, READ_AT, READ_AT + 1, KEY_REACH] {
        let (c, s) = rope_at(at);
        for &d in KEY_DIMS.iter().chain([&COMMIT]) {
            assert_eq!(c[d], 1.0, "key pair {} turns at position {at}: cos {}", d % HALF, c[d]);
            assert!(s[d].abs() < (-25f32).exp2() && f16(s[d]) == 0.0, "key pair {} turns at position {at}: sin {}", d % HALF, s[d]);
        }
        for n in [0, (1 << KEY_BITS) - 1, 0x0A_AAAA & ((1 << KEY_BITS) - 1), 0x15_5555 & ((1 << KEY_BITS) - 1)] {
            for m in [[1.0; KEY_BITS], [3.0; KEY_BITS]] {
                let mut q = [0f32; HEAD];
                for (i, &d) in KEY_DIMS.iter().enumerate() {
                    q[d] = m[i] * if n >> i & 1 == 1 { p.s } else { -p.s };
                }
                q[COMMIT] = p.b;
                let mut roped = q;
                rope(&mut roped, at);
                held(&roped, &q, 3.0 * p.s * (-25f32).exp2(), &format!("a query for line {n:#x} at position {at}"));
            }
            held(&key(Kind::Cell, n, at), &key(Kind::Cell, n, 0), 0.0, &format!("line {n:#x}'s key at position {at}"));
        }
    }
}

/// Panics unless `roped` equals `plain` exactly on every key dim and the committed dim,
/// and every other dim stays within `partner` (0: rounds to zero in f16).
fn held(roped: &[f32; HEAD], plain: &[f32; HEAD], partner: f32, what: &str) {
    for d in 0..HEAD {
        if KEY_DIMS.contains(&d) || d == COMMIT {
            assert_eq!(roped[d].to_bits(), plain[d].to_bits(), "{what}: dim {d} turns, {} against {}", roped[d], plain[d]);
        } else if partner == 0.0 {
            assert_eq!(f16(roped[d]).abs(), 0.0, "{what}: dim {d} holds {} after the turn", roped[d]);
        } else {
            assert!(roped[d].abs() < partner, "{what}: dim {d} holds {} after the turn", roped[d]);
        }
    }
}

/// What a token's key says beyond its line address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A memory cell: a line as loaded or last written.
    Cell,
    /// Not a memory token: a record's own address and line tokens, or a step position
    /// that holds the address. Appears in the margin check only.
    Uncommitted,
}

/// A token's key for line `n` at position `p`, roped as the cache stores it.
pub fn key(kind: Kind, n: u32, p: usize) -> [f32; HEAD] {
    let mut k = [0f32; HEAD];
    for (i, &d) in KEY_DIMS.iter().enumerate() {
        k[d] = if n >> i & 1 == 1 { 1.0 } else { -1.0 };
    }
    if kind == Kind::Cell {
        k[COMMIT] = 1.0;
    }
    rope(&mut k, p);
    k
}

/// A line's value vector: byte k's bit i in dim `8k + i`.
pub fn value(line: &[u8; LINE]) -> [f32; HEAD] {
    std::array::from_fn(|j| if j < 8 * LINE { (line[j / 8] >> (j % 8) & 1) as f32 } else { 0.0 })
}

/// The line a value vector spells.
pub fn line_of(v: &[f32; HEAD]) -> [u8; LINE] {
    std::array::from_fn(|k| (0..8).fold(0u8, |acc, i| acc | ((v[8 * k + i] == 1.0) as u8) << i))
}

/// The dense form, with one cell per line and a full softmax attention per read. A write
/// replaces its line's cell with the line after it.
pub struct AttnMemory {
    pub params: Params,
    rom_len: usize,
    keys: Vec<[f32; HEAD]>,
    lines: Vec<[u8; LINE]>,
}

impl AttnMemory {
    /// Loads a cartridge: one cell per ROM line, then `ram_len` zero bytes.
    pub fn load(params: Params, rom: &[u8], ram_len: usize) -> AttnMemory {
        assert!(rom.len() + ram_len <= 1 << ADDR_BITS, "{} bytes exceed the {ADDR_BITS}-bit address space", rom.len() + ram_len);
        assert!(rom.len().is_multiple_of(LINE), "the ROM ends mid-line");
        params.check();
        let n = (rom.len() + ram_len).div_ceil(LINE);
        let mut lines = vec![[0u8; LINE]; n];
        for (i, &b) in rom.iter().enumerate() {
            lines[i / LINE][i % LINE] = b;
        }
        AttnMemory { params, rom_len: rom.len(), keys: (0..n).map(|l| key(Kind::Cell, l as u32, l)).collect(), lines }
    }

    pub fn tokens(&self) -> usize {
        self.keys.len()
    }

    fn attend(&self, n: u32) -> [u8; LINE] {
        let q = self.params.query(n, READ_AT);
        let scale = Params::scale();
        let logits: Vec<f32> = self.keys.iter().map(|k| (0..HEAD).fold(0f32, |acc, d| acc + q[d] * k[d]) * scale).collect();
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let mut out = [0f32; HEAD];
        for (j, w) in e.iter().enumerate() {
            if *w != 0.0 {
                for (o, v) in out.iter_mut().zip(value(&self.lines[j])) {
                    *o += w / z * v;
                }
            }
        }
        assert!(out.iter().all(|&o| o == 0.0 || o == 1.0), "read of line {n:#x}: a value dim is not exactly 0 or 1");
        line_of(&out)
    }
}

impl Memory for AttnMemory {
    fn read(&mut self, addr: Addr) -> u8 {
        self.attend(addr >> LINE_BITS)[addr as usize % LINE]
    }

    /// The runtime's eviction and the record's cell. The line's one token then holds the
    /// line read back with `value` written over the byte.
    fn write(&mut self, addr: Addr, value: u8) {
        assert!(addr as usize >= self.rom_len, "write to ROM address {addr:#x}");
        let mut line = self.attend(addr >> LINE_BITS);
        line[addr as usize % LINE] = value;
        self.lines[(addr >> LINE_BITS) as usize] = line;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{random_stream, Bytes, Lockstep, Rng};

    #[test]
    fn margins_hold_in_f32_and_f16() {
        let p = Params::memory();
        assert_eq!((p.s, p.b), (512.0, 1024.0));
        p.check();
        assert_eq!(p.margin_at(READ_AT - 1, true), 128.0);
    }

    /// A line's cell scores bit for bit the same at any of eight positions (H3's byte
    /// cells of one line) and in any summation order, in f32 and with f16 q and k.
    #[test]
    fn a_cells_logit_is_one_number_in_any_order() {
        let p = Params::memory();
        key_pairs_turn_by_nothing();
        for n in [0u32, 1, 0x2A5A5, 0x15_A5A5, (READ_AT - 1) as u32] {
            for m in [[1.0; KEY_BITS], [3.0; KEY_BITS]] {
                let q = p.query_scaled(n, READ_AT, &m);
                let want = (0..KEY_BITS).map(|i| m[i] * p.s).sum::<f32>() + p.b;
                for at in (0..8).map(|k| 8 * n as usize % KEY_REACH + k).chain([n as usize]) {
                    let k = key(Kind::Cell, n, at);
                    for round in [false, true] {
                        let r = |x: f32| if round { f16(x) } else { x };
                        let terms: Vec<f32> = (0..HEAD).map(|d| r(q[d]) * r(k[d])).collect();
                        let forward = terms.iter().fold(0f32, |a, &t| a + t);
                        let backward = terms.iter().rev().fold(0f32, |a, &t| a + t);
                        let pairwise = terms.chunks(2).map(|c| c.iter().sum::<f32>()).collect::<Vec<_>>().chunks(4).map(|c| c.iter().sum::<f32>()).fold(0f32, |a, t| a + t);
                        for got in [forward, backward, pairwise] {
                            assert_eq!(got.to_bits(), want.to_bits(), "line {n:#x} at {at} (f16 {round}): {got} against {want}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn values_spell_their_line() {
        let line = [0x00, 0xFF, 0xA5, 0x5A, 0x01, 0x80, 0x3C, 0xC3];
        assert_eq!(line_of(&value(&line)), line);
        assert_eq!(value(&line)[8..16], [1.0; 8]);
        assert_eq!(value(&line)[16..24], [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn dense_reads_match_the_reference() {
        let p = Params::memory();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let rom: Vec<u8> = (0..4096).map(|_| rng.draw() as u8).collect();
        let ram_len = 512u32;
        let dense = AttnMemory::load(p, &rom, ram_len as usize);
        let mut bytes = rom.clone();
        bytes.resize(rom.len() + ram_len as usize, 0);
        let mut both = Lockstep::new(Bytes(bytes), dense);
        for a in random_stream(&mut rng, rom.len() as u32, ram_len, 20_000, 60) {
            match a.write {
                Some(v) => both.write(a.addr, v),
                None => {
                    both.read(a.addr);
                }
            }
        }
        assert_eq!(both.model.tokens(), (4096 + 512) / LINE);
    }
}
