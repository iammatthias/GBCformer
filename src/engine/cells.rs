//! The memory as the K/V rows a stock runtime's cache holds (`docs/stock-llama.md` §4,
//! `docs/harness.md`): one cell per 8-byte line, and per cell, per layer, the key and
//! value rows in f16 that llama.cpp keeps for the cell token of the line's write record.
//!
//! **The rows.** [`RowBuilder`] computes a cell's rows from the weights, as the runtime
//! does when it decodes the write record `REC WP0a..c WP1a..c WL0..WL7` at `P` with the
//! cell token `CELL` at `P + 1`, then moves the cell to the line's position `n` (the
//! K-shift re-ropes its key). The keys and values past layer 0 read only channels that
//! the embedding and layer 0's attention write. No MLP and no later attention writes
//! them (`RowBuilder::new` proves it from the weights). So a cell's rows are layer 0's
//! attention of the cell token over the record, its output projection onto those
//! channels, and each layer's `W_k` and `W_v`. The builder rounds every matmul input to
//! f16, as the runtime's half tiles hold it, and stores keys and values in f16.
//!
//! **A read.** A memory head's query is the lattice point of its address (`MemPlan`
//! proves the pattern from `W_q`): `±S` on the 21 address dims, `B` on the committed
//! dim. The fast path looks the line up by its key. It serves the cell's value only when
//! (a) every row in the cache has the canonical key of its own line, ±1 on the address
//! dims and 1 on the committed dim (checked when the row is written, so every other row
//! differs from the query's line in at least one address bit and scores at least
//! `2·S·scale = 256` lower, where f32 `exp` is exactly 0), (b) the row found holds the
//! canonical key of the line read, and (c) its value is 0 or 1 on every dim. Then the
//! softmax over every row weighs that row exactly 1 and the read is its value. The group
//! tokens' keys at the memory layers hold no committed bit and at most ±1 per address
//! dim, which `Params::check` bounds the same way. Otherwise the read is the full scan:
//! the softmax over every row, in f32. `GBCFORMER_SCAN=1` runs the scan on every read
//! as an oracle and stops at the first difference.

use crate::gguf::{f16_to_f32, f32_to_f16};
use crate::llama::{attn_scale, tok, Config, Model, CONSTANTS, HEAD_DIM, ROPE_THETA};
use crate::memory::attn::{rope, Params, COMMIT, EXP_ZERO, KEY_BITS, KEY_DIMS, READ_AT};
use crate::memory::dense::{Records, RECORD};
use crate::memory::layout::{LINE, LINES, LINE_BITS};
use super::rows::Store;
use std::sync::{Arc, LazyLock};

/// One head's key or value in f16 bits.
pub type Row = [u16; HEAD_DIM];

const NONE: u32 = u32::MAX;
/// Every this many memo hits, the rows are built again and compared (`Cells::put`).
const MEMO_CHECK: u64 = 64;

/// The memo holds at most `2^MEMO_BITS` records' rows (about 70 MB), and starts at
/// `2^MEMO_MIN_BITS` (about 1 MB) with the first write record after the insertion. It
/// grows fourfold each time it has missed as many records as it holds, so a stream
/// whose records rarely repeat (the PPU's) keeps it small.
const MEMO_BITS: u32 = 16;
/// The memo's entries per set. Text's records repeat: the first 800,000 kept (every
/// engine's) hold 45,518 distinct ones. With one entry per set the memo built 116,679
/// of 877,197 records' rows, with four 53,319.
const MEMO_WAYS: usize = 4;
const MEMO_MIN_BITS: u32 = 10;

/// A write record's tokens: its byte address and its line.
pub(super) type Key = (u32, [u8; LINE]);
/// One record's rows: a key and a value per group.
pub(super) type Pair = ([Row; MAX_GROUPS], [Row; MAX_GROUPS]);
pub(super) const NO_PAIR: Pair = ([[0; HEAD_DIM]; MAX_GROUPS], [[0; HEAD_DIM]; MAX_GROUPS]);

/// What a lookup needs of a record's rows (`Cells::serve`): per group whether the key is
/// the line's canonical key, whether every memory layer's is, and the line each value
/// row spells.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct Facts {
    regular: bool,
    canon: [bool; MAX_GROUPS],
    lines: [Option<[u8; LINE]>; MAX_GROUPS],
}

impl Facts {
    pub(super) const NONE: Facts = Facts { regular: false, canon: [false; MAX_GROUPS], lines: [None; MAX_GROUPS] };
}

/// The record's set among `len` memo entries (a power of two).
fn memo_set((addr, line): Key, len: usize) -> usize {
    let h = (addr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from_le_bytes(line).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (h ^ h >> 29) as usize & (len / MEMO_WAYS - 1)
}

/// The number of turns `Cells` caches.
const TURNS: usize = 4096;
/// Bounds on the row groups, the live layer-0 heads and the read channels, which
/// `RowBuilder::new` checks, so a cell's rows build in fixed arrays.
const MAX_GROUPS: usize = 4;
const MAX_LIVE: usize = 2;
const MAX_READ: usize = 128;
const ONE: u16 = 0x3C00;
const MINUS_ONE: u16 = 0xBC00;

fn half(x: &[f32; HEAD_DIM]) -> [f32; HEAD_DIM] {
    x.map(crate::llama::f16)
}

/// RoPE by a signed position offset, HF rotate-half, with that offset's `turn`.
fn rope_by(x: &mut [f32; HEAD_DIM], turn: &Turn) {
    let h = HEAD_DIM / 2;
    let y = *x;
    for (t, &(c, s)) in turn.iter().enumerate() {
        // A zero pair stays zero (up to the sign of a zero, `same`).
        if y[t] == 0.0 && y[t + h] == 0.0 {
            continue;
        }
        x[t] = y[t] * c - y[t + h] * s;
        x[t + h] = y[t + h] * c + y[t] * s;
    }
}

/// RoPE with a position's (cos, sin) table (`llama::rope_at`), as `memory::attn::rope` turns.
fn rope_with(x: &mut [f32; HEAD_DIM], (c, s): &([f32; HEAD_DIM], [f32; HEAD_DIM])) {
    let h = HEAD_DIM / 2;
    let y = *x;
    for t in 0..h {
        x[t] = y[t] * c[t] - y[t + h] * s[t];
        x[t + h] = y[t + h] * c[t + h] + y[t] * s[t + h];
    }
}

/// A sparse row: (input index, weight).
type Sparse = Vec<(u32, f32)>;

/// RoPE's (cos, sin) per pair for one position offset, formed in f64 as `llama::rope_at`
/// forms them.
pub type Turn = [(f32, f32); HEAD_DIM / 2];

/// The turn by `delta` positions.
pub fn turn(delta: i64) -> Turn {
    static DENOM: LazyLock<[f64; HEAD_DIM / 2]> = LazyLock::new(|| std::array::from_fn(|t| ROPE_THETA.powf(2.0 * t as f64 / HEAD_DIM as f64)));
    std::array::from_fn(|t| {
        let ang = delta as f64 / DENOM[t];
        (ang.cos() as f32, ang.sin() as f32)
    })
}

/// Computes a cell's rows from the weights.
///
/// Layers whose `W_k` and `W_v` rows over the read channels are equal weight for weight
/// read the same input, so they hold the same rows. `new` groups them, and a cell keeps
/// one row per group (layers 1 to 13 of the one model are one group).
pub struct RowBuilder {
    cfg: Config,
    slots: tok::RecordSlots,
    layers: usize,
    /// Per layer-0 head that writes a read channel, its score against every token's key
    /// at P (`q · k`, then times the attention scale, in f32 in dim order), and against
    /// the cell token's key at P + 1. The record's attention compares these scores.
    logit: Vec<Vec<f32>>,
    logit_cell: Vec<f32>,
    rope_pc: ([f32; HEAD_DIM], [f32; HEAD_DIM]),
    /// Per token, layer 0's value (one KV head), f16.
    v0: Vec<[f32; HEAD_DIM]>,
    /// Per layer-0 head that writes a read channel: its index and |q|·scale, bounding a
    /// cell's score with |k0(CELL)|.
    live: Vec<(usize, f32)>,
    cell_norm: f32,
    /// Per channel a key or value past layer 0 reads: its embedding value on the cell
    /// token and layer 0's `W_o` row over the live heads' outputs, each entry's index
    /// `HEAD_DIM · (live head) + dim`.
    embed_cell: Vec<f32>,
    o_rows: Vec<Sparse>,
    /// Per layer, its group; per group past the first (layer 0's), `W_k` and `W_v` per dim
    /// over `read`'s indices.
    group: Vec<usize>,
    kv: Vec<[Vec<Sparse>; 2]>,
    /// Layer 0's key and value of the cell token (they read only its embedding).
    cell_k0: [f32; HEAD_DIM],
    cell_v0: [f32; HEAD_DIM],
    /// The layers holding a memory head, whose keys must be canonical, and their groups.
    pub memory_layers: Vec<usize>,
    memory_groups: Vec<usize>,
    /// Per group, whether a memory layer is in it.
    memory_group: Vec<bool>,
    pub params: Params,
}

impl RowBuilder {
    /// The builder, after proving from the weights what it relies on.
    pub fn new(m: &Model) -> Result<RowBuilder, String> {
        let spec = m.memory.as_ref().ok_or("not a memory checkpoint")?;
        let c = &m.cfg;
        let (d, a, kvd, inter) = (c.hidden, c.attn(), c.kv_dim(), c.intermediate);
        if c.kv_heads != 1 {
            return Err(format!("{} KV heads; the rows are one KV head's", c.kv_heads));
        }
        let nc = CONSTANTS.len();
        let inv = 1.0 / crate::llama::UNIT;
        let l0 = &m.layers[0];
        // Layer 0's query reads only the constants, which every embedding holds alike.
        if (0..a).any(|r| l0.q[r * d..][..d].iter().skip(nc).any(|&w| w != 0.0)) {
            return Err("layer 0's query reads a channel past the constants".into());
        }
        if !(0..c.vocab).all(|t| (0..nc).all(|i| m.embed[t * d + i] == m.embed[i])) {
            return Err("an embedding holds other constants".into());
        }
        let normed = |t: usize| -> Vec<(usize, f32)> { (0..d).filter_map(|ch| Some((ch, crate::llama::f16(m.embed[t * d + ch] * inv))).filter(|x| x.1 != 0.0)).collect() };
        let project = |w: &[f32], row: usize, h: &[(usize, f32)]| -> f32 { h.iter().fold(0f32, |s, &(ch, x)| s + w[row * d + ch] * x) };
        let cell = tok::cell(c);
        let h_cell = normed(cell);
        let q0: Vec<[f32; HEAD_DIM]> = (0..c.heads).map(|h| half(&std::array::from_fn(|i| project(&l0.q, h * HEAD_DIM + i, &h_cell)))).collect();
        let (mut k0, mut v0) = (Vec::with_capacity(c.vocab), Vec::with_capacity(c.vocab));
        for t in 0..c.vocab {
            let h = normed(t);
            k0.push(half(&std::array::from_fn(|i| project(&l0.k, i, &h))));
            v0.push(half(&std::array::from_fn(|i| project(&l0.v, i, &h))));
        }
        // What the keys and values past layer 0 read, against what MLPs and later
        // attention write.
        let mut written = vec![false; d];
        let mut reads = vec![false; d];
        for (l, layer) in m.layers.iter().enumerate() {
            for ch in 0..d {
                written[ch] |= (0..inter).any(|j| layer.down[ch * inter + j] != 0.0) || (l > 0 && (0..a).any(|j| layer.o[ch * a + j] != 0.0));
                reads[ch] |= l > 0 && (0..kvd).any(|r| layer.k[r * d + ch] != 0.0 || layer.v[r * d + ch] != 0.0);
            }
        }
        if let Some(ch) = (0..d).find(|&ch| reads[ch] && written[ch]) {
            return Err(format!("channel {ch}: a key or value past layer 0 reads it and an MLP or later attention writes it"));
        }
        let read: Vec<usize> = (0..d).filter(|&ch| reads[ch]).collect();
        let o_rows: Vec<Sparse> = read.iter().map(|&ch| (0..a).filter(|&j| l0.o[ch * a + j] != 0.0).map(|j| (j as u32, l0.o[ch * a + j])).collect()).collect();
        let live: Vec<(usize, f32)> = (0..c.heads)
            .filter(|&h| o_rows.iter().any(|r| r.iter().any(|&(j, _)| j as usize / HEAD_DIM == h)))
            .map(|h| (h, q0[h].iter().map(|x| x * x).sum::<f32>().sqrt() * attn_scale()))
            .collect();
        let kv: Vec<[Vec<Sparse>; 2]> = m
            .layers
            .iter()
            .map(|l| {
                let rows = |w: &[f32]| -> Vec<Sparse> { (0..HEAD_DIM).map(|r| read.iter().enumerate().filter(|&(_, &ch)| w[r * d + ch] != 0.0).map(|(i, &ch)| (i as u32, w[r * d + ch])).collect()).collect() };
                [rows(&l.k), rows(&l.v)]
            })
            .collect();
        let mut memory_layers: Vec<usize> = spec.heads.iter().map(|h| h.0).collect();
        memory_layers.sort_unstable();
        memory_layers.dedup();
        if memory_layers.contains(&0) {
            return Err("a memory head at layer 0".into());
        }
        // A token other than a cell holds no committed bit at a memory layer. Only the
        // cell token's embedding sets the committed channel, no layer-0 head writes it, and
        // the key's committed dim reads nothing else. So such a token scores at least
        // `B·scale` below the cell of the line its address names (`Params::check`, the
        // uncommitted margin), as long as its address entries are within ±1, which the
        // address gathers give and the runtime comparison checks.
        let committed = m.layout.committed();
        if (0..c.vocab).any(|t| (t == cell) != (m.embed[t * d + committed] != 0.0)) {
            return Err("a token other than the cell token sets the committed channel".into());
        }
        let ci = read.iter().position(|&ch| ch == committed).ok_or("no key reads the committed channel")?;
        if !o_rows[ci].is_empty() {
            return Err("layer 0's attention writes the committed channel".into());
        }
        if let Some(&l) = memory_layers.iter().find(|&&l| kv[l][0][COMMIT].iter().any(|&(i, _)| i as usize != ci)) {
            return Err(format!("layer {l}: the key's committed dim reads a channel besides the committed one"));
        }
        // Layers ≥ 1 with equal `W_k` and `W_v` rows over the read channels share a group;
        // layer 0 (its key and value read only the cell token's embedding) is group 0.
        let mut group = vec![0usize; c.layers];
        let mut reps: Vec<usize> = Vec::new();
        for l in 1..c.layers {
            let bits = |l: usize| kv[l].iter().flatten().map(|row| row.iter().map(|&(i, w)| (i, w.to_bits())).collect::<Vec<_>>()).collect::<Vec<_>>();
            group[l] = match reps.iter().position(|&r| bits(r) == bits(l)) {
                Some(g) => g + 1,
                None => {
                    reps.push(l);
                    reps.len()
                }
            };
        }
        let mut memory_groups: Vec<usize> = memory_layers.iter().map(|&l| group[l]).collect();
        memory_groups.sort_unstable();
        memory_groups.dedup();
        let memory_group = (0..=reps.len()).map(|g| memory_groups.contains(&g)).collect();
        if reps.len() + 1 > MAX_GROUPS || live.len() > MAX_LIVE || read.len() > MAX_READ {
            return Err(format!("{} row groups, {} live layer-0 heads, {} read channels: above the row builder's arrays", reps.len() + 1, live.len(), read.len()));
        }
        let kv: Vec<[Vec<Sparse>; 2]> = reps.iter().map(|&l| kv[l].clone()).collect();
        let roped = |k: &[f32; HEAD_DIM], p: usize| {
            let mut k = *k;
            rope(&mut k, p);
            half(&k)
        };
        // Layer 0's scores of the cell token's query against each token's key, in the
        // record's order of operations.
        let scale = attn_scale();
        let score = |q: &[f32; HEAD_DIM], k: &[f32; HEAD_DIM]| (0..HEAD_DIM).fold(0f32, |s, i| s + q[i] * k[i]) * scale;
        let q_cell: Vec<[f32; HEAD_DIM]> = live.iter().map(|&(h, _)| roped(&q0[h], READ_AT + 1)).collect();
        let k_cell = roped(&k0[cell], READ_AT + 1);
        let logit: Vec<Vec<f32>> = q_cell.iter().map(|q| k0.iter().map(|k| score(q, &roped(k, READ_AT))).collect()).collect();
        let logit_cell = q_cell.iter().map(|q| score(q, &k_cell)).collect();
        let slot = |j: u32| live.iter().position(|&(h, _)| h == j as usize / HEAD_DIM).expect("a live head") * HEAD_DIM + j as usize % HEAD_DIM;
        let o_rows = o_rows.iter().map(|r| r.iter().map(|&(j, w)| (slot(j) as u32, w)).collect()).collect();
        Ok(RowBuilder {
            cfg: c.clone(),
            logit,
            logit_cell,
            rope_pc: crate::llama::rope_at(READ_AT + 1),
            slots: spec.slots,
            layers: c.layers,
            cell_norm: k0[cell].iter().map(|x| x * x).sum::<f32>().sqrt(),
            cell_k0: k0[cell],
            cell_v0: v0[cell],
            embed_cell: read.iter().map(|&ch| m.embed[cell * d + ch]).collect(),
            v0,
            live,
            o_rows,
            group,
            kv,
            memory_layers,
            memory_groups,
            memory_group,
            params: spec.params,
        })
    }

    /// The number of row groups.
    pub fn groups(&self) -> usize {
        self.kv.len() + 1
    }

    /// Layer `l`'s row group.
    pub fn group(&self, l: usize) -> usize {
        self.group[l]
    }

    /// `rows`, one per group into `ks` and `vs`, with `turn` the turn by the line's index
    /// less P + 1.
    pub fn group_rows(&self, addr: u32, line: [u8; LINE], turn: &Turn, ks: &mut [Row], vs: &mut [Row]) -> Result<(), String> {
        let toks: [usize; RECORD] = Records { cfg: &self.cfg, slots: &self.slots }.write_tokens(addr, line);
        let cell_at = Records::CELL;
        // Layer 0's attention of the cell token over its record, with the scratch tokens
        // at P and the cell token itself at P + 1 (`logit`, `logit_cell`).
        let mut attn = [[0f32; HEAD_DIM]; MAX_LIVE];
        for (a, &(h, qn)) in self.live.iter().enumerate() {
            let logits: [f32; RECORD] = std::array::from_fn(|j| if j == cell_at { self.logit_cell[a] } else { self.logit[a][toks[j]] });
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Every cell already in the cache holds the cell token's layer-0 key, turned,
            // so its score is at most |q|·|k|.
            if m - qn * self.cell_norm < EXP_ZERO {
                return Err(format!("layer 0 head {h}: the record's best score {m} does not beat a cell's bound {} by {EXP_ZERO}", qn * self.cell_norm));
            }
            let w: [f32; RECORD] = logits.map(|l| (l - m).exp());
            let z: f32 = w.iter().sum();
            let out = &mut attn[a];
            for (j, &t) in toks.iter().enumerate() {
                if w[j] != 0.0 {
                    let wj = w[j] / z;
                    for i in 0..HEAD_DIM {
                        out[i] += wj * self.v0[t][i];
                    }
                }
            }
        }
        let attn = attn.map(|a| half(&a));
        let attn = attn.as_flattened();
        let inv = 1.0 / crate::llama::UNIT;
        let mut h = [0f32; MAX_READ];
        for ((x, row), &e) in h.iter_mut().zip(&self.o_rows).zip(&self.embed_cell) {
            *x = crate::llama::f16((e + row.iter().fold(0f32, |s, &(j, w)| s + w * attn[j as usize])) * inv);
        }
        assert!(ks.len() == self.groups() && vs.len() == self.groups());
        for g in 0..self.groups() {
            let (k, v): ([f32; HEAD_DIM], [f32; HEAD_DIM]) = if g == 0 {
                (self.cell_k0, self.cell_v0)
            } else {
                let [wk, wv] = &self.kv[g - 1];
                let dot = |row: &Sparse| row.iter().fold(0f32, |s, &(i, w)| s + w * h[i as usize]);
                (std::array::from_fn(|r| dot(&wk[r])), std::array::from_fn(|r| dot(&wv[r])))
            };
            // The runtime stores the key at P + 1, then shifts it to n, and each step
            // rounds to the f16 cache.
            let mut r = k;
            rope_with(&mut r, &self.rope_pc);
            let mut r = half(&r);
            rope_by(&mut r, turn);
            ks[g] = r.map(f32_to_f16);
            vs[g] = v.map(f32_to_f16);
        }
        Ok(())
    }

    /// Whether every memory layer's key among `ks` (one per group) is line `n`'s
    /// canonical key.
    fn canonical_at(&self, ks: &[Row], n: u32) -> bool {
        let want = canonical(n);
        self.memory_groups.iter().all(|&g| same(&ks[g], &want))
    }
}

/// The canonical key of line `n` in f16 bits: ±1 on the address dims, 1 on committed.
fn canonical(n: u32) -> Row {
    let mut k = [0u16; HEAD_DIM];
    for (i, &d) in KEY_DIMS.iter().enumerate() {
        k[d] = if n >> i & 1 == 1 { ONE } else { MINUS_ONE };
    }
    k[COMMIT] = ONE;
    k
}

/// Equal as f16 numbers: bit for bit, but +0 and -0 alike. Where a rotated key's
/// partner dim holds a vanishing product, llama.cpp's RoPE and this one may round it to
/// zeros of opposite sign, which no dot product or weighted sum can tell apart.
fn same(a: &Row, b: &Row) -> bool {
    a.iter().zip(b).all(|(&x, &y)| x == y || (x & 0x7FFF == 0 && y & 0x7FFF == 0))
}

/// A value row as a line, when every dim is exactly 0 or 1.
fn line_of(v: &Row) -> Option<[u8; LINE]> {
    let mut line = [0u8; LINE];
    for (j, &x) in v.iter().enumerate() {
        match x {
            ONE => line[j / 8] |= 1 << (j % 8),
            0 | 0x8000 => {}
            _ => return None,
        }
    }
    Some(line)
}

/// How a read was served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Served {
    /// The key lookup, its guard held.
    Lookup,
    /// The full softmax over every row (the guard failed).
    Scan,
}

/// The rows of the record `key`, built from the weights, and the facts a lookup reads.
pub(super) fn build_rows(builder: &RowBuilder, (addr, line): Key, turn: &Turn) -> (Pair, Facts) {
    let n = addr >> LINE_BITS;
    let groups = builder.groups();
    let (mut ks, mut vs) = NO_PAIR;
    builder.group_rows(addr, line, turn, &mut ks[..groups], &mut vs[..groups]).unwrap_or_else(|e| panic!("the rows of line {n:#07x}: {e}"));
    let want = canonical(n);
    let mut facts = Facts { regular: builder.canonical_at(&ks[..groups], n), ..Facts::NONE };
    for g in 0..groups {
        facts.canon[g] = same(&ks[g], &want);
        facts.lines[g] = line_of(&vs[g]);
    }
    ((ks, vs), facts)
}

/// The cache's cells: what a lookup reads of each (`Facts`, per row), the key index, the
/// memo of recent records by their tokens, and the rows (`rows::Store`), per layer group
/// (`RowBuilder::group`: a group's layers hold equal rows).
pub struct Cells {
    pub builder: Arc<RowBuilder>,
    store: Store,
    /// Per line, its row, or `NONE`.
    row_of: Vec<u32>,
    /// Per row, its line, and the facts of the rows it holds.
    line_of_row: Vec<u32>,
    facts: Vec<Facts>,
    /// The memo: per entry the record whose rows it holds and their facts, and per set of
    /// `MEMO_WAYS` entries the way found or filled last. `rows::Rows::memo` holds the rows.
    memo_keys: Vec<Key>,
    memo_facts: Vec<Facts>,
    recent: Vec<u8>,
    /// The count of records served from the memo, every `MEMO_CHECK`th of which is built
    /// again and compared.
    pub memo_hits: u64,
    pub memo_checked: u64,
    /// Records the memo missed since it last grew.
    memo_misses: u64,
    /// Rows whose key at a memory layer is not their line's canonical key. While any
    /// exists, every read is the full scan.
    irregular: usize,
    /// The last turns by a line's index less P + 1, by line modulo their count. A turn
    /// depends on the line alone.
    turns: Vec<(u32, Turn)>,
    /// Run the full scan beside every lookup and compare (`GBCFORMER_SCAN=1`).
    pub scan_all: bool,
}

impl Cells {
    pub fn new(builder: Arc<RowBuilder>) -> Cells {
        let turns = vec![(NONE, [(0.0, 0.0); HEAD_DIM / 2]); TURNS];
        Cells { store: Store::new(builder.clone()), builder, row_of: vec![NONE; LINES as usize], line_of_row: Vec::new(), facts: Vec::new(), memo_keys: Vec::new(), memo_facts: Vec::new(), recent: Vec::new(), memo_hits: 0, memo_checked: 0, memo_misses: 0, irregular: 0, turns, scan_all: std::env::var("GBCFORMER_SCAN").is_ok_and(|v| v == "1") }
    }

    /// Moves the rows behind a worker thread (`rows::Store::defer`): inference management,
    /// which changes no row. What a lookup reads stays on the calling thread.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn defer(&mut self) {
        self.store.defer();
    }

    /// Whether a worker thread writes the rows.
    pub fn deferred(&self) -> bool {
        self.store.deferred()
    }

    /// Waits until every record's rows are in their cells. A memo check that failed on
    /// the worker thread stops the run here.
    pub fn settle(&self) {
        self.store.settle();
    }

    pub fn len(&self) -> usize {
        self.line_of_row.len()
    }

    pub fn is_empty(&self) -> bool {
        self.line_of_row.is_empty()
    }

    /// The memo's entry count, 0 until the first write record after the insertion.
    pub fn memo_len(&self) -> usize {
        self.memo_keys.len()
    }

    /// The rows of the record of `addr` holding `line`, built from the weights, with
    /// their facts.
    fn build(&mut self, key: Key) -> (Pair, Facts) {
        let n = key.0 >> LINE_BITS;
        let slot = n as usize % TURNS;
        if self.turns[slot].0 != n {
            self.turns[slot] = (n, turn(n as i64 - (READ_AT + 1) as i64));
        }
        build_rows(&self.builder, key, &self.turns[slot].1)
    }

    /// Gives line `n`'s cell the facts of its new rows, and returns its row.
    fn index(&mut self, n: u32, facts: &Facts) -> u32 {
        let r = match self.row_of[n as usize] {
            NONE => {
                let r = self.line_of_row.len() as u32;
                self.line_of_row.push(n);
                self.facts.push(*facts);
                self.row_of[n as usize] = r;
                r
            }
            r => {
                self.irregular -= !self.facts[r as usize].regular as usize;
                self.facts[r as usize] = *facts;
                r
            }
        };
        self.irregular += !facts.regular as usize;
        r
    }

    /// Puts `line` in the cell of `addr`'s line at the insertion, building its rows from
    /// the weights without the memo (an inserted line is rarely written again as it was).
    pub fn fill(&mut self, addr: u32, line: [u8; LINE]) {
        let (pair, facts) = self.build((addr, line));
        let r = self.index(addr >> LINE_BITS, &facts);
        self.store.with(|rows| rows.place(r as usize, &pair));
    }

    /// Puts `line` in the cell of `addr`'s line. The rows come from the memo when it
    /// holds rows built for a record with the same tokens (the rows depend on the tokens
    /// alone), and from the weights otherwise. Every `MEMO_CHECK`th memo hit builds the
    /// rows again and stops the run if they or their facts differ. Returns whether it
    /// built the rows.
    pub fn put(&mut self, addr: u32, line: [u8; LINE]) -> bool {
        if self.memo_keys.is_empty() {
            self.resize(1 << MEMO_MIN_BITS);
        }
        let (key, n) = ((addr, line), addr >> LINE_BITS);
        if let Some(at) = self.find(key) {
            self.memo_hits += 1;
            let facts = self.memo_facts[at];
            let r = self.index(n, &facts);
            if self.memo_hits % MEMO_CHECK == 0 {
                self.memo_checked += 1;
                self.store.check(r, at as u32, key, &facts);
            } else {
                self.store.hit(r, at as u32);
            }
            return false;
        }
        let (pair, facts) = self.build(key);
        self.memo_misses += 1;
        if self.memo_misses >= self.memo_keys.len() as u64 && self.memo_keys.len() < 1 << MEMO_BITS {
            self.grow();
        }
        let at = self.victim(key);
        self.memo_keys[at] = key;
        self.memo_facts[at] = facts;
        let r = self.index(n, &facts);
        self.store.miss(r, at as u32, &pair);
        true
    }

    /// An empty memo of `len` entries.
    fn resize(&mut self, len: usize) {
        self.memo_keys = vec![(NONE, [0; LINE]); len];
        self.memo_facts = vec![Facts::NONE; len];
        self.recent = vec![0; len / MEMO_WAYS];
        self.store.clear_memo(len);
    }

    /// Grows the memo fourfold (to at most `2^MEMO_BITS` entries) and moves each entry to
    /// a way of its new set.
    fn grow(&mut self) {
        let len = (self.memo_keys.len() * 4).min(1 << MEMO_BITS);
        let (keys, facts) = (std::mem::take(&mut self.memo_keys), std::mem::take(&mut self.memo_facts));
        let pairs = self.store.with(|rows| std::mem::take(&mut rows.memo));
        self.resize(len);
        let mut moved = vec![NO_PAIR; len];
        for (i, &key) in keys.iter().enumerate().filter(|(_, k)| k.0 != NONE) {
            let at = self.victim(key);
            self.memo_keys[at] = key;
            self.memo_facts[at] = facts[i];
            moved[at] = pairs[i];
        }
        self.store.with(|rows| rows.memo = moved);
        self.memo_misses = 0;
    }

    /// The memo entry holding `key`, marked the most recent of its set.
    fn find(&mut self, key: Key) -> Option<usize> {
        let set = memo_set(key, self.memo_keys.len());
        let way = (0..MEMO_WAYS).find(|&w| self.memo_keys[set * MEMO_WAYS + w] == key)?;
        self.recent[set] = way as u8;
        Some(set * MEMO_WAYS + way)
    }

    /// The entry a new `key` replaces: the way after its set's most recent one.
    fn victim(&mut self, key: Key) -> usize {
        let set = memo_set(key, self.memo_keys.len());
        let way = (self.recent[set] as usize + 1) % MEMO_WAYS;
        self.recent[set] = way as u8;
        set * MEMO_WAYS + way
    }

    /// The line a memory head at `layer` reads for line `n`, and how it was served;
    /// None when the softmax over the rows does not give a line (no exact one-hot).
    pub fn read(&self, layer: usize, n: u32) -> Option<([u8; LINE], Served)> {
        let got = self.serve(layer, n);
        if self.scan_all && let Some((line, Served::Lookup)) = got {
            let scan = self.scan(layer, n);
            assert_eq!(scan, Some(line), "cell oracle: the lookup of line {n:#07x} at layer {layer} read {line:02x?}, the full scan {scan:02x?}");
        }
        got
    }

    /// `read` without the oracle.
    pub fn serve(&self, layer: usize, n: u32) -> Option<([u8; LINE], Served)> {
        let fast = || -> Option<[u8; LINE]> {
            if self.irregular > 0 {
                return None;
            }
            let r = *self.row_of.get(n as usize)?;
            let g = self.builder.group(layer);
            // The row's key was compared with line n's canonical key when it was written
            // (`Facts::canon`; at a memory layer `regular`, all true while `irregular` is 0).
            if r == NONE || !(self.builder.memory_group[g] || self.facts[r as usize].canon[g]) {
                return None;
            }
            self.facts[r as usize].lines[g]
        };
        match fast() {
            Some(line) => Some((line, Served::Lookup)),
            None => self.scan(layer, n).map(|l| (l, Served::Scan)),
        }
    }

    /// The memory head's softmax over every row at `layer`, its query the lattice point
    /// of line `n` at `READ_AT`, rounded to f16 as the runtime's tiles hold it.
    pub fn scan(&self, layer: usize, n: u32) -> Option<[u8; LINE]> {
        static F16: LazyLock<Vec<f32>> = LazyLock::new(|| (0..=u16::MAX).map(f16_to_f32).collect());
        let q = self.builder.params.query(n & ((1 << KEY_BITS) - 1), READ_AT).map(crate::llama::f16);
        let dims: Vec<(usize, f32)> = q.iter().copied().enumerate().filter(|x| x.1 != 0.0).collect();
        let scale = attn_scale();
        let g = self.builder.group(layer);
        let rows = self.store.rows();
        let keys = &rows.k[g];
        let logits: Vec<f32> = keys.iter().map(|k| dims.iter().fold(0f32, |s, &(d, x)| s + x * F16[k[d] as usize]) * scale).collect();
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let z: f32 = w.iter().sum();
        let mut out = [0f32; HEAD_DIM];
        for (r, &wr) in w.iter().enumerate() {
            if wr != 0.0 {
                let wr = wr / z;
                for (o, &x) in out.iter_mut().zip(&rows.v[g][r]) {
                    *o += wr * F16[x as usize];
                }
            }
        }
        let mut line = [0u8; LINE];
        for (j, &x) in out.iter().enumerate() {
            match x {
                1.0 => line[j / 8] |= 1 << (j % 8),
                0.0 => {}
                _ => return None,
            }
        }
        Some(line)
    }

    /// Every row at every layer, keys in llama.cpp's interleaved pair order (`gguf::interleave`),
    /// with each row's line. This is what a runtime's cache state holds.
    pub fn rows_by_line(&self) -> impl Iterator<Item = (u32, Vec<(Row, Row)>)> + '_ {
        let rows = self.store.rows();
        let hf_to_gg = |k: &Row| -> Row {
            let mut g = [0u16; HEAD_DIM];
            for j in 0..2 {
                for i in 0..HEAD_DIM / 2 {
                    g[2 * i + j] = k[j * HEAD_DIM / 2 + i];
                }
            }
            g
        };
        let b = &self.builder;
        let by_line: Vec<(u32, Vec<(Row, Row)>)> = self.line_of_row.iter().enumerate().map(|(r, &n)| (n, (0..b.layers).map(|l| (hf_to_gg(&rows.k[b.group(l)][r]), rows.v[b.group(l)][r])).collect())).collect();
        by_line.into_iter()
    }
}
