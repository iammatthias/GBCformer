//! The integer form of the MLPs and the virtual channels they read and write: the
//! tables `Marks::layer` indexes without bounds checks, built only by `IntLayer::new`.

use super::ONE;
use crate::llama::{CONSTANTS, K, UNIT};
use crate::llama::{Config, Layer, Model};

/// Bit of a gate entry's unit index marking a fused step pair (see `IntLayer::cap`).
pub(super) const PAIR: u16 = 0x8000;

/// Sparse integer matrix grouped by one index: per index, its (other, weight) pairs.
#[derive(Clone)]
pub(super) struct Cols {
    start: Vec<u32>,
    /// Per index: its first entry, its first entry built with the `PAIR` bit (they sort
    /// last), and its end.
    spans: Vec<[u32; 3]>,
    e: Vec<(u16, i32)>,
}

impl Cols {
    pub(super) fn from_pairs(n: usize, mut pairs: Vec<(usize, u16, i32)>) -> Cols {
        pairs.sort_unstable_by_key(|p| (p.0, p.1));
        let mut start = vec![0u32; n + 1];
        for p in &pairs {
            start[p.0 + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let e: Vec<(u16, i32)> = pairs.into_iter().map(|(_, o, w)| (o, w)).collect();
        let spans = (0..n).map(|i| [start[i], start[i] + e[start[i] as usize..start[i + 1] as usize].partition_point(|&(o, _)| o & PAIR == 0) as u32, start[i + 1]]).collect();
        Cols { start, spans, e }
    }
    #[inline]
    pub(super) fn get(&self, i: usize) -> &[(u16, i32)] {
        &self.e[self.start[i] as usize..self.start[i + 1] as usize]
    }
    /// Index `i`'s entries built without the `PAIR` bit, then those built with it.
    #[inline(always)]
    pub(super) fn split(&self, i: usize) -> (&[(u16, i32)], &[(u16, i32)]) {
        let [a, f, b] = self.spans[i].map(|x| x as usize);
        // SAFETY: `from_pairs`, the only constructor, makes each span ascending and
        // within `e`.
        unsafe { (self.e.get_unchecked(a..f), self.e.get_unchecked(f..b)) }
    }

    /// The table with `tag(unit, pair)` beside each entry, in the same order, `pair`
    /// telling whether it sits in its index's fused-pair part (`split`).
    pub(super) fn tagged<T: Copy>(&self, tag: impl Fn(u16, bool) -> T) -> Tagged<T> {
        let mut e = Vec::with_capacity(self.e.len());
        for &[a, f, b] in &self.spans {
            e.extend(self.e[a as usize..b as usize].iter().zip(a..).map(|(&(u, w), k)| (u, w, tag(u, k >= f))));
        }
        let units = e.iter().map(|x: &(u16, i32, T)| x.0 as usize + 1).max().unwrap_or(0);
        Tagged { start: self.start.clone(), e, units }
    }
}

/// A `Cols` whose entries carry a payload, so a loop over an index's entries reads it
/// in sequence with them instead of by unit.
#[derive(Clone)]
pub(super) struct Tagged<T> {
    start: Vec<u32>,
    e: Vec<(u16, i32, T)>,
    /// One past the largest unit an entry names.
    units: usize,
}

impl<T> Default for Tagged<T> {
    /// No index.
    fn default() -> Tagged<T> {
        Tagged { start: vec![0], e: Vec::new(), units: 0 }
    }
}

impl<T> Tagged<T> {
    /// Index `i`'s entries.
    ///
    /// # Safety
    ///
    /// `i` is below `indices()`.
    #[inline(always)]
    pub(super) unsafe fn get_unchecked(&self, i: usize) -> &[(u16, i32, T)] {
        // SAFETY: the caller's bound on `i`. `Cols::tagged`, the only constructor, copies
        // `start`, which `from_pairs` made ascending and ending at the entry count, and
        // every entry.
        unsafe {
            let (a, b) = (*self.start.get_unchecked(i) as usize, *self.start.get_unchecked(i + 1) as usize);
            self.e.get_unchecked(a..b)
        }
    }

    /// The indices the table holds entries for.
    pub(super) fn indices(&self) -> usize {
        self.start.len() - 1
    }

    /// One past the largest unit an entry names.
    pub(super) fn units(&self) -> usize {
        self.units
    }

    pub(super) fn entries(&self) -> impl Iterator<Item = &(u16, i32, T)> {
        self.e.iter()
    }
}

/// One MLP in integer form: gate = 256·(bias + Σ w·x), act = max(gate, 0)/256,
/// residual += Σ c·act.
#[derive(Clone)]
pub(super) struct IntLayer {
    /// Gate weights by input channel.
    gate: Cols,
    /// Gate weights by unit.
    rows: Cols,
    bias: Vec<i32>,
    /// Each unit's one output channel and down weight (every unit writes one wire).
    out: Vec<u16>,
    c: Vec<i32>,
    /// Each unit's largest activation: 1 for a fused step pair, otherwise unbounded.
    /// A step `relu(K(s-θ+1)) - relu(K(s-θ))` is two units with the same gate row,
    /// biases one apart and opposite down weights; in integers their difference is
    /// exactly clamp(g, 0, 1) of the first unit's gate, so the pair is one unit.
    cap: Vec<i32>,
    /// The gate table again, as blocks of consecutive units (`Blocks`).
    blocks: Blocks,
}

/// Units a block's update covers at once: one NEON register of i32 lanes.
pub(super) const LANES: usize = 4;

/// One channel's run of readers as a block: the gates of the units from `unit` on,
/// `LANES` to a vector, with their weights at `w` (0 for a unit inside the run that does
/// not read the channel, whose gate then does not move).
#[derive(Clone, Copy)]
pub(super) struct Block {
    pub(super) unit: u32,
    pub(super) w: u32,
    pub(super) vectors: u32,
}

/// Per channel, where its blocks and its lone entries (readers no block holds) start.
#[derive(Clone, Copy, Default)]
pub(super) struct Head {
    block: u32,
    /// Where its blocks' vectors start in `Blocks::vectors`.
    vector: u32,
    lone: u32,
    /// The lone entries of fused pairs, after the plain ones.
    pair: u32,
    /// The most sends the channel's change can stage, one per block lane and lone entry.
    pub(super) lanes: u32,
    /// The number of gate updates the channel's change makes, one per entry.
    pub(super) entries: u32,
}

/// The gate table by channel, laid out for a vector update. A channel's readers that
/// sit close together (`IntLayer::packed_order` puts units with the same inputs side by
/// side) form blocks of consecutive units, each updated `LANES` gates at a time; the
/// rest are lone entries, one at a time, plain units before fused pairs. Every gate
/// entry is in exactly one block lane or lone entry, with its weight
/// (`IntLayer::with_blocks` checks it).
#[derive(Clone, Default)]
pub(super) struct Blocks {
    /// One per channel, and one past the last.
    head: Vec<Head>,
    blocks: Vec<Block>,
    /// Every block's vectors in order, each as (first unit, first weight): a channel's
    /// vectors are one run, which one loop walks whatever its blocks.
    vectors: Vec<(u32, u32)>,
    w: Vec<i32>,
    lone: Vec<(u16, i32)>,
    /// Per unit and `LANES - 1` past the last: all ones for a fused pair, else 0.
    pair: Vec<u32>,
    /// Per channel with exactly one reader in the layer, that gate entry as one word: the
    /// unit in bits 0..16, 1 in bit 16 for a fused pair, the weight in the high half;
    /// `NOT_SINGLE` for every other channel. The template routes such a channel's changes
    /// to the layer's single bucket (`Template::single_bucket`), which it walks without the
    /// block table. These are about a third of a text pass's (channel, layer) pairs.
    single: Vec<u64>,
    /// Per channel whose readers in the layer are exactly one block and no lone entry,
    /// that block's index in the low half and the channel's gate entries in the high;
    /// `NOT_SINGLE` for every other channel. The template routes such a channel's changes
    /// to the layer's block bucket (`Template::block_bucket`).
    one_block: Vec<u64>,
    /// The most sends any one channel's change stages (the largest `Head::lanes`).
    max_lanes: u32,
}

/// `Blocks::single` of a channel with no reader or more than one in the layer.
pub(super) const NOT_SINGLE: u64 = u64::MAX;

impl Blocks {
    /// Readers this close (in units) share a block, the units between with weight 0.
    const GAP: u32 = LANES as u32;

    fn new(gate: &Cols, cap: &[i32]) -> Blocks {
        let n = cap.len();
        let channels = gate.start.len() - 1;
        let mut t = Blocks { head: Vec::with_capacity(channels + 1), pair: (0..n + LANES - 1).map(|u| if u < n && cap[u] == 1 { !0 } else { 0 }).collect(), ..Blocks::default() };
        for ch in 0..channels {
            let mut e: Vec<(u16, i32)> = gate.get(ch).to_vec();
            t.single.push(match e[..] { [(u, w)] => u as u64 | ((cap[u as usize] == 1) as u64) << 16 | (w as u32 as u64) << 32, _ => NOT_SINGLE });
            e.sort_unstable_by_key(|x| x.0);
            let mut head = Head { block: t.blocks.len() as u32, vector: t.vectors.len() as u32, lone: 0, pair: 0, lanes: 0, entries: e.len() as u32 };
            let mut lone = Vec::new();
            let mut i = 0;
            while i < e.len() {
                let mut j = i + 1;
                while j < e.len() && (e[j].0 - e[j - 1].0) as u32 <= Self::GAP {
                    j += 1;
                }
                let (first, last) = (e[i].0 as u32, e[j - 1].0 as u32);
                let vectors = (last - first + 1).div_ceil(LANES as u32);
                // A vector costs about what two lone updates do, plus about one a block.
                if 2 * vectors + 1 < 3 * (j - i) as u32 {
                    let base = t.w.len();
                    t.blocks.push(Block { unit: first, w: base as u32, vectors });
                    t.vectors.extend((0..vectors).map(|v| (first + v * LANES as u32, base as u32 + v * LANES as u32)));
                    t.w.resize(base + vectors as usize * LANES, 0);
                    for &(u, w) in &e[i..j] {
                        t.w[base + (u as u32 - first) as usize] = w;
                    }
                    head.lanes += vectors * LANES as u32;
                } else {
                    lone.extend_from_slice(&e[i..j]);
                }
                i = j;
            }
            lone.sort_unstable_by_key(|&(u, _)| (cap[u as usize] == 1, u));
            head.lone = t.lone.len() as u32;
            head.pair = head.lone + lone.iter().filter(|&&(u, _)| cap[u as usize] != 1).count() as u32;
            head.lanes += lone.len() as u32;
            let blocks_here = t.blocks.len() as u32 - head.block;
            t.one_block.push(if blocks_here == 1 && lone.is_empty() { head.block as u64 | (head.entries as u64) << 32 } else { NOT_SINGLE });
            t.lone.extend(lone);
            t.head.push(head);
        }
        t.head.push(Head { block: t.blocks.len() as u32, vector: t.vectors.len() as u32, lone: t.lone.len() as u32, pair: t.lone.len() as u32, lanes: 0, entries: 0 });
        t.max_lanes = t.head.iter().map(|h| h.lanes).max().unwrap_or(0);
        t
    }

    /// Channel `ch`'s blocks, lone plain entries, lone pair entries, and its head.
    #[inline(always)]
    pub(super) fn get(&self, ch: usize) -> (&[Block], &[(u16, i32)], &[(u16, i32)], Head) {
        let (h, next) = (self.head[ch], self.head[ch + 1]);
        // SAFETY: `new`, the only constructor, pushes heads in ascending order, the last
        // one at the lengths of `blocks` and `lone`, each `pair` between its `lone`
        // and the next head's.
        unsafe { (self.blocks.get_unchecked(h.block as usize..next.block as usize), self.lone.get_unchecked(h.lone as usize..h.pair as usize), self.lone.get_unchecked(h.pair as usize..next.lone as usize), h) }
    }

    /// Channel `ch`'s block vectors (first unit, first weight), its lone plain entries,
    /// its lone pair entries, and its head.
    #[inline(always)]
    pub(super) fn flat(&self, ch: usize) -> (&[(u32, u32)], &[(u16, i32)], &[(u16, i32)], Head) {
        let (h, next) = (self.head[ch], self.head[ch + 1]);
        // SAFETY: as in `get`; `vector` ascends with `block`, and the last head holds the
        // length of `vectors`.
        unsafe { (self.vectors.get_unchecked(h.vector as usize..next.vector as usize), self.lone.get_unchecked(h.lone as usize..h.pair as usize), self.lone.get_unchecked(h.pair as usize..next.lone as usize), h) }
    }

    /// `flat` without the bounds check on `ch`.
    ///
    /// # Safety
    ///
    /// `ch` is below the channel count.
    #[inline(always)]
    pub(super) unsafe fn flat_unchecked(&self, ch: usize) -> (&[(u32, u32)], &[(u16, i32)], &[(u16, i32)], Head) {
        // SAFETY: `head` holds one entry per channel and one past the last; the rest as in `flat`.
        unsafe {
            let (h, next) = (*self.head.get_unchecked(ch), *self.head.get_unchecked(ch + 1));
            (self.vectors.get_unchecked(h.vector as usize..next.vector as usize), self.lone.get_unchecked(h.lone as usize..h.pair as usize), self.lone.get_unchecked(h.pair as usize..next.lone as usize), h)
        }
    }

    /// The most sends any one channel's change stages.
    #[inline(always)]
    pub(super) fn max_lanes(&self) -> u32 {
        self.max_lanes
    }

    #[inline(always)]
    pub(super) fn weights(&self) -> &[i32] {
        &self.w
    }

    #[inline(always)]
    pub(super) fn pair(&self) -> &[u32] {
        &self.pair
    }

    #[inline(always)]
    pub(super) fn single(&self) -> &[u64] {
        &self.single
    }

    #[inline(always)]
    pub(super) fn one_block(&self) -> &[u64] {
        &self.one_block
    }

    #[inline(always)]
    pub(super) fn block_list(&self) -> &[Block] {
        &self.blocks
    }
}

impl IntLayer {
    /// The one constructor, from each unit's gate terms `rows` as (unit, channel,
    /// weight) over `channels` channels. It builds both gate tables, a fused pair's
    /// (cap 1) entries sorted after the plain ones so the hot loop needs no cap load, and
    /// checks what `Marks::layer` indexes without bounds checks: every unit a gate
    /// entry names is below `bias.len()`, the per-unit arrays are that long, and every
    /// unit's output channel is below `channels`.
    pub(super) fn new(channels: usize, rows: Vec<(usize, u16, i32)>, bias: Vec<i32>, out: Vec<u16>, c: Vec<i32>, cap: Vec<i32>) -> IntLayer {
        let n = bias.len();
        assert!(out.len() == n && c.len() == n && cap.len() == n);
        assert!(n <= PAIR as usize, "{n} units; a gate entry names up to {}", PAIR);
        assert!(rows.iter().all(|&(u, _, _)| u < n), "a gate entry names a unit outside the layer");
        assert!(out.iter().all(|&o| (o as usize) < channels), "a unit writes a channel outside the layer");
        let gate = rows.iter().map(|&(u, ch, w)| (ch as usize, if cap[u] == 1 { u as u16 | PAIR } else { u as u16 }, w)).collect();
        let mut gate = Cols::from_pairs(channels, gate);
        // The flag sorted the pairs last in each span; `split` now tells them apart, so
        // the entries carry the bare unit and the loop needs no mask.
        for e in &mut gate.e {
            e.0 &= !PAIR;
        }
        IntLayer { gate, rows: Cols::from_pairs(n, rows), bias, out, c, cap, blocks: Blocks::default() }
    }

    /// The units that read channel `ch`.
    pub(super) fn readers_of(&self, ch: usize) -> impl Iterator<Item = u16> + '_ {
        self.gate.get(ch).iter().map(|&(u, _)| u)
    }

    /// The gates before any input, in the form `Marks::layer` keeps them: g − 1, so
    /// that a unit's send test is one sign test (see `Marks::layer`).
    pub(super) fn start(&self) -> Vec<i32> {
        self.bias.iter().map(|&b| b - 1).collect()
    }

    /// `start` with `LANES - 1` gates past the last unit, which a block's last vector
    /// may cover (with weight 0) and which no unit owns. `Marks::blocks` walks these gates.
    pub(super) fn start_padded(&self) -> Vec<i32> {
        let mut g = self.start();
        g.resize(g.len() + LANES - 1, -1);
        g
    }

    /// The block table (`Blocks`), empty until `with_blocks` builds it.
    #[inline]
    pub(super) fn blocks(&self) -> &Blocks {
        &self.blocks
    }

    /// Whether `with_blocks` built the block table.
    pub(super) fn has_blocks(&self) -> bool {
        self.blocks.head.len() == self.channels() + 1
    }

    /// The layer with its block table built and checked.
    pub(super) fn with_blocks(mut self) -> IntLayer {
        let (gate, cap, n, channels) = (&self.gate, &self.cap, self.bias.len(), self.channels());
        let blocks = Blocks::new(gate, cap);
        // What the block walk indexes without bounds checks: every block's weights within
        // `w` and its lanes within the unit count padded by `LANES - 1`.
        assert_eq!(blocks.single.len(), channels);
        assert_eq!(blocks.one_block.len(), channels);
        assert!(blocks.blocks.iter().all(|b| (b.w + b.vectors * LANES as u32) as usize <= blocks.w.len() && (b.unit + b.vectors * LANES as u32) as usize <= n + LANES - 1));
        // Every gate entry is in exactly one block lane or lone entry, with its weight.
        for ch in 0..channels {
            let (bl, plain, pairs, h) = blocks.get(ch);
            // The flat walk's view: the blocks' vectors in order, the lone entries joined.
            let (vectors, fplain, fpairs, _) = blocks.flat(ch);
            let want: Vec<(u32, u32)> = bl.iter().flat_map(|b| (0..b.vectors).map(move |v| (b.unit + v * LANES as u32, b.w + v * LANES as u32))).collect();
            assert!(vectors == &want[..] && fplain == plain && fpairs == pairs, "channel {ch}: the flat table differs from the blocks");
            let mut got: Vec<(u16, i32)> = plain.iter().chain(pairs).copied().collect();
            for b in bl {
                for k in 0..b.vectors as usize * LANES {
                    let w = blocks.w[b.w as usize + k];
                    if w != 0 {
                        got.push(((b.unit as usize + k) as u16, w));
                    }
                }
            }
            got.sort_unstable();
            let mut want = gate.get(ch).to_vec();
            want.sort_unstable();
            assert!(got == want && h.entries as usize == want.len(), "channel {ch}: the blocks do not hold its gate entries");
            assert!(plain.iter().all(|&(u, _)| cap[u as usize] != 1) && pairs.iter().all(|&(u, _)| cap[u as usize] == 1));
            // A single entry is the channel's one gate entry, with its pair flag, which the
            // single walk indexes without bounds checks.
            let ob = blocks.one_block[ch];
            assert!(ob == NOT_SINGLE || (bl.len() == 1 && plain.is_empty() && pairs.is_empty() && ob as u32 == h.block && (ob >> 32) as u32 == h.entries));
            let s = blocks.single[ch];
            assert!(s == NOT_SINGLE || (want.len() == 1 && want[0] == (s as u16, (s >> 32) as i32) && (s >> 16 & 1 == 1) == (cap[s as u16 as usize] == 1)));
        }
        self.blocks = blocks;
        self
    }

    /// The number of channels the layer reads and writes.
    #[inline]
    pub(super) fn channels(&self) -> usize {
        self.gate.start.len() - 1
    }

    // Read-only views: the fields stay private, so nothing outside this module can
    // break what `new` checked.
    #[inline]
    pub(super) fn gate(&self) -> &Cols {
        &self.gate
    }
    #[inline]
    pub(super) fn rows(&self) -> &Cols {
        &self.rows
    }
    #[inline]
    pub(super) fn bias(&self) -> &[i32] {
        &self.bias
    }
    #[inline]
    pub(super) fn out(&self) -> &[u16] {
        &self.out
    }
    #[inline]
    pub(super) fn c(&self) -> &[i32] {
        &self.c
    }
    #[inline]
    pub(super) fn cap(&self) -> &[i32] {
        &self.cap
    }

    /// The layer in integer form with each unit's largest possible |gate|, or None if
    /// some weight is not of the integer shape described in the module docs. `largest`
    /// is each channel's largest |value| (1 for a 0/1 channel, 15 for a nibble).
    pub(super) fn from(l: &Layer, cfg: &Config, readable: &[bool], largest: &[i64]) -> Result<(IntLayer, Vec<i64>), String> {
        let (d, inter, nc) = (cfg.hidden, cfg.intermediate, cfg.n_const);
        let (mut gate, mut down, mut bias, mut max) = (Vec::new(), Vec::new(), vec![0i32; inter], vec![0i64; inter]);
        for u in 0..inter {
            let r = &l.gate[u * d..][..d];
            if r.iter().all(|&w| w == 0.0) {
                continue;
            }
            if l.up[u * d..][..d].iter().enumerate().any(|(c, &w)| if c == 0 { w * ONE != 1.0 } else { w != 0.0 }) {
                return Err(format!("unit {u}: up is not exactly 1"));
            }
            bias[u] = units_of(CONSTANTS.iter().zip(r).map(|(k, w)| k * w).sum(), K).ok_or(format!("unit {u}: bias is not a multiple of K"))?;
            max[u] = bias[u].unsigned_abs() as i64;
            for (c, &w) in r.iter().enumerate().skip(nc) {
                if w != 0.0 {
                    if c < nc || !readable[c] {
                        return Err(format!("unit {u} reads channel {c}"));
                    }
                    let m = units_of(w, K).ok_or(format!("unit {u}: gate weight on channel {c} is not a multiple of K"))?;
                    gate.push((c, u as u16, m));
                    max[u] += m.unsigned_abs() as i64 * largest[c];
                }
            }
            for ch in nc..d {
                let w = l.down[ch * inter + u];
                if w != 0.0 {
                    down.push((u, ch as u16, units_of(w * K / UNIT, 1.0).ok_or(format!("unit {u}: down weight to {ch} is not c·UNIT/K"))?));
                }
            }
            if l.down[..nc * inter].iter().skip(u).step_by(inter).any(|&w| w != 0.0) {
                return Err(format!("unit {u} writes a constant channel"));
            }
        }
        if max.iter().any(|&g| g >= 1 << 23) {
            return Err("a gate can reach 2^23 units".into());
        }
        // Down sums stay exact only while every channel's Σ |act·c| is below 2^24.
        let mut sum = vec![0i64; d];
        for &(u, ch, c) in &down {
            sum[ch as usize] += max[u] * c.unsigned_abs() as i64;
        }
        if sum.iter().any(|&s| s >= 1 << 24) {
            return Err("a down sum can reach 2^24".into());
        }
        let rows = Cols::from_pairs(inter, gate.iter().map(|&(c, u, w)| (u as usize, c as u16, w)).collect());
        let (mut out, mut c) = (vec![0u16; inter], vec![0i32; inter]);
        let mut seen = vec![false; inter];
        for &(u, ch, w) in &down {
            if std::mem::replace(&mut seen[u], true) {
                return Err(format!("unit {u} writes more than one channel"));
            }
            (out[u], c[u]) = (ch, w);
        }
        // Fuse each step pair into its first unit; the second unit is dropped.
        let mut cap = vec![i32::MAX; inter];
        let mut dropped = vec![false; inter];
        let mut u = 0;
        while u + 1 < inter {
            let pair = c[u] != 0 && c[u] == -c[u + 1] && out[u] == out[u + 1] && bias[u] == bias[u + 1] + 1 && rows.get(u) == rows.get(u + 1);
            if pair {
                cap[u] = 1;
                dropped[u + 1] = true;
                c[u + 1] = 0;
                u += 2;
            } else {
                u += 1;
            }
        }
        gate.retain(|&(_, u, _)| !dropped[u as usize]);
        if inter > PAIR as usize {
            return Err("more than 32768 units".into());
        }
        let rows = gate.iter().map(|&(c, u, w)| (u as usize, c as u16, w)).collect();
        Ok((IntLayer::new(d, rows, bias, out, c, cap), max))
    }

    pub(super) fn writes(&self) -> impl Iterator<Item = usize> + '_ {
        self.c.iter().zip(&self.out).filter(|(c, _)| **c != 0).map(|(_, &ch)| ch as usize)
    }

    /// The layer with the units `drop` marks emptied, so they read and write nothing.
    pub(super) fn without(&self, drop: &[bool], channels: usize) -> IntLayer {
        let keep = |u: usize| !drop[u];
        let rows = (0..self.bias.len()).filter(|&u| keep(u)).flat_map(|u| self.rows.get(u).iter().map(move |&(ch, w)| (u, ch, w))).collect();
        let c = (0..self.bias.len()).map(|u| if keep(u) { self.c[u] } else { 0 }).collect();
        IntLayer::new(channels, rows, self.bias.clone(), self.out.clone(), c, self.cap.clone())
    }

    /// The layer with unit `order[i]` as unit `i`, and every unit `order` leaves out
    /// dropped. Panics unless `order` names distinct units and holds every unit that
    /// writes (c ≠ 0). A unit that writes nothing sends nothing, so dropping it changes
    /// no channel.
    pub(super) fn permuted(&self, order: &[u16]) -> IntLayer {
        let n = self.bias.len();
        let mut seen = vec![false; n];
        for &u in order {
            assert!(!std::mem::replace(&mut seen[u as usize], true), "unit {u} named twice");
        }
        assert!((0..n).all(|u| seen[u] || self.c[u] == 0), "a unit that writes is left out");
        let pick = |v: &[i32]| order.iter().map(|&u| v[u as usize]).collect::<Vec<i32>>();
        let rows = order.iter().enumerate().flat_map(|(i, &u)| self.rows.get(u as usize).iter().map(move |&(ch, w)| (i, ch, w))).collect();
        IntLayer::new(self.channels(), rows, pick(&self.bias), order.iter().map(|&u| self.out[u as usize]).collect(), pick(&self.c), pick(&self.cap))
    }

    /// The units that write, ordered so that units reading the same channels sit
    /// together: by their read channels in ascending order, compared as sequences.
    /// Units with the same inputs (a decoder's one-hots) become one run of consecutive
    /// units under every channel they read.
    pub(super) fn packed_order(&self) -> Vec<u16> {
        let mut order: Vec<u16> = (0..self.bias.len() as u16).filter(|&u| self.c[u as usize] != 0).collect();
        order.sort_by(|&a, &b| self.rows.get(a as usize).iter().map(|e| e.0).cmp(self.rows.get(b as usize).iter().map(|e| e.0)).then(a.cmp(&b)));
        order
    }

    /// Whether unit `u` is a clearing unit: it reads the channel it writes.
    pub(super) fn clears(&self, u: usize) -> bool {
        self.c[u] != 0 && self.rows.get(u).iter().any(|&(ch, _)| ch == self.out[u])
    }

    /// The layer over virtual channels (`Segments`), read and written at `event`, with
    /// its clearing units left out: a virtual channel starts at zero, which is what the
    /// clearing makes of the dense channel.
    pub(super) fn remap(&self, seg: &Segments, event: usize) -> IntLayer {
        let inter = self.bias.len();
        let mut rows = Vec::new();
        let (mut out, mut c) = (self.out.clone(), self.c.clone());
        for u in 0..inter {
            if self.clears(u) {
                c[u] = 0;
                continue;
            }
            for &(ch, w) in self.rows.get(u) {
                rows.push((u, seg.read(ch as usize, event), w));
            }
            if c[u] != 0 {
                out[u] = seg.write(out[u] as usize, event);
            }
        }
        IntLayer::new(seg.count, rows, self.bias.clone(), out, c, self.cap.clone())
    }
}

/// The residual as the engine keeps it. A dense channel that circuit wires hold in
/// turn (a wire takes a dead wire's channel, and a clearing unit, which reads only that
/// channel, subtracts the dead value first) is one virtual channel per occupant: the
/// first keeps the dense index, the later ones count up from `hidden` (see `renumber`). Then every
/// virtual channel has one writer and the hot loop is as before. Events are numbered
/// attention of layer L = 2L, its MLP = 2L + 1; a reader at an event sees the occupant
/// written before it.
#[derive(Clone)]
pub(super) struct Segments {
    /// Per dense channel, the virtual id of its first occupant, and its occupants after
    /// the first: (event of the clearing that started it, virtual id).
    pub(super) first: Vec<u16>,
    pub(super) later: Vec<Vec<(usize, u16)>>,
    pub(super) count: usize,
    /// Per virtual channel, the values its clearing negates exactly: a value outside
    /// leaves the dense channel holding a remainder, so the fast pass refuses it.
    pub(super) range: Vec<(i32, i32)>,
}

impl Segments {
    /// `attn[L]`: the dense channels layer L's attention writes. `mlps`: every layer's
    /// MLP in dense integer form, the routing layer last.
    pub(super) fn new(d: usize, attn: &[Vec<usize>], mlps: &[&IntLayer]) -> Result<Segments, String> {
        let mut events: Vec<Vec<(usize, bool)>> = vec![Vec::new(); d];
        for (li, chans) in attn.iter().enumerate() {
            for &ch in chans {
                events[ch].push((2 * li, false));
            }
        }
        let mut clear_units: Vec<Vec<(usize, usize)>> = vec![Vec::new(); d];
        for (li, l) in mlps.iter().enumerate() {
            let mut seen: Vec<usize> = Vec::new();
            for u in 0..l.bias.len() {
                if l.c[u] == 0 {
                    continue;
                }
                let ch = l.out[u] as usize;
                if l.clears(u) {
                    if l.rows.get(u).len() != 1 {
                        return Err(format!("layer {li} unit {u} reads the channel it writes and another"));
                    }
                    clear_units[ch].push((li, u));
                } else {
                    seen.push(ch);
                }
            }
            seen.sort_unstable();
            seen.dedup();
            for ch in seen {
                events[ch].push((2 * li + 1, false));
            }
        }
        for (ch, us) in clear_units.iter().enumerate() {
            let mut layers: Vec<usize> = us.iter().map(|&(li, _)| li).collect();
            layers.dedup();
            for li in layers {
                events[ch].push((2 * li + 1, true));
            }
        }
        let mut seg = Segments { first: (0..d as u16).collect(), later: vec![Vec::new(); d], count: d, range: vec![(-1, 1); d] };
        for ch in 0..d {
            // A clearing sorts before a write at the same event: the write lands in the
            // new occupant.
            events[ch].sort_unstable_by_key(|&(e, clear)| (e, !clear));
            let (mut current, mut written) = (ch, false);
            for &(e, clear) in &events[ch] {
                if clear {
                    let l = mlps[e / 2];
                    let f = |x: i32| -> i32 { clear_units[ch].iter().filter(|&&(li, _)| li == e / 2).map(|&(_, u)| l.c[u] * (l.bias[u] + l.rows.get(u)[0].1 * x).clamp(0, l.cap[u])).sum() };
                    let exact: Vec<i32> = (-1..=1).filter(|&x| f(x) == -x).collect();
                    let (lo, hi) = (*exact.first().unwrap_or(&1), *exact.last().unwrap_or(&-1));
                    if !(lo <= 0 && 0 <= hi && exact.len() as i32 == hi - lo + 1) {
                        return Err(format!("the clearing of channel {ch} at layer {} is exact on {exact:?}, not on a range around 0", e / 2));
                    }
                    seg.range[current] = (lo, hi);
                    if seg.count > u16::MAX as usize {
                        return Err("more than 65536 virtual channels".into());
                    }
                    current = seg.count;
                    seg.count += 1;
                    seg.range.push((-1, 1));
                    seg.later[ch].push((e, current as u16));
                    written = false;
                } else {
                    if written {
                        return Err(format!("channel {ch} has more than one writer"));
                    }
                    written = true;
                }
            }
        }
        Ok(seg)
    }

    /// The occupant of dense channel `ch` a reader at `event` sees.
    pub(super) fn read(&self, ch: usize, event: usize) -> u16 {
        self.later[ch].iter().rev().find(|&&(e, _)| e < event).map_or(self.first[ch], |&(_, v)| v)
    }

    /// The occupant a writer at `event` writes.
    pub(super) fn write(&self, ch: usize, event: usize) -> u16 {
        self.read(ch, event + 1)
    }

    /// Renumbers occupant k of dense channel ch to `id(ch, k)` where it says so: the
    /// hot tables are indexed by virtual id, and wires that change together (a byte's
    /// bits) sit together only in creation order, not in the order channels free up.
    /// Ids must stay distinct; the ones left unused are empty rows.
    pub(super) fn renumber(&mut self, id: impl Fn(usize, usize) -> Option<usize>) -> Result<(), String> {
        let mut map: Vec<(u16, usize)> = Vec::new();
        for ch in 0..self.first.len() {
            if let Some(v) = id(ch, 0) {
                map.push((self.first[ch], v));
            }
            for (k, &(_, old)) in self.later[ch].iter().enumerate() {
                if let Some(v) = id(ch, k + 1) {
                    map.push((old, v));
                }
            }
        }
        let count = map.iter().map(|&(_, v)| v + 1).max().unwrap_or(0).max(self.count);
        if count > u16::MAX as usize + 1 {
            return Err("more than 65536 virtual channels".into());
        }
        let mut new: Vec<u16> = (0..self.count as u16).collect();
        let mut used = vec![false; count];
        for &(old, v) in &map {
            new[old as usize] = v as u16;
        }
        for &v in &new {
            if std::mem::replace(&mut used[v as usize], true) {
                return Err(format!("virtual channel {v} renumbered twice"));
            }
        }
        let mut range = vec![(-1, 1); count];
        for (old, &v) in new.iter().enumerate() {
            range[v as usize] = self.range[old];
        }
        for f in &mut self.first {
            *f = new[*f as usize];
        }
        for (_, v) in self.later.iter_mut().flatten() {
            *v = new[*v as usize];
        }
        (self.range, self.count) = (range, count);
        Ok(())
    }
}

/// An f32 value that is exactly `m·unit` for an integer `m` with |m| < 2^24.
pub(super) fn units_of(v: f32, unit: f32) -> Option<i32> {
    let m = v / unit;
    (m.fract() == 0.0 && m.abs() < 16777216.0).then_some(m as i32)
}

/// Per virtual channel, the interval its value lies in: a wire's declared range
/// (`Model::wire_ranges`, the circuit's `ranges`: a node's signal is 0/1), and 0/1 for
/// every other channel (token bits, the gathered address, kinds, the copy bits, which
/// attention fills with a convex combination of 0/1 values). The positional stage
/// leaves units out on the same declared ranges. Also, per routing unit, whether it is gated by a token channel: it reads one with a
/// positive weight (an output slot identity, or RD), so at a position whose token does
/// not carry it, it sees `field + 0 >= 2`, which is 0 because an output field bit is
/// 0/1. That is the contract every routing layer rests on already: it is what keeps
/// the other slots' units at 0 at each decoded position of a single checkpoint.
pub(super) fn declared_bounds(m: &Model, seg: &Segments, routing: &IntLayer, token: &[bool]) -> Result<(Vec<(i64, i64)>, Vec<bool>), String> {
    let lay = &m.layout;
    let mut b = vec![(0i64, 1i64); seg.count];
    for (k, occ) in lay.occupants.iter().enumerate() {
        let ch = lay.wire_base() + k;
        // An overlay stream's channel that only other parts' wires hold stays 0.
        if occ.is_empty() && seg.later[ch].is_empty() {
            continue;
        }
        if occ.len() != seg.later[ch].len() + 1 {
            return Err(format!("wire channel {ch} holds {} wires and {} virtual channels", occ.len(), seg.later[ch].len() + 1));
        }
        for (j, s) in occ.iter().enumerate() {
            let v = if j == 0 { seg.first[ch] } else { seg.later[ch][j - 1].1 };
            b[v as usize] = m.wire_ranges[s.0 as usize];
        }
    }
    let gated = (0..routing.bias.len()).map(|u| routing.c[u] == 0 || routing.rows.get(u).iter().any(|&(ch, w)| token[ch as usize] && w > 0)).collect();
    Ok((b, gated))
}

/// The RMSNorm argument over `bounds`: (Σ of each non-constant dense channel's
/// largest square over its occupants, route channels as `route_square`; the bound it
/// must stay below). The mean square of the residual is that sum · UNIT² / d, which
/// must stay below half an ulp of `EPS` (2^-10) for every norm to divide by exactly
/// 2^7. That puts the bound at 16·d.
pub(super) fn norm_bound(bounds: &[(i64, i64)], seg: &Segments, nc: usize, routes: &[usize], route_square: i64) -> (i64, i64) {
    let sum = (nc..seg.first.len())
        .map(|ch| {
            if routes.contains(&ch) {
                return route_square;
            }
            std::iter::once(seg.first[ch]).chain(seg.later[ch].iter().map(|&(_, v)| v)).map(|v| bounds[v as usize].0.abs().max(bounds[v as usize].1.abs()).pow(2)).max().unwrap_or(0)
        })
        .sum::<i64>()
        + CONSTANTS[..nc].iter().map(|&k| (k * k) as i64).sum::<i64>();
    let d = seg.first.len() as f64;
    let limit = (crate::llama::EPS as f64 * 2f64.powi(-24)) * d / (crate::llama::UNIT as f64).powi(2);
    (sum, limit as i64)
}
