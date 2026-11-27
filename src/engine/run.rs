//! The per-pass hot loop: the integer state and marks, `fill_fast`, `advance`, the
//! memory heads, the positional stage's pass, and the decode. Everything a pass runs
//! through sits in this one module.

use super::heads::{Bus, Unit};
use super::int::{Block, IntLayer, Tagged, LANES};
use super::layer0::AttentionPlan;
use super::plan::{Demand, MemPlan, Owner, RouteTag, Template};
use super::pool::Pool;
use super::positional::{PosPart, PosState, Positional, ROUTE_SLOT};
use super::probe::Stage;
use super::stats::{Refusal, Stats};
use super::{Engine, Readers, NO_TOKEN, ROUTE_CAP};
use crate::llama::{tok, Config, Denoiser};
use crate::sequence::{Decoded, Pass, Positions, Sequence, Stamp};
use crate::memory::attn::KEY_BITS;
use super::plan::NO_CHANNEL;
use super::records::Memory;

/// ROM's lines.
const ROM_LINES: u32 = (crate::memory::layout::MAX_ROM >> crate::memory::layout::LINE_BITS) as u32;

/// The integer state after the last fast pass.
#[derive(Default)]
pub(super) struct State {
    pub(super) valid: bool,
    /// Wire values: the layer-0 input plus every circuit layer's writes.
    pub(super) x: Vec<i32>,
    pub(super) g: Vec<Vec<i32>>,
    /// The routing layer on the blank row (no token channels): gates and route totals.
    pub(super) route_g: Vec<i32>,
    pub(super) route: [i32; ROUTE_CAP],
    /// The inputs of the last pass: tokens, attention plan, attention output, and the
    /// decoded token per position.
    pub(super) tokens: Vec<usize>,
    pub(super) weights: AttentionPlan,
    /// The positions the attention plan was built for, by list and as a flag per
    /// position: its logits are checked equal at each, so it holds for any part of them.
    pub(super) plan_positions: Vec<usize>,
    /// The id of the positions the plan was built for (`Engine::positions_id`), or 0.
    pub(super) plan_id: u64,
    pub(super) plan_verified: Vec<bool>,
    pub(super) out: Vec<f32>,
    pub(super) got: Vec<usize>,
    /// The positions the last pass's decode ran at, unless the pass before it was `dense`.
    pub(super) fresh: Vec<u32>,
    /// The last pass decoded half its positions or more (the PPU's pixels decode nearly
    /// all, each pass): the next writes every position back and lists none here.
    pub(super) dense: bool,
    /// The positions an input of whose decode moved since they last decoded: their
    /// token (layer 0's moved positions), a routing unit their token owns
    /// (`Template::owners`), the blank-row route totals, or their positional route
    /// totals. Pushed by those producers, cleared by the decode; one spare bit past the
    /// sequence takes the owner table's fill.
    pub(super) stale: Bits,
    /// Per position (and the spare `seq`), what its tracked token's own routing units
    /// add to the blank-row route totals (`Template::tracked`, `Template::owners`).
    pub(super) own_route: Vec<[i32; ROUTE_CAP]>,
    /// The positions the last pass decoded, as a list and as bits.
    pub(super) want_positions: Vec<usize>,
    /// The id `want_positions` was last held equal to (`Engine::positions_id`), or 0.
    pub(super) want_id: u64,
    pub(super) want: Bits,
    /// One past the largest of `want_positions` (0 for none).
    pub(super) want_top: usize,
    /// The words of `want` that hold a position.
    pub(super) want_words: std::ops::Range<usize>,
    /// Per memory head, the line its output channels were last read at, and that
    /// line's bits packed as they are held (they start at 0, as every wire does).
    pub(super) head_addr: Vec<u32>,
    pub(super) head_out: Vec<u64>,
    /// The memory heads whose address may have moved this pass, as `MemPlan::words` u64s
    /// (`MemPlan::reach`): a column select whose gate changed, and of a changed unit's
    /// heads those that follow it and its first.
    pub(super) head_dirty: Vec<u64>,
    /// The heads whose line depends on their unit's forms as of their last read: a head
    /// without a gate, a column select whose gate was on or whose row was not all 0 or
    /// 1 (`Unit::column`). Any other column sits on its off line while its unit's forms
    /// stay 0 or 1.
    pub(super) head_follows: Vec<u64>,
    /// Per address unit, its forms as last formed, and whether a form changed since.
    pub(super) units: Vec<Unit>,
    pub(super) unit_stale: Vec<bool>,
    /// The memory heads a write record hit (one at `head_addr`) since the last pass, in
    /// the same words, and the image generation as of this engine's last write record:
    /// when the image's differs, something else changed it and every head is hit.
    pub(super) head_hit: Vec<u64>,
    pub(super) seen_gen: u64,
    /// Per address bit of every head (`MemPlan::bit_terms`), its form's value over the
    /// wires as of the last head-bucket drain.
    pub(super) head_bits: Vec<i32>,
    /// Per column head, the lane byte it drives; per bus, the byte its channels hold;
    /// the buses a layer's columns moved.
    pub(super) col_byte: Vec<u8>,
    pub(super) buses: Vec<Bus>,
    pub(super) bus_moved: Vec<u16>,
    /// This pass's lines read, by (layer, line): the columns of one row, and the off
    /// lines, read the cells once.
    pub(super) lines_read: Vec<(u32, u32, u64)>,
    /// The positional stage, when the plan has one.
    pub(super) pos: PosState,
}

/// A set of positions as one byte each (0 or 1), padded to whole 8-byte words: a
/// member is one byte store, and a walk reads eight positions per word.
#[derive(Default)]
pub(super) struct Bits(Vec<u8>);

impl Bits {
    fn empty(n: usize) -> Bits {
        Bits(vec![0; n.div_ceil(8) * 8])
    }
    fn full(n: usize) -> Bits {
        Bits(vec![1; n.div_ceil(8) * 8])
    }
    fn bits(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    fn set(&mut self, p: usize) {
        self.0[p] = 1;
    }
    fn set_all(&mut self) {
        self.0.fill(1);
    }
    /// Word `w`: bit 8·i holds position 8·w + i.
    #[inline(always)]
    fn word(&self, w: usize) -> u64 {
        u64::from_le_bytes(self.0[w * 8..][..8].try_into().unwrap())
    }
}

/// What the per-layer drain reads of a virtual channel, side by side: its range
/// (`Segments::range`) as its low end and width, and its reader bits.
#[derive(Clone, Copy)]
pub(super) struct Wire {
    pub(super) lo: i32,
    pub(super) width: u32,
    pub(super) readers: Readers,
}

impl Wire {
    pub(super) fn new((lo, hi): (i32, i32), readers: Readers) -> Wire {
        assert!(lo <= hi);
        Wire { lo, width: hi.abs_diff(lo), readers }
    }

    /// lo <= v <= hi, as one unsigned compare.
    #[inline(always)]
    fn holds(self, v: i32) -> bool {
        v.wrapping_sub(self.lo) as u32 <= self.width
    }
}

/// Pending input changes per reader bit (`Readers`, the positional stage's masks
/// too): the circuit layers, then the routing layer, the head bucket, the
/// positional stage and the circuit layers' single and block buckets. They share one
/// flat array, in which bucket `b` owns the cells from `start[b]` to `start[b + 1]`,
/// one per channel bit `b` reads. A pass changes a channel at most once and routes the
/// change to each of its readers once, so a bucket never fills. A push past its end is
/// dropped and flagged, and the pass is refused.
pub(super) struct Buckets {
    cells: Vec<(u16, i32)>,
    start: [u32; 65],
    end: [u32; 64],
    overflowed: bool,
}

impl Default for Buckets {
    /// No cells: every push overflows.
    fn default() -> Buckets {
        Buckets::new(std::iter::empty())
    }
}

impl Buckets {
    /// Buckets sized for channels with these reader masks.
    pub(super) fn new(readers: impl Iterator<Item = u64>) -> Buckets {
        let mut count = [0u32; 64];
        for mut mask in readers {
            while mask != 0 {
                count[mask.trailing_zeros() as usize] += 1;
                mask &= mask - 1;
            }
        }
        let mut start = [0u32; 65];
        for b in 0..64 {
            start[b + 1] = start[b] + count[b];
        }
        let end = std::array::from_fn(|b| start[b]);
        Buckets { cells: vec![(0, 0); start[64] as usize], start, end, overflowed: false }
    }

    /// Routes a change to the bucket of every reader bit in `mask`.
    #[inline(always)]
    pub(super) fn route(&mut self, mut mask: u64, ch: u16, dv: i32) {
        while mask != 0 {
            let b = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            // SAFETY: b < 64 (a set bit of a u64), and `new` made start[64] the cells'
            // length with start ascending, so a cell below start[b + 1] is in bounds.
            unsafe {
                let i = *self.end.get_unchecked(b);
                if i < *self.start.get_unchecked(b + 1) {
                    *self.cells.get_unchecked_mut(i as usize) = (ch, dv);
                    *self.end.get_unchecked_mut(b) = i + 1;
                } else {
                    self.overflowed = true;
                }
            }
        }
    }

    #[inline(always)]
    pub(super) fn get(&self, b: usize) -> &[(u16, i32)] {
        let (from, to) = (self.start[b] as usize, self.end[b] as usize);
        // SAFETY: `new` made the starts ascend to the cells' length, and `end[b]` stays
        // from `start[b]` to `start[b + 1]`: `route` stops it there, and `clear` and
        // `clear_all` put it back at `start[b]`.
        unsafe { self.cells.get_unchecked(from..to) }
    }

    pub(super) fn is_empty(&self, b: usize) -> bool {
        self.end[b] == self.start[b]
    }

    pub(super) fn clear(&mut self, b: usize) {
        self.end[b] = self.start[b];
    }

    pub(super) fn clear_all(&mut self) {
        self.end = std::array::from_fn(|b| self.start[b]);
        self.overflowed = false;
    }

    /// Whether a push was dropped since the last `clear_all`.
    pub(super) fn overflowed(&self) -> bool {
        self.overflowed
    }
}

/// Reusable marks, so a pass touches only the units and channels it reaches.
#[derive(Default)]
pub(super) struct Marks {
    pub(super) chan: Vec<bool>,
    pub(super) touched: Vec<u16>,
    pub(super) acc: Vec<i32>,
    /// This pass's changes to the layer-0 input.
    pub(super) changes: Vec<(u16, i32)>,
    /// `layer`'s sends, each a unit in the low half and its activation change in the
    /// high half: the gate walk writes them, then one loop applies them.
    pub(super) sends: Vec<u64>,
    /// `blocks`' sends: the units and their activation changes, side by side.
    pub(super) send_u: Vec<u32>,
    pub(super) send_d: Vec<i32>,
    /// The positional stage's helper threads, once it runs in parts.
    pub(super) pool: Option<Pool>,
}

impl Marks {
    #[inline]
    pub(super) fn add(&mut self, ch: u16, v: i32) {
        if !self.chan[ch as usize] {
            self.chan[ch as usize] = true;
            self.touched.push(ch);
        }
        self.acc[ch as usize] += v;
    }
    /// Hands each touched channel's accumulated value to `f` and clears it.
    pub(super) fn drain(&mut self, mut f: impl FnMut(usize, i32)) {
        for &ch in &self.touched {
            let c = ch as usize;
            f(c, self.acc[c]);
            self.acc[c] = 0;
            self.chan[c] = false;
        }
        self.touched.clear();
    }

    /// Adds each touched channel's accumulated change to its wire in `x` and routes a
    /// nonzero one to the buckets of the layers `readers` names, clearing the marks.
    /// False when a wire leaves its `range`.
    #[inline(never)]
    fn drain_wires(&mut self, x: &mut [i32], wires: &[Wire], q: &mut Buckets) -> bool {
        let mut ok = true;
        let Marks { touched, acc, chan, .. } = self;
        // One bound per channel: every array is cut to the marks' length.
        let n = chan.len();
        let (acc, chan, x, wires) = (&mut acc[..n], &mut chan[..n], &mut x[..n], &wires[..n]);
        for &ch in touched.iter() {
            let c = ch as usize;
            assert!(c < n);
            let dv = std::mem::take(&mut acc[c]);
            chan[c] = false;
            if dv != 0 {
                let v = x[c] + dv;
                x[c] = v;
                let w = wires[c];
                ok &= w.holds(v);
                q.route(w.readers, ch, dv);
            }
        }
        touched.clear();
        ok
    }

    /// Applies input changes to one layer's gates and pushes activation differences
    /// down into the accumulator. Each gate change sends max(g', 0) - max(g, 0) down at
    /// once; in integers the intermediate differences sum to exactly the final one.
    /// `g` holds each gate minus one (`IntLayer::start`), so each send test is one sign
    /// test. On a fresh state (gates at their biases, nothing sent yet) the units whose
    /// bias alone fires send their activation first. `changed` sees every gate update
    /// as (unit, old, new, whether the unit is a fused pair), both gates kept as g - 1.
    /// Returns the gate updates made.
    ///
    /// `STAGED` picks the walk. Staged, every update stores its send and a second loop
    /// applies the kept ones, with no branch per update. The branchy walk sends in place
    /// and pays a mispredict whenever its send test goes the other way. Which is cheaper
    /// is measured per stage (docs/perf-log.md, "Overworld: the PPU thread and the staged
    /// walk"): the shared circuit layers stage, about one update in six sending; the
    /// positional stage, where a third to a half of the PPU's pixel updates send, walks
    /// branchy.
    #[inline(never)]
    pub(super) fn layer<const STAGED: bool>(&mut self, l: &IntLayer, g: &mut [i32], changes: &[(u16, i32)], fresh: bool, mut changed: impl FnMut(u16, i32, i32, bool)) -> usize {
        let send = |m: &mut Marks, u: usize, da: i32| {
            m.add(l.out()[u], l.c()[u] * da);
        };
        if fresh {
            for u in (0..l.bias().len()).filter(|&u| g[u] >= 0) {
                send(self, u, (g[u] + 1).min(l.cap()[u]));
                changed(u as u16, g[u], g[u], l.cap()[u] == 1);
            }
        }
        let mut updates = 0;
        // The indexing below is unchecked: `IntLayer::new` proved every gate entry's
        // unit below `bias.len()` and every output channel below `channels()`, and
        // these asserts tie `g` and the marks to those bounds.
        let n = l.bias().len();
        assert!(g.len() >= n && l.out().len() == n && l.c().len() == n);
        assert!(self.chan.len() >= l.channels() && self.acc.len() >= l.channels());
        if !STAGED {
            // SAFETY: the asserts above.
            return unsafe { self.walk_branchy(l, g, changes, changed) };
        }
        let gp = g.as_mut_ptr();
        // Each gate update's activation change, with a constant cap per loop: plain
        // units and fused pairs walk separate spans.
        let mut walk = |u: u16, w: i32, dx: i32, pair: bool| -> i32 {
            // SAFETY: u < n (IntLayer::new), the length of g.
            let old = unsafe { *gp.add(u as usize) };
            let new = old + w * dx;
            unsafe { *gp.add(u as usize) = new };
            changed(u, old, new, pair);
            // With each gate kept as g - 1, a pair's activation is the sign test
            // g - 1 >= 0, one minus the sign bit, and a plain unit's is max(g - 1 + 1, 0).
            if pair { ((old as u32) >> 31) as i32 - ((new as u32) >> 31) as i32 } else { (new + 1).max(0) - (old + 1).max(0) }
        };
        // About nine updates in ten send nothing, too unpredictably for a branch per
        // update. So the walk stores every update's unit and change at `k`, packed in
        // one word, and moves `k` on only when the change is not 0; a second loop sends
        // what it kept.
        let mut k = 0;
        for &(ch, dx) in changes {
            let (plain, pairs) = l.gate().split(ch as usize);
            let m = plain.len() + pairs.len();
            updates += m;
            if k + m >= self.sends.len() {
                self.sends.resize((k + m + 1).next_power_of_two(), 0);
            }
            let sp = self.sends.as_mut_ptr();
            // SAFETY: k + m < sends.len(), and each store is at k + (stores so far).
            for &(u, w) in plain {
                let da = walk(u, w, dx, false);
                unsafe { *sp.add(k) = u as u64 | (da as u32 as u64) << 32 };
                k += (da != 0) as usize;
            }
            for &(u, w) in pairs {
                let da = walk(u, w, dx, true);
                unsafe { *sp.add(k) = u as u64 | (da as u32 as u64) << 32 };
                k += (da != 0) as usize;
            }
        }
        let (out, c) = (l.out().as_ptr(), l.c().as_ptr());
        let (chan, acc) = (self.chan.as_mut_ptr(), self.acc.as_mut_ptr());
        // A channel enters `touched` when its mark goes up, and no mark comes down
        // here, so this call pushes at most `channels()`, and each send stores one slot past the
        // last push.
        self.touched.reserve(l.channels() + 1);
        let (tp, mut tn) = (self.touched.as_mut_ptr(), self.touched.len());
        for &s in &self.sends[..k] {
            let (u, da) = (s as u16 as usize, (s >> 32) as i32);
            // SAFETY: u is a gate entry's unit, so below n, the length of out and c;
            // o < channels() <= the lengths of chan and acc; and the reserve above
            // leaves room for every push.
            unsafe {
                let o = *out.add(u) as usize;
                // About one send in five finds its channel marked, so the push is a
                // store at tn either way, kept only when the mark was down.
                let was = *chan.add(o);
                *chan.add(o) = true;
                *tp.add(tn) = o as u16;
                tn += !was as usize;
                *acc.add(o) += *c.add(u) * da;
            }
        }
        // SAFETY: the first `tn` entries are written (above, or before this call).
        unsafe { self.touched.set_len(tn) };
        updates
    }

    /// `layer` (staged, without the census) over the layer's block table (`Blocks`).
    /// Each change updates its channel's blocks `LANES` gates at a time and its lone
    /// entries one at a time, the same updates `layer` makes, then the kept sends go
    /// down. A lane of weight 0 leaves its gate as it is, so its activation change is 0
    /// and it sends nothing. `g` holds the gates padded by `IntLayer::start_padded`.
    /// Returns the gate updates made.
    #[inline(never)]
    pub(super) fn blocks(&mut self, l: &IntLayer, g: &mut [i32], changes: &[(u16, i32)], singles: &[(u16, i32)], one_blocks: &[(u16, i32)], fresh: bool) -> usize {
        let n = l.bias().len();
        // The bounds the unchecked indexing below rests on: `IntLayer::with_blocks`
        // proved every block's lanes below n + LANES - 1 and its weights inside the
        // table, `IntLayer::new` every lone unit below n and every output below channels().
        assert!(l.has_blocks() && g.len() == n + LANES - 1 && l.out().len() == n && l.c().len() == n);
        // Every bucket holds channels below the marks' length: layer 0's changes, the
        // memory heads' output channels and `drain_wires` route only channels of the plan,
        // which this ties to the layer's tables for the unchecked lookups below.
        assert!(self.chan.len() == l.channels() && self.acc.len() >= l.channels());
        if fresh {
            for u in (0..n).filter(|&u| g[u] >= 0) {
                self.add(l.out()[u], l.c()[u] * (g[u] + 1).min(l.cap()[u]));
            }
        }
        let t = l.blocks();
        let (gp, wp, pp) = (g.as_mut_ptr(), t.weights().as_ptr(), t.pair().as_ptr());
        let (mut k, mut updates) = (0usize, 0usize);
        // Each change stages at most its channel's lanes and a block's last vector LANES
        // slots past them, and each single one slot: one check covers the whole call.
        let need = (changes.len() + one_blocks.len()) * (t.max_lanes() as usize + LANES) + singles.len() + 1;
        if need > self.send_u.len() {
            self.send_u.resize(need.next_power_of_two(), 0);
            self.send_d.resize(need.next_power_of_two(), 0);
        }
        let (su, sd) = (self.send_u.as_mut_ptr(), self.send_d.as_mut_ptr());
        for &(ch, dx) in changes {
            // SAFETY: ch < channels() (the assert above).
            let (vectors, plain, pairs, h) = unsafe { t.flat_unchecked(ch as usize) };
            updates += h.entries as usize;
            // The channel's vectors in one loop, whatever blocks they came from.
            for &(unit, w) in vectors {
                // SAFETY: the asserts above (`with_blocks` checked each vector inside a
                // block), and k + the channel's lanes + LANES within the sends.
                k = unsafe { block(gp, wp, pp, &Block { unit, w, vectors: 1 }, dx, su, sd, k) };
            }
            let mut lone = |u: u16, w: i32, pair: bool| {
                // SAFETY: u < n (IntLayer::new); k < the sends' length (reserved above).
                unsafe {
                    let old = *gp.add(u as usize);
                    let new = old + w * dx;
                    *gp.add(u as usize) = new;
                    let da = if pair { ((old as u32) >> 31) as i32 - ((new as u32) >> 31) as i32 } else { (new + 1).max(0) - (old + 1).max(0) };
                    *su.add(k) = u as u32;
                    *sd.add(k) = da;
                    k += (da != 0) as usize;
                }
            };
            for &(u, w) in plain {
                lone(u, w, false);
            }
            for &(u, w) in pairs {
                lone(u, w, true);
            }
        }
        // The block bucket: each change updates its channel's one block, found whole in
        // `Blocks::one_block`, with no head and no lone loops.
        if !one_blocks.is_empty() {
            let (ob, bl) = (t.one_block(), t.block_list());
            assert!(ob.len() == l.channels());
            for &(ch, dx) in one_blocks {
                // SAFETY: ch < channels(), the length of `one_block`; a channel reaches this
                // bucket only when its entry names one block (`Template::index`), which
                // `IntLayer::with_blocks` checked; the sends are sized below.
                unsafe {
                    let e = *ob.get_unchecked(ch as usize);
                    debug_assert!(e != super::int::NOT_SINGLE);
                    let b = bl.get_unchecked(e as u32 as usize);
                    updates += (e >> 32) as usize;
                    k = block(gp, wp, pp, b, dx, su, sd, k);
                }
            }
        }
        // The single bucket: each change moves its channel's one gate entry, read whole
        // from `Blocks::single`, with no block table and no loop per change.
        if !singles.is_empty() {
            let one = t.single();
            assert!(one.len() == l.channels());
            updates += singles.len();
            for &(ch, dx) in singles {
                // SAFETY: ch < channels() (a bucket holds channels of the layout), the length
                // of `single`; a channel reaches this bucket only when its entry is a single
                // (`Template::index`), whose unit `IntLayer::with_blocks` checked below n; k
                // stays below `need`.
                unsafe {
                    let e = *one.get_unchecked(ch as usize);
                    debug_assert!(e != super::int::NOT_SINGLE);
                    let u = e as u16 as usize;
                    let old = *gp.add(u);
                    let new = old + (e >> 32) as i32 * dx;
                    *gp.add(u) = new;
                    let pair = ((old as u32) >> 31) as i32 - ((new as u32) >> 31) as i32;
                    let plain = (new + 1).max(0) - (old + 1).max(0);
                    let da = if e >> 16 & 1 == 1 { pair } else { plain };
                    *su.add(k) = u as u32;
                    *sd.add(k) = da;
                    k += (da != 0) as usize;
                }
            }
        }
        let (out, c) = (l.out().as_ptr(), l.c().as_ptr());
        let (chan, acc) = (self.chan.as_mut_ptr(), self.acc.as_mut_ptr());
        self.touched.reserve(l.channels() + 1);
        let (tp, mut tn) = (self.touched.as_mut_ptr(), self.touched.len());
        for (&u, &da) in self.send_u[..k].iter().zip(&self.send_d[..k]) {
            // SAFETY: as in `layer`'s send loop, a kept send's unit is below n.
            unsafe {
                let o = *out.add(u as usize) as usize;
                let was = *chan.add(o);
                *chan.add(o) = true;
                *tp.add(tn) = o as u16;
                tn += !was as usize;
                *acc.add(o) += *c.add(u as usize) * da;
            }
        }
        // SAFETY: the first `tn` entries are written (above, or before this call).
        unsafe { self.touched.set_len(tn) };
        updates
    }

    /// `layer`'s branchy walk: each update sends in place when its sign test says so.
    ///
    /// # Safety
    ///
    /// `l` comes from `IntLayer::new`, `g`, `l.out()` and `l.c()` have one entry per
    /// unit, and `chan` and `acc` cover `l.channels()`: the bounds `layer` asserts.
    #[inline(always)]
    unsafe fn walk_branchy(&mut self, l: &IntLayer, g: &mut [i32], changes: &[(u16, i32)], mut changed: impl FnMut(u16, i32, i32, bool)) -> usize {
        let mut updates = 0;
        let (out, c) = (l.out().as_ptr(), l.c().as_ptr());
        let gp = g.as_mut_ptr();
        let (chan, acc) = (self.chan.as_mut_ptr(), self.acc.as_mut_ptr());
        // A channel enters `touched` when its mark goes up, and no mark comes down
        // here, so this call pushes at most `channels()`.
        self.touched.reserve(l.channels());
        let (tp, mut tn) = (self.touched.as_mut_ptr(), self.touched.len());
        let mut update = |u: usize, w: i32, dx: i32, pair: bool| {
            // SAFETY: u < n (IntLayer::new), the length of g, out and c.
            let old = unsafe { *gp.add(u) };
            let new = old + w * dx;
            unsafe { *gp.add(u) = new };
            changed(u as u16, old, new, pair);
            // With each gate kept as g - 1 (gates stay below 2^23 in magnitude), a pair
            // sends exactly when g crosses from <= 0 to >= 1 or back, a differing sign
            // bit, and a plain unit exactly when g is above 0 before or after, a clear
            // sign bit in the AND. That needs every update to move g: gate entries hold
            // nonzero weights, and every change list the layer reads carries only
            // nonzero changes.
            let sends = if pair { (new ^ old) < 0 } else { (new & old) >= 0 };
            if sends {
                let da = if pair { (new >= 0) as i32 - (old >= 0) as i32 } else { (new + 1).max(0) - (old + 1).max(0) };
                // SAFETY: o < channels() <= the lengths of chan and acc, and the
                // reserve above leaves room for every push.
                unsafe {
                    let (o, cu) = (*out.add(u) as usize, *c.add(u));
                    if !*chan.add(o) {
                        *chan.add(o) = true;
                        *tp.add(tn) = o as u16;
                        tn += 1;
                    }
                    *acc.add(o) += cu * da;
                }
            }
        };
        for &(ch, dx) in changes {
            let (plain, pairs) = l.gate().split(ch as usize);
            updates += plain.len() + pairs.len();
            for &(u, w) in plain {
                update(u as usize, w, dx, false);
            }
            for &(u, w) in pairs {
                update(u as usize, w, dx, true);
            }
        }
        // SAFETY: the first `tn` entries are written (above, or before this call).
        unsafe { self.touched.set_len(tn) };
        updates
    }

    /// `layer` for the routing layer, over its gate entries with each unit's pair flag
    /// and first owner beside them (`Template::route_gate`). A routing channel feeds
    /// about one unit, so one loop a channel with the flag in the entry beats `layer`'s
    /// two loops; the circuit layers, about seven units a channel, are the other way
    /// round. Folded into `layer` as one generic walk (perf-log), the circuit loop lost
    /// about 230 cycles a step and this one most of its gain, so the send is repeated.
    /// Each update also moves each owner's own route total (`own_route` at its home and
    /// bit) by the change in its clamped contribution, and marks the home in `stale`
    /// when that is not 0. A fresh state's owners need nothing: their totals start at
    /// the start gates (`Template::own_start`), and only gate changes move them. The
    /// caller proves every owner's home below the lengths of `own_route` and `stale`.
    ///
    /// An update takes the blank row's activation change `da` once: it is what the unit
    /// sends, and an owner's part moves by c · (its change with the owner's token − da).
    /// A linear unit (`RouteTag::linear`) needs no clamp: its gate moves by w · dx, its
    /// owner's total by c · w · dx, and it sends nothing.
    #[inline(never)]
    pub(super) fn route_layer(&mut self, l: &IntLayer, gate: &Tagged<RouteTag>, owners: &[[Owner; 2]], g: &mut [i32], changes: &[(u16, i32)], fresh: bool, own_route: &mut [[i32; ROUTE_CAP]], stale: &mut [u8]) -> usize {
        if fresh {
            for u in (0..g.len()).filter(|&u| g[u] >= 0) {
                self.add(l.out()[u], l.c()[u] * (g[u] + 1).min(l.cap()[u]));
            }
        }
        let mut updates = 0;
        // Unchecked as in `layer`, with the entries' units tied to the layer here.
        let n = l.bias().len();
        assert!(gate.units() <= n && g.len() == n && l.out().len() == n && l.c().len() == n && owners.len() == n);
        assert!(self.chan.len() >= l.channels() && self.acc.len() >= l.channels());
        let channels = gate.indices();
        let (out, c) = (l.out().as_ptr(), l.c().as_ptr());
        let gp = g.as_mut_ptr();
        let (chan, acc) = (self.chan.as_mut_ptr(), self.acc.as_mut_ptr());
        let (own_route, stale) = (own_route.as_mut_ptr(), stale.as_mut_ptr());
        self.touched.reserve(l.channels());
        let (tp, mut tn) = (self.touched.as_mut_ptr(), self.touched.len());
        // The change in clamp(g + 1 + own) from gate `old` to `new`, both kept as g - 1:
        // a pair's clamp to [0, 1] is the sign test g + own >= 0, a plain unit's is max(·, 0).
        let step = |old: i32, new: i32, own: i32, pair: bool| -> i32 {
            if pair { (((old + own) as u32) >> 31) as i32 - (((new + own) as u32) >> 31) as i32 } else { (new + 1 + own).max(0) - (old + 1 + own).max(0) }
        };
        let mut update = |u: usize, w: i32, dx: i32, t: RouteTag, pair: bool| {
            // SAFETY: u < gate.units() <= n, the length of g, out, c and owners.
            let old = unsafe { *gp.add(u) };
            let new = old + w * dx;
            unsafe { *gp.add(u) = new };
            let da = step(old, new, 0, pair);
            let mv = |o: Owner| {
                let d = o.c * (step(old, new, o.own, pair) - da);
                // SAFETY: the caller's bound on the owners' homes, and `Engine::build`
                // proved every owner's bit below ROUTE_CAP.
                unsafe {
                    *(*own_route.add(o.p as usize)).get_unchecked_mut(o.bit as usize) += d;
                    if d != 0 {
                        *stale.add(o.p as usize) = 1;
                    }
                }
            };
            mv(t.owner);
            // A second owner is rare (no routing update of the CPU on text.sh reaches
            // one), so it is read from the unit's owners only when the entry flags it.
            if t.second {
                mv(unsafe { owners.get_unchecked(u) }[1]);
            }
            if da != 0 {
                // SAFETY: as in `layer`.
                unsafe {
                    let (o, cu) = (*out.add(u) as usize, *c.add(u));
                    if !*chan.add(o) {
                        *chan.add(o) = true;
                        *tp.add(tn) = o as u16;
                        tn += 1;
                    }
                    *acc.add(o) += cu * da;
                }
            }
        };
        for &(ch, dx) in changes {
            assert!((ch as usize) < channels);
            // SAFETY: the assert above.
            let entries = unsafe { gate.get_unchecked(ch as usize) };
            updates += entries.len();
            for &(u, w, t) in entries {
                if t.linear {
                    // SAFETY: as in `update`. The change is w · dx on the gate and
                    // c · w · dx on the owner's total, neither 0, and nothing is sent.
                    unsafe {
                        *gp.add(u as usize) += w * dx;
                        *(*own_route.add(t.owner.p as usize)).get_unchecked_mut(t.owner.bit as usize) += t.owner.c * (w * dx);
                        *stale.add(t.owner.p as usize) = 1;
                    }
                    continue;
                }
                // Each arm inlines `update` with a constant cap.
                if t.pair {
                    update(u as usize, w, dx, t, true);
                } else {
                    update(u as usize, w, dx, t, false);
                }
            }
        }
        // SAFETY: the first `tn` entries are written (above, or before this call).
        unsafe { self.touched.set_len(tn) };
        updates
    }
}

impl Positional {
    /// Brings every specialised position forward by this pass's changes to the shared
    /// channels it reads, as `Engine::advance` does the shared layers, each part on its
    /// own thread when there is a pool. None when a positional wire leaves its declared
    /// range.
    fn advance(&self, s: &mut PosState, pool: Option<&Pool>, memory: Option<&Memory>, shared: &[(u16, i32)], fresh: bool, stale: &mut Bits, stats: &mut Stats) -> Option<u64> {
        if fresh {
            *s = self.fresh();
        }
        // After a change to the image since the heads last read (a write record, a fill),
        // every positional head reads again, each distinct line once per part.
        let generation = memory.map_or(0, |m| m.image.generation());
        let hit = std::mem::replace(&mut s.seen_gen, generation) != generation;
        let run = |p: &mut PosPart| self.advance_part(p, memory, hit, shared, fresh);
        match pool {
            Some(pool) => pool.each(&mut s.parts, &run),
            None => s.parts.iter_mut().for_each(run),
        }
        let (mut ok, mut units) = (true, 0);
        for p in &mut s.parts {
            ok &= p.ok;
            units += p.units;
            stats.memory_reads += std::mem::take(&mut p.reads);
            for &q in &p.stale {
                stale.set(q as usize);
            }
        }
        ok.then_some(units)
    }

    /// One part's pass.
    fn advance_part(&self, p: &mut PosPart, memory: Option<&Memory>, hit: bool, shared: &[(u16, i32)], fresh: bool) {
        let layers = &self.parts[p.part];
        p.stale.clear();
        let (mut ok, mut units) = (true, 0);
        for &(ch, dv) in shared {
            p.q.route(self.readers[ch as usize], ch, dv);
        }
        p.read.clear();
        for (k, groups) in layers.iter().enumerate() {
            if k < self.heads[p.part].by_layer.len() && !self.heads[p.part].by_layer[k].is_empty() {
                ok &= self.memory_reads(p, memory, k, hit);
            }
            for (l, g) in groups.iter().zip(&mut p.g[k]) {
                units += p.m.layer::<false>(l, g, p.q.get(k), fresh, |_, _, _, _| {});
            }
            p.q.clear(k);
            ok &= self.drain(p);
        }
        p.ok = ok && !p.q.overflowed();
        p.units = units as u64;
    }

    /// Serves the part's positional memory heads at circuit layer `k`, as
    /// `Engine::memory_reads` serves the shared ones. An instance whose query channels
    /// moved forms its address, reads again when the address moved or `hit`, and diffs
    /// the line into its channels. A line depends only on its (layer, address) and the
    /// cells, which no pass changes, so equal queries in a pass share one read. False
    /// when a query bit is not 0/1, the softmax is not one row, or there is no memory.
    fn memory_reads(&self, p: &mut PosPart, memory: Option<&Memory>, k: usize, hit: bool) -> bool {
        let Some(memory) = memory else { return false };
        let ph = &self.heads[p.part];
        let b = self.head_bit + k;
        for &(ch, dv) in p.q.get(b) {
            for &(bit, w) in ph.terms[k].get(ch as usize) {
                p.head_bits[bit as usize] += w * dv;
                p.head_dirty[bit as usize / KEY_BITS] = true;
            }
        }
        p.q.clear(b);
        let n = self.slot.len();
        for &i in &ph.by_layer[k] {
            let i = i as usize;
            if !std::mem::take(&mut p.head_dirty[i]) && !hit {
                continue;
            }
            let (mut addr, mut bad) = (0u32, 0u32);
            for (j, &v) in p.head_bits[i * KEY_BITS..][..KEY_BITS].iter().enumerate() {
                bad |= (v + 1) as u32 >> 2;
                addr |= ((v >= 1) as u32) << j;
            }
            if bad != 0 {
                return false;
            }
            if addr == p.head_addr[i] && !hit {
                continue;
            }
            p.head_addr[i] = addr;
            let (layer, out) = &ph.inst[i];
            let line = match p.read.iter().find(|r| r.0 == (*layer, addr)) {
                Some(r) => r.1,
                None => {
                    let Some((line, _)) = memory.cells.read(*layer, addr) else { return false };
                    let line = u64::from_le_bytes(line);
                    p.read.push(((*layer, addr), line));
                    p.reads += 1;
                    line
                }
            };
            let mut flips = line ^ p.head_out[i];
            p.head_out[i] = line;
            let (x, range) = (&mut p.x[..n], &self.range[..n]);
            while flips != 0 {
                let j = flips.trailing_zeros() as usize;
                flips &= flips - 1;
                let ch = out[j];
                if ch == NO_CHANNEL {
                    continue;
                }
                let dv = 2 * (line >> j & 1) as i32 - 1;
                let off = self.slot[ch as usize - self.base] as usize;
                x[off] += dv;
                if !(range[off].0 <= x[off] && x[off] <= range[off].1) {
                    return false;
                }
                p.q.route(self.readers[ch as usize], ch, dv);
            }
        }
        true
    }

    /// `Marks::drain_wires` for the positional channels: a touched channel is a
    /// position's wire, routed on, or one of its route totals. False when a wire
    /// leaves its range.
    #[inline(never)]
    pub(super) fn drain(&self, p: &mut PosPart) -> bool {
        let mut ok = true;
        let (m, q) = (&mut p.m, &mut p.q);
        let touched = std::mem::take(&mut m.touched);
        let n = self.slot.len();
        let (x, range, route) = (&mut p.x[..n], &self.range[..n], p.route.as_flattened_mut());
        for &ch in &touched {
            let c = ch as usize;
            let dv = std::mem::take(&mut m.acc[c]);
            m.chan[c] = false;
            if dv == 0 {
                continue;
            }
            let slot = self.slot[c - self.base];
            if slot & ROUTE_SLOT != 0 {
                let r = (slot & !ROUTE_SLOT) as usize;
                route[r] += dv;
                p.stale.push(self.at[self.first[p.part] + r / ROUTE_CAP] as u32);
            } else {
                let off = slot as usize;
                let v = x[off] + dv;
                x[off] = v;
                ok &= range[off].0 <= v && v <= range[off].1;
                q.route(self.readers[c], ch, dv);
            }
        }
        m.touched = touched;
        m.touched.clear();
        ok
    }
}

impl Engine {
    /// Serves layer `li`'s memory heads from the index: each head's address from the
    /// wires its query reads, the byte's bits diffed into the head's output wires and
    /// routed like any other change. None when a query form is not 0/1. The compiler
    /// inlines it into `advance`; kept out with `#[inline(never)]` it measured flat
    /// (`docs/perf-log.md`). With the bracketed copy of the pass it has two callers,
    /// and the attribute keeps it inlined into both.
    #[inline(always)]
    fn memory_reads<const P: bool>(mem: Option<(&MemPlan, &Memory)>, li: usize, s: &mut State, q: &mut Buckets, plan: &Template, stats: &mut Stats) -> Option<()> {
        let (mp, memory) = mem?;
        let image = &memory.image;
        let heads = &mp.layer_heads[li];
        if heads.is_empty() {
            return Some(());
        }
        // Nothing routed to the head bucket, the image as last seen and no head of this
        // layer dirty or hit: every head stays as it is, and no bus moved.
        if q.is_empty(plan.head_bucket()) && image.generation() == s.seen_gen && (0..mp.words).all(|w| (s.head_dirty[w] | s.head_hit[w]) & mp.layer_mask[li][w] == 0) {
            return Some(());
        }
        stats.probe.enter::<P>(Stage::HeadBucket(li as u8));
        let bucket = q.get(plan.head_bucket()).len() as u32;
        // The changes routed to the head bucket since the last drain move their units'
        // address bits and mark the heads they reach: the columns a channel gates, and
        // of a unit it moves the heads that follow the unit and the unit's first head,
        // which forms it at the unit's first layer whether or not any head follows. Every
        // change to a channel a query reads passes through this bucket once, so each bit
        // holds its form over the wires now.
        if !q.is_empty(plan.head_bucket()) {
            for &(ch, dv) in q.get(plan.head_bucket()) {
                for (w, &[always, following]) in mp.reach[ch as usize * mp.words..][..mp.words].iter().enumerate() {
                    s.head_dirty[w] |= always | following & s.head_follows[w];
                }
                for &(b, w) in mp.bit_terms.get(ch as usize) {
                    s.head_bits[b as usize] += w * dv;
                    s.unit_stale[b as usize / KEY_BITS] = true;
                }
            }
            q.clear(plan.head_bucket());
        }
        stats.probe.leave::<P>(Stage::HeadBucket(li as u8), [bucket, 0, 0, 0]);
        stats.probe.enter::<P>(Stage::Heads(li as u8));
        let before = [stats.memory_reads, s.lines_read.len() as u64];
        // A change to the image that no write record of this engine made (a new image, a
        // fill, a revert) may have moved any byte, so every head reads again.
        if image.generation() != s.seen_gen {
            s.head_hit.fill(!0);
            s.seen_gen = image.generation();
        }
        let group = memory.cells.builder.group(li) as u32;
        // The heads of this layer that are dirty or hit, in index order.
        let mut todo = [0u64; 8];
        assert!(mp.words <= todo.len(), "{} head words", mp.words);
        for w in 0..mp.words {
            todo[w] = (s.head_dirty[w] | s.head_hit[w]) & mp.layer_mask[li][w];
        }
        let todo_heads = todo.iter().map(|w| w.count_ones()).sum::<u32>();
        for w in 0..mp.words {
          while todo[w] != 0 {
            let hi = w * 64 + todo[w].trailing_zeros() as usize;
            todo[w] &= todo[w] - 1;
            let h = &mp.heads[hi];
            // The head's byte depends only on its address and the byte there. Its output
            // channels hold the byte at `head_addr` as of its last read, so it reads again
            // only when the address moved or a write hit it (`Engine::write_record`).
            let bit = 1u64 << (hi % 64);
            let (dirty, hit) = (s.head_dirty[hi / 64] & bit != 0, s.head_hit[hi / 64] & bit != 0);
            if dirty {
                let u = mp.unit_of[hi] as usize;
                let bits = &s.head_bits[u * KEY_BITS..][..KEY_BITS];
                if std::mem::take(&mut s.unit_stale[u]) {
                    s.units[u] = Unit::form(bits);
                    // A row with a form outside 0 and 1 moves its columns whatever their
                    // gates, so every head of the unit follows it from here and reads this
                    // pass. The unit's first head forms it, so the others of this layer
                    // are still ahead in `todo`.
                    if !s.units[u].is_row() {
                        for other in (0..mp.heads.len()).filter(|&o| mp.unit_of[o] as usize == u) {
                            let (v, b) = (other / 64, 1u64 << (other % 64));
                            s.head_follows[v] |= b;
                            s.head_dirty[v] |= b;
                            if other > hi {
                                todo[v] |= b & mp.layer_mask[li][v];
                            }
                        }
                    }
                }
                let addr = match mp.gate[hi] {
                    None => s.units[u].line()?,
                    Some((g, off)) => {
                        let (line, follows) = s.units[u].column(bits, s.x[g as usize], off)?;
                        s.head_follows[hi / 64] = s.head_follows[hi / 64] & !bit | if follows { bit } else { 0 };
                        line
                    }
                };
                if addr == s.head_addr[hi] && !hit {
                    continue;
                }
                s.head_addr[hi] = addr;
            }
            // The head's attention over the cells' rows; a softmax that is not one row
            // refuses the pass.
            let at = s.head_addr[hi];
            // Layers of one row group serve one cell's value (`Cells::serve`).
            let line = match s.lines_read.iter().find(|r| r.0 == group && r.1 == at) {
                Some(r) => r.2,
                None => {
                    let (line, served) = memory.cells.read(h.layer, at)?;
                    stats.count_read(served, memory.cells.scan_all);
                    stats.memory_reads += 1;
                    let line = u64::from_le_bytes(line);
                    s.lines_read.push((group, at, line));
                    line
                }
            };
            if let Some((lane, b)) = mp.col[hi] {
                let byte = (line >> (8 * lane)) as u8;
                let old = std::mem::replace(&mut s.col_byte[hi], byte);
                if byte != old && s.buses[b as usize].drive(old, byte) {
                    s.bus_moved.push(b);
                }
                continue;
            }
            let byte = line & mp.live[hi];
            // Only this loop writes the output channels, and `head_out` holds them packed
            // (both start at 0), so the changed channels are the set bits of the XOR, each
            // moving by +1 to a set bit or -1 to a clear one.
            let mut flips = byte ^ s.head_out[hi];
            s.head_out[hi] = byte;
            while flips != 0 {
                let i = flips.trailing_zeros() as usize;
                flips &= flips - 1;
                let ch = h.out[i];
                if ch == NO_CHANNEL {
                    continue;
                }
                let dv = 2 * (byte >> i & 1) as i32 - 1;
                s.x[ch as usize] += dv;
                // A query that reads the channel hears of it through the head bucket.
                q.route(plan.readers[ch as usize], ch, dv);
            }
          }
        }
        stats.probe.leave::<P>(Stage::Heads(li as u8), [todo_heads, (stats.memory_reads - before[0]) as u32, s.bus_moved.len() as u32, 0]);
        stats.probe.enter::<P>(Stage::Buses(li as u8));
        let buses = s.bus_moved.len() as u32;
        // For each bus a column moved, its drivers' bytes must not overlap (each channel 0
        // or 1, at most one column on its line), and the bits their sum changed are routed.
        for k in 0..s.bus_moved.len() {
            let b = s.bus_moved[k] as usize;
            let (acc, mut flips) = s.buses[b].settle()?;
            let chans = &mp.buses[b].0;
            while flips != 0 {
                let i = flips.trailing_zeros() as usize;
                flips &= flips - 1;
                let ch = chans[i];
                if ch == NO_CHANNEL {
                    continue;
                }
                let dv = 2 * (acc >> i & 1) as i32 - 1;
                s.x[ch as usize] += dv;
                q.route(plan.readers[ch as usize], ch, dv);
            }
        }
        stats.probe.leave::<P>(Stage::Buses(li as u8), [buses, 0, 0, 0]);
        s.bus_moved.clear();
        Some(())
    }

    /// For the cost model: the block vectors and the lone entries one layer's walk of these
    /// buckets updates (`Marks::blocks`).
    fn walk_shape(l: &IntLayer, bucket: &[(u16, i32)], singles: &[(u16, i32)], one_blocks: &[(u16, i32)]) -> [u32; 2] {
        let t = l.blocks();
        let (mut vectors, mut lone) = (0, singles.len());
        for &(ch, _) in bucket {
            let (v, plain, pairs, _) = t.flat(ch as usize);
            vectors += v.len();
            lone += plain.len() + pairs.len();
        }
        for &(ch, _) in one_blocks {
            vectors += t.block_list()[t.one_block()[ch as usize] as u32 as usize].vectors as usize;
        }
        [vectors as u32, lone as u32]
    }

    /// The census of one layer's bucket: the gate updates each channel's change causes,
    /// and each change once, in the first layer that reads the channel.
    fn count_sources(plan: &Template, l: &IntLayer, li: usize, bucket: &[(u16, i32)], d: usize, stats: &mut Stats) {
        if stats.source_updates.len() < d {
            stats.source_updates.resize(d, 0);
            stats.channel_changes.resize(d, 0);
        }
        for &(ch, _) in bucket {
            stats.source_updates[ch as usize] += l.gate().get(ch as usize).len() as u64;
            if plan.reading_layers(plan.readers[ch as usize]).trailing_zeros() as usize == li {
                stats.channel_changes[ch as usize] += 1;
            }
        }
    }

    /// Brings the integer state forward by `changes` to the layer-0 input, touching
    /// only what they reach. Each virtual channel has one writer, so a pass's changes
    /// form one list that later layers read. Leaves the state invalid if a channel
    /// leaves {-1, 0, 1}, or the range its clearing is exact on.
    #[inline(never)]
    fn advance<const P: bool>(plan: &Template, mem: Option<(&MemPlan, &Memory)>, memory: Option<&Memory>, fam: Option<&[Vec<u16>]>, demand: Option<&Demand>, s: &mut State, m: &mut Marks, q: &mut Buckets, d: usize, stats: &mut Stats) -> Result<u64, Refusal> {
        if let Some(f) = fam {
            let n = f.iter().flatten().copied().max().map_or(0, |x| x as usize + 1);
            if stats.family_updates.len() < n {
                stats.family_updates.resize(n, 0);
            }
        }
        let fresh = !s.valid;
        if fresh {
            // A refused pass can leave changes routed to later layers; they belong to no state.
            q.clear_all();
            let (tokens, weights, out) = (std::mem::take(&mut s.tokens), s.weights.take(), std::mem::take(&mut s.out));
            let (plan_positions, plan_verified) = (std::mem::take(&mut s.plan_positions), std::mem::take(&mut s.plan_verified));
            *s = State {
                valid: true,
                x: vec![0; d],
                g: plan.circuit.iter().map(IntLayer::start_padded).collect(),
                route_g: plan.routing.start(),
                stale: Bits::full(plan.seq + 1),
                own_route: plan.own_start.clone(),
                head_addr: vec![0; mem.map_or(0, |(mp, _)| mp.heads.len())],
                head_out: vec![0; mem.map_or(0, |(mp, _)| mp.heads.len())],
                head_dirty: vec![!0; mem.map_or(0, |(mp, _)| mp.words)],
                head_follows: mem.map_or(Vec::new(), |(mp, _)| mp.ungated.clone()),
                units: vec![Unit::default(); mem.map_or(0, |(mp, _)| mp.bit_consts.len() / KEY_BITS)],
                unit_stale: vec![true; mem.map_or(0, |(mp, _)| mp.bit_consts.len() / KEY_BITS)],
                head_hit: vec![!0; mem.map_or(0, |(mp, _)| mp.words)],
                seen_gen: u64::MAX,
                head_bits: mem.map_or(Vec::new(), |(mp, _)| mp.bit_consts.clone()),
                lines_read: Vec::new(),
                col_byte: vec![0; mem.map_or(0, |(mp, _)| mp.heads.len())],
                buses: vec![Bus::default(); mem.map_or(0, |(mp, _)| mp.buses.len())],
                tokens,
                weights,
                plan_positions,
                plan_verified,
                out,
                ..State::default()
            };
        }
        s.lines_read.clear();
        // Each change goes only to the layers that read its channel (the routing layer
        // is the last bucket).
        let n = plan.circuit.len();
        stats.changes += m.changes.len() as u64;
        stats.layer_units.resize(n + 1, 0);
        // A channel's value must stay where its clearing (if any) is exact.
        let range = &plan.seg.range;
        let in_range = |ch: usize, x: i32| range[ch].0 <= x && x <= range[ch].1;
        let mut ok = true;
        let changes = std::mem::take(&mut m.changes);
        stats.probe.enter::<P>(Stage::Scatter);
        for &(ch, dx) in &changes {
            s.x[ch as usize] += dx;
            ok &= in_range(ch as usize, s.x[ch as usize]);
            q.route(plan.readers[ch as usize], ch, dx);
        }
        m.changes = changes;
        stats.probe.leave::<P>(Stage::Scatter, [m.changes.len() as u32, 0, 0, 0]);
        let mut units = 0;
        for (li, l) in plan.circuit.iter().enumerate() {
            let before = [q.get(plan.head_bucket()).len() as u64, stats.memory_reads];
            stats.probe.enter::<P>(Stage::Reads(li as u8));
            let served = Self::memory_reads::<P>(mem, li, s, q, plan, stats);
            stats.probe.leave::<P>(Stage::Reads(li as u8), [before[0] as u32, (stats.memory_reads - before[1]) as u32, 0, 0]);
            if served.is_none() && mem.is_some() {
                s.valid = false;
                return Err(Refusal::MemoryQuery);
            }
            let (sb, bb) = (plan.single_bucket(li), plan.block_bucket(li));
            let (bucket, singles, one_blocks) = (q.get(li), sb.map_or(&[][..], |b| q.get(b)), bb.map_or(&[][..], |b| q.get(b)));
            stats.routed += (bucket.len() + singles.len() + one_blocks.len()) as u64;
            if fam.is_some() {
                for part in [bucket, singles, one_blocks] {
                    Self::count_sources(plan, l, li, part, d, stats);
                }
            }
            let shape = if stats.probe.wants::<P>(Stage::Walk(li as u8)) { Self::walk_shape(l, bucket, singles, one_blocks) } else { [0; 2] };
            stats.probe.shape::<P>([shape[0], shape[1], 0, 0]);
            stats.probe.enter::<P>(Stage::Walk(li as u8));
            // The census closure only when counting: the plain loop tests nothing per update.
            let here = match fam {
                Some(f) => {
                    let mut census = |u: u16, _: i32, _: i32, _: bool| {
                        stats.family_updates[f[li][plan.unit_ids[li][u as usize] as usize] as usize] += 1;
                        stats.unreached[0] += demand.is_some_and(|d| d.reach.drops[li][u as usize]) as u64;
                    };
                    m.layer::<true>(l, &mut s.g[li], bucket, fresh, &mut census) + m.layer::<true>(l, &mut s.g[li], singles, false, &mut census) + m.layer::<true>(l, &mut s.g[li], one_blocks, false, &mut census)
                }
                None => m.blocks(l, &mut s.g[li], bucket, singles, one_blocks, fresh),
            };
            stats.probe.leave::<P>(Stage::Walk(li as u8), [(bucket.len() + singles.len() + one_blocks.len()) as u32, here as u32, shape[0], shape[1]]);
            stats.layer_units[li] += here as u64;
            units += here;
            let touched = m.touched.len() as u32;
            stats.probe.shape::<P>([0, 0, touched, 0]);
            stats.probe.enter::<P>(Stage::Drain(li as u8));
            q.clear(li);
            if let (Some(sb), Some(bb)) = (sb, bb) {
                q.clear(sb);
                q.clear(bb);
            }
            ok &= m.drain_wires(&mut s.x, &plan.wires, q);
            stats.probe.leave::<P>(Stage::Drain(li as u8), [touched, 0, 0, 0]);
        }
        let bucket = q.get(n);
        stats.routed += bucket.len() as u64;
        if fam.is_some() {
            Self::count_sources(plan, &plan.routing, n, bucket, d, stats);
        }
        // A routing update moves the own route totals of the homes whose decode reads its
        // gate (`Marks::route_layer`). Every owner's home is at most `seq` (`Engine::build`),
        // and `stale` and `own_route` hold `seq + 1` entries.
        assert!(s.stale.bits() > plan.seq && s.own_route.len() > plan.seq);
        if let Some(f) = fam {
            // The census `Marks::layer` takes: a fresh state's firing units, then each update.
            if fresh {
                for u in (0..s.route_g.len()).filter(|&u| s.route_g[u] >= 0) {
                    stats.family_updates[f[n][u] as usize] += 1;
                }
            }
            for &(ch, _) in bucket {
                for u in plan.routing.readers_of(ch as usize) {
                    stats.family_updates[f[n][u as usize] as usize] += 1;
                    stats.unreached[1] += demand.is_some_and(|d| !d.route[u as usize]) as u64;
                }
            }
        }
        stats.probe.enter::<P>(Stage::Route);
        let here = m.route_layer(&plan.routing, &plan.route_gate, &plan.owners, &mut s.route_g, bucket, fresh, &mut s.own_route, &mut s.stale.0);
        stats.probe.leave::<P>(Stage::Route, [bucket.len() as u32, here as u32, 0, 0]);
        stats.layer_units[n] += here as u64;
        units += here;
        let touched = m.touched.len() as u32;
        stats.probe.enter::<P>(Stage::RouteDrain);
        q.clear(n);
        let mut moved = false;
        m.drain(|c, dv| {
            if dv != 0 {
                s.route[plan.route_index[c] as usize] += dv;
                moved = true;
            }
        });
        // Every position reads the blank-row route totals.
        if moved {
            s.stale.set_all();
        }
        stats.probe.leave::<P>(Stage::RouteDrain, [touched, moved as u32, 0, 0]);
        if let Some(pos) = &plan.pos {
            stats.probe.enter::<P>(Stage::Positional);
            let pool = (pos.parts.len() > 1 && !cfg!(target_arch = "wasm32")).then(|| &*m.pool.get_or_insert_with(|| Pool::new(pos.parts.len() - 1)));
            match pos.advance(&mut s.pos, pool, memory, q.get(plan.pos_bucket()), fresh, &mut s.stale, stats) {
                Some(u) => {
                    stats.layer_units[n] += u;
                    units += u as usize;
                }
                None => ok = false,
            }
            q.clear(plan.pos_bucket());
            stats.probe.leave::<P>(Stage::Positional, [0, 0, 0, 0]);
        }
        if stats.probe.wants::<P>(Stage::Tail) {
            stats.probe.shape::<P>([0, 0, 0, s.head_dirty.iter().map(|w| w.count_ones()).sum()]);
        }
        stats.probe.enter::<P>(Stage::Tail);
        s.head_dirty.fill(0);
        s.head_hit.fill(0);
        let overflowed = q.overflowed();
        stats.probe.leave::<P>(Stage::Tail, [0, 0, 0, 0]);
        if overflowed {
            s.valid = false;
            return Err(Refusal::BucketOverflow);
        }
        s.valid = ok;
        if ok { Ok(units as u64) } else { Err(Refusal::WireOutOfRange) }
    }

    /// The decoded tokens and the units re-evaluated, or why the pass must take the
    /// dense model.
    fn fill_fast<const P: bool>(&mut self, tokens: &[usize], positions: &[usize], sink: &mut Sink) -> Result<u64, Refusal> {
        if self.force_fallback {
            return Err(Refusal::Forced);
        }
        let clock = self.timed.then(std::time::Instant::now);
        self.stats.probe.enter::<P>(Stage::Layer0);
        let changed = self.layer0_changes(tokens, positions);
        if let Some(c1) = self.replan_from.take()
            && let Some(c2) = super::counters::cpi()
        {
            self.stats.replan_rest = self.stats.replan_rest.plus(c2.since(c1));
        }
        self.stats.probe.leave::<P>(Stage::Layer0, [self.moved.len() as u32, self.marks.changes.len() as u32, 0, 0]);
        let lap = |stage: usize, stats: &mut Stats, from: &mut Option<std::time::Instant>| {
            if let Some(t) = from {
                let now = std::time::Instant::now();
                stats.stage_ns[stage] += (now - *t).as_nanos() as u64;
                *t = now;
            }
        };
        let mut clock = clock;
        lap(0, &mut self.stats, &mut clock);
        let Ok(whole) = &self.plan else { return Err(Refusal::NoPlan) };
        let plan = &whole.t;
        if changed.is_none() {
            self.state.valid = false;
            return Err(Refusal::Layer0Attention);
        }
        let mem = whole.mem.as_ref().map(|mp| (mp, self.memory.as_ref().expect("a memory checkpoint needs an inserted cartridge")));
        let fam = self.families.as_ref().map(|f| f.1.as_slice());
        let demand = match fam {
            Some(_) => Some(&*self.demands.entry(positions.to_vec()).or_insert_with(|| whole.demand(positions.iter().map(|&p| tok::output_mask(&self.dense.cfg, p))))),
            None => None,
        };
        let units = Self::advance::<P>(plan, mem, self.memory.as_ref(), fam, demand, &mut self.state, &mut self.marks, &mut self.pending, plan.seg.count, &mut self.stats)?;
        lap(1, &mut self.stats, &mut clock);
        self.stats.probe.enter::<P>(Stage::DecodeSetup);
        let s = &mut self.state;
        // A position's token depends only on its own token, the blank-row route totals,
        // the routing units its token channels reach and its positional route totals:
        // unless one of those moved since it last decoded (`stale`), it decodes to the
        // same token. Layer 0's moved positions are those whose token changed.
        for &j in &self.moved {
            s.stale.set(j);
        }
        s.got.resize(tokens.len(), 0);
        let id = self.positions_id;
        if (id == 0 || s.want_id != id) && s.want_positions != positions {
            s.want_positions.clear();
            s.want_positions.extend_from_slice(positions);
            s.want = Bits::empty(s.stale.bits());
            for &p in positions {
                s.want.set(p);
            }
            let (lo, hi) = positions.iter().fold((usize::MAX, 0), |(lo, hi), &p| (lo.min(p), hi.max(p)));
            s.want_words = if positions.is_empty() { 0..0 } else { lo / 8..hi / 8 + 1 };
            s.want_top = if positions.is_empty() { 0 } else { hi + 1 };
        }
        s.want_id = id;
        if let Some(pos) = &plan.pos {
            // The stage folded each position's mask token's slot identity into its units.
            if positions.iter().any(|&p| pos.block[p].is_some() && tokens[p] != tok::output_mask(&self.dense.cfg, p)) {
                s.valid = false;
                return Err(Refusal::PositionalToken);
            }
        }
        self.stats.probe.leave::<P>(Stage::DecodeSetup, [self.moved.len() as u32, 0, 0, 0]);
        let before = [self.stats.decoded, self.stats.own];
        self.stats.probe.enter::<P>(Stage::Decode);
        // A store that missed a decode holds another token than this state at a position
        // that is not stale: every position decodes again.
        if std::mem::take(&mut self.resync) {
            s.stale.set_all();
        }
        s.fresh.clear();
        let listed = !s.dense;
        for w in s.want_words.clone() {
            let mut due = s.stale.word(w) & s.want.word(w);
            while due != 0 {
                let p = w * 8 + due.trailing_zeros() as usize / 8;
                due &= due - 1;
                let t = tokens[p];
                let mut route = s.route;
                if let Some(pos) = &plan.pos
                    && let Some(b) = pos.block[p]
                {
                    for (r, v) in route.iter_mut().zip(pos.route(&s.pos, b as usize)) {
                        *r += v;
                    }
                }
                let home = plan.tracked[t] as usize == p;
                s.got[p] = Self::decode(plan, s, &mut self.stats, t, p, home, route)?;
                if listed {
                    s.fresh.push(p as u32);
                }
                if home {
                    s.stale.0[p] = 0;
                }
            }
        }
        self.stats.probe.leave::<P>(Stage::Decode, [(self.stats.decoded - before[0]) as u32, (self.stats.own - before[1]) as u32, 0, 0]);
        s.dense = 2 * (self.stats.decoded - before[0]) as usize >= positions.len();
        self.stats.probe.enter::<P>(Stage::Out);
        // One bound for the whole list: `positions` is the list `want_top` was taken over,
        // which the block above found equal by id or by content.
        assert!(s.want_top <= s.got.len());
        match sink {
            Sink::List(out) => {
                debug_assert!(positions.iter().all(|&p| p < s.want_top), "the positions are not the list the bound was taken over");
                if out.len() != positions.len() {
                    out.resize(positions.len(), 0);
                }
                for (o, &p) in out.iter_mut().zip(positions) {
                    // SAFETY: p < want_top <= got.len().
                    *o = unsafe { *s.got.get_unchecked(p) };
                }
            }
            // A position whose decode did not run was not stale, so the store holds its token.
            Sink::Store { decoded, own } => {
                match self.view.as_ref() {
                    // The engine's token is current at every position that is not stale, so
                    // writing every position back is right whenever it is the cheaper way.
                    _ if s.dense || !listed => decoded.put_all(own, positions.iter().map(|&q| s.got[q])),
                    Some(v) => {
                        for &q in &s.fresh {
                            decoded.put(v.stream_of[q as usize] as usize, s.got[q as usize]);
                        }
                    }
                    None => {
                        for &q in &s.fresh {
                            decoded.put(q as usize, s.got[q as usize]);
                        }
                    }
                }
                if self.verify {
                    for (&p, &q) in own.iter().zip(positions) {
                        assert!(decoded.at(p) == s.got[q], "position {p} ({}): the sequence's store holds token {}, the engine's state {}", self.dense.slots[q], decoded.at(p), s.got[q]);
                    }
                }
            }
        }
        self.stats.probe.leave::<P>(Stage::Out, [positions.len() as u32, 0, 0, 0]);
        lap(2, &mut self.stats, &mut clock);
        Ok(units)
    }

    /// The token position `p` holding `t` decodes to from `route` (the blank-row route
    /// totals, plus its own positional routes): its token's own routing units, kept
    /// summed in `State::own_route` when `t` is tracked at `p` (`home`), then the head
    /// table. Refused, with the state invalid, when a route is not 0/1.
    #[inline(always)]
    fn decode(plan: &Template, s: &mut State, stats: &mut Stats, t: usize, p: usize, home: bool, mut route: [i32; ROUTE_CAP]) -> Result<usize, Refusal> {
        stats.decoded += 1;
        if home {
            for (r, v) in route.iter_mut().zip(s.own_route[p]) {
                *r += v;
            }
        } else {
            let own = &plan.own_units[t];
            stats.own += own.len() as u64;
            // `Engine::build` proved every own unit below the routing layer's length and every
            // route bit below ROUTE_CAP; this ties the gates to that length.
            assert!(s.route_g.len() == plan.routing.bias().len());
            for o in own {
                // SAFETY: the bounds above.
                // The gate itself: `route_g` keeps g - 1.
                let g = unsafe { *s.route_g.get_unchecked(o.u as usize) } + 1;
                let cap = if o.pair { 1 } else { i32::MAX };
                let da = (g + o.own).min(cap).max(0) - g.min(cap).max(0);
                unsafe { *route.get_unchecked_mut(o.bit as usize) += o.c * da };
            }
        }
        // A nibble head reads routes 0 and 1 as the byte's nibbles and the rest as bits;
        // a route past the table's stays 0.
        if plan.value_head {
            let (pattern, ok) = value_pattern(&route);
            return match plan.head_table.get(pattern) {
                Some(&token) if ok => Ok(if token == NO_TOKEN { usize::MAX } else { token as usize }),
                _ => {
                    s.valid = false;
                    Err(Refusal::RouteNot01)
                }
            };
        }
        // Every route total must be 0 or 1. The routes past `plan.routes` stay 0, so a
        // pattern outside the head table is refused too.
        let (mut pattern, mut any) = (0usize, 0u32);
        for (i, &v) in route.iter().enumerate() {
            pattern |= ((v & 1) as usize) << i;
            any |= v as u32;
        }
        match plan.head_table.get(pattern) {
            Some(&token) if any <= 1 => Ok(if token == NO_TOKEN { usize::MAX } else { token as usize }),
            _ => {
                s.valid = false;
                Err(Refusal::RouteNot01)
            }
        }
    }
}

/// Where a pass's decoded tokens go.
pub(super) enum Sink<'a, 'b> {
    /// Every position's token, in the positions' order.
    List(&'a mut Vec<usize>),
    /// A sequence's decoded store, written at the positions whose decode ran. `own` is
    /// the pass's positions as the stream names them, in the positions' order.
    Store { decoded: &'a mut Decoded<'b>, own: &'a [usize] },
}

impl Sink<'_, '_> {
    /// Every position's token, in the positions' order.
    fn put_all(&mut self, tokens: impl Iterator<Item = usize>) {
        match self {
            Sink::List(out) => {
                out.clear();
                out.extend(tokens);
            }
            Sink::Store { decoded, own } => decoded.put_all(own, tokens),
        }
    }

    /// The pass's tokens in the positions' order, as the caller will read them.
    fn list<'s>(&'s self, scratch: &'s mut Vec<usize>) -> &'s [usize] {
        match self {
            Sink::List(out) => out,
            Sink::Store { decoded, own } => {
                scratch.clear();
                scratch.extend(own.iter().map(|&p| decoded.at(p)));
                scratch
            }
        }
    }
}

/// The tokens a pass reads.
enum Placing<'a> {
    /// A slice, compared whole with the last placement.
    Whole(&'a [usize]),
    /// A sequence's tokens and the positions it lists as changed; `synced` when the last
    /// placement holds that sequence at its last pass.
    Changed { tokens: &'a [usize], changed: &'a [u32], stamp: Stamp, synced: bool },
}

impl Placing<'_> {
    fn tokens(&self) -> &[usize] {
        match self {
            Placing::Whole(tokens) | Placing::Changed { tokens, .. } => tokens,
        }
    }
}

/// With a stream's view, the stream's own tokens and positions, mapped onto the
/// checkpoint's sequence and back.
impl Denoiser for Engine {
    fn cfg(&self) -> &Config {
        self.view.as_ref().map_or(&self.dense.cfg, |v| &v.cfg)
    }
    fn slots(&self) -> &[String] {
        self.view.as_ref().map_or(&self.dense.slots, |v| &v.slots)
    }
    fn fill_into(&mut self, tokens: &[usize], positions: &[usize], out: &mut Vec<usize>) {
        // The tokens decoded here reach no sequence's store.
        self.out_stamp = None;
        if self.replay.is_none() {
            self.fill_stream(Placing::Whole(tokens), positions, 0, &mut Sink::List(out));
            return;
        }
        let mut r = self.replay.take().expect("checked above");
        r.pass(&self.memory.as_ref().expect("a recorded stream holds the memory").image, tokens, positions);
        self.fill_stream(Placing::Whole(tokens), positions, 0, &mut Sink::List(out));
        if r.decoded(out) {
            self.replay = Some(r);
        }
    }

    /// Places the positions the sequence lists as changed and writes back the positions
    /// whose decode ran, when this engine's last pass was this sequence's last pass;
    /// otherwise the whole sequence is compared and every position decoded again. Under
    /// the step oracle, and in a debug build or the library's unit tests, every pass also
    /// compares the whole sequence with the placement and every position's stored token
    /// with the engine's.
    fn fill_seq(&mut self, seq: &mut Sequence, positions: &Positions) {
        let before = seq.stamp();
        let mut replay = self.replay.take();
        if let Some(r) = replay.as_mut() {
            r.pass(&self.memory.as_ref().expect("a recorded stream holds the memory").image, seq.tokens(), positions);
        }
        let Pass { tokens, changed, mut decoded, stamp } = seq.begin();
        // A position that is not stale holds in the store the token its last decode gave
        // only while every decode since reached this store.
        self.resync |= self.out_stamp != Some(before);
        self.out_stamp = None;
        let synced = self.view.as_ref().is_some_and(|v| v.placed_stamp == Some(before));
        let fast = self.fill_stream(Placing::Changed { tokens, changed, stamp, synced }, positions, positions.id(), &mut Sink::Store { decoded: &mut decoded, own: positions });
        if fast {
            self.out_stamp = Some(stamp);
        }
        if let Some(mut r) = replay {
            let mut list = std::mem::take(&mut self.list);
            list.clear();
            list.extend(positions.iter().map(|&p| seq.decoded()[p]));
            if r.decoded(&list) {
                self.replay = Some(r);
            }
            self.list = list;
        }
    }
}

impl Engine {
    /// Whether a pass also proves its sequence's change list and decoded store complete.
    fn verifies(&self) -> bool {
        cfg!(any(debug_assertions, test)) || self.step_oracle > 0 || self.check_next
    }

    /// A pass in the stream's tokens: the bracketed copy while the replay bench has armed
    /// a stage (`probe`), else the plain one. True when the fast path decoded it.
    fn fill_stream(&mut self, placing: Placing, positions: &[usize], key: u64, sink: &mut Sink) -> bool {
        if self.stats.probe.is_armed() {
            return self.fill_stream_as::<true>(placing, positions, key, sink);
        }
        self.fill_stream_as::<false>(placing, positions, key, sink)
    }

    fn fill_stream_as<const P: bool>(&mut self, placing: Placing, positions: &[usize], key: u64, sink: &mut Sink) -> bool {
        let verify = self.verifies();
        self.verify = verify;
        let Some(v) = self.view.as_mut() else {
            #[cfg(feature = "record")]
            if self.delegate(placing.tokens(), positions, sink) {
                return false;
            }
            self.positions_id = key;
            return self.fill_merged::<P>(placing.tokens(), positions, sink);
        };
        self.stats.probe.enter::<P>(Stage::View);
        match placing {
            Placing::Changed { tokens, changed, stamp, synced: true } => {
                v.place_changed(tokens, changed, verify);
                v.placed_stamp = Some(stamp);
            }
            Placing::Changed { tokens, stamp, synced: false, .. } => {
                v.place_tokens(tokens);
                v.placed_stamp = Some(stamp);
            }
            Placing::Whole(tokens) => v.place_tokens(tokens),
        }
        v.place_positions(positions, key);
        self.stats.probe.leave::<P>(Stage::View, [v.dirty.len() as u32, 0, 0, 0]);
        self.positions_id = v.positions_id;
        let (toks, pos) = (std::mem::take(&mut v.tokens), std::mem::take(&mut v.positions));
        #[cfg(feature = "record")]
        let delegated = self.delegate(&toks, &pos, sink);
        #[cfg(not(feature = "record"))]
        let delegated = false;
        let fast = !delegated && self.fill_merged::<P>(&toks, &pos, sink);
        #[cfg(feature = "record")]
        {
            let mut list = std::mem::take(&mut self.list);
            self.record("fill", &toks, &pos, sink.list(&mut list));
            self.list = list;
        }
        let v = self.view.as_mut().expect("the view stays in place");
        (v.tokens, v.positions) = (toks, pos);
        fast
    }
}

#[cfg(feature = "record")]
impl Engine {
    /// Hands the pass to the external runtime, when one is installed, and writes its
    /// tokens to `sink` in the stream's vocabulary (the checkpoint's, without a view).
    fn delegate(&mut self, tokens: &[usize], positions: &[usize], sink: &mut Sink) -> bool {
        let Some(ext) = self.external.as_mut() else { return false };
        // A delegated pass is no fast pass the step oracle checks: a check asked for it is spent.
        self.check_next = false;
        self.stats.passes += 1;
        let got = ext.fill(tokens, positions, self.memory.as_ref().map(|m| &m.image));
        match self.view.as_ref() {
            Some(v) => sink.put_all(got.into_iter().map(|t| v.back(t))),
            None => sink.put_all(got.into_iter()),
        }
        true
    }

    /// Records an event in the checkpoint's token space. `out` holds the stream's tokens.
    pub(super) fn record(&mut self, kind: &str, tokens: &[usize], positions: &[usize], out: &[usize]) {
        let Some(r) = self.recorder.as_mut() else { return };
        if !r.due() {
            return;
        }
        let v = self.view.as_ref().expect("only a view records");
        let out: Vec<usize> = out.iter().map(|&t| v.to[t] as usize).collect();
        let heads: Vec<(u32, [u8; crate::memory::layout::LINE])> = match (&self.plan, &self.memory) {
            (Ok(super::Plan { t, mem: Some(mp) }), Some(m)) if self.external.is_none() => {
                let line = |layer: usize, n: u32| (n, m.cells.serve(layer, n).map_or([0; crate::memory::layout::LINE], |r| r.0));
                let mut heads: Vec<_> = (0..mp.heads.len()).filter_map(|h| self.state.head_addr.get(h).map(|&n| line(mp.heads[h].layer, n))).collect();
                // A positional head reads a line at every position. Each distinct line follows
                // the shared heads' once, so a reader of the record sees every line the pass read.
                if let Some(pos) = &t.pos {
                    let mut seen = std::collections::BTreeSet::new();
                    for p in &self.state.pos.parts {
                        for (&n, (layer, _)) in p.head_addr.iter().zip(&pos.heads[p.part].inst) {
                            if n != u32::MAX && seen.insert(n) {
                                heads.push(line(*layer, n));
                            }
                        }
                    }
                }
                heads
            }
            _ => Vec::new(),
        };
        let position = self.memory.as_ref().map_or(0, |m| m.kv.len());
        let r = self.recorder.as_mut().expect("checked above");
        let cells: &[(u32, [u8; crate::memory::layout::LINE])] = match &self.memory {
            Some(m) if r.cells_max > 0 && m.kv.len() <= r.cells_max => m.kv.cells(),
            _ => &[],
        };
        let event = r.last();
        r.line(kind, event, position, tokens, positions, &out, &heads, cells);
    }
}

impl Engine {
    /// The pass over the checkpoint's tokens. True when the fast path decoded it.
    fn fill_merged<const P: bool>(&mut self, tokens: &[usize], positions: &[usize], sink: &mut Sink) -> bool {
        self.stats.passes += 1;
        // A check asked for this pass is spent here, fast or not: a dense fallback is the
        // oracle's own pass, and the next pass is not the one asked for.
        let forced = std::mem::take(&mut self.check_next);
        if self.timed && (self.stats.passes == 1 || self.stats.passes % 4096 == 0) {
            // A thread's counters are its own: the span restarts when the engine moves thread.
            let at = (self.stats.passes - 1, std::thread::current().id(), super::counters::thread().unwrap_or_default());
            match self.stats.thread_first {
                Some((_, id, _)) if id == at.1 => self.stats.thread_last = Some(at),
                _ => (self.stats.thread_first, self.stats.thread_last) = (Some(at), None),
            }
        }
        let t = self.timed.then(std::time::Instant::now);
        // Every 1,024th timed pass is also bracketed by the thread's own counters.
        let sampled = self.timed && self.stats.passes % 1024 == 0;
        if sampled && let (Some(a), Some(b)) = (super::counters::thread(), super::counters::thread()) {
            self.stats.brackets += 1;
            self.stats.bracket = self.stats.bracket.plus(b.since(a));
        }
        let sample = sampled.then(super::counters::thread).flatten();
        match self.fill_fast::<P>(tokens, positions, sink) {
            Ok(units) => {
                let sampled = sample.and_then(|c0| super::counters::thread().map(|c1| c1.since(c0)));
                self.stats.fast_passes += 1;
                self.stats.units += units;
                let kind = self.stats.kind(positions.len());
                kind.passes += 1;
                kind.units += units;
                if let Some(t) = t {
                    kind.nanos += t.elapsed().as_nanos() as u64;
                }
                if let Some(c) = sampled {
                    kind.sampled += 1;
                    kind.counts = kind.counts.plus(c);
                }
                #[cfg(feature = "testing")]
                if let Some(x) = self.fault.take() {
                    match sink {
                        Sink::List(out) => {
                            if let Some(t) = out.first_mut() {
                                *t ^= x;
                            }
                        }
                        Sink::Store { decoded, own } => {
                            if let Some(&p) = own.first() {
                                decoded.corrupt(p, x);
                            }
                        }
                    }
                }
                self.stats.forced_checks += forced as u64;
                if forced {
                    self.check_sink(tokens, positions, sink);
                } else if self.step_oracle > 0 || self.rom_oracle > 0 {
                    // ROM past 2 MiB is the only memory with key bits 18 and 19 set.
                    let rom = |b: u32| self.state.head_addr.iter().any(|&n| n < ROM_LINES && n >> b & 1 == 1);
                    let (b18, b19) = (rom(18), rom(19));
                    self.stats.high_rom_passes[0] += b18 as u64;
                    self.stats.high_rom_passes[1] += b19 as u64;
                    let high = b18 || b19;
                    self.stats.high_rom_seen += high as u64;
                    let uniform = self.step_oracle > 0 && (self.stats.fast_passes - 1) % self.step_oracle == 0;
                    if uniform || high && self.rom_oracle > 0 && (self.stats.high_rom_seen - 1) % self.rom_oracle == 0 {
                        self.check_sink(tokens, positions, sink);
                    }
                }
                return true;
            }
            Err(why) => self.stats.refused = Some(why),
        }
        self.stats.fallback_passes += 1;
        let all = match &self.memory {
            // A memory checkpoint's stream without the machine's memory is not the pass
            // a stock runtime runs, since its foreign memory heads would find no cell.
            None if self.dense.memory.is_some() => panic!("a pass of this stream needs the dense fallback ({:?}), which needs the machine's memory the stream does not hold", self.stats.refused),
            None => self.dense.forward(tokens, false),
            Some(mem) => self.dense.forward_cached(&mem.kv, tokens, crate::memory::attn::READ_AT).0,
        };
        // A view's fast passes decode straight to its own tokens (`Plan::prune` maps its
        // head table); the dense pass gives the checkpoint's.
        let back = |t: usize| self.view.as_ref().map_or(t, |v| v.back(t));
        sink.put_all(positions.iter().map(|&p| back(all[p])));
        false
    }

    /// `check_step` on the tokens the pass's caller will read.
    fn check_sink(&mut self, tokens: &[usize], positions: &[usize], sink: &Sink) {
        let mut list = std::mem::take(&mut self.list);
        self.check_step(tokens, positions, sink.list(&mut list));
        self.list = list;
    }

    /// The step oracle: the dense pass of this engine's network over the same tokens and
    /// the same KV cache, each decoded token against the fast pass's `out`. A stream of
    /// the one checkpoint runs its own network, its slice (`Model::stream`). A stream
    /// that holds no memory of its own (the APU's, control's) reads a cache of line 0
    /// alone, the one cell its copy heads read, since its record slots hold masks;
    /// `every_slice_is_the_checkpoint_at_its_positions` holds those two slices over that
    /// cache to the whole checkpoint.
    fn check_step(&mut self, tokens: &[usize], positions: &[usize], out: &[usize]) {
        let clock = self.timed.then(std::time::Instant::now);
        let all = match (&self.memory, &self.dense.memory) {
            (Some(mem), _) => self.dense.forward_cached(&mem.kv, tokens, crate::memory::attn::READ_AT).0,
            (None, Some(spec)) => {
                let cache = self.step_cache.get_or_insert_with(|| {
                    let mut c = crate::llama::KvCache::new(spec.heads.clone());
                    c.set(0, [0; crate::memory::layout::LINE]);
                    c
                });
                self.dense.forward_cached(cache, tokens, crate::memory::attn::READ_AT).0
            }
            (None, None) => self.dense.forward(tokens, false),
        };
        let back = |t: usize| self.view.as_ref().map_or(t, |v| v.back(t));
        for (&p, &got) in positions.iter().zip(out) {
            let want = back(all[p]);
            assert_eq!(got, want, "step oracle: fast pass {} at position {p} ({}) decoded token {got}, the dense pass {want}", self.stats.fast_passes, self.dense.slots[p]);
        }
        self.stats.step_checks += 1;
        for (b, n) in self.stats.step_high.iter_mut().enumerate() {
            *n += self.state.head_addr.iter().any(|&a| a >> (18 + b) & 1 == 1) as u64;
        }
        if let Some(t) = clock {
            self.stats.step_check_ns += t.elapsed().as_nanos() as u64;
        }
    }
}

/// `value_pattern` one route at a time: the definition, and the path where there is no NEON.
#[cfg_attr(target_arch = "aarch64", cfg(test))]
fn value_pattern_scalar(route: &[i32; ROUTE_CAP]) -> (usize, bool) {
    let (mut pattern, mut any) = ((route[0] | route[1] << 4) as usize, 0u32);
    for (i, &v) in route.iter().enumerate().skip(2) {
        pattern |= ((v & 1) as usize) << (i + 6);
        any |= v as u32;
    }
    (pattern, (0..16).contains(&route[0]) && (0..16).contains(&route[1]) && any <= 1)
}

#[cfg(not(target_arch = "aarch64"))]
use value_pattern_scalar as value_pattern;

/// A nibble head's route pattern (routes 0 and 1 as nibbles at bits 0 and 4, route i from
/// 2 on as bit i + 6), and whether every route is in range: 0 to 15 for the nibbles, 0 or
/// 1 for the rest. An out-of-range route may give any pattern. Four routes at a
/// time.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn value_pattern(route: &[i32; ROUTE_CAP]) -> (usize, bool) {
    use std::arch::aarch64::*;
    const KEEP: [i32; ROUTE_CAP] = [15, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    const SHIFT: [i32; ROUTE_CAP] = [0, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    // SAFETY: the three loads of each array cover its 12 entries; NEON is part of every
    // aarch64 target.
    unsafe {
        let (mut pattern, mut bad) = (vdupq_n_s32(0), vdupq_n_s32(0));
        for k in 0..ROUTE_CAP / 4 {
            let (v, keep) = (vld1q_s32(route.as_ptr().add(4 * k)), vld1q_s32(KEEP.as_ptr().add(4 * k)));
            bad = vorrq_s32(bad, vbicq_s32(v, keep));
            pattern = vorrq_s32(pattern, vshlq_s32(vandq_s32(v, keep), vld1q_s32(SHIFT.as_ptr().add(4 * k))));
        }
        // The kept bits of the twelve routes land on distinct bits, so the lanes' sum is their OR.
        (vaddvq_u32(vreinterpretq_u32_s32(pattern)) as usize, vmaxvq_u32(vreinterpretq_u32_s32(bad)) == 0)
    }
}

/// One block's update by `dx`: each lane's gate moves by its weight times `dx`, and each
/// lane whose activation changes stores its unit and change at `k` on, in lane order.
/// Returns `k` past the stored sends.
///
/// # Safety
///
/// `g` and `pair` hold the block's lanes, `w` its weights, and `su` and `sd` at least
/// `k + b.vectors · LANES + LANES` slots.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block(g: *mut i32, w: *const i32, pair: *const u32, b: &Block, dx: i32, su: *mut u32, sd: *mut i32, mut k: usize) -> usize {
    use std::arch::aarch64::*;
    // SAFETY: the caller's bounds; NEON is part of every aarch64 target.
    unsafe {
        let (one, zero) = (vdupq_n_s32(1), vdupq_n_s32(0));
        let bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        let mut units = vaddq_u32(vdupq_n_u32(b.unit), vld1q_u32([0u32, 1, 2, 3].as_ptr()));
        let step = vdupq_n_u32(LANES as u32);
        let (gp, wp, pp) = (g.add(b.unit as usize), w.add(b.w as usize), pair.add(b.unit as usize));
        for v in 0..b.vectors as usize {
            let old = vld1q_s32(gp.add(v * LANES));
            let new = vmlaq_n_s32(old, vld1q_s32(wp.add(v * LANES)), dx);
            vst1q_s32(gp.add(v * LANES), new);
            let plain = vsubq_s32(vmaxq_s32(vaddq_s32(new, one), zero), vmaxq_s32(vaddq_s32(old, one), zero));
            let pairs = vreinterpretq_s32_u32(vsubq_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(old)), vshrq_n_u32::<31>(vreinterpretq_u32_s32(new))));
            let da = vbslq_s32(vld1q_u32(pp.add(v * LANES)), pairs, plain);
            let mask = vaddvq_u32(vandq_u32(vtstq_s32(da, da), bits)) as usize;
            // mask is a sum of distinct bits of 1, 2, 4 and 8, so below 16.
            let pick = vld1q_u8(PACK.get_unchecked(mask).as_ptr());
            vst1q_u32(su.add(k), vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(units), pick)));
            vst1q_s32(sd.add(k), vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(da), pick)));
            k += *POP.get_unchecked(mask) as usize;
            units = vaddq_u32(units, step);
        }
    }
    k
}

/// `block` one lane at a time, where there is no NEON.
///
/// # Safety
///
/// As the NEON `block`.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block(g: *mut i32, w: *const i32, pair: *const u32, b: &Block, dx: i32, su: *mut u32, sd: *mut i32, mut k: usize) -> usize {
    for i in 0..b.vectors as usize * LANES {
        // SAFETY: the caller's bounds.
        unsafe {
            let u = b.unit as usize + i;
            let old = *g.add(u);
            let new = old + *w.add(b.w as usize + i) * dx;
            *g.add(u) = new;
            let da = if *pair.add(u) != 0 { ((old as u32) >> 31) as i32 - ((new as u32) >> 31) as i32 } else { (new + 1).max(0) - (old + 1).max(0) };
            *su.add(k) = u as u32;
            *sd.add(k) = da;
            k += (da != 0) as usize;
        }
    }
    k
}

/// Per 4-bit lane mask, its set bits.
#[cfg(target_arch = "aarch64")]
const POP: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// Per 4-bit lane mask, the byte shuffle that moves the masked i32 lanes to the front
/// in order (the rest of the register is don't-care).
#[cfg(target_arch = "aarch64")]
const PACK: [[u8; 16]; 16] = {
    let mut t = [[0x80u8; 16]; 16];
    let mut m = 0;
    while m < 16 {
        let (mut lane, mut at) = (0, 0);
        while lane < 4 {
            if m >> lane & 1 == 1 {
                let mut byte = 0;
                while byte < 4 {
                    t[m][at * 4 + byte] = (lane * 4 + byte) as u8;
                    byte += 1;
                }
                at += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
};

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::{value_pattern, value_pattern_scalar, ROUTE_CAP};

    fn scalar(route: &[i32; ROUTE_CAP]) -> Option<usize> {
        let (pattern, ok) = value_pattern_scalar(route);
        ok.then_some(pattern)
    }

    #[test]
    fn the_vector_route_pattern_is_the_scalar_one() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let (mut valid, mut invalid) = (0, 0);
        for round in 0..200_000 {
            let mut route = [0i32; ROUTE_CAP];
            for (i, r) in route.iter_mut().enumerate() {
                *r = (next() % if i < 2 { 16 } else { 2 }) as i32;
            }
            // Every third round puts one route out of its range, by a little or a lot.
            if round % 3 == 0 {
                let i = next() as usize % ROUTE_CAP;
                route[i] = [-1, 2, 16, 17, -16, i32::MIN, i32::MAX, 1 << 20][next() as usize % 8];
            }
            let (pattern, ok) = value_pattern(&route);
            match scalar(&route) {
                Some(want) => {
                    assert!(ok && pattern == want, "{route:?}: {pattern:#x} against {want:#x}");
                    valid += 1;
                }
                None => {
                    assert!(!ok, "{route:?} passed the range test");
                    invalid += 1;
                }
            }
        }
        assert!(valid > 100_000 && invalid > 50_000, "{valid} valid, {invalid} invalid");
    }
}
