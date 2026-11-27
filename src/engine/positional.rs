//! The positional stage: the units an output position computes for itself, built
//! from the template's layers. `run` holds its per-pass `advance` and `drain`.

use super::int::{Cols, IntLayer, Segments, PAIR};
use super::plan::NO_CHANNEL;
use crate::memory::attn::KEY_BITS;
use super::run::{Buckets, Marks};
use super::{Readers, ROUTE_CAP};
use crate::llama::Model;

/// One unit of a positional layer before grouping: bias, gate terms, output channel,
/// down weight and cap.
pub(super) type PosUnit = (i32, Vec<(usize, i32)>, u16, i32, i32);

/// Output positions that compute wires of their own. A position wire
/// (`Circuit::positions`) reads its position's slot identity, so a unit that reads one,
/// or a wire such a unit writes, has its own value at each output position. The
/// template leaves those units out; this stage runs them per position after the shared
/// layers, specialised at each: the position's slot channels are constants folded into
/// the bias, and a unit whose gate cannot reach 1 over its inputs' declared ranges
/// (`Model::wire_ranges`) is left out there. The pass refuses any read wire outside its
/// declared range, which guards that proof. The stage's channels are the shared
/// virtual channels, then per specialised position a block of `stride`: its positional
/// channels, then its route channels.
///
/// The positions fall into parts of consecutive blocks. A position's units read only
/// the shared channels and its own block, so the parts run side by side
/// (`Positional::advance`), each over its own state.
#[derive(Clone)]
pub(super) struct Positional {
    /// Per part, per positional layer (each circuit layer, then the routing layer), its
    /// units in groups of at most 32,767.
    pub(super) parts: Vec<Vec<Vec<IntLayer>>>,
    /// Per block, its part; per part its first block, then the end. A part numbers its
    /// own blocks from 0, after the shared channels.
    pub(super) owner: Vec<u8>,
    pub(super) first: Vec<usize>,
    /// Per channel, a bit per positional layer that reads it.
    pub(super) readers: Vec<Readers>,
    /// Per channel from `base` on, the range its value must stay in.
    pub(super) range: Vec<(i32, i32)>,
    pub(super) base: usize,
    /// The specialised positions (sequence indices), and each sequence index's block.
    pub(super) at: Vec<usize>,
    pub(super) block: Vec<Option<u32>>,
    /// Per shared virtual channel, whether a positional unit writes it (it holds no
    /// shared value: the template never writes it).
    pub(super) positional: Vec<bool>,
    /// Per channel from `base` on, where its value lives: its index into `range` and
    /// `PosState::x`, or `ROUTE_SLOT` plus its index into the flattened
    /// `PosState::route`.
    pub(super) slot: Vec<u32>,
    /// The memory heads whose query reads a positional wire, as (layer, head). Each
    /// position asks for its own address, so the stage serves them (`attach`).
    pub(super) owned: Vec<(usize, usize)>,
    /// Per part, those heads at each of its positions.
    pub(super) heads: Vec<PartHeads>,
    /// Reader bit of the heads at circuit layer `li`: `head_bit + li`.
    pub(super) head_bit: usize,
    /// Per shared virtual channel, its index among the positional ones; a block's width.
    pub(super) pc: Vec<u32>,
    pub(super) stride: usize,
}

/// The positional memory heads one part serves: one instance per (head, position).
#[derive(Clone)]
pub(super) struct PartHeads {
    /// Per instance: its layer, and per line bit the stage channel it writes.
    pub(super) inst: Vec<(usize, Vec<u16>)>,
    /// Per circuit layer and stage channel, that layer's instances' address bits whose form reads it
    /// (`instance · KEY_BITS + bit`), with the weight; and each bit's constant.
    pub(super) terms: Vec<Cols>,
    pub(super) pairs: Vec<Vec<(usize, u16, i32)>>,
    pub(super) consts: Vec<i32>,
    /// Per circuit layer, its instances.
    pub(super) by_layer: Vec<Vec<u32>>,
}

/// A memory head as `Template::check` finds it before the split: its layer and head,
/// the virtual channels its query reads and those it writes.
pub(super) struct HeadIo {
    pub(super) at: (usize, usize),
    pub(super) reads: Vec<usize>,
    pub(super) writes: Vec<usize>,
}

/// Marks a `Positional::slot` that is a route total.
pub(super) const ROUTE_SLOT: u32 = 1 << 31;

/// The positional stage's state after the last pass, per part.
#[derive(Default)]
pub(super) struct PosState {
    pub(super) parts: Vec<PosPart>,
    /// The memory image's generation as the positional heads last read it.
    pub(super) seen_gen: u64,
}

/// One part's state and scratch: its wires and route totals (indexed as every part's,
/// holding only its own blocks'), its gates per layer and group, its marks and buckets
/// over the stage's channels, and what its last pass did.
pub(super) struct PosPart {
    pub(super) part: usize,
    pub(super) x: Vec<i32>,
    pub(super) g: Vec<Vec<Vec<i32>>>,
    pub(super) route: Vec<[i32; ROUTE_CAP]>,
    pub(super) m: Marks,
    pub(super) q: Buckets,
    /// The positions whose route totals moved, the units updated, and whether every
    /// wire stayed in its range.
    pub(super) stale: Vec<u32>,
    pub(super) units: u64,
    pub(super) ok: bool,
    /// Per positional head instance: its address bits' forms, the line it last read and
    /// that line's bits as its channels hold them, and whether a query channel moved.
    pub(super) head_bits: Vec<i32>,
    pub(super) head_addr: Vec<u32>,
    pub(super) head_out: Vec<u64>,
    pub(super) head_dirty: Vec<bool>,
    /// This pass's distinct reads, mapping (layer, line) to the line's bits, so equal
    /// queries read once.
    pub(super) read: Vec<((usize, u32), u64)>,
    pub(super) reads: u64,
}

/// The number of parts the positional stage splits into: `GBCFORMER_POS_PARTS`, 4 by
/// default natively, and 1 where there are no threads.
fn parts_wanted() -> usize {
    if cfg!(target_arch = "wasm32") {
        return 1;
    }
    std::env::var("GBCFORMER_POS_PARTS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 64)
}

impl Positional {
    /// Takes the positional units out of `circuit` and `routing` (layers over virtual
    /// channels), or returns None when no unit reads an output slot.
    pub(super) fn split(m: &Model, circuit: &mut [IntLayer], routing: &mut IntLayer, seg: &mut Segments, route_index: &[u8], routes: usize, heads: &[HeadIo]) -> Result<Option<Positional>, String> {
        let (cfg, lay) = (&m.cfg, &m.layout);
        let slots = lay.output_ids(cfg.n_inputs);
        let dv = seg.count;
        let mut positional = vec![false; dv];
        let mut taken: Vec<Vec<bool>> = Vec::new();
        let mut owned = Vec::new();
        for (li, l) in circuit.iter().enumerate() {
            // A head at this layer writes before its units read. A head whose query reads a
            // positional wire has its own line at each position.
            let own: Vec<&HeadIo> = heads.iter().filter(|h| h.at.0 == li && h.reads.iter().any(|&c| positional[c])).collect();
            for h in own {
                owned.push(h.at);
                for &c in &h.writes {
                    positional[c] = true;
                }
            }
            let n = l.bias().len();
            let reads = |u: usize, positional: &[bool]| l.rows().get(u).iter().any(|&(ch, _)| slots.contains(&(ch as usize)) || positional[ch as usize]);
            let mut is: Vec<bool> = (0..n).map(|u| l.c()[u] != 0 && reads(u, &positional)).collect();
            for u in (0..n).filter(|&u| is[u]) {
                positional[l.out()[u] as usize] = true;
            }
            // A node's units write one channel; all of them move with it.
            for u in 0..n {
                is[u] |= l.c()[u] != 0 && positional[l.out()[u] as usize];
            }
            taken.push(is);
        }
        if !taken.iter().flatten().any(|&b| b) {
            return Ok(None);
        }
        let route_taken: Vec<bool> = (0..routing.bias().len()).map(|u| routing.c()[u] != 0 && routing.rows().get(u).iter().any(|&(ch, _)| positional[ch as usize])).collect();
        // Each identity channel's slot, among the stream's own (`Layout::slot_of_id`).
        let slot_at: Vec<Option<usize>> = slots.clone().map(|ch| lay.slot_of_id(ch, cfg.n_inputs)).collect();
        let slot_of = |ch: usize| slot_at[ch - slots.start];
        let mut at: Vec<usize> = Vec::new();
        for u in (0..routing.bias().len()).filter(|&u| route_taken[u]) {
            let own: Vec<usize> = routing.rows().get(u).iter().filter(|&&(ch, _)| slots.contains(&(ch as usize))).filter_map(|&(ch, _)| slot_of(ch as usize)).collect();
            if own.len() != 1 {
                return Err(format!("positional routing unit {u} reads {} slot identities", own.len()));
            }
            at.push(own[0]);
        }
        at.sort_unstable();
        at.dedup();

        // Declared ranges by virtual channel: occupant k of a wire channel is wire
        // `occupants[k]`, which the k-th clearing started.
        let mut declared = vec![(-1i32, 1i32); dv];
        let base_w = lay.wire_base();
        for (ch, occ) in lay.occupants.iter().enumerate().map(|(i, o)| (base_w + i, o)).filter(|(ch, o)| !(o.is_empty() && seg.later[*ch].is_empty())) {
            if occ.len() != seg.later[ch].len() + 1 {
                return Err(format!("channel {ch} holds {} wires and {} clearings", occ.len(), seg.later[ch].len()));
            }
            for (k, s) in occ.iter().enumerate() {
                let v = if k == 0 { seg.first[ch] } else { seg.later[ch][k - 1].1 };
                let (lo, hi) = m.wire_ranges[s.0 as usize];
                declared[v as usize] = (lo as i32, hi as i32);
            }
        }
        // Positions without a stage of their own run no positional unit, so a positional
        // routing unit must stay silent there, with its slot identity 0.
        for u in (0..routing.bias().len()).filter(|&u| route_taken[u]) {
            let max = routing.rows().get(u).iter().filter(|&&(ch, _)| !slots.contains(&(ch as usize))).fold(routing.bias()[u] as i64, |m, &(ch, w)| {
                let (lo, hi) = declared[ch as usize];
                m + (w as i64 * lo as i64).max(w as i64 * hi as i64)
            });
            if max > 0 {
                return Err(format!("positional routing unit {u} can fire at a position without a stage"));
            }
        }

        let list: Vec<usize> = (0..dv).filter(|&c| positional[c]).collect();
        let mut pc = vec![u32::MAX; dv];
        for (i, &c) in list.iter().enumerate() {
            pc[c] = i as u32;
        }
        let n_pc = list.len();
        let stride = n_pc + routes;
        // At least as many parts as keep a part's channels within the tables' u16 index.
        let need = (1..=at.len().max(1)).find(|&p| dv + at.len().div_ceil(p) * stride <= 1 << 16).unwrap_or(at.len());
        let n_parts = parts_wanted().max(need).min(at.len()).max(1);
        let part_of = |i: usize| i * n_parts / at.len();
        // Each part numbers its own blocks from the shared channels on. A position reads
        // only the shared channels and its own block, so two parts' channels never meet.
        let first: Vec<usize> = (0..=n_parts).map(|p| (0..at.len()).find(|&i| part_of(i) >= p).unwrap_or(at.len())).collect();
        let blocks = (0..n_parts).map(|p| first[p + 1] - first[p]).max().unwrap_or(0);
        let total = dv + blocks * stride;
        if total > 1 << 16 {
            return Err(format!("{total} positional channels in a part; the tables index up to 65536"));
        }
        let mut block = vec![None; cfg.seq()];
        for (i, &q) in at.iter().enumerate() {
            block[q] = Some(i as u32);
        }
        let mut readers: Vec<Readers> = vec![0; total];
        let mut shared_read = vec![false; dv];
        let n_layers = circuit.len() + 1;
        let owner: Vec<u8> = (0..at.len()).map(|i| part_of(i) as u8).collect();
        let mut parts: Vec<Vec<Vec<IntLayer>>> = vec![Vec::new(); n_parts];
        for (k, part) in (0..n_layers).flat_map(|k| (0..n_parts).map(move |p| (k, p))) {
            let (l, is) = if k < circuit.len() { (&circuit[k], &taken[k]) } else { (&*routing, &route_taken) };
            let mut groups: Vec<Vec<PosUnit>> = vec![Vec::new()];
            for (i, &q) in at.iter().enumerate().filter(|&(i, _)| part_of(i) == part) {
                let blk = dv + (i - first[part]) * stride;
                for u in (0..l.bias().len()).filter(|&u| is[u]) {
                    let (mut bias, mut max, mut terms) = (l.bias()[u], l.bias()[u] as i64, Vec::new());
                    for &(ch, w) in l.rows().get(u) {
                        let c = ch as usize;
                        if slots.contains(&c) {
                            if slot_of(c) == Some(q) {
                                bias += w;
                                max += w as i64;
                            }
                            continue;
                        }
                        // The proof below reads this channel's declared range, so the pass
                        // must hold it there, whether or not the unit survives.
                        shared_read[c] |= !positional[c];
                        let (lo, hi) = declared[c];
                        max += (w as i64 * lo as i64).max(w as i64 * hi as i64);
                        terms.push((if positional[c] { blk + pc[c] as usize } else { c }, w));
                    }
                    // The unit's activation is max(gate, 0) capped, which is never above 0
                    // here.
                    if max <= 0 {
                        continue;
                    }
                    let o = l.out()[u] as usize;
                    let out = if k < circuit.len() { blk + pc[o] as usize } else { blk + n_pc + route_index[o] as usize };
                    if groups.last().unwrap().len() == PAIR as usize - 1 {
                        groups.push(Vec::new());
                    }
                    groups.last_mut().unwrap().push((bias, terms, out as u16, l.c()[u], l.cap()[u]));
                }
            }
            let built: Vec<IntLayer> = groups.into_iter().filter(|g| !g.is_empty()).map(|g| Self::layer(total, g)).collect();
            for gl in &built {
                for ch in 0..total {
                    if !gl.gate().get(ch).is_empty() {
                        readers[ch] |= 1 << k;
                    }
                }
            }
            parts[part].push(built);
        }
        // The pruning proofs read the shared channels' declared ranges, so the pass's
        // range guard (`seg.range`) narrows to them.
        for c in (0..dv).filter(|&c| shared_read[c]) {
            let (lo, hi) = declared[c];
            seg.range[c] = (seg.range[c].0.max(lo), seg.range[c].1.min(hi));
        }
        let mut range = vec![(i32::MIN, i32::MAX); total - dv];
        for i in 0..blocks {
            for (j, &c) in list.iter().enumerate() {
                range[i * stride + j] = (declared[c].0.max(seg.range[c].0), declared[c].1.min(seg.range[c].1));
            }
        }
        for (li, l) in circuit.iter_mut().enumerate() {
            *l = l.without(&taken[li], dv);
        }
        *routing = routing.without(&route_taken, dv);
        let slot = (0..total - dv).map(|off| if off % stride < n_pc { off as u32 } else { ROUTE_SLOT | ((off / stride) * ROUTE_CAP + off % stride - n_pc) as u32 }).collect();
        if n_layers + circuit.len() > Readers::BITS as usize {
            return Err(format!("{} positional layers and as many head buckets; the reader masks hold {}", n_layers, Readers::BITS));
        }
        let heads = vec![PartHeads { inst: Vec::new(), terms: vec![Cols::from_pairs(total, Vec::new()); circuit.len()], pairs: vec![Vec::new(); circuit.len()], consts: Vec::new(), by_layer: vec![Vec::new(); circuit.len()] }; n_parts];
        Ok(Some(Positional { parts, owner, first, readers, range, base: dv, at, block, positional, slot, owned, heads, head_bit: n_layers, pc, stride }))
    }

    /// Serves memory head `h` at circuit layer `li` at every specialised position: its
    /// address bits (virtual channels, weights, constants) and the virtual channel each
    /// line bit writes (`NO_CHANNEL` for none), all of them positional.
    pub(super) fn attach(&mut self, li: usize, bits: &[(Vec<(u16, i32)>, i32)], out: &[u16]) -> Result<(), String> {
        let dv = self.base;
        if out.iter().any(|&o| o != NO_CHANNEL && !self.positional[o as usize]) {
            return Err(format!("positional memory head at layer {li} writes a shared channel"));
        }
        let mut pairs: Vec<Vec<(usize, u16, i32)>> = vec![Vec::new(); self.heads.len()];
        for i in 0..self.at.len() {
            let part = self.owner[i] as usize;
            let blk = dv + (i - self.first[part]) * self.stride;
            let at = |c: u16| if self.positional[c as usize] { blk + self.pc[c as usize] as usize } else { c as usize };
            let ph = &mut self.heads[part];
            let n = ph.inst.len();
            if (n + 1) * KEY_BITS > PAIR as usize {
                return Err(format!("{} positional head instances in a part; the bit tables index up to {}", n + 1, PAIR as usize / KEY_BITS));
            }
            for (b, (terms, c)) in bits.iter().enumerate() {
                for &(ch, w) in terms {
                    pairs[part].push((at(ch), (n * KEY_BITS + b) as u16, w));
                    self.readers[at(ch)] |= 1 << (self.head_bit + li);
                }
                ph.consts.push(*c);
            }
            ph.inst.push((li, out.iter().map(|&o| if o == NO_CHANNEL { NO_CHANNEL } else { at(o) as u16 }).collect()));
            ph.by_layer[li].push(n as u32);
        }
        let total = self.base + self.range.len();
        for (ph, new) in self.heads.iter_mut().zip(pairs) {
            ph.pairs[li].extend(new);
            ph.terms[li] = Cols::from_pairs(total, ph.pairs[li].clone());
        }
        Ok(())
    }

    pub(super) fn layer(channels: usize, units: Vec<PosUnit>) -> IntLayer {
        let n = units.len();
        let mut rows = Vec::new();
        let (mut bias, mut out, mut c, mut cap) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for (u, (b, terms, o, cu, cp)) in units.into_iter().enumerate() {
            rows.extend(terms.into_iter().map(|(ch, w)| (u, ch as u16, w)));
            bias.push(b);
            out.push(o);
            c.push(cu);
            cap.push(cp);
        }
        IntLayer::new(channels, rows, bias, out, c, cap)
    }

    pub(super) fn fresh(&self) -> PosState {
        let total = self.base + self.range.len();
        let part = |(p, layers): (usize, &Vec<Vec<IntLayer>>)| PosPart {
            part: p,
            x: vec![0; self.range.len()],
            g: layers.iter().map(|gs| gs.iter().map(IntLayer::start).collect()).collect(),
            route: vec![[0; ROUTE_CAP]; self.first[p + 1] - self.first[p]],
            m: Marks { chan: vec![false; total], acc: vec![0; total], ..Marks::default() },
            q: Buckets::new(self.readers.iter().copied()),
            stale: Vec::new(),
            units: 0,
            ok: true,
            head_bits: self.heads[p].consts.clone(),
            head_addr: vec![u32::MAX; self.heads[p].inst.len()],
            head_out: vec![0; self.heads[p].inst.len()],
            head_dirty: vec![true; self.heads[p].inst.len()],
            read: Vec::new(),
            reads: 0,
        };
        PosState { parts: self.parts.iter().enumerate().map(part).collect(), seen_gen: u64::MAX }
    }

    /// Block `b`'s route totals.
    pub(super) fn route<'a>(&self, s: &'a PosState, b: usize) -> &'a [i32; ROUTE_CAP] {
        let p = self.owner[b] as usize;
        &s.parts[p].route[b - self.first[p]]
    }
}
