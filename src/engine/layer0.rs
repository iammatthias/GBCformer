//! Layer 0: the per-slot q, k and v cache, the interned softmax plans, and the diff of
//! the shared attention output into the circuit's layer-0 input.

use super::{Engine, INV};
use crate::llama::{apply_rope, attn_scale, tok, Config, HEAD_DIM};

/// The nonzero cells of a `rows × d` projection, by row with channels ascending, so a
/// product sums the same terms in the same order as a walk of the dense row that skips
/// zero weights.
pub(super) struct Sparse {
    pub(super) starts: Vec<u32>,
    pub(super) cells: Vec<(u16, f32)>,
}

impl Sparse {
    pub(super) fn new(w: &[f32], rows: usize, d: usize) -> Sparse {
        assert!(d <= u16::MAX as usize + 1, "hidden size {d} does not fit a u16 channel");
        let mut starts = Vec::with_capacity(rows + 1);
        let mut cells = Vec::new();
        for o in 0..rows {
            starts.push(cells.len() as u32);
            cells.extend((0..d).filter(|&c| w[o * d + c] != 0.0).map(|c| (c as u16, w[o * d + c])));
        }
        starts.push(cells.len() as u32);
        Sparse { starts, cells }
    }

    /// `w · x` row by row, skipping zero terms.
    pub(super) fn apply(&self, x: &[f32]) -> Vec<f32> {
        self.starts
            .windows(2)
            .map(|r| self.cells[r[0] as usize..r[1] as usize].iter().fold(0.0, |acc, &(c, w)| if x[c as usize] != 0.0 { acc + w * x[c as usize] } else { acc }))
            .collect()
    }
}

pub(super) struct Qkv {
    pub(super) q_id: u32,
    pub(super) k_id: u32,
    /// Per head, an id shared by bit-identical blocks of v.
    pub(super) v_ids: Vec<u32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) heads: Vec<usize>,
}

/// Where the protocol places a token.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Home {
    /// At one position: an input slot's bytes and mask, or an output slot's mask.
    At(u16),
    /// At any output position: the shared output byte tokens.
    Outputs,
    Nowhere,
}

/// The (token, position) pairs the protocol can present, numbered compactly, so the
/// per-pair caches hold seq · 257 entries rather than vocab · seq.
pub(super) struct Slots {
    pub(super) home: Vec<Home>,
    /// Per token, its first slot minus its first position: (t, p) is `base[t] + p`.
    pub(super) base: Vec<u32>,
    pub(super) n_inputs: usize,
    pub(super) len: usize,
}

impl Slots {
    pub(super) fn new(cfg: &Config) -> Slots {
        let (mut home, mut base, mut len) = (vec![Home::Nowhere; cfg.vocab], vec![0u32; cfg.vocab], 0);
        let mut place = |t: usize, h: Home, first: usize, count: usize| {
            home[t] = h;
            base[t] = (len - first) as u32;
            len += count;
        };
        for j in 0..cfg.n_inputs {
            for v in 0..=255 {
                place(tok::input(cfg, j, v), Home::At(j as u16), j, 1);
            }
            place(tok::input_mask(cfg, j), Home::At(j as u16), j, 1);
        }
        for v in 0..=255 {
            place(tok::output(cfg, v), Home::Outputs, cfg.n_inputs, cfg.n_outputs);
        }
        for j in cfg.n_inputs..cfg.seq() {
            place(tok::output_mask(cfg, j), Home::At(j as u16), j, 1);
        }
        Slots { home, base, n_inputs: cfg.n_inputs, len }
    }

    /// The slot of a pair `valid` accepts.
    pub(super) fn of(&self, t: usize, p: usize) -> usize {
        self.base[t] as usize + p
    }

    fn valid(&self, t: usize, p: usize) -> bool {
        match self.home[t] {
            Home::At(h) => h as usize == p,
            Home::Outputs => p >= self.n_inputs,
            Home::Nowhere => false,
        }
    }
}

/// Per head, the nonzero softmax weights in key order.
pub(super) type Weights = Vec<Vec<(usize, f32)>>;

/// Interned softmax weights with, per key position, the heads that read it.
/// Identical weights share one allocation.
pub(super) struct Interned {
    pub(super) weights: Weights,
    pub(super) readers: Vec<Vec<usize>>,
    /// Whether every head reads exactly one key and every gathered channel is fed by
    /// heads of a single key. Then a position's wires are a function of its token
    /// alone, cached in `wires` by slot; empty for other plans. A slot's list holds every
    /// channel its position can reach, by channel, zeros kept: two tokens of one position
    /// hold the same channels in the same order (`Engine::wire_list`).
    pub(super) exclusive: bool,
    pub(super) wires: Vec<Option<Box<[(u16, i32)]>>>,
}

/// An index into `Engine::interned`, or None when the decoded positions' logits
/// differ, so they do not share one attention output.
pub(super) type AttentionPlan = Option<u32>;

impl Engine {
    /// Roped q and k, v, and the heads where k is nonzero, for a token at a position,
    /// computed as the dense pass does from a dominated embedding. None when the
    /// protocol never places the token there, so the pair has no slot.
    pub(super) fn qkv(&mut self, t: usize, p: usize) -> Option<()> {
        if !self.slots.valid(t, p) {
            return None;
        }
        let slot = self.slots.of(t, p);
        if self.qkv[slot].is_some() {
            return Some(());
        }
        let (d, a) = (self.dense.cfg.hidden, self.dense.cfg.attn());
        let x: Vec<f32> = self.dense.embed[t * d..][..d].iter().map(|v| v * INV).collect();
        let rope_at = |row: Vec<f32>| apply_rope(row, &self.cos[p * HEAD_DIM..][..HEAD_DIM], &self.sin[p * HEAD_DIM..][..HEAD_DIM], a);
        let (q, k, v) = (rope_at(self.l0[0].apply(&x)), rope_at(self.l0[1].apply(&x)), self.l0[2].apply(&x));
        let live = self.plan.as_ref().ok().map(|p| &p.t.live_heads);
        let heads = (0..a / HEAD_DIM).filter(|&h| live.is_none_or(|l| l[h]) && k[h * HEAD_DIM..][..HEAD_DIM].iter().any(|&x| x != 0.0)).collect();
        let mut id = |x: &[f32]| {
            let n = self.vector_ids.len() as u32;
            *self.vector_ids.entry(x.iter().map(|v| v.to_bits()).collect()).or_insert(n)
        };
        let (q_id, k_id) = (id(&q), id(&k));
        let v_ids = v.chunks(HEAD_DIM).map(|b| id(b)).collect();
        self.qkv[slot] = Some(Box::new(Qkv { q_id, k_id, v_ids, q, k, v, heads }));
        Some(())
    }

    /// q·k over one head, scaled. It skips zero products, since they cannot change a
    /// nonzero sum, and a zero logit's sign never reaches the softmax.
    fn logit(q: &[f32], k: &[f32], head: usize) -> f32 {
        let r = head * HEAD_DIM..(head + 1) * HEAD_DIM;
        q[r.clone()].iter().zip(&k[r]).fold(0.0, |acc, (&a, &b)| if a != 0.0 && b != 0.0 { acc + a * b } else { acc }) * attn_scale()
    }

    /// The first decoded position's layer-0 softmax weights, provided every decoded
    /// position has bit-identical logits. Keys whose head block is zero have logit ±0
    /// and are dropped only when their shared exponential is exactly 0.
    fn plan_attention(qkv: &[&Qkv], positions: &[usize], live_heads: &[bool]) -> Option<Weights> {
        let mut live: Vec<Vec<usize>> = vec![Vec::new(); qkv[0].q.len() / HEAD_DIM];
        for (j, e) in qkv.iter().enumerate() {
            for &h in &e.heads {
                live[h].push(j);
            }
        }
        let p0 = positions[0];
        // A bit-identical query (the same id) has bit-identical logits.
        let same = |p: usize| qkv[p].q_id == qkv[p0].q_id || live.iter().enumerate().all(|(h, keys)| keys.iter().all(|&j| Self::logit(&qkv[p].q, &qkv[j].k, h).to_bits() == Self::logit(&qkv[p0].q, &qkv[j].k, h).to_bits()));
        if !positions[1..].iter().all(|&p| same(p)) {
            return None;
        }
        live.iter()
            .enumerate()
            .map(|(h, keys)| {
                // A dead head (its output channels read by no layer) gets no weights.
                if !live_heads[h] {
                    return Some(Vec::new());
                }
                let logit = |j: usize| Self::logit(&qkv[p0].q, &qkv[j].k, h);
                let zero_keys = keys.len() < qkv.len();
                let m = keys.iter().map(|&j| logit(j)).fold(if zero_keys { 0.0 } else { f32::NEG_INFINITY }, f32::max);
                if zero_keys && (0.0 - m).exp() != 0.0 {
                    return None;
                }
                let z: f32 = keys.iter().map(|&j| (logit(j) - m).exp()).sum();
                Some(keys.iter().map(|&j| (j, (logit(j) - m).exp() / z)).filter(|&(_, w)| w != 0.0).collect())
            })
            .collect()
    }

    /// Builds, once per slot, the wires position `j`'s token `t` drives under the exclusive
    /// plan `p`: every channel fed by a value dim the position can reach (`v_reach`), by
    /// channel, with the value the token gives it, zeros kept. A position no head of the
    /// plan reads has no list. None when a wire is not the integer the circuit reads.
    #[allow(clippy::too_many_arguments)]
    fn wire_list(p: &mut Interned, plan: &super::plan::Template, slots: &Slots, reach: &[Vec<usize>], qkv: &[Option<Box<Qkv>>], a: usize, j: usize, t: usize) -> Option<()> {
        let slot = slots.of(t, j);
        if p.readers[j].is_empty() || p.wires[slot].is_some() {
            return Some(());
        }
        let v = &qkv[slot].as_deref().expect("a placed token has its vectors").v;
        let mut out = vec![0f32; a];
        // Exclusive: in every dim j's token can reach, j is the only key that can, so the
        // other keys of a group head add nothing there.
        for &h in &p.readers[j] {
            for &(_, w) in p.weights[h].iter().filter(|&&(k, _)| k == j) {
                for d in h * HEAD_DIM..(h + 1) * HEAD_DIM {
                    out[d] += w * v[d];
                }
            }
        }
        let mut list: Vec<(u16, i32)> = Vec::new();
        for &h in &p.readers[j] {
            for d in (h * HEAD_DIM..(h + 1) * HEAD_DIM).filter(|d| reach[*d].contains(&j)) {
                for &(ch, _) in &plan.o_cols[d] {
                    if list.iter().any(|&(c, _)| c == ch) {
                        continue;
                    }
                    let v = plan.o_rows[ch as usize].iter().fold(0.0f32, |acc, &(i, w)| if out[i as usize] != 0.0 { acc + w * out[i as usize] } else { acc });
                    if v.abs() > GATHER_MAX || v.fract() != 0.0 {
                        return None;
                    }
                    list.push((ch, v as i32));
                }
            }
        }
        list.sort_unstable_by_key(|&(ch, _)| ch);
        p.wires[slot] = Some(list.into_boxed_slice());
        Some(())
    }

    /// The changes to the circuit's layer-0 input (the shared attention output in its
    /// wire channels, as exact integers: token bits and nibbles) since the last fast pass. With
    /// the same attention plan only heads reading a changed token are recomputed;
    /// otherwise the whole output is, and diffed against the state.
    pub(super) fn layer0_changes(&mut self, tokens: &[usize], positions: &[usize]) -> Option<()> {
        let a = self.dense.cfg.attn();
        let continuing = self.state.valid && self.state.tokens.len() == tokens.len();
        self.moved.clear();
        // With a view, a position whose token differs from the state's was rewritten by a
        // place since the last diff, so it is in `dirty`.
        match (continuing, self.view.as_mut()) {
            (true, Some(v)) => self.moved.extend(v.dirty.iter().copied().filter(|&j| tokens[j] != self.state.tokens[j])),
            _ => self.moved.extend((0..tokens.len()).filter(|&j| !continuing || tokens[j] != self.state.tokens[j])),
        }
        if let Some(v) = self.view.as_mut() {
            v.clean();
        }
        self.stats.layer0_moved += self.moved.len() as u64;
        for i in 0..self.moved.len() {
            let j = self.moved[i];
            self.qkv(tokens[j], j)?;
        }
        let moved = &self.moved;
        let plan = &self.plan.as_ref().ok()?.t;
        let slots = &self.slots;
        let reach = &self.v_reach;
        let qkv = |j: usize, t: usize| self.qkv[slots.of(t, j)].as_deref().unwrap();

        // The last pass's attention plan stays valid while no moved position's key or
        // query id changed and every decoded position is one its logits were checked
        // equal at. A plan checked for a list of positions is valid for any part of it, so the
        // request and result passes (the result positions a suffix of the request
        // pass's) share the plan built for the longer list.
        let ids = |e: &Qkv| (e.k_id, e.q_id);
        let s = &mut self.state;
        let planned = continuing && s.weights.is_some();
        // The first moved position whose token breaks the plan.
        let broke = moved.iter().copied().filter(|_| planned).find(|&j| {
            // A moved key with another id keeps the plan valid when it scores, bit for
            // bit, what the old one scored in every head against the plan's query,
            // since the softmax weights depend on the logits alone.
            let (old, new) = (qkv(j, s.tokens[j]), qkv(j, tokens[j]));
            !(ids(old) == ids(new)
                || (old.q_id == new.q_id
                    && s.plan_positions.first().is_some_and(|&p0| {
                        let q = &qkv(p0, tokens[p0]).q;
                        (0..q.len() / HEAD_DIM).all(|h| Self::logit(q, &old.k, h).to_bits() == Self::logit(q, &new.k, h).to_bits())
                    })))
        });
        let held = planned
            && broke.is_none()
            && ((self.positions_id != 0 && s.plan_id == self.positions_id) || s.plan_positions == positions || {
                // A position whose query is bit-identical to the plan's first position's
                // (the same q id) has bit-identical logits, so the plan is valid there too.
                let verified = &mut s.plan_verified;
                verified.resize(tokens.len(), false);
                let q0 = s.plan_positions.first().map(|&p0| qkv(p0, tokens[p0]).q_id);
                positions.iter().all(|&p| {
                    verified[p] |= q0 == Some(qkv(p, tokens[p]).q_id);
                    verified[p]
                })
            });
        let id = match held {
            true => s.weights,
            false => {
                self.stats.layer0_replans += 1;
                let why = match broke {
                    _ if !planned => 0,
                    Some(j) if j < self.dense.cfg.n_inputs => 1,
                    Some(_) => 2,
                    None => 3,
                };
                self.stats.replan_why[why] += 1;
                let c0 = self.timed.then(super::counters::cpi).flatten();
                // The plan depends only on every position's query and key (their ids name them
                // bit for bit) and on the decoded positions, so a key set seen before reuses its
                // plan. A view's other positions hold one token for good (`View::tokens`), so
                // the stream's own positions tell its key sets apart.
                let mut key = std::mem::take(&mut self.plan_key);
                key.clear();
                let own = self.view.as_ref().map(|v| &v.own[..]);
                key.resize(2 * own.map_or(tokens.len(), <[u32]>::len), 0);
                for (i, ids) in key.chunks_exact_mut(2).enumerate() {
                    let j = own.map_or(i, |own| own[i] as usize);
                    let e = qkv(j, tokens[j]);
                    ids.copy_from_slice(&[e.k_id, e.q_id]);
                }
                key.extend(positions.iter().map(|&p| p as u32 | 1 << 31));
                let hit = self.plan_memo.get(&key[..]).copied();
                self.stats.replan_memo_hits += hit.is_some() as u64;
                self.stats.replan_same += (hit.is_some() && hit == s.weights) as u64;
                if let (Some(c0), Some(c1)) = (c0, super::counters::cpi()) {
                    self.stats.replan_timed += 1;
                    self.stats.replan_key = self.stats.replan_key.plus(c1.since(c0));
                    self.replan_from = Some(c1);
                }
                let made = match hit {
                    Some(id) => Some(id),
                    None => Self::plan_attention(&tokens.iter().enumerate().map(|(j, &t)| qkv(j, t)).collect::<Vec<&Qkv>>(), positions, &plan.live_heads).map(|w| {
                    let bits: Vec<(usize, usize, u32)> = w.iter().enumerate().flat_map(|(h, ws)| ws.iter().map(move |&(j, x)| (h, j, x.to_bits()))).collect();
                    *self.interned_ids.entry(bits).or_insert_with(|| {
                        let mut readers = vec![Vec::new(); tokens.len()];
                        for (h, ws) in w.iter().enumerate() {
                            for &(j, _) in ws {
                                readers[j].push(h);
                            }
                        }
                        // Per value dim, the one key whose token can make it nonzero
                        // (Err when several can): a head gathering a group of slots
                        // reads each slot's lane in its own dims.
                        let key_of = |i: usize| {
                            let mut ks = w[i / HEAD_DIM].iter().map(|&(j, _)| j).filter(|j| reach[i].contains(j));
                            match (ks.next(), ks.next()) {
                                (k, None) => Ok(k),
                                _ => Err(()),
                            }
                        };
                        let exclusive = (0..a).all(|i| key_of(i).is_ok())
                            && plan.gathered.iter().all(|&ch| {
                                let mut keys = plan.o_rows[ch as usize].iter().filter_map(|&(i, _)| key_of(i as usize).ok().flatten());
                                keys.next().is_none_or(|k| keys.all(|j| j == k))
                            });
                        let wires = if exclusive { vec![None; slots.len] } else { Vec::new() };
                        self.interned.push(Interned { weights: w, readers, exclusive, wires });
                        (self.interned.len() - 1) as u32
                    })
                }),
                };
                if let (None, Some(id)) = (hit, made) {
                    if self.plan_memo.len() >= 1 << 16 {
                        self.plan_memo.clear();
                    }
                    self.plan_memo.insert(key.clone(), id);
                }
                self.plan_key = key;
                if made.is_some() {
                    s.plan_positions.clear();
                    s.plan_positions.extend_from_slice(positions);
                    s.plan_id = self.positions_id;
                    s.plan_verified.clear();
                    s.plan_verified.resize(tokens.len(), false);
                    for &p in positions {
                        s.plan_verified[p] = true;
                    }
                }
                made
            }
        };
        let id = id?;
        let s = &mut self.state;
        let same_plan = continuing && s.weights == Some(id);
        // Exclusive plans: diff the moved positions' cached wire lists.
        if same_plan && self.interned[id as usize].exclusive {
            self.stats.layer0_exclusive += 1;
            // An exclusive plan's gathered channel has one key, so the moved positions'
            // channels are disjoint, and two tokens of one position hold the same channels
            // in the same order: each channel's change is one subtraction, kept when it
            // is not 0. A position no head reads (every output slot) contributes no
            // wires. A wire list that cannot be built abandons the pass.
            let interned = &mut self.interned[id as usize];
            let changes = &mut self.marks.changes;
            changes.clear();
            for &j in moved {
                if interned.readers[j].is_empty() {
                    continue;
                }
                let (old, new) = (s.tokens[j], tokens[j]);
                Self::wire_list(interned, plan, slots, reach, &self.qkv, a, j, old)?;
                Self::wire_list(interned, plan, slots, reach, &self.qkv, a, j, new)?;
                let (before, after) = (interned.wires[slots.of(old, j)].as_deref().unwrap(), interned.wires[slots.of(new, j)].as_deref().unwrap());
                if before.len() != after.len() {
                    return None;
                }
                changes.reserve(before.len());
                let (at, mut n, mut same) = (changes.as_mut_ptr(), changes.len(), true);
                for (x, y) in before.iter().zip(after) {
                    same &= x.0 == y.0;
                    let d = y.1 - x.1;
                    // SAFETY: n is below the length at the reserve plus before.len(), which it holds.
                    unsafe { *at.add(n) = (x.0, d) };
                    n += (d != 0) as usize;
                }
                // SAFETY: the first n entries are written.
                unsafe { changes.set_len(n) };
                if !same {
                    return None;
                }
            }
            // Continuing: the state's tokens differ from `tokens` only at the moved positions.
            for &j in moved {
                s.tokens[j] = tokens[j];
            }
            return Some(());
        }
        let interned = &self.interned[id as usize];
        let (weights, readers) = (&interned.weights, &interned.readers);

        // A head's output changes only if one of its keys' value blocks did.
        let heads: Vec<usize> = if same_plan {
            let mut hs: Vec<usize> = moved
                .iter()
                .flat_map(|&j| readers[j].iter().map(move |&h| (h, j)))
                .filter(|&(h, j)| qkv(j, s.tokens[j]).v_ids[h] != qkv(j, tokens[j]).v_ids[h])
                .map(|(h, _)| h)
                .collect();
            hs.sort_unstable();
            hs.dedup();
            hs
        } else {
            s.out = vec![0.0; a];
            (0..weights.len()).collect()
        };
        let mut chans: Vec<u16> = Vec::new();
        let m = &mut self.marks;
        let mark = &mut m.chan;
        for &h in &heads {
            let block = h * HEAD_DIM..(h + 1) * HEAD_DIM;
            s.out[block.clone()].fill(0.0);
            self.stats.layer0_terms += weights[h].len() as u64;
            for &(j, w) in &weights[h] {
                let v = &qkv(j, tokens[j]).v;
                for t in block.clone() {
                    s.out[t] += w * v[t];
                }
            }
            for i in block {
                for &(ch, _) in &plan.o_cols[i] {
                    if !mark[ch as usize] {
                        mark[ch as usize] = true;
                        chans.push(ch);
                    }
                }
            }
        }
        if !same_plan {
            chans = plan.gathered.clone();
        }
        m.changes.clear();
        let mut ok = true;
        for &ch in &chans {
            mark[ch as usize] = false;
            let v = plan.o_rows[ch as usize].iter().fold(0.0f32, |acc, &(i, w)| if s.out[i as usize] != 0.0 { acc + w * s.out[i as usize] } else { acc });
            ok &= v.abs() <= GATHER_MAX && v.fract() == 0.0;
            let old = if continuing && s.x.len() > ch as usize { s.x[ch as usize] } else { 0 };
            if v as i32 != old {
                m.changes.push((ch, v as i32 - old));
            }
        }
        if !ok {
            return None;
        }
        s.weights = Some(id);
        if continuing {
            for &j in moved {
                s.tokens[j] = tokens[j];
            }
        } else {
            s.tokens.clear();
            s.tokens.extend_from_slice(tokens);
        }
        Some(())
    }
}

/// A hasher for the plan memo's keys, lists of u32 ids: eight bytes a step. The map
/// compares whole keys, so the hash only spreads them.
#[derive(Default)]
pub(super) struct Words(u64);

impl std::hash::Hasher for Words {
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            h = (h.rotate_left(5) ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        for &b in chunks.remainder() {
            h = (h.rotate_left(5) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        self.0 = h;
    }

    fn finish(&self) -> u64 {
        // The table reads the hash's low bits and its top seven: fold the halves together.
        (self.0 ^ self.0 >> 32).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// The largest gathered channel value: a token bit is 0/1, an input's nibble
/// (`Circuit::input_nibbles`, `docs/byte-channels.md`) up to 15. The run's range guard
/// then checks each wire against its declared range.
const GATHER_MAX: f32 = 15.0;
