//! The fast-path plan: the tables the integer pass reads, checked from the weights
//! (`Plan::check`) and pruned to one stream's reach (`Plan::prune`).

use super::int::{declared_bounds, norm_bound, Cols, IntLayer, Segments, Tagged, PAIR};
use super::layer0::{Home, Slots};
use super::positional::{HeadIo, Positional};
use super::run::Wire;
use super::view::View;
use super::{Readers, C, NO_TOKEN, ROUTE_CAP};
use crate::llama::{memory_head, Layer, MemorySpec, Model, QueryRows, CONSTANTS, HEAD_DIM, UNIT};
use crate::circuit::{LINE, LINE_BITS};
use crate::memory::attn::{KEY_BITS, KEY_DIMS};

/// A memory head's line bit that no channel holds.
pub(super) const NO_CHANNEL: u16 = u16::MAX;

/// The fast path's tables, built only when the checkpoint has the shape they need: the
/// template, and for a memory checkpoint the memory heads it serves from the index.
#[derive(Clone)]
pub(super) struct Plan {
    pub(super) t: Template,
    pub(super) mem: Option<MemPlan>,
}

impl Plan {
    pub(super) fn check(m: &Model) -> Result<Plan, String> {
        let mut t = Template::check(m, m.memory.as_ref())?;
        let mem = m.memory.as_ref().map(|spec| MemPlan::check(m, spec, &mut t)).transpose()?;
        if let Some(mp) = &mem {
            t.index(&mp.heads);
        }
        Ok(Plan { t, mem })
    }
    /// The plan for one stream (`view`): the units, memory heads and positional stage
    /// that the stream's tokens (its whole vocabulary) and decoded positions can reach,
    /// the rest left out, with the head table giving the stream's own tokens. Virtual
    /// channels have one writer each, so the reach is a walk from the routing units
    /// those tokens own, through each kept unit's and head's read channels, back to
    /// layer 0. What is left out still runs in the dense pass and can change the
    /// stream's tokens two ways only, both refused here: through RMSNorm (the hard
    /// bounds must keep every norm at exactly 2^15), and through a routing unit it
    /// leaves out firing at a decoded position (each must be gated by a token channel
    /// the decoded positions do not carry, so it adds exactly 0 there).
    pub(super) fn prune(&self, view: &View) -> Result<Plan, String> {
        let t = &self.t;
        let (n, dv) = (t.circuit.len(), t.seg.count);
        if t.norm.0 >= t.norm.1 {
            return Err(format!("the declared ranges' squares sum to {}, not below half an ulp of the norm ({})", t.norm.0, t.norm.1));
        }
        let mut keep_route = vec![false; t.routing.bias().len()];
        for tk in view.to.iter().filter(|&&tk| tk != NO_TOKEN) {
            for &Own { u, .. } in &t.own_units[*tk as usize] {
                keep_route[u as usize] = true;
            }
        }
        for (u, keep) in keep_route.iter_mut().enumerate() {
            if t.routing.c()[u] != 0 && t.routing.rows().get(u).iter().all(|&(ch, _)| !t.token[ch as usize]) {
                *keep = true;
            }
        }
        let positions = || view.at[view.cfg.n_inputs..].iter().flatten().map(|&q| q as usize);
        let pos = t.pos.as_ref().filter(|p| positions().any(|q| p.block[q].is_some()));
        // Positional routing units are 0 in the template's routing layer; with the stage
        // left out they run nowhere, so they too must stay 0 at a decoded position.
        if let Some(u) = (0..t.route_gated.len()).find(|&u| !t.route_gated[u] && !(keep_route[u] && t.routing.c()[u] != 0) && (t.routing.c()[u] != 0 || pos.is_none())) {
            return Err(format!("routing unit {u}, left out, reads no token channel, so it could fire at a decoded position"));
        }
        let all_heads = self.mem.as_ref().map_or(&[][..], |mp| &mp.heads[..]);
        let Reach { need, drops, keep_head } = self.reach(&keep_route, pos);
        let circuit: Vec<IntLayer> = t.circuit.iter().zip(&drops).map(|(l, drop)| l.without(drop, dv)).collect();
        let routing = t.routing.without(&keep_route.iter().map(|k| !k).collect::<Vec<_>>(), dv);
        let heads: Vec<MemHead> = all_heads.iter().zip(&keep_head).filter(|(_, k)| **k).map(|(h, _)| h.clone()).collect();
        // Layer 0 too: a gather whose channels no kept unit reads (another part's
        // slots) is left out, so the stream's attention plan has no keys at the
        // positions it holds fixed.
        let o_cols: Vec<Vec<(u16, f32)>> = t.o_cols.iter().map(|col| if col.iter().any(|&(ch, _)| need[ch as usize]) { col.clone() } else { Vec::new() }).collect();
        // A fast pass decodes straight to the stream's tokens; the dense pass gives the
        // checkpoint's, which `View::back` maps.
        let head_table = t.head_table.iter().map(|&tk| view.from[tk as usize]).collect();
        let mut template = Template { circuit, routing, pos: pos.cloned(), o_cols, head_table, ..t.clone() };
        template.pack();
        template.index(&heads);
        let mem = if heads.is_empty() { None } else { Some(MemPlan::new(heads, dv, n + 1, &t.seg.range)?) };
        Ok(Plan { t: template, mem })
    }
}

/// What a set of routing units reads, back to layer 0 (`Plan::reach`): the channels
/// needed, per circuit layer the units left out, and the memory heads kept.
pub(super) struct Reach {
    pub(super) need: Vec<bool>,
    pub(super) drops: Vec<Vec<bool>>,
    pub(super) keep_head: Vec<bool>,
}

impl Plan {
    /// The same plan over only the virtual channels it reads or writes, numbered in their
    /// order. A stream's pruned plan keeps the checkpoint's numbering, in which the other
    /// parts' channels sit between its own: the GBC CPU stream of the one model uses
    /// 2,578 of 13,826, and every per-channel table of every layer was sized and strided
    /// by all of them (`docs/perf-log.md`, "The stream's own channels"). A plan with a
    /// positional stage, whose tables index channels too, keeps its numbering.
    pub(super) fn compact(self) -> Result<Plan, String> {
        if self.t.pos.is_some() {
            return Ok(self);
        }
        let t = &self.t;
        let dv = t.seg.count;
        let heads = self.mem.as_ref().map_or(&[][..], |mp| &mp.heads[..]);
        let mut used = vec![false; dv];
        let mut mark = |ch: u16| {
            if ch != NO_CHANNEL {
                used[ch as usize] = true;
            }
        };
        for l in t.circuit.iter().chain([&t.routing]) {
            for u in 0..l.bias().len() {
                if l.c()[u] != 0 {
                    mark(l.out()[u]);
                }
                for &(ch, _) in l.rows().get(u) {
                    mark(ch);
                }
            }
        }
        for &(ch, _) in t.o_cols.iter().flatten() {
            mark(ch);
        }
        for h in heads {
            for &(ch, _) in h.bits.iter().flat_map(|(terms, _)| terms) {
                mark(ch);
            }
            for &o in &h.out {
                mark(o);
            }
        }
        let mut to = vec![NO_CHANNEL; dv];
        let mut n = 0usize;
        for ch in (0..dv).filter(|&ch| used[ch]) {
            to[ch] = n as u16;
            n += 1;
        }
        let map = |ch: u16| if ch == NO_CHANNEL { NO_CHANNEL } else { to[ch as usize] };
        let layer = |l: &IntLayer| {
            let rows = (0..l.bias().len()).flat_map(|u| l.rows().get(u).iter().map(move |&(ch, w)| (u, map(ch), w))).collect();
            // A unit that writes nothing sends nothing: its output channel is any.
            let out = (0..l.bias().len()).map(|u| if l.c()[u] != 0 { map(l.out()[u]) } else { 0 }).collect();
            IntLayer::new(n, rows, l.bias().to_vec(), out, l.c().to_vec(), l.cap().to_vec())
        };
        let kept = |v: &[bool]| (0..dv).filter(|&ch| used[ch]).map(|ch| v[ch]).collect::<Vec<bool>>();
        let seg = Segments {
            first: t.seg.first.iter().map(|&v| map(v)).collect(),
            later: t.seg.later.iter().map(|occ| occ.iter().map(|&(e, v)| (e, map(v))).collect()).collect(),
            count: n,
            range: (0..dv).filter(|&ch| used[ch]).map(|ch| t.seg.range[ch]).collect(),
        };
        let heads: Vec<MemHead> = heads.iter().map(|h| MemHead { layer: h.layer, bits: h.bits.iter().map(|(terms, c)| (terms.iter().map(|&(ch, w)| (map(ch), w)).collect(), *c)).collect(), out: h.out.iter().map(|&o| map(o)).collect() }).collect();
        let mut template = Template {
            circuit: t.circuit.iter().map(layer).collect(),
            routing: layer(&t.routing),
            route_index: (0..dv).filter(|&ch| used[ch]).map(|ch| t.route_index[ch]).collect(),
            token: kept(&t.token),
            o_cols: t.o_cols.iter().map(|col| col.iter().map(|&(ch, w)| (map(ch), w)).collect()).collect(),
            seg,
            ..t.clone()
        };
        template.pack();
        template.index(&heads);
        let mem = if heads.is_empty() { None } else { Some(MemPlan::new(heads, n, t.circuit.len() + 1, &template.seg.range)?) };
        Ok(Plan { t: template, mem })
    }
}

/// What the decoded positions of one pass can reach (`Plan::demand`): its routing units
/// and their `Reach`.
pub(super) struct Demand {
    pub(super) route: Vec<bool>,
    pub(super) reach: Reach,
}

impl Plan {
    /// What decoding `tokens` (one per decoded position) can reach: the routing units
    /// those tokens own and every routing unit no token channel gates, walked back.
    pub(super) fn demand(&self, tokens: impl Iterator<Item = usize>) -> Demand {
        let t = &self.t;
        let mut route = vec![false; t.routing.bias().len()];
        for tk in tokens {
            for &Own { u, .. } in &t.own_units[tk] {
                route[u as usize] = true;
            }
        }
        for (u, keep) in route.iter_mut().enumerate() {
            if t.routing.c()[u] != 0 && t.routing.rows().get(u).iter().all(|&(ch, _)| !t.token[ch as usize]) {
                *keep = true;
            }
        }
        let reach = self.reach(&route, t.pos.as_ref());
        Demand { route, reach }
    }

    /// The walk back from the routing units `keep_route` keeps (and the positional
    /// stage's readers): every circuit unit writing a needed channel is kept and needs
    /// what it reads, and so is every memory head writing one, with its query's channels.
    /// Virtual channels have one writer each, so one walk from the last layer suffices.
    pub(super) fn reach(&self, keep_route: &[bool], pos: Option<&Positional>) -> Reach {
        let t = &self.t;
        let (n, dv) = (t.circuit.len(), t.seg.count);
        let mut need = vec![false; dv];
        for u in (0..keep_route.len()).filter(|&u| keep_route[u]) {
            for &(ch, _) in t.routing.rows().get(u) {
                need[ch as usize] = true;
            }
        }
        if let Some(p) = pos {
            for (ch, nd) in need.iter_mut().enumerate() {
                *nd |= p.readers[ch] != 0;
            }
        }
        let all_heads = self.mem.as_ref().map_or(&[][..], |mp| &mp.heads[..]);
        let mut keep_head = vec![false; all_heads.len()];
        let mut drops: Vec<Vec<bool>> = t.circuit.iter().map(|l| vec![true; l.bias().len()]).collect();
        for li in (0..n).rev() {
            let l = &t.circuit[li];
            for u in 0..l.bias().len() {
                if l.c()[u] != 0 && need[l.out()[u] as usize] {
                    drops[li][u] = false;
                }
            }
            for u in (0..l.bias().len()).filter(|&u| !drops[li][u]) {
                for &(ch, _) in l.rows().get(u) {
                    need[ch as usize] = true;
                }
            }
            for &h in self.mem.as_ref().map_or(&[][..], |mp| &mp.layer_heads[li][..]) {
                let head = &all_heads[h as usize];
                if head.out.iter().any(|&o| o != NO_CHANNEL && need[o as usize]) {
                    keep_head[h as usize] = true;
                    for &(ch, _) in head.bits.iter().flat_map(|(terms, _)| terms) {
                        need[ch as usize] = true;
                    }
                }
            }
        }
        Reach { need, drops, keep_head }
    }
}

/// One memory head as the engine serves it: the address its query names, formed from
/// wire channels, and the eight channels its byte lands in.
#[derive(Clone)]
pub(super) struct MemHead {
    pub(super) layer: usize,
    /// Per address bit: (channel, weight) terms and a constant; the form is 0 or 1.
    pub(super) bits: Vec<(Vec<(u16, i32)>, i32)>,
    /// Per line bit, the channel it writes, or `NO_CHANNEL` for a bit no node reads.
    pub(super) out: Vec<u16>,
}

/// The memory heads and the argument that the index equals their attention result:
/// every head has the record pattern (`llama::memory_head`) with one `Params`, whose
/// margins `Params::check` holds at the cache's longest position when a cartridge is
/// inserted; the address channels no head reads from the index are dead in every CPU
/// pass (the address gathers find only mask tokens there), so the copy head's query
/// is the constant 0.
#[derive(Clone)]
pub(super) struct MemPlan {
    pub(super) heads: Vec<MemHead>,
    /// Per channel, two bitsets over `heads`, `words` pairs of u64s per channel. The
    /// first: the column selects the channel gates, and of each address unit whose forms
    /// read it the first head by (layer, index), which forms the unit before any other
    /// reads it. The second: every head of those units. A change of the channel reaches
    /// the first set, and of the second the heads that follow their unit
    /// (`State::head_follows`).
    pub(super) reach: Vec<[u64; 2]>,
    pub(super) words: usize,
    /// The heads without a gate: each follows its unit always.
    pub(super) ungated: Vec<u64>,
    /// Per layer, the indices into `heads` of the heads that read there, and the same
    /// set as `words` u64s.
    pub(super) layer_heads: Vec<Vec<u16>>,
    pub(super) layer_mask: Vec<Vec<u64>>,
    /// The address units, `KEY_BITS` bits a unit: per channel, the bits whose form reads
    /// it and the weight; and each bit's constant. `State::head_bits` keeps the forms'
    /// values from the changes the head bucket routes. A head reads its own unit, or a
    /// column select (`circuit::Query::gated`) the unit of the line it shares with the
    /// other columns of its row.
    pub(super) bit_terms: Cols,
    pub(super) bit_consts: Vec<i32>,
    pub(super) unit_of: Vec<u16>,
    /// Per head, a column select's gate (its virtual channel) and off line. While its
    /// unit's forms are all 0 or 1, its address is the unit's line when the gate is 1 and
    /// the off line when it is 0 (`Engine::memory_reads` holds both at every read, and
    /// forms the address bit by bit when either fails).
    pub(super) gate: Vec<Option<(u16, u32)>>,
    /// Per head, its line bits that land in a channel: a column select's one lane, a
    /// line head's read bytes. The read diffs only these.
    pub(super) live: Vec<u64>,
    /// Per head whose live bits lie in one lane, the bus it drives: (lane, bus). Per bus, its
    /// eight channels (bit order) and its driving heads. A layer's drivers set their
    /// lane bytes, then each bus they moved diffs the sum of its drivers' bytes.
    pub(super) col: Vec<Option<(u8, u16)>>,
    pub(super) buses: Vec<([u16; 8], Vec<u16>)>,
}

impl MemPlan {
    pub(super) fn check(m: &Model, spec: &MemorySpec, t: &mut Template) -> Result<MemPlan, String> {
        let cfg = &m.cfg;
        let (d, a) = (cfg.hidden, cfg.attn());
        let lay = &m.layout;
        let qs = spec.params.s;
        let wire_base = lay.wire_base();
        let is_wire = |ch: usize| ch >= wire_base && ch < lay.after_wires();
        let mut heads = Vec::new();
        let kvd = cfg.kv_dim();
        let mut blank = Layer { q: vec![0.0; a * d], k: vec![0.0; kvd * d], v: vec![0.0; kvd * d], o: vec![0.0; d * a], gate: Vec::new(), up: Vec::new(), down: Vec::new() };
        for &(li, h) in &spec.heads {
            let l = &m.layers[li];
            let base = h * HEAD_DIM;
            let copy = spec.copy_heads.iter().position(|&c| c == (li, h));
            // Recover the query: per line address bit, the wire channels' terms and the
            // constant. A copy head's query is a gathered address instead.
            let mut bits: Vec<(Vec<(usize, i64)>, i64)> = Vec::new();
            for (i, &dim) in KEY_DIMS.iter().enumerate() {
                let row = &l.q[(base + dim) * d..][..d];
                let mut terms = Vec::new();
                for (ch, &w) in row.iter().enumerate().skip(1) {
                    if w == 0.0 {
                        continue;
                    }
                    if let Some(cw) = copy
                        && ch == lay.addr(cw, LINE_BITS + i)
                    {
                        continue;
                    }
                    if !is_wire(ch) {
                        return Err(format!("memory head ({li}, {h}) address bit {i} reads channel {ch}, not a wire"));
                    }
                    let wt = (w / (2.0 * qs)).round() as i64;
                    if 2.0 * qs * wt as f32 != w {
                        return Err(format!("memory head ({li}, {h}) address bit {i}: weight {w} on channel {ch} is not 2·S·m"));
                    }
                    terms.push((ch, wt));
                }
                let c = ((row[0] / qs + 1.0) / 2.0).round() as i64;
                if qs * (2 * c - 1) as f32 != row[0] {
                    return Err(format!("memory head ({li}, {h}) address bit {i}: constant {} is not S·(2c − 1)", row[0]));
                }
                bits.push((terms, c));
            }
            // The value is the line's bits, dim j to one channel or none.
            let out: Vec<Option<usize>> = (0..8 * LINE).map(|j| (0..d).find(|&ch| l.o[ch * a + base + j] != 0.0)).collect();
            let copy_out = |ch: usize| copy.is_some_and(|w| (lay.copy(w, 0)..=lay.copy(w, 8 * LINE - 1)).contains(&ch));
            if out.iter().flatten().any(|&ch| !(is_wire(ch) || copy_out(ch))) {
                return Err(format!("memory head ({li}, {h}) writes a non-wire channel"));
            }
            // Re-emit the head from the recovered query and compare every weight. The
            // blank layer is shared; only this head's rows are written, then cleared.
            let query = match copy {
                Some(w) => QueryRows::Gathered(w),
                None => QueryRows::Bits(bits.clone()),
            };
            memory_head(&mut blank, lay, cfg, h, &query, &out, &spec.params);
            let kb = cfg.kv_head(h) * HEAD_DIM;
            for t in 0..HEAD_DIM {
                let (r, kr) = ((base + t) * d, (kb + t) * d);
                if blank.q[r..r + d] != l.q[r..r + d] || blank.k[kr..kr + d] != l.k[kr..kr + d] || blank.v[kr..kr + d] != l.v[kr..kr + d] {
                    return Err(format!("memory head ({li}, {h}) dim {t} is not the record pattern"));
                }
            }
            if (0..d).any(|ch| blank.o[ch * a + base..][..HEAD_DIM] != l.o[ch * a + base..][..HEAD_DIM]) {
                return Err(format!("memory head ({li}, {h}): W_o is not the record pattern"));
            }
            for t in 0..HEAD_DIM {
                let (r, kr) = ((base + t) * d, (kb + t) * d);
                blank.q[r..r + d].fill(0.0);
                blank.k[kr..kr + d].fill(0.0);
                blank.v[kr..kr + d].fill(0.0);
            }
            for ch in 0..d {
                blank.o[ch * a + base..][..HEAD_DIM].fill(0.0);
            }
            // A copy head's query is a gathered address, which no step sets, so the plan
            // serves it the constant 0, whose line has a cell.
            if copy.is_some() && t.gathered.iter().any(|&ch| (lay.addr(0, 0)..=lay.addr(1, 23)).contains(&(ch as usize))) {
                return Err("a copy head's query is not the dead gathered address".into());
            }
            // The query reads the occupants its attention sees; the line lands in the
            // occupants the head writes, whose clearing (if any) must negate 0 and 1.
            let bits: Vec<(Vec<(u16, i32)>, i32)> = match copy {
                Some(_) => vec![(Vec::new(), 0); KEY_BITS],
                None => bits.iter().map(|(terms, c)| (terms.iter().map(|&(ch, w)| (t.seg.read(ch, 2 * li), w as i32)).collect(), *c as i32)).collect(),
            };
            let out: Vec<u16> = out.iter().map(|ch| ch.map_or(NO_CHANNEL, |ch| t.seg.write(ch, 2 * li))).collect();
            if out.iter().filter(|&&v| v != NO_CHANNEL).any(|&v| t.seg.range[v as usize].0 > 0 || t.seg.range[v as usize].1 < 1) {
                return Err(format!("memory head ({li}, {h}) writes a channel whose clearing is not exact on its bits"));
            }
            // A positional wire has a value per position, so its head has an address per
            // position. The positional stage serves it there, the index one per position.
            if let Some(p) = &mut t.pos {
                if p.owned.contains(&(li, h)) {
                    p.attach(li, &bits, &out)?;
                    continue;
                }
                if bits.iter().flat_map(|(terms, _)| terms).any(|&(v, _)| p.positional[v as usize]) {
                    return Err(format!("memory head ({li}, {h}) reads a positional wire the split did not see"));
                }
            }
            heads.push(MemHead { layer: li, bits, out });
        }
        if heads.iter().any(|h| h.layer == 0) {
            return Err("a memory head at layer 0 keys without its address".into());
        }
        MemPlan::new(heads, t.seg.count, t.circuit.len() + 1, &t.seg.range)
    }

    /// The plan over `channels` virtual channels and `layers` layers (the circuit layers
    /// and the routing layer, as `Template` holds them), with every table derived from
    /// `heads`: the query bitsets (`reach`, at least one word a channel), the heads
    /// by layer, and the address-bit tables.
    pub(super) fn new(heads: Vec<MemHead>, channels: usize, layers: usize, range: &[(i32, i32)]) -> Result<MemPlan, String> {
        if heads.len() * KEY_BITS > PAIR as usize {
            return Err(format!("{} memory heads; the address-bit tables index up to {}", heads.len(), PAIR as usize / KEY_BITS));
        }
        let words = heads.len().div_ceil(64).max(1);
        let mut reach = vec![[0u64; 2]; channels * words];
        let mut layer_heads = vec![Vec::new(); layers];
        // A column select's bit forms are its row's forms plus `(2·off_b − 1)·(1 − g)`
        // for one gate channel g: g in every bit at weight ∓1, the constant moved by ±1.
        // The row's forms are a unit the columns share; any other head is its own unit.
        let mut units: Vec<Vec<(Vec<(u16, i32)>, i32)>> = Vec::new();
        let mut unit_of = Vec::new();
        let mut gate = Vec::new();
        for (h, head) in heads.iter().enumerate() {
            // The gate is a channel declared 0 or 1 that every bit reads once at weight
            // ±1. With it 0 or 1 and every row form 0 or 1, the column's address is the
            // row's line or the off line, as the forms' signs say; the pass checks both
            // where it reads (`Engine::memory_reads`). Of several such channels, one whose
            // rows are 0 or 1 over their channels' declared ranges is the gate.
            let within = |terms: &[(u16, i32)], c: i32| {
                let (lo, hi) = terms.iter().fold((c, c), |(lo, hi), &(ch, w)| {
                    let (a, b) = range[ch as usize];
                    (lo + (w * a).min(w * b), hi + (w * a).max(w * b))
                });
                lo >= 0 && hi <= 1
            };
            let gates = |rows_declared: bool| {
                head.bits[0].0.iter().map(|&(ch, _)| ch).find(|&g| {
                    within(&[(g, 1)], 0)
                        && head.bits.iter().all(|(terms, c)| {
                            let w = terms.iter().filter(|&&(ch, _)| ch == g).map(|&(_, w)| w).collect::<Vec<_>>();
                            let row: Vec<(u16, i32)> = terms.iter().copied().filter(|&(ch, _)| ch != g).collect();
                            w.len() == 1 && w[0].abs() == 1 && (!rows_declared || within(&row, c + w[0]))
                        })
                })
            };
            let column = gates(true).or_else(|| gates(false));
            let (unit, g) = match column {
                Some(g) => {
                    let mut off = 0u32;
                    let row: Vec<(Vec<(u16, i32)>, i32)> = head
                        .bits
                        .iter()
                        .enumerate()
                        .map(|(i, (terms, c))| {
                            let w = terms.iter().find(|&&(ch, _)| ch == g).unwrap().1;
                            off |= ((w == -1) as u32) << i;
                            (terms.iter().copied().filter(|&(ch, _)| ch != g).collect(), c + w)
                        })
                        .collect();
                    (row, Some((g, off)))
                }
                None => (head.bits.clone(), None),
            };
            let u = match units.iter().position(|x| *x == unit) {
                Some(u) if g.is_some() => u,
                _ => {
                    units.push(unit);
                    units.len() - 1
                }
            };
            unit_of.push(u as u16);
            gate.push(g);
            if let Some((g, _)) = g {
                reach[g as usize * words + h / 64][0] |= 1 << (h % 64);
            }
            layer_heads[head.layer].push(h as u16);
        }
        if units.len() * KEY_BITS > PAIR as usize {
            return Err(format!("{} address units; the address-bit tables index up to {}", units.len(), PAIR as usize / KEY_BITS));
        }
        for (u, unit) in units.iter().enumerate() {
            let of_unit = || (0..heads.len()).filter(|&h| unit_of[h] as usize == u);
            let first = of_unit().min_by_key(|&h| (heads[h].layer, h)).expect("a unit has a head");
            for &(ch, _) in unit.iter().flat_map(|(terms, _)| terms) {
                reach[ch as usize * words + first / 64][0] |= 1 << (first % 64);
                for h in of_unit() {
                    reach[ch as usize * words + h / 64][1] |= 1 << (h % 64);
                }
            }
        }
        let mut ungated = vec![0u64; words];
        for h in (0..heads.len()).filter(|&h| gate[h].is_none()) {
            ungated[h / 64] |= 1 << (h % 64);
        }
        let slots = || units.iter().flatten().enumerate();
        let bit_terms = Cols::from_pairs(channels, slots().flat_map(|(b, (terms, _))| terms.iter().map(move |&(ch, w)| (ch as usize, b as u16, w))).collect());
        let bit_consts = slots().map(|(_, &(_, c))| c).collect();
        let live: Vec<u64> = heads.iter().map(|h| h.out.iter().enumerate().filter(|(_, o)| **o != NO_CHANNEL).fold(0u64, |m, (j, _)| m | 1 << j)).collect();
        let layer_mask = layer_heads
            .iter()
            .map(|hs| {
                let mut m = vec![0u64; words];
                for &h in hs {
                    m[h as usize / 64] |= 1 << (h % 64);
                }
                m
            })
            .collect();
        let mut buses: Vec<([u16; 8], Vec<u16>)> = Vec::new();
        let col: Vec<Option<(u8, u16)>> = heads
            .iter()
            .enumerate()
            .map(|(h, head)| {
                let m: u64 = live[h];
                let lane = m.trailing_zeros() as usize / 8;
                // A head whose live bits lie in one lane (a bus bit no node reads has no
                // channel) drives its lane's channels as a bus, alone or with others.
                if m == 0 || m & !(0xFF << (8 * lane)) != 0 {
                    return None;
                }
                let chans: [u16; 8] = std::array::from_fn(|i| head.out[8 * lane + i]);
                let b = match buses.iter().position(|(c, _)| *c == chans) {
                    Some(b) => b,
                    None => {
                        buses.push((chans, Vec::new()));
                        buses.len() - 1
                    }
                };
                buses[b].1.push(h as u16);
                Some((lane as u8, b as u16))
            })
            .collect();
        // Only one-lane heads drive a bus, and they leave its channels to its diff.
        if let Some((chans, _)) = buses.iter().find(|(chans, members)| heads.iter().enumerate().any(|(h, hd)| !members.contains(&(h as u16)) && hd.out.iter().any(|o| chans.contains(o)))) {
            return Err(format!("bus channels {chans:?} have a writer that is not a column"));
        }
        // Every other channel has one writer, and that head's read diffs it alone.
        let mut writers = vec![0u8; channels];
        for (h, head) in heads.iter().enumerate() {
            if col[h].is_none() {
                for &o in head.out.iter().filter(|&&o| o != NO_CHANNEL) {
                    writers[o as usize] += 1;
                    if writers[o as usize] > 1 {
                        return Err(format!("channel {o} has two memory heads writing it outside a bus "));
                    }
                }
            }
        }
        Ok(MemPlan { heads, reach, words, ungated, layer_heads, layer_mask, bit_terms, bit_consts, unit_of, gate, live, col, buses })
    }
}

/// The template strategy: the circuit layers are shared by every decoded position
/// and evaluated once per pass, the routing layer runs on a blank row and each
/// position corrects its own units, and the LM head is a table over the route bits.
#[derive(Clone)]
pub(super) struct Template {
    pub(super) circuit: Vec<IntLayer>,
    pub(super) routing: IntLayer,
    /// Per token: the routing units that read one of its nonzero token channels, each
    /// with the gate contribution those channels make.
    pub(super) own_units: Vec<Vec<Own>>,
    /// Per routing unit, its part in the decode of each tracked token's home
    /// (`tracked`) whose `own_units` hold it: at most two a unit, the rest filled with
    /// `Owner::none`. A routing update moves each home's own route totals
    /// (`State::own_route`) by the change in its clamped contribution, and marks the
    /// home stale only when that moved.
    pub(super) owners: Vec<[Owner; 2]>,
    /// The routing layer's gate table with each entry's `RouteTag` beside it,
    /// which the routing update reads in sequence with the entries.
    pub(super) route_gate: Tagged<RouteTag>,
    /// Per token, the position whose own route totals its decode reads, or `seq` when
    /// none does. A token is tracked at its home when it has one nonzero channel and
    /// one home position (the mask tokens, which are what a decoded position holds),
    /// and it is the only such token there. A position holding an untracked token, or a
    /// token away from its tracked home, sums its own units and decodes on every pass.
    pub(super) tracked: Vec<u16>,
    /// Per position (and the spare `seq`), its tracked token's own route totals at the
    /// routing layer's start gates: what `State::own_route` starts from.
    pub(super) own_start: Vec<[i32; ROUTE_CAP]>,
    pub(super) seq: usize,
    /// Route channel → bit index in the head table's pattern.
    pub(super) route_index: Vec<u8>,
    /// Per route pattern, the argmax token: the checkpoint's, or with a view the
    /// stream's (`NO_TOKEN` for one the stream lacks).
    pub(super) head_table: Vec<u32>,
    /// The head scores one route channel as a byte value, not route bits.
    pub(super) value_head: bool,
    pub(super) readers: Vec<Readers>,
    /// `seg.range` and `readers` per virtual channel, side by side for the drain.
    pub(super) wires: Vec<Wire>,
    /// Layer-0 attention output dim → (wire channel, weight), and the reverse.
    pub(super) o_cols: Vec<Vec<(u16, f32)>>,
    pub(super) o_rows: Vec<Vec<(u16, f32)>>,
    pub(super) gathered: Vec<u16>,
    /// Layer-0 heads whose output the circuit reads; the others (the address gathers)
    /// feed only channels no layer reads, so the fast path leaves them out.
    pub(super) live_heads: Vec<bool>,
    /// The virtual channels every table above indexes.
    pub(super) seg: Segments,
    /// The units output positions compute for themselves, when the circuit has
    /// position wires.
    pub(super) pos: Option<Positional>,
    /// Whether a virtual channel is a token channel (only dense ids below `hidden`).
    pub(super) token: Vec<bool>,
    /// Per routing unit (positional ones included), whether a token channel gates it
    /// (`declared_bounds`). A pruned plan cannot check the channels it leaves out at run time;
    /// this and `norm` stand in for that check (`Plan::prune`).
    pub(super) route_gated: Vec<bool>,
    /// Σ over non-constant dense channels of the largest square any occupant can hold,
    /// route channels counted as one each, against half an ulp of the constants' n·2^40:
    /// below it, every RMSNorm divides by exactly 2^15 whatever the wires hold.
    pub(super) norm: (i64, i64),
    /// Per circuit layer, each unit's index in the checkpoint's layer: `pack` drops
    /// and reorders units, and the update census names them by family.
    pub(super) unit_ids: Vec<Vec<u16>>,
}

impl Template {
    /// The bucket after the routing layer's: changes to channels a memory head's query
    /// reads, drained by `Engine::memory_reads`.
    #[inline]
    pub(super) fn head_bucket(&self) -> usize {
        self.circuit.len() + 1
    }

    /// The bucket after the head bucket: changes to shared channels the positional
    /// stage reads.
    #[inline]
    pub(super) fn pos_bucket(&self) -> usize {
        self.circuit.len() + 2
    }

    /// Circuit layer `li`'s single bucket: changes to the channels it reads through one
    /// gate entry (`Blocks::single`), after the positional bucket; None when the reader
    /// bits cannot hold one per layer.
    #[inline]
    pub(super) fn single_bucket(&self, li: usize) -> Option<usize> {
        let n = self.circuit.len();
        (n <= super::BUCKET_LAYERS as usize).then_some(n + 3 + li)
    }

    /// Circuit layer `li`'s block bucket: changes to the channels it reads through
    /// exactly one block (`Blocks::one_block`), after the single buckets; None when the
    /// reader bits cannot hold it.
    #[inline]
    pub(super) fn block_bucket(&self, li: usize) -> Option<usize> {
        let n = self.circuit.len();
        (n <= super::BUCKET_LAYERS as usize).then_some(2 * n + 3 + li)
    }

    /// The circuit layers a reader mask names, single buckets folded onto their layers.
    #[inline]
    pub(super) fn reading_layers(&self, r: Readers) -> Readers {
        let n = self.circuit.len();
        let layers = (1 << n) - 1;
        r & layers | self.single_bucket(0).map_or(0, |b| r >> b & layers) | self.block_bucket(0).map_or(0, |b| r >> b & layers)
    }

    pub(super) fn check(m: &Model, spec: Option<&MemorySpec>) -> Result<Template, String> {
        let cfg = &m.cfg;
        let (d, a, nc) = (cfg.hidden, cfg.attn(), cfg.n_const);
        // Channels are u16 indices in the plan's tables; tokens index its vectors as usize (the
        // one model with the stepper's stream holds more than 65,536).
        if d > 1 << 16 {
            return Err(format!("{d} channels; the tables index up to 65536"));
        }
        let row = |t: usize| &m.embed[t * d..][..d];
        if !(0..cfg.vocab).all(|t| row(t)[..nc].iter().zip(CONSTANTS).all(|(&v, k)| v == C * k) && row(t)[nc..].iter().all(|&v| v == 0.0 || v == UNIT)) {
            return Err("an embedding is not constants plus 0/1 values".into());
        }
        // The channels a token sets.
        let token: Vec<bool> = (0..d).map(|c| c >= nc && (0..cfg.vocab).any(|t| row(t)[c] != 0.0)).collect();
        let (first, rest) = m.layers.split_first().ok_or("no layers")?;
        // Attention after layer 0 is live only in the declared memory heads.
        let memory_dims = |li: usize| -> Vec<bool> {
            let mut dims = vec![false; a];
            for &(l, h) in &spec.map_or(Vec::new(), |s| s.heads.clone()) {
                if l == li {
                    dims[h * HEAD_DIM..(h + 1) * HEAD_DIM].fill(true);
                }
            }
            dims
        };
        if first.o.iter().all(|&w| w == 0.0) {
            return Err("attention is live outside layer 0".into());
        }
        for (li, l) in rest.iter().enumerate() {
            let dims = memory_dims(li + 1);
            if (0..d).any(|ch| (0..a).any(|i| l.o[ch * a + i] != 0.0 && !dims[i])) {
                return Err("attention is live outside layer 0".into());
            }
        }
        let dead_dims = {
            let mut dims = vec![false; a];
            for &h in spec.map_or(&[][..], |s| &s.gathers[..]) {
                dims[h * HEAD_DIM..(h + 1) * HEAD_DIM].fill(true);
            }
            dims
        };
        // Layer 0's W_o by output dim, nonzeros only. A head whose W_o is zero (an
        // address gather, or a head slot another layer needs and layer 0 leaves empty)
        // changes no channel whatever it attends to; `index` leaves it out.
        // In wire units: a head writes v·UNIT into the residual for a wire of value v.
        let w_o: Vec<Vec<(u16, f32)>> = (0..a).map(|i| (0..d).filter(|&ch| first.o[ch * a + i] != 0.0).map(|ch| (ch as u16, first.o[ch * a + i] / UNIT)).collect()).collect();
        let o_cols: Vec<Vec<(u16, f32)>> = w_o.iter().zip(&dead_dims).map(|(col, &dead)| if dead { Vec::new() } else { col.clone() }).collect();
        let gathered: Vec<usize> = o_cols.iter().flatten().map(|&(ch, _)| ch as usize).collect();
        if gathered.iter().any(|&c| c < nc || token[c]) {
            return Err("layer-0 attention writes a constant or token channel".into());
        }
        // A dead head's channels must be read by no layer, else leaving it out would matter.
        let dead_channels: Vec<usize> = w_o.iter().zip(&dead_dims).filter(|(_, dead)| **dead).flat_map(|(col, _)| col.iter().map(|&(ch, _)| ch as usize)).collect();
        // The reader tokens' routing units read the gathered addresses' low bits; each is
        // gated by its reader token's channel (form ≤ θ − 1 without it), which no step
        // or record but a reader carries, so it is 0 wherever the gathers are left out.
        let rd: Vec<usize> = if m.memory.is_some() { (0..crate::circuit::LINE).map(|k| m.layout.rd(k)).collect() } else { Vec::new() };
        let gated = |l: &Layer, u: usize| rd.iter().any(|&ch| l.gate[u * d + ch] > 0.0);
        let routing = m.layers.len() - 1;
        if dead_channels.iter().any(|&ch| m.layers.iter().enumerate().any(|(li, l)| (0..cfg.intermediate).any(|u| l.gate[u * d + ch] != 0.0 && !(li == routing && gated(l, u)))) || (0..cfg.vocab).any(|t| m.lm_head[t * d + ch] != 0.0)) {
            return Err("an address-gather channel is read by a layer".into());
        }
        // Per layer, the channels its attention writes: layer 0's gathers, then the
        // memory heads.
        let n = m.layers.len();
        let mut attn_writes: Vec<Vec<usize>> = vec![Vec::new(); n];
        attn_writes[0] = gathered.clone();
        for &(li, h) in &spec.map_or(Vec::new(), |s| s.heads.clone()) {
            attn_writes[li].extend((0..d).filter(|&ch| (0..HEAD_DIM).any(|i| m.layers[li].o[ch * a + h * HEAD_DIM + i] != 0.0)));
        }
        for w in &mut attn_writes {
            w.sort_unstable();
            w.dedup();
        }

        // Circuit units read wires, and position wires read an output position's slot
        // identity (`Positional` takes those units).
        let wires = readable(&token, m.layout.output_ids(cfg.n_inputs));
        // Each channel's largest |value|: a wire channel's over its occupants' declared
        // ranges (a nibble's 15), every other channel 0/1.
        let mut largest = vec![1i64; d];
        for (k, occ) in m.layout.occupants.iter().enumerate() {
            largest[m.layout.wire_base() + k] = occ.iter().map(|s| m.wire_ranges[s.0 as usize].0.abs().max(m.wire_ranges[s.0 as usize].1.abs())).max().unwrap_or(1).max(1);
        }
        let mut dense = Vec::new();
        for l in &m.layers[..n - 1] {
            dense.push(IntLayer::from(l, cfg, &wires, &largest).map_err(|e| format!("layer {}: {e}", dense.len()))?.0);
        }
        dense.push(IntLayer::from(&m.layers[n - 1], cfg, &vec![true; d], &largest).map_err(|e| format!("routing layer: {e}"))?.0);
        if dense[..n - 1].iter().flat_map(IntLayer::writes).any(|c| token[c]) {
            return Err("a circuit layer writes a token channel".into());
        }
        let mut seg = Segments::new(d, &attn_writes, &dense.iter().collect::<Vec<_>>())?;
        // Shared channels: each wire's occupant becomes `hidden + wire`, the order the
        // wires were created in. Any numbering is exact; this one keeps locality.
        let lay = &m.layout;
        if seg.later.iter().any(|l| !l.is_empty()) {
            let base = lay.wire_base();
            seg.renumber(|ch, k| (ch >= base && ch < lay.after_wires()).then(|| lay.occupants[ch - base].get(k).map(|s| d + s.0 as usize)).flatten())?;
        }
        let written: Vec<bool> = {
            let mut w = vec![false; d];
            for c in dense[..n - 1].iter().flat_map(IntLayer::writes).chain(attn_writes.iter().flatten().copied()) {
                w[c] = true;
            }
            w
        };
        let mut circuit: Vec<IntLayer> = dense.iter().enumerate().map(|(li, l)| l.remap(&seg, 2 * li + 1)).collect();
        let mut routing = circuit.pop().unwrap();
        let dv = seg.count;
        let token_v: Vec<bool> = (0..dv).map(|v| v < d && token[v]).collect();
        let bounds = declared_bounds(m, &seg, &routing, &token_v);
        // A channel's last occupant is never cleared, so its guard is the wire's declared
        // range: 0/1, or 0..15 for a nibble (`docs/byte-channels.md`). An earlier
        // occupant keeps its clearing's exact range.
        if let Ok((b, _)) = &bounds {
            for ch in 0..d {
                let v = seg.later[ch].last().map_or(seg.first[ch], |&(_, v)| v) as usize;
                let (lo, hi) = b[v];
                seg.range[v] = (seg.range[v].0.min(lo as i32), seg.range[v].1.max(hi as i32));
            }
        }

        // The head reads only the constant and route channels, which only the routing
        // layer writes; then its argmax is a function of the route bits.
        // Each head row's nonzero weights in channel order: a fold over them adds the
        // same terms in the same order as a walk of the dense row that skips zeros.
        let head_nz: Vec<Vec<(usize, f32)>> = (0..cfg.vocab).map(|t| m.lm_head[t * d..][..d].iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(c, &w)| (c, w)).collect()).collect();
        let mut route_channels: Vec<usize> = head_nz.iter().flatten().map(|&(c, _)| c).filter(|&c| c >= nc).collect();
        route_channels.sort_unstable();
        route_channels.dedup();
        if route_channels.len() > ROUTE_CAP {
            return Err(format!("{} route channels; at most {ROUTE_CAP}", route_channels.len()));
        }
        if route_channels.iter().any(|&c| c < nc || token[c] || written[c] || !seg.later[c].is_empty()) || routing.writes().any(|c| !route_channels.contains(&c)) {
            return Err(format!("the head does not read exactly the routing layer's outputs ({} route channels)", route_channels.len()));
        }
        let mut route_index = vec![u8::MAX; dv];
        for (i, &c) in route_channels.iter().enumerate() {
            route_index[c] = i as u8;
        }
        // Each memory head's query and line channels, so the split can tell a head that
        // asks a position-dependent address (`MemPlan::check` checks the heads in full).
        let head_io: Vec<HeadIo> = spec
            .map_or(&[][..], |s| &s.heads[..])
            .iter()
            .filter(|at| !spec.unwrap().copy_heads.contains(at))
            .map(|&(li, h)| {
                let l = &m.layers[li];
                let reads = (1..d).filter(|&ch| KEY_DIMS.iter().any(|&dim| l.q[(h * HEAD_DIM + dim) * d + ch] != 0.0)).map(|ch| seg.read(ch, 2 * li) as usize).collect();
                let writes = (0..d).filter(|&ch| (0..HEAD_DIM).any(|j| l.o[ch * a + h * HEAD_DIM + j] != 0.0)).map(|ch| seg.write(ch, 2 * li) as usize).collect();
                HeadIo { at: (li, h), reads, writes }
            })
            .collect();
        let pos = Positional::split(m, &mut circuit, &mut routing, &mut seg, &route_index, route_channels.len(), &head_io)?;
        // A nibble head (`docs/byte-channels.md`) scores the first two route channels as
        // the byte's nibbles, 2l·L − l² + 2h·H − h², and the rest as bits. Its table runs
        // over L | H << 4 | bits << 8.
        let value_head = route_channels.len() >= 2 && head_nz.iter().flatten().any(|&(c, w)| c >= nc && w.abs() > 1.0);
        let entries = if value_head { 256 << (route_channels.len() - 2) } else { 1usize << route_channels.len() };
        let head_table = (0..entries)
            .map(|pat| {
                let h = |c: usize| match route_channels.iter().position(|&r| r == c) {
                    _ if c < nc => CONSTANTS[c],
                    Some(i) if value_head && i < 2 => (pat >> (4 * i) & 15) as f32,
                    Some(i) if value_head => ((pat >> (i + 6)) & 1) as f32,
                    Some(i) => ((pat >> i) & 1) as f32,
                    None => 0.0,
                };
                let logit = |t: usize| head_nz[t].iter().fold(0.0f32, |acc, &(c, w)| if h(c) != 0.0 { acc + w * h(c) } else { acc });
                (0..cfg.vocab).fold((f32::NEG_INFINITY, 0), |b, t| {
                    let v = logit(t);
                    if v > b.0 { (v, t as u32) } else { b }
                })
                .1
            })
            .collect();

        let token_bits: Vec<Vec<(u16, i32)>> = (0..cfg.vocab).map(|t| (nc..d).filter(|&c| row(t)[c] != 0.0).map(|c| (c as u16, (row(t)[c] / UNIT) as i32)).collect()).collect();
        let own_units: Vec<Vec<Own>> = token_bits
            .iter()
            .map(|bits| {
                let mut us: Vec<u16> = bits.iter().flat_map(|&(c, _)| routing.readers_of(c as usize)).collect();
                us.sort_unstable();
                us.dedup();
                us.into_iter()
                    .map(|u| {
                        let own = routing.rows().get(u as usize).iter().map(|&(c, w)| bits.iter().find(|b| b.0 == c).map_or(0, |b| w * b.1)).sum();
                        let ui = u as usize;
                        let (c, bit) = (routing.c()[ui], route_index[routing.out()[ui] as usize]);
                        if c != 0 && bit as usize >= ROUTE_CAP {
                            return Err(format!("routing unit {u} writes no route channel"));
                        }
                        Ok(Own { u, bit: if c != 0 { bit } else { 0 }, pair: routing.cap()[ui] == 1, own, c })
                    })
                    .collect()
            })
            .collect::<Result<_, String>>()?;
        let o_cols: Vec<Vec<(u16, f32)>> = o_cols.iter().map(|col| col.iter().map(|&(ch, w)| (seg.write(ch as usize, 0), w)).collect()).collect();
        // Reader bits: one per circuit layer, the routing layer, the head bucket, the
        // positional stage.
        if circuit.len() + 3 > Readers::BITS as usize {
            return Err(format!("more than {} circuit layers", Readers::BITS - 3));
        }
        let single = |t: usize| token_bits[t].len() == 1;
        let home = Slots::new(cfg).home;
        let seq = cfg.seq();
        if seq >= u16::MAX as usize {
            return Err(format!("{seq} positions; the owner table indexes up to 65534"));
        }
        let mut tracked = vec![seq as u16; cfg.vocab];
        let mut held = vec![false; seq];
        // In a stream's network another part's output mask carries the identity channel
        // one of the stream's own slots holds (`Layout::identities`), so it reaches that
        // slot's routing units too. The stream never decodes there, and the units' one
        // owner is the stream's own slot.
        let own = |p: u16| (p as usize) < cfg.n_inputs || m.layout.own_outputs.as_ref().is_none_or(|o| o.contains(&(p as usize)));
        for t in 0..cfg.vocab {
            if let Home::At(p) = home[t] {
                if single(t) && own(p) && !std::mem::replace(&mut held[p as usize], true) {
                    tracked[t] = p;
                }
            }
        }
        let mut owners = vec![[Owner::none(seq); 2]; routing.bias().len()];
        let mut own_start = vec![[0i32; ROUTE_CAP]; seq + 1];
        let start = routing.start();
        for (t, us) in own_units.iter().enumerate() {
            let p = tracked[t];
            if p as usize == seq {
                continue;
            }
            for o in us.iter().filter(|o| o.c != 0) {
                let owner = Owner { p, bit: o.bit, own: o.own, c: o.c };
                let free = owners[o.u as usize].iter_mut().find(|q| q.c == 0).ok_or(format!("routing unit {} feeds the decode of more than two positions", o.u))?;
                *free = owner;
                own_start[p as usize][o.bit as usize] += owner.part(start[o.u as usize], o.pair);
            }
        }
        // A routing unit that is linear over its wires' declared ranges (`route_span`,
        // `Owner::linear`) is updated without its clamps. `index` proves that from the
        // ranges the pass guards (`seg.range`), so those narrow to the declared ones here,
        // as they do for the wires the positional stage's proofs read.
        if let Ok((b, _)) = &bounds {
            let declared: Vec<(i64, i64)> = seg.range.iter().zip(b).map(|(&(lo, hi), &(dlo, dhi))| ((lo as i64).max(dlo), (hi as i64).min(dhi))).collect();
            for u in 0..owners.len() {
                if owners[u][1].c == 0 && owners[u][0].linear(start[u], routing.cap()[u] == 1, route_span(&routing, &token_v, u, |ch| declared[ch])) {
                    for &(ch, _) in routing.rows().get(u).iter().filter(|e| !token_v[e.0 as usize]) {
                        let (lo, hi) = declared[ch as usize];
                        seg.range[ch as usize] = (lo as i32, hi as i32);
                    }
                }
            }
        }
        let (norm, route_gated) = match bounds {
            Ok((b, blank)) => (norm_bound(&b, &seg, nc, &route_channels, if value_head { 225 } else { 1 }), blank),
            Err(_) => ((i64::MAX, 0), Vec::new()),
        };
        let mut t = Template {
            circuit,
            routing,
            own_units,
            owners,
            tracked,
            route_gate: Tagged::default(),
            own_start,
            seq,
            route_index,
            head_table,
            value_head,
            o_cols,
            seg,
            pos,
            token: token_v,
            route_gated,
            norm,
            unit_ids: Vec::new(),
            readers: Vec::new(),
            wires: Vec::new(),
            o_rows: Vec::new(),
            gathered: Vec::new(),
            live_heads: Vec::new(),
        };
        t.unit_ids = t.circuit.iter().map(|l| (0..l.bias().len() as u16).collect()).collect();
        t.pack();
        t.index(&[]);
        Ok(t)
    }

    /// Renumbers each circuit layer's units to the ones that write, in
    /// `IntLayer::packed_order`, so the gates the walk updates for one channel's change sit
    /// near each other, mostly in runs. The renumbering is a bijection on the writing
    /// units, so every gate, send and channel is the same, and only `unit_ids` keeps the
    /// old names.
    fn pack(&mut self) {
        for (l, ids) in self.circuit.iter_mut().zip(&mut self.unit_ids) {
            let order = l.packed_order();
            *l = l.permuted(&order).with_blocks();
            *ids = order.iter().map(|&u| ids[u as usize]).collect();
        }
    }

    /// Derives the tables that index the layers, the layer-0 columns, the memory
    /// `heads` and the positional stage: `o_rows`, `gathered`, `live_heads`, `readers`,
    /// `wires` and `route_gate`. Every builder ends here, so they always describe the plan's own
    /// layers.
    fn index(&mut self, heads: &[MemHead]) {
        let dv = self.seg.count;
        let mut o_rows: Vec<Vec<(u16, f32)>> = vec![Vec::new(); dv];
        for (i, col) in self.o_cols.iter().enumerate() {
            for &(ch, w) in col {
                o_rows[ch as usize].push((i as u16, w));
            }
        }
        let mut gathered: Vec<u16> = self.o_cols.iter().flatten().map(|&(ch, _)| ch).collect();
        gathered.sort_unstable();
        gathered.dedup();
        // A head with no column left feeds no channel any layer reads.
        let live_heads = self.o_cols.chunks(HEAD_DIM).map(|cols| cols.iter().any(|c| !c.is_empty())).collect();
        let mut readers: Vec<Readers> = vec![0; dv];
        for (li, l) in self.circuit.iter().chain([&self.routing]).enumerate() {
            for (ch, r) in readers.iter_mut().enumerate() {
                if !l.gate().get(ch).is_empty() {
                    let single = li < self.circuit.len() && l.blocks().single()[ch] != super::int::NOT_SINGLE;
                    let one_block = li < self.circuit.len() && l.blocks().one_block()[ch] != super::int::NOT_SINGLE;
                    *r |= 1 << match (self.single_bucket(li), self.block_bucket(li)) { (Some(b), _) if single => b, (_, Some(b)) if one_block => b, _ => li };
                }
            }
        }
        for &(ch, _) in heads.iter().flat_map(|h| h.bits.iter().flat_map(|(terms, _)| terms)) {
            readers[ch as usize] |= 1 << self.head_bucket();
        }
        if let Some(p) = &self.pos {
            for (ch, r) in readers.iter_mut().enumerate() {
                if p.readers[ch] != 0 {
                    *r |= 1 << self.pos_bucket();
                }
            }
        }
        assert_eq!(self.seg.range.len(), dv);
        self.wires = self.seg.range.iter().zip(&readers).map(|(&r, &rd)| Wire::new(r, rd)).collect();
        (self.o_rows, self.gathered, self.live_heads, self.readers) = (o_rows, gathered, live_heads, readers);
        // Linear units, proven over the ranges the pass guards: `wires` above holds the
        // same `seg.range`, a wire no unit and no gather writes stays 0, and a memory
        // head's output is a bit, which `MemPlan` holds inside its channel's range.
        let start = self.routing.start();
        let linear: Vec<bool> = (0..start.len())
            .map(|u| {
                let [o, second] = self.owners[u];
                second.c == 0 && o.linear(start[u], self.routing.cap()[u] == 1, route_span(&self.routing, &self.token, u, |ch| (self.seg.range[ch].0 as i64, self.seg.range[ch].1 as i64)))
            })
            .collect();
        self.route_gate = self.routing.gate().tagged(|u, pair| RouteTag { pair, second: self.owners[u as usize][1].c != 0, linear: linear[u as usize], owner: self.owners[u as usize][0] });
    }
}

/// A routing unit a token's channels reach, as decode reads it: the unit, its route
/// bit, whether it is a fused pair (cap 1), the gate contribution of the token's
/// channels, and its down weight.
#[derive(Clone, Copy)]
pub(super) struct Own {
    pub(super) u: u16,
    pub(super) bit: u8,
    pub(super) pair: bool,
    pub(super) own: i32,
    pub(super) c: i32,
}

/// A routing unit's part in one tracked home's decode, as the routing layer's update
/// reads it: the home, its route bit, the gate contribution of the home's token
/// channels, and the unit's down weight (0 for none). Whether the unit is a fused pair
/// (cap 1) comes with each update, from its gate entry's `RouteTag`.
#[derive(Clone, Copy)]
pub(super) struct Owner {
    pub(super) p: u16,
    pub(super) bit: u8,
    pub(super) own: i32,
    pub(super) c: i32,
}

/// What a routing gate entry carries for its unit (`Template::route_gate`): whether it
/// is a fused pair, its first owner, and whether `Template::owners` holds a second.
#[derive(Clone, Copy)]
pub(super) struct RouteTag {
    pub(super) pair: bool,
    pub(super) second: bool,
    /// The unit has one owner and is linear over its wires' guarded ranges
    /// (`Owner::linear`): an update moves the owner's total by c · w · dx and sends nothing.
    pub(super) linear: bool,
    pub(super) owner: Owner,
}

impl Owner {
    /// No part, at home `p`: a down weight of 0 moves nothing.
    pub(super) const fn none(p: usize) -> Owner {
        Owner { p: p as u16, bit: 0, own: 0, c: 0 }
    }

    /// What the home's route total gains from this unit with its gate kept at `g`
    /// (g - 1, as `Marks::layer` keeps it): c · (clamp(g + own) - clamp(g)), the clamp
    /// to [0, 1] for a pair and to [0, ∞) otherwise. A pair's clamp(g + 1 + x, 0, 1) is
    /// (g + x >= 0), so its part is a difference of two sign tests.
    #[inline(always)]
    pub(super) fn part(self, g: i32, pair: bool) -> i32 {
        let on = if pair { (g + self.own >= 0) as i32 - (g >= 0) as i32 } else { (g + 1 + self.own).max(0) - (g + 1).max(0) };
        self.c * on
    }

    /// Whether this owner's part is c · (gate + own) wherever the unit's wires put its
    /// gate: with the gate before any wire at `g` (kept as g - 1) and the wires adding
    /// `lo` to `hi`, the blank row's clamp stays at 0 and the owner's is the identity
    /// (at least 0, and at most 1 for a pair). An update of w · dx then moves the part
    /// by c · w · dx, and the unit sends nothing on the blank row.
    pub(super) fn linear(self, g: i32, pair: bool, (lo, hi): (i64, i64)) -> bool {
        let (blank, with) = (g as i64 + 1, g as i64 + 1 + self.own as i64);
        self.c != 0 && blank + hi <= 0 && with + lo >= 0 && (!pair || with + hi <= 1)
    }
}

/// What routing unit `u`'s wires (the channels of its row no token sets) add to its
/// gate, lowest and highest, with each wire in `range` or at 0, a fresh state's value.
pub(super) fn route_span(routing: &IntLayer, token: &[bool], u: usize, range: impl Fn(usize) -> (i64, i64)) -> (i64, i64) {
    routing.rows().get(u).iter().filter(|e| !token[e.0 as usize]).fold((0, 0), |(lo, hi), &(ch, w)| {
        let (a, z) = range(ch as usize);
        let (a, z) = (a.min(0) * w as i64, z.max(0) * w as i64);
        (lo + a.min(z), hi + a.max(z))
    })
}

/// The channels a circuit unit may read: every channel no token sets (the wires) and the
/// output slots' identities, which position wires read. A unit reading any other token
/// channel is refused (`IntLayer::from`).
pub(super) fn readable(token: &[bool], output_ids: std::ops::Range<usize>) -> Vec<bool> {
    token.iter().enumerate().map(|(c, &t)| !t || output_ids.contains(&c)).collect()
}
