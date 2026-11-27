//! Runs a [`Model`] fast, giving exactly the dense pass's tokens.
//!
//! The compiled checkpoint computes only exact integers, and `Plan::check` proves it
//! from the weights:
//! - Every residual value is 0 or ±UNIT, so `rms_norm_eps` dominates and RMSNorm
//!   divides by exactly 2^7 in any summation order (`int::norm_bound`); with the norm
//!   weight, every matmul input is the wire's integer.
//! - Every gate term is then K·m for an integer m, the bias is K·b, `up` is 1, and
//!   every down weight is c·UNIT/K for an integer c. With the bounds checked here,
//!   no f32 sum ever rounds.
//! - A gate is therefore 0 or at least 256 in magnitude, where silu is exactly 0 or
//!   the identity.
//!
//! So the engine computes those integers directly: gates in units of 256, wires as
//! integers. The circuit layers are shared by every decoded position (no attention
//! after layer 0, and they never read the token channels). They are evaluated
//! incrementally: only input bits that changed since the last pass are scattered, and
//! only activation differences flow down. The routing layer runs once on a blank row,
//! and each position corrects only the units its own token reaches. The LM head is a
//! table over the route bits. When any check fails, the pass falls back to the dense
//! model and is counted.

//!
//! The modules: `int` holds the integer layers and the virtual channels, `plan` the
//! fast-path tables built from the weights, `positional` the per-position stage,
//! `layer0` the layer-0 attention diff, `view` a stream's view of a merged
//! checkpoint, `records` the memory checkpoint's cache and index, `cells` the K/V rows
//! every memory read attends over, `heads` a memory head's line from its query's forms and a
//! bus's byte from its columns, `stats` the counters, and `run` the per-pass hot loop.

mod cells;
mod rows;
pub mod counters;
mod heads;
mod int;
mod layer0;
mod plan;
mod pool;
mod positional;
pub mod ols;
pub mod probe;
#[cfg(feature = "record")]
pub mod stock;
mod records;
pub mod replay;
mod run;
mod stats;
#[cfg(test)]
mod tests;
mod view;

pub use cells::{turn, Cells, Row, RowBuilder, Served, Turn};
pub use records::{Memory, Video, VideoRecords, DEST, SOURCE};
pub use stats::{Kind, Refusal, Stats};
pub use view::StreamSpec;

use crate::llama::{rope, Model};
use layer0::{Home, Interned, Qkv, Slots, Sparse};
use plan::Plan;
use run::{Buckets, Marks, State};
use std::collections::HashMap;
use std::sync::Arc;
use view::View;

/// The embedding's constant channel.
const C: f32 = crate::llama::UNIT;
/// The factor RMSNorm and its weight apply from residual to matmul input (x / 2^7 · 2^14).
const INV: f32 = 1.0 / crate::llama::UNIT;
/// The normalised constant channel.
const ONE: f32 = 1.0;

/// Route channels a plan may have. Each decoded position copies its route totals to
/// the stack, and the head table holds one entry per route pattern.
const ROUTE_CAP: usize = 12;

/// A view's mapping for a checkpoint token its stream lacks.
const NO_TOKEN: u32 = u32::MAX;

/// Per channel, a bit per layer that reads it: the circuit layers, then the routing
/// layer, the head and positional buckets, and each circuit layer's single and block
/// buckets (`Template::single_bucket`, `Template::block_bucket`). Its width bounds the
/// circuit depth.
type Readers = u64;

/// The most circuit layers a stream runs with the template's single and block buckets: three
/// reader bits a layer and three more (`Template::single_bucket`). A deeper stream runs
/// without them, as the APU's does, so `one_model::PLACEMENT` stops the scanline PPU and
/// its length here.
pub const BUCKET_LAYERS: u32 = (Readers::BITS - 3) / 3;

/// A new nonzero id for a list of positions (`View::positions_id`), unique in the process.
/// It shares `Positions`' ids' counter: an engine without a view takes those as they are.
fn fresh_positions_id() -> u64 {
    crate::sequence::fresh_id()
}

pub struct Engine {
    dense: Arc<Model>,
    /// The id of this pass's positions when its view names them (`View::positions_id`),
    /// else 0. Equal ids mean equal lists, so the pass need not compare them.
    positions_id: u64,
    /// The stream this engine runs, when the checkpoint holds several.
    view: Option<View>,
    /// The sequence state whose decoded store holds every token this engine's state
    /// decoded since its positions were last all stale (`Sequence::stamp`), or None.
    out_stamp: Option<crate::sequence::Stamp>,
    /// The next fast pass decodes every position again (the store missed a decode).
    resync: bool,
    /// This pass also proves its sequence's change list and decoded store complete.
    verify: bool,
    /// A pass's tokens in its positions' order, for the checks that read them so.
    list: Vec<usize>,
    /// The thread's counters after a timed replan's memo lookup (`Stats::replan_rest`).
    replan_from: Option<counters::Counts>,
    plan: Result<Plan, String>,
    state: State,
    marks: Marks,
    /// The pending changes per reader layer, sized from the plan.
    pending: Buckets,
    slots: Slots,
    /// Layer 0's q, k and v projections as nonzero cells.
    l0: [Sparse; 3],
    /// Per layer-0 value dim, the positions holding a token whose value is nonzero there.
    v_reach: Vec<Vec<usize>>,
    /// By slot.
    qkv: Vec<Option<Box<Qkv>>>,
    /// Positions whose token differs from the last pass (reused buffer).
    moved: Vec<usize>,
    vector_ids: HashMap<Vec<u32>, u32>,
    /// Attention plans, one per distinct set of softmax weights, and their ids by
    /// those weights' bits.
    interned: Vec<Interned>,
    interned_ids: HashMap<Vec<(usize, usize, u32)>, u32>,
    /// Layer-0 plans by every position's (key id, query id) and the decoded positions.
    plan_memo: HashMap<Vec<u32>, u32, std::hash::BuildHasherDefault<layer0::Words>>,
    /// The memo key of the last replan (reused buffer).
    plan_key: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// Test hook: always take the dense fallback.
    pub force_fallback: bool,
    /// Time each fast pass into `stats.kinds`.
    pub timed: bool,
    /// Experiment switch for the replay bench's paired runs (`examples/replay.rs`,
    /// `--variants`), which sets it. Nothing reads it here: a candidate cut under
    /// measurement branches on it and leaves with its measurement.
    pub variant: u8,
    /// Run every `n`th fast pass, the first included, through the dense pass as well,
    /// and stop at the first decoded token that differs (0: never).
    pub step_oracle: u64,
    /// Also every `n`th fast pass that reads ROM past 2 MiB (0: only at `step_oracle`).
    pub rom_oracle: u64,
    /// Check the next fast pass whatever the cadence (`MemoryDenoiser::check_next`).
    pub check_next: bool,
    /// Fault injection for a test that the step oracle catches a wrong token: the next fast
    /// pass's first decoded token is XORed with this before any check
    /// (`tests/step_oracle_faults.rs`; feature `testing` only).
    #[cfg(feature = "testing")]
    fault: Option<usize>,
    /// The cache the step oracle's dense pass reads in a stream that holds no memory of
    /// its own, built at its first check.
    step_cache: Option<crate::llama::KvCache>,
    pub stats: Stats,
    pub memory: Option<Memory>,
    /// The record slots in the stream's own sequence, for a view that holds them.
    stream_records: Option<crate::llama::tok::RecordSlots>,
    /// Per list of decoded positions, what it reaches, for the update census.
    demands: HashMap<Vec<usize>, plan::Demand>,
    /// The replay bench's recording of this stream (`replay`), from its insertion.
    replay: Option<Box<replay::Recorder>>,
    /// Node families and each unit's family per circuit layer, for the update census.
    pub families: Option<(Vec<String>, Vec<Vec<u16>>)>,
    #[cfg(feature = "record")]
    recorder: Option<stock::Recorder>,
    #[cfg(feature = "record")]
    external: Option<Box<dyn stock::External>>,
}

impl Engine {
    pub fn new(dense: Model) -> Engine {
        let plan = Plan::check(&dense);
        Self::build(Arc::new(dense), plan, None)
    }

    /// One engine per stream over one checkpoint: the weights are shared, the plan is
    /// checked once and pruned to each stream's reach (`Plan::prune`), and each engine
    /// presents its stream's own slots and tokens (`StreamSpec`).
    ///
    /// An overlay checkpoint's parts share channels, so each stream runs its own
    /// network (`Model::stream` of the spec's part: its prefix of channels, its layers,
    /// its units and heads). Its dense pass is that network with the shared record units
    /// (`records` true). `Plan::check` runs on the network without them (`records`
    /// false), and the prune starts from that plan.
    pub fn streams(dense: Model, specs: &[StreamSpec]) -> Vec<Engine> {
        Self::streams_with(dense, specs, &|_, _| {})
    }

    /// `streams`, with `load` given each overlay stream's network and its slice before
    /// the engine takes it (`gguf::slice_from_env` reads its weights from a file).
    pub fn streams_with(dense: Model, specs: &[StreamSpec], load: &dyn Fn(&mut Model, &crate::llama::Slice)) -> Vec<Engine> {
        if dense.overlay.is_some() {
            return specs
                .iter()
                .map(|spec| {
                    let view = View::new(&dense.cfg, spec);
                    let plan = Plan::check(&dense.stream(spec.part, false)).and_then(|p| p.prune(&view).map_err(|e| format!("stream plan refused: {e}")));
                    let mut own = dense.stream(spec.part, true);
                    load(&mut own, &dense.slice(spec.part, true));
                    Self::build(Arc::new(own), plan, Some(view))
                })
                .collect();
        }
        let full = Plan::check(&dense);
        let dense = Arc::new(dense);
        specs
            .iter()
            .map(|spec| {
                let view = View::new(&dense.cfg, spec);
                let plan = full.as_ref().map_err(|e| e.clone()).and_then(|p| p.prune(&view).map_err(|e| format!("stream plan refused: {e}")));
                Self::build(dense.clone(), plan, Some(view))
            })
            .collect()
    }

    fn build(dense: Arc<Model>, plan: Result<Plan, String>, view: Option<View>) -> Engine {
        let plan = plan.and_then(Plan::compact);
        if let Ok(p) = &plan {
            // What `Engine::decode` indexes without bounds checks.
            let units = p.t.routing.bias().len();
            assert!(p.t.own_units.iter().flatten().all(|o| (o.u as usize) < units && (o.bit as usize) < ROUTE_CAP), "an own unit outside the routing layer or its routes");
            // What the routing layer's update indexes without bounds checks.
            assert!(p.t.route_gate.entries().all(|(u, _, _)| (*u as usize) < units) && p.t.owners.len() == units && p.t.owners.iter().flatten().all(|o| o.p as usize <= p.t.seq && (o.bit as usize) < ROUTE_CAP), "a routing entry outside its layer, or an owner outside the sequence or its routes");
            assert!(p.t.own_start.len() == p.t.seq + 1 && p.t.tracked.iter().all(|&h| h as usize <= p.t.seq));
        }
        let cfg = dense.cfg.clone();
        let (cos, sin) = rope(cfg.seq(), dense.group_position());
        let slots = Slots::new(&cfg);
        let l0 = &dense.layers[0];
        let (d, a) = (cfg.hidden, cfg.attn());
        // Layer 0 per query head, as HF's repeat_kv lays out the shared key and value heads.
        let (k, v) = l0.repeat_kv(&cfg);
        let v = Sparse::new(&v, a, d);
        let mut v_reach = vec![Vec::new(); a];
        for t in 0..cfg.vocab {
            let at = match slots.home[t] {
                Home::At(p) => p as usize..p as usize + 1,
                Home::Outputs => cfg.n_inputs..cfg.seq(),
                Home::Nowhere => continue,
            };
            let x: Vec<f32> = dense.embed[t * d..][..d].iter().map(|v| v * INV).collect();
            for (i, _) in v.apply(&x).iter().enumerate().filter(|&(_, &y)| y != 0.0) {
                v_reach[i].extend(at.clone());
            }
        }
        for r in &mut v_reach {
            r.sort_unstable();
            r.dedup();
        }
        let channels = plan.as_ref().map_or(d, |p| p.t.seg.count);
        let pending = plan.as_ref().map_or_else(|_| Buckets::default(), |p| Buckets::new(p.t.readers.iter().copied()));
        Engine {
            positions_id: 0,
            out_stamp: None,
            resync: false,
            verify: false,
            list: Vec::new(),
            replan_from: None,
            plan,
            l0: [Sparse::new(&l0.q, a, d), Sparse::new(&k, a, d), v],
            v_reach,
            state: State::default(),
            marks: Marks { chan: vec![false; channels], acc: vec![0; channels], ..Marks::default() },
            pending,
            qkv: (0..slots.len).map(|_| None).collect(),
            slots,
            moved: Vec::new(),
            vector_ids: HashMap::new(),
            interned: Vec::new(),
            interned_ids: HashMap::new(),
            plan_memo: HashMap::default(),
            plan_key: Vec::new(),
            cos,
            sin,
            force_fallback: false,
            timed: false,
            variant: 0,
            step_oracle: 0,
            rom_oracle: 0,
            check_next: false,
            #[cfg(feature = "testing")]
            fault: None,
            step_cache: None,
            stats: Stats::default(),
            memory: None,
            stream_records: view.as_ref().and_then(|v| crate::llama::tok::RecordSlots::find(&v.slots)),
            replay: None,
            demands: HashMap::new(),
            families: std::env::var_os("GBCFORMER_UNITS").map(|_| Self::families_of(&dense)),
            #[cfg(feature = "record")]
            recorder: view.as_ref().and_then(|v| stock::Recorder::open(&v.slots, v.cfg.n_inputs, &v.at)),
            #[cfg(feature = "record")]
            external: match view.as_ref() {
                Some(v) => stock::external(&v.slots, v.cfg.n_inputs),
                None => stock::external(&dense.slots, dense.cfg.n_inputs),
            },
            dense,
            view,
        }
    }

    /// Node families: a unit's name up to its first `.` or `=`, with opcode and CB
    /// one-hots folded into `op*` and `cb*`, and the routing layer's units as `route`.
    fn families_of(m: &Model) -> (Vec<String>, Vec<Vec<u16>>) {
        let mut names: Vec<String> = vec!["route".to_string()];
        let mut per_layer = Vec::new();
        for layer in &m.unit_names {
            let mut ids = vec![0u16; m.cfg.intermediate];
            for (u, n) in layer.iter().enumerate() {
                let stem = n.split(['.', '=']).next().unwrap_or("");
                let fam = if stem.len() == 4 && (stem.starts_with("op") || stem.starts_with("cb")) && stem[2..].chars().all(|c| c.is_ascii_hexdigit()) { format!("{}*", &stem[..2]) } else { stem.to_string() };
                let id = names.iter().position(|f| *f == fam).unwrap_or_else(|| {
                    names.push(fam);
                    names.len() - 1
                });
                ids[u] = id as u16;
            }
            per_layer.push(ids);
        }
        (names, per_layer)
    }

    /// The update census, largest families first, as (family, updates per fast pass).
    pub fn census(&self) -> Vec<(String, f64)> {
        let Some((names, _)) = &self.families else { return Vec::new() };
        let fast = self.stats.fast_passes.max(1) as f64;
        let mut rows: Vec<(String, f64)> = names.iter().zip(&self.stats.family_updates).map(|(n, &u)| (n.clone(), u as f64 / fast)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }

    /// The gate updates each changed channel caused, largest first, as (channel name,
    /// updates per fast pass). A wire channel is named by its wire, the others by their
    /// place in the layout.
    pub fn sources(&self) -> Vec<(String, f64)> {
        if self.families.is_none() {
            return Vec::new();
        }
        self.per_channel(&self.stats.source_updates)
    }

    /// The changes per fast pass of each channel that changed, largest first, named as
    /// `sources` names them: the change frequencies a schedule can weigh.
    pub fn wire_changes(&self) -> Vec<(String, f64)> {
        self.per_channel(&self.stats.channel_changes)
    }

    fn per_channel(&self, counts: &[u64]) -> Vec<(String, f64)> {
        let fast = self.stats.fast_passes.max(1) as f64;
        let lay = &self.dense.layout;
        let w0 = lay.wire_base();
        // A virtual channel is a dense channel's k-th occupant.
        let seg = self.plan.as_ref().ok().map(|p| &p.t.seg);
        let occupant = |v: usize| -> (usize, usize) {
            seg.and_then(|s| {
                (0..s.first.len()).find_map(|ch| if s.first[ch] as usize == v { Some((ch, 0)) } else { s.later[ch].iter().position(|&(_, id)| id as usize == v).map(|k| (ch, k + 1)) })
            })
            .unwrap_or((v, 0))
        };
        let names = crate::circuit::unique_names(&self.dense.wire_names);
        let name = |v: usize| -> String {
            let (ch, k) = occupant(v);
            if ch >= w0 && ch < lay.after_wires() {
                lay.occupants.get(ch - w0).and_then(|o| o.get(k)).and_then(|s| names.get(s.0 as usize)).cloned().unwrap_or_else(|| format!("wire{}", ch - w0))
            } else if ch >= lay.after_wires() {
                format!("mem{}", ch - lay.after_wires())
            } else {
                format!("ch{ch}")
            }
        };
        let mut rows: Vec<(String, f64)> = counts.iter().enumerate().filter(|&(_, &u)| u > 0).map(|(ch, &u)| (name(ch), u as f64 / fast)).collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }

    pub fn dense(&self) -> &Model {
        &self.dense
    }

    /// The dense pass's weights, for a test that changes one and shows the step oracle
    /// catches it; the fast plan keeps the weights it was checked from, so a normal build
    /// has no such hook (feature `testing` only). Panics when the weights are shared with
    /// another engine.
    #[cfg(feature = "testing")]
    pub fn dense_mut(&mut self) -> &mut Model {
        Arc::get_mut(&mut self.dense).expect("the dense weights are this engine's alone")
    }

    /// XORs the next fast pass's first decoded token with `x` before any check, for a test
    /// that the step oracle catches a wrong token (feature `testing` only).
    #[cfg(feature = "testing")]
    pub fn inject_fault(&mut self, x: usize) {
        self.fault = Some(x);
    }

    /// Why the integer fast path is off for this checkpoint, if it is.
    pub fn exact(&self) -> Result<(), &str> {
        self.plan.as_ref().map(|_| ()).map_err(String::as_str)
    }

    /// The fast plan's size: units the circuit and routing layers evaluate, memory
    /// heads it serves, whether it runs the positional stage, and the norm argument
    /// (Σ of the hard bounds' squares, the limit it must stay below).
    pub fn plan_size(&self) -> Option<(usize, usize, bool, (i64, i64))> {
        let p = self.plan.as_ref().ok()?;
        let t = &p.t;
        let units = t.circuit.iter().chain([&t.routing]).map(|l| l.c().iter().filter(|&&c| c != 0).count()).sum();
        let heads = p.mem.as_ref().map_or(0, |m| m.heads.len()) + t.pos.as_ref().map_or(0, |pos| pos.owned.len());
        Some((units, heads, t.pos.is_some(), t.norm))
    }

    pub fn nonzeros(&self) -> usize {
        let m = &self.dense;
        let count = |t: &[f32]| t.iter().filter(|&&w| w != 0.0).count();
        m.layers.iter().map(|l| [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down].iter().map(|t| count(t)).sum::<usize>()).sum::<usize>() + count(&m.lm_head)
    }
}

/// The engine moves between threads whole: no Rc, RefCell or raw pointer inside.
fn _assert_send() {
    fn f<T: Send>() {}
    f::<Engine>();
}
