//! Threshold-circuit IR. Every signal is a 0/1 wire (or an integer 0..255 in a
//! byte-valued node, `Node::ramp`, `docs/byte-channels.md`); every node is a weighted
//! sum of step functions of affine forms over earlier wires:
//!
//!   node = Σ coef · [lin ≥ θ]
//!
//! One node costs one level, so a level is exactly one transformer MLP. NOT, constants
//! and sums of mutually exclusive wires are affine and free.

use crate::f16acc::{self, Form, Unit, Vars, NO_GROUP};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct Sig(pub u32);

/// Affine form Σ w·sig + c over wires.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lin {
    pub terms: BTreeMap<Sig, i64>,
    pub c: i64,
}

impl Lin {
    pub fn konst(c: i64) -> Lin {
        Lin { terms: BTreeMap::new(), c }
    }
    fn add_term(mut self, s: Sig, w: i64) -> Lin {
        let e = self.terms.entry(s).or_insert(0);
        *e += w;
        if *e == 0 {
            self.terms.remove(&s);
        }
        self
    }
    pub fn max_level(&self, levels: &[u32]) -> u32 {
        self.terms.keys().map(|s| levels[s.0 as usize]).max().unwrap_or(0)
    }
    pub fn eval(&self, v: &[i64]) -> i64 {
        self.c + self.terms.iter().map(|(s, w)| w * v[s.0 as usize]).sum::<i64>()
    }
    /// Range of the form given each wire's range.
    pub fn bounds_in(&self, ranges: &[(i64, i64)]) -> (i64, i64) {
        self.terms.iter().fold((self.c, self.c), |(lo, hi), (s, &w)| {
            let (a, b) = ranges[s.0 as usize];
            (lo + (w * a).min(w * b), hi + (w * a).max(w * b))
        })
    }
}

impl From<Sig> for Lin {
    fn from(s: Sig) -> Lin {
        Lin::konst(0).add_term(s, 1)
    }
}
impl From<i64> for Lin {
    fn from(c: i64) -> Lin {
        Lin::konst(c)
    }
}
impl<T: Into<Lin>> Add<T> for Lin {
    type Output = Lin;
    fn add(self, rhs: T) -> Lin {
        let rhs = rhs.into();
        let mut out = self;
        out.c += rhs.c;
        for (s, w) in rhs.terms {
            out = out.add_term(s, w);
        }
        out
    }
}
impl<T: Into<Lin>> Sub<T> for Lin {
    type Output = Lin;
    fn sub(self, rhs: T) -> Lin {
        self + (-rhs.into())
    }
}
impl Neg for Lin {
    type Output = Lin;
    fn neg(self) -> Lin {
        self * -1
    }
}
impl Mul<i64> for Lin {
    type Output = Lin;
    fn mul(self, k: i64) -> Lin {
        let terms = if k == 0 { BTreeMap::new() } else { self.terms.into_iter().map(|(s, w)| (s, w * k)).collect() };
        Lin { terms, c: self.c * k }
    }
}
impl<T: Into<Lin>> Add<T> for Sig {
    type Output = Lin;
    fn add(self, rhs: T) -> Lin {
        Lin::from(self) + rhs
    }
}
impl<T: Into<Lin>> Sub<T> for Sig {
    type Output = Lin;
    fn sub(self, rhs: T) -> Lin {
        Lin::from(self) - rhs
    }
}
impl Mul<i64> for Sig {
    type Output = Lin;
    fn mul(self, k: i64) -> Lin {
        Lin::from(self) * k
    }
}

/// The largest live side and gate coefficient an f16 checkpoint holds, in units of the
/// step scale `llama::K` = 128: `128 · 511 ≤ 65,504` (`docs/stock-llama.md` §3). A step
/// whose form can exceed θ by more is emitted the other way round when that side fits.
pub const F16_SPAN: i64 = 511;
/// The largest |c − θ| of a step the two constant channels hold (`llama::CONSTANTS`).
pub const F16_BIAS: i64 = 32255;
/// The largest partial sum of a gate an f16 accumulator holds exactly, in units of K:
/// every term is a multiple of K = 2^7, exact up to the f16 maximum 65,504 = 511.75·K.
pub const F16_ACC_GATE: i64 = 511;
/// The largest partial sum of a node's down row an f16 accumulator holds exactly, in
/// units of `llama::UNIT` = 2^-7: f16 holds every multiple of 2^-7 up to 2^11 of them.
/// Two are left for the residual entry a fused accumulate adds and a clearing unit.
pub const F16_ACC_DOWN: i64 = 2046;

/// `coef · [lin ≥ theta]`
#[derive(Clone, Debug)]
pub struct Step {
    pub coef: i64,
    pub lin: Lin,
    pub theta: i64,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub out: Sig,
    pub steps: Vec<Step>,
    /// Always-true steps folded out of the wire; the signal is `out + konst`, which is
    /// 0/1 even when the wire alone is not.
    pub konst: i64,
    pub name: String,
    /// Byte-valued channels (`docs/byte-channels.md`): each step contributes
    /// `coef · max(0, lin − θ + 1)`, the emitter's first unit alone, instead of
    /// `coef · [lin ≥ θ]`. The two agree wherever `lin ≤ θ`.
    pub ramp: bool,
}

impl Node {
    /// The wire's value under the wire values `v` (without `konst`).
    pub fn value(&self, v: &[i64]) -> i64 {
        self.steps.iter().map(|s| {
            let x = s.lin.eval(v);
            if self.ramp { s.coef * (x - s.theta + 1).max(0) } else if x >= s.theta { s.coef } else { 0 }
        }).sum()
    }
}

/// What a memory head asks the cache for: the 24-bit physical byte address formed by
/// these wires (least significant first), or one fixed address. The head reads the line
/// holding it (bits `LINE_BITS..24`); the low bits pick the byte in the nodes that read
/// the line (`low_match`).
#[derive(Clone, Debug)]
pub enum Query {
    Wires(Vec<Lin>),
    Const(u32),
}

impl Query {
    /// A column select: the line `bits` name when the 0/1 `gate` is 1, else the line
    /// holding `off`, as one linear query. Address bit b's form is
    /// `a_b + (2·off_b − 1)·(1 − gate)`: `a_b` with the gate on, and with it off a form
    /// in {−1, 0} or {1, 2} whose sign is `off_b`. The head's query is `S·(2·form − 1)`,
    /// ±S or ±3S per bit, so the line its signs spell beats every other by 2S per
    /// mismatched bit, as an ungated query does (`memory::attn::Params::check`).
    pub fn gated(bits: &[Lin], gate: &Lin, off: u32) -> Query {
        assert_eq!(bits.len(), ADDR_BITS, "a query is {ADDR_BITS} address bits");
        Query::Wires((0..ADDR_BITS).map(|b| if b < LINE_BITS { bits[b].clone() } else { bits[b].clone() + (one() - gate.clone()) * (2 * (off >> b & 1) as i64 - 1) }).collect())
    }
}

/// The address bit a query form spells: its sign, `[form ≥ 1]`, over a form in [−1, 2]
/// (`Query::gated`); an ungated form is 0 or 1 and spells itself.
pub fn address_bit(form: i64) -> u32 {
    assert!((-1..=2).contains(&form), "an address bit form is {form}, outside [−1, 2]");
    (form >= 1) as u32
}

/// Bytes per memory line: a cell of the cache holds one line, and a head reads one
/// (`docs/stock-llama.md` §10).
pub const LINE: usize = crate::memory::layout::LINE;
/// The address bits that pick a byte within a line.
pub const LINE_BITS: usize = crate::memory::layout::LINE_BITS;
/// Bits of a physical byte address, a query's length.
pub const ADDR_BITS: usize = crate::memory::layout::ADDR_BITS;

/// A line's wires: bit i of byte k is `line[k][i]`.
pub type Line = [[Sig; 8]; LINE];

/// An attention head at `layer` that reads one line of the memory and writes its bits
/// into `8·LINE` wires of level `layer`. `after` is the node count when the head was
/// declared, so evaluation interleaves heads and nodes in creation order.
#[derive(Clone, Debug)]
pub struct Head {
    pub name: String,
    pub layer: u32,
    pub query: Query,
    /// The line's bits, byte k's bit i at `8k + i`.
    pub out: Vec<Sig>,
    after: usize,
}

/// `Σ_b (k_b ? low_b : 1 − low_b)`: `LINE_BITS` exactly when the 0/1 forms `low` spell
/// `k`, at most `LINE_BITS − 1` otherwise.
pub fn low_match(low: &[Lin], k: usize) -> Lin {
    low.iter().enumerate().fold(zero(), |s, (b, l)| s + if k >> b & 1 == 1 { l.clone() } else { one() - l.clone() })
}

/// How an output slot's byte is tokenised, and what a memory record reads from it: the
/// shared byte tokens; byte `byte` of write `write`'s physical address; or byte `byte`
/// of the line write `write` leaves at that address (the whole line, both writes merged
/// when they share it). Write 0's address and line slots decode to per-slot token
/// families, which a record group repeats; write 1's line slots to the shared tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutKind {
    Shared,
    Address { write: u8, byte: u8 },
    Line { write: u8, byte: u8 },
}

/// A nibble-valued signal (`docs/byte-channels.md`): an integer form whose value lies
/// in [0, hi] on every input, as the builder that made it proves, though its terms'
/// ranges may allow more.
#[derive(Clone, Debug)]
pub struct Nib {
    pub lin: Lin,
    pub hi: i64,
}

impl Nib {
    pub fn konst(v: i64) -> Nib {
        Nib { lin: Lin::konst(v), hi: v }
    }
    /// The value of up to four 0/1 bits, least significant first.
    pub fn of_bits(bits: &[Lin]) -> Nib {
        assert!(bits.len() <= 4);
        Nib { lin: value(bits, bits.len()), hi: (1 << bits.len()) - 1 }
    }
}

/// A byte-wide input or output field, least significant bit first.
pub type Byte = [Lin; 8];

#[derive(Default)]
pub struct Circuit {
    pub levels: Vec<u32>,
    /// Value range of each wire. A node's signal `out + konst` is 0/1, so its wire
    /// lies in [-konst, 1 - konst].
    pub ranges: Vec<(i64, i64)>,
    /// Per wire, its one-hot group (`NO_GROUP` for none): at most one wire of a group
    /// is 1 at any position (the position wires, an input's one-hot, a head's nibble
    /// one-hots), so a form's terms over one group reach only one weight at a time.
    pub groups: Vec<u32>,
    pub names: Vec<String>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<(String, [Sig; 8])>,
    /// Inputs also delivered as a 256-way one-hot of their byte: (input index, wires).
    pub onehots: Vec<(usize, Vec<Sig>)>,
    pub outputs: Vec<(String, Byte)>,
    /// Per output, how its byte is tokenised.
    pub out_kinds: Vec<OutKind>,
    /// Per output, its byte as two nibble forms, low first, instead of eight bits
    /// (`output_nibbles`); the bits are then unused.
    pub value_outputs: Vec<Option<[Lin; 2]>>,
    /// Level-0 nibble wires: (input index, nibble, wire), nibble 0 the input byte's low
    /// four bits as one integer 0..15, nibble 1 its high four (`input_nibbles`).
    pub values: Vec<(usize, usize, Sig)>,
    pub heads: Vec<Head>,
    /// Position wires: wire k is 1 at output position k and 0 everywhere else. They
    /// take no residual channel: the checkpoint reads each from its position's slot
    /// identity, so a node reading one computes a different value at each position.
    pub positions: Vec<Sig>,
    /// Per part `concat` placed side by side, the input slot of each of its inputs, in
    /// its own order (empty for one part). The parts share the input slots: a part's
    /// k-th input without nibble wires holds slot k, and an input with nibble wires a
    /// slot no other part holds. `inputs` names each slot after its first holder.
    pub parts: Vec<Vec<usize>>,
    /// The other holders of a shared input slot: the slot, the part's own name for it and
    /// its wires. Layer 0 gathers a slot once, so every holder's bit wire sits in the one
    /// channel the gather writes (`overlay_channels`).
    pub held: Vec<(usize, String, [Sig; 8])>,
    /// The first wire of each part `concat` placed side by side (empty for one part).
    pub offsets: Vec<u32>,
    /// The output count of each part `concat` kept (empty for one part).
    pub out_parts: Vec<usize>,
    /// Selector forms already materialised as wires, by their terms and constant.
    selectors: std::collections::HashMap<(Vec<(Sig, i64)>, i64), Lin>,
    /// The digit comparisons f16 splitting made, by form and threshold.
    shared_ge: std::collections::HashMap<(Vec<(Sig, i64)>, i64, i64), Lin>,
    /// Per node wire, its twin of the other polarity (`balance`).
    twins: std::collections::HashMap<usize, Sig>,
    next_group: u32,
}

impl Circuit {
    fn wire(&mut self, level: u32, range: (i64, i64), name: String) -> Sig {
        self.levels.push(level);
        self.ranges.push(range);
        self.groups.push(NO_GROUP);
        self.names.push(name);
        Sig(self.levels.len() as u32 - 1)
    }

    /// Marks `wires` as one one-hot group.
    fn one_hot(&mut self, wires: &[Sig]) {
        let g = self.next_group;
        self.next_group += 1;
        for s in wires {
            self.groups[s.0 as usize] = g;
        }
    }

    /// The form's range when the wires of each one-hot group are 1 one at a time.
    pub fn bounds_hot(&self, l: &Lin) -> (i64, i64) {
        f16acc::bounds(&Self::form(l), self)
    }

    fn form(l: &Lin) -> Form {
        Form { terms: l.terms.iter().map(|(s, &w)| (s.0 as usize, w)).collect(), c: l.c }
    }

    pub fn input(&mut self, name: &str) -> [Sig; 8] {
        let bits = std::array::from_fn(|i| self.wire(0, (0, 1), format!("{name}.{i}")));
        self.inputs.push((name.to_string(), bits));
        bits
    }

    /// One position wire per output slot from the first: `n` level-0 wires, the k-th 1
    /// exactly at output position k.
    pub fn positions(&mut self, n: usize) -> Vec<Sig> {
        assert!(self.positions.is_empty(), "a circuit has one set of position wires");
        self.positions = (0..n).map(|k| self.wire(0, (0, 1), format!("@{k}"))).collect();
        let p = self.positions.clone();
        self.one_hot(&p);
        self.positions.clone()
    }

    /// Whether wire `w` is a position wire.
    pub fn is_position(&self, w: usize) -> bool {
        self.positions.first().is_some_and(|f| (f.0 as usize..f.0 as usize + self.positions.len()).contains(&w))
    }

    /// A byte input together with a one-hot wire per value. The one-hot costs no
    /// level: the checkpoint's token embeddings carry it and layer 0 gathers it.
    pub fn input_onehot(&mut self, name: &str) -> ([Sig; 8], Vec<Sig>) {
        let bits = self.input(name);
        let hot: Vec<Sig> = (0..256).map(|v| self.wire(0, (0, 1), format!("{name}={v:02x}"))).collect();
        self.one_hot(&hot);
        self.onehots.push((self.inputs.len() - 1, hot.clone()));
        (bits, hot)
    }

    pub fn output(&mut self, name: &str, bits: Byte) {
        self.output_kind(name, bits, OutKind::Shared);
    }

    pub fn output_kind(&mut self, name: &str, bits: Byte, kind: OutKind) {
        self.outputs.push((name.to_string(), bits));
        self.out_kinds.push(kind);
        self.value_outputs.push(None);
    }

    /// The byte of input `name` as two level-0 wires of range 0..15, its low and high
    /// nibbles, beside its bits (`docs/byte-channels.md`).
    pub fn input_nibbles(&mut self, name: &str) -> [Nib; 2] {
        let idx = self.inputs.iter().position(|(n, _)| n == name).unwrap_or_else(|| panic!("no input {name}"));
        std::array::from_fn(|h| {
            let s = match self.values.iter().find(|&&(i, k, _)| i == idx && k == h) {
                Some(&(_, _, s)) => s,
                None => {
                    let s = self.wire(0, (0, 15), format!("{name}#{}", ["lo", "hi"][h]));
                    self.values.push((idx, h, s));
                    s
                }
            };
            Nib { lin: Lin::from(s), hi: 15 }
        })
    }

    /// An output whose byte is two nibble forms, low first, not eight bits.
    pub fn output_nibbles(&mut self, name: &str, kind: OutKind, v: [Nib; 2]) {
        for n in &v {
            assert!((0..=15).contains(&n.hi), "{name}: a nibble ranges over [0, {}]", n.hi);
        }
        self.output_kind(name, std::array::from_fn(|_| zero()), kind);
        *self.value_outputs.last_mut().unwrap() = Some(v.map(|n| n.lin));
    }

    /// Every form an output reads: its bits, or its two nibbles.
    pub fn output_lins(&self) -> impl Iterator<Item = &Lin> {
        self.outputs.iter().flat_map(|(_, b)| b.iter()).chain(self.value_outputs.iter().flatten().flatten())
    }

    /// Marks `wires` as one one-hot group: on every input at most one of them is 1, a
    /// contract the evaluator asserts. The f16 analysis then counts a form's terms over
    /// them one at a time.
    pub fn declare_one_hot(&mut self, wires: &[Lin]) {
        let sigs: Vec<Sig> = wires.iter().flat_map(|l| l.terms.keys().copied()).collect();
        for s in &sigs {
            assert!(self.groups[s.0 as usize] == NO_GROUP && self.ranges[s.0 as usize] == (0, 1), "{} cannot join a one-hot group", self.names[s.0 as usize]);
        }
        self.one_hot(&sigs);
    }

    /// A node of nibble-valued channels, placed as given: no f16 legalisation. With
    /// `ramp` each step contributes `coef · max(0, lin − θ + 1)`, otherwise
    /// `coef · [lin ≥ θ]`. Steps constant over their inputs' ranges fold into the
    /// returned form. The wire's range is what the steps can sum to unless `range`
    /// gives it, a contract the evaluator asserts and the engine guards (a mux's
    /// exclusive cases).
    pub fn value_node_in(&mut self, name: &str, steps: Vec<Step>, ramp: bool, range: Option<(i64, i64)>) -> Lin {
        let mut folded = Lin::konst(0);
        let mut live = Vec::new();
        let (mut lo, mut hi) = (0i64, 0i64);
        for s in steps {
            let (a, b) = self.bounds(&s.lin);
            let (x, y) = if ramp { ((a - s.theta + 1).max(0), (b - s.theta + 1).max(0)) } else { ((a >= s.theta) as i64, (b >= s.theta) as i64) };
            if x == y && (!ramp || x == 0) {
                folded = folded + s.coef * x;
                continue;
            }
            lo += (s.coef * x).min(s.coef * y);
            hi += (s.coef * x).max(s.coef * y);
            live.push(s);
        }
        if live.is_empty() {
            return folded;
        }
        let (lo, hi) = range.unwrap_or((lo, hi));
        let level = 1 + live.iter().map(|s| s.lin.max_level(&self.levels)).max().unwrap();
        let out = self.wire(level, (lo, hi), name.to_string());
        self.nodes.push(Node { out, steps: live, konst: 0, name: name.to_string(), ramp });
        folded + Lin::from(out)
    }

    /// `value` when every gate is 1, else 0: one ramp unit per case,
    /// `max(0, v + m·(Σ gates − n))` with m the largest value v's terms allow, so an
    /// unselected case is 0. Gates are 0/1 and at most one case fires, so the wire is the
    /// selected case's value, in [0, the largest case's `hi`].
    pub fn value_mux(&mut self, name: &str, cases: &[(Vec<Lin>, Nib)]) -> Nib {
        let level = 1 + cases.iter().flat_map(|(gs, v)| gs.iter().chain([&v.lin])).map(|l| l.max_level(&self.levels)).max().unwrap_or(0);
        let mut top = 0;
        let steps = cases
            .iter()
            .map(|(gs, v)| {
                assert!(v.hi >= 0, "{name}: a case value ranges over [0, {}]", v.hi);
                top = top.max(v.hi);
                let m = self.bounds(&v.lin).1.max(1);
                let mut gs: Vec<Lin> = gs.iter().map(|g| if g.terms.len() > 3 && g.max_level(&self.levels) + 1 < level { self.selector(g) } else { g.clone() }).collect();
                // An f16 unit's live side, v + m·(Σ gates − n) at the gates' interval
                // maxima, stays within F16_SPAN: a gate whose terms can sum past 1 is
                // read through its 0/1 selector wire instead.
                let live = |c: &Circuit, gs: &[Lin]| m + gs.iter().map(|g| m * (c.bounds(g).1 - 1)).sum::<i64>();
                if live(self, &gs) > F16_SPAN {
                    gs = gs.iter().map(|g| if self.bounds(g).1 > 1 { self.selector(g) } else { g.clone() }).collect();
                }
                let lin = gs.iter().fold(v.lin.clone(), |a, g| a + g.clone() * m);
                Step { coef: 1, lin, theta: m * gs.len() as i64 + 1 }
            })
            .collect();
        Nib { lin: self.value_node_in(name, steps, true, Some((0, top))), hi: top }
    }

    /// A memory head at `layer`: `8·LINE` new wires of that level holding the line of
    /// the queried address. The query's line bits must be readable by that layer (level
    /// at most `layer`); its low bits are for the nodes that pick a byte.
    pub fn head(&mut self, name: &str, layer: u32, query: Query) -> Line {
        self.head_onto(name, layer, query, &[])
    }

    /// A bus: eight wires of level `layer` that several heads drive (`head_onto`). Each
    /// wire is the sum of its drivers' bits, 0 or 1 because at most one driver reads a
    /// line other than a zero one.
    pub fn bus(&mut self, name: &str, layer: u32) -> [Sig; 8] {
        std::array::from_fn(|i| self.wire(layer, (0, 1), format!("{name}[{i}]")))
    }

    /// `head` with the bytes `lanes` names written onto existing bus wires of level
    /// `layer` (`bus`) instead of new ones; `W_o` adds every driver's bits into the
    /// bus's channels.
    pub fn head_onto(&mut self, name: &str, layer: u32, query: Query, lanes: &[(usize, [Sig; 8])]) -> Line {
        if let Query::Wires(w) = &query {
            assert_eq!(w.len(), ADDR_BITS, "{name}: a query is {ADDR_BITS} address bits");
            let lvl = self.level_of(&w[LINE_BITS..]);
            assert!(lvl <= layer, "{name}: query wires are at level {lvl}, above layer {layer}");
        }
        assert!(lanes.iter().flat_map(|(_, b)| b).all(|&s| self.level(s) == layer), "{name}: a bus wire is not of level {layer}");
        let line: Line = std::array::from_fn(|k| match lanes.iter().find(|(l, _)| *l == k) {
            Some((_, byte)) => *byte,
            None => std::array::from_fn(|i| self.wire(layer, (0, 1), format!("{name}.{k}.{i}"))),
        });
        self.heads.push(Head { name: name.to_string(), layer, query, out: line.iter().flatten().copied().collect(), after: self.nodes.len() });
        line
    }

    /// Every (wire, layer) a head reads: its query's line bits.
    fn head_reads(&self) -> impl Iterator<Item = (Sig, u32)> + '_ {
        self.heads.iter().flat_map(|h| {
            let q: &[Lin] = match &h.query {
                Query::Wires(w) => &w[LINE_BITS..],
                Query::Const(_) => &[],
            };
            q.iter().flat_map(|l| l.terms.keys().copied()).map(move |s| (s, h.layer))
        })
    }

    /// The level of the highest wire these forms read.
    pub fn level_of(&self, lins: &[Lin]) -> u32 {
        lins.iter().map(|l| l.max_level(&self.levels)).max().unwrap_or(0)
    }

    /// The address a head's query names under the wire values `v`.
    pub fn query_address(&self, q: &Query, v: &[i64]) -> u32 {
        match q {
            Query::Const(a) => *a,
            Query::Wires(w) => w.iter().enumerate().fold(0u32, |acc, (i, l)| acc | address_bit(l.eval(v)) << i),
        }
    }

    pub fn level(&self, s: Sig) -> u32 {
        self.levels[s.0 as usize]
    }

    pub fn bounds(&self, l: &Lin) -> (i64, i64) {
        l.bounds_in(&self.ranges)
    }

    /// New wire = Σ coef·[lin ≥ θ]. Steps that are constant over the inputs' range fold away.
    pub fn node(&mut self, name: &str, steps: Vec<Step>) -> Lin {
        self.node_at(name, 0, steps)
    }

    /// `node`, placed no lower than level `floor`. The residual keeps every earlier
    /// wire, so a node may sit above the level its inputs force; a floor spreads nodes
    /// across the levels without copy wires.
    pub fn node_at(&mut self, name: &str, floor: u32, steps: Vec<Step>) -> Lin {
        let mut konst = 0;
        let mut live = Vec::new();
        let mut todo = steps;
        let mut i = 0;
        while i < todo.len() {
            let mut s = todo[i].clone();
            i += 1;
            let (lo, hi) = self.bounds(&s.lin);
            if lo >= s.theta {
                konst += s.coef;
                continue;
            } else if hi < s.theta {
                continue;
            }
            if !self.step_ok(&s.lin, s.theta) {
                // Rewrite the step as the same truth table over smaller weights. The
                // tries, in order: reduction with literal runs merged only where that
                // costs no level; the same with some wires read through their twins of
                // the other polarity (`balance`), which costs no level either; twins
                // alone; reduction with runs merged at any level, without and with twins.
                let tries = [(Some(false), false), (Some(false), true), (None, true), (Some(true), false), (Some(true), true)];
                for (reduce, balanced) in tries {
                    let mark = self.mark();
                    let (lin, theta) = match reduce {
                        Some(any_level) => self.reduce(&s.lin, s.theta, any_level),
                        None => (s.lin.clone(), s.theta),
                    };
                    let lin = if balanced { self.balance(&lin, theta) } else { Some(lin) };
                    if let Some(lin) = lin.filter(|l| self.step_ok(l, theta)) {
                        s = Step { coef: s.coef, lin, theta };
                        break;
                    }
                    self.rollback(mark);
                }
            }
            let (lo, hi) = self.bounds(&s.lin);
            let split = if self.step_ok(&s.lin, s.theta) { None } else { [256, 64, 16, 4].into_iter().find_map(|b| self.split_step(name, &s, b)) };
            if let Some(split) = split {
                todo.extend(split);
            } else if hi - s.theta + 1 > F16_SPAN && s.theta - lo <= F16_SPAN {
                // coef·[s ≥ θ] = coef − coef·[−s ≥ 1 − θ]: the unit's live side is then
                // θ − lo instead of hi − θ + 1, and the coef joins the folded constant.
                konst += s.coef;
                live.push(Step { coef: -s.coef, lin: s.lin * -1, theta: 1 - s.theta });
            } else {
                live.push(s);
            }
        }
        if live.is_empty() {
            return Lin::konst(konst);
        }
        if self.down_mass(&live) > F16_ACC_DOWN {
            konst += self.reorient(&mut live);
        }
        if self.down_mass(&live) > F16_ACC_DOWN
            && let Some(runs) = self.runs(&live, konst)
        {
            let mut sum = Lin::konst(konst);
            for (k, (steps, rk)) in runs.into_iter().enumerate() {
                sum = sum + self.place(&format!("{name}#{k}"), floor, steps, rk) - rk;
            }
            return sum;
        }
        self.place(name, floor, live, konst)
    }

    /// A node wire for live steps whose signal, with `konst`, is 0/1.
    fn place(&mut self, name: &str, floor: u32, live: Vec<Step>, konst: i64) -> Lin {
        let level = (1 + live.iter().map(|s| s.lin.max_level(&self.levels)).max().unwrap()).max(floor);
        let out = self.wire(level, (-konst, 1 - konst), name.to_string());
        self.nodes.push(Node { out, steps: live, konst, name: name.to_string(), ramp: false });
        Lin::from(out) + konst
    }

    /// The wire and node counts, for `rollback`.
    fn mark(&self) -> (usize, usize) {
        (self.levels.len(), self.nodes.len())
    }

    /// Drops every wire and node made since `mark`, and the shared forms naming them,
    /// to undo a rejected rewrite.
    fn rollback(&mut self, (wires, nodes): (usize, usize)) {
        self.levels.truncate(wires);
        self.ranges.truncate(wires);
        self.groups.truncate(wires);
        self.names.truncate(wires);
        self.nodes.truncate(nodes);
        let fresh = |l: &Lin| l.terms.keys().any(|s| s.0 as usize >= wires);
        self.shared_ge.retain(|_, l| !fresh(l));
        self.selectors.retain(|_, l| !fresh(l));
        self.twins.retain(|&x, t| x < wires && (t.0 as usize) < wires);
    }

    /// `[lin ≥ θ]` over twins of some of its node wires: a twin carries the same 0/1
    /// signal over the opposite wire range, so its products fall on the other side of a
    /// gate's partial sums. Greedily turns the wires that most lower the gate's mass;
    /// None when no set brings it within `F16_ACC_GATE`.
    fn balance(&mut self, lin: &Lin, theta: i64) -> Option<Lin> {
        let konst = |c: &Circuit, x: usize| -c.ranges[x].0;
        let units = self.units_of(lin, theta);
        let mass = |c: &Circuit, flip: &BTreeSet<usize>| -> i64 {
            let shift: i64 = flip.iter().map(|&x| lin.terms[&Sig(x as u32)] * (1 - 2 * konst(c, x))).sum();
            let vars = Flipped { c, flip };
            let terms: Vec<(usize, i64)> = lin.terms.iter().map(|(s, &w)| (s.0 as usize, w)).collect();
            units.iter().map(|&(off, _)| f16acc::gate_mass(&Form { terms: terms.clone(), c: lin.c + shift - theta + off }, &vars)).max().unwrap()
        };
        let cands: Vec<usize> = lin.terms.keys().map(|s| s.0 as usize).filter(|&x| self.groups[x] == NO_GROUP && self.twin_steps(x).is_some()).collect();
        let mut flip = BTreeSet::new();
        let mut m = mass(self, &flip);
        while m > F16_ACC_GATE {
            let (best, x) = cands.iter().filter(|x| !flip.contains(*x)).map(|&x| (mass(self, &flip.iter().copied().chain([x]).collect()), x)).min()?;
            if best >= m {
                return None;
            }
            flip.insert(x);
            m = best;
        }
        let mut out = lin.clone();
        for x in flip {
            let w = lin.terms[&Sig(x as u32)];
            let k = konst(self, x);
            let t = self.twin(x);
            out = out - Lin::from(Sig(x as u32)) * w + Lin::from(t) * w + w * (1 - 2 * k);
        }
        Some(out)
    }

    /// The steps of a twin of node wire `x`, the same signal with the other constant:
    /// one step of coefficient `1 − 2κ` turned, so the folded constant κ becomes 1 − κ.
    fn twin_steps(&self, x: usize) -> Option<(Vec<Step>, i64, u32, String)> {
        if self.ranges[x].1 - self.ranges[x].0 != 1 {
            return None;
        }
        let n = self.nodes.iter().rev().find(|n| n.out.0 as usize == x)?;
        let i = n.steps.iter().position(|s| s.coef == 1 - 2 * n.konst)?;
        let mut steps = n.steps.clone();
        let s = &steps[i];
        steps[i] = Step { coef: -s.coef, lin: s.lin.clone() * -1, theta: 1 - s.theta };
        if !self.step_ok(&steps[i].lin, steps[i].theta) || self.down_mass(&steps) > F16_ACC_DOWN {
            return None;
        }
        Some((steps, 1 - n.konst, self.levels[x], format!("{}~", n.name)))
    }

    /// The twin of node wire `x`, made once.
    fn twin(&mut self, x: usize) -> Sig {
        if let Some(&t) = self.twins.get(&x) {
            return t;
        }
        let (steps, konst, level, name) = self.twin_steps(x).expect("a twin of a wire that has none");
        self.place(&name, level, steps, konst);
        let t = Sig(self.levels.len() as u32 - 1);
        self.twins.insert(x, t);
        t
    }

    /// Whether one unit pair holds `[lin ≥ θ]` in f16 weights and f16 sums.
    pub fn step_ok(&self, lin: &Lin, theta: i64) -> bool {
        self.f16_ok(lin, theta) && self.gate_mass(lin, theta) <= F16_ACC_GATE
    }

    /// Turns each step toward the end of its form's range it is nearer, where its units
    /// stay small: `coef·[s ≥ θ] = coef − coef·[−s ≥ 1 − θ]`. Returns the constant the
    /// turned steps fold out.
    fn reorient(&self, live: &mut [Step]) -> i64 {
        let mut konst = 0;
        for s in live.iter_mut() {
            let (lo, hi) = self.bounds_hot(&s.lin);
            let turned = Step { coef: -s.coef, lin: s.lin.clone() * -1, theta: 1 - s.theta };
            if s.theta - lo < hi - s.theta + 1 && self.step_ok(&turned.lin, turned.theta) {
                konst += s.coef;
                *s = turned;
            }
        }
        konst
    }

    /// A staircase over one form whose down row f16 cannot sum, cut into runs of
    /// consecutive thresholds it can, each with the constant that makes its signal 0/1.
    /// Read as `Σ c·[s ≥ θ]` over the form s (a turned step `−c·[−s ≥ 1 − θ]` is
    /// `c·[s ≥ θ] − c`), the node is 0/1 everywhere, so a run's sum is the node's value
    /// past the run less its value before: 0/1 when that is 0, −1/0 when it is 1.
    /// None when the steps do not share one form.
    fn runs(&self, live: &[Step], konst: i64) -> Option<Vec<(Vec<Step>, i64)>> {
        let base = live[0].lin.clone();
        let neg = base.clone() * -1;
        if !live.iter().all(|s| s.lin == base || s.lin == neg) {
            return None;
        }
        // (θ on s, c, the turned step's folded constant) per step.
        let plain = |s: &Step| if s.lin == base { (s.theta, s.coef, 0) } else { (1 - s.theta, -s.coef, -s.coef) };
        let mut steps: Vec<Step> = live.to_vec();
        steps.sort_by_key(|s| plain(s).0);
        let mut value = konst - steps.iter().map(|s| plain(s).2).sum::<i64>();
        let mut runs: Vec<(Vec<Step>, i64)> = Vec::new();
        let (mut run, mut turned, mut start) = (Vec::new(), 0, value);
        for s in steps {
            run.push(s.clone());
            if run.len() > 1 && self.down_mass(&run) > F16_ACC_DOWN {
                run.pop();
                runs.push((std::mem::take(&mut run), turned + start));
                run.push(s.clone());
                (turned, start) = (0, value);
            }
            let (_, c, t) = plain(&s);
            turned += t;
            value += c;
        }
        runs.push((run, turned + start));
        Some(runs)
    }

    /// `[lin ≥ θ]` with the same value on every input in the wires' ranges, over smaller
    /// weights. Shannon expansion: when one value of a 0/1 term ℓ (weight w) decides
    /// the step alone, the step is `ℓ ∨ [rest ≥ θ]` or `ℓ ∧ [rest ≥ θ − w]`; peeling such
    /// terms leaves a chain of ANDs and ORs around a core step. Rebuilt from the core
    /// out, each literal takes the weight that makes it decide, no more, so
    /// binary-weighted comparisons reduce to Fibonacci-like weights. A run of ANDs (or
    /// ORs) costs one such weight per literal; the literals of a run that sit below the
    /// step's last input become one shared AND (OR) wire, which costs no level.
    fn reduce(&mut self, lin: &Lin, theta: i64, any_level: bool) -> (Lin, i64) {
        #[derive(Clone, Copy, PartialEq)]
        enum Op {
            And,
            Or,
        }
        let top = lin.max_level(&self.levels);
        let (mut chain, mut r, mut t): (Vec<(Op, Lin)>, Lin, i64) = (Vec::new(), lin.clone(), theta);
        'peel: loop {
            // A constant core ends the chain, since peeling it further would add literals
            // it does not depend on.
            let (lo, hi) = self.bounds(&r);
            if lo >= t || hi < t {
                break;
            }
            let mut terms: Vec<(Sig, i64)> = r.terms.iter().filter(|(s, _)| self.groups[s.0 as usize] == NO_GROUP && self.ranges[s.0 as usize].1 - self.ranges[s.0 as usize].0 == 1).map(|(&s, &w)| (s, w)).collect();
            terms.sort_by_key(|&(s, w)| (std::cmp::Reverse(w.abs()), s));
            for (x, w) in terms {
                let (a, b) = self.ranges[x.0 as usize];
                // r = rest + |w|·ℓ, ℓ = 1 at x's favourable end.
                let (lit, rest) = if w > 0 { (Lin::from(x) - a, r.clone() - Lin::from(x) * w + w * a) } else { (Lin::konst(b) - x, r.clone() - Lin::from(x) * w + w * b) };
                let (rlo, rhi) = self.bounds(&rest);
                if t - w.abs() <= rlo {
                    chain.push((Op::Or, lit));
                    r = rest;
                    continue 'peel;
                }
                if t > rhi {
                    chain.push((Op::And, lit));
                    (r, t) = (rest, t - w.abs());
                    continue 'peel;
                }
            }
            break;
        }
        // A core that no longer depends on its form: the last literal is the core.
        let (lo, hi) = self.bounds(&r);
        if (lo >= t || hi < t) && let Some((_, lit)) = chain.pop() {
            (r, t) = (lit, 1);
        }
        // Runs of one op: the literals below the top level become one wire.
        let mut merged: Vec<(Op, Lin)> = Vec::new();
        let mut i = 0;
        while i < chain.len() {
            let op = chain[i].0;
            let j = (i..chain.len()).find(|&j| chain[j].0 != op).unwrap_or(chain.len());
            let (early, late): (Vec<Lin>, Vec<Lin>) = chain[i..j].iter().map(|(_, l)| l.clone()).partition(|l| any_level || l.max_level(&self.levels) < top);
            if early.len() > 1 {
                let sum = early.iter().cloned().fold(Lin::konst(0), |a, l| a + l);
                let n = if op == Op::And { early.len() as i64 } else { 1 };
                merged.push((op, self.ge_shared(if op == Op::And { "and" } else { "or" }, sum, n)));
            } else {
                merged.extend(early.into_iter().map(|l| (op, l)));
            }
            merged.extend(late.into_iter().map(|l| (op, l)));
            i = j;
        }
        for (op, lit) in merged.into_iter().rev() {
            let (lo, hi) = self.bounds(&r);
            match op {
                Op::Or => r = r + lit * (t - lo),
                Op::And => {
                    let b = hi - t + 1;
                    (r, t) = (r + lit * b, t + b);
                }
            }
        }
        (r, t)
    }

    /// Whether an f16 checkpoint holds `[lin ≥ θ]` as one unit pair, in one of its two
    /// orientations: every coefficient within `F16_SPAN`, the live side within it, and
    /// the bias within what the two constant channels hold (`llama::CONSTANTS`).
    pub fn f16_ok(&self, lin: &Lin, theta: i64) -> bool {
        let (lo, hi) = self.bounds(lin);
        let up = hi - theta + 1 <= F16_SPAN && (lin.c - theta).abs() + 1 <= F16_BIAS;
        let down = theta - lo <= F16_SPAN && (theta - 1 - lin.c).abs() + 1 <= F16_BIAS;
        lin.terms.values().all(|w| w.abs() <= F16_SPAN) && (up || down)
    }

    /// The emitted units of `[lin ≥ θ]`: the bias offset of each (`relu(K(s − θ + off))`)
    /// and the sign of its down weight relative to the step's coefficient.
    fn units_of(&self, lin: &Lin, theta: i64) -> &'static [(i64, i64)] {
        if self.bounds(lin).1 <= theta { &[(1, 1)] } else { &[(1, 1), (0, -1)] }
    }

    /// The emitted units of a step, as `f16acc` reads them.
    fn step_units(&self, st: &Step) -> Vec<Unit> {
        let f = Self::form(&st.lin);
        self.units_of(&st.lin, st.theta).iter().map(|&(off, sign)| Unit { form: Form { terms: f.terms.clone(), c: f.c - st.theta + off }, down: sign * st.coef }).collect()
    }

    /// The largest magnitude a partial sum of any of the step's gate rows reaches, in
    /// any order, in units of K (`f16acc::gate_mass`). An f16 accumulator holds every
    /// such sum exactly while it is at most `F16_ACC_GATE`.
    pub fn gate_mass(&self, lin: &Lin, theta: i64) -> i64 {
        self.step_units(&Step { coef: 1, lin: lin.clone(), theta }).iter().map(|u| f16acc::gate_mass(&u.form, self)).max().unwrap()
    }

    /// The largest magnitude a partial sum of a node's down row reaches, in any order,
    /// in units of the residual's `UNIT` (`f16acc::down_mass`).
    pub fn down_mass(&self, steps: &[Step]) -> i64 {
        let units: Vec<Unit> = steps.iter().flat_map(|s| self.step_units(s)).collect();
        f16acc::down_mass(&units, self)
    }

    /// The emitted units of one of a node's steps: a ramp's is the first unit alone.
    fn node_step_units(&self, st: &Step, ramp: bool) -> Vec<Unit> {
        let mut u = self.step_units(st);
        if ramp {
            u.truncate(1);
        }
        u
    }

    /// The node's steps an f16 checkpoint cannot store (T1, `f16_ok`); a ramp has only
    /// the orientation `lin − θ + 1`.
    pub fn node_t1_violations(&self, n: &Node) -> usize {
        n.steps
            .iter()
            .filter(|s| {
                if !n.ramp {
                    return !self.f16_ok(&s.lin, s.theta);
                }
                let hi = self.bounds(&s.lin).1;
                !(s.lin.terms.values().all(|w| w.abs() <= F16_SPAN) && hi - s.theta + 1 <= F16_SPAN && (s.lin.c - s.theta).abs() + 1 <= F16_BIAS)
            })
            .count()
    }

    /// Per step of the node, the largest partial sum of its gate rows (`gate_mass`).
    pub fn node_gate_masses(&self, n: &Node) -> Vec<i64> {
        n.steps.iter().map(|s| self.node_step_units(s, n.ramp).iter().map(|u| f16acc::gate_mass(&u.form, self)).max().unwrap()).collect()
    }

    /// The largest partial sum of the node's down row (`down_mass`).
    pub fn node_down_mass(&self, n: &Node) -> i64 {
        let units: Vec<Unit> = n.steps.iter().flat_map(|s| self.node_step_units(s, n.ramp)).collect();
        f16acc::down_mass(&units, self)
    }

    /// Whether a node fits f16 accumulation (`docs/stock-llama.md` §6): every gate
    /// within `F16_ACC_GATE` and the down row within `F16_ACC_DOWN`.
    pub fn f16_acc_ok(&self, steps: &[Step]) -> bool {
        steps.iter().all(|s| self.gate_mass(&s.lin, s.theta) <= F16_ACC_GATE) && self.down_mass(steps) <= F16_ACC_DOWN
    }

    /// `s = base·H + L` with every term's weight in L, and the constant, in
    /// [1 − base/2, base/2] ([−127, 128] for base 256).
    fn digits(s: &Lin, base: i64) -> (Lin, Lin) {
        let high = |w: i64| (w + base / 2 - 1).div_euclid(base);
        let (mut h, mut l) = (Lin::konst(high(s.c)), Lin::konst(s.c - base * high(s.c)));
        for (&sig, &w) in &s.terms {
            h = h + Lin::from(sig) * high(w);
            l = l + Lin::from(sig) * (w - base * high(w));
        }
        (h, l)
    }

    /// `[lin ≥ θ]` as one wire shared by every step that asks for it.
    fn ge_shared(&mut self, name: &str, lin: Lin, theta: i64) -> Lin {
        let key = (lin.terms.iter().map(|(s, w)| (*s, *w)).collect(), lin.c, theta);
        if let Some(l) = self.shared_ge.get(&key) {
            return l.clone();
        }
        let l = self.ge(name, lin, theta);
        self.shared_ge.insert(key, l.clone());
        l
    }

    /// A step no f16 unit pair holds, one level later as steps that fit. With
    /// `s = base·H + L` (`digits`) and L narrow,
    /// `[s ≥ θ] = [H ≥ h₁] + Σ_h [H = h ∧ L ≥ θ − base·h]` over the few h where L
    /// decides. Each `[H ≥ h]` is a node of its own, split again while H is wide. None
    /// when L itself is wide, and then the builder must narrow that sum.
    fn split_step(&mut self, name: &str, s: &Step, base: i64) -> Option<Vec<Step>> {
        let (h, l) = Self::digits(&s.lin, base);
        let (llo, lhi) = self.bounds(&l);
        if lhi - llo > F16_SPAN || h.terms.is_empty() || (base < 256 && (lhi - llo) / base > 8) {
            return None;
        }
        let h1 = (s.theta - llo + base - 1).div_euclid(base);
        let h0 = (s.theta - lhi + base - 1).div_euclid(base);
        let ge_h = |c: &mut Circuit, v: i64| c.ge_shared(&format!("{name}^H≥{v}"), h.clone(), v);
        let mut steps = vec![Step { coef: s.coef, lin: ge_h(self, h1), theta: 1 }];
        for v in h0..h1 {
            let lin = ge_h(self, v) - ge_h(self, v + 1) + self.ge_shared(&format!("{name}^L≥{}", s.theta - base * v), l.clone(), s.theta - base * v);
            steps.push(Step { coef: s.coef, lin, theta: 2 });
        }
        Some(steps)
    }

    /// `[lin ≥ θ]` as the exclusive steps of its digit split, over wires one level past
    /// the form's, for a consumer that gates them and sums them in its own node.
    pub fn split(&mut self, name: &str, lin: &Lin, theta: i64) -> Option<Vec<Step>> {
        let s = Step { coef: 1, lin: lin.clone(), theta };
        [256, 64, 16, 4].into_iter().find_map(|b| self.split_step(name, &s, b))
    }

    pub fn ge(&mut self, name: &str, lin: Lin, theta: i64) -> Lin {
        self.node(name, vec![Step { coef: 1, lin, theta }])
    }

    pub fn and(&mut self, name: &str, xs: &[Lin]) -> Lin {
        let n = xs.len() as i64;
        let sum = xs.iter().cloned().fold(Lin::konst(0), |a, x| a + x);
        self.ge(name, sum, n)
    }

    pub fn or(&mut self, name: &str, xs: &[Lin]) -> Lin {
        let sum = xs.iter().cloned().fold(Lin::konst(0), |a, x| a + x);
        self.ge(name, sum, 1)
    }

    /// `[lin == v]`, for an integer-valued form.
    pub fn eq(&mut self, name: &str, lin: Lin, v: i64) -> Lin {
        self.node(name, vec![Step { coef: 1, lin: lin.clone(), theta: v }, Step { coef: -1, lin, theta: v + 1 }])
    }

    /// Bit `k` of a non-negative integer form `s`, in one level: a staircase that
    /// toggles at every multiple of 2^k in the form's range. A staircase f16 cannot hold
    /// is taken one byte digit at a time (`staircase`).
    pub fn bit_of(&mut self, name: &str, s: Lin, k: u32) -> Lin {
        let (lo, _) = self.bounds(&s);
        assert!(lo >= 0, "{name}: bit_of needs a non-negative form, range starts at {lo}");
        self.staircase(name, s, k)
    }

    /// A non-negative form congruent to `s` modulo `m`, over the narrower range: every
    /// weight taken to its residue nearest 0, the constant shifted to make the low end
    /// 0. Bits below `log2 m` of a non-negative value depend only on its residue.
    pub fn modulo(&self, s: Lin, m: i64) -> Lin {
        let near = |w: i64| w - m * (w + m / 2 - 1).div_euclid(m);
        let mut r = Lin::konst(0);
        for (&x, &w) in &s.terms {
            r = r + Lin::from(x) * near(w);
        }
        let lo = self.bounds(&r).0 + s.c.rem_euclid(m);
        let r = r + (s.c.rem_euclid(m) - m * lo.div_euclid(m));
        let ((a, b), (c, d)) = (self.bounds(&s), self.bounds(&r));
        if d - c < b - a { r } else { s }
    }

    /// Bit `k` of `s`, whose value is non-negative whatever its declared range. With
    /// `s = 256·H + L` and L narrow, bit k < 8 is bit k of L shifted non-negative by a
    /// multiple of 256, and bit k ≥ 8 is bit k − 8 of `H + ⌊L/256⌋`, the carry made of
    /// `[L ≥ 256j]` wires: one level per byte.
    fn staircase(&mut self, name: &str, s: Lin, k: u32) -> Lin {
        let p = 1i64 << k;
        let s = self.modulo(s, 2 * p);
        let hi = self.bounds(&s).1;
        let steps: Vec<Step> = (1..=hi / p).map(|j| Step { coef: if j % 2 == 1 { 1 } else { -1 }, lin: s.clone(), theta: j * p }).collect();
        if steps.iter().all(|st| self.step_ok(&st.lin, st.theta)) {
            return self.node(name, steps);
        }
        let (h, l) = Self::digits(&s, 256);
        let (llo, lhi) = self.bounds(&l);
        if lhi - llo > F16_SPAN || h.terms.is_empty() {
            return self.node(name, steps);
        }
        if k < 8 {
            let shift = 256 * ((-llo).max(0) + 255).div_euclid(256);
            return self.staircase(name, l + shift, k);
        }
        // Reduction may narrow every step in place (an increment's carries are ANDs),
        // while the byte digits cost a level each. The shallower result wins, and the
        // direct steps win a tie.
        let mark = self.mark();
        let direct = self.node(name, steps.clone());
        let direct_level = direct.max_level(&self.levels);
        if direct_level <= 1 + s.max_level(&self.levels) {
            return direct;
        }
        self.rollback(mark);
        let mut carry = Lin::konst(llo.div_euclid(256));
        for j in llo.div_euclid(256) + 1..=lhi.div_euclid(256) {
            carry = carry + self.ge_shared(&format!("{name}^c{j}"), l.clone(), 256 * j);
        }
        let split = self.staircase(name, h + carry, k - 8);
        if split.max_level(&self.levels) < direct_level {
            return split;
        }
        self.rollback(mark);
        self.node(name, steps)
    }

    /// The low `n` bits of a sum given by its prefix forms: `low(k)` is the sum of the
    /// operands' low `k` bits plus carry-in, so bit k reads only `low(k + 1)`.
    pub fn sum_bits(&mut self, name: &str, n: usize, low: impl Fn(usize) -> Lin) -> Vec<Lin> {
        (0..n).map(|k| self.bit_of(&format!("{name}.{k}"), low(k + 1), k as u32)).collect()
    }

    /// Selects a multi-bit value by mutually exclusive cases in one level. A case fires
    /// when all its gates are 1; at most one case may fire.
    pub fn mux(&mut self, name: &str, n: usize, cases: &[(Vec<Lin>, Vec<Lin>)]) -> Vec<Lin> {
        // A wide gate would be repeated in every bit's step, and an opcode change would
        // reach all of them. With a level of slack, one shared wire [gate ≥ 1] carries
        // it instead (gates are 0/1 by contract), and it changes only when it flips.
        let level = 1 + cases.iter().flat_map(|(gs, v)| gs.iter().chain(v.iter().take(n))).map(|l| l.max_level(&self.levels)).max().unwrap_or(0);
        let cases: Vec<(Vec<Lin>, Vec<Lin>)> = cases
            .iter()
            .map(|(gs, v)| {
                let gs = gs.iter().map(|g| if g.terms.len() > 3 && g.max_level(&self.levels) + 1 < level { self.selector(g) } else { g.clone() }).collect();
                (gs, v.clone())
            })
            .collect();
        (0..n)
            .map(|k| {
                let steps = cases
                    .iter()
                    .map(|(gates, v)| {
                        let lin = gates.iter().cloned().fold(v[k].clone(), |a, g| a + g);
                        Step { coef: 1, lin, theta: gates.len() as i64 + 1 }
                    })
                    .collect();
                self.node(&format!("{name}.{k}"), steps)
            })
            .collect()
    }

    /// A 0/1 form as a single wire, shared between identical forms.
    pub fn selector(&mut self, g: &Lin) -> Lin {
        let key = (g.terms.iter().map(|(s, w)| (*s, *w)).collect(), g.c);
        if let Some(l) = self.selectors.get(&key) {
            return l.clone();
        }
        let l = self.ge("sel", g.clone(), 1);
        self.selectors.insert(key, l.clone());
        l
    }

    /// Little-endian sum bits of `a + b + cin` over `n` bits, one level, plus
    /// carry-out and carry out of bit 3. Each bit k reads only the low k+1 bits.
    pub fn add(&mut self, name: &str, a: &[Lin], b: &[Lin], cin: Lin, n: usize) -> (Vec<Lin>, Lin, Lin) {
        let low = |k: usize| value(a, k) + value(b, k) + cin.clone();
        let sum = self.sum_bits(name, n, low);
        let carry = self.ge(&format!("{name}.c"), low(n), 1 << n);
        let half = self.ge(&format!("{name}.h"), low(4), 16);
        (sum, carry, half)
    }

    /// Asserts every one-hot group's contract: at most one of its wires is 1.
    fn assert_one_hot(&self, v: &[i64]) {
        let mut on = vec![0u32; self.next_group as usize];
        for (w, &g) in self.groups.iter().enumerate() {
            if g != NO_GROUP && v[w] != 0 {
                on[g as usize] += 1;
                assert!(on[g as usize] == 1 && v[w] == 1, "one-hot group {g}: {} = {} beside another", self.names[w], v[w]);
            }
        }
    }

    /// Evaluates a circuit with no memory heads.
    pub fn eval(&self, input: &BTreeMap<String, u8>) -> BTreeMap<String, u8> {
        assert!(self.heads.is_empty(), "this circuit reads memory; use eval_mem");
        self.eval_mem(input, &|_| [0; LINE])
    }

    /// Evaluates the circuit, serving each memory head's line from `mem` (by line
    /// index), at an input position (every position wire 0).
    pub fn eval_mem(&self, input: &BTreeMap<String, u8>, mem: &dyn Fn(u32) -> [u8; LINE]) -> BTreeMap<String, u8> {
        self.eval_at(input, mem, None)
    }

    /// `eval_mem` at output position `at`, where position wire `at` is 1.
    pub fn eval_at(&self, input: &BTreeMap<String, u8>, mem: &dyn Fn(u32) -> [u8; LINE], at: Option<usize>) -> BTreeMap<String, u8> {
        let mut v = vec![0i64; self.levels.len()];
        if let Some(k) = at.and_then(|k| self.positions.get(k)) {
            v[k.0 as usize] = 1;
        }
        for (name, bits) in self.inputs.iter().map(|(n, b)| (n, b)).chain(self.held.iter().map(|(_, n, b)| (n, b))) {
            let byte = input.get(name).copied().unwrap_or(0);
            for (i, s) in bits.iter().enumerate() {
                v[s.0 as usize] = ((byte >> i) & 1) as i64;
            }
        }
        for (idx, hot) in &self.onehots {
            let byte = input.get(&self.inputs[*idx].0).copied().unwrap_or(0);
            v[hot[byte as usize].0 as usize] = 1;
        }
        for &(idx, h, s) in &self.values {
            v[s.0 as usize] = (input.get(&self.inputs[idx].0).copied().unwrap_or(0) >> (4 * h) & 15) as i64;
        }
        let mut heads = self.heads.iter().peekable();
        let mut serve = |v: &mut Vec<i64>, upto: usize| {
            while let Some(h) = heads.next_if(|h| h.after <= upto) {
                let line = mem(self.query_address(&h.query, v) >> LINE_BITS);
                // A bus wire sums its drivers; every other head wire has one, and starts at 0.
                for (j, s) in h.out.iter().enumerate() {
                    v[s.0 as usize] += ((line[j / 8] >> (j % 8)) & 1) as i64;
                }
            }
        };
        for (ni, n) in self.nodes.iter().enumerate() {
            serve(&mut v, ni);
            let x = n.value(&v);
            let (lo, hi) = self.ranges[n.out.0 as usize];
            assert!(lo <= x && x <= hi, "{} = {} outside [{lo}, {hi}]", n.name, x);
            v[n.out.0 as usize] = x;
        }
        serve(&mut v, self.nodes.len());
        self.assert_one_hot(&v);
        self.outputs
            .iter()
            .enumerate()
            .map(|(j, (name, bits))| {
                if let Some(nibbles) = &self.value_outputs[j] {
                    let b = nibbles.iter().enumerate().fold(0u8, |acc, (h, l)| {
                        let n = l.eval(&v);
                        assert!((0..=15).contains(&n), "{name} nibble {h} = {n}");
                        acc | (n as u8) << (4 * h)
                    });
                    return (name.clone(), b);
                }
                let byte = bits.iter().enumerate().fold(0u8, |acc, (i, l)| {
                    let b = l.eval(&v);
                    assert!(b == 0 || b == 1, "{name} bit {i} = {b}");
                    acc | (b as u8) << i
                });
                (name.clone(), byte)
            })
            .collect()
    }

    /// Every line each head can name, over every value of the inputs `input` leaves open
    /// (`None`), with the memory bytes `mem` gives fixed and every other byte open. An
    /// interval evaluation: each wire's range under the open inputs, intersected with its
    /// declared range; a step whose form's range lies wholly on one side of θ is known. A
    /// head names the lines whose address bits match each bit its query fixes, so the
    /// result, per head, is (fixed bits, their values) over the address bits; a head
    /// whose line bits are all fixed reads `mem`'s bytes of that line. Position wires are
    /// left open.
    pub fn reach(&self, input: &dyn Fn(&str) -> Option<u8>, mem: &dyn Fn(u32) -> [Option<u8>; LINE]) -> Vec<(u32, u32)> {
        let mut r = self.ranges.clone();
        for (name, bits) in self.inputs.iter().map(|(n, b)| (n, b)).chain(self.held.iter().map(|(_, n, b)| (n, b))) {
            if let Some(byte) = input(name) {
                for (i, s) in bits.iter().enumerate() {
                    let b = ((byte >> i) & 1) as i64;
                    r[s.0 as usize] = (b, b);
                }
            }
        }
        for (idx, hot) in &self.onehots {
            if let Some(byte) = input(&self.inputs[*idx].0) {
                for (v, s) in hot.iter().enumerate() {
                    let b = (v == byte as usize) as i64;
                    r[s.0 as usize] = (b, b);
                }
            }
        }
        let mut out = Vec::with_capacity(self.heads.len());
        let mut heads = self.heads.iter().peekable();
        let mut driven = vec![false; self.levels.len()];
        let mut serve = |r: &mut Vec<(i64, i64)>, upto: usize, out: &mut Vec<(u32, u32)>| {
            while let Some(h) = heads.next_if(|h| h.after <= upto) {
                let (mut fixed, mut value) = (0u32, 0u32);
                match &h.query {
                    Query::Const(a) => (fixed, value) = ((1 << ADDR_BITS) - 1, *a),
                    Query::Wires(w) => {
                        for (i, l) in w.iter().enumerate() {
                            // The head's query entry is S·(2·form − 1), so the bit is the
                            // form's sign, `[form ≥ 1]`, whatever the form's value. The
                            // interval of a sum of exclusive terms can pass [−1, 2]
                            // (each term at its own bound), and a sign known at both ends
                            // is known in between.
                            let (lo, hi) = l.bounds_in(r);
                            if (lo >= 1) == (hi >= 1) {
                                fixed |= 1 << i;
                                value |= ((lo >= 1) as u32) << i;
                            }
                        }
                    }
                }
                let line_bits = ((1u32 << ADDR_BITS) - 1) & !((1 << LINE_BITS) - 1);
                let line = if fixed & line_bits == line_bits { mem(value >> LINE_BITS) } else { [None; LINE] };
                for (j, s) in h.out.iter().enumerate() {
                    let b = match line[j / 8] {
                        Some(byte) => {
                            let b = ((byte >> (j % 8)) & 1) as i64;
                            (b, b)
                        }
                        None => (0, 1),
                    };
                    // A bus wire's drivers add, within its declared range.
                    let s = s.0 as usize;
                    r[s] = if driven[s] { ((r[s].0 + b.0).max(self.ranges[s].0), (r[s].1 + b.1).min(self.ranges[s].1)) } else { b };
                    driven[s] = true;
                }
                out.push((fixed, value));
            }
        };
        for (ni, n) in self.nodes.iter().enumerate() {
            serve(&mut r, ni, &mut out);
            let (mut lo, mut hi) = (0i64, 0i64);
            for s in &n.steps {
                let (a, b) = s.lin.bounds_in(&r);
                let (x, y) = if n.ramp {
                    let (p, q) = (s.coef * (a - s.theta + 1).max(0), s.coef * (b - s.theta + 1).max(0));
                    (p.min(q), p.max(q))
                } else if a >= s.theta { (s.coef, s.coef) } else if b < s.theta { (0, 0) } else { (s.coef.min(0), s.coef.max(0)) };
                lo += x;
                hi += y;
            }
            let (dlo, dhi) = self.ranges[n.out.0 as usize];
            let (lo, hi) = (lo.max(dlo), hi.min(dhi));
            assert!(lo <= hi, "{}: no value in its declared range", n.name);
            r[n.out.0 as usize] = (lo, hi);
        }
        serve(&mut r, self.nodes.len(), &mut out);
        out
    }

    pub fn depth(&self) -> u32 {
        self.levels.iter().copied().max().unwrap_or(0)
    }

    /// MLP units of a node: a step is one unit, two when its form can exceed θ (the
    /// emitter's `relu(K(s−θ+1)) − relu(K(s−θ))`).
    pub fn units(&self, n: &Node) -> usize {
        n.steps.iter().map(|s| if n.ramp || self.bounds(&s.lin).1 <= s.theta { 1 } else { 2 }).sum()
    }

    /// MLP units per level, level 1 first.
    pub fn units_per_level(&self) -> Vec<usize> {
        let mut per = vec![0; self.depth() as usize];
        for n in &self.nodes {
            per[self.level(n.out) as usize - 1] += self.units(n);
        }
        per
    }

    /// Removes every node whose wire nothing reads (no node, head query or output), and
    /// then those that only removed nodes read, so no unit is evaluated for a value no
    /// one uses. The wires stay, written by nothing, and take no channel, and the shared
    /// forms naming them go, as in `rollback`: a node built after the prune makes its
    /// own wire instead of reading one nothing writes. Returns the count removed.
    pub fn prune_unread(&mut self) -> usize {
        let start = self.nodes.len();
        let mut gone = vec![false; self.levels.len()];
        loop {
            let mut read = vec![false; self.levels.len()];
            let terms = self.nodes.iter().flat_map(|n| n.steps.iter().flat_map(|s| s.lin.terms.keys())).chain(self.output_lins().flat_map(|l| l.terms.keys()));
            for s in terms.copied().chain(self.head_reads().map(|(s, _)| s)) {
                read[s.0 as usize] = true;
            }
            if self.nodes.iter().all(|n| read[n.out.0 as usize]) {
                break;
            }
            for n in self.nodes.iter().filter(|n| !read[n.out.0 as usize]) {
                gone[n.out.0 as usize] = true;
            }
            // A head evaluates after the nodes declared before it: count only those kept.
            let kept: Vec<usize> = self.nodes.iter().scan(0, |k, n| { *k += read[n.out.0 as usize] as usize; Some(*k) }).collect();
            for h in &mut self.heads {
                h.after = if h.after == 0 { 0 } else { kept[h.after - 1] };
            }
            self.nodes.retain(|n| read[n.out.0 as usize]);
        }
        let stale = |l: &Lin| l.terms.keys().any(|s| gone[s.0 as usize]);
        self.shared_ge.retain(|_, l| !stale(l));
        self.selectors.retain(|_, l| !stale(l));
        self.twins.retain(|&x, t| !gone[x] && !gone[t.0 as usize]);
        start - self.nodes.len()
    }

    /// Per wire, whether it is a line bit of a head that nothing reads (no node, query or
    /// output): it takes no channel and its head's `W_o` leaves it out.
    pub fn unread(&self) -> Vec<bool> {
        let mut read = vec![false; self.levels.len()];
        let terms = self.nodes.iter().flat_map(|n| n.steps.iter().flat_map(|s| s.lin.terms.keys())).chain(self.output_lins().flat_map(|l| l.terms.keys()));
        for s in terms.copied().chain(self.head_reads().map(|(s, _)| s)) {
            read[s.0 as usize] = true;
        }
        let mut unread = vec![false; self.levels.len()];
        for s in self.heads.iter().flat_map(|h| &h.out) {
            unread[s.0 as usize] = !read[s.0 as usize];
        }
        unread
    }

    /// Per wire, the last layer that reads it (a node at level r reads at layer r − 1, a
    /// head's query at its layer, an output at the routing layer, the depth), or its own
    /// level when nothing reads it.
    fn last_reads(&self) -> Vec<u32> {
        let mut last: Vec<u32> = self.levels.clone();
        for nd in &self.nodes {
            let at = self.level(nd.out) - 1;
            for s in nd.steps.iter().flat_map(|s| s.lin.terms.keys()) {
                last[s.0 as usize] = last[s.0 as usize].max(at);
            }
        }
        for (s, layer) in self.head_reads() {
            last[s.0 as usize] = last[s.0 as usize].max(layer);
        }
        for (j, (_, b)) in self.outputs.iter().enumerate() {
            for s in b.iter().chain(self.value_outputs[j].iter().flatten()).flat_map(|l| l.terms.keys()) {
                last[s.0 as usize] = last[s.0 as usize].max(self.out_depth(j));
            }
        }
        last
    }

    /// The depth of part `p` of an overlay: its deepest wire's level, the layer that
    /// routes its outputs (the circuit's depth for a circuit of one part).
    pub fn part_depth(&self, p: usize) -> u32 {
        match self.offsets.len() {
            0 | 1 => self.depth(),
            n => {
                let end = if p + 1 < n { self.offsets[p + 1] as usize } else { self.levels.len() };
                self.levels[self.offsets[p] as usize..end].iter().copied().max().unwrap_or(0)
            }
        }
    }

    /// The part output `j` belongs to (0 for a circuit of one part).
    pub fn out_part(&self, j: usize) -> usize {
        let mut end = 0;
        self.out_parts.iter().position(|&n| {
            end += n;
            j < end
        }).unwrap_or(0)
    }

    /// The layer that routes output `j`: its part's depth.
    pub fn out_depth(&self, j: usize) -> u32 {
        self.part_depth(self.out_part(j))
    }

    /// Per residual boundary b (the input of layer b), the wires live there: written at
    /// level ≤ b and read at layer ≥ b. The widest is the wire channels `channels` needs.
    pub fn live_per_boundary(&self) -> Vec<usize> {
        (0..=self.depth()).map(|b| self.live_at(b).len()).collect()
    }

    /// The wires live at boundary `b`, as `live_per_boundary` counts them.
    pub fn live_at(&self, b: u32) -> Vec<Sig> {
        let (last, unread) = (self.last_reads(), self.unread());
        self.levels.iter().zip(&last).enumerate().filter(|&(w, (&l, &u))| l <= b && b <= u && !self.is_position(w) && !unread[w]).map(|(w, _)| Sig(w as u32)).collect()
    }

    /// A channel plan in which a wire may take the channel of a wire no layer reads any
    /// more. Wire A's channel is free for wire B once A's last reader is at a layer
    /// below B's level (B is visible from boundary `level(B)`, a head's byte from its
    /// layer's MLP on); then some MLP from A's last reading layer up to `level(B) − 1`
    /// subtracts A, so the channel holds 0 when B is written. That clearing unit reads
    /// the channel it writes, which tells it apart, so B must not read A itself. Wires
    /// are placed in level order, each taking the free channel that freed first, so the
    /// channel count is the widest `live_per_boundary`; each clearing goes to the layer
    /// of its window with the fewest units so far, given `units` per level.
    pub fn channels(&self, units: &[usize]) -> Channels {
        let (of, occupants, clears) = self.pooled_channels(units, 1, &|_| 0, &[]);
        Channels { of, occupants: occupants.into_iter().next().unwrap_or_default(), clears }
    }

    /// The part a wire belongs to (0 for a circuit of one part).
    pub fn part_of(&self, w: usize) -> usize {
        self.offsets.iter().rposition(|&o| o as usize <= w).unwrap_or(0)
    }

    /// The overlay's channel plan (`docs/one-model.md` §3): each part plans its wires as
    /// `channels` would on its own, and channel k of every part is one residual
    /// channel. A position computes only its own part's units, so the parts' occupants
    /// never meet. `of` is each wire's channel; `occupants[k]` lists every part's
    /// occupants of channel k, part by part (`Layout::part_occupants` splits them).
    ///
    /// The parts share the input slots, and layer 0 gathers a slot once, so bit i of slot
    /// s sits in channel 8·s + i of every part that holds the slot. A part that does not
    /// hold the slot keeps byte 0 there, the gather writes 0, and the channel is free for
    /// its other wires. Every input bit is pinned, read or not, so a holder that reads none
    /// of a slot's bit still counts the channel as occupied (`Model::compile_with` asserts
    /// that every holder's wire has the gather's channel).
    pub fn overlay_channels(&self, units: &[usize]) -> Channels {
        let parts = self.offsets.len().max(1);
        let mut pins = Vec::new();
        for (slot, bits) in self.inputs.iter().enumerate().map(|(j, (_, b))| (j, b)).chain(self.held.iter().map(|(j, _, b)| (*j, b))) {
            for (i, w) in bits.iter().enumerate() {
                if !self.parts.is_empty() {
                    pins.push((w.0 as usize, (8 * slot + i) as u32));
                }
            }
        }
        let (of, pools, clears) = self.pooled_channels(units, parts, &|w| self.part_of(w), &pins);
        let width = pools.iter().map(Vec::len).max().unwrap_or(0);
        let occupants = (0..width).map(|k| pools.iter().flat_map(|p| p.get(k).cloned().unwrap_or_default()).collect()).collect();
        Channels { of, occupants, clears }
    }

    /// `channels` with wires drawn from `pools` separate channel pools: a wire takes a
    /// free channel of its own pool only. Returns each wire's channel within its pool,
    /// each pool's occupants, and the clearings, placed by the units of every pool.
    /// `pins` gives level-0 wires their channels beforehand. A channel below a pool's
    /// last pin that no wire holds yet is free with nothing to clear, and a wire takes
    /// the lowest such before any channel that needs a clearing unit.
    fn pooled_channels(&self, units: &[usize], pools: usize, pool_of: &dyn Fn(usize) -> usize, pins: &[(usize, u32)]) -> (Vec<u32>, Vec<Vec<Vec<Sig>>>, Vec<(Sig, u32)>) {
        let last = self.last_reads();
        let mut inputs_of: Vec<Vec<Sig>> = vec![Vec::new(); self.levels.len()];
        for nd in &self.nodes {
            let mut ins: Vec<Sig> = nd.steps.iter().flat_map(|s| s.lin.terms.keys().copied()).collect();
            ins.sort_unstable();
            ins.dedup();
            inputs_of[nd.out.0 as usize] = ins;
        }
        let unread = self.unread();
        let mut order: Vec<usize> = (0..self.levels.len()).filter(|&w| !self.is_position(w) && !unread[w]).collect();
        order.sort_by_key(|&w| (self.levels[w], w));
        let mut of = vec![u32::MAX; self.levels.len()];
        let mut pooled: Vec<Vec<Vec<Sig>>> = vec![Vec::new(); pools];
        let mut windows: Vec<(Sig, u32, u32)> = Vec::new();
        for &(w, ch) in pins {
            assert!(self.levels[w] == 0 && !unread[w] && !self.is_position(w), "{}: a pinned wire is a level-0 wire some node reads", self.names[w]);
            let occupants = &mut pooled[pool_of(w)];
            if occupants.len() <= ch as usize {
                occupants.resize(ch as usize + 1, Vec::new());
            }
            assert!(occupants[ch as usize].is_empty(), "{}: channel {ch} is pinned twice in one part", self.names[w]);
            occupants[ch as usize].push(Sig(w as u32));
            of[w] = ch;
        }
        for w in order {
            if of[w] != u32::MAX {
                continue;
            }
            let start = self.levels[w];
            let occupants = &mut pooled[pool_of(w)];
            // The wire a channel must be cleared of, or None for a channel never held.
            let holder = |occ: &Vec<Sig>| occ.last().copied();
            let free = (0..occupants.len()).filter(|&ch| {
                let Some(a) = holder(&occupants[ch]) else { return true };
                // One clearing unit undoes a wire of range 0/1 or −1/0, not a nibble.
                let (lo, hi) = self.ranges[a.0 as usize];
                last[a.0 as usize] < start && inputs_of[w].binary_search(&a).is_err() && hi - lo == 1
            });
            match free.min_by_key(|&ch| (holder(&occupants[ch]).map(|a| last[a.0 as usize]), ch)) {
                Some(ch) => {
                    if let Some(a) = holder(&occupants[ch]) {
                        windows.push((a, last[a.0 as usize], start - 1));
                    }
                    occupants[ch].push(Sig(w as u32));
                    of[w] = ch as u32;
                }
                None => {
                    of[w] = occupants.len() as u32;
                    occupants.push(vec![Sig(w as u32)]);
                }
            }
        }
        // Narrow windows first, each into its least loaded MLP layer (level layer + 1).
        windows.sort_by_key(|&(a, lo, hi)| (hi - lo, lo, a));
        let mut load: Vec<usize> = units.to_vec();
        let clears = windows
            .into_iter()
            .map(|(a, lo, hi)| {
                let k = (lo..=hi).min_by_key(|&k| (load[k as usize], k)).unwrap();
                load[k as usize] += 1;
                (a, k)
            })
            .collect();
        (of, pooled, clears)
    }

    /// Moves nodes between levels to narrow the widest level, which sets the MLP width
    /// of every layer. Depth, heads and every node's steps stay as they are; a node only
    /// changes the layer that computes it, so every value is unchanged. A node may sit
    /// anywhere from one above its inputs to one below its earliest reader (a head's
    /// query reads at the head's layer, an output at the depth), as the residual keeps
    /// every wire. List scheduling, earliest deadline first, under each width cap in
    /// turn; widths are compared rounded up to `quantum` (the checkpoint's unit
    /// rounding), and among the narrowest the placement whose wires are read in the
    /// fewest layers wins, the original placement included.
    pub fn schedule(&mut self, quantum: usize) {
        let depth = self.depth();
        let n = self.nodes.len();
        let mut producer = vec![usize::MAX; self.levels.len()];
        for (i, nd) in self.nodes.iter().enumerate() {
            producer[nd.out.0 as usize] = i;
        }
        // Per node: the nodes it reads, and the highest level among its other inputs.
        let mut reads: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fixed = vec![0u32; n];
        for (i, nd) in self.nodes.iter().enumerate() {
            for s in nd.steps.iter().flat_map(|s| s.lin.terms.keys()) {
                match producer[s.0 as usize] {
                    usize::MAX => fixed[i] = fixed[i].max(self.level(*s)),
                    p => reads[i].push(p),
                }
            }
            reads[i].sort_unstable();
            reads[i].dedup();
        }
        let mut alap = vec![depth; n];
        for h in &self.heads {
            if let Query::Wires(w) = &h.query {
                for s in w[LINE_BITS..].iter().flat_map(|l| l.terms.keys()) {
                    if let Some(a) = alap.get_mut(producer[s.0 as usize]) {
                        *a = (*a).min(h.layer);
                    }
                }
            }
        }
        // Creation order is topological, since a node reads only earlier wires.
        for i in (0..n).rev() {
            for &p in &reads[i] {
                alap[p] = alap[p].min(alap[i] - 1);
            }
        }
        let units: Vec<usize> = self.nodes.iter().map(|nd| self.units(nd)).collect();
        let original: Vec<u32> = self.nodes.iter().map(|nd| self.level(nd.out)).collect();
        // `stay`: a node is placed early only as far as its original level, so nodes move
        // only later, off a full level; otherwise any ready node fills the level.
        let place = |cap: usize, stay: bool| -> (usize, Vec<u32>) {
            let mut level = vec![0u32; n];
            let mut widest = 0;
            for l in 1..=depth {
                let mut ready: Vec<usize> = (0..n).filter(|&i| level[i] == 0 && fixed[i] < l && reads[i].iter().all(|&p| level[p] != 0 && level[p] < l)).collect();
                ready.sort_by_key(|&i| (alap[i], i));
                let mut load = 0;
                for i in ready {
                    if alap[i] == l || load + units[i] <= cap && (!stay || original[i] <= l) {
                        level[i] = l;
                        load += units[i];
                    }
                }
                widest = widest.max(load);
            }
            assert!(level.iter().all(|&l| l != 0), "a node missed its deadline");
            (widest, level)
        };
        // The engine pays per changed wire per layer that reads it, so among placements
        // of the same width, the one whose wires are read in the fewest layers wins.
        let spread = |level: &[u32]| -> usize {
            let mut pairs: Vec<(usize, u32)> = (0..n).flat_map(|i| reads[i].iter().map(move |&p| (p, level[i]))).collect();
            pairs.sort_unstable();
            pairs.dedup();
            pairs.len()
        };
        let current = self.units_per_level().into_iter().max().unwrap_or(0);
        let total: usize = units.iter().sum();
        let floor = total / depth.max(1) as usize;
        let mut candidates: Vec<(usize, Vec<u32>)> = vec![(current, original.clone())];
        for cap in (floor..current).step_by((current.saturating_sub(floor) / 256).max(1)) {
            candidates.push(place(cap, false));
            candidates.push(place(cap, true));
        }
        let narrowest = candidates.iter().map(|c| c.0.next_multiple_of(quantum)).min().unwrap();
        let (_, best) = candidates.into_iter().filter(|c| c.0.next_multiple_of(quantum) == narrowest).min_by_key(|c| (spread(&c.1), c.0)).unwrap();
        for (nd, l) in self.nodes.iter().zip(best) {
            self.levels[nd.out.0 as usize] = l;
        }
    }

    /// The (changed wire, gate-reading layer) pairs per pass the placement predicts,
    /// given each wire's `changes` per pass.
    pub fn changed_pairs(&self, changes: &[f64]) -> f64 {
        Search::new(self, changes).pairs()
    }

    /// Moves nodes between levels, keeping depth, heads and steps, toward two goals the
    /// width schedule leaves open. Channels: at most `goal.channels` wires live at any
    /// boundary, so `channels` fits the wires in that many residual channels; the
    /// placement that does so with the narrowest MLP, its clearing units included,
    /// wins. Speed: the engine walks a gate row for every changed wire in every layer
    /// that reads it, so the placement minimises Σ changes(w) · layers reading w, the
    /// (changed wire, reading layer) pairs per pass, after the channels and the width.
    /// When no placement fits the channels, nothing moves. Returns whether
    /// the wires fit the channels.
    pub fn fit(&mut self, quantum: usize, goal: &Fit) -> bool {
        let width = self.units_per_level().into_iter().max().unwrap_or(0).next_multiple_of(quantum);
        let levels = self.depth() as usize + 1;
        let start = self.levels.clone();
        let mut best: Option<(usize, f64, Vec<u32>)> = None;
        if self.live_per_boundary().into_iter().max().unwrap_or(0) > goal.channels {
            for cap in [width, width + quantum, width + 2 * quantum] {
                let mut s = Search::new(self, goal.changes);
                let mut caps = vec![cap; levels];
                let mut last: Option<(usize, f64)> = None;
                // The clearing units land where the channels free up, mostly just past
                // the widest boundary; each round leaves their room in the node caps,
                // aiming a quantum narrower, until a round no longer narrows.
                for _ in 0..6 {
                    s.run(&caps, goal.channels, 2_000_000);
                    if s.live.iter().max().copied().unwrap_or(0) as usize > goal.channels {
                        break;
                    }
                    self.levels.clone_from(&s.level);
                    let nodes = self.units_per_level();
                    let mut units = nodes.clone();
                    for (_, k) in self.channels(&units).clears {
                        units[k as usize] += 1;
                    }
                    self.levels.clone_from(&start);
                    let w = units.iter().copied().max().unwrap_or(0).next_multiple_of(quantum);
                    let pairs = s.pairs();
                    if last.is_some_and(|l| (w, pairs) >= l) {
                        break;
                    }
                    last = Some((w, pairs));
                    if best.as_ref().is_none_or(|b| (w, pairs) < (b.0, b.1)) {
                        best = Some((w, pairs, s.level.clone()));
                    }
                    for l in 1..levels {
                        caps[l] = (w - quantum).saturating_sub(units[l - 1] - nodes[l - 1]);
                    }
                }
            }
        }
        if let Some((_, _, level)) = best {
            self.levels = level;
        }
        self.live_per_boundary().into_iter().max().unwrap_or(0) <= goal.channels
    }

    /// Moves nodes between levels, each move with the nodes it pushes, to cut the
    /// changed (wire, gate-reading layer) pairs per pass: the engine walks one gate
    /// column per changed wire per layer that reads it, so a wire whose readers share a
    /// layer costs one walk. A move takes every reader of one wire in one layer to
    /// another layer that reads it, or one node to another level; the nodes that must
    /// follow (its readers when it moves later, its inputs' nodes when earlier) move
    /// just far enough. A move is kept when the pairs fall, no boundary holds more than
    /// `goal.channels` live wires and no level more than `units` node units. Depth,
    /// heads and steps are unchanged, and so is every value. Deterministic; returns the
    /// pairs before and after.
    pub fn gather(&mut self, goal: &Fit, units: usize) -> (f64, f64) {
        let mut s = Search::new(self, goal.changes);
        // `Search::new` counts every wire; the channel plan counts neither a position
        // wire nor a head's line bit nothing reads.
        let unread = self.unread();
        let counted: Vec<bool> = (0..self.levels.len()).map(|w| !unread[w] && !self.is_position(w)).collect();
        s.live.fill(0);
        for w in (0..counted.len()).filter(|&w| counted[w]) {
            for b in s.level[w]..=s.end(w) {
                s.live[b as usize] += 1;
            }
        }
        assert!(s.live.iter().map(|&x| x as usize).eq(self.live_per_boundary()), "the search's live wires differ from the channel plan's");
        let mut producer = vec![u32::MAX; self.levels.len()];
        for (i, &w) in s.out.iter().enumerate() {
            producer[w] = i as u32;
        }
        let before = s.pairs();
        let n = s.out.len();
        let depth = s.depth as u32;
        let (mut tent, mut stamp, mut epoch) = (vec![0u32; n], vec![0u32; n], 0u32);
        let (mut seen, mut affected): (Vec<u32>, Vec<usize>) = (vec![0; self.levels.len()], Vec::new());
        let mut moves: Vec<(usize, u32)> = Vec::new();
        // Applies `moves` (node, new level) and returns the change in pairs.
        fn apply(s: &mut Search, counted: &[bool], moves: &[(usize, u32)], seen: &mut [u32], epoch: u32, affected: &mut Vec<usize>) -> f64 {
            affected.clear();
            for &(i, _) in moves {
                for k in s.ins_at[i]..s.ins_at[i + 1] {
                    affected.push(s.ins[k as usize] as usize);
                }
                affected.push(s.out[i]);
            }
            affected.retain(|&w| std::mem::replace(&mut seen[w], epoch) != epoch);
            let mut d = 0.0;
            for &w in affected.iter() {
                d -= s.weight[w] * s.read[w].count_ones() as f64;
                if counted[w] {
                    for b in s.level[w]..=s.end(w) {
                        s.live[b as usize] -= 1;
                    }
                }
            }
            for &(i, nl) in moves {
                let l = s.level[s.out[i]];
                for k in s.ins_at[i]..s.ins_at[i + 1] {
                    s.move_reader(s.ins[k as usize] as usize, l as usize - 1, nl as usize - 1);
                }
                s.level[s.out[i]] = nl;
                s.load[l as usize] -= s.units[i];
                s.load[nl as usize] += s.units[i];
            }
            for &w in affected.iter() {
                d += s.weight[w] * s.read[w].count_ones() as f64;
                if counted[w] {
                    for b in s.level[w]..=s.end(w) {
                        s.live[b as usize] += 1;
                    }
                }
            }
            d
        }
        // Plans node `i` at level `l` with what it pushes, into `tent`; false when a
        // pushed node has nowhere to go.
        fn plan(s: &Search, producer: &[u32], i: usize, l: u32, later: bool, tent: &mut [u32], stamp: &mut [u32], epoch: u32, moves: &mut Vec<(usize, u32)>) -> bool {
            let at = |j: usize, tent: &[u32], stamp: &[u32]| if stamp[j] == epoch { tent[j] } else { s.level[s.out[j]] };
            if l < 1 || l > (s.depth as u32).min(s.deadline[s.out[i]]) {
                return false;
            }
            if stamp[i] != epoch {
                moves.push((i, 0));
            }
            (stamp[i], tent[i]) = (epoch, l);
            if later {
                for k in s.readers_at[s.out[i]]..s.readers_at[s.out[i] + 1] {
                    let r = s.readers[k as usize] as usize;
                    if at(r, tent, stamp) <= l && !plan(s, producer, r, l + 1, later, tent, stamp, epoch, moves) {
                        return false;
                    }
                }
            } else {
                for k in s.ins_at[i]..s.ins_at[i + 1] {
                    let w = s.ins[k as usize] as usize;
                    match producer[w] {
                        u32::MAX => {
                            if s.level[w] >= l {
                                return false;
                            }
                        }
                        p => {
                            if at(p as usize, tent, stamp) >= l && !plan(s, producer, p as usize, l - 1, later, tent, stamp, epoch, moves) {
                                return false;
                            }
                        }
                    }
                }
            }
            true
        }
        let limit = i64::try_from(goal.channels).unwrap_or(i64::MAX);
        let mut order: Vec<usize> = (0..self.levels.len()).collect();
        order.sort_by(|&a, &b| s.weight[b].total_cmp(&s.weight[a]).then(a.cmp(&b)));
        loop {
            let mut improved = false;
            // Each candidate: the nodes to move, all to `to`.
            let mut attempt = |s: &mut Search, group: &[usize], to: u32, improved: &mut bool| {
                epoch += 1;
                moves.clear();
                let ok = group.iter().all(|&i| {
                    let l = s.level[s.out[i]];
                    let now = if stamp[i] == epoch { tent[i] } else { l };
                    now == to || plan(s, &producer, i, to, to > l, &mut tent, &mut stamp, epoch, &mut moves)
                });
                if !ok {
                    return;
                }
                for m in moves.iter_mut() {
                    m.1 = tent[m.0];
                }
                moves.retain(|&(i, nl)| s.level[s.out[i]] != nl);
                // A node pushed both ways by two of the group's moves has no consistent level.
                let consistent = moves.iter().all(|&(i, nl)| {
                    let lo = 1 + s.ins(i).iter().map(|&w| match producer[w as usize] { u32::MAX => s.level[w as usize], p if stamp[p as usize] == epoch => tent[p as usize], _ => s.level[w as usize] }).max().unwrap_or(0);
                    lo <= nl
                });
                if !consistent || moves.is_empty() {
                    return;
                }
                let back: Vec<(usize, u32)> = moves.iter().map(|&(i, _)| (i, s.level[s.out[i]])).collect();
                epoch += 1;
                let d = apply(s, &counted, &moves, &mut seen, epoch, &mut affected);
                let fits = s.live.iter().all(|&x| x <= limit) && s.load.iter().all(|&u| u <= units);
                if d < -1e-9 && fits {
                    *improved = true;
                } else {
                    epoch += 1;
                    apply(s, &counted, &back, &mut seen, epoch, &mut affected);
                }
            };
            for &w in &order {
                let layers: Vec<u32> = (0..depth).filter(|&l| s.read[w] >> l & 1 != 0).collect();
                for &a in &layers {
                    for &b in &layers {
                        if a != b && s.read[w] >> a & 1 != 0 {
                            let group: Vec<usize> = s.readers[s.readers_at[w] as usize..s.readers_at[w + 1] as usize].iter().map(|&r| r as usize).filter(|&r| s.level[s.out[r]] == a + 1).collect();
                            attempt(&mut s, &group, b + 1, &mut improved);
                        }
                    }
                }
            }
            for i in 0..n {
                for to in 1..=depth {
                    if to != s.level[s.out[i]] {
                        attempt(&mut s, &[i], to, &mut improved);
                    }
                }
            }
            if !improved {
                break;
            }
        }
        let after = s.pairs();
        self.levels = s.level;
        (before, after)
    }
}

/// What `Circuit::fit` aims at beyond the width.
pub struct Fit<'a> {
    /// Wire channels the layout has room for.
    pub channels: usize,
    /// Per wire, its changes per pass in a recorded run.
    pub changes: &'a [f64],
}

/// The state of `Circuit::fit`'s local search: a level per wire, and per wire the
/// count of its gate readers in each layer (the node readers and, at the depth, the
/// routing layer), from which its live interval and its reading layers follow.
struct Search {
    depth: usize,
    level: Vec<u32>,
    /// Per node: its wire, its input wires (`ins_at`), its units.
    out: Vec<usize>,
    ins: Vec<u32>,
    ins_at: Vec<u32>,
    units: Vec<usize>,
    /// Per wire: the nodes reading it (`readers_at`), the earliest head reading it (the
    /// latest level it may sit at), the latest, the gate readers per layer (row `w` of
    /// `depth + 1` counts) and, bit l, whether layer l has any.
    readers: Vec<u32>,
    readers_at: Vec<u32>,
    deadline: Vec<u32>,
    head_last: Vec<u32>,
    gate: Vec<u16>,
    read: Vec<u32>,
    weight: Vec<f64>,
    live: Vec<i64>,
    load: Vec<usize>,
}

/// A list of lists as one array and its starts.
fn flat(lists: &[Vec<usize>]) -> (Vec<u32>, Vec<u32>) {
    let mut at = vec![0u32];
    for l in lists {
        at.push(at.last().unwrap() + l.len() as u32);
    }
    (lists.iter().flatten().map(|&x| x as u32).collect(), at)
}

impl Search {
    fn new(c: &Circuit, changes: &[f64]) -> Search {
        let depth = c.depth() as usize;
        assert!(depth < 32, "the search keeps a wire's reading layers in 32 bits");
        let nw = c.levels.len();
        let out: Vec<usize> = c.nodes.iter().map(|nd| nd.out.0 as usize).collect();
        let ins: Vec<Vec<usize>> = c
            .nodes
            .iter()
            .map(|nd| {
                let mut v: Vec<usize> = nd.steps.iter().flat_map(|s| s.lin.terms.keys()).map(|s| s.0 as usize).filter(|&w| !c.is_position(w)).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();
        let mut readers = vec![Vec::new(); nw];
        let mut gate = vec![0u16; nw * (depth + 1)];
        for (i, v) in ins.iter().enumerate() {
            for &w in v {
                readers[w].push(i);
                gate[w * (depth + 1) + c.levels[out[i]] as usize - 1] += 1;
            }
        }
        for s in c.output_lins().flat_map(|l| l.terms.keys()) {
            gate[s.0 as usize * (depth + 1) + depth] += 1;
        }
        let read = gate.chunks(depth + 1).map(|g| g.iter().enumerate().filter(|&(_, &n)| n > 0).fold(0u32, |m, (l, _)| m | 1 << l)).collect();
        let (mut deadline, mut head_last) = (vec![depth as u32; nw], vec![0u32; nw]);
        for (s, layer) in c.head_reads() {
            deadline[s.0 as usize] = deadline[s.0 as usize].min(layer);
            head_last[s.0 as usize] = head_last[s.0 as usize].max(layer);
        }
        let units: Vec<usize> = c.nodes.iter().map(|nd| c.units(nd)).collect();
        let mut load = vec![0; depth + 1];
        for (i, &u) in units.iter().enumerate() {
            load[c.levels[out[i]] as usize] += u;
        }
        let weight = (0..nw).map(|w| changes.get(w).copied().unwrap_or(0.0)).collect();
        let ((ins, ins_at), (readers, readers_at)) = (flat(&ins), flat(&readers));
        let mut s = Search { depth, level: c.levels.clone(), out, ins, ins_at, units, readers, readers_at, deadline, head_last, gate, read, weight, live: vec![0; depth + 1], load };
        for w in 0..nw {
            for b in s.level[w]..=s.end(w) {
                s.live[b as usize] += 1;
            }
        }
        s
    }

    fn ins(&self, i: usize) -> &[u32] {
        &self.ins[self.ins_at[i] as usize..self.ins_at[i + 1] as usize]
    }

    /// The last boundary wire `w` is live at.
    fn end(&self, w: usize) -> u32 {
        let g = 31u32.saturating_sub(self.read[w].leading_zeros());
        g.max(self.head_last[w]).max(self.level[w])
    }

    /// Changed (wire, gate-reading layer) pairs per pass.
    fn pairs(&self) -> f64 {
        self.read.iter().zip(&self.weight).map(|(&r, &f)| f * r.count_ones() as f64).sum()
    }

    /// Moves one gate reader of wire `w` from layer `from` to layer `to`.
    fn move_reader(&mut self, w: usize, from: usize, to: usize) {
        let g = &mut self.gate[w * (self.depth + 1)..][..self.depth + 1];
        g[from] -= 1;
        g[to] += 1;
        let r = &mut self.read[w];
        *r = (*r & !(((g[from] == 0) as u32) << from)) | 1 << to;
    }

    /// Hill-climbs single-node moves, deterministically: live wires above `channels`
    /// at a boundary cost most, then node units above their level's entry in `caps`, then the
    /// changed pairs, then the spread of units over levels (room for clearing units).
    fn run(&mut self, caps: &[usize], channels: usize, iters: usize) {
        let limit = i64::try_from(channels).unwrap_or(i64::MAX);
        let over_live = |live: &[i64]| -> f64 { live.iter().map(|&x| (x.saturating_sub(limit).max(0) as f64).powi(2)).sum::<f64>() * 1e6 };
        let over_load = |load: &[usize]| -> f64 { load.iter().zip(caps).map(|(&l, &cap)| l.saturating_sub(cap) as f64 * 1e9 + (l * l) as f64 * 1e-3).sum() };
        let n = self.out.len();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut cost = over_live(&self.live) + over_load(&self.load);
        // The move's live counts, scored before it is kept: one buffer for every move.
        let mut live = self.live.clone();
        for _ in 0..iters {
            let i = (next() % n as u64) as usize;
            let ow = self.out[i];
            let l = self.level[ow];
            let lo = 1 + self.ins(i).iter().map(|&w| self.level[w as usize]).max().unwrap_or(0);
            let readers = &self.readers[self.readers_at[ow] as usize..self.readers_at[ow + 1] as usize];
            let hi = readers.iter().map(|&r| self.level[self.out[r as usize]] - 1).min().unwrap_or(self.depth as u32).min(self.deadline[ow]);
            if lo > hi {
                continue;
            }
            let nl = lo + (next() % (hi - lo + 1) as u64) as u32;
            if nl == l {
                continue;
            }
            // Apply the move, scoring the live and pair changes it makes.
            live.copy_from_slice(&self.live);
            let mut d_pairs = 0.0;
            let own_end = self.end(ow);
            for k in self.ins_at[i]..self.ins_at[i + 1] {
                let w = self.ins[k as usize] as usize;
                let (e0, before) = (self.end(w), self.read[w].count_ones());
                self.move_reader(w, l as usize - 1, nl as usize - 1);
                d_pairs += self.weight[w] * (self.read[w].count_ones() as i32 - before as i32) as f64;
                let e1 = self.end(w);
                for b in e0.min(e1) + 1..=e0.max(e1) {
                    live[b as usize] += if e1 > e0 { 1 } else { -1 };
                }
            }
            self.level[ow] = nl;
            let new_end = self.end(ow);
            for b in l..=own_end {
                live[b as usize] -= 1;
            }
            for b in nl..=new_end {
                live[b as usize] += 1;
            }
            self.load[l as usize] -= self.units[i];
            self.load[nl as usize] += self.units[i];
            let pairs_cost = 1e3 * d_pairs;
            let new = over_live(&live) + over_load(&self.load);
            if new + pairs_cost <= cost {
                cost = new;
                std::mem::swap(&mut self.live, &mut live);
            } else {
                self.load[l as usize] += self.units[i];
                self.load[nl as usize] -= self.units[i];
                self.level[ow] = l;
                for k in self.ins_at[i]..self.ins_at[i + 1] {
                    self.move_reader(self.ins[k as usize] as usize, nl as usize - 1, l as usize - 1);
                }
            }
        }
    }
}

impl Circuit {
    /// The circuits side by side as one: every wire, node and head of each, renumbered
    /// so no two share a wire; slot names prefixed `tag:`; outputs in part order with the
    /// outputs `keep` refuses left out (a part's unused record slots). The parts share
    /// the input slots (`Circuit::parts`): a part's k-th input without nibble wires holds
    /// slot k, and the inputs with nibble wires follow the widest part, each in a slot of
    /// its own, since a gather writes a nibble wire into a channel no clearing unit
    /// empties. At most one part may have position wires, and it must come first so its
    /// position 0 is the first output slot. Returns the circuit and, per part, its wire
    /// offset.
    pub fn concat(parts: &[(&str, &Circuit)], keep: impl Fn(usize, OutKind) -> bool) -> (Circuit, Vec<u32>) {
        let mut c = Circuit::default();
        let mut offsets = Vec::new();
        // An input with nibble wires is the part's alone; the rest share slots by index.
        let own = |p: &Circuit, i: usize| p.values.iter().any(|&(j, _, _)| j == i);
        let mut next_own = parts.iter().map(|(_, p)| (0..p.inputs.len()).filter(|&i| !own(p, i)).count()).max().unwrap_or(0);
        let mut slots: Vec<Option<(String, [Sig; 8])>> = vec![None; next_own + parts.iter().map(|(_, p)| (0..p.inputs.len()).filter(|&i| own(p, i)).count()).sum::<usize>()];
        for (pi, (tag, p)) in parts.iter().enumerate() {
            assert!(p.positions.is_empty() || pi == 0, "{tag}: only the first part may have position wires");
            assert!(p.onehots.is_empty(), "{tag}: a part of several has no one-hot input");
            let off = c.levels.len() as u32;
            let s = |x: Sig| Sig(x.0 + off);
            let lin = |l: &Lin| Lin { terms: l.terms.iter().map(|(&k, &w)| (s(k), w)).collect(), c: l.c };
            c.levels.extend(&p.levels);
            c.ranges.extend(&p.ranges);
            let base = c.next_group;
            c.groups.extend(p.groups.iter().map(|&g| if g == NO_GROUP { g } else { base + g }));
            c.next_group += p.next_group;
            c.names.extend(p.names.iter().map(|n| format!("{tag}:{n}")));
            for h in &p.heads {
                let query = match &h.query {
                    Query::Wires(w) => Query::Wires(w.iter().map(lin).collect()),
                    Query::Const(a) => Query::Const(*a),
                };
                c.heads.push(Head { name: format!("{tag}:{}", h.name), layer: h.layer, query, out: h.out.iter().map(|&x| s(x)).collect(), after: h.after + c.nodes.len() });
            }
            c.nodes.extend(p.nodes.iter().map(|n| Node { out: s(n.out), steps: n.steps.iter().map(|st| Step { coef: st.coef, lin: lin(&st.lin), theta: st.theta }).collect(), konst: n.konst, name: format!("{tag}:{}", n.name), ramp: n.ramp }));
            let mut shared = 0;
            let at: Vec<usize> = (0..p.inputs.len())
                .map(|i| {
                    let slot = if own(p, i) { &mut next_own } else { &mut shared };
                    *slot += 1;
                    *slot - 1
                })
                .collect();
            c.values.extend(p.values.iter().map(|&(i, h, x)| (at[i], h, s(x))));
            for ((n, b), &slot) in p.inputs.iter().zip(&at) {
                let (name, bits) = (format!("{tag}:{n}"), b.map(s));
                match &slots[slot] {
                    None => slots[slot] = Some((name, bits)),
                    Some(_) => c.held.push((slot, name, bits)),
                }
            }
            c.parts.push(at);
            c.offsets.push(off);
            c.positions.extend(p.positions.iter().map(|&x| s(x)));
            offsets.push(off);
        }
        c.inputs = slots.into_iter().map(|s| s.expect("every input slot has a holder")).collect();
        for (pi, (tag, p)) in parts.iter().enumerate() {
            let lin = |l: &Lin| Lin { terms: l.terms.iter().map(|(&k, &w)| (Sig(k.0 + offsets[pi]), w)).collect(), c: l.c };
            c.out_parts.push(p.outputs.iter().zip(&p.out_kinds).filter(|&(_, &kind)| keep(pi, kind)).count());
            for (((n, b), &kind), v) in p.outputs.iter().zip(&p.out_kinds).zip(&p.value_outputs) {
                if keep(pi, kind) {
                    c.outputs.push((format!("{tag}:{n}"), std::array::from_fn(|i| lin(&b[i]))));
                    c.out_kinds.push(kind);
                    c.value_outputs.push(v.as_ref().map(|v| v.each_ref().map(lin)));
                }
            }
        }
        (c, offsets)
    }
}

impl Vars for Circuit {
    fn range(&self, v: usize) -> (i64, i64) {
        self.ranges[v]
    }
    fn group(&self, v: usize) -> u32 {
        self.groups[v]
    }
}

/// The circuit's wire ranges with the wires of `flip` read through their twins.
struct Flipped<'a> {
    c: &'a Circuit,
    flip: &'a BTreeSet<usize>,
}

impl Vars for Flipped<'_> {
    fn range(&self, v: usize) -> (i64, i64) {
        let (a, b) = self.c.ranges[v];
        if self.flip.contains(&v) { (-b, -a) } else { (a, b) }
    }
    fn group(&self, v: usize) -> u32 {
        self.c.groups[v]
    }
}

/// Residual channels for the wires when a wire may take a dead wire's channel
/// (`Circuit::channels`).
pub struct Channels {
    /// Per wire, its channel.
    pub of: Vec<u32>,
    /// Per channel, the wires it holds, in turn.
    pub occupants: Vec<Vec<Sig>>,
    /// Per reuse: the dead wire, and the layer whose MLP subtracts it from its channel.
    pub clears: Vec<(Sig, u32)>,
}

/// Wire names made unique in wire order: the k-th repeat of a name gets `#k`.
pub fn unique_names(names: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    names
        .iter()
        .map(|n| {
            let k = seen.entry(n).or_insert(0);
            *k += 1;
            if *k == 1 { n.clone() } else { format!("{n}#{}", *k - 1) }
        })
        .collect()
}

pub fn byte(bits: [Sig; 8]) -> Byte {
    bits.map(Lin::from)
}

/// Integer value Σ 2^i·bit_i of the low `n` bits.
pub fn value(bits: &[Lin], n: usize) -> Lin {
    bits.iter().take(n).enumerate().fold(Lin::konst(0), |a, (i, b)| a + b.clone() * (1 << i))
}

/// Constant bits of `x`, least significant first.
pub fn bits(x: u64, n: usize) -> Vec<Lin> {
    (0..n).map(|i| Lin::konst(((x >> i) & 1) as i64)).collect()
}

pub fn not(w: &[Lin]) -> Vec<Lin> {
    w.iter().map(|b| Lin::konst(1) - b.clone()).collect()
}

/// `lo` then `hi`, least significant bit first.
pub fn cat(lo: &[Lin], hi: &[Lin]) -> Vec<Lin> {
    lo.iter().chain(hi).cloned().collect()
}

pub fn zero() -> Lin {
    Lin::konst(0)
}

pub fn one() -> Lin {
    Lin::konst(1)
}

/// Prefix form for `a + konst - hb` over the low `k` bits: `-hb` is added as
/// `hb·(2^k - 1)` so the form stays non-negative.
pub fn plus_const(a: &[Lin], konst: u64, hb: &Lin, k: usize) -> Lin {
    value(a, k) + ((konst & ((1 << k) - 1)) as i64) + hb.clone() * ((1i64 << k) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_narrows_the_widest_level_and_keeps_every_value() {
        let mut c = Circuit::default();
        let x = byte(c.input("X"));
        let a = c.and("a", &[x[0].clone(), x[1].clone()]);
        let b = c.and("b", &[a.clone(), x[2].clone()]);
        let deep = c.and("deep", &[b, x[3].clone()]);
        let side: Vec<Lin> = (3..8).map(|k| c.and(&format!("n{k}"), &[a.clone(), x[k].clone()])).collect();
        let mut out = [zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()];
        out[0] = deep;
        out[1..6].clone_from_slice(&side);
        c.output("Y", out);
        let eval = |c: &Circuit| (0..=255u8).map(|v| c.eval(&BTreeMap::from([("X".to_string(), v)]))["Y"]).collect::<Vec<u8>>();
        let before = eval(&c);
        assert_eq!(c.units_per_level(), vec![1, 6, 1]);
        c.schedule(1);
        assert_eq!(c.units_per_level(), vec![1, 3, 4]);
        assert_eq!(c.depth(), 3);
        assert_eq!(eval(&c), before);
    }

    #[test]
    fn prune_unread_drops_the_chains_nothing_reads_and_keeps_every_value() {
        let mut c = Circuit::default();
        let x = byte(c.input("X"));
        let a = c.and("a", &[x[0].clone(), x[1].clone()]);
        let dead = c.and("dead", &[x[6].clone(), x[7].clone()]);
        let _ = c.and("deader", &[dead, a.clone()]);
        let b = c.and("b", &[a, x[2].clone()]);
        let mut out = [zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()];
        out[0] = b;
        c.output("Y", out);
        let eval = |c: &Circuit| (0..=255u8).map(|v| c.eval(&BTreeMap::from([("X".to_string(), v)]))["Y"]).collect::<Vec<u8>>();
        let before = eval(&c);
        assert_eq!(c.units_per_level(), vec![2, 2]);
        // `deader` goes first, and `dead` once nothing reads it; `a` and `b` reach the output.
        assert_eq!(c.prune_unread(), 2);
        assert_eq!(c.nodes.iter().map(|n| n.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(c.units_per_level(), vec![1, 1]);
        assert_eq!(eval(&c), before);
        assert_eq!(c.prune_unread(), 0);
    }

    #[test]
    fn a_node_built_after_a_prune_reads_no_pruned_wire() {
        let mut c = Circuit::default();
        let x = byte(c.input("X"));
        let a = c.and("a", &[x[0].clone(), x[1].clone()]);
        let wire = |l: &Lin| l.terms.keys().next().unwrap().0 as usize;
        let any = x[2].clone() + x[3].clone();
        let both = x[4].clone() + x[5].clone();
        // Three shared forms nothing reads yet: a selector, a shared step, a twin of `a`.
        let _ = c.selector(&any);
        let _ = c.ge_shared("both", both.clone(), 2);
        let _ = c.twin(wire(&a));
        let mut out = [zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()];
        out[0] = a.clone();
        c.output("Y", out);
        assert_eq!(c.prune_unread(), 3);
        // The same three asked for again, and read this time.
        let mut out = [zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()];
        out[0] = c.selector(&any);
        out[1] = c.ge_shared("both", both, 2);
        out[2] = Lin::from(c.twin(wire(&a))) + 1;
        c.output("Z", out);
        assert_eq!(c.nodes.len(), 4);
        for v in 0..=255u8 {
            let want = (v >> 2 & 3 != 0) as u8 | ((v >> 4 & 3 == 3) as u8) << 1 | ((v & 3 == 3) as u8) << 2;
            assert_eq!(c.eval(&BTreeMap::from([("X".to_string(), v)]))["Z"], want, "X = {v:#04x}");
        }
        assert_eq!(c.prune_unread(), 0);
    }

    #[test]
    fn gather_moves_a_reader_with_the_node_it_pushes() {
        let make = || {
            let mut c = Circuit::default();
            let xs = c.input("X");
            let x = byte(xs);
            let s = c.and("s", &[x[0].clone(), x[1].clone()]);
            let t = c.and("t", &[x[2].clone(), x[3].clone()]);
            let u = c.and("u", &[s, x[4].clone()]);
            let w = c.and("w", &[t.clone(), x[4].clone()]);
            let v = c.and("v", &[u, x[5].clone()]);
            let v2 = c.and("v2", &[v, x[7].clone()]);
            // `y` reads X.5 a layer before `v` does; it can read it beside `v` only if `z` moves too.
            let y = c.and("y", &[t, x[5].clone()]);
            let z = c.and("z", &[y, x[6].clone()]);
            let mut out = [zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()];
            (out[0], out[1], out[2]) = (v2, w, z);
            c.output("Y", out);
            (c, xs)
        };
        let (mut c, xs) = make();
        let eval = |c: &Circuit| (0..=255u8).map(|v| c.eval(&BTreeMap::from([("X".to_string(), v)]))["Y"]).collect::<Vec<u8>>();
        let before = eval(&c);
        let level = |c: &Circuit, name: &str| c.level(c.nodes.iter().find(|n| n.name == name).unwrap().out);
        assert_eq!((level(&c, "y"), level(&c, "z"), c.units_per_level()), (2, 3, vec![2, 3, 2, 1]));
        let mut changes = vec![0.0; c.levels.len()];
        changes[xs[5].0 as usize] = 1.0;
        // A level may hold one unit, and the move would leave two in each: nothing moves.
        let mut tight = make().0;
        assert_eq!(tight.gather(&Fit { channels: usize::MAX, changes: &changes }, 1), (2.0, 2.0));
        assert_eq!(tight.levels, c.levels);
        // No room for another live wire: nothing moves.
        let live = c.live_per_boundary().into_iter().max().unwrap();
        let mut tight = make().0;
        tight.gather(&Fit { channels: live - 1, changes: &changes }, usize::MAX);
        assert_eq!(tight.levels, c.levels);
        assert_eq!(c.gather(&Fit { channels: usize::MAX, changes: &changes }, usize::MAX), (2.0, 1.0));
        assert_eq!((level(&c, "y"), level(&c, "z"), c.units_per_level()), (3, 4, vec![2, 2, 2, 2]));
        assert_eq!(c.depth(), 4);
        assert_eq!(eval(&c), before);
    }

    fn all_f16_acc(c: &Circuit) -> bool {
        c.nodes.iter().all(|n| c.f16_acc_ok(&n.steps))
    }

    #[test]
    fn a_wide_step_reduces_to_the_literal_that_decides_it() {
        let mut c = Circuit::default();
        let x = byte(c.input("X"));
        let t = c.ge("t", x[0].clone() * 600 + x[1].clone() + x[2].clone(), 600);
        c.output("T", std::array::from_fn(|i| if i == 0 { t.clone() } else { zero() }));
        assert!(all_f16_acc(&c));
        for v in 0..8u8 {
            assert_eq!(c.eval(&BTreeMap::from([("X".to_string(), v)]))["T"], v & 1, "{v}");
        }
    }

    #[test]
    fn a_16_bit_increment_is_one_level() {
        let mut c = Circuit::default();
        let (lo, hi) = (byte(c.input("L")), byte(c.input("H")));
        let x: Vec<Lin> = lo.iter().chain(&hi).cloned().collect();
        let y = c.sum_bits("x+1", 16, |n| value(&x, n) + 1);
        c.output("YL", std::array::from_fn(|i| y[i].clone()));
        c.output("YH", std::array::from_fn(|i| y[8 + i].clone()));
        assert_eq!(c.depth(), 1);
        assert!(all_f16_acc(&c));
        for v in (0..=0xFFFFu32).step_by(7).chain([0x7FFF, 0xFFFF, 0x00FF]) {
            let out = c.eval(&BTreeMap::from([("L".to_string(), v as u8), ("H".to_string(), (v >> 8) as u8)]));
            assert_eq!(out["YL"] as u32 | (out["YH"] as u32) << 8, (v + 1) & 0xFFFF, "{v:04x}");
        }
    }

    #[test]
    fn a_carry_over_512_values_is_one_level_over_its_operands() {
        let mut c = Circuit::default();
        let (a, b, z) = (byte(c.input("A")), byte(c.input("B")), byte(c.input("Z")));
        // Node wires, so the carry may read twins of them.
        let a: Vec<Lin> = (0..8).map(|k| c.and(&format!("a{k}"), &[a[k].clone()])).collect();
        let b: Vec<Lin> = (0..8).map(|k| c.and(&format!("b{k}"), &[b[k].clone()])).collect();
        let carry = c.ge("carry", value(&a, 8) + value(&b, 8) + z[0].clone() + z[1].clone(), 256);
        c.output("C", std::array::from_fn(|i| if i == 0 { carry.clone() } else { zero() }));
        assert_eq!(c.depth(), 2);
        assert!(all_f16_acc(&c));
        for (x, y, w) in (0..=255u32).flat_map(|x| (0..=255u32).step_by(3).flat_map(move |y| (0..4u32).map(move |w| (x, y, w)))) {
            let out = c.eval(&BTreeMap::from([("A".to_string(), x as u8), ("B".to_string(), y as u8), ("Z".to_string(), w as u8)]));
            assert_eq!(out["C"] as u32, (x + y + (w & 1) + (w >> 1) >= 256) as u32, "{x} {y} {w}");
        }
    }
}
