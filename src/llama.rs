//! Emits a circuit as a Llama checkpoint (RMSNorm, RoPE attention, SwiGLU MLP, no
//! biases) and runs it with a plain dense forward pass that follows HF Llama semantics.
//!
//! Exactness by construction (`docs/stock-llama.md`):
//! - `rms_norm_eps` dominates RMSNorm. The residual's sum of squares in wire units
//!   stays below 16·d (`engine::int::norm_bound`), so every norm divides by exactly
//!   2^7, and the norm weight `GAMMA` turns each wire back into its integer before
//!   every matmul.
//! - Every weight is a small integer times a power of two, so every activation is an
//!   exact integer.
//! - Each step is relu(K(s-θ)+K) - relu(K(s-θ)) with K = 128. Pre-activations are
//!   multiples of K, where silu is exactly relu. A layer-0 gather's targets lead every
//!   other logit by at least 128, and every other head's chosen key leads by at least
//!   `memory::attn::EXP_ZERO` = 104, where f32 exp underflows to exactly 0, so softmax
//!   puts weight 0 on every other token.
//! - RoPE only touches the lowest-frequency pair, whose angle rounds cos to 1.
//!
//! Sequence: one position per input slot, then one per output slot. Layer 0's
//! attention copies every input slot into every position; the circuit's levels are the
//! MLPs; one final routing MLP moves each output position's own field into eight
//! shared channels that the LM head scores against the 256 shared byte tokens.
//! Inputs precede outputs, so a causal mask gives the same answers.
//!
//! A circuit with memory heads (`sm83::build_mem`) compiles with `compile_mem`: every
//! head of the circuit becomes an attention head that reads the KV cache of memory
//! records with the key, value and query pattern of `memory::record`
//! (`docs/memory-in-weights.md` §3), the output slots the records need decode to
//! per-slot token families, and layer 0 gains the three heads that gather a record's
//! address bytes and the copy head that compaction reads through.

use crate::circuit::{Circuit, Lin, OutKind, Query, Sig, LINE, LINE_BITS};
use crate::f16acc::{self, Form, Unit, NO_GROUP};
use crate::memory::attn::{key, Kind, Params, COMMIT, KEY_DIMS};
use crate::sequence::{Positions, Sequence};
use std::sync::LazyLock;

/// A wire of value v holds `v·UNIT` in the residual stream, and the first constant
/// channel holds `UNIT`. A wire in [−1, 1] then sits in [−UNIT, UNIT]. The second
/// constant (64) and nibble wires (up to 15) sit above that, inside `int::norm_bound`.
pub const UNIT: f32 = 1.0 / 128.0;
/// `rms_norm_eps`. It dominates `mean(x²)`, which `int::norm_bound` keeps below
/// 16·UNIT² = 2^-10. Half an ulp of 2^14 is
/// 2^-10 in f32 and 8 in f16, so `mean + EPS` rounds to 2^14 in either reduction and
/// every norm divides by exactly 2^7.
pub const EPS: f32 = 16384.0;
/// The norm weight on every channel: `x / 2^7 · 2^14 = x / UNIT`, so a matmul reads a
/// wire's own integer and the constant as 1.
pub const GAMMA: f32 = 16384.0;
/// The step scale. Pre-activations are multiples of K, where f32 silu is exactly relu
/// (`exp(K)` overflows for K > 88.7, and `exp(−K)` is below 2^-24 for K > 16.7).
pub const K: f32 = 128.0;
/// The constant channels in wire units: 1, and 64 for the bias a single f16 weight
/// cannot hold. A bias `K·n` puts `K·r` on the first and `1024·p` on the second, with
/// `n = 512p + r`, so |n| up to 32,255 fits (|r| ≤ 256, |p| ≤ 63). The second adds
/// `(64·UNIT)² = 1/4` to the sum of squares, inside `int::norm_bound`'s 16·d.
pub const CONSTANTS: [f32; 2] = [1.0, 64.0];
pub const HEAD_DIM: usize = 64;
/// Input slots one layer-0 head gathers at most. The head attends to its group's slots
/// with equal logits, so each weighs exactly 1/len (a power of two); each slot's byte
/// sits in its own lane, so the 64 value dims carry eight bytes.
pub const GROUP: usize = HEAD_DIM / 8;
/// The intermediate size is the widest layer's units rounded up to this.
pub const UNIT_QUANTUM: usize = 64;
/// Hidden is the number of channels the layout uses, rounded up to this. It is one f16
/// MMA tile, which Metal's `mul_mm` needs of an inner dimension. Nothing else needs
/// more, since the norm is `eps`-dominated at every width above 273 channels
/// (`int::norm_bound`: the 64-valued constant adds 4,096 to a sum that must stay below
/// 16 per channel).
pub const HIDDEN_QUANTUM: usize = 64;
/// The gather pair: dims 31 and 63, the lowest RoPE frequency (1e38^(-62/64), so its
/// angle rounds cos to 1 at every position a 24-bit memory can reach).
const ROPE_DIM: usize = HEAD_DIM / 2 - 1;
/// The low-frequency dims (RoPE pairs 10..20, both halves). Pair t turns by
/// 1e38^(-2t/64) ≤ 1.3·10⁻¹² rad per position, so the angle stays below 6·10⁻⁶ rad at
/// every position the layout reaches and a rotated entry rounds to itself in f16.
/// Layer 0's shared key codes its targets on them.
pub const LOW_DIMS: [usize; 22] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52];
/// A memory record's layer-0 gathers, each a group of record slots in lane order:
/// write 0's address with write 1's first byte, write 1's other two bytes, write 0's
/// line. Groups of 4, 2 and 8 tokens at one position, so each weighs exactly 1/len.
const MEMORY_GATHERS: [&[Fam]; 3] = [
    &[Fam::Addr { write: 0, byte: 0 }, Fam::Addr { write: 0, byte: 1 }, Fam::Addr { write: 0, byte: 2 }, Fam::Addr { write: 1, byte: 0 }],
    &[Fam::Addr { write: 1, byte: 1 }, Fam::Addr { write: 1, byte: 2 }],
    &[Fam::Line { byte: 0 }, Fam::Line { byte: 1 }, Fam::Line { byte: 2 }, Fam::Line { byte: 3 }, Fam::Line { byte: 4 }, Fam::Line { byte: 5 }, Fam::Line { byte: 6 }, Fam::Line { byte: 7 }],
];

/// A family token's lane in its record gather, and the channel bit i of its byte lands
/// in: write w's address bit `8·byte + i`, or line bit `8·byte + i`.
fn gather_lane(f: Fam) -> (usize, usize, Option<usize>) {
    match f {
        Fam::Addr { write: 0, byte } => (byte as usize, 8 * byte as usize, Some(0)),
        Fam::Addr { write: _, byte: 0 } => (3, 0, Some(1)),
        Fam::Addr { write: _, byte } => (byte as usize - 1, 8 * byte as usize, Some(1)),
        Fam::Line { byte } => (byte as usize, 8 * byte as usize, None),
    }
}

/// The channel bit `i` of family `f`'s byte lands in after its gather.
fn gathered(l: &Layout, f: Fam, i: usize) -> usize {
    match gather_lane(f) {
        (_, at, Some(w)) => l.addr(w, at + i),
        (_, at, None) => l.line(at + i),
    }
}

/// The sequence position of the slot whose tokens family `f` holds.
fn slot_of_fam(c: &Circuit, f: Fam) -> usize {
    c.inputs.len() + c.out_kinds.iter().position(|&k| Fam::of(k) == Some(f)).unwrap_or_else(|| panic!("a memory circuit has a {f:?} slot"))
}

/// Routing units of the reader tokens: per token and bit, a step per source byte and
/// one (two units) for the kept byte, plus up to four family bits. An upper count
/// (`reader_units` is the exact one).
pub const READER_UNITS: usize = LINE * (8 * (LINE + 2) + 4);

/// The reader tokens' routing units as `compile_with` writes them: `READER_UNITS` with
/// each token's family index at a unit per set bit.
pub fn reader_units() -> usize {
    (0..LINE).map(|k| 8 * (LINE + 2) + (Fam::Line { byte: k as u8 }.index() + 1).count_ones() as usize).sum()
}

/// Input group `g`'s two dims of the shared layer-0 key: the g-th pair of the first 16
/// low dims, so two groups share at most one dim. The record gathers use the last six.
fn gather_code(g: usize) -> [usize; 2] {
    let pairs = (0..16).flat_map(|i| (i + 1..16).map(move |j| [LOW_DIMS[i], LOW_DIMS[j]]));
    pairs.clone().nth(g).unwrap_or_else(|| panic!("input group {g}: the shared key codes {} groups", pairs.count()))
}

/// rope_theta. HF forms `base ** (2t/dim)` in f32, so it must stay below f32::MAX;
/// 1e38 gives the memory heads (`crate::memory::attn`) their recency and address pairs.
pub const ROPE_THETA: f64 = 1e38;

/// HF's `head_dim ** -0.5`, a Python float rounded to the activations' f32.
pub fn attn_scale() -> f32 {
    (1.0 / (HEAD_DIM as f64).sqrt()) as f32
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    /// Key and value heads (HF `num_key_value_heads`). Query head `h` reads KV head
    /// `h / (heads / kv_heads)`, as HF's `repeat_kv` does.
    pub kv_heads: usize,
    pub vocab: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_const: usize,
    /// Per-slot token families (0 without memory).
    pub families: usize,
}

impl Config {
    pub fn seq(&self) -> usize {
        self.n_inputs + self.n_outputs
    }
    pub fn attn(&self) -> usize {
        self.heads * HEAD_DIM
    }
    /// The width of the key and value projections.
    pub fn kv_dim(&self) -> usize {
        self.kv_heads * HEAD_DIM
    }
    /// The KV head query head `h` reads.
    pub fn kv_head(&self, h: usize) -> usize {
        h / (self.heads / self.kv_heads)
    }
    pub fn params(&self) -> usize {
        let (d, a, kv, i) = (self.hidden, self.attn(), self.kv_dim(), self.intermediate);
        2 * self.vocab * d + self.layers * (2 * a * d + 2 * kv * d + 3 * d * i + 2 * d) + d
    }
    /// Bytes a stock runtime's f32 KV cache holds per cached token: a key and a value
    /// of every KV head at every layer.
    pub fn kv_bytes_per_token(&self) -> usize {
        self.layers * 2 * self.kv_dim() * 4
    }
}

/// The per-slot token families of a memory checkpoint, in token order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fam {
    /// Byte `byte` of write `write`'s physical address.
    Addr { write: u8, byte: u8 },
    /// Byte `byte` of the line a record keeps: write 0's line slots, and what a reader
    /// record's `RD` tokens decode to.
    Line { byte: u8 },
}

impl Fam {
    pub const ALL: [Fam; 6 + LINE] = {
        let mut all = [Fam::Line { byte: 0 }; 6 + LINE];
        let mut i = 0;
        while i < 6 {
            all[i] = Fam::Addr { write: (i / 3) as u8, byte: (i % 3) as u8 };
            i += 1;
        }
        while i < 6 + LINE {
            all[i] = Fam::Line { byte: (i - 6) as u8 };
            i += 1;
        }
        all
    };
    pub fn index(self) -> usize {
        Fam::ALL.iter().position(|&f| f == self).unwrap()
    }
    fn of(kind: OutKind) -> Option<Fam> {
        match kind {
            OutKind::Shared | OutKind::Line { write: 1.., .. } => None,
            OutKind::Address { write, byte } => Some(Fam::Addr { write, byte }),
            OutKind::Line { write: 0, byte } => Some(Fam::Line { byte }),
        }
    }
}

/// Token ids. Input slots have their own byte tokens; output slots share one set; a
/// memory checkpoint adds the per-slot families, then `REC`, `CELL` and `RD0..RD7`.
pub mod tok {
    use super::{Config, Fam, LINE};
    pub fn input(_: &Config, slot: usize, v: u8) -> usize {
        slot * 256 + v as usize
    }
    pub fn input_mask(c: &Config, slot: usize) -> usize {
        c.n_inputs * 256 + slot
    }
    pub fn output(c: &Config, v: u8) -> usize {
        c.n_inputs * 257 + v as usize
    }
    pub fn output_mask(c: &Config, slot: usize) -> usize {
        c.n_inputs * 257 + 256 + (slot - c.n_inputs)
    }
    pub fn output_value(c: &Config, t: usize) -> Option<u8> {
        let base = c.n_inputs * 257;
        (base..base + 256).contains(&t).then(|| (t - base) as u8)
    }
    fn families(c: &Config) -> usize {
        c.n_inputs * 257 + 256 + c.n_outputs
    }
    pub fn family(c: &Config, f: Fam, v: u8) -> usize {
        families(c) + f.index() * 256 + v as usize
    }
    /// The family and byte of a per-slot token.
    pub fn family_value(c: &Config, t: usize) -> Option<(Fam, u8)> {
        let base = families(c);
        (base..base + 256 * c.families).contains(&t).then(|| (Fam::ALL[(t - base) / 256], ((t - base) % 256) as u8))
    }
    /// A record's first token: it carries every input slot's identity, so the input
    /// gathers see exact zeros.
    pub fn rec(c: &Config) -> usize {
        families(c) + 256 * c.families
    }
    /// A write record's cell token. It gathers the record's address and line at layer 0
    /// and carries `committed`, so its key and value are the line's cell.
    pub fn cell(c: &Config) -> usize {
        rec(c) + 1
    }
    /// Reader token k. A reader record decodes it to byte k of the destination's line
    /// with the source's byte written over the destination's (a `Line` token).
    pub fn rd(c: &Config, k: usize) -> usize {
        assert!(k < LINE);
        rec(c) + 2 + k
    }
    /// The tokens after the families: `REC`, `CELL`, `RD0..RD7`.
    pub const SPECIAL: usize = 2 + LINE;
    /// The first token of `seq` that no record holds, when `seq` is a record (it holds
    /// `REC`); None for a record of its own tokens and for any other sequence. A record
    /// holds `REC`, family tokens, `CELL` or the reader tokens, and a reader record the
    /// masks of write 0's line slots.
    ///
    /// A record gather attends every token of its window that carries one of its slots'
    /// identities, each with the same weight. An overlay's parts share the output identity
    /// channels, so another part's mask on those identities scores with the family tokens
    /// and would scale the gathered address and line. Only the sequence's composition
    /// keeps such a token out, and this is its guard.
    pub fn stray(c: &Config, slots: &RecordSlots, seq: &[usize]) -> Option<usize> {
        let record = families(c)..rec(c) + SPECIAL;
        if !seq.contains(&rec(c)) {
            return None;
        }
        seq.iter().copied().find(|t| !record.contains(t) && !slots.line[0].iter().any(|&p| output_mask(c, p) == *t))
    }
    /// An overlay's null token, the vocabulary's last, after a memory checkpoint's special
    /// tokens: the cell every memory head attends to outside its part's positions. Its
    /// key is 1 on the null dims and its value 0.
    pub fn null(c: &Config) -> usize {
        rec(c) + if c.families > 0 { SPECIAL } else { 0 }
    }
    /// The byte a decoded token spells, whatever its family.
    pub fn byte(c: &Config, t: usize) -> Option<u8> {
        output_value(c, t).or_else(|| family_value(c, t).map(|(_, v)| v))
    }

    /// `byte` with the token ranges read once: the shared output bytes, then the
    /// per-slot families.
    #[derive(Clone, Copy, Debug)]
    pub struct Bytes {
        output: usize,
        families: usize,
        family_tokens: usize,
    }

    impl Bytes {
        pub fn new(c: &Config) -> Bytes {
            Bytes { output: c.n_inputs * 257, families: families(c), family_tokens: 256 * c.families }
        }

        #[inline]
        pub fn of(&self, t: usize) -> Option<u8> {
            let o = t.wrapping_sub(self.output);
            if o < 256 {
                return Some(o as u8);
            }
            let f = t.wrapping_sub(self.families);
            (f < self.family_tokens).then_some(f as u8)
        }
    }

    /// Where a memory circuit's record slots sit in its sequence: per write, the three
    /// address bytes (`WPwa..c`), then the line's bytes (`WLw0..7`). Found by slot name,
    /// with or without a part's `tag:` prefix. A GBC CPU memory circuit adds the OAM bug's
    /// five rows (`OWta..c`), which share one line (`OL0..7`): records `OAM_RECORDS` on.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct RecordSlots {
        pub addr: [[usize; 3]; 2],
        pub line: [[usize; LINE]; 2],
        #[serde(default)]
        pub oam: Option<([[usize; 3]; 5], [usize; LINE])>,
    }

    /// The record number of the OAM bug's first row (`RecordSlots::oam`).
    pub const OAM_RECORDS: u8 = 2;

    impl RecordSlots {
        pub fn find(slots: &[String]) -> Option<RecordSlots> {
            let at = |name: String| slots.iter().position(|s| s.rsplit(':').next() == Some(name.as_str()));
            let mut r = RecordSlots { addr: [[0; 3]; 2], line: [[0; LINE]; 2], oam: None };
            for w in 0..2 {
                for (b, n) in ["a", "b", "c"].iter().enumerate() {
                    r.addr[w][b] = at(format!("WP{w}{n}"))?;
                }
                for k in 0..LINE {
                    r.line[w][k] = at(format!("WL{w}{k}"))?;
                }
            }
            let rows: Option<Vec<[usize; 3]>> = (0..5).map(|t| Some([at(format!("OW{t}a"))?, at(format!("OW{t}b"))?, at(format!("OW{t}c"))?])).collect();
            let line: Option<Vec<usize>> = (0..LINE).map(|k| at(format!("OL{k}"))).collect();
            r.oam = rows.zip(line).map(|(rows, line)| (rows.try_into().unwrap(), line.try_into().unwrap()));
            Some(r)
        }

        /// Record `w`'s address and line positions: writes 0 and 1, then the OAM bug's rows.
        pub fn of(&self, w: u8) -> ([usize; 3], [usize; LINE]) {
            match w.checked_sub(OAM_RECORDS) {
                None => (self.addr[w as usize], self.line[w as usize]),
                Some(t) => {
                    let (rows, line) = self.oam.expect("an OAM bug record from a checkpoint without the OAM bug's slots");
                    (rows[t as usize], line)
                }
            }
        }
    }

    /// A line a finished group keeps: which of the step's writes, the physical byte
    /// address its address tokens spell, and the line its line tokens spell.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Record {
        pub write: u8,
        pub addr: u32,
        pub line: [u8; LINE],
    }

    impl Record {
        /// The line's index, which is the cell's position and key.
        pub fn line_index(&self) -> u32 {
            self.addr >> crate::memory::layout::LINE_BITS
        }
    }

    /// The records of writes `writes` of a decoded group.
    pub fn records(c: &Config, at: &RecordSlots, tokens: &[usize], writes: &[u8]) -> Vec<Record> {
        let b = |p: usize| byte(c, tokens[p]).unwrap_or_else(|| panic!("position {p} holds token {}, not a byte", tokens[p]));
        writes
            .iter()
            .map(|&w| {
                let (addr, line) = at.of(w);
                let a = addr.map(b);
                Record { write: w, addr: a[0] as u32 | (a[1] as u32) << 8 | (a[2] as u32) << 16, line: line.map(b) }
            })
            .collect()
    }
}

pub struct Layer {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

impl Layer {
    /// `W_k` and `W_v` with one block of rows per query head, each the block of the KV
    /// head it reads: HF's `repeat_kv`, applied to the weights.
    pub fn repeat_kv(&self, cfg: &Config) -> (Vec<f32>, Vec<f32>) {
        let rows = HEAD_DIM * cfg.hidden;
        let expand = |w: &[f32]| (0..cfg.heads).flat_map(|h| w[cfg.kv_head(h) * rows..][..rows].iter().copied()).collect();
        (expand(&self.k), expand(&self.v))
    }
}

/// Which heads read the memory, and with what margins.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemorySpec {
    pub params: Params,
    /// (layer, head) of every memory head: the circuit's heads and the copy heads.
    pub heads: Vec<(usize, usize)>,
    /// Layer-0 heads that gather a record's addresses and line (`MEMORY_GATHERS`).
    pub gathers: [usize; 3],
    /// The heads a reader record reads through: at write 0's gathered address (the
    /// destination) and at write 1's (the source).
    pub copy_heads: [(usize, usize); 2],
    /// Where the record slots sit in the sequence.
    pub slots: tok::RecordSlots,
}

pub struct Model {
    pub cfg: Config,
    pub embed: Vec<f32>,
    pub layers: Vec<Layer>,
    pub lm_head: Vec<f32>,
    pub slots: Vec<String>,
    /// Per layer, the circuit node each MLP unit belongs to (debugging metadata).
    pub unit_names: Vec<Vec<String>>,
    /// Each circuit wire's name, by wire (debugging metadata; `Layout::occupants` says
    /// which channel holds it).
    pub wire_names: Vec<String>,
    /// Each wire's declared value range (`Circuit::ranges`), by wire.
    pub wire_ranges: Vec<(i64, i64)>,
    /// Each wire's level and one-hot group (`Circuit::levels`, `Circuit::groups`), by wire.
    pub wire_levels: Vec<u32>,
    pub wire_groups: Vec<u32>,
    pub memory: Option<MemorySpec>,
    pub layout: Layout,
    /// How the dense pass rounds its matmul inputs, as the runtime it stands in for.
    pub tiles: Tiles,
    pub overlay: Option<Overlay>,
}

/// Residual channel layout: constants, slot identities, the token's own byte and
/// one-hot, the circuit's wires, then (with memory) the gathered address, the kind
/// channels, the reader identity and the copy head's byte, then the route channels.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Layout {
    pub n_const: usize,
    pub seq: usize,
    /// Wire channels.
    pub wires: usize,
    /// Per wire, its wire channel. Wires hold a channel in turn when the circuit's
    /// channel plan (`Circuit::channels`) reuses a dead wire's.
    pub wire_ch: Vec<u32>,
    /// Per wire channel, the wires it holds, in turn.
    pub occupants: Vec<Vec<Sig>>,
    /// Whether tokens carry a 256-wide one-hot of their byte (host mode's opcode slots).
    pub hots: bool,
    pub memory: bool,
    /// The circuit's position wires: (first wire, count, slot of position 0). Each
    /// reads its position's slot-identity channel instead of a wire channel.
    pub positions: (u32, u32, usize),
    /// An overlay's parts by first wire (`Circuit::offsets`; empty without one). Each
    /// part has a mode channel and the overlay one null channel, after the lanes.
    pub offsets: Vec<u32>,
    /// Per slot, its identity channel among the identity channels; empty: one per slot
    /// (`Layout::identities`).
    #[serde(default)]
    pub ids: Vec<u32>,
    /// In a stream's network (`Model::stream`), the stream's own output slots. An
    /// overlay's parts share the output identity channels, so there an identity channel
    /// names one slot only among these. None for a whole checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub own_outputs: Option<std::ops::Range<usize>>,
}

impl Layout {
    /// An overlay's identity channels: one per gather group of input slots, then one per
    /// output slot of a part, which every part's outputs share in turn: a part's k-th
    /// output slot holds the k-th (empty, one per slot, for any other checkpoint). A
    /// layer-0 key reads an input slot's identity only for its group's code, the same
    /// for every slot of the group, so the keys are the per-slot channels' bit for bit.
    /// An output identity is 1 at one slot of each part. Every row that reads one is
    /// silent outside its own part's positions whatever the identities hold there
    /// (`docs/one-model.md` §4; `tests/one_model.rs` reads that off the weights).
    fn identities(c: &Circuit, groups: &[std::ops::Range<usize>]) -> Vec<u32> {
        if c.offsets.len() <= 1 || !c.onehots.is_empty() {
            return Vec::new();
        }
        assert_eq!(c.out_parts.iter().sum::<usize>(), c.outputs.len(), "an overlay names every output slot's part");
        let inputs = groups.iter().enumerate().flat_map(|(g, r)| r.clone().map(move |_| g as u32));
        inputs.chain(c.out_parts.iter().flat_map(|&n| (0..n).map(|k| (groups.len() + k) as u32))).collect()
    }
    fn id_count(&self) -> usize {
        self.ids.iter().max().map_or(self.seq, |&k| k as usize + 1)
    }
    /// One channel per wire, the position wires and unread head bits aside.
    fn plain(c: &Circuit) -> (Vec<u32>, Vec<Vec<Sig>>) {
        let unread = c.unread();
        let wires: Vec<u32> = (0..c.levels.len() as u32).filter(|&w| !c.is_position(w as usize) && !unread[w as usize]).collect();
        let mut of = vec![u32::MAX; c.levels.len()];
        for (ch, &w) in wires.iter().enumerate() {
            of[w as usize] = ch as u32;
        }
        (of, wires.into_iter().map(|s| vec![Sig(s)]).collect())
    }
    fn positions_of(c: &Circuit) -> (u32, u32, usize) {
        c.positions.first().map_or((0, 0, 0), |f| (f.0, c.positions.len() as u32, c.inputs.len()))
    }
    /// The channels before the circuit's: constants, slot identities, the own byte and
    /// one-hot.
    fn front(&self) -> usize {
        self.n_const + self.id_count() + 8 + self.hot_width()
    }
    /// The first wire channel. An overlay puts the wires last, after every shared
    /// channel, so each part's wires end its prefix (`Layout::prefix`).
    pub fn wire_base(&self) -> usize {
        if self.offsets.is_empty() { self.front() } else { self.shared_end() }
    }
    /// Where the memory block, the routes and the lanes start: after the wires, or with
    /// an overlay right after the front.
    fn tail(&self) -> usize {
        if self.offsets.is_empty() { self.after_wires() } else { self.front() }
    }
    /// The end of the shared channels: the lanes, then an overlay's mode and null
    /// channels.
    fn shared_end(&self) -> usize {
        self.route(self.routes()) + 8 * GROUP + self.offsets.len() + usize::from(!self.offsets.is_empty())
    }
    /// The channels part `p` of an overlay uses: the shared ones and its own wires.
    /// Every channel above holds only other parts' wires.
    pub fn prefix(&self, p: usize) -> usize {
        let own = self.occupants.iter().rposition(|occ| occ.iter().any(|&s| self.part_of(s) == p)).map_or(0, |k| k + 1);
        self.wire_base() + own
    }
    pub fn slot_id(&self, j: usize) -> usize {
        self.n_const + self.ids.get(j).map_or(j, |&k| k as usize)
    }
    /// The output slots' identity channels, after the inputs' (one per gather group in an
    /// overlay, one per slot otherwise) and before the own byte.
    pub fn output_ids(&self, n_inputs: usize) -> std::ops::Range<usize> {
        self.slot_id(n_inputs)..self.n_const + self.id_count()
    }
    /// The output slot whose identity is channel `ch`: the one slot that holds it, among
    /// a stream's own outputs when the layout is a stream's (`own_outputs`). None for a
    /// channel no such slot holds, or one that several do (an overlay's whole checkpoint).
    pub fn slot_of_id(&self, ch: usize, n_inputs: usize) -> Option<usize> {
        let mut at = self.own_outputs.clone().unwrap_or(n_inputs..self.seq).filter(|&j| self.slot_id(j) == ch);
        at.next().filter(|_| at.next().is_none())
    }
    pub fn own_bit(&self, i: usize) -> usize {
        self.n_const + self.id_count() + i
    }
    pub fn own_hot(&self, v: usize) -> usize {
        assert!(self.hots);
        self.n_const + self.id_count() + 8 + v
    }
    fn hot_width(&self) -> usize {
        if self.hots { 256 } else { 0 }
    }
    pub fn wire(&self, s: Sig) -> usize {
        let (first, n, slot) = self.positions;
        if (first..first + n).contains(&s.0) {
            return self.slot_id(slot + (s.0 - first) as usize);
        }
        self.wire_opt(s).unwrap_or_else(|| panic!("wire {} has no channel: a head bit nothing reads", s.0))
    }
    /// A wire's channel, or None for a head bit nothing reads.
    pub fn wire_opt(&self, s: Sig) -> Option<usize> {
        let (first, n, slot) = self.positions;
        if (first..first + n).contains(&s.0) {
            return Some(self.slot_id(slot + (s.0 - first) as usize));
        }
        let ch = self.wire_ch[s.0 as usize];
        (ch != u32::MAX).then(|| self.wire_base() + ch as usize)
    }
    pub fn after_wires(&self) -> usize {
        self.wire_base() + self.wires
    }
    /// Bit `i` of a record's gathered 24-bit address of write `w`. Write 0's address is
    /// the cell's key and a reader's destination, and write 1's is a reader's source.
    pub fn addr(&self, w: usize, i: usize) -> usize {
        assert!(w < 2 && i < 24);
        self.tail() + 24 * w + i
    }
    pub fn committed(&self) -> usize {
        self.tail() + 48
    }
    /// Reader token k's identity.
    pub fn rd(&self, k: usize) -> usize {
        assert!(k < LINE);
        self.tail() + 49 + k
    }
    /// Bit `j` of the line a record keeps, gathered from its line tokens. These bits are
    /// every memory head's value (byte `j / 8`'s bit `j % 8`).
    pub fn line(&self, j: usize) -> usize {
        assert!(j < 8 * LINE);
        self.tail() + 49 + LINE + j
    }
    /// Bit `j` of the line a copy head reads: copy head 0 at write 0's gathered address
    /// (the destination), copy head 1 at write 1's (the source).
    pub fn copy(&self, h: usize, j: usize) -> usize {
        assert!(h < 2 && j < 8 * LINE);
        self.tail() + 49 + LINE + 8 * LINE * (1 + h) + j
    }
    fn memory_width() -> usize {
        49 + LINE + 3 * 8 * LINE
    }
    pub fn route(&self, i: usize) -> usize {
        self.tail() + if self.memory { Self::memory_width() } else { 0 } + i
    }
    /// Route channels: eight value bits, plus four family bits with memory.
    pub fn routes(&self) -> usize {
        if self.memory { 12 } else { 8 }
    }
    /// Bit `i` of a token's byte in lane `k`: the copy a layer-0 head gathering a group
    /// of slots reads, one lane per slot of the group.
    pub fn lane(&self, k: usize, i: usize) -> usize {
        assert!(k < GROUP);
        self.route(self.routes()) + 8 * k + i
    }
    /// Part `p`'s mode channel: 1 at the part's output positions (their mask tokens
    /// carry it) and 0 at every other position. Every unit of the part reads it as its
    /// `up` input, so the unit is 0 wherever another part's wires hold the channels.
    pub fn mode(&self, p: usize) -> usize {
        assert!(p < self.offsets.len());
        self.route(self.routes()) + 8 * GROUP + p
    }
    /// The null token's channel. Every memory head of an overlay keys its null target,
    /// the one it attends to outside its part's positions, on this channel.
    pub fn null(&self) -> usize {
        assert!(!self.offsets.is_empty());
        self.route(self.routes()) + 8 * GROUP + self.offsets.len()
    }
    /// The part wire `s` belongs to (0 without an overlay).
    pub fn part_of(&self, s: Sig) -> usize {
        self.offsets.iter().rposition(|&o| o <= s.0).unwrap_or(0)
    }
    fn used(&self) -> usize {
        if self.offsets.is_empty() { self.shared_end() } else { self.after_wires() }
    }
}

/// The part a unit or head belongs to when it belongs to none. Routing, the reader
/// tokens' units and the record gathers serve every part.
pub const NO_PART: u8 = u8::MAX;

/// The head dims an overlay's null target keys on: both halves of RoPE pairs 11 down to
/// 8, below the key pairs (`memory::attn::KEY_PAIR0`), slowest first. The null target
/// needs a margin, not exact logits. With both halves equal, its score is `2·w·k·cos` of
/// the turn between query and key, which rounds to `2·w·k` in f32 at every position the
/// layout reaches on these pairs.
const NULL_DIMS: [usize; 8] = [11, 43, 10, 42, 9, 41, 8, 40];

/// What `Model::census` counts.
pub struct Census {
    pub cfg: Config,
    /// MLP units per layer: nodes, clearing units, routing units, reader units.
    pub units: Vec<usize>,
    /// Heads per layer.
    pub heads: Vec<usize>,
    /// Per part, its clearing units per layer.
    pub clears: Vec<Vec<usize>>,
}

/// What `Model::slice` keeps: the prefix width, and per kept layer the units and heads
/// by the checkpoint's index.
#[derive(Clone, Debug)]
pub struct Slice {
    pub hidden: usize,
    pub units: Vec<Vec<usize>>,
    pub heads: Vec<Vec<usize>>,
    /// Per kept layer and head, whether the head is the part's own (its null term is
    /// folded away); the null dims the overlay uses; the part's mode channel (its own
    /// units' `up` is folded onto the constant).
    pub own_heads: Vec<Vec<bool>>,
    pub null_dims: Vec<usize>,
    pub mode: usize,
    /// The embedding channels the stream's network clears, which none of its rows read
    /// (every mode channel and the null channel).
    pub cleared: Vec<usize>,
}

/// What an overlay checkpoint (`docs/one-model.md` §3) records for `Model::stream`:
/// the part of every MLP unit and attention head by layer (`NO_PART` for the shared
/// ones), and the null term each memory head's query carries per null dim.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Overlay {
    pub units: Vec<Vec<u8>>,
    pub heads: Vec<Vec<u8>>,
    /// Per layer-0 head, the parts whose input slots it gathers, a bit each (0 for a
    /// record gather and for a head that gathers nothing). The parts share the input
    /// slots, so a gather serves every part that holds its group.
    pub holders: Vec<u32>,
    pub null_q: Vec<f32>,
    /// The null token's key on each null dim: 1, or the power of two that keeps the
    /// null term within `NULL_DIMS` when nibble wires (0..15) raise the foreign score.
    pub null_k: f32,
}

/// An upper bound on the residual's sum of squares at hidden `d` (with `d` 0, before
/// the channels past the wires are counted). It counts each wire channel at its
/// occupants' largest square, a nibble route at 15², every other channel at most 1,
/// and the constants.
fn norm_squares(c: &Circuit, layout: &Layout, d: usize) -> i64 {
    let valued = c.value_outputs.iter().any(Option::is_some);
    layout.occupants.iter().map(|occ| occ.iter().map(|s| c.ranges[s.0 as usize].0.abs().max(c.ranges[s.0 as usize].1.abs()).pow(2)).max().unwrap_or(0)).sum::<i64>()
        + d.saturating_sub(layout.occupants.len()) as i64
        + if valued { 2 * 224 } else { 0 }
        + CONSTANTS.iter().map(|&k| (k * k) as i64).sum::<i64>()
}

/// The largest logit, before `attn_scale`, that a memory head's query can give any key
/// when each channel it reads holds any value its occupants' ranges allow (`mag`; a
/// nibble wire of another part reaches 15). Every address bit's query is
/// `S·(2·form − 1)` against a key of ±1, and the committed dim adds `B`.
fn foreign_score(q: &QueryRows, p: &Params, mag: &dyn Fn(usize) -> i64) -> f32 {
    let bits: f32 = match q {
        QueryRows::Bits(b) => b.iter().map(|(t, c)| (2 * c - 1).abs() as f32 + 2.0 * t.iter().map(|&(ch, w)| (w.abs() * mag(ch)) as f32).sum::<f32>()).sum(),
        QueryRows::Const(_) => crate::memory::attn::KEY_BITS as f32,
        QueryRows::Gathered(_) => 3.0 * crate::memory::attn::KEY_BITS as f32,
    };
    p.s * bits + p.b
}

/// One MLP unit pair per step, written into a layer's gate/up/down.
struct Mlp<'a> {
    l: &'a mut Layer,
    d: usize,
    inter: usize,
    next: usize,
    /// The channel the next units' `up` reads (the constant, or an overlay part's
    /// mode), and the part each unit is recorded under.
    up: usize,
    part: u8,
    parts: Vec<u8>,
}

impl Mlp<'_> {
    /// out += coef · [Σ w·channel + c ≥ θ] for integer channels. The step is
    /// relu(K(s-θ+1)) - relu(K(s-θ)); when the form never exceeds θ (`max`) the
    /// second unit is always 0 and is left out.
    fn step(&mut self, terms: &[(usize, i64)], c: i64, theta: i64, max: i64, coef: i64, out: usize) {
        let units: &[(f32, f32)] = if max <= theta { &[(K, 1.0)] } else { &[(K, 1.0), (0.0, -1.0)] };
        for &(off, sign) in units {
            let u = self.next;
            self.next += 1;
            for &(ch, w) in terms {
                self.l.gate[u * self.d + ch] += K * w as f32;
            }
            let n = c - theta + (off / K) as i64;
            // A bias beyond both channels stays whole on the first. That is exact in f32,
            // and `Model::check_f16` names the tensor that f16 cannot hold.
            let p = (n as f64 / 512.0).round() as i64;
            let (p, r) = if n.abs() <= 511 || p.abs() > 63 { (0, n) } else { (p, n - 512 * p) };
            self.l.gate[u * self.d] = K * r as f32;
            self.l.gate[u * self.d + 1] = K * 512.0 * p as f32 / CONSTANTS[1];
            self.l.up[u * self.d + self.up] = 1.0;
            self.parts.push(self.part);
            self.l.down[out * self.inter + u] = sign * coef as f32 * UNIT / K;
        }
    }
}

fn lin_terms(layout: &Layout, l: &Lin) -> Vec<(usize, i64)> {
    l.terms.iter().map(|(s, w)| (layout.wire(*s), *w)).collect()
}

/// A memory head's query over residual channels: per line address bit (byte address
/// bit `LINE_BITS + i`), (channel, weight) terms and a constant; one fixed address; or
/// a record's gathered address of write `w` (the copy heads).
#[derive(Clone, Debug)]
pub enum QueryRows {
    Bits(Vec<(Vec<(usize, i64)>, i64)>),
    Const(u32),
    Gathered(usize),
}

impl QueryRows {
    fn of(layout: &Layout, q: &Query) -> QueryRows {
        match q {
            Query::Const(a) => QueryRows::Const(*a),
            Query::Wires(w) => QueryRows::Bits(w[LINE_BITS..].iter().map(|l| (lin_terms(layout, l), l.c)).collect()),
        }
    }
}

/// The copy heads a memory checkpoint holds past layer 0 (`MemorySpec::copy_heads`).
pub const COPY_HEADS: usize = 2;

/// The layers of the two copy heads. Their queries are the gathered addresses and their
/// lines go to the routing layer, so any circuit layer after 0 serves. It picks the two
/// with the fewest circuit heads, which keeps the widest layer's head count down.
fn copy_layers(c: &Circuit) -> [usize; COPY_HEADS] {
    let depth = record_depth(c).max(2);
    let mut load: Vec<usize> = (0..depth).map(|l| c.heads.iter().filter(|h| h.layer as usize == l).count()).collect();
    std::array::from_fn(|_| {
        let l = (1..depth).min_by_key(|&l| (load[l], l)).unwrap();
        load[l] += 1;
        l
    })
}

/// An upper count of the routing units of output `j`: its value's units
/// (`value_routing_units`), and with memory four family bits, which only a record slot
/// has and rarely all four. The checkpoints of one circuit place their clearing units by
/// this count, so it stays as it is; an overlay counts `routed_units`.
pub fn routing_units(c: &Circuit, j: usize, memory: bool) -> usize {
    value_routing_units(c, j) + if memory { 4 } else { 0 }
}

/// The routing units of output `j` as `compile_with` writes them: its value's units, and
/// for a record slot its family index at a unit per set bit.
pub fn routed_units(c: &Circuit, j: usize) -> usize {
    value_routing_units(c, j) + Fam::of(c.out_kinds[j]).map_or(0, |f| (f.index() + 1).count_ones() as usize)
}

/// The units that route output `j`'s value: one per bit of its byte that is not constant
/// 0 (two when the bit's form can pass 1), and one ramp per nibble of a nibble output.
fn value_routing_units(c: &Circuit, j: usize) -> usize {
    let nibbles = c.value_outputs[j].iter().flatten().filter(|l| *l != &Lin::konst(0)).count();
    c.outputs[j].1.iter().filter(|l| *l != &Lin::konst(0)).map(|l| if c.bounds(l).1 <= 1 { 1 } else { 2 }).sum::<usize>() + nibbles
}

/// The layer that routes the record slots and the reader tokens: the depth of the
/// part that writes records (the circuit's depth for a circuit of one part).
fn record_depth(c: &Circuit) -> usize {
    if c.out_parts.len() <= 1 { c.depth() as usize } else { c.out_depth(slot_of_fam(c, Fam::ALL[0]) - c.inputs.len()) as usize }
}

/// Writes a memory head's weights (`docs/stock-llama.md` §10): the query is the line
/// address the circuit computed (or a constant, or a gathered address), scaled by `S`
/// per bit, plus `B` on the committed dim; the key is write 0's gathered line address as
/// ±1 plus the committed channel; the value is the line's 64 bits, which `W_o` copies
/// into `out` (None for a bit no node reads, whose channel the head leaves alone).
pub fn memory_head(l: &mut Layer, layout: &Layout, cfg: &Config, head: usize, query: &QueryRows, out: &[Option<usize>], p: &Params) {
    assert_eq!(out.len(), 8 * LINE);
    let (d, a) = (cfg.hidden, cfg.attn());
    let base = head * HEAD_DIM;
    let kv = cfg.kv_head(head) * HEAD_DIM;
    let qs = p.s;
    for (i, &dim) in KEY_DIMS.iter().enumerate() {
        let r = (base + dim) * d;
        // q_i = S·(2·bit − 1); the bit is an affine form over channels, or a constant's bit.
        match query {
            QueryRows::Const(addr) => l.q[r] = if addr >> (LINE_BITS + i) & 1 == 1 { qs } else { -qs },
            QueryRows::Bits(bits) => {
                let (terms, c) = &bits[i];
                l.q[r] = qs * (2 * c - 1) as f32;
                for &(ch, wt) in terms {
                    l.q[r + ch] += 2.0 * qs * wt as f32;
                }
            }
            QueryRows::Gathered(w) => {
                l.q[r] = -qs;
                l.q[r + layout.addr(*w, LINE_BITS + i)] = 2.0 * qs;
            }
        }
        let r = (kv + dim) * d;
        l.k[r] = -1.0;
        l.k[r + layout.addr(0, LINE_BITS + i)] = 2.0;
    }
    l.q[(base + COMMIT) * d] = p.b;
    l.k[(kv + COMMIT) * d + layout.committed()] = 1.0;
    for j in 0..8 * LINE {
        l.v[(kv + j) * d + layout.line(j)] = 1.0;
    }
    for (j, ch) in out.iter().enumerate() {
        if let Some(ch) = ch {
            l.o[ch * a + base + j] = UNIT;
        }
    }
}

/// A line's value vector: byte k's bit i in dim `8k + i`.
pub fn value_of(line: &[u8; LINE]) -> [f32; HEAD_DIM] {
    crate::memory::attn::value(line)
}

/// The line a value vector spells.
pub fn line_of_value(v: &[f32; HEAD_DIM]) -> [u8; LINE] {
    crate::memory::attn::line_of(v)
}

/// The embedding (rows of `d` channels) with every mode and null channel cleared,
/// once no row of `layers` reads them. A stream's network after its folds reads
/// neither, and the engine's owner tables track a token whose only channel is its slot
/// identity.
fn unread_modes(layers: &[Layer], embed: &[f32], lay: &Layout, d: usize) -> Vec<f32> {
    let chans: Vec<usize> = (0..lay.offsets.len()).map(|p| lay.mode(p)).chain([lay.null()]).filter(|&ch| ch < d).collect();
    for (li, l) in layers.iter().enumerate() {
        for w in [&l.q, &l.k, &l.v, &l.gate, &l.up] {
            assert!(w.chunks(d).all(|r| chans.iter().all(|&ch| r[ch] == 0.0)), "layer {li}: a row still reads a mode or null channel");
        }
    }
    let mut e = embed.to_vec();
    for row in e.chunks_mut(d) {
        for &ch in &chans {
            row[ch] = 0.0;
        }
    }
    e
}

impl Model {
    /// The host-fed circuit's checkpoint.
    pub fn compile(c: &Circuit) -> Model {
        assert!(c.heads.is_empty(), "a circuit with memory heads compiles with compile_mem");
        Self::compile_with(c, None)
    }

    /// The memory circuit's checkpoint, with the memory heads' margins `p`.
    pub fn compile_mem(c: &Circuit, p: Params) -> Model {
        Self::compile_with(c, Some(p))
    }

    /// The wire channels a checkpoint of `c` has room for at hidden 1,024 · `blocks`.
    pub fn wire_budget(c: &Circuit, memory: bool, blocks: usize) -> usize {
        let l = Layout { n_const: CONSTANTS.len(), seq: c.inputs.len() + c.outputs.len(), wires: 0, wire_ch: Vec::new(), occupants: Vec::new(), hots: !c.onehots.is_empty(), memory, positions: Layout::positions_of(c), offsets: Self::overlay_offsets(c), ids: Layout::identities(c, &Self::groups(c)), own_outputs: None };
        (1024 * blocks).saturating_sub(l.used())
    }

    /// The input slots in groups of GROUP, the remainder in power-of-two groups, so every
    /// gather weight is exact. A group ends where the parts that hold the slots change
    /// (`Circuit::parts`), so each part holds a group whole or not at all: a stream's
    /// pass carries its own inputs and no other slot, and every gather of a group it
    /// holds still weighs each slot 1/len.
    fn groups(c: &Circuit) -> Vec<std::ops::Range<usize>> {
        let holders = Self::slot_holders(c);
        let (mut groups, mut at, n) = (Vec::new(), 0, c.inputs.len());
        while at < n {
            let end = (at..n).find(|&j| holders[j] != holders[at]).unwrap_or(n);
            while at < end {
                let len = GROUP.min(1 << (end - at).ilog2());
                groups.push(at..at + len);
                at += len;
            }
        }
        groups
    }

    /// Per input slot, the parts that hold it, a bit each (0 for a circuit of one part).
    fn slot_holders(c: &Circuit) -> Vec<u32> {
        let mut holders = vec![0u32; c.inputs.len()];
        for (p, slots) in c.parts.iter().enumerate() {
            for &j in slots {
                holders[j] |= 1 << p;
            }
        }
        holders
    }

    /// Layer 0's heads: one gather per group of input slots and per 64 values of a
    /// one-hot block, plus, with memory, the gathers of a record's addresses and line.
    pub fn layer0_heads(c: &Circuit, memory: bool) -> usize {
        Self::groups(c).len() + (256 / HEAD_DIM) * c.onehots.len() + if memory { MEMORY_GATHERS.len() } else { 0 }
    }

    /// An overlay's part offsets, empty unless the circuit has several parts (which
    /// makes it an overlay).
    fn overlay_offsets(c: &Circuit) -> Vec<u32> {
        if c.offsets.len() > 1 { c.offsets.clone() } else { Vec::new() }
    }

    /// The configuration `c` compiles to, from the circuit alone (no weight is written),
    /// with what its widest layer is the maximum of: the MLP units and the heads of each
    /// layer, and each part's clearing units by layer.
    pub fn census(c: &Circuit, memory: bool) -> Census {
        let (_, cfg, _, clears, (units, heads)) = Self::shape_census(c, memory);
        let mut by_part = vec![vec![0; cfg.layers]; c.offsets.len().max(1)];
        for (w, layer) in clears {
            by_part[c.part_of(w.0 as usize)][layer as usize] += 1;
        }
        Census { cfg, units, heads, clears: by_part }
    }

    #[allow(clippy::type_complexity)]
    fn shape_census(c: &Circuit, memory: bool) -> (Layout, Config, Vec<std::ops::Range<usize>>, Vec<(Sig, u32)>, (Vec<usize>, Vec<usize>)) {
        let (n_in, n_out) = (c.inputs.len(), c.outputs.len());
        let seq = n_in + n_out;
        let groups = Self::groups(c);
        let ids = Layout::identities(c, &groups);
        let fit = |(wire_ch, occupants): (Vec<u32>, Vec<Vec<Sig>>)| {
            let layout = Layout { n_const: CONSTANTS.len(), seq, wires: occupants.len(), wire_ch, occupants, hots: !c.onehots.is_empty(), memory, positions: Layout::positions_of(c), offsets: Self::overlay_offsets(c), ids: ids.clone(), own_outputs: None };
            // Nibble wires (0..15) can need more channels than they use for the norm's
            // bound (`norm_squares`). Each channel past the wires adds at most 1 to the
            // squares and 16 to the bound.
            let (s0, w) = (norm_squares(c, &layout, 0), layout.occupants.len() as i64);
            let floor = if s0 - w < 15 * layout.used() as i64 { 0 } else { ((s0 - w) / 15 + 1) as usize };
            let d = layout.used().max(floor).next_multiple_of(HIDDEN_QUANTUM);
            (layout, d)
        };
        // The routing units, each output's at its part's depth (a nibble output's one
        // ramp per nibble), and the reader tokens' at the record part's. An overlay
        // counts them exactly: its layers' widths are the uniform file's cost, and its
        // parts are packed into the room these leave (`one_model::level`).
        let overlay = c.offsets.len() > 1;
        let mut units = c.units_per_level();
        units.push(0);
        for j in 0..c.outputs.len() {
            units[c.out_depth(j) as usize] += if overlay { routed_units(c, j) } else { routing_units(c, j, memory) };
        }
        if memory {
            units[record_depth(c)] += if overlay { reader_units() } else { READER_UNITS };
        }
        // Wires share channels only when that narrows hidden; an overlay's parts always
        // do, each on its own (`Circuit::overlay_channels`). A clearing unit goes to the
        // emptiest layer of its window, counting the routing units a shallower part has
        // there.
        let plan = if overlay { c.overlay_channels(&units) } else { c.channels(&units) };
        let (reuse, reuse_d) = fit((plan.of.clone(), plan.occupants.clone()));
        let (plain, plain_d) = fit(Layout::plain(c));
        let (layout, d, clears) = if overlay || reuse_d < plain_d { (reuse, reuse_d, plan.clears) } else { (plain, plain_d, Vec::new()) };
        for &(_, k) in &clears {
            units[k as usize] += 1;
        }
        let depth = c.depth() as usize;
        let families = if memory { Fam::ALL.len() } else { 0 };

        let inter = units.iter().copied().max().unwrap().next_multiple_of(UNIT_QUANTUM);
        // Heads: layer 0 gathers every input slot (and each one-hot block), plus, with
        // memory, the gathers of a record's addresses and line; later layers hold the
        // circuit's heads, and two of them the copy heads.
        let copies = memory.then(|| copy_layers(c));
        let heads_at = |layer: usize| c.heads.iter().filter(|h| h.layer as usize == layer).count() + copies.iter().flatten().filter(|&&l| l == layer).count();
        assert!(heads_at(0) == 0, "memory heads cannot sit at layer 0: a kept token's key there would lack its address");
        let l0_heads = Self::layer0_heads(c, memory);
        let per_layer: Vec<usize> = std::iter::once(l0_heads).chain((1..=depth).map(heads_at)).collect();
        let heads = per_layer.iter().copied().max().unwrap();
        let vocab = n_in * 257 + 256 + n_out + if memory { 256 * families + tok::SPECIAL } else { 0 } + usize::from(overlay);
        // One key and value head for every query head (HF `num_key_value_heads = 1`)
        // unless an input carries a one-hot, whose gathers need values of their own.
        let kv_heads = if c.onehots.is_empty() { 1 } else { heads };
        assert!(!memory || kv_heads == 1, "a memory checkpoint shares one key and value head: its cells serve every memory head");
        let cfg = Config { hidden: d, intermediate: inter, layers: depth + 1, heads, kv_heads, vocab, n_inputs: n_in, n_outputs: n_out, n_const: layout.n_const, families };
        (layout, cfg, groups, clears, (units, per_layer))
    }

    fn compile_with(c: &Circuit, memory: Option<Params>) -> Model {
        let slots: Vec<String> = c.inputs.iter().map(|x| x.0.clone()).chain(c.outputs.iter().map(|x| x.0.clone())).collect();
        let (layout, cfg, groups, clears, (census_units, _)) = Self::shape_census(c, memory.is_some());
        let (n_in, seq, d, depth, inter) = (cfg.n_inputs, cfg.seq(), cfg.hidden, cfg.layers - 1, cfg.intermediate);
        let overlay = !layout.offsets.is_empty();
        // Per input and output slot, its part (an overlay's; 0 otherwise).
        let part_of_slot: Vec<usize> = std::iter::repeat_n(usize::MAX, n_in).chain(c.out_parts.iter().enumerate().flat_map(|(p, &n)| std::iter::repeat_n(p, n))).collect();
        assert!(!overlay || part_of_slot.len() == seq, "an overlay names every slot's part");
        let hot_inputs: Vec<usize> = c.onehots.iter().map(|(i, _)| *i).collect();
        let (a, kvd) = (cfg.attn(), cfg.kv_dim());

        let mut embed = vec![0f32; cfg.vocab * d];
        let lane_of = |j: usize| j - groups.iter().find(|g| g.contains(&j)).unwrap().start;
        let row = |embed: &mut Vec<f32>, t: usize, slot: Option<usize>, v: Option<u8>, hot: bool| {
            let r = &mut embed[t * d..][..d];
            r[..CONSTANTS.len()].copy_from_slice(&CONSTANTS);
            if let Some(j) = slot {
                r[layout.slot_id(j)] = 1.0;
            }
            if let Some(v) = v {
                for i in 0..8 {
                    r[layout.own_bit(i)] = ((v >> i) & 1) as f32;
                }
                if hot {
                    r[layout.own_hot(v as usize)] = 1.0;
                }
            }
        };
        for j in 0..n_in {
            for v in 0..=255u8 {
                let t = tok::input(&cfg, j, v);
                row(&mut embed, t, Some(j), Some(v), hot_inputs.contains(&j));
                for i in 0..8 {
                    embed[t * d + layout.lane(lane_of(j), i)] = ((v >> i) & 1) as f32;
                }
            }
            row(&mut embed, tok::input_mask(&cfg, j), Some(j), None, false);
        }
        for v in 0..=255u8 {
            row(&mut embed, tok::output(&cfg, v), None, Some(v), false);
        }
        for j in n_in..seq {
            row(&mut embed, tok::output_mask(&cfg, j), Some(j), None, false);
            if overlay {
                embed[tok::output_mask(&cfg, j) * d + layout.mode(part_of_slot[j])] = 1.0;
            }
        }
        if overlay {
            let null = &mut embed[tok::null(&cfg) * d..][..d];
            null[..CONSTANTS.len()].copy_from_slice(&CONSTANTS);
            null[layout.null()] = 1.0;
        }
        if memory.is_some() {
            // A family token carries its slot's identity, so the record gathers find it,
            // and its byte in its lane of that gather.
            for f in Fam::ALL {
                let lane = gather_lane(f).0;
                let slot = slot_of_fam(c, f);
                for v in 0..=255u8 {
                    let t = tok::family(&cfg, f, v);
                    row(&mut embed, t, Some(slot), Some(v), false);
                    for i in 0..8 {
                        embed[t * d + layout.lane(lane, i)] = ((v >> i) & 1) as f32;
                    }
                }
            }
            let rec = &mut embed[tok::rec(&cfg) * d..][..d];
            rec[..CONSTANTS.len()].copy_from_slice(&CONSTANTS);
            for j in 0..n_in {
                rec[layout.slot_id(j)] = 1.0;
            }
            let cell = &mut embed[tok::cell(&cfg) * d..][..d];
            cell[..CONSTANTS.len()].copy_from_slice(&CONSTANTS);
            cell[layout.committed()] = 1.0;
            for k in 0..LINE {
                let rd = &mut embed[tok::rd(&cfg, k) * d..][..d];
                rd[..CONSTANTS.len()].copy_from_slice(&CONSTANTS);
                rd[layout.rd(k)] = 1.0;
            }
        }
        // Every embedding entry above is a value of 1; the residual holds it as UNIT.
        embed.iter_mut().for_each(|e| *e *= UNIT);

        let mut layers: Vec<Layer> = (0..cfg.layers)
            .map(|_| Layer { q: vec![0.0; a * d], k: vec![0.0; kvd * d], v: vec![0.0; kvd * d], o: vec![0.0; d * a], gate: vec![0.0; inter * d], up: vec![0.0; inter * d], down: vec![0.0; d * inter] })
            .collect();

        // Layer 0 gathers the input slots into every position. One head per group of
        // slots, its key every slot of the group and its values each slot's lane; and one
        // per block of 32 values of a one-hot, its key that slot and its values the block.
        let l0 = &mut layers[0];
        let lay = &layout;
        // Per gather: the slots it keys on, and (lane channel, destination channel) per bit.
        let mut copies: Vec<(Vec<usize>, Vec<(usize, usize)>)> = groups
            .iter()
            .map(|g| (g.clone().collect(), g.clone().flat_map(|j| c.inputs[j].1.iter().enumerate().map(move |(i, s)| (lay.lane(j - g.start, i), lay.wire_opt(*s).unwrap_or(usize::MAX)))).collect()))
            .collect();
        // A shared slot's other holders read the channels its first holder's wires have
        // (`Circuit::overlay_channels`), so the one gather serves them all.
        for (j, name, bits) in &c.held {
            for (i, s) in bits.iter().enumerate() {
                assert_eq!(lay.wire_opt(*s), lay.wire_opt(c.inputs[*j].1[i]), "{name}: bit {i} sits in another channel than slot {j}'s gather writes");
            }
        }
        for (j, hot) in &c.onehots {
            for block in hot.chunks(HEAD_DIM).enumerate() {
                copies.push((vec![*j], block.1.iter().enumerate().map(|(i, s)| (layout.own_hot(HEAD_DIM * block.0 + i), layout.wire(*s))).collect()));
            }
        }
        // A record's gathers: its addresses and its line, from the family tokens' lanes
        // (in a step, from the record slots' mask tokens, whose lanes are zero).
        let mut gathers = [0usize; 3];
        if memory.is_some() {
            for (g, fams) in MEMORY_GATHERS.iter().enumerate() {
                gathers[g] = copies.len();
                let slots = fams.iter().map(|&f| slot_of_fam(c, f)).collect();
                let chans = fams.iter().flat_map(|&f| (0..8).map(move |i| (lay.lane(gather_lane(f).0, i), gathered(lay, f, i)))).collect();
                copies.push((slots, chans));
            }
        }
        let shared = cfg.kv_heads == 1;
        // A checkpoint of several parts gives every gather a fallback. Every output
        // slot's mask (its lanes zero) also keys on the gather pair's dim, so a stream's
        // pass that holds none of a gather's targets attends to its own masks and the
        // gather writes 0, as another part's byte-0 slots do. The code dims weigh twice
        // as much to make room: targets score 2·64·32/8 = 512, a mask 64·48/8 = 384 (a
        // record slot's mask, a record gather's target, 896), and a group sharing one
        // code dim 256.
        let fallback = c.parts.len() > 1;
        let q_code = if fallback { 64.0 } else { 32.0 };
        if fallback && shared {
            for j in n_in..seq {
                l0.k[ROPE_DIM * d + layout.slot_id(j)] = 48.0;
            }
        }
        for (h, (keys, chans)) in copies.iter().enumerate() {
            if shared {
                // One key serves every gather. Group h's slots carry its two code dims,
                // and its query names them. Its own slots score 2·32·32/8 = 256, a group
                // sharing one dim 128, every other token 0. The value is the eight
                // lanes, times 4; W_o undoes the 1/len weight and the 4.
                if fallback {
                    l0.q[(h * HEAD_DIM + ROPE_DIM) * d] = 64.0;
                }
                let code = match gathers.iter().position(|&g| memory.is_some() && g == h) {
                    // A record gather keys on dims no input group uses: REC carries every
                    // input group's identity, so it scores on every input group's dims.
                    Some(g) => [LOW_DIMS[16 + 2 * g], LOW_DIMS[17 + 2 * g]],
                    None => gather_code(h),
                };
                for &dim in &code {
                    l0.q[(h * HEAD_DIM + dim) * d] = q_code;
                    for &j in keys {
                        l0.k[dim * d + layout.slot_id(j)] = 32.0;
                    }
                }
                for &(own, dst) in chans {
                    let i = own - layout.lane(0, 0);
                    l0.v[i * d + own] = 4.0;
                    if dst != usize::MAX {
                        l0.o[dst * a + h * HEAD_DIM + i] = UNIT * keys.len() as f32 / 4.0;
                    }
                }
                continue;
            }
            let r = h * HEAD_DIM + ROPE_DIM;
            l0.q[r * d] = 32.0;
            for &j in keys {
                // k = 32, logit = 32·32/sqrt(64) = 128 > 104, so every other key's weight is exactly 0
                l0.k[r * d + layout.slot_id(j)] = 32.0;
            }
            // Each of the group's keys weighs exactly 1/len; the value undoes it.
            for (i, &(own, dst)) in chans.iter().enumerate() {
                let vr = h * HEAD_DIM + i;
                l0.v[vr * d + own] = keys.len() as f32;
                if dst != usize::MAX {
                    l0.o[dst * a + vr] = UNIT;
                }
            }
        }
        // A nibble wire (`Circuit::input_nibbles`) is four of its input's gathered bits
        // weighted 1, 2, 4 and 8 into one channel (`docs/byte-channels.md`).
        for &(j, nib, s) in &c.values {
            let Some(dst) = layout.wire_opt(s) else { continue };
            let h = groups.iter().position(|g| g.contains(&j)).expect("every input slot is gathered");
            let g = &groups[h];
            for i in 4 * nib..4 * nib + 4 {
                let own = layout.lane(j - g.start, i);
                let w = (1 << (i - 4 * nib)) as f32;
                if shared {
                    l0.o[dst * a + h * HEAD_DIM + own - layout.lane(0, 0)] = w * UNIT * g.len() as f32 / 4.0;
                } else {
                    let k = copies[h].1.iter().position(|&(o, _)| o == own).unwrap();
                    l0.o[dst * a + h * HEAD_DIM + k] = w * UNIT;
                }
            }
        }
        let mut next_head = vec![0usize; cfg.layers];
        next_head[0] = copies.len();
        let mut head_parts: Vec<Vec<u8>> = vec![vec![NO_PART; cfg.heads]; cfg.layers];
        let slot_holders = Self::slot_holders(c);
        let holders: Vec<u32> = (0..cfg.heads).map(|h| groups.get(h).map_or(0, |g| slot_holders[g.start])).collect();
        // Outside its part's positions an overlay's memory head attends to the null
        // token. Its query's null term, `N·(1 − mode)`, beats every key its foreign
        // channels can score (`foreign_score`) by `EXP_ZERO` logits, split over null
        // dims in f16-exact pieces of at most 2^15, against a null key of 1 (or the power
        // of two that keeps it within the null dims).
        let mut null_q = Vec::new();
        let mut null_k = 1f32;
        if let (true, Some(p)) = (overlay, memory) {
            let base = layout.wire_base();
            let mag = |ch: usize| ch.checked_sub(base).and_then(|k| layout.occupants.get(k)).map_or(1, |occ| occ.iter().map(|s| { let (lo, hi) = c.ranges[s.0 as usize]; lo.abs().max(hi.abs()) }).max().unwrap_or(1).max(1));
            let worst = c.heads.iter().map(|h| foreign_score(&QueryRows::of(&layout, &h.query), &p, &mag)).fold(0f32, f32::max);
            let n = worst + crate::memory::attn::EXP_ZERO / attn_scale();
            while (n / null_k / 32768.0).ceil() as usize > NULL_DIMS.len() {
                null_k *= 2.0;
            }
            let k = (n / null_k / 32768.0).ceil() as usize;
            null_q = vec![(n / null_k / k as f32 / 1024.0).ceil() * 1024.0; k];
        }
        let null_gate = |l: &mut Layer, head: usize, part: usize| {
            let base = head * HEAD_DIM;
            for (&dim, &w) in NULL_DIMS.iter().zip(&null_q) {
                l.q[(base + dim) * d] = w;
                l.q[(base + dim) * d + layout.mode(part)] = -w;
            }
        };
        // This writes the null cell as the null token's own forward pass makes it, at
        // every layer whose heads read the cells (the copy heads' too, which never ask
        // for it). The null channel keys `null_k` on the null dims and cancels each key
        // dim's −1 (`memory_head`'s `2·bit − 1`), so the key is `null_kv`'s, and no value
        // reads it. A runtime holds the null cell only as it decodes the null token.
        let null_key = |l: &mut Layer, kv: usize| {
            for &dim in &NULL_DIMS[..null_q.len()] {
                l.k[(kv * HEAD_DIM + dim) * d + layout.null()] = null_k;
            }
            for &dim in &KEY_DIMS {
                l.k[(kv * HEAD_DIM + dim) * d + layout.null()] = 1.0;
            }
        };

        let mut spec = None;
        if let Some(p) = memory {
            let mut heads = Vec::new();
            for head in &c.heads {
                let li = head.layer as usize;
                let hi = next_head[li];
                next_head[li] += 1;
                let out: Vec<Option<usize>> = head.out.iter().map(|&s| layout.wire_opt(s)).collect();
                let query = QueryRows::of(&layout, &head.query);
                memory_head(&mut layers[li], &layout, &cfg, hi, &query, &out, &p);
                if overlay {
                    let part = layout.part_of(head.out[0]);
                    null_gate(&mut layers[li], hi, part);
                    head_parts[li][hi] = part as u8;
                }
                heads.push((li, hi));
            }
            // The copy heads: a record's gathered addresses as queries, the lines into the
            // copy channels, which the reader tokens' routing decodes.
            let mut copy_heads = [(0, 0); 2];
            for (w, &cl) in copy_layers(c).iter().enumerate() {
                copy_heads[w] = (cl, next_head[cl]);
                next_head[cl] += 1;
                let out: Vec<Option<usize>> = (0..8 * LINE).map(|j| Some(layout.copy(w, j))).collect();
                // The copy heads are shared and ungated. A reader record's tokens carry
                // no part's mode, and the copy channels are no part's wires (only the
                // record part's units, which its mode gates, and the reader tokens'
                // routing read them).
                memory_head(&mut layers[cl], &layout, &cfg, copy_heads[w].1, &QueryRows::Gathered(w), &out, &p);
                heads.push(copy_heads[w]);
            }
            if overlay {
                for &(li, h) in &heads {
                    null_key(&mut layers[li], cfg.kv_head(h));
                }
            }
            let slots = tok::RecordSlots::find(&slots).expect("a memory circuit has every record slot");
            spec = Some(MemorySpec { params: p, heads, gathers, copy_heads, slots });
        }

        let mut unit_names: Vec<Vec<String>> = vec![Vec::new(); depth + 1];
        let mut mlps: Vec<Mlp> = layers.iter_mut().map(|l| Mlp { l, d, inter, next: 0, up: 0, part: NO_PART, parts: Vec::new() }).collect();
        // An overlay's unit reads its part's mode as `up`.
        let gate = |m: &mut Mlp, w: Sig| {
            if overlay {
                let p = layout.part_of(w);
                (m.up, m.part) = (layout.mode(p), p as u8);
            }
        };
        // Part by part, its nodes and then its clearings, so each part's units in a layer
        // are one contiguous range (an overlay stream's slice of the FFN).
        let n_parts = layout.offsets.len().max(1);
        let part_of = |w: Sig| if overlay { layout.part_of(w) } else { 0 };
        for p in 0..n_parts {
            for n in c.nodes.iter().filter(|n| part_of(n.out) == p) {
                let m = &mut mlps[c.level(n.out) as usize - 1];
                gate(m, n.out);
                for st in &n.steps {
                    let before = m.next;
                    // A ramp is the first unit alone (`Node::ramp`).
                    let max = if n.ramp { st.theta } else { c.bounds(&st.lin).1 };
                    m.step(&lin_terms(&layout, &st.lin), st.lin.c, st.theta, max, st.coef, layout.wire(n.out));
                    let li = c.level(n.out) as usize - 1;
                    unit_names[li].resize(before, String::new());
                    unit_names[li].extend((before..m.next).map(|_| n.name.clone()));
                }
            }
            // On a reused channel, one unit that reads only that channel subtracts its
            // dead wire (0/1 or −1/0), as -[x ≥ 1] or [−x ≥ 1].
            for &(dead, k) in clears.iter().filter(|(dead, _)| part_of(*dead) == p) {
                let (ch, lo) = (layout.wire(dead), c.ranges[dead.0 as usize].0);
                let m = &mut mlps[k as usize];
                gate(m, dead);
                let before = m.next;
                match lo {
                    0 => m.step(&[(ch, 1)], 0, 1, 1, -1, ch),
                    -1 => m.step(&[(ch, -1)], 0, 1, 1, 1, ch),
                    _ => panic!("wire {} lies in [{lo}, {}], which one clearing unit cannot undo", c.names[dead.0 as usize], lo + 1),
                }
                unit_names[k as usize].resize(before, String::new());
                unit_names[k as usize].push("clear".to_string());
            }
        }
        for m in &mut mlps {
            (m.up, m.part) = (0, NO_PART);
        }
        // Folded constants of nodes are carried by the affine forms that read them. With
        // nibble outputs (`docs/byte-channels.md`) every field routes as its two nibbles
        // into routes 0 and 1: a nibble as one ramp gated by its slot, `max(0, v + m·slot
        // − m)` with m the largest value v's terms allow, a bit field as its bits weighted
        // 1, 2, 4 and 8.
        let valued = c.value_outputs.iter().any(Option::is_some);
        let bit_route = |k: usize| if valued { (layout.route(k / 4), 1 << (k % 4)) } else { (layout.route(k), 1) };
        for (j, (_, bits)) in c.outputs.iter().enumerate() {
            // An overlay part routes at its own depth, so its stream can stop there, and
            // only at its own positions, because a folded constant can put another
            // part's field form at 2 over this part's wires.
            let route_at = c.out_depth(j) as usize;
            if overlay {
                let p = c.out_part(j);
                (mlps[route_at].up, mlps[route_at].part) = (layout.mode(p), p as u8);
            }
            let slot = layout.slot_id(n_in + j);
            if let Some(nibbles) = &c.value_outputs[j] {
                for (h, v) in nibbles.iter().enumerate().filter(|(_, v)| **v != Lin::konst(0)) {
                    let m = c.bounds(v).1.max(1);
                    let mut terms = lin_terms(&layout, v);
                    terms.push((slot, m));
                    mlps[route_at].step(&terms, v.c, m + 1, m + 1, 1, layout.route(h));
                }
            }
            for (k, bit) in bits.iter().enumerate() {
                if bit == &Lin::konst(0) {
                    continue;
                }
                let mut terms = lin_terms(&layout, bit);
                terms.push((slot, 1));
                let (route, coef) = bit_route(k);
                mlps[route_at].step(&terms, bit.c, 2, c.bounds(bit).1 + 1, coef, route);
            }
            // A per-slot output also routes its family's index into the family bits.
            if let Some(f) = Fam::of(c.out_kinds[j]) {
                let fam = f.index() + 1;
                for bit in 0..4 {
                    if fam >> bit & 1 == 1 {
                        mlps[route_at].step(&[(layout.slot_id(n_in + j), 1)], 0, 1, 1, 1, layout.route(8 + bit));
                    }
                }
            }
        }
        for m in &mut mlps {
            (m.up, m.part) = (0, NO_PART);
        }
        if memory.is_some() {
            let depth = record_depth(c);
            // Reader token k decodes byte k of the destination's line (copy head 0) with
            // the source's byte (copy head 1, picked by the source's low address bits)
            // written over the byte the destination's low bits name. Each bit has one
            // step per source byte j, [rd_k ∧ dest low = k ∧ src low = j ∧ src bit], and
            // one for the kept byte, [rd_k ∧ dest bit ∧ dest low ≠ k]. At most one fires.
            let low_match = |w: usize, k: usize| -> (Vec<(usize, i64)>, i64) {
                let terms = (0..LINE_BITS).map(|b| (layout.addr(w, b), if k >> b & 1 == 1 { 1 } else { -1 })).collect();
                (terms, (0..LINE_BITS).filter(|b| k >> b & 1 == 0).count() as i64)
            };
            let full = LINE_BITS as i64;
            for k in 0..LINE {
                let (dk, dc) = low_match(0, k);
                for i in 0..8 {
                    for j in 0..LINE {
                        let (sj, sc) = low_match(1, j);
                        let mut terms = vec![(layout.rd(k), 1), (layout.copy(1, 8 * j + i), 1)];
                        terms.extend(&dk);
                        terms.extend(&sj);
                        let (route, coef) = bit_route(i);
                        mlps[depth].step(&terms, dc + sc, 2 + 2 * full, 2 + 2 * full, coef, route);
                    }
                    let mut terms: Vec<(usize, i64)> = vec![(layout.rd(k), 3), (layout.copy(0, 8 * k + i), 3)];
                    terms.extend(dk.iter().map(|&(ch, w)| (ch, -w)));
                    let (route, coef) = bit_route(i);
                    mlps[depth].step(&terms, -dc, 4, 6, coef, route);
                }
                let fam = Fam::Line { byte: k as u8 }.index() + 1;
                for bit in 0..4 {
                    if fam >> bit & 1 == 1 {
                        mlps[depth].step(&[(layout.rd(k), 1)], 0, 1, 1, 1, layout.route(8 + bit));
                    }
                }
            }
        }

        // An overlay's census counts every layer's units exactly, so the shape it gives
        // (`Model::census`, which the placement is packed by) is the compiled one.
        if overlay {
            assert_eq!(mlps.iter().map(|m| m.next).collect::<Vec<_>>(), census_units, "the units written, layer by layer, against the census");
        }
        let unit_parts: Vec<Vec<u8>> = mlps.into_iter().map(|m| m.parts).collect();
        // The head scores a token by how many of the route bits (eight value bits, and the
        // family bits with memory) it matches, so the exact match wins.
        let mut lm_head = vec![0f32; cfg.vocab * d];
        let routes = layout.routes();
        for t in 0..cfg.vocab {
            let r = &mut lm_head[t * d..][..d];
            let pattern = match (tok::output_value(&cfg, t), tok::family_value(&cfg, t)) {
                (Some(v), _) => Some(v as usize),
                (None, Some((f, v))) => Some(v as usize | (f.index() + 1) << 8),
                _ => None,
            };
            match pattern {
                // The nibble head: 2l·L − l² + 2h·H − h² on the two nibble routes, so the
                // token whose nibbles are the routes' scores highest, by at least 1; the
                // squares split over the two constants (1 and 64). Family bits score as
                // the bit head's do.
                Some(pat) if valued => {
                    let (lo, hi) = ((pat & 15) as f32, (pat >> 4 & 15) as f32);
                    r[layout.route(0)] = 2.0 * lo;
                    r[layout.route(1)] = 2.0 * hi;
                    let sq = lo * lo + hi * hi;
                    r[1] = -(sq / 64.0).floor();
                    let mut konst = -(sq % 64.0);
                    for k in 8..routes {
                        let one = (pat >> k) & 1 == 1;
                        r[layout.route(k)] = if one { 1.0 } else { -1.0 };
                        konst += if one { 0.0 } else { 1.0 };
                    }
                    r[0] = konst;
                }
                Some(pat) => {
                    let mut konst = 0.0;
                    for k in 0..routes {
                        let one = (pat >> k) & 1 == 1;
                        r[layout.route(k)] = if one { 1.0 } else { -1.0 };
                        konst += if one { 0.0 } else { 1.0 };
                    }
                    r[0] = konst;
                }
                None => r[0] = -1024.0,
            }
        }
        // Every norm divides by exactly 2^7 while the residual's squares sum below 16·d
        // (`EPS`): each wire channel at its occupants' largest square, a nibble route at
        // 15², every other channel at most 1 (`engine::int::norm_bound` checks the same
        // over the engine's channels).
        let squares = norm_squares(c, &layout, d);
        assert!(squares < 16 * d as i64, "the residual's squares can sum to {squares}, not below 16·{d}");
        Model { cfg, embed, layers, lm_head, slots, unit_names, wire_names: c.names.clone(), wire_ranges: c.ranges.clone(), wire_levels: c.levels.clone(), wire_groups: c.groups.clone(), memory: spec, layout, tiles: Tiles::F32, overlay: overlay.then_some(Overlay { units: unit_parts, heads: head_parts, holders, null_q, null_k }) }
    }

    /// Per layer, upper bounds on `|q|` and `|k|`, every entry of every head after
    /// RoPE, at any position a runtime computes. Metal's flash attention holds the query
    /// in f16, where a larger entry is inf and `inf·0` is a NaN logit. At a position of
    /// part p (any position of p's stream: its slots, the cells' and records' positions,
    /// which hold no other part's wires) a wire channel holds one of p's occupants, at
    /// most its declared range. There p's mode channel is 1 and every other mode is 0,
    /// the null channel is 0 (1 on the null token, which no query reads), a nibble route
    /// is at most 15, and any other channel is at most 1. A record's positions hold any
    /// part's occupants, so the bound also takes every occupant at once. An entry turned
    /// by RoPE is `a·cos − b·sin` of the pair's two rows, at most `|a| + |b|·|sin|`, with
    /// `|sin|` at most the pair's turn over `KEY_REACH`.
    pub fn attention_bounds(&self) -> Vec<(f32, f32)> {
        let (d, lay) = (self.cfg.hidden, &self.layout);
        let base = lay.wire_base();
        let parts = lay.offsets.len();
        let routes: Vec<usize> = (0..lay.routes()).map(|i| lay.route(i)).collect();
        let modes: Vec<usize> = (0..parts).map(|p| lay.mode(p)).collect();
        // Per class (part p, or `parts` for every occupant), each channel's magnitude.
        let mags: Vec<Vec<f32>> = (0..=parts)
            .map(|p| {
                (0..d)
                    .map(|ch| {
                        if ch < CONSTANTS.len() {
                            CONSTANTS[ch].abs()
                        } else if parts > 0 && ch == lay.null() {
                            0.0
                        } else if let Some(q) = modes.iter().position(|&m| m == ch) {
                            if p == parts || q == p { 1.0 } else { 0.0 }
                        } else if let Some(occ) = ch.checked_sub(base).and_then(|k| lay.occupants.get(k)) {
                            occ.iter().filter(|&&s| p == parts || lay.part_of(s) == p).map(|s| { let (lo, hi) = self.wire_ranges[s.0 as usize]; lo.abs().max(hi.abs()) as f32 }).fold(0.0, f32::max)
                        } else if routes.contains(&ch) {
                            15.0
                        } else {
                            1.0
                        }
                    })
                    .collect()
            })
            .collect();
        let half = HEAD_DIM / 2;
        let turn: Vec<f32> = (0..half).map(|t| ((crate::memory::attn::KEY_REACH as f64 / ROPE_THETA.powf(2.0 * t as f64 / HEAD_DIM as f64)).min(1.0)) as f32).collect();
        let bound = |w: &[f32], rows: usize| -> f32 {
            let mut worst = 0f32;
            for mag in &mags {
                let r: Vec<f32> = (0..rows).map(|r| w[r * d..][..d].iter().zip(mag).map(|(w, m)| w.abs() * m).sum()).collect();
                for (i, &x) in r.iter().enumerate() {
                    let (t, partner) = (i % HEAD_DIM % half, if i % HEAD_DIM < half { i + half } else { i - half });
                    worst = worst.max(x + r[partner] * turn[t]);
                }
            }
            worst
        };
        self.layers.iter().map(|l| (bound(&l.q, self.cfg.attn()), bound(&l.k, self.cfg.kv_dim()))).collect()
    }

    /// What stream `part` of an overlay checkpoint keeps (`Model::stream`): the prefix
    /// of channels its rows read and write, the layers up to the one that routes its
    /// outputs, and per kept layer the units and heads it keeps, by index into the
    /// checkpoint's. With `records`, it also keeps the shared record units in layers
    /// before its last (the reader tokens' routing, at the record part's depth). A dense
    /// pass over a reader record needs them, and a fast plan, whose last layer must route
    /// alone, does not.
    pub fn slice(&self, part: usize, records: bool) -> Slice {
        let o = self.overlay.as_ref().expect("Model::slice takes an overlay checkpoint");
        let own = part as u8;
        // The prefix, widened while the norm's bound (`norm_squares`) needs more channels
        // than it holds. A part whose wires are nibbles can need more than it uses.
        let lay = &self.layout;
        let base = lay.wire_base();
        let square = |ch: usize| ch.checked_sub(base).and_then(|k| lay.occupants.get(k)).map_or(1, |occ| occ.iter().filter(|&&s| lay.part_of(s) == part).map(|s| { let (lo, hi) = self.wire_ranges[s.0 as usize]; (lo.abs().max(hi.abs())).pow(2) }).max().unwrap_or(0));
        // A part with nibble wires routes nibbles, so each route channel counts at 15²,
        // as the engine's bound counts it (`engine::int::norm_bound`).
        let nibbles = (base..lay.after_wires()).any(|ch| square(ch) > 1);
        let routes: Vec<usize> = (0..lay.routes()).map(|i| lay.route(i)).collect();
        let squares = |h: usize| (0..h).map(|ch| if nibbles && routes.contains(&ch) { 225 } else { square(ch) }).sum::<i64>() + CONSTANTS.iter().map(|&k| (k * k) as i64).sum::<i64>();
        let mut hidden = lay.prefix(part).next_multiple_of(HIDDEN_QUANTUM).min(self.cfg.hidden);
        while hidden < self.cfg.hidden && squares(hidden) >= 16 * hidden as i64 {
            hidden = (hidden + HIDDEN_QUANTUM).min(self.cfg.hidden);
        }
        // The loop also stops at the checkpoint's full width, where nothing is left to widen.
        assert!(squares(hidden) < 16 * hidden as i64, "part {part}: the norm's bound fails at the checkpoint's full width {hidden}: its squares sum to {}, the bound is {}", squares(hidden), 16 * hidden);
        let last = (0..self.cfg.layers).rev().find(|&li| o.units[li].contains(&own)).expect("a part with no units");
        let units = (0..=last).map(|li| (0..o.units[li].len()).filter(|&u| o.units[li][u] == own || o.units[li][u] == NO_PART && (records || li == last)).collect()).collect();
        let gathers = self.memory.as_ref().map_or(&[][..], |m| &m.gathers[..]);
        let writes = |li: usize, h: usize| (0..self.cfg.hidden).any(|ch| self.layers[li].o[ch * self.cfg.attn() + h * HEAD_DIM..][..HEAD_DIM].iter().any(|&w| w != 0.0));
        let heads: Vec<Vec<usize>> = (0..=last).map(|li| (0..self.cfg.heads).filter(|&h| if li == 0 { o.holders[h] >> part & 1 == 1 || gathers.contains(&h) } else { o.heads[li][h] == own || o.heads[li][h] == NO_PART && writes(li, h) }).collect()).collect();
        let own_heads = heads.iter().enumerate().map(|(li, hs)| hs.iter().map(|&h| li > 0 && o.heads[li][h] == own).collect()).collect();
        let cleared = (0..lay.offsets.len()).map(|p| lay.mode(p)).chain([lay.null()]).filter(|&ch| ch < hidden).collect();
        Slice { hidden, units, heads, own_heads, null_dims: NULL_DIMS[..o.null_q.len()].to_vec(), mode: lay.mode(part), cleared }
    }

    /// Stream `part`'s own network out of an overlay checkpoint (`Model::slice`): the
    /// checkpoint's vocabulary and slots, its prefix of channels, its layers, and in
    /// each its own units and heads with the shared ones. The engine plans the stream
    /// from this network (`records` false) and runs it as its dense pass (`records` true).
    ///
    /// At a position the stream decodes (one of its own output slots, whose mask token
    /// carries its mode), the checkpoint's residual equals this network's on every kept
    /// channel after every kept layer, and every other channel holds 0 there:
    /// - its own units read `up` from its mode channel, which is 1 there, so here they
    ///   read it from the constant;
    /// - another part's unit, its routing units too, reads a mode channel that is 0
    ///   there, so its `up` is 0 and it adds exactly 0 (its gate is finite, since f32
    ///   never overflows a sum of weights over the channels' ranges), and the network
    ///   leaves it out. The mode is what silences another part's routing unit: the output
    ///   identity it reads is shared (`Layout::identities`) and can be 1 there;
    /// - its own memory heads' null term is `N − N·mode` = 0 there, so the network leaves
    ///   it out, with the null dims of the key and the null channel its key dims read,
    ///   which only the null token sets;
    /// - another part's memory head scores the null token `N` above every key its
    ///   query can reach (`foreign_score`) by `EXP_ZERO` logits, so it attends to the null
    ///   token alone and adds its value, 0, and the network leaves it out;
    /// - a layer-0 gather of input slots the part does not hold (`Overlay::holders`)
    ///   attends to slots the stream holds at byte 0 or the mask, whose values are 0
    ///   (asserted here), and the network leaves it out;
    /// - the network keeps the shared heads (the record gathers, the copy heads) as they
    ///   are;
    /// - a channel above the prefix is no kept row's input or output (asserted here) and
    ///   no token's embedding the stream places, so it holds 0 there; a layer past the
    ///   last kept holds no unit or head of the part (asserted), so it adds 0.
    ///
    /// The norm divides by `sqrt(eps)` exactly at any width (`docs/stock-llama.md` §2:
    /// every residual entry is at most `UNIT` and `mean(x²)` stays below half an ulp of
    /// `EPS`), so dropping zero channels changes no normalised value.
    pub fn stream(&self, part: usize, records: bool) -> Model {
        let o = self.overlay.as_ref().expect("Model::stream takes an overlay checkpoint");
        let (cfg, lay) = (&self.cfg, &self.layout);
        let (d, a, inter, kvd) = (cfg.hidden, cfg.attn(), cfg.intermediate, cfg.kv_dim());
        let own = part as u8;
        let s = self.slice(part, records);
        let h = s.hidden;
        let (n_layers, n_inter, n_heads) = (s.units.len(), s.units.iter().map(Vec::len).max().unwrap_or(0).next_multiple_of(UNIT_QUANTUM), s.heads.iter().map(Vec::len).max().unwrap_or(1));
        let row = |t: usize| &self.embed[t * d..][..d];
        let l0 = &self.layers[0];
        // Every value a layer-0 head can read at a held slot is 0.
        for j in 0..cfg.n_inputs {
            for t in [tok::input(cfg, j, 0), tok::input_mask(cfg, j)] {
                assert!(l0.v.chunks(d).all(|w| w.iter().zip(row(t)).all(|(w, x)| w * x == 0.0)), "slot {j}'s held token has a nonzero layer-0 value");
            }
        }
        for li in n_layers..cfg.layers {
            assert!(!o.units[li].contains(&own) && !o.heads[li].contains(&own), "part {part} has work past its routing layer {}", n_layers - 1);
        }
        let above = |r: &[f32]| r[h..].iter().all(|&w| w == 0.0);
        let layers: Vec<Layer> = s
            .units
            .iter()
            .zip(&s.heads)
            .enumerate()
            .map(|(li, (units, heads))| {
                let l = &self.layers[li];
                let mut n = Layer { q: vec![0.0; n_heads * HEAD_DIM * h], k: vec![0.0; kvd * h], v: vec![0.0; kvd * h], o: vec![0.0; h * n_heads * HEAD_DIM], gate: vec![0.0; n_inter * h], up: vec![0.0; n_inter * h], down: vec![0.0; h * n_inter] };
                let na = n_heads * HEAD_DIM;
                for (nh, &fh) in heads.iter().enumerate() {
                    for t in 0..HEAD_DIM {
                        let r = &l.q[(fh * HEAD_DIM + t) * d..][..d];
                        let null = o.heads[li][fh] == own && NULL_DIMS[..o.null_q.len()].contains(&t) && li > 0;
                        if !null {
                            assert!(above(r), "layer {li} head {fh}: the query reads above the prefix");
                            n.q[(nh * HEAD_DIM + t) * h..][..h].copy_from_slice(&r[..h]);
                        }
                        for ch in 0..d {
                            let w = l.o[ch * a + fh * HEAD_DIM + t];
                            assert!(ch < h || w == 0.0, "layer {li} head {fh}: W_o writes above the prefix");
                            if ch < h {
                                n.o[ch * na + nh * HEAD_DIM + t] = w;
                            }
                        }
                    }
                }
                for r in 0..kvd {
                    let null = li > 0 && NULL_DIMS[..o.null_q.len()].contains(&(r % HEAD_DIM));
                    let (k, v) = (&l.k[r * d..][..d], &l.v[r * d..][..d]);
                    assert!(above(v) && (null || above(k)), "layer {li}: a key or value row reads above the prefix");
                    n.v[r * h..][..h].copy_from_slice(&v[..h]);
                    if !null {
                        n.k[r * h..][..h].copy_from_slice(&k[..h]);
                        // The null channel, which only the null token sets, cancels a key
                        // dim's −1 there; the stream holds no null token.
                        n.k[r * h + lay.null()] = 0.0;
                    }
                }
                for (nu, &fu) in units.iter().enumerate() {
                    let (g, u) = (&l.gate[fu * d..][..d], &l.up[fu * d..][..d]);
                    assert!(above(g), "layer {li} unit {fu}: the gate reads above the prefix");
                    n.gate[nu * h..][..h].copy_from_slice(&g[..h]);
                    if o.units[li][fu] == own && u[0] == 0.0 {
                        assert!(u.iter().enumerate().all(|(ch, &w)| w == 0.0 || ch == lay.mode(part)), "layer {li} unit {fu}: up reads more than its mode");
                        n.up[nu * h] = 1.0;
                    } else {
                        assert!(above(u), "layer {li} unit {fu}: up reads above the prefix");
                        n.up[nu * h..][..h].copy_from_slice(&u[..h]);
                    }
                    for ch in 0..d {
                        let w = l.down[ch * inter + fu];
                        assert!(ch < h || w == 0.0, "layer {li} unit {fu}: the down projection writes above the prefix");
                        if ch < h {
                            n.down[ch * n_inter + nu] = w;
                        }
                    }
                }
                n
            })
            .collect();
        let rows = |w: &[f32]| -> Vec<f32> {
            w.chunks(d).flat_map(|r| r[..h].iter().copied()).collect()
        };
        assert!(self.lm_head.chunks(d).all(|r| above(r)), "the LM head reads above the prefix");
        let foreign = |sig: Sig| lay.part_of(sig) != part;
        let mut wire_ch = lay.wire_ch.clone();
        for (w, ch) in wire_ch.iter_mut().enumerate() {
            if foreign(Sig(w as u32)) {
                *ch = u32::MAX;
            }
        }
        let wires = h.saturating_sub(lay.wire_base()).min(lay.wires);
        // The part's output slots: the ones whose mask carries its mode.
        let masked = |j: usize| row(tok::output_mask(cfg, j))[lay.mode(part)] != 0.0;
        let first = (cfg.n_inputs..cfg.seq()).find(|&j| masked(j)).expect("a part with no output slot");
        let own_outputs = first..(first..cfg.seq()).find(|&j| !masked(j)).unwrap_or(cfg.seq());
        assert!((cfg.n_inputs..cfg.seq()).all(|j| masked(j) == own_outputs.contains(&j)), "part {part}'s output slots are not one run");
        let occupants = lay.occupants[..wires].iter().map(|occ| occ.iter().copied().filter(|&s| !foreign(s)).collect()).collect();
        let remap = |(li, fh): (usize, usize)| (li, s.heads.get(li).and_then(|hs| hs.iter().position(|&x| x == fh)).unwrap_or(usize::MAX));
        let memory = self.memory.as_ref().map(|m| MemorySpec {
            heads: m.heads.iter().copied().filter(|&at| remap(at).1 != usize::MAX).map(remap).collect(),
            gathers: m.gathers.map(|g| remap((0, g)).1),
            copy_heads: m.copy_heads.map(remap),
            ..m.clone()
        });
        Model {
            cfg: Config { hidden: h, intermediate: n_inter, layers: n_layers, heads: n_heads, ..cfg.clone() },
            embed: unread_modes(&layers, &rows(&self.embed), lay, h),
            layers,
            lm_head: rows(&self.lm_head),
            slots: self.slots.clone(),
            unit_names: s.units.iter().enumerate().map(|(li, us)| us.iter().map(|&u| self.unit_names[li].get(u).cloned().unwrap_or_default()).collect()).collect(),
            wire_names: self.wire_names.clone(),
            wire_ranges: self.wire_ranges.clone(),
            wire_levels: self.wire_levels.clone(),
            wire_groups: self.wire_groups.clone(),
            memory,
            layout: Layout { wires, wire_ch, occupants, own_outputs: Some(own_outputs), ..lay.clone() },
            tiles: self.tiles,
            overlay: None,
        }
    }

    /// The position every token of a group sits at: above every memory cell for a
    /// memory checkpoint (`memory::attn::READ_AT`), else 0. One position for the whole
    /// group keeps a gather's keys bit-identical, so each weighs exactly 1/len.
    pub fn group_position(&self) -> usize {
        if self.memory.is_some() { crate::memory::attn::READ_AT } else { 0 }
    }

    /// Dense forward pass over token ids; returns the argmax token per position.
    pub fn forward(&self, tokens: &[usize], causal: bool) -> Vec<usize> {
        let d = self.cfg.hidden;
        let x = self.residual(tokens, causal);
        let logits = self.tiles.mm(&self.tiles.all(self.tiles.norm(&x, d)), &self.lm_head, d, self.cfg.vocab);
        logits.chunks(self.cfg.vocab).map(argmax).collect()
    }

    /// Final residual stream, before the output norm.
    pub fn residual(&self, tokens: &[usize], causal: bool) -> Vec<f32> {
        let (d, a, kvd) = (self.cfg.hidden, self.cfg.attn(), self.cfg.kv_dim());
        let s = tokens.len();
        let mut x: Vec<f32> = tokens.iter().flat_map(|&t| self.embed[t * d..][..d].iter().copied()).collect();
        let (cos, sin) = rope(s, self.group_position());
        for l in &self.layers {
            let h = self.tiles.all(self.tiles.norm(&x, d));
            let q = self.tiles.all(self.tiles.rope(self.tiles.mm(&h, &l.q, d, a), &cos, &sin, a));
            let k = self.tiles.all(self.tiles.rope(self.tiles.mm(&h, &l.k, d, kvd), &cos, &sin, kvd));
            let v = self.tiles.all(self.tiles.mm(&h, &l.v, d, kvd));
            let mut attn = vec![0f32; s * a];
            for head in 0..self.cfg.heads {
                let kv = (kvd, self.cfg.kv_head(head));
                for i in 0..s {
                    let visible = if causal { i + 1 } else { s };
                    attend(&q, &k, &v, a, head, kv, i, visible, self.tiles, &mut attn[i * a..][..a]);
                }
            }
            self.tiles.add(&mut x, &self.tiles.mm(&self.tiles.all(attn), &l.o, a, d));
            let h = self.tiles.all(self.tiles.norm(&x, d));
            let i = self.cfg.intermediate;
            let g = self.tiles.mm(&h, &l.gate, d, i);
            let u = self.tiles.mm(&h, &l.up, d, i);
            let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| self.tiles.act(*g, *u)).collect();
            self.tiles.add(&mut x, &self.tiles.mm(&self.tiles.all(act), &l.down, i, d));
        }
        x
    }
}

/// The precision of the dense pass's matmul inputs and KV entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Tiles {
    /// f32 throughout, as llama.cpp's CPU backend and HF in f32 compute.
    F32,
    /// Every matmul input, every query, key and value and every softmax weight rounded
    /// to f16, with products summed in f32, as llama.cpp's Metal `mul_mm` with half
    /// tiles and an f16 KV cache, or MLX in f16, computes. Exact only for a checkpoint
    /// `Model::check_f16` accepts.
    F16,
    /// f16 throughout (T2, Core ML on the Neural Engine). Every product, every partial
    /// sum and every elementwise result is rounded to f16, and the sums are taken in
    /// `Order`. Exact only for a checkpoint `Model::check_f16_acc` accepts.
    F16Acc(Order),
}

/// The order an f16-accumulation pass sums a row's products in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Order {
    /// Input order.
    Forward,
    /// Every positive product first, then the negative ones, which reaches the largest
    /// partial sums a row can.
    PositivesFirst,
    /// The negative products first.
    NegativesFirst,
}

impl Tiles {
    pub fn round(self, x: f32) -> f32 {
        match self {
            Tiles::F32 => x,
            _ => f16(x),
        }
    }
    fn all(self, mut v: Vec<f32>) -> Vec<f32> {
        if self != Tiles::F32 {
            v.iter_mut().for_each(|x| *x = f16(*x));
        }
        v
    }
    /// Σ of the products, each rounded and summed in f16 in `order`.
    fn sum16(order: Order, products: &mut Vec<f32>) -> f32 {
        match order {
            Order::Forward => {}
            Order::PositivesFirst => products.sort_by_key(|p| *p < 0.0),
            Order::NegativesFirst => products.sort_by_key(|p| *p > 0.0),
        }
        products.iter().fold(0.0, |acc, &p| f16(acc + f16(p)))
    }
    /// y = x·Wᵀ.
    fn mm(self, x: &[f32], w: &[f32], din: usize, dout: usize) -> Vec<f32> {
        let Tiles::F16Acc(order) = self else { return matmul(x, w, din, dout) };
        let mut y = vec![0f32; x.len() / din * dout];
        let mut products = Vec::new();
        for (xi, yi) in x.chunks(din).zip(y.chunks_mut(dout)) {
            let nz: Vec<usize> = (0..din).filter(|&i| xi[i] != 0.0).collect();
            for (o, yo) in yi.iter_mut().enumerate() {
                let row = &w[o * din..][..din];
                products.clear();
                products.extend(nz.iter().filter(|&&i| row[i] != 0.0).map(|&i| row[i] * xi[i]));
                *yo = Self::sum16(order, &mut products);
            }
        }
        y
    }
    fn norm(self, x: &[f32], d: usize) -> Vec<f32> {
        if !matches!(self, Tiles::F16Acc(_)) {
            return rms_norm(x, d);
        }
        x.chunks(d)
            .flat_map(|r| {
                let ss = r.iter().fold(0.0, |acc, v| f16(acc + f16(v * v)));
                let inv = f16(1.0 / f16(f16(f16(ss / d as f32) + EPS).sqrt()));
                r.iter().map(move |v| f16(f16(v * inv) * GAMMA))
            })
            .collect()
    }
    fn act(self, g: f32, u: f32) -> f32 {
        match self {
            Tiles::F16Acc(_) => f16(f16(g * f16(1.0 / f16(1.0 + f16((-g).exp())))) * u),
            _ => silu(g) * u,
        }
    }
    fn add(self, x: &mut [f32], y: &[f32]) {
        match self {
            Tiles::F16Acc(_) => x.iter_mut().zip(y).for_each(|(a, b)| *a = f16(*a + b)),
            _ => add(x, y),
        }
    }
    fn rope(self, x: Vec<f32>, cos: &[f32], sin: &[f32], a: usize) -> Vec<f32> {
        if !matches!(self, Tiles::F16Acc(_)) {
            return apply_rope(x, cos, sin, a);
        }
        let half = HEAD_DIM / 2;
        let mut y = x.clone();
        for (p, row) in x.chunks(a).enumerate() {
            for h in 0..a / HEAD_DIM {
                for t in 0..HEAD_DIM {
                    let rot = if t < half { -row[h * HEAD_DIM + t + half] } else { row[h * HEAD_DIM + t - half] };
                    let (c, s) = (f16(cos[p * HEAD_DIM + t]), f16(sin[p * HEAD_DIM + t]));
                    y[p * a + h * HEAD_DIM + t] = f16(f16(row[h * HEAD_DIM + t] * c) + f16(rot * s));
                }
            }
        }
        y
    }
    /// A scaled attention logit.
    fn dot(self, q: &[f32], k: &[f32], scale: f32) -> f32 {
        match self {
            Tiles::F16Acc(order) => {
                let mut p: Vec<f32> = (0..HEAD_DIM).map(|t| q[t] * k[t]).filter(|p| *p != 0.0).collect();
                f16(Self::sum16(order, &mut p) * f16(scale))
            }
            _ => (0..HEAD_DIM).map(|t| q[t] * k[t]).sum::<f32>() * scale,
        }
    }
    fn exp(self, z: f32, m: f32) -> f32 {
        match self {
            Tiles::F16Acc(_) => f16(f16(z - m).exp()),
            _ => (z - m).exp(),
        }
    }
    fn sum(self, e: impl Iterator<Item = f32>) -> f32 {
        match self {
            Tiles::F16Acc(_) => e.fold(0.0, |acc, x| f16(acc + x)),
            _ => e.sum(),
        }
    }
    fn div(self, w: f32, z: f32) -> f32 {
        self.round(w / z)
    }
    /// out += Σ w·v over the (weight, value) terms.
    fn mix(self, terms: &[(f32, [f32; HEAD_DIM])], out: &mut [f32]) {
        match self {
            Tiles::F16Acc(order) => {
                for t in 0..HEAD_DIM {
                    let mut p: Vec<f32> = std::iter::once(out[t]).chain(terms.iter().map(|(w, v)| w * v[t])).filter(|p| *p != 0.0).collect();
                    out[t] = Self::sum16(order, &mut p);
                }
            }
            _ => {
                for (w, v) in terms {
                    for t in 0..HEAD_DIM {
                        out[t] += w * v[t];
                    }
                }
            }
        }
    }
}

/// `x` rounded to the nearest f16, ties to even, as an f32; ±inf beyond the f16 range.
pub fn f16(x: f32) -> f32 {
    let a = x.abs();
    if a == 0.0 || !a.is_finite() {
        return x;
    }
    // The f16 quantum at `a`: 2^(e − 10) for a normal f16, 2^-24 below 2^-14.
    let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
    let q = e.max(-14) - 10;
    let pow2 = |n: i32| f32::from_bits(((n + 127) as u32) << 23);
    let r = (a * pow2(-q)).round_ties_even() * pow2(q);
    if r > 65504.0 { f32::INFINITY.copysign(x) } else { r.copysign(x) }
}

impl Model {
    /// Checks that every weight of the checkpoint holds its value exactly in f16. `Err`
    /// names each tensor that does not, with how many entries fail and the largest of
    /// them, and a scale constant that f16 cannot hold.
    pub fn check_f16(&self) -> Result<(), String> {
        let mut tensors: Vec<(String, &[f32])> = vec![("embed".into(), &self.embed), ("lm_head".into(), &self.lm_head)];
        for (i, l) in self.layers.iter().enumerate() {
            for (n, w) in [("q", &l.q), ("k", &l.k), ("v", &l.v), ("o", &l.o), ("gate", &l.gate), ("up", &l.up), ("down", &l.down)] {
                tensors.push((format!("layer {i} {n}"), w));
            }
        }
        let mut failures = Vec::new();
        for (name, w) in tensors {
            let bad: Vec<f32> = w.iter().copied().filter(|&x| f16(x) != x).collect();
            if !bad.is_empty() {
                failures.push(format!("{name}: {} entries, largest |w| {}", bad.len(), bad.iter().fold(0f32, |m, x| m.max(x.abs()))));
            }
        }
        if [EPS, GAMMA, UNIT, K].iter().any(|&x| f16(x) != x) {
            failures.push("a scale constant".into());
        }
        if failures.is_empty() { Ok(()) } else { Err(failures.join("; ")) }
    }

    /// Every channel's range and one-hot group where layer `layer` reads the residual:
    /// its attention, or with `mlp` its MLP. A wire channel holds the latest occupant
    /// written by then. A cleared occupant still counts, since its range holds the 0 it
    /// is cleared to. The output slot identities, a token's own one-hot and the reader
    /// tokens' identities are one-hot groups.
    pub fn channel_vars(&self, layer: usize, mlp: bool) -> ChannelVars {
        self.channel_vars_of(layer, mlp, None)
    }

    /// `channel_vars` as an overlay part's rows see it at its own positions, where
    /// every other part's occupant holds 0 (`Model::stream`). With `part` None, it
    /// covers every part's occupants.
    pub fn channel_vars_of(&self, layer: usize, mlp: bool, part: Option<usize>) -> ChannelVars {
        let lay = &self.layout;
        let mut v = vec![(0i64, 1i64, NO_GROUP); self.cfg.hidden];
        for x in v.iter_mut().skip(lay.used()) {
            *x = (0, 0, NO_GROUP);
        }
        for (i, &c) in CONSTANTS.iter().enumerate() {
            v[i] = (c as i64, c as i64, NO_GROUP);
        }
        let special = self.wire_groups.iter().filter(|&&g| g != NO_GROUP).max().map_or(0, |g| g + 1);
        for j in self.cfg.n_inputs..self.cfg.seq() {
            v[lay.slot_id(j)].2 = special;
        }
        if lay.hots {
            for x in 0..256 {
                v[lay.own_hot(x)].2 = special + 1;
            }
        }
        if lay.memory {
            for k in 0..LINE {
                v[lay.rd(k)].2 = special + 2;
            }
        }
        // Wires of level ≤ `upto` are in the residual. A layer's attention writes its
        // heads' wires (layer 0 the inputs'), and its MLP writes its nodes'.
        let upto = if mlp { Some(layer as u32) } else { (layer as u32).checked_sub(1) };
        // An overlay's channel holds each part's latest occupant at that part's
        // positions; a stock runtime runs every row at every position, so the channel
        // ranges over all of them, in no one-hot group when two parts meet.
        let parts = lay.offsets.len().max(1);
        for (ch, occ) in lay.occupants.iter().enumerate() {
            let at = lay.wire_base() + ch;
            v[at] = (0, 0, NO_GROUP);
            let mut met = 0;
            for p in (0..parts).filter(|&p| part.is_none_or(|q| q == p)) {
                if let Some(s) = upto.and_then(|up| occ.iter().rev().find(|s| lay.part_of(**s) == p && self.wire_levels[s.0 as usize] <= up)) {
                    let (a, b) = self.wire_ranges[s.0 as usize];
                    let g = if met == 0 { self.wire_groups[s.0 as usize] } else { NO_GROUP };
                    v[at] = (v[at].0.min(a), v[at].1.max(b), g);
                    met += 1;
                }
            }
        }
        ChannelVars(v)
    }

    /// Checks that every matmul of the checkpoint holds its partial sums exactly in an
    /// f16 accumulator, in any order (T2, `docs/stock-llama.md` §6). Per row, its products
    /// are multiples of one power of two q, and the largest subset sum
    /// (`f16acc::gate_mass` over the input channels' ranges, or `f16acc::down_mass` over
    /// the MLP units) is at most 2^11·q and below 65,504. The residual entry a fused
    /// accumulate adds counts too. `Err` names each failing row family and its worst row.
    pub fn check_f16_acc(&self) -> Result<(), String> {
        self.f16_acc_peaks().map(|_| ())
    }

    /// `check_f16_acc`, and on success the largest magnitude a partial sum reaches in
    /// each tensor family (`q k v o gate down lm_head`), over every layer. The Neural
    /// Engine's matmul overflows at 2^15, not f16's 65,504 (`docs/stock-llama.md` §10).
    pub fn f16_acc_peaks(&self) -> Result<std::collections::BTreeMap<&'static str, f64>, String> {
        let (d, a, kvd, inter) = (self.cfg.hidden, self.cfg.attn(), self.cfg.kv_dim(), self.cfg.intermediate);
        let mut peaks = std::collections::BTreeMap::new();
        let mut peak = |family: &'static str, m: f64| {
            let p = peaks.entry(family).or_insert(0.0f64);
            *p = p.max(m);
        };
        let mut failures: Vec<String> = Vec::new();
        let mut count = 0usize;
        let mut fail = |what: String| {
            count += 1;
            if failures.len() < 40 {
                failures.push(what);
            }
        };
        // Integer weights over the row's quantum, or None when a weight is not dyadic.
        let integral = |w: &[f32]| -> Option<(f64, Vec<(usize, i64)>)> {
            let q = w.iter().filter(|x| **x != 0.0).map(|&x| lowbit(x)).fold(f64::INFINITY, f64::min);
            let terms: Vec<(usize, i64)> = w.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, &x)| (i, (x as f64 / q) as i64)).collect();
            terms.iter().all(|&(i, n)| n as f64 * q == w[i] as f64).then_some((q, terms))
        };
        // The row's largest partial-sum magnitude.
        let row_ok = |w: &[f32], vars: &ChannelVars| -> Result<f64, String> {
            if w.iter().all(|&x| x == 0.0) {
                return Ok(0.0);
            }
            let (q, terms) = integral(w).ok_or("a weight is not dyadic")?;
            let m = f16acc::gate_mass(&Form { terms, c: 0 }, vars);
            if m <= 2048 && m as f64 * q <= 65504.0 { Ok(m as f64 * q) } else { Err(format!("mass {m} of quantum {q}")) }
        };
        for (li, l) in self.layers.iter().enumerate() {
            let at = self.channel_vars(li, false);
            // An overlay head's query counts where its read does, at its part's positions.
            // Elsewhere the null term outscores it, and a runtime that runs it there needs f32
            // sums or the part's slice (`Model::stream`), as an overlay unit's gate does.
            let own_q: Vec<ChannelVars> = (0..self.layout.offsets.len()).map(|p| self.channel_vars_of(li, false, Some(p))).collect();
            let q_vars = |r: usize| self.overlay.as_ref().and_then(|o| own_q.get(o.heads[li][r / HEAD_DIM] as usize)).unwrap_or(&at);
            for (name, w, rows) in [("q", &l.q, a), ("k", &l.k, kvd), ("v", &l.v, kvd)] {
                for r in 0..rows {
                    match row_ok(&w[r * d..][..d], if name == "q" { q_vars(r) } else { &at }) {
                        Ok(m) => peak(name, m),
                        Err(e) => fail(format!("layer {li} {name} row {r}: {e}")),
                    }
                }
            }
            // For W_o, each attention dim is a convex mix of its value row's values, on the
            // value weights' quantum over the uniform weight of a gather (1/GROUP).
            let vmax: Vec<(f64, f64)> = (0..a)
                .map(|t| {
                    let r = &l.v[(self.cfg.kv_head(t / HEAD_DIM) * HEAD_DIM + t % HEAD_DIM) * d..][..d];
                    let m = r.iter().enumerate().map(|(ch, &x)| (x as f64 * at.0[ch].1 as f64).abs().max((x as f64 * at.0[ch].0 as f64).abs())).sum::<f64>();
                    let q = r.iter().filter(|x| **x != 0.0).map(|&x| lowbit(x)).fold(f64::INFINITY, f64::min) / GROUP as f64;
                    (m, q)
                })
                .collect();
            for ch in 0..d {
                let r = &l.o[ch * a..][..a];
                let (mut mass, mut q) = (UNIT as f64, UNIT as f64);
                for (t, &x) in r.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                    mass += (x as f64).abs() * vmax[t].0;
                    q = q.min(lowbit(x) * vmax[t].1);
                }
                if mass > 2048.0 * q || mass > 65504.0 {
                    fail(format!("layer {li} o row {ch}: mass {mass} of quantum {q}"));
                }
                peak("o", mass);
            }
            let at = self.channel_vars(li, true);
            // An overlay unit's gate counts where its output does, at its part's
            // positions. Elsewhere its `up` is 0, and a runtime that runs it there needs f32
            // sums or the part's slice (`Model::stream`).
            let own: Vec<ChannelVars> = (0..self.layout.offsets.len()).map(|p| self.channel_vars_of(li, true, Some(p))).collect();
            let vars_of = |u: usize| self.overlay.as_ref().and_then(|o| own.get(o.units[li][u] as usize)).unwrap_or(&at);
            let mut units: Vec<Option<Form>> = Vec::with_capacity(inter);
            for u in 0..inter {
                let g = &l.gate[u * d..][..d];
                let up = &l.up[u * d..][..d];
                if g.iter().all(|&x| x == 0.0) {
                    units.push(None);
                    continue;
                }
                // The constant, or an overlay part's mode (0 or 1, so the unit's output is
                // at most what the constant gives it).
                let modes: Vec<usize> = (0..self.layout.offsets.len()).map(|p| self.layout.mode(p)).collect();
                let reads: Vec<usize> = (0..d).filter(|&ch| up[ch] != 0.0).collect();
                if !matches!(reads[..], [ch] if up[ch] == 1.0 && (ch == 0 || modes.contains(&ch))) {
                    fail(format!("layer {li} unit {u}: up is not the constant 1 or a mode channel"));
                }
                match row_ok(g, vars_of(u)) {
                    Ok(m) => peak("gate", m),
                    Err(e) => fail(format!("layer {li} gate {u} ({}): {e}", self.unit_names[li].get(u).map_or("", |s| s))),
                }
                // The activation in units of K is the gate's integer form, with the
                // constant channels folded into its constant.
                let mut f = Form { terms: Vec::new(), c: 0 };
                for (ch, &x) in g.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                    let n = x / K;
                    if n.fract() != 0.0 {
                        fail(format!("layer {li} gate {u}: weight {x} is not a multiple of K"));
                    }
                    match at.0[ch] {
                        (lo, hi, _) if lo == hi => f.c += n as i64 * lo,
                        _ => f.terms.push((ch, n as i64)),
                    }
                }
                units.push(Some(f));
            }
            for ch in 0..d {
                let row = &l.down[ch * inter..][..inter];
                let mut us = Vec::new();
                for (u, &x) in row.iter().enumerate().filter(|(_, x)| **x != 0.0) {
                    let coef = x * K / UNIT;
                    if coef.fract() != 0.0 || units[u].is_none() {
                        fail(format!("layer {li} down row {ch}: unit {u}'s weight {x} is off the grid"));
                        continue;
                    }
                    us.push((u, Unit { form: units[u].clone().unwrap(), down: coef as i64 }));
                }
                if !us.is_empty() {
                    // An overlay's down row sums one part's units at a time (every other
                    // part's `up` is 0 there), with the shared ones.
                    let part = |u: usize| self.overlay.as_ref().map_or(NO_PART, |o| o.units[li][u]);
                    let mut parts: Vec<u8> = us.iter().map(|&(u, _)| part(u)).filter(|&p| p != NO_PART).collect();
                    parts.sort_unstable();
                    parts.dedup();
                    if parts.is_empty() {
                        parts.push(NO_PART);
                    }
                    let m = parts
                        .iter()
                        .map(|&p| {
                            let group: Vec<Unit> = us.iter().filter(|&&(u, _)| part(u) == p || part(u) == NO_PART).map(|(_, x)| x.clone()).collect();
                            f16acc::down_mass(&group, own.get(p as usize).unwrap_or(&at)) + 1
                        })
                        .max()
                        .unwrap();
                    if m > 2048 {
                        fail(format!("layer {li} down row {ch}: mass {m}"));
                    }
                    peak("down", m as f64 * UNIT as f64);
                }
            }
        }
        let at = self.channel_vars(self.layers.len() - 1, true);
        for t in 0..self.cfg.vocab {
            match row_ok(&self.lm_head[t * d..][..d], &at) {
                Ok(m) => peak("lm_head", m),
                Err(e) => fail(format!("lm_head row {t}: {e}")),
            }
        }
        if failures.is_empty() { Ok(peaks) } else { Err(format!("{count} rows: {}", failures.join("; "))) }
    }
}

/// The largest power of two dividing a nonzero f32.
fn lowbit(x: f32) -> f64 {
    let b = x.abs().to_bits();
    let (e, m) = ((b >> 23) as i32, b & 0x7F_FFFF);
    let (e, m) = if e == 0 { (1, m) } else { (e, m | 0x80_0000) };
    2f64.powi(e - 150 + m.trailing_zeros() as i32)
}

/// Channel ranges and one-hot groups at one read of the residual (`Model::channel_vars`).
pub struct ChannelVars(pub Vec<(i64, i64, u32)>);

impl f16acc::Vars for ChannelVars {
    fn range(&self, v: usize) -> (i64, i64) {
        (self.0[v].0, self.0[v].1)
    }
    fn group(&self, v: usize) -> u32 {
        self.0[v].2
    }
}

pub fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, t| if row[t] > row[b] { t } else { b })
}

/// One query row of one head, written into `out` (that position's attention output);
/// `kv` is the key and value width and the KV head `head` reads.
#[allow(clippy::too_many_arguments)]
pub fn attend(q: &[f32], k: &[f32], v: &[f32], a: usize, head: usize, (kvd, kvh): (usize, usize), i: usize, visible: usize, tiles: Tiles, out: &mut [f32]) {
    let (base, kb) = (head * HEAD_DIM, kvh * HEAD_DIM);
    let logits: Vec<f32> = (0..visible).map(|j| tiles.dot(&q[i * a + base..][..HEAD_DIM], &k[j * kvd + kb..][..HEAD_DIM], attn_scale())).collect();
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|z| tiles.exp(*z, m)).collect();
    let z = tiles.sum(e.iter().copied());
    let terms: Vec<(f32, [f32; HEAD_DIM])> = e.iter().enumerate().map(|(j, w)| (tiles.div(*w, z), std::array::from_fn(|t| v[j * kvd + kb + t]))).collect();
    tiles.mix(&terms, &mut out[base..][..HEAD_DIM]);
}

pub fn rms_norm(x: &[f32], d: usize) -> Vec<f32> {
    x.chunks(d)
        .flat_map(|r| {
            let ms = r.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            r.iter().map(move |v| v * inv * GAMMA)
        })
        .collect()
}

/// y = x·Wᵀ. Zero inputs are skipped without reordering the remaining terms, so the
/// result is bit-identical to the full dot product.
fn matmul(x: &[f32], w: &[f32], din: usize, dout: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len() / din * dout];
    each_row(&mut y, dout, x.len() * dout, |r, yi| {
        let xi = &x[r * din..][..din];
        let nz: Vec<usize> = (0..din).filter(|&i| xi[i] != 0.0).collect();
        for (o, yo) in yi.iter_mut().enumerate() {
            let row = &w[o * din..][..din];
            *yo = nz.iter().fold(0.0, |acc, &i| acc + row[i] * xi[i]);
        }
    });
    y
}

/// Multiply-adds above which `each_row` splits its rows across threads.
const PARALLEL_WORK: usize = 1 << 26;

/// `f(r, row)` on each `width`-element row of `out`. When `work` (about the
/// multiply-adds it takes) passes `PARALLEL_WORK`, blocks of rows run on threads of their
/// own. Each row is computed alone, so the result is the serial loop's bit for bit.
fn each_row(out: &mut [f32], width: usize, work: usize, f: impl Fn(usize, &mut [f32]) + Sync) {
    let rows = out.len() / width;
    let threads = if cfg!(target_arch = "wasm32") || work < PARALLEL_WORK { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows) };
    if threads <= 1 {
        out.chunks_mut(width).enumerate().for_each(|(r, row)| f(r, row));
        return;
    }
    let per = rows.div_ceil(threads);
    let f = &f;
    std::thread::scope(|s| {
        for (b, block) in out.chunks_mut(per * width).enumerate() {
            s.spawn(move || block.chunks_mut(width).enumerate().for_each(|(k, row)| f(b * per + k, row)));
        }
    });
}

fn add(x: &mut [f32], y: &[f32]) {
    x.iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

pub fn silu(z: f32) -> f32 {
    z / (1.0 + (-z).exp())
}

/// cos and sin per head dim at one position. The angle is formed in f64 and the
/// trigonometric values are cast to f32; HF forms the angle in f32, which this project
/// does not reproduce (see `docs/memory-in-weights.md`, §3).
pub fn rope_at(p: usize) -> ([f32; HEAD_DIM], [f32; HEAD_DIM]) {
    // `ROPE_THETA.powf(2t / dim)` per pair, computed once: the quotient below is the
    // same f64 as dividing by the freshly computed power, so the tables are bit-identical
    // to HF's `position / base ** (2t / dim)` formed in f64.
    static DENOM: LazyLock<[f64; HEAD_DIM / 2]> = LazyLock::new(|| std::array::from_fn(|t| ROPE_THETA.powf(2.0 * t as f64 / HEAD_DIM as f64)));
    let half = HEAD_DIM / 2;
    let (mut cos, mut sin) = ([0f32; HEAD_DIM], [0f32; HEAD_DIM]);
    for t in 0..half {
        let ang = p as f64 / DENOM[t];
        let (c, s) = (ang.cos() as f32, ang.sin() as f32);
        (cos[t], cos[t + half]) = (c, c);
        (sin[t], sin[t + half]) = (s, s);
    }
    (cos, sin)
}

/// cos and sin tables for `s` tokens that all sit at position `p`, laid out per token.
pub fn rope(s: usize, p: usize) -> (Vec<f32>, Vec<f32>) {
    let (c, si) = rope_at(p);
    (c.repeat(s), si.repeat(s))
}

/// The memory cells a stock runtime's KV cache holds (`docs/stock-llama.md` §4, §10):
/// one per 8-byte line, at the line's index as its position, its key
/// `memory::attn::key` and its value the line's 64 bits. Every memory head has the same
/// key and value weights, so one key and value per cell serves them all. Every other
/// head's key on a cell is zero by the vocabulary (no cell carries a slot identity), so
/// those heads attend over the current group alone.
pub struct KvCache {
    /// The heads that read the cells.
    pub heads: Vec<(usize, usize)>,
    /// (line index, line), by index.
    cells: Vec<(u32, [u8; LINE])>,
    /// Per line, where `set` last found its cell in `cells`. A hint, checked before use:
    /// an insertion or a removal moves the cells after it.
    hint: Vec<u32>,
}

pub type Kv = ([f32; HEAD_DIM], [f32; HEAD_DIM]);

impl KvCache {
    pub fn new(heads: Vec<(usize, usize)>) -> KvCache {
        KvCache { heads, cells: Vec::new(), hint: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Bytes a stock runtime's f32 cache holds for these cells at one memory layer's
    /// KV head; `Config::kv_bytes_per_token` gives the whole model's.
    pub fn bytes(&self) -> usize {
        self.len() * 2 * HEAD_DIM * 4
    }

    /// Sets the cell of line `n` to `line`, for a write's eviction and record or for a
    /// load.
    pub fn set(&mut self, n: u32, line: [u8; LINE]) {
        let hint = self.hint.get(n as usize).map_or(usize::MAX, |&i| i as usize);
        if let Some(cell) = self.cells.get_mut(hint).filter(|c| c.0 == n) {
            cell.1 = line;
            return;
        }
        match self.cells.binary_search_by_key(&n, |c| c.0) {
            Ok(i) => {
                self.cells[i].1 = line;
                // The table comes with the first cell written again: a cache of one or two
                // cells never pays for it.
                if self.hint.is_empty() {
                    self.hint = vec![u32::MAX; crate::memory::layout::LINES as usize];
                }
                self.hint[n as usize] = i as u32;
            }
            Err(i) => self.cells.insert(i, (n, line)),
        }
    }

    /// Every cell of `image`, in line order.
    pub fn fill(&mut self, image: &crate::memory::image::Image) {
        self.cells = (0..crate::memory::layout::LINES).filter(|&n| image.has_line(n)).map(|n| (n, image.line(n))).collect();
    }

    /// Drops the cell of line `n`, as the runtime evicts it before a write's record.
    pub fn remove(&mut self, n: u32) {
        if let Ok(i) = self.cells.binary_search_by_key(&n, |c| c.0) {
            self.cells.remove(i);
        }
    }

    /// Every cell, in line order.
    pub fn cells(&self) -> &[(u32, [u8; LINE])] {
        &self.cells
    }

    pub fn get(&self, n: u32) -> Option<[u8; LINE]> {
        self.cells.binary_search_by_key(&n, |c| c.0).ok().map(|i| self.cells[i].1)
    }

    /// Cell `j`'s key and value, roped at its line index.
    pub fn kv(&self, j: usize) -> Kv {
        let (n, line) = self.cells[j];
        (key(Kind::Cell, n, n as usize), value_of(&line))
    }

    fn materialised(&self) -> Vec<Kv> {
        (0..self.len()).map(|j| self.kv(j)).collect()
    }
}

/// The null token's key and value as the cache holds it: `null_k` on each null dim
/// the overlay uses, roped at position 0 (no turn), and the value 0.
pub fn null_kv(o: &Overlay) -> Kv {
    let mut k = [0f32; HEAD_DIM];
    for &dim in &NULL_DIMS[..o.null_q.len()] {
        k[dim] = o.null_k;
    }
    (k, [0f32; HEAD_DIM])
}

/// A group's keys and values per layer, as the dense pass computed them: `k[l]` and
/// `v[l]` hold `tokens × attn` values, roped keys included.
pub struct GroupKv {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl GroupKv {
    /// The roped key and value query head `(layer, head)` reads at group position `i`.
    pub fn head(&self, cfg: &Config, (layer, head): (usize, usize), i: usize) -> ([f32; HEAD_DIM], [f32; HEAD_DIM]) {
        let (kvd, base) = (cfg.kv_dim(), cfg.kv_head(head) * HEAD_DIM);
        let (mut k, mut v) = ([0f32; HEAD_DIM], [0f32; HEAD_DIM]);
        k.copy_from_slice(&self.k[layer][i * kvd + base..][..HEAD_DIM]);
        v.copy_from_slice(&self.v[layer][i * kvd + base..][..HEAD_DIM]);
        (k, v)
    }
}

impl Model {
    /// Dense forward of a token group that all sits at `position`, with the cache's
    /// tokens before it. The memory heads attend over the cache and the group; a head
    /// whose `W_o` is zero contributes nothing whatever it attends to; any other head
    /// attends over the group and must find a logit gap of at least 104 there, else the
    /// cached tokens it cannot see would matter and the pass panics rather than guess.
    /// Returns the argmax token per group position and the group's keys and values.
    pub fn forward_cached(&self, cache: &KvCache, tokens: &[usize], position: usize) -> (Vec<usize>, GroupKv) {
        let (x, group) = self.residual_cached(cache, tokens, position);
        (self.tokens_of(&x), group)
    }

    /// The argmax token per position of a final residual stream (`residual_cached`'s).
    pub fn tokens_of(&self, x: &[f32]) -> Vec<usize> {
        let d = self.cfg.hidden;
        let logits = self.tiles.mm(&self.tiles.all(self.tiles.norm(x, d)), &self.lm_head, d, self.cfg.vocab);
        logits.chunks(self.cfg.vocab).map(argmax).collect()
    }

    /// `forward_cached`'s final residual stream, before the output norm, and the
    /// group's keys and values.
    pub fn residual_cached(&self, cache: &KvCache, tokens: &[usize], position: usize) -> (Vec<f32>, GroupKv) {
        if let Some(t) = self.memory.as_ref().and_then(|m| tok::stray(&self.cfg, &m.slots, tokens)) {
            panic!("a record's sequence holds token {t}, which no record holds: its gathers would weigh it with the record's own");
        }
        let null = self.overlay.as_ref().map(|o| vec![null_kv(o); self.cfg.layers]);
        self.residual_with_null(cache, tokens, position, null.as_deref())
    }

    /// The null token's key and value at each layer as a runtime computes them when it
    /// decodes the token alone, in a sequence of its own at position 0, so every head
    /// attends to it alone and RoPE turns nothing.
    pub fn null_cell(&self) -> Vec<Kv> {
        let empty = KvCache::new(self.memory.as_ref().map_or(Vec::new(), |m| m.heads.clone()));
        let (_, g) = self.residual_with_null(&empty, &[tok::null(&self.cfg)], 0, None);
        (0..self.cfg.layers).map(|li| g.head(&self.cfg, (li, 0), 0)).collect()
    }

    /// `residual_cached` with `null`, per layer, the one cell the memory heads find
    /// besides the cache's (an overlay's null cell), or with no such cell.
    pub fn residual_with_null(&self, cache: &KvCache, tokens: &[usize], position: usize, null: Option<&[Kv]>) -> (Vec<f32>, GroupKv) {
        let (d, a, kvd, s) = (self.cfg.hidden, self.cfg.attn(), self.cfg.kv_dim(), tokens.len());
        let scale = attn_scale();
        let mut x: Vec<f32> = tokens.iter().flat_map(|&t| self.embed[t * d..][..d].iter().copied()).collect();
        let (c1, s1) = rope_at(position);
        let (cos, sin): (Vec<f32>, Vec<f32>) = ((0..s).flat_map(|_| c1).collect(), (0..s).flat_map(|_| s1).collect());
        let mut group = GroupKv { k: Vec::new(), v: Vec::new() };
        let mut cached_kv = cache.materialised();
        if let Some(n) = null {
            assert_eq!(n.len(), self.cfg.layers, "a null cell per layer");
            cached_kv.push(n[0]);
        }
        for (li, l) in self.layers.iter().enumerate() {
            if let Some(n) = null {
                *cached_kv.last_mut().unwrap() = n[li];
            }
            let h = self.tiles.all(self.tiles.norm(&x, d));
            let q = self.tiles.all(self.tiles.rope(self.tiles.mm(&h, &l.q, d, a), &cos, &sin, a));
            let k = self.tiles.all(self.tiles.rope(self.tiles.mm(&h, &l.k, d, kvd), &cos, &sin, kvd));
            let v = self.tiles.all(self.tiles.mm(&h, &l.v, d, kvd));
            let mut attn = vec![0f32; s * a];
            // The cached keys as every memory head of this layer reads them.
            let keys: Vec<[f32; HEAD_DIM]> = if cache.heads.iter().any(|h| h.0 == li) { cached_kv.iter().map(|kv| kv.0.map(|x| self.tiles.round(x))).collect() } else { Vec::new() };
            for head in 0..self.cfg.heads {
                let (base, kb) = (head * HEAD_DIM, self.cfg.kv_head(head) * HEAD_DIM);
                if (0..d).all(|ch| l.o[ch * a + base..][..HEAD_DIM].iter().all(|&w| w == 0.0)) {
                    continue;
                }
                let memory = cache.heads.contains(&(li, head));
                let value = |j: usize| cached_kv[j].1.map(|x| self.tiles.round(x));
                let work = s * (if memory { cached_kv.len() } else { 0 } + s) * HEAD_DIM;
                each_row(&mut attn, a, work, |i, out| {
                    let qi = &q[i * a + base..][..HEAD_DIM];
                    let dot = |kj: &[f32]| self.tiles.dot(qi, kj, scale);
                    let cached: Vec<f32> = if memory { keys.iter().map(|k| dot(k)).collect() } else { Vec::new() };
                    let own: Vec<f32> = (0..s).map(|j| dot(&k[j * kvd + kb..][..HEAD_DIM])).collect();
                    let m = cached.iter().chain(&own).copied().fold(f32::NEG_INFINITY, f32::max);
                    // With nothing cached there is nothing this head cannot see.
                    if !memory && !cached_kv.is_empty() {
                        assert!(m >= 104.0 || (0.0f32 - m).exp() == 0.0, "layer {li} head {head}: no exact match in the group (max logit {m}); the cached tokens' keys for this head are not held");
                    }
                    let ec: Vec<f32> = cached.iter().map(|z| self.tiles.exp(*z, m)).collect();
                    let eo: Vec<f32> = own.iter().map(|z| self.tiles.exp(*z, m)).collect();
                    let z = self.tiles.sum(ec.iter().chain(&eo).copied());
                    let mut terms: Vec<(f32, [f32; HEAD_DIM])> = ec.iter().enumerate().filter(|(_, w)| **w != 0.0).map(|(j, w)| (self.tiles.div(*w, z), value(j))).collect();
                    terms.extend(eo.iter().enumerate().map(|(j, w)| (self.tiles.div(*w, z), std::array::from_fn(|t| v[j * kvd + kb + t]))));
                    self.tiles.mix(&terms, &mut out[base..][..HEAD_DIM]);
                });
            }
            self.tiles.add(&mut x, &self.tiles.mm(&self.tiles.all(attn), &l.o, a, d));
            group.k.push(k);
            group.v.push(v);
            let h = self.tiles.all(self.tiles.norm(&x, d));
            let i = self.cfg.intermediate;
            let g = self.tiles.mm(&h, &l.gate, d, i);
            let u = self.tiles.mm(&h, &l.up, d, i);
            let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| self.tiles.act(*g, *u)).collect();
            self.tiles.add(&mut x, &self.tiles.mm(&self.tiles.all(act), &l.down, i, d));
        }
        (x, group)
    }
}

/// HF rotate_half convention: dim t pairs with t ± HEAD_DIM/2.
pub fn apply_rope(x: Vec<f32>, cos: &[f32], sin: &[f32], a: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut y = x.clone();
    for (p, row) in x.chunks(a).enumerate() {
        for h in 0..a / HEAD_DIM {
            for t in 0..HEAD_DIM {
                let rot = if t < half { -row[h * HEAD_DIM + t + half] } else { row[h * HEAD_DIM + t - half] };
                y[p * a + h * HEAD_DIM + t] = row[h * HEAD_DIM + t] * cos[p * HEAD_DIM + t] + rot * sin[p * HEAD_DIM + t];
            }
        }
    }
    y
}

/// A checkpoint that fills masked positions of a token sequence by argmax. The
/// protocol over the passes lives in [`crate::checkpoint::CheckpointDevice`].
pub trait Denoiser {
    fn cfg(&self) -> &Config;
    fn slots(&self) -> &[String];
    /// Writes the argmax tokens at `positions` into `out`, replacing what it held, so a
    /// caller that keeps `out` allocates nothing per pass.
    fn fill_into(&mut self, tokens: &[usize], positions: &[usize], out: &mut Vec<usize>);
    /// One pass over a device's sequence: afterwards its decoded store holds the argmax
    /// token at each of `positions`, and lists the positions the pass wrote
    /// (`Sequence::fresh`). A denoiser that keeps the sequence it placed last may place
    /// only what the sequence lists as changed, and write only what it decoded again.
    fn fill_seq(&mut self, seq: &mut Sequence, positions: &Positions) {
        seq.whole(positions, |tokens, out| self.fill_into(tokens, positions, out));
    }
    /// The argmax tokens at `positions`, in a new `Vec`.
    fn fill(&mut self, tokens: &[usize], positions: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(positions.len());
        self.fill_into(tokens, positions, &mut out);
        out
    }
}

impl Denoiser for Model {
    fn cfg(&self) -> &Config {
        &self.cfg
    }
    fn slots(&self) -> &[String] {
        &self.slots
    }
    fn fill_into(&mut self, tokens: &[usize], positions: &[usize], out: &mut Vec<usize>) {
        let all = self.forward(tokens, false);
        out.clear();
        out.extend(positions.iter().map(|&p| all[p]));
    }
}

/// A denoiser whose cache holds one memory cell per 8-byte line.
pub trait MemoryDenoiser: Denoiser {
    /// Keeps writes `writes` (0 and/or 1, in order) of the finished step `tokens` as
    /// write records. Each record is the line its `WLw` tokens spell, at the address its
    /// `WPw` tokens spell, and replaces that line's cell.
    fn keep(&mut self, tokens: &[usize], writes: &[u8]);
    /// Keeps a byte the host serves as a write record at `addr` in the HOST block: an
    /// opcode byte fetched from an address the host serves. Labelled host input, which
    /// the audit holds to the bus through the bytes the model then fetches.
    fn keep_host(&mut self, addr: u32, byte: u8);
    /// The byte the memory holds at `addr`: a constant cell's byte the host copies into a
    /// HOST record when the model fetches from it.
    fn cell(&mut self, addr: u32) -> u8;
    /// With the step oracle on, the next fast pass is checked whatever its cadence (an OAM
    /// rewrite pass, rare enough that a cadence could miss every one).
    fn check_next(&mut self) {}
    /// Loads a bus image (SingleStepTests) into the memory, when the denoiser holds one.
    fn load_bus(&mut self, _ram: &[(u16, u8)]) -> Result<(), String> {
        Err("this denoiser holds no bus image".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tok::Bytes` reads every token as `tok::byte` does, families or none.
    #[test]
    fn byte_ranges_read_every_token_as_byte_does() {
        for (n_inputs, n_outputs, families) in [(20, 27, 0), (20, 27, 3), (65, 65, 0), (7, 40, 5)] {
            let c = Config { hidden: 0, intermediate: 0, layers: 0, heads: 0, kv_heads: 0, vocab: 0, n_inputs, n_outputs, n_const: 0, families };
            let b = tok::Bytes::new(&c);
            for t in 0..tok::rec(&c) + tok::SPECIAL + 300 {
                assert_eq!(b.of(t), tok::byte(&c, t), "token {t} of {n_inputs}, {n_outputs}, {families}");
            }
        }
    }

    #[test]
    fn f16_rounds_to_the_nearest_half_ties_to_even() {
        for (x, want) in [(1.0, 1.0), (2049.0, 2048.0), (2051.0, 2052.0), (65504.0, 65504.0), (65519.0, 65504.0), (65520.0, f32::INFINITY), (1.0 / 128.0, 1.0 / 128.0), (2.9e-8, 0.0), (3e-8, 5.9604645e-8), (6.0e-8, 5.9604645e-8), (1.0 + 1.0 / 4096.0, 1.0), (-0.1, -0.099975586)] {
            assert_eq!(f16(x), want, "f16({x})");
        }
        assert_eq!(f16(2f32.powi(-15)), 2f32.powi(-15));
        assert_eq!(f16(1048576.0), f32::INFINITY);
    }

    /// The precomputed denominators give the same f64 quotient as the power formed per
    /// call, so the tables are bit-identical to the literal HF formula.
    #[test]
    fn rope_tables_equal_the_literal_formula() {
        let half = HEAD_DIM / 2;
        for p in [0usize, 1, 2, 29_444, 1_131_049, 1_160_493, 2_208_803, 4_400_000] {
            let (cos, sin) = rope_at(p);
            for t in 0..HEAD_DIM {
                let ang = p as f64 / ROPE_THETA.powf(2.0 * (t % half) as f64 / HEAD_DIM as f64);
                assert_eq!(cos[t].to_bits(), (ang.cos() as f32).to_bits(), "cos at position {p}, dim {t}");
                assert_eq!(sin[t].to_bits(), (ang.sin() as f32).to_bits(), "sin at position {p}, dim {t}");
            }
        }
    }

    /// A cell holds the formula's key at its line index and the line's bits as its
    /// value; setting the line again replaces it.
    #[test]
    fn cells_are_keyed_at_their_address() {
        let mut c = KvCache::new(vec![(1, 0)]);
        let (a, b) = ([0xA5; LINE], [1, 2, 3, 4, 5, 6, 7, 0x5A]);
        c.set(0x4_4001, a);
        c.set(0x10, [0x3C; LINE]);
        c.set(0x4_4001, b);
        assert_eq!((c.len(), c.get(0x4_4001), c.get(0x10)), (2, Some(b), Some([0x3C; LINE])));
        assert_eq!(c.kv(1).0.map(f32::to_bits), key(Kind::Cell, 0x4_4001, 0x4_4001).map(f32::to_bits));
        assert_eq!(line_of_value(&c.kv(1).1), b);
    }
}
