//! One Llama checkpoint for the whole Game Boy (`docs/one-model.md`). The five
//! circuits (the GBC CPU with its memory, the APU, the control logic, the scanline PPU and its mode-3 length)
//! are placed side by side in one circuit (`Circuit::concat`) and compiled once: one
//! config, one set of weights, one vocabulary, one forward pass. Each part runs as its
//! own stream over those weights: its own sequence, with every slot of the other parts
//! held at a fixed token, and its own engine state (`Engine::streams`). The CPU
//! stream owns the memory's KV cache; the PPU stream reads VRAM and OAM as of each
//! line through the records the CPU stream keeps, as before.

use crate::circuit::{Circuit, OutKind};
use crate::engine::{Engine, StreamSpec, BUCKET_LAYERS};
use crate::level::Over;
use crate::llama::{Fam, Model};
use crate::memory::attn::Params;

/// A part of the checkpoint, in its slot order: [`parts`], the specs [`circuit`] gives
/// and the engines `Engine::streams` builds from them are indexed by `Part as usize`.
/// The PPU comes first: its position wires name output positions from the first output
/// slot on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Ppu,
    Cpu,
    Apu,
    Control,
    /// Each line's mode-3 length on its own and the judge of each display write to it
    /// (`ppu_circuit::circuit::build_length`).
    Length,
    /// The pixel FIFO's step, which draws the lines the judge marks
    /// (`ppu_circuit::stepper`).
    Stepper,
}

impl Part {
    pub const ALL: [Part; 6] = [Part::Ppu, Part::Cpu, Part::Apu, Part::Control, Part::Length, Part::Stepper];

    /// The prefix of the part's unit names in the merged circuit.
    pub fn name(self) -> &'static str {
        match self {
            Part::Ppu => "ppu",
            Part::Cpu => "cpu",
            Part::Apu => "apu",
            Part::Control => "ctl",
            Part::Length => "len",
            Part::Stepper => "stp",
        }
    }

    /// The part's own circuit, from its own builder.
    fn build(self) -> Circuit {
        match self {
            Part::Ppu => crate::ppu_circuit::circuit::build(),
            Part::Cpu => crate::sm83::build_mem(),
            Part::Apu => crate::apu_circuit::build(),
            Part::Control => crate::control_circuit::build(),
            Part::Length => crate::ppu_circuit::circuit::build_length(),
            Part::Stepper => crate::ppu_circuit::stepper::build(),
        }
    }

    /// Whether the merged circuit keeps an output of this part: the PPU's, the length's and
    /// the stepper's write-record slots, which they leave zero, are left out, so the CPU's
    /// are the checkpoint's.
    fn keeps(self, kind: OutKind) -> bool {
        !matches!(self, Part::Ppu | Part::Length | Part::Stepper) || kind == OutKind::Shared
    }
}

/// The six circuits, in [`Part::ALL`] order.
pub fn parts() -> [Circuit; 6] {
    Part::ALL.map(Part::build)
}

/// Where the PPU's three parts sit (`level`). The GBC CPU, the APU and the control logic
/// keep the levels their own builds gave them.
pub struct Placement {
    /// The parts re-levelled, each with the depth it ends at.
    pub free: [(Part, u32); 3],
    /// The MLP units a layer may hold.
    pub units: usize,
    /// The memory heads a layer may hold besides the copy heads, which the compiler
    /// then gives the two emptiest layers (`llama::copy_layers`).
    pub heads: usize,
    /// The wires a re-levelled part keeps live where it can.
    pub live: usize,
}

/// The scanline PPU and its length stop at the most levels a stream runs with the
/// engine's reader buckets (`engine::BUCKET_LAYERS`); the stepper takes all 24, the
/// APU's depth. `ROOM_CAPS=3336:3448:8 ROOM_LIVES=1371:1395:2 examples/room -- --scan 20 24 20`
/// lists the placements at those depths by cost under the head cap, 6 of them, all at width
/// 3,456 and hidden 1,920; this one is round 2's (`docs/perf-log.md`, "Re-levelling, round
/// 2" and "Round 13"). The scanline PPU keeps 1,377 wires live, which with the 526
/// channels the parts share is 1,903 of hidden's 1,920.
pub const PLACEMENT: Placement = Placement { free: [(Part::Ppu, BUCKET_LAYERS), (Part::Stepper, 24), (Part::Length, BUCKET_LAYERS)], units: 3480, heads: HEADS, live: 1371 };

/// The head cap of `PLACEMENT`. The parts share the input slots, so layer 0 holds fewer
/// gathers than this, and the cap is what the packer can hold at the placement's width
/// and hidden: `examples/room -- --try 3480 <heads> ppu=20,stp=24,len=20 1371` ends
/// within its caps at 26 and 2 heads over at 25 (`docs/perf-log.md`, "Round 13").
pub const HEADS: usize = 26;

/// The GBC CPU's slice of the one model is this wide, the checkpoint's own hidden: the
/// norm's bound sets it (`Model::slice` widens the slice past its channels until the
/// bound holds, and asserts it at the checkpoint's full width). A placement whose
/// checkpoint is narrower is not one, which `examples/room -- --scan` leaves out;
/// `tests/one_model.rs` holds the constant to the compiled slice.
pub const CPU_SLICE_HIDDEN: usize = 1920;

/// The merged circuit and each part's stream spec in [`Part::ALL`] order. A stock
/// runtime's file gives every layer the widest layer's MLP and heads, and a layer's
/// units and heads are the sum over the parts placed there, so the PPU's three parts
/// are spread over the layers the others leave empty (`PLACEMENT`).
pub fn circuit(parts: &[Circuit; 6]) -> (Circuit, Vec<StreamSpec>) {
    placed(parts, &PLACEMENT)
}

/// The merged circuit with the PPU's parts where `placement` puts them. It refuses a
/// placement that ends over its caps, and one whose checkpoint is not `CPU_SLICE_HIDDEN`
/// wide. `PLACEMENT` leaves 17 wires free in hidden, so a circuit change that keeps more
/// wires live than that stops here by name, where it would otherwise compile a wider
/// checkpoint.
pub fn placed(parts: &[Circuit; 6], placement: &Placement) -> (Circuit, Vec<StreamSpec>) {
    let (mut c, specs) = concat(parts);
    let over = level(&mut c, &placement.free, placement.units, placement.heads, placement.live).expect("a part does not fit its depth");
    // The canonical placement ends within its caps by the packer's own count, so no
    // layer past layer 0 holds more heads than the cap. The compiled widths and heads are
    // `Model::census`'s, pinned in `tests/one_model.rs`.
    assert_eq!(over, Over { units: 0, heads: 0 }, "the placement ends over its caps of {} units and {} heads a layer", placement.units, placement.heads);
    assert!(Model::layer0_heads(&c, true) <= placement.heads, "layer 0 holds {} gathers, more than the cap of {} heads", Model::layer0_heads(&c, true), placement.heads);
    let cfg = Model::census(&c, true).cfg;
    // The packer's cap leaves out the copy heads, so the compiled count is held here too.
    assert_eq!(cfg.heads, placement.heads, "the placement compiles {} heads a layer, not its cap of {}", cfg.heads, placement.heads);
    let hidden = cfg.hidden;
    assert_eq!(hidden, CPU_SLICE_HIDDEN, "the placement compiles a checkpoint of hidden {hidden}, not {CPU_SLICE_HIDDEN}: the parts keep more wires live than the placement has channels for, so place them again (`examples/room -- --scan`) and move `PLACEMENT`, the digests and the pinned shape on purpose");
    (c, specs)
}

/// Re-levels the parts `free` of the merged circuit, each to the depth given with it, in
/// the room the others leave of `cap` MLP units and `heads` heads a layer, keeping at
/// most `live` of a free part's wires live where it can (`level::level`). The result is
/// how far the fullest layer ended over the caps; None, with the circuit unchanged, when
/// a depth is too few for its part.
pub fn level(c: &mut Circuit, free: &[(Part, u32)], cap: usize, heads: usize, live: usize) -> Option<Over> {
    let clears: Vec<usize> = Model::census(c, true).clears.iter().map(|p| p.iter().sum()).collect();
    let free: Vec<(usize, u32)> = free.iter().map(|&(p, depth)| (p as usize, depth)).collect();
    // The routing units, counted as the compiler writes them: a free part's at the
    // depth it will end at, the others' at their parts' depths, with the reader
    // tokens' at the record part's.
    let mut spoken = vec![0; c.depth() as usize + 1];
    for j in 0..c.outputs.len() {
        let at = free.iter().find(|f| f.0 == c.out_part(j)).map_or(c.out_depth(j), |f| f.1);
        spoken[at as usize] += crate::llama::routed_units(c, j);
    }
    spoken[c.part_depth(Part::Cpu as usize) as usize] += crate::llama::reader_units();
    crate::level::level(c, &free, &spoken, &clears, cap, heads, live)
}

/// The six circuits side by side, each with the levels its own build gave it.
pub fn concat(parts: &[Circuit; 6]) -> (Circuit, Vec<StreamSpec>) {
    let named: Vec<(&str, &Circuit)> = Part::ALL.iter().map(|p| p.name()).zip(parts.iter()).collect();
    // The overlay (`Circuit::overlay_channels`) gives each part's wires the channels its own plan does.
    let (c, _) = Circuit::concat(&named, |pi, kind| Part::ALL[pi].keeps(kind));
    // A part's inputs sit at the input slots it holds, which the parts share (`Circuit::parts`).
    let mut out_at = c.inputs.len();
    let specs = Part::ALL
        .iter()
        .zip(parts)
        .map(|(&part, p)| {
            let mut slots: Vec<String> = p.inputs.iter().map(|(n, _)| n.clone()).collect();
            let mut at: Vec<Option<usize>> = c.parts[part as usize].iter().copied().map(Some).collect();
            let mut memory = false;
            for ((n, _), &kind) in p.outputs.iter().zip(&p.out_kinds) {
                if part.keeps(kind) {
                    memory |= kind != OutKind::Shared;
                    slots.push(n.clone());
                    at.push(Some(out_at));
                    out_at += 1;
                }
            }
            StreamSpec { slots, at, n_inputs: p.inputs.len(), families: if memory { Fam::ALL.len() } else { 0 }, part: part as usize }
        })
        .collect();
    (c, specs)
}

/// The checkpoint's engines for the parts asked for, one per part's stream, each
/// running its own slice of the one set of weights (`Model::stream`).
pub struct Streams {
    pub cpu: Option<Engine>,
    pub apu: Option<Engine>,
    pub control: Option<Engine>,
    pub ppu: Option<Engine>,
    pub length: Option<Engine>,
    pub stepper: Option<Engine>,
}

/// Compiles the one checkpoint and builds the streams of `want`, each reading its
/// slice of the weights, and only that, from `GBCFORMER_GGUF` when set
/// (`gguf::slice_from_env`).
pub fn streams(want: &[Part]) -> Streams {
    // `GBCFORMER_GGUF_ONLY=<model.gguf>`: no compiler at all; the configuration, layout,
    // ranges and streams come from the file's metadata and every weight from its tensors.
    let (model, specs, from_file) = match std::env::var_os("GBCFORMER_GGUF_ONLY") {
        Some(path) => {
            let p = std::path::Path::new(&path);
            let (manifest, _) = crate::wiring::Manifest::read(p).unwrap_or_else(|e| panic!("GBCFORMER_GGUF_ONLY: {e}"));
            assert_eq!(manifest.wiring_sha1, crate::gbc_wiring::wiring_sha1(), "GBCFORMER_GGUF_ONLY: {} was written with another wiring than this build's; export it again", p.display());
            let (m, specs, loaded) = crate::gguf::open_model(p).unwrap_or_else(|e| panic!("GBCFORMER_GGUF_ONLY: {e}"));
            assert_eq!(m.cfg.vocab, manifest.vocab, "GBCFORMER_GGUF_ONLY: the manifest's vocabulary");
            eprintln!("one model from {} alone: {} tensors, {} values, no circuit compiled", std::path::Path::new(&path).display(), loaded.tensors, loaded.values);
            (m, specs, true)
        }
        None => {
            let (c, specs) = circuit(&parts());
            (Model::compile_mem(&c, Params::memory()), specs, false)
        }
    };
    let (keep, d) = (crate::gguf::keep(&model, 1), model.cfg.hidden);
    let mut specs: Vec<Option<StreamSpec>> = specs.into_iter().map(Some).collect();
    let chosen: Vec<StreamSpec> = want.iter().map(|&p| specs[p as usize].take().expect("each part once")).collect();
    let load = |m: &mut Model, s: &crate::llama::Slice| {
        if !from_file {
            crate::gguf::slice_from_env(m, s, &keep, d);
        }
    };
    let mut engines: Vec<Option<Engine>> = Engine::streams_with(model, &chosen, &load).into_iter().map(Some).collect();
    let mut take = |p: Part| want.iter().position(|&q| q == p).and_then(|i| engines[i].take());
    Streams { cpu: take(Part::Cpu), apu: take(Part::Apu), control: take(Part::Control), ppu: take(Part::Ppu), length: take(Part::Length), stepper: take(Part::Stepper) }
}

/// One layer-0 head as a stream's pass sees it. It holds the positions at its largest
/// logit in the whole sequence (its targets, each weighing 1/len), and the ones at its
/// largest logit among the tokens the stream's group keeps (with `cell`, the cache's
/// cells too).
pub struct Gather {
    pub head: usize,
    pub targets: Vec<usize>,
    pub attended: Vec<usize>,
    pub cell: bool,
}

/// A stream's reduced group, the checkpoint positions a stock runtime's pass decodes in
/// place of the whole sequence, with every kept row bit for bit the whole sequence's.
pub struct Group {
    /// The positions kept, ascending.
    pub keep: Vec<usize>,
    pub gathers: Vec<Gather>,
    /// The smallest logit gap, in the whole sequence or the group, between the tokens a
    /// gather attends to and any other token, cells included.
    pub gap: f32,
}

/// The logit gap past which a softmax weight is exactly 0 in f32 (`exp(−104) < 2^-149`)
/// and in f16.
const ZERO_WEIGHT_GAP: f32 = 104.0;

/// The group a part's stream decodes. It holds the stream's own positions, and the
/// other parts' positions a layer-0 head needs to write what it writes over the whole
/// sequence. Other parts' slots hold byte 0 or a mask in the stream's view.
///
/// Why each kept row is unchanged, read from the weights:
/// - Every head's query reads only channels equal in every embedding row, so it is one
///   vector at every position; no key reads a channel that differs between the tokens
///   of one input slot, so own inputs score alike whatever their bytes. All group tokens
///   share one position, and a cell's layer-0 key is zero, so RoPE changes no score.
/// - A head attends to the tokens at its largest logit, each alike, and every other
///   token scores at least `ZERO_WEIGHT_GAP` below and weighs exactly 0. The group keeps
///   a head's targets whole, or attends to tokens whose values are all zero where the
///   targets' values are all zero too, since both write 0. Otherwise it keeps one target when
///   all of them are other parts' zero-valued slots, and all of them when not.
/// - Later layers hold only memory heads, where every group token loses to the cell
///   its query names by `Params`' gap (`tests/stream_audit.rs`: one cell per head).
pub fn group(m: &Model, specs: &[StreamSpec], part: Part) -> Group {
    use crate::llama::{attn_scale, rms_norm, tok, HEAD_DIM};
    let c = &m.cfg;
    let (d, a, seq) = (c.hidden, c.attn(), c.seq());
    let l0 = &m.layers[0];
    let row = |t: usize| &m.embed[t * d..][..d];
    let mut own = vec![false; seq];
    for q in specs[part as usize].at.iter().flatten() {
        own[*q] = true;
    }
    let mut varies = vec![false; d];
    for r in m.embed.chunks(d).skip(1) {
        for (ch, (x, y)) in r.iter().zip(row(0)).enumerate() {
            varies[ch] |= x != y;
        }
    }
    assert!(l0.q.chunks(d).all(|r| r.iter().enumerate().all(|(ch, &w)| w == 0.0 || !varies[ch])), "a layer-0 query reads a channel some token changes");
    // The tokens a stream can place at each position: every token at its own slots, and
    // byte 0 or the mask at another part's.
    let at = |p: usize| -> Vec<usize> {
        match (p < c.n_inputs, own[p]) {
            (true, true) => (0..=255u8).map(|v| tok::input(c, p, v)).chain([tok::input_mask(c, p)]).collect(),
            (true, false) => vec![tok::input(c, p, 0)],
            (false, true) => [tok::output_mask(c, p)].into_iter().chain((0..=255u8).map(|v| tok::output(c, v))).collect(),
            (false, false) => vec![tok::output_mask(c, p)],
        }
    };
    // No layer-0 key may read a channel that differs between the tokens one input slot holds, or
    // between the output byte tokens.
    let mut byte_ch = vec![false; d];
    let mut differ = |ts: &mut dyn Iterator<Item = usize>| {
        let base = row(ts.next().unwrap());
        for t in ts {
            for (ch, (x, y)) in row(t).iter().zip(base).enumerate() {
                byte_ch[ch] |= x != y;
            }
        }
    };
    for j in 0..c.n_inputs {
        differ(&mut (0..=255u8).map(|v| tok::input(c, j, v)).chain([tok::input_mask(c, j)]));
    }
    differ(&mut (0..=255u8).map(|v| tok::output(c, v)));
    assert!(l0.k.chunks(d).all(|r| r.iter().enumerate().all(|(ch, &w)| w == 0.0 || !byte_ch[ch])), "a layer-0 key reads a byte");
    let project = |w: &[f32], t: usize| -> Vec<f32> {
        let h = rms_norm(row(t), d);
        w.chunks(d).map(|r| r.iter().zip(&h).map(|(x, y)| x * y).sum()).collect()
    };
    let token = |p: usize| if p < c.n_inputs { tok::input(c, p, 0) } else { tok::output_mask(c, p) };
    let keys: Vec<Vec<f32>> = (0..seq).map(|p| project(&l0.k, token(p))).collect();
    let values: Vec<Vec<Vec<f32>>> = (0..seq).map(|p| at(p).into_iter().map(|t| project(&l0.v, t)).collect()).collect();
    let decoded_key = project(&l0.k, tok::output(c, 0));
    let (cell_key, cell_value) = (project(&l0.k, tok::cell(c)), project(&l0.v, tok::cell(c)));
    assert!(cell_key.iter().all(|&x| x == 0.0), "a cell's layer-0 key is zero");
    let query = project(&l0.q, token(0));
    let heads: Vec<usize> = (0..c.heads).filter(|h| (0..d).any(|ch| l0.o[ch * a + h * HEAD_DIM..][..HEAD_DIM].iter().any(|&w| w != 0.0))).collect();
    // Per head: every position's logit, the cells', and which values are zero.
    let attn = |h: usize| {
        let kv = c.kv_head(h) * HEAD_DIM;
        let q = &query[h * HEAD_DIM..][..HEAD_DIM];
        let score = |k: &[f32]| q.iter().zip(&k[kv..kv + HEAD_DIM]).map(|(x, y)| x * y).sum::<f32>() * attn_scale();
        let zero = |v: &[f32]| v[kv..kv + HEAD_DIM].iter().all(|&x| x == 0.0);
        (keys.iter().map(|k| score(k)).collect::<Vec<f32>>(), score(&cell_key), values.iter().map(|vs| vs.iter().all(|v| zero(v))).collect::<Vec<bool>>(), zero(&cell_value), score(&decoded_key))
    };
    let attended = |scores: &[f32], cell: f32, kept: &dyn Fn(usize) -> bool| {
        let top = (0..seq).filter(|&p| kept(p)).map(|p| scores[p]).fold(cell, f32::max);
        let at: Vec<usize> = (0..seq).filter(|&p| kept(p) && scores[p] == top).collect();
        let next = (0..seq).filter(|&p| kept(p) && scores[p] != top).map(|p| scores[p]).chain((cell != top).then_some(cell)).fold(f32::MIN, f32::max);
        (at, cell == top, top - next)
    };
    let mut keep = own.clone();
    loop {
        let mut added = false;
        for &h in &heads {
            let (scores, cell, zero, cell_zero, _) = attn(h);
            let (targets, _, _) = attended(&scores, cell, &|_| true);
            let (at, at_cell, _) = attended(&scores, cell, &|p| keep[p]);
            let same = targets.iter().all(|&p| keep[p]);
            let both_zero = targets.iter().all(|&p| zero[p]) && at.iter().all(|&p| zero[p]) && (!at_cell || cell_zero);
            if same || both_zero {
                continue;
            }
            let rep = targets.iter().all(|&p| !own[p] && zero[p]);
            for &p in if rep { &targets[..1] } else { &targets[..] } {
                keep[p] = true;
            }
            added = true;
        }
        if !added {
            break;
        }
    }
    let (mut gathers, mut gap) = (Vec::new(), f32::INFINITY);
    for &h in &heads {
        let (scores, cell, _, _, decoded) = attn(h);
        let (targets, target_cell, full_gap) = attended(&scores, cell, &|_| true);
        let (at, at_cell, kept_gap) = attended(&scores, cell, &|p| keep[p]);
        assert!(!target_cell, "{part:?}: layer-0 head {h} attends to the cells");
        // A decoded output byte (pass 2's request slots) must weigh 0 wherever it sits.
        let top = at.first().map_or(cell, |&p| scores[p]);
        for g in [full_gap, kept_gap, top - decoded] {
            assert!(g >= ZERO_WEIGHT_GAP, "{part:?}: layer-0 head {h}'s attended tokens lead by {g}");
            gap = gap.min(g);
        }
        gathers.push(Gather { head: h, targets, attended: at, cell: at_cell });
    }
    Group { keep: (0..seq).filter(|&p| keep[p]).collect(), gathers, gap }
}
