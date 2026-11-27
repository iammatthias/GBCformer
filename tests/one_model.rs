//! The one checkpoint (`docs/one-model.md`): every stream presents its part's own
//! slots, runs its own slice of the weights with a pruned fast plan, the APU stream
//! decodes what the APU's own checkpoint decodes, and a slice computes exactly what
//! the whole checkpoint computes at the part's positions.

use gbcformer::circuit::Circuit;
use gbcformer::engine::{Engine, StreamSpec};
use gbcformer::llama::{tok, Denoiser, KvCache, Model, HEAD_DIM};
use gbcformer::memory::attn::{Params, READ_AT};
use gbcformer::memory::image::Image;
use gbcformer::one_model::{circuit, parts, placed, Part, Placement, PLACEMENT};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::OnceLock;

fn pass(e: &mut Engine, rng: &mut u64) -> Vec<usize> {
    let cfg = e.cfg().clone();
    let tokens: Vec<usize> = (0..cfg.seq())
        .map(|p| {
            let byte = xorshift(rng) as u8;
            if p < cfg.n_inputs { tok::input(&cfg, p, byte) } else { tok::output_mask(&cfg, p) }
        })
        .collect();
    e.fill(&tokens, &(cfg.n_inputs..cfg.seq()).collect::<Vec<_>>())
}

/// A control pass's input bytes in `control_circuit::INPUTS` order: the state of a
/// reference `Control` on a random console after random register writes, and a read of a random
/// control register over no M-cycle.
fn control_inputs(rng: &mut u64) -> Vec<u8> {
    use gbcformer::control_circuit::{encode_state, INPUTS};
    use gbcformer::machine::control::{Control, ControlDevice};
    let mut c = Control::post_boot(xorshift(rng) & 1 == 0, 0x80);
    for addr in [0xFF05, 0xFF06, 0xFF07, 0xFF0F, 0xFFFF, 0xFF41, 0xFF45, 0xFF72, 0xFF75] {
        c.write(addr, xorshift(rng) as u8, &mut gbcformer::device::Signals::default());
    }
    let state = encode_state(&c.state());
    let mut bytes = vec![0u8; INPUTS.len()];
    bytes[..state.len()].copy_from_slice(&state);
    let at = |n: &str| INPUTS.iter().position(|&x| x == n).expect("an input");
    bytes[at("EV")] = 2;
    bytes[at("EA")] = [0x00, 0x04, 0x0F, 0x41, 0x44, 0x45, 0xFF][xorshift(rng) as usize % 7];
    bytes[at("SI")] = 0xFF;
    bytes
}

fn xorshift(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

#[test]
fn every_stream_is_its_part_on_its_slice_of_the_weights() {
    let s = gbcformer::one_model::streams(&Part::ALL);
    let (sc, mut sa, sctl, sp) = (s.cpu.unwrap(), s.apu.unwrap(), s.control.unwrap(), s.ppu.unwrap());
    let mut apu = gbcformer::apu_circuit::engine();
    let names = |e: &Engine| e.slots().to_vec();
    assert_eq!(names(&sa), names(&apu), "the APU stream presents the APU's slots");
    assert_eq!(names(&sctl), names(&gbcformer::control_circuit::engine()));
    let cpu = gbcformer::sm83::build_mem();
    assert!(names(&sc).iter().eq(cpu.inputs.iter().map(|x| &x.0).chain(cpu.outputs.iter().map(|x| &x.0))), "the CPU stream presents the memory checkpoint's slots");
    assert_eq!(names(&sp).len(), 3 + 160 + 2 + 1, "the PPU stream leaves out the record slots a scanline never writes: 160 pixels, the window's two outputs and the mode-3 length");
    // Each stream stops at its own depth, plus the routing layer: the GBC CPU's 10
    // levels, the APU 24 (SameBoy's rules, docs/apu-in-weights.md §15), control 16 (the
    // HBlank request's edge and HDMA5's count at a step's end, docs/control-in-weights.md
    // §26, added one), and the PPU the 20 its placement gives it (13 on its own,
    // `one_model::PLACEMENT`).
    let layers: Vec<usize> = [&sc, &sa, &sctl, &sp].iter().map(|e| e.dense().cfg.layers).collect();
    assert_eq!(layers, vec![11, 25, 17, 21]);
    // Every stream must keep its integer fast path, because a slice narrower than its
    // norm bound needs (`Model::slice`) would run every pass dense.
    for (name, e) in [("cpu", &sc), ("apu", &sa), ("control", &sctl), ("ppu", &sp)] {
        assert!(e.exact().is_ok(), "the {name} stream's fast plan is refused: {:?}", e.exact().err());
    }
    let (cpu, apu_plan, ctl, ppu) = (sc.plan_size().unwrap(), sa.plan_size().unwrap(), sctl.plan_size().unwrap(), sp.plan_size().unwrap());
    assert_eq!((cpu.1, apu_plan.1, ctl.1, ppu.1), (55, 0, 0, 61), "memory heads per stream: the CPU's (its column heads, docs/hardware-shaped.md §20), the two copy heads and the OAM bug's four row heads, none, none, the PPU's line heads (the display registers, OAM, the objects' VRAM rows, the positional fetch, palette and compatibility-palette heads, and the two tile-0 rows the CGB's WX glitch reads)");
    assert_eq!(ppu.1 + 2, gbcformer::ppu_circuit::engine().plan_size().unwrap().1, "the PPU stream reads with the PPU checkpoint's own heads but the two copy heads, which the CPU stream holds");
    assert_eq!((cpu.2, apu_plan.2, ctl.2, ppu.2), (false, false, false, true), "only the PPU stream runs the positional stage");
    // The length stream: the length checkpoint's slots and heads but the two copy heads,
    // on its integer fast path, with no positional stage.
    let sl = s.length.expect("the length stream");
    let own = gbcformer::ppu_circuit::length_engine();
    assert_eq!(names(&sl), ["MLY", "MFLG", "MWH", "MWI", "MWV", "MML", "MWAR'", "MJ", "MWT0", "MWT1"], "the length stream presents its inputs and outputs (the judge's among them), not the record slots it leaves zero");
    assert!(sl.exact().is_ok(), "the length stream's fast plan is refused: {:?}", sl.exact().err());
    let (len_plan, own_plan) = (sl.plan_size().unwrap(), own.plan_size().unwrap());
    assert_eq!((len_plan.1 + 2, len_plan.2), (own_plan.1, false), "the length stream reads with its own heads but the two copy heads, and runs no positional stage");
    // The stepper stream: the stepper checkpoint's slots but the record slots, its heads but
    // the two copy heads, on its integer fast path.
    let st = s.stepper.expect("the stepper stream");
    let own = gbcformer::ppu_circuit::stepper::engine();
    let want: Vec<String> = own.slots().iter().filter(|n| !n.starts_with("WP") && !n.starts_with("WL")).cloned().collect();
    assert_eq!(names(&st), want, "the stepper stream presents its inputs and outputs, not the record slots it leaves zero");
    assert!(st.exact().is_ok(), "the stepper stream's fast plan is refused: {:?}", st.exact().err());
    let (st_plan, own_plan) = (st.plan_size().unwrap(), own.plan_size().unwrap());
    assert_eq!((st_plan.1 + 2, st_plan.2), (own_plan.1, false), "the stepper stream reads with its own heads but the two copy heads, and runs no positional stage");
    let (mut r1, mut r2) = (0x2545_F491_4F6C_DD1Du64, 0x2545_F491_4F6C_DD1Du64);
    for k in 0..500 {
        assert_eq!(pass(&mut sa, &mut r1), pass(&mut apu, &mut r2), "segment {k}");
    }
    assert_eq!((sa.stats.fast_passes, sa.stats.fallback_passes), (500, 0));
}

/// Gate 2 for the one checkpoint, and the truncation proof's check. At a part's output
/// positions, the whole checkpoint's dense pass and the part's slice (`Model::stream`)
/// give bit-identical residuals on the slice's channels, 0 on every channel above them,
/// and the same tokens. The comparison skips the mode and null channels, which the slice
/// clears and none of its rows read. The APU and control streams' fast passes then
/// decode those tokens. Eight random passes per stream take about two minutes (133 s at
/// load 20 to 26), most of it the whole checkpoint's dense passes.
#[test]
fn every_slice_is_the_checkpoint_at_its_positions() {
    const PASSES: usize = 8;
    let Whole { c, specs, model: full, .. } = whole();
    // A step's copy heads read one cell, line 0, because its record slots hold masks and
    // their gathered address is 0. The whole checkpoint adds the null token.
    let mut cache = KvCache::new(full.memory.as_ref().unwrap().heads.clone());
    cache.set(0, [0; 8]);
    let cfg = full.cfg.clone();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let tested = [Part::Apu, Part::Control];
    let dup = |s: &gbcformer::engine::StreamSpec| gbcformer::engine::StreamSpec { slots: s.slots.clone(), at: s.at.clone(), n_inputs: s.n_inputs, families: s.families, part: s.part };
    let engines = Engine::streams(Model::compile_mem(c, Params::memory()), &tested.map(|p| dup(&specs[p as usize])));
    for (part, mut fast) in tested.into_iter().zip(engines) {
        let spec = &specs[part as usize];
        let slice = full.stream(part as usize, true);
        let mut own_cache = KvCache::new(slice.memory.as_ref().unwrap().heads.clone());
        own_cache.set(0, [0; 8]);
        let h = slice.cfg.hidden;
        // The mode and null channels, which the slice clears from its embedding and none
        // of its rows read.
        let cleared = full.slice(part as usize, true).cleared;
        let outputs: Vec<usize> = spec.at[spec.n_inputs..].iter().flatten().copied().collect();
        for k in 0..PASSES {
            let mut tokens: Vec<usize> = (0..cfg.seq()).map(|q| if q < cfg.n_inputs { tok::input(&cfg, q, 0) } else { tok::output_mask(&cfg, q) }).collect();
            let mut own = Vec::new();
            // Control's state and event bytes must be ones the reference reaches: random
            // bytes break the exclusive cases its selects declare (TIMA's reload, the
            // HDMA moves), and the engine then refuses the pass. Its state is a reference
            // `Control` after random writes, and its event a read with no M-cycle.
            let legal = (part == Part::Control).then(|| control_inputs(&mut rng));
            for (j, q) in spec.at[..spec.n_inputs].iter().enumerate() {
                let byte = legal.as_ref().map_or_else(|| xorshift(&mut rng) as u8, |b| b[j]);
                tokens[q.unwrap()] = tok::input(&cfg, q.unwrap(), byte);
                own.push(tok::input(fast.cfg(), j, byte));
            }
            let (a, _) = full.residual_cached(&cache, &tokens, READ_AT);
            let (b, _) = slice.residual_cached(&own_cache, &tokens, READ_AT);
            for &q in &outputs {
                let (x, y) = (&a[q * cfg.hidden..][..cfg.hidden], &b[q * h..][..h]);
                let diff: Vec<(usize, f32, f32)> = x[..h].iter().zip(y).enumerate().filter(|&(c, (x, y))| x.to_bits() != y.to_bits() && !cleared.contains(&c)).map(|(c, (x, y))| (c, *x / gbcformer::llama::UNIT, *y / gbcformer::llama::UNIT)).take(12).collect();
                assert!(diff.is_empty(), "{part:?} pass {k} position {q}: the slice's residual differs (wire base {}, mode {:?}): {diff:?}", slice.layout.wire_base(), (0..4).map(|p| slice.layout.mode(p)).collect::<Vec<_>>());
                assert!(x[h..].iter().all(|&v| v == 0.0), "{part:?} pass {k} position {q}: a channel above the prefix is not 0");
            }
            let (ta, tb) = (full.tokens_of(&a), slice.tokens_of(&b));
            let pick = |t: &[usize]| outputs.iter().map(|&q| t[q]).collect::<Vec<_>>();
            assert_eq!(pick(&ta), pick(&tb), "{part:?} pass {k}: tokens");
            let n_out = fast.cfg().seq() - fast.cfg().n_inputs;
            let stream_tokens: Vec<usize> = own.iter().copied().chain((spec.n_inputs..spec.n_inputs + n_out).map(|p| tok::output_mask(fast.cfg(), p))).collect();
            let got = fast.fill(&stream_tokens, &(spec.n_inputs..spec.n_inputs + n_out).collect::<Vec<_>>());
            let want: Vec<usize> = pick(&ta).iter().map(|&t| tok::output_value(&cfg, t).map_or(usize::MAX, |v| tok::output(fast.cfg(), v))).collect();
            assert_eq!(got, want, "{part:?} pass {k}: the stream's fast pass and the checkpoint's dense pass disagree");
        }
        assert!(fast.stats.fast_passes == PASSES as u64, "the {part:?} stream ran fast");
    }
}

/// The null cell a stock runtime must hold (`docs/stock-llama.md` §4). The dense pass
/// gives every overlay memory head the null cell the formula spells (`llama::null_kv`).
/// A runtime holds only what it decodes, so the null token's own forward pass, alone at
/// position 0, must give that key and value at every layer whose heads read the cells.
/// The cache then holds enough random lines that a memory head attending anywhere but
/// the null cell reads nonzero bytes. Over it, the whole checkpoint's dense pass equals
/// the control stream's slice at its output positions with the decoded null cell, and
/// differs with no null cell. Each dense pass takes minutes, so the test runs on request.
#[test]
#[ignore]
fn the_null_token_decodes_the_null_cell() {
    use gbcformer::llama::null_kv;
    use gbcformer::memory::layout::{LINES, LINE_BITS, VOID};
    let Whole { specs, model: full, .. } = whole();
    let o = full.overlay.as_ref().unwrap();
    let heads = full.memory.as_ref().unwrap().heads.clone();
    let memory_layers: std::collections::BTreeSet<usize> = heads.iter().map(|h| h.0).collect();
    let decoded = full.null_cell();
    let formula = null_kv(o);
    let mut wrong = Vec::new();
    for &li in &memory_layers {
        let (k, v) = decoded[li];
        if k.iter().zip(&formula.0).any(|(a, b)| a.to_bits() != b.to_bits()) || v.iter().any(|&x| x != 0.0) {
            wrong.push((li, k.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, x)| (i, *x)).collect::<Vec<_>>(), v.iter().filter(|&&x| x != 0.0).count()));
        }
    }
    let mut cells: Vec<(u32, [u8; 8])> = vec![(0, [0; 8]), (VOID >> LINE_BITS, [0; 8])];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || xorshift(&mut rng);
    for _ in 0..2048 {
        let n = (next() % LINES as u64) as u32;
        if n != 0 && n != VOID >> LINE_BITS {
            cells.push((n, next().to_le_bytes()));
        }
    }
    let cfg = full.cfg.clone();
    let part = Part::Control;
    let spec = &specs[part as usize];
    let slice = full.stream(part as usize, true);
    let (mut cache, mut own_cache) = (KvCache::new(heads), KvCache::new(slice.memory.as_ref().unwrap().heads.clone()));
    for &(n, l) in &cells {
        cache.set(n, l);
        own_cache.set(n, l);
    }
    let h = slice.cfg.hidden;
    let cleared = full.slice(part as usize, true).cleared;
    let outputs: Vec<usize> = spec.at[spec.n_inputs..].iter().flatten().copied().collect();
    let mut tokens: Vec<usize> = (0..cfg.seq()).map(|q| if q < cfg.n_inputs { tok::input(&cfg, q, 0) } else { tok::output_mask(&cfg, q) }).collect();
    for q in spec.at[..spec.n_inputs].iter().flatten() {
        tokens[*q] = tok::input(&cfg, *q, next() as u8);
    }
    let (want, _) = slice.residual_cached(&own_cache, &tokens, READ_AT);
    let differing = |null: Option<&[gbcformer::llama::Kv]>| -> usize {
        let (got, _) = full.residual_with_null(&cache, &tokens, READ_AT, null);
        outputs.iter().filter(|&&q| got[q * cfg.hidden..][..h].iter().zip(&want[q * h..][..h]).enumerate().any(|(ch, (x, y))| x.to_bits() != y.to_bits() && !cleared.contains(&ch))).count()
    };
    let (none, with_formula, with_decoded) = (differing(None), differing(Some(&vec![formula; cfg.layers])), differing(Some(&decoded)));
    println!("{} cells; decoded null cell wrong at {} of {} memory layers {:?}; control output positions differing from the slice: no null cell {none}, the formula's null cell {with_formula}, the decoded null cell {with_decoded}, of {}", cache.len(), wrong.len(), memory_layers.len(), wrong.first(), outputs.len());
    assert!(none > 0, "the cache is too small to show a foreign head off the null cell");
    assert_eq!(with_formula, 0, "with the formula's null cell the checkpoint is not the slice");
    assert!(wrong.is_empty(), "the null token's own forward pass does not give the null cell");
    assert_eq!(with_decoded, 0, "with the decoded null cell the checkpoint is not the slice");
}

/// The GBC CPU stream's records. Until the parts shared the input slots the one model held
/// 65,804 tokens and every record token (`REC`, `CELL`, the reader tokens, the upper
/// families) sat above 65,535, which this test was written to exercise. It holds 27,511
/// now, so no token of any checkpoint is that large and no run exercises one that is. The
/// token maps are held above 65,535 by `engine::view`'s unit test on a 300-slot `Config`
/// (`a_views_tokens_map_above_65535_and_back`); no token is stored narrower than 32 bits in
/// the engine, the GGUF reader, the runner or the harness (the review's search).
/// The GBC CPU stream runs SingleStepTests' `LD (HL),A` (a write each case) on
/// its fast plan, and every 4th kept record goes through the dense pass as the write
/// record a stock runtime decodes (`REC`, the address families, `CELL`), its cell's key
/// and value held to the formula; every 25th fast pass is also decoded by the dense pass
/// over the same cache, token for token.
#[test]
fn the_cpu_stream_keeps_records_through_the_oracles() {
    use gbcformer::memory::image::Image;
    use gbcformer::memory::layout::{Banking, Mapper};
    use gbcformer::sm83::MemCheckpointCore;
    let dir = std::path::Path::new("vendor/sm83/v1");
    if !dir.exists() {
        eprintln!("skip: no vendor/sm83/v1 (tools/link-data.sh)");
        return;
    }
    let mut e = gbcformer::one_model::streams(&[Part::Cpu]).cpu.expect("the GBC CPU stream");
    let cfg = e.dense().cfg.clone();
    eprintln!("{} tokens: REC is token {}", cfg.vocab, tok::rec(&cfg));
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
    e.insert(Image::new(&[0; 0x8000], 0x2000, banking).with_writable_rom());
    e.memory.as_mut().unwrap().oracle_every = 4;
    e.step_oracle = 25;
    let mut core = MemCheckpointCore::new(e);
    let (cases, failed) = gbcformer::sst::run_all(&mut core, dir, "77", false);
    let s = &core.model().stats;
    eprintln!("{cases} cases, {failed} failed; {s}");
    assert_eq!((cases, failed), (1000, 0));
    assert!(s.records >= 1000 && s.oracle_checks > 0 && s.step_checks > 0, "{} records, {} record oracle checks, {} step checks", s.records, s.oracle_checks, s.step_checks);
    assert_eq!(s.fallback_passes, 0);
}

/// The parts share the output identity channels (`Layout::identities`,
/// `docs/one-model.md` §4): a part's k-th output mask carries the k-th. So a row that
/// reads one must be silent outside its own part's positions whatever the identities
/// hold there. Read off the compiled weights, every such row is one of:
/// - an MLP unit whose `up` reads one mode channel and nothing else, so it adds 0 where
///   that mode is 0;
/// - a query of a memory head with the null term `N·(1 − mode)`, where `N` leads every
///   score the query can reach with each channel at its largest magnitude by `EXP_ZERO`
///   logits, so off its part the head reads the null cell's 0 (no query reads one today:
///   the PPU's positional heads read wires its units compute from the position wires);
/// - a layer-0 key. A key only decides which tokens a gather weighs, and the tokens it
///   newly raises are other parts' output masks, whose layer-0 values are all 0.
/// No value, `up` or LM-head row reads one, and no unit or head writes one.
///
/// A stream's own network (`Model::stream`) has no mode gate, so its units also run at
/// another part's position that holds one of its identities. No key or value row reads
/// a channel an MLP writes, so what they compute there stays at that position.
#[test]
fn every_reader_of_an_output_identity_is_gated_by_its_part() {
    use gbcformer::llama::{attn_scale, HEAD_DIM, NO_PART};
    use gbcformer::memory::attn::EXP_ZERO;
    let m = &whole().model;
    let (cfg, lay, o) = (&m.cfg, &m.layout, m.overlay.as_ref().unwrap());
    let (d, a, inter) = (cfg.hidden, cfg.attn(), cfg.intermediate);
    let ids = lay.output_ids(cfg.n_inputs);
    let parts = lay.offsets.len();
    let modes: Vec<usize> = (0..parts).map(|p| lay.mode(p)).collect();
    let row = |t: usize| &m.embed[t * d..][..d];

    // Each output mask carries its part's mode and one identity; a part's masks carry
    // distinct identities; the widest part sets how many there are.
    let mut held = vec![Vec::new(); parts];
    for j in cfg.n_inputs..cfg.seq() {
        let r = row(tok::output_mask(cfg, j));
        let part: Vec<usize> = (0..parts).filter(|&p| r[modes[p]] != 0.0).collect();
        let id: Vec<usize> = ids.clone().filter(|&ch| r[ch] != 0.0).collect();
        assert!(part.len() == 1 && id == [lay.slot_id(j)], "output slot {j}'s mask carries modes {part:?} and identities {id:?}");
        assert!((cfg.n_const..d).all(|ch| r[ch] == 0.0 || ch == id[0] || ch == modes[part[0]]), "output slot {j}'s mask carries another channel");
        assert!(!held[part[0]].contains(&id[0]), "two output slots of part {} share identity channel {}", part[0], id[0]);
        held[part[0]].push(id[0]);
    }
    assert_eq!(ids.len(), held.iter().map(Vec::len).max().unwrap(), "one identity channel per output slot of the widest part");
    assert!(held.iter().filter(|h| !h.is_empty()).count() > 1 && ids.len() < cfg.n_outputs, "the parts share the output identities");
    eprintln!("{} output identity channels for {} output slots (parts of {:?}); hidden {d}", ids.len(), cfg.n_outputs, held.iter().map(Vec::len).collect::<Vec<_>>());

    // A channel's largest magnitude at any position: a wire channel its occupants'
    // declared ranges, the constants themselves, every other channel 0/1.
    let base = lay.wire_base();
    let mag = |ch: usize| -> f32 {
        if ch < cfg.n_const {
            return gbcformer::llama::CONSTANTS[ch];
        }
        ch.checked_sub(base).and_then(|k| lay.occupants.get(k)).map_or(1, |occ| occ.iter().map(|s| { let (lo, hi) = m.wire_ranges[s.0 as usize]; lo.abs().max(hi.abs()) }).max().unwrap_or(1).max(1)) as f32
    };
    let reads = |r: &[f32]| ids.clone().any(|ch| r[ch] != 0.0);
    let null_dims = m.slice(0, true).null_dims;
    let null_score: f32 = o.null_q.iter().map(|w| w * o.null_k).sum();
    let (mut units, mut queries, mut keys) = (0, 0, 0);
    let mlp_writes: Vec<bool> = (0..d).map(|ch| m.layers.iter().any(|l| l.down[ch * inter..][..inter].iter().any(|&w| w != 0.0))).collect();
    for (li, l) in m.layers.iter().enumerate() {
        for (name, w) in [("key", &l.k), ("value", &l.v)] {
            assert!(w.chunks(d).all(|r| r.iter().zip(&mlp_writes).all(|(&w, &written)| w == 0.0 || !written)), "layer {li}: a {name} row reads a channel an MLP writes");
        }
        for u in 0..inter {
            let (g, up) = (&l.gate[u * d..][..d], &l.up[u * d..][..d]);
            assert!(!reads(up), "layer {li} unit {u}: up reads an output identity");
            if reads(g) {
                let read: Vec<usize> = (0..d).filter(|&ch| up[ch] != 0.0).collect();
                assert!(read.len() == 1 && modes.contains(&read[0]) && up[read[0]] == 1.0, "layer {li} unit {u} reads an output identity and its up reads channels {read:?}, not one mode");
                assert_eq!(o.units[li][u] as usize, modes.iter().position(|&ch| ch == read[0]).unwrap(), "layer {li} unit {u}: its mode is not its part's");
                units += 1;
            }
        }
        for h in 0..cfg.heads {
            let q = |t: usize| &l.q[(h * HEAD_DIM + t) * d..][..d];
            if !(0..HEAD_DIM).any(|t| reads(q(t))) {
                continue;
            }
            let part = o.heads[li][h];
            assert!(li > 0 && part != NO_PART, "layer {li} head {h}: a query that reads an output identity belongs to no part");
            // The null term on each null dim, and nothing else there.
            for (&dim, &w) in null_dims.iter().zip(&o.null_q) {
                let r = q(dim);
                assert!(r[0] == w && r[modes[part as usize]] == -w && (0..d).all(|ch| r[ch] == 0.0 || ch == 0 || ch == modes[part as usize]), "layer {li} head {h}: null dim {dim} is not N·(1 − mode)");
            }
            // Every key entry a cell or a group token holds on the other dims is at most 1
            // in magnitude (`memory_head`: ±1 per address bit, the committed bit).
            let reach: f32 = (0..HEAD_DIM).filter(|t| !null_dims.contains(t)).map(|t| q(t).iter().enumerate().map(|(ch, w)| w.abs() * mag(ch)).sum::<f32>()).sum();
            assert!(null_score >= reach + EXP_ZERO / attn_scale(), "layer {li} head {h}: the null cell scores {null_score}, the query can reach {reach}");
            queries += 1;
        }
        for r in 0..cfg.kv_dim() {
            assert!(!reads(&l.v[r * d..][..d]), "layer {li}: value row {r} reads an output identity");
            if reads(&l.k[r * d..][..d]) {
                assert_eq!(li, 0, "layer {li}: key row {r} reads an output identity");
                keys += 1;
            }
        }
        for ch in ids.clone() {
            assert!(l.o[ch * a..][..a].iter().all(|&w| w == 0.0) && l.down[ch * inter..][..inter].iter().all(|&w| w == 0.0), "layer {li} writes output identity channel {ch}");
        }
    }
    for j in cfg.n_inputs..cfg.seq() {
        let x = row(tok::output_mask(cfg, j));
        assert!(m.layers[0].v.chunks(d).all(|w| w.iter().zip(x).all(|(w, x)| w * x == 0.0)), "output slot {j}'s mask has a layer-0 value");
    }
    assert!(m.lm_head.chunks(d).all(|r| !reads(r)), "the LM head reads an output identity");
    // A family token carries an output identity too (its record slot's), with its byte in
    // its lanes and no mode. No mode means every unit above, gated by one, is silent at
    // it. Its layer-0 key is the one thing left that reads its identity, and that is by
    // design inside a record, whose gathers target it. Outside a record no sequence holds
    // one: a pass holds its own slots' masks and decoded bytes and byte 0 or the mask at
    // every other slot (the tokens `one_model::group` takes its margins over), and
    // `tok::stray` stops a record's sequence that holds anything else.
    for f in gbcformer::llama::Fam::ALL {
        for v in [0u8, 255] {
            let r = row(tok::family(cfg, f, v));
            let id: Vec<usize> = ids.clone().filter(|&ch| r[ch] != 0.0).collect();
            assert!(id.len() == 1 && modes.iter().all(|&ch| r[ch] == 0.0), "family token {f:?} {v} carries identities {id:?} or a mode");
        }
    }
    // Pinned, so a change in what reads an output identity is noticed: the mode-gated
    // units, no head query (the null-gated branch above has no instance), and the fallback
    // dim with the record gathers' six code dims.
    assert_eq!((units, queries, keys), (4223, 0, 7), "the readers of an output identity: mode-gated units, null-gated head queries, layer-0 key rows");
}

/// The six circuits, the merged circuit with its stream specs, and the whole checkpoint,
/// compiled once for the tests that only read it.
struct Whole {
    parts: [Circuit; 6],
    c: Circuit,
    specs: Vec<StreamSpec>,
    model: Model,
}

fn whole() -> &'static Whole {
    static WHOLE: OnceLock<Whole> = OnceLock::new();
    WHOLE.get_or_init(|| {
        let parts = parts();
        let (c, specs) = circuit(&parts);
        let model = Model::compile_mem(&c, Params::memory());
        Whole { parts, c, specs, model }
    })
}

/// A memory every part's circuit can read: a random ROM, random cartridge RAM, WRAM,
/// VRAM, OAM, HRAM and palettes, and the registers and constant blocks as a cartridge's
/// image has them.
fn scene(rng: &mut u64) -> Image {
    use gbcformer::memory::layout::{Banking, Mapper, CRAM, HRAM, HRAM_LEN, OAM, OAM_LEN, PAL, PAL_LEN, VRAM, VRAM_LEN, WRAM, WRAM_LEN};
    let mut bytes = |n: usize| (0..n).map(|_| xorshift(rng) as u8).collect::<Vec<u8>>();
    let rom = bytes(0x8000);
    let mut image = Image::new(&rom, 0x2000, Banking::new(Mapper::Rom, 0x8000, 0x2000, true).unwrap()).with_palettes();
    for (base, len) in [(CRAM, 0x2000), (WRAM, WRAM_LEN), (VRAM, VRAM_LEN), (OAM, OAM_LEN), (HRAM, HRAM_LEN), (PAL, PAL_LEN)] {
        image.fill(base, &bytes(len));
    }
    image
}

/// What `own` computes on `input` over `image`: every output's byte, an output with a
/// position wire at its own position, and the lines its heads read at any position.
/// None when the circuit refuses the input (a declared range or exclusive case broken,
/// or a line with no cell).
fn evaluate(own: &Circuit, input: &BTreeMap<String, u8>, image: &Image) -> Option<(BTreeSet<u32>, BTreeMap<String, u8>)> {
    let seen = RefCell::new(BTreeSet::new());
    let mem = |n: u32| {
        seen.borrow_mut().insert(n);
        image.line(n)
    };
    let out = catch_unwind(AssertUnwindSafe(|| {
        let mut out = own.eval_at(input, &mem, None);
        for k in 0..own.positions.len() {
            let name = &own.outputs[k].0;
            let at = own.eval_at(input, &mem, Some(k));
            out.insert(name.clone(), at[name]);
        }
        out
    }))
    .ok()?;
    Some((seen.into_inner(), out))
}

/// What an unpatched runtime computes is the whole checkpoint at every position, so
/// every part's slice (`Model::stream`) must be the whole checkpoint at the part's
/// output positions: bit-identical residuals on the slice's channels (but the mode and
/// null channels, which the slice clears and none of its rows read), 0 on every channel
/// above them, the same tokens, and those tokens the part's own circuit's bytes, the
/// record slots' family tokens included. Each part runs one pass on random input bytes
/// its circuit accepts, over a cache that holds every line the circuit reads there and
/// 1,024 other lines of a random memory.
///
/// The same pass with no null cell must differ at some output position of every part:
/// the other parts' heads then read a line of the cache, so the comparison sees a head
/// that reads where it should not.
#[test]
fn every_parts_slice_is_the_checkpoint_at_its_positions() {
    use gbcformer::memory::layout::{LINES, LINE_BITS, VOID};
    let w = whole();
    let (full, cfg) = (&w.model, &w.model.cfg);
    let heads = full.memory.as_ref().unwrap().heads.clone();
    let mut rng = 0x1234_5678_9ABC_DEF1u64;
    let image = scene(&mut rng);
    for part in Part::ALL {
        let p = part as usize;
        let (spec, own) = (&w.specs[p], &w.parts[p]);
        // Random input bytes the part's own circuit accepts. Control's must be a state
        // the reference reaches (`control_inputs`).
        let (bytes, lines, want) = (0..64)
            .find_map(|_| {
                let bytes: Vec<u8> = if part == Part::Control { control_inputs(&mut rng) } else { (0..spec.n_inputs).map(|_| xorshift(&mut rng) as u8).collect() };
                let input: BTreeMap<String, u8> = own.inputs.iter().zip(&bytes).map(|((n, _), &b)| (n.clone(), b)).collect();
                evaluate(own, &input, &image).map(|(lines, out)| (bytes, lines, out))
            })
            .unwrap_or_else(|| panic!("{part:?}: its circuit refused 64 random inputs"));
        assert!(bytes.iter().any(|&b| b != 0), "{part:?}: an input of zeros");
        let slice = full.stream(p, true);
        let (mut cache, mut own_cache) = (KvCache::new(heads.clone()), KvCache::new(slice.memory.as_ref().unwrap().heads.clone()));
        let mut cells: BTreeSet<u32> = lines.iter().copied().chain([0, VOID >> LINE_BITS]).collect();
        while cells.len() < lines.len() + 1024 {
            let n = (xorshift(&mut rng) % LINES as u64) as u32;
            if image.has_line(n) {
                cells.insert(n);
            }
        }
        for &n in &cells {
            cache.set(n, image.line(n));
            own_cache.set(n, image.line(n));
        }
        let mut tokens: Vec<usize> = (0..cfg.seq()).map(|q| if q < cfg.n_inputs { tok::input(cfg, q, 0) } else { tok::output_mask(cfg, q) }).collect();
        for (j, q) in spec.at[..spec.n_inputs].iter().enumerate() {
            tokens[q.unwrap()] = tok::input(cfg, q.unwrap(), bytes[j]);
        }
        let h = slice.cfg.hidden;
        let cleared = full.slice(p, true).cleared;
        let outputs: Vec<usize> = spec.at[spec.n_inputs..].iter().flatten().copied().collect();
        let t = std::time::Instant::now();
        let (a, _) = full.residual_cached(&cache, &tokens, READ_AT);
        let (b, _) = slice.residual_cached(&own_cache, &tokens, READ_AT);
        let differs = |x: &[f32], q: usize| x[q * cfg.hidden..][..h].iter().zip(&b[q * h..][..h]).enumerate().any(|(ch, (x, y))| x.to_bits() != y.to_bits() && !cleared.contains(&ch));
        for &q in &outputs {
            assert!(!differs(&a, q), "{part:?} position {q} ({}): the slice's residual differs from the whole checkpoint's", full.slots[q]);
            assert!(a[q * cfg.hidden + h..(q + 1) * cfg.hidden].iter().all(|&v| v == 0.0), "{part:?} position {q}: a channel above the slice's {h} is not 0");
        }
        let (ta, tb) = (full.tokens_of(&a), slice.tokens_of(&b));
        let mut families = 0;
        for (name, &q) in spec.slots[spec.n_inputs..].iter().zip(&outputs) {
            assert_eq!(ta[q], tb[q], "{part:?} {name}: the slice's token");
            assert_eq!(tok::byte(cfg, ta[q]), Some(want[name]), "{part:?} {name}: the whole checkpoint decodes token {}, the part's own circuit gives {:#04x}", ta[q], want[name]);
            families += tok::output_value(cfg, ta[q]).is_none() as usize;
        }
        // Only the GBC CPU keeps its record slots, which decode to family tokens.
        assert_eq!(families, if part == Part::Cpu { 6 + gbcformer::circuit::LINE } else { 0 }, "{part:?}: output slots that decode a family token");
        let (bare, _) = full.residual_with_null(&cache, &tokens, READ_AT, None);
        let without: usize = outputs.iter().filter(|&&q| differs(&bare, q)).count();
        eprintln!(
            "{part:?}: slice hidden {h}, {} layers; {} lines read, {} cells; {} output positions equal to the slice and to its own circuit ({families} family tokens); with no null cell {without} differ; {:.0} s",
            slice.cfg.layers,
            lines.len(),
            cache.len(),
            outputs.len(),
            t.elapsed().as_secs_f64()
        );
        assert!(without > 0, "{part:?}: with no null cell the whole checkpoint still equals the slice, so this cache cannot show a head that reads off its part");
    }
}

/// A cell's rows are what a runtime's cache holds for a line: the key and value its
/// write record's `CELL` token has at each layer. An unpatched runtime decodes the
/// record through the whole checkpoint, and the PPU's parts' heads sit in layers past
/// the GBC CPU's own, so this holds the whole checkpoint's dense pass over write
/// records, at every memory head, to the formula's key and the line's value (what
/// `KvCache` gives the dense pass for a cell), and to the rows the engine's builder
/// computes from the whole checkpoint's weights for every memory layer. The formula's
/// key must differ between a line and the next, and its value between a line and its
/// complement, in f16, so that those comparisons hold a record to its own line.
#[test]
fn a_write_records_cell_is_the_caches_at_every_memory_head() {
    use gbcformer::engine::{turn, RowBuilder};
    use gbcformer::llama::f16;
    use gbcformer::memory::attn::{key, value, Kind};
    use gbcformer::memory::dense::Records;
    use gbcformer::memory::layout::{Banking, Mapper, HRAM, LINE, LINE_BITS, N_REGS, OAM, PAL, REGS, VOID, VRAM, WRAM};
    let w = whole();
    let (full, cfg) = (&w.model, &w.model.cfg);
    let spec = full.memory.as_ref().unwrap();
    // Every layer a circuit head sits at, and the two copy heads' layers.
    let circuit_layers: BTreeSet<usize> = w.c.heads.iter().map(|h| h.layer as usize).collect();
    let layers: BTreeSet<usize> = circuit_layers.iter().copied().chain(spec.copy_heads.iter().map(|h| h.0)).collect();
    assert_eq!(spec.heads.iter().map(|h| h.0).collect::<BTreeSet<_>>(), layers, "the layers of the memory heads");
    assert_eq!(spec.heads.len(), w.c.heads.len() + 2, "every circuit head and the two copy heads");
    let cpu_layers = w.c.part_depth(Part::Cpu as usize) as usize;
    assert!(layers.iter().any(|&l| l > cpu_layers), "no memory head sits past the GBC CPU's {cpu_layers} layers: this test would check no more than the stream's own oracle");
    let builder = RowBuilder::new(full).expect("the whole checkpoint's row builder");
    assert_eq!(builder.memory_layers, layers.iter().copied().collect::<Vec<_>>(), "the row builder's memory layers");
    eprintln!("{} memory heads at layers {layers:?}; {} row groups", spec.heads.len(), builder.groups());

    let mut rng = 0x0C0F_FEE0_1234_5678u64;
    let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, true).unwrap();
    let mut regs = [0u8; 2 * LINE];
    regs[..N_REGS].copy_from_slice(&banking.0);
    // VOID's line first (an off column reads it at a record's positions too), then the
    // registers, then lines across the address space: ROM bank 0 and the key's high
    // bits, each RAM, and a line written twice.
    let mut records: Vec<(u32, [u8; LINE])> = vec![(VOID, [0; LINE]), (REGS, regs[..LINE].try_into().unwrap()), (REGS + LINE as u32, regs[LINE..].try_into().unwrap())];
    for addr in [0u32, 0x4000, 0x20_0005, 0x40_0003, 0x7F_FFFE, WRAM + 0x1234, VRAM + 7, VRAM + 0x2001, OAM + 0x9C, HRAM + 3, PAL + 0x41, WRAM + 0x1236] {
        records.push((addr, xorshift(&mut rng).to_le_bytes()));
    }
    for _ in 0..12 {
        records.push((VRAM + (xorshift(&mut rng) % 0x4000) as u32, xorshift(&mut rng).to_le_bytes()));
    }
    let bits = |x: &[f32; HEAD_DIM]| x.map(f32::to_bits);
    // Equal as f16 numbers, +0 and −0 alike (`engine::cells::same`).
    let same16 = |row: &[u16; HEAD_DIM], want: &[f32; HEAD_DIM]| row.iter().zip(want).all(|(&x, &y)| gbcformer::gguf::f16_to_f32(x) == f16(y));
    let mut cache = KvCache::new(spec.heads.clone());
    let r = Records { cfg, slots: &spec.slots };
    for &(addr, line) in &records {
        let n = addr >> LINE_BITS;
        cache.remove(n);
        let (_, group) = full.forward_cached(&cache, &r.write(addr, line), READ_AT);
        let (want_k, want_v) = (key(Kind::Cell, n, READ_AT), value(&line));
        let (other_k, other_v) = (key(Kind::Cell, n ^ 1, READ_AT), value(&line.map(|b| !b)));
        // The comparisons below tell one line from the next, and one line's bytes from
        // another's, only if the formula does, as f32 and as the f16 rows a runtime stores.
        let stored = key(Kind::Cell, n, n as usize);
        let other_stored = key(Kind::Cell, n ^ 1, (n ^ 1) as usize);
        let apart = |a: &[f32; HEAD_DIM], b: &[f32; HEAD_DIM]| a.iter().zip(b).any(|(&x, &y)| f16(x) != f16(y));
        assert!(apart(&want_k, &other_k) && apart(&stored, &other_stored), "line {n:#07x}'s key and the next line's are one key in f16: no comparison with it holds a record to its line");
        assert!(apart(&want_v, &other_v), "the value of {addr:#08x}'s line and of its complement are one value in f16");
        for &head in &spec.heads {
            let (k, v) = group.head(cfg, head, Records::CELL);
            assert_eq!(bits(&k), bits(&want_k), "the record of {addr:#08x}: its cell's key at head {head:?}");
            assert_eq!(bits(&v), bits(&want_v), "the record of {addr:#08x}: its cell's value at head {head:?}");
        }
        // The engine's rows, as the runtime stores them: decoded at `READ_AT + 1`, then
        // moved to the line's position.
        let (mut ks, mut vs) = (vec![[0u16; HEAD_DIM]; builder.groups()], vec![[0u16; HEAD_DIM]; builder.groups()]);
        builder.group_rows(addr, line, &turn(n as i64 - (READ_AT + 1) as i64), &mut ks, &mut vs).unwrap_or_else(|e| panic!("the rows of {addr:#08x}: {e}"));
        for &l in &builder.memory_layers {
            let g = builder.group(l);
            assert!(same16(&ks[g], &stored), "the record of {addr:#08x}: the engine's key row at layer {l} is not the cache's");
            assert!(same16(&vs[g], &want_v), "the record of {addr:#08x}: the engine's value row at layer {l} is not the line");
        }
        cache.set(n, line);
    }
    eprintln!("{} write records, each cell held at {} heads and {} memory layers", records.len(), spec.heads.len(), builder.memory_layers.len());
}

/// The one model's shape, pinned: what the uniform file stores for every layer (hidden,
/// the widest layer's MLP units and heads, the layers) and each stream's slice. A
/// placement or a circuit that moves any of these moves the unpatched runtime's cost,
/// so the change shows here (`docs/perf-log.md`, "Re-levelling").
#[test]
fn the_one_models_shape_is_pinned() {
    let w = whole();
    let cfg = &w.model.cfg;
    assert_eq!((cfg.hidden, cfg.intermediate, cfg.heads, cfg.layers), (1920, 3456, 26, 25), "hidden, the widest layer's MLP units, the widest layer's heads, layers");
    assert_eq!((cfg.n_inputs, cfg.vocab), (90, 27511), "the input slots the parts share (the stepper's 79 and the GBC CPU's 11 with nibble wires) and the vocabulary");
    let census = Model::census(&w.c, true);
    assert_eq!((census.units.iter().copied().max(), census.heads[0], census.heads[1..].iter().copied().max()), (Some(3424), 24, Some(26)), "the widest layer's units, layer 0's gathers, the most memory heads in a layer");
    let streams: Vec<(usize, usize)> = Part::ALL.iter().map(|&p| w.model.slice(p as usize, true)).map(|s| (s.hidden, s.units.len())).collect();
    assert_eq!(streams, [(1920, 21), (1920, 11), (1664, 25), (1664, 17), (1088, 21), (1920, 25)], "each stream's hidden and layers: the scanline PPU, the GBC CPU, the APU, control, the length, the stepper");
    // The GBC CPU's slice is as wide as the checkpoint: the norm's bound sets its width
    // (`Model::slice`), which is the floor `examples/room` holds a placement to.
    assert_eq!(streams[Part::Cpu as usize].0, gbcformer::one_model::CPU_SLICE_HIDDEN);
}

/// `PLACEMENT` leaves 17 wires free in hidden, so a circuit change that keeps more wires
/// live than that would compile a wider checkpoint. `placed` must refuse that by name. The
/// stand-in here is a placement that lets a part keep 18 more wires live: it ends within
/// its caps (`examples/room -- --try 3480 26 ppu=20,stp=24,len=20 1389`), the stepper
/// keeps 1,402 wires, and its checkpoint has hidden 1,984.
#[test]
#[should_panic(expected = "compiles a checkpoint of hidden 1984, not 1920")]
fn a_placement_that_widens_hidden_is_refused() {
    placed(&whole().parts, &Placement { live: PLACEMENT.live + 18, ..PLACEMENT });
}
