//! Prints the compiled checkpoint's circuit statistics and per-layer shape, for the
//! host-fed CPU circuit, the memory circuit with `--mem`, the APU circuit with
//! `--apu`, the scanline circuit with `--ppu`, or the FIFO stepper with `--stepper`; `--one` gives the one model by part and layer.

use gbcformer::engine::Engine;
use gbcformer::llama::Model;
use gbcformer::memory::attn::Params;
use gbcformer::sm83::{build, build_mem};

fn main() {
    if std::env::args().any(|a| a == "--one") {
        return one();
    }
    let mem = std::env::args().any(|a| a == "--mem");
    let apu = std::env::args().any(|a| a == "--apu");
    let ppu = std::env::args().any(|a| a == "--ppu");
    let stepper = std::env::args().any(|a| a == "--stepper");
    let c = if stepper { gbcformer::ppu_circuit::stepper::build() } else if ppu { gbcformer::ppu_circuit::circuit::build() } else if apu { gbcformer::apu_circuit::build() } else if mem { build_mem() } else { build() };
    let mut per_level = vec![0usize; c.depth() as usize + 1];
    for n in &c.nodes {
        per_level[c.level(n.out) as usize] += n.steps.len();
    }
    println!("circuit: {} wires, {} nodes, depth {}, steps per level {:?}, units per level {:?}, wires live per layer {:?}, {} memory heads at layers {:?}", c.levels.len(), c.nodes.len(), c.depth(), &per_level[1..], c.units_per_level(), c.live_per_boundary(), c.heads.len(), c.heads.iter().map(|h| h.layer).collect::<Vec<_>>());
    if mem {
        println!("predicted (changed wire, reading layer) pairs per pass: {:.1}; wire channels at hidden 1,024: {}", c.changed_pairs(&gbcformer::sm83::changes(&c)), Model::wire_budget(&c, true, 1));
        // What holds the width: the widest boundary's live wires by name family.
        let live = c.live_per_boundary();
        let b = (0..live.len()).max_by_key(|&b| live[b]).unwrap();
        let mut fam: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for s in c.live_at(b as u32) {
            let n = &c.names[s.0 as usize];
            *fam.entry(n.split(['.', '=']).next().unwrap_or(n).to_string()).or_default() += 1;
        }
        let mut fam: Vec<(String, usize)> = fam.into_iter().collect();
        fam.sort_by_key(|(n, k)| (std::cmp::Reverse(*k), n.clone()));
        println!("live at boundary {b} by family: {}", fam.iter().map(|(n, k)| format!("{n} {k}")).collect::<Vec<_>>().join(", "));
    }
    let model = if mem || ppu || stepper { Model::compile_mem(&c, Params::memory()) } else { Model::compile(&c) };
    if let Some(m) = &model.memory {
        println!("memory heads (layer, head) {:?}, address gathers {:?}, copy heads {:?}, vocab {}", m.heads, m.gathers, m.copy_heads, model.cfg.vocab);
    }
    let e = Engine::new(model);
    let cfg = &e.dense().cfg;
    println!("checkpoint: hidden {}, intermediate {}, {} layers, {:.1}M parameters, {} nonzero; integer fast path {:?}", cfg.hidden, cfg.intermediate, cfg.layers, cfg.params() as f64 / 1e6, e.nonzeros(), e.exact());
    println!("heads {}, vocab {}", cfg.heads, cfg.vocab);
    let (d, i) = (cfg.hidden, cfg.intermediate);
    for (k, l) in e.dense().layers.iter().enumerate() {
        let live = (0..i).filter(|&u| l.gate[u * d..][..d].iter().any(|&w| w != 0.0)).count();
        let gate = l.gate.iter().filter(|&&w| w != 0.0).count();
        println!("layer {k}: {live} live units, {gate} gate nonzeros ({:.1}/unit)", gate as f64 / live.max(1) as f64);
    }
}

/// `--one`: the one model's shape, what sets its hidden width, its FFN width and its
/// stored size, per part and per layer.
fn one() {
    use gbcformer::one_model::{circuit, parts, Part};
    let ps = parts();
    for (p, c) in Part::ALL.iter().zip(&ps) {
        let live = c.live_per_boundary();
        println!("part {}: {} inputs, {} outputs ({} record slots), depth {}, {} heads, units per level {:?}, widest live {}", p.name(), c.inputs.len(), c.outputs.len(), c.out_kinds.iter().filter(|&&k| k != gbcformer::circuit::OutKind::Shared).count(), c.depth(), c.heads.len(), c.units_per_level(), live.iter().max().unwrap_or(&0));
    }
    let (c, _) = circuit(&ps);
    let m = Model::compile_mem(&c, Params::memory());
    let (cfg, lay) = (&m.cfg, &m.layout);
    let keep = gbcformer::gguf::keep(&m, 64);
    println!("config: hidden {}, intermediate {}, {} layers, {} heads, vocab {} ({} inputs, {} outputs, seq {})", cfg.hidden, cfg.intermediate, cfg.layers, cfg.heads, cfg.vocab, cfg.n_inputs, cfg.n_outputs, cfg.seq());
    // What the file stores of the embedding and the LM head: every token's prefix of
    // columns, and the head's run of rows at its prefix.
    let head = gbcformer::gguf::HeadRows::of(&m);
    let (embed_cols, head_cols) = gbcformer::gguf::col_prefixes(&m, &head);
    let (emb, lm) = (cfg.vocab * embed_cols, head.count * head_cols);
    println!("uniform {:.1}M, stored layer-shaped (align 64) {:.1}M, align 1 {:.1}M; of the stored, embedding {:.1}M ({} of {} columns) + LM head {:.1}M ({} of {} rows at {} columns); uniform embedding + LM head {:.1}M", cfg.params() as f64 / 1e6, gbcformer::gguf::stored_params(&m, &keep) as f64 / 1e6, gbcformer::gguf::stored_params(&m, &gbcformer::gguf::keep(&m, 1)) as f64 / 1e6, emb as f64 / 1e6, embed_cols, cfg.hidden, lm as f64 / 1e6, head.count, cfg.vocab, head_cols, (2 * cfg.vocab * cfg.hidden) as f64 / 1e6);
    println!("layout: {} constants, {} slot ids, wire base {}, wires {} (after {}); route 0 at {}, lane 0 at {}, null {}", lay.n_const, lay.output_ids(cfg.n_inputs).end - lay.n_const, lay.wire_base(), lay.wires, lay.after_wires(), lay.route(0), lay.lane(0, 0), lay.null());
    for p in Part::ALL {
        let s = m.slice(p as usize, false);
        let own = lay.occupants.iter().filter(|occ| occ.iter().any(|&w| lay.part_of(w) == p as usize)).count();
        println!("part {}: prefix {} ({} own wire channels), slice hidden {}, layers {}, widest units {}, widest heads {}", p.name(), lay.prefix(p as usize), own, s.hidden, s.units.len(), s.units.iter().map(Vec::len).max().unwrap_or(0), s.heads.iter().map(Vec::len).max().unwrap_or(0));
    }
    let o = m.overlay.as_ref().unwrap();
    let w = lay.occupants.len() as i64;
    let squares: i64 = lay.occupants.iter().map(|occ| occ.iter().map(|s| { let (lo, hi) = m.wire_ranges[s.0 as usize]; lo.abs().max(hi.abs()).pow(2) }).max().unwrap_or(0)).sum::<i64>() + 2 * 224 + 1 + 64 * 64;
    let nibble_channels = lay.occupants.iter().filter(|occ| occ.iter().any(|s| { let (lo, hi) = m.wire_ranges[s.0 as usize]; lo.abs().max(hi.abs()) > 1 })).count();
    let floor = if squares - w < 15 * lay.after_wires() as i64 { 0 } else { (squares - w) / 15 + 1 };
    println!("norm: squares over the wire channels and constants {squares}, {nibble_channels} nibble wire channels, floor {floor} (0: the channels used bind)");
    let zero_lm = (0..cfg.vocab).filter(|&t| m.lm_head[t * cfg.hidden..][..cfg.hidden].iter().all(|&x| x == 0.0)).count();
    let clears: Vec<usize> = (0..Part::ALL.len()).map(|p| (0..cfg.layers).map(|li| m.unit_names[li].iter().zip(&o.units[li]).filter(|(n, q)| n.as_str() == "clear" && **q == p as u8).count()).sum()).collect();
    println!("clearing units by part {clears:?}");
    println!("LM head: {zero_lm} of {} rows all zero", cfg.vocab);
    let names: Vec<&str> = Part::ALL.iter().map(|p| p.name()).collect();
    println!("layer: kept units (width) heads | units by part {names:?} shared | heads by part, shared (layer 0: a gather counts for every part that holds its slots, and shared are the heads no part holds)");
    for (li, k) in keep.iter().enumerate() {
        let by = |v: &[u8]| -> (Vec<usize>, usize) { ((0..Part::ALL.len()).map(|p| v.iter().filter(|&&x| x == p as u8).count()).collect(), v.iter().filter(|&&x| x == gbcformer::llama::NO_PART).count()) };
        let (u, us) = by(&o.units[li]);
        // Layer 0's gathers have holders, not one part (`Overlay::holders`).
        let (h, hs) = if li == 0 { ((0..Part::ALL.len()).map(|p| o.holders.iter().filter(|&&x| x >> p & 1 == 1).count()).collect(), o.holders.iter().filter(|&&x| x == 0).count()) } else { by(&o.heads[li]) };
        println!("layer {li:2}: {:6} ({:6}) {:3} | {u:?} {us} | {h:?} {hs}", k.units.len(), k.width, k.heads.len());
    }
}
