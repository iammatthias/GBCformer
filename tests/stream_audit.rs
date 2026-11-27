//! Each part of the one model computes only exact integers, and each of its memory heads
//! reads exactly one cell (`docs/stock-llama.md` §4). The test evaluates the merged
//! circuit at the state Yellow reaches after its boot: every state the first way below, and
//! each random state (the APU's, the only part drawn at random) the second way too.
//!
//! First with the part's bytes under its own input names and every other name at byte 0.
//! The circuit asserts that every node is in its range, so every query is an exact
//! address, and the memory closure asserts that every queried line has a cell.
//!
//! Then as the checkpoint has it since the parts share the input slots: layer 0 gathers a
//! slot once, so every holder of a slot sees the stream's byte. The part's own outputs
//! must be the first evaluation's. The other parts then compute on bytes that are not
//! theirs, and the merged circuit refuses some of those states: a node of another part
//! leaves its declared range, or a head queries a line that holds no cell. The test
//! requires that a refusal names another part's node, never the stream's own, and prints
//! the counts. A refused state is not a wrong result. Those values sit at the other
//! parts' output positions, which only a whole-sequence pass holds (`STOCK_GROUP=full`).
//! A stream's own group holds none of them, at the stream's own positions the other
//! parts' units and heads are silent
//! (`every_reader_of_an_output_identity_is_gated_by_its_part`), and
//! `every_parts_slice_is_the_checkpoint_at_its_positions` holds the whole-sequence pass to
//! each part's own circuit there.

use gbcformer::machine::io::Buttons;
use gbcformer::one_model::{circuit, parts, Part};
use gbcformer::{Hardware, Session};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[test]
fn every_stream_reads_one_cell_per_head_and_stays_integral() {
    let Ok(rom) = std::fs::read("pokemon-yellow.gbc") else {
        eprintln!("skip: no pokemon-yellow.gbc (tools/link-data.sh)");
        return;
    };
    let mut session = Session::new(rom, None, Hardware::Cgb).unwrap();
    for _ in 0..60 {
        session.run_frame(Buttons(0));
    }
    let image = session.memory_image().unwrap().with_palettes();
    let own_parts = parts();
    let (merged, _) = circuit(&own_parts);
    let mem = |n: u32| {
        assert!(image.has_line(n), "a memory head queries line {n:#07x}, where no cell is");
        image.line(n)
    };
    // The second evaluation serves a line without a cell as zeros and counts it.
    let absent = Cell::new(0usize);
    let lenient = |n: u32| {
        if image.has_line(n) {
            return image.line(n);
        }
        absent.set(absent.get() + 1);
        [0; gbcformer::memory::layout::LINE]
    };
    // The merged circuit's slot of an input name, and every name that holds a slot.
    let slot_of = |n: &str| merged.inputs.iter().position(|(m, _)| m == n).or_else(|| merged.held.iter().find(|(_, m, _)| m == n).map(|(j, _, _)| *j)).expect("an input name");
    let holders = |j: usize| std::iter::once(&merged.inputs[j].0).chain(merged.held.iter().filter(move |(k, _, _)| *k == j).map(|(_, m, _)| m));
    let own_outputs = |o: &BTreeMap<String, u8>, tag: &str| o.iter().filter(|(k, _)| k.starts_with(tag)).map(|(k, v)| (k.clone(), *v)).collect::<Vec<_>>();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    // Each part's own slots are zero, plus random states for the APU. The APU's circuit
    // refuses about 1 random state in 600 (a frame-sequencer fall and rise at once, which
    // the host never sends); the 200 drawn here pass by their seed.
    for (pi, part) in Part::ALL.iter().enumerate() {
        let tag = format!("{}:", part.name());
        // The part's inputs under its own names, in its own order: the merged circuit lists
        // the input slots the parts share, each under its first holder's name.
        let own: Vec<String> = own_parts[pi].inputs.iter().map(|(n, _)| format!("{tag}{n}")).collect();
        assert!(own.iter().all(|n| merged.inputs.iter().map(|(m, _)| m).chain(merged.held.iter().map(|(_, m, _)| m)).any(|m| m == n)), "the merged circuit holds every input of {tag}");
        let trials = if *part == Part::Apu { 200 } else { 1 };
        let (mut refused, mut equal) = (0, 0);
        absent.set(0);
        for t in 0..trials {
            let mut input = BTreeMap::new();
            if t > 0 {
                for n in &own {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    input.insert(n.clone(), rng as u8);
                }
            }
            // The PPU's position wires differ per output position, so evaluate every eighth
            // pixel.
            let mut first = None;
            for at in std::iter::once(None).chain((0..160).step_by(8).map(Some)) {
                let out = merged.eval_at(&input, &mem, at);
                assert!(!out.is_empty(), "stream {pi} decodes");
                first.get_or_insert(out);
            }
            // As the checkpoint has it: every holder of a slot sees the part's byte. With
            // every byte 0 that is the evaluation above.
            let mut shared = input.clone();
            for (n, &b) in &input {
                for m in holders(slot_of(n)) {
                    shared.insert(m.clone(), b);
                }
            }
            if shared == input {
                continue;
            }
            match catch_unwind(AssertUnwindSafe(|| merged.eval_at(&shared, &lenient, None))) {
                Ok(out) => {
                    assert!(!own_outputs(&out, &tag).is_empty(), "{tag} names its outputs");
                    assert_eq!(own_outputs(&out, &tag), own_outputs(first.as_ref().unwrap(), &tag), "{tag} trial {t}: the part's outputs change when the other holders see its bytes");
                    equal += 1;
                }
                Err(e) => {
                    let msg = e.downcast_ref::<String>().cloned().unwrap_or_default();
                    assert!(msg.contains(" outside [") && !msg.starts_with(&tag), "{tag} trial {t}: refused, and not at another part's node: {msg}");
                    refused += 1;
                }
            }
        }
        if trials > 1 {
            eprintln!("{tag} of {} random states with every holder seeing the part's bytes, {equal} give the part's own outputs and {refused} are refused at another part's node; {} queried lines hold no cell", trials - 1, absent.get());
            assert!(equal > 0, "{tag} no state with every holder fed was evaluated to the end");
        }
    }
}

/// A stream's reduced group (`one_model::group`, the positions a stock runtime decodes)
/// is its own slots and nothing else. A layer-0 gather serves every part that holds its
/// group of input slots, and a gather whose targets the group lacks attends to the
/// stream's own output masks, whose values are zero like the targets'. `group` asserts each gather's margins from the
/// weights.
#[test]
fn every_stream_decodes_only_its_own_slots() {
    use gbcformer::one_model::group;
    let (c, specs) = circuit(&parts());
    let m = gbcformer::llama::Model::compile_mem(&c, gbcformer::memory::attn::Params::memory());
    let sizes: Vec<(Part, usize, usize)> = Part::ALL.iter().map(|&p| (p, group(&m, &specs, p).keep.len(), specs[p as usize].at.iter().flatten().count())).collect();
    // The APU's 63 state bytes and events in, 63 and its mix, read, sample distance,
    // segment end and cycles left out (SameBoy's rules); control's rise distance adds FR0 and FR1, the transport's
    // rules in the model take six inputs off control and add RT and NE, and FF72-FF75 add UR0-UR2 in and out; the OAM bug's pass adds OB in and
    // OT, OW0-OW4 and OL out to the GBC CPU's.
    assert_eq!(sizes, vec![(Part::Ppu, 166, 166), (Part::Cpu, 146, 146), (Part::Apu, 139, 139), (Part::Control, 150, 150), (Part::Length, 10, 10), (Part::Stepper, 158, 158)]);
    let g = group(&m, &specs, Part::Cpu);
    assert_eq!(g.gap, 128.0);
    // The part that alone holds a position: an output slot's.
    let owner = |p: usize| Part::ALL.iter().copied().find(|&q| specs[q as usize].at.contains(&Some(p))).expect("a target some part owns");
    // The record gathers are shared. The parts share the output identity channels
    // (`Layout::identities`), so over the whole sequence a record gather's targets are
    // its record slots' masks and every other part's mask on the same identities, all
    // with zero values. In the CPU's group it attends to its own record slots alone.
    let spec = m.memory.as_ref().unwrap();
    let record = |h: &gbcformer::one_model::Gather| spec.gathers.contains(&h.head);
    let records: Vec<usize> = spec.slots.addr.iter().flatten().chain(&spec.slots.line[0]).copied().collect();
    for h in g.gathers.iter().filter(|h| record(h)) {
        assert!(!h.attended.is_empty() && h.attended.iter().all(|p| records.contains(p)), "record gather {} attends to {:?} in the CPU's group", h.head, h.attended);
        assert!(h.attended.iter().all(|p| h.targets.contains(p)) && h.targets.iter().any(|&p| owner(p) != Part::Cpu), "record gather {}'s targets {:?}", h.head, h.targets);
    }
    assert_eq!(g.gathers.iter().filter(|h| record(h)).count(), 3);
    // The parts share the input slots, in 21 groups that each part holds whole or not at
    // all (`Model::groups`). Per part: the input gathers whose slots its group holds, and
    // the ones that fall back on its own masks, whose values are zero like the byte 0 the
    // stream holds at a slot it does not own. `group` asserts every gather's margins either way.
    let gathers: Vec<(Part, usize, usize)> = Part::ALL
        .iter()
        .map(|&q| {
            let g = group(&m, &specs, q);
            let held = g.gathers.iter().filter(|h| !record(h) && h.targets.iter().all(|p| g.keep.contains(p))).count();
            let fallback = g.gathers.iter().filter(|h| !record(h) && !h.targets.iter().any(|p| g.keep.contains(p))).count();
            (q, held, fallback)
        })
        .collect();
    assert_eq!(gathers, vec![(Part::Ppu, 2, 19), (Part::Cpu, 9, 12), (Part::Apu, 15, 6), (Part::Control, 12, 9), (Part::Length, 3, 18), (Part::Stepper, 18, 3)], "per part, the input gathers it holds and the ones that fall back: no gather is held in part");
}

/// Every checkpoint holds every weight exactly in f16 and every partial sum exactly in
/// an f16 accumulator, the one model included, and the APU's and control's f16-tile
/// and f16-accumulation dense passes (in three summation orders) decode what their
/// engines decode.
#[test]
fn every_checkpoint_is_f16_exact() {
    use gbcformer::llama::{tok, Denoiser, Model, Order, Tiles};
    let p = gbcformer::memory::attn::Params::memory();
    for (name, c, memory) in [
        ("cpu", gbcformer::sm83::build(), false),
        ("cpu-memory", gbcformer::sm83::build_mem(), true),
        ("apu", gbcformer::apu_circuit::build(), false),
        ("control", gbcformer::control_circuit::build(), false),
        ("ppu", gbcformer::ppu_circuit::circuit::build(), true),
        ("one-model", circuit(&parts()).0, true),
    ] {
        let mut m = if memory { Model::compile_mem(&c, p) } else { Model::compile(&c) };
        m.check_f16().unwrap_or_else(|e| panic!("{name}: {e}"));
        m.check_f16_acc().unwrap_or_else(|e| panic!("{name}: f16 sums: {e}"));
        if name == "apu" || name == "control" {
            let mut e = gbcformer::engine::Engine::new(Model::compile(&c));
            let cfg = m.cfg.clone();
            let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            for k in 0..8 {
                m.tiles = [Tiles::F16, Tiles::F16Acc(Order::Forward), Tiles::F16Acc(Order::PositivesFirst), Tiles::F16Acc(Order::NegativesFirst)][k % 4];
                let tokens: Vec<usize> = (0..cfg.seq())
                    .map(|j| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        if j < cfg.n_inputs { tok::input(&cfg, j, rng as u8) } else { tok::output_mask(&cfg, j) }
                    })
                    .collect();
                assert_eq!(e.fill(&tokens, &outputs), m.fill(&tokens, &outputs), "{name}: {:?} pass {k}", m.tiles);
            }
        }
    }
}

/// The f16-accumulation pass catches an overflow. A step whose gate's positive terms sum
/// past the f16 maximum (built by hand, bypassing `node_at`'s legalisation) fails
/// `check_f16_acc` and decodes wrong when the pass sums those terms first, while f32
/// sums decode it right.
#[test]
fn a_gate_past_the_f16_range_fails_in_f16_sums() {
    use gbcformer::circuit::{byte, value, Circuit, Lin, Step};
    use gbcformer::llama::{tok, Denoiser, Model, Order, Tiles};
    let mut c = Circuit::default();
    let a = byte(c.input("A"));
    let b = byte(c.input("B"));
    let form = value(&a, 8) + value(&b, 8) * 2;
    let n = c.and("seed", &[a[0].clone(), b[0].clone()]);
    // Reuse the seed node's wire slot for a node [A + 2B ≥ 384] with mass 765 units of K.
    let out = *n.terms.keys().next().unwrap();
    c.nodes.last_mut().unwrap().steps = vec![Step { coef: 1, lin: form, theta: 384 }];
    c.output("Y", [Lin::from(out), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0), Lin::konst(0)]);
    let mut m = Model::compile(&c);
    assert!(m.check_f16().is_ok());
    assert!(m.check_f16_acc().is_err());
    let cfg = m.cfg.clone();
    let tokens = vec![tok::input(&cfg, 0, 255), tok::input(&cfg, 1, 255), tok::output_mask(&cfg, 2)];
    let want = tok::output(&cfg, 1);
    let mut out = Vec::new();
    m.fill_into(&tokens, &[2], &mut out);
    assert_eq!(out, vec![want], "f32 sums");
    m.tiles = Tiles::F16Acc(Order::PositivesFirst);
    m.fill_into(&tokens, &[2], &mut out);
    assert_ne!(out, vec![want], "f16 sums, positives first");
}
