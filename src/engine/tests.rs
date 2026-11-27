use super::Engine;
use crate::llama::{tok, Config, Denoiser, Model};

/// Random input bytes with every output masked, as a step's first pass sees them.
fn state(cfg: &Config, mut rng: u64) -> Vec<usize> {
    (0..cfg.seq())
        .map(|j| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            if j < cfg.n_inputs { tok::input(cfg, j, rng as u8) } else { tok::output_mask(cfg, j) }
        })
        .collect()
}

/// A token whose cached value vector gives non-integer wires makes the exclusive
/// plan's wire list unbuildable half way through a pass. The pass falls back, and
/// the marks it had already made must not reach the next pass.
#[test]
fn unbuildable_wire_list_leaves_no_marks_behind() {
    let mut fast = Engine::new(Model::compile(&crate::sm83::build()));
    let mut dense = Engine::new(Model::compile(&crate::sm83::build()));
    dense.force_fallback = true;
    let cfg = fast.dense().cfg.clone();
    let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
    let base = state(&cfg, 0x2545F4914F6CDD1D);
    assert_eq!(fast.fill(&base, &outputs), dense.fill(&base, &outputs));
    let bad = tok::input(&cfg, 0, (base[0] as u8).wrapping_add(1));
    fast.qkv(bad, 0);
    let slot = fast.slots.of(bad, 0);
    for v in &mut fast.qkv[slot].as_mut().unwrap().v {
        *v = 0.5;
    }
    let mut poisoned = base.clone();
    poisoned[0] = bad;
    fast.fill(&poisoned, &outputs);
    assert_eq!((fast.stats.fast_passes, fast.stats.fallback_passes), (1, 1));
    let m = &fast.marks;
    assert_eq!((m.touched.len(), m.acc.iter().filter(|&&a| a != 0).count(), m.chan.iter().filter(|&&c| c).count()), (0, 0, 0));
    for seed in 1..=6u64 {
        let s = state(&cfg, seed.wrapping_mul(0x9E3779B97F4A7C15));
        assert_eq!(fast.fill(&s, &outputs), dense.fill(&s, &outputs), "pass after the failed one, seed {seed}");
    }
    assert_eq!((fast.stats.fast_passes, fast.stats.fallback_passes), (7, 1));
}

/// Twelve levels of 96 wires, each level reading only the one before: 1,168 wires,
/// but under 200 live at once. The channel
/// plan reuses dead wires' channels, with clearing units for 0/1 and for −1/0 wires
/// (the NOR nodes fold a constant), and the fast path must decode what the dense
/// pass and the circuit compute.
#[test]
fn reused_channels_match_the_dense_pass_and_the_circuit() {
    use crate::circuit::{byte, Circuit, Lin, Step};
    let mut c = Circuit::default();
    let (x, y) = (byte(c.input("X")), byte(c.input("Y")));
    let mut prev: Vec<Lin> = x.iter().chain(&y).cloned().collect();
    for level in 0..12 {
        prev = (0..96)
            .map(|k| {
                let (a, b) = (prev[k % prev.len()].clone(), prev[(7 * k + 3) % prev.len()].clone());
                let name = format!("n{level}.{k}");
                match k % 3 {
                    0 => c.node(&name, vec![Step { coef: 1, lin: Lin::konst(0), theta: 0 }, Step { coef: -1, lin: a + b, theta: 1 }]),
                    1 => c.and(&name, &[a, b]),
                    _ => c.eq(&name, a + b, 1),
                }
            })
            .collect();
    }
    c.output("Z", std::array::from_fn(|i| prev[i].clone()));
    c.output("W", std::array::from_fn(|i| prev[8 + i].clone()));
    let model = Model::compile(&c);
    assert!(model.cfg.hidden < 1168, "the wires share channels: hidden {}", model.cfg.hidden);
    assert!(model.unit_names.iter().flatten().any(|n| n == "clear"));
    let mut fast = Engine::new(model);
    assert_eq!(fast.exact(), Ok(()));
    let cfg = fast.dense().cfg.clone();
    let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
    for seed in 1..=40u64 {
        let tokens = state(&cfg, seed.wrapping_mul(0x9E3779B97F4A7C15));
        let got = fast.fill(&tokens, &outputs);
        assert_eq!(got, fast.dense().forward(&tokens, false)[cfg.n_inputs..].to_vec(), "seed {seed}");
        let input = std::collections::BTreeMap::from([("X".to_string(), tokens[0] as u8), ("Y".to_string(), (tokens[1] - tok::input(&cfg, 1, 0)) as u8)]);
        let want = c.eval(&input);
        assert_eq!(got.iter().map(|&t| tok::output_value(&cfg, t).unwrap()).collect::<Vec<u8>>(), vec![want["Z"], want["W"]], "seed {seed}");
    }
    assert_eq!(fast.stats.fallback_passes, 0);
}

/// A token at a position the protocol never gives it has no cache slot, so the
/// pass takes the dense model rather than another pair's cached entry.
#[test]
fn token_outside_its_home_takes_the_dense_path() {
    let mut fast = Engine::new(Model::compile(&crate::sm83::build()));
    let cfg = fast.dense().cfg.clone();
    let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
    let mut tokens = state(&cfg, 0x9E3779B97F4A7C15);
    tokens[0] = tok::output(&cfg, 0x42);
    let want: Vec<usize> = fast.dense().forward(&tokens, false)[cfg.n_inputs..].to_vec();
    assert_eq!(fast.fill(&tokens, &outputs), want);
    assert_eq!((fast.stats.fast_passes, fast.stats.fallback_passes), (0, 1));
}

/// The block walk (`Marks::blocks`) sends exactly what the scalar walk (`Marks::layer`)
/// sends, layer call by layer call, and leaves every unit's gate at the same value,
/// over packed layers whose units often share their inputs (so blocks form, with fused
/// pairs and plain units in one block) and random 0/1 changes, from a fresh state on.
#[test]
fn block_walk_sends_what_the_scalar_walk_sends() {
    use super::int::IntLayer;
    use super::run::Marks;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |m: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % m
    };
    for _ in 0..20 {
        // Channels from `shared` on have one reader each, for the single bucket.
        let (channels, shared, n) = (64, 48, 300);
        let pool: Vec<Vec<u16>> = (0..12).map(|_| {
            let mut s: Vec<u16> = (0..1 + next(6)).map(|_| next(shared as u64) as u16).collect();
            s.sort_unstable();
            s.dedup();
            s
        }).collect();
        let (mut rows, mut bias, mut out, mut c, mut cap) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for u in 0..n {
            let inputs = if next(3) == 0 { pool[next(pool.len() as u64) as usize].clone() } else {
                let mut s: Vec<u16> = (0..1 + next(5)).map(|_| next(shared as u64) as u16).collect();
                s.sort_unstable();
                s.dedup();
                s
            };
            for ch in inputs {
                rows.push((u, ch, [-3, -2, -1, 1, 2, 3][next(6) as usize]));
            }
            bias.push(next(9) as i32 - 4);
            out.push(next(channels as u64) as u16);
            c.push(if next(2) == 0 { 1 } else { -2 });
            cap.push(if next(3) == 0 { 1 } else { i32::MAX });
        }
        for ch in shared..channels {
            rows.push((next(n as u64) as usize, ch as u16, [-3, -2, -1, 1, 2, 3][next(6) as usize]));
        }
        let l = IntLayer::new(channels, rows, bias, out, c, cap);
        let order = l.packed_order();
        let l = l.permuted(&order).with_blocks();
        assert!((0..channels).any(|ch| !l.blocks().get(ch).0.is_empty()), "the layers have blocks");
        assert!((0..channels).any(|ch| l.blocks().single()[ch] != super::int::NOT_SINGLE), "the layers have single-entry channels");
        let new_marks = || Marks { chan: vec![false; channels], acc: vec![0; channels], ..Marks::default() };
        let (mut a, mut b) = (new_marks(), new_marks());
        let (mut ga, mut gb) = (l.start_padded(), l.start_padded());
        let mut x = vec![0i32; channels];
        let sent = |m: &mut Marks| {
            let mut v = Vec::new();
            m.drain(|ch, d| if d != 0 { v.push((ch, d)) });
            v.sort_unstable();
            v
        };
        for step in 0..200 {
            let mut changes: Vec<(u16, i32)> = Vec::new();
            if step > 0 {
                for _ in 0..1 + next(4) {
                    let ch = next(channels as u64) as usize;
                    if changes.iter().all(|&(c, _)| c as usize != ch) {
                        let dx = 1 - 2 * x[ch];
                        x[ch] += dx;
                        changes.push((ch as u16, dx));
                    }
                }
            }
            a.layer::<true>(&l, &mut ga, &changes, step == 0, |_, _, _, _| {});
            // The block walk takes the single-entry channels as its single bucket, as the
            // template routes them.
            let (singles, rest): (Vec<(u16, i32)>, Vec<(u16, i32)>) = changes.iter().partition(|&&(ch, _)| l.blocks().single()[ch as usize] != super::int::NOT_SINGLE);
            let (one_blocks, rest): (Vec<(u16, i32)>, Vec<(u16, i32)>) = rest.iter().partition(|&&(ch, _)| l.blocks().one_block()[ch as usize] != super::int::NOT_SINGLE);
            b.blocks(&l, &mut gb, &rest, &singles, &one_blocks, step == 0);
            assert_eq!(sent(&mut a), sent(&mut b), "step {step}");
            assert_eq!(ga, gb, "step {step}");
        }
    }
}

/// In an overlay the inputs share one identity channel per gather group, so the output
/// slots' identities end well before `seq` identity channels would. A circuit unit may
/// read the last output identity (a position wire does) but not the channel just past
/// it, the token's own byte, which only tokens set.
#[test]
fn a_unit_reading_past_the_output_identities_is_refused() {
    use crate::llama::{Layer, Layout, CONSTANTS, K};
    let (n_inputs, n_outputs) = (8, 3);
    let lay = Layout { n_const: CONSTANTS.len(), seq: n_inputs + n_outputs, wires: 0, wire_ch: Vec::new(), occupants: Vec::new(), hots: false, memory: false, positions: (0, 0, 0), offsets: vec![0, 4], ids: vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4], own_outputs: None };
    let ids = lay.output_ids(n_inputs);
    assert_eq!(ids, 4..7);
    assert_eq!(lay.own_bit(0), ids.end);
    let d = 16;
    let cfg = Config { hidden: d, intermediate: 1, layers: 1, heads: 0, kv_heads: 0, vocab: 0, n_inputs, n_outputs, n_const: CONSTANTS.len(), families: 0 };
    // The identities and the own byte are token channels; the rest are wires.
    let token: Vec<bool> = (0..d).map(|c| (CONSTANTS.len()..lay.own_bit(8)).contains(&c)).collect();
    let readable = super::plan::readable(&token, ids.clone());
    let reading = |ch: usize| {
        let mut gate = vec![0.0; d];
        gate[ch] = K;
        let mut up = vec![0.0; d];
        up[0] = 1.0;
        Layer { q: Vec::new(), k: Vec::new(), v: Vec::new(), o: Vec::new(), gate, up, down: vec![0.0; d] }
    };
    let largest = vec![1i64; d];
    assert!(super::int::IntLayer::from(&reading(ids.end - 1), &cfg, &readable, &largest).is_ok());
    let past = super::int::IntLayer::from(&reading(ids.end), &cfg, &readable, &largest).err();
    assert_eq!(past.as_deref(), Some(format!("unit 0 reads channel {}", ids.end).as_str()));
    assert!(super::int::IntLayer::from(&reading(lay.own_bit(8)), &cfg, &readable, &largest).is_ok(), "a wire channel");
}

/// An engine over the GBC CPU checkpoint through a view that keeps every slot where it is.
fn viewed() -> Engine {
    let model = Model::compile(&crate::sm83::build());
    let spec = super::StreamSpec { slots: model.slots.clone(), at: (0..model.cfg.seq()).map(Some).collect(), n_inputs: model.cfg.n_inputs, families: 0, part: 0 };
    Engine::streams(model, &[spec]).pop().unwrap()
}

/// Passes through a sequence decode what passes over the whole slice decode: with a few
/// inputs changed each pass, over two position lists, across a slice pass on the same
/// engine (its placement and the store both out of step afterwards) and a dense
/// fallback. The sequence's passes place only the listed changes and write only the
/// positions whose decode ran.
#[test]
fn sequence_passes_decode_what_slice_passes_decode() {
    use crate::sequence::{Positions, Sequence};
    let (mut seq_engine, mut slice_engine) = (viewed(), viewed());
    let cfg = seq_engine.cfg().clone();
    let lists = [Positions::new((cfg.n_inputs..cfg.seq()).collect()), Positions::new((cfg.n_inputs + 5..cfg.seq()).collect())];
    let mut seq = Sequence::new(state(&cfg, 0x2545F4914F6CDD1D));
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move |m: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % m
    };
    let (mut sparse, mut by_changes) = (0, 0);
    for pass in 0..300 {
        for _ in 0..next(4) {
            let j = next(cfg.n_inputs as u64) as usize;
            seq.set(j, tok::input(&cfg, j, next(256) as u8));
        }
        let list = &lists[(pass % 5 == 3) as usize];
        if pass % 41 == 17 {
            // A slice pass on other tokens: the engine's placement and state leave the sequence's.
            let other = state(&cfg, next(u64::MAX));
            assert_eq!(seq_engine.fill(&other, list), slice_engine.fill(&other, list), "slice pass {pass}");
        }
        seq_engine.force_fallback = pass % 67 == 29;
        by_changes += (seq_engine.view.as_ref().unwrap().placed_stamp == Some(seq.stamp())) as u32;
        seq_engine.fill_seq(&mut seq, list);
        let got: Vec<usize> = list.iter().map(|&p| seq.decoded()[p]).collect();
        assert_eq!(got, slice_engine.fill(seq.tokens(), list), "pass {pass}");
        assert!(seq.fresh().iter().all(|&p| list.contains(&(p as usize))), "pass {pass}: a position outside the pass was written");
        sparse += (seq.fresh().len() < list.len()) as u32;
    }
    assert_eq!((seq_engine.stats.fallback_passes, slice_engine.stats.fallback_passes), (5, 0));
    assert!(sparse > 20 && sparse < 280 && by_changes > 280, "{sparse} passes wrote part of their positions, {by_changes} placed by the change list");
}

/// The guard on the change list: a token that changed with no record of it stops the
/// next checked pass.
#[test]
#[should_panic(expected = "the sequence lists no change there")]
fn an_unrecorded_change_stops_the_pass() {
    use crate::sequence::{Positions, Sequence};
    let mut e = viewed();
    let cfg = e.cfg().clone();
    let list = Positions::new((cfg.n_inputs..cfg.seq()).collect());
    let mut seq = Sequence::new(state(&cfg, 0x2545F4914F6CDD1D));
    e.fill_seq(&mut seq, &list);
    seq.set_unrecorded(3, tok::input(&cfg, 3, (seq.get(3) as u8).wrapping_add(1)));
    e.fill_seq(&mut seq, &list);
}

/// The guard on the decoded store: a stored token that is not the engine's stops the
/// next checked pass, though that pass decodes nothing again.
#[test]
#[should_panic(expected = "the sequence's store holds token")]
fn a_store_out_of_step_stops_the_pass() {
    use crate::sequence::{Positions, Sequence};
    let mut e = viewed();
    let cfg = e.cfg().clone();
    let list = Positions::new((cfg.n_inputs..cfg.seq()).collect());
    let mut seq = Sequence::new(state(&cfg, 0x2545F4914F6CDD1D));
    // The first pass decodes every position, so the second writes them all back; the
    // third writes only what it decoded again, which is nothing.
    for _ in 0..3 {
        e.fill_seq(&mut seq, &list);
    }
    let kept = (cfg.n_inputs..cfg.seq()).find(|p| !seq.fresh().contains(&(*p as u32))).expect("a position the third pass did not decode again");
    seq.corrupt_decoded(kept, 1);
    e.fill_seq(&mut seq, &list);
}

/// A push to a full bucket is dropped, reaches no other bucket, and is found by
/// `overflowed`, before and after the bucket is cleared, until `clear_all`.
#[test]
fn a_full_bucket_drops_the_push_and_says_so() {
    use super::run::Buckets;
    // Bucket 0 holds two channels' changes, bucket 1 one, bucket 3 one.
    let mut q = Buckets::new([0b0001, 0b0011, 0b1000].into_iter());
    q.route(0b0011, 5, 1);
    q.route(0b1000, 7, -1);
    q.route(0, 9, 3);
    assert!(!q.overflowed());
    assert_eq!((q.get(0), q.get(1), q.get(2), q.get(3)), (&[(5, 1)][..], &[(5, 1)][..], &[][..], &[(7, -1)][..]));
    // A second change for bucket 1, which holds one.
    q.route(0b0011, 6, 2);
    assert!(q.overflowed());
    assert_eq!((q.get(0), q.get(1), q.get(2), q.get(3)), (&[(5, 1), (6, 2)][..], &[(5, 1)][..], &[][..], &[(7, -1)][..]));
    q.clear(1);
    assert!(q.overflowed() && q.is_empty(1));
    q.clear_all();
    assert!(!q.overflowed() && (0..64).all(|b| q.is_empty(b)));
    // No bucket holds anything by default.
    let mut none = Buckets::default();
    none.route(1, 0, 1);
    assert!(none.overflowed());
}

/// The routing update (`Marks::route_layer`) leaves every gate where the scalar walk
/// (`Marks::layer`) leaves it and sends what it sends, and keeps each home's own route
/// totals at the sum of its owners' parts over the gates (`Owner::part`), marking every
/// home whose total moved. The units are plain and fused pairs with no owner, one or
/// two, from a fresh state on.
#[test]
fn routing_update_keeps_the_own_totals_and_sends_what_the_scalar_walk_sends() {
    use super::int::IntLayer;
    use super::plan::{route_span, Owner, RouteTag};
    use super::run::Marks;
    use super::ROUTE_CAP;
    let mut rng = 0xD1B5_4A32_D192_ED03u64;
    let mut next = move |m: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % m
    };
    let (mut seconds, mut moved_homes, mut sent_any, mut linears) = (0, 0, 0, 0);
    for _ in 0..20 {
        let (channels, n, seq) = (40, 120, 9);
        let (mut rows, mut bias, mut out, mut c, mut cap) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for u in 0..n {
            let mut inputs: Vec<u16> = (0..1 + next(3)).map(|_| next(channels as u64) as u16).collect();
            inputs.sort_unstable();
            inputs.dedup();
            for ch in inputs {
                rows.push((u, ch, [-3, -2, -1, 1, 2, 3][next(6) as usize]));
            }
            // Low biases, so that many units stay at 0 without their owner's token.
            bias.push(next(60) as i32 - 55);
            out.push(next(channels as u64) as u16);
            c.push(if next(2) == 0 { 1 } else { -2 });
            cap.push(if next(2) == 0 { 1 } else { i32::MAX });
        }
        let l = IntLayer::new(channels, rows, bias, out, c, cap);
        let mut owners = vec![[Owner::none(seq); 2]; n];
        for os in &mut owners {
            // No owner, one, or two.
            let held = [0, 1, 1, 2][next(4) as usize];
            for o in &mut os[..held] {
                *o = Owner { p: next(seq as u64) as u16, bit: next(ROUTE_CAP as u64) as u8, own: next(60) as i32 - 2, c: [-2, 1, 3][next(3) as usize] };
            }
        }
        seconds += owners.iter().filter(|o| o[1].c != 0).count();
        // The channels below 8 hold nibbles and the rest bits, as the steps below move them.
        let start = l.start();
        let linear: Vec<bool> = (0..n).map(|u| owners[u][1].c == 0 && owners[u][0].linear(start[u], l.cap()[u] == 1, route_span(&l, &[false; 40], u, |ch| (0, if ch < 8 { 15 } else { 1 })))).collect();
        linears += linear.iter().filter(|&&x| x).count();
        let gate = l.gate().tagged(|u, pair| RouteTag { pair, second: owners[u as usize][1].c != 0, linear: linear[u as usize], owner: owners[u as usize][0] });
        let totals = |g: &[i32]| {
            let mut t = vec![[0i32; ROUTE_CAP]; seq + 1];
            for (u, os) in owners.iter().enumerate() {
                for o in os {
                    t[o.p as usize][o.bit as usize] += o.part(g[u], l.cap()[u] == 1);
                }
            }
            t
        };
        let new_marks = || Marks { chan: vec![false; channels], acc: vec![0; channels], ..Marks::default() };
        let (mut a, mut b) = (new_marks(), new_marks());
        let (mut ga, mut gb) = (l.start(), l.start());
        let mut own_route = totals(&gb);
        let mut x = vec![0i32; channels];
        let sent = |m: &mut Marks| {
            let mut v = Vec::new();
            m.drain(|ch, d| if d != 0 { v.push((ch, d)) });
            v.sort_unstable();
            v
        };
        for step in 0..200 {
            let mut changes: Vec<(u16, i32)> = Vec::new();
            if step > 0 {
                for _ in 0..1 + next(4) {
                    let ch = next(channels as u64) as usize;
                    if changes.iter().all(|&(c, _)| c as usize != ch) {
                        // Mostly 0/1 wires, some nibbles.
                        let to = if ch < 8 { next(16) as i32 } else { 1 - x[ch] };
                        if to != x[ch] {
                            changes.push((ch as u16, to - x[ch]));
                            x[ch] = to;
                        }
                    }
                }
            }
            let before = own_route.clone();
            let mut stale = vec![0u8; seq + 1];
            let ua = a.layer::<true>(&l, &mut ga, &changes, step == 0, |_, _, _, _| {});
            let ub = b.route_layer(&l, &gate, &owners, &mut gb, &changes, step == 0, &mut own_route, &mut stale);
            let (sa, sb) = (sent(&mut a), sent(&mut b));
            sent_any += !sa.is_empty() as usize;
            assert_eq!(sa, sb, "step {step}: the sends");
            assert_eq!((ua, &ga), (ub, &gb), "step {step}: the gates");
            assert_eq!(own_route, totals(&gb), "step {step}: the own route totals");
            for p in 0..=seq {
                if own_route[p] != before[p] {
                    moved_homes += 1;
                    assert_eq!(stale[p], 1, "step {step}: home {p} moved and is not stale");
                }
            }
        }
    }
    assert!(seconds > 300 && moved_homes > 3000 && sent_any > 1000 && linears > 100 && linears < 1500, "the rounds cover {seconds} second owners, {moved_homes} moved homes, {sent_any} steps with sends, {linears} linear units of 2400");
}

/// Where `Owner::linear` holds, the owner's part is c · (gate + own) at every gate the
/// wires can reach and the blank row's activation is 0 there, for plain units and fused
/// pairs; and it holds wherever that is so over the whole span.
#[test]
fn a_linear_owner_moves_by_its_weight_over_the_whole_span() {
    use super::plan::Owner;
    let (mut held, mut refused) = (0, 0);
    for g in -12..6 {
        for own in -3..14 {
            for (lo, hi) in [(0i64, 0i64), (0, 1), (0, 5), (-2, 0), (-1, 3), (0, 9)] {
                for pair in [false, true] {
                    let o = Owner { p: 0, bit: 0, own, c: 3 };
                    // Gates are kept as g - 1: the blank row's activation is the clamp of v, the owner's of v + own.
                    let clamp = |v: i64| if pair { v.clamp(0, 1) } else { v.max(0) };
                    let exact = (lo..=hi).all(|s| {
                        let v = g as i64 + 1 + s;
                        clamp(v) == 0 && clamp(v + own as i64) == v + own as i64 && o.part((g as i64 + s) as i32, pair) as i64 == 3 * (v + own as i64)
                    });
                    assert_eq!(o.linear(g, pair, (lo, hi)), exact, "gate {g}, own {own}, wires {lo} to {hi}, pair {pair}");
                    held += exact as usize;
                    refused += !exact as usize;
                }
            }
        }
    }
    assert!(held > 100 && refused > 100, "{held} linear and {refused} not");
    assert!(!Owner::none(0).linear(-5, false, (0, 1)), "no owner, nothing to move");
}
