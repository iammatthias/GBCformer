//! The step oracle catches a wrong result. The APU checkpoint's engine runs random segments
//! with every fast pass also run through its dense pass (`Engine::step_oracle`): unchanged,
//! every pass agrees; with one weight of the dense pass changed (`Engine::dense_mut`), or
//! one token the fast pass decoded flipped (`Engine::inject_fault`), the oracle stops the run with
//! its own message. The same passes run in all three, so a perturbation the oracle never
//! sampled cannot pass for a catch.

use gbcformer::engine::Engine;
use gbcformer::llama::{tok, Denoiser};

const PASSES: usize = 40;

fn run(e: &mut Engine) {
    let cfg = e.cfg().clone();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..PASSES {
        let tokens: Vec<usize> = (0..cfg.seq())
            .map(|p| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if p < cfg.n_inputs { tok::input(&cfg, p, rng as u8) } else { tok::output_mask(&cfg, p) }
            })
            .collect();
        e.fill(&tokens, &(cfg.n_inputs..cfg.seq()).collect::<Vec<_>>());
    }
}

fn engine() -> Engine {
    let mut e = gbcformer::apu_circuit::engine();
    e.step_oracle = 1;
    e
}

#[test]
fn the_step_oracle_passes_the_unchanged_engine() {
    let mut e = engine();
    run(&mut e);
    assert_eq!((e.stats.fast_passes, e.stats.step_checks, e.stats.fallback_passes), (PASSES as u64, PASSES as u64, 0));
}

#[test]
#[should_panic(expected = "step oracle: fast pass")]
fn the_step_oracle_catches_one_changed_weight() {
    let mut e = engine();
    // The LM head's weight from the constant channel (always 1) to the byte token 0x5A: the
    // dense pass now decodes 0x5A at every output position, the fast pass what it did.
    let model = e.dense_mut();
    let (hidden, t) = (model.cfg.hidden, tok::output(&model.cfg, 0x5A));
    model.lm_head[t * hidden] = 1.0e6;
    run(&mut e);
}

#[test]
#[should_panic(expected = "step oracle: fast pass")]
fn the_step_oracle_catches_one_changed_token() {
    let mut e = engine();
    e.inject_fault(1);
    run(&mut e);
}

/// The same through a sequence: the oracle reads the tokens from the sequence's store, as
/// a device does, so a wrong token there stops the run.
#[test]
#[should_panic(expected = "step oracle: fast pass")]
fn the_step_oracle_catches_one_changed_token_in_a_sequence_s_store() {
    use gbcformer::sequence::{Positions, Sequence};
    let mut e = engine();
    let cfg = e.cfg().clone();
    let list = Positions::new((cfg.n_inputs..cfg.seq()).collect());
    let mut seq = Sequence::new((0..cfg.seq()).map(|p| if p < cfg.n_inputs { tok::input(&cfg, p, 0) } else { tok::output_mask(&cfg, p) }).collect());
    e.fill_seq(&mut seq, &list);
    assert_eq!((e.stats.fast_passes, e.stats.step_checks), (1, 1));
    seq.set(0, tok::input(&cfg, 0, 1));
    e.inject_fault(1);
    e.fill_seq(&mut seq, &list);
}
