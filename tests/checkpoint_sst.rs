//! The compiled SM83 checkpoint: the engine gives the dense HF-semantics pass's
//! tokens on both of its paths, and the engine-backed CPU passes SingleStepTests.

use gbcformer::engine::Engine;
use gbcformer::llama::{tok, Denoiser};
use gbcformer::sm83::{self, CheckpointCore};
use gbcformer::sst;
#[path = "common/sst.rs"]
mod sst_data;

fn engine() -> Engine {
    let e = sm83::engine();
    let c = &e.dense().cfg;
    eprintln!("hidden {} intermediate {} layers {} heads {} vocab {} params {:.1}M", c.hidden, c.intermediate, c.layers, c.heads, c.vocab, c.params() as f64 / 1e6);
    e.exact().expect("integer fast path preconditions");
    e
}

/// Random input bytes with every output masked, as pass 1 of a step sees them.
fn random_states(cfg: &gbcformer::llama::Config, n: usize) -> Vec<Vec<usize>> {
    let mut rng = 0x2545F4914F6CDD1Du64;
    (0..n)
        .map(|_| {
            (0..cfg.seq())
                .map(|j| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    if j < cfg.n_inputs { tok::input(cfg, j, rng as u8) } else { tok::output_mask(cfg, j) }
                })
                .collect()
        })
        .collect()
}

#[test]
fn engine_matches_dense_pass() {
    let mut fast = engine();
    let cfg = fast.dense().cfg.clone();
    let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
    for (case, tokens) in random_states(&cfg, 12).iter().enumerate() {
        for causal in [false, true] {
            let want: Vec<usize> = fast.dense().forward(tokens, causal)[cfg.n_inputs..].to_vec();
            assert_eq!(fast.fill(tokens, &outputs), want, "case {case} causal={causal}");
        }
    }
    assert_eq!((fast.stats.fast_passes, fast.stats.fallback_passes), (24, 0));
}

#[test]
fn forced_fallback_matches_fast_path() {
    let mut fast = engine();
    let mut slow = engine();
    slow.force_fallback = true;
    let cfg = fast.dense().cfg.clone();
    let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
    for tokens in random_states(&cfg, 3) {
        assert_eq!(slow.fill(&tokens, &outputs), fast.fill(&tokens, &outputs));
    }
    assert_eq!((slow.stats.fast_passes, slow.stats.fallback_passes), (0, 3));
}

#[test]
fn checkpoint_matches_singlesteptests() {
    let Some(dir) = sst_data::dir() else { return };
    let filter = std::env::var("SST").unwrap_or_default();
    let mut core = CheckpointCore::new(engine());
    let t = std::time::Instant::now();
    let (cases, failed) = sst::run_all(&mut core, dir, &filter, true);
    let s = &core.model().stats;
    eprintln!("checkpoint: {cases} cases, {failed} failed, {:.1}s, {} passes ({} fast, {} fallback)", t.elapsed().as_secs_f64(), s.passes, s.fast_passes, s.fallback_passes);
    assert_eq!(failed, 0);
    assert_eq!(s.fallback_passes, 0);
}

mod memory {
    use gbcformer::engine::Engine;
    use gbcformer::llama::{tok, Denoiser, Model};
    use gbcformer::memory::attn::Params;
    use gbcformer::memory::image::Image;
    use gbcformer::memory::layout::{Banking, Mapper};
    use gbcformer::sm83::{build_mem, MemCheckpointCore};
    use gbcformer::sst;

    /// The memory checkpoint's engine over a 32 KiB ROM-only cartridge whose ROM region
    /// is writable (SingleStepTests put program bytes anywhere), with 8 KiB of RAM.
    fn memory_engine() -> Engine {
        let mut e = Engine::new(Model::compile_mem(&build_mem(), Params::memory()));
        let c = &e.dense().cfg;
        eprintln!("memory checkpoint: hidden {} intermediate {} layers {} heads {} vocab {} params {:.1}M nonzeros {}", c.hidden, c.intermediate, c.layers, c.heads, c.vocab, c.params() as f64 / 1e6, e.nonzeros());
        e.exact().expect("integer fast path preconditions with memory heads");
        let banking = Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap();
        e.insert(Image::new(&[0; 0x8000], 0x2000, banking).with_writable_rom());
        e
    }

    /// Random input bytes with the late slots and every output masked, as pass 1 sees them.
    fn random_states(cfg: &gbcformer::llama::Config, n: usize) -> Vec<Vec<usize>> {
        let mut rng = 0x2545F4914F6CDD1Du64;
        (0..n)
            .map(|_| {
                (0..cfg.seq())
                    .map(|j| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        if j < 15 { tok::input(cfg, j, rng as u8) } else if j < cfg.n_inputs { tok::input_mask(cfg, j) } else { tok::output_mask(cfg, j) }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn memory_engine_matches_dense_pass_with_cache() {
        let mut fast = memory_engine();
        let cfg = fast.dense().cfg.clone();
        let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
        for (case, tokens) in random_states(&cfg, 4).iter().enumerate() {
            let want: Vec<usize> = fast.dense().forward_cached(&fast.memory.as_ref().unwrap().kv, tokens, gbcformer::memory::attn::READ_AT).0[cfg.n_inputs..].to_vec();
            assert_eq!(fast.fill(tokens, &outputs), want, "case {case}");
        }
        assert_eq!((fast.stats.fast_passes, fast.stats.fallback_passes), (4, 0), "refused: {:?}", fast.stats.refused);
        let mut slow = memory_engine();
        slow.force_fallback = true;
        for tokens in random_states(&cfg, 2) {
            assert_eq!(slow.fill(&tokens, &outputs), fast.fill(&tokens, &outputs));
        }
        assert_eq!((slow.stats.fast_passes, slow.stats.fallback_passes), (0, 2));
    }

    /// The checkpoint holds every weight in f16, and the dense pass with every matmul
    /// input, query, key, value and softmax weight rounded to f16 (a half-tile runtime
    /// with an f16 cache) decodes what the engine decodes.
    #[test]
    fn memory_checkpoint_is_exact_in_f16() {
        let mut fast = memory_engine();
        let mut half = Model::compile_mem(&build_mem(), Params::memory());
        half.check_f16().expect("every weight f16-exact");
        half.tiles = gbcformer::llama::Tiles::F16;
        let cfg = half.cfg.clone();
        let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
        let mut cache = gbcformer::llama::KvCache::new(half.memory.as_ref().unwrap().heads.clone());
        cache.fill(&fast.memory.as_ref().unwrap().image);
        for (case, tokens) in random_states(&cfg, 4).iter().enumerate() {
            let want: Vec<usize> = half.forward_cached(&cache, tokens, gbcformer::memory::attn::READ_AT).0[cfg.n_inputs..].to_vec();
            assert_eq!(fast.fill(tokens, &outputs), want, "case {case}");
        }
    }

    /// The same with every product, partial sum and elementwise result in f16 (T2), the
    /// sums taken in three orders, over the whole memory as cached cells.
    #[test]
    fn memory_checkpoint_is_exact_in_f16_sums() {
        use gbcformer::llama::{Order, Tiles};
        let mut fast = memory_engine();
        let mut half = Model::compile_mem(&build_mem(), Params::memory());
        half.check_f16_acc().expect("every partial sum f16-exact");
        let cfg = half.cfg.clone();
        let outputs: Vec<usize> = (cfg.n_inputs..cfg.seq()).collect();
        let mut cache = gbcformer::llama::KvCache::new(half.memory.as_ref().unwrap().heads.clone());
        cache.fill(&fast.memory.as_ref().unwrap().image);
        for (case, tokens) in random_states(&cfg, 3).iter().enumerate() {
            half.tiles = Tiles::F16Acc([Order::Forward, Order::PositivesFirst, Order::NegativesFirst][case]);
            let want: Vec<usize> = half.forward_cached(&cache, tokens, gbcformer::memory::attn::READ_AT).0[cfg.n_inputs..].to_vec();
            assert_eq!(fast.fill(tokens, &outputs), want, "case {case}, {:?}", half.tiles);
        }
    }

    /// A bus with nothing behind it: the step below reads and writes only the model's memory.
    struct Empty;

    impl gbcformer::step::Bus for Empty {
        fn read(&mut self, _addr: u16) -> u8 {
            0
        }
        fn write(&mut self, _addr: u16, _value: u8) {}
        fn ie(&self) -> u8 {
            0
        }
        fn iflag(&self) -> u8 {
            0
        }
        fn ack(&mut self, _bits: u8) {}
    }

    /// A PUSH BC whose two writes fall in different lines keeps each write's record when
    /// the host reaches that write's M-cycle (`Core::land_write`), not when the step's
    /// outcome is decoded: a DMA byte or a line drawn before the write sees the old byte.
    #[test]
    fn a_write_record_lands_at_its_write_not_at_the_outcome() {
        use gbcformer::step::{Core, Cpu, Inputs};
        let mut core = MemCheckpointCore::new(memory_engine());
        core.load_bus(&[(0xC100, 0xC5), (0xC008, 0xAA), (0xC007, 0xBB)]).unwrap();
        let cpu = Cpu { b: 0x12, c: 0x34, sp: 0xC009, pc: 0xC100, ..Cpu::default() };
        let inp = Inputs { op: [0xC5, 0, 0], ..Inputs::default() };
        let req = gbcformer::step::request(&mut core, &cpu, &inp, |_| 0);
        assert_eq!(req.reads, [None, None]);
        let out = core.execute_on(&cpu, &inp, [0, 0], &Empty);
        assert_eq!(out.writes, [Some((0xC008, 0x12)), Some((0xC007, 0x34))]);
        let byte = |core: &MemCheckpointCore<Engine>, i: usize| {
            let p = core.facts().write_targets[i].unwrap();
            core.model().memory.as_ref().unwrap().image.line(p >> gbcformer::memory::layout::LINE_BITS)[p as usize % 8]
        };
        assert_eq!([byte(&core, 0), byte(&core, 1)], [0xAA, 0xBB], "the outcome kept a record before its write's M-cycle");
        core.land_write(0);
        assert_eq!([byte(&core, 0), byte(&core, 1)], [0x12, 0xBB], "write 0's record at write 0");
        core.land_write(1);
        assert_eq!([byte(&core, 0), byte(&core, 1)], [0x12, 0x34], "write 1's record at write 1");
    }

    #[test]
    fn memory_checkpoint_matches_singlesteptests() {
        let Some(dir) = crate::sst_data::dir() else { return };
        let filter = std::env::var("SST").unwrap_or_default();
        let mut engine = memory_engine();
        engine.memory.as_mut().unwrap().oracle_every = std::env::var("ORACLE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let mut core = MemCheckpointCore::new(engine);
        let t = std::time::Instant::now();
        let (cases, failed) = sst::run_all(&mut core, dir, &filter, true);
        let s = &core.model().stats;
        eprintln!("memory checkpoint: {cases} cases, {failed} failed, {:.1}s, {} passes ({} fast, {} fallback), {} host-served steps, {} fetched bytes kept as HOST records; {s}", t.elapsed().as_secs_f64(), s.passes, s.fast_passes, s.fallback_passes, core.host_served_steps, s.host_fetches);
        assert!(!filter.is_empty() || s.host_fetches > 0, "no case fetched from an address the host serves");
        assert_eq!(failed, 0);
        assert_eq!(s.fallback_passes, 0);
    }
}
