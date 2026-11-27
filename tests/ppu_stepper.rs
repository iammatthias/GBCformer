//! The FIFO stepper (`ppu_circuit::stepper`, `docs/ppu-in-weights.md` §14.2): its circuit
//! computes exactly `fifo::State::step` and `State::new` on every pass of random lines with
//! writes, the checkpoint's tokens equal the circuit's on every pass and the dense pass's on
//! a sample, and a line driven by the checkpoint alone draws what `fifo::draw` draws.

use gbcformer::circuit::Circuit;
use gbcformer::llama::{tok, Denoiser};
use gbcformer::machine::fifo::{self, Phase, Sample, State};
use gbcformer::machine::ppu::Mode;
use gbcformer::memory::layout::PAL_REGS;
use gbcformer::memory::Rng;
use gbcformer::ppu_circuit::stepper::{self, Constants, Decoded, Stepper};
use std::collections::BTreeMap;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn constants(s: &fifo::Start) -> Constants {
    Constants { ly: s.ly, mode: s.mode, double: s.double, blocked: s.blocked }
}

/// The fields of `got` that differ from `want`'s.
fn diff(want: &State, got: &State) -> String {
    let (a, b) = (stepper::fields(want), stepper::fields(got));
    a.iter().filter(|(k, v)| b[*k] != **v).map(|(k, v)| format!("{k} {} want {v}; ", b[k])).collect()
}

/// What a pass decoded against what the reference's pass did; empty when equal.
fn check(got: &Decoded, p: &fifo::Pass) -> String {
    let mut why = String::new();
    if got.state != p.after {
        why += &diff(&p.after, &got.state);
    }
    if got.out != p.out {
        why += &format!("out {:?} want {:?}; ", got.out, p.out);
    }
    if p.out.done && (got.window, got.dots) != (p.after.window_after(), p.after.mode0_dots()) {
        why += &format!("window {:?} dots {} want {:?} {}; ", got.window, got.dots, p.after.window_after(), p.after.mode0_dots());
    }
    why
}

#[derive(Default)]
struct Tally {
    lines: usize,
    passes: usize,
    by_phase: BTreeMap<String, usize>,
    by_line: BTreeMap<String, usize>,
    bad: BTreeMap<String, (usize, String)>,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        self.lines += o.lines;
        self.passes += o.passes;
        for (k, v) in o.by_phase {
            *self.by_phase.entry(k).or_default() += v;
        }
        for (k, v) in o.by_line {
            *self.by_line.entry(k).or_default() += v;
        }
        for (k, (n, first)) in o.bad {
            let e = self.bad.entry(k).or_insert((0, first));
            e.0 += n;
        }
    }
    fn fail(&mut self, kind: String, first: impl FnOnce() -> String) {
        let e = self.bad.entry(kind).or_insert((0, String::new()));
        e.0 += 1;
        if e.1.is_empty() {
            e.1 = first();
        }
    }
}

/// The kinds of line a trace took, for the coverage report.
fn kinds(s: &fifo::Start, trace: &[fifo::Pass]) -> Vec<&'static str> {
    let mut k = vec![match (s.mode, s.double) {
        (Mode::Dmg, _) => "DMG",
        (Mode::Cgb, false) => "CGB single speed",
        (Mode::Cgb, true) => "CGB double speed",
        (Mode::Compat, false) => "compatibility single speed",
        (Mode::Compat, true) => "compatibility double speed",
    }];
    let aborted = trace.windows(2).any(|p| p[0].before.phase == Phase::Sleep && p[1].before.phase == Phase::Render && p[0].out.applied);
    for (on, name) in [
        (aborted, "a DMG object fetch aborted"),
        (trace.iter().any(|p| p.out.applied), "a write applied"),
        (trace.iter().any(|p| p.before.cgb_wx_glitch), "the CGB's WX glitch"),
        (trace.iter().any(|p| p.before.tile_sel_glitch), "the CGB's tile-select glitch"),
        (trace.iter().any(|p| p.before.insert_bg_pixel), "a pixel inserted at WX"),
        (trace.iter().any(|p| p.before.phase == Phase::WindowOn), "the DMG's extra dot at WX 0"),
        (trace.iter().any(|p| p.before.bg.size == 0 && p.after.bg.size == 1), "DMG pixel inserted with the window off"),
        (trace.first().is_some_and(|p| p.after.wx_triggered), "window carried from WX 166"),
        (s.blocked, "blocked"),
    ] {
        if on {
            k.push(name);
        }
    }
    k
}

fn circuit_lines(c: &Circuit, seed: u64, from: usize, to: usize) -> Tally {
    let mut rng = Rng(seed);
    let mut t = Tally::default();
    for case in 0..to {
        let sample = Sample::random(&mut rng, case);
        if case >= from {
            check_line(c, case, &sample, &mut t);
        }
    }
    t
}

/// Line `case`'s start pass and every step pass through the circuit, into `t`.
fn check_line(c: &Circuit, case: usize, sample: &Sample, t: &mut Tally) {
    {
        let start = sample.start();
        let memory = stepper::image(&sample.vram, &sample.oam, &sample.display);
        let k = constants(&start);
        let got = stepper::eval(c, &stepper::start_inputs(k, start.window), &memory, k);
        let want = State::new(&start);
        if got.state != want {
            t.fail("start".into(), || format!("line {case}: {}", diff(&want, &got.state)));
        }
        let trace = fifo::trace(&start, &sample.writes);
        for p in &trace {
            let got = stepper::eval(c, &stepper::step_inputs(&p.before, p.pending), &memory, k);
            let why = check(&got, p);
            *t.by_phase.entry(format!("{:?}", p.before.phase)).or_default() += 1;
            if !why.is_empty() {
                t.fail(format!("{:?}/{:?}", p.before.phase, p.before.step), || format!("line {case} {:?} double {} pending {:?}: {why}", start.mode, start.double, p.pending));
            }
        }
        for name in kinds(&start, &trace) {
            *t.by_line.entry(name.into()).or_default() += 1;
        }
        t.lines += 1;
        t.passes += trace.len() + 1;
    }
}

fn report(t: &Tally) {
    println!("{} lines, {} passes (start passes included)", t.lines, t.passes);
    for (k, n) in &t.by_line {
        println!("  {n:6} lines: {k}");
    }
    for (k, n) in &t.by_phase {
        println!("  {n:8} passes in {k}");
    }
    for (k, (n, first)) in &t.bad {
        println!("DIFFERS {k}: {n} passes; first {first}");
    }
}

/// Every pass of `LINES` random lines (default 2,000) through the circuit, against
/// `State::step`, and every line's start pass against `State::new`. The lines run on
/// `THREADS` threads (default 4).
#[test]
fn the_circuit_computes_every_step_of_random_lines() {
    let (lines, threads) = (env("LINES", 2000), env("THREADS", 4));
    let c = stepper::build();
    let mut t = Tally::default();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|i| {
            let c = &c;
            scope.spawn(move || circuit_lines(c, 0x57E9_9E12_0000_0001, lines * i / threads, lines * (i + 1) / threads))
        }).collect();
        for h in handles {
            t.merge(h.join().expect("a circuit thread"));
        }
    });
    report(&t);
    assert!(t.bad.is_empty(), "{} kinds of pass differ", t.bad.len());
    for need in ["DMG", "CGB single speed", "CGB double speed", "compatibility single speed", "compatibility double speed", "a DMG object fetch aborted", "the CGB's WX glitch", "the CGB's tile-select glitch", "a pixel inserted at WX", "the DMG's extra dot at WX 0", "DMG pixel inserted with the window off", "window carried from WX 166", "blocked"] {
        assert!(t.by_line.get(need).is_some_and(|&n| n > 0), "no line took: {need}");
    }
}

/// `CP_LINES` random lines (default 200) drawn by the checkpoint alone: each pass's output
/// tokens equal the circuit's (every output slot) and the reference's pass, every
/// `DENSE_EVERY`th pass (default 2,000) also equals the dense pass over the KV cache, the
/// engine never falls back to the dense pass, and the line's colours, window and length are
/// `fifo::draw`'s.
#[test]
fn the_checkpoint_draws_lines_as_the_circuit_and_the_reference() {
    let (lines, dense_every) = (env("CP_LINES", 200), env("DENSE_EVERY", 2000));
    let c = stepper::build();
    let mut rng = Rng(0x57E9_9E12_0000_0002);
    let mut st: Option<Stepper> = None;
    let (mut passes, mut dense_checks) = (0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();
    for case in 0..lines {
        let sample = Sample::random(&mut rng, case);
        let start = sample.start();
        match st.as_mut() {
            Some(s) => s.load(&sample.vram, &sample.oam, &sample.display),
            None => st = Some(Stepper::new(stepper::engine(), &sample.vram, &sample.oam, &sample.display)),
        }
        let s = st.as_mut().unwrap();
        let memory = stepper::image(&sample.vram, &sample.oam, &sample.display);
        let k = constants(&start);
        let trace = fifo::trace(&start, &sample.writes);
        let want = fifo::draw(&start, &sample.writes);
        let mut n = 0usize;
        let mut compare = |s: &mut Stepper, inputs: Vec<u8>, ref_pass: Option<&fifo::Pass>, bad: &mut Vec<String>| -> Decoded {
            let got = s.run(&inputs);
            let map: BTreeMap<String, u8> = c.inputs.iter().map(|(n, _)| n.clone()).zip(inputs.iter().copied()).collect();
            let circuit = c.eval_mem(&map, &|l| memory.line(l));
            if let Some((slot, b)) = got.iter().find(|(slot, b)| circuit[*slot] != **b) {
                bad.push(format!("line {case} pass {n}: slot {slot} engine {b:#04x} circuit {:#04x}", circuit[slot]));
            }
            let e = s.engine();
            if passes % dense_every == 0 {
                let mem = e.memory.as_ref().expect("a memory checkpoint");
                let (tokens, positions) = s.tokens();
                let (dense, _) = e.dense().forward_cached(&mem.kv, tokens, mem.kv.len());
                let cfg = e.cfg();
                for (&p, name) in positions.iter().zip(c.outputs.iter().map(|(n, _)| n).filter(|n| !n.starts_with("WP") && !n.starts_with("WL"))) {
                    let d = tok::output_value(cfg, dense[p]).expect("a byte token");
                    if d != got[name] {
                        bad.push(format!("line {case} pass {n}: slot {name} engine {:#04x} dense {d:#04x}", got[name]));
                    }
                }
                dense_checks += 1;
            }
            passes += 1;
            let d = stepper::decode(k, &|name| got[name]);
            if let Some(p) = ref_pass {
                let why = check(&d, p);
                if !why.is_empty() {
                    bad.push(format!("line {case} pass {n} {:?}: against the reference: {why}", p.before.phase));
                }
            }
            n += 1;
            d
        };
        let first = compare(s, stepper::start_inputs(k, start.window), None, &mut bad).state;
        if first != State::new(&start) {
            bad.push(format!("line {case}: the start pass {}", diff(&State::new(&start), &first)));
        }
        let mut colours = [0; 160];
        let mut last = None;
        let mut i = 0;
        fifo::drive(first, &sample.writes, |state, pending| {
            let d = compare(s, stepper::step_inputs(state, pending), trace.get(i), &mut bad);
            i += 1;
            *state = d.state;
            if let Some((x, colour)) = d.out.pixel {
                colours[x as usize] = colour;
            }
            last = Some(d);
            d.out
        });
        let last = last.expect("a line takes passes");
        let got = fifo::Drawn { colours, window: last.window, dots: last.dots };
        // The stepper gives the dots to mode 0 (`fifo::mode0_dots`).
        let want = fifo::Drawn { dots: fifo::mode0_dots(want.dots, start.display[PAL_REGS as usize + 2], start.mode.rgb555()), ..want };
        if got != want {
            bad.push(format!("line {case} {:?}: the checkpoint drew {:?}, fifo::draw {:?}", start.mode, (got.window, got.dots), (want.window, want.dots)));
        }
        let e = s.engine();
        assert_eq!(e.stats.fallback_passes, 0, "line {case}: the engine took the dense fallback ({:?})", e.stats.refused);
        assert!(bad.is_empty(), "{bad:#?}");
    }
    println!("{lines} lines, {passes} passes, {dense_checks} against the dense pass, 0 fallbacks");
}

/// Paths the random lines rarely take, and two the generator never makes: the DMG's extra
/// dot at WX 0, a pixel inserted at WX, each of the four places a DMG object fetch aborts,
/// the lower half of 8x16 objects (rows 8 to 15, flipped or not), and lines with more than
/// ten objects, which the start pass's OAM scan cuts to ten. `RARE_SCAN` lines (default
/// 40,000) are traced, a tenth of them with tall objects and a tenth crowded; up to `QUOTA`
/// (default 6) of each kind go through the circuit on every pass and through the
/// checkpoint alone (`Stepper::draw`) against `fifo::draw`.
#[test]
fn the_rare_paths_and_tall_and_crowded_objects() {
    let (scan, quota) = (env("RARE_SCAN", 40_000), env("QUOTA", 6));
    let lcdc = gbcformer::memory::layout::PAL_REGS as usize;
    const KINDS: [&str; 8] = ["extra dot at WX 0", "pixel inserted at WX", "abort waiting on the fetcher", "abort before its tile", "abort before its low byte", "abort before its high byte", "8x16 lower half", "more than ten objects"];
    let mut rng = Rng(0x57E9_9E12_0000_0004);
    let mut alter = Rng(0x57E9_9E12_0000_0005);
    let mut picked: Vec<(usize, Sample)> = Vec::new();
    let mut found = [0usize; KINDS.len()];
    for case in 0..scan {
        let mut sample = Sample::random(&mut rng, case);
        let ly = sample.ly;
        match case % 10 {
            7 => {
                sample.display[lcdc] |= 4;
                for _ in 0..6 {
                    let o = &mut sample.oam[(alter.draw() % 40) as usize * 4..][..4];
                    o[0] = ly + 16 - 8 - (alter.draw() % 8) as u8;
                    o[1] = 1 + (alter.draw() % 167) as u8;
                    o[3] = alter.draw() as u8;
                }
            }
            8 => {
                for i in 0..16 {
                    let o = &mut sample.oam[(i * 2 + (alter.draw() % 2) as usize) * 4..][..2];
                    o[0] = ly + 16 - (alter.draw() % 8) as u8;
                    o[1] = (alter.draw() % 176) as u8;
                }
            }
            _ => {}
        }
        sample.objects = fifo::scan(&sample.oam, sample.display[lcdc], ly);
        let trace = fifo::trace(&sample.start(), &sample.writes);
        let resumed = |r: Phase| trace.windows(2).any(|p| p[0].before.phase == Phase::Sleep && p[0].before.resume == r && p[1].before.phase == Phase::Render && p[0].out.applied);
        let tall = case % 10 == 7 && trace.iter().any(|p| p.before.phase == Phase::FetchC && p.before.regs[3] & 4 != 0 && {
            let i = p.before.objects[0].0 as usize;
            (ly.wrapping_sub(sample.oam[i * 4]) & 15) >= 8
        });
        let crowded = (0..40).filter(|&i| (0..if sample.display[lcdc] & 4 != 0 { 16 } else { 8 }).contains(&(ly as i16 - (sample.oam[i * 4] as i16 - 16)))).count() > 10;
        let has = [
            trace.iter().any(|p| p.before.phase == Phase::WindowOn),
            trace.iter().any(|p| p.before.insert_bg_pixel),
            resumed(Phase::FetchWait),
            resumed(Phase::FetchB),
            resumed(Phase::FetchC),
            resumed(Phase::FetchD),
            tall,
            crowded,
        ];
        if has.iter().zip(&found).any(|(&h, &n)| h && n < quota) {
            for (n, &h) in found.iter_mut().zip(&has) {
                *n += h as usize;
            }
            picked.push((case, sample));
        }
    }
    for (name, n) in KINDS.iter().zip(&found) {
        println!("  {n:3} lines: {name}");
    }
    assert!(found.iter().all(|&n| n > 0), "a kind no line took: {:?}", KINDS.iter().zip(&found).collect::<Vec<_>>());
    let c = stepper::build();
    let mut t = Tally::default();
    let mut st: Option<Stepper> = None;
    for (case, sample) in &picked {
        check_line(&c, *case, sample, &mut t);
        match st.as_mut() {
            Some(s) => s.load(&sample.vram, &sample.oam, &sample.display),
            None => st = Some(Stepper::new(stepper::engine(), &sample.vram, &sample.oam, &sample.display)),
        }
        let s = st.as_mut().unwrap();
        let start = sample.start();
        let want = fifo::draw(&start, &sample.writes);
        assert_eq!(s.draw(&start, &sample.writes), fifo::Drawn { dots: fifo::mode0_dots(want.dots, start.display[PAL_REGS as usize + 2], start.mode.rgb555()), ..want }, "line {case}: the checkpoint alone against fifo::draw, in dots to mode 0");
        assert_eq!(s.engine().stats.fallback_passes, 0, "line {case}: the engine took the dense fallback");
    }
    report(&t);
    assert!(t.bad.is_empty(), "{} kinds of pass differ", t.bad.len());
}
