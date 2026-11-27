//! The GBC CPU stream's replay bench (`src/engine/replay.rs`): the passes a run recorded,
//! run again through the one model's CPU stream, every pass's decoded tokens held to the
//! recording.
//!
//! ```sh
//! GBCFORMER_REPLAY_OUT=t.rpl GBCFORMER_REPLAY_SKIP=500000 tools/bench/text.sh --weights all --one-model
//! cargo run --release --example replay -- t.rpl pokemon-yellow.gbc [--reps 9]
//! ```
//!
//! Each repetition starts the engine fresh on the recorded image. It prints each
//! repetition's wall time and the thread's own CPU time, cycles and instructions per pass,
//! then their minimum and median. The cycles count only while the thread runs, so a busy
//! machine moves them far less than wall time.
//!
//! `--model N` then fits the cost model (`model`): every pass bracketed once by the
//! thread's counters (the passes at each phase modulo N, N prime, in a replay of their
//! own), its cycles and instructions fitted by least squares against what the pass did;
//! one stage of every Nth pass bracketed, the stages in turn, `--stage-reps` replays of
//! them; and the same two for a pass run a second time on the same tokens, in which
//! nothing changed. A stage's instructions are exact. Its cycles hold what the bracket's
//! two system calls disturb, about 47 a bracket on this trace, so `sample` on the plain
//! replay is the measure of where the cycles go.

use gbcformer::engine::ols::{least_squares, Fit};
use gbcformer::engine::probe::{Stage, SHAPE};
use gbcformer::engine::replay::{Bracket, Sample, Trace, DID};
use gbcformer::engine::Engine;
use gbcformer::memory::image::Image;
use gbcformer::one_model::{streams, Part};
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(trace), Some(rom)) = (args.first(), args.get(1)) else {
        eprintln!("usage: replay <trace> <rom> [--reps K] [--variants 0,1] [--model N [--stage-reps K]]");
        std::process::exit(2);
    };
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let reps: usize = flag("--reps").map_or(9, |v| v.parse().expect("--reps K"));
    let variants: Vec<u8> = flag("--variants").map_or(vec![0], |v| v.split(',').map(|x| x.parse().expect("--variants a,b")).collect());
    gbcformer::qos::interactive();
    let trace = Trace::read(std::path::Path::new(trace)).unwrap_or_else(|e| panic!("{e}"));
    let rom = std::fs::read(rom).unwrap_or_else(|e| panic!("{rom}: {e}"));
    let image = trace.image(&rom).unwrap_or_else(|e| panic!("{e}"));
    let passes = trace.passes() as f64;
    eprintln!("trace: {passes} passes, {} events", trace.events.len());
    let mut engine = streams(&[Part::Cpu]).cpu.expect("the CPU stream");
    engine.insert(image.clone());
    // Each variant's first repetition warms the caches and memos and is not counted; then
    // the variants take turns, so drift on a busy machine reaches each alike.
    let mut runs: Vec<Vec<[f64; 4]>> = vec![Vec::new(); variants.len()];
    for rep in 0..(reps + 1) * variants.len() {
        let (vi, warm) = (rep % variants.len(), rep < variants.len());
        engine.variant = variants[vi];
        let r = engine.replay(&trace, &image).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.fallbacks, 0, "replay {rep}: {} passes took the dense fallback ({:?})", r.fallbacks, engine.stats.refused);
        let c = r.counts.expect("this platform gives no thread counters");
        assert!(c.cycles > 0 && c.instructions > 0, "the kernel counted no cycles or instructions for this thread (thread_selfcounts)");
        let row = [r.wall_ns as f64 / passes, c.cpu_ns as f64 / passes, c.cycles as f64 / passes, c.instructions as f64 / passes];
        eprintln!("rep {rep} variant {}{}: {:.1} ns wall, {:.1} ns CPU, {:.0} cycles, {:.0} instructions per pass, {:.2} GHz", variants[vi], if warm { " (warm-up)" } else { "" }, row[0], row[1], row[2], row[3], c.cycles as f64 / c.cpu_ns as f64);
        if !warm {
            runs[vi].push(row);
        }
    }
    for (vi, runs) in runs.iter().enumerate() {
        let stat = |i: usize| {
            let mut v: Vec<f64> = runs.iter().map(|r| r[i]).collect();
            v.sort_by(f64::total_cmp);
            (v[0], v[v.len() / 2], v[v.len() - 1])
        };
        let names = ["ns wall", "ns CPU", "cycles", "instructions"];
        let line: Vec<String> = names.iter().enumerate().map(|(i, n)| {
            let (min, med, max) = stat(i);
            format!("{n} min {min:.1} median {med:.1} max {max:.1}")
        }).collect();
        println!("replay variant {} {passes} passes x {reps}: {}", variants[vi], line.join("; "));
    }
    eprintln!("engine over every repetition: {}", engine.stats);
    eprintln!("layer 0 over every repetition: {} replans", engine.stats.layer0_replans);
    if let Some(every) = flag("--model") {
        let median = |i: usize| {
            let mut v: Vec<f64> = runs[0].iter().map(|r| r[i]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        engine.variant = variants[0];
        let stage_reps = flag("--stage-reps").map_or(14, |v| v.parse().expect("--stage-reps K"));
        model(&mut engine, &trace, &image, every.parse().expect("--model N"), stage_reps, [median(2), median(3)]);
    }
}

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for x in v {
        sum += x;
        n += 1;
    }
    sum / n.max(1) as f64
}

/// The values at `at` (shares of the sorted list).
fn quantiles(mut v: Vec<f64>, at: &[f64]) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    at.iter().map(|q| v[((v.len() - 1) as f64 * q).round() as usize]).collect()
}

/// One fit as a table: each term's coefficient and standard error, the term's mean, and
/// what it adds to the mean pass.
fn print_fit(title: &str, unit: &str, names: &[&str], rows: usize, x: impl Fn(usize, usize) -> f64, y: impl Fn(usize) -> f64) -> Option<Fit> {
    let fit = match least_squares(rows, names.len(), &x, &y) {
        Ok(f) => f,
        Err(e) => {
            println!("\n{title}: not fitted ({e})");
            return None;
        }
    };
    println!("\n{title}: {rows} passes, mean {:.1} {unit}, R2 {:.4}, residual sd {:.1}", fit.mean_y, fit.r2, fit.residual_sd);
    println!("  {:<24} {:>10} {:>8} {:>10} {:>10} {:>7}", "term", unit, "se", "mean", "adds", "share");
    println!("  {:<24} {:>10.1} {:>8.1} {:>10} {:>10.1} {:>6.1}%", "intercept", fit.intercept, fit.intercept_se, "1", fit.intercept, 100.0 * fit.intercept / fit.mean_y);
    for (j, name) in names.iter().enumerate() {
        let adds = fit.coef[j] * fit.mean_x[j];
        println!("  {:<24} {:>10.2} {:>8.2} {:>10.2} {:>10.1} {:>6.1}%", name, fit.coef[j], fit.se[j], fit.mean_x[j], adds, 100.0 * adds / fit.mean_y);
    }
    let residuals: Vec<f64> = (0..rows).map(|i| y(i) - fit.predict(&(0..names.len()).map(|j| x(i, j)).collect::<Vec<f64>>())).collect();
    let q = quantiles(residuals, &[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]);
    println!("  residuals at 1/5/25/50/75/95/99%: {}", q.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "));
    Some(fit)
}

/// A stage's kind and what its items count.
fn stage_kind(s: Stage) -> (&'static str, [&'static str; 4]) {
    match s {
        Stage::View => ("view", ["positions rewritten", "-", "-", "-"]),
        Stage::Layer0 => ("layer 0", ["moved positions", "wire changes", "-", "-"]),
        Stage::Scatter => ("scatter", ["wire changes", "-", "-", "-"]),
        Stage::HeadBucket(_) => ("head bucket", ["head-bucket changes", "-", "-", "-"]),
        Stage::Heads(_) => ("heads", ["heads dirty or hit", "lines read", "buses moved", "-"]),
        Stage::Buses(_) => ("buses", ["buses moved", "-", "-", "-"]),
        Stage::Reads(_) => ("reads", ["head-bucket changes", "lines read", "-", "-"]),
        Stage::Walk(_) => ("walk", ["pairs", "units", "vectors", "lone entries"]),
        Stage::Drain(_) => ("drain", ["touched channels", "-", "-", "-"]),
        Stage::Route => ("route", ["pairs", "units", "-", "-"]),
        Stage::RouteDrain => ("route drain", ["touched channels", "blank routes moved", "-", "-"]),
        Stage::Positional => ("positional", ["-", "-", "-", "-"]),
        Stage::Tail => ("tail", ["-", "-", "-", "-"]),
        Stage::DecodeSetup => ("decode setup", ["moved positions", "-", "-", "-"]),
        Stage::Decode => ("decode", ["positions decoded", "own units", "-", "-"]),
        Stage::Out => ("out", ["positions", "-", "-", "-"]),
        Stage::Shape => ("shape", SHAPE),
    }
}

/// The cost model of the GBC CPU stream's pass over `trace` (the module comment), printed.
/// `whole` is the unbracketed replay's median cycles and instructions per pass.
/// Whether a stage is a part of the `Reads` stage of its layer.
fn nested(s: Stage) -> bool {
    matches!(s, Stage::HeadBucket(_) | Stage::Heads(_) | Stage::Buses(_))
}

fn model(engine: &mut Engine, trace: &Trace, image: &Image, every: usize, stage_reps: usize, whole: [f64; 2]) {
    // The trace lists the decoded positions anew at each change: a kind is a list's content.
    let kind: Vec<u32> = trace.positions.iter().map(|p| trace.positions.iter().position(|q| q == p).expect("itself") as u32).collect();
    let mut turn = 0;
    let mut run = |bracket: Bracket, every: usize, phases: &mut dyn Iterator<Item = usize>| -> Vec<Sample> {
        let mut samples = Vec::new();
        for phase in phases {
            engine.replay_bracketed(trace, image, bracket, every, phase, &mut turn, &mut samples).unwrap_or_else(|e| panic!("{bracket:?}: {e}"));
        }
        // A replay's first passes build the state and warm the caches.
        samples.retain(|s| s.pass >= 64);
        for s in &mut samples {
            s.kind = kind[s.kind as usize];
        }
        samples
    };
    let passes = run(Bracket::Pass, every, &mut (0..every));
    let stages = run(Bracket::Stages, every, &mut (0..stage_reps).map(|r| r % every));
    let nulls = run(Bracket::NullPass, every, &mut (0..every.min(3)));
    let null_stages = run(Bracket::NullStages, every, &mut (0..stage_reps).map(|r| r % every));
    // Each pass's shape, by pass: the same in every replay.
    let mut shape = vec![[0u32; 4]; trace.passes()];
    for s in run(Bracket::Shape, 1, &mut (0..1)) {
        shape[s.pass as usize] = s.items;
    }

    let n = passes.len();
    let empty = [mean(passes.iter().map(|s| s.empty.cycles as f64)), mean(passes.iter().map(|s| s.empty.instructions as f64))];
    let cycles = |s: &Sample| s.counts.cycles as f64 - empty[0];
    let instructions = |s: &Sample| s.counts.instructions as f64 - empty[1];
    println!("\n== cost model: {n} passes bracketed once each (every {every}th, each phase), figures net of an empty bracket ({:.0} cycles, {:.0} instructions)", empty[0], empty[1]);
    let pass = [mean(passes.iter().map(cycles)), mean(passes.iter().map(instructions))];
    let between = [mean(passes.iter().map(|s| s.between.cycles as f64)) - empty[0], mean(passes.iter().map(|s| s.between.instructions as f64)) - empty[1]];
    let q = quantiles(passes.iter().map(cycles).collect(), &[0.05, 0.25, 0.5, 0.75, 0.95, 0.999]);
    println!("a pass: {:.0} cycles, {:.0} instructions (cycles at 5/25/50/75/95/99.9%: {})", pass[0], pass[1], q.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "));
    println!("between passes (hash check, write records, token changes): {:.0} cycles, {:.0} instructions", between[0], between[1]);
    println!("their sum {:.0} cycles, {:.0} instructions, against the unbracketed replay's {:.0} and {:.0} ({:+.1}%, {:+.1}%)", pass[0] + between[0], pass[1] + between[1], whole[0], whole[1], 100.0 * ((pass[0] + between[0]) / whole[0] - 1.0), 100.0 * ((pass[1] + between[1]) / whole[1] - 1.0));

    println!("\npasses by the positions they decode:");
    let mut kinds: Vec<u32> = passes.iter().map(|s| s.kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let mut main = (0, 0usize);
    for &k in &kinds {
        let of: Vec<&Sample> = passes.iter().filter(|s| s.kind == k).collect();
        if of.len() > main.1 {
            main = (k, of.len());
        }
        let did: Vec<String> = (0..DID.len()).map(|j| format!("{} {:.1}", DID[j], mean(of.iter().map(|s| s.did[j] as f64)))).collect();
        println!("  {} positions: {} passes ({:.1}%), {:.0} cycles, {:.0} instructions; {}", trace.positions[k as usize].len(), of.len(), 100.0 * of.len() as f64 / n as f64, mean(of.iter().map(|s| cycles(s))), mean(of.iter().map(|s| instructions(s))), did.join(", "));
    }

    let all: Vec<&Sample> = passes.iter().collect();
    let of_main: Vec<&Sample> = passes.iter().filter(|s| s.kind == main.0).collect();
    let main_name = format!("the passes decoding {} positions", trace.positions[main.0 as usize].len());
    for (name, set) in [("every pass", &all), (main_name.as_str(), &of_main)] {
        let x = |i: usize, j: usize| set[i].did[j] as f64;
        print_fit(&format!("instructions of {name} against what it did"), "instr", &DID, set.len(), x, |i| instructions(set[i]));
        print_fit(&format!("cycles of {name} against what it did"), "cycles", &DID, set.len(), x, |i| cycles(set[i]));
        let short = [0usize, 1, 9];
        let names: Vec<&str> = short.iter().map(|&j| DID[j]).collect();
        print_fit(&format!("cycles of {name} against units, pairs and layers walked"), "cycles", &names, set.len(), |i, j| set[i].did[short[j]] as f64, |i| cycles(set[i]));
        print_fit(&format!("cycles of {name} against units alone"), "cycles", &DID[..1], set.len(), x, |i| cycles(set[i]));
        // With the shape, the walk's vectors and lone entries stand in for its units.
        let long: Vec<&str> = DID[1..].iter().chain(&SHAPE).copied().collect();
        let xl = |i: usize, j: usize| if j < DID.len() - 1 { set[i].did[j + 1] as f64 } else { shape[set[i].pass as usize][j + 1 - DID.len()] as f64 };
        print_fit(&format!("instructions of {name} against what it did and its shape"), "instr", &long, set.len(), xl, |i| instructions(set[i]));
        print_fit(&format!("cycles of {name} against what it did and its shape"), "cycles", &long, set.len(), xl, |i| cycles(set[i]));
    }

    println!("\ncorrelations over every pass (cycles, then what the pass did):");
    let cols: Vec<Vec<f64>> = std::iter::once(passes.iter().map(cycles).collect()).chain((0..DID.len()).map(|j| passes.iter().map(|s| s.did[j] as f64).collect())).collect();
    let names: Vec<&str> = std::iter::once("cycles").chain(DID).collect();
    for (a, ca) in cols.iter().enumerate() {
        let (ma, mb): (f64, Vec<f64>) = (mean(ca.iter().copied()), cols.iter().map(|c| mean(c.iter().copied())).collect());
        let r: Vec<String> = cols[..=a]
            .iter()
            .enumerate()
            .map(|(b, cb)| {
                let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
                for (x, y) in ca.iter().zip(cb) {
                    sab += (x - ma) * (y - mb[b]);
                    saa += (x - ma) * (x - ma);
                    sbb += (y - mb[b]) * (y - mb[b]);
                }
                format!("{:>5.2}", sab / (saa * sbb).sqrt())
            })
            .collect();
        println!("  {:<24} {}", names[a], r.join(" "));
    }

    let null = [mean(nulls.iter().map(cycles)), mean(nulls.iter().map(instructions))];
    let did: Vec<String> = (0..DID.len()).map(|j| format!("{} {:.2}", DID[j], mean(nulls.iter().map(|s| s.did[j] as f64)))).collect();
    let q = quantiles(nulls.iter().map(cycles).collect(), &[0.05, 0.5, 0.95]);
    println!("\n== a pass in which nothing changed (each sampled pass run again at once): {} passes, {:.0} cycles (5/50/95%: {:.0} {:.0} {:.0}), {:.0} instructions; {}", nulls.len(), null[0], q[0], q[1], q[2], null[1], did.join(", "));

    println!("\n== stages: one stage of every {every}th pass bracketed, {stage_reps} replays; net cycles and instructions per pass, each stage's cycles fitted against its two items, and the same stage of a pass in which nothing changed");
    println!("  {:<12} {:>7} {:>8} {:>8} {:>8} {:>8} {:>26} {:>8} {:>30}  items a, b, c, d", "stage", "n", "cycles", "instr", "null cyc", "null ins", "mean items", "fixed", "cycles per item");
    let mut sums: Vec<(&str, [f64; 4])> = Vec::new();
    let mut total = [0.0f64; 4];
    for st in Stage::all(engine.circuit_layers()) {
        let of: Vec<&Sample> = stages.iter().filter(|s| s.stage == Some(st)).collect();
        let null_of: Vec<&Sample> = null_stages.iter().filter(|s| s.stage == Some(st)).collect();
        if of.is_empty() {
            continue;
        }
        let (kind, items) = stage_kind(st);
        let row = [mean(of.iter().map(|s| cycles(s))), mean(of.iter().map(|s| instructions(s))), mean(null_of.iter().map(|s| cycles(s))), mean(null_of.iter().map(|s| instructions(s)))];
        let used = items.iter().filter(|n| **n != "-").count();
        let fit = least_squares(of.len(), used, |i, j| of[i].items[j] as f64, |i| cycles(of[i]));
        let (fixed, per) = fit.map_or((f64::NAN, vec![f64::NAN; used]), |f| (f.intercept, f.coef));
        let list = |v: Vec<f64>| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" ");
        println!("  {:<12} {:>7} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>26} {:>8.1} {:>30}  {}", format!("{st:?}"), of.len(), row[0], row[1], row[2], row[3], list((0..used).map(|j| mean(of.iter().map(|s| s.items[j] as f64))).collect()), fixed, list(per), items[..used].join(", "));
        match sums.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, sum)) => (0..4).for_each(|i| sum[i] += row[i]),
            None => sums.push((kind, row)),
        }
        if !nested(st) {
            (0..4).for_each(|i| total[i] += row[i]);
        }
    }
    println!("\n  by kind of stage, summed over the layers (head bucket, heads and buses are parts of reads):");
    println!("  {:<12} {:>8} {:>8} {:>8} {:>8} {:>8}", "stage", "cycles", "share", "instr", "null cyc", "null ins");
    for (kind, row) in sums.iter().chain([&("outside them", [pass[0] - total[0], pass[1] - total[1], null[0] - total[2], null[1] - total[3]]), &("the pass", [pass[0], pass[1], null[0], null[1]])]) {
        println!("  {:<12} {:>8.1} {:>7.1}% {:>8.1} {:>8.1} {:>8.1}", kind, row[0], 100.0 * row[0] / pass[0], row[1], row[2], row[3]);
    }
    // The walks and drains of every layer pooled: one call's fixed cost and its cost per item.
    for (kind, names) in [("walk", &["pairs", "units", "vectors", "lone entries"][..]), ("walk", &["pairs", "units"][..]), ("drain", &["touched channels"][..]), ("reads", &["head-bucket changes", "lines read"][..]), ("head bucket", &["head-bucket changes"][..]), ("heads", &["heads dirty or hit", "lines read", "buses moved"][..]), ("buses", &["buses moved"][..])] {
        let of: Vec<&Sample> = stages.iter().filter(|s| s.stage.is_some_and(|st| stage_kind(st).0 == kind)).collect();
        print_fit(&format!("one layer's {kind}, every layer pooled, instructions"), "instr", names, of.len(), |i, j| of[i].items[j] as f64, |i| instructions(of[i]));
        print_fit(&format!("one layer's {kind}, every layer pooled, cycles"), "cycles", names, of.len(), |i, j| of[i].items[j] as f64, |i| cycles(of[i]));
    }
}
