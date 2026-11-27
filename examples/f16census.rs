//! Lists every circuit step an f16 checkpoint cannot hold (`docs/stock-llama.md` §3,
//! §6). Under T1 (f16 weights, f32 sums) a step fails with a coefficient above 511, a
//! live side above 511 in both orientations, or a bias above what the constant channels
//! hold. Under T2 (f16 sums) a gate fails when its partial sums can pass `F16_ACC_GATE`,
//! and a node fails when its down row's can pass `F16_ACC_DOWN`.
//! `cargo run --release --example f16census -- [cpu|mem|apu|control|ppu|all] [-v] [--model] [--path]`; `--model` also compiles the checkpoint and runs its f16 checks, and `--path` prints the longest chain of levels.

use gbcformer::circuit::{Circuit, F16_ACC_DOWN, F16_ACC_GATE};
use gbcformer::llama::Model;
use gbcformer::memory::attn::Params;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let which = args.iter().find(|a| !a.starts_with('-')).cloned().unwrap_or_else(|| "all".into());
    let verbose = args.iter().any(|a| a == "-v");
    let all: [(&str, fn() -> Circuit); 7] = [
        ("cpu", gbcformer::sm83::build),
        ("mem", gbcformer::sm83::build_mem),
        ("apu", gbcformer::apu_circuit::build),
        ("control", gbcformer::control_circuit::build),
        ("ppu", gbcformer::ppu_circuit::circuit::build),
        ("length", gbcformer::ppu_circuit::circuit::build_length),
        ("stepper", gbcformer::ppu_circuit::stepper::build),
    ];
    for (name, build) in all.iter().filter(|(n, _)| which == "all" || *n == which) {
        let c = build();
        let (mut t1, mut gate, mut down) = (0, 0, 0);
        let mut families: BTreeMap<String, (usize, usize, i64)> = BTreeMap::new();
        for n in &c.nodes {
            let t1_here = c.node_t1_violations(n);
            t1 += t1_here;
            if t1_here > 0 {
                println!("  T1: {} ({} steps, ramp {})", n.name, t1_here, n.ramp);
            }
            let masses = c.node_gate_masses(n);
            let g = masses.iter().copied().max().unwrap_or(0);
            let bad_g = masses.iter().filter(|&&m| m > F16_ACC_GATE).count();
            let d = c.node_down_mass(n);
            gate += bad_g;
            down += (d > F16_ACC_DOWN) as usize;
            if bad_g > 0 || d > F16_ACC_DOWN {
                let fam: String = n.name.split(['.', '^', '#']).next().unwrap_or("").to_string();
                let e = families.entry(fam).or_default();
                e.0 += bad_g;
                e.1 += (d > F16_ACC_DOWN) as usize;
                e.2 = e.2.max(g.max(d));
                if verbose {
                    let ws: Vec<i64> = n.steps[0].lin.terms.values().copied().collect();
                    println!("  {} | steps {} | gate {g} | down {d} | w {:?} | c {} | θ {}", n.name, n.steps.len(), ws, n.steps[0].lin.c, n.steps[0].theta);
                }
            }
        }
        println!("{name}: depth {}, {} nodes, {} steps; T1 violations {t1}; T2 gate steps over {F16_ACC_GATE}: {gate}; T2 down rows over {F16_ACC_DOWN}: {down}", c.depth(), c.nodes.len(), c.nodes.iter().map(|n| n.steps.len()).sum::<usize>());
        println!("  units per level {:?}", c.units_per_level());
        for (fam, (g, d, worst)) in &families {
            println!("  {fam}: gate {g}, down {d}, worst {worst}");
        }
        if args.iter().any(|a| a == "--model") {
            let m = if c.heads.is_empty() && *name != "mem" { Model::compile(&c) } else { Model::compile_mem(&c, Params::memory()) };
            let show = |r: Result<(), String>| r.err().map_or("ok".to_string(), |e| e.chars().take(3000).collect());
            println!("  checkpoint: {} layers, hidden {}, intermediate {}, {:.1}M dense", m.cfg.layers, m.cfg.hidden, m.cfg.intermediate, m.cfg.params() as f64 / 1e6);
            println!("  check_f16: {}", show(m.check_f16()));
            println!("  check_f16_acc: {}", show(m.check_f16_acc()));
        }
        if args.iter().any(|a| a == "--path") {
            // The longest chain by earliest level, a head's byte counted at its layer.
            let mut asap: Vec<u32> = vec![0; c.levels.len()];
            let mut from: Vec<Option<usize>> = vec![None; c.levels.len()];
            for h in &c.heads {
                for s in &h.out {
                    asap[s.0 as usize] = h.layer;
                }
            }
            for n in &c.nodes {
                let (mut l, mut f) = (0, None);
                for s in n.steps.iter().flat_map(|s| s.lin.terms.keys()) {
                    if asap[s.0 as usize] >= l {
                        (l, f) = (asap[s.0 as usize], Some(s.0 as usize));
                    }
                }
                asap[n.out.0 as usize] = l + 1;
                from[n.out.0 as usize] = f;
            }
            // A head's byte continues the chain through its query's latest wire.
            for h in &c.heads {
                if let gbcformer::circuit::Query::Wires(q) = &h.query {
                    let f = q.iter().flat_map(|l| l.terms.keys()).max_by_key(|s| asap[s.0 as usize]).map(|s| s.0 as usize);
                    for s in &h.out {
                        from[s.0 as usize] = f;
                    }
                }
            }
            let mut w = (0..c.levels.len()).max_by_key(|&w| asap[w]);
            while let Some(x) = w {
                println!("  L{} {}", asap[x], c.names[x]);
                w = from[x];
            }
        }
    }
}
