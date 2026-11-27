//! The room for re-levelling the one model (`docs/perf-log.md`, "Re-levelling"): each
//! part's units and heads by layer, every item's earliest and latest layer, lower bounds
//! on the widest layer at each layer count, and what packing the parts reaches. Reads
//! the circuits only; no checkpoint is compiled.
//!
//! `room` prints the census, the bounds and the curve; `room -- --try <cap> <heads>
//! <part>=<depth>,... [<live wires>]` packs the named parts, each to its depth, under a
//! unit cap, a head cap and a part's live-wire limit and prints the shape it compiles to
//! (`ROOM_LIVE=1` adds each part's live wires per boundary); `room -- --scan <ppu depths> <stepper
//! depths> <length depths>` (each a comma list, 0 for a part left where it is) tries those
//! placements of the PPU's three parts that end within their caps and lists the cheapest
//! (`ROOM_CAPS=lo:hi:step` and `ROOM_LIVES=lo:hi:step` set the unit caps and live limits it
//! tries); `room -- --path <part>` lists the part's critical path by level.

use gbcformer::circuit::Circuit;
use gbcformer::level::{Graph, Over};
use gbcformer::llama::{routed_units, Model, HEAD_DIM};
use gbcformer::one_model::{concat, level, parts, Part, CPU_SLICE_HIDDEN, HEADS};

/// The multiplies one token pays in one uniform layer: the three MLP matrices, the query
/// and output projections, one key and one value head.
fn multiplies(hidden: usize, width: usize, heads: usize) -> f64 {
    (hidden * (3 * width + 2 * HEAD_DIM * heads + 2 * HEAD_DIM)) as f64
}

fn line(tag: &str, c: &Circuit, base: f64) -> f64 {
    let s = Model::census(c, true);
    let depths: Vec<u32> = (0..6).map(|p| c.part_depth(p)).collect();
    let mut live = vec![vec![0usize; s.cfg.layers]; 6];
    for b in 0..s.cfg.layers {
        for w in c.live_at(b as u32) {
            live[c.part_of(w.0 as usize)][b] += 1;
        }
    }
    let m = s.cfg.layers as f64 * multiplies(s.cfg.hidden, s.cfg.intermediate, s.cfg.heads);
    println!(
        "{tag}: hidden {}, {} layers x width {} = {}, heads {}, multiplies per token {:.1}M ({:.1}% of today's); part depths {depths:?}, widest live {:?}, clearing units {:?}; units per layer {:?}; heads per layer {:?}",
        s.cfg.hidden,
        s.cfg.layers,
        s.cfg.intermediate,
        s.cfg.layers * s.cfg.intermediate,
        s.cfg.heads,
        m / 1e6,
        100.0 * m / if base == 0.0 { m } else { base },
        live.iter().map(|p| p.iter().copied().max().unwrap()).collect::<Vec<_>>(),
        s.clears.iter().map(|p| p.iter().sum::<usize>()).collect::<Vec<_>>(),
        s.units,
        s.heads
    );
    if std::env::var_os("ROOM_LIVE").is_some() {
        for (p, l) in Part::ALL.iter().zip(&live) {
            println!("  {} live per boundary {l:?}", p.name());
        }
    }
    m
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ps = parts();
    let (now, _) = concat(&ps);
    let base = line("today", &now, 0.0);
    let census = Model::census(&now, true);
    let part = |s: &str| *Part::ALL.iter().find(|p| p.name() == s).unwrap_or_else(|| panic!("no part {s}"));

    if let Some(i) = args.iter().position(|a| a == "--scan") {
        let lists: Vec<Vec<u32>> = args[i + 1..i + 4].iter().map(|l| l.split(',').map(|d| d.parse().unwrap()).collect()).collect();
        return scan(&ps, &lists, HEADS, base);
    }
    if let Some(i) = args.iter().position(|a| a == "--try") {
        let (cap, heads): (usize, usize) = (args[i + 1].parse().unwrap(), args[i + 2].parse().unwrap());
        let live: usize = args.get(i + 4).map_or(usize::MAX / 2, |a| a.parse().unwrap());
        let free: Vec<(Part, u32)> = args[i + 3].split(',').map(|s| s.split_once('=').map(|(p, d)| (part(p), d.parse().unwrap())).unwrap_or_else(|| panic!("--try takes part=depth, not {s}"))).collect();
        let (mut c, _) = concat(&ps);
        if let Some(over) = level(&mut c, &free, cap, heads, live) {
            line(&format!("cap {cap}, heads {heads}, {} (the fullest layer ends {} units and {} heads over)", args[i + 3], over.units, over.heads), &c, base);
        } else {
            println!("cap {cap}, heads {heads}, {}: does not fit", args[i + 3]);
        }
        return;
    }

    if let Some(i) = args.iter().position(|a| a == "--path") {
        return path(&ps[part(&args[i + 1]) as usize]);
    }

    // Per part: its items, and how far each can move in a model of today's depth.
    let graphs: Vec<Graph> = ps.iter().map(Graph::of).collect();
    let depth = census.cfg.layers as u32 - 1;
    for (k, p) in Part::ALL.iter().enumerate() {
        let (g, c) = (&graphs[k], &ps[k]);
        let free = vec![0; g.len()];
        let (early, late) = (g.earliest(&free), g.latest(depth, &free));
        let own = g.latest(c.depth(), &free);
        let nodes: Vec<usize> = (0..g.len()).filter(|&i| g.is_node(i)).collect();
        let units: usize = nodes.iter().map(|&i| g.weight(i).0).sum();
        let heads: usize = (0..g.len()).map(|i| g.weight(i).1).sum();
        let mut slack = [0usize; 10];
        for &i in &nodes {
            slack[((own[i] - early[i] as i64) as usize).min(9)] += g.weight(i).0;
        }
        let per = |f: &dyn Fn(usize) -> usize, of: &dyn Fn(usize) -> usize| -> Vec<usize> {
            let mut v = vec![0; depth as usize + 1];
            for i in 0..g.len() {
                v[of(i)] += f(i);
            }
            v
        };
        println!(
            "part {}: {} nodes, {} node units, {} heads in {} groups (the largest {}), depth {} (critical path {}), largest node {} units, routing units {}; node units by slack within its own depth 0 to 8 and 9+: {:?}; node units by earliest level {:?}; by latest level of {depth} {:?}; heads by layer today {:?}, by earliest layer {:?}",
            p.name(),
            nodes.len(),
            units,
            heads,
            g.len() - nodes.len(),
            (0..g.len()).map(|i| g.weight(i).1).max().unwrap_or(0),
            c.depth(),
            early.iter().copied().max().unwrap(),
            nodes.iter().map(|&i| g.weight(i).0).max().unwrap(),
            (0..now.outputs.len()).filter(|&j| now.out_part(j) == k).map(|j| routed_units(&now, j)).sum::<usize>(),
            slack,
            &per(&|i| g.weight(i).0, &|i| early[i] as usize)[1..],
            &per(&|i| g.weight(i).0, &|i| late[i] as usize)[1..],
            per(&|i| g.weight(i).1, &|i| g.original()[i] as usize),
            per(&|i| g.weight(i).1, &|i| early[i] as usize),
        );
        println!("part {} heads by latest level in a model of {depth} levels: {:?}", p.name(), per(&|i| g.weight(i).1, &|i| late[i] as usize));
    }

    // Lower bounds. Whatever the placement, the units confined to a window of layers
    // need that many layers' width: the nodes whose earliest and latest levels both fall
    // in it. Every unit is confined to the whole model, clearing and routing units too.
    let total: usize = census.units.iter().sum();
    println!("units {total} (nodes, clearing units, routing units, reader units), heads past layer 0 {}, layer 0 heads {}", census.heads[1..].iter().sum::<usize>(), census.heads[0]);
    let bound = |depth: u32, pin_heads: bool, own_depth: bool| -> (usize, usize) {
        let levels = depth as usize;
        let mut confined = vec![vec![0usize; levels + 2]; levels + 2];
        let mut heads = vec![vec![0usize; levels + 2]; levels + 2];
        for (k, g) in graphs.iter().enumerate() {
            let pins: Vec<u32> = (0..g.len()).map(|i| if pin_heads && !g.is_node(i) { g.original()[i] } else { 0 }).collect();
            let (early, late) = (g.earliest(&pins), g.latest(if own_depth { ps[k].depth() } else { depth }, &pins));
            for i in 0..g.len() {
                confined[early[i] as usize][late[i] as usize] += g.weight(i).0;
                heads[early[i] as usize][late[i] as usize] += g.weight(i).1;
            }
        }
        let over = |m: &[Vec<usize>]| (1..=levels).flat_map(|a| (a..=levels).map(move |b| (a, b))).map(|(a, b)| (a..=b).flat_map(|x| (x..=b).map(move |y| (x, y))).map(|(x, y)| m[x][y]).sum::<usize>().div_ceil(b - a + 1)).max().unwrap();
        (over(&confined).max(total.div_ceil(levels + 1)), over(&heads).max(census.heads[0]))
    };
    let (w, h) = bound(depth, true, true);
    println!("bound (a), {} layers, every head and every part's depth where it is, nodes moved within their slack: width at least {w}, heads at least {h}", depth + 1);
    println!("bound (b), every node and head free in L layers (the heads' floor is layer 0's gathers):");
    for layers in 25..=40u32 {
        let (w, h) = bound(layers - 1, false, false);
        let width = w.next_multiple_of(64);
        let m = layers as f64 * multiplies(census.cfg.hidden, width, h);
        println!("  L {layers}: width at least {w} ({width}), heads at least {h}, L x width {}, multiplies per token at today's hidden {:.1}M ({:.1}% of today's)", layers as usize * width, m / 1e6, 100.0 * m / base);
    }

}

/// `--path <part>`: the part's critical path, the nodes with no slack in its own depth, by
/// level and by name (digits dropped), with their units. A shallower part must shorten
/// every one of these chains.
fn path(c: &Circuit) {
    let g = Graph::of(c);
    let free = vec![0; g.len()];
    let (early, late) = (g.earliest(&free), g.latest(c.depth(), &free));
    let mut by: std::collections::BTreeMap<(u32, String), (usize, usize)> = Default::default();
    for i in (0..g.len()).filter(|&i| late[i] == early[i] as i64) {
        let name = if g.is_node(i) { c.nodes[i].name.chars().filter(|ch| !ch.is_ascii_digit()).collect() } else { "(heads)".to_string() };
        let e = by.entry((early[i], name)).or_default();
        (e.0, e.1) = (e.0 + 1, e.1 + g.weight(i).0);
    }
    for level in 1..=c.depth() {
        let mut names: Vec<(usize, String)> = by.iter().filter(|((l, _), _)| *l == level).map(|((_, n), (count, units))| (*units, format!("{n} x{count} ({units})"))).collect();
        names.sort_by(|a, b| b.0.cmp(&a.0));
        println!("  level {level}: {} units with no slack: {}", names.iter().map(|n| n.0).sum::<usize>(), names.iter().take(8).map(|n| n.1.as_str()).collect::<Vec<_>>().join(", "));
    }
}

/// The values `lo:hi:step` of the environment variable `name` spans, or `default`'s.
fn range(name: &str, default: (usize, usize, usize)) -> impl Iterator<Item = usize> + Clone {
    let (lo, hi, step) = std::env::var(name).map_or(default, |v| {
        let n: Vec<usize> = v.split(':').map(|x| x.parse().unwrap_or_else(|_| panic!("{name} takes lo:hi:step, not {v}"))).collect();
        (n[0], n[1], n[2])
    });
    (lo..=hi).step_by(step)
}

/// `--scan`: the scanline PPU, the stepper and the length at each depth of their lists
/// (0 leaves a part where it is), under each cap and live limit; the cheapest thirty by a
/// token's multiplies.
///
/// A stream of more than 20 circuit layers runs without the template's reader buckets
/// (`Template::single_bucket`), as the APU's does.
fn scan(ps: &[Circuit; 6], lists: &[Vec<u32>], heads: usize, base: f64) {
    let mut found: Vec<(f64, String)> = Vec::new();
    for &ppu in &lists[0] {
        for &stp in &lists[1] {
            for &len in &lists[2] {
                for (cap, live) in range("ROOM_CAPS", (3392, 4160, 64)).flat_map(|cap| range("ROOM_LIVES", (1313, 1377, 32)).map(move |live| (cap, live))) {
                    let free: Vec<(Part, u32)> = [(Part::Ppu, ppu), (Part::Stepper, stp), (Part::Length, len)].into_iter().filter(|f| f.1 != 0).collect();
                    let (mut c, _) = concat(ps);
                    if level(&mut c, &free, cap, heads, live) == Some(Over { units: 0, heads: 0 }) {
                        let s = Model::census(&c, true);
                        if s.cfg.hidden < CPU_SLICE_HIDDEN {
                            continue;
                        }
                        let m = s.cfg.layers as f64 * multiplies(s.cfg.hidden, s.cfg.intermediate, s.cfg.heads);
                        found.push((m, format!("ppu={ppu},stp={stp},len={len} cap {cap} live {live}: hidden {}, width {} (widest {}), heads {}, {:.1}% of today's", s.cfg.hidden, s.cfg.intermediate, s.units.iter().max().unwrap(), s.cfg.heads, 100.0 * m / base)));
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("{} placements fit; the cheapest:", found.len());
    for (_, s) in found.iter().take(30) {
        println!("  {s}");
    }
}
