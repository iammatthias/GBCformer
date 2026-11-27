//! Lists circuit nodes at a given level, with their step counts and fan-in, for the
//! host-fed circuit or the memory circuit with `--mem`.
//!
//! With `--table PART` it prints the whole circuit instead. PART is ppu, cpu, apu or
//! ctl for the one model's parts as `one_model::parts` builds them, or host for the
//! host-fed GBC CPU. The table lists MLP units per node family and level, their weights
//! (gate terms, bias, up and down per unit), and the wires each family keeps live at the
//! widest residual boundary. A family is a node name's stem, with an opcode or value
//! suffix (`op3e`, `OP0=3e`) collapsed to `*`.

use gbcformer::circuit::Circuit;
use gbcformer::sm83::{build, build_mem};
use std::collections::BTreeMap;

fn family(name: &str) -> String {
    let stem = name.split('.').next().unwrap_or("");
    if let Some((head, tail)) = stem.split_once('=') {
        return if tail.chars().all(|c| c.is_ascii_hexdigit()) { format!("{head}=*") } else { stem.to_string() };
    }
    if stem.len() == 4 && (stem.starts_with("op") || stem.starts_with("cb")) && stem[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        return format!("{}*", &stem[..2]);
    }
    stem.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()
}

fn table(c: &Circuit) {
    let depth = c.depth() as usize;
    #[derive(Default)]
    struct Row {
        units: Vec<usize>,
        nonzeros: usize,
        live: usize,
    }
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for n in &c.nodes {
        let r = rows.entry(family(&n.name)).or_insert_with(|| Row { units: vec![0; depth], ..Default::default() });
        r.units[c.level(n.out) as usize - 1] += c.units(n);
        r.nonzeros += n.steps.iter().map(|s| (if c.bounds(&s.lin).1 <= s.theta { 1 } else { 2 }) * (s.lin.terms.len() + 3)).sum::<usize>();
    }
    let live = c.live_per_boundary();
    let widest = (0..live.len()).max_by_key(|&b| live[b]).unwrap();
    let mut other_live: BTreeMap<String, usize> = BTreeMap::new();
    for s in c.live_at(widest as u32) {
        let f = family(&c.names[s.0 as usize]);
        match rows.get_mut(&f) {
            Some(r) => r.live += 1,
            None => *other_live.entry(f).or_default() += 1,
        }
    }
    let total: usize = c.units_per_level().iter().sum();
    println!("depth {depth}, {} nodes, {total} units, units per level {:?}, live wires per boundary {live:?} (widest {widest})", c.nodes.len(), c.units_per_level());
    let mut rows: Vec<(String, Row)> = rows.into_iter().collect();
    rows.sort_by_key(|(n, r)| (std::cmp::Reverse(r.units.iter().sum::<usize>()), n.clone()));
    println!("{:14} {:>6} {:>5} {:>8} {:>5}  units per level", "family", "units", "%", "nonzero", "live");
    for (n, r) in &rows {
        let u: usize = r.units.iter().sum();
        let per: Vec<String> = r.units.iter().map(|k| if *k == 0 { ".".to_string() } else { k.to_string() }).collect();
        println!("{n:14} {u:6} {:5.1} {:8} {:5}  {}", 100.0 * u as f64 / total as f64, r.nonzeros, r.live, per.join(" "));
    }
    let other: Vec<String> = other_live.iter().map(|(n, k)| format!("{n} {k}")).collect();
    println!("live at boundary {widest} that no node writes (inputs, one-hots, head bytes, positions): {}", other.join(", "));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--table") {
        let part = args.get(i + 1).map(String::as_str).unwrap_or("cpu");
        if part == "host" {
            return table(&build());
        }
        let k = gbcformer::one_model::Part::ALL.iter().position(|p| p.name() == part).unwrap_or_else(|| panic!("--table takes ppu, cpu, apu or ctl, not {part}"));
        table(&gbcformer::one_model::parts()[k]);
        return;
    }
    let level: u32 = args.iter().find_map(|a| a.parse().ok()).unwrap_or(5);
    let c = if args.iter().any(|a| a == "--mem") { build_mem() } else { build() };
    let mut by_prefix: BTreeMap<String, (usize, usize, usize)> = Default::default();
    for n in c.nodes.iter().filter(|n| c.level(n.out) == level) {
        let prefix = n.name.split(['.', '\'']).next().unwrap_or("").to_string();
        let e = by_prefix.entry(prefix).or_default();
        e.0 += 1;
        e.1 += n.steps.len();
        e.2 += n.steps.iter().map(|s| s.lin.terms.len() + 1).sum::<usize>();
    }
    for (p, (nodes, steps, weights)) in by_prefix {
        println!("{p:12} nodes {nodes:4} steps {steps:5} gate weights {:6}", 2 * weights);
    }
}
