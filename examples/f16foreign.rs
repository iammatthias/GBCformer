//! The one model's rows whose f16 partial sums can pass 65,504 at another part's
//! positions (`docs/hardware-shaped.md` §15, "What stock runtimes must know").
//! `Model::check_f16_acc` holds a part's gate and query rows over that part's own channel
//! ranges, where their outputs count. A runtime that runs the whole model at every
//! position and sums in f16 (llama.cpp's Mac CPU backend) also runs each row over every
//! other part's ranges, and an infinite gate times its `up` of 0 is NaN. This lists those
//! rows: per layer, per (row's part, position's part), with the largest subset sum.
//! `cargo run --release --example f16foreign -- [-v] [--unit <layer>:<unit>]`; `-v` names
//! every row, `--unit` prints one gate row's terms with each channel's occupants.

use gbcformer::f16acc::{self, Form};
use gbcformer::llama::{ChannelVars, HEAD_DIM, Model, NO_PART};
use gbcformer::memory::attn::Params;
use gbcformer::one_model::{self, Part};
use std::collections::BTreeMap;

const F16_MAX: f64 = 65504.0;

/// The row as integers over its quantum, the largest power of two dividing every weight.
fn integral(w: &[f32]) -> Option<(f64, Form)> {
    let low = |x: f32| {
        let b = x.abs().to_bits();
        let (e, m) = ((b >> 23) as i32, b & 0x7F_FFFF);
        let (e, m) = if e == 0 { (1, m) } else { (e, m | 0x80_0000) };
        2f64.powi(e - 150 + m.trailing_zeros() as i32)
    };
    let q = w.iter().filter(|x| **x != 0.0).map(|&x| low(x)).fold(f64::INFINITY, f64::min);
    q.is_finite().then(|| (q, Form { terms: w.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, &x)| (i, (x as f64 / q) as i64)).collect(), c: 0 }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "-v");
    let unit: Option<(usize, usize)> = args.iter().position(|a| a == "--unit").map(|i| {
        let (l, u) = args[i + 1].split_once(':').expect("--unit <layer>:<unit>");
        (l.parse().expect("layer"), u.parse().expect("unit"))
    });
    let m = Model::compile_mem(&one_model::circuit(&one_model::parts()).0, Params::memory());
    let overlay = m.overlay.as_ref().expect("the one model is an overlay");
    let (d, inter) = (m.cfg.hidden, m.cfg.intermediate);
    let parts = m.layout.offsets.len();
    let name = |p: usize| Part::ALL[p].name();
    let shift = gbcformer::gguf::qk_shift(&m);
    println!("hidden {d}, FFN width {inter}, {} layers, {} heads, {parts} parts", m.layers.len(), m.cfg.heads);

    if let Some((li, u)) = unit {
        let g = &m.layers[li].gate[u * d..][..d];
        let own = overlay.units[li][u];
        println!("layer {li} unit {u}: {} (part {})", m.unit_names[li].get(u).map_or("", |s| s), if own == NO_PART { "shared" } else { name(own as usize) });
        let vars: Vec<ChannelVars> = (0..parts).map(|p| m.channel_vars_of(li, true, Some(p))).collect();
        for (ch, &w) in g.iter().enumerate().filter(|(_, x)| **x != 0.0) {
            let mut line = format!("  channel {ch}: weight {w}");
            for (p, v) in vars.iter().enumerate() {
                let (a, b, _) = v.0[ch];
                if (a, b) != (0, 0) {
                    line += &format!("; {} [{a}, {b}]", name(p));
                }
            }
            if ch >= m.layout.wire_base() && ch - m.layout.wire_base() < m.layout.occupants.len() {
                for s in &m.layout.occupants[ch - m.layout.wire_base()] {
                    line += &format!("; {} {} (level {}, {:?})", name(m.layout.part_of(*s)), m.wire_names.get(s.0 as usize).map_or("", |s| s), m.wire_levels[s.0 as usize], m.wire_ranges[s.0 as usize]);
                }
            }
            println!("{line}");
        }
        return;
    }

    // (kind, row's part, position's part) -> rows over, worst sum.
    let mut pairs: BTreeMap<(&str, usize, usize), (usize, f64)> = BTreeMap::new();
    let mut rows = [0usize; 2];
    let mut total = [0usize; 2];
    for (li, l) in m.layers.iter().enumerate() {
        let mut here = [0usize; 2];
        let mut worst = [0f64; 2];
        let mlp: Vec<ChannelVars> = (0..parts).map(|p| m.channel_vars_of(li, true, Some(p))).collect();
        let attn: Vec<ChannelVars> = (0..parts).map(|p| m.channel_vars_of(li, false, Some(p))).collect();
        let mut row = |kind: usize, label: &'static str, w: &[f32], own: &dyn Fn() -> u8, vars: &[ChannelVars], scale: f64, what: &dyn Fn() -> String| {
            let Some((q, f)) = integral(w) else { return };
            let own = own();
            if own == NO_PART {
                return;
            }
            rows[kind] += 1;
            let mut over = false;
            for p in (0..parts).filter(|&p| p != own as usize) {
                let sum = f16acc::gate_mass(&f, &vars[p]) as f64 * q * scale;
                if sum > F16_MAX {
                    over = true;
                    let e = pairs.entry((label, own as usize, p)).or_insert((0, 0.0));
                    *e = (e.0 + 1, e.1.max(sum));
                    worst[kind] = worst[kind].max(sum);
                    if verbose {
                        println!("  layer {li} {label} {}: {} at {} positions, largest partial sum {sum}", what(), name(own as usize), name(p));
                    }
                }
            }
            here[kind] += over as usize;
        };
        for u in 0..inter {
            row(0, "gate", &l.gate[u * d..][..d], &|| overlay.units[li][u], &mlp, 1.0, &|| format!("unit {u} ({})", m.unit_names[li].get(u).map_or("", |s| s)));
        }
        for r in 0..m.cfg.attn() {
            row(1, "query", &l.q[r * d..][..d], &|| overlay.heads[li][r / HEAD_DIM], &attn, 1.0 / (1u64 << shift[li]) as f64, &|| format!("head {} row {}", r / HEAD_DIM, r % HEAD_DIM));
        }
        println!("layer {li}: {} gate rows over (worst {}), {} query rows over (worst {}, GGUF shift {})", here[0], worst[0], here[1], worst[1], shift[li]);
        total[0] += here[0];
        total[1] += here[1];
    }
    println!("{} of {} part-owned gate rows and {} of {} part-owned query rows can pass {F16_MAX} at another part's positions", total[0], rows[0], total[1], rows[1]);
    for ((label, own, at), (n, worst)) in &pairs {
        println!("  {label} rows of {} at {} positions: {n}, worst {worst}", name(*own), name(*at));
    }
}
