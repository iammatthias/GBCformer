//! `stock-llama shapes`: what a pass of a given shape costs in the runtime, through the
//! pass's own path (`Cpp::scope`, `Runtime::scoped_pass`). It reads no token, so it runs
//! any uniform Llama GGUF, the one model's or a synthetic one of another shape
//! (`examples/export -- synth`). Values do not change what the runtime computes, so the
//! tokens are arbitrary ids.
//!
//! ```sh
//! stock-llama shapes <model.gguf> --shapes NAME=TOKENS:ATTENDING:ROWS:CELLS,... [--cells N | --state FILE --top N] [--reps N] [--lanes N]
//! ```
//!
//! A shape is a group of TOKENS tokens, of which ATTENDING attend a scope of CELLS cells
//! and ROWS of those reach the LM head. The cache is `--cells` cells decoded here, or a
//! saved cache of `--top` cells (`STOCK_STATE`'s file). The shapes decode in turn, `--reps`
//! rounds after 5 discarded ones, and each prints its median.

use crate::runtime::{Cpp, Runtime, Scoped};
use std::time::Instant;

pub struct Shape {
    pub name: String,
    pub tokens: usize,
    pub attending: usize,
    pub rows: usize,
    pub cells: usize,
}

impl Shape {
    pub fn parse(s: &str) -> Shape {
        let (name, rest) = s.split_once('=').unwrap_or_else(|| panic!("--shapes {s}: NAME=TOKENS:ATTENDING:ROWS:CELLS"));
        let n: Vec<usize> = rest.split(':').map(|x| x.parse().unwrap_or_else(|_| panic!("--shapes {s}: {x} is not a number"))).collect();
        assert!(n.len() == 4 && n[2] <= n[1] && n[1] <= n[0] && n[3] >= 2, "--shapes {s}: ROWS <= ATTENDING <= TOKENS, and at least 2 cells");
        Shape { name: name.to_string(), tokens: n[0], attending: n[1], rows: n[2], cells: n[3] }
    }
}

const WARM: usize = 5;

pub fn run(model: &str, shapes: &[Shape], cells: usize, state: Option<&str>, reps: usize, lanes: usize, threads: usize) {
    let n_ctx = cells + (gbcformer::memory::dense::Records::CELL + 1) * lanes + 1024;
    let mut rt = Cpp::open(model, n_ctx, lanes, threads);
    let (lib, build) = crate::cpp::build();
    eprintln!("runtime: {build:?} ({lib}); vocabulary {}", rt.cpp.vocab);
    let vocab = rt.cpp.vocab;
    let t = Instant::now();
    match state {
        Some(path) => assert!(rt.load(path), "the cache at {path} loads"),
        None => {
            for at in (0..cells).step_by(512) {
                let batch: Vec<crate::cpp::Tok> = (at..cells.min(at + 512)).map(|p| crate::cpp::Tok { id: p % vocab, pos: p, seq: 0, also: -1, logits: false }).collect();
                rt.cpp.decode(&batch);
            }
        }
    }
    let top = rt.cpp.seq_pos_max(0);
    eprintln!("cache: top cell at position {top}, {:.1}s", t.elapsed().as_secs_f64());
    // One scoped sequence per size: the first cells and the top cell, which a scope holds.
    let mut sizes: Vec<usize> = shapes.iter().map(|s| s.cells).collect();
    sizes.sort();
    sizes.dedup();
    let scopes: Vec<(usize, Scoped)> = sizes.iter().map(|&n| (n, rt.scope(&[(0, n as u32 - 1), (top as u32, top as u32 + 1)]).expect("a scoped sequence"))).collect();
    let mut ns: Vec<Vec<u128>> = vec![Vec::new(); shapes.len()];
    let mut inside: Vec<Vec<u128>> = vec![Vec::new(); shapes.len()];
    for round in 0..WARM + reps {
        for (i, s) in shapes.iter().enumerate() {
            let tokens: Vec<usize> = (0..s.tokens).map(|k| (k * 257 + 1) % vocab).collect();
            let outs: Vec<usize> = (s.tokens - s.rows..s.tokens).collect();
            let also: Vec<usize> = (s.tokens - s.attending..s.tokens - s.rows).collect();
            let scope = scopes.iter().find(|(n, _)| *n == s.cells).expect("the shape's scope").1;
            let d0 = rt.decode_nanos();
            let t = Instant::now();
            let (rows, _) = rt.scoped_pass(&tokens, &outs, &also, scope);
            let e = t.elapsed().as_nanos();
            assert_eq!(rows.len(), s.rows);
            if round >= WARM {
                ns[i].push(e);
                inside[i].push(rt.decode_nanos() - d0);
            }
        }
    }
    let ms = |v: &mut Vec<u128>, q: usize| {
        v.sort();
        v[(v.len() - 1) * q / 100] as f64 / 1e6
    };
    for (i, s) in shapes.iter().enumerate() {
        println!("shape {} tokens {} attending {} rows {} cells {}: median {:.2} ms a pass (min {:.2}, p90 {:.2}; {:.2} inside the runtime's decode), {} passes", s.name, s.tokens, s.attending, s.rows, s.cells, ms(&mut ns[i], 50), ms(&mut ns[i], 0), ms(&mut ns[i], 90), ms(&mut inside[i], 50), reps);
    }
}
