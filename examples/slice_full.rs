//! Every stream's passes from a real run, held to the whole one-model checkpoint. A run of
//! the full program with `--features stock` and `GBCFORMER_RECORD` writes each stream's
//! events in the checkpoint's token space: the whole sequence (every other part's slots at
//! their fixed tokens), the positions decoded and the tokens decoded there, and each memory
//! head's cell and the 8 bytes it read. Here the whole checkpoint (not the stream's slice)
//! runs its dense pass over a cache of those cells, for the first `PASSES` events of each
//! stream that read memory or decode, and every decoded token must equal the recorded one.
//! A stream memory the run recorded whole (`GBCFORMER_RECORD_CELLS` at or above its cells)
//! comes whole: every cell, as the stream's own cache held it. Otherwise the cache holds every
//! cell the stream's heads read in that event and line 0 (the cell a head whose gathered
//! address is 0 finds, zero unless the event read it), a selection the engine under test
//! made (`docs/one-model.md` §4).
//!
//!   cargo run --release --example slice_full -- <record dir>... [--passes N] [--cells N]
//!
//! `--cells N` is the runs' `GBCFORMER_RECORD_CELLS`: an event of a stream memory of at most N
//! cells without its `cells` field fails. Exits 1 at the first token that differs, at a record
//! whose positions and tokens differ in length, or unless six distinct streams (by their slot
//! lists) were checked and every stream with a memory was checked over its whole cache at
//! least once.

use gbcformer::llama::{KvCache, Model};
use gbcformer::memory::attn::{Params, READ_AT};
use gbcformer::one_model::{circuit, parts};
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str, default: usize| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map_or(default, |n| n.parse().unwrap_or_else(|_| panic!("{name} N")));
    let (passes, cells_max) = (flag("--passes", 4), flag("--cells", 0));
    let dirs: Vec<&String> = args.iter().enumerate().filter(|&(i, a)| !a.starts_with("--") && (i == 0 || !args[i - 1].starts_with("--"))).map(|(_, a)| a).collect();
    assert!(!dirs.is_empty(), "usage: slice_full <record dir>... [--passes N] [--cells N]");
    let t = std::time::Instant::now();
    let (c, _) = circuit(&parts());
    let full = Model::compile_mem(&c, Params::memory());
    let heads = full.memory.as_ref().expect("the one model reads memory").heads.clone();
    eprintln!("the whole checkpoint: {} layers, hidden {}, {} tokens, compiled in {:.1} s", full.cfg.layers, full.cfg.hidden, full.cfg.vocab, t.elapsed().as_secs_f64());
    let mut failed = false;
    // Per stream, by its slot list (the record's header): events checked, those over the whole
    // cache, and whether it has a memory. The name is its first slot's.
    let mut streams: std::collections::BTreeMap<Vec<String>, (String, usize, usize, bool)> = std::collections::BTreeMap::new();
    for dir in dirs {
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir}: {e}")).filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect();
        files.sort();
        for path in files {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let mut lines = std::io::BufReader::new(std::fs::File::open(&path).unwrap()).lines();
            let header: serde_json::Value = serde_json::from_str(&lines.next().expect("a header").unwrap()).expect("the header");
            let id: Vec<String> = header["slots"].as_array().expect("the header's slots").iter().map(|s| s.as_str().unwrap().to_string()).collect();
            assert!(!id.is_empty(), "{name}: a stream with no slots");
            let (mut checked, mut tokens_checked, mut high, mut kinds, mut wrong, mut whole, mut memory) = (0usize, 0usize, 0usize, std::collections::BTreeMap::new(), 0usize, 0usize, false);
            for line in lines {
                if checked == passes {
                    break;
                }
                let e: serde_json::Value = serde_json::from_str(&line.unwrap()).expect("a record line");
                let ids = |k: &str| -> Vec<usize> { e[k].as_array().expect(k).iter().map(|x| x.as_u64().unwrap() as usize).collect() };
                let (seq, positions, out) = (ids("tokens"), ids("positions"), ids("out"));
                if positions.is_empty() {
                    continue;
                }
                if positions.len() != out.len() {
                    println!("FAIL {name} event {}: {} positions, {} tokens", e["event"], positions.len(), out.len());
                    failed = true;
                    break;
                }
                // The stream memory's cell count (0: none), and its whole image when recorded.
                let size = e["position"].as_u64().unwrap_or(0) as usize;
                let image = e["cells"].as_array();
                memory |= size > 0;
                if size > 0 && size <= cells_max && image.is_none_or(|c| c.is_empty()) {
                    println!("FAIL {name} event {}: a memory of {size} cells, at most --cells {cells_max}, recorded without its cells", e["event"]);
                    failed = true;
                    break;
                }
                whole += image.is_some_and(|c| !c.is_empty()) as usize;
                let mut cache = KvCache::new(heads.clone());
                cache.set(0, [0; 8]);
                // A small stream memory (the PPU's image) comes whole: its positional heads read
                // cells the plan's head list leaves out.
                for h in e["cells"].as_array().into_iter().flatten().chain(e["heads"].as_array().unwrap()) {
                    let h = h.as_array().unwrap();
                    let line: Vec<u8> = h[1..].iter().map(|b| b.as_u64().unwrap() as u8).collect();
                    cache.set(h[0].as_u64().unwrap() as u32, line.try_into().expect("8 bytes a line"));
                }
                let t = std::time::Instant::now();
                let (all, _) = full.forward_cached(&cache, &seq, READ_AT);
                for (&p, &want) in positions.iter().zip(&out) {
                    if all[p] != want && wrong < 5 {
                        println!("FAIL {name} event {}: position {p} ({}) the whole checkpoint decodes {}, the stream decoded {want}", e["event"], full.slots[p], all[p]);
                    }
                    wrong += (all[p] != want) as usize;
                    high += (want > usize::from(u16::MAX)) as usize;
                }
                tokens_checked += positions.len();
                *kinds.entry(e["kind"].as_str().unwrap_or("?").to_string()).or_insert(0) += 1;
                checked += 1;
                eprintln!("  {name} event {} ({}): {} tokens, {} cells, {:.1} s", e["event"], e["kind"], positions.len(), cache.len(), t.elapsed().as_secs_f64());
            }
            if checked == 0 {
                println!("skip  {dir}/{name}: no event decodes a token");
            } else if wrong > 0 {
                println!("FAIL  {dir}/{name}: {wrong} of {tokens_checked} tokens differ from the whole checkpoint's");
                failed = true;
            } else {
                let s = streams.entry(id).or_insert((name.clone(), 0, 0, false));
                (s.1, s.2, s.3) = (s.1 + checked, s.2 + whole, s.3 | memory);
                println!("ok    {dir}/{name}: {checked} events ({whole} over the whole cache) {kinds:?}, {tokens_checked} tokens equal to the whole checkpoint's, {high} of them above 65,535");
            }
        }
    }
    let names: Vec<&str> = streams.values().map(|s| s.0.as_str()).collect();
    if streams.len() != 6 {
        println!("FAIL  {} distinct streams checked, want six: {names:?}", streams.len());
        failed = true;
    }
    for (name, _, whole, memory) in streams.values() {
        if *memory && *whole == 0 {
            println!("FAIL  {name}: a stream with a memory, never checked over its whole cache");
            failed = true;
        }
    }
    std::process::exit(failed as i32);
}
