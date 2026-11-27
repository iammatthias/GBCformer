//! Prints one SHA-1 per compiled checkpoint: configuration, slot names, every weight
//! and the unit names, plus the memory spec for the memory checkpoints. A refactor of
//! the circuit or the emitter that keeps every digest left the checkpoints
//! bit-identical, which is stronger than the token gates. `--gguf <model.gguf>` prints
//! the one model's digest as compiled and again with every weight read from the file.
//! `--check <file>` holds every digest to the file's (`tools/digests.txt`) and exits 1 on
//! one that moved: the part checkpoints must not move when only the one model's
//! placement does.

use gbcformer::circuit::Circuit;
use gbcformer::llama::Model;
use gbcformer::memory::attn::Params;
use gbcformer::{apu_circuit, control_circuit, one_model, ppu_circuit, sm83};

const CHECKPOINTS: [(&str, fn() -> Circuit, bool); 8] = [
    ("cpu", sm83::build, false),
    ("cpu-memory", sm83::build_mem, true),
    ("apu", apu_circuit::build, false),
    ("control", control_circuit::build, false),
    ("ppu", ppu_circuit::circuit::build, true),
    ("one-model", one_model_circuit, true),
    ("ppu-stepper", ppu_circuit::stepper::build, true),
    ("ppu-length", ppu_circuit::circuit::build_length, true),
];

/// The merged circuit `--one-model` compiles (`one_model::streams`).
fn one_model_circuit() -> Circuit {
    one_model::circuit(&one_model::parts()).0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(p) = args.iter().position(|a| a == "--gguf") {
        let path = std::path::Path::new(&args[p + 1]);
        let mut m = Model::compile_mem(&one_model_circuit(), Params::memory());
        println!("{} (compiled)", line("one-model", &m));
        let compiled = gbcformer::gguf::digest(&m).0;
        let (manifest, _) = gbcformer::wiring::Manifest::read(path).unwrap_or_else(|e| panic!("{e}"));
        let l = gbcformer::gguf::load(&mut m, path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("{} (weights from {}: {} tensors, {} values in the file expanded to {}, {} differ)", line("one-model", &m), path.display(), l.tensors, l.file_values, l.values, l.differ);
        let loaded = gbcformer::gguf::digest(&m).0;
        println!("manifest: wiring version {}, wiring sha1 {}, tensors sha1 {}, vocabulary {}, {}", manifest.wiring_version, manifest.wiring_sha1, manifest.tensors_sha1, manifest.vocab, manifest.llama_cpp);
        println!("this build: wiring version {}, wiring sha1 {}", gbcformer::wiring::WIRING_VERSION, gbcformer::gbc_wiring::wiring_sha1());
        let ok = manifest.tensors_sha1 == compiled && loaded == compiled && manifest.vocab == m.cfg.vocab && manifest.wiring_sha1 == gbcformer::gbc_wiring::wiring_sha1();
        println!("{}: the manifest's tensors, the file's expanded tensors and this build's compiled checkpoint{}", if ok { "manifest ok" } else { "manifest MISMATCH" }, if ok { " agree, and its wiring is this build's" } else { " or wiring differ" });
        std::process::exit(!ok as i32);
    }
    let lines: Vec<String> = std::thread::scope(|s| {
        let jobs: Vec<_> = CHECKPOINTS
            .iter()
            .map(|&(name, build, memory)| {
                s.spawn(move || {
                    let c = build();
                    let m = if memory { Model::compile_mem(&c, Params::memory()) } else { Model::compile(&c) };
                    if name == "one-model" {
                        // The canonical artifact's size: the layer-shaped GGUF's stored parameters.
                        let stored = gbcformer::gguf::stored_params(&m, &gbcformer::gguf::keep(&m, 64));
                        let (digest, nonzeros) = gbcformer::gguf::digest(&m);
                        format!("{name:<10} sha1 {digest} ({:.1}M parameters stored layer-shaped ({stored}, align 64), {:.2}M nonzeros; {:.1}M as the padded uniform checkpoint)", stored as f64 / 1e6, nonzeros as f64 / 1e6, m.cfg.params() as f64 / 1e6)
                    } else {
                        line(name, &m)
                    }
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().unwrap()).collect()
    });
    for l in &lines {
        println!("{l}");
    }
    if let Some(p) = args.iter().position(|a| a == "--check") {
        let path = args.get(p + 1).expect("--check <file>");
        let pinned = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let pinned: Vec<(&str, &str)> = pinned.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()).map(|l| l.split_once(' ').unwrap_or_else(|| panic!("{path}: `{l}` is not a name and a sha1"))).collect();
        let compiled: Vec<(&str, &str)> = lines.iter().map(|l| { let mut w = l.split_whitespace(); (w.next().unwrap(), w.nth(1).unwrap()) }).collect();
        let moved: Vec<String> = pinned.iter().filter(|p| !compiled.contains(p)).map(|(name, sha1)| format!("{name} (pinned {sha1})")).collect();
        let unpinned: Vec<&str> = compiled.iter().filter(|c| !pinned.iter().any(|p| p.0 == c.0)).map(|c| c.0).collect();
        if moved.is_empty() && unpinned.is_empty() {
            println!("digests ok: {} checkpoints equal to {path}", pinned.len());
        } else {
            println!("digests MOVED against {path}: {moved:?}; not pinned: {unpinned:?}");
            std::process::exit(1);
        }
    }
}

fn line(name: &str, m: &Model) -> String {
    let (digest, nonzeros) = gbcformer::gguf::digest(m);
    format!("{name:<10} sha1 {digest} ({:.1}M parameters, {:.2}M nonzeros)", m.cfg.params() as f64 / 1e6, nonzeros as f64 / 1e6)
}
