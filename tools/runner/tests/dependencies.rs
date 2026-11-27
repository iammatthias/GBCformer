//! The runner links the inference core alone: its crate graph enables no Game Boy
//! feature of gbcformer, and its binary holds no symbol of the machine, the reference
//! interpreter, a device or a circuit.

use std::process::Command;

/// Features of gbcformer that bring in the Game Boy.
const GBC_FEATURES: [&str; 3] = ["gbc", "stock", "frontend"];
/// Modules of gbcformer behind the `gbc` feature, and the memory's fixed layout (the
/// console's address map), which the core keeps beside the generic memory model.
const GBC_MODULES: [&str; 14] = ["memory::layout", "machine", "reference", "device", "sm83", "control_circuit", "apu_circuit", "ppu_circuit", "one_model", "stack", "step", "session", "lockstep", "gbc_wiring"];

/// gbcformer's features as `cargo tree` resolves them, with `extra` arguments.
fn resolved(extra: &[&str]) -> Vec<String> {
    let out = Command::new(env!("CARGO")).args(["tree", "-e", "features", "--prefix", "none", "-i", "gbcformer"]).args(extra).current_dir(env!("CARGO_MANIFEST_DIR")).output().expect("cargo tree runs");
    assert!(out.status.success(), "cargo tree: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.strip_prefix("gbcformer feature \"").and_then(|r| r.split('"').next()).map(str::to_string)).collect()
}

fn gbc_features(features: &[String]) -> Vec<String> {
    features.iter().filter(|f| GBC_FEATURES.contains(&f.as_str())).cloned().collect()
}

#[test]
fn the_crate_graph_enables_no_game_boy_feature() {
    for extra in [&[][..], &["--features", "record"]] {
        let features = resolved(extra);
        assert_eq!(gbc_features(&features), Vec::<String>::new(), "gbcformer's features with {extra:?}: {features:?}");
    }
}

#[test]
fn the_check_sees_the_game_boy_when_it_is_enabled() {
    let features = resolved(&["--features", "gbcformer/gbc"]);
    assert_eq!(gbc_features(&features), vec!["gbc".to_string()], "the check must see the gbc feature it guards against");
}

#[test]
fn the_binary_holds_no_symbol_of_the_machine() {
    let bin = env!("CARGO_BIN_EXE_gbcformer-runner");
    let out = Command::new("nm").arg(bin).output().expect("nm runs");
    assert!(out.status.success(), "nm: {}", String::from_utf8_lossy(&out.stderr));
    let symbols = String::from_utf8_lossy(&out.stdout);
    let core = symbols.lines().filter(|l| l.contains("9gbcformer")).count();
    assert!(core > 100, "the binary links gbcformer's core ({core} symbols)");
    for module in GBC_MODULES {
        // Rust's legacy mangling writes `gbcformer::memory::layout` as `9gbcformer6memory6layout`.
        let mangled: String = std::iter::once("gbcformer").chain(module.split("::")).map(|s| format!("{}{s}", s.len())).collect();
        let hits: Vec<&str> = symbols.lines().filter(|l| l.contains(&mangled)).take(3).collect();
        assert!(hits.is_empty(), "the binary links gbcformer::{module}: {hits:?}");
    }
}
