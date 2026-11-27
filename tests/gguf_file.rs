//! The file-alone path (`gguf::open_model`, which the runner and `GBCFORMER_GGUF_ONLY`
//! use) on a small exported file: the GBC CPU checkpoint as `export --layer-shapes` writes
//! it. The file as exported opens. A copy whose placement key names another run of the
//! same size keeps every shape, so only the digest can refuse it, and it must. The
//! uniform file, which unpatched llama.cpp loads, opens too and stores every value.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The file the export example writes with `flags`, in a directory of its own.
fn exported(flags: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gbcformer-gguf-file-{}{}", std::process::id(), flags.concat()));
    let mut cargo = Command::new(env!("CARGO"));
    cargo.args(["run", "--quiet", "--example", "export"]);
    if !cfg!(debug_assertions) {
        cargo.arg("--release");
    }
    if cfg!(feature = "testing") {
        cargo.args(["--features", "testing"]);
    }
    let out = cargo.arg("--").arg("cpu").arg(&dir).args(flags).current_dir(env!("CARGO_MANIFEST_DIR")).output().expect("cargo runs the export example");
    assert!(out.status.success(), "export: {}", String::from_utf8_lossy(&out.stderr));
    dir.join("model.gguf")
}

/// The byte offset of the u32 values of the metadata array `key`, and their count.
fn u32_array(bytes: &[u8], key: &str) -> Option<(usize, usize)> {
    let mut named = (key.len() as u64).to_le_bytes().to_vec();
    named.extend(key.as_bytes());
    let at = bytes.windows(named.len()).position(|w| w == named)? + named.len();
    let u32_at = |p: usize| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
    // GGUF value type 9 is an array, element type 4 a u32.
    assert_eq!((u32_at(at), u32_at(at + 4)), (9, 4), "{key} is an array of u32");
    Some((at + 16, u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap()) as usize))
}

fn values(bytes: &[u8], (at, n): (usize, usize)) -> Vec<u32> {
    (0..n).map(|i| u32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap())).collect()
}

/// A copy of the file with value `i` of the array at `at` replaced.
fn with_value(bytes: &[u8], at: usize, i: usize, v: u32, path: &Path) {
    let mut lie = bytes.to_vec();
    lie[at + 4 * i..at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
    std::fs::write(path, lie).unwrap();
}

#[test]
fn a_file_whose_placement_key_lies_is_refused() {
    let path = exported(&["--gguf-only", "--layer-shapes", "--align", "64"]);
    let (m, _, loaded) = gbcformer::gguf::open_model(&path).expect("the exported file opens");
    assert!(loaded.values > loaded.file_values, "the file stores less than the compiled layout ({} of {})", loaded.file_values, loaded.values);
    let (d, vocab) = (m.cfg.hidden as u32, m.cfg.vocab as u32);
    let bytes = std::fs::read(&path).unwrap();
    let lie = path.with_file_name("lie.gguf");
    let mut lies = 0;
    let mut refuse = |what: &str| {
        let err = gbcformer::gguf::open_model(&lie).err().unwrap_or_else(|| panic!("{what}: the file opened"));
        assert!(err.contains("a tensor or a placement key is not the exported one"), "{what}: refused for another reason: {err}");
        lies += 1;
    };
    // A layer's run of `ffn_up` columns moved by one quantum, the same count.
    if let Some(run) = u32_array(&bytes, gbcformer::gguf::UP_COLS_KEY) {
        let v = values(&bytes, run);
        if let Some((l, first)) = (0..v.len() / 2).find_map(|l| {
            let (first, count) = (v[2 * l], v[2 * l + 1]);
            if first + count + 64 <= d { Some((l, first + 64)) } else { first.checked_sub(64).map(|f| (l, f)) }
        }) {
            with_value(&bytes, run.0, 2 * l, first, &lie);
            refuse("a moved run of up columns");
        }
    }
    // The LM head's run of rows moved by one token, the same count.
    if let Some(run) = u32_array(&bytes, gbcformer::gguf::HEAD_ROWS_KEY) {
        let v = values(&bytes, run);
        if let Some(first) = if v[0] + v[1] < vocab { Some(v[0] + 1) } else { v[0].checked_sub(1) } {
            with_value(&bytes, run.0, 0, first, &lie);
            refuse("a moved run of head rows");
        }
    }
    assert!(lies > 0, "the exported file has a run that can move within its tensor");
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

/// The uniform export keeps every layer at the checkpoint's head count and FFN width,
/// so llama.cpp needs no patch to load it. It opens from the file alone, which holds
/// the rebuilt weights to the digest the export named, and it stores every value.
#[test]
fn the_uniform_file_opens_and_stores_every_value() {
    let path = exported(&["--gguf-only"]);
    let (m, _, loaded) = gbcformer::gguf::open_model(&path).expect("the uniform file opens");
    assert_eq!(loaded.file_values, loaded.values, "the uniform file stores the compiled layout whole");
    let c = &m.cfg;
    let per_layer = 2 * c.attn() * c.hidden + 2 * c.kv_dim() * c.hidden + 3 * c.intermediate * c.hidden;
    assert_eq!(loaded.values, 2 * c.vocab * c.hidden + c.layers * per_layer, "every weight matrix at its full shape");
    let bytes = std::fs::read(&path).unwrap();
    for key in [gbcformer::gguf::UP_COLS_KEY, gbcformer::gguf::HEAD_ROWS_KEY, gbcformer::gguf::EMBED_COLS_KEY, gbcformer::gguf::HEAD_COLS_KEY] {
        assert!(!bytes.windows(key.len()).any(|w| w == key.as_bytes()), "the uniform file holds {key}, which only patched llama.cpp reads");
    }
    assert!(!bytes.windows(17).any(|w| w == b"gbcformer.stream."), "the uniform file names a stream's slice");
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}
