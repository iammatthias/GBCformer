//! Checkpoints of a `play` in the runtime, so a long run goes in chunks
//! (`docs/stock-llama.md` §14).
//!
//! A checkpoint is the runtime's saved cache (sequence 0: every cell's position, key rows
//! and value rows) and a manifest: the event it was taken after, what the harness held
//! then, the counts so far, and what the run is bound to (`Identity`). An event is one
//! call of an engine into the harness: a pass, a write record or a reader record. The
//! engines and the machine are not saved. A resumed run runs the host again from frame 0
//! with every pass answered from the engine's record and no model, up to the checkpoint's
//! event; what the harness holds there must equal the manifest's `held`; then the runtime
//! loads the saved cache and decodes from the next event on (`play`).
//!
//! `Store::commit` writes the state to a temporary file, syncs and renames it, then does
//! the same for the manifest, which names the state with its size and digest. The
//! manifest's rename is the commit: a state without a manifest is ignored, and the two
//! newest checkpoints are kept, so a crash at any point leaves a whole one.
//!
//! The prefill runs inside the first event, so it has its own checkpoints (`Prefill`):
//! every sequence of the context (the cache so far and the open lanes), saved between a
//! lane batch's decode and the move of its cells, where no cell waits for a shift. A run
//! that finds one loads it, moves that batch's cells and decodes the batches after it.
//!
//! `state_diff` holds two saved caches to each other cell by cell.

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x100_0000_01b3;

/// FNV-1a over `bytes`, continuing from `h`.
pub fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(PRIME))
}

/// The length of the first `limit` bytes of `path` past `from`, and their FNV-1a continuing
/// from `h`.
pub fn fnv_file(path: &Path, from: u64, limit: u64, mut h: u64) -> std::io::Result<(u64, u64)> {
    use std::io::Seek;
    let mut f = std::fs::File::open(path)?;
    f.seek(std::io::SeekFrom::Start(from))?;
    let mut f = f.take(limit);
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            return Ok((n, h));
        }
        h = fnv(h, &buf[..k]);
        n += k as u64;
    }
}

/// A saved state's length and digest: FNV-1a over its 8-byte words (the tail as bytes).
fn state_digest(path: &Path) -> std::io::Result<(u64, u64)> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 22];
    let (mut n, mut h, mut held) = (0u64, FNV0, 0usize);
    loop {
        let k = f.read(&mut buf[held..])?;
        let have = held + k;
        let whole = if k == 0 { 0 } else { have / 8 * 8 };
        for w in buf[..whole].chunks_exact(8) {
            h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(PRIME);
        }
        n += whole as u64;
        buf.copy_within(whole..have, 0);
        held = have - whole;
        if k == 0 {
            n += held as u64;
            return Ok((n, fnv(h, &buf[..held])));
        }
    }
}

/// Stores every listed numeric field of a counts struct in a manifest and reads it back.
macro_rules! numbers {
    ($t:ty { $($f:ident),* $(,)? }) => {
        impl $t {
            pub fn to_json(&self) -> serde_json::Value {
                let mut m = serde_json::Map::new();
                $(m.insert(stringify!($f).to_string(), serde_json::Value::from(self.$f as u64));)*
                serde_json::Value::Object(m)
            }
            pub fn from_json(v: &serde_json::Value) -> Self {
                let mut c = Self::default();
                $(c.$f = v[stringify!($f)].as_u64().unwrap_or_else(|| panic!("a checkpoint's counts hold {}", stringify!($f))) as _;)*
                c
            }
        }
    };
}
pub(crate) use numbers;

/// What a checkpoint is bound to. A resume refuses one written under another value of
/// any field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity(pub Vec<(&'static str, String)>);

impl Identity {
    fn to_json(&self) -> Value {
        Value::Object(self.0.iter().map(|(k, v)| (k.to_string(), json!(v))).collect())
    }

    /// The fields whose value in `manifest` is not this run's, each with both values.
    fn differs(&self, manifest: &Value) -> Vec<String> {
        self.0.iter().filter(|(k, v)| manifest[*k].as_str() != Some(v)).map(|(k, v)| format!("{k}: the checkpoint's is {}, this run's is {v}", manifest[*k])).collect()
    }
}

/// The SHA-1 of a file's bytes.
pub fn sha1_file(path: &Path) -> String {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut h = sha1_smol::Sha1::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let k = f.read(&mut buf).unwrap();
        if k == 0 {
            return h.digest().to_string();
        }
        h.update(&buf[..k]);
    }
}

const VERSION: u64 = 1;

/// Set by SIGTERM and SIGINT: the run ends at its next checkpoint.
pub static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PROCESS: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// Seconds since the first call, which `play` makes as it starts.
pub fn process_s() -> f64 {
    PROCESS.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

/// What asks a run to stop at its next checkpoint: a signal, the file `STOCK_STOP` names
/// once it exists, or `STOCK_CHUNK_SECS` seconds of this process.
#[derive(Clone)]
pub struct Stop {
    file: Option<PathBuf>,
    chunk_secs: Option<f64>,
    polled: std::time::Instant,
}

impl Stop {
    pub fn from_env() -> Stop {
        let chunk_secs = std::env::var("STOCK_CHUNK_SECS").ok().map(|v| v.parse().unwrap_or_else(|_| panic!("STOCK_CHUNK_SECS={v}: not a number of seconds")));
        Stop { file: std::env::var_os("STOCK_STOP").map(Into::into), chunk_secs, polled: std::time::Instant::now() }
    }

    /// Why the run should stop now, if it should.
    pub fn asked(&mut self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        if self.polled.elapsed().as_secs_f64() >= 1.0 {
            self.polled = std::time::Instant::now();
            if self.file.as_ref().is_some_and(|f| f.exists()) {
                STOP.store(true, Relaxed);
            }
        }
        if STOP.load(Relaxed) {
            return Some("asked to stop (a signal or the stop file)".to_string());
        }
        self.chunk_secs.filter(|&c| process_s() >= c).map(|c| format!("the chunk's {c:.0} seconds"))
    }
}

/// The prefill's checkpoints, in the run's checkpoint directory: `prefill.json` names the
/// one state kept. The first checkpoint after an event removes them (`Store::commit`).
pub struct Prefill {
    dir: PathBuf,
    identity: Identity,
    every: f64,
    last: std::time::Instant,
    stop: Stop,
    /// This process's number among the run's chunks, and the seconds of those before it.
    pub chunk: u64,
    pub prior_s: f64,
    pub saved: u64,
    pub cost_s: f64,
}

/// A prefill another chunk began: its state, the lane batches it decoded (the last one's
/// cells not yet moved) and the seconds it prefilled.
pub struct Begun {
    pub state: String,
    pub batches: usize,
    pub prefill_s: f64,
}

impl Prefill {
    pub fn new(dir: &str, identity: Identity, every: f64, stop: Stop) -> Prefill {
        Prefill { dir: PathBuf::from(dir), identity, every, last: std::time::Instant::now(), stop, chunk: 1, prior_s: 0.0, saved: 0, cost_s: 0.0 }
    }

    /// The prefill to continue, of the same `cells` (the cache's sidecar text) over the
    /// same `lanes`, or None. Err when the checkpoint is another run's.
    pub fn begun(&mut self, cells: &str, lanes: usize) -> Result<Option<Begun>, String> {
        let path = self.dir.join("prefill.json");
        let Ok(text) = std::fs::read_to_string(&path) else { return Ok(None) };
        let m: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut d = self.identity.differs(&m["identity"]);
        if m["cells"].as_str() != Some(cells) {
            d.push(format!("cells: the checkpoint's are {}, this run's are {cells}", m["cells"]));
        }
        if m["lanes"].as_u64() != Some(lanes as u64) {
            d.push(format!("lanes: the checkpoint's {}, this run's {lanes}", m["lanes"]));
        }
        if m["version"].as_u64() != Some(VERSION) || !d.is_empty() {
            return Err(format!("{} was written by another run:
  {}", path.display(), d.join("
  ")));
        }
        let state = self.dir.join(m["state"]["file"].as_str().unwrap_or("?"));
        match state_digest(&state) {
            Ok((bytes, digest)) if Some(bytes) == m["state"]["bytes"].as_u64() && Some(format!("{digest:016x}").as_str()) == m["state"]["fnv"].as_str() => {}
            other => {
                eprintln!("the prefill's checkpoint is passed over: {} is not the state its manifest names ({other:?}); the prefill starts again", state.display());
                return Ok(None);
            }
        }
        self.chunk = m["chunk"].as_u64().unwrap_or(0) + 1;
        self.prior_s = m["elapsed_s"].as_f64().unwrap_or(0.0);
        self.saved = m["checkpoints"].as_u64().unwrap_or(0);
        self.cost_s = m["checkpoints_s"].as_f64().unwrap_or(0.0);
        Ok(Some(Begun { state: state.to_str().expect("a UTF-8 path").to_string(), batches: m["batches"].as_u64().expect("the batches decoded") as usize, prefill_s: m["prefill_s"].as_f64().unwrap_or(0.0) }))
    }

    /// Whether a checkpoint is due: None when not, Some(None) on the timer, and
    /// Some(Some(why)) when the run is asked to stop.
    pub fn due(&mut self) -> Option<Option<String>> {
        match self.stop.asked() {
            Some(why) => Some(Some(why)),
            None => (self.last.elapsed().as_secs_f64() >= self.every).then_some(None),
        }
    }

    /// Commits the prefill after `batches` lane batches decoded (`records` of `total` lane
    /// records): `save` writes every sequence to the path it is given.
    #[allow(clippy::too_many_arguments)]
    pub fn commit(&mut self, batches: usize, records: usize, total: usize, cells: &str, lanes: usize, prefill_s: f64, save: impl FnOnce(&str) -> bool) -> Result<(), String> {
        let io = |e: std::io::Error| format!("{}: {e}", self.dir.display());
        let t = std::time::Instant::now();
        let state = self.dir.join(format!("prefill-{batches:06}.state"));
        let tmp = state.with_extension("state.tmp");
        if !save(tmp.to_str().expect("a UTF-8 path")) {
            return Err("the runtime saved no state".into());
        }
        sync(&tmp).map_err(io)?;
        let (bytes, digest) = state_digest(&tmp).map_err(io)?;
        std::fs::rename(&tmp, &state).map_err(io)?;
        let elapsed = self.prior_s + process_s();
        let manifest = json!({
            "version": VERSION, "identity": self.identity.to_json(), "cells": cells, "lanes": lanes, "batches": batches, "records": records, "total": total, "prefill_s": prefill_s,
            "chunk": self.chunk, "elapsed_s": elapsed, "checkpoints": self.saved + 1, "checkpoints_s": self.cost_s,
            "state": { "file": state.file_name().unwrap().to_str(), "bytes": bytes, "fnv": format!("{digest:016x}") },
        });
        let path = self.dir.join("prefill.json");
        let tmp = path.with_extension("json.tmp");
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        f.write_all(serde_json::to_string_pretty(&manifest).unwrap().as_bytes()).map_err(io)?;
        f.sync_all().map_err(io)?;
        std::fs::rename(&tmp, &path).map_err(io)?;
        sync(&self.dir).map_err(io)?;
        for e in std::fs::read_dir(&self.dir).into_iter().flatten().flatten() {
            if e.path() != state && e.file_name().to_str().is_some_and(|n| n.starts_with("prefill-") && n.ends_with(".state")) {
                std::fs::remove_file(e.path()).ok();
            }
        }
        self.saved += 1;
        self.cost_s += t.elapsed().as_secs_f64();
        self.last = std::time::Instant::now();
        println!("checkpoint: the prefill, {records} of {total} lane records, chunk {}; {bytes} bytes saved in {:.1}s; {} checkpoints took {:.1}s of {:.0}s over all chunks; at unix {}", self.chunk, t.elapsed().as_secs_f64(), self.saved, self.cost_s, elapsed, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        Ok(())
    }
}

/// A directory of checkpoints.
pub struct Store {
    dir: PathBuf,
}

/// What one commit cost.
pub struct Cost {
    pub bytes: u64,
    pub save_s: f64,
    pub digest_s: f64,
}

fn sync(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

impl Store {
    pub fn open(dir: &str) -> Store {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{dir}: {e}"));
        Store { dir: PathBuf::from(dir) }
    }

    fn state(&self, event: u64) -> PathBuf {
        self.dir.join(format!("ckpt-{event:012}.state"))
    }

    fn manifest(&self, event: u64) -> PathBuf {
        self.dir.join(format!("ckpt-{event:012}.json"))
    }

    /// The checkpoints' events, newest first.
    fn events(&self) -> Vec<u64> {
        let mut events: Vec<u64> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("ckpt-")?.strip_suffix(".json")?.parse().ok()))
            .collect();
        events.sort_unstable_by(|a, b| b.cmp(a));
        events
    }

    /// Commits the checkpoint after `event`: `save` writes the runtime's state to the path
    /// it is given, and `manifest` gains the state's name, size and digest and `identity`.
    /// Older checkpoints than the one before this go.
    pub fn commit(&self, event: u64, mut manifest: Value, identity: &Identity, save: impl FnOnce(&str) -> bool) -> Result<Cost, String> {
        let io = |e: std::io::Error| format!("{}: {e}", self.dir.display());
        let state = self.state(event);
        let tmp = state.with_extension("state.tmp");
        let t = std::time::Instant::now();
        if !save(tmp.to_str().expect("a UTF-8 path")) {
            return Err("the runtime saved no state".into());
        }
        sync(&tmp).map_err(io)?;
        let save_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let (bytes, digest) = state_digest(&tmp).map_err(io)?;
        let digest_s = t.elapsed().as_secs_f64();
        std::fs::rename(&tmp, &state).map_err(io)?;
        manifest["version"] = json!(VERSION);
        manifest["event"] = json!(event);
        manifest["identity"] = identity.to_json();
        manifest["state"] = json!({ "file": state.file_name().unwrap().to_str(), "bytes": bytes, "fnv": format!("{digest:016x}") });
        let path = self.manifest(event);
        let tmp = path.with_extension("json.tmp");
        let mut f = std::fs::File::create(&tmp).map_err(io)?;
        f.write_all(serde_json::to_string_pretty(&manifest).unwrap().as_bytes()).map_err(io)?;
        f.sync_all().map_err(io)?;
        std::fs::rename(&tmp, &path).map_err(io)?;
        sync(&self.dir).map_err(io)?;
        for old in self.events().into_iter().skip(2) {
            std::fs::remove_file(self.manifest(old)).ok();
            std::fs::remove_file(self.state(old)).ok();
        }
        // A state whose manifest never landed (a crash between the two renames), and the
        // prefill's checkpoint, which this one supersedes.
        let kept = self.events();
        for e in std::fs::read_dir(&self.dir).into_iter().flatten().flatten() {
            let name = e.file_name();
            let stray = name.to_str().and_then(|n| n.strip_prefix("ckpt-")?.strip_suffix(".state")?.parse::<u64>().ok()).is_some_and(|ev| !kept.contains(&ev));
            if stray || name.to_str().is_some_and(|n| n.ends_with(".tmp") || n.starts_with("prefill")) {
                std::fs::remove_file(e.path()).ok();
            }
        }
        Ok(Cost { bytes, save_s, digest_s })
    }

    /// The newest whole checkpoint: its manifest and its state's path. None without one.
    /// Refused (Err) when the newest was written under another identity or by another
    /// version of this file. A checkpoint whose state is missing, short or changed is
    /// passed over for the one before it, and said so.
    pub fn latest(&self, identity: &Identity) -> Result<Option<(Value, PathBuf)>, String> {
        for event in self.events() {
            let path = self.manifest(event);
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let m: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            if m["version"].as_u64() != Some(VERSION) || m["event"].as_u64() != Some(event) {
                return Err(format!("{}: not a version {VERSION} checkpoint of event {event}", path.display()));
            }
            let d = identity.differs(&m["identity"]);
            if !d.is_empty() {
                return Err(format!("{} was written by another run:\n  {}", path.display(), d.join("\n  ")));
            }
            let state = self.dir.join(m["state"]["file"].as_str().unwrap_or("?"));
            match state_digest(&state) {
                Ok((bytes, digest)) if Some(bytes) == m["state"]["bytes"].as_u64() && Some(format!("{digest:016x}").as_str()) == m["state"]["fnv"].as_str() => return Ok(Some((m, state))),
                Ok((bytes, digest)) => eprintln!("checkpoint {event} passed over: {} holds {bytes} bytes with digest {digest:016x}, its manifest names {} and {}", state.display(), m["state"]["bytes"], m["state"]["fnv"]),
                Err(e) => eprintln!("checkpoint {event} passed over: {}: {e}", state.display()),
            }
        }
        Ok(None)
    }
}

/// One cell of a saved sequence: its position, a digest of its key and value rows, and
/// the digest with every f16 key entry that is -0 read as +0.
type CellDigest = (i32, u64, u64);

/// The cells of a sequence llama.cpp saved (`llama_state_seq_save_file` over
/// `llama_kv_cache::state_write`): a 12-byte header and no prompt, the stream count (1),
/// the cell count, per cell its position, its sequence count (1), an optional 12-byte
/// extension and its sequence, then four 32-bit counts (`v_trans`, which is 0, the
/// layers and two rotation counts), per layer the key's type, its row size and every
/// cell's row, and the same per layer for the values. The cell extension is there or not:
/// the layout that ends exactly at the end of the file is taken.
fn cells_of_state(path: &str) -> Result<Vec<CellDigest>, String> {
    const F16: u32 = 1;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let word = |i: usize| bytes.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let long = |i: usize| bytes.get(i..i + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize);
    if word(8) != Some(0) || word(12) != Some(1) {
        return Err(format!("{path}: not a saved sequence of one stream with no prompt"));
    }
    let n = word(16).ok_or("short file")? as usize;
    'meta: for meta in [12usize, 24] {
        let data = 20 + n * meta;
        let (Some(0), Some(layers)) = (word(data), word(data + 4)) else { continue };
        // Per row family (keys, then values) and layer: the type, the row size, the first row.
        let mut rows = Vec::new();
        let mut at = data + 16;
        for _ in 0..2 * layers {
            let (Some(kind), Some(len)) = (word(at), long(at + 4)) else { continue 'meta };
            rows.push((kind, len, at + 12));
            at += 12 + len * n;
        }
        if at != bytes.len() {
            continue;
        }
        let mut cells = Vec::with_capacity(n);
        for c in 0..n {
            let m = 20 + c * meta;
            if word(m + 4) != Some(1) {
                return Err(format!("{path}: cell {c} is in {:?} sequences, not one", word(m + 4)));
            }
            let (mut exact, mut signless) = (FNV0, FNV0);
            for (r, &(kind, len, first)) in rows.iter().enumerate() {
                let row = &bytes[first + c * len..first + (c + 1) * len];
                exact = fnv(exact, row);
                if r < layers as usize && kind == F16 {
                    for e in row.chunks_exact(2) {
                        signless = fnv(signless, if e == [0, 0x80] { &[0, 0] } else { e });
                    }
                } else {
                    signless = fnv(signless, row);
                }
            }
            cells.push((word(m).unwrap() as i32, exact, signless));
        }
        cells.sort_unstable();
        return Ok(cells);
    }
    Err(format!("{path}: no layout of {n} cells ends at the end of its {} bytes", bytes.len()))
}

/// Holds the cells of two saved caches to each other by position (a position may hold
/// two cells: line 0's and the null cell). Prints the counts; true when every cell of one
/// has its equal in the other, byte for byte or but for the sign of zero f16 key entries.
pub fn state_diff(a: &str, b: &str) -> bool {
    let (ca, cb) = match (cells_of_state(a), cells_of_state(b)) {
        (Ok(ca), Ok(cb)) => (ca, cb),
        (ra, rb) => {
            for e in [ra.err(), rb.err()].into_iter().flatten() {
                println!("STATE UNREADABLE: {e}");
            }
            return false;
        }
    };
    let (mut exact, mut signs, mut differ, mut alone) = (0u64, 0u64, 0u64, 0u64);
    let (mut i, mut j) = (0, 0);
    let mut first: Option<i32> = None;
    while i < ca.len() || j < cb.len() {
        let p = match (ca.get(i), cb.get(j)) {
            (Some(x), Some(y)) => x.0.min(y.0),
            (Some(x), None) => x.0,
            (None, Some(y)) => y.0,
            (None, None) => unreachable!(),
        };
        let (i0, j0) = (i, j);
        while ca.get(i).is_some_and(|c| c.0 == p) {
            i += 1;
        }
        while cb.get(j).is_some_and(|c| c.0 == p) {
            j += 1;
        }
        let (xa, xb) = (&ca[i0..i], &cb[j0..j]);
        if xa.len() != xb.len() {
            alone += xa.len().abs_diff(xb.len()) as u64;
            first.get_or_insert(p);
            continue;
        }
        // The cells of a position are sorted by their exact digest, so equal sets pair up.
        if xa.iter().zip(xb).all(|(x, y)| x.1 == y.1) {
            exact += xa.len() as u64;
            continue;
        }
        let mut sb: Vec<u64> = xb.iter().map(|c| c.2).collect();
        let mut sa: Vec<u64> = xa.iter().map(|c| c.2).collect();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa == sb {
            signs += xa.len() as u64;
        } else {
            differ += xa.len() as u64;
            first.get_or_insert(p);
        }
    }
    println!("state diff: {} and {} cells; {exact} equal byte for byte, {signs} at positions whose cells are equal but for the sign of zero f16 key entries, {differ} differ, {alone} at a position only one holds{}", ca.len(), cb.len(), first.map_or(String::new(), |p| format!("; first at position {p:#x}")));
    differ == 0 && alone == 0 && !ca.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("stock-llama-ckpt-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        d.to_str().unwrap().to_string()
    }

    fn id(model: &str) -> Identity {
        Identity(vec![("model", model.to_string()), ("rom", "r".to_string())])
    }

    fn put(s: &Store, event: u64, bytes: &'static [u8]) {
        s.commit(event, json!({ "held": { "spent": event } }), &id("m"), |p| std::fs::write(p, bytes).is_ok()).unwrap();
    }

    #[test]
    fn the_newest_whole_checkpoint_is_the_one_resumed() {
        let s = Store::open(&dir("newest"));
        assert!(s.latest(&id("m")).unwrap().is_none());
        put(&s, 5, b"five");
        put(&s, 9, b"nine and more");
        let (m, state) = s.latest(&id("m")).unwrap().unwrap();
        assert_eq!((m["event"].as_u64(), m["held"]["spent"].as_u64(), m["state"]["bytes"].as_u64()), (Some(9), Some(9), Some(13)));
        assert_eq!(std::fs::read(state).unwrap(), b"nine and more");
    }

    #[test]
    fn a_checkpoint_of_another_run_is_refused_and_names_the_field() {
        let s = Store::open(&dir("identity"));
        put(&s, 5, b"five");
        let e = s.latest(&id("other")).unwrap_err();
        assert!(e.contains("model: the checkpoint's is \"m\", this run's is other"), "{e}");
        assert!(!e.contains("rom"), "{e}");
    }

    #[test]
    fn a_state_that_changed_or_is_short_is_passed_over_for_the_checkpoint_before() {
        let s = Store::open(&dir("short"));
        put(&s, 5, b"five");
        put(&s, 9, b"nine and more");
        std::fs::write(s.state(9), b"nine and mor").unwrap();
        assert_eq!(s.latest(&id("m")).unwrap().unwrap().0["event"].as_u64(), Some(5));
        std::fs::write(s.state(9), b"nine and morE").unwrap();
        assert_eq!(s.latest(&id("m")).unwrap().unwrap().0["event"].as_u64(), Some(5));
        std::fs::remove_file(s.state(9)).unwrap();
        assert_eq!(s.latest(&id("m")).unwrap().unwrap().0["event"].as_u64(), Some(5));
    }

    #[test]
    fn a_state_without_a_manifest_is_no_checkpoint() {
        let s = Store::open(&dir("uncommitted"));
        put(&s, 5, b"five");
        std::fs::write(s.state(7), b"seven").unwrap();
        std::fs::write(s.state(8).with_extension("state.tmp"), b"eight").unwrap();
        assert_eq!(s.latest(&id("m")).unwrap().unwrap().0["event"].as_u64(), Some(5));
        // A save that fails leaves the last checkpoint as it was.
        assert!(s.commit(9, json!({}), &id("m"), |_| false).is_err());
        assert_eq!(s.latest(&id("m")).unwrap().unwrap().0["event"].as_u64(), Some(5));
    }

    #[test]
    fn the_two_newest_checkpoints_are_kept() {
        let s = Store::open(&dir("prune"));
        std::fs::write(s.state(1), b"stray").unwrap();
        for e in [2, 4, 6, 8] {
            put(&s, e, b"state");
        }
        assert_eq!(s.events(), [8, 6]);
        let mut names: Vec<String> = std::fs::read_dir(&s.dir).unwrap().flatten().map(|e| e.file_name().into_string().unwrap()).collect();
        names.sort();
        assert_eq!(names, ["ckpt-000000000006.json", "ckpt-000000000006.state", "ckpt-000000000008.json", "ckpt-000000000008.state"]);
    }

    #[test]
    fn a_prefill_s_checkpoint_is_continued_by_the_same_run_only_and_goes_with_the_first_event_s() {
        let d = dir("prefill");
        let s = Store::open(&d);
        let stop = Stop { file: None, chunk_secs: None, polled: std::time::Instant::now() };
        let mut p = Prefill::new(&d, id("m"), 1e9, stop.clone());
        assert!(p.begun("cells", 255).unwrap().is_none());
        assert!(p.due().is_none());
        p.commit(3, 765, 9000, "cells", 255, 12.5, |path| std::fs::write(path, b"three").is_ok()).unwrap();
        p.commit(7, 1785, 9000, "cells", 255, 30.0, |path| std::fs::write(path, b"seven").is_ok()).unwrap();
        assert!(!Path::new(&d).join("prefill-000003.state").exists(), "one state is kept");
        let mut q = Prefill::new(&d, id("m"), 1e9, stop.clone());
        let b = q.begun("cells", 255).unwrap().unwrap();
        assert_eq!((b.batches, b.prefill_s, q.chunk, q.saved), (7, 30.0, 2, 2));
        assert_eq!(std::fs::read(&b.state).unwrap(), b"seven");
        // Another model, other cells, other lanes: refused, and the field named.
        assert!(Prefill::new(&d, id("other"), 1e9, stop.clone()).begun("cells", 255).err().unwrap().contains("model:"));
        assert!(q.begun("other cells", 255).err().unwrap().contains("cells:"));
        assert!(q.begun("cells", 96).err().unwrap().contains("lanes:"));
        // A changed state: the prefill starts again.
        std::fs::write(&b.state, b"seveN").unwrap();
        assert!(q.begun("cells", 255).unwrap().is_none());
        // The first checkpoint after an event removes the prefill's.
        put(&s, 5, b"five");
        assert!(!Path::new(&d).join("prefill.json").exists() && !Path::new(&b.state).exists());
    }

    #[test]
    fn a_run_is_asked_to_stop_by_its_stop_file() {
        let d = dir("stop");
        std::fs::create_dir_all(&d).unwrap();
        let file = Path::new(&d).join("stop");
        let mut stop = Stop { file: Some(file.clone()), chunk_secs: None, polled: std::time::Instant::now() - std::time::Duration::from_secs(2) };
        assert!(stop.asked().is_none());
        std::fs::write(&file, b"").unwrap();
        stop.polled -= std::time::Duration::from_secs(2);
        assert!(stop.asked().is_some());
        STOP.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn the_state_digest_reads_every_byte() {
        let d = dir("digest");
        std::fs::create_dir_all(&d).unwrap();
        let p = Path::new(&d).join("s");
        let base: Vec<u8> = (0..100_003u32).map(|i| (i * 7) as u8).collect();
        std::fs::write(&p, &base).unwrap();
        let want = state_digest(&p).unwrap();
        assert_eq!(want.0, base.len() as u64);
        for at in [0, 8, 99_999, 100_002] {
            let mut b = base.clone();
            b[at] ^= 1;
            std::fs::write(&p, &b).unwrap();
            assert_ne!(state_digest(&p).unwrap().1, want.1, "byte {at}");
        }
        assert_eq!(fnv_file(&p, 3, 10, FNV0).unwrap().0, 10);
    }

    /// A saved sequence of `cells` (position, f16 key row, value row), one layer.
    fn saved(cells: &[(i32, [u8; 4], [u8; 2])]) -> Vec<u8> {
        let mut b = Vec::new();
        for w in [1u32, 2, 0, 1, cells.len() as u32] {
            b.extend(w.to_le_bytes());
        }
        for c in cells {
            b.extend(c.0.to_le_bytes());
            b.extend(1u32.to_le_bytes());
            b.extend(0u32.to_le_bytes());
        }
        for w in [0u32, 1, 0, 0] {
            b.extend(w.to_le_bytes());
        }
        for (len, row) in [(4u64, 0), (2, 1)] {
            b.extend(1u32.to_le_bytes());
            b.extend(len.to_le_bytes());
            for c in cells {
                b.extend(if row == 0 { &c.1[..] } else { &c.2[..] });
            }
        }
        b
    }

    #[test]
    fn two_states_are_held_to_each_other_by_position_whatever_their_order() {
        let d = dir("diff");
        std::fs::create_dir_all(&d).unwrap();
        let write = |name: &str, cells: &[(i32, [u8; 4], [u8; 2])]| {
            let p = format!("{d}/{name}");
            std::fs::write(&p, saved(cells)).unwrap();
            p
        };
        let (k, v) = ([0, 0, 5, 0x3c], [9, 9]);
        let a = write("a", &[(0, k, v), (0, [1, 1, 1, 1], v), (7, k, v)]);
        // The same cells in another order, position 0's two swapped.
        assert!(state_diff(&a, &write("b", &[(7, k, v), (0, [1, 1, 1, 1], v), (0, k, v)])));
        // A zero key entry of the other sign.
        assert!(state_diff(&a, &write("c", &[(0, k, v), (0, [1, 1, 1, 1], v), (7, [0, 0x80, 5, 0x3c], v)])));
        // Another key, another value, a cell moved, a cell missing.
        assert!(!state_diff(&a, &write("d", &[(0, k, v), (0, [1, 1, 1, 1], v), (7, [0, 0, 5, 0x3d], v)])));
        assert!(!state_diff(&a, &write("e", &[(0, k, v), (0, [1, 1, 1, 1], v), (7, k, [9, 8])])));
        assert!(!state_diff(&a, &write("f", &[(0, k, v), (0, [1, 1, 1, 1], v), (8, k, v)])));
        assert!(!state_diff(&a, &write("g", &[(0, k, v), (7, k, v)])));
        assert!(!state_diff(&a, &format!("{d}/none")));
    }
}
