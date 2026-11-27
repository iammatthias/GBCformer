//! Gameplay with every pass of every stream in a stock runtime. A `Session` runs every
//! component in weights as streams of the one checkpoint (`--one-model`), and each
//! stream's engine hands its passes, writes and reader records to the runtime
//! (`engine::stock::set_external`). Every stream decodes over one cache sequence that
//! holds the machine's full memory. The host references run beside it (`--lockstep
//! all`), and the engine's own run is recorded the same way, so `diff` holds every
//! pass's tokens to the engine's.
//!
//! The PPU's, the stepper's and the length's streams follow the GBC CPU's VRAM and OAM
//! records instead of keeping their own, and each applies them at a line's cut, the start
//! of its mode 3. The engine draws the line when it closes, at the next line's cut, so by
//! that pass the GBC CPU has kept a line's worth of later records, and the cache's cells
//! hold them. `Stream::own` holds each such stream's lines as its engine wrote them. A
//! pass first lends the stream's line to every cell that holds another and puts the cell's
//! line back after its last decode (`Cache::lend`, `Cache::put_back`), so the pass attends
//! exactly the records its engine's image held. The counts name the passes and lines
//! swapped, and the cache's the cells put back that it held to the cells it found, byte for
//! byte or but for the sign of a zero key entry (`Cache::decoded`). `STOCK_SWAP=0` lends
//! nothing, and `STOCK_LENDS=<file>` logs every pass that lends: the frame, the stream, its
//! first input (LY) and the lines.
//!
//! `play replay` runs the same harness with no model: each pass decodes the tokens of the
//! engine's record (`STOCK_REFERENCE`, required), and a reader record reads the mirror. It
//! is not a check of any model. It runs the scopes, the loans, the mirror check and the
//! asserts in the run's true order, so it counts what a run in the runtime will lend.
//!
//! A stream decodes its reduced group (`one_model::group`, its own slots and what the
//! layer-0 gathers need of the others, `docs/stock-llama.md` §12a). `STOCK_GROUP=full`
//! decodes every slot of the checkpoint instead, and `STOCK_GROUP=check` decodes both and
//! requires bit-identical rows. A full group is not exact in llama.cpp: another part's
//! mask carries that part's mode while the channels hold this stream's wires, so that
//! part's units can leave f16's range at `ffn_down`'s input (§12a).
//! A reduced group also leaves out each own mask the pass does not decode, unless a
//! layer-0 gather targets it (`Stream::reduced`). `STOCK_MASKS=0` keeps them, and
//! `STOCK_MASKS=check` decodes both groups and requires bit-identical rows.
//! Each stream decodes its rows over a scoped sequence of the lines its heads can name
//! (`scope`): in an overlay its own heads' lines, since every other part's head attends
//! the null cell at its tokens (`STOCK_OWN=0` keeps every head's lines). A GBC CPU pass
//! that decodes the address of each of its reads and writes attends the narrow scope
//! (`CpuScope`): a few fixed regions, the lines the bytes at PC and at SP + 1 sit in, and
//! the lines the pass names, with at most `STOCK_CPU_LRU` (256) named lines kept. A pass that names a
//! line outside it decodes again with the line. `STOCK_CPU_SCOPE=windows` decodes every GBC
//! CPU pass over the banked windows instead. `STOCK_SCOPE=check` also decodes every scoped
//! pass over every cell and requires bit-identical rows (from pass `STOCK_CHECK_FROM` on,
//! when that is set, and only every `STOCK_CHECK_EVERY`th pass of the run, when that is
//! set: the form a long run can afford), and reads the pass's sequence back from the
//! runtime and holds its cells to the lines the harness takes it to hold (`Cache::audit`).
//! `STOCK_SCOPE=0` scopes nothing. Under `STOCK_REFERENCE` a scoped pass's scope must hold
//! every line the engine's record names for the pass's heads, the positional heads' too,
//! or the run stops at `SCOPE MISSES LINE`.
//!
//! What holds the narrow scope without the engine's record:
//! - at start-up, the wiring asserts and a sample of the GBC CPU's own circuit's reads
//!   (`scope::named_slots_wiring`, `scope::sampled_reads`);
//! - a pass must name the windows its lines were placed under (`MAP`), since no write
//!   record changed a banking register between: `NARROW SCOPE NOT HELD`, or `WINDOWS SCOPE
//!   NOT HELD` for a pass over the windows' scope;
//! - a reader record must decode the mirror's line (`READ NOT HELD`), and a row that
//!   differs between two decodes a check compares stops the run (`CHECK NOT HELD`), with
//!   or without the record;
//! - the sampled `STOCK_SCOPE=check`.
//!
//! A head whose line the scope lacks reads whatever its scores pick among the cells the
//! scope holds, and the model gives no sign of it. So without the record a misplaced line
//! that none of these sees decodes a wrong token unseen. The proof runs under the record.
//!
//! Deliberate faults, each to show a stop: `STOCK_FAULT_SCOPE=<k>` drops a line narrow
//! pass k names and decodes it again (`SCOPE MISSES LINE` under the record, the scope
//! check's rows without it); `STOCK_FAULT_MAP=<k>` and `STOCK_FAULT_PROBE=<k>` move a
//! window (`NARROW SCOPE NOT HELD`, both forms); `STOCK_FAULT_NAMED=<k>` moves an address
//! a pass names when it decodes again (`NARROW SCOPE NOT HELD`); `STOCK_FAULT_READ=<k>`
//! flips a bit of a reader record's line (`READ NOT HELD`); `STOCK_FAULT_SEQUENCE=<k>`
//! takes a cell out of a sequence behind the harness's account (`SCOPE SEQUENCE DIFFERS`).
//! `tools/stock-llama/faults.sh` runs those a replay can show.
//!
//! Each pass runs only its stream's slice of the model when the runtime can
//! (`Runtime::stream`: llama.cpp with `stream-slices.patch`), and `play` says for each
//! stream which it is. `STOCK_STREAMS=0` runs the whole model for every pass,
//! `STOCK_STREAMS=check` decodes every pass both ways and requires bit-identical rows, and
//! `STOCK_STREAMS=require` refuses a runtime that cannot run a stream's slice. In every
//! mode the first sliced pass of each form (stream, count of tokens) is decoded by the
//! whole model too, and then one sliced pass in `STOCK_SLICE_EVERY` (default 1,000): a
//! row that differs stops the run (`CHECK NOT HELD`). A backend that reads a slice's view
//! of the weights wrong gives finite scores, so nothing else would show it without the
//! record. `STOCK_SLICE_EVERY=0` holds no sliced pass to the whole model, and `play` says
//! so: a BLAS-off build's Mac CPU backend needs it, because it sums f16 products in f16 and
//! its whole model's decode of a GBC CPU pass is NaN, patched or not (`docs/perf-log.md`,
//! "The Mac CPU backend's NaN").
//! Whether a pass ran its slice is the runtime's own report (`Runtime::sliced`), never
//! inferred. The runtime runs a group of at most its fence of tokens (`Runtime::fence`)
//! on the whole model whatever stream is named; those passes are counted and printed. A
//! pass above the fence that names a stream and still runs the whole model stops the run.
//!
//! A write record decodes over the cells the prefill's lanes saw (`Cache::scope_records`).
//! `STOCK_WRITES=0` decodes it over every cell, and `STOCK_WRITES=check` decodes it both
//! ways and requires the two cells' saved bytes equal. A reader record decodes over the
//! same cells and the two lines it names (`Cache::read`). `STOCK_READS=0` decodes it over
//! every cell, and `STOCK_READS=check` decodes it both ways and requires bit-identical
//! rows. The line a reader record decodes must be the line the mirror gives.
//! `STOCK_READ_PROBE=N` decodes N reader records of evenly spaced lines after the prefill
//! and prints what each took (`Cache::probe_reads`).
//!
//! `STOCK_CELLS=<file>` logs a hash of every written cell's saved bytes, to compare the
//! cells two builds of the runtime write.
//!
//! `STOCK_REFERENCE=<dir>` holds every decoded pass, as it decodes, to the engine's record
//! of the same run in `<dir>` (`GBCFORMER_RECORD`): the sequence, the positions and the
//! tokens. The first difference, or a record that ends first, stops the run, and so does
//! the first `MIRROR DIFFERS` and the first cell put back that differs from the cell found
//! or is never compared (`CacheStats::unheld`).
//! `STOCK_PROGRESS=N` prints the counts every N passes (default 10,000).
//!
//! `STOCK_CHECKPOINT=<dir>` runs in chunks (`checkpoint`, `docs/stock-llama.md` §14). After
//! an event (a pass, a write record or a reader record) at which the cache is quiet
//! (`Cache::quiet`), a checkpoint saves the runtime's cache and what the harness holds
//! (`held`): every `STOCK_CHECKPOINT_SECS` seconds (default 600), and at once when the run
//! is asked to stop, which then ends it with exit 0: by SIGTERM or SIGINT, by the file
//! `STOCK_STOP` names, after `STOCK_CHUNK_SECS` seconds, or at `--max-passes`. A run that
//! finds a checkpoint in the directory resumes it: the host runs again from frame 0 over
//! `runtime::Resuming`, with no model and every pass answered from the engine's record
//! (`STOCK_REFERENCE`, required), up to the checkpoint's event. There it must hold what the
//! checkpoint holds (each stream's counts, its place in the engine's record with a digest
//! of the lines read, its own lines, the mirror, the GBC CPU's scope, and a digest of every
//! record file the engines rewrote), or the run stops at `RESUME DIFFERS`. Then the runtime
//! loads the saved cache, the counts carry on from the checkpoint's, and llama.cpp decodes
//! every later pass. A checkpoint written under another model file, harness, ROM,
//! llama.cpp, engine's record or setting is refused (`RESUME REFUSED`). The frames and
//! passes of the replayed stretch are labelled as such; `--max-passes` counts them too.
//! `STOCK_SHIFT_PROBE=1` reads each written cell's saved bytes before and after the
//! runtime's shift and counts how they compare (the cache's `probe_` counts).

use crate::cache::{Cache, P};
use crate::checkpoint::{fnv, Identity, Store, FNV0};
use crate::runtime::Scoped;
use crate::cpp::Row;
use gbcformer::engine::stock::{set_external, External};
use gbcformer::machine::io::Buttons;
use gbcformer::memory::image::Image;
use gbcformer::llama::tok;
use gbcformer::memory::layout::{LINE, LINE_BITS, N_REGS, OAM, OAM_LEN, PAL, PAL_LEN, REGS, VRAM, VRAM_LEN};
use gbcformer::one_model::Part;
use gbcformer::output::frame_sha1;
use gbcformer::{Hardware, Instrumentation, Session, Stack};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Options {
    pub model: Option<String>,
    pub rom: String,
    pub frames: u64,
    pub lockstep: String,
    pub threads: usize,
    pub max_passes: u64,
    /// Frames run first on the reference interpreter, labelled, before the hand-off.
    pub warmup: u64,
    pub lanes: usize,
    /// The console: None follows the header, as the main binary's `--hardware auto`.
    pub hardware: Option<Hardware>,
    /// A boot ROM file the run starts in, from power-on, as the main binary's `--boot`;
    /// None starts at the hand-off.
    pub boot: Option<String>,
}

#[derive(Default, Clone, Debug)]
pub struct Counts {
    pub passes: u64,
    pub tokens: u64,
    /// Decodes of the stream's groups (a pass may decode more than once), the tokens their
    /// batches held, the logit rows they asked for and the tokens that attended the cells.
    pub decodes: u64,
    pub batch: u64,
    pub asked: u64,
    pub attending: u64,
    pub nanos: u128,
    pub decode_nanos: u128,
    pub writes: u64,
    pub skipped_writes: u64,
    pub reads: u64,
    pub mirror_checks: u64,
    pub mirror_differs: u64,
    pub group_checks: u64,
    pub group_rows_differ: u64,
    pub mask_checks: u64,
    pub mask_rows_differ: u64,
    pub scope_checks: u64,
    pub scope_rows_differ: u64,
    /// Time in the decodes over every cell that `STOCK_SCOPE=check` adds.
    pub scope_check_nanos: u128,
    /// Scoped passes whose scope held every line the engine's record names for the pass,
    /// of the passes whose record names any: a stream without memory heads has none.
    pub scope_held: u64,
    pub sliced: u64,
    /// Passes of at most the runtime's fence of tokens under a named stream, which it runs on
    /// the whole model.
    pub fenced: u64,
    pub slice_checks: u64,
    pub slice_rows_differ: u64,
    pub reference_equal: u64,
    /// Passes that decoded with at least one of the stream's own lines lent to the cache,
    /// and the lines lent.
    pub swap_passes: u64,
    pub swap_lines: u64,
    /// Time in those passes' loans: the saved bytes, lending and putting back.
    pub swap_nanos: u128,
}

crate::checkpoint::numbers!(Counts {
    passes, tokens, nanos, decode_nanos, writes, skipped_writes, reads, mirror_checks, mirror_differs, group_checks, group_rows_differ, mask_checks, mask_rows_differ, scope_checks, scope_rows_differ, scope_check_nanos, scope_held, sliced, fenced, slice_checks,
    slice_rows_differ, reference_equal, swap_passes, swap_lines, swap_nanos
});

/// The regions some stream follows.
const FOLLOWED: [(u32, usize); 2] = [(VRAM, VRAM_LEN), (OAM, OAM_LEN)];

/// The regions whose write records a stream follows from the GBC CPU's cells instead of
/// decoding its own (the length's image leaves VRAM out).
fn follows(part: Part) -> &'static [(u32, usize)] {
    match part {
        Part::Ppu | Part::Stepper => &FOLLOWED,
        Part::Length => &FOLLOWED[1..],
        _ => &[],
    }
}

/// Whether byte address `addr` is in one of `regions`.
fn within(regions: &[(u32, usize)], addr: u32) -> bool {
    regions.iter().any(|&(base, len)| (base..base + len as u32).contains(&addr))
}

/// A stream's events in the engine's record (`STOCK_REFERENCE`), read as the run reaches
/// them.
struct Reference {
    path: std::path::PathBuf,
    lines: std::io::Lines<std::io::BufReader<std::fs::File>>,
    /// The lines read so far, the header included, and their digest.
    read: u64,
    fnv: u64,
}

impl Reference {
    /// The record in `dir` whose header names `slots`.
    fn open(dir: &str, slots: &[String]) -> Reference {
        use std::io::BufRead;
        for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{dir}: {e}")).flatten() {
            let mut lines = std::io::BufReader::new(std::fs::File::open(e.path()).unwrap()).lines();
            let Some(Ok(header)) = lines.next() else { continue };
            let header: serde_json::Value = serde_json::from_str(&header).unwrap();
            if header["slots"].as_array().is_some_and(|a| a.iter().map(|s| s.as_str()).eq(slots.iter().map(|s| Some(s.as_str())))) {
                return Reference { path: e.path(), lines, read: 1, fnv: fnv(FNV0, serde_json::to_string(&header).unwrap().as_bytes()) };
            }
        }
        panic!("{dir}: no record names these {} slots", slots.len())
    }

    /// The next decoded event's sequence, positions and tokens, and the line each of the
    /// stream's memory heads named in the engine's pass, past the write records (`keep`),
    /// which nothing decodes.
    fn next(&mut self) -> Option<(u64, [Vec<usize>; 3], Vec<u32>)> {
        for line in self.lines.by_ref() {
            let line = line.unwrap();
            self.read += 1;
            self.fnv = fnv(self.fnv, line.as_bytes());
            let v: serde_json::Value = serde_json::from_str(&line).unwrap();
            if v["kind"] == "keep" {
                continue;
            }
            let get = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let heads = v["heads"].as_array().map_or(Vec::new(), |h| h.iter().filter_map(|x| x[0].as_u64()).map(|n| n as u32).collect());
            return Some((v["event"].as_u64().unwrap(), [get("tokens"), get("positions"), get("out")], heads));
        }
        None
    }
}

/// What every stream shares: the cache, the cells the engines inserted before the
/// first decode, and each stream's counts.
struct Shared {
    cache: Cache,
    pending: Vec<(String, Image)>,
    counts: HashMap<String, Counts>,
    /// The merged circuit, until the prefill derives the scoped sets from it.
    circuit: Option<gbcformer::circuit::Circuit>,
    /// Per part (`Part::ALL` order), the sequence its decoded rows attend, or None for
    /// every cell.
    scopes: [Option<Scoped>; Part::ALL.len()],
    scope: ScopeMode,
    /// Whether a stream's scope holds only its own heads' lines (`scope::own`): the
    /// checkpoint is an overlay and `STOCK_OWN` is not 0.
    own: bool,
    /// Under `STOCK_SCOPE=check`, whether the decode at hand is held to every cell. A GBC
    /// CPU pass over the narrow scope is held once its scope has settled: a decode before
    /// that names the lines the scope still lacks, and its rows are not the pass's.
    settled: bool,
    /// The pass `STOCK_SCOPE=check` starts at (`STOCK_CHECK_FROM`, default 0), and the
    /// passes between two it checks (`STOCK_CHECK_EVERY`, default 1: every pass).
    check_from: u64,
    check_every: u64,
    /// `STOCK_FAULT_SEQUENCE=<k>`: at the first scoped pass from pass k on, a cell leaves
    /// the pass's sequence in the runtime behind the cache's account. A deliberate fault,
    /// to show the audit of the sequences stops a run.
    fault_sequence: Option<u64>,
    /// The GBC CPU's own circuit, until the prefill samples its reads (`scope::sampled_reads`).
    cpu_own: Option<gbcformer::circuit::Circuit>,
    streams: StreamMode,
    /// The (part, tokens) forms of sliced pass already held to the whole model
    /// (`Stream::decode`'s probe).
    slice_forms: HashSet<(usize, usize)>,
    /// Every how many of a stream's sliced passes one more is held to the whole model
    /// (`STOCK_SLICE_EVERY`; 0 holds none, the first of a form included).
    slice_every: u64,
    /// The most tokens of a group the runtime runs on the whole model whatever stream is
    /// named (`Runtime::fence`: its fence against Metal's few-row kernels).
    fence: Option<usize>,
    cpu: CpuScope,
    /// Under `STOCK_REFERENCE`: a failed mirror or loan check stops the run.
    strict: bool,
    /// Whether a following stream's pass lends its own lines (`STOCK_SWAP=0`: not).
    swap: bool,
    /// `STOCK_LENDS=<file>`: one line per pass that lends.
    lends: Option<std::fs::File>,
    /// Per following stream, the lines its own heads can name outside the regions it
    /// follows and the palettes, and the write records of other streams that changed one.
    /// The stream's engine holds its own copy of such a line, taken at the insert, so after
    /// such a record its pass attends bytes its engine's image does not hold. The mirror
    /// check does not cover these lines; the counts print them.
    unfollowed: Vec<(Part, Vec<u32>, u64)>,
    /// `STOCK_CHECKPOINT=<dir>`: the run's checkpoints.
    ckpt: Option<Ckpt>,
    /// Per stream, what its last event left outside `Shared`.
    ledger: BTreeMap<String, Ledger>,
}

/// A stream's place in the engine's record (the lines read and their digest) and a digest
/// of its own lines (`Stream::own`), which a checkpoint holds and a resume must reach.
#[derive(Default, Clone)]
struct Ledger {
    reference: (u64, u64),
    own: u64,
}

/// The run's checkpoints (`checkpoint`, `docs/stock-llama.md` §14).
struct Ckpt {
    store: Store,
    identity: Identity,
    /// Seconds between checkpoints (`STOCK_CHECKPOINT_SECS`, default 600).
    every: f64,
    last: Instant,
    /// The event the last checkpoint was taken after.
    last_event: Option<u64>,
    /// What asks the run to stop at its next checkpoint.
    stop: crate::checkpoint::Stop,
    /// The prefill's checkpoints, until the cache takes them.
    prefill: Option<crate::checkpoint::Prefill>,
    /// A prefill ended in this process: the next quiet event takes a checkpoint.
    due_now: bool,
    /// This process's number among the run's chunks, from 1, and the seconds of the chunks
    /// before it up to the checkpoint it resumed.
    chunk: u64,
    prior_s: f64,
    /// The checkpoint to resume: its event, manifest and state. While it is set no model
    /// runs: every pass decodes the tokens of the engine's record.
    resume: Option<(u64, serde_json::Value, std::path::PathBuf)>,
    /// `GBCFORMER_RECORD`: per record file, the bytes digested so far and their digest.
    record_dir: Option<std::path::PathBuf>,
    records: BTreeMap<String, (u64, u64)>,
    /// Checkpoints saved over all chunks, and what they cost.
    saved: u64,
    cost_s: f64,
}

static EVENTS: AtomicU64 = AtomicU64::new(0);
static BUDGET_HIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

extern "C" fn on_stop(_: i32) {
    crate::checkpoint::STOP.store(true, Ordering::Relaxed);
}

fn hex(h: u64) -> String {
    format!("{h:016x}")
}

/// What the harness holds after an event that does not depend on the runtime or the
/// clock: a checkpoint stores it, and a resumed run must hold exactly it when its replay
/// reaches the checkpoint's event.
fn held(s: &mut Shared) -> serde_json::Value {
    use serde_json::json;
    let k = s.ckpt.as_mut().expect("a run with checkpoints");
    if let Some(dir) = &k.record_dir {
        let mut names: Vec<String> = std::fs::read_dir(dir).into_iter().flatten().flatten().filter_map(|e| e.file_name().into_string().ok()).collect();
        names.sort();
        for name in names {
            let e = k.records.entry(name.clone()).or_insert((0, FNV0));
            let (n, h) = crate::checkpoint::fnv_file(&dir.join(&name), e.0, u64::MAX, e.1).unwrap_or_else(|err| panic!("{name}: {err}"));
            *e = (e.0 + n, h);
        }
    }
    let records: serde_json::Map<String, serde_json::Value> = k.records.iter().map(|(n, &(bytes, h))| (n.clone(), json!({ "bytes": bytes, "fnv": hex(h) }))).collect();
    let (cells, mirror) = s.cache.mirror_digest();
    let mut names: Vec<&String> = s.counts.keys().chain(s.ledger.keys()).collect();
    names.sort();
    names.dedup();
    let streams: serde_json::Map<String, serde_json::Value> = names
        .into_iter()
        .map(|n| {
            let c = s.counts.get(n).cloned().unwrap_or_default();
            let l = s.ledger.get(n).cloned().unwrap_or_default();
            let v = json!({ "passes": c.passes, "tokens": c.tokens, "writes": c.writes, "skipped_writes": c.skipped_writes, "reads": c.reads, "mirror_checks": c.mirror_checks, "swap_passes": c.swap_passes, "swap_lines": c.swap_lines, "reference_lines": l.reference.0, "reference_fnv": hex(l.reference.1), "own_lines": hex(l.own) });
            (n.clone(), v)
        })
        .collect();
    json!({
        "passes": SPENT.load(Ordering::Relaxed),
        "mirror": { "cells": cells, "fnv": hex(mirror) },
        "streams": streams,
        "cache": { "passes": s.cache.stats.passes, "writes": s.cache.stats.writes, "reads": s.cache.stats.reads, "swaps": s.cache.stats.swaps },
        "cpu_scope": { "unscoped": s.cpu.unscoped, "moves": s.cpu.moves, "map": s.cpu.map, "held": s.cpu.held, "narrow_passes": s.cpu.narrow_passes, "reruns": s.cpu.reruns, "joined": s.cpu.joined, "probes": s.cpu.probes, "recent": s.cpu.recent.lines().collect::<Vec<u32>>() },
        "unfollowed": s.unfollowed.iter().map(|(_, _, changed)| *changed).collect::<Vec<u64>>(),
        "records": records,
    })
}

/// The paths at which two manifests' values differ, with both values.
fn differences(a: &serde_json::Value, b: &serde_json::Value, path: &str, out: &mut Vec<String>) {
    match (a, b) {
        (serde_json::Value::Object(x), serde_json::Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                differences(x.get(k).unwrap_or(&serde_json::Value::Null), y.get(k).unwrap_or(&serde_json::Value::Null), &format!("{path}.{k}"), out);
            }
        }
        _ if a != b => out.push(format!("{path}: the checkpoint holds {a}, the replay reached {b}")),
        _ => {}
    }
}

/// Saves the checkpoint after `event`. The cache is quiet (`Cache::quiet`).
fn save_checkpoint(s: &mut Shared, event: u64) -> bool {
    use serde_json::json;
    assert!(s.cache.quiet(), "a checkpoint is taken of a quiet cache");
    let t = Instant::now();
    let held = held(s);
    let counts: serde_json::Map<String, serde_json::Value> = s.counts.iter().map(|(n, c)| (n.clone(), c.to_json())).collect();
    let mut cache = s.cache.stats.to_json();
    cache["prefill_s"] = json!(s.cache.stats.prefill_s);
    let k = s.ckpt.as_mut().expect("a run with checkpoints");
    let elapsed = k.prior_s + crate::checkpoint::process_s();
    // The frames done are a label: the checkpoint at the run's end is taken after its last
    // frame is counted, and a replay reaches the same event before that.
    let manifest = json!({ "held": held, "frames": FRAME.load(Ordering::Relaxed), "counts": counts, "cache": cache, "chunk": k.chunk, "elapsed_s": elapsed, "unix": unix(), "checkpoints": k.saved + 1, "checkpoints_s": k.cost_s });
    let rt = &mut s.cache.rt;
    match k.store.commit(event, manifest, &k.identity, |p| rt.save(p)) {
        Ok(cost) => {
            let took = t.elapsed().as_secs_f64();
            k.saved += 1;
            k.cost_s += took;
            k.last = Instant::now();
            k.last_event = Some(event);
            k.due_now = false;
            println!(
                "checkpoint: event {event}, frame {}, {} passes, chunk {}; {} bytes saved in {took:.1}s (the state {:.1}s, its digest {:.1}s); {} checkpoints took {:.1}s of {:.0}s over all chunks ({:.2}%); at unix {}",
                FRAME.load(Ordering::Relaxed),
                SPENT.load(Ordering::Relaxed),
                k.chunk,
                cost.bytes,
                cost.save_s,
                cost.digest_s,
                k.saved,
                k.cost_s,
                elapsed + took,
                100.0 * k.cost_s / (elapsed + took).max(1e-9),
                unix()
            );
            true
        }
        Err(e) => {
            println!("CHECKPOINT NOT SAVED after event {event}: {e}; the one before stands");
            false
        }
    }
}

/// The replay has reached the checkpoint's event: what the harness holds must be what the
/// checkpoint holds, then the runtime loads the saved cache and the counts carry on.
fn go_live(s: &mut Shared, event: u64) {
    let (at, m, state) = s.ckpt.as_mut().and_then(|k| k.resume.take()).expect("a checkpoint to resume");
    assert_eq!(at, event);
    let now = held(s);
    let mut d = Vec::new();
    differences(&m["held"], &now, "held", &mut d);
    if !d.is_empty() {
        println!("RESUME DIFFERS at event {event}: the host run again from frame 0 over the engine's record does not hold what the checkpoint holds ({} values):", d.len());
        d.iter().take(8).for_each(|l| println!("  {l}"));
        crate::quit(1);
    }
    if !s.cache.quiet() {
        println!("RESUME DIFFERS at event {event}: the replay's cache is not quiet where the checkpoint's was");
        crate::quit(1);
    }
    let t = Instant::now();
    if !s.cache.go_live(state.to_str().expect("a UTF-8 path")) {
        println!("RESUME FAILED: the runtime did not load {}", state.display());
        crate::quit(1);
    }
    for (name, c) in m["counts"].as_object().expect("a checkpoint's counts") {
        s.counts.insert(name.clone(), Counts::from_json(c));
    }
    let prefill_s = m["cache"]["prefill_s"].as_f64().unwrap_or(0.0);
    s.cache.stats = crate::cache::CacheStats { prefill_s, ..crate::cache::CacheStats::from_json(&m["cache"]) };
    s.fence = s.cache.rt.fence();
    let k = s.ckpt.as_mut().expect("a run with checkpoints");
    k.chunk = m["chunk"].as_u64().unwrap_or(0) + 1;
    k.prior_s = m["elapsed_s"].as_f64().unwrap_or(0.0);
    k.saved = m["checkpoints"].as_u64().unwrap_or(0);
    k.cost_s = m["checkpoints_s"].as_f64().unwrap_or(0.0);
    k.last = Instant::now();
    k.last_event = Some(event);
    println!(
        "resumed: chunk {} goes live after event {event} (frame {}, {} passes), {:.0}s into this process and {:.0}s into the run; the host ran again from frame 0 with every pass answered from the engine's record and no model, and holds what the checkpoint holds ({} cells, {} record files); the cache loaded in {:.1}s; at unix {}",
        k.chunk,
        FRAME.load(Ordering::Relaxed),
        SPENT.load(Ordering::Relaxed),
        crate::checkpoint::process_s(),
        k.prior_s,
        now["mirror"]["cells"],
        k.records.len(),
        t.elapsed().as_secs_f64(),
        unix()
    );
}

/// Called at the end of every event (a pass, a write record or a reader record). Goes
/// live when a resumed run's replay reaches its checkpoint, and takes a checkpoint when one
/// is due and the cache is quiet: after `Ckpt::every` seconds, or at once when the run is
/// asked to stop (a signal, the stop file, the chunk's seconds or the pass budget), which
/// then ends it.
fn after_event(s: &mut Shared) {
    let event = EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
    // A pass can come before the cartridge is inserted. Until the prefill there is no
    // cache to save, and the prefill keeps its own checkpoints.
    if s.cache.stats.cells == 0 {
        return;
    }
    let Some(k) = s.ckpt.as_mut() else { return };
    if let Some((at, ..)) = &k.resume {
        if event < *at {
            return;
        }
        go_live(s, event);
    }
    let k = s.ckpt.as_mut().expect("a run with checkpoints");
    let why = k.stop.asked().or_else(|| BUDGET_HIT.load(Ordering::Relaxed).then(|| format!("the pass budget: {} passes", SPENT.load(Ordering::Relaxed))));
    if why.is_none() && !k.due_now && k.last.elapsed().as_secs_f64() < k.every {
        return;
    }
    // A cell waits for its shift or its comparison: the next quiet event takes it.
    if !s.cache.quiet() {
        return;
    }
    let saved = k.last_event == Some(event) || save_checkpoint(s, event);
    if let Some(why) = why {
        print_counts(s);
        println!("stopped at a checkpoint: event {event}, {why}{}", if saved { "" } else { "; THE CHECKPOINT WAS NOT SAVED" });
        crate::quit(if saved { 0 } else { 1 });
    }
}

/// The GBC CPU stream's scopes (`docs/stock-llama.md` §12a).
///
/// The windows' scope holds the banked windows the model names in its `MAP` slots, the
/// regions every other bus address lands in, and the lines of the heads that translate no
/// GBC CPU address. A write to a banking register's line can move a window, so it leaves
/// the next GBC CPU pass unscoped, and that pass names the windows again.
///
/// The narrow scope serves a pass that decodes the physical address of each of its reads
/// and writes (`scope::NAMED_SLOTS`). It holds `base` and the lines named lately
/// (`scope::Recent`). Before the pass, the lines its first heads read join it: the three
/// bytes at PC and the byte at SP + 1 (`scope::first_addresses`, `scope::bus_lines` under
/// the windows the model last named). After it, every line the decoded addresses name
/// must be in it. A line that is not joins, and the pass decodes again, since its read
/// or write of that line found no cell. Every other pass (an OAM DMA
/// placement, an OAM rewrite) decodes over the windows' scope.
#[derive(Default)]
struct CpuScope {
    /// Whether standard passes decode over the narrow scope (`STOCK_CPU_SCOPE=windows`: not).
    narrow_on: bool,
    /// The lines the narrow scope always holds: `others`, `scope::small_unbanked` and the
    /// top cell's, which every scope holds.
    base: Vec<u32>,
    narrow: Option<Scoped>,
    recent: crate::scope::Recent,
    /// Passes decoded over the narrow scope, the decodes repeated because a pass named a
    /// line outside it, and the lines that joined it.
    narrow_passes: u64,
    reruns: u64,
    joined: u64,
    /// Passes that first decoded for their `MAP` slots alone, the windows unknown.
    probes: u64,
    /// `STOCK_FAULT_SCOPE=<k>`: narrow pass k (from 0), or the first after it that names a
    /// line beyond `base`, decodes once more with that line dropped and the rule that
    /// would put it back switched off, held to every cell as under `STOCK_SCOPE=check`. A
    /// deliberate fault, to show the checks stop a run.
    fault: Option<u64>,
    /// Three more deliberate faults, each to show a stop: `STOCK_FAULT_MAP=<k>` moves the
    /// windows narrow pass k's lines are placed under, `STOCK_FAULT_PROBE=<k>` the windows
    /// probe k decodes, and `STOCK_FAULT_NAMED=<k>` an address the first pass after narrow
    /// pass k that decodes again names in its second decode.
    fault_map: Option<u64>,
    fault_probe: Option<u64>,
    fault_named: Option<u64>,
    /// The lines of the heads that translate no GBC CPU address (`scope::cpu_others`).
    others: Vec<u32>,
    fixed: Option<crate::scope::Fixed>,
    seq: Option<Scoped>,
    /// The windows' first bytes the last GBC CPU pass decoded, None after a banking write.
    map: Option<[u32; 4]>,
    /// The windows `seq` holds.
    held: Option<[u32; 4]>,
    unscoped: u64,
    moves: u64,
}

impl Shared {
    /// The GBC CPU's scope for its next pass, moved to the windows the model last named,
    /// or None while no pass has named them since the last banking write.
    fn cpu_scope(&mut self) -> Option<Scoped> {
        let Some(map) = self.cpu.map else {
            self.cpu.unscoped += 1;
            return None;
        };
        if self.cpu.held != Some(map) {
            let lines = crate::scope::cpu_lines(&self.cpu.others, map, P);
            match self.cpu.seq {
                Some(s) => self.cache.rescope(s, &lines),
                None => {
                    let held = lines.iter().filter(|&&n| self.cache.mirror[n as usize].is_some()).count();
                    eprintln!("scope Cpu: its rows attend {held} cells ({} lines) under the windows {:x?}", lines.len(), map);
                    self.cpu.seq = Some(self.cache.scope(&lines, self.cpu.fixed.clone().expect("the fixed bytes")).expect("the runtime keeps scoped sequences"));
                }
            }
            self.cpu.held = Some(map);
            self.cpu.moves += 1;
        }
        self.cpu.seq
    }

    /// Whether the decode at hand is also decoded over every cell: under
    /// `STOCK_SCOPE=check`, once its scope has settled, from pass `STOCK_CHECK_FROM` on, at
    /// every `STOCK_CHECK_EVERY`th pass of the run.
    fn checks_scope(&self) -> bool {
        let pass = SPENT.load(Ordering::Relaxed);
        self.scope == ScopeMode::Check && self.settled && pass >= self.check_from && pass % self.check_every == 0
    }

    /// Whether line `n` is in the narrow scope, joining it when it holds a cell and is not.
    fn narrow_hold(&mut self, n: u32) -> bool {
        let seq = self.cpu.narrow.expect("the narrow scope");
        if self.cpu.base.binary_search(&n).is_ok() {
            return true;
        }
        if !self.cache.scope_lines(seq).contains(&n) {
            if !self.cache.scope_add(seq, n) {
                return false;
            }
            self.cpu.joined += 1;
        }
        if let Some(out) = self.cpu.recent.touch(n) {
            self.cache.scope_drop(seq, out);
        }
        true
    }

    /// The narrow scope for a GBC CPU pass whose PC is `pc` and whose SP is `sp`, holding
    /// the lines its first heads read (`scope::first_addresses`): the three fetched bytes'
    /// and read 1's. None while no pass has named the windows since the last banking
    /// write, when one of those addresses is in cartridge RAM (`scope::bus_lines` gives no
    /// line there), or when one of those lines holds no cell.
    fn cpu_narrow(&mut self, pc: u16, sp: u16) -> Option<Scoped> {
        let map = self.cpu.map?;
        let first: Vec<u32> = crate::scope::first_addresses(pc, sp).into_iter().map(|a| crate::scope::bus_lines(a, map)).collect::<Option<Vec<_>>>()?.concat();
        let seq = self.cpu_narrow_scope();
        first.into_iter().all(|n| self.narrow_hold(n)).then_some(seq)
    }

    /// The narrow scope's sequence, made of its fixed lines at its first use.
    fn cpu_narrow_scope(&mut self) -> Scoped {
        if self.cpu.narrow.is_none() {
            let held = self.cpu.base.iter().filter(|&&n| self.cache.mirror[n as usize].is_some()).count();
            eprintln!("scope Cpu: a pass that names its reads and writes attends {held} cells ({} lines) and the lines named lately", self.cpu.base.len());
            let base = self.cpu.base.clone();
            self.cpu.narrow = Some(self.cache.scope(&base, self.cpu.fixed.clone().expect("the fixed bytes")).expect("the runtime keeps scoped sequences"));
        }
        self.cpu.narrow.expect("the narrow scope")
    }
}

/// Whether a pass runs its stream's slice of the model (`Runtime::stream`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Whole,
    Sliced,
    /// `Sliced`, and no stream may fall back to the whole model.
    Require,
    /// Every sliced pass decoded by the whole model too, its rows required bit-identical.
    Check,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScopeMode {
    Off,
    On,
    /// Every pass of a scoped stream decoded over every cell too, its rows required
    /// bit-identical.
    Check,
}

impl Shared {
    /// Prefills the cells, which are the GBC CPU's image's lines and any line only the
    /// PPU's image holds (the palettes). Where both hold one, the PPU's VRAM and OAM must
    /// equal the GBC CPU's.
    fn prefill(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let cpu = &pending.iter().find(|(n, _)| n == "Cpu").expect("the CPU stream inserts the cartridge").1;
        let mut cells = crate::cache::cells_of(cpu);
        for (name, image) in &pending {
            if name == "Cpu" {
                continue;
            }
            let part = Part::ALL.into_iter().find(|p| format!("{p:?}") == *name).expect("a part's stream");
            for &(base, len) in follows(part) {
                for a in base..base + len as u32 {
                    assert_eq!(image.read(a), cpu.read(a), "{name}'s image and the CPU's differ at {a:#08x}");
                }
            }
            cells.extend(crate::cache::cells_of(image).into_iter().filter(|&(n, _)| (PAL >> LINE_BITS..(PAL + PAL_LEN as u32).div_ceil(LINE as u32)).contains(&n) && !cpu.has_line(n)));
        }
        cells.sort_unstable();
        self.cache.prefill(&cells);
        // The prefill ran in this process, in whole or from another chunk's checkpoint: the
        // run's chunks and seconds are the prefill's, and the first quiet event is saved.
        if let (Some(k), Some(p)) = (self.ckpt.as_mut(), self.cache.prefill_ckpt.take()) {
            (k.chunk, k.prior_s, k.saved, k.cost_s) = (p.chunk, p.prior_s, p.saved, p.cost_s);
            k.due_now = true;
        }
        let circuit = self.circuit.take().expect("the merged circuit");
        if self.scope == ScopeMode::Off {
            return;
        }
        if self.cache.writes != crate::cache::Writes::Whole {
            let held = self.cache.scope_records();
            eprintln!("scope of the write records: {held} cells");
        }
        // The streams other than the GBC CPU's attend only the lines their heads can name
        // (`scope`); streams with the same lines share one sequence.
        let fixed = crate::scope::Fixed::of(cpu);
        let mut made: Vec<(Vec<u32>, Scoped)> = Vec::new();
        for part in [Part::Ppu, Part::Apu, Part::Control, Part::Length, Part::Stepper] {
            let cubes = crate::scope::cubes(&circuit, part, cpu, &fixed);
            let lines = if self.own { own_lines(&crate::scope::own(&circuit, part, &cubes)) } else { crate::scope::lines_of(&cubes, P) };
            let held = lines.iter().filter(|&&n| self.cache.mirror[n as usize].is_some()).count();
            assert!(held <= crate::scope::CELLS, "scope {part:?}: its rows would attend {held} cells, more than {}; heads left open: {:?}", crate::scope::CELLS, crate::scope::wide(&circuit, &cubes));
            let s = match made.iter().find(|(l, _)| *l == lines) {
                Some(&(_, s)) => s,
                None => {
                    let s = self.cache.scope(&lines, fixed.clone()).expect("the runtime keeps scoped sequences");
                    made.push((lines.clone(), s));
                    s
                }
            };
            eprintln!("scope {part:?}: its rows attend {held} cells ({} lines named), {s:?}", lines.len());
            self.scopes[part as usize] = Some(s);
            if !follows(part).is_empty() {
                let prefix = format!("{}:", part.name());
                let own: Vec<(u32, u32)> = cubes.iter().zip(&circuit.heads).filter(|(_, h)| h.name.starts_with(&prefix)).map(|(&q, _)| q).collect();
                let followed = |n: u32| within(follows(part), n << LINE_BITS) || within(&[(PAL, PAL_LEN)], n << LINE_BITS);
                let rest: Vec<u32> = crate::scope::lines_of(&own, P).into_iter().filter(|&n| self.cache.mirror[n as usize].is_some() && !followed(n)).collect();
                eprintln!("scope {part:?}: its own heads can name {} cells outside the regions it follows and the palettes, lines {:x?}", rest.len(), crate::scope::runs(&rest));
                self.unfollowed.push((part, rest, 0));
            }
        }
        self.cpu.others = if self.own { own_lines(&crate::scope::cpu_fixed(&circuit, cpu, &fixed)) } else { crate::scope::lines_of(&crate::scope::cpu_others(&circuit, cpu, &fixed), P) };
        assert!(self.cpu.others.len() <= crate::scope::CELLS, "scope Cpu: the heads that translate no GBC CPU address name {} lines, more than {}", self.cpu.others.len(), crate::scope::CELLS);
        self.cpu.base = self.cpu.others.iter().copied().chain(crate::scope::small_unbanked().flat_map(|(base, len)| crate::cache::lines(base, len as usize))).chain([P as u32 - 1]).collect();
        self.cpu.base.sort_unstable();
        self.cpu.base.dedup();
        if self.cpu.narrow_on {
            let wiring = crate::scope::named_slots_wiring(&circuit, cpu, &fixed);
            eprintln!("scope Cpu: the MAP slots depend on no head but the banking registers' ({:?}), and the read and write address slots besides on {} heads whose own addresses the inputs and those registers give: {:?}", wiring.fixed, wiring.first.len(), wiring.first);
            let own = self.cpu_own.take().expect("the GBC CPU's own circuit");
            let t = Instant::now();
            let sampled = crate::scope::sampled_reads(&own, &wiring, cpu, &self.cpu.base, &|pc, sp| crate::scope::first_addresses(pc, sp).to_vec(), crate::scope::SAMPLES);
            eprintln!("scope Cpu: {sampled} ({:.1} s)", t.elapsed().as_secs_f64());
        }
        self.cpu.fixed = Some(fixed);
        if let Some(n) = std::env::var("STOCK_READ_PROBE").ok().map(|v| v.parse::<usize>().expect("STOCK_READ_PROBE=<reader records>")).filter(|_| !self.cache.rt.replays()) {
            let (read, ms) = self.cache.probe_reads(n);
            eprintln!("read probe: {read} reader records of evenly spaced lines, {ms:.1} ms each; {} decoded another line than the mirror's, {} rows differ between the scope and every cell (STOCK_READS={})", self.cache.stats.read_differs, self.cache.stats.read_rows_differ, std::env::var("STOCK_READS").unwrap_or_default());
            if self.cache.stats.read_differs + self.cache.stats.read_rows_differ > 0 {
                println!("READ NOT HELD: a probe's reader record decoded another line than the mirror gives, or other rows over its scope than over every cell");
                crate::quit(1);
            }
        }
    }
}

/// The lines a stream's own heads can name (`cubes`, theirs) and the copy heads' line.
fn own_lines(cubes: &[(u32, u32)]) -> Vec<u32> {
    let mut lines = crate::scope::lines_of(cubes, P);
    lines.extend(crate::scope::OWN_BESIDES);
    lines.sort_unstable();
    lines.dedup();
    lines
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    Full,
    Own,
    Check,
}

/// Whether a reduced group leaves out the own masks a pass does not decode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Masks {
    Keep,
    Drop,
    /// Both groups decoded, their rows required bit-identical.
    Check,
}

struct Stream {
    part: Part,
    name: String,
    shared: Arc<Mutex<Shared>>,
    /// The checkpoint positions a reduced group keeps, in order.
    keep: Vec<usize>,
    /// The mask token of output position `p` is `mask + p`.
    mask: usize,
    /// Every output position's mask token.
    masks_at: std::ops::Range<usize>,
    /// Per checkpoint position, whether a pass may leave it out while it holds a
    /// decoded byte rather than its mask. Such a position is an own output slot no gather
    /// targets (a byte token's layer-0 key scores far below every gather's targets,
    /// `one_model::group`).
    settled: Vec<bool>,
    /// Per layer-0 gather, the positions it attends in the reduced group (`one_model::Gather`).
    attended: Vec<Vec<usize>>,
    group: Group,
    masks: Masks,
    /// The GBC CPU's `MAP` slots' checkpoint positions (`scope::MAP_SLOTS`).
    map_at: Option<[[usize; 3]; 4]>,
    /// The GBC CPU's read and write address slots' checkpoint positions
    /// (`scope::NAMED_SLOTS`), and its PC's and SP's input slots', low byte first.
    named_at: Option<[[usize; 3]; 4]>,
    pc_sp_at: Option<[[usize; 2]; 2]>,
    reference: Option<Reference>,
    /// The stream's own lines of the regions it follows (`follows`): its image's at the
    /// insert, then each write record its engine kept there.
    own: BTreeMap<u32, [u8; LINE]>,
    /// The checkpoint position of the stream's first input slot (LY in a following stream).
    first_at: Option<usize>,
}

/// One own line's part of `Ledger::own`, the XOR over the stream's own lines.
fn own_digest(n: u32, line: &[u8; LINE]) -> u64 {
    fnv(fnv(FNV0, &n.to_le_bytes()), line)
}

static SPENT: AtomicU64 = AtomicU64::new(0);
static BUDGET: AtomicU64 = AtomicU64::new(u64::MAX);
static FRAME: AtomicU64 = AtomicU64::new(0);
static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

impl Stream {
    /// The rows at `outs`, attending the stream's scope. Under `STOCK_SCOPE=check` it
    /// decodes the same group again over every cell, on the stream's slice when the scoped
    /// pass ran it, and holds its rows to the scoped ones bit for bit.
    ///
    /// Every mask token of the group attends the cells, read or not. A mask carries its
    /// part's mode, so the part's units run there, and a memory head with no cell to find
    /// reads whatever its scores pick: a bus of two columns gives a bit wire 2, a gate
    /// leaves the f16 range, and `ffn_down`'s f16 input turns it into inf, then NaN in
    /// every later key that row holds (`docs/stock-llama.md` §12a).
    fn decode(&self, s: &mut Shared, c: &mut Counts, tokens: &[usize], outs: &[usize]) -> Vec<Row> {
        let scope = s.scopes[self.part as usize];
        // A stream's run of the head rests on its passes holding no record token (`gguf::stream_head`).
        assert!(tokens.iter().all(|&t| t < tok::rec(&s.cache.cfg)), "{}: a pass holds a record token", self.name);
        let also: Vec<usize> = (0..tokens.len()).filter(|i| self.masks_at.contains(&tokens[*i]) && !outs.contains(i)).collect();
        let named = s.streams != StreamMode::Whole && s.cache.rt.stream(Some(self.part.name()));
        (c.decodes, c.batch, c.asked, c.attending) = (c.decodes + 1, c.batch + tokens.len() as u64, c.asked + outs.len() as u64, c.attending + (outs.len() + also.len()) as u64);
        let rows = s.cache.pass(tokens, outs, &also, scope);
        let mut sliced = false;
        if named {
            let (ran, fence) = (s.cache.rt.sliced(), s.fence);
            s.cache.rt.stream(None);
            let (Some(ran), Some(fence)) = (ran, fence) else {
                println!("SLICE UNKNOWN in {} pass {}: the runtime names the stream but does not report whether the pass ran its slice, or its fence (llama.cpp built without this tree's stored-runs.patch and stream-slices.patch)", self.name, c.passes);
                crate::quit(1);
            };
            if ran != (tokens.len() > fence) {
                println!("SLICE NOT RUN in {} pass {}: a group of {} tokens under the named stream ran {}, and the runtime's fence is {fence} tokens", self.name, c.passes, tokens.len(), if ran { "the stream's slice" } else { "the whole model" });
                crate::quit(1);
            }
            sliced = ran;
            c.fenced += !sliced as u64;
        }
        if sliced {
            c.sliced += 1;
            // A slice is a set of views of the weights, and a backend can read a view wrong
            // without a sign (Metal did: a finite, wrong product). The kernels a pass runs
            // follow from its stream and its count of tokens, so the first sliced pass of
            // each such form is decoded by the whole model too, then one in
            // `STOCK_SLICE_EVERY` (0: none of either); `STOCK_STREAMS=check` holds every pass.
            let probe = s.streams == StreamMode::Check || (s.slice_every > 0 && (s.slice_forms.insert((self.part as usize, tokens.len())) || c.sliced % s.slice_every == 0));
            if probe {
                let whole = s.cache.pass(tokens, outs, &also, scope);
                if s.cache.rt.sliced() != Some(false) {
                    println!("SLICE PROBE NOT WHOLE in {} pass {}: the decode meant for the whole model did not report that it ran the whole model, so the slice would be held to itself", self.name, c.passes);
                    crate::quit(1);
                }
                c.slice_checks += 1;
                // The slice may score a run of the whole model's tokens (`Row::within`).
                for (k, (a, b)) in whole.iter().zip(&rows).enumerate() {
                    if !b.within(a) {
                        c.slice_rows_differ += 1;
                        eprintln!("SLICE ROW DIFFERS in {} pass {} row {k}, the stream's slice (tokens {}..{}) against the whole model ({}..{}): tokens {} vs {}", self.name, c.passes, b.first, b.first + b.v.len(), a.first, a.first + a.v.len(), b.best(), a.best());
                    }
                }
            }
        }
        if s.checks_scope() && scope.is_some() {
            // Over every cell on the same network as the scoped pass, the stream's slice
            // when that ran, so both score the same run of tokens.
            if sliced {
                s.cache.rt.stream(Some(self.part.name()));
            }
            let t = Instant::now();
            let full = s.cache.pass(tokens, outs, &also, None);
            s.cache.rt.stream(None);
            c.scope_check_nanos += t.elapsed().as_nanos();
            c.scope_checks += 1;
            for (k, (a, b)) in full.iter().zip(&rows).enumerate() {
                if !a.same(b) {
                    c.scope_rows_differ += 1;
                    eprintln!("GROUP ROW DIFFERS in {} pass {} row {k}, scoped against every cell: tokens {} vs {}", self.name, c.passes, a.best(), b.best());
                }
            }
        }
        rows
    }

    /// The windows' first bytes a GBC CPU pass decoded in its `MAP` slots, or None in
    /// another stream, in a pass that does not decode them, or when a slot decoded no byte.
    fn windows(&self, s: &Shared, rows: &[Row], positions: &[usize], pass: u64) -> Option<[u32; 4]> {
        let byte = |p: usize| {
            let i = positions.iter().position(|&q| q == p)?;
            let t = rows[i].token(&s.cache.head, s.cache.cfg.vocab);
            let v = tok::output_value(&s.cache.cfg, t);
            if v.is_none() {
                let r = &rows[i].v;
                let (nan, max, min) = (r.iter().filter(|x| x.is_nan()).count(), r.iter().cloned().fold(f32::NEG_INFINITY, f32::max), r.iter().cloned().fold(f32::INFINITY, f32::min));
                eprintln!("MAP SLOT NOT A BYTE in {} pass {pass}: position {p} decoded token {t} (the mask is {}); its row: {} values, {nan} NaN, max {max}, min {min}; the reference check names the pass", self.name, self.mask + p, r.len());
            }
            v.map(|b| b as u32)
        };
        let map: Vec<u32> = self.map_at?.iter().map(|w| Some(byte(w[0])? | byte(w[1])? << 8 | byte(w[2])? << 16)).collect::<Option<_>>()?;
        Some(map.try_into().expect("four windows"))
    }

    /// The rows at `positions` of the pass's group as `STOCK_GROUP` and `STOCK_MASKS` choose
    /// it, and the tokens decoded.
    fn decode_group(&self, s: &mut Shared, c: &mut Counts, tokens: &[usize], positions: &[usize]) -> (Vec<Row>, usize) {
        let mut decoded = tokens.len();
        let rows = match self.group {
            Group::Full => self.decode(s, c, tokens, positions),
            Group::Own => {
                let (t, o) = self.reduced(tokens, positions, self.masks != Masks::Keep);
                decoded = t.len();
                let rows = self.decode(s, c, &t, &o);
                if self.masks == Masks::Check {
                    let (t, o) = self.reduced(tokens, positions, false);
                    let kept = self.decode(s, c, &t, &o);
                    c.mask_checks += 1;
                    for (k, (a, b)) in kept.iter().zip(&rows).enumerate() {
                        if !a.same(b) {
                            c.mask_rows_differ += 1;
                            eprintln!("MASK ROW DIFFERS in {} pass {} row {k}, the group with every own mask against the one without the undecoded: tokens {} vs {}", self.name, c.passes, a.best(), b.best());
                        }
                    }
                }
                rows
            }
            Group::Check => {
                let full = self.decode(s, c, tokens, positions);
                let (t, o) = self.reduced(tokens, positions, self.masks != Masks::Keep);
                decoded = t.len();
                let own = self.decode(s, c, &t, &o);
                c.group_checks += 1;
                // The full group runs the stream's slice, and a reduced group of at most
                // the runtime's fence the whole model, so the two may score different runs.
                for (k, (a, b)) in full.iter().zip(&own).enumerate() {
                    if !a.agrees(b) {
                        c.group_rows_differ += 1;
                        eprintln!("GROUP ROW DIFFERS in {} pass {} row {k}: tokens {} vs {} (rows over tokens {}..{} and {}..{})", self.name, c.passes, a.best(), b.best(), a.first, a.first + a.v.len(), b.first, b.first + b.v.len());
                    }
                }
                own
            }
        };
        (rows, decoded)
    }

    /// The reduced group of a pass that decodes `positions`, and their indexes in it.
    ///
    /// It leaves out a settled position (an own output slot no layer-0 gather targets)
    /// that holds a decoded byte, and with `drop_masks` one that holds its mask and is
    /// not decoded. Neither changes a row that is read:
    /// - the row of a position left out is not read;
    /// - past layer 0 a group token attends only the cell its query names
    ///   (`one_model::group`), so a token left out reaches the others through layer 0 alone;
    /// - at layer 0 every head gives it exactly zero weight while one token the head
    ///   attends stays. A head whose targets the group keeps whole keeps them all, since
    ///   no target is settled. Every other head attends tokens whose values are all zero
    ///   (`one_model::group` keeps a head's targets whole unless they are), so any of
    ///   them writes the 0 they all wrote: its targets are left out of the group
    ///   altogether and it attends the fallback's masks, or the group keeps them in part,
    ///   as it does a record gather's in a stream that owns slots on the record
    ///   identities (the parts share the output identity channels). Those own slots are
    ///   targets, so they are not settled and stay in every pass's group. The assert
    ///   below holds each head to one attended token that stays, holding its mask.
    fn reduced(&self, tokens: &[usize], positions: &[usize], drop_masks: bool) -> (Vec<usize>, Vec<usize>) {
        let n_inputs = self.masks_at.start - self.mask;
        let masked = |p: usize| tokens[p] == self.mask + p;
        let live: Vec<usize> = self.keep.iter().copied().filter(|&p| !(self.settled[p] && (!masked(p) || drop_masks && !positions.contains(&p)))).collect();
        for (h, at) in self.attended.iter().enumerate() {
            assert!(at.iter().any(|&p| live.binary_search(&p).is_ok() && (p < n_inputs || masked(p))), "{}: layer-0 gather {h} attends no token of the reduced group", self.name);
        }
        let idx: HashMap<usize, usize> = live.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let toks = live.iter().map(|&p| tokens[p]).collect();
        let outs = positions.iter().map(|p| *idx.get(p).unwrap_or_else(|| panic!("{}: decoded position {p} is not in the reduced group", self.name))).collect();
        (toks, outs)
    }
}

impl External for Stream {
    fn fill(&mut self, tokens: &[usize], positions: &[usize], memory: Option<&Image>) -> Vec<usize> {
        let mut s = self.shared.lock().unwrap();
        s.prefill();
        let mut c = s.counts.get(&self.name).cloned().unwrap_or_default();
        // The cells that hold another line than the stream's own: the GBC CPU kept records
        // there after the stream's engine last applied them. Each takes the stream's line
        // for this pass's decodes; `lent` keeps what goes back after them.
        let t = Instant::now();
        let lent: Vec<crate::cache::Lent> = if s.swap { self.own.iter().filter_map(|(&n, &mine)| s.cache.lend(n, mine)).collect() } else { Vec::new() };
        if !lent.is_empty() {
            c.swap_passes += 1;
            c.swap_lines += lent.len() as u64;
            c.swap_nanos += t.elapsed().as_nanos();
            let first = self.first_at.and_then(|p| tokens[p].checked_sub(tok::input(&s.cache.cfg, p, 0))).filter(|&v| v < 256);
            if let Some(f) = s.lends.as_mut() {
                writeln!(f, "frame {} {} pass {} first input {} lines {}", FRAME.load(Ordering::Relaxed) + 1, self.name, c.passes, first.map_or("?".to_string(), |v| v.to_string()), lent.iter().map(|l| format!("{:#x}", l.line())).collect::<Vec<_>>().join(",")).unwrap();
            }
        }
        // The engine's record of this pass, read before it decodes: a replay decodes its tokens.
        let want = self.reference.as_mut().map(|r| r.next());
        if let Some(r) = &self.reference {
            s.ledger.entry(self.name.clone()).or_default().reference = (r.read, r.fnv);
        }
        if s.cache.rt.replays() {
            match &want {
                Some(Some((_, [rt, rp, ro], _))) if rt == tokens && rp == positions => s.cache.rt.answer(ro),
                // The check after the pass names the difference.
                _ => s.cache.rt.answer(&vec![0; positions.len()]),
            }
        }
        // A pass's time leaves out the prefill, which runs inside the first, and the swaps.
        let t = Instant::now();
        // The engine's index is the oracle for the cache's mirror (a check, never an input),
        // here with the stream's own lines lent.
        if let Some(image) = memory
            && (!follows(self.part).is_empty() || c.passes % 200 == 0)
        {
            // A following stream's image holds only what its heads read.
            let regions: Vec<(u32, usize)> = match follows(self.part) {
                [] => vec![(0, gbcformer::memory::layout::TOTAL as usize)],
                held => held.iter().copied().chain([(PAL, PAL_LEN)]).collect(),
            };
            let d = s.cache.differs(image, &regions);
            c.mirror_checks += 1;
            if !d.is_empty() {
                c.mirror_differs += 1;
                eprintln!("MIRROR DIFFERS in {} pass {}: {} addresses, first {:#08x}", self.name, c.passes, d.len(), d[0]);
                if s.strict {
                    s.counts.insert(self.name.clone(), c);
                    print_counts(&s);
                    println!("stopped at MIRROR DIFFERS in {}: the cells its pass would attend are not its engine's image", self.name);
                    crate::quit(1);
                }
            }
        }
        // A GBC CPU pass that decodes its reads' and writes' addresses and the windows
        // attends the narrow scope, every other the windows' scope.
        let named_at = self.named_at.zip(self.map_at).filter(|(n, m)| s.cpu.narrow_on && s.scope != ScopeMode::Off && n.iter().chain(m.iter()).flatten().all(|p| positions.contains(p))).map(|(n, _)| n);
        let (sliced0, fenced0, checks0, slice_checks0) = (c.sliced, c.fenced, c.scope_checks, c.slice_checks);
        let differ0 = c.scope_rows_differ + c.mask_rows_differ + c.group_rows_differ + c.slice_rows_differ;
        let d0 = s.cache.rt.decode_nanos();
        // No pass has named the windows since the start or a banking write, so the lines the
        // bytes at PC sit in are not known. The pass first decodes over the narrow scope's
        // fixed lines for its `MAP` slots alone, which read the banking registers' cells
        // and nothing the fetch reads. Then it decodes as any other, and must name the
        // same windows.
        let mut probed = None;
        s.settled = named_at.is_none();
        if named_at.is_some() && self.pc_sp_at.is_some() && s.cpu.map.is_none() {
            s.scopes[Part::Cpu as usize] = Some(s.cpu_narrow_scope());
            let (rows, _) = self.decode_group(&mut s, &mut c, tokens, positions);
            probed = self.windows(&s, &rows, positions, c.passes);
            if let Some(k) = s.cpu.fault_probe.filter(|&k| s.cpu.probes >= k)
                && let Some(map) = probed.as_mut()
            {
                s.cpu.fault_probe = None;
                map[1] ^= 0x4000;
                eprintln!("FAULT INJECTED (STOCK_FAULT_PROBE={k}): {} pass {} takes the windows {map:x?} from its first decode, with the second window moved", self.name, c.passes);
            }
            s.cpu.map = probed;
            s.cpu.probes += 1;
        }
        let cpu = &mut s.cpu;
        if let Some(k) = cpu.fault_map.filter(|&k| named_at.is_some() && probed.is_none() && cpu.narrow_passes >= k)
            && let Some(map) = cpu.map.as_mut()
        {
            cpu.fault_map = None;
            map[1] ^= 0x4000;
            eprintln!("FAULT INJECTED (STOCK_FAULT_MAP={k}): {} pass {} places its lines under the windows {map:x?}, with the second window moved", self.name, c.passes);
        }
        // The windows this pass's lines are placed under: those the last pass named, or
        // this pass's own first decode. No write record has changed a banking register
        // since (`write` forgets the windows at one), so the pass must name the same.
        let under = s.cpu.map;
        let narrow = named_at.zip(self.pc_sp_at).and_then(|(_, at)| {
            let [pc, sp] = at.map(|w| w.map(|p| (tokens[p] - tok::input(&s.cache.cfg, p, 0)) as u16)).map(|[low, high]| low | high << 8);
            s.cpu_narrow(pc, sp)
        });
        if narrow.is_some() {
            s.scopes[Part::Cpu as usize] = narrow;
            s.cpu.narrow_passes += 1;
        } else {
            s.settled = true;
            if self.map_at.is_some() && s.scope != ScopeMode::Off {
                s.scopes[Part::Cpu as usize] = s.cpu_scope();
            }
        }
        let (mut rows, mut decoded) = self.decode_group(&mut s, &mut c, tokens, positions);
        if let (Some(_), Some(at)) = (narrow, named_at) {
            // Every line the decoded addresses name must be in the scope the pass attended.
            // A line that is not held no cell for its read or write, so it joins and the
            // pass decodes again. The addresses are the model's own account. They depend
            // on the pass's inputs, the banking registers and what the first heads read
            // (`scope::named_slots_wiring`), all of which the scope held, so they may not
            // change when the pass decodes again.
            let lines = |s: &Shared, rows: &[Row]| -> Option<Vec<u32>> {
                let byte = |p: usize| tok::byte(&s.cache.cfg, rows[positions.iter().position(|&q| q == p)?].token(&s.cache.head, s.cache.cfg.vocab)).map(|b| b as u32);
                at.iter().map(|w| Some((byte(w[0])? | byte(w[1])? << 8 | byte(w[2])? << 16) >> LINE_BITS)).collect()
            };
            let stop = |pass: u64, why: String| -> ! {
                println!("NARROW SCOPE NOT HELD in {} pass {pass}: {why}; STOCK_CPU_SCOPE=windows decodes every pass over the windows' scope", self.name);
                crate::quit(1)
            };
            let Some(named) = lines(&s, &rows) else { stop(c.passes, "a read or write address slot decoded no byte".to_string()) };
            let outside: Vec<u32> = named.iter().copied().filter(|&n| !s.cache.scope_lines(narrow.expect("the narrow scope")).contains(&n)).collect();
            if !outside.is_empty() {
                if s.cpu.reruns < 3 {
                    eprintln!("scope Cpu: pass {} named line {:#07x} outside its scope; the line joins and the pass decodes again (the first such passes are printed, the counts keep the total)", c.passes, outside[0]);
                }
                if let Some(&n) = outside.iter().find(|&&n| !s.narrow_hold(n)) {
                    stop(c.passes, format!("the pass names line {n:#07x}, which holds no cell"));
                }
                s.cpu.reruns += 1;
                (rows, decoded) = self.decode_group(&mut s, &mut c, tokens, positions);
                let mut again = lines(&s, &rows);
                if let Some(k) = s.cpu.fault_named.filter(|&k| s.cpu.narrow_passes > k)
                    && let Some(lines) = again.as_mut()
                {
                    s.cpu.fault_named = None;
                    lines[0] ^= 1;
                    eprintln!("FAULT INJECTED (STOCK_FAULT_NAMED={k}): {} pass {} is taken to name line {:#07x} for read 0 in its second decode", self.name, c.passes, lines[0]);
                }
                if again.as_ref() != Some(&named) {
                    stop(c.passes, format!("the addresses the pass names changed when it decoded again over the lines it first named ({named:x?}, then {again:x?})"));
                }
            }
            for &n in &named {
                s.narrow_hold(n);
            }
            s.settled = true;
            if s.checks_scope() {
                (rows, decoded) = self.decode_group(&mut s, &mut c, tokens, positions);
            }
            if let Some(k) = s.cpu.fault.filter(|&k| s.cpu.narrow_passes > k)
                && let Some(&n) = named.iter().find(|&&n| s.cpu.base.binary_search(&n).is_err())
            {
                s.cpu.fault = None;
                s.cache.scope_drop(narrow.expect("the narrow scope"), n);
                eprintln!("FAULT INJECTED (STOCK_FAULT_SCOPE={k}): {} pass {} decodes again over its scope without line {n:#07x}, which it names, and nothing puts the line back; the decode is also held to every cell", self.name, c.passes);
                let (mode, from) = (std::mem::replace(&mut s.scope, ScopeMode::Check), std::mem::replace(&mut s.check_from, 0));
                (rows, decoded) = self.decode_group(&mut s, &mut c, tokens, positions);
                (s.scope, s.check_from) = (mode, from);
            }
        }
        // A pass that decodes its `MAP` slots names the windows. The GBC CPU's scopes rest
        // on them: the lines of this pass were placed under `under`, and the next pass's
        // will be placed under what this one names. So the pass must name windows, the
        // ones its first decode named, and the ones its lines were placed under. Nothing
        // it did not confirm stays for the next pass.
        if self.map_at.is_some_and(|m| m.iter().flatten().all(|p| positions.contains(p))) {
            let map = self.windows(&s, &rows, positions, c.passes);
            let why = match (map, probed, under) {
                _ if s.scope == ScopeMode::Off => None,
                (None, ..) => Some("a MAP slot decoded no byte, so the pass names no windows".to_string()),
                (Some(map), Some(first), _) if first != map => Some(format!("the windows the pass names changed when it decoded again over the lines its PC names under them ({first:x?}, then {map:x?})")),
                (Some(map), None, Some(under)) if under != map => Some(format!("the pass names the windows {map:x?}, and its lines were placed under {under:x?}, the windows the last pass named, with no write record to a banking register between them")),
                _ => None,
            };
            if let Some(why) = why {
                s.counts.insert(self.name.clone(), c);
                print_counts(&s);
                println!("{} in {} pass {}: {why}; STOCK_CPU_SCOPE=windows decodes every pass over the windows' scope, STOCK_SCOPE=0 over every cell", if narrow.is_some() { "NARROW SCOPE NOT HELD" } else { "WINDOWS SCOPE NOT HELD" }, self.name, s.counts[&self.name].passes);
                crate::quit(1);
            }
            s.cpu.map = map;
        }
        // A check that decodes a pass two ways holds the rows equal. A row that differs stops
        // the run in every mode: without the engine's record nothing else would.
        if c.scope_rows_differ + c.mask_rows_differ + c.group_rows_differ + c.slice_rows_differ > differ0 {
            s.counts.insert(self.name.clone(), c);
            print_counts(&s);
            println!("CHECK NOT HELD in {} pass {}: a row differs between two decodes of the pass that a check holds equal (the ROW DIFFERS lines above name the check and the rows)", self.name, s.counts[&self.name].passes);
            crate::quit(1);
        }
        // Under `STOCK_SCOPE=check`, at the passes it samples, the runtime's own account of
        // the pass's sequence is held to the cache's (`Cache::audit`).
        if let Some(scope) = s.scopes[self.part as usize] {
            let pass = SPENT.load(Ordering::Relaxed);
            if let Some(k) = s.fault_sequence.filter(|&k| pass >= k)
                && let Some(n) = s.cache.scope_lines(scope).iter().copied().find(|&n| n != 0 && n as usize != P - 1)
            {
                s.fault_sequence = None;
                s.cache.rt.fault_drop(scope, n);
                eprintln!("FAULT INJECTED (STOCK_FAULT_SEQUENCE={k}): after {} pass {}, line {n:#07x}'s cell leaves its sequence in the runtime, and the cache's account keeps the line", self.name, c.passes);
            }
            if s.scope == ScopeMode::Check && pass >= s.check_from && pass % s.check_every == 0
                && let Some(why) = s.cache.audit(scope)
            {
                s.counts.insert(self.name.clone(), c);
                print_counts(&s);
                println!("SCOPE SEQUENCE DIFFERS in {} pass {}: {why}", self.name, s.counts[&self.name].passes);
                crate::quit(1);
            }
        }
        // The counts are of passes: one that decoded more than once counts once.
        (c.sliced, c.fenced) = (sliced0 + (c.sliced > sliced0) as u64, fenced0 + (c.fenced > fenced0) as u64);
        (c.scope_checks, c.slice_checks) = (checks0 + (c.scope_checks > checks0) as u64, slice_checks0 + (c.slice_checks > slice_checks0) as u64);
        c.decode_nanos += s.cache.rt.decode_nanos() - d0;
        c.passes += 1;
        c.tokens += decoded as u64;
        let out: Vec<usize> = rows.iter().map(|r| r.token(&s.cache.head, s.cache.cfg.vocab)).collect();
        c.nanos += t.elapsed().as_nanos();
        if !lent.is_empty() {
            let t = Instant::now();
            for l in lent {
                s.cache.put_back(l);
            }
            c.swap_nanos += t.elapsed().as_nanos();
        }
        if let Some(next) = want {
            let r = self.reference.as_ref().expect("the record the event came from");
            let Some((event, [rt, rp, ro], heads)) = next else {
                s.counts.insert(self.name.clone(), c);
                print_counts(&s);
                println!("REFERENCE ENDS for {} at pass {}: {} holds no more decoded events", self.name, s.counts[&self.name].passes, r.path.display());
                crate::quit(1);
            };
            if rt != tokens || rp != positions || ro != out {
                s.counts.insert(self.name.clone(), c);
                print_counts(&s);
                let k = ro.iter().zip(&out).position(|(a, b)| a != b);
                println!(
                    "REFERENCE DIFFERS in {} pass {} (reference event {event} of {}): sequence equal {}, positions equal {}, first differing token {k:?} (reference {:?}, runtime {:?})",
                    self.name,
                    s.counts[&self.name].passes,
                    r.path.display(),
                    rt == tokens,
                    rp == positions,
                    k.map(|k| ro[k]),
                    k.map(|k| out[k])
                );
                crate::quit(1);
            }
            c.reference_equal += 1;
            // The engine's record names the line each of the stream's heads read in this
            // pass. The scope the pass decoded over must hold every one that has a cell.
            if let Some(scope) = s.scopes[self.part as usize].filter(|_| !heads.is_empty()) {
                let held = s.cache.scope_lines(scope);
                if let Some(n) = heads.iter().find(|&&n| !held.contains(&n) && s.cache.mirror.get(n as usize).is_some_and(|l| l.is_some())) {
                    s.counts.insert(self.name.clone(), c);
                    print_counts(&s);
                    println!("SCOPE MISSES LINE in {} pass {} (reference event {event}): the engine's pass read line {n:#07x} ({:#08x}), which holds a cell the pass's scope lacks", self.name, s.counts[&self.name].passes, n << LINE_BITS);
                    crate::quit(1);
                }
                c.scope_held += 1;
            }
        }
        s.counts.insert(self.name.clone(), c);
        hold(&s);
        let spent = SPENT.fetch_add(1, Ordering::Relaxed) + 1;
        let every = std::env::var("STOCK_PROGRESS").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000u64);
        let replayed = s.ckpt.as_ref().is_some_and(|k| k.resume.is_some());
        if spent % every == 0 && !replayed {
            println!("progress: frame {}, {spent} passes, {:.0}s elapsed, at unix {}", FRAME.load(Ordering::Relaxed), START.get().map_or(0.0, |t| t.elapsed().as_secs_f64()), unix());
            print_counts(&s);
        }
        if spent >= BUDGET.load(Ordering::Relaxed) {
            // With checkpoints, the run ends at the next quiet event, after its checkpoint.
            if s.ckpt.is_some() {
                BUDGET_HIT.store(true, Ordering::Relaxed);
            } else {
                print_counts(&s);
                println!("stopped at the pass budget: {} passes", SPENT.load(Ordering::Relaxed));
                crate::quit(0);
            }
        }
        after_event(&mut s);
        out
    }

    fn insert(&mut self, image: &Image) {
        self.own = follows(self.part).iter().flat_map(|&(base, len)| crate::cache::lines(base, len)).filter(|&n| image.has_line(n)).map(|n| (n, image.line(n))).collect();
        let mut s = self.shared.lock().unwrap();
        s.ledger.entry(self.name.clone()).or_default().own = self.own.iter().fold(0, |h, (&n, l)| h ^ own_digest(n, l));
        s.pending.push((self.name.clone(), image.clone()));
    }

    fn write(&mut self, addr: u32, line: [u8; LINE]) {
        let mut s = self.shared.lock().unwrap();
        s.prefill();
        if (REGS..REGS + N_REGS as u32).any(|a| a >> LINE_BITS == addr >> LINE_BITS) {
            s.cpu.map = None;
        }
        // A following stream's VRAM and OAM records are the GBC CPU's, whose cells the
        // cache already holds, so the cache decodes none of them again, only the stream's
        // palette writes. The GBC CPU may since have written the same line again: the
        // stream keeps its own line, and its passes lend it to the cell (`fill`).
        let c = s.counts.entry(self.name.clone()).or_default();
        if let Some(mine) = self.own.get_mut(&(addr >> LINE_BITS)) {
            let n = addr >> LINE_BITS;
            let moved = own_digest(n, mine) ^ own_digest(n, &line);
            *mine = line;
            c.skipped_writes += 1;
            s.ledger.entry(self.name.clone()).or_default().own ^= moved;
            after_event(&mut s);
            return;
        }
        assert!(self.part == Part::Cpu || !within(&FOLLOWED, addr), "{}: a write record at {addr:#08x}, a line of the GBC CPU's the stream does not follow", self.name);
        c.writes += 1;
        let n = addr >> LINE_BITS;
        if s.cache.mirror[n as usize] != Some(line) {
            let part = self.part;
            for (p, _, changed) in s.unfollowed.iter_mut().filter(|(p, lines, _)| *p != part && lines.binary_search(&n).is_ok()) {
                if *changed == 0 {
                    eprintln!("UNFOLLOWED LINE CHANGED: {}'s write record at {addr:#08x} changes line {n:#07x}, which {p:?}'s own heads can name and its engine holds its own copy of (first such record; the counts keep the total)", self.name);
                }
                *changed += 1;
            }
        }
        s.cache.write(addr, line);
        hold(&s);
        after_event(&mut s);
    }

    fn read(&mut self, dest: u32, source: u32) -> Option<[u8; LINE]> {
        let mut s = self.shared.lock().unwrap();
        s.prefill();
        s.counts.entry(self.name.clone()).or_default().reads += 1;
        // A reader record decodes over the cells as they are, with no line lent.
        assert!([dest, source].iter().all(|a| !self.own.contains_key(&(a >> LINE_BITS))), "{}: a reader record of {source:#08x} into {dest:#08x} reads a line the stream follows", self.name);
        let line = s.cache.read(dest, source);
        // In every mode: without the engine's record the mirror is the only account of what
        // a reader record must decode.
        if s.cache.stats.read_differs + s.cache.stats.read_rows_differ > 0 {
            print_counts(&s);
            println!("READ NOT HELD: {}'s reader record of {source:#08x} into {dest:#08x} decoded another line than the mirror gives, or other rows over its scope than over every cell", self.name);
            crate::quit(1);
        }
        hold(&s);
        after_event(&mut s);
        Some(line)
    }
}

/// Under `STOCK_REFERENCE`, stops the run at the first cell put back that differs from the
/// cell its pass found or was never compared.
fn hold(s: &Shared) {
    if s.strict
        && let Some(why) = s.cache.stats.unheld()
    {
        print_counts(s);
        println!("SWAP NOT HELD: {why}");
        crate::quit(1);
    }
}

fn print_counts(s: &Shared) {
    println!("cache: {:?}", s.cache.stats);
    for (p, lines, changed) in &s.unfollowed {
        println!("unfollowed {p:?}: {} lines its own heads can name outside what it follows, {changed} records of other streams changed one", lines.len());
    }
    println!("cpu scope: {} passes unscoped (no windows named since the start or a banking write), {} moves of the windows; {} passes over the narrow scope, {} lines joined it, {} passes decoded again after naming a line outside it, {} passes decoded first for the windows alone", s.cpu.unscoped, s.cpu.moves, s.cpu.narrow_passes, s.cpu.joined, s.cpu.reruns, s.cpu.probes);
    println!("decodes right after a write record, which shift its cell: {}, {:.1} ms each", s.cache.stats.shifting_passes, s.cache.stats.shifting_pass_ns as f64 / 1e6 / s.cache.stats.shifting_passes.max(1) as f64);
    println!("reader records: {} over the records' scope and their two lines, {} also over every cell, {} rows differ; {} decoded another line than the mirror's", s.cache.stats.scoped_reads, s.cache.stats.read_checks, s.cache.stats.read_rows_differ, s.cache.stats.read_differs);
    println!("scoped sequences read back from the runtime and held to the cache's account of their lines: {}, {:.1} ms each", s.cache.stats.scope_audits, s.cache.stats.scope_audit_ns as f64 / 1e6 / s.cache.stats.scope_audits.max(1) as f64);
    let mut names: Vec<&String> = s.counts.keys().collect();
    names.sort();
    for n in names {
        let k = &s.counts[n];
        let per = |x: u128| x as f64 / 1e6 / k.passes.max(1) as f64;
        let each = |x: u64| x as f64 / k.decodes.max(1) as f64;
        println!("{n} shape: {} decodes of its groups, each {:.1} tokens, {:.1} logit rows, {:.1} tokens attending the cells", k.decodes, each(k.batch), each(k.asked), each(k.attending));
        println!("{n}: {} passes, {:.0} tokens per pass, {:.1} ms per pass ({:.1} ms inside the runtime's decode); {} writes ({} already held), {} reader records; mirror {}/{} checks differ; slices {} passes, check {} passes, {} rows differ; group check {} passes, {} rows differ; mask check {} passes, {} rows differ; scope check {} passes, {} rows differ, {:.1} s in the decodes over every cell; {} scoped passes held every line the reference's heads name; {} passes equal the reference; {} passes of at most {} tokens on the whole model; {} passes with {} own lines lent to the cache, {:.1} s in those loans", k.passes, k.tokens as f64 / k.passes.max(1) as f64, per(k.nanos), per(k.decode_nanos), k.writes, k.skipped_writes, k.reads, k.mirror_differs, k.mirror_checks, k.sliced, k.slice_checks, k.slice_rows_differ, k.group_checks, k.group_rows_differ, k.mask_checks, k.mask_rows_differ, k.scope_checks, k.scope_rows_differ, k.scope_check_nanos as f64 / 1e9, k.scope_held, k.reference_equal, k.fenced, s.fence.map_or("?".to_string(), |f| f.to_string()), k.swap_passes, k.swap_lines, k.swap_nanos as f64 / 1e9);
    }
}

/// `STOCK_SLICE_EVERY`: every how many of a stream's sliced passes one is held to the whole
/// model (1,000 when unset, 0 for none).
fn slice_every() -> u64 {
    let Ok(v) = std::env::var("STOCK_SLICE_EVERY") else { return 1000 };
    v.parse().unwrap_or_else(|_| {
        println!("STOCK_SLICE_EVERY={v} is not a count of passes (0 holds no sliced pass to the whole model)");
        crate::quit(2)
    })
}

fn unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
}

/// What this run's checkpoints are bound to: the model file (its manifest's digests and
/// its size), this executable, the ROM, the llama.cpp the process loaded, the engine's
/// record (each file's size; a checkpoint also holds a digest of every line read) and
/// every setting that changes what the harness decodes.
fn identity(o: &Options, model: &str, rom_sha1: &str, hardware: Hardware) -> Identity {
    let path = std::path::Path::new(model);
    let size = std::fs::metadata(path).map_or(0, |m| m.len());
    let model = match gbcformer::wiring::Manifest::read(path) {
        Ok((m, _)) => format!("tensors {}, wiring {}, written for {}, {size} bytes", m.tensors_sha1, m.wiring_sha1, m.llama_cpp),
        Err(_) => format!("no manifest, sha1 {}, {size} bytes", crate::checkpoint::sha1_file(path)),
    };
    let harness = std::env::current_exe().map_or("unknown".to_string(), |p| crate::checkpoint::sha1_file(&p));
    let (lib, build) = crate::cpp::build();
    let llama = format!("{lib}, {build:?}, sha1 {}", crate::checkpoint::sha1_file(std::path::Path::new(&lib)));
    let reference = std::env::var("STOCK_REFERENCE").map_or("none".to_string(), |dir| {
        let mut files: Vec<String> = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir}: {e}")).flatten().map(|e| format!("{} {}", e.file_name().to_string_lossy(), e.metadata().map_or(0, |m| m.len()))).collect();
        files.sort();
        files.join(", ")
    });
    let env = |names: &[&str]| names.iter().map(|n| format!("{n}={}", std::env::var(n).unwrap_or_default())).collect::<Vec<_>>().join(" ");
    let settings = format!(
        "hardware {hardware:?}, boot {}, warmup {}, lanes {}, lockstep {}, threads {}; {}",
        o.boot.as_ref().map_or("none".to_string(), |b| crate::checkpoint::sha1_file(std::path::Path::new(b))),
        o.warmup,
        o.lanes,
        o.lockstep,
        o.threads,
        env(&["STOCK_GPU_LAYERS", "STOCK_GROUP", "STOCK_MASKS", "STOCK_SCOPE", "STOCK_CHECK_FROM", "STOCK_CHECK_EVERY", "STOCK_OWN", "STOCK_CPU_SCOPE", "STOCK_CPU_LRU", "STOCK_STREAMS", "STOCK_WRITES", "STOCK_READS", "STOCK_SPLIT", "STOCK_SWAP", "STOCK_KV", "STOCK_FA", "STOCK_PURE"])
    );
    Identity(vec![("model", model), ("harness", harness), ("rom", rom_sha1.to_string()), ("llama.cpp", llama), ("reference", reference), ("settings", settings)])
}

/// `STOCK_CHECKPOINT=<dir>`: the run's checkpoints, and the newest whole one to resume.
/// A checkpoint of another run stops this one.
fn checkpoints(o: &Options, model: &str, rom_sha1: &str, hardware: Hardware) -> Option<Ckpt> {
    let dir = std::env::var("STOCK_CHECKPOINT").ok()?;
    let number = |name: &str| std::env::var(name).ok().map(|v| v.parse::<f64>().unwrap_or_else(|_| panic!("{name}={v}: not a number of seconds")));
    let store = Store::open(&dir);
    let identity = identity(o, model, rom_sha1, hardware);
    let resume = match store.latest(&identity) {
        Ok(found) => found.map(|(m, state)| (m["event"].as_u64().expect("a checkpoint's event"), m, state)),
        Err(e) => {
            println!("RESUME REFUSED: {e}");
            crate::quit(1);
        }
    };
    match &resume {
        Some((event, m, state)) => {
            if std::env::var_os("STOCK_REFERENCE").is_none() {
                println!("RESUME REFUSED: a resume answers the passes before the checkpoint from the engine's record: set STOCK_REFERENCE");
                crate::quit(1);
            }
            eprintln!("checkpoint found: event {event} of chunk {} in {dir} ({}, {} bytes): the host runs again from frame 0 to it with NO MODEL, every pass answered from the engine's record", m["chunk"], state.display(), m["state"]["bytes"]);
        }
        None => eprintln!("checkpoints: none in {dir}; the run starts at frame 0"),
    }
    unsafe {
        signal(2, on_stop);
        signal(15, on_stop);
    }
    let every = number("STOCK_CHECKPOINT_SECS").unwrap_or(600.0);
    let stop = crate::checkpoint::Stop::from_env();
    // A run that starts at frame 0 prefills here, with the prefill's own checkpoints.
    let prefill = resume.is_none().then(|| crate::checkpoint::Prefill::new(&dir, identity.clone(), every, stop.clone()));
    Some(Ckpt {
        store,
        identity,
        every,
        last: Instant::now(),
        last_event: None,
        stop,
        prefill,
        due_now: false,
        chunk: 1,
        prior_s: 0.0,
        resume,
        record_dir: std::env::var_os("GBCFORMER_RECORD").map(Into::into),
        records: BTreeMap::new(),
        saved: 0,
        cost_s: 0.0,
    })
}

pub fn run(o: Options) {
    crate::checkpoint::process_s();
    BUDGET.store(o.max_passes, Ordering::Relaxed);
    let bytes = std::fs::read(&o.rom).unwrap_or_else(|e| panic!("{}: {e}", o.rom));
    let rom_sha1 = sha1_smol::Sha1::from(&bytes[..]).digest().to_string();
    let header = gbcformer::rom::identify(&bytes).unwrap();
    let hardware = Hardware::resolve(&header, o.hardware).unwrap();
    let mut warm = match &o.boot {
        Some(boot) => {
            assert_eq!(o.warmup, 0, "--boot starts at power-on: no warmup");
            eprintln!("boot: the run starts at power-on with the boot ROM {boot} mapped");
            Session::power_on(bytes, None, hardware, std::fs::read(boot).unwrap_or_else(|e| panic!("{boot}: {e}"))).unwrap()
        }
        None => Session::new(bytes, None, hardware).unwrap(),
    };
    while warm.frame_number() < o.warmup.min(o.frames) {
        warm.run_frame(Buttons::default());
    }
    if o.warmup > 0 {
        eprintln!("warmup: frames 0..{} ran on the reference interpreter, not the selected stack ({} steps)", warm.frame_number(), warm.counters().steps);
    }
    let group = match std::env::var("STOCK_GROUP").as_deref() {
        Ok("full") => Group::Full,
        Ok("check") => Group::Check,
        _ => Group::Own,
    };
    let drop_masks = match std::env::var("STOCK_MASKS").as_deref() {
        Ok("0") => Masks::Keep,
        Ok("check") => Masks::Check,
        _ => Masks::Drop,
    };
    let mut shared = None;
    let strict = std::env::var_os("STOCK_REFERENCE").is_some();
    let replay = o.model.as_deref() == Some("replay");
    assert!(strict || !replay, "play replay decodes the tokens of the engine's record: set STOCK_REFERENCE");
    if let Some(model) = o.model.clone() {
        let (m, specs, circuit, cpu_own) = crate::checkpoint_parts();
        let cfg = m.cfg.clone();
        let t = Instant::now();
        // Room for a cell per line the cartridge's image holds (its ROM and RAM, the fixed
        // regions) and the palettes' lines, every prefill lane's record and a group. The
        // cells sit at their lines' positions, up to P, but a context's size is its cell
        // count: sizing it by P would make a small ROM pay for the 8 MiB layout.
        let cells = warm.memory_image().expect("the cartridge's memory image").lines() + PAL_LEN.div_ceil(LINE);
        let replayed = crate::runtime::Replay::default();
        let fault = std::env::var("STOCK_FAULT").ok().filter(|_| replay).map(|k| (k.parse().expect("STOCK_FAULT=<the loans' write record to corrupt>"), replayed.fault.clone()));
        let mut ckpt = None;
        let rt: Box<dyn crate::runtime::Runtime> = if replay {
            eprintln!("REPLAY: no model runs. Every pass decodes the tokens of the engine's record, so this run checks the harness's scopes, loans and mirror, and no token");
            Box::new(replayed)
        } else {
            // The runtime opens first, so a build or a file the run refuses stops it here.
            let rt = crate::runtime::open(&model, &cfg, cells + (gbcformer::memory::dense::Records::CELL + 1) * o.lanes + 1024, o.lanes, o.threads);
            ckpt = checkpoints(&o, &model, &rom_sha1, hardware);
            match &ckpt {
                Some(Ckpt { resume: Some(_), .. }) => Box::new(crate::runtime::Resuming::new(rt)),
                _ => rt,
            }
        };
        eprintln!("runtime over {model} opened in {:.1}s", t.elapsed().as_secs_f64());
        let slots = m.memory.as_ref().expect("a memory checkpoint").slots;
        let head = gbcformer::gguf::HeadRows::of(&m);
        // In an overlay a stream's rows read the cells with its own heads alone
        // (`scope::own`). STOCK_OWN=0 keeps every head's lines in every scope.
        let own = m.overlay.is_some() && std::env::var("STOCK_OWN").as_deref() != Ok("0");
        let groups: Vec<(Vec<usize>, Vec<bool>, Vec<Vec<usize>>)> = Part::ALL
            .iter()
            .map(|&p| {
                let g = gbcformer::one_model::group(&m, &specs, p);
                let mut settled = vec![false; cfg.seq()];
                for &q in specs[p as usize].at[specs[p as usize].n_inputs..].iter().flatten() {
                    settled[q] = true;
                }
                for &q in g.gathers.iter().flat_map(|h| &h.targets) {
                    settled[q] = false;
                }
                (g.keep, settled, g.gathers.into_iter().map(|h| h.attended).collect())
            })
            .collect();
        drop(m);
        let mut cache = Cache::new(rt, cfg.clone(), slots, head);
        cache.fault = fault;
        cache.prefill_ckpt = ckpt.as_mut().and_then(|k| k.prefill.take());
        cache.fault_put_back = std::env::var("STOCK_FAULT_PUT_BACK").ok().map(|k| k.parse().expect("STOCK_FAULT_PUT_BACK=<the put-back to corrupt, from 0>"));
        cache.fault_read = std::env::var("STOCK_FAULT_READ").ok().map(|k| k.parse().expect("STOCK_FAULT_READ=<the reader record to corrupt, from 0>"));
        // With STOCK_SPLIT=0, every group token attends the cells.
        cache.split = std::env::var("STOCK_SPLIT").as_deref() != Ok("0");
        // With STOCK_SCOPE=0, every stream's rows attend every cell. STOCK_SCOPE=check
        // decodes both ways and holds the rows equal.
        let scope = match std::env::var("STOCK_SCOPE").as_deref() {
            Ok("0") => ScopeMode::Off,
            Ok("check") => ScopeMode::Check,
            _ => ScopeMode::On,
        };
        // With STOCK_WRITES=0, every write record decodes over every cell. STOCK_WRITES=check
        // decodes both ways and holds the two cells' saved bytes equal.
        cache.writes = match std::env::var("STOCK_WRITES").as_deref() {
            Ok("0") => crate::cache::Writes::Whole,
            Ok("check") => crate::cache::Writes::Check,
            _ => crate::cache::Writes::Scoped,
        };
        // With STOCK_READS=0, every reader record decodes over every cell. STOCK_READS=check
        // decodes both ways and holds the rows equal.
        cache.reads = match std::env::var("STOCK_READS").as_deref() {
            Ok("0") => crate::cache::Writes::Whole,
            Ok("check") => crate::cache::Writes::Check,
            _ => crate::cache::Writes::Scoped,
        };
        // With STOCK_CPU_SCOPE=windows, every GBC CPU pass decodes over the windows' scope.
        // STOCK_CPU_LRU is the most lines the narrow scope keeps beyond its fixed ones.
        let cpu = CpuScope {
            narrow_on: std::env::var("STOCK_CPU_SCOPE").as_deref() != Ok("windows"),
            recent: crate::scope::Recent::new(std::env::var("STOCK_CPU_LRU").ok().map_or(256, |v| v.parse().expect("STOCK_CPU_LRU=<lines>"))),
            fault: std::env::var("STOCK_FAULT_SCOPE").ok().map(|k| k.parse().expect("STOCK_FAULT_SCOPE=<the narrow pass to fault, from 0>")),
            fault_map: std::env::var("STOCK_FAULT_MAP").ok().map(|k| k.parse().expect("STOCK_FAULT_MAP=<the narrow pass to fault, from 0>")),
            fault_probe: std::env::var("STOCK_FAULT_PROBE").ok().map(|k| k.parse().expect("STOCK_FAULT_PROBE=<the probe to fault, from 0>")),
            fault_named: std::env::var("STOCK_FAULT_NAMED").ok().map(|k| k.parse().expect("STOCK_FAULT_NAMED=<the narrow pass to fault after, from 0>")),
            ..CpuScope::default()
        };
        // With STOCK_STREAMS=0, every pass runs the whole model. STOCK_STREAMS=check
        // decodes both ways and holds the rows equal.
        let streams = match std::env::var("STOCK_STREAMS").as_deref() {
            Ok("0") => StreamMode::Whole,
            Ok("check") => StreamMode::Check,
            Ok("require") => StreamMode::Require,
            _ => StreamMode::Sliced,
        };
        let fence = cache.rt.fence();
        let s = Arc::new(Mutex::new(Shared { cache, pending: Vec::new(), counts: HashMap::new(), circuit: Some(circuit), scopes: [None; Part::ALL.len()], scope, own, settled: true, check_from: std::env::var("STOCK_CHECK_FROM").ok().map_or(0, |v| v.parse().expect("STOCK_CHECK_FROM=<pass>")), check_every: std::env::var("STOCK_CHECK_EVERY").ok().map_or(1, |v| v.parse::<u64>().expect("STOCK_CHECK_EVERY=<passes>").max(1)), fault_sequence: std::env::var("STOCK_FAULT_SEQUENCE").ok().map(|k| k.parse().expect("STOCK_FAULT_SEQUENCE=<pass>")), cpu_own: Some(cpu_own), streams, slice_forms: HashSet::new(), slice_every: slice_every(), fence, cpu, strict, swap: std::env::var("STOCK_SWAP").as_deref() != Ok("0"), lends: std::env::var("STOCK_LENDS").ok().map(|p| std::fs::File::create(&p).unwrap_or_else(|e| panic!("{p}: {e}"))), unfollowed: Vec::new(), ckpt, ledger: BTreeMap::new() }));
        shared = Some(s.clone());
        set_external(Box::new(move |slots: &[String], _n_inputs: usize| {
            let Some(part) = Part::ALL.into_iter().find(|&p| specs[p as usize].slots == slots) else {
                eprintln!("no part has these {} slots: left on the engine", slots.len());
                return None;
            };
            let (keep, settled, attended) = groups[part as usize].clone();
            eprintln!("stream {part:?}: every pass in the runtime; a reduced group keeps {} of {} positions", keep.len(), cfg.seq());
            let mask = tok::output_mask(&cfg, cfg.n_inputs) - cfg.n_inputs;
            let masks = mask + cfg.n_inputs..mask + cfg.seq();
            let spec = &specs[part as usize];
            let at = |name: &str| spec.slots.iter().position(|n| n == name).and_then(|i| spec.at[i]);
            let map_triples = |names: &[[&str; 3]; 4]| names.iter().map(|w| w.iter().map(|n| at(n)).collect::<Option<Vec<usize>>>().map(|v| [v[0], v[1], v[2]])).collect::<Option<Vec<_>>>().map(|v| [v[0], v[1], v[2], v[3]]);
            let (map_at, named_at) = (map_triples(&crate::scope::MAP_SLOTS), map_triples(&crate::scope::NAMED_SLOTS));
            let word = |low: &str, high: &str| at(low).zip(at(high)).map(|(l, h)| [l, h]);
            let pc_sp_at = word("PCL", "PCH").zip(word("SPL", "SPH")).map(|(pc, sp)| [pc, sp]);
            let reference = std::env::var("STOCK_REFERENCE").ok().map(|dir| Reference::open(&dir, slots));
            if let Some(r) = &reference {
                eprintln!("stream {part:?}: every decoded pass held to {}", r.path.display());
            }
            if streams != StreamMode::Whole {
                let live = s.lock().unwrap().cache.rt.can_stream(part.name());
                if live {
                    eprintln!("stream {part:?}: its passes run its slice of the model (the GGUF's gbcformer.stream.{} keys; a group of at most {} tokens runs the whole model, the runtime's fence)", part.name(), fence.map_or("?".to_string(), |f| f.to_string()));
                    if streams != StreamMode::Check && s.lock().unwrap().slice_every == 0 {
                        eprintln!("stream {part:?}: NO sliced pass is held to the whole model (STOCK_SLICE_EVERY=0)");
                    }
                } else {
                    eprintln!("stream {part:?}: its passes run the WHOLE model: the runtime lacks stream-slices.patch, or the GGUF does not name the stream");
                    if streams == StreamMode::Require {
                        eprintln!("STOCK_STREAMS=require: refusing to run without the slice");
                        crate::quit(1);
                    }
                }
            }
            Some(Box::new(Stream { part, name: format!("{part:?}"), shared: s.clone(), keep, mask, masks_at: masks, settled, attended, group, masks: drop_masks, map_at, named_at, pc_sp_at, reference, own: BTreeMap::new(), first_at: spec.at[0] }) as Box<dyn External>)
        }));
    }
    let stack = Stack::parse("all", Some(&o.lockstep)).unwrap().with_one_model(true);
    let instr = Instrumentation { timed: false, oracle_every: 0, step_oracle: Default::default(), apu_same_thread: true, ppu_same_thread: true, rows_same_thread: true };
    let t = Instant::now();
    let mut s = warm.with_stack(stack, instr).unwrap();
    eprintln!("{stack}; built in {:.1}s; hardware {hardware:?}", t.elapsed().as_secs_f64());
    let start = *START.get_or_init(Instant::now);
    while s.frame_number() < o.frames {
        let f0 = Instant::now();
        s.run_frame(Buttons::default());
        FRAME.store(s.frame_number(), Ordering::Relaxed);
        // Under a resume, a frame that ended before the checkpoint's event ran no model.
        let replayed = shared.as_ref().is_some_and(|sh| sh.lock().unwrap().ckpt.as_ref().is_some_and(|k| k.resume.is_some()));
        let label = if replayed { "; REPLAYED from the engine's record, no model ran" } else { "" };
        println!("frame {} sha1 {} in {:.1}s ({:.1}s total, at unix {}){label}", s.frame_number(), frame_sha1(s.frame()), f0.elapsed().as_secs_f64(), start.elapsed().as_secs_f64(), unix());
        if let Some(sh) = shared.as_ref().filter(|_| !replayed) {
            print_counts(&sh.lock().unwrap());
        }
    }
    // No checkpoint is taken at the run's end. By then the engines have written the last
    // event's record line, which a replay has not when it reaches that event, so such a
    // checkpoint could not be resumed. A finished run that is started again continues
    // from its last checkpoint inside the run and decodes the rest again.
    if let Some(at) = shared.as_ref().and_then(|sh| sh.lock().unwrap().ckpt.as_ref().and_then(|k| k.resume.as_ref().map(|r| r.0))) {
        println!("the run ended before the checkpoint's event {at}: every pass was answered from the engine's record and none ran in the model");
    }
    for line in s.report(start.elapsed().as_secs_f64(), false) {
        eprintln!("{line}");
    }
}

/// Holds recording `a` to recording `b`, stream by stream and event by event: the
/// sequence, the positions and the decoded tokens. `a` may be a prefix of `b` (a run stopped
/// at a pass budget against a longer one). False when a stream is in one recording only, an
/// event differs, `a` has events `b` lacks, or nothing was compared.
pub fn diff(a: &str, b: &str) -> bool {
    let streams = |d: &str| -> HashMap<String, Vec<String>> {
        std::fs::read_dir(d)
            .unwrap()
            .flatten()
            .map(|e| {
                let text = std::fs::read_to_string(e.path()).unwrap();
                let mut lines = text.lines();
                (lines.next().unwrap_or("").to_string(), lines.map(String::from).collect())
            })
            .collect()
    };
    let (sa, sb) = (streams(a), streams(b));
    let mut total = (0u64, 0u64, 0u64);
    let mut ok = true;
    for header in sb.keys().filter(|h| !sa.contains_key(*h)) {
        println!("stream {}: only in {b}", &header[..60.min(header.len())]);
        ok = false;
    }
    for (header, la) in &sa {
        let Some(lb) = sb.get(header) else {
            println!("stream {}: only in {a}", &header[..60.min(header.len())]);
            ok = false;
            continue;
        };
        let (mut same, mut differ, mut tokens) = (0u64, 0u64, 0u64);
        for (x, y) in la.iter().zip(lb) {
            let (x, y): (serde_json::Value, serde_json::Value) = (serde_json::from_str(x).unwrap(), serde_json::from_str(y).unwrap());
            let eq = ["kind", "tokens", "positions", "out"].iter().all(|k| x[k] == y[k]);
            tokens += x["out"].as_array().map_or(0, |v| v.len() as u64);
            if eq {
                same += 1;
            } else {
                differ += 1;
                if differ <= 3 {
                    println!("  differs at event {} ({}): out {} vs {}", x["event"], x["kind"], x["out"], y["out"]);
                }
            }
        }
        let first: String = header.chars().skip(10).take(24).collect();
        println!("stream {first}…: {} vs {} events, {} compared, {same} equal, {differ} differ, {tokens} decoded tokens compared", la.len(), lb.len(), la.len().min(lb.len()));
        if la.len() > lb.len() {
            println!("  {} events of {a} have none in {b}", la.len() - lb.len());
            ok = false;
        }
        total = (total.0 + same, total.1 + differ, total.2 + tokens);
    }
    println!("all streams: {} events equal, {} differ, {} decoded tokens compared", total.0, total.1, total.2);
    ok && total.1 == 0 && total.0 > 0
}

/// Why `Runtime::split_pass` decodes the same rows as `pass`, from the weights. A group
/// token whose row is not read attends only the group (and two cells) instead of every
/// cell. The rows read are the same if every such token's keys and values at every layer
/// are, since the rows read see the same keys, values and mask entries either way. Three
/// facts show that:
/// 1. layer 0's query reads only the constant channels, which every token's embedding
///    holds alike, so every token asks layer 0 the same query;
/// 2. against that query, a cell's layer-0 key (every cell is the `CELL` token) scores at
///    least `EXP_ZERO` below the best key of every part's reduced group, whatever the
///    group's input bytes, so the cells weigh exactly 0 at layer 0 for every token;
/// 3. every key and value at layers 1 and up reads only channels no MLP (layer 0's
///    included) and no attention past layer 0 writes. Each is then a function of the
///    embedding and the layer-0 attention, which 1 and 2 make the same with or without
///    the cells;
/// 4. those keys and values pass through the row's norm, over every channel, so the
///    rest of the row must stay finite in the runtime, whose `ffn_down` takes its input
///    in f16: one unit past 65504 is inf there, and the norm spreads it to the whole row.
///    `Model::f16_acc_peaks` bounds a unit where its wires hold their ranges, and they
///    hold at a token that skips the cells only if no memory head there needs a cell.
///    Every mask token attends the cells (`Stream::decode`), so the tokens that skip them
///    are the rest, and no embedding but a mask's sets a mode channel: every overlay
///    memory head there reads its null term (`Layout::mode`) and finds the null cell,
///    which the split keeps.
pub fn split_check() {
    use gbcformer::llama::{attn_scale, rms_norm, CONSTANTS, HEAD_DIM};
    use gbcformer::memory::attn::EXP_ZERO;
    let (m, specs) = crate::checkpoint_cfg();
    let c = &m.cfg;
    let (d, a, kvd, inter) = (c.hidden, c.attn(), c.kv_dim(), c.intermediate);
    let nc = CONSTANTS.len();
    let mut ok = true;
    let committed = m.layout.committed();
    let holders: Vec<usize> = (0..c.vocab).filter(|&t| m.embed[t * d + committed] != 0.0).collect();
    println!("committed channel {committed}: set in the embedding of tokens {holders:?} (CELL is {})", tok::cell(c));
    ok &= holders == [tok::cell(c)];
    // 1. Layer 0's query.
    let l0 = &m.layers[0];
    let q_reads: Vec<usize> = (0..d).filter(|&ch| (0..a).any(|r| l0.q[r * d + ch] != 0.0)).collect();
    let consts_alike = (0..c.vocab).all(|t| (0..nc).all(|i| m.embed[t * d + i] == m.embed[i]));
    println!("1. layer 0's query reads channels {q_reads:?}; every embedding holds the same constants: {consts_alike}");
    ok &= q_reads.iter().all(|&ch| ch < nc) && consts_alike;
    let normed = |t: usize| rms_norm(&m.embed[t * d..][..d], d);
    let project = |w: &[f32], rows: usize, h: &[f32]| -> Vec<f32> { (0..rows).map(|r| (0..d).filter(|&i| h[i] != 0.0).map(|i| w[r * d + i] * h[i]).sum()).collect() };
    let q = project(&l0.q, a, &normed(tok::cell(c)));
    for t in [0, tok::rec(c), tok::output_mask(c, c.n_inputs)] {
        ok &= project(&l0.q, a, &normed(t)) == q;
    }
    // 2. The cells' layer-0 weight. A cell's key turned by RoPE keeps its length, so
    // |q|·|k| bounds its score at any position. The group's keys share the query's
    // position, where RoPE cancels. An overlay's null cell (`Runtime::null`) counts as a
    // cell here too, and the larger of the two keys bounds both.
    let cached: Vec<usize> = std::iter::once(tok::cell(c)).chain((tok::null(c) < c.vocab).then(|| tok::null(c))).collect();
    let keys: Vec<Vec<f32>> = cached.iter().map(|&t| project(&l0.k, kvd, &normed(t))).collect();
    let scores = |k: &[f32]| -> Vec<f32> {
        (0..c.heads).map(|h| (0..HEAD_DIM).map(|i| q[h * HEAD_DIM + i] * k[c.kv_head(h) * HEAD_DIM + i]).sum::<f32>() * attn_scale()).collect()
    };
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cell_bound: Vec<f32> = (0..c.heads).map(|h| keys.iter().map(|kc| norm(&q[h * HEAD_DIM..][..HEAD_DIM]) * norm(&kc[c.kv_head(h) * HEAD_DIM..][..HEAD_DIM]) * attn_scale()).fold(0f32, f32::max)).collect();
    // Per position, each head's lowest score over the tokens that can sit there.
    let lowest: Vec<Vec<f32>> = (0..c.seq())
        .map(|p| {
            let toks: Vec<usize> = if p < c.n_inputs { (0..=255u8).map(|v| tok::input(c, p, v)).collect() } else { vec![tok::output_mask(c, p)] };
            toks.iter().map(|&t| scores(&project(&l0.k, kvd, &normed(t)))).fold(vec![f32::INFINITY; c.heads], |acc, s| acc.iter().zip(&s).map(|(x, y)| x.min(*y)).collect())
        })
        .collect();
    // A head that writes nothing (its `W_o` columns zero) may weigh the cells as it likes.
    let live: Vec<usize> = (0..c.heads).filter(|&h| (0..d).any(|ch| (0..HEAD_DIM).any(|j| l0.o[ch * a + h * HEAD_DIM + j] != 0.0))).collect();
    println!("2. layer 0's heads that write: {live:?} of {}", c.heads);
    for part in Part::ALL {
        let keep = gbcformer::one_model::group(&m, &specs, part).keep;
        let margin = (0..c.heads).filter(|h| live.contains(h)).map(|h| keep.iter().map(|&p| lowest[p][h]).fold(f32::NEG_INFINITY, f32::max) - cell_bound[h]).fold(f32::INFINITY, f32::min);
        println!("2. {part:?}: the group's best layer-0 key beats any cell's by at least {margin} (needs {EXP_ZERO}) in every head that writes");
        ok &= margin >= EXP_ZERO;
    }
    // 3. What the later keys and values read, against what MLPs and later attention write.
    let mut written = vec![false; d];
    for (l, layer) in m.layers.iter().enumerate() {
        for ch in 0..d {
            written[ch] |= (0..inter).any(|j| layer.down[ch * inter + j] != 0.0) || (l > 0 && (0..a).any(|j| layer.o[ch * a + j] != 0.0));
        }
    }
    let mut clash = Vec::new();
    for (l, layer) in m.layers.iter().enumerate().skip(1) {
        for ch in 0..d {
            if written[ch] && (0..kvd).any(|r| layer.k[r * d + ch] != 0.0 || layer.v[r * d + ch] != 0.0) {
                clash.push((l, ch));
            }
        }
    }
    println!("3. channels a key or value past layer 0 reads that an MLP or later attention writes: {clash:?}");
    ok &= clash.is_empty();
    // 4. The tokens that skip the cells carry no mode.
    let masks = tok::output_mask(c, c.n_inputs)..tok::output_mask(c, c.seq() - 1) + 1;
    let modes: Vec<usize> = (0..m.layout.offsets.len()).map(|p| m.layout.mode(p)).collect();
    let moded: Vec<usize> = (0..c.vocab).filter(|t| !masks.contains(t) && modes.iter().any(|&ch| m.embed[t * d + ch] != 0.0)).collect();
    println!("4. tokens other than masks whose embedding sets a mode channel {modes:?}: {moded:?}");
    ok &= moded.is_empty();
    println!("split: {}", if ok { "the rows read are the same with or without the cells for the rest" } else { "NOT shown" });
    if !ok {
        crate::quit(1);
    }
}
