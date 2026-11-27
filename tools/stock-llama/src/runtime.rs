//! The cache operations of `docs/stock-llama.md` §4, as llama.cpp implements them.
//!
//! One sequence (0) holds one cell per 8-byte line, at the line's index as its position.
//! A pass decodes its group at the next position over the cache and drops it. A write
//! evicts the line's cell, decodes the record's scratch tokens (`REC WP0a..c WP1a..c
//! WL0..7`) at the next position and its cell token one after, drops the scratch and
//! shifts the cell token to the line. Stock llama.cpp requires a sequence's batch to
//! start at its largest position plus one, so "the next position" means that position.
//! It is `P = 0x108800` (`READ_AT`, the 24-bit layout's line count) while the last line has
//! its cell, which is true after the prefill except while that cell is rewritten.

use crate::cpp::{LlamaCpp, Row, Tok};

/// Where one decode sat.
#[derive(Clone, Copy, Debug, Default)]
pub struct At {
    pub pos: usize,
}

pub trait Runtime: Send {
    /// A group over the cache, at the next position; the rows at `outs` (indexes into
    /// `tokens`), in `outs` order. Nothing of it stays.
    fn pass(&mut self, tokens: &[usize], outs: &[usize]) -> (Vec<Row>, At);
    /// `pass`, with only the tokens at `outs` and at `also` attending the cells; no row is
    /// read at `also`. It gives the same rows when every other token's later keys and
    /// values depend only on its embedding and its layer-0 attention, which gives the
    /// cells zero weight, and every other token's row stays finite, which holds when none
    /// is a mask (`stock-llama split-check`, facts 1 to 4; `Cache::pass` asserts the last).
    fn split_pass(&mut self, tokens: &[usize], outs: &[usize], _also: &[usize]) -> (Vec<Row>, At) {
        self.pass(tokens, outs)
    }
    /// A sequence holding the cells of the lines in `runs` (each `first..end`), kept so
    /// by every later write, or None when the runtime has no such sequences.
    fn scope(&mut self, _runs: &[(u32, u32)]) -> Option<Scoped> {
        None
    }
    /// `split_pass` with the tokens at `outs` and `also` attending only the cells of `scope`. It
    /// gives the same rows when each of their heads gives zero weight to every cell `scope` lacks.
    fn scoped_pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize], _scope: Scoped) -> (Vec<Row>, At) {
        self.split_pass(tokens, outs, also)
    }
    /// `scope`'s cells replaced by those of the lines in `runs`.
    fn rescope(&mut self, _scope: Scoped, _runs: &[(u32, u32)]) {
        unimplemented!("this runtime keeps no scoped sequences")
    }
    /// Line `n`'s cell joins `scope`.
    fn scope_add(&mut self, _scope: Scoped, _n: u32) {
        unimplemented!("this runtime keeps no scoped sequences")
    }
    /// Line `n`'s cell leaves `scope`.
    fn scope_drop(&mut self, _scope: Scoped, _n: u32) {
        unimplemented!("this runtime keeps no scoped sequences")
    }
    /// The positions of the cells the runtime itself holds in `scope`, sorted, read back
    /// from the runtime and not from any account of the harness's; None when the runtime
    /// keeps no sequences or cannot say.
    fn scope_positions(&mut self, _scope: Scoped) -> Option<Vec<i32>> {
        None
    }
    /// Line `n`'s cell leaves `scope` in the runtime alone, behind the cache's account of
    /// the scope: a deliberate fault (`STOCK_FAULT_SEQUENCE`).
    fn fault_drop(&mut self, _scope: Scoped, _n: u32) {}
    /// The cell of line `n` replaced (or created, without `evict`) by the cell token of
    /// the write record `rec`, its last token. With `scope`, the record's tokens attend
    /// only that sequence's cells, which gives the same cell when no head of a record
    /// gives weight to a cell the scope lacks (as a prefill lane's record sees only the
    /// cells decoded before the lanes opened). The new cell joins the scopes in `join`,
    /// the ones whose lines hold `n` (`Cache` keeps each scope's lines), and no other.
    fn write(&mut self, n: u32, rec: &[usize], evict: bool, scope: Option<Scoped>, join: &[Scoped]) -> At;
    /// The saved bytes of line `n`'s cell (its position and every layer's key and value
    /// row), or None when the runtime cannot save one.
    fn cell_bytes(&mut self, _n: u32) -> Option<Vec<u8>> {
        None
    }
    /// Names the tokens the next pass decodes, to a runtime that replays a record.
    fn answer(&mut self, _out: &[usize]) {}
    /// Whether the runtime replays a record's tokens instead of running a model (`Replay`).
    fn replays(&self) -> bool {
        false
    }
    /// Write records for lines without a cell, each decoded in its own lane. It sees the
    /// cache as the lanes opened and its lane's earlier records (other lines), and its
    /// own tokens. Every record has the same length. `open_lanes` first, `close_lanes`
    /// after. The batch's cells stay where they decoded until `lanes_place`.
    fn lanes_decode(&mut self, recs: &[(u32, Vec<usize>)]);
    /// Drops the scratch tokens of the batch `lanes_decode` last decoded and moves its
    /// cells to `lines`, the batch's lines in lane order. It decodes nothing, so between
    /// the two calls no cell waits for a shift and `save_all` saves the whole of the
    /// prefill so far.
    fn lanes_place(&mut self, lines: &[u32]);
    /// Saves every sequence (the cache and the open lanes) to `path`, or false when the
    /// runtime cannot.
    fn save_all(&mut self, _path: &str) -> bool {
        false
    }
    /// Loads what `save_all` wrote in place of every cell, or false.
    fn load_all(&mut self, _path: &str) -> bool {
        false
    }
    fn open_lanes(&mut self, lanes: usize);
    fn close_lanes(&mut self, lanes: usize);
    fn lanes(&self) -> usize;
    /// Decodes an overlay's null token alone at position 0 of the empty sequence 0, so its
    /// own forward pass makes the null cell every foreign memory head attends to
    /// (`docs/stock-llama.md` §4); every later sequence holds it.
    fn null(&mut self, _token: usize) {
        unimplemented!("this runtime keeps no null cell")
    }
    /// Saves the cache (sequence 0) to `path`, or false when the runtime cannot.
    fn save(&mut self, _path: &str) -> bool {
        false
    }
    /// Loads a cache `save` wrote, or false.
    fn load(&mut self, _path: &str) -> bool {
        false
    }
    /// Wall time inside the runtime's forward calls, summed.
    fn decode_nanos(&self) -> u128;
    /// The passes that follow run only the slice of the overlay checkpoint's stream
    /// `name` (its layers, heads and FFN units, `gbcformer.stream.<name>.*` in the GGUF),
    /// or the whole model again with None. False when the runtime cannot, and then it runs
    /// the whole model. A slice gives the same rows at a group of that stream: every
    /// other unit's `up` reads another part's mode channel, which is 0 at every token of
    /// the group, every other head attends the null cell, and the stream has neither in a
    /// later layer (`Model::stream`).
    fn stream(&mut self, _name: Option<&str>) -> bool {
        false
    }
    /// Whether the last pass's rows ran a stream's slice, as the runtime itself reports
    /// it, or None when it cannot say.
    fn sliced(&self) -> Option<bool> {
        None
    }
    /// The most tokens of a group the runtime runs on the whole model whatever stream is
    /// named, or None when it does not say.
    fn fence(&self) -> Option<usize> {
        None
    }
    /// Whether the runtime that decodes this run's passes runs stream `name`'s slice
    /// (`stream`), leaving the whole model selected. A resuming runtime answers for the
    /// runtime it goes live on.
    fn can_stream(&mut self, name: &str) -> bool {
        let live = self.stream(Some(name));
        self.stream(None);
        live
    }
    /// A resuming runtime (`Resuming`) stops replaying: it loads the cache saved at
    /// `state` and makes the scoped sequences the replay asked for, each from its runs of
    /// lines in `scopes`. False when the state does not load.
    fn go_live(&mut self, _state: &str, _scopes: &[(Scoped, Vec<(u32, u32)>)]) -> bool {
        unimplemented!("this runtime resumes nothing")
    }
}

/// Rows of a decode that asked for `outs` (in any order), put in `outs` order, since the
/// runtime returns them in batch order.
fn ordered(rows: Vec<Row>, outs: &[usize]) -> Vec<Row> {
    if outs.windows(2).all(|w| w[0] < w[1]) {
        return rows;
    }
    let mut sorted: Vec<usize> = outs.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    outs.iter().map(|o| rows[sorted.binary_search(o).unwrap()].clone()).collect()
}

/// The positions of the cells in a sequence llama.cpp saved for sequence `seq`, sorted, or
/// None when the bytes are not such a file. llama.cpp b12070 writes a magic, a version
/// and a token count (0 here), then its cache's stream count (1), the cell count and, per
/// cell, its position, the number of sequences listed (1, the one saved) and that
/// sequence (`llama_context::state_seq_save_file`, `llama_kv_cache::state_write`,
/// `state_write_meta`). Each of those is required as read, and after the last cell the
/// rows' header must follow (`v_trans`, 0 or 1, and a layer count).
pub fn saved_positions(bytes: &[u8], seq: i32) -> Option<Vec<i32>> {
    let word = |i: usize| bytes.get(4 * i..4 * i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    if word(2)? != 0 || word(3)? != 1 {
        return None;
    }
    let cells = word(4)? as usize;
    let mut positions = Vec::with_capacity(cells);
    for k in 0..cells {
        if word(6 + 3 * k)? != 1 || word(7 + 3 * k)? as i32 != seq {
            return None;
        }
        positions.push(word(5 + 3 * k)? as i32);
    }
    if cells > 0 && (word(5 + 3 * cells)? > 1 || !(1..=4096).contains(&word(6 + 3 * cells)?)) {
        return None;
    }
    positions.sort_unstable();
    Some(positions)
}

/// A scoped sequence (`Runtime::scope`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scoped(i32);

/// The sequence `split_pass` puts the tokens that do not attend the cells in: a prefill
/// lane's, empty once the lanes close.
const SPLIT: i32 = 1;
/// Scoped sequences follow `SPLIT`, at most this many. The runtime makes them after the
/// prefill, so they share ids with prefill lanes, which are empty by then. They are the
/// records' scope, one per stream with lines of its own (the PPU's, the length's, the
/// stepper's, and one the APU and control share) and the GBC CPU's two.
const SCOPES: usize = 8;
/// The null cell's position: line 0's, a ROM line no write evicts, where the dense pass
/// ropes the null key (`llama::null_kv`). A sequence may hold two cells at a position.
const NULL_AT: i32 = 0;

pub struct Cpp {
    pub cpp: LlamaCpp,
    lanes: usize,
    /// Whether sequence 0 holds the null cell, at position `NULL_AT`.
    null: bool,
    /// The scoped sequences made; sequence `2 + i` is the `i`th.
    scopes: usize,
}

impl Cpp {
    pub fn open(path: &str, n_ctx: usize, lanes: usize, threads: usize) -> Cpp {
        let n_batch = 512.max((gbcformer::memory::dense::Records::CELL + 1) * lanes);
        Cpp { cpp: LlamaCpp::open(path, n_ctx, n_batch, lanes.max(SPLIT as usize + SCOPES) + 1, threads), lanes, null: false, scopes: 0 }
    }

    fn next(&self, seq: i32) -> usize {
        (self.cpp.seq_pos_max(seq) + 1) as usize
    }

    /// Every token is in sequence `cells` (0, or a scope) and in `SPLIT`, and llama.cpp
    /// masks a token's attention by the sequence it lists second (`shim_decode`). The
    /// tokens at `outs` and `also` list `cells` second and see its cells and the group. The rest
    /// list `SPLIT` second and see the group and the two cells `SPLIT` shares for the
    /// pass (stock llama.cpp requires coupled sequences to span the same positions).
    /// Flash attention skips each block of 8 query rows by 64 cells that is masked whole,
    /// so only those rows pay for the cells, and the batch gathers only the cells of
    /// `cells` (kv_rows). Only the rows at `outs` reach the LM head.
    fn split_in(&mut self, tokens: &[usize], outs: &[usize], also: &[usize], cells: i32) -> (Vec<Row>, At) {
        let y = self.next(0) as i32;
        assert_eq!(self.next(cells) as i32, y, "sequence {cells} ends at the top cell");
        assert_eq!(self.cpp.seq_pos_max(SPLIT), -1, "sequence {SPLIT} is empty between passes");
        let lo = self.cpp.seq_pos_min(cells);
        self.cpp.seq_cp(cells, SPLIT, lo, lo + 1);
        self.cpp.seq_cp(cells, SPLIT, y - 1, y);
        let mut want = vec![false; tokens.len()];
        for &o in outs {
            want[o] = true;
        }
        let mut attends = want.clone();
        for &o in also {
            attends[o] = true;
        }
        let batch: Vec<Tok> = tokens
            .iter()
            .zip(want.iter().zip(&attends))
            .map(|(&id, (&logits, &attends))| {
                let (seq, also) = if attends { (SPLIT, cells) } else { (cells, SPLIT) };
                Tok { id, pos: y as usize, seq, also, logits }
            })
            .collect();
        let rows = self.cpp.decode(&batch);
        self.cpp.seq_rm(-1, y, -1);
        self.cpp.seq_rm(SPLIT, -1, -1);
        (ordered(rows, outs), At { pos: y as usize })
    }
}

impl Runtime for Cpp {
    fn null(&mut self, token: usize) {
        assert_eq!(self.cpp.seq_pos_max(0), -1, "the null cell is the first cell");
        self.cpp.decode(&[Tok { id: token, pos: NULL_AT as usize, seq: 0, also: -1, logits: false }]);
        self.null = true;
    }

    fn pass(&mut self, tokens: &[usize], outs: &[usize]) -> (Vec<Row>, At) {
        let y = self.next(0);
        let mut want = vec![false; tokens.len()];
        for &o in outs {
            want[o] = true;
        }
        let batch: Vec<Tok> = tokens.iter().zip(&want).map(|(&id, &logits)| Tok { id, pos: y, seq: 0, also: -1, logits }).collect();
        let rows = self.cpp.decode(&batch);
        self.cpp.seq_rm(0, y as i32, -1);
        (ordered(rows, outs), At { pos: y })
    }

    fn split_pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize]) -> (Vec<Row>, At) {
        self.split_in(tokens, outs, also, 0)
    }

    /// Copies each run's cells from sequence 0; `write` keeps them.
    fn scope(&mut self, runs: &[(u32, u32)]) -> Option<Scoped> {
        assert!(self.scopes < SCOPES, "at most {SCOPES} scoped sequences");
        let seq = SPLIT + 1 + self.scopes as i32;
        assert_eq!(self.cpp.seq_pos_max(seq), -1, "sequence {seq} is empty before it is scoped");
        for &(a, e) in runs {
            self.cpp.seq_cp(0, seq, a as i32, e as i32);
        }
        if self.null {
            self.cpp.seq_cp(0, seq, NULL_AT, NULL_AT + 1);
        }
        assert_eq!(self.next(seq), self.next(0), "a scope holds the top cell");
        self.scopes += 1;
        Some(Scoped(seq))
    }

    fn rescope(&mut self, scope: Scoped, runs: &[(u32, u32)]) {
        self.cpp.seq_rm(scope.0, -1, -1);
        for &(a, e) in runs {
            self.cpp.seq_cp(0, scope.0, a as i32, e as i32);
        }
        if self.null {
            self.cpp.seq_cp(0, scope.0, NULL_AT, NULL_AT + 1);
        }
        assert_eq!(self.next(scope.0), self.next(0), "a scope holds the top cell");
    }

    fn scope_add(&mut self, scope: Scoped, n: u32) {
        self.cpp.seq_cp(0, scope.0, n as i32, n as i32 + 1);
    }

    fn scope_drop(&mut self, scope: Scoped, n: u32) {
        assert!(!self.null || n as i32 != NULL_AT, "line {n} shares its position with the null cell, which every scope keeps");
        self.cpp.seq_rm(scope.0, n as i32, n as i32 + 1);
    }

    fn scoped_pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize], scope: Scoped) -> (Vec<Row>, At) {
        self.split_in(tokens, outs, also, scope.0)
    }

    /// Saves the sequence to a file, as llama.cpp writes one, and reads the cells' positions
    /// from its head (`saved_positions`).
    fn scope_positions(&mut self, scope: Scoped) -> Option<Vec<i32>> {
        let path = std::env::temp_dir().join(format!("stock-llama-scope-{}", std::process::id()));
        let saved = self.cpp.seq_save(path.to_str()?, scope.0);
        let bytes = saved.then(|| std::fs::read(&path).ok()).flatten();
        std::fs::remove_file(&path).ok();
        saved_positions(&bytes?, scope.0)
    }

    fn fault_drop(&mut self, scope: Scoped, n: u32) {
        self.cpp.seq_rm(scope.0, n as i32, n as i32 + 1);
    }

    /// Evicts the line's cell from every sequence, so no scope keeps an old line; the new
    /// cell joins every scope in `join`.
    /// A scoped record's tokens attend the scope's cells. The new cell leaves the scope
    /// again unless `join` names the scope.
    fn write(&mut self, n: u32, rec: &[usize], evict: bool, scope: Option<Scoped>, join: &[Scoped]) -> At {
        if evict {
            assert!(!self.null || n as i32 != NULL_AT, "a write would evict the null cell with line {n}'s");
            self.cpp.seq_rm(-1, n as i32, n as i32 + 1);
        }
        let y = self.next(0);
        let cell = rec.len() - 1;
        let batch = |seq: i32, also: i32| -> Vec<Tok> { rec.iter().enumerate().map(|(i, &id)| Tok { id, pos: y + (i == cell) as usize, seq, also, logits: i == cell }).collect() };
        let (at, to) = (y as i32, n as i32);
        match scope {
            None => {
                self.cpp.decode(&batch(0, -1));
                self.cpp.seq_rm(-1, at, at + 1);
            }
            Some(s) => {
                // As in `split_in`: the record's tokens sit in the scope and in `SPLIT`,
                // which shares the scope's lowest cell and the top cell for the decode,
                // so the batch gathers the scope's cells alone. In sequence 0 it would
                // gather every cell and mask all but the scope's. The new cell then joins
                // sequence 0.
                assert_eq!(self.next(s.0), y, "sequence {} ends at the top cell", s.0);
                assert_eq!(self.cpp.seq_pos_max(SPLIT), -1, "sequence {SPLIT} is empty between decodes");
                let lo = self.cpp.seq_pos_min(s.0);
                self.cpp.seq_cp(s.0, SPLIT, lo, lo + 1);
                self.cpp.seq_cp(s.0, SPLIT, at - 1, at);
                self.cpp.decode(&batch(SPLIT, s.0));
                self.cpp.seq_rm(-1, at, at + 1);
                self.cpp.seq_cp(s.0, 0, at + 1, at + 2);
                self.cpp.seq_rm(SPLIT, -1, -1);
            }
        }
        self.cpp.seq_add(0, at + 1, at + 2, to - at - 1);
        if let Some(s) = scope {
            self.cpp.seq_rm(s.0, to, to + 1);
        }
        for s in join {
            self.cpp.seq_cp(0, s.0, n as i32, n as i32 + 1);
        }
        At { pos: y }
    }

    fn open_lanes(&mut self, lanes: usize) {
        assert_eq!(self.scopes, 0, "lanes open before any scope: they share sequence ids");
        assert!(lanes <= self.lanes, "{lanes} lanes, the context has {}", self.lanes);
        for j in 1..=lanes as i32 {
            self.cpp.seq_cp(0, j, -1, -1);
        }
    }

    fn close_lanes(&mut self, lanes: usize) {
        for j in 1..=lanes as i32 {
            self.cpp.seq_cp(j, 0, -1, -1);
            self.cpp.seq_rm(j, -1, -1);
        }
    }

    fn lanes(&self) -> usize {
        self.lanes
    }

    fn lanes_decode(&mut self, recs: &[(u32, Vec<usize>)]) {
        let y = self.next(1);
        let mut batch = Vec::with_capacity(recs.iter().map(|(_, r)| r.len()).sum());
        for (j, (_, rec)) in recs.iter().enumerate() {
            let seq = 1 + j as i32;
            assert_eq!(self.next(seq), y, "every lane holds the same top cell");
            let cell = rec.len() - 1;
            batch.extend(rec.iter().enumerate().map(|(i, &id)| Tok { id, pos: y + (i == cell) as usize, seq, also: -1, logits: false }));
        }
        self.cpp.decode(&batch);
    }

    fn lanes_place(&mut self, lines: &[u32]) {
        // Lane 1 holds a record of every batch: its scratch sits at y and its cell at y + 1.
        let y = self.next(1) as i32 - 2;
        assert_eq!(y, self.next(0) as i32, "the batch decoded one above the top cell");
        self.cpp.seq_rm(-1, y, y + 1);
        for (j, n) in lines.iter().enumerate() {
            self.cpp.seq_add(1 + j as i32, y + 1, y + 2, *n as i32 - y - 1);
        }
    }

    fn save_all(&mut self, path: &str) -> bool {
        self.cpp.state_save(path)
    }

    fn load_all(&mut self, path: &str) -> bool {
        self.cpp.state_load(path)
    }

    fn decode_nanos(&self) -> u128 {
        self.cpp.decode_ns
    }

    fn stream(&mut self, name: Option<&str>) -> bool {
        self.cpp.stream(name)
    }

    fn sliced(&self) -> Option<bool> {
        self.cpp.sliced()
    }

    fn fence(&self) -> Option<usize> {
        self.cpp.fence()
    }

    fn save(&mut self, path: &str) -> bool {
        self.cpp.seq_save(path, 0)
    }

    fn cell_bytes(&mut self, n: u32) -> Option<Vec<u8>> {
        assert_eq!(self.cpp.seq_pos_max(SPLIT), -1, "sequence {SPLIT} is empty between passes");
        let path = std::env::temp_dir().join(format!("stock-llama-cell-{}", std::process::id()));
        self.cpp.seq_cp(0, SPLIT, n as i32, n as i32 + 1);
        let saved = self.cpp.seq_save(path.to_str()?, SPLIT);
        self.cpp.seq_rm(SPLIT, -1, -1);
        let bytes = saved.then(|| std::fs::read(&path).ok()).flatten();
        std::fs::remove_file(&path).ok();
        bytes
    }

    fn load(&mut self, path: &str) -> bool {
        self.cpp.seq_load(path, 0)
    }
}

/// A runtime for `model`: llama.cpp over a GGUF, with room for `n_ctx` cells and `lanes`
/// prefill lanes.
pub fn open(model: &str, cfg: &gbcformer::llama::Config, n_ctx: usize, lanes: usize, threads: usize) -> Box<dyn Runtime> {
    let rt = Cpp::open(model, n_ctx, lanes, threads);
    assert_eq!(rt.cpp.vocab, cfg.vocab, "the GGUF's vocabulary is the checkpoint's");
    announce(model);
    Box::new(rt)
}

/// Says which llama.cpp the process loaded and what the model file was written for.
/// `STOCK_PURE=1` refuses anything but unpatched llama.cpp over a file written for it.
fn announce(model: &str) {
    use crate::cpp::Build;
    let (lib, build) = crate::cpp::build();
    let file = gbcformer::wiring::Manifest::read(std::path::Path::new(model)).map_or_else(|e| format!("no manifest ({e})"), |(m, _)| m.llama_cpp);
    match &build {
        Build::Unpatched => eprintln!("llama.cpp unpatched; whole model every pass ({lib} exports none of the functions and holds none of the GGUF keys this tree's patches add)"),
        Build::Patched => eprintln!("llama.cpp patched with this tree's stream-slices.patch and stored-runs.patch ({lib})"),
        Build::Partly(what) => eprintln!("llama.cpp PARTLY patched ({lib}): {what}"),
    }
    eprintln!("model file written for: {file}");
    if std::env::var("STOCK_PURE").as_deref() == Ok("1") && (build != Build::Unpatched || !file.contains("unpatched")) {
        eprintln!("STOCK_PURE=1: refusing to run: the claim is unpatched llama.cpp over the uniform file");
        crate::quit(1);
    }
}

/// What a test makes `Replay`'s next written cell save (`Replay::fault`).
pub mod fault {
    /// Other bytes than its record's.
    pub const CORRUPT: u8 = 1;
    /// A zero entry of the other sign, which the next shift of another cell clears.
    pub const SIGN: u8 = 2;
    /// A zero entry of the other sign that no shift clears.
    pub const SIGN_KEPT: u8 = 3;
}

/// A cell of `Replay`: its write record, whether a later decode has shifted it, and its
/// fault.
struct Kept {
    rec: Vec<usize>,
    shifted: bool,
    fault: u8,
}

/// A runtime without a model, for `play replay` and the cache's tests. A pass decodes the
/// tokens `answer` last named (the engine's record of the pass), so nothing it returns is
/// a model's result. It keeps each cell's write record and shifts a new cell at the next
/// decode, as llama.cpp does, so a cell's saved bytes differ before and after its shift.
#[derive(Default)]
pub struct Replay {
    cells: std::collections::HashMap<u32, Kept>,
    fresh: Option<u32>,
    answer: Vec<usize>,
    scopes: i32,
    /// The fault of the next written cell (`fault`), then 0 again.
    pub fault: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

impl Replay {
    /// A decode: the cell the last one wrote takes its shift, and that shift clears every
    /// other cell's `fault::SIGN`.
    fn shift(&mut self) {
        if let Some(n) = self.fresh.take() {
            for (&m, c) in self.cells.iter_mut() {
                c.shifted = true;
                if m != n && c.fault == fault::SIGN {
                    c.fault = 0;
                }
            }
        }
    }
}

impl Runtime for Replay {
    fn pass(&mut self, _tokens: &[usize], outs: &[usize]) -> (Vec<Row>, At) {
        assert_eq!(outs.len(), self.answer.len(), "a replayed pass decodes the positions the record's pass decoded");
        self.shift();
        (self.answer.iter().map(|&t| Row { first: t, v: vec![f32::MAX] }).collect(), At { pos: gbcformer::memory::attn::READ_AT })
    }

    fn answer(&mut self, out: &[usize]) {
        self.answer = out.to_vec();
    }

    fn replays(&self) -> bool {
        true
    }

    fn scope(&mut self, _runs: &[(u32, u32)]) -> Option<Scoped> {
        self.scopes += 1;
        Some(Scoped(SPLIT + self.scopes))
    }

    fn rescope(&mut self, _scope: Scoped, _runs: &[(u32, u32)]) {}

    fn scope_add(&mut self, _scope: Scoped, _n: u32) {}

    fn scope_drop(&mut self, _scope: Scoped, _n: u32) {}

    fn write(&mut self, n: u32, rec: &[usize], _evict: bool, _scope: Option<Scoped>, _join: &[Scoped]) -> At {
        self.shift();
        let fault = self.fault.swap(0, std::sync::atomic::Ordering::Relaxed);
        self.cells.insert(n, Kept { rec: rec.to_vec(), shifted: false, fault });
        self.fresh = Some(n);
        At { pos: gbcformer::memory::attn::READ_AT }
    }

    /// Laid out as llama.cpp saves a sequence of one cell and one layer: a header (whether
    /// the cell has its shift), the counts, an f16 key row (two entries, the first a zero
    /// whose sign is the fault's) and an f16 value row (the record's line tokens: a cell
    /// is the same whichever byte of its line the record's address names).
    fn cell_bytes(&mut self, n: u32) -> Option<Vec<u8>> {
        let c = self.cells.get(&n)?;
        let sign = if matches!(c.fault, fault::SIGN | fault::SIGN_KEPT) { 0x80 } else { 0 };
        let value: Vec<u8> = c.rec[7..].iter().flat_map(|&t| (t as u32 ^ (c.fault == fault::CORRUPT) as u32).to_le_bytes()).collect();
        let mut bytes = Vec::new();
        for word in [c.shifted as u32, 0, 0, 1, 0, 0] {
            bytes.extend(word.to_le_bytes());
        }
        for row in [&[0, sign, 0, 0][..], &value] {
            bytes.extend(1u32.to_le_bytes());
            bytes.extend((row.len() as u64).to_le_bytes());
            bytes.extend(row);
        }
        Some(bytes)
    }

    fn lanes_decode(&mut self, recs: &[(u32, Vec<usize>)]) {
        for (n, rec) in recs {
            self.cells.insert(*n, Kept { rec: rec.clone(), shifted: true, fault: 0 });
        }
    }

    fn lanes_place(&mut self, _lines: &[u32]) {}

    fn open_lanes(&mut self, _lanes: usize) {}

    fn close_lanes(&mut self, _lanes: usize) {}

    fn lanes(&self) -> usize {
        0
    }

    fn null(&mut self, _token: usize) {}

    fn decode_nanos(&self) -> u128 {
        0
    }
}

/// The runtime of a resumed run (`docs/stock-llama.md` §14). Up to the checkpoint's event
/// it is `Replay`: no model runs, and each pass decodes the tokens of the engine's
/// record. `go_live` then loads the checkpoint's cache into the real runtime, copies the
/// scoped sequences the replay asked for from it, and every later call is the real
/// runtime's. A scoped sequence is a set of sequence 0's cells by position (`Cpp::scope`,
/// `Cpp::write`), and `Cache` holds each one's lines, so copying those lines' cells from
/// the loaded cache gives the set the straight run held.
pub struct Resuming {
    replay: Replay,
    real: Box<dyn Runtime>,
    live: bool,
}

impl Resuming {
    pub fn new(real: Box<dyn Runtime>) -> Resuming {
        Resuming { replay: Replay::default(), real, live: false }
    }

    fn rt(&mut self) -> &mut dyn Runtime {
        if self.live { self.real.as_mut() } else { &mut self.replay }
    }
}

impl Runtime for Resuming {
    fn pass(&mut self, tokens: &[usize], outs: &[usize]) -> (Vec<Row>, At) {
        self.rt().pass(tokens, outs)
    }

    fn split_pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize]) -> (Vec<Row>, At) {
        self.rt().split_pass(tokens, outs, also)
    }

    fn scope(&mut self, runs: &[(u32, u32)]) -> Option<Scoped> {
        self.rt().scope(runs)
    }

    fn scoped_pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize], scope: Scoped) -> (Vec<Row>, At) {
        self.rt().scoped_pass(tokens, outs, also, scope)
    }

    fn rescope(&mut self, scope: Scoped, runs: &[(u32, u32)]) {
        self.rt().rescope(scope, runs);
    }

    fn scope_add(&mut self, scope: Scoped, n: u32) {
        self.rt().scope_add(scope, n);
    }

    fn scope_drop(&mut self, scope: Scoped, n: u32) {
        self.rt().scope_drop(scope, n);
    }

    fn scope_positions(&mut self, scope: Scoped) -> Option<Vec<i32>> {
        self.rt().scope_positions(scope)
    }

    fn fault_drop(&mut self, scope: Scoped, n: u32) {
        self.rt().fault_drop(scope, n)
    }

    fn write(&mut self, n: u32, rec: &[usize], evict: bool, scope: Option<Scoped>, join: &[Scoped]) -> At {
        self.rt().write(n, rec, evict, scope, join)
    }

    fn cell_bytes(&mut self, n: u32) -> Option<Vec<u8>> {
        self.rt().cell_bytes(n)
    }

    fn answer(&mut self, out: &[usize]) {
        self.rt().answer(out)
    }

    fn replays(&self) -> bool {
        !self.live
    }

    fn lanes_decode(&mut self, recs: &[(u32, Vec<usize>)]) {
        self.rt().lanes_decode(recs)
    }

    fn lanes_place(&mut self, lines: &[u32]) {
        self.rt().lanes_place(lines)
    }

    fn open_lanes(&mut self, lanes: usize) {
        self.rt().open_lanes(lanes)
    }

    fn close_lanes(&mut self, lanes: usize) {
        self.rt().close_lanes(lanes)
    }

    fn lanes(&self) -> usize {
        if self.live { self.real.lanes() } else { self.replay.lanes() }
    }

    /// The real runtime decodes its null cell at once: the cache it loads later replaces
    /// sequence 0 and holds that cell too.
    fn null(&mut self, token: usize) {
        self.real.null(token);
    }

    /// Nothing is saved or loaded while it replays: the prefill's cells are the replay's.
    fn save(&mut self, path: &str) -> bool {
        self.live && self.real.save(path)
    }

    fn load(&mut self, path: &str) -> bool {
        self.live && self.real.load(path)
    }

    fn decode_nanos(&self) -> u128 {
        self.real.decode_nanos()
    }

    fn stream(&mut self, name: Option<&str>) -> bool {
        self.live && self.real.stream(name)
    }

    fn sliced(&self) -> Option<bool> {
        if self.live { self.real.sliced() } else { None }
    }

    fn fence(&self) -> Option<usize> {
        self.real.fence()
    }

    fn can_stream(&mut self, name: &str) -> bool {
        self.real.can_stream(name)
    }

    fn go_live(&mut self, state: &str, scopes: &[(Scoped, Vec<(u32, u32)>)]) -> bool {
        assert!(!self.live, "a run goes live once");
        if !self.real.load(state) {
            return false;
        }
        for (s, runs) in scopes {
            assert_eq!(self.real.scope(runs), Some(*s), "the runtime numbers the scoped sequences as the replay did");
        }
        self.live = true;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::saved_positions;

    fn file(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A saved sequence's head: magic, version, no tokens, one stream, the cells (each its
    /// position, one sequence listed, that sequence), then the rows' header.
    #[test]
    fn a_saved_sequence_gives_its_cells_positions_sorted() {
        let head = [0x6767_7371, 2, 0, 1];
        let saved = file(&[&head[..], &[3, 0x108_7FF, 1, 4, 0, 1, 4, 0x20, 1, 4, 0, 25, 1, 128], &[0xAAAA; 40]].concat());
        assert_eq!(saved_positions(&saved, 4), Some(vec![0, 0x20, 0x108_7FF]));
        assert_eq!(saved_positions(&file(&[&head[..], &[0]].concat()), 4), Some(vec![]), "an empty sequence");
    }

    #[test]
    fn a_file_that_is_not_one_sequence_s_cells_gives_no_positions() {
        let cells = [2, 7, 1, 4, 9, 1, 4, 0, 25];
        let with = |head: [u32; 4], cells: &[u32]| file(&[&head[..], cells].concat());
        assert_eq!(saved_positions(&with([1, 2, 0, 1], &cells), 4), Some(vec![7, 9]));
        assert_eq!(saved_positions(&with([1, 2, 0, 1], &cells), 5), None, "cells of another sequence");
        assert_eq!(saved_positions(&with([1, 2, 3, 1], &cells), 4), None, "a file with tokens");
        assert_eq!(saved_positions(&with([1, 2, 0, 2], &cells), 4), None, "two streams");
        assert_eq!(saved_positions(&with([1, 2, 0, 1], &[2, 7, 2, 4, 9, 1, 4, 0, 25]), 4), None, "a cell in two sequences");
        assert_eq!(saved_positions(&with([1, 2, 0, 1], &[2, 7, 1, 4, 9, 1, 4, 5, 25]), 4), None, "no rows' header after the cells");
        assert_eq!(saved_positions(&with([1, 2, 0, 1], &[3, 7, 1, 4, 9, 1, 4]), 4), None, "fewer cells than counted");
    }
}
