//! llama.cpp through `shim.c`. It loads a GGUF, decodes token batches at given positions
//! and sequences, reads logit rows, and edits the cache with the `llama_memory_seq_*`
//! calls.

use std::ffi::{c_char, c_int, CString};

#[repr(C)]
struct Shim {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn shim_open(path: *const c_char, n_ctx: c_int, n_batch: c_int, n_seq_max: c_int, n_threads: c_int, verbose: c_int) -> *mut Shim;
    fn shim_n_vocab(s: *mut Shim) -> c_int;
    fn shim_decode(s: *mut Shim, n: c_int, tokens: *const c_int, pos: *const c_int, seq: *const c_int, also: *const c_int, logits: *const i8) -> c_int;
    fn shim_logits(s: *mut Shim, i: c_int) -> *const f32;
    fn shim_logits_cols(s: *mut Shim, first: *mut c_int) -> c_int;
    fn shim_stream(name: *const c_char) -> c_int;
    fn shim_fence() -> c_int;
    fn shim_sliced(s: *mut Shim) -> c_int;
    fn shim_patch_symbols() -> c_int;
    fn shim_lib() -> *const c_char;
    fn shim_seq_rm(s: *mut Shim, seq: c_int, p0: c_int, p1: c_int) -> c_int;
    fn shim_seq_add(s: *mut Shim, seq: c_int, p0: c_int, p1: c_int, delta: c_int);
    fn shim_seq_cp(s: *mut Shim, src: c_int, dst: c_int, p0: c_int, p1: c_int);
    fn shim_seq_pos_max(s: *mut Shim, seq: c_int) -> c_int;
    fn shim_seq_pos_min(s: *mut Shim, seq: c_int) -> c_int;
    fn shim_seq_save(s: *mut Shim, path: *const c_char, seq: c_int) -> c_int;
    fn shim_seq_load(s: *mut Shim, path: *const c_char, seq: c_int) -> c_int;
    fn shim_state_save(s: *mut Shim, path: *const c_char) -> c_int;
    fn shim_state_load(s: *mut Shim, path: *const c_char) -> c_int;
    fn shim_close(s: *mut Shim);
}

/// One token of a batch: its id, position, sequence, and whether its logit row is read.
#[derive(Clone, Copy, Debug)]
pub struct Tok {
    pub id: usize,
    pub pos: usize,
    pub seq: i32,
    /// A second sequence the token also belongs to, or -1. Its attention follows `also`
    /// when set (`shim_decode`).
    pub also: i32,
    pub logits: bool,
}

/// One decoded position's scores: token `first + i` scores `v[i]`. A runtime that scores
/// only a run of the LM head's rows (`shim_logits_cols`) gives rows narrower than the
/// vocabulary.
#[derive(Clone, Debug)]
pub struct Row {
    pub first: usize,
    pub v: Vec<f32>,
}

impl Row {
    /// The token the whole vocabulary's scores would choose, the first with the largest
    /// score. A row over a run of the head leaves out rows that all score `head.fill`
    /// (`gguf::HeadRows`), so its own choice is the vocabulary's when its largest score is
    /// above that; anything else stops the run, since the choice would need the rows left out.
    pub fn token(&self, head: &gbcformer::gguf::HeadRows, vocab: usize) -> usize {
        let i = argmax(&self.v);
        assert!(self.v.len() == vocab || head.decides(vocab, self.v[i]), "a row over tokens {}..{} has its largest score {} at or below the {} of every row left out", self.first, self.first + self.v.len(), self.v[i], head.fill);
        self.first + i
    }

    /// The first token with the largest score among the row's own, unchecked (for messages).
    pub fn best(&self) -> usize {
        self.first + argmax(&self.v)
    }

    /// Whether both rows score the same tokens with the same bits.
    pub fn same(&self, other: &Row) -> bool {
        self.first == other.first && self.v.len() == other.v.len() && self.v.iter().zip(&other.v).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Whether the two rows agree where one scores a run of the other's tokens, as a
    /// stream's slice and the whole model do, or a group above the patch's fence and one
    /// below it.
    pub fn agrees(&self, other: &Row) -> bool {
        self.within(other) || other.within(self)
    }

    /// Whether this row is `wide` on a run of its tokens: the same bits on every token
    /// it scores, and `wide`'s first largest score among them.
    pub fn within(&self, wide: &Row) -> bool {
        let Some(at) = self.first.checked_sub(wide.first) else { return false };
        at + self.v.len() <= wide.v.len() && self.v.iter().zip(&wide.v[at..]).all(|(x, y)| x.to_bits() == y.to_bits()) && wide.best() == self.best()
    }
}

/// The functions this tree's patches add to llama.cpp, in `shim_patch_symbols`' bit order.
const PATCH_SYMBOLS: [&str; 4] = ["llama_gbcformer_set_stream", "llama_gbcformer_fence", "llama_get_logits_stream", "llama_get_logits_cols"];
/// The GGUF keys the patched loader reads, whose text a patched libllama holds.
const PATCH_KEYS: [&str; 5] = ["gbcformer.stream.", "gbcformer.output.rows", "gbcformer.token_embd.cols", "gbcformer.output.cols", "gbcformer.ffn_up.cols"];

/// What the libllama this process loaded is, by the marks `stream-slices.patch` and
/// `stored-runs.patch` leave in it. `layer-shapes.patch` adds neither a function nor a
/// key, so it leaves no mark: only the tree a build came from shows its absence, and a
/// uniform GGUF never reaches its code.
#[derive(Debug, PartialEq, Eq)]
pub enum Build {
    /// None of `PATCH_SYMBOLS` resolves and the file holds none of `PATCH_KEYS`.
    Unpatched,
    /// Every one of them is there.
    Patched,
    /// Some are, or the file could not be read: the text says which.
    Partly(String),
}

/// The file the process loaded llama.cpp from, and what it is.
pub fn build() -> (String, Build) {
    let lib = unsafe { std::ffi::CStr::from_ptr(shim_lib()) }.to_string_lossy().into_owned();
    let found = unsafe { shim_patch_symbols() };
    let symbols: Vec<&str> = PATCH_SYMBOLS.iter().enumerate().filter(|(i, _)| found >> i & 1 == 1).map(|(_, n)| *n).collect();
    let Ok(bytes) = std::fs::read(&lib) else {
        return (lib, Build::Partly(format!("the library file could not be read; functions found: {symbols:?}")));
    };
    let keys: Vec<&str> = PATCH_KEYS.into_iter().filter(|k| bytes.windows(k.len()).any(|w| w == k.as_bytes())).collect();
    let build = match (symbols.len(), keys.len()) {
        (0, 0) => Build::Unpatched,
        (4, 5) => Build::Patched,
        _ => Build::Partly(format!("functions found: {symbols:?}; keys found: {keys:?}")),
    };
    (lib, build)
}

/// `STOCK_ROWHASH=<file>` writes one line per logit row the runtime returns, in decode
/// order: the first token it scores, how many, and a hash of the scores' bits.
/// `STOCK_ROWHASH_CHECK=<file>` holds this run's rows to such a file from another run of
/// the same passes: each row hashed over the tokens the line names, which a wider row
/// holds as a run of its own (the whole model against a stream's run of the head). The
/// first row that differs, or that does not hold the line's tokens, stops the run.
enum RowLog {
    Off,
    Write(std::fs::File),
    Check { lines: std::iter::Peekable<std::io::Lines<std::io::BufReader<std::fs::File>>>, path: String, equal: u64, ended: bool },
}

impl RowLog {
    fn from_env() -> RowLog {
        let open = |p: &str| std::fs::File::open(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        match (std::env::var("STOCK_ROWHASH"), std::env::var("STOCK_ROWHASH_CHECK")) {
            (Ok(_), Ok(_)) => panic!("STOCK_ROWHASH writes a file and STOCK_ROWHASH_CHECK reads one: set one of them"),
            (Ok(p), _) => RowLog::Write(std::fs::File::create(&p).unwrap_or_else(|e| panic!("{p}: {e}"))),
            (_, Ok(p)) => RowLog::Check { lines: std::io::BufRead::lines(std::io::BufReader::new(open(&p))).peekable(), path: p, equal: 0, ended: false },
            _ => RowLog::Off,
        }
    }

    fn hash(v: &[f32]) -> u64 {
        v.iter().flat_map(|x| x.to_bits().to_le_bytes()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
    }

    fn rows(&mut self, rows: &[Row]) {
        match self {
            RowLog::Off => {}
            RowLog::Write(file) => {
                let text: String = rows.iter().map(|r| format!("{} {} {:016x}\n", r.first, r.v.len(), RowLog::hash(&r.v))).collect();
                std::io::Write::write_all(file, text.as_bytes()).expect("the row hashes are written");
            }
            RowLog::Check { lines, path, equal, ended } => {
                for r in rows {
                    let Some(line) = lines.next() else {
                        if !*ended {
                            eprintln!("ROW HASH ENDS: {path} holds no line for this run's row {}; the rows after it are not compared", *equal + 1);
                            *ended = true;
                        }
                        return;
                    };
                    let line = line.expect("the row hashes are read");
                    let mut words = line.split(' ');
                    let (first, len): (usize, usize) = (words.next().unwrap().parse().unwrap(), words.next().unwrap().parse().unwrap());
                    let want = words.next().unwrap();
                    let got = first.checked_sub(r.first).filter(|at| at + len <= r.v.len()).map(|at| format!("{:016x}", RowLog::hash(&r.v[at..at + len])));
                    if got.as_deref() != Some(want) {
                        eprintln!("ROW HASH DIFFERS at row {}: {path} scores tokens {first}..{} with hash {want}; this run scores {}..{} and hashes that run to {}", *equal + 1, first + len, r.first, r.first + r.v.len(), got.unwrap_or_else(|| "nothing, since it does not hold it".to_string()));
                        crate::quit(1);
                    }
                    *equal += 1;
                    if lines.peek().is_none() {
                        eprintln!("row hashes: all {equal} rows of {path} equal this run's, bit for bit on the tokens both score");
                    } else if *equal % 100_000 == 0 {
                        eprintln!("row hashes: {equal} rows equal {path}'s so far");
                    }
                }
            }
        }
    }
}

pub struct LlamaCpp {
    s: *mut Shim,
    pub vocab: usize,
    /// Wall time spent inside llama_decode, summed.
    pub decode_ns: u128,
    rows: RowLog,
}

// Only one thread uses the context at a time.
unsafe impl Send for LlamaCpp {}

impl LlamaCpp {
    pub fn open(path: &str, n_ctx: usize, n_batch: usize, n_seq_max: usize, n_threads: usize) -> LlamaCpp {
        let p = CString::new(path).unwrap();
        let verbose = std::env::var_os("STOCK_VERBOSE").is_some() as c_int;
        let s = unsafe { shim_open(p.as_ptr(), n_ctx as c_int, n_batch as c_int, n_seq_max as c_int, n_threads as c_int, verbose) };
        assert!(!s.is_null(), "llama.cpp could not load {path} with a context of {n_ctx} cells");
        let vocab = unsafe { shim_n_vocab(s) as usize };
        LlamaCpp { s, vocab, decode_ns: 0, rows: RowLog::from_env() }
    }

    /// Decodes the batch; returns the logit rows asked for, in batch order.
    pub fn decode(&mut self, batch: &[Tok]) -> Vec<Row> {
        let ids: Vec<c_int> = batch.iter().map(|t| t.id as c_int).collect();
        let pos: Vec<c_int> = batch.iter().map(|t| t.pos as c_int).collect();
        let seq: Vec<c_int> = batch.iter().map(|t| t.seq).collect();
        let also: Vec<c_int> = batch.iter().map(|t| t.also).collect();
        let l: Vec<i8> = batch.iter().map(|t| t.logits as i8).collect();
        let t = std::time::Instant::now();
        let rc = unsafe { shim_decode(self.s, batch.len() as c_int, ids.as_ptr(), pos.as_ptr(), seq.as_ptr(), also.as_ptr(), l.as_ptr()) };
        assert_eq!(rc, 0, "llama_decode returned {rc} for a batch of {} tokens", batch.len());
        let mut first: c_int = 0;
        let width = unsafe { shim_logits_cols(self.s, &mut first) } as usize;
        assert!(width > 0 && first as usize + width <= self.vocab, "logit rows over tokens {first}..{} of {}", first as usize + width, self.vocab);
        let rows: Vec<Row> = batch
            .iter()
            .enumerate()
            .filter(|(_, t)| t.logits)
            .map(|(i, _)| {
                let p = unsafe { shim_logits(self.s, i as c_int) };
                assert!(!p.is_null(), "no logits for batch row {i}");
                Row { first: first as usize, v: unsafe { std::slice::from_raw_parts(p, width) }.to_vec() }
            })
            .collect();
        self.decode_ns += t.elapsed().as_nanos();
        self.rows.rows(&rows);
        rows
    }

    /// Names the stream the next decodes run (`shim_stream`), or the whole model with
    /// None. True when the runtime runs that stream's slice.
    pub fn stream(&mut self, name: Option<&str>) -> bool {
        let n = name.map(|n| CString::new(n).unwrap());
        unsafe { shim_stream(n.as_ref().map_or(std::ptr::null(), |n| n.as_ptr())) != 0 }
    }

    /// The most tokens of a batch the runtime runs on the whole model whatever stream is
    /// named (`shim_fence`), or None from a runtime that does not say.
    pub fn fence(&self) -> Option<usize> {
        usize::try_from(unsafe { shim_fence() }).ok()
    }

    /// Whether the last decode's logit rows ran a stream's slice, as the runtime reports
    /// it (`shim_sliced`), or None from a runtime that cannot say.
    pub fn sliced(&self) -> Option<bool> {
        match unsafe { shim_sliced(self.s) } {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    pub fn seq_rm(&mut self, seq: i32, p0: i32, p1: i32) {
        assert!(unsafe { shim_seq_rm(self.s, seq, p0, p1) } != 0, "seq_rm({seq}, {p0}, {p1}) refused");
    }

    pub fn seq_add(&mut self, seq: i32, p0: i32, p1: i32, delta: i32) {
        unsafe { shim_seq_add(self.s, seq, p0, p1, delta) }
    }

    pub fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) {
        unsafe { shim_seq_cp(self.s, src, dst, p0, p1) }
    }

    pub fn seq_pos_max(&self, seq: i32) -> i32 {
        unsafe { shim_seq_pos_max(self.s, seq) }
    }

    pub fn seq_pos_min(&self, seq: i32) -> i32 {
        unsafe { shim_seq_pos_min(self.s, seq) }
    }

    /// Saves sequence `seq` (positions, keys and values) to `path`.
    pub fn seq_save(&mut self, path: &str, seq: i32) -> bool {
        let p = CString::new(path).unwrap();
        unsafe { shim_seq_save(self.s, p.as_ptr(), seq) != 0 }
    }

    /// Saves every sequence of the context: each cell with the sequences it is in.
    pub fn state_save(&mut self, path: &str) -> bool {
        let p = CString::new(path).unwrap();
        unsafe { shim_state_save(self.s, p.as_ptr()) != 0 }
    }

    /// Loads what `state_save` wrote, in place of every cell the context holds.
    pub fn state_load(&mut self, path: &str) -> bool {
        let p = CString::new(path).unwrap();
        unsafe { shim_state_load(self.s, p.as_ptr()) != 0 }
    }

    /// Loads a sequence `seq_save` wrote.
    pub fn seq_load(&mut self, path: &str, seq: i32) -> bool {
        let p = CString::new(path).unwrap();
        unsafe { shim_seq_load(self.s, p.as_ptr(), seq) != 0 }
    }
}

impl Drop for LlamaCpp {
    fn drop(&mut self) {
        unsafe { shim_close(self.s) }
    }
}

/// The first index of the largest value, as `llama::argmax` picks: 0 when the first value
/// is a NaN, since nothing compares greater than one, and otherwise the first index of the
/// largest value that is not a NaN. The largest comes from 16 independent running maxima,
/// which the compiler turns into vector code: a GBC CPU pass decodes about 5 million
/// logits, and one comparison at a time took a third of a run's wall time.
pub fn argmax(row: &[f32]) -> usize {
    if row.first().is_none_or(|x| x.is_nan()) {
        return 0;
    }
    let mut lanes = [f32::NEG_INFINITY; 16];
    let chunks = row.chunks_exact(16);
    let rest = chunks.remainder();
    for c in chunks {
        for (l, &x) in lanes.iter_mut().zip(c) {
            *l = l.max(x);
        }
    }
    let max = lanes.iter().chain(rest).copied().fold(f32::NEG_INFINITY, f32::max);
    // The first entry equal to the largest: the first block that holds one, then its place
    // in the block. A row's choice sits near its end (the byte tokens follow every input
    // token), and a block without it is one vector comparison.
    const BLOCK: usize = 256;
    let block = row.chunks(BLOCK).position(|c| c.iter().fold(false, |any, &x| any | (x == max)));
    block.map_or(0, |b| b * BLOCK + row[b * BLOCK..].iter().position(|&x| x == max).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::{argmax, Row};
    use gbcformer::gguf::HeadRows;

    #[test]
    fn a_row_over_a_run_decides_only_above_the_fill() {
        let head = HeadRows { first: 10, count: 3, fill: -1024.0 };
        let row = |v: &[f32]| Row { first: 10, v: v.to_vec() };
        assert_eq!(row(&[-5.0, 7.0, 7.0]).token(&head, 20), 11);
        assert_eq!(row(&[-1023.0, -2000.0, -3000.0]).token(&head, 20), 10);
        let all_below = std::panic::catch_unwind(|| row(&[-1024.0, -2000.0, -3000.0]).token(&head, 20));
        assert!(all_below.is_err(), "a row whose best is the fill's score needs the rows left out");
    }

    #[test]
    fn a_narrow_row_is_within_the_wide_row_it_is_a_run_of() {
        let wide = Row { first: 4, v: vec![-1.0, 3.0, 9.0, 9.0, 2.0] };
        assert!(Row { first: 5, v: vec![3.0, 9.0, 9.0] }.within(&wide));
        assert!(Row { first: 4, v: wide.v.clone() }.within(&wide));
        // another score, the wide row's choice outside the run, and a run past its end
        assert!(!Row { first: 5, v: vec![3.0, 9.0, 8.0] }.within(&wide));
        assert!(!Row { first: 7, v: vec![9.0, 2.0] }.within(&Row { first: 4, v: vec![-1.0, 3.0, 9.5, 9.0, 2.0] }));
        assert!(!Row { first: 7, v: vec![9.0, 2.0, 0.0] }.within(&wide));
        assert!(!Row { first: 3, v: vec![0.0, -1.0] }.within(&wide));
        let narrow = Row { first: 5, v: vec![3.0, 9.0, 9.0] };
        assert!(narrow.agrees(&wide) && wide.agrees(&narrow));
        assert!(!Row { first: 5, v: vec![3.0, 9.0, 8.0] }.agrees(&wide));
    }

    fn one_at_a_time(row: &[f32]) -> usize {
        (0..row.len()).fold(0, |b, t| if row[t] > row[b] { t } else { b })
    }

    #[test]
    fn argmax_picks_what_one_comparison_at_a_time_picks() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let special = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 7.0, -7.0];
        for case in 0..20_000 {
            let n = (next() % if case % 16 == 0 { 1500 } else { 70 }) as usize;
            let row: Vec<f32> = (0..n)
                .map(|_| match next() % 4 {
                    0 => special[(next() % special.len() as u64) as usize],
                    _ if case % 2 == 0 => (next() % 5) as f32 - 2.0,
                    _ => (next() % 2001) as f32 - 1000.0,
                })
                .collect();
            assert_eq!(argmax(&row), one_at_a_time(&row), "{row:?}");
        }
    }
    /// The largest value planted past the first block, and ties across a block's edge:
    /// the choice is the first index of the largest, wherever the blocks fall.
    #[test]
    fn argmax_finds_a_maximum_in_a_later_block_and_the_first_of_a_tie_across_blocks() {
        const BLOCK: usize = 256;
        for n in [BLOCK + 1, 2 * BLOCK, 2 * BLOCK + 17, 5 * BLOCK + 3, 70_000] {
            for at in [BLOCK, BLOCK + 1, n / 2, n - 2, n - 1].into_iter().filter(|&at| at < n) {
                let mut row = vec![-3.0f32; n];
                row[at] = 9.0;
                assert_eq!(argmax(&row), at, "a row of {n} with its maximum planted at {at}");
                assert_eq!(one_at_a_time(&row), at);
            }
            // A tie whose two entries sit either side of each block edge, then two blocks apart.
            for edge in (BLOCK..n).step_by(BLOCK).take(4) {
                for (first, second) in [(edge - 1, edge), (edge - 1, (edge + BLOCK).min(n - 1)), (edge, n - 1)] {
                    let mut row = vec![-3.0f32; n];
                    (row[first], row[second]) = (9.0, 9.0);
                    assert_eq!(argmax(&row), first, "a row of {n} with a tie at {first} and {second}");
                    assert_eq!(one_at_a_time(&row), first);
                }
            }
            // The maximum in a later block with a NaN and a −0 before it, and a row of zeros of both signs.
            let mut row = vec![-0.0f32; n];
            let late = (BLOCK + 5).min(n - 1);
            (row[3], row[late]) = (f32::NAN, 0.5);
            assert_eq!(argmax(&row), late);
            let mut zeros = vec![-0.0f32; n];
            zeros[BLOCK] = 0.0;
            assert_eq!(argmax(&zeros), one_at_a_time(&zeros), "+0 equals −0: the first entry is the choice");
            assert_eq!(argmax(&zeros), 0);
        }
    }
}
