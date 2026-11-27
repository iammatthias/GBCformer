//! The machine's memory as cells of one runtime sequence (`docs/stock-llama.md` §4), one
//! cell per 8-byte line, shared by every stream, and a mirror of the line each cell
//! holds. Only a write record the runtime decodes writes the mirror: an engine's (its line
//! comes from the step's decoded tokens) or a loan's (`Cache::lend`, `Cache::put_back`: a
//! following stream's own line, then the cell's line again). No pass's tokens read it. The
//! harness holds it to the engines' indexes as a check, and `play` reads it to find the
//! cells that hold a newer line than a following stream's own.

use crate::cpp::Row;
use crate::runtime::{Runtime, Scoped};
use gbcformer::llama::{tok, Config, Fam};
use gbcformer::memory::attn::READ_AT;
use gbcformer::memory::dense::Records;
use gbcformer::memory::layout::{self, BLOCK, FF, HOST, IO, IO_BLOCKS, LINE, LINE_BITS, OAM, OAM_LEN, PAL, PAL_LEN, REGS, TAB, TAB_BLOCKS, TOTAL, VOID, VRAM, VRAM_LEN};
use std::collections::BTreeSet;
use std::io::Write;
use std::time::Instant;

/// The position every group and every record's scratch tokens sit at: one above the last
/// line's cell.
pub const P: usize = READ_AT;

/// A line's cell: its index (the cell's position and key) and its bytes.
pub type Cell = (u32, [u8; LINE]);

#[derive(Default, Clone, Debug)]
pub struct CacheStats {
    pub cells: usize,
    pub prefill_s: f64,
    pub prefill_sequential: usize,
    pub prefill_lanes: usize,
    pub passes: u64,
    pub writes: u64,
    pub reads: u64,
    /// Decodes whose group sat elsewhere than P (the layout's last line evicted).
    pub off_p: u64,
    pub pass_ns: u128,
    pub write_ns: u128,
    pub read_ns: u128,
    /// Passes whose decode was the first after a write record, and their time: the
    /// runtime shifts the new cell to its line inside that decode, over its whole buffer.
    pub shifting_passes: u64,
    pub shifting_pass_ns: u128,
    /// Write records decoded over the records' scope, those also decoded over every cell
    /// (`STOCK_WRITES=check`), and the cells that then differed in any byte.
    pub scoped_writes: u64,
    pub write_checks: u64,
    pub write_cells_differ: u64,
    /// Reader records decoded over the records' scope and the two lines they name, those
    /// also decoded over every cell (`STOCK_READS=check`) and the rows that then differed,
    /// and the reader records whose decoded line is not the line the mirror gives.
    pub scoped_reads: u64,
    pub read_checks: u64,
    pub read_rows_differ: u64,
    pub read_differs: u64,
    /// Scoped sequences read back from the runtime and held to the cache's account of
    /// their lines (`Cache::audit`), and the time that took.
    pub scope_audits: u64,
    pub scope_audit_ns: u128,
    /// Write records decoded to lend a following stream's line to a cell or to put the
    /// cell's line back (`Cache::lend`, `Cache::put_back`), outside `writes`.
    pub swaps: u64,
    pub swap_ns: u128,
    /// Time in the comparisons of cells put back, which run inside the next decode.
    pub swap_check_ns: u128,
    /// Cells put back that were held to the cell their pass found, both after their
    /// shifts, and those held to it before either's shift (the pass found a cell the last
    /// decode had written).
    pub swap_checks: u64,
    pub swap_unshifted_checks: u64,
    /// Of `swap_checks`, the cells equal but for the sign of zero key entries
    /// (`zero_signs`); of those, the cells byte for byte equal after a later shift, and
    /// the cells that kept the sign through `SHIFTS_WATCHED` later shifts.
    pub swap_zero_signs: u64,
    pub swap_signs_cleared: u64,
    pub swap_signs_kept: u64,
    /// Cells put back that differed from the cell their pass found.
    pub swap_cells_differ: u64,
    /// Cells put back that the next decode lent again, so nothing read them: the cell that
    /// loan put back owed the first loan's comparison.
    pub swap_relent: u64,
    /// Cells put back that an engine's write record replaced at the next decode, so
    /// nothing read them and nothing was left to compare.
    pub swap_replaced: u64,
    /// Cells put back and never compared though a decode could read them: the runtime
    /// saved no cell. Every other case is counted above.
    pub swap_unchecked: u64,
    /// `STOCK_SHIFT_PROBE=1`: cells a write record made whose saved bytes were read before
    /// and after the runtime's shift, and those equal, equal but for zero signs, or other.
    pub probe_equal: u64,
    pub probe_signs: u64,
    pub probe_differ: u64,
}

crate::checkpoint::numbers!(CacheStats {
    cells, prefill_sequential, prefill_lanes, passes, writes, reads, off_p, pass_ns, write_ns, read_ns, shifting_passes, shifting_pass_ns, scoped_writes, write_checks, write_cells_differ, scoped_reads, read_checks, read_rows_differ, read_differs, scope_audits, scope_audit_ns, swaps, swap_ns, swap_check_ns, swap_checks,
    swap_unshifted_checks, swap_zero_signs, swap_signs_cleared, swap_signs_kept, swap_cells_differ, swap_relent, swap_replaced, swap_unchecked, probe_equal, probe_signs, probe_differ
});

impl CacheStats {
    /// Why the loans do not hold: a cell put back differed from the cell its pass found, or
    /// was left for a decode to read without a comparison.
    pub fn unheld(&self) -> Option<String> {
        (self.swap_cells_differ > 0 || self.swap_unchecked > 0).then(|| format!("{} cells put back differ from the cells found, {} were never compared", self.swap_cells_differ, self.swap_unchecked))
    }
}

/// The later shifts a cell with a zero of the other sign is read again after.
const SHIFTS_WATCHED: u8 = 4;

/// A cell lent to a following stream's pass: its line, the line it held and holds again
/// after the pass, and what the cell put back is held to.
pub struct Lent {
    n: u32,
    held: [u8; LINE],
    saved: Saved,
}

impl Lent {
    pub fn line(&self) -> u32 {
        self.n
    }
}

/// The saved bytes a cell put back is held to.
enum Saved {
    /// The found cell's after the runtime shifted it. The cell put back is compared once
    /// it has its shift too (`Cache::decoded`).
    Shifted(Vec<u8>),
    /// The found cell's before its shift: the last decode wrote it, and the loan replaces
    /// it before any decode shifts it. The cell put back is compared before its own shift.
    Unshifted(Vec<u8>),
    /// The runtime saves no cell.
    Missing,
}

/// How a write record decodes once the records' scope exists (`Cache::scope_records`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Writes {
    /// Over every cell.
    Whole,
    /// Over the records' scope.
    Scoped,
    /// Both ways, the two cells' saved bytes required equal.
    Check,
}

/// The regions whose lines a record's heads query on its zero inputs, besides the top
/// cell, VOID's first line and line 0. The prefill decodes them before the lanes open,
/// and the records' scope holds them.
const FIXED: [(u32, usize); 9] = [(VOID, BLOCK as usize), (FF, BLOCK as usize), (REGS, layout::N_REGS), (HOST, BLOCK as usize), (IO, (IO_BLOCKS * BLOCK) as usize), (TAB, (TAB_BLOCKS * BLOCK) as usize), (PAL, PAL_LEN), (VRAM, VRAM_LEN), (OAM, OAM_LEN)];

pub struct Cache {
    pub rt: Box<dyn Runtime>,
    pub cfg: Config,
    slots: tok::RecordSlots,
    /// Per line, the line its cell holds, or None without a cell.
    pub mirror: Vec<Option<[u8; LINE]>>,
    pub stats: CacheStats,
    /// Logit rows of every `every`th pass, appended: (pass, rows, width, first token, f32 rows).
    rows: Option<(std::fs::File, u64)>,
    /// Groups decode with only their logit rows attending the cells (`Runtime::split_pass`).
    pub split: bool,
    /// What the scoped sets take as fixed, held to every write record once a scope exists.
    fixed: Option<crate::scope::Fixed>,
    /// Whether the cache holds an overlay's null cell (`Runtime::null`).
    null: bool,
    /// The sequence write records decode over (`scope_records`), and how.
    records_scope: Option<Scoped>,
    pub writes: Writes,
    /// How a reader record decodes once the records' scope exists.
    pub reads: Writes,
    /// Each scoped sequence's lines. A write record's new cell joins the scopes that hold
    /// its line, and a resumed run copies each scope's cells again from these.
    scoped: Vec<(Scoped, BTreeSet<u32>)>,
    /// `STOCK_CELLS=<file>`: one line per write record, the line and a hash of its cell's
    /// saved bytes, so two builds' cells can be compared.
    cells_log: Option<std::fs::File>,
    /// The rows of the LM head the checkpoint's file stores, which bound a narrowed row's
    /// choice (`Row::token`).
    pub head: gbcformer::gguf::HeadRows,
    /// The line whose cell the last decode wrote. The runtime shifts a new cell to its
    /// line at its next decode, so until then the cell's saved bytes are not its final ones.
    unsettled: Option<u32>,
    /// The cell the last decode put back, with the saved bytes it is held to once the
    /// next decode has shifted it.
    owed: Vec<(u32, Vec<u8>)>,
    /// Cells put back that equal the cell found but for zero signs, each with the found
    /// cell's saved bytes and the later shifts it kept the sign through.
    watch: Vec<(u32, Vec<u8>, u8)>,
    /// `STOCK_FAULT=<k>` under `play replay`: the loans' `k`th write record saves other bytes.
    pub fault: Option<(u64, std::sync::Arc<std::sync::atomic::AtomicU8>)>,
    /// `STOCK_SHIFT_PROBE=1`: the last written cell's saved bytes before its shift.
    probe: Option<Option<(u32, Vec<u8>)>>,
    /// `STOCK_FAULT_PUT_BACK=<k>`, in any runtime: the `k`th cell put back (from 0) takes
    /// its line with one bit flipped while the mirror keeps the line, so the cell put back
    /// is not the cell found. A deliberate fault, to show the loan checks stop a run.
    pub fault_put_back: Option<u64>,
    put_backs: u64,
    /// `STOCK_FAULT_READ=<k>`: reader record k (from 0, the probes' included) is taken to
    /// decode its line with one bit flipped. A deliberate fault, to show the mirror check
    /// stops a run with or without the engine's record.
    pub fault_read: Option<u64>,
    /// The prefill's checkpoints, in a run with checkpoints that starts at frame 0.
    pub prefill_ckpt: Option<crate::checkpoint::Prefill>,
}

/// The lines a region of byte addresses touches.
pub fn lines(base: u32, len: usize) -> std::ops::Range<u32> {
    base >> LINE_BITS..(base + len as u32 - 1) / LINE as u32 + 1
}

impl Cache {
    /// The cache of `cfg`'s checkpoint; an overlay's (one with the null token) holds the
    /// null cell first, decoded before any record or pass.
    pub fn new(mut rt: Box<dyn Runtime>, cfg: Config, slots: tok::RecordSlots, head: gbcformer::gguf::HeadRows) -> Cache {
        let null = tok::null(&cfg) < cfg.vocab;
        if null {
            rt.null(tok::null(&cfg));
        }
        let rows = std::env::var("STOCK_ROWS").ok().map(|spec| {
            let (path, every) = spec.split_once(':').unwrap_or((&spec, "50"));
            (std::fs::File::create(path).unwrap_or_else(|e| panic!("{path}: {e}")), every.parse().expect("STOCK_ROWS=<file>:<every>"))
        });
        Cache { rt, cfg, slots, mirror: vec![None; P], stats: CacheStats::default(), rows, split: false, fixed: None, null, records_scope: None, writes: Writes::Scoped, reads: Writes::Scoped, scoped: Vec::new(), cells_log: std::env::var("STOCK_CELLS").ok().map(|p| std::fs::File::create(&p).unwrap_or_else(|e| panic!("{p}: {e}"))), head, unsettled: None, owed: Vec::new(), watch: Vec::new(), fault: None, probe: std::env::var_os("STOCK_SHIFT_PROBE").map(|_| None), fault_put_back: None, put_backs: 0, fault_read: None, prefill_ckpt: None }
    }

    fn records(&self) -> Records<'_> {
        Records { cfg: &self.cfg, slots: &self.slots }
    }

    /// Stops unless `rec` holds only a record's tokens (`tok::stray`). The runtime decodes
    /// a record as its own batch over the cells, every pass's tokens removed, so the
    /// batch is the whole of what its layer-0 gathers weigh.
    fn alone(&self, rec: &[usize]) {
        assert!(rec.contains(&tok::rec(&self.cfg)), "a record's batch starts a record");
        if let Some(t) = tok::stray(&self.cfg, &self.slots, rec) {
            panic!("a record's batch holds token {t}, which no record holds");
        }
    }

    /// Every cell, each through its write record. The fixed regions go first, one record
    /// per decode, in this order:
    /// - the layout's last line, the readback table's (the top cell, so every later
    ///   decode sits at P),
    /// - VOID's first line (a column's off line),
    /// - line 0,
    /// - the lines of VOID, FF, REGS, HOST, the I/O ports, the readback table, PAL, VRAM
    ///   and OAM, which a record's heads
    ///   query on its zero inputs,
    /// - the top cell again, now over those cells, and VOID's line again.
    ///
    /// The rest decode in lanes, each record in its own sequence over the fixed cells.
    pub fn prefill(&mut self, cells: &[Cell]) {
        let t = Instant::now();
        // With STOCK_STATE=<file>, the prefill loads a cache it saved before for the same
        // cells (the file's sidecar holds their digest) instead of decoding it again.
        let state = std::env::var("STOCK_STATE").ok();
        let digest = cells.iter().flat_map(|(n, l)| n.to_le_bytes().into_iter().chain(*l)).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
        let sidecar = format!("lines {digest:016x} {}{}", cells.len(), if self.null { " and the null cell" } else { "" });
        if let Some(path) = &state
            && std::fs::read_to_string(format!("{path}.cells")).ok().as_deref() == Some(&sidecar)
            && self.rt.load(path)
        {
            for &(n, l) in cells {
                self.mirror[n as usize] = Some(l);
            }
            self.stats.cells = cells.len();
            self.stats.prefill_s = t.elapsed().as_secs_f64();
            eprintln!("prefill: {} line cells loaded from {path} (saved by an earlier prefill of the same cells) in {:.1}s", cells.len(), self.stats.prefill_s);
            self.check_loaded(cells);
            return;
        }
        self.decode_cells(cells, &sidecar);
        if let Some(path) = &state
            && self.rt.save(path)
        {
            std::fs::write(format!("{path}.cells"), sidecar).unwrap();
            eprintln!("prefill: cache saved to {path}");
        }
    }

    /// `STOCK_STATE_CHECK=N`: N of the loaded cells, evenly spaced, each held to the cell
    /// this runtime and this model file decode from the same line's write record. The
    /// sidecar names the cells, not the file that decoded them, so this is what shows a
    /// cache saved under one build and file is the cache another would have made. A cell
    /// may differ only by the sign of zero key entries (`zero_signs`), counted apart.
    fn check_loaded(&mut self, cells: &[Cell]) {
        let Some(want) = std::env::var("STOCK_STATE_CHECK").ok().map(|v| v.parse::<usize>().expect("STOCK_STATE_CHECK=<cells>")) else { return };
        let t = Instant::now();
        // Not line 0, whose position the null cell shares, nor the top cell and VOID's,
        // which every record decodes over.
        let apart = [0, P as u32 - 1, VOID >> LINE_BITS];
        let sample: Vec<Cell> = cells.iter().filter(|(n, _)| !apart.contains(n)).step_by((cells.len() / want.max(1)).max(1)).copied().collect();
        let (mut equal, mut signs, mut differ) = (0, 0, 0);
        for &(n, l) in &sample {
            let loaded = self.rt.cell_bytes(n).expect("the runtime saves a cell");
            self.put(n << LINE_BITS, l, true);
            let decoded = self.rt.cell_bytes(n).expect("the runtime saves a cell");
            if decoded == loaded {
                equal += 1;
            } else if zero_signs(&decoded, &loaded) {
                signs += 1;
            } else {
                differ += 1;
                eprintln!("STATE CELL DIFFERS at line {n:#07x}: the loaded cell against the cell decoded here, {} of {} bytes", decoded.iter().zip(&loaded).filter(|(a, b)| a != b).count(), loaded.len().max(decoded.len()));
            }
        }
        eprintln!("state check: {} loaded cells decoded again here in {:.1}s: {equal} equal byte for byte, {signs} equal but for the sign of zero key entries, {differ} differ", sample.len(), t.elapsed().as_secs_f64());
        if differ > 0 {
            crate::quit(1);
        }
    }

    /// Decodes every cell, or the rest of them when another chunk began the prefill
    /// (`checkpoint::Prefill`): then the fixed cells, the open lanes and the lane batches
    /// that chunk decoded load from its checkpoint, the last of those batches takes its
    /// place, and the batches after it decode here. A checkpoint is taken between a
    /// batch's decode and its place, on the timer or when the run is asked to stop.
    fn decode_cells(&mut self, cells: &[Cell], sidecar: &str) {
        let t = Instant::now();
        let mut line = vec![None; P];
        for &(n, l) in cells {
            line[n as usize] = Some(l);
        }
        let top = (TOTAL - 1) >> LINE_BITS;
        assert_eq!(top as usize, P - 1, "the readback table ends the layout");
        // VOID's first line goes right after the top cell. Every column head off its line
        // reads it (`docs/hardware-shaped.md` §20), the records' own positions included,
        // so the records after it hold exact rows. The top cell decodes again below.
        let void = VOID >> LINE_BITS;
        let mut fixed: Vec<u32> = vec![top, void, 0];
        for (base, len) in FIXED {
            fixed.extend(lines(base, len).filter(|&n| n != top && n != 0 && n != void));
        }
        fixed.retain(|&n| line[n as usize].is_some());
        assert_eq!(fixed[0], top, "the readback table's last line holds a cell");
        let lanes = self.rt.lanes();
        let begun = match self.prefill_ckpt.as_mut().filter(|_| lanes > 0).map(|p| p.begun(sidecar, lanes)) {
            Some(Err(e)) => {
                println!("RESUME REFUSED: {e}");
                crate::quit(1);
            }
            Some(Ok(b)) => b,
            None => None,
        };
        for &n in &fixed {
            let l = line[n as usize].take().unwrap();
            match begun {
                None => self.put(n << LINE_BITS, l, false),
                Some(_) => self.mirror[n as usize] = Some(l),
            }
        }
        if begun.is_none() {
            // The top cell and VOID's first decoded over an incomplete cache, so they decode
            // again over every fixed cell.
            self.put(top << LINE_BITS, self.mirror[top as usize].unwrap(), true);
            self.put(VOID, self.mirror[void as usize].unwrap(), true);
        }
        self.stats.prefill_sequential = fixed.len() + 2;
        let rest: Vec<(u32, Vec<usize>)> = (0..P as u32).filter_map(|n| line[n as usize].map(|l| (n, self.records().write(n << LINE_BITS, l)))).collect();
        rest.iter().for_each(|(_, rec)| self.alone(rec));
        let mut prior_s = 0.0;
        if lanes > 0 {
            let batches: Vec<&[(u32, Vec<usize>)]> = rest.chunks(lanes).collect();
            let place = |rt: &mut dyn Runtime, k: usize| rt.lanes_place(&batches[k].iter().map(|(n, _)| *n).collect::<Vec<u32>>());
            let records = |k: usize| (k * lanes).min(rest.len());
            let first = match &begun {
                Some(b) => {
                    if !self.rt.load_all(&b.state) {
                        println!("RESUME FAILED: the runtime did not load {}", b.state);
                        crate::quit(1);
                    }
                    place(self.rt.as_mut(), b.batches - 1);
                    prior_s = b.prefill_s;
                    eprintln!("prefill: resumed after {} of {} lane records, which earlier chunks decoded in {:.0}s; {} loaded", records(b.batches), rest.len(), prior_s, b.state);
                    b.batches
                }
                None => {
                    self.rt.open_lanes(lanes);
                    0
                }
            };
            // A progress line every 200 batches, or every STOCK_PREFILL_EVERY.
            let every = std::env::var("STOCK_PREFILL_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(200usize);
            for k in first..batches.len() {
                self.rt.lanes_decode(batches[k]);
                if let Some(p) = self.prefill_ckpt.as_mut()
                    && let Some(stop) = p.due()
                {
                    let rt = &mut self.rt;
                    let saved = p.commit(k + 1, records(k + 1), rest.len(), sidecar, lanes, prior_s + t.elapsed().as_secs_f64(), |path| rt.save_all(path));
                    if let Err(e) = &saved {
                        println!("CHECKPOINT NOT SAVED in the prefill after {} lane records: {e}; the one before stands", records(k + 1));
                    }
                    if let Some(why) = stop {
                        println!("stopped at a checkpoint: the prefill, {} of {} lane records, {why}", records(k + 1), rest.len());
                        crate::quit(if saved.is_ok() { 0 } else { 1 });
                    }
                }
                place(self.rt.as_mut(), k);
                if k % every == 0 {
                    eprintln!("prefill: {} of {} lane records, {:.0}s", k * lanes, rest.len(), prior_s + t.elapsed().as_secs_f64());
                }
            }
            self.rt.close_lanes(lanes);
        } else {
            for (n, rec) in &rest {
                self.rt.write(*n, rec, false, None, &[]);
            }
        }
        for &(n, l) in cells {
            self.mirror[n as usize] = Some(l);
        }
        self.stats.prefill_lanes = rest.len();
        self.stats.cells = cells.len();
        self.stats.prefill_s = prior_s + t.elapsed().as_secs_f64();
        if self.rt.replays() {
            eprintln!("prefill: REPLAYED, no model ran: {} line cells' records kept by the replay in {:.1}s", cells.len(), self.stats.prefill_s);
        } else {
            eprintln!("prefill: {} line cells ({} one by one, {} in lanes) in {:.1}s", cells.len(), self.stats.prefill_sequential, rest.len(), self.stats.prefill_s);
        }
    }

    /// The write record of `line` at byte address `addr`'s line, decoded into its cell.
    fn put(&mut self, addr: u32, line: [u8; LINE], evict: bool) {
        let n = addr >> LINE_BITS;
        // The rows stay exact only in this order (`docs/stock-llama.md` §4). Every record
        // but the first two (the top cell and VOID's, both decoded again once the fixed
        // cells exist) decodes over VOID's cell, which an off column reads.
        let void = VOID >> LINE_BITS;
        assert!(n == void || self.mirror[void as usize].is_some() || (n as usize == P - 1 && !evict), "the record of line {n:#07x} decodes before VOID's cell");
        let rec = self.records().write(addr, line);
        self.alone(&rec);
        // A cell put back that an engine's record replaces at the next decode: nothing
        // read it (`lend` takes the entry first when the record is a loan's).
        if let Some(i) = self.owed.iter().position(|&(m, _)| m == n) {
            self.owed.swap_remove(i);
            self.stats.swap_replaced += 1;
        }
        self.watch.retain(|&(m, _, _)| m != n);
        let scope = if self.writes == Writes::Whole { None } else { self.records_scope };
        if matches!(&self.probe, Some(Some((m, _))) if *m == n) {
            self.probe = Some(None);
        }
        let join: Vec<Scoped> = self.scoped.iter().filter(|(_, lines)| lines.contains(&n)).map(|(s, _)| *s).collect();
        let at = self.rt.write(n, &rec, evict, scope, &join);
        self.stats.off_p += (at.pos != P) as u64;
        self.stats.scoped_writes += scope.is_some() as u64;
        if scope.is_some() && self.writes == Writes::Check {
            let scoped = self.rt.cell_bytes(n).expect("the runtime saves a cell");
            self.rt.write(n, &rec, true, None, &join);
            let whole = self.rt.cell_bytes(n).expect("the runtime saves a cell");
            self.stats.write_checks += 1;
            if self.stats.write_checks == 1 {
                eprintln!("write check: a cell's saved bytes are {} long", whole.len());
            }
            if scoped != whole {
                self.stats.write_cells_differ += 1;
                eprintln!("WRITE CELL DIFFERS at line {n:#07x}: the record over its scope against every cell, {} of {} bytes", scoped.iter().zip(&whole).filter(|(a, b)| a != b).count(), whole.len());
            }
        }
        if self.cells_log.is_some() {
            let bytes = self.rt.cell_bytes(n).expect("the runtime saves a cell");
            let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
            writeln!(self.cells_log.as_mut().unwrap(), "{n:#07x} {} {hash:016x}", bytes.len()).unwrap();
        }
        self.mirror[n as usize] = Some(line);
        self.decoded(Some(n));
        if self.probe.is_some() {
            self.probe = Some(self.rt.cell_bytes(n).map(|b| (n, b)));
        }
    }

    /// Whether the runtime's saved cache is the cache: no cell waits for its shift (only a
    /// write record leaves one, and llama.cpp saves a cell's new position with its key rows
    /// as they were before the shift), and no cell put back waits for its comparison. A
    /// checkpoint is taken only here, so a run with checkpoints decodes exactly what a run
    /// without them does.
    pub fn quiet(&self) -> bool {
        self.unsettled.is_none() && self.owed.is_empty() && self.watch.is_empty()
    }

    /// The cells the mirror holds, and a digest of their lines.
    pub fn mirror_digest(&self) -> (u64, u64) {
        self.mirror.iter().enumerate().filter_map(|(n, l)| l.map(|l| (n as u32, l))).fold((0, crate::checkpoint::FNV0), |(k, h), (n, l)| (k + 1, crate::checkpoint::fnv(crate::checkpoint::fnv(h, &n.to_le_bytes()), &l)))
    }

    /// Called after every decode, with the line whose cell it wrote, if any. Every earlier
    /// cell has its shift by now, so the cell the decode before put back is held to the
    /// bytes saved before its loan, byte for byte or but for the sign of zero key entries
    /// (`zero_signs`). A cell that differed by such signs is read again after each later
    /// shift, which rewrites every cell's key rows, until it equals the saved bytes.
    fn decoded(&mut self, wrote: Option<u32>) {
        let shifted = self.unsettled.is_some();
        self.unsettled = wrote;
        if let Some(Some((n, before))) = self.probe.as_mut().map(Option::take) {
            match self.rt.cell_bytes(n) {
                Some(now) if now == before => self.stats.probe_equal += 1,
                Some(now) if zero_signs(&now, &before) => self.stats.probe_signs += 1,
                _ => self.stats.probe_differ += 1,
            }
        }
        if self.owed.is_empty() && self.watch.is_empty() {
            return;
        }
        let t = Instant::now();
        if shifted {
            for (n, saved, shifts) in std::mem::take(&mut self.watch) {
                match self.rt.cell_bytes(n) {
                    Some(now) if now == saved => self.stats.swap_signs_cleared += 1,
                    Some(now) if zero_signs(&now, &saved) && shifts + 1 < SHIFTS_WATCHED => self.watch.push((n, saved, shifts + 1)),
                    Some(now) if zero_signs(&now, &saved) => self.stats.swap_signs_kept += 1,
                    _ => self.cell_differs(n, "after a later shift"),
                }
            }
        }
        for (n, saved) in std::mem::take(&mut self.owed) {
            self.stats.swap_checks += 1;
            if self.stats.swap_checks == 1 {
                eprintln!("swap check: a cell's saved bytes are {} long; its f16 key rows, where a zero's sign is tolerated: {}", saved.len(), key_rows(&saved).map_or("NOT FOUND, so no difference is tolerated".to_string(), |k| format!("{} layers, {} bytes", k.len(), k.iter().map(|r| r.len()).sum::<usize>())));
            }
            match self.rt.cell_bytes(n) {
                Some(now) if now == saved => {}
                Some(now) if zero_signs(&now, &saved) => {
                    self.stats.swap_zero_signs += 1;
                    self.watch.push((n, saved, 0));
                }
                _ => self.cell_differs(n, "after its shift"),
            }
        }
        self.stats.swap_check_ns += t.elapsed().as_nanos();
    }

    fn cell_differs(&mut self, n: u32, when: &str) {
        self.stats.swap_cells_differ += 1;
        eprintln!("SWAP CELL DIFFERS at line {n:#07x} {when}: the cell put back after a pass against the cell the pass found");
    }

    /// Makes the sequence write records decode over from here on: the cells of the lines
    /// the prefill decodes before its lanes open (`FIXED`, the top cell, VOID's first line
    /// and line 0), which every later write keeps. A lane's record sees exactly those
    /// cells and its lane's earlier records, so a record over this scope decodes as the
    /// prefill's records did. Returns the cells it holds.
    pub fn scope_records(&mut self) -> usize {
        let mut held: Vec<u32> = FIXED.iter().flat_map(|&(base, len)| lines(base, len)).chain([0, P as u32 - 1]).filter(|&n| self.mirror[n as usize].is_some()).collect();
        held.sort_unstable();
        held.dedup();
        self.records_scope = self.sequence(&held);
        held.len()
    }

    /// A sequence of the cells of `lines` and the top cell (so its next position is P),
    /// which every later write keeps; `fixed` is what the lines were derived under.
    pub fn scope(&mut self, lines: &[u32], fixed: crate::scope::Fixed) -> Option<Scoped> {
        self.fixed = Some(fixed);
        self.sequence(lines)
    }

    fn sequence(&mut self, lines: &[u32]) -> Option<Scoped> {
        let lines = with_top(lines);
        let s = self.rt.scope(&crate::scope::runs(&lines.iter().copied().collect::<Vec<u32>>()))?;
        self.scoped.push((s, lines));
        Some(s)
    }

    fn lines_mut(&mut self, scope: Scoped) -> &mut BTreeSet<u32> {
        &mut self.scoped.iter_mut().find(|(s, _)| *s == scope).expect("a scope this cache made").1
    }

    /// The lines whose cells `scope` holds, the top cell's among them.
    pub fn scope_lines(&self, scope: Scoped) -> &BTreeSet<u32> {
        &self.scoped.iter().find(|(s, _)| *s == scope).expect("a scope this cache made").1
    }

    /// `scope`'s cells replaced by those of `lines` and the top cell.
    pub fn rescope(&mut self, scope: Scoped, lines: &[u32]) {
        let lines = with_top(lines);
        self.rt.rescope(scope, &crate::scope::runs(&lines.iter().copied().collect::<Vec<u32>>()));
        *self.lines_mut(scope) = lines;
    }

    /// Line `n`'s cell joins `scope`, unless it holds it. False when the line holds no cell.
    pub fn scope_add(&mut self, scope: Scoped, n: u32) -> bool {
        if self.mirror.get(n as usize).is_none_or(|l| l.is_none()) {
            return false;
        }
        if self.lines_mut(scope).insert(n) {
            self.rt.scope_add(scope, n);
        }
        true
    }

    /// Line `n`'s cell leaves `scope`.
    pub fn scope_drop(&mut self, scope: Scoped, n: u32) {
        assert!(n as usize != P - 1, "a scope keeps the top cell");
        if self.lines_mut(scope).remove(&n) {
            self.rt.scope_drop(scope, n);
        }
    }

    /// Reads `scope`'s cells back from the runtime and holds their positions to the cache's
    /// own account of the scope: a cell at every line of `scoped` that holds one (a scope
    /// names lines, and a line the image lacks has no cell anywhere), the null cell beside
    /// them in an overlay, and no other. The scoped passes and the engine's record's lines
    /// are held to that account, so this is what holds the account to the runtime. Returns
    /// the difference, or None when the two agree or the runtime keeps no sequence to read.
    pub fn audit(&mut self, scope: Scoped) -> Option<String> {
        let t = Instant::now();
        let got = self.rt.scope_positions(scope)?;
        // The null cell sits at line 0's position (`runtime::NULL_AT`).
        let mut want: Vec<i32> = self.scope_lines(scope).iter().filter(|&&n| self.mirror[n as usize].is_some()).map(|&n| n as i32).chain(self.null.then_some(0)).collect();
        want.sort_unstable();
        self.stats.scope_audits += 1;
        self.stats.scope_audit_ns += t.elapsed().as_nanos();
        if got == want {
            return None;
        }
        let only = |a: &[i32], b: &[i32]| a.iter().find(|p| !b.contains(p)).map_or("none".to_string(), |p| format!("{p:#07x}"));
        Some(format!("the runtime's sequence holds {} cells and the cache's account {} lines with a cell; the first position only the runtime holds: {}, the first only the account holds: {}", got.len(), want.len(), only(&got, &want), only(&want, &got)))
    }

    /// A resumed run's runtime stops replaying: it loads the cache saved at `state` and
    /// copies every scope's cells from it (`Runtime::go_live`).
    pub fn go_live(&mut self, state: &str) -> bool {
        let scopes: Vec<(Scoped, Vec<(u32, u32)>)> = self.scoped.iter().map(|(s, lines)| (*s, crate::scope::runs(&lines.iter().copied().collect::<Vec<u32>>()))).collect();
        self.rt.go_live(state, &scopes)
    }

    /// A write record: the cell of `addr`'s line replaced by `line`.
    pub fn write(&mut self, addr: u32, line: [u8; LINE]) {
        let n = addr >> LINE_BITS;
        if let Some(f) = &self.fixed {
            f.guard(n, &line);
        }
        assert!(self.mirror[n as usize].is_some(), "a write to line {n:#07x} ({addr:#08x}), which holds no cell");
        let t = Instant::now();
        self.put(addr, line, true);
        self.stats.writes += 1;
        self.stats.write_ns += t.elapsed().as_nanos();
    }

    /// Lends `mine`, a following stream's own line `n`, to the cell when the cell holds
    /// another line, for that stream's pass; `put_back` takes what it returns.
    pub fn lend(&mut self, n: u32, mine: [u8; LINE]) -> Option<Lent> {
        let held = self.mirror[n as usize].expect("a followed line holds a cell");
        if held == mine {
            return None;
        }
        let saved = if let Some(i) = self.owed.iter().position(|&(m, _)| m == n) {
            // The last decode put this cell back and nothing has read it (a CGB line's
            // second pass follows its first): the cell this loan puts back owes the
            // comparison with the cell the first loan found.
            self.stats.swap_relent += 1;
            Saved::Shifted(self.owed.swap_remove(i).1)
        } else {
            match self.rt.cell_bytes(n) {
                Some(bytes) if self.unsettled == Some(n) => Saved::Unshifted(bytes),
                Some(bytes) => Saved::Shifted(bytes),
                None => Saved::Missing,
            }
        };
        self.swap(n, mine);
        Some(Lent { n, held, saved })
    }

    /// Puts a lent cell's line back. The cell is held to the found cell's saved bytes: at
    /// once when those were taken before its shift, else once the runtime has shifted this
    /// one too (`decoded`).
    pub fn put_back(&mut self, l: Lent) {
        if self.fault_put_back == Some(self.put_backs) {
            let mut other = l.held;
            other[0] ^= 1;
            eprintln!("FAULT INJECTED (STOCK_FAULT_PUT_BACK={}): line {:#07x}'s cell is put back with byte 0 as {:#04x}, the cell found held {:#04x}", self.put_backs, l.n, other[0], l.held[0]);
            self.swap(l.n, other);
            self.mirror[l.n as usize] = Some(l.held);
        } else {
            self.swap(l.n, l.held);
        }
        self.put_backs += 1;
        match l.saved {
            Saved::Shifted(saved) => self.owed.push((l.n, saved)),
            Saved::Unshifted(saved) => {
                self.stats.swap_unshifted_checks += 1;
                if self.rt.cell_bytes(l.n).as_ref() != Some(&saved) {
                    self.cell_differs(l.n, "before either's shift");
                }
            }
            Saved::Missing => {
                self.stats.swap_unchecked += 1;
                eprintln!("SWAP UNCHECKED at line {:#07x}: the runtime saved no cell to hold the cell put back to", l.n);
            }
        }
    }

    /// The cell of line `n` replaced by `line`'s, through its write record as `write`
    /// decodes one, outside the engines' write counts.
    fn swap(&mut self, n: u32, line: [u8; LINE]) {
        if let Some(f) = &self.fixed {
            f.guard(n, &line);
        }
        let t = Instant::now();
        if let Some((k, fault)) = &self.fault
            && self.stats.swaps == *k
        {
            fault.store(crate::runtime::fault::CORRUPT, std::sync::atomic::Ordering::Relaxed);
        }
        self.put(n << LINE_BITS, line, true);
        self.stats.swaps += 1;
        self.stats.swap_ns += t.elapsed().as_nanos();
    }

    /// A reader record: the line at `dest` with the byte at `source` over byte
    /// `dest mod LINE`, as its eight `RD` positions decode it.
    pub fn read(&mut self, dest: u32, source: u32) -> [u8; LINE] {
        let t = Instant::now();
        let rd = self.records().reader(dest, source);
        self.alone(&rd);
        let outs: Vec<usize> = (rd.len() - LINE..rd.len()).collect();
        if self.rt.replays() {
            // No model reads the cells: the record's line is the mirror's.
            let byte = |a: u32| self.mirror[(a >> LINE_BITS) as usize].expect("a reader record's line holds a cell")[a as usize % LINE];
            let line: [u8; LINE] = std::array::from_fn(|k| if k == dest as usize % LINE { byte(source) } else { byte((dest & !(LINE as u32 - 1)) + k as u32) });
            let tokens: Vec<usize> = (0..LINE).map(|k| tok::family(&self.cfg, Fam::Line { byte: k as u8 }, line[k])).collect();
            self.rt.answer(&tokens);
        }
        let scope = if self.reads == Writes::Whole { None } else { self.records_scope };
        let (rows, at) = match scope {
            None => self.rt.pass(&rd, &outs),
            Some(s) => {
                // The record's copy heads name the destination's line and the source's, and
                // its other heads the lines they name on its zero inputs, which the records'
                // scope holds as it does for a write record. The two lines join the scope
                // for this decode. Every token of the record attends the scope.
                let named: BTreeSet<u32> = [dest >> LINE_BITS, source >> LINE_BITS].into_iter().filter(|n| !self.scope_lines(s).contains(n)).collect();
                for &n in &named {
                    assert!(self.scope_add(s, n), "a reader record's line {n:#07x} holds a cell");
                }
                let also: Vec<usize> = (0..rd.len() - LINE).collect();
                let scoped = self.rt.scoped_pass(&rd, &outs, &also, s);
                self.stats.scoped_reads += 1;
                if self.reads == Writes::Check {
                    let (whole, _) = self.rt.pass(&rd, &outs);
                    self.stats.read_checks += 1;
                    for (k, (a, b)) in whole.iter().zip(&scoped.0).enumerate() {
                        if !a.same(b) {
                            self.stats.read_rows_differ += 1;
                            eprintln!("READ ROW DIFFERS in the reader record of {source:#08x} into {dest:#08x}, row {k}, its scope against every cell: tokens {} vs {}", b.best(), a.best());
                        }
                    }
                }
                for n in named {
                    self.scope_drop(s, n);
                }
                scoped
            }
        };
        self.decoded(None);
        self.stats.off_p += (at.pos != P) as u64;
        let mut out = vec![0; rd.len()];
        for (&o, r) in outs.iter().zip(&rows) {
            out[o] = r.token(&self.head, self.cfg.vocab);
        }
        self.stats.reads += 1;
        self.stats.read_ns += t.elapsed().as_nanos();
        let mut line = self.records().read_back(&out);
        if self.fault_read == Some(self.stats.reads - 1) {
            line[0] ^= 1;
            eprintln!("FAULT INJECTED (STOCK_FAULT_READ={}): the reader record of {source:#08x} into {dest:#08x} is taken to decode byte 0 as {:#04x}", self.stats.reads - 1, line[0]);
        }
        // The mirror holds what the decoded write records wrote, so it gives the line a
        // reader record must decode. A check, never the record's result.
        let byte = |a: u32| self.mirror[(a >> LINE_BITS) as usize].map(|l| l[a as usize % LINE]);
        let want: Option<Vec<u8>> = (0..LINE).map(|k| if k == dest as usize % LINE { byte(source) } else { byte((dest & !(LINE as u32 - 1)) + k as u32) }).collect();
        if want.as_deref() != Some(&line[..]) {
            self.stats.read_differs += 1;
            eprintln!("READ DIFFERS in the reader record of {source:#08x} into {dest:#08x}: decoded {line:02x?}, the mirror gives {want:02x?}");
        }
        line
    }

    /// `STOCK_READ_PROBE=N`: N reader records of evenly spaced lines, each read whole
    /// (`dest = source`), decoded as `reads` says and held to the mirror as every reader
    /// record is. A reader record writes nothing, so the cache is as it was, and the counts
    /// are put back. Returns the records decoded and the milliseconds each took.
    pub fn probe_reads(&mut self, n: usize) -> (usize, f64) {
        let held: Vec<u32> = (0..P as u32).filter(|&m| self.mirror[m as usize].is_some()).collect();
        let before = self.stats.clone();
        let t = Instant::now();
        let lines: Vec<u32> = held.iter().copied().step_by((held.len() / n.max(1)).max(1)).take(n).collect();
        for &m in &lines {
            self.read(m << LINE_BITS, m << LINE_BITS);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / lines.len().max(1) as f64;
        let (differ, rows) = (self.stats.read_differs - before.read_differs, self.stats.read_rows_differ - before.read_rows_differ);
        self.stats = CacheStats { read_differs: before.read_differs + differ, read_rows_differ: before.read_rows_differ + rows, ..before };
        (lines.len(), ms)
    }

    /// A stream's group: the logit row at each of `outs`, the tokens at `outs` and `also`
    /// attending the cells of `scope` (every cell without one).
    ///
    /// On a split or scoped pass, every mask token must be among `outs` and `also`: a token
    /// outside them skips the cells, and a mask carries its part's mode, so its memory heads
    /// would find no line and its units could leave f16's range (`split-check` fact 4).
    pub fn pass(&mut self, tokens: &[usize], outs: &[usize], also: &[usize], scope: Option<Scoped>) -> Vec<Row> {
        if scope.is_some() || self.split {
            let c = &self.cfg;
            let masks = tok::output_mask(c, c.n_inputs)..tok::output_mask(c, c.seq() - 1) + 1;
            let skipped = (0..tokens.len()).find(|i| masks.contains(&tokens[*i]) && !outs.contains(i) && !also.contains(i));
            assert!(skipped.is_none(), "a split pass leaves mask token {} (group index {}) outside the tokens that attend the cells", tokens[skipped.unwrap_or(0)], skipped.unwrap_or(0));
        }
        let t = Instant::now();
        let shifts = self.unsettled.is_some();
        let (rows, at) = match scope {
            Some(s) => self.rt.scoped_pass(tokens, outs, also, s),
            None if self.split => self.rt.split_pass(tokens, outs, also),
            None => self.rt.pass(tokens, outs),
        };
        self.decoded(None);
        self.stats.off_p += (at.pos != P) as u64;
        self.stats.pass_ns += t.elapsed().as_nanos();
        if shifts {
            self.stats.shifting_passes += 1;
            self.stats.shifting_pass_ns += t.elapsed().as_nanos();
        }
        if let Some((f, every)) = &mut self.rows
            && self.stats.passes % *every == 0
        {
            let head = [self.stats.passes, rows.len() as u64, rows.first().map_or(0, |r| r.v.len()) as u64, rows.first().map_or(0, |r| r.first) as u64];
            f.write_all(&head.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            for r in &rows {
                f.write_all(&r.v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
        }
        self.stats.passes += 1;
        rows
    }

    /// Byte addresses in `regions` where the mirror differs from `image` (both holding
    /// the byte).
    pub fn differs(&self, image: &gbcformer::memory::image::Image, regions: &[(u32, usize)]) -> Vec<u32> {
        let mut out = Vec::new();
        for &(base, len) in regions {
            for a in base..base + len as u32 {
                if let Some(l) = self.mirror[(a >> LINE_BITS) as usize]
                    && image.has(a)
                    && image.read(a) != l[a as usize % LINE]
                {
                    out.push(a);
                }
            }
        }
        out
    }
}

/// Whether two cells' saved bytes differ only in the sign of zero entries of their f16 key
/// rows: entries that are +0 in one and -0 in the other. The runtime's shift of a new cell
/// to its line rounds a key entry to a zero of either sign, and a zero of either sign adds
/// the same to every score. Nothing else is tolerated: not a value row, not a header
/// field, and no difference at all when the key rows are not f16 (`key_rows`).
fn zero_signs(a: &[u8], b: &[u8]) -> bool {
    let Some(keys) = key_rows(a) else { return false };
    let entry_top = |i: usize| keys.iter().any(|r| r.contains(&i) && (i - r.start) % 2 == 1);
    a.len() == b.len() && key_rows(b).as_ref() == Some(&keys) && (0..a.len()).all(|i| a[i] == b[i] || (entry_top(i) && a[i - 1] == 0 && b[i - 1] == 0 && a[i] | b[i] == 0x80))
}

/// The byte ranges of the f16 key rows in a cell's saved bytes, one per layer, or None when
/// the bytes are not a saved sequence of one cell with f16 keys and values. llama.cpp
/// (`llama_kv_cache::state_write_data`) ends the file with four 32-bit counts (`v_trans`,
/// which is 0 with flash attention, the layers, and two rotation counts), then per layer
/// the key's type, its row size in 64 bits and the row, then the same per layer for the
/// values. The ranges are those of the one offset from which that layout ends exactly at
/// the end of the bytes.
fn key_rows(bytes: &[u8]) -> Option<Vec<std::ops::Range<usize>>> {
    const F16: u32 = 1;
    let word = |i: usize| bytes.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let long = |i: usize| bytes.get(i..i + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize);
    (0..bytes.len()).step_by(4).find_map(|start| {
        let layers = word(start + 4).filter(|&n| word(start) == Some(0) && (1..=4096).contains(&n))? as usize;
        let (mut at, mut keys) = (start + 16, Vec::new());
        for row in 0..2 * layers {
            let len = long(at + 4).filter(|&len| word(at) == Some(F16) && len > 0 && len % 2 == 0 && len <= bytes.len())?;
            if row < layers {
                keys.push(at + 12..at + 12 + len);
            }
            at += 12 + len;
        }
        (at == bytes.len()).then_some(keys)
    })
}

/// `lines` and the top cell's line.
fn with_top(lines: &[u32]) -> BTreeSet<u32> {
    lines.iter().copied().chain([P as u32 - 1]).collect()
}

/// Every line `image` holds a cell for, with its bytes.
pub fn cells_of(image: &gbcformer::memory::image::Image) -> Vec<Cell> {
    (0..P as u32).filter(|&n| image.has_line(n)).map(|n| (n, image.line(n))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{fault, Replay};
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::Arc;

    const A: [u8; LINE] = [1; LINE];
    const B: [u8; LINE] = [2; LINE];
    const C: [u8; LINE] = [3; LINE];

    /// A cache over `Replay` with VOID's cell and two OAM lines that hold `A`.
    fn cache() -> (Cache, Arc<AtomicU8>, u32, u32) {
        let cfg = Config { hidden: 0, intermediate: 0, layers: 1, heads: 1, kv_heads: 1, vocab: 0, n_inputs: 1, n_outputs: 2 * LINE, n_const: 0, families: Fam::ALL.len() };
        let cfg = Config { vocab: tok::null(&cfg), ..cfg };
        let slots = tok::RecordSlots { addr: [[0; 3]; 2], line: [std::array::from_fn(|k| 1 + k), std::array::from_fn(|k| 1 + LINE + k)], oam: None };
        let rt = Replay::default();
        let fault = rt.fault.clone();
        let mut c = Cache::new(Box::new(rt), cfg, slots, gbcformer::gguf::HeadRows { first: 0, count: 1, fill: -1024.0 });
        let (n, m) = (OAM >> LINE_BITS, (OAM >> LINE_BITS) + 1);
        c.put(VOID, [0; LINE], false);
        c.put(n << LINE_BITS, A, false);
        c.put(m << LINE_BITS, A, false);
        c.idle();
        (c, fault, n, m)
    }

    impl Cache {
        /// A pass that reads no row: any decode shifts the cell the last one wrote.
        fn idle(&mut self) {
            self.rt.answer(&[]);
            self.pass(&[], &[], &[], None);
        }
    }

    #[test]
    fn a_cell_put_back_is_held_to_the_cell_found_once_it_has_its_shift() {
        let (mut c, _, n, _) = cache();
        assert!(c.lend(n, A).is_none(), "the cell already holds the stream's line");
        let lent = c.lend(n, B).unwrap();
        assert_eq!(c.mirror[n as usize], Some(B));
        c.put_back(lent);
        assert_eq!(c.mirror[n as usize], Some(A));
        assert_eq!((c.stats.swaps, c.stats.swap_checks), (2, 0));
        c.idle();
        assert_eq!((c.stats.swap_checks, c.stats.swap_unshifted_checks, c.stats.swap_cells_differ, c.stats.swap_unchecked), (1, 0, 0, 0));
        assert_eq!(c.stats.unheld(), None);
    }

    #[test]
    fn two_lines_lent_to_one_pass_are_both_compared() {
        let (mut c, _, n, m) = cache();
        let lent = [c.lend(n, B).unwrap(), c.lend(m, C).unwrap()];
        lent.into_iter().for_each(|l| c.put_back(l));
        c.idle();
        assert_eq!((c.stats.swap_checks, c.stats.swap_cells_differ, c.stats.swap_unchecked), (2, 0, 0));
    }

    /// A CGB line's two passes run back to back and lend the same line.
    #[test]
    fn a_line_lent_again_at_once_carries_the_first_loan_s_comparison() {
        let (mut c, _, n, _) = cache();
        for _ in 0..2 {
            let lent = c.lend(n, B).unwrap();
            c.idle();
            c.put_back(lent);
        }
        assert_eq!((c.stats.swap_relent, c.stats.swap_checks), (1, 0));
        c.idle();
        assert_eq!((c.stats.swap_checks, c.stats.swap_unshifted_checks, c.stats.swap_cells_differ, c.stats.swap_unchecked, c.stats.swap_replaced), (1, 0, 0, 0, 0));
    }

    #[test]
    fn a_carried_comparison_still_catches_the_last_cell_put_back() {
        let (mut c, fault, n, _) = cache();
        let lent = c.lend(n, B).unwrap();
        c.put_back(lent);
        let lent = c.lend(n, B).unwrap();
        fault.store(fault::CORRUPT, Ordering::Relaxed);
        c.put_back(lent);
        c.idle();
        assert_eq!((c.stats.swap_checks, c.stats.swap_cells_differ), (1, 1));
        assert!(c.stats.unheld().is_some());
    }

    /// The GBC CPU's record of the line is the last decode before the following pass.
    #[test]
    fn a_cell_found_before_its_shift_is_compared_before_the_shift_of_the_cell_put_back() {
        let (mut c, fault, n, _) = cache();
        c.write(n << LINE_BITS, C);
        let lent = c.lend(n, B).unwrap();
        c.idle();
        c.put_back(lent);
        assert_eq!((c.stats.swap_unshifted_checks, c.stats.swap_cells_differ, c.stats.swap_unchecked), (1, 0, 0));
        assert_eq!(c.mirror[n as usize], Some(C));
        c.write(n << LINE_BITS, C);
        let lent = c.lend(n, B).unwrap();
        fault.store(fault::CORRUPT, Ordering::Relaxed);
        c.put_back(lent);
        c.idle();
        assert_eq!((c.stats.swap_unshifted_checks, c.stats.swap_checks, c.stats.swap_cells_differ), (2, 0, 1));
    }

    #[test]
    fn a_cell_put_back_that_differs_fails_the_loans() {
        let (mut c, fault, n, _) = cache();
        let lent = c.lend(n, B).unwrap();
        fault.store(fault::CORRUPT, Ordering::Relaxed);
        c.put_back(lent);
        c.idle();
        assert_eq!(c.stats.swap_cells_differ, 1);
        assert!(c.stats.unheld().is_some());
    }

    #[test]
    fn a_put_back_of_another_line_than_the_cell_held_fails_the_loans() {
        let (mut c, _, n, m) = cache();
        c.fault_put_back = Some(1);
        let lent = [c.lend(n, B).unwrap(), c.lend(m, B).unwrap()];
        lent.into_iter().for_each(|l| c.put_back(l));
        assert_eq!(c.mirror[m as usize], Some(A), "the mirror keeps the line the cell held");
        c.idle();
        assert_eq!((c.stats.swap_checks, c.stats.swap_cells_differ), (2, 1));
        assert!(c.stats.unheld().is_some());
    }

    #[test]
    fn a_cache_is_quiet_once_every_cell_has_its_shift_and_its_comparison() {
        let (mut c, _, n, _) = cache();
        assert!(c.quiet());
        c.write(n << LINE_BITS, C);
        assert!(!c.quiet(), "the written cell waits for its shift");
        c.idle();
        assert!(c.quiet());
        let lent = c.lend(n, B).unwrap();
        c.idle();
        c.put_back(lent);
        assert!(!c.quiet(), "the cell put back waits for its shift and its comparison");
        c.idle();
        assert!(c.quiet());
        let before = c.mirror_digest();
        c.write(n << LINE_BITS, A);
        assert_eq!(c.mirror_digest().0, before.0);
        assert_ne!(c.mirror_digest().1, before.1);
    }

    #[test]
    fn a_cell_put_back_that_an_engine_s_record_replaces_unread_is_counted_apart() {
        let (mut c, _, n, _) = cache();
        let lent = c.lend(n, B).unwrap();
        c.put_back(lent);
        c.write(n << LINE_BITS, C);
        c.idle();
        assert_eq!((c.stats.swap_replaced, c.stats.swap_checks, c.stats.swap_unchecked, c.stats.swap_cells_differ), (1, 0, 0, 0));
    }

    #[test]
    fn a_zero_of_the_other_sign_is_read_again_after_each_later_shift() {
        let (mut c, fault, n, m) = cache();
        let lent = c.lend(n, B).unwrap();
        fault.store(fault::SIGN, Ordering::Relaxed);
        c.put_back(lent);
        c.idle();
        assert_eq!((c.stats.swap_zero_signs, c.stats.swap_signs_cleared, c.stats.swap_cells_differ), (1, 0, 0));
        c.idle();
        assert_eq!(c.stats.swap_signs_cleared, 0, "a pass shifts nothing");
        c.write(m << LINE_BITS, C);
        c.idle();
        assert_eq!((c.stats.swap_signs_cleared, c.stats.swap_signs_kept), (1, 0));

        let lent = c.lend(n, B).unwrap();
        fault.store(fault::SIGN_KEPT, Ordering::Relaxed);
        c.put_back(lent);
        for _ in 0..SHIFTS_WATCHED {
            c.write(m << LINE_BITS, C);
        }
        c.idle();
        assert_eq!((c.stats.swap_zero_signs, c.stats.swap_signs_cleared, c.stats.swap_signs_kept, c.stats.swap_cells_differ), (2, 1, 1, 0));
        assert_eq!(c.stats.unheld(), None);
    }

    /// A saved sequence of one cell: `head` bytes, the counts, then per layer an f16 key
    /// row and per layer an f16 value row (type 1), or f32 rows (type 0).
    fn saved(head: usize, kind: u32, keys: &[&[u8]], values: &[&[u8]]) -> Vec<u8> {
        let mut b = vec![7u8; head];
        for w in [0, keys.len() as u32, 0, 0] {
            b.extend(w.to_le_bytes());
        }
        for row in keys.iter().chain(values) {
            b.extend(kind.to_le_bytes());
            b.extend((row.len() as u64).to_le_bytes());
            b.extend(*row);
        }
        b
    }

    #[test]
    fn only_the_sign_of_a_zero_f16_key_entry_is_tolerated() {
        let key = [0, 0, 5, 0x3c, 0, 0];
        let value = [0, 0, 9, 9];
        let base = saved(24, 1, &[&key, &key], &[&value, &value]);
        let rows = key_rows(&base).unwrap();
        assert_eq!(rows.iter().map(|r| r.len()).collect::<Vec<_>>(), [6, 6]);
        assert_eq!(&base[rows[1].clone()], &key);
        let with = |i: usize, v: u8| {
            let mut b = base.clone();
            b[i] = v;
            b
        };
        assert!(zero_signs(&base, &base));
        // -0 for +0 in the first and the last entry of a key row, either way round
        for i in [rows[0].start + 1, rows[1].start + 5] {
            assert!(zero_signs(&base, &with(i, 0x80)) && zero_signs(&with(i, 0x80), &base));
        }
        // the sign of a nonzero entry, the low byte of a zero, and another value
        assert!(!zero_signs(&base, &with(rows[0].start + 3, 0xbc)));
        assert!(!zero_signs(&base, &with(rows[0].start, 0x80)));
        assert!(!zero_signs(&base, &with(rows[0].start + 1, 0x01)));
        // the same word in a value row and in the header
        let value_row = base.len() - 4;
        assert_eq!(&base[value_row..], &value);
        assert!(!zero_signs(&base, &with(value_row + 1, 0x80)));
        assert!(!zero_signs(&with(1, 0), &with(1, 0x80)));
        // another length
        assert!(!zero_signs(&base, &base[..base.len() - 2]));
        // f32 rows: bit 15 of a mantissa is a real difference
        let wide = saved(24, 0, &[&[0, 0, 0, 0]], &[&[0, 0, 0, 0]]);
        assert_eq!(key_rows(&wide), None);
        let mut other = wide.clone();
        let at = wide.len() - 4 - 12 - 3;
        other[at] = 0x80;
        assert!(!zero_signs(&wide, &other));
    }
}
