//! The fast-pass counters.

/// Why a fast pass took the dense model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The test hook `Engine::force_fallback`.
    Forced,
    /// The checkpoint has no fast plan (`Engine::exact` says why).
    NoPlan,
    /// The layer-0 attention output is not the exact integers the circuit reads, or
    /// a token sits where the protocol never places it.
    Layer0Attention,
    /// A memory head's address bit is not 0 or 1.
    MemoryQuery,
    /// A wire left the range its clearing, or its declared range, is exact on.
    WireOutOfRange,
    /// A position with a positional stage holds a token other than its mask.
    PositionalToken,
    /// A route total is not 0 or 1.
    RouteNot01,
    /// A pass routed one channel to one layer twice, past its bucket.
    BucketOverflow,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Refusal::Forced => "forced",
            Refusal::NoPlan => "no plan",
            Refusal::Layer0Attention => "layer-0 attention",
            Refusal::MemoryQuery => "memory query",
            Refusal::WireOutOfRange => "wire out of range",
            Refusal::PositionalToken => "positional token",
            Refusal::RouteNot01 => "route not 0/1",
            Refusal::BucketOverflow => "bucket overflow",
        })
    }
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub passes: u64,
    pub fast_passes: u64,
    pub fallback_passes: u64,
    /// Units re-evaluated by fast passes.
    pub units: u64,
    /// Fast passes by the number of positions they decoded, in first-seen order.
    pub kinds: Vec<(usize, Kind)>,
    /// Units re-evaluated per circuit layer, then the routing layer.
    pub layer_units: Vec<u64>,
    /// Layer-0 input changes scattered, and routing units corrected per position.
    pub changes: u64,
    pub own: u64,
    /// Positions decoded by fast passes (the rest reused their cached token).
    pub decoded: u64,
    /// Memory: bytes served from the index, records kept, and dense
    /// oracle checks run (passes and kept tokens recomputed by `forward_cached`).
    pub memory_reads: u64,
    pub records: u64,
    pub oracle_checks: u64,
    /// Copy oracle checks whose source is ROM past 2 MiB.
    pub oracle_high_rom: u64,
    /// Fast passes the step oracle ran through the dense pass, every decoded token equal.
    pub step_checks: u64,
    /// Of them, the passes checked whatever the cadence (`MemoryDenoiser::check_next`).
    pub forced_checks: u64,
    /// Their wall time, when timed.
    pub step_check_ns: u64,
    /// Under the step oracle: fast passes whose memory heads read a ROM line with key
    /// bit 18, with bit 19, and with either (ROM past 2 MiB); and checked passes that read
    /// a line with key bit 18, 19 or 20 set. Bit 20 is the region bit of every line above
    /// ROM.
    pub high_rom_passes: [u64; 2],
    pub high_rom_seen: u64,
    pub step_high: [u64; 3],
    /// DMA bytes kept as copy-mode records (`Engine::copy`), and those whose source byte
    /// the host served.
    pub copies: u64,
    pub host_copies: u64,
    /// Opcode bytes the host served, kept as records in the HOST block (`keep_host`).
    pub host_fetches: u64,
    /// Wall time of fast passes by stage, when timed: layer 0 (the changed inputs and
    /// the attention diff), the shared layers (circuit layers, memory reads, routing),
    /// and the decoded positions (own routing units and the head table).
    pub stage_ns: [u64; 3],
    /// Layer 0: positions whose token moved, and passes that planned the attention again.
    pub layer0_moved: u64,
    pub layer0_replans: u64,
    /// The replans by cause: no plan held, a moved input's key or query, a moved output's,
    /// the decoded positions. Then those whose key the memo held, and of them those that
    /// gave the plan already held.
    pub replan_why: [u64; 4],
    pub replan_memo_hits: u64,
    pub replan_same: u64,
    /// When timed, by the thread's counters: the replans read, the memo's key and lookup,
    /// and the rest of layer 0 from there.
    pub replan_timed: u64,
    pub replan_key: super::counters::Counts,
    pub replan_rest: super::counters::Counts,
    /// Layer 0: passes on an exclusive plan's cached wire lists, and the (head, key)
    /// terms the other passes mixed.
    pub layer0_exclusive: u64,
    pub layer0_terms: u64,
    /// The cells' rows built from the weights (a cartridge's lines, then one per record).
    pub cell_rows: u64,
    /// Write records whose rows came from the memo, and how many of those `Cells::put`
    /// built again and compared.
    pub cell_memo: u64,
    pub cell_memo_checked: u64,
    /// The memo's entry count at the last record.
    pub cell_memo_len: usize,
    /// Wall time building the rows of inserted cartridges.
    pub cell_insert_ns: u64,
    /// Reads served from the cells by the key lookup, by the full scan after a guard
    /// failed, and lookups the scan also checked (`GBCFORMER_SCAN=1`).
    pub cell_lookups: u64,
    pub cell_scans: u64,
    pub cell_oracle_scans: u64,
    /// Why the last refused pass took the dense model (None before any).
    pub refused: Option<Refusal>,
    /// Gate updates per node family (`Engine::families`), when GBCFORMER_UNITS is set.
    pub family_updates: Vec<u64>,
    /// Gate updates per changed source channel, when GBCFORMER_UNITS is set.
    pub source_updates: Vec<u64>,
    /// (changed channel, reading layer) pairs the fast passes walked: each is one gate
    /// row, so this is the per-change work the unit count leaves out.
    pub routed: u64,
    /// Changes per channel, when GBCFORMER_UNITS is set.
    pub channel_changes: Vec<u64>,
    /// Census: gate updates of circuit units, then routing units, that no position the
    /// pass decodes can reach (`Plan::demand`).
    pub unreached: [u64; 2],
    /// When timed: this engine's thread's counters (`counters::thread`) at its first pass
    /// and at its latest 4,096th, each with the passes before it.
    pub thread_first: Option<(u64, std::thread::ThreadId, super::counters::Counts)>,
    pub thread_last: Option<(u64, std::thread::ThreadId, super::counters::Counts)>,
    /// When timed: empty brackets of the thread's counters, one beside each sampled pass,
    /// and their sum: what the bracket itself adds to each sampled figure.
    pub brackets: u64,
    pub bracket: super::counters::Counts,
    /// The replay bench's stage bracket (`probe`).
    pub probe: super::probe::Probe,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Kind {
    pub passes: u64,
    pub units: u64,
    /// Wall time, only when `Engine::timed` is set.
    pub nanos: u64,
    /// When timed: the passes bracketed by the thread's counters, and their sum.
    pub sampled: u64,
    pub counts: super::counters::Counts,
}

impl Stats {
    /// The counters for fast passes that decode `positions` positions.
    #[inline]
    pub(super) fn kind(&mut self, positions: usize) -> &mut Kind {
        let i = self.kinds.iter().position(|k| k.0 == positions).unwrap_or_else(|| {
            self.kinds.push((positions, Kind::default()));
            self.kinds.len() - 1
        });
        &mut self.kinds[i].1
    }

    pub(super) fn count_read(&mut self, served: super::cells::Served, oracle: bool) {
        match served {
            super::cells::Served::Lookup => {
                self.cell_lookups += 1;
                self.cell_oracle_scans += oracle as u64;
            }
            super::cells::Served::Scan => self.cell_scans += 1,
        }
    }

    /// The step oracle's coverage of the key's high bits.
    pub fn step_high_line(&self) -> String {
        format!("fast passes reading ROM past 2 MiB: {}, {} with key bit 18, {} with bit 19; checked passes reading a line with key bit 18, 19, 20: {}, {}, {}", self.high_rom_seen, self.high_rom_passes[0], self.high_rom_passes[1], self.step_high[0], self.step_high[1], self.step_high[2])
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let fast = self.fast_passes.max(1) as f64;
        write!(f, "passes {} ({} fast, {} dense fallback); {:.0} units re-evaluated per fast pass", self.passes, self.fast_passes, self.fallback_passes, self.units as f64 / fast)?;
        if let Some(why) = self.refused {
            write!(f, " (last refusal: {why})")?;
        }
        if self.step_checks > 0 {
            write!(f, "; {} fast passes checked by the step oracle, every token equal to the dense pass's", self.step_checks)?;
            if self.step_check_ns > 0 {
                write!(f, " ({:.2} s each)", self.step_check_ns as f64 / self.step_checks as f64 / 1e9)?;
            }
            write!(f, "; {}", self.step_high_line())?;
        }
        writeln!(f)?;
        for (positions, k) in &self.kinds {
            let n = k.passes.max(1) as f64;
            write!(f, "  passes decoding {positions} positions: {} ({:.0} units, {:.2} µs each", k.passes, k.units as f64 / n, k.nanos as f64 / n / 1e3)?;
            if k.sampled > 0 {
                let s = k.sampled as f64;
                write!(f, "; {} sampled by the thread's counters: {:.2} µs CPU, {:.0} cycles, {:.0} instructions each", k.sampled, k.counts.cpu_ns as f64 / s / 1e3, k.counts.cycles as f64 / s, k.counts.instructions as f64 / s)?;
            }
            writeln!(f, ")")?;
        }
        if self.brackets > 0 {
            let b = self.brackets as f64;
            writeln!(f, "  an empty bracket of the thread's counters, which each sampled figure includes: {:.0} cycles, {:.0} instructions", self.bracket.cycles as f64 / b, self.bracket.instructions as f64 / b)?;
        }
        let per: Vec<String> = self.layer_units.iter().map(|&u| format!("{:.0}", u as f64 / fast)).collect();
        write!(f, "  per pass: {:.1} layer-0 changes, {:.1} (changed channel, reading layer) pairs, units by layer [{}], {:.0} own routing units, {:.1} positions decoded", self.changes as f64 / fast, self.routed as f64 / fast, per.join(", "), self.own as f64 / fast, self.decoded as f64 / fast)?;
        if self.stage_ns.iter().any(|&n| n > 0) {
            let us = |n: u64| n as f64 / fast / 1e3;
            write!(f, "\n  stages per fast pass: layer 0 {:.2} µs, shared layers {:.2} µs, decode {:.2} µs", us(self.stage_ns[0]), us(self.stage_ns[1]), us(self.stage_ns[2]))?;
            write!(f, "; layer 0: {:.1} moved positions and {:.3} replans per pass, {:.3} on exclusive plans, {:.0} (head, key) terms mixed", self.layer0_moved as f64 / fast, self.layer0_replans as f64 / fast, self.layer0_exclusive as f64 / fast, self.layer0_terms as f64 / fast)?;
            if self.layer0_replans > 0 {
                let [none, input, output, positions] = self.replan_why;
                write!(f, "\n  layer-0 replans: {} ({none} with no plan held, {input} at a moved input, {output} at a moved output, {positions} at the decoded positions); the memo held {}, {} of them the plan in place", self.layer0_replans, self.replan_memo_hits, self.replan_same)?;
            }
            if self.replan_timed > 0 {
                let n = self.replan_timed as f64;
                write!(f, "; each: the memo's key and lookup {:.0} cycles and {:.0} instructions, the rest of layer 0 {:.0} and {:.0}", self.replan_key.cycles as f64 / n, self.replan_key.instructions as f64 / n, self.replan_rest.cycles as f64 / n, self.replan_rest.instructions as f64 / n)?;
            }
        }
        if self.unreached.iter().any(|&u| u > 0) {
            write!(f, "\n  census: {:.1} circuit and {:.1} routing gate updates per pass reach no position the pass decodes", self.unreached[0] as f64 / fast, self.unreached[1] as f64 / fast)?;
        }
        if let (Some((p0, _, c0)), Some((p1, _, c1))) = (self.thread_first, self.thread_last)
            && p1 > p0
        {
            let (d, n) = (c1.since(c0), (p1 - p0) as f64);
            write!(f, "\n  this engine's thread, by its own counters over passes {p0} to {p1}: {:.3} µs CPU, {:.0} cycles, {:.0} instructions per pass", d.cpu_ns as f64 / n / 1e3, d.cycles as f64 / n, d.instructions as f64 / n)?;
        }
        if self.memory_reads > 0 {
            write!(f, "\n  memory: {:.1} head reads per fast pass, {} records kept, {} dense oracle checks ({} copies from ROM past 2 MiB)", self.memory_reads as f64 / fast, self.records, self.oracle_checks, self.oracle_high_rom)?;
            write!(f, "\n  cells: {} rows built ({:.1} s inserting), {} records' rows from the memo by their tokens ({} built again and compared; {} entries); reads {} by lookup, {} by full scan (a guard failed), {} lookups checked by the scan", self.cell_rows, self.cell_insert_ns as f64 / 1e9, self.cell_memo, self.cell_memo_checked, self.cell_memo_len, self.cell_lookups, self.cell_scans, self.cell_oracle_scans)?;
        }
        Ok(())
    }
}
