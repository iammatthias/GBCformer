//! The transport between a model's streams: when a stream of spans runs its next pass,
//! the copies a pass hands the host to perform, when a battery save is due, the output
//! filter over the mixed samples and the files a run's outputs go to. Every rule here is the host's (counting cycles,
//! concatenating address bits, keeping order); what the machine does comes from the
//! passes' outputs and the numbers the caller hands in (the wiring table's data). The
//! full program (`control_circuit::ControlModel`, `machine`, `sm83`) and the minimal
//! runner (`tools/runner`) both run this module, so their transport is one by
//! construction. Each side supplies only its passes (the full program's circuit fields,
//! the runner's wiring table) through the traits here: `Passes`, `Blocks`, `Sink`, `Accesses`.

/// A copy a pass asks the host to perform, between bus addresses whose bits the pass
/// emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Copy {
    /// One byte of a run.
    Byte { source: u16, dest: u16 },
    /// One byte of a run with no source: `value`.
    Fill { value: u8, dest: u16 },
    /// A block of `len` bytes: byte k moves from `source | k` to `dest | k`.
    Block { source: u16, dest: u16, len: u16 },
}

impl Copy {
    /// The (source, dest) pairs in the order they move, each placed by concatenating the
    /// byte's low address bits onto the addresses the pass named. A fill has no source.
    pub fn bytes(self) -> impl Iterator<Item = (Option<u16>, u16)> {
        let (source, dest, len) = match self {
            Copy::Byte { source, dest } => (Some(source), dest, 1),
            Copy::Fill { dest, .. } => (None, dest, 1),
            Copy::Block { source, dest, len } => {
                // Concatenation needs the low bits free: the length a power of two, both
                // addresses aligned to it.
                assert!((source | dest) & (len - 1) == 0 && len.is_power_of_two(), "a block of {len} bytes at {source:#06x} -> {dest:#06x} is not aligned to its length");
                (Some(source), dest, len)
            }
        };
        (0..len).map(move |k| (source.map(|s| s | k), dest | k))
    }
}

/// A run of copied bytes a pass emitted: within the line at `line` of 2^`bits` bytes,
/// from page `page` to page `dest`, a bit per byte moved in the span's cycles before its
/// last (`before`) and in its last (`last`); a nonzero `fill` means no source. The bytes
/// before the last cycle land before a line that cycle starts is drawn; the last cycle's
/// byte lands before it when `last_early` says so, else after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Run {
    pub page: u8,
    pub line: u8,
    pub before: u8,
    pub last: u8,
    pub last_early: bool,
    pub dest: u8,
    pub fill: u8,
}

impl Run {
    /// The run's bytes, those that land before a line the span's last cycle starts into
    /// `early`, the rest into `late`, each at the address bits the pass emitted.
    pub fn bytes(self, bits: u8, early: &mut Vec<Copy>, late: &mut Vec<Copy>) {
        let size = 1u16 << bits;
        assert!(self.line as u16 & (size - 1) == 0, "a run's line {:#04x} is not aligned to its {size} bytes", self.line);
        let (at, to, fill) = ((self.page as u16) << 8 | self.line as u16, (self.dest as u16) << 8 | self.line as u16, self.fill);
        let bytes = move |moved: u8| (0..size).filter(move |j| moved >> j & 1 != 0).map(move |j| if fill != 0 { Copy::Fill { value: fill, dest: to | j } } else { Copy::Byte { source: at | j, dest: to | j } });
        early.extend(bytes(self.before));
        if self.last_early { &mut *early } else { late }.extend(bytes(self.last));
    }

    /// A run moves one byte a cycle, its last in the span's last cycle.
    pub fn check(self, cycles: u64) {
        let moved = self.before | self.last;
        assert!(self.before & self.last == 0 && self.last.count_ones() == 1 && moved.count_ones() as u64 == cycles, "a run of {:#010b} and {:#010b} after a span of {cycles} cycles", self.before, self.last);
    }
}

/// Starts a pass scheduled ahead: the first `first` cycles after the pass, then one every
/// `period`, `count` of them, each at its line of `lines` and the phase `phase`; the next
/// pass `tail` cycles after the last.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    pub count: usize,
    pub first: u64,
    pub period: u64,
    pub tail: u64,
    pub phase: u8,
    pub lines: [u8; MAX_STARTS],
}

/// The most starts a pass schedules.
pub const MAX_STARTS: usize = 8;

/// What a pass of spans tells the host about cutting the next ones.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cut {
    /// Cycles from the pass to the next one.
    pub horizon: u64,
    pub schedule: Schedule,
    /// Where, in cycles since the pass, the edge the segments stream steps at falls by
    /// counting (None: held), and where the edge a reset dropped falls (0: none).
    pub edge: Option<u64>,
    pub drop: u64,
    /// Where it next rises by counting (None: held).
    pub rise: Option<u64>,
    /// The cycles the steps stream stays stopped once the step that runs has ended, from
    /// the pass: a total, counted down only while it is stopped.
    pub owed: u64,
}

/// The segments stream's edges a block's last cycle holds: the bit's fall (`reset`: the
/// one a reset of the counter dropped) or its rise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Edges {
    pub fall: bool,
    pub reset: bool,
    pub rise: bool,
}

/// Where the next pass of spans starts: the cycles since the last full span, and after
/// a scheduled start the position the model emitted for it (phase, line) with the
/// cycles since it.
#[derive(Clone, Copy, Debug)]
pub struct Start {
    pub cycles: u64,
    pub since: u64,
    pub position: Option<(u8, u8)>,
}

impl Start {
    /// `state` as the pass starts: after a scheduled start, its phase and line at the
    /// position the model emitted for it (`phase_at`, `line_at`).
    pub fn place(&self, state: &mut [u8], phase_at: usize, line_at: usize) {
        if let Some((phase, line)) = self.position {
            state[phase_at] = phase;
            state[line_at] = line;
        }
    }
}

/// The locks `at` cycles after the last pass, while a span that pass emitted holds them.
pub fn held_lock<U: std::marker::Copy + Into<u64>>(locks: &[(u8, U)], at: u64) -> Option<u8> {
    locks.iter().find(|&&(_, until)| at < until.into()).map(|&(l, _)| l)
}

/// The bus state an access `at` cycles after the last pass sees: with `locks`, the locks a
/// span the last pass emitted holds there, else those of a look-ahead pass (`query`); and
/// the busy copy's bit while `at` lies in its `busy` cycles.
pub fn bus_state<U: std::marker::Copy + Into<u64>>(held: &[(u8, U)], busy: std::ops::Range<u64>, at: u64, locks: bool, write: bool, bits: StateBits, query: impl FnOnce() -> u8) -> u8 {
    let l = if locks { held_lock(held, at).unwrap_or_else(query) } else { 0 };
    let shift = if write { bits.write_shift } else { 0 };
    (l >> shift) & bits.locks | if busy.contains(&at) { bits.busy } else { 0 }
}

/// What the cycles just run reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    /// Nothing: the next stop lies ahead.
    Nothing,
    /// A scheduled start: draw this line, no pass.
    Start(u8),
    /// The horizon: a pass.
    Pass,
}

/// The countdown that cuts a stream of spans into passes: at each pass's horizon, at each
/// scheduled start, at each edge. It keeps cycles, nothing of the machine.
#[derive(Clone, Debug, Default)]
pub struct Spans {
    elapsed: u64,
    horizon: u64,
    schedule: Schedule,
    passed: usize,
    countdown: u64,
    since_start: u64,
    /// The fall a reset dropped, the counted fall, the rise.
    edges: [Option<u64>; 3],
    owed: u64,
    max_span: u64,
}

impl Spans {
    pub fn new(max_span: u64) -> Spans {
        Spans { max_span, ..Spans::default() }
    }

    /// Cycles since the last pass.
    pub fn elapsed(&self) -> u64 {
        self.elapsed
    }

    /// Cycles to the next stop: the first scheduled start or the horizon, then the
    /// countdown from each start passed.
    pub fn until_stop(&self) -> u64 {
        if self.passed == 0 { self.horizon - self.elapsed } else { self.countdown }
    }

    /// Cycles to the next edge.
    pub fn until_edge(&self) -> u64 {
        self.edges.iter().flatten().filter(|&&e| e > self.elapsed).map(|e| e - self.elapsed).min().unwrap_or(u64::MAX)
    }

    /// The cycles the steps stream stays stopped once the step that runs has ended: what the
    /// last pass emitted, less the stopped cycles since. Each pass takes it as an input.
    pub fn owed(&self) -> u64 {
        self.owed
    }

    /// The next stop or edge, whichever comes first.
    pub fn horizon(&self) -> u64 {
        self.until_stop().min(self.until_edge())
    }

    /// Opens a pass: where it starts. The cycles since the last pass are its span.
    pub fn open(&mut self) -> Start {
        let mut start = Start { cycles: self.elapsed, since: self.elapsed, position: None };
        if self.passed > 0 {
            start.position = Some((self.schedule.phase, self.schedule.lines[self.passed - 1]));
            start.since = self.since_start;
        }
        start
    }

    /// The same start, `extra` cycles further on: a pass no one keeps (a look ahead).
    pub fn ahead(&self, extra: u64) -> Start {
        let mut start = Start { cycles: self.elapsed + extra, since: self.elapsed + extra, position: None };
        if self.passed > 0 {
            start.position = Some((self.schedule.phase, self.schedule.lines[self.passed - 1]));
            start.since = self.since_start + extra;
        }
        start
    }

    /// Closes the pass `open` started with its cut: the next pass, a schedule's first
    /// start, lies within the longest span.
    pub fn close(&mut self, cut: &Cut) {
        let sched = cut.schedule;
        assert!(sched.count == 0 || (sched.count <= MAX_STARTS && sched.first == cut.horizon), "a schedule of {} starts from {} with the horizon {}", sched.count, sched.first, cut.horizon);
        self.horizon = cut.horizon;
        assert!((1..=self.max_span).contains(&self.horizon), "horizon {} out of range", self.horizon);
        self.passed = 0;
        self.elapsed = 0;
        self.schedule = sched;
        self.edges = [(cut.drop > 0).then_some(cut.drop), cut.edge, cut.rise];
        self.owed = cut.owed;
    }

    /// `cycles` cycles run, no further than the next stop or edge, the steps stream stopped
    /// through them when `stopped`: the edges they end at, and what they reach.
    pub fn advance(&mut self, cycles: u64, stopped: bool) -> (Edges, Due) {
        let (due, edge) = (self.until_stop(), self.until_edge());
        assert!(cycles <= due.min(edge), "advanced {cycles} cycles past the next stop {due} or edge {edge}");
        if stopped {
            assert!(cycles <= self.owed, "{cycles} cycles stopped past the {} the passes emitted", self.owed);
            self.owed -= cycles;
        }
        self.elapsed += cycles;
        self.countdown -= self.countdown.min(cycles);
        self.since_start += cycles;
        let at = |i: usize| self.edges[i] == Some(self.elapsed);
        let at_edge = Edges { fall: at(0) || at(1), reset: at(0), rise: at(2) };
        if cycles < due {
            return (at_edge, Due::Nothing);
        }
        if self.passed < self.schedule.count {
            let line = self.schedule.lines[self.passed];
            self.passed += 1;
            self.countdown = if self.passed < self.schedule.count { self.schedule.period } else { self.schedule.tail };
            self.since_start = 0;
            return (at_edge, Due::Start(line));
        }
        (at_edge, Due::Pass)
    }
}

/// When a battery save is due, from the model's own outputs: a step whose enabling window
/// leaves the saved regions after a write to them finishes a save.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Battery {
    enabled: Option<bool>,
    dirty: bool,
    committed: bool,
}

impl Battery {
    /// At a hand-off: what was written and not yet handed over (`written`), and whether
    /// it finished (`committed`).
    pub fn seeded(written: bool, committed: bool) -> Battery {
        Battery { enabled: None, dirty: written && !committed, committed }
    }

    /// One step: whether its window enables the saved regions, and whether it wrote to one.
    pub fn step(&mut self, enabled: bool, wrote: bool) {
        if self.enabled.replace(enabled) == Some(true) && !enabled && self.dirty {
            (self.dirty, self.committed) = (false, true);
        }
        self.dirty |= wrote;
    }

    /// A finished save since the last take.
    pub fn take_committed(&mut self) -> bool {
        std::mem::take(&mut self.committed)
    }

    /// Anything written since the last take, finished or not.
    pub fn take_unsaved(&mut self) -> bool {
        let any = self.committed || self.dirty;
        (self.committed, self.dirty) = (false, false);
        any
    }
}

/// Whether a copy's byte at physical address `addr` shares its line of 2^`line_bits` bytes
/// with a write record of the step not landed yet (`staged`, their addresses): the record
/// would land over the byte, or the copy would read the line before the write. Both
/// programs stop the run on it (`docs/hardware-tests.md` §4.1 argues it unreachable while
/// the transfer blocks the write).
pub fn staged_in_line(staged: impl IntoIterator<Item = u32>, line_bits: u8, addr: u32) -> bool {
    staged.into_iter().any(|r| r >> line_bits == addr >> line_bits)
}

/// Whether `p` lies in one of `regions`.
pub fn within(regions: &[(u32, u32)], p: u32) -> bool {
    regions.iter().any(|&(a, b)| (a..b).contains(&p))
}

/// The output's high-pass filter over each side of a mixed sample: `input = mix · gain`,
/// `out = input − (charge >> shift)`, the charge taking `out`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HighPass {
    pub charge: [i32; 2],
    pub gain: i32,
    pub shift: u8,
}

impl HighPass {
    pub fn apply(&mut self, mix: [i16; 2]) -> [i16; 2] {
        let mut out = [0i16; 2];
        for side in 0..2 {
            let input = mix[side] as i32 * self.gain;
            let filtered = input - (self.charge[side] >> self.shift);
            self.charge[side] += filtered;
            out[side] = filtered.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        out
    }
}

/// A run's outputs as files in one directory: every frame's hash (`frames.txt`, one
/// [`frame_sha1`] a line), every sample played (`audio.raw`, i16 little-endian, left then
/// right) and every save's bytes (`save-<n>.bin`). The full program (`GBCFORMER_OUTPUTS`)
/// and the minimal runner (`--outputs`) write them through this one type, and the
/// runner's proof compares the two directories.
pub struct Outputs {
    dir: std::path::PathBuf,
    frames: std::fs::File,
    audio: std::fs::File,
    saves: u32,
}

impl Outputs {
    pub fn new(dir: &std::path::Path) -> Result<Outputs, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let file = |name: &str| std::fs::File::create(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()));
        Ok(Outputs { frames: file("frames.txt")?, audio: file("audio.raw")?, saves: 0, dir: dir.to_path_buf() })
    }

    pub fn frame(&mut self, frame: &[u32]) -> Result<(), String> {
        use std::io::Write;
        writeln!(self.frames, "{}", frame_sha1(frame)).map_err(|e| e.to_string())
    }

    pub fn audio(&mut self, samples: &[[i16; 2]]) -> Result<(), String> {
        use std::io::Write;
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.iter().flat_map(|v| v.to_le_bytes())).collect();
        self.audio.write_all(&bytes).map_err(|e| e.to_string())
    }

    pub fn save(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.saves += 1;
        std::fs::write(self.dir.join(format!("save-{}.bin", self.saves)), bytes).map_err(|e| e.to_string())
    }
}

/// SHA-1 of a frame's R, G, B bytes (each pixel's low three bytes, red first), the
/// `frame_sha1` the gates compare.
pub fn frame_sha1(frame: &[u32]) -> String {
    let rgb: Vec<u8> = frame.iter().flat_map(|p| p.to_be_bytes()[1..].to_vec()).collect();
    sha1_smol::Sha1::from(rgb).digest().to_string()
}

/// The segments stream's two countdowns, each a number the last pass emitted: the halves
/// to the next sample, counted down by each cycle's halves (the rate the spans stream's
/// pass emitted) until reached or passed, and the cycles after which the next segment ends
/// (0: none before its sample).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentClock {
    to_sample: u64,
    to_end: u64,
}

/// What a counted cycle does to the open segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Close {
    Open,
    /// It ends at the cycle the pass emitted.
    End,
    /// It ends in its sample.
    Sample,
}

impl SegmentClock {
    pub fn emitted(to_sample: u64, to_end: u64) -> SegmentClock {
        assert!(to_sample >= 1, "the model put the next sample {to_sample} halves away");
        SegmentClock { to_sample, to_end }
    }

    /// The pass emitted an end after the next cycle.
    pub fn ends_next(&self) -> bool {
        self.to_end == 1
    }

    /// One cycle of `rate` halves.
    pub fn cycle(&mut self, rate: u64) -> Close {
        assert!(rate >= 1, "a cycle of no halves");
        self.to_sample = self.to_sample.saturating_sub(rate);
        let end = self.ends_next();
        self.to_end = self.to_end.saturating_sub(1);
        match (self.to_sample, end) {
            (0, _) => Close::Sample,
            (_, true) => Close::End,
            _ => Close::Open,
        }
    }
}

/// One segment of the segments stream as its pass reads it: a write that opens it, the
/// events at its first cycle, its cycles at one speed, and the sample or the read that
/// ends it. `rate` is the halves of each of its cycles, which only the countdown reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segment {
    pub write: Option<(u8, u8)>,
    pub events: Edges,
    pub cycles: u64,
    pub fast: bool,
    pub rate: u64,
    pub sample: bool,
    pub read: Option<u8>,
}

impl Segment {
    fn is_empty(&self) -> bool {
        self.write.is_none() && self.events == Edges::default() && self.cycles == 0 && !self.sample && self.read.is_none()
    }
}

/// The numbers a segments pass emits that the cutter counts: the halves to the next
/// sample, the cycles after which the next segment ends, and the cycles of this segment
/// the pass left for the segments after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Emitted {
    pub to_sample: u64,
    pub to_end: u64,
    pub rest: u64,
}

/// What runs a segment's pass, for [`Cutter`].
pub trait SegmentPass {
    fn pass(&mut self, seg: &Segment) -> Emitted;
    /// The segment's pass left no cycles: its sample and its read are the segment's.
    fn keep(&mut self, seg: &Segment);
}

/// The segments stream's cut, one for both programs. A write opens a segment, a cycle's
/// events open one at that cycle, and a change of speed closes the open one (a segment
/// holds one speed); otherwise a segment closes only where the countdowns the last pass
/// emitted reach 0, or at a read. The cycles a pass leaves (`rest`) are counted again
/// from the cycle after the part it ran, with the countdowns it emitted, as if they had
/// just run; then the close that ran the pass is delivered once, so every close leaves
/// the open segment empty.
#[derive(Clone, Debug, Default)]
pub struct Cutter {
    seg: Segment,
    clock: SegmentClock,
    /// Cycles counted, and the first cycle of the open segment.
    at: u64,
    start: u64,
    /// The cycles (counted from 0) after which a segment ended where a pass emitted it.
    pub ended: Vec<u64>,
}

impl Cutter {
    /// The cutter after a first pass over no cycle, which emits the countdowns.
    pub fn new(p: &mut impl SegmentPass) -> Cutter {
        let mut c = Cutter::default();
        c.close(p);
        c
    }

    /// Cycles counted up to the open segment: every cycle before it has run in a pass.
    pub fn settled(&self) -> u64 {
        self.start
    }

    pub fn ends_next(&self) -> bool {
        self.clock.ends_next()
    }

    fn close(&mut self, p: &mut impl SegmentPass) {
        let fast = self.seg.fast;
        let seg = std::mem::replace(&mut self.seg, Segment { fast, ..Segment::default() });
        let e = p.pass(&seg);
        self.clock = SegmentClock::emitted(e.to_sample, e.to_end);
        assert!(e.rest <= seg.cycles, "a pass left {} of its {} cycles", e.rest, seg.cycles);
        self.start = self.at - e.rest;
        if e.rest == 0 {
            p.keep(&seg);
            return;
        }
        self.at = self.start;
        let mut last = Close::Open;
        for _ in 0..e.rest {
            last = self.count(p, seg.fast, seg.rate);
        }
        assert!(!seg.sample || last == Close::Sample, "the cycles a pass left reach no sample where its segment ended in one");
        if !seg.sample && (self.seg.cycles > 0 || seg.read.is_some()) {
            self.seg.read = seg.read;
            self.close(p);
        }
    }

    pub fn write(&mut self, p: &mut impl SegmentPass, address: u8, value: u8) {
        if !self.seg.is_empty() {
            self.close(p);
        }
        self.seg.write = Some((address, value));
    }

    /// A cycle's events, at its start: they open a segment. Returns whether a segment closed.
    pub fn events(&mut self, p: &mut impl SegmentPass, events: Edges, fast: bool) -> bool {
        if events == Edges::default() {
            return false;
        }
        let closed = self.seg.cycles > 0;
        if closed {
            self.close(p);
        }
        self.seg.events = events;
        self.seg.fast = fast;
        closed
    }

    /// One cycle at the speed `fast`, of `rate` halves. Returns how it left the segment.
    pub fn count(&mut self, p: &mut impl SegmentPass, fast: bool, rate: u64) -> Close {
        if fast != self.seg.fast {
            if self.seg.cycles > 0 {
                self.close(p);
            }
            self.seg.fast = fast;
        }
        assert!(self.seg.cycles == 0 || self.seg.rate == rate, "a segment of two rates");
        self.seg.rate = rate;
        self.seg.cycles += 1;
        self.at += 1;
        if self.clock.ends_next() {
            self.ended.push(self.at - 1);
        }
        let close = self.clock.cycle(rate);
        match close {
            Close::Open => {}
            Close::End => self.close(p),
            Close::Sample => {
                self.seg.sample = true;
                self.close(p);
            }
        }
        close
    }

    /// Closes the open segment with a read of `address`.
    pub fn read(&mut self, p: &mut impl SegmentPass, address: u8) {
        self.seg.read = Some(address);
        self.close(p);
    }
}

/// What one block of cycles, or one pass, asks of the host, applied after it in one order
/// ([`Signals::apply`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Signals {
    /// The fast speed: the segments stream's cycle takes two of the spans stream's.
    pub fast: bool,
    /// The halves each cycle of the block lasts, as the spans stream's pass emitted them.
    pub rate: u64,
    /// The block's last cycle is the edge the segments stream steps at.
    pub edge: bool,
    /// That fall came from a reset of the counter.
    pub reset: bool,
    /// The bit rose at the block's last cycle.
    pub rise: bool,
    /// The block's last cycle starts this line: draw it now.
    pub render: Option<u8>,
    /// The block's last cycle ends the frame: show it.
    pub frame: bool,
    /// The screen blanks.
    pub blank: bool,
    /// A byte a write reached in the lines stream's own memory: its index and value.
    pub own: Option<(u8, u8)>,
    /// Copies from the cycles before the block's last, landed before the line is drawn.
    pub early: Vec<Copy>,
    /// Copies after the block, in request order.
    pub copies: Vec<Copy>,
    /// Copies the spans stream holds to the end of the step that runs (its own output):
    /// they land when the step ends.
    pub held: Vec<Copy>,
    /// The lines stream is kept from its memory (Some(true)), or no longer (Some(false)).
    pub stopped: Option<bool>,
}

/// Where a block's signals go, as [`Signals::apply`] orders them.
pub trait Sink {
    fn land(&mut self, copies: &[Copy]);
    fn render(&mut self, line: u8);
    fn stopped(&mut self, stopped: bool);
    /// Copies held to the end of the step that runs.
    fn hold(&mut self, copies: &mut Vec<Copy>);
    fn frame(&mut self);
    fn blank(&mut self);
    fn own(&mut self, index: u8, value: u8);
}

impl Signals {
    /// Clears everything but the speed and the rate for the next block.
    pub fn reset(&mut self, fast: bool) {
        let (mut early, mut copies, mut held) = (std::mem::take(&mut self.early), std::mem::take(&mut self.copies), std::mem::take(&mut self.held));
        early.clear();
        copies.clear();
        held.clear();
        *self = Signals { fast, rate: self.rate, early, copies, held, ..Signals::default() };
    }

    /// Lands the early copies, keeping the rest for after them.
    pub fn land_early(&mut self, sink: &mut impl Sink) {
        if !self.early.is_empty() {
            let early = std::mem::take(&mut self.early);
            sink.land(&early);
            self.early = early;
            self.early.clear();
        }
    }

    /// The block's signals: the early copies, the line drawn as of the cycle it starts at,
    /// the stop, the frame, the blank, the own byte, the copies held to the step's end, then
    /// the copies, which stay in `copies` for the caller to hand on. Only a line and the
    /// bytes it reads share state, and the passes order those: the bytes before its cycle
    /// and the run's last byte by its own bit land early (`Run::last_early`). The rest
    /// reach state no other signal of the block reads or writes (the palette byte, the
    /// stop's hold, the frame shown, the screen blanked, memory), so their order decides
    /// nothing; a line drawn beside one that reaches what it reads stops the run.
    pub fn apply(&mut self, sink: &mut impl Sink) {
        assert!(self.render.is_none() || (self.own.is_none() && self.stopped.is_none() && !self.blank && !self.frame && self.held.is_empty() && !self.early.iter().chain(&self.copies).any(|c| matches!(c, Copy::Block { .. }))), "a line drawn in the block of an own byte, a stop, a blank, a frame or a block copy: {self:?}");
        assert!(!(self.frame && self.blank), "a frame shown in the block that blanks the screen");
        self.land_early(sink);
        if let Some(line) = self.render {
            sink.render(line);
        }
        if let Some(stopped) = self.stopped {
            sink.stopped(stopped);
        }
        if self.frame {
            sink.frame();
        }
        if self.blank {
            sink.blank();
        }
        if let Some((index, value)) = self.own {
            sink.own(index, value);
        }
        if !self.held.is_empty() {
            sink.hold(&mut self.held);
        }
        if !self.copies.is_empty() {
            let copies = std::mem::take(&mut self.copies);
            sink.land(&copies);
            self.copies = copies;
        }
    }
}

/// A host that runs its streams in blocks of cycles.
pub trait Blocks {
    /// How far the next block may run.
    fn horizon(&self) -> u64;
    /// Runs one block of `cycles`, the steps stream stopped through it when `stopped`, and
    /// applies its signals.
    fn block(&mut self, cycles: u64, stopped: bool);
    /// A stop the last step ran, now its pass.
    fn stop(&mut self);
    /// The cycles the steps stream stays stopped, as the last pass emitted them and the
    /// stopped cycles since counted down (`Spans::owed`).
    fn owed(&self) -> u64;
    /// The cycles from here to the next pass are a pause (the last pass's flag).
    fn paused(&self) -> bool;
    /// After a step's cycles: the copies held while it ran land.
    fn settle(&mut self);
}

/// `cycles` cycles in blocks no longer than the horizon. Within a step (`in_step`, the
/// devices run to an access's cycle) that is all. Otherwise the copies held while the step
/// ran land, a stop the step ran is a pass, and the steps stream stays stopped while the
/// passes say it is owed, counted down in blocks: (stalled, paused) cycles run, a pause's
/// by the passes' flag.
pub fn advance(host: &mut impl Blocks, cycles: u64, in_step: bool) -> (u64, u64) {
    let mut left = cycles;
    while left > 0 {
        let block = left.min(host.horizon());
        host.block(block, false);
        left -= block;
    }
    if in_step {
        return (0, 0);
    }
    host.settle();
    host.stop();
    let (mut stalled, mut paused) = (0, 0);
    loop {
        let owed = host.owed();
        if owed == 0 {
            return (stalled, paused);
        }
        let block = owed.min(host.horizon());
        *if host.paused() { &mut paused } else { &mut stalled } += block;
        host.block(block, true);
    }
}

/// A pass's copies into `s`: its run's bytes, and its block, held to the end of the step
/// that runs when its held bit says so.
pub fn transfers(bits: u8, asks: &Asks, s: &mut Signals) {
    if let Some(r) = asks.run {
        r.bytes(bits, &mut s.early, &mut s.copies);
    }
    if let Some((source, dest, len, held)) = asks.block {
        if held { &mut s.held } else { &mut s.copies }.push(Copy::Block { source, dest, len });
    }
}

/// What a pass of a stream of spans asks of the bus: its run's bytes, its block (source,
/// destination, length, held to the step's end), the renderer's entry points and the
/// palette byte a write reached.
#[derive(Clone, Copy, Debug, Default)]
pub struct Asks {
    pub run: Option<Run>,
    pub block: Option<(u16, u16, u16, bool)>,
    pub render: bool,
    pub frame: bool,
    pub blank: bool,
    pub own: Option<u8>,
}

/// The events a stream of spans runs on its own, beside the bus's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Own {
    /// The span to here, ended by the horizon or a cut.
    Tick,
    /// STOP.
    Stop,
}

/// What `Passes` counts, for a caller's statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counted {
    Block,
    Render,
    Scheduled,
    Frame,
    Cut,
    Stop,
    Tick,
}

/// A stream of spans as the bus drives it: the one way its passes hand the bus their
/// copies, the renderer's entry points and STOP's outcome. A stream supplies its pass, its
/// decoding of a pass's asks and of the state bits read here, and the countdown; the
/// full program's control model and the runner's spans stream both run these methods.
pub trait Passes {
    type Event: std::marker::Copy + std::fmt::Debug;
    type Out;
    /// What a pass needs from its caller (the runner's length stream), or `()`.
    type Cx<'a>: ?Sized;
    /// One pass at `event` after the span since the last.
    fn once(&mut self, event: Self::Event, cx: &mut Self::Cx<'_>) -> Self::Out;
    fn asks(out: &Self::Out) -> Asks;
    /// The event of a further pass the pass asked for at once, at the same cycle.
    fn follow(out: &Self::Out) -> Option<Self::Event>;
    fn own(&self, event: Own) -> Self::Event;
    fn countdown(&mut self) -> &mut Spans;
    /// The low address bits of a run's line.
    fn run_bits(&self) -> u8;
    /// The state says STOP holds the GBC CPU.
    fn held(&self) -> bool;
    /// The line the state names for the renderer.
    fn line(&self) -> u8;
    fn counted(&mut self, _what: Counted) {}

    /// A pass whose asks go to the bus, then each further pass a pass asks for, at the
    /// event it names. `write` is the byte written when the event is a write.
    fn pass_with(&mut self, event: Self::Event, s: &mut Signals, cx: &mut Self::Cx<'_>, write: Option<u8>) {
        let first = self.once(event, cx);
        let out = Self::asks(&first);
        let bits = self.run_bits();
        transfers(bits, &out, s);
        let mut next = Self::follow(&first);
        while let Some(event) = next {
            self.counted(Counted::Block);
            let further = self.once(event, cx);
            let more = Self::asks(&further);
            assert!(more.run.is_none() && !more.render && !more.frame && !more.blank, "a further pass asked for a run byte or the renderer");
            transfers(bits, &more, s);
            next = Self::follow(&further);
        }
        if out.render {
            s.render = Some(self.line());
            self.counted(Counted::Render);
        }
        if out.frame {
            s.frame = true;
            self.counted(Counted::Frame);
        }
        s.blank |= out.blank;
        if let Some(value) = write {
            s.own = out.own.map(|i| (i, value));
        }
    }

    /// A pass whose event can ask the bus for nothing but its run's bytes, which go to
    /// `landed` for the caller to land at once.
    fn pass_alone(&mut self, event: Self::Event, landed: &mut Vec<Copy>, cx: &mut Self::Cx<'_>) -> Self::Out {
        let owed = self.countdown().owed();
        let out = self.once(event, cx);
        assert_eq!(self.countdown().owed(), owed, "{event:?} changed the cycles the steps stream stays stopped");
        let asks = Self::asks(&out);
        if let Some(r) = asks.run {
            let mut last = Vec::new();
            r.bytes(self.run_bits(), landed, &mut last);
            landed.append(&mut last);
        }
        assert!(asks.block.is_none() && !asks.render && !asks.frame && !asks.blank && Self::follow(&out).is_none(), "{event:?} asked the bus for a copy, the renderer or a further pass");
        out
    }

    /// A pass over the span to an access the busy copy blocked, which ends its run there.
    fn cut(&mut self, s: &mut Signals, cx: &mut Self::Cx<'_>) {
        self.counted(Counted::Cut);
        let tick = self.own(Own::Tick);
        self.pass_with(tick, s, cx, None);
    }

    /// `pass_alone`, and whether the pass held (Some(true)) or released (Some(false)) the
    /// lines stream: the change it made to the state's held bit, as STOP's pass reports it.
    fn pass_holding(&mut self, event: Self::Event, landed: &mut Vec<Copy>, cx: &mut Self::Cx<'_>) -> Option<bool> {
        let was = self.held();
        self.pass_alone(event, landed, cx);
        let now = self.held();
        (now != was).then_some(now)
    }

    /// STOP's pass, and whether it held or released the GBC CPU.
    fn stop(&mut self, s: &mut Signals, cx: &mut Self::Cx<'_>) {
        self.counted(Counted::Stop);
        let was = self.held();
        let stop = self.own(Own::Stop);
        self.pass_with(stop, s, cx, None);
        if self.held() != was {
            s.stopped = Some(!was);
        }
    }

    /// Cycles run, the steps stream stopped through them when `stopped`: at the edge the
    /// APU steps, at a scheduled start the renderer draws the line the model named for it,
    /// at the horizon a pass.
    fn advance(&mut self, cycles: u64, stopped: bool, s: &mut Signals, cx: &mut Self::Cx<'_>) {
        let (edges, due) = self.countdown().advance(cycles, stopped);
        (s.edge, s.reset, s.rise) = (edges.fall, edges.reset, edges.rise);
        match due {
            Due::Nothing => {}
            Due::Start(line) => {
                s.render = Some(line);
                self.counted(Counted::Render);
                self.counted(Counted::Scheduled);
            }
            Due::Pass => {
                self.counted(Counted::Tick);
                let tick = self.own(Own::Tick);
                self.pass_with(tick, s, cx, None);
            }
        }
    }
}

/// The bus state's bits as a step's accesses see them: the locks (as the spans stream
/// holds them), the busy copy's bit, the bit that marks an access's own state, and the
/// width of one access's nibble (two to an input byte). The spans stream's locks hold the
/// reads' in `locks` and the writes' `write_shift` bits above them; `writes` marks the
/// slots (a bit each, in [`SLOTS`] order) that take the writes'.
#[derive(Clone, Copy, Debug)]
pub struct StateBits {
    pub locks: u8,
    pub busy: u8,
    pub replace: u8,
    pub shift: u8,
    pub writes: u8,
    pub write_shift: u8,
}

impl StateBits {
    /// The locks slot `s` takes, from the spans stream's.
    pub fn slot_locks(&self, locks: u8, s: usize) -> u8 {
        let shift = if self.writes >> s & 1 != 0 { self.write_shift } else { 0 };
        (locks >> shift) & self.locks
    }
}

/// A step's accesses on its cycles: read 0, read 1, write 0, write 1 (the data accesses),
/// then each fetched byte after the first (two).
pub const DATA: usize = 4;
pub const SLOTS: usize = DATA + 2;

/// A step's request as the access loop needs it: each access's cycle in [`SLOTS`] order (0
/// when the step makes none), which reads the step makes, and each access's mask, the bits
/// of the bus state the model reads at its cycle, from the request's own outputs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Asked {
    pub cycles: [u8; SLOTS],
    pub reads: [bool; 2],
    pub masks: [u8; SLOTS],
}

/// An access's mask: the busy copy's bit always, and the locks where the request's own
/// flag says the access reaches the regions they cover. Which bus the busy copy blocks
/// stays the model's, from its own input.
pub fn access_mask(bits: StateBits, locks: bool) -> u8 {
    bits.busy | if locks { bits.locks } else { 0 }
}

/// The bus state at a step's start: the locks while they hold, else those the step asked
/// for, and the busy copy's bit.
pub fn start_state(bits: StateBits, locks: Option<u8>, asked: u8, busy: bool) -> u8 {
    locks.unwrap_or(asked) & bits.locks | if busy { bits.busy } else { 0 }
}

/// A bus event a step's pass flagged (`wiring::Steps::events`), as the host that runs it
/// supplies it: the row the spans stream's query names there, and the rewrite pass at it.
pub trait Rewrites {
    fn row(&mut self) -> u8;
    fn rewrite(&mut self, row: u8, read: bool);
}

/// One bus event: the row query, then the rewrite pass when the row is not 0 (none).
pub fn bus_event(h: &mut impl Rewrites, read: bool) {
    let row = h.row();
    if row != 0 {
        h.rewrite(row, read);
    }
}

/// A step's opening as the host that runs it supplies it.
pub trait Opening {
    type Request;
    /// The step's request under the locks at its start (None: the host's copy lapsed),
    /// no access's own state set.
    fn request(&mut self, locks: Option<u8>) -> Self::Request;
    /// The request's mask for the first fetched byte after the opcode reads the locks.
    fn fetch_reads_locks(&self, req: &Self::Request) -> bool;
    /// The request asks for the locks the host could not give.
    fn wants_locks(&self, req: &Self::Request) -> bool;
    /// A pass for the locks now.
    fn lock_query(&mut self) -> u8;
}

/// A step's first request, the locks at its start and the locks it asked for. Where the
/// host's copy of the locks (`held`) lapsed and the first fetched byte's mask reads them,
/// the opcode, fetched at the step's start, reads under them too: a pass gives them and
/// the step asks again. A request that asks for the locks gets them from a pass.
pub fn open_step<O: Opening>(h: &mut O, held: Option<u8>) -> (O::Request, Option<u8>, u8) {
    let mut locks = held;
    let mut req = h.request(locks);
    if locks.is_none() && h.fetch_reads_locks(&req) {
        locks = Some(h.lock_query());
        req = h.request(locks);
    }
    let asked = if h.wants_locks(&req) { h.lock_query() } else { 0 };
    (req, locks, asked)
}

/// What a step's accesses need from the host that runs it: the step's request again under
/// each access's state, the bus state ahead and now, and the other streams run further
/// into the step.
pub trait Accesses {
    /// The step's request under `access`, a nibble per slot, two to a byte.
    fn request(&mut self, access: &[u8]) -> Asked;
    /// Read `read` of the last request, served now.
    fn serve(&mut self, read: usize) -> u8;
    /// The bus state `cycles` from now as the spans stream emitted it: the busy bit, and
    /// the locks with `locks` (a look-ahead pass where its spans lapse), a write's with
    /// `write`.
    fn ahead(&mut self, cycles: u8, locks: bool, write: bool) -> u8;
    /// The other streams run `cycles` further into the step.
    fn tick(&mut self, cycles: u8);
    /// The locks while they hold, the busy bit now, and a pass for the locks where they lapsed.
    fn locks(&mut self) -> Option<u8>;
    fn busy(&mut self) -> bool;
    fn lock_query(&mut self) -> u8;
}

/// A step's accesses on their cycles: where each falls, how far the step has run, and the
/// bus state predicted at each, held to the host's as the cycles reach it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timeline {
    pub done: u8,
    pub cycles: [u8; SLOTS],
    pub expect: [Option<(u8, u8)>; SLOTS],
}

impl Timeline {
    /// Runs the other streams to cycle `k` of the step, stopping at each access on the
    /// way to hold its predicted state.
    pub fn reach(&mut self, h: &mut impl Accesses, bits: StateBits, k: u8) {
        while self.done < k {
            let next = self.cycles.iter().copied().filter(|&c| c > self.done && c < k).min().unwrap_or(k);
            h.tick(next - self.done);
            self.done = next;
            for s in (0..SLOTS).filter(|&s| self.cycles[s] == next) {
                let Some((want, mask)) = self.expect[s] else { continue };
                let locks = if mask & bits.locks != 0 { h.locks().unwrap_or_else(|| h.lock_query()) } else { 0 };
                let got = (bits.slot_locks(locks, s) | if h.busy() { bits.busy } else { 0 }) & mask;
                assert_eq!(got, want, "the bus state at access {s}'s cycle {next} is {got:#x}; the host predicted {want:#x} (bits {mask:#x})");
            }
        }
    }
}

/// Each access's bus state where it differs from the step's `start`, as the spans stream
/// says it will be at the access's cycle, settled over the step's requests: the bits of
/// the access's mask from its cycle, the rest from the start. A busy copy started within the step blocks after its
/// last access (`docs/hardware-tests.md` §3.11). When the busy copy ends before an access,
/// the other streams run to it first, so the request that asks again reads its last bytes;
/// the reads before that access are served on the way, each at its own cycle under the
/// state it had (the copy still blocking). Every state is read as the spans stream said it
/// at the step's start, the cycles run since then taken off. Returns the accesses'
/// timeline, the access bytes the last request ran under, and the reads served early.
pub fn settle_accesses(h: &mut impl Accesses, bits: StateBits, first: Asked, start: u8) -> (Timeline, [u8; SLOTS / 2], [Option<u8>; 2]) {
    let Asked { cycles, mut reads, mut masks } = first;
    let mut early = [None; 2];
    let mut line = Timeline { cycles, ..Timeline::default() };
    let all = bits.locks | bits.busy;
    // Each slot's state ahead, and whether it was read with the locks.
    let mut ahead: [Option<(u8, bool)>; SLOTS] = [None; SLOTS];
    let mut access = [0u8; SLOTS / 2];
    let mut ticked = false;
    for round in 0.. {
        let mut nibble = [0u8; SLOTS];
        for s in (0..SLOTS).filter(|&s| cycles[s] > 0) {
            let mask = masks[s];
            let locks = mask & bits.locks != 0;
            let raw = match ahead[s] {
                Some((raw, with)) if with || !locks => raw,
                _ => {
                    assert!(cycles[s] >= line.done, "access {s} at cycle {} after the step ran to {}", cycles[s], line.done);
                    let raw = h.ahead(cycles[s] - line.done, locks, bits.writes >> s & 1 != 0);
                    ahead[s] = Some((raw, locks));
                    raw
                }
            };
            let state = raw & mask | start & !mask & all;
            line.expect[s] = Some((state & mask, mask));
            if state != start {
                nibble[s] = bits.replace | state;
            }
        }
        let next: [u8; SLOTS / 2] = std::array::from_fn(|i| nibble[2 * i] | nibble[2 * i + 1] << bits.shift);
        if next == access {
            break;
        }
        assert!(round < 4, "the access states {next:02x?} did not settle over the step's requests");
        access = next;
        let unblocked = (0..SLOTS).filter(|&s| nibble[s] & bits.replace != 0 && nibble[s] & bits.busy == 0).map(|s| cycles[s]).min();
        if !ticked
            && start & bits.busy != 0
            && let Some(end) = unblocked
        {
            for i in (0..2).filter(|&i| reads[i] && cycles[i] < end) {
                line.reach(h, bits, cycles[i]);
                early[i] = Some(h.serve(i));
            }
            line.reach(h, bits, end);
            ticked = true;
        }
        let again = h.request(&access);
        assert_eq!(again.cycles, cycles, "the access cycles changed when the step asked again under {access:02x?}");
        (reads, masks) = (again.reads, again.masks);
    }
    (line, access, early)
}
