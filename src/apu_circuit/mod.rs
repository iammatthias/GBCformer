//! The APU as a threshold circuit compiled into its own Llama checkpoint, run as the
//! machine's APU one segment per pass (`docs/apu-in-weights.md`). The reference it is
//! held to is `machine::apu`.
//!
//! A segment is at most one register write, at most one frame-sequencer step, M-cycles
//! at one speed that finish up to [`MAX_DELTA`] channel ticks (the model converts
//! M-cycles to ticks and keeps the tick's phase at double speed), then a sample or a CPU
//! read. [`ApuModel`] hands the M-cycle stream to `transport::Cutter`, and every pass turns the previous pass's state bytes and the segment's event bytes into the
//! next state, the sample's mix and the read's byte. The mix is the chip's digital
//! output. The board's high-pass filter after it is presentation on the host
//! (`machine::apu::HighPass`), whatever computes the APU. The frame
//! sequencer's steps come from the control logic (`Signals::edge`), and the speed is its
//! KEY1 bit, carried over. The model keeps the 48 kHz sample clock in its state and emits
//! the halves to the next sample (`SD`), where envelope steps wait for the next M-cycle
//! the M-cycles after which the next segment ends (`SE`), and the M-cycles of its
//! segment a pass that opens with an event leaves (`DL`); the host counts them down and
//! counts the left ones again (`transport::Cutter`).

pub(crate) mod build;
mod circuit;
mod events;
mod lfsr;
mod run;
mod write;
pub mod fields;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker;

use crate::checkpoint::{CheckpointDevice, SlotAbi};
use crate::circuit::Circuit;
use crate::device::{Device, Signals};
use crate::engine::Engine;
use crate::llama::Model;
use crate::machine::apu::{Apu, AudioDevice, Div, Mix, SAMPLE_CAPACITY};
use fields::{BYTES, State};
use std::collections::BTreeMap;
use crate::transport::{Close, Cutter, Edges, Emitted, SegmentPass};
use std::ops::Range;

pub use circuit::MAX_DELTA;

/// `EV`'s bits: the segment starts with a write, with the frame sequencer's fall (by a
/// reset of the counter with `EV_RESET`) or its rise, ends at a sample, and its M-cycles
/// run at double speed.
pub const EV_WRITE: u8 = 1;
pub const EV_FALL: u8 = 2;
pub const EV_SAMPLE: u8 = 4;
pub const EV_FAST: u8 = 8;
pub const EV_RESET: u8 = 16;
pub const EV_RISE: u8 = 32;

/// The input slots: the 63 state bytes (`fields::FIELDS` packed), then the events.
pub const INPUTS: [&str; BYTES + 5] = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47", "S48", "S49", "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57", "S58", "S59", "S60", "S61", "S62", "WA", "WV", "EV", "D", "RA"];

/// The result slots: the next state, the stereo mix (`Mix`, little-endian i16, left then
/// right), the read's byte, the halves to the next sample, the M-cycles of the next
/// segment after which it ends (0: none before its sample) and the M-cycles of this one
/// the pass left.
pub const RESULT: [&str; BYTES + 8] = ["S0'", "S1'", "S2'", "S3'", "S4'", "S5'", "S6'", "S7'", "S8'", "S9'", "S10'", "S11'", "S12'", "S13'", "S14'", "S15'", "S16'", "S17'", "S18'", "S19'", "S20'", "S21'", "S22'", "S23'", "S24'", "S25'", "S26'", "S27'", "S28'", "S29'", "S30'", "S31'", "S32'", "S33'", "S34'", "S35'", "S36'", "S37'", "S38'", "S39'", "S40'", "S41'", "S42'", "S43'", "S44'", "S45'", "S46'", "S47'", "S48'", "S49'", "S50'", "S51'", "S52'", "S53'", "S54'", "S55'", "S56'", "S57'", "S58'", "S59'", "S60'", "S61'", "S62'", "SL0", "SL1", "SR0", "SR1", "RV", "SD", "SE", "DL"];

/// The four-channel segment circuit, its slots checked against [`INPUTS`] and [`RESULT`].
pub fn build() -> Circuit {
    let b = circuit::build();
    assert_eq!(b.inputs, INPUTS, "the circuit's inputs are the ABI's");
    assert_eq!(b.outputs, RESULT, "the circuit's outputs are the ABI's");
    let mut c = b.c;
    c.schedule(crate::llama::UNIT_QUANTUM);
    let changes = changes(&c);
    // `fit` moves nodes only when the wires overflow its channel target, and where it
    // lands depends on how far below the scheduled live count the target sits. Try a few
    // targets and keep the narrowest MLP (clearing units included), then the fewest
    // (changed wire, reading layer) pairs; the schedule alone is a candidate.
    let live = c.live_per_boundary().into_iter().max().unwrap_or(0);
    let score = |c: &Circuit| {
        let mut units = c.units_per_level();
        for (_, k) in c.channels(&units).clears {
            units[k as usize] += 1;
        }
        let width = units.into_iter().max().unwrap_or(0).next_multiple_of(crate::llama::UNIT_QUANTUM);
        (width, c.changed_pairs(&changes))
    };
    let scheduled = c.levels.clone();
    // The fewest blocks of 1,024 channels whose budget the wires fit.
    for blocks in 1..=4 {
    let budget = Model::wire_budget(&c, false, blocks);
    let mut best = (score(&c), scheduled.clone());
    for below in [1, 16] {
        let target = budget.min(live.saturating_sub(below));
        if c.fit(crate::llama::UNIT_QUANTUM, &crate::circuit::Fit { channels: target, changes: &changes }) {
            let sc = score(&c);
            if sc.0 < best.0.0 || sc.0 == best.0.0 && sc.1 < best.0.1 {
                best = (sc, c.levels.clone());
            }
        }
        c.levels.clone_from(&scheduled);
    }
    c.levels = best.1;
    if c.live_per_boundary().into_iter().max().unwrap_or(0) <= budget {
        return c;
    }
    c.levels.clone_from(&scheduled);
    }
    panic!("the APU's {live} live wires fit no hidden size up to 4,096");
}

/// Per wire, its changes per fast pass: the mean of text.sh's and overworld.sh's model
/// windows on `--weights apu`, from the `apu wire changes` line `GBCFORMER_UNITS=1`
/// reports, keyed by `unique_names`. A wire the record lacks weighs a thousandth of a
/// change. Re-record after a circuit change.
fn changes(c: &Circuit) -> Vec<f64> {
    let by_name: std::collections::HashMap<&str, f64> = include_str!("changes.txt").lines().filter_map(|l| l.rsplit_once(' ')).filter_map(|(n, f)| Some((n, f.parse().ok()?))).collect();
    crate::circuit::unique_names(&c.names).iter().map(|n| by_name.get(n.as_str()).copied().unwrap_or(0.0) + 1e-3).collect()
}

/// The APU circuit compiled into its checkpoint and loaded into the fast engine.
pub fn engine() -> Engine {
    Engine::new(Model::compile(&build()))
}

/// One segment's inputs: its cycles are M-cycles, and its addresses the low byte of
/// FF10-FF3F (and FF76, FF77).
pub use crate::transport::Segment;

/// What one pass decodes. `sample` is meaningful when the segment ended in a sample,
/// `read` when it ended in a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub state: State,
    pub sample: Mix,
    pub read: u8,
    /// Halves (2-dot units) from the end of the segment to the next sample.
    pub to_sample: u8,
    /// M-cycles of the next segment after which it ends, 0 for none before its sample: 1
    /// after a counted fall at double speed, whose envelope steps wait for the next M-cycle.
    pub to_end: u8,
    /// M-cycles of this segment the pass left: all of them when it opens with a
    /// frame-sequencer event, whose pass runs the write and the event only.
    pub rest: u8,
}

/// The APU slot ABI: no request and no late inputs, so every segment is one pass.
pub struct Apu4;

impl SlotAbi for Apu4 {
    const INPUTS: &'static [&'static str] = &INPUTS;
    const REQUEST: &'static [&'static str] = &[];
    const RESULT: &'static [&'static str] = &RESULT;
    const LATE: Range<usize> = 0..0;
    type State = State;
    type Inputs = Segment;
    type Late = [u8; 0];
    type Request = ();
    type Outcome = Outcome;

    fn encode(state: &State, seg: &Segment, slots: &mut [u8]) {
        let most = if seg.fast { 2 * MAX_DELTA } else { MAX_DELTA };
        assert!(seg.cycles as i64 <= most, "a segment of {} M-cycles", seg.cycles);
        let (wa, wv) = seg.write.unwrap_or((0, 0));
        let e = seg.events;
        let ev = EV_WRITE * seg.write.is_some() as u8 | EV_FALL * e.fall as u8 | EV_SAMPLE * seg.sample as u8 | EV_FAST * seg.fast as u8 | EV_RESET * e.reset as u8 | EV_RISE * e.rise as u8;
        slots[..BYTES].copy_from_slice(state);
        slots[BYTES..].copy_from_slice(&[wa, wv, ev, seg.cycles as u8, seg.read.unwrap_or(0)]);
    }

    fn decode_request(_bytes: &[u8]) {}

    fn decode_outcome(bytes: &[u8]) -> Outcome {
        let s = &bytes[BYTES..];
        Outcome { state: bytes[..BYTES].try_into().unwrap(), sample: [i16::from_le_bytes([s[0], s[1]]), i16::from_le_bytes([s[2], s[3]])], read: s[4], to_sample: s[5], to_end: s[6], rest: s[7] }
    }

    fn needs_late(_request: &()) -> bool {
        false
    }
}

/// Whatever runs one segment: the checkpoint, or the circuit evaluated directly.
pub trait Segments: Send {
    fn run(&mut self, state: &State, seg: &Segment) -> Outcome;
    fn describe(&self) -> Vec<String> {
        Vec::new()
    }
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The checkpoint, stepped by the generic protocol: one pass, settled at once.
pub type ApuCheckpoint<M> = CheckpointDevice<Apu4, M>;

/// The engine's shape and its fast path, as the run's label prints them.
fn engine_lines(e: &Engine) -> Vec<String> {
    let c = &e.dense().cfg;
    let fast_path = e.exact().map_or_else(|err| format!("OFF ({err}): every pass runs dense"), |_| "on".into());
    vec![format!("apu checkpoint: {} layers, hidden {}, intermediate {}, {} heads, vocab {}, {:.1}M parameters ({} nonzero), integer fast path {fast_path}", c.layers, c.hidden, c.intermediate, c.heads, c.vocab, c.params() as f64 / 1e6, e.nonzeros())]
}

impl Segments for ApuCheckpoint<Engine> {
    fn run(&mut self, state: &State, seg: &Segment) -> Outcome {
        self.request(state, seg);
        self.execute(&[])
    }

    fn describe(&self) -> Vec<String> {
        engine_lines(self.model())
    }

    fn report(&self) -> Vec<String> {
        let e = self.model();
        let s = &e.stats;
        let mut lines = vec![format!("apu engine: {} passes, {} on the dense fallback", s.passes, s.fallback_passes), s.to_string()];
        let census = e.census();
        if !census.is_empty() {
            let top: Vec<String> = census.iter().take(40).map(|(n, u)| format!("{n} {u:.0}")).collect();
            lines.push(format!("apu gate updates per fast pass by node family: {}", top.join(", ")));
            let sources: Vec<String> = e.sources().iter().take(80).map(|(n, u)| format!("{n} {u:.1}")).collect();
            lines.push(format!("apu gate updates per fast pass by changed wire: {}", sources.join(", ")));
            let changes: Vec<String> = e.wire_changes().iter().map(|(n, f)| format!("{n} {f:.4}")).collect();
            lines.push(format!("apu wire changes per fast pass: {}", changes.join("; ")));
        }
        lines
    }
}

/// The circuit evaluated directly (`Circuit::eval`), for tests that need no compile.
pub struct CircuitSegments(pub Circuit);

impl Segments for CircuitSegments {
    fn run(&mut self, state: &State, seg: &Segment) -> Outcome {
        let mut slots = [0u8; BYTES + 5];
        Apu4::encode(state, seg, &mut slots);
        let input: BTreeMap<String, u8> = INPUTS.iter().map(|n| n.to_string()).zip(slots).collect();
        let out = self.0.eval(&input);
        let bytes: Vec<u8> = RESULT.iter().map(|n| out[*n]).collect();
        Apu4::decode_outcome(&bytes)
    }
}

/// Counters over a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub passes: u64,
    pub samples: u64,
    pub writes: u64,
    pub edges: u64,
    /// CPU reads of FF10-FF3F, each a pass of its own.
    pub reads: u64,
    /// Emulated time in 2-dot halves: two an M-cycle at normal speed, one at double.
    pub halves: u64,
}

/// The APU computed by a segment evaluator: the bus contract, with the segments cut by
/// `transport::Cutter`. The tick's phase at double speed, a frame-sequencer step waiting
/// for its tick and the sample clock are the model's state; the host counts down the
/// halves to the next sample and the cycles to the segment's end that each pass emits,
/// and counts again the cycles a pass leaves.
pub struct ApuModel<E: Segments> {
    core: Core<E>,
    cut: Cutter,
}

/// The evaluator and what its passes produced, run by the cutter.
struct Core<E: Segments> {
    eval: E,
    state: State,
    out: Option<Outcome>,
    samples: Vec<Mix>,
    /// The sample the last kept pass mixed, when its segment ended in one.
    last_sample: Option<Mix>,
    /// The byte the last sync's pass returned, valid until the next event.
    read: Option<(u8, u8)>,
    /// The last pass's input state and segment, for a divergence's report.
    last: (State, Segment),
    counts: Counts,
}

impl<E: Segments> SegmentPass for Core<E> {
    fn pass(&mut self, seg: &Segment) -> Emitted {
        self.last = (self.state, *seg);
        let out = self.eval.run(&self.state, seg);
        self.state = out.state;
        self.out = Some(out);
        self.counts.passes += 1;
        Emitted { to_sample: out.to_sample as u64, to_end: out.to_end as u64, rest: out.rest as u64 }
    }

    fn keep(&mut self, seg: &Segment) {
        let out = self.out.expect("a pass ran");
        self.last_sample = seg.sample.then_some(out.sample);
        if seg.sample {
            self.counts.samples += 1;
            if self.samples.len() < SAMPLE_CAPACITY {
                self.samples.push(out.sample);
            }
        }
        if let Some(a) = seg.read {
            self.read = Some((a, out.read));
        }
    }
}

impl<E: Segments> ApuModel<E> {
    /// Takes over from `reference` at its current M-cycle: its state, sample clock
    /// included, as the first pass's input, and the samples it mixed that nobody drained
    /// yet, so the stream continues across the hand-off. A first pass over no M-cycle
    /// emits the distance to the next sample.
    pub fn new(eval: E, reference: &mut Apu) -> Self {
        let mut samples = Vec::with_capacity(SAMPLE_CAPACITY);
        reference.take_samples(&mut samples);
        let mut core = Core { eval, state: fields::state_of(reference), out: None, samples, last_sample: None, read: None, last: ([0; BYTES], Segment::default()), counts: Counts::default() };
        let cut = Cutter::new(&mut core);
        core.counts = Counts::default();
        ApuModel { core, cut }
    }

    pub fn state(&self) -> &State {
        &self.core.state
    }

    pub fn last_sample(&self) -> Option<Mix> {
        self.core.last_sample
    }

    pub fn counts(&self) -> &Counts {
        &self.core.counts
    }

    /// M-cycles counted up to the open segment, every one of them run in a pass.
    pub fn settled(&self) -> u64 {
        self.cut.settled()
    }

    /// The M-cycles (counted from the hand-off) after which a segment ended where a pass
    /// emitted it, since the last call.
    pub fn take_ended(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.cut.ended)
    }

    /// A register write. Returns whether it closed a segment first.
    pub fn write_event(&mut self, addr: u16, value: u8) -> bool {
        self.core.read = None;
        self.core.counts.writes += 1;
        let passes = self.core.counts.passes;
        self.cut.write(&mut self.core, addr as u8, value);
        self.core.counts.passes != passes
    }

    /// The frame-sequencer events at the start of an M-cycle open a segment. Returns
    /// whether a segment closed.
    pub fn begin_cycle(&mut self, div: Div, double: bool) -> bool {
        self.core.read = None;
        if div.fall || div.rise {
            self.core.counts.edges += 1;
        }
        self.cut.events(&mut self.core, Edges { fall: div.fall, reset: div.reset, rise: div.rise }, double)
    }

    /// The M-cycle at the speed `double` says, of `rate` halves (the control logic's).
    /// Returns whether a sample closed the segment.
    pub fn end_cycle(&mut self, double: bool, rate: u64) -> bool {
        self.core.counts.halves += rate;
        self.cut.count(&mut self.core, double, rate) == Close::Sample
    }

    /// Closes the open segment with a read of `addr`, so the read sees every M-cycle
    /// ticked so far.
    pub fn read_event(&mut self, addr: u16) {
        self.core.counts.reads += 1;
        self.cut.read(&mut self.core, addr as u8);
    }
}

impl<E: Segments> Device for ApuModel<E> {
    fn read(&self, addr: u16) -> u8 {
        match self.core.read {
            Some((a, v)) if a == addr as u8 => v,
            _ => panic!("APU read of {addr:#06x} without a sync at this M-cycle: the bus must call AudioDevice::sync first"),
        }
    }

    fn write(&mut self, addr: u16, value: u8, _s: &mut Signals) {
        self.write_event(addr, value);
    }

    fn advance(&mut self, mcycles: u64, s: &mut Signals) {
        for k in 1..=mcycles {
            let div = if k == mcycles { Div::of(s) } else { Div::default() };
            self.begin_cycle(div, s.fast);
            self.end_cycle(s.fast, s.rate);
        }
    }
}

fn count_lines(c: &Counts) -> String {
    let frames = c.halves as f64 / crate::machine::HALVES_PER_FRAME as f64;
    let per = |n: u64| n as f64 / frames.max(1e-9);
    format!(
        "apu: {} segments ({:.1} per frame): {} samples, {} writes, {} frame-sequencer steps, {} CPU reads of FF10-FF3F ({:.2} per frame)",
        c.passes,
        per(c.passes),
        c.samples,
        c.writes,
        c.edges,
        c.reads,
        per(c.reads)
    )
}

impl<E: Segments> AudioDevice for ApuModel<E> {
    fn sync(&mut self, addr: u16) {
        self.read_event(addr);
    }

    fn take_samples(&mut self, out: &mut Vec<Mix>) {
        out.append(&mut self.core.samples);
    }

    fn describe(&self) -> Vec<String> {
        let mut lines = self.core.eval.describe();
        lines.push("apu: every sample's mix (the chip's digital output, before the board's high-pass filter), register readback and state byte comes from the model, which turns M-cycles into 4-dot ticks at the speed it is handed and keeps the 48 kHz sample clock, emitting the halves to each next sample, the M-cycles to each segment end it needs and the M-cycles a pass leaves, which the host counts down (`transport::Cutter`); the host hands over the frame-sequencer steps, the speed and the halves per M-cycle the control logic emits".into());
        lines
    }

    fn report(&mut self) -> Vec<String> {
        let mut lines = vec![count_lines(&self.core.counts)];
        lines.extend(self.core.eval.report());
        lines
    }
}

/// The model beside the reference, compared after every pass that settles the model at
/// the reference's M-cycle: the whole carried state, the sample's mix (before the host's
/// filter) when the segment ended in one, every read's byte, and the M-cycles after which
/// the model ended a segment against those after which the reference left envelope steps
/// waiting. The first difference stops the run.
pub struct ApuLockstep<E: Segments> {
    pub reference: Apu,
    pub model: ApuModel<E>,
    scratch: Vec<Mix>,
    /// M-cycles the reference ran since the hand-off, and those after which it left steps
    /// waiting that the model's settled M-cycles have not reached.
    cycles: u64,
    waits: std::collections::VecDeque<u64>,
    ended: std::collections::VecDeque<u64>,
}

impl<E: Segments> ApuLockstep<E> {
    pub fn new(reference: Apu, model: ApuModel<E>) -> Self {
        ApuLockstep { reference, model, scratch: Vec::new(), cycles: 0, waits: Default::default(), ended: Default::default() }
    }

    fn check(&self, at: &str) {
        let want = fields::state_of(&self.reference);
        if *self.model.state() != want {
            let last = &self.model.core.last;
            panic!("apu lockstep: state diverged at pass {} ({at}); fields (model/reference): {}", self.model.counts().passes, fields::diff(self.model.state(), &want) + &format!("\nsegment {:?}\nbefore: {:?}", last.1, fields::unpack(&last.0)));
        }
    }

    /// The ends the model emitted against the reference's waiting steps, over the M-cycles
    /// the model has settled.
    fn check_ends(&mut self) {
        self.ended.extend(self.model.take_ended());
        let settled = self.model.settled();
        let take = |q: &mut std::collections::VecDeque<u64>| {
            let mut v = Vec::new();
            while q.front().is_some_and(|&c| c < settled) {
                v.push(q.pop_front().unwrap());
            }
            v
        };
        let (got, want) = (take(&mut self.ended), take(&mut self.waits));
        assert_eq!(got, want, "apu lockstep: the model ended the segment after the M-cycles {got:?}, the reference left envelope steps waiting after {want:?}, at pass {}", self.model.counts().passes);
    }
}

impl<E: Segments> Device for ApuLockstep<E> {
    fn read(&self, addr: u16) -> u8 {
        let (want, got) = (self.reference.read(addr), self.model.read(addr));
        assert_eq!(got, want, "apu lockstep: read {addr:#06x} diverged at pass {}", self.model.counts().passes);
        got
    }

    fn write(&mut self, addr: u16, value: u8, _s: &mut Signals) {
        if self.model.write_event(addr, value) {
            self.check("before a write");
            self.check_ends();
        }
        self.reference.write(addr, value);
    }

    fn advance(&mut self, mcycles: u64, s: &mut Signals) {
        for k in 1..=mcycles {
            let div = if k == mcycles { Div::of(s) } else { Div::default() };
            if self.model.begin_cycle(div, s.fast) {
                self.check("at a frame-sequencer event");
            }
            self.reference.mcycle(s.fast, div);
            if self.reference.steps_wait() {
                self.waits.push_back(self.cycles);
            }
            self.cycles += 1;
            self.reference.take_samples(&mut self.scratch);
            let sampled = self.model.end_cycle(s.fast, s.rate);
            match (sampled, self.scratch.pop()) {
                (false, None) => {}
                (true, Some(want)) => {
                    self.check("at a sample");
                    let got = self.model.last_sample().expect("a sample pass");
                    assert_eq!(got, want, "apu lockstep: sample diverged at pass {}", self.model.counts().passes);
                }
                (got, want) => panic!("apu lockstep: the model sampled {got}, the reference {want:?}, at pass {}", self.model.counts().passes),
            }
            self.check_ends();
        }
    }
}

impl<E: Segments> AudioDevice for ApuLockstep<E> {
    fn sync(&mut self, addr: u16) {
        self.model.read_event(addr);
        self.check("at a read");
        self.check_ends();
    }

    fn take_samples(&mut self, out: &mut Vec<Mix>) {
        self.model.take_samples(out);
    }

    fn describe(&self) -> Vec<String> {
        let mut lines = self.model.describe();
        lines.push("apu lockstep: machine::apu runs beside the model; state, mixes (before the filter), reads and the M-cycles each segment's emitted end falls after compared after every pass that settles the model".into());
        lines
    }

    fn report(&mut self) -> Vec<String> {
        let mut lines = self.model.report();
        lines.push(format!("apu lockstep: {} passes equal to the reference in state, mix, read and segment end", self.model.counts().passes));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The addresses a random stream writes, the triggers and wave RAM twice as often.
    const WRITTEN: [u16; 26] = [0xFF10, 0xFF11, 0xFF12, 0xFF13, 0xFF14, 0xFF16, 0xFF17, 0xFF18, 0xFF19, 0xFF1A, 0xFF1B, 0xFF1C, 0xFF1D, 0xFF1E, 0xFF20, 0xFF21, 0xFF22, 0xFF23, 0xFF24, 0xFF25, 0xFF26, 0xFF30, 0xFF35, 0xFF3F, 0xFF14, 0xFF23];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Random writes, reads, ticks and frame-sequencer steps through the circuit in lockstep with
    /// the reference; any difference in state, sample or read byte panics.
    fn random_stream<E: Segments>(eval: E, cgb: bool, seed: u64, events: usize) -> Counts {
        let mut reference = Apu::post_boot(cgb);
        let model = ApuModel::new(eval, &mut reference);
        let mut pair = ApuLockstep::new(reference, model);
        let mut rng = Rng(seed);
        // M-cycles since the last step: the control logic's steps are 2,048 or more apart.
        let mut since_step = 0u64;
        let mut s = Signals::default();
        for _ in 0..events {
            match rng.below(10) {
                0..=3 => {
                    let addr = WRITTEN[rng.below(WRITTEN.len() as u64) as usize];
                    let mut value = rng.next() as u8;
                    if addr == 0xFF26 && rng.below(4) != 0 {
                        value |= 0x80;
                    }
                    pair.write(addr, value, &mut s);
                }
                4 => {
                    let addr = if rng.below(6) == 0 { 0xFF76 + rng.below(2) as u16 } else { 0xFF10 + rng.below(0x30) as u16 };
                    pair.sync(addr);
                    pair.read(addr);
                }
                _ => {
                    let span = if rng.below(8) == 0 { 400 } else { 24 };
                    let m = 1 + rng.below(span);
                    // CGB blocks at either speed, so segments also close at a change of
                    // speed and blocks start in either half of a tick, some ending in a
                    // frame-sequencer step.
                    let double = cgb && rng.below(2) == 0;
                    s.reset(double);
                    s.rate = if double { 1 } else { 2 };
                    // Falls (some from a DIV reset) and rises, never two within a few
                    // M-cycles, as the control logic's are 1,024 or more apart.
                    let event = since_step + m > 8 && rng.below(3) == 0;
                    s.edge = event && rng.below(2) == 0;
                    s.reset = s.edge && rng.below(3) == 0;
                    s.rise = event && !s.edge;
                    since_step = if event { 0 } else { since_step + m };
                    pair.advance(m, &mut s);
                }
            }
        }
        *pair.model.counts()
    }

    #[test]
    fn random_register_traffic_matches_the_reference_on_dmg_and_cgb() {
        for (cgb, seed) in [(false, 0x9E37_79B9_7F4A_7C15), (true, 0xD1B5_4A32_D192_ED03)] {
            let c = random_stream(CircuitSegments(build()), cgb, seed, 6_000);
            assert!(c.samples > 1_000 && c.reads > 400 && c.writes > 2_000 && c.edges > 20, "{c:?}");
        }
    }

    #[test]
    fn a_read_closes_the_segment_and_sees_its_ticks() {
        let mut reference = Apu::post_boot(true);
        for (a, v) in [(0xFF1A, 0x80), (0xFF1D, 0xF0), (0xFF1E, 0x87)] {
            reference.write(a, v);
        }
        let mut m = ApuModel::new(CircuitSegments(build()), &mut reference);
        let mut s = Signals { rate: 2, ..Signals::default() };
        m.advance(3, &mut s);
        Device::advance(&mut reference, 3, &mut s);
        m.sync(0xFF30);
        assert_eq!(m.read(0xFF30), reference.read(0xFF30));
        m.sync(0xFF26);
        assert_eq!(m.read(0xFF26), 0xF5, "power, channels 1 and 3 on");
        assert_eq!(m.counts().passes, 2, "two reads, one pass each");
    }

    /// The circuit with one carried field forced after every pass that changed it.
    struct Mutant(CircuitSegments, &'static str);

    impl Segments for Mutant {
        fn run(&mut self, state: &State, seg: &Segment) -> Outcome {
            let mut out = self.0.run(state, seg);
            let mut f = fields::unpack(&out.state);
            if seg.cycles % 2 == 1 {
                *f.get_mut(self.1).unwrap() ^= 1;
            }
            out.state = fields::pack(&f);
            out
        }
    }

    #[test]
    fn the_lockstep_catches_a_wrong_tick_phase_or_waiting_step() {
        for field in ["half", "pend"] {
            let caught = std::panic::catch_unwind(|| random_stream(Mutant(CircuitSegments(build()), field), true, 0xD1B5_4A32_D192_ED03, 6_000)).is_err();
            assert!(caught, "{field} flipped: not caught");
        }
    }

    /// The circuit with its emitted end replaced: never one, or the host's old rule (a
    /// counted fall at double speed, whether or not the APU leaves steps waiting).
    struct EndMutant(CircuitSegments, fn(&Segment) -> u8);

    impl Segments for EndMutant {
        fn run(&mut self, state: &State, seg: &Segment) -> Outcome {
            Outcome { to_end: (self.1)(seg), ..self.0.run(state, seg) }
        }
    }

    #[test]
    fn the_lockstep_catches_an_end_the_model_did_not_emit() {
        let rules: [(&str, fn(&Segment) -> u8, &str); 2] = [("never", |_| 0, "apu lockstep"), ("the host's rule", |s| (s.events.fall && !s.events.reset && s.fast) as u8, "the model ended the segment")];
        for (name, rule, expected) in rules {
            let err = std::panic::catch_unwind(|| random_stream(EndMutant(CircuitSegments(build()), rule), true, 0xD1B5_4A32_D192_ED03, 6_000)).expect_err(name);
            let msg = err.downcast_ref::<String>().map_or("", String::as_str);
            assert!(msg.contains(expected), "the end by {name}: caught as {msg}");
        }
    }

    /// The circuit with the M-cycles it leaves replaced by none: the host's old rule that
    /// an event's segment holds no M-cycles only holds if the host cuts there itself.
    struct RestMutant(CircuitSegments);

    impl Segments for RestMutant {
        fn run(&mut self, state: &State, seg: &Segment) -> Outcome {
            Outcome { rest: 0, ..self.0.run(state, seg) }
        }
    }

    #[test]
    fn the_lockstep_catches_cycles_the_model_did_not_leave() {
        assert!(std::panic::catch_unwind(|| random_stream(RestMutant(CircuitSegments(build())), true, 0xD1B5_4A32_D192_ED03, 6_000)).is_err(), "rest forced to 0: not caught");
    }

    /// A segment that opens with an event, whatever M-cycles and end it carries, runs the
    /// write and the event only: the same state and countdowns as the event over no M-cycle,
    /// and all its M-cycles left.
    #[test]
    fn an_event_s_pass_leaves_its_cycles() {
        let mut eval = CircuitSegments(build());
        let mut reference = Apu::post_boot(true);
        for (a, v) in [(0xFF26, 0x80), (0xFF12, 0xF3), (0xFF14, 0x87), (0xFF1A, 0x80), (0xFF1E, 0x87)] {
            reference.write(a, v);
        }
        let state = fields::state_of(&reference);
        for fast in [false, true] {
            for events in [Edges { fall: true, ..Edges::default() }, Edges { fall: true, reset: true, rise: false }, Edges { rise: true, ..Edges::default() }] {
                let bare = Segment { events, fast, write: Some((0x12, 0xA5)), ..Segment::default() };
                let want = eval.run(&state, &bare);
                assert_eq!(want.rest, 0, "no M-cycles, none left");
                for (cycles, sample, read) in [(1, false, None), (7, true, None), (21, false, Some(0x26)), (13, true, None)] {
                    let got = eval.run(&state, &Segment { cycles, sample, read, ..bare });
                    assert_eq!((got.state, got.to_sample, got.to_end, got.rest), (want.state, want.to_sample, want.to_end, cycles as u8), "{events:?} fast {fast}, {cycles} M-cycles");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "without a sync")]
    fn a_read_without_a_sync_is_refused() {
        let m = ApuModel::new(CircuitSegments(build()), &mut Apu::post_boot(false));
        m.read(0xFF26);
    }
}
