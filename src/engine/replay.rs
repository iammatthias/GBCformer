//! A replay bench for the GBC CPU stream: a run records its passes once, and
//! `examples/replay` runs them again through a fresh engine, each pass's decoded tokens
//! held to the recorded ones.
//!
//! **Recording.** `GBCFORMER_REPLAY_OUT=<file>` makes the engine whose stream holds the GBC
//! CPU's slots (`PCL`) and a memory record, after `GBCFORMER_REPLAY_SKIP` passes (default
//! 0) from the cartridge's insertion, the memory image as it stands and then the next
//! `GBCFORMER_REPLAY_PASSES` passes (default 100,000) with every write record kept between
//! them. The file holds no ROM and no boot ROM byte: the replay takes the ROM from the
//! cartridge file, checked against the recorded SHA-1, and the boot ROM's block as zeros
//! (no read reaches it once FF50 has unmapped it; every pass's tokens checked say so).
//!
//! **The file**, little-endian: `GBCRPL01`, the image's ROM, RAM and palette lengths
//! (u32 each), the ROM's SHA-1 (20 bytes), the image's bytes from the ROM's end to the boot
//! ROM's block, its presence flags packed 8 a byte, the stream's sequence length (u32),
//! then the events: 1 a pass (u16 changed positions, each u16 position and u32 token, the
//! decoded positions as u16 count and u16 each when they differ from the last pass's or
//! u16::MAX, and the FNV-1a hash of the decoded tokens), 2 a write record (u32 byte
//! address, 8 line bytes), 0 the end.

use super::counters::{cpi, Counts};
use super::probe::Stage;
use super::Engine;
use crate::memory::image::Image;
use crate::memory::layout::{BOOT, LINE, LINE_BITS, TOTAL};
use crate::sequence::{Positions, Sequence};
use std::io::{BufWriter, Read, Write};

const MAGIC: &[u8; 8] = b"GBCRPL01";
const SAME_POSITIONS: u16 = u16::MAX;

/// One recorded event.
pub enum Event {
    /// A pass: the stream positions whose token changed since the last pass, the decoded
    /// positions (an index into `Trace::positions`), and the hash of the decoded tokens.
    Pass { changes: Vec<(u16, u32)>, positions: u32, out: u64 },
    Write { addr: u32, line: [u8; LINE] },
}

pub struct Trace {
    lens: [usize; 3],
    rom_sha1: [u8; 20],
    /// The image's bytes from the ROM's end to the boot ROM's block, and every presence flag.
    ram: Vec<u8>,
    present: Vec<bool>,
    pub seq: usize,
    pub positions: Vec<Vec<usize>>,
    pub events: Vec<Event>,
}

fn fnv(out: &[usize]) -> u64 {
    out.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100_0000_01b3))
}

/// `fnv` of a sequence's decoded tokens at `positions`, in their order.
fn fnv_at(decoded: &[usize], positions: &[usize]) -> u64 {
    positions.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &p| (h ^ decoded[p] as u64).wrapping_mul(0x100_0000_01b3))
}

/// The recording of one engine, from its cartridge's insertion.
pub(super) struct Recorder {
    path: std::path::PathBuf,
    skip: u64,
    left: u64,
    file: Option<BufWriter<std::fs::File>>,
    last_tokens: Vec<usize>,
    last_positions: Vec<usize>,
}

impl Recorder {
    /// A recorder when `GBCFORMER_REPLAY_OUT` is set and `slots` are the GBC CPU stream's.
    pub(super) fn from_env(slots: &[String]) -> Option<Box<Recorder>> {
        let path = std::env::var_os("GBCFORMER_REPLAY_OUT")?;
        if !slots.iter().any(|s| s == "PCL") {
            return None;
        }
        let num = |name: &str, default: u64| std::env::var(name).ok().map_or(default, |v| v.parse().unwrap_or_else(|e| panic!("{name}={v}: {e}")));
        Some(Box::new(Recorder { path: path.into(), skip: num("GBCFORMER_REPLAY_SKIP", 0), left: num("GBCFORMER_REPLAY_PASSES", 100_000), file: None, last_tokens: Vec::new(), last_positions: Vec::new() }))
    }

    /// Before a pass of `tokens` at `positions`, with the image as it stands.
    pub(super) fn pass(&mut self, image: &Image, tokens: &[usize], positions: &[usize]) {
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        if self.file.is_none() {
            self.start(image, tokens.len());
        }
        let f = self.file.as_mut().expect("started");
        let changes: Vec<(u16, u32)> = tokens.iter().enumerate().filter(|&(p, &t)| self.last_tokens[p] != t).map(|(p, &t)| (p as u16, t as u32)).collect();
        f.write_all(&[1]).unwrap();
        f.write_all(&(changes.len() as u16).to_le_bytes()).unwrap();
        for (p, t) in &changes {
            f.write_all(&p.to_le_bytes()).unwrap();
            f.write_all(&t.to_le_bytes()).unwrap();
        }
        if self.last_positions != positions {
            f.write_all(&(positions.len() as u16).to_le_bytes()).unwrap();
            for &p in positions {
                f.write_all(&(p as u16).to_le_bytes()).unwrap();
            }
            self.last_positions = positions.to_vec();
        } else {
            f.write_all(&SAME_POSITIONS.to_le_bytes()).unwrap();
        }
        self.last_tokens.copy_from_slice(tokens);
    }

    /// After the pass: its decoded tokens. False once the recording is complete.
    pub(super) fn decoded(&mut self, out: &[usize]) -> bool {
        let Some(f) = self.file.as_mut() else { return true };
        f.write_all(&fnv(out).to_le_bytes()).unwrap();
        self.left -= 1;
        if self.left > 0 {
            return true;
        }
        f.write_all(&[0]).unwrap();
        f.flush().unwrap();
        eprintln!("replay: recorded the passes to {}", self.path.display());
        false
    }

    pub(super) fn write(&mut self, addr: u32, line: [u8; LINE]) {
        if let Some(f) = self.file.as_mut() {
            f.write_all(&[2]).unwrap();
            f.write_all(&addr.to_le_bytes()).unwrap();
            f.write_all(&line).unwrap();
        }
    }

    fn start(&mut self, image: &Image, seq: usize) {
        let mut f = BufWriter::new(std::fs::File::create(&self.path).unwrap_or_else(|e| panic!("{}: {e}", self.path.display())));
        let (bytes, present, lens) = image.parts();
        f.write_all(MAGIC).unwrap();
        for l in lens {
            f.write_all(&(l as u32).to_le_bytes()).unwrap();
        }
        f.write_all(&sha1_smol::Sha1::from(&bytes[..lens[0]]).digest().bytes()).unwrap();
        f.write_all(&bytes[lens[0]..BOOT as usize]).unwrap();
        let packed: Vec<u8> = present.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |b, (i, &p)| b | (p as u8) << i)).collect();
        f.write_all(&packed).unwrap();
        f.write_all(&(seq as u32).to_le_bytes()).unwrap();
        self.last_tokens = vec![usize::MAX; seq];
        self.file = Some(f);
    }
}

impl Trace {
    pub fn read(path: &std::path::Path) -> Result<Trace, String> {
        let mut b = Vec::new();
        std::fs::File::open(path).and_then(|mut f| f.read_to_end(&mut b)).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = b.get(at..at + n).ok_or("the trace ends early")?;
            at += n;
            Ok(s)
        };
        if take(8)? != MAGIC {
            return Err(format!("{}: not a replay trace", path.display()));
        }
        let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
        let u16_ = |s: &[u8]| u16::from_le_bytes(s.try_into().unwrap());
        let mut lens = [0usize; 3];
        for l in &mut lens {
            *l = u32_(take(4)?) as usize;
        }
        let rom_sha1: [u8; 20] = take(20)?.try_into().unwrap();
        let ram = take(BOOT as usize - lens[0])?.to_vec();
        let packed = take(TOTAL as usize / 8)?;
        let present: Vec<bool> = (0..TOTAL as usize).map(|i| packed[i / 8] >> (i % 8) & 1 == 1).collect();
        let seq = u32_(take(4)?) as usize;
        let (mut positions, mut events): (Vec<Vec<usize>>, Vec<Event>) = (Vec::new(), Vec::new());
        loop {
            match take(1)?[0] {
                0 => break,
                1 => {
                    let n = u16_(take(2)?) as usize;
                    let mut changes = Vec::with_capacity(n);
                    for _ in 0..n {
                        let p = u16_(take(2)?);
                        changes.push((p, u32_(take(4)?)));
                    }
                    let m = u16_(take(2)?);
                    if m != SAME_POSITIONS {
                        let mut list = Vec::with_capacity(m as usize);
                        for _ in 0..m {
                            list.push(u16_(take(2)?) as usize);
                        }
                        positions.push(list);
                    }
                    let index = positions.len().checked_sub(1).ok_or("a pass before any positions")? as u32;
                    let out = u64::from_le_bytes(take(8)?.try_into().unwrap());
                    events.push(Event::Pass { changes, positions: index, out });
                }
                2 => {
                    let addr = u32_(take(4)?);
                    events.push(Event::Write { addr, line: take(LINE)?.try_into().unwrap() });
                }
                k => return Err(format!("event kind {k}")),
            }
        }
        Ok(Trace { lens, rom_sha1, ram, present, seq, positions, events })
    }

    pub fn passes(&self) -> usize {
        self.events.iter().filter(|e| matches!(e, Event::Pass { .. })).count()
    }

    /// The recorded image, its ROM from `rom` (checked against the recorded SHA-1).
    pub fn image(&self, rom: &[u8]) -> Result<Image, String> {
        let r = rom.get(..self.lens[0]).ok_or("the cartridge is shorter than the recorded ROM")?;
        if sha1_smol::Sha1::from(r).digest().bytes() != self.rom_sha1 {
            return Err("the cartridge is not the recorded one".into());
        }
        let mut bytes = Vec::with_capacity(TOTAL as usize);
        bytes.extend_from_slice(r);
        bytes.extend_from_slice(&self.ram);
        bytes.resize(TOTAL as usize, 0);
        Image::from_parts(bytes, self.present.clone(), self.lens)
    }
}

/// One replay's result.
pub struct Run {
    pub wall_ns: u64,
    pub counts: Option<super::counters::Counts>,
    /// Passes the engine refused (dense fallbacks), which the bench must see at 0.
    pub fallbacks: u64,
}

impl Engine {
    /// Replays `trace` from `image`: the engine starts fresh on it, runs every pass and
    /// keeps every write record, and each pass's decoded tokens must hash as recorded.
    /// Leaves the memory as the trace found it, so the next replay starts from the same
    /// image: every written line gets its recorded line back, and the state starts fresh.
    pub fn replay(&mut self, trace: &Trace, image: &Image) -> Result<Run, String> {
        assert!(self.memory.is_some(), "replay needs an inserted image");
        self.state.valid = false;
        let fallbacks = self.stats.fallback_passes;
        // Variant bit 0: the passes as a whole slice (`fill_into`), not through a sequence.
        let whole = self.variant & 1 != 0;
        let mut seq = Sequence::new(vec![0usize; trace.seq]);
        let lists: Vec<Positions> = trace.positions.iter().map(|l| Positions::new(l.clone())).collect();
        let mut out = Vec::new();
        let (start, c0) = (std::time::Instant::now(), super::counters::thread());
        let mut pass = 0usize;
        for e in &trace.events {
            match e {
                Event::Pass { changes, positions, out: want } => {
                    for &(p, t) in changes {
                        seq.set(p as usize, t as usize);
                    }
                    let list = &lists[*positions as usize];
                    let got = match whole {
                        true => {
                            crate::llama::Denoiser::fill_into(self, seq.tokens(), list, &mut out);
                            fnv(&out)
                        }
                        false => {
                            crate::llama::Denoiser::fill_seq(self, &mut seq, list);
                            fnv_at(seq.decoded(), list)
                        }
                    };
                    if got != *want {
                        return Err(format!("replay pass {pass}: the decoded tokens differ from the recording"));
                    }
                    pass += 1;
                }
                Event::Write { addr, line } => self.write_record(*addr, *line),
            }
        }
        let wall_ns = start.elapsed().as_nanos() as u64;
        let counts = super::counters::thread().zip(c0).map(|(c1, c0)| c1.since(c0));
        self.restore(trace, image);
        Ok(Run { wall_ns, counts, fallbacks: self.stats.fallback_passes - fallbacks })
    }

    /// After a replay: every written line gets its recorded line back, and the state starts fresh.
    fn restore(&mut self, trace: &Trace, image: &Image) {
        let mut restored = std::collections::HashSet::new();
        for e in &trace.events {
            if let Event::Write { addr, .. } = e
                && restored.insert(addr >> LINE_BITS)
            {
                self.write_record(*addr, image.line(addr >> LINE_BITS));
            }
        }
        self.state.valid = false;
    }
}

/// What a bracketed replay puts between two readings of the thread's counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bracket {
    /// Each sampled pass whole, then the span from it to the next pass: its hash check,
    /// the write records after it and the next pass's token changes.
    Pass,
    /// One stage of each sampled pass, the stages in turn.
    Stages,
    /// Each sampled pass run a second time at once, on the same tokens with no write
    /// record between: a pass in which nothing changed, whole.
    NullPass,
    /// One stage of that second pass.
    NullStages,
    /// No reading: each sampled pass's shape (`probe::SHAPE`) as its items.
    Shape,
}

/// What a pass did, as the engine's counters moved over it (`Sample::did`).
pub const DID: [&str; 10] = ["units", "pairs", "head reads", "moved positions", "layer-0 wires", "positions decoded", "own units", "layer-0 replans", "write records before", "circuit layers walked"];

/// One bracket of a bracketed replay. Every count holds one bracket's own cost, which
/// `empty` (two readings with nothing between) measures beside it.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub pass: u32,
    /// The pass's decoded positions, as an index into `Trace::positions`.
    pub kind: u32,
    pub stage: Option<Stage>,
    pub counts: Counts,
    pub empty: Counts,
    /// `Bracket::Pass`: the span to the next pass.
    pub between: Counts,
    pub items: [u32; 4],
    /// The bracketed pass by `DID`.
    pub did: [u32; 10],
}

/// The counters `DID` names, as they stand.
struct Snap {
    v: [u64; 8],
    layers: [u64; 64],
}

impl Snap {
    fn of(s: &super::Stats) -> Snap {
        let mut layers = [0u64; 64];
        // The routing layer's entry is the last and is not a circuit layer.
        let n = s.layer_units.len().saturating_sub(1);
        layers[..n].copy_from_slice(&s.layer_units[..n]);
        Snap { v: [s.units, s.routed, s.memory_reads, s.layer0_moved, s.changes, s.decoded, s.own, s.layer0_replans], layers }
    }

    fn did(&self, after: &Snap, records: u32) -> [u32; 10] {
        let mut d = [0u32; 10];
        for i in 0..8 {
            d[i] = (after.v[i] - self.v[i]) as u32;
        }
        d[8] = records;
        d[9] = self.layers.iter().zip(&after.layers).filter(|(a, b)| a != b).count() as u32;
        d
    }
}

impl Engine {
    /// The fast plan's circuit layers (0 without a plan).
    pub fn circuit_layers(&self) -> usize {
        self.plan.as_ref().map_or(0, |p| p.t.circuit.len())
    }

    /// `replay`, with every pass whose index is `phase` modulo `every` bracketed as
    /// `bracket` says, each bracket pushed to `samples`. `turn` counts the stage brackets
    /// across calls, so the stages take turns evenly. The readings perturb the run, so
    /// its own totals are not returned.
    #[allow(clippy::too_many_arguments)]
    pub fn replay_bracketed(&mut self, trace: &Trace, image: &Image, bracket: Bracket, every: usize, phase: usize, turn: &mut usize, samples: &mut Vec<Sample>) -> Result<(), String> {
        assert!(self.memory.is_some(), "replay needs an inserted image");
        assert!(self.circuit_layers() < 64, "the layer counters hold 64 layers");
        self.state.valid = false;
        let fallbacks = self.stats.fallback_passes;
        let stages = Stage::all(self.circuit_layers());
        let mut seq = Sequence::new(vec![0usize; trace.seq]);
        let lists: Vec<Positions> = trace.positions.iter().map(|l| Positions::new(l.clone())).collect();
        let read = || cpi().expect("this platform gives no thread counters");
        let (mut pass, mut records) = (0usize, 0u32);
        // The sample whose span to the next pass is open, and that span's first reading.
        let mut open: Option<(usize, Counts)> = None;
        for e in &trace.events {
            match e {
                Event::Pass { changes, positions: kind, out: want } => {
                    if let Some((i, from)) = open.take() {
                        samples[i].between = read().since(from);
                    }
                    for &(p, t) in changes {
                        seq.set(p as usize, t as usize);
                    }
                    let positions = &lists[*kind as usize];
                    let sampled = pass % every == phase;
                    let null = matches!(bracket, Bracket::NullPass | Bracket::NullStages);
                    if !sampled || null {
                        crate::llama::Denoiser::fill_seq(self, &mut seq, positions);
                        if fnv_at(seq.decoded(), positions) != *want {
                            return Err(format!("replay pass {pass}: the decoded tokens differ from the recording"));
                        }
                    }
                    if sampled {
                        let stage = match bracket {
                            Bracket::Stages | Bracket::NullStages => {
                                *turn += 1;
                                Some(stages[(*turn - 1) % stages.len()])
                            }
                            Bracket::Shape => Some(Stage::Shape),
                            Bracket::Pass | Bracket::NullPass => None,
                        };
                        if let Some(s) = stage {
                            self.stats.probe.arm(s);
                        }
                        let before = Snap::of(&self.stats);
                        let c0 = read();
                        crate::llama::Denoiser::fill_seq(self, &mut seq, positions);
                        let c1 = read();
                        let c2 = read();
                        let did = before.did(&Snap::of(&self.stats), if null { 0 } else { records });
                        let reading = self.stats.probe.take();
                        if fnv_at(seq.decoded(), positions) != *want {
                            return Err(format!("replay pass {pass}: the decoded tokens differ from the recording"));
                        }
                        let sample = Sample { pass: pass as u32, kind: *kind, stage, counts: c1.since(c0), empty: c2.since(c1), between: Counts::default(), items: [0; 4], did };
                        match (stage, reading) {
                            (None, _) => {
                                samples.push(sample);
                                if bracket == Bracket::Pass {
                                    open = Some((samples.len() - 1, read()));
                                }
                            }
                            (Some(_), Some(r)) => samples.push(Sample { counts: r.counts, items: r.items, ..sample }),
                            // A stage the pass did not run (no positional stage, say).
                            (Some(_), None) => {}
                        }
                    }
                    pass += 1;
                    records = 0;
                }
                Event::Write { addr, line } => {
                    self.write_record(*addr, *line);
                    records += 1;
                }
            }
        }
        self.restore(trace, image);
        if self.stats.fallback_passes != fallbacks {
            return Err(format!("{} passes took the dense fallback ({:?})", self.stats.fallback_passes - fallbacks, self.stats.refused));
        }
        Ok(())
    }
}
