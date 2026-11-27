//! Test-only hooks for running a stream's passes in another runtime (`--features record`).
//!
//! **Recording.** `GBCFORMER_RECORD=<dir>` makes every engine with a view write
//! `<dir>/stream-<n>-<first slot>.jsonl`, one line per pass and per keep, in the
//! checkpoint's token space: the full sequence, the positions decoded and the tokens
//! decoded there, the memory position, and the line and its bytes every memory head of
//! the plan read. `GBCFORMER_RECORD_WINDOW=W` and `GBCFORMER_RECORD_EVERY=E` record the
//! first `W` events of every `E` (default: all).
//!
//! **Delegation.** A process that calls [`set_external`] before building its engines
//! hands every pass of every stream to the [`External`] the factory returns for it. The
//! engine then evaluates no unit and no head, and only keeps its record index (the
//! lines the records hold, which the external runtime prefills from). A cell is one
//! 8-byte line (`docs/stock-llama.md` §4).

use crate::memory::image::Image;
use crate::memory::layout::LINE;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// A runtime that computes a stream's passes in the checkpoint's token space.
pub trait External: Send {
    /// The tokens decoded at `positions` of the checkpoint's sequence `tokens`, with the
    /// memory's record index as it stands (for a stream with memory).
    fn fill(&mut self, tokens: &[usize], positions: &[usize], memory: Option<&Image>) -> Vec<usize>;
    /// Called when a cartridge is inserted. Every line of `image` becomes a cell.
    fn insert(&mut self, _image: &Image) {}
    /// Called when a write record is kept. The cell of byte address `addr`'s line now
    /// holds `line`. A GBC CPU write's address and line are the step's decoded tokens, a
    /// DMA write's line is its reader record's, and a HOST record's byte and a test's bus
    /// come from the host.
    fn write(&mut self, _addr: u32, _line: [u8; LINE]) {}
    /// Decodes a reader record, the line at `dest` with the byte at `source` over byte
    /// `dest mod LINE`, as the runtime does, or returns None to leave the read to the
    /// engine's index.
    fn read(&mut self, _dest: u32, _source: u32) -> Option<[u8; LINE]> {
        None
    }
}

/// Builds the external runtime for a stream from its slot names and input count, or
/// None to leave the stream on the engine.
pub type Factory = Box<dyn Fn(&[String], usize) -> Option<Box<dyn External>> + Send + Sync>;

static FACTORY: OnceLock<Factory> = OnceLock::new();

/// Installs the factory every engine with a view built afterwards consults.
pub fn set_external(f: Factory) {
    assert!(FACTORY.set(f).is_ok(), "the external factory is set once");
}

pub(super) fn external(slots: &[String], n_inputs: usize) -> Option<Box<dyn External>> {
    FACTORY.get().and_then(|f| f(slots, n_inputs))
}

pub(super) struct Recorder {
    out: BufWriter<File>,
    events: u64,
    window: u64,
    every: u64,
    /// Each event also writes every cell of a stream memory holding at most this many
    /// (`GBCFORMER_RECORD_CELLS`; 0: none), for a check over the whole cache.
    pub(super) cells_max: usize,
}

static STREAMS: AtomicUsize = AtomicUsize::new(0);

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Recorder {
    pub(super) fn open(slots: &[String], n_inputs: usize, at: &[Option<u32>]) -> Option<Recorder> {
        let dir = std::env::var_os("GBCFORMER_RECORD")?;
        let n = STREAMS.fetch_add(1, Ordering::Relaxed);
        let first: String = slots.first().map_or("none".into(), |s| s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect());
        let path = std::path::Path::new(&dir).join(format!("stream-{n}-{first}.jsonl"));
        let mut out = BufWriter::new(File::create(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())));
        let names: Vec<String> = slots.iter().map(|s| format!("\"{s}\"")).collect();
        // Each slot's position in the checkpoint's sequence (-1: none), which a comparison that
        // leaves slots out reads.
        let at: Vec<String> = at.iter().map(|p| p.map_or("-1".into(), |p| p.to_string())).collect();
        writeln!(out, "{{\"slots\":[{}],\"n_inputs\":{n_inputs},\"at\":[{}]}}", names.join(","), at.join(",")).unwrap();
        out.flush().unwrap();
        Some(Recorder { out, events: 0, window: env("GBCFORMER_RECORD_WINDOW", u64::MAX), every: env("GBCFORMER_RECORD_EVERY", u64::MAX), cells_max: env("GBCFORMER_RECORD_CELLS", 0) as usize })
    }

    /// Counts the next event and returns whether it is recorded.
    pub(super) fn due(&mut self) -> bool {
        let e = self.events;
        self.events += 1;
        e % self.every < self.window
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn line(&mut self, kind: &str, event: u64, position: usize, tokens: &[usize], positions: &[usize], out: &[usize], heads: &[(u32, [u8; LINE])], cells: &[(u32, [u8; LINE])]) {
        let join = |v: &[usize]| v.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        let lines = |v: &[(u32, [u8; LINE])]| v.iter().map(|(n, l)| format!("[{n},{}]", l.map(|b| b.to_string()).join(","))).collect::<Vec<_>>().join(",");
        let cells = if cells.is_empty() { String::new() } else { format!(",\"cells\":[{}]", lines(cells)) };
        writeln!(self.out, "{{\"kind\":\"{kind}\",\"event\":{event},\"position\":{position},\"tokens\":[{}],\"positions\":[{}],\"out\":[{}],\"heads\":[{}]{cells}}}", join(tokens), join(positions), join(out), lines(heads)).unwrap();
        // Streams run on threads a process exit does not join.
        self.out.flush().unwrap();
    }

    /// The number of the event `due` last counted.
    pub(super) fn last(&self) -> u64 {
        self.events - 1
    }
}
