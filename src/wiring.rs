//! How a runtime moves bytes between the one model's streams: the wiring table the
//! compiler writes into the GGUF next to the weights (`gbcformer.wiring`), and the
//! snapshot a run starts from. Both are data. This module names no device and holds no
//! rule of the machine the model computes; the `gbc` side fills them from each part's
//! slot ABI (`one_model::wiring`, `session::Session::snapshot`), and the minimal runner
//! (`tools/runner`, `docs/minimal-runner.md`) interprets them.
//!
//! A stream plays one of six roles, named by how its passes are cut:
//! - **steps**: each pass is one step of the machine, whose length in clock cycles is an
//!   output, and which reads and writes the other streams through ports;
//! - **spans**: each pass runs after the cycles since the last one, at an event or at the
//!   horizon the last pass emitted;
//! - **segments**: each pass runs the cycles and events since the last one, closed at the
//!   events the role lists;
//! - **lines**: each pass is triggered by the spans stream and yields one row of pixels;
//! - **length**: each pass runs as a line is queued for the lines stream and yields the
//!   length the spans stream may ask for, so it never waits for the lines stream.
//! - **stepper**: each pass is one step of the pixel FIFO, drawing a line the length
//!   stream's verdict marked, from the queue of display writes it took.
//!
//! Every slot is named as the stream's own slot (`StreamSpec::slots`): input names among
//! its inputs, output names among its outputs, so an input and an output may share a name.

use serde::{Deserialize, Serialize};

/// The GGUF metadata key holding [`Wiring`] as JSON.
pub const WIRING_KEY: &str = "gbcformer.wiring";

/// The schema of [`Wiring`] and [`Snapshot`] this build reads. Bump it with any change to
/// their fields or what a field means; a file written for another is refused, as is a
/// field this build does not know. 2: the steps role no longer carries `reread` (the
/// HALT bug's two reads are each the model's).
pub const WIRING_VERSION: u32 = 2;

/// The GGUF metadata key holding [`Manifest`] as JSON.
pub const MANIFEST_KEY: &str = "gbcformer.manifest";

/// What a one-model GGUF was built as, beside its tensors: the wiring's schema and digest,
/// the compiled checkpoint's tensor digest (`gguf::digest`, which `digest --gguf` holds the
/// file's expanded tensors to), the vocabulary and the llama.cpp build the file runs in.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub wiring_version: u32,
    pub wiring_sha1: String,
    pub tensors_sha1: String,
    pub vocab: usize,
    pub llama_cpp: String,
}

impl Manifest {
    /// The manifest and the wiring of the GGUF in `path`, checked: refused without a
    /// manifest (a file written before manifests), for another wiring schema, or when the
    /// wiring the file holds is not the one its manifest names.
    pub fn read(path: &std::path::Path) -> Result<(Manifest, Wiring), String> {
        let json = crate::gguf::read_string(path, MANIFEST_KEY).map_err(|e| format!("{e}: the GGUF predates manifests; export it again"))?;
        let m: Manifest = serde_json::from_str(&json).map_err(|e| format!("{MANIFEST_KEY}: {e}"))?;
        if m.wiring_version != WIRING_VERSION {
            return Err(format!("{}: written for wiring version {}, this build reads version {WIRING_VERSION}; export it again", path.display(), m.wiring_version));
        }
        let text = crate::gguf::read_string(path, WIRING_KEY)?;
        if sha1(&text) != m.wiring_sha1 {
            return Err(format!("{}: the wiring's digest {} is not the manifest's {}", path.display(), sha1(&text), m.wiring_sha1));
        }
        let w: Wiring = serde_json::from_str(&text).map_err(|e| format!("{WIRING_KEY}: {e}"))?;
        Ok((m, w))
    }

    /// Refuses a snapshot written by a program whose wiring or model is not this file's.
    pub fn check_snapshot(&self, s: &Snapshot) -> Result<(), String> {
        if s.wiring_sha1 != self.wiring_sha1 {
            return Err(format!("the snapshot was written for wiring {}, the model's is {}", s.wiring_sha1, self.wiring_sha1));
        }
        if s.tensors_sha1 != self.tensors_sha1 {
            return Err(format!("the snapshot was written by the model {}, this file is {} (write it with GBCFORMER_GGUF_ONLY on this file)", s.tensors_sha1, self.tensors_sha1));
        }
        Ok(())
    }
}

/// The SHA-1 of `text` in hexadecimal.
pub fn sha1(text: &str) -> String {
    sha1_smol::Sha1::from(text).digest().to_string()
}

/// An integer read from output bytes: `slots` little-endian, then `>> shift`, `& mask`
/// and `+ bias`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub slots: Vec<String>,
    pub shift: u8,
    pub mask: u64,
    pub bias: i64,
}

impl Field {
    pub fn of(slots: &[&str]) -> Field {
        Field { slots: slots.iter().map(|s| s.to_string()).collect(), shift: 0, mask: u64::MAX, bias: 0 }
    }

    pub fn masked(slots: &[&str], mask: u64, bias: i64) -> Field {
        Field { mask, bias, ..Field::of(slots) }
    }

    pub fn bits(slot: &str, shift: u8, mask: u64) -> Field {
        Field { shift, mask, ..Field::of(&[slot]) }
    }
}

/// One bit of a byte slot, as a mask.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Flag {
    pub slot: String,
    pub mask: u8,
}

impl Flag {
    pub fn of(slot: &str, mask: u8) -> Flag {
        Flag { slot: slot.into(), mask }
    }
}

/// An event a stream of spans takes: its code bytes (the `event` slots but the last,
/// which carries the event's argument).
pub type Code = [u8; 2];

/// A read or a write that reaches another stream through a port: where it goes, and for
/// each of the 2^`Ports::low_bits` low address bits, the address byte that stream takes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum Target {
    /// The spans stream: an event whose first code byte is the one given and whose second
    /// is the table's at the low bits, with the access's value (0 for a read) as its
    /// argument; a read and a write each have their own.
    Spans { read: Option<(u8, Vec<u8>)>, write: Option<(u8, Vec<u8>)> },
    /// The segments stream: a read or write of `address[low]`.
    Segments { address: Vec<u8> },
    /// The lines stream's own memory: the byte at `Lines::own_memory + index`.
    Own { index: u8 },
}

/// The port blocks: `base + code · 2^low_bits + low`, each code's [`Target`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ports {
    pub base: u32,
    pub low_bits: u8,
    pub codes: Vec<(u32, Target)>,
    /// Reads at these low bits of a code are indirect: when the spans pass names an
    /// own-memory index, the read takes the lines stream's own byte there.
    pub indirect_reads: Vec<(u32, Vec<u8>)>,
}

/// The steps stream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Steps {
    pub stream: usize,
    /// Output to input, from one step to the next.
    pub carry: Vec<(String, String)>,
    /// An input built from an output of the last step and the spans stream's signals.
    pub composed: Vec<Composed>,
    /// Output slots each pass decodes, by pass.
    pub step: Vec<String>,
    pub result: Vec<String>,
    pub fetch: Vec<String>,
    pub line: Vec<String>,
    pub second: Vec<String>,
    /// Output slots no pass of a run decodes; they hold their mask token.
    pub unread: Vec<String>,
    /// Inputs filled between the step pass and the result pass, a 0 byte as the mask.
    pub late: Vec<String>,
    /// The step pass asks for late bytes: a read the host serves (`served`), the spans
    /// stream's locks (`locks`); for a fetched byte the host serves (`fetched`).
    pub served: Vec<Flag>,
    pub locks: Flag,
    /// The bits of the spans stream's locks the late locks input takes.
    pub locks_mask: u8,
    pub fetched: Vec<Flag>,
    /// Each served read's physical address.
    pub read_at: Vec<Field>,
    pub fetch_at: Vec<Field>,
    /// Each enabled read's flag: an enabled read that lands in `blocked` ends the busy copy's run.
    pub reads: Vec<Flag>,
    /// A fetched byte's record goes to `host_block | (address & host_mask)`.
    pub host_block: u32,
    pub host_mask: u32,
    /// Writes: enabled, value, physical address, kept as a record, heard by a second port.
    pub writes: Vec<Flag>,
    pub value: Vec<String>,
    pub write_at: Vec<Field>,
    pub records: Vec<Flag>,
    pub seconds: Vec<Flag>,
    pub second_at: Vec<Field>,
    /// Each data access's cycle within its step (read 0, read 1, write 0, write 1; 0 when
    /// the step makes none). A busy copy write 0 starts blocks after the step's last access.
    pub access_at: Vec<Field>,
    /// Each access's flag, in `access_at`'s order and then `fetch_cycles`', that it reaches
    /// the regions the locks cover: the locks at its cycle then matter, and its mask takes
    /// `locks_bits` beside `busy_bit`.
    pub access_locks: Vec<Flag>,
    /// Each fetched byte after the first: the flag that the step fetches it and its cycle.
    pub fetch_cycles: Vec<(Flag, u8)>,
    /// The inputs that hand each access its bus state where it differs from the step's
    /// start: a nibble of `access_shift` bits per access, in `access_at`'s order and then
    /// `fetch_cycles`', two to a byte, its `replace` bit set when it replaces the start's
    /// state. A bus state is the spans stream's locks (`locks_bits`) and the busy copy's bit
    /// (`busy_bit`), each access's masked by `access_locks`.
    pub access_in: Vec<String>,
    pub access_shift: u8,
    pub replace: u8,
    pub locks_bits: u8,
    pub busy_bit: u8,
    /// The slots (a bit each, in `access_at`'s order and then `fetch_cycles`') that take the
    /// writes' locks, which sit `write_shift` bits above the reads' in the spans stream's.
    pub write_slots: u8,
    pub write_shift: u8,
    /// The IF bits acknowledged and the step's M-cycles, each the low bits of its slot.
    pub ack: Field,
    pub cycles: Field,
    pub halted: Flag,
    pub stop: Flag,
    /// The first byte of each window the step names (its last memory map).
    pub windows: Vec<Field>,
    /// A copy line's inputs (source low, high, destination low, high) and its two
    /// translated addresses; a line is 2^`line_bits` bytes.
    pub line_in: Vec<String>,
    pub line_at: Vec<Field>,
    pub line_bits: u8,
    /// A copy with no source reads `fill_block | (destination & fill_mask)`.
    pub fill_block: u32,
    pub fill_mask: u32,
    /// An access landing in this block is one the busy copy blocks: the spans stream ends
    /// its run there.
    pub blocked: (u32, u32),
    /// A save from a cartridge that counts wall time ends with a footer of `footer_len`
    /// bytes: bit `bit` of the spans stream's state input `slot` at bit `to_bit` of byte
    /// `to_byte`, and the wall time last read, little-endian, at `time_at`.
    pub footer_len: usize,
    pub footer_bits: Vec<(String, u8, usize, u8)>,
    pub time_at: usize,
    /// Battery saves: the memory window that enables them, the regions it enables, and
    /// where the saved bytes start.
    pub save_window: usize,
    pub save_regions: Vec<(u32, u32)>,
    pub save_base: u32,
    /// The bus events a step's pass flags, each a row query of the spans stream
    /// (`Spans::row_query`) and, when the row it returns is not 0, a pass of its own.
    pub events: Events,
}

/// A step's bus events: addresses on the bus at which the row the spans stream names
/// there is rewritten (a DMG's OAM corruption bug). In cycle order: each fetch's, at its
/// cycle (the first at the step's start); each read's, before it is served; each idle
/// cycle's, at `idle_at`; each write's, before its record lands. A read or a fetch sets
/// `read_bit` in the event byte (`input`) beside the row. The pass decodes `pass` and
/// keeps the record each `rows` flag names (its number), whose address field gives the
/// row it rewrites.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Events {
    pub fetches: Vec<Flag>,
    pub reads: Vec<Flag>,
    pub idle: Vec<Flag>,
    pub idle_at: Field,
    pub writes: Vec<Flag>,
    pub input: String,
    pub read_bit: u8,
    pub pass: Vec<String>,
    pub rows: Vec<(Flag, u8, Field)>,
}

/// An input of the steps stream built from parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Composed {
    pub slot: String,
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum Part {
    /// The last step's output byte, masked.
    Carry { from: String, mask: u8 },
    /// A byte of the spans stream's state input.
    State { slot: String },
    /// The spans stream's busy flag (its last pass's `Spans::busy`) as `bit`.
    Busy { bit: u8 },
    /// Output slot `slot` of the spans stream's last pass.
    Emitted { slot: String },
    /// The spans stream's locks while they hold: `present | (locks & mask) << shift`.
    Locks { present: u8, mask: u8, shift: u8 },
}

/// The spans stream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Spans {
    pub stream: usize,
    /// Output to input, pass to pass: the state.
    pub carry: Vec<(String, String)>,
    /// The position inputs a scheduled start overrides: the phase and the line.
    pub phase: String,
    pub position: String,
    /// The cycles since the last pass, or as a pause's, and the cycles the position moves,
    /// each little-endian bytes.
    pub cycles: Vec<String>,
    pub paused_cycles: Vec<String>,
    pub moved: Vec<String>,
    /// The event's code bytes, then its argument.
    pub event: Vec<String>,
    /// Inputs every pass holds at a fixed byte (a link with nothing plugged in).
    pub fixed: Vec<(String, u8)>,
    /// The late input (a line's length from the lines stream) and the flag asking for it.
    /// The request slots, decoded with the result in the first pass.
    pub request: Vec<String>,
    pub late: String,
    pub asks: Flag,
    /// Event codes.
    pub tick: Code,
    pub ack: Code,
    pub raise: Code,
    pub buttons: Code,
    pub stop: Code,
    /// The steps stream came to rest while the state says a block copy runs (`idle` clear),
    /// and it woke from a rest the state keeps (`kept`): each a pass.
    pub rest: Code,
    pub wake_up: Code,
    /// A step of the steps stream ended while the state says a block copy waits to be
    /// counted (`uncounted`): a pass.
    pub step_end: Code,
    /// The wall time elapsed: one elapse event for each binary digit k (below `digits`) set in
    /// its seconds, k its argument.
    pub elapse: Code,
    pub digits: u32,
    pub lock_query: Code,
    /// The event whose read is the row a step's bus event rewrites (`Steps::events`), 0
    /// for none.
    pub row_query: Code,
    /// The read's byte, and the cycles to the next pass.
    pub read: Field,
    pub horizon: Field,
    /// The cycles the busy copy still blocks for after the pass, and the cycles before a
    /// copy the pass started blocks: it blocks from `busy_from` up to `busy_left`.
    pub busy_left: Field,
    pub busy_from: Field,
    pub serial_start: Flag,
    /// The busy flag the steps stream reads (`Part::Busy`).
    pub busy: Flag,
    pub render: Flag,
    pub frame: Flag,
    pub blank: Flag,
    /// The cycles since this pass are a pause.
    pub pause: Flag,
    /// A run of copied bytes: source page, line, a bit per byte for the cycles before the
    /// last and for the last, the destination page and the fill byte (nonzero: no source).
    pub run: Flag,
    pub run_page: Field,
    pub run_line: Field,
    /// The last cycle's byte lands before a line that cycle starts is drawn.
    pub run_last_early: Flag,
    pub run_before: Field,
    pub run_last: Field,
    pub run_dest: Field,
    pub run_fill: Field,
    /// A run moves bytes within one line of 2^`run_bits` bytes.
    pub run_bits: u8,
    /// A block copy: source, destination, length.
    pub block_copy: Flag,
    pub block_source: Field,
    pub block_dest: Field,
    pub block_len: Field,
    /// The block lands at the end of the step that runs (the pass's own output), and the
    /// input that tells a pass whether a step runs.
    pub block_held: Flag,
    pub step_input: String,
    /// The cycles the steps stream stays stopped once the step that runs has ended: a
    /// total each pass emits, from the count it took in `owed` (little-endian bytes).
    pub stall: Field,
    pub owed: Vec<String>,
    /// An own-memory index a write event reaches.
    pub own: Flag,
    pub own_index: Field,
    /// Scheduled starts: the first, count, period, the cycles from the last to the next
    /// pass, phase, each start's line.
    pub first: Field,
    pub count: Field,
    pub period: Field,
    pub tail: Field,
    pub start_phase: Field,
    pub lines: Vec<Field>,
    /// Where the segments stream's edge falls by counting (unless held), and the cycles to
    /// where the edge a reset dropped falls (0: none); the halted steps stream's wake.
    pub edge: Field,
    pub edge_held: Flag,
    pub edge_drop: Field,
    /// Where the segments stream's bit next rises by counting (unless held).
    pub rise: Field,
    pub wake: Flag,
    /// Real time per cycle in half units, and the rate that means the fast speed.
    pub rate: Field,
    pub fast_rate: u64,
    /// The rates a pass may emit.
    pub rates: Vec<u64>,
    /// Real time a frame lasts, in half units.
    pub frame_time: Field,
    /// The locks after the pass, each with the cycles from the pass it holds to.
    pub locks: Vec<(Field, Field)>,
    /// The first code byte of a further pass at once (0: none); the pass takes no other
    /// code byte or argument.
    pub follow: Field,
    /// The state input flags: no block copy runs, so a rest is no pass; a rest is kept, so
    /// its wake is one.
    pub idle: Flag,
    pub kept: Flag,
    /// The state input flag: a block copy waits for the step's end to be counted.
    pub uncounted: Flag,
    /// The state input flag that keeps the lines stream from its memory.
    pub held: Flag,
    /// The state input the serial log records at a start.
    pub serial: String,
    pub max_span: u64,
}

/// The segments stream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Segments {
    pub stream: usize,
    pub carry: Vec<(String, String)>,
    pub write_address: String,
    pub write_value: String,
    pub events: String,
    pub on_write: u8,
    pub on_fall: u8,
    pub on_reset: u8,
    pub on_rise: u8,
    pub on_sample: u8,
    pub on_fast: u8,
    pub cycles: String,
    pub read_address: String,
    /// Outputs: the sample's two sides (little-endian i16), the read byte, the cycles to
    /// the next sample in halves, the cycles of the next segment after which it ends (0:
    /// none before its sample), and the cycles of its own segment the pass left.
    pub sample: Vec<Field>,
    pub read: String,
    pub to_sample: String,
    pub to_end: String,
    pub left: String,
    pub max_cycles: u64,
    pub max_fast_cycles: u64,
}

/// The lines stream.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Lines {
    pub stream: usize,
    /// Inputs: the line, the window's counter, the flag byte.
    pub line: String,
    pub counter: String,
    pub flags: String,
    pub armed: u8,
    pub features: u8,
    pub high: u8,
    pub compat: u8,
    pub held: u8,
    /// Outputs: each pixel, the window's counter and arming after the line, its length.
    pub pixels: Vec<String>,
    pub counter_out: String,
    pub armed_out: String,
    pub armed_mask: u8,
    /// The flag bit of the window carried into the line, and its bit in `armed_out`.
    pub carried: u8,
    pub carried_mask: u8,
    pub length: String,
    /// Where the own-memory bytes a port names sit in the lines stream's memory.
    pub own_memory: u32,
    /// A record at `mode_cell` is no memory byte: its bit `mode_mask` picks the line
    /// flags from the snapshot's `line_modes`.
    pub mode_cell: u32,
    pub mode_mask: u8,
    /// The memory ranges whose changed bytes the lines stream follows, beside the mode cell.
    pub follows: Vec<(u32, u32)>,
}

/// The length stream: a pass per line the lines stream draws, on the scheduler's thread
/// as the line is queued, over its own memory, which takes the same records but the ones
/// in `skip`. It gives the spans stream each line's length. While a line is open, a pass
/// per own-memory write judges it: whether the stepper draws the line, whether the
/// stepper's queue takes the write, and the write's time, which that queue carries.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Length {
    pub stream: usize,
    pub line: String,
    pub flags: String,
    pub armed: u8,
    pub carried: u8,
    pub length: String,
    pub armed_out: String,
    pub armed_mask: u8,
    pub carried_mask: u8,
    pub skip: (u32, u32),
    /// The write's inputs: half units since the line was triggered, its own-memory index
    /// and its value; the flag bit of a pass that judges one.
    pub write: [String; 3],
    pub write_flag: u8,
    /// The verdict's output and its bits (the line is the stepper's; the queue takes the
    /// write), and the write's time, two bytes little-endian.
    pub verdict: String,
    pub marks: u8,
    pub takes: u8,
    pub time: [String; 2],
}

/// The stepper: the lines the length stream's verdicts mark are drawn by it, a pass a
/// step, over its own memory, which follows the lines stream's records. A line's first
/// pass carries the `start` flag and the window before the line; each later pass carries
/// every `carry` output to its input and the queue's head write, until a pass says `done`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stepper {
    pub stream: usize,
    /// Inputs: the line, the flag byte, the window's counter at the start.
    pub line: String,
    pub flags: String,
    pub counter: String,
    /// The flag bits: the first pass, the window armed and carried before the line, the
    /// fast rate, the stream held; the line's flags by the mode cell's bit are the
    /// snapshot's `stepper_modes`.
    pub start: u8,
    pub armed: u8,
    pub carried: u8,
    pub fast: u8,
    pub held: u8,
    /// The head write's inputs: its time (two bytes), its index, its value, and its changes
    /// applied so far, with `pending` set when there is a head write.
    pub write: [String; 5],
    pub pending: u8,
    /// Each output fed to its input at the next pass.
    pub carry: Vec<(String, String)>,
    /// The pass's report and its bits: the head write's next change applied, that change
    /// its last, a pixel out, the line done.
    pub report: String,
    pub applied: u8,
    pub finished: u8,
    pub pixel: u8,
    pub done: u8,
    /// The pixel's place and its two colour bytes.
    pub x: String,
    pub colour: [String; 2],
    /// Once done: the window's counter and its flags (`armed_mask`, `carried_mask`), and the
    /// line's length, two bytes, shifted right by `shift` and plus `bias` into the spans
    /// stream's unit.
    pub counter_out: String,
    pub window_out: String,
    pub armed_mask: u8,
    pub carried_mask: u8,
    pub length: [String; 2],
    pub shift: u8,
    pub bias: i64,
}

/// What the host shows and plays: presentation only.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Present {
    pub width: usize,
    pub height: usize,
    /// Two-bit shades, and 15-bit colours as three 5-bit channels widened by `widen`.
    pub shades: Vec<u32>,
    pub channel_shifts: Vec<u8>,
    pub widen: Vec<u8>,
    pub blank_shade: u32,
    pub blank_colour: u32,
    /// The board's high-pass filter: `input = mix · gain`, `out = input − (cap >> shift)`.
    pub gain: i32,
    pub shift: u8,
    /// The buttons by name, each its bit in the buttons byte.
    pub buttons: Vec<(String, u8)>,
}

/// The whole table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Wiring {
    pub steps: Steps,
    pub spans: Spans,
    pub segments: Segments,
    pub lines: Lines,
    pub length: Length,
    pub stepper: Stepper,
    pub ports: Ports,
    pub present: Present,
    pub power_on: PowerOn,
}

/// Where a run from power-on starts, built from the ROM and the boot ROM alone.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PowerOn {
    /// The ROM's two header bytes that name its board, and where the board's register
    /// cells and its RAM sit in the memory.
    pub header_type: usize,
    pub header_ram: usize,
    pub regs: u32,
    pub ram: u32,
    /// One per console, told apart by the boot ROM's length.
    pub consoles: Vec<Console>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Console {
    pub boot_len: usize,
    /// The boot image the memory holds while the boot ROM is mapped: ROM bank 0 with the
    /// boot ROM's bytes over it in these ranges of offsets.
    pub boot_overlay: Vec<(usize, usize)>,
    /// The power-on snapshot of a blank cartridge: its ROM cells and registers zero,
    /// the images and frames given here rather than as files.
    pub snapshot: Snapshot,
    pub memory: Sparse,
    pub own: Sparse,
    pub len: Sparse,
    pub frames: Vec<u32>,
    pub boards: Vec<Board>,
}

/// A board the header names, at one ROM length: its register cells at power-on, its
/// RAM's length and fill byte, the saved bytes' length and whether it counts wall time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Board {
    pub kind: u8,
    pub ram: u8,
    pub rom_len: usize,
    pub regs: Vec<u8>,
    pub ram_len: usize,
    pub ram_fill: u8,
    pub save_len: usize,
    pub clock: bool,
}

/// An image as its nonzero bytes and its runs of addresses that hold a token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Sparse {
    pub len: usize,
    pub bytes: Vec<(u32, u8)>,
    pub present: Vec<(u32, u32)>,
    pub lengths: [usize; 3],
}

/// Where a run starts: every stream's first input bytes by slot, the memories' bytes,
/// and the facts of the cartridge and console a run keeps (all data a full program
/// writes at its hand-off, `GBCFORMER_SNAPSHOT`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub steps: Vec<(String, u8)>,
    pub spans: Vec<(String, u8)>,
    pub segments: Vec<(String, u8)>,
    /// The window's counter, arming and carry.
    pub window: (u8, bool, bool),
    /// The lines stream's flag bits with the mode cell's bit clear and set, the bit at the
    /// hand-off, whether the stream starts held, and its passes per line.
    pub line_modes: [u8; 2],
    pub line_mode: bool,
    pub line_held: bool,
    pub line_passes: u8,
    /// Shades (false) or 15-bit colours (true).
    pub colours: bool,
    /// The saved bytes' length, a pending save at the hand-off (written, committed), and
    /// whether the cartridge counts wall time.
    pub save_len: usize,
    pub save_pending: (bool, bool),
    pub clock: bool,
    /// The lines stream's last length, which a spans pass may ask for.
    pub length: u8,
    /// The frame number at the hand-off, how far its last step ran past the frame's end
    /// (in half units), the wall time the clock last read, and the filter's charge.
    pub frame: u64,
    pub overshoot: u64,
    pub clock_reading: Option<u64>,
    pub filter: [i32; 2],
    /// The steps stream's memory and the lines stream's: `memory.bin`, `own.bin` beside
    /// the snapshot file, as [`MemoryFile`].
    pub memory: MemoryFile,
    pub own: MemoryFile,
    /// The length stream's memory, and its flag bits with the mode cell's bit clear and set.
    pub len: MemoryFile,
    pub length_modes: [u8; 2],
    /// The stepper's flag bits with the mode cell's bit clear and set.
    pub stepper_modes: [u8; 2],
    /// The frame being drawn and the frame shown, `frames.bin`: each pixel a u32
    /// little-endian, the drawn frame first.
    pub frames: String,
    /// The cells the runner fills from the files you supply, which the snapshot leaves zero:
    /// the cartridge ROM in the memory (base, length) and in the own memory, and the boot
    /// ROM while it is mapped.
    pub rom: (u32, usize),
    pub own_rom: (u32, usize),
    pub boot: Option<(u32, usize)>,
    /// The digest of the wiring the program that wrote it ran (`Manifest::wiring_sha1`);
    /// empty in a console's power-on snapshot, which the wiring itself holds.
    #[serde(default)]
    pub wiring_sha1: String,
    /// The tensor digest of the model the program that wrote it ran (`Manifest::tensors_sha1`,
    /// from `GBCFORMER_GGUF_ONLY`'s manifest); empty where it ran no GGUF. A state byte's
    /// meaning is the model's, so a snapshot binds to the model as well as to the wiring.
    #[serde(default)]
    pub tensors_sha1: String,
}

/// A memory image as files: its bytes and a byte per address that has a token (1) or not.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MemoryFile {
    pub bytes: String,
    pub present: String,
    /// `Image::parts`' three region lengths.
    pub lengths: [usize; 3],
}
