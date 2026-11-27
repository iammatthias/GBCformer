//! The control logic as a threshold circuit compiled into its own Llama checkpoint, run
//! as the machine's control device one span per pass (`docs/control-in-weights.md`).
//! The reference it is held to is `machine::control`.
//!
//! A span is the M-cycles since the previous pass, then at most one event. The device
//! runs a pass only at an event (a CPU access to a control register, an
//! acknowledgement, another device's interrupt, a change of the buttons held) or when
//! the M-cycles reach the horizon the previous pass returned (the timer's reload, a
//! serial transfer's end, the end of the 8-byte OAM line an OAM DMA transfer is in, a point of the PPU's
//! timing (a visible line's mode-3 start, its mode-0 start where an HBlank HDMA block moves or
//! the STAT line rises while IF bit 1 is clear, a line start where the line rises or
//! VBlank runs, the end of a speed switch's pause), or [`MAX_SPAN`] with the LCD off);
//! between passes nothing a reader can see changes but the counter and, at double speed,
//! the M-cycle within the PPU's tick, which the next full span moves on by the M-cycles
//! the host counted. Where only the PPU's mode-3 starts can need a pass before line 144, a pass
//! schedules up to [`SCHEDULE`] of them before its horizon ([`Schedule`]). The host
//! counts down to each and calls the renderer with the line the pass named, without
//! running a pass (`docs/control-in-weights.md` §19).
//! An MBC3 cartridge's clock runs here too, at the clock's bus accesses and
//! when the host hands over the wall time elapsed (`clock`, §20).
//! Each pass also emits where the frame sequencer's counter bit next falls and the HALT
//! wake bit; the host hands the APU a step at each such M-cycle with no pass (§22).
//! Every DMA byte and stall is a model output: a pass moves one run of OAM DMA bytes within
//! an 8-byte OAM line and at most one HDMA block, by the address bits it emits, and names
//! the M-cycles the CPU stays stopped (§21, §21.5).

pub mod circuit;
pub mod clock;

use crate::checkpoint::{CheckpointDevice, SlotAbi};
use crate::circuit::Circuit;
use crate::device::{Copy, Signals};
use crate::transport::Passes;
use crate::engine::Engine;
use crate::llama::Model;
use crate::machine::control::{Control, ControlDevice, ControlState, Lines, PaletteIndex, PaletteRead};
use crate::machine::io::{Buttons, Link};
use crate::machine::timer::TimerState;
use crate::machine::lcd::LcdState;
use crate::machine::ppu::Lengths;
use crate::machine::rtc::Rtc;
use clock::{CLOCK_ENABLE, CLOCK_LATCH, CLOCK_READ, CLOCK_SELECT, CLOCK_TIME, CLOCK_WRITE, EV_CLOCK};
use circuit::{FE_DROP, FE_DROP_SHIFT, FE_HELD, FE_WAKE};
use circuit::{DX_LAST_EARLY, OWED};
use circuit::{EV_ACK, EV_BLOCK, EV_BUTTONS, EV_RAISE, EV_READ, EV_STOP, EV_WRITE, EVENTS, EXTRA, FX_BLOCK, FX_FRAME, FX_LCD_OFF, FX_OAM_BYTE, FX_OAM_DMA, FX_PAUSE, FX_RENDER, FX_SERIAL_START, MAX_SPAN, NEXT, SCHEDULE, STATE};
use std::collections::BTreeMap;
use std::ops::Range;

pub type State = [u8; STATE.len() + clock::STATE.len()];

/// State byte indices.
/// `PA`: a palette byte a data port reaches (bit 7) and its index. `LK`: each lock span's
/// locks, four bits apiece; `MX`, `MY`: each span's last M-cycle, eleven bits. `HR`: the
/// 2-dot halves an M-cycle lasts, one at double speed.
pub const PA_PRESENT: u8 = 0x80;
pub const PA_INDEX: u8 = 0x7F;
/// `LK`: the locks after the event in bits 0-3 (reads' VRAM and OAM, then writes', as
/// `Lcd::locks` holds them) and the locks after those in bits 4-7.
pub const LK_MASK: u8 = 15;
pub const LK_SHIFT: u8 = 4;
pub const LOCK_SPAN: u16 = 0x7FF;
/// `FE`: where the frame sequencer's edge falls, twelve bits.
pub const FE_EDGE: u16 = 0xFFF;
pub const RATES: [u8; 2] = [1, 2];
pub const FAST_RATE: u8 = 1;
/// The serial input with no cable plugged in.
pub const CABLE_IDLE: u8 = 0xFF;
/// `CL`: an HDMA block's length, and bit 7, the block lands at the end of the GBC CPU step
/// that runs.
pub const CL_LEN: u8 = 0x7F;
pub const CL_HELD: u8 = 0x80;

/// An OAM DMA run moves bytes within one memory line, 2^this bytes.
pub const OAM_LINE_BITS: u8 = crate::memory::layout::LINE_BITS as u8;

const TC: usize = 4;
/// `TC` bits: ordinary STOP holds the GBC CPU, and it keeps a CGB's PPU from VRAM.
const TC_STOPPED: u8 = 0x20;
pub const TC_BLOCKED: u8 = 0x40;
/// `TC` bit 7: the console is a CGB.
const TC_CGB: u8 = 0x80;
const IF: usize = 5;
const IE: usize = 6;
const SB: usize = 9;
#[cfg(test)]
const K1: usize = 14;
const HDMA3: usize = 20;
const HDMA5: usize = 22;
const FL: usize = 31;
const LM: usize = 23;
const LY: usize = 24;
/// The clock's first state byte.
const CK: usize = STATE.len();

pub fn encode_state(s: &ControlState) -> State {
    let (r, q) = (&s.clock.regs, &s.clock.latched);
    let code = s.clock.select.map_or(0, |i| i + 1);
    let t = &s.timer;
    let [c0, c1] = t.counter.to_le_bytes();
    let (sb, sc, rem) = s.serial;
    let [sr0, sr1] = rem.to_le_bytes();
    let (src, dst, h5, uncounted, line_done, ran_off) = s.hdma;
    let l = &s.lcd;
    [
        c0,
        c1,
        t.tima,
        t.tma,
        t.tac & 7 | (t.reload_pending as u8) << 3 | (s.dropped as u8) << 4 | (s.stopped as u8) << 5 | (s.blocked as u8) << 6 | (s.cgb as u8) << 7,
        s.iflag,
        s.ie,
        s.joypad.0,
        s.joypad.1,
        sb,
        sc,
        sr0,
        sr1,
        s.key0,
        s.key1,
        s.svbk,
        s.dma.0,
        s.dma.1,
        (src >> 8) as u8,
        src as u8 & 0xF0,
        (dst >> 8) as u8 & 0x1F | if uncounted { circuit::HP } else { 0 } | if line_done { circuit::DN } else { 0 },
        dst as u8 & 0xF0 | if ran_off { circuit::HE } else { 0 },
        h5,
        2 * l.m + s.half as u8,
        l.ly,
        l.lyc,
        l.enables | l.line as u8 | (l.first as u8) << 1 | (l.coincidence as u8) << 2 | (l.on as u8) << 7,
        s.palette.0[0],
        s.palette.0[1],
        s.pause as u8,
        (s.pause >> 8) as u8,
        s.dma.2 as u8 | (s.timer.reloaded as u8) << 1 | (s.waking as u8) << 2 | s.spare[3] | s.halt.map_or(0, |allow| circuit::FL_HALT | if allow { circuit::FL_HALT_ALLOW } else { 0 }),
        s.spare[0],
        s.spare[1],
        s.spare[2],
        r[0] | r[4] & 0xC0,
        r[1] | (r[4] & 1) << 6 | (s.clock.enabled as u8) << 7,
        r[2] | code << 5,
        r[3],
        q[0] | q[4] & 0xC0,
        q[1] | (q[4] & 1) << 6 | (s.clock.armed as u8) << 7,
        q[2],
        q[3],
    ]
}

pub fn decode_state(b: &State) -> ControlState {
    ControlState {
        timer: TimerState { counter: u16::from_le_bytes([b[0], b[1]]), tima: b[2], tma: b[3], tac: b[4] & 7, reload_pending: b[4] & 8 != 0, reloaded: b[31] & 2 != 0 },
        iflag: b[5],
        ie: b[6],
        joypad: (b[7], b[8]),
        serial: (b[9], b[10], u16::from_le_bytes([b[11], b[12]])),
        key0: b[13],
        key1: b[14],
        svbk: b[15],
        dma: (b[16], b[17], b[31] & 1 != 0),
        hdma: (u16::from_le_bytes([b[19], b[18]]), u16::from_le_bytes([b[21] & !circuit::HE, b[20] & !(circuit::HP | circuit::DN)]), b[22], b[20] & circuit::HP != 0, b[20] & circuit::DN != 0, b[21] & circuit::HE != 0),
        lcd: LcdState { on: b[26] & 0x80 != 0, ly: b[24], m: b[23] >> 1, lyc: b[25], enables: b[26] & 0x78, line: b[26] & 1 != 0, coincidence: b[26] & 4 != 0, first: b[26] & 2 != 0 },
        palette: PaletteIndex([b[27], b[28]]),
        half: b[23] & 1 != 0,
        pause: u16::from_le_bytes([b[29], b[30]]),
        clock: decode_clock(&b[CK..]),
        dropped: b[4] & 0x10 != 0,
        stopped: b[4] & TC_STOPPED != 0,
        blocked: b[4] & TC_BLOCKED != 0,
        cgb: b[4] & TC_CGB != 0,
        waking: b[31] & 4 != 0,
        halt: (b[31] & circuit::FL_HALT != 0).then_some(b[31] & circuit::FL_HALT_ALLOW != 0),
        spare: [b[32], b[33], b[34], b[31] & 0x70],
    }
}

/// The clock's packed state bytes (`clock::STATE`) as the reference holds them.
fn decode_clock(b: &[u8]) -> Rtc {
    let regs = |k: &[u8]| [k[0] & 0x3F, k[1] & 0x3F, k[2] & 0x1F, k[3], k[1] >> 6 & 1 | k[0] & 0xC0];
    let code = b[2] >> 5;
    Rtc { regs: regs(&b[..4]), latched: regs(&b[4..8]), select: (code > 0).then(|| code - 1), enabled: b[1] & 0x80 != 0, armed: b[5] & 0x80 != 0 }
}

/// What happens after a span's M-cycles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// No event. The span reached the horizon.
    Tick,
    Write(u16, u8),
    Read(u16),
    Ack(u8),
    Raise(u8),
    /// The buttons held from now on.
    Buttons(u8),
    /// The CPU ran STOP.
    Stop,
    /// 2^k seconds of wall time elapsed, for the cartridge clock: one binary digit of the
    /// seconds the host hands over.
    Elapse(u8),
    /// The next block of a general-purpose HDMA transfer, the further pass a pass asked for.
    Block,
    /// The serial cable's other end clocked this byte across.
    Receive(u8),
    /// The GBC CPU halted while an HBlank transfer ran.
    Halt,
    /// The GBC CPU left a HALT the model kept (`FL_HALT`).
    Wake,
    /// A GBC CPU step ended while an HBlank block waits to be counted (`HP`).
    StepEnd,
}

impl Event {
    /// The further pass `NE` asks for: its `EV` code, 0 for none.
    fn follow(code: u8) -> Option<Event> {
        match code {
            0 => None,
            EV_BLOCK => Some(Event::Block),
            c => panic!("control: a further pass at the event code {c:#04x}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub mcycles: u16,
    /// The M-cycles the PPU's position moves from the state's `LM` and `LY`. That is all
    /// of them, or after a scheduled mode-3 start, the M-cycles since it.
    pub since: u16,
    /// The M-cycles are a speed switch's pause, counted in `Q`.
    pub paused: bool,
    pub event: Event,
    /// The byte the serial cable's other end shows, 0xFF with no cable.
    pub cable: u8,
    /// A GBC CPU step runs (`IN`).
    pub step: bool,
    /// The M-cycles the GBC CPU stays stopped as the pass begins (`SQ`).
    pub owed: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub state: State,
    pub read: u8,
    /// M-cycles from the end of this span to the next pass, 1 to [`MAX_SPAN`].
    pub horizon: u16,
    /// [`circuit::EXTRA`]'s `FX` bits.
    pub flags: u8,
    /// The OAM DMA bytes the span moved, a run within one 8-byte OAM line: the source page,
    /// the line (index bits 7-3), and a bit per byte of the line moved, for the bytes before
    /// the span's last M-cycle and for its last.
    pub oam: Option<(u8, u8, u8, u8)>,
    /// The run's last byte lands before a line the span's last M-cycle starts (`DX` bit 0).
    pub last_early: bool,
    /// The HDMA block the pass moves: its source and destination, each with the low four
    /// bits clear, its length (`CL`), and whether it waits for the end of the GBC CPU step
    /// that runs (`CL` bit 7).
    pub block: Option<(u16, u16, u16, bool)>,
    /// M-cycles the CPU stays stopped once the step that runs has ended: the span's `owed`
    /// and what the pass added.
    pub stall: u16,
    /// The palette byte a BCPD or OCPD access reaches, when mode 3 does not lock it.
    pub palette: Option<u8>,
    pub schedule: Schedule,
    /// M-cycles from the pass to where the frame sequencer's counter bit next falls by
    /// counting, None while a pause holds the counter.
    pub edge: Option<u16>,
    /// M-cycles from the pass to where it next rises by counting, None while a pause
    /// holds the counter.
    pub rise: Option<u16>,
    /// M-cycles from the pass to where a set bit a DIV write or a speed switch reset falls
    /// (`FE_DROP`), 0 for none.
    pub drop: u8,
    /// An interrupt is enabled and requested: the GBC CPU leaves HALT.
    pub wake: bool,
    /// The 2-dot halves each M-cycle after the pass lasts (`HR`).
    pub rate: u8,
    /// The page OAM DMA bytes land in (`DD`).
    pub oam_dest: u8,
    /// The halves a frame of emulated time lasts (`FH`).
    pub frame: u16,
    /// The GBC CPU's access locks after the pass (`LK` bits 0-3: reads' VRAM and OAM, then
    /// writes') and the M-cycles they hold at the least (`MX`), then the locks after those
    /// (bits 4-7) and the M-cycles from the pass they hold to (`MY`).
    pub locks: [(u8, u16); 2],
    /// The cable (`SX`): this end's transfer ended and sends `SO`; the other end's byte landed.
    pub sent: Option<u8>,
    pub took: bool,
    /// The byte OAM DMA writes when its source drives none (`DF`, nonzero), else 0.
    pub fill: u8,
    /// The M-cycles OAM DMA still blocks the bus after the pass (`DY`).
    pub dma_left: u8,
    /// `Lines::dma_buses` after the pass (`DB`).
    pub dma_buses: u8,
    /// The M-cycles after the pass before a fresh OAM DMA blocks (`DZ`).
    pub dma_from: u8,
    /// A further pass at once, at the same M-cycle, at this event (`NE`).
    pub follow: Option<Event>,
}

/// The mode-3 starts a pass scheduled. The first is `first` M-cycles after the pass's
/// position, each next one `period` after the one before, and start j draws `lines[j]`,
/// with the PPU's position there `phase` (`LM`'s byte) in line `lines[j]`. The next pass
/// runs `tail` after the last. Every field is a model output, and `first` is its
/// horizon; the host counts down from each, adding nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    pub count: u8,
    pub first: u16,
    pub period: u16,
    pub tail: u16,
    pub phase: u8,
    pub lines: [u8; SCHEDULE],
}

pub const INPUTS: [&str; STATE.len() + clock::STATE.len() + EVENTS.len() + OWED.len() + 3] = [
    "C0", "C1", "TIMA", "TMA", "TC", "IF", "IE", "JP", "JB", "SB", "SC", "SR0", "SR1", "K0", "K1", "SV", "DR", "DI", "HDMA1", "HDMA2", "HDMA3", "HDMA4", "HDMA5", "LM", "LY", "LYC", "ST", "BP", "OP", "PS0", "PS1", "FL", "UR0", "UR1", "UR2", "CK0", "CK1", "CK2", "CK3", "CKL0", "CKL1", "CKL2", "CKL3", "D0", "D1", "EV", "EA", "EX", "Q0", "Q1", "H0", "H1", "SQ0", "SQ1", "SI", "IN", "ML",
];
/// The request: `MQ` bit 0, the pass needs the drawn line's mode-3 length (`ML`, late).
pub const REQUEST: [&str; 1] = ["MQ"];
/// `MQ`'s bit: the pass needs the line's length.
pub const MQ_LENGTH: u8 = 1;
pub const RESULT: [&str; NEXT.len() + clock::NEXT.len() + EXTRA.len()] = [
    "C0'", "C1'", "TIMA'", "TMA'", "TC'", "IF'", "IE'", "JP'", "JB'", "SB'", "SC'", "SR0'", "SR1'", "K0'", "K1'", "SV'", "DR'", "DI'", "HDMA1'", "HDMA2'", "HDMA3'", "HDMA4'", "HDMA5'", "LM'", "LY'", "LYC'", "ST'", "BP'", "OP'", "PS0'", "PS1'", "FL'", "UR0'", "UR1'", "UR2'", "CK0'", "CK1'", "CK2'", "CK3'", "CKL0'", "CKL1'", "CKL2'", "CKL3'", "RV", "H0", "H1", "FX", "DP", "DX", "DM", "DL", "CS0", "CS1", "CD0", "CD1", "SS0", "SS1", "PA", "RF", "RS", "RP", "RM", "SL0", "SL1", "SL2", "SL3", "SL4", "SL5", "SL6", "SL7", "FE0", "FE1", "HR", "DD", "CL", "LK", "MX0", "MX1", "MY0", "MY1", "SX", "SO", "DF", "FH0", "FH1", "DY", "DB", "DZ", "FR0", "FR1", "RT", "NE"];

pub fn build() -> Circuit {
    let b = circuit::build();
    assert_eq!(b.inputs, INPUTS, "the circuit's inputs are the ABI's");
    assert_eq!(b.outputs, REQUEST.iter().chain(&RESULT).copied().collect::<Vec<_>>(), "the circuit's outputs are the ABI's");
    b.c
}

pub fn engine() -> Engine {
    Engine::new(Model::compile(&build()))
}

/// The control slot ABI: a pass past M-cycle 63 of a visible line asks for that line's
/// mode-3 length, a late input, and decodes again with it.
pub struct Ctl;

impl SlotAbi for Ctl {
    const INPUTS: &'static [&'static str] = &INPUTS;
    const REQUEST: &'static [&'static str] = &REQUEST;
    const RESULT: &'static [&'static str] = &RESULT;
    const LATE: Range<usize> = INPUTS.len() - 1..INPUTS.len();
    type State = State;
    type Inputs = Span;
    type Late = [u8; 1];
    type Request = bool;
    type Outcome = Outcome;

    fn encode(state: &State, span: &Span, slots: &mut [u8]) {
        assert!(span.mcycles as u64 <= MAX_SPAN, "a span of {} M-cycles", span.mcycles);
        let (ev, ea, ex) = match span.event {
            Event::Tick => (0, 0, 0),
            // A cartridge address is the clock's.
            Event::Write(a, v) if a < 0xC000 => {
                let code = match a {
                    0x0000..=0x1FFF => CLOCK_ENABLE,
                    0x4000..=0x5FFF => CLOCK_SELECT,
                    0x6000..=0x7FFF => CLOCK_LATCH,
                    0xA000..=0xBFFF => CLOCK_WRITE,
                    _ => panic!("the clock sees no write to {a:#06x}"),
                };
                (EV_CLOCK, code, v)
            }
            Event::Write(a, v) => (EV_WRITE, a as u8, v),
            Event::Read(a) if a < 0xC000 => (EV_CLOCK, CLOCK_READ, 0),
            Event::Read(a) => (EV_READ, a as u8, 0),
            Event::Elapse(k) => {
                assert!((k as u32) < clock::DIGITS, "2^{k} seconds elapsed");
                (EV_CLOCK, CLOCK_TIME, k)
            }
            Event::Ack(b) => (EV_ACK, 0, b),
            Event::Raise(b) => (EV_RAISE, 0, b),
            Event::Buttons(b) => (EV_BUTTONS, 0, b),
            Event::Stop => (EV_STOP, 0, 0),
            Event::Block => (EV_BLOCK, 0, 0),
            Event::Receive(b) => (EV_WRITE, circuit::LINK_PORT, b),
            Event::Halt => (EV_WRITE, circuit::HALT_PORT, 0),
            Event::StepEnd => (EV_WRITE, circuit::STEP_END_PORT, 0),
            Event::Wake => (EV_WRITE, circuit::WAKE_PORT, 0),
        };
        // The span in M-cycles, for the timer (`D`) or, as the last pass flagged, the pause
        // (`Q`), and the M-cycles the PPU's position moves (`H`).
        let (d, q) = if span.paused { (0, span.mcycles) } else { (span.mcycles, 0) };
        let h = span.since;
        let ([d0, d1], [q0, q1], [h0, h1]) = (d.to_le_bytes(), q.to_le_bytes(), h.to_le_bytes());
        let n = state.len();
        slots[..n].copy_from_slice(state);
        slots[n..n + EVENTS.len()].copy_from_slice(&[d0, d1, ev, ea, ex, q0, q1, h0, h1]);
        slots[n + EVENTS.len()..n + EVENTS.len() + OWED.len()].copy_from_slice(&span.owed.to_le_bytes());
        slots[n + EVENTS.len() + OWED.len()] = span.cable;
        slots[n + EVENTS.len() + OWED.len() + 1] = span.step as u8;
    }

    fn decode_request(bytes: &[u8]) -> bool {
        bytes[0] & MQ_LENGTH != 0
    }

    fn decode_outcome(b: &[u8]) -> Outcome {
        let n = CK + clock::STATE.len();
        let flags = b[n + 3];
        let word = |i: usize| u16::from_le_bytes([b[n + i], b[n + i + 1]]);
        let oam = (flags & FX_OAM_BYTE != 0).then_some((b[n + 4], b[n + 5] & !DX_LAST_EARLY, b[n + 6], b[n + 7]));
        let block = (flags & FX_BLOCK != 0).then(|| (word(8), word(10), (b[n + 31] & CL_LEN) as u16, b[n + 31] & CL_HELD != 0));
        let palette = (b[n + 14] & PA_PRESENT != 0).then_some(b[n + 14] & PA_INDEX);
        let schedule = Schedule { count: b[n + 16], first: b[n + 15] as u16 + 1, period: b[n + 17] as u16, tail: b[n + 47] as u16, phase: b[n + 18], lines: b[n + 19..n + 19 + SCHEDULE].try_into().unwrap() };
        let fe = word(27);
        let edge = (b[n + 28] & FE_HELD == 0).then_some((fe & FE_EDGE) + 1);
        let rise = (b[n + 28] & FE_HELD == 0).then_some((word(45) & FE_EDGE) + 1);
        Outcome { state: b[..n].try_into().unwrap(), read: b[n], horizon: word(1) + 1, flags, oam, last_early: b[n + 5] & DX_LAST_EARLY != 0, block, stall: word(12), palette, schedule, edge, rise, drop: (b[n + 28] & FE_DROP) >> FE_DROP_SHIFT, wake: b[n + 28] & FE_WAKE != 0, rate: b[n + 29], oam_dest: b[n + 30], locks: [(b[n + 32] & LK_MASK, (word(33) & LOCK_SPAN) + 1), (b[n + 32] >> LK_SHIFT & LK_MASK, (word(35) & LOCK_SPAN) + 1)], sent: (b[n + 37] & 1 != 0).then_some(b[n + 38]), took: b[n + 37] & 2 != 0, fill: b[n + 39], frame: word(40), dma_left: b[n + 42], dma_buses: b[n + 43], dma_from: b[n + 44], follow: Event::follow(b[n + 48]) }
    }


    fn needs_late(request: &bool) -> bool {
        *request
    }
}

/// Whatever runs one span: the checkpoint, or the circuit evaluated directly. `length`
/// gives the drawn line's mode-3 length, waiting for the renderer, when the pass asks.
pub trait Spans: Send {
    fn run(&mut self, state: &State, span: &Span, length: &mut dyn FnMut() -> u8) -> Outcome;
    fn describe(&self) -> Vec<String> {
        Vec::new()
    }
    fn report(&self) -> Vec<String> {
        Vec::new()
    }
}

pub type ControlCheckpoint<M> = CheckpointDevice<Ctl, M>;

impl Spans for ControlCheckpoint<Engine> {
    fn run(&mut self, state: &State, span: &Span, length: &mut dyn FnMut() -> u8) -> Outcome {
        let late = if self.request(state, span) { length() } else { 0 };
        self.execute(&[late])
    }

    fn describe(&self) -> Vec<String> {
        let e = self.model();
        let c = &e.dense().cfg;
        let fast_path = e.exact().map_or_else(|err| format!("OFF ({err}): every pass runs dense"), |_| "on".into());
        vec![format!("control checkpoint: {} layers, hidden {}, intermediate {}, {} heads, vocab {}, {:.1}M parameters ({} nonzero), integer fast path {fast_path}", c.layers, c.hidden, c.intermediate, c.heads, c.vocab, c.params() as f64 / 1e6, e.nonzeros())]
    }

    fn report(&self) -> Vec<String> {
        let e = self.model();
        let s = &e.stats;
        let mut lines = vec![format!("control engine: {} passes, {} on the dense fallback", s.passes, s.fallback_passes), s.to_string()];
        let census = e.census();
        if !census.is_empty() {
            let top: Vec<String> = census.iter().take(40).map(|(n, u)| format!("{n} {u:.1}")).collect();
            lines.push(format!("control gate updates per fast pass by node family: {}", top.join(", ")));
            let sources: Vec<String> = e.sources().iter().take(60).map(|(n, u)| format!("{n} {u:.1}")).collect();
            lines.push(format!("control gate updates per fast pass by changed wire: {}", sources.join(", ")));
        }
        lines
    }
}

pub struct CircuitSpans(pub Circuit);

impl Spans for CircuitSpans {
    fn run(&mut self, state: &State, span: &Span, length: &mut dyn FnMut() -> u8) -> Outcome {
        let mut slots = [0u8; INPUTS.len()];
        Ctl::encode(state, span, &mut slots);
        let mut input: BTreeMap<String, u8> = INPUTS.iter().map(|n| n.to_string()).zip(slots).collect();
        let mut out = self.0.eval(&input);
        if out[REQUEST[0]] & MQ_LENGTH != 0 {
            input.insert(circuit::LENGTH.to_string(), length());
            out = self.0.eval(&input);
        }
        let bytes: Vec<u8> = RESULT.iter().map(|n| out[*n]).collect();
        Ctl::decode_outcome(&bytes)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub passes: u64,
    /// Passes at the horizon, `renders` and `frames` of them where the renderer drew a
    /// line or showed a frame.
    pub ticks: u64,
    pub renders: u64,
    /// Lines drawn at a scheduled mode-3 start, with no pass.
    pub scheduled: u64,
    pub frames: u64,
    pub writes: u64,
    pub reads: u64,
    pub acks: u64,
    pub raises: u64,
    pub buttons: u64,
    pub stops: u64,
    /// Passes at a HALT and at its wake while an HBlank transfer runs.
    pub halts: u64,
    /// Cartridge clock events: writes the clock's chip saw, reads of its registers, and
    /// wall time handed over. Each is a pass, also counted in `writes` and `reads`.
    pub clock_writes: u64,
    pub clock_reads: u64,
    pub clock_times: u64,
    /// OAM DMA bytes moved, the passes (runs) that moved them, and the passes for a
    /// general-purpose HDMA transfer's blocks after its first.
    pub oam_bytes: u64,
    pub oam_runs: u64,
    /// Passes that ended a run at a CPU access the bus blocked.
    pub cuts: u64,
    /// Passes that asked for a line's mode-3 length, and the time they waited for it.
    pub lengths: u64,
    pub length_nanos: u64,
    /// Passes the memory model asked for the access locks by.
    pub lock_queries: u64,
    /// Passes over the span to a GBC CPU access's M-cycle past the last pass's lock spans,
    /// their state kept by nobody.
    pub lock_lookaheads: u64,
    /// Passes for the OAM row a DMG's scan reads, at an FE page address the memory model
    /// flagged, and how many of them named a row.
    pub row_queries: u64,
    pub rows: u64,
    /// Passes that offered a byte the cable's other end clocked across.
    pub received: u64,
    pub blocks: u64,
    /// Emulated time over the spans, in half M-cycles of normal speed (one per M-cycle
    /// at double speed), as the machine counts it.
    pub halves: u64,
}

/// The control logic computed by a span evaluator. The host counts the M-cycles since
/// the last pass and runs a pass at every event and at the horizon.
pub struct ControlModel<E: Spans> {
    eval: E,
    state: State,
    /// Bytes sent over the serial port: SB whenever a pass started a transfer.
    serial: Vec<u8>,
    /// The last pass's `FX` bits.
    flags: u8,
    /// The last pass's wake bit.
    wake: bool,
    /// The last pass's `HR`: the 2-dot halves an M-cycle lasts from it on.
    rate: u8,
    /// The last pass's `FH`.
    frame: u64,
    /// The last pass's access locks and the M-cycles from it they hold to, in order.
    locks: [(u8, u16); 2],
    /// The M-cycles OAM DMA blocks the bus for from the last pass (`DY`).
    dma_left: u64,
    /// `DB` from the last pass.
    dma_buses: u8,
    /// The M-cycles OAM DMA waits to block from the last pass (`DZ`).
    dma_from: u64,
    /// The countdown to the next pass, each mode-3 start the last pass scheduled and each
    /// frame sequencer edge (`transport::Spans`), shared with the minimal runner.
    t: crate::transport::Spans,
    /// Each drawn line's mode-3 length, from the PPU.
    lengths: Lengths,
    /// The serial cable's end, when one is plugged in.
    link: Option<Link>,
    /// A GBC CPU step runs (`ControlDevice::step_running`).
    step: bool,
    pub counts: Counts,
}

impl<E: Spans> ControlModel<E> {
    /// The M-cycles since the last pass are a speed switch's pause or ordinary STOP.
    fn paused(&self) -> bool {
        self.flags & FX_PAUSE != 0
    }

    /// CGB double speed after the last pass: an M-cycle of one half (`HR`).
    fn double(&self) -> bool {
        self.rate == FAST_RATE
    }
}

impl<E: Spans> ControlModel<E> {
    /// The locks a STAT read `extra` M-cycles from now would see: a pass over the span to
    /// there, from the position the next pass would start at, whose state no one keeps.
    fn locks_at(&mut self, extra: u64) -> u8 {
        let start = self.t.ahead(extra);
        let mut state = self.state;
        start.place(&mut state, LM, LY);
        let cable = self.link.as_ref().map_or(CABLE_IDLE, |l| l.their_sb());
        let span = Span { mcycles: start.cycles as u16, since: start.since as u16, paused: self.paused(), event: Event::Read(0xFF41), cable, step: self.step, owed: self.t.owed() as u16 };
        self.counts.lock_lookaheads += 1;
        let lengths = &self.lengths;
        self.eval.run(&state, &span, &mut || lengths.latest()).locks[0].0
    }
}

impl<E: Spans> ControlModel<E> {
    /// Takes over from `reference` at its current M-cycle. The first pass runs at once
    /// (a span of 0 M-cycles) to learn the horizon.
    pub fn new(eval: E, reference: &ControlState, lengths: Lengths) -> Self {
        let mut m = ControlModel { eval, state: encode_state(reference), serial: Vec::new(), flags: 0, wake: false, rate: 2, frame: 0, locks: [(0, 0); 2], dma_left: 0, dma_buses: 0, dma_from: 0, t: crate::transport::Spans::new(MAX_SPAN), lengths, link: None, step: false, counts: Counts::default() };
        m.pass(Event::Tick);
        m.counts = Counts::default();
        m
    }

    pub fn state(&self) -> ControlState {
        decode_state(&self.state)
    }

    /// Runs `event` after the span since the last pass: every event follows the ticks
    /// before it, so the host chooses no pass shape (`docs/host-in-weights.md` §4 item 8).
    fn pass(&mut self, event: Event) -> Outcome {
        let elapsed = self.t.elapsed();
        let halves = if self.double() { elapsed } else { 2 * elapsed };
        // After a scheduled mode-3 start, the span runs from there. It starts at the
        // position the model emitted for that start (its line and `LM`) and counts the
        // PPU's M-cycles since it.
        let start = self.t.open();
        let mut state = self.state;
        start.place(&mut state, LM, LY);
        let out = self.run(state, start.cycles, start.since, event);
        self.counts.halves += halves;
        out
    }

    /// The run of OAM DMA bytes a pass moved, as the transport's `Run` within an OAM line.
    fn run_of(out: &Outcome) -> Option<crate::transport::Run> {
        out.oam.map(|(page, line, before, last)| crate::transport::Run { page, line, before, last, last_early: out.last_early, dest: out.oam_dest, fill: out.fill })
    }


    fn run(&mut self, state: State, mcycles: u64, since: u64, event: Event) -> Outcome {
        let cable = self.link.as_ref().map_or(CABLE_IDLE, |l| l.their_sb());
        let span = Span { mcycles: mcycles as u16, since: since as u16, paused: self.paused(), event, cable, step: self.step, owed: self.t.owed() as u16 };
        let (lengths, counts) = (&self.lengths, &mut self.counts);
        let out = self.eval.run(&state, &span, &mut || {
            let t = std::time::Instant::now();
            let ml = lengths.latest();
            counts.lengths += 1;
            counts.length_nanos += t.elapsed().as_nanos() as u64;
            ml
        });
        self.state = out.state;
        // With a schedule the horizon is its first start, and the next pass is `tail` after
        // its last (`transport::Spans::close`).
        let sched = out.schedule;
        let schedule = crate::transport::Schedule { count: sched.count as usize, first: sched.first as u64, period: sched.period as u64, tail: sched.tail as u64, phase: sched.phase, lines: sched.lines };
        assert!(out.stall >= span.owed, "control: the stop owed fell from {} to {} M-cycles in a pass", span.owed, out.stall);
        self.t.close(&crate::transport::Cut { horizon: out.horizon as u64, schedule, edge: out.edge.map(u64::from), drop: out.drop as u64, rise: out.rise.map(u64::from), owed: out.stall as u64 });
        if out.flags & FX_SERIAL_START != 0 {
            self.serial.push(self.state[SB]);
        }
        if let Some(run) = Self::run_of(&out) {
            run.check(mcycles);
            self.counts.oam_bytes += mcycles;
            self.counts.oam_runs += 1;
        }
        self.flags = out.flags;
        self.wake = out.wake;
        assert!(RATES.contains(&out.rate), "control: an M-cycle of {} halves", out.rate);
        self.rate = out.rate;
        self.frame = out.frame as u64;
        self.locks = out.locks;
        self.dma_left = out.dma_left as u64;
        self.dma_buses = out.dma_buses;
        self.dma_from = out.dma_from as u64;
        if let Some(l) = &self.link {
            if let Some(b) = out.sent {
                l.send(b);
            }
            l.show(self.state[SB]);
        }
        self.counts.passes += 1;
        out
    }
}

impl<E: Spans> crate::transport::Passes for ControlModel<E> {
    type Event = Event;
    type Out = Outcome;
    type Cx<'a> = ();

    fn once(&mut self, event: Event, _: &mut ()) -> Outcome {
        self.pass(event)
    }

    fn asks(out: &Outcome) -> crate::transport::Asks {
        crate::transport::Asks {
            run: Self::run_of(out),
            block: out.block,
            render: out.flags & FX_RENDER != 0,
            frame: out.flags & FX_FRAME != 0,
            blank: out.flags & FX_LCD_OFF != 0,
            own: out.palette,
        }
    }

    fn follow(out: &Outcome) -> Option<Event> {
        out.follow
    }

    fn own(&self, event: crate::transport::Own) -> Event {
        match event {
            crate::transport::Own::Tick => Event::Tick,
            crate::transport::Own::Stop => Event::Stop,
        }
    }

    fn countdown(&mut self) -> &mut crate::transport::Spans {
        &mut self.t
    }

    fn run_bits(&self) -> u8 {
        OAM_LINE_BITS
    }

    fn held(&self) -> bool {
        self.state[TC] & TC_BLOCKED != 0
    }

    fn line(&self) -> u8 {
        self.state[LY]
    }

    fn counted(&mut self, what: crate::transport::Counted) {
        use crate::transport::Counted::*;
        let c = &mut self.counts;
        match what {
            Block => c.blocks += 1,
            Render => c.renders += 1,
            Scheduled => c.scheduled += 1,
            Frame => c.frames += 1,
            Cut => c.cuts += 1,
            Stop => c.stops += 1,
            Tick => c.ticks += 1,
        }
    }
}

impl<E: Spans> ControlDevice for ControlModel<E> {
    fn read(&mut self, addr: u16, landed: &mut Vec<Copy>) -> u8 {
        self.counts.reads += 1;
        self.counts.clock_reads += (addr < 0xC000) as u64;
        let read = self.pass_alone(Event::Read(addr), landed, &mut ()).read;
        if addr == crate::machine::control::OAM_ROW_ADDR {
            self.counts.row_queries += 1;
            self.counts.rows += (read != 0) as u64;
        }
        read
    }

    fn palette(&mut self, addr: u16, landed: &mut Vec<Copy>) -> PaletteRead {
        self.counts.reads += 1;
        let out = self.pass_alone(Event::Read(addr), landed, &mut ());
        out.palette.map_or(PaletteRead::Byte(out.read), PaletteRead::At)
    }

    fn write(&mut self, addr: u16, value: u8, s: &mut Signals) {
        self.counts.writes += 1;
        self.counts.clock_writes += (addr < 0xC000) as u64;
        self.pass_with(Event::Write(addr, value), s, &mut (), Some(value));
    }

    /// A pass for each binary digit of the seconds.
    fn elapse(&mut self, seconds: u64, landed: &mut Vec<Copy>) {
        self.counts.clock_times += 1;
        for k in (0..u64::BITS).filter(|k| seconds >> k & 1 == 1) {
            self.pass_alone(Event::Elapse(k as u8), landed, &mut ());
        }
    }

    fn clock(&self) -> Rtc {
        self.state().clock
    }

    fn stop(&mut self, s: &mut Signals) {
        Passes::stop(self, s, &mut ());
    }

    /// A pass that ends the OAM DMA run here, when one runs and M-cycles have passed.
    fn cut(&mut self, s: &mut Signals) {
        Passes::cut(self, s, &mut ());
    }

    fn advance(&mut self, mcycles: u64, stopped: bool, s: &mut Signals) {
        // A byte the cable's other end clocked across: a pass offers it, and the cable
        // keeps it while a transfer of this end's counts.
        if let Some(b) = self.link.as_ref().and_then(Link::waiting) {
            self.counts.received += 1;
            if self.pass_alone(Event::Receive(b), &mut s.early, &mut ()).took {
                self.link.as_ref().expect("the cable").take();
            }
        }
        Passes::advance(self, mcycles, stopped, s, &mut ());
    }

    fn raise(&mut self, bits: u8, landed: &mut Vec<Copy>) {
        self.counts.raises += 1;
        self.pass_alone(Event::Raise(bits), landed, &mut ());
    }

    fn ack(&mut self, bits: u8, landed: &mut Vec<Copy>) {
        self.counts.acks += 1;
        self.pass_alone(Event::Ack(bits), landed, &mut ());
    }

    fn buttons(&mut self, pressed: Buttons, landed: &mut Vec<Copy>) -> Option<bool> {
        self.counts.buttons += 1;
        self.pass_holding(Event::Buttons(pressed.0), landed, &mut ())
    }

    /// IE, IF, SVBK, the speed, the pause and whether OAM DMA runs from the last pass, and
    /// the M-cycles to the next stop (a scheduled start, the horizon or an edge), and the
    /// wake bit the last pass emitted.
    fn lines(&self) -> Lines {
        let paused = self.paused();
        Lines {
            ie: self.state[IE],
            iflag: self.state[IF],
            horizon: self.t.horizon(),
            dma_active: self.flags & FX_OAM_DMA != 0,
            dma_buses: self.dma_buses,
            svbk: self.state[15],
            double: self.double(),
            paused,
            wake: self.wake,
            owed: self.t.owed(),
            rate: self.rate,
            frame: self.frame,
            locks: crate::transport::held_lock(&self.locks, self.t.elapsed()),
        }
    }

    fn serial_output(&self) -> &[u8] {
        &self.serial
    }

    /// At a step's end, a pass only while the model's state says an HBlank block waits to be
    /// counted (`HP`).
    fn step_running(&mut self, running: bool, landed: &mut Vec<Copy>) {
        self.step = running;
        if !running && self.state[HDMA3] & circuit::HP != 0 {
            self.counts.halts += 1;
            self.pass_alone(Event::StepEnd, landed, &mut ());
        }
    }

    /// A pass only while the model's state says an HBlank transfer runs (HDMA5 bit 7
    /// clear): no other HALT changes what the model keeps.
    fn halt(&mut self, landed: &mut Vec<Copy>) {
        if self.state[HDMA5] & 0x80 == 0 {
            self.counts.halts += 1;
            self.pass_alone(Event::Halt, landed, &mut ());
        }
    }

    /// A pass only while the model's state keeps a HALT (`FL_HALT`).
    fn woke(&mut self, s: &mut Signals) {
        if self.state[FL] & circuit::FL_HALT != 0 {
            self.counts.halts += 1;
            self.pass_with(Event::Wake, s, &mut (), None);
        }
    }

    fn plug(&mut self, link: Link) -> Result<(), String> {
        self.link = Some(link);
        Ok(())
    }

    /// A pass in place, as a STAT read is, for the locks after it.
    fn locks(&mut self, landed: &mut Vec<Copy>) -> u8 {
        self.counts.lock_queries += 1;
        self.pass_alone(Event::Read(0xFF41), landed, &mut ()).locks[0].0
    }

    /// From the last pass's outputs: OAM DMA blocks from its `DZ`-th M-cycle to its `DY`-th, and the
    /// locks are the span's that holds there. Past the spans, a pass over the span to that
    /// M-cycle gives them and leaves no state. (A pass in place first rarely helps: the
    /// spans lapse mostly in mode 3 at its 63rd M-cycle, where a pass not yet past it
    /// does not know the line's length; 307 of 312 refreshes on Yellow 300 still lapsed.)
    fn ahead(&mut self, mcycles: u8, locks: bool, write: bool) -> u8 {
        let (held, at) = (self.locks, self.t.elapsed() + mcycles as u64);
        crate::transport::bus_state(&held, self.dma_from..self.dma_left, at, locks, write, crate::step::STATE_BITS, || self.locks_at(mcycles as u64))
    }

    fn describe(&self) -> Vec<String> {
        let mut lines = self.eval.describe();
        lines.push("control: the timer (DIV, TIMA, TMA, TAC), the interrupt flags (IF, IE), the joypad register (P1, with the buttons held as an input), serial (SB, SC, the transfer and its interrupt), OAM DMA and HDMA (each run of OAM DMA bytes within an 8-byte OAM line, by its source page, line and a bit per byte, ended at the line's end, every line start and mode-3 start, every event and every CPU access the transfer blocks, with which bytes land before a line the pass draws; each 16-byte HDMA block, by its source and destination bits, one a pass, a general-purpose transfer's blocks after its first a pass each; the transfers' starts and ends; the M-cycles each block and the pause stop the CPU for), KEY0, KEY1, SVBK, the speed switch (the speed, DIV's reset and the pause), the PPU's timing (LY, LYC, STAT, the mode, the VBlank and STAT interrupts, and when a line is drawn, the frame shown and the screen blanked, a tick every second M-cycle at double speed, the span's M-cycles turned into 2-dot halves at the model's speed), the frame sequencer's edges and the HALT wake and an HBlank transfer held through a HALT, the palette index registers (BCPS, OCPS, which byte a BCPD or OCPD access reaches and the mode-3 lock, 0xFF read through it), ordinary STOP (DIV reset, the wait for a falling joypad line, a DMG's held timing and blank screen, a CGB's PPU kept from VRAM), each visible line's mode 0 from the mode-3 length the PPU checkpoint emits for it, the GBC CPU's VRAM and OAM locks and how long they hold and an MBC3 cartridge's real-time clock (the RAM enable and select it sees, the latch, halt, the counting with its carries, the day counter's 9 bits and sticky carry, register writes and reads) come from the model, every read, interrupt bit and horizon included; the host formats the wall time elapsed since its last reading as days and the time of day for the clock, counts M-cycles between passes (as the pause's while the model flags one), reads the horizon as one more than the model's H, counts down to each M-cycle where the model said the frame sequencer's counter bit falls (the count's carry, and the M-cycle after a DIV write or speed switch that reset a set bit) and hands the APU a step there, hands over each wake bit the model emits for a halted GBC CPU, places each DMA byte at the address bits the model emitted (an OAM DMA run's bytes at the destination page, line and bit positions the model set, an HDMA block's bytes by the length the model emitted, at their low address bits), runs a pass at a CPU access the memory model sends to the BLOCKED block, hands the PPU checkpoint's mode-3 length to a pass that asks for it (waiting for the line to be drawn), hands the memory model the locks while they hold and runs a pass for them when the memory model asks, runs a pass for each further block while the model's state says a general-purpose transfer has more, runs a pass at a HALT the GBC CPU model emitted while the model's state says an HBlank transfer runs and at the wake while the state keeps that HALT, tells each pass whether a GBC CPU step runs and lands a block the model holds to that step's end there, stops the CPU for the M-cycles each pass names, records SB when the model starts a transfer, calls the renderer when a pass flags it and at each mode-3 start a pass scheduled (counting down to the instants and naming the lines the model emitted, and handing the next pass the position the model emitted for the last one passed with the M-cycles since it)".into());
        lines
    }

    fn report(&mut self) -> Vec<String> {
        let c = &self.counts;
        let frames = c.halves as f64 / crate::machine::HALVES_PER_FRAME as f64;
        let per = |n: u64| n as f64 / frames.max(1e-9);
        let mut lines = vec![format!(
            "control: {} passes ({:.1} per frame): {} at the horizon ({:.1} per frame; {} drew a line, {} showed a frame), {} lines drawn at a scheduled mode-3 start with no pass ({:.1} per frame), {} writes, {} reads, {} acknowledgements, {} raises, {} button changes, {} STOPs, {} passes at a HALT, its wake or a step's end; DMA: {} OAM DMA bytes in {} runs, a pass each ({:.1} runs per frame, {} of them ended at a CPU access the transfer blocked), {} general-purpose HDMA blocks after a transfer's first, a pass each; the cartridge clock: {} writes, {} reads, {} wall times handed over",
            c.passes,
            per(c.passes),
            c.ticks,
            per(c.ticks),
            c.renders - c.scheduled,
            c.frames,
            c.scheduled,
            per(c.scheduled),
            c.writes,
            c.reads,
            c.acks,
            c.raises,
            c.buttons,
            c.stops,
            c.halts,
            c.oam_bytes,
            c.oam_runs,
            per(c.oam_runs),
            c.cuts,
            c.blocks,
            c.clock_writes,
            c.clock_reads,
            c.clock_times,
        )];
        lines.push(format!("control: {} passes asked for a line's mode-3 length ({:.1} per frame), waiting {:.2} ms in all for the renderer; {} passes for the GBC CPU's access locks ({:.1} per frame), {} over the span to an access's M-cycle for its locks ({:.2} per frame); {} for the OAM row at an FE page address ({:.1} per frame), {} of them naming a row", c.lengths, per(c.lengths), c.length_nanos as f64 / 1e6, c.lock_queries, per(c.lock_queries), c.lock_lookaheads, per(c.lock_lookaheads), c.row_queries, per(c.row_queries), c.rows));
        lines.extend(self.eval.report());
        lines
    }
}

/// `s` without what the ticks since the last full span have not yet changed in the
/// model: the timer, a serial transfer's countdown, the OAM DMA index within a run, the
/// PPU's position (LY, the tick, the M-cycle within it) with the STAT line that follows
/// it, a speed switch's pause, and a dropped frame-sequencer bit the next M-cycle hands
/// over.
fn untimed(s: &ControlState) -> ControlState {
    ControlState {
        timer: TimerState { counter: 0, tima: 0, tma: s.timer.tma, tac: s.timer.tac, reload_pending: false, reloaded: false },
        serial: (s.serial.0, s.serial.1, 0),
        dma: (s.dma.0, 0, false),
        // A line's request clears `DN` between passes when no transfer runs, which needs no pass.
        hdma: (s.hdma.0, s.hdma.1, s.hdma.2, s.hdma.3, false, s.hdma.5),
        lcd: LcdState { ly: 0, m: 0, line: false, coincidence: false, first: false, ..s.lcd },
        half: false,
        pause: 0,
        dropped: false,
        ..*s
    }
}

/// The model beside the reference. The reference ticks every M-cycle, and after every
/// pass and every advance the state (all of it after a full span, all but the timer, the
/// serial countdown and the OAM DMA index between passes), every read byte, the copies
/// and stall each call asks the bus for, and the lines the machine latches must agree.
/// The model moves OAM DMA bytes in runs, one pass a run, so its bytes are held to the
/// reference's at every pass boundary instead: the run must be the bytes the reference
/// moved, one an M-cycle, since the last. The first difference stops the run.
pub struct ControlLockstep<E: Spans> {
    pub reference: Control,
    pub model: ControlModel<E>,
    /// OAM DMA bytes each side asked for since the last pass boundary, with each line drawn
    /// among them (None) where it fell.
    got_bytes: Vec<Option<Copy>>,
    want_bytes: Vec<Option<Copy>>,
}

impl<E: Spans> ControlLockstep<E> {
    pub fn new(reference: Control, model: ControlModel<E>) -> Self {
        // The reference reads the lengths board as a follower: the model's read alone decides
        // which length revisions the PPU's stepper owes (`Lengths::reissue`), as without it.
        let lengths = reference.lengths().follower();
        let reference = reference.with_lengths(lengths);
        ControlLockstep { reference, model, got_bytes: Vec::new(), want_bytes: Vec::new() }
    }

    fn check(&self, at: &str) {
        let (got, want) = (self.model.state(), self.reference.state());
        if self.model.t.elapsed() == 0 {
            assert_eq!(got, want, "control lockstep: state diverged at pass {} ({at})", self.model.counts.passes);
        } else {
            assert_eq!(untimed(&got), untimed(&want), "control lockstep: state diverged {} M-cycles after pass {} ({at})", self.model.t.elapsed(), self.model.counts.passes);
        }
    }

    /// The bus requests a call added to `got` (the model's) and `want` (the reference's)
    /// from the early and late copies `early` and `from` on: equal but for OAM DMA bytes, which must be equal at each pass
    /// boundary, where the model has landed every byte of its run.
    fn requests(&mut self, got: &Signals, want: &Signals, (early, from): (usize, usize), what: &str) {
        // A line drawn joins the byte stream where it lands among the bytes (None).
        let split = |s: &Signals, bytes: &mut Vec<Option<Copy>>| {
            let mut rest = s.clone();
            rest.early.truncate(early);
            rest.copies.truncate(from);
            let drawn = s.render.map(|_| None);
            for c in s.early[early..].iter().map(Some).chain(drawn).chain(s.copies[from..].iter().map(Some)) {
                match c {
                    Some(c) if !matches!(c, Copy::Byte { .. } | Copy::Fill { .. }) => rest.copies.push(*c),
                    c => bytes.push(c.copied()),
                }
            }
            rest
        };
        let (g, w) = (split(got, &mut self.got_bytes), split(want, &mut self.want_bytes));
        assert_eq!(g, w, "control lockstep: the bus requests of {what} diverged at pass {}", self.model.counts.passes);
        self.boundary(what);
    }

    /// OAM DMA bytes the model's pass at a call without signals moved (`landed`): the
    /// reference moves none there.
    fn landed(&mut self, landed: &[Copy], what: &str) {
        self.got_bytes.extend(landed.iter().copied().map(Some));
        self.boundary(what);
    }

    /// At a pass boundary the model has landed every byte of its run.
    fn boundary(&mut self, what: &str) {
        if self.model.t.elapsed() == 0 {
            assert_eq!(self.got_bytes, self.want_bytes, "control lockstep: the OAM DMA bytes up to pass {} diverged ({what})", self.model.counts.passes);
            self.got_bytes.clear();
            self.want_bytes.clear();
        }
    }
}

impl<E: Spans> ControlDevice for ControlLockstep<E> {
    fn read(&mut self, addr: u16, landed: &mut Vec<Copy>) -> u8 {
        let from = landed.len();
        let (got, want) = (self.model.read(addr, landed), self.reference.read(addr, &mut Vec::new()));
        assert_eq!(got, want, "control lockstep: read {addr:#06x} diverged at pass {}", self.model.counts.passes);
        self.landed(&landed[from..], "a read");
        self.check("at a read");
        got
    }

    fn palette(&mut self, addr: u16, landed: &mut Vec<Copy>) -> PaletteRead {
        let from = landed.len();
        let (got, want) = (self.model.palette(addr, landed), self.reference.palette(addr, &mut Vec::new()));
        assert_eq!(got, want, "control lockstep: the palette byte a read of {addr:#06x} reaches diverged at pass {}", self.model.counts.passes);
        self.landed(&landed[from..], "a palette read");
        self.check("at a palette read");
        got
    }

    fn write(&mut self, addr: u16, value: u8, s: &mut Signals) {
        let (mut want, from) = (s.clone(), (s.early.len(), s.copies.len()));
        self.model.write(addr, value, s);
        self.reference.write(addr, value, &mut want);
        self.requests(s, &want, from, &format!("a write of {addr:#06x}"));
        self.check("after a write");
        assert_eq!(self.model.serial_output(), self.reference.serial_output(), "control lockstep: serial output diverged at pass {}", self.model.counts.passes);
    }

    fn stop(&mut self, s: &mut Signals) {
        let (mut want, from) = (s.clone(), (s.early.len(), s.copies.len()));
        ControlDevice::stop(&mut self.model, s);
        self.reference.stop(&mut want);
        self.requests(s, &want, from, "a STOP");
        self.check("after a STOP");
    }

    fn advance(&mut self, mcycles: u64, stopped: bool, s: &mut Signals) {
        let (mut want, from) = (s.clone(), (s.early.len(), s.copies.len()));
        ControlDevice::advance(&mut self.model, mcycles, stopped, s);
        self.reference.advance(mcycles, stopped, &mut want);
        self.requests(s, &want, from, "an advance");
        self.check("after an advance");
    }

    fn cut(&mut self, s: &mut Signals) {
        let (want, from) = (s.clone(), (s.early.len(), s.copies.len()));
        ControlDevice::cut(&mut self.model, s);
        self.requests(s, &want, from, "a blocked access");
        self.check("after a blocked access");
    }

    fn raise(&mut self, bits: u8, landed: &mut Vec<Copy>) {
        let from = landed.len();
        self.model.raise(bits, landed);
        self.reference.raise(bits, &mut Vec::new());
        self.landed(&landed[from..], "a raise");
        self.check("after a raise");
    }

    fn ack(&mut self, bits: u8, landed: &mut Vec<Copy>) {
        let from = landed.len();
        self.model.ack(bits, landed);
        self.reference.ack(bits, &mut Vec::new());
        self.landed(&landed[from..], "an acknowledgement");
        self.check("after an acknowledgement");
    }

    fn buttons(&mut self, pressed: Buttons, landed: &mut Vec<Copy>) -> Option<bool> {
        let from = landed.len();
        let (got, want) = (self.model.buttons(pressed, landed), self.reference.buttons(pressed, &mut Vec::new()));
        assert_eq!(got, want, "control lockstep: the hold a button change released diverged at pass {}", self.model.counts.passes);
        self.landed(&landed[from..], "a button change");
        self.check("after a button change");
        got
    }

    fn elapse(&mut self, seconds: u64, landed: &mut Vec<Copy>) {
        let from = landed.len();
        self.model.elapse(seconds, landed);
        self.reference.elapse(seconds, &mut Vec::new());
        self.landed(&landed[from..], "the wall time elapsed");
        self.check("after the wall time elapsed");
    }

    fn clock(&self) -> Rtc {
        let (got, want) = (self.model.clock(), self.reference.clock());
        assert_eq!(got, want, "control lockstep: the cartridge clock diverged at pass {}", self.model.counts.passes);
        got
    }

    fn lines(&self) -> Lines {
        let (got, want) = (self.model.lines(), self.reference.lines());
        // The model's locks hold for the M-cycles it said; past them it has none to give.
        assert!(got.locks.is_none_or(|l| Some(l) == want.locks), "control lockstep: the access locks {:?} diverged from {:?} {} M-cycles after pass {} (the model's locks {:?}, LY {} LM {}, the reference {:?})", got.locks, want.locks, self.model.t.elapsed(), self.model.counts.passes, self.model.locks, self.model.state[LY], self.model.state[LM], self.reference.state().lcd);
        assert_eq!(Lines { horizon: 0, locks: None, ..got }, Lines { horizon: 0, locks: None, ..want }, "control lockstep: the lines diverged {} M-cycles after pass {}", self.model.t.elapsed(), self.model.counts.passes);
        got
    }

    fn serial_output(&self) -> &[u8] {
        self.model.serial_output()
    }

    fn step_running(&mut self, running: bool, landed: &mut Vec<Copy>) {
        let from = landed.len();
        self.model.step_running(running, landed);
        self.reference.step_running(running, &mut Vec::new());
        self.landed(&landed[from..], "a step's end");
        self.check("after a step's start or end");
    }

    fn halt(&mut self, landed: &mut Vec<Copy>) {
        let from = landed.len();
        self.model.halt(landed);
        self.reference.halt(&mut Vec::new());
        self.landed(&landed[from..], "a HALT");
        self.check("after a HALT");
    }

    fn woke(&mut self, s: &mut Signals) {
        let (mut want, from) = (s.clone(), (s.early.len(), s.copies.len()));
        self.model.woke(s);
        self.reference.woke(&mut want);
        self.requests(s, &want, from, "a wake");
        self.check("after a wake");
    }

    fn plug(&mut self, _link: Link) -> Result<(), String> {
        Err("a link cable with the control logic in lockstep: both sides would drive the cable".into())
    }

    fn locks(&mut self, landed: &mut Vec<Copy>) -> u8 {
        let from = landed.len();
        let (got, want) = (self.model.locks(landed), self.reference.locks(&mut Vec::new()));
        assert_eq!(got, want, "control lockstep: the access locks a pass gave diverged at pass {}", self.model.counts.passes);
        self.landed(&landed[from..], "a lock query");
        self.check("at a lock query");
        got
    }

    fn ahead(&mut self, mcycles: u8, locks: bool, write: bool) -> u8 {
        let (got, want) = (self.model.ahead(mcycles, locks, write), self.reference.ahead(mcycles, locks, write));
        assert_eq!(got, want, "control lockstep: the bus state {mcycles} M-cycles ahead diverged {} M-cycles after pass {}", self.model.t.elapsed(), self.model.counts.passes);
        got
    }

    fn describe(&self) -> Vec<String> {
        let mut lines = self.model.describe();
        lines.push("control lockstep: machine::control runs beside the model; state, reads, the serial bytes and the machine's lines compared after every pass and every advance".into());
        lines
    }

    fn report(&mut self) -> Vec<String> {
        let mut lines = self.model.report();
        lines.push(format!("control lockstep: {} passes equal to the reference", self.model.counts.passes));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use circuit::{DMA_IDLE, GD};

    /// Stands in for the PPU: a line the pass had drawn gets a length of 44 to 163, 172 to
    /// 291 dots to mode 0 at every residue mod 4, by its LY, on the board both sides read.
    fn drew<E: Spans>(pair: &ControlLockstep<E>, s: &Signals) {
        if let Some(ly) = s.render {
            let lengths = pair.reference.lengths();
            lengths.issue();
            lengths.publish(44 + ly.wrapping_mul(13) % 120);
        }
    }

    const DI: usize = 17;
    const PS: usize = 29;

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

    const ADDRS: [u16; 30] = [0xFF04, 0xFF05, 0xFF06, 0xFF07, 0xFF0F, 0xFFFF, 0xFF00, 0xFF01, 0xFF02, 0xFF4C, 0xFF4D, 0xFF70, 0xFF46, 0xFF51, 0xFF52, 0xFF53, 0xFF54, 0xFF55, 0xFF40, 0xFF41, 0xFF44, 0xFF45, 0xFF68, 0xFF69, 0xFF6A, 0xFF6B, 0xFF72, 0xFF73, 0xFF74, 0xFF75];

    /// The registers a game touches while the line-timing schedule is in force. None of
    /// them starts the timer, a serial transfer, OAM DMA or HDMA, and IF writes mostly set bit 1.
    const QUIET: [u16; 16] = [0xFF00, 0xFF01, 0xFF0F, 0xFF4C, 0xFF40, 0xFF41, 0xFF44, 0xFF45, 0xFF68, 0xFF69, 0xFF6A, 0xFF70, 0xFF72, 0xFF73, 0xFF74, 0xFF75];

    /// Random writes, reads, acknowledgements, raises, button changes and advances
    /// through `eval` in lockstep with the reference; advances stop at the model's
    /// horizon as the machine's scheduler cuts them.
    pub(crate) fn random_stream<E: Spans>(eval: E, cgb: bool, seed: u64, events: usize) -> Counts {
        random_stream_in(eval, cgb, seed, events, false)
    }

    /// `random_stream`, or with `quiet` the traffic of a game the line-timing schedule
    /// serves. That traffic touches only `QUIET`'s registers, raises IF bit 1 often and
    /// rarely acknowledges it.
    pub(crate) fn random_stream_in<E: Spans>(eval: E, cgb: bool, seed: u64, events: usize, quiet: bool) -> Counts {
        let reference = Control::post_boot(cgb, 0x80);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut rng = Rng(seed);
        let mut s = Signals::default();
        let addrs: &[u16] = if quiet { &QUIET } else { &ADDRS };
        for _ in 0..events {
            match rng.below(17) {
                0..=3 => {
                    let addr = addrs[rng.below(addrs.len() as u64) as usize];
                    let mut value = rng.next() as u8;
                    if addr == 0xFF07 && rng.below(3) != 0 {
                        value |= 4;
                    }
                    if addr == 0xFF05 && rng.below(2) == 0 {
                        value |= 0xF0;
                    }
                    if addr == 0xFF06 && rng.below(3) == 0 {
                        value = 0xFF;
                    }
                    if addr == 0xFF02 && rng.below(2) == 0 {
                        value |= 0x81;
                    }
                    if addr == 0xFF53 && rng.below(2) == 0 {
                        value |= 0x1E;
                    }
                    if addr == 0xFF55 && rng.below(3) != 0 {
                        value &= 0x8F;
                    }
                    if addr == 0xFF40 && rng.below(8) != 0 {
                        value |= 0x80;
                    }
                    if addr == 0xFF45 {
                        value = rng.below(155) as u8;
                    }
                    if quiet && addr == 0xFF0F && rng.below(8) != 0 {
                        value |= 2;
                    }
                    if addr == 0xFF46 && pair.lines().dma_active && rng.below(4) != 0 {
                        continue;
                    }
                    s.reset(pair.lines().double);
                    pair.write(addr, value, &mut s); drew(&pair, &s);
                }
                4 | 5 => {
                    // LCDC reads come from the PPU, which holds all of its bits.
                    let addr = match addrs[rng.below(addrs.len() as u64) as usize] {
                        0xFF40 => 0xFF44,
                        a => a,
                    };
                    if matches!(addr, 0xFF69 | 0xFF6B) {
                        pair.palette(addr, &mut Vec::new());
                    } else if rng.below(3) == 0 {
                        // The OAM row the scan reads, which the host asks at an FE page address.
                        pair.oam_row(&mut Vec::new());
                    } else {
                        pair.read(addr, &mut Vec::new());
                    }
                }
                6 => pair.ack(1 << if quiet && rng.below(16) != 0 { [0, 2, 3, 4][rng.below(4) as usize] } else { rng.below(5) }, &mut Vec::new()),
                7 => pair.raise(1 << if quiet && rng.below(2) == 0 { 1 } else { rng.below(5) }, &mut Vec::new()),
                8 => {
                    pair.buttons(Buttons(rng.next() as u8), &mut Vec::new());
                }
                // A STOP, mostly with a speed switch armed, then its pause as the machine
                // runs it, the CPU stopped.
                9 if cgb && rng.below(6) == 0 => {
                    if rng.below(4) != 0 {
                        s.reset(pair.lines().double);
                        pair.write(0xFF4D, 1, &mut s); drew(&pair, &s);
                    }
                    s.reset(pair.lines().double);
                    pair.stop(&mut s); drew(&pair, &s);
                    stopped(&mut pair, &mut s);
                }
                _ => {
                    let span = [3, 24, 300, 3000][rng.below(4) as usize];
                    let mut m = 1 + rng.below(span);
                    while m > 0 {
                        let block = m.min(pair.lines().horizon);
                        s.reset(pair.lines().double);
                        pair.advance(block, false, &mut s); drew(&pair, &s);
                        m -= block;
                    }
                }
            }
        }
        pair.model.counts
    }

    /// Mode 0's source (mode 2's off) or the LYC match holds the STAT line, and the next
    /// line's start has a source on (LYC, or line 144's mode 1 or, on a DMG, mode 2): the line
    /// drops in the line's last M-cycle and rises at the next line's start, a STAT interrupt
    /// after IF bit 1 was cleared. LYC 153 holds it only in line 153's first M-cycle, and
    /// line 0's mode 2 raises it again. One span covers both from the IF write unless the
    /// model passes at the drop.
    #[test]
    fn a_stat_line_dropped_in_the_last_m_cycle_rises_at_the_next_line() {
        let cases = [(true, 0x48, 52, 51, 100), (false, 0x48, 52, 51, 100), (true, 0x18, 0, 143, 100), (true, 0x50, 143, 143, 100), (false, 0x50, 143, 143, 100), (false, 0x60, 143, 143, 100), (true, 0x60, 153, 153, 0), (false, 0x60, 153, 153, 0)];
        for c in cases {
            stat_dip(c);
        }
    }

    fn stat_dip((cgb, stat, lyc, ly, m): (bool, u8, u8, u8, u8)) {
        let reference = Control::post_boot(cgb, 0x80);
        let model = ControlModel::new(CircuitSpans(build()), &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut s = Signals::default();
        for (addr, value) in [(0xFF41, stat), (0xFF45, lyc)] {
            s.reset(pair.lines().double);
            pair.write(addr, value, &mut s);
            drew(&pair, &s);
        }
        while (pair.reference.state().lcd.ly, pair.reference.state().lcd.m) != (ly, m) {
            s.reset(pair.lines().double);
            pair.advance(1, false, &mut s);
            drew(&pair, &s);
        }
        s.reset(pair.lines().double);
        pair.write(0xFF0F, 0, &mut s);
        drew(&pair, &s);
        run_for(&mut pair, &mut s, 134 - m as u64);
        assert_eq!(pair.reference.state().iflag & 2, 2, "cgb {cgb}, line {ly}: the reference raised no STAT interrupt");
    }

    /// STAT traffic alone, against the reference: random enables, LYC on this line, the
    /// next, 0, 143, 144 or 153, IF bit 1 cleared at random M-cycles, spans cut at the
    /// horizon or single M-cycles. Every STAT rise must come out of a pass the same.
    #[test]
    fn random_stat_traffic_matches_the_reference() {
        for (cgb, seed) in [(false, 0x5DEE_CE66_D1CE_4E5B), (true, 0x2545_F491_4F6C_DD1D)] {
            let reference = Control::post_boot(cgb, 0x80);
            let model = ControlModel::new(CircuitSpans(build()), &reference.state(), reference.lengths());
            let mut pair = ControlLockstep::new(reference, model);
            let mut rng = Rng(seed);
            let mut s = Signals::default();
            let mut rises = 0;
            for _ in 0..20_000 {
                let ly = pair.reference.state().lcd.ly;
                let write = |pair: &mut ControlLockstep<CircuitSpans>, s: &mut Signals, addr: u16, value: u8| {
                    s.reset(pair.lines().double);
                    pair.write(addr, value, s);
                    drew(pair, s);
                };
                match rng.below(8) {
                    0 => write(&mut pair, &mut s, 0xFF41, rng.next() as u8 & 0x78),
                    1 => {
                        let lyc = [ly, (ly + 1) % 154, 0, 143, 144, 153][rng.below(6) as usize];
                        write(&mut pair, &mut s, 0xFF45, lyc);
                    }
                    2 | 3 => {
                        let before = pair.reference.state().iflag;
                        write(&mut pair, &mut s, 0xFF0F, before & !2);
                    }
                    4 => run_for(&mut pair, &mut s, 1),
                    _ => run_for(&mut pair, &mut s, 1 + rng.below(300)),
                }
                rises += (pair.reference.state().iflag & 2 != 0) as u64;
            }
            assert!(rises > 1_000, "cgb {cgb}: {rises} events with IF bit 1 set");
        }
    }

    #[test]
    fn random_control_traffic_matches_the_reference_on_dmg_and_cgb() {
        for (cgb, seed) in [(false, 0x9E37_79B9_7F4A_7C15), (true, 0xD1B5_4A32_D192_ED03)] {
            let c = random_stream(CircuitSpans(build()), cgb, seed, 20_000);
            assert!(c.ticks > 1_000 && c.reads > 1_000 && c.writes > 3_000 && c.buttons > 500 && c.renders > 10_000 && c.frames > 50 && (c.stops > 30) == cgb, "{c:?}");
        }
    }

    /// The line-timing schedule against the reference on a game's quiet traffic. Scheduled
    /// starts must draw over 10,000 lines and more than a quarter of all lines drawn, each
    /// equal to the reference's.
    #[test]
    fn scheduled_mode_3_starts_match_the_reference() {
        for (cgb, seed) in [(false, 0x2545_F491_4F6C_DD1D), (true, 0x9E37_79B9_7F4A_7C15)] {
            let c = random_stream_in(CircuitSpans(build()), cgb, seed, 20_000, true);
            // A CGB stream's ordinary STOPs hold the GBC CPU, and the schedule waits them out.
            assert!(c.scheduled > 10_000 && 5 * c.scheduled > c.renders, "{c:?}");
        }
    }

    /// The circuit with one output changed after the fact.
    struct Mutant(CircuitSpans, fn(&Span, &mut Outcome));

    impl Spans for Mutant {
        fn run(&mut self, state: &State, span: &Span, length: &mut dyn FnMut() -> u8) -> Outcome {
            let mut out = self.0.run(state, span, length);
            (self.1)(span, &mut out);
            out
        }
    }

    #[test]
    fn the_lockstep_catches_a_late_horizon_and_a_lost_reload() {
        let mutants: [(&str, fn(&Span, &mut Outcome)); 19] = [
            ("a frame one half long", |_, out| out.frame += 1),
            ("a speed switch that keeps the speed", |span, out| {
                if span.event == Event::Stop {
                    out.state[K1] = out.state[K1] & 0x7F | !out.state[K1] & 0x80;
                }
            }),
            ("a speed switch that keeps real time's rate", |span, out| {
                if span.event == Event::Stop {
                    out.rate = 3 - out.rate;
                }
            }),
            ("a pause one M-cycle short", |span, out| {
                if span.event == Event::Stop && out.state[PS] > 0 {
                    out.state[PS] -= 1;
                }
            }),
            ("DIV ticking in the pause", |span, out| {
                if span.paused && span.mcycles > 0 {
                    out.state[0] = out.state[0].wrapping_add(4);
                }
            }),
            ("horizon one late", |_, out| out.horizon = (out.horizon + 1).min(MAX_SPAN as u16)),
            ("reload without the interrupt", |span, out| {
                if span.event == Event::Tick && span.mcycles > 0 {
                    out.state[IF] &= !4;
                }
            }),
            ("DIV read off by one", |span, out| {
                if span.event == Event::Read(0xFF04) {
                    out.read ^= 1;
                }
            }),
            ("serial completion without the interrupt", |span, out| {
                if span.event == Event::Tick {
                    out.state[IF] &= !8;
                }
            }),
            ("a button press without the joypad interrupt", |span, out| {
                if matches!(span.event, Event::Buttons(_)) {
                    out.state[IF] &= !0x10;
                }
            }),
            ("a transfer that never reports its byte", |_, out| out.flags &= !FX_SERIAL_START),
            ("OAM DMA ends a byte early", |_, out| {
                if out.state[DI] == DMA_IDLE - 1 {
                    out.state[DI] = DMA_IDLE;
                }
            }),
            ("a general-purpose HDMA a block short", |_, out| {
                if out.state[HDMA3] & GD != 0 && out.state[HDMA5] == 0 {
                    out.state[HDMA3] &= !GD;
                }
            }),
            ("a STAT rising edge without its interrupt", |span, out| {
                if span.event == Event::Tick && out.state[IF] & 2 != 0 && out.state[26] & 1 != 0 {
                    out.state[IF] &= !2;
                }
            }),
            ("VBlank without its interrupt", |_, out| {
                if out.flags & FX_FRAME != 0 {
                    out.state[IF] &= !1;
                }
            }),
            ("a line never drawn", |_, out| {
                if out.state[LY] == 77 {
                    out.flags &= !FX_RENDER;
                }
            }),
            ("LY read off by one", |span, out| {
                if span.event == Event::Read(0xFF44) {
                    out.read ^= 1;
                }
            }),
            ("a palette write to the wrong byte", |span, out| {
                if matches!(span.event, Event::Write(0xFF69, _)) {
                    out.palette = Some(0);
                }
            }),
            ("a palette read that reaches the wrong byte", |span, out| {
                if let (Event::Read(0xFF6B), Some(i)) = (span.event, &mut out.palette) {
                    *i ^= 1;
                }
            }),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| random_stream(Mutant(CircuitSpans(build()), mutate), true, 7, 20_000)).is_err();
            assert!(caught, "{name}: not caught");
        }
    }

    #[test]
    fn the_lockstep_catches_a_wrong_edge_or_wake() {
        // The stream itself passes, so each mutant's panic is the lockstep's.
        let c = random_stream(Mutant(CircuitSpans(build()), |_, _| {}), true, 7, 20_000);
        assert!(c.stops > 30 && c.writes > 3_000, "{c:?}");
        let mutants: [(&str, fn(&Span, &mut Outcome)); 6] = [
            ("a wake bit flipped", |_, out| out.wake = !out.wake),
            ("a DIV reset's drop one M-cycle late", |_, out| out.drop *= 2),
            ("an edge one M-cycle late", |_, out| out.edge = out.edge.map(|e| e + 1)),
            ("an edge one M-cycle early", |_, out| out.edge = out.edge.map(|e| e.max(2) - 1)),
            ("a DIV reset's drop lost", |_, out| out.drop = 0),
            ("an edge while the pause holds the counter", |_, out| out.edge = out.edge.or(Some(1))),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| random_stream(Mutant(CircuitSpans(build()), mutate), true, 7, 20_000)).is_err();
            assert!(caught, "{name}: not caught");
        }
    }

    #[test]
    fn the_lockstep_catches_a_wrong_schedule() {
        let mutants: [(&str, fn(&Span, &mut Outcome)); 6] = [
            ("a scheduled period one M-cycle long", |_, out| out.schedule.period += 1),
            ("the pass after a schedule one M-cycle late", |_, out| {
                if out.schedule.count > 0 {
                    out.schedule.tail += 1;
                }
            }),
            ("a scheduled line one off", |_, out| out.schedule.lines[1] ^= 1),
            // One start short is no fault, because the next pass then draws that line itself.
            ("a schedule one start long", |_, out| {
                if out.schedule.count > 0 {
                    out.schedule.count += 1;
                }
            }),
            ("a scheduled start one M-cycle late", |_, out| {
                if out.schedule.count > 0 {
                    out.schedule.first += 1;
                }
            }),
            ("a scheduled position one M-cycle late", |_, out| out.schedule.phase = out.schedule.phase.wrapping_add(2)),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| random_stream_in(Mutant(CircuitSpans(build()), mutate), true, 7, 20_000, true)).is_err();
            assert!(caught, "{name}: not caught");
        }
    }

    /// Random cartridge clock traffic (the RAM enable, the select, the latch, register
    /// writes and reads, and wall time from a second to past u64's days) mixed with control
    /// traffic, through `eval` in lockstep with the reference. Every clock event is a pass.
    pub(crate) fn clock_stream<E: Spans>(eval: E, seed: u64, events: usize) -> Counts {
        let reference = Control::post_boot(true, 0x80);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut rng = Rng(seed);
        let mut s = Signals::default();
        for _ in 0..events {
            s.reset(pair.lines().double);
            match rng.below(16) {
                0 => {
                    pair.write(0x1000, if rng.below(4) != 0 { 0x0A } else { rng.next() as u8 }, &mut s);
                    drew(&pair, &s)
                }
                1 | 2 => {
                    pair.write(0x4000 | rng.below(0x2000) as u16, if rng.below(6) != 0 { 8 + rng.below(5) as u8 } else { rng.next() as u8 }, &mut s);
                    drew(&pair, &s)
                }
                3 | 4 => {
                    pair.write(0x6000 | rng.below(0x2000) as u16, [0, 1, 1, rng.next() as u8][rng.below(4) as usize], &mut s);
                    drew(&pair, &s)
                }
                5 | 6 => {
                    // Halt set or cleared in DH often, so both the halted and the running
                    // clock see time pass; S, M and H past their ranges.
                    let value = match rng.below(3) {
                        0 => [0x00, 0x01, 0x40, 0x41, 0x80, 0xC1][rng.below(6) as usize],
                        1 => 50 + rng.below(30) as u8,
                        _ => rng.next() as u8,
                    };
                    pair.write(0xA000 | rng.below(0x2000) as u16, value, &mut s); drew(&pair, &s);
                }
                7..=9 => {
                    pair.read(0xA000 | rng.below(0x2000) as u16, &mut Vec::new());
                }
                10..=12 => {
                    let seconds = match rng.below(6) {
                        0 => 1 + rng.below(3),
                        1 => rng.below(4000),
                        2 => rng.below(86400 * 3),
                        3 => rng.below(86400 * 600),
                        4 => rng.next() >> rng.below(64),
                        _ => u64::MAX - rng.below(1000),
                    };
                    pair.elapse(seconds, &mut Vec::new());
                }
                13 => {
                    pair.write(ADDRS[rng.below(ADDRS.len() as u64) as usize], rng.next() as u8 & 0x7F, &mut s);
                    drew(&pair, &s)
                }
                _ => {
                    let mut m = 1 + rng.below(300);
                    while m > 0 {
                        let block = m.min(pair.lines().horizon);
                        s.reset(pair.lines().double);
                        pair.advance(block, false, &mut s); drew(&pair, &s);
                        m -= block;
                    }
                }
            }
        }
        assert_eq!(pair.clock(), pair.reference.clock());
        pair.model.counts
    }

    #[test]
    fn random_clock_traffic_matches_the_reference() {
        let c = clock_stream(CircuitSpans(build()), 0x5DEE_CE66_D1CE_4E5B, 20_000);
        assert!(c.clock_writes > 7_000 && c.clock_reads > 3_000 && c.clock_times > 3_000, "{c:?}");
    }

    #[test]
    fn random_clock_traffic_through_the_engine_matches_the_reference() {
        let c = clock_stream(ControlCheckpoint::new(engine()), 0xA076_1D64_78BD_642F, 5_000);
        assert!(c.clock_writes > 1_500 && c.clock_reads > 800 && c.clock_times > 800, "{c:?}");
    }

    #[test]
    fn the_lockstep_catches_a_wrong_clock() {
        fn ck(out: &Outcome, i: usize) -> u8 {
            out.state[CK + i]
        }
        let mutants: [(&str, fn(&Span, &mut Outcome)); 7] = [
            ("a minute short after time passes", |span, out| {
                if matches!(span.event, Event::Elapse(_)) && ck(out, 1) & 0x3F > 0 {
                    out.state[CK + 1] -= 1;
                }
            }),
            ("the day counter's carry never set", |_, out| out.state[CK] &= 0x7F),
            ("time passing while halted", |span, out| {
                if matches!(span.event, Event::Elapse(_)) && ck(out, 0) & 0x40 != 0 {
                    out.state[CK] = ck(out, 0) & 0xC0 | ((ck(out, 0) & 0x3F) + 1) % 60;
                }
            }),
            ("a latch that drops halt and the carry", |span, out| {
                if matches!(span.event, Event::Write(0x6000..=0x7FFF, 1)) {
                    out.state[CK + 4] &= 0x3F;
                }
            }),
            ("a clock read off by one", |span, out| {
                if matches!(span.event, Event::Read(0xA000..=0xBFFF)) {
                    out.read ^= 1;
                }
            }),
            ("RAM disable ignored", |span, out| {
                if matches!(span.event, Event::Write(0x0000..=0x1FFF, _)) {
                    out.state[CK + 1] |= 0x80;
                }
            }),
            ("a seconds write not reduced modulo 60", |span, out| {
                if let Event::Write(0xA000..=0xBFFF, v) = span.event
                    && (60..64).contains(&v)
                    && ck(out, 2) >> 5 == 1
                    && ck(out, 1) & 0x80 != 0
                {
                    out.state[CK] = ck(out, 0) & 0xC0 | v;
                }
            }),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| clock_stream(Mutant(CircuitSpans(build()), mutate), 11, 20_000)).is_err();
            assert!(caught, "{name}: not caught");
        }
    }

    /// The M-cycles the pair says the GBC CPU stays stopped, in blocks cut at its horizon, as
    /// the machine's scheduler runs them after a step.
    fn stopped<E: Spans>(pair: &mut ControlLockstep<E>, s: &mut Signals) {
        while pair.lines().owed > 0 {
            let block = pair.lines().owed.min(pair.lines().horizon);
            s.reset(pair.lines().double);
            pair.advance(block, true, s);
            drew(pair, s);
        }
    }

    /// Advances `m` M-cycles in blocks cut at the pair's horizon, as the machine's
    /// scheduler cuts them.
    fn run_for<E: Spans>(pair: &mut ControlLockstep<E>, s: &mut Signals, mut m: u64) {
        while m > 0 {
            let block = m.min(pair.lines().horizon);
            s.reset(pair.lines().double);
            pair.advance(block, false, s);
            drew(pair, s);
            m -= block;
        }
    }

    /// What a DMA stream ran: the control counts, the general-purpose transfers the end
    /// of VRAM cut short, those that started at its last block, and the HBlank blocks
    /// moved at double speed.
    #[derive(Debug)]
    pub(crate) struct DmaRun {
        counts: Counts,
        cut: u64,
        at_end: u64,
        double_hblank: u64,
        /// HDMA blocks held to the end of a GBC CPU step.
        held: u64,
        /// Wakes from a HALT.
        halts: u64,
        /// Accesses in the step after a block it held (HDMA3's `HP` set): FF55 reads, FF55
        /// writes and FF53 or FF54 writes.
        after_held: [u64; 3],
        /// Of those, accesses after a held block that ran off the end of VRAM (`HE`).
        after_held_off: u64,
        /// HBlank starts and HALTs between STAT's mode 0 and the line's request (`Lcd::in_gap`).
        gap: [u64; 2],
    }

    /// DMA traffic on CGB in lockstep with the reference: general-purpose transfers of
    /// up to 128 blocks, many from near the end of VRAM; HBlank transfers across lines,
    /// cancelled and restarted; OAM DMA, sometimes restarted mid-transfer; speed
    /// switches; and every stall run as the machine runs it, the CPU stopped.
    pub(crate) fn dma_stream<E: Spans>(eval: E, seed: u64, events: usize) -> DmaRun {
        let reference = Control::post_boot(true, 0x80);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut rng = Rng(seed);
        let mut s = Signals::default();
        let mut run = DmaRun { counts: Counts::default(), cut: 0, at_end: 0, double_hblank: 0, held: 0, halts: 0, after_held: [0; 3], after_held_off: 0, gap: [0; 2] };
        // An HDMA access later in a step that held a block, from a stream of its own too.
        let mut after = Rng(seed ^ 0x1405_7B7E_F767_814F);
        // Whether a GBC CPU step runs, from a stream of its own so the traffic stays the same.
        let mut steps = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        // HALTs and wakes, from a stream of their own too.
        let (mut halts, mut halted) = (Rng(seed ^ 0x5851_F42D_4C95_7F2D), false);
        s.reset(pair.lines().double);
        pair.write(0xFF40, 0x91, &mut s); drew(&pair, &s);
        for _ in 0..events {
            let (_, dst, h5, ..) = pair.reference.state().hdma;
            s.reset(pair.lines().double);
            // A HALT and a wake fall between steps, as the machine runs them.
            if halts.below(8) == 0 {
                if halted {
                    pair.woke(&mut s);
                    drew(&pair, &s);
                    run.halts += 1;
                    stopped(&mut pair, &mut s);
                    s.reset(pair.lines().double);
                } else {
                    run.gap[1] += (pair.reference.in_gap() && pair.reference.state().hdma.2 & 0x80 == 0) as u64;
                    pair.halt(&mut Vec::new());
                }
                halted = !halted;
            }
            // A step runs a few M-cycles; its stall runs after its end (`stall`).
            let running = steps.below(2) == 0;
            pair.step_running(running, &mut Vec::new());
            let mut stall = false;
            match rng.below(16) {
                0 => {
                    pair.write(0xFF51, rng.next() as u8, &mut s);
                    drew(&pair, &s)
                }
                1 => {
                    pair.write(0xFF52, rng.next() as u8, &mut s);
                    drew(&pair, &s)
                }
                2 => {
                    pair.write(0xFF53, rng.next() as u8 | if rng.below(2) == 0 { 0x1F } else { 0 }, &mut s);
                    drew(&pair, &s)
                }
                3 => {
                    pair.write(0xFF54, if rng.below(4) == 0 { 0xF0 } else { rng.next() as u8 }, &mut s);
                    drew(&pair, &s)
                }
                4 | 5 => {
                    let value = if rng.below(3) == 0 { rng.next() as u8 & 0x7F } else { rng.below(12) as u8 };
                    if h5 & 0x80 != 0 {
                        run.cut += (dst / 16 + value as u16 + 1 > 512) as u64;
                        run.at_end += (dst == 0x1FF0) as u64;
                    }
                    pair.write(0xFF55, value, &mut s); drew(&pair, &s);
                    run.held += s.held.len() as u64;
                    stall = true;
                }
                6 => {
                    run.gap[0] += pair.reference.in_gap() as u64;
                    pair.write(0xFF55, 0x80 | rng.below(24) as u8, &mut s);
                    drew(&pair, &s)
                }
                10 if pair.lines().dma_active => {
                    pair.cut(&mut s);
                    drew(&pair, &s)
                }
                7 => {
                    if !pair.lines().dma_active || rng.below(8) == 0 {
                        pair.write(0xFF46, [0xC0, 0xD0, 0xE3, 0x80, rng.next() as u8][rng.below(5) as usize], &mut s); drew(&pair, &s);
                    }
                }
                8 if rng.below(4) == 0 => {
                    pair.write(0xFF40, if rng.below(8) != 0 { 0x91 } else { 0x11 }, &mut s);
                    drew(&pair, &s)
                }
                9 if rng.below(3) == 0 => {
                    pair.write(0xFF4D, 1, &mut s); drew(&pair, &s);
                    s.reset(pair.lines().double);
                    pair.stop(&mut s); drew(&pair, &s);
                    stall = true;
                }
                _ => {
                    let span = if running { 24 } else { [24, 300, 3000][rng.below(3) as usize] };
                    let mut left = 1 + rng.below(span);
                    while left > 0 {
                        let block = left.min(pair.lines().horizon);
                        s.reset(pair.lines().double);
                        pair.advance(block, false, &mut s); drew(&pair, &s);
                        run.held += s.held.len() as u64;
                        let hblank = s.copies.iter().chain(&s.held).any(|c| matches!(c, Copy::Block { .. }));
                        run.double_hblank += (hblank && pair.lines().double) as u64;
                        stall = true;
                        left -= block;
                    }
                    // A step that held a block goes on to an HDMA register: the held count
                    // meets a read, a new count, or a new destination.
                    if running && pair.reference.state().hdma.3 {
                        s.reset(pair.lines().double);
                        let k = after.below(3) as usize;
                        match k {
                            0 => {
                                pair.read(0xFF55, &mut Vec::new());
                            }
                            1 => {
                                pair.write(0xFF55, after.next() as u8 & 0x87, &mut s);
                                drew(&pair, &s);
                                stall = true;
                            }
                            _ => {
                                pair.write(0xFF53 + after.below(2) as u16, after.next() as u8, &mut s);
                                drew(&pair, &s)
                            }
                        }
                        run.after_held[k] += 1;
                        run.after_held_off += pair.reference.state().hdma.5 as u64;
                    }
                }
            }
            if running {
                pair.step_running(false, &mut Vec::new());
            }
            if stall {
                stopped(&mut pair, &mut s);
            }
        }
        run.counts = pair.model.counts;
        run
    }

    #[test]
    fn random_dma_traffic_matches_the_reference() {
        let r = dma_stream(CircuitSpans(build()), 0x3C6E_F372_FE94_F82B, 20_000);
        let c = &r.counts;
        assert!(c.blocks > 20_000 && c.oam_bytes > 50_000 && c.stops > 100 && r.cut > 200 && r.at_end > 20 && r.double_hblank > 500 && r.held > 1_000 && r.halts > 1_000 && r.counts.halts > 500 && r.after_held.iter().all(|&n| n >= 5) && r.after_held_off >= 1 && r.gap.iter().all(|&n| n >= 1), "{r:?}");
    }

    #[test]
    fn random_dma_traffic_through_the_engine_matches_the_reference() {
        let r = dma_stream(ControlCheckpoint::new(engine()), 0xBB67_AE85_84CA_A73B, 4_000);
        assert!(r.counts.blocks > 3_000 && r.counts.oam_bytes > 10_000 && r.cut > 40 && r.held > 100, "{r:?}");
    }

    #[test]
    fn the_lockstep_catches_a_wrong_dma_byte_or_stall() {
        let mutants: [(&str, fn(&Span, &mut Outcome)); 14] = [
            ("a stop emitted as what the pass added, not the total", |span, out| out.stall -= span.owed),
            ("a general-purpose transfer's further blocks never asked for", |_, out| out.follow = None),
            ("an HDMA block's source bit 4 flipped", |_, out| {
                if let Some((source, _, _, _)) = &mut out.block {
                    *source ^= 0x10;
                }
            }),
            ("an HDMA block's destination bit 12 flipped", |_, out| {
                if let Some((_, dest, _, _)) = &mut out.block {
                    *dest ^= 0x1000;
                }
            }),
            ("a stall one M-cycle short", |_, out| out.stall = out.stall.saturating_sub(1)),
            ("a double-speed block's stall at normal speed", |_, out| {
                if out.stall == 16 {
                    out.stall = 8;
                }
            }),
            ("an OAM DMA run one byte to the side", |_, out| {
                if let Some((_, _, before, last)) = &mut out.oam
                    && (*before | *last) < 0x80
                {
                    (*before, *last) = (*before << 1, *last << 1);
                }
            }),
            ("an OAM DMA run in the next line", |_, out| {
                if let Some((_, line, _, _)) = &mut out.oam {
                    *line ^= 8;
                }
            }),
            ("an OAM DMA source in the echo not mapped to WRAM", |_, out| {
                if let Some((page, _, _, _)) = &mut out.oam
                    && *page >= 0xC0
                {
                    *page |= 0x20;
                }
            }),
            ("an HBlank block that never moves", |span, out| {
                if span.event == Event::Tick {
                    out.block = None;
                }
            }),
            ("an HDMA block half as long", |_, out| {
                if let Some((_, _, len, _)) = &mut out.block {
                    *len /= 2;
                }
            }),
            ("OAM DMA into the page below OAM", |_, out| out.oam_dest = out.oam_dest.wrapping_sub(1)),
            ("an HDMA block inside a GBC CPU step landed at once", |span, out| {
                if let Some((_, _, _, held)) = &mut out.block
                    && span.step
                {
                    *held = false;
                }
            }),
            ("an HDMA block outside a step held", |span, out| {
                if let Some((_, _, _, held)) = &mut out.block
                    && !span.step
                {
                    *held = true;
                }
            }),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| dma_stream(Mutant(CircuitSpans(build()), mutate), 13, 5_000)).is_err();
            assert!(caught, "{name}: not caught");
        }
    }


    /// OAM DMA transfers started at random points of the PPU's lines, mode 2 and mode 3
    /// included, each run beside CPU accesses the transfer blocks, acknowledgements,
    /// raises and button changes, in lockstep with the reference. Returns the counts and
    /// how many transfers started in modes 0, 1, 2 and 3.
    fn oam_dma_stream<E: Spans>(eval: E, cgb: bool, seed: u64, transfers: usize) -> (Counts, [u64; 4]) {
        let reference = Control::post_boot(cgb, 0x80);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut rng = Rng(seed);
        let mut s = Signals::default();
        let mut modes = [0u64; 4];
        s.reset(pair.lines().double);
        pair.write(0xFF40, 0x91, &mut s); drew(&pair, &s);
        for _ in 0..transfers {
            run_for(&mut pair, &mut s, 1 + rng.below(456));
            let lcd = pair.reference.state().lcd;
            modes[if lcd.ly >= 144 { 1 } else if lcd.m < 20 { 2 } else if lcd.m < 63 { 3 } else { 0 }] += 1;
            s.reset(pair.lines().double);
            pair.write(0xFF46, [0xC0, 0xD1, 0xE2, 0x80][rng.below(4) as usize], &mut s); drew(&pair, &s);
            // The transfer runs from the write, though it blocks only from two M-cycles on.
            while pair.reference.state().dma.1 != 160 {
                s.reset(pair.lines().double);
                match rng.below(12) {
                    0..=3 => {
                    pair.cut(&mut s);
                    drew(&pair, &s)
                }
                    4 => pair.ack(1 << rng.below(5), &mut Vec::new()),
                    5 => pair.raise(1 << rng.below(5), &mut Vec::new()),
                    6 => {
                        pair.buttons(Buttons(rng.next() as u8), &mut Vec::new());
                    }
                    _ => run_for(&mut pair, &mut s, 1 + rng.below(6)),
                }
            }
        }
        (pair.model.counts, modes)
    }

    /// An HBlank block held to the end of a GBC CPU step that runs STOP, a speed switch:
    /// the pause adds to the block's stall, the stops one after the other.
    fn held_block_then_switch<E: Spans>(eval: E) -> u64 {
        let reference = Control::post_boot(true, 0x80);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut s = Signals::default();
        s.reset(pair.lines().double);
        pair.write(0xFF40, 0x91, &mut s); drew(&pair, &s);
        s.reset(pair.lines().double);
        pair.write(0xFF55, 0x85, &mut s); drew(&pair, &s);
        stopped(&mut pair, &mut s);
        pair.step_running(true, &mut Vec::new());
        loop {
            let block = pair.lines().horizon;
            s.reset(pair.lines().double);
            pair.advance(block, false, &mut s); drew(&pair, &s);
            if !s.held.is_empty() {
                break;
            }
        }
        s.reset(pair.lines().double);
        pair.write(0xFF4D, 1, &mut s); drew(&pair, &s);
        pair.step_running(false, &mut Vec::new());
        s.reset(pair.lines().double);
        pair.stop(&mut s); drew(&pair, &s);
        let owed = pair.lines().owed;
        stopped(&mut pair, &mut s);
        owed
    }

    #[test]
    fn a_switch_after_a_held_block_runs_after_its_stall() {
        assert_eq!(held_block_then_switch(CircuitSpans(build())), 8 + crate::machine::dma::RUN_MCYCLES + crate::machine::control::SWITCH_MCYCLES as u64);
        let caught = std::panic::catch_unwind(|| held_block_then_switch(Mutant(CircuitSpans(build()), |span, out| {
            if span.event == Event::Stop {
                out.stall -= span.owed;
            }
        })))
        .is_err();
        assert!(caught, "a switch's pause that drops the stall owed: not caught");
    }

    /// OAM DMA transfers, each written at a random M-cycle, then the bus state the next
    /// four M-cycles as a GBC CPU step's accesses ask for it (`ControlDevice::ahead`): the
    /// transfer the write starts blocks from the M-cycle the model emits (`DZ`).
    fn oam_dma_ahead<E: Spans>(eval: E) {
        let reference = Control::post_boot(false, 0);
        let model = ControlModel::new(eval, &reference.state(), reference.lengths());
        let mut pair = ControlLockstep::new(reference, model);
        let mut rng = Rng(0x5851_F42D_4C95_7F2D);
        let mut s = Signals::default();
        for _ in 0..200 {
            run_for(&mut pair, &mut s, 1 + rng.below(700));
            s.reset(pair.lines().double);
            pair.write(0xFF46, 0xC0, &mut s); drew(&pair, &s);
            for m in 1..=4 {
                pair.ahead(m, false, false);
            }
        }
    }

    #[test]
    fn the_lockstep_catches_a_fresh_oam_dma_blocking_a_cycle_off() {
        oam_dma_ahead(CircuitSpans(build()));
        let mutants: [(&str, fn(&Span, &mut Outcome)); 2] = [
            ("a fresh transfer blocking one M-cycle early", |_, out| out.dma_from = out.dma_from.saturating_sub(1)),
            ("a fresh transfer blocking one M-cycle late", |_, out| {
                if out.dma_from > 0 {
                    out.dma_from += 1;
                }
            }),
        ];
        for (name, mutate) in mutants {
            let caught = std::panic::catch_unwind(|| oam_dma_ahead(Mutant(CircuitSpans(build()), mutate))).is_err();
            assert!(caught, "{name}: not caught");
        }
    }

    #[test]
    fn the_lockstep_catches_a_run_s_last_byte_before_the_line_it_starts() {
        let caught = std::panic::catch_unwind(|| oam_dma_stream(Mutant(CircuitSpans(build()), |_, out| out.last_early = true), true, 0x0FED_CBA9_8765_4321, 1_500)).is_err();
        assert!(caught, "the last byte before the line: not caught");
    }

    #[test]
    fn oam_dma_runs_end_at_mode_2_mode_3_and_every_access() {
        for (cgb, seed) in [(false, 0x1234_5678_9ABC_DEF1), (true, 0x0FED_CBA9_8765_4321)] {
            let (c, modes) = oam_dma_stream(CircuitSpans(build()), cgb, seed, 1_500);
            assert!(modes[2] > 150 && modes[3] > 250 && c.cuts > 15_000 && c.oam_bytes == 1_500 * 160 && c.oam_runs > c.oam_bytes / 4 && c.renders > 1_000, "{c:?} {modes:?}");
        }
    }

}
