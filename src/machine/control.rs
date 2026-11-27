//! The control logic `control_circuit` compiles into weights: the timer (DIV/TIMA/TMA/TAC), the
//! interrupt flags (IF, IE), the joypad register (P1), serial (SB, SC), OAM DMA and
//! HDMA control (which bytes move and when), the CGB registers KEY0, KEY1 and SVBK,
//! the speed switch, the PPU's timing (LY, LYC, STAT and when lines are drawn,
//! `machine::lcd`), and an MBC3 cartridge's real-time clock (`machine::rtc`).
//!
//! CGB double speed (Pan Docs, "CGB Registers", KEY1): STOP with KEY1 bit 0 set toggles
//! the speed (bit 7), clears bit 0, resets DIV (Pan Docs, "Timer and Divider
//! Registers") and pauses the CPU for [`SWITCH_MCYCLES`] while DIV holds. At
//! double speed an M-cycle is 2 dots: the timer, serial and OAM DMA, which count
//! M-cycles, run twice as fast; the PPU's timing ticks every second M-cycle, and an
//! HDMA block stops the CPU for 16 M-cycles instead of 8, the same 8 µs.
//! [`Control`] is the reference; a checkpoint replaces it through [`ControlDevice`].

use super::dma::{Hdma, OamDma};
use super::io::{Buttons, Joypad, Serial};
use super::lcd::{Lcd, LcdState, MODE3_START, STAT_IRQ};
use super::ppu::Lengths;
use super::rtc::Rtc;
use super::timer::{Timer, TimerState};
use crate::device::{Copy, Device, Signals};

/// What the bus and the CPU step ask of the control logic. The machine raises other
/// devices' interrupt bits through [`ControlDevice::raise`] after each block; the
/// control logic's own bits (timer, serial, joypad, VBlank, STAT) never pass through
/// [`Signals`]. The DMA copies it asks for, the M-cycles HDMA stops the CPU for, and
/// the renderer's three entry points (draw a line, show the frame, blank the screen)
/// do.
/// What a BCPD or OCPD read gives: the palette byte at an index, which the PPU holds, or
/// a byte of the control logic's own (0xFF while mode 3 locks the port).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteRead {
    At(u8),
    Byte(u8),
}

pub trait ControlDevice: Send {
    // A call without a `Signals` may still be a pass that moves bytes of a copy (a run of
    // OAM DMA bytes up to its M-cycle): they go to `landed`, which the caller lands at once.
    /// A bus read of one of the control registers at this M-cycle, or of A000-BFFF while
    /// the cartridge selects its clock's chip.
    fn read(&mut self, addr: u16, landed: &mut Vec<Copy>) -> u8;
    /// A bus read of BCPD or OCPD: the palette byte's index (`machine::ppu::Palettes`)
    /// the index register points at, or the byte itself while mode 3 locks palette RAM.
    fn palette(&mut self, addr: u16, landed: &mut Vec<Copy>) -> PaletteRead;
    /// A control register write, or a cartridge write the clock's chip sees
    /// (`Cartridge::clock_hears`).
    fn write(&mut self, addr: u16, value: u8, s: &mut Signals);
    /// The seconds of wall time elapsed since the host's last reading, for the cartridge
    /// clock: a host input, as the buttons are.
    fn elapse(&mut self, seconds: u64, landed: &mut Vec<Copy>);
    /// The cartridge clock's state, for the battery save's footer.
    fn clock(&self) -> Rtc;
    /// The CPU ran STOP. With a speed switch armed it switches, and adds the switch's pause
    /// to the M-cycles the CPU stays stopped (`Lines::owed`).
    fn stop(&mut self, s: &mut Signals);
    /// The CPU made a bus access that OAM DMA blocks. A device that moves the transfer's
    /// bytes in runs lands the run so far.
    fn cut(&mut self, _s: &mut Signals) {}
    /// Advances `mcycles` M-cycles, at most the horizon of [`ControlDevice::lines`], the
    /// GBC CPU stopped through them when `stopped` (they count down `Lines::owed`).
    fn advance(&mut self, mcycles: u64, stopped: bool, s: &mut Signals);
    /// IF bits other devices raised in a block.
    fn raise(&mut self, bits: u8, landed: &mut Vec<Copy>);
    /// IF bits the CPU acknowledged by dispatching.
    fn ack(&mut self, bits: u8, landed: &mut Vec<Copy>);
    /// The buttons held from now on, the host's input as a keyboard's is. A selected line
    /// that falls ends ordinary STOP: whether that released the PPU's VRAM (Some(false)).
    fn buttons(&mut self, pressed: Buttons, landed: &mut Vec<Copy>) -> Option<bool>;
    /// What the bus and the scheduler read between calls. The machine latches it after
    /// every call, so the CPU's IE and IF and the scheduler's horizon are field loads.
    fn lines(&self) -> Lines;
    /// Bytes the game sent over the serial port.
    fn serial_output(&self) -> &[u8];
    /// The GBC CPU's access locks at this M-cycle (`Lines::locks`), a pass if need be.
    fn locks(&mut self, landed: &mut Vec<Copy>) -> u8;
    /// The bus state `mcycles` M-cycles from now (`step::Bus::ahead`): bit 2 while OAM DMA
    /// still blocks, and with `locks` the access locks in bits 0-1, a write's with `write`
    /// (else a read's). The GBC CPU's step asks it for each data access's M-cycle before
    /// the devices run there.
    fn ahead(&mut self, mcycles: u8, locks: bool, write: bool) -> u8;
    /// A GBC CPU step starts (true) or ends (false). An HDMA block the step's M-cycles move
    /// waits for its end (`Signals::held`), and HDMA5 counts an HBlank block there. A run's
    /// bytes a pass for it moves go to `landed`.
    fn step_running(&mut self, running: bool, landed: &mut Vec<Copy>);
    /// The GBC CPU's step left it halted (its `halted` output), before the HALT's M-cycle.
    /// A run's bytes a pass for it moves go to `landed`.
    fn halt(&mut self, landed: &mut Vec<Copy>);
    /// The GBC CPU left HALT, at its wake step, before the step's M-cycles.
    fn woke(&mut self, s: &mut Signals);
    /// The OAM row, by byte, a DMG's PPU reads at this M-cycle of a visible line's mode 2
    /// (SameBoy's `accessed_oam_row`), which the OAM corruption bug rewrites; 0 elsewhere
    /// and on a CGB. A read of the control logic's own port for it, which no bus access
    /// reaches ([`OAM_ROW_ADDR`]).
    fn oam_row(&mut self, landed: &mut Vec<Copy>) -> u8 {
        self.read(OAM_ROW_ADDR, landed)
    }
    /// Plugs one end of a serial link cable in.
    fn plug(&mut self, link: super::io::Link) -> Result<(), String>;
    fn describe(&self) -> Vec<String> {
        Vec::new()
    }
    fn report(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// What the control device shows the rest of the machine at this M-cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lines {
    pub ie: u8,
    /// IF's five bits.
    pub iflag: u8,
    /// M-cycles until the device must run again: 1 while OAM DMA runs, since the PPU
    /// may read OAM on the next M-cycle, and never past a visible line's mode-3 start
    /// or line 144's start, where a block must end for the renderer.
    pub horizon: u64,
    /// An OAM DMA transfer blocks the GBC CPU's accesses on the buses `dma_buses` names.
    pub dma_active: bool,
    /// The buses OAM DMA takes (`layout::Dma` bits): the low nibble those of the transfer
    /// FF46 last started, from its source page and the console; the high nibble those
    /// blocked now, 0 while no transfer blocks.
    pub dma_buses: u8,
    /// SVBK as written; its low 3 bits pick the WRAM bank at D000, 0 selecting bank 1.
    pub svbk: u8,
    /// KEY1 bit 7: CGB double speed.
    pub double: bool,
    /// The 2-dot halves of real time an M-cycle lasts: 2 at normal speed, 1 at double.
    pub rate: u8,
    /// The 2-dot halves a frame of emulated time lasts: 154 lines of 456 dots.
    pub frame: u64,
    /// A speed switch's pause or ordinary STOP runs: the CPU is stopped and the system
    /// counter holds.
    pub paused: bool,
    /// An interrupt is enabled and requested (`IE & IF & 0x1F != 0`) and no longer waits an
    /// M-cycle (`ControlState::waking`): the GBC CPU leaves HALT.
    pub wake: bool,
    /// The M-cycles the GBC CPU stays stopped once the step that runs has ended: HDMA
    /// blocks' and a speed switch's pause, one after the other.
    pub owed: u64,
    /// The GBC CPU's access locks, reads' VRAM and OAM in bits 0 and 1 and writes' in bits
    /// 2 and 3 (`Lcd::locks`), when the device can say without a pass; else
    /// [`ControlDevice::locks`] asks.
    pub locks: Option<u8>,
}

/// The control logic's read port for the OAM row a DMG's OAM scan reads
/// (`ControlDevice::oam_row`). FF0A is unmapped, so no bus access reaches it.
pub const OAM_ROW_ADDR: u16 = 0xFF0A;

/// CGB's palette index registers, BCPS and OCPS: bits 0-5 the byte, bit 7
/// auto-increment. A data port write locked out by mode 3 is dropped, but the index
/// still advances.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaletteIndex(pub [u8; 2]);

impl PaletteIndex {
    /// 0 for BCPS/BCPD (FF68/FF69), 1 for OCPS/OCPD (FF6A/FF6B).
    fn which(addr: u16) -> usize {
        (addr >= 0xFF6A) as usize
    }

    fn read(&self, addr: u16) -> u8 {
        self.0[Self::which(addr)] | 0x40
    }

    fn write(&mut self, addr: u16, value: u8) {
        self.0[Self::which(addr)] = value & 0xBF;
    }

    /// The byte the data port at `addr` reaches, at its index in the palettes.
    fn byte(&self, addr: u16) -> u8 {
        let w = Self::which(addr);
        w as u8 * 64 + (self.0[w] & 0x3F)
    }

    /// A data port write: the byte to write unless locked, then the index advances.
    fn data_write(&mut self, addr: u16, locked: bool) -> Option<u8> {
        let byte = (!locked).then(|| self.byte(addr));
        let i = &mut self.0[Self::which(addr)];
        if *i & 0x80 != 0 {
            *i = 0x80 | (*i + 1) & 0x3F;
        }
        byte
    }
}

/// The carried state, the same fields a checkpoint holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlState {
    pub timer: TimerState,
    pub iflag: u8,
    pub ie: u8,
    /// P1's select bits (4-5) and the buttons held.
    pub joypad: (u8, u8),
    /// SB, SC and the M-cycles left in a transfer.
    pub serial: (u8, u8, u16),
    pub key0: u8,
    /// Bit 7 the speed (double when set), bit 0 a switch armed.
    pub key1: u8,
    pub svbk: u8,
    /// FF46, the next byte's index (160 when no transfer runs, `dma::DMA_START` and the
    /// next for the two M-cycles before byte 0) and whether the start restarted a transfer
    /// that blocked.
    pub dma: (u8, u8, bool),
    /// The HDMA source and destination (VRAM offset), HDMA5 as it reads, whether a block a
    /// step moved waits to be counted (`Hdma::count`), and whether this line's block moved
    /// before its request (`Hdma::line_done`).
    pub hdma: (u16, u16, u8, bool, bool, bool),
    pub lcd: LcdState,
    pub palette: PaletteIndex,
    /// At double speed with the LCD on: one M-cycle of the PPU's current tick has passed.
    pub half: bool,
    /// M-cycles left in a speed switch's pause.
    pub pause: u16,
    /// The cartridge clock, all zeros on a board without one.
    pub clock: Rtc,
    /// A DIV write or a speed switch reset the counter while the frame sequencer's bit
    /// was set, and no M-cycle has passed since: the bit falls at the next one.
    pub dropped: bool,
    /// In ordinary STOP: the GBC CPU, the counter and (on DMG) the PPU's timing wait for
    /// a joypad line to fall.
    pub stopped: bool,
    /// The STOP keeps a CGB's PPU from VRAM, so its lines draw black: it did not come in
    /// mode 3.
    pub blocked: bool,
    /// IE & IF became nonzero in the latest M-cycle and a halted GBC CPU wakes one M-cycle
    /// later. SameBoy's `GB_cpu_run` samples IE & IF at the start of a CGB's halted
    /// M-cycle and halfway through a DMG's, so a DMG wakes in the M-cycle itself unless
    /// only bits that rise in its second half are pending: the timer's, an internal serial
    /// transfer's and STAT's from mode 2 (gbmicrotest `int_timer_halt`, `int_oam_halt`
    /// against `vblank_int_halt_a`, `lyc_int_halt_a`; mooneye `halt_ime1_timing2-GS`,
    /// `di_timing-GS`, `intr_1_2_timing-GS`, `intr_2_*`, `hblank_ly_scx_timing-GS`;
    /// daid `speed_switch_timing_ly` on CGB).
    pub waking: bool,
    /// The GBC CPU halted while an HBlank transfer ran (`ControlDevice::halt`): no block
    /// moves until the wake, and at the wake one moves when the HALT began outside mode 0
    /// (the flag) and STAT reads mode 0 (SameBoy's `allow_hdma_on_wake`; Mealybug
    /// `dma/hdma_during_halt-C`).
    pub halt: Option<bool>,
    /// The console is a CGB, fixed at load.
    pub cgb: bool,
    /// FF72, FF73 and FF74 as written, and FF75's bits 4-6 (Pan Docs, "Undocumented
    /// registers"); none does anything else.
    pub spare: [u8; 4],
}

/// M-cycles of the new speed a speed switch pauses the CPU for, DIV held. Pan Docs
/// ("CGB Registers", KEY1) gives 2050; hardware does not agree. daid's
/// speed_switch_timing tests (GBEmulatorShootout, MIT) read LY 0x85 after a switch to
/// double speed made at line 144, about 16,400 ticks later, and SameBoy's countdown is
/// 0x20008 T-cycles of the new speed (32,770 M-cycles). 32,772 makes the DIV and STAT
/// screenshots pixel-exact and leaves the LY one off by a read at three of its
/// transitions (the PPU's timing here has whole-M-cycle modes).
pub const SWITCH_MCYCLES: u16 = 32772;

#[derive(Clone)]
pub struct Control {
    pub timer: Timer,
    ie: u8,
    iflag: u8,
    joypad: Joypad,
    serial: Serial,
    dma: OamDma,
    hdma: Hdma,
    /// KEY0: what the boot ROM wrote (the header byte for a CGB game), read-only after it.
    key0: u8,
    key1: u8,
    svbk: u8,
    lcd: Lcd,
    palette: PaletteIndex,
    half: bool,
    pause: u16,
    clock: Rtc,
    cgb: bool,
    /// Each drawn line's mode-3 length, which places its mode 0.
    lengths: Lengths,
    /// In ordinary STOP: the GBC CPU and the counter wait for a joypad line to fall.
    stopped: bool,
    /// The STOP keeps a CGB's PPU from VRAM.
    blocked: bool,
    /// A DIV write or a speed switch reset the system counter while the frame
    /// sequencer's bit was set: the bit falls at the next M-cycle.
    dropped: bool,
    /// `ControlState::waking`.
    waking: bool,
    /// `ControlState::halt`.
    halt: Option<bool>,
    /// `ControlState::spare`.
    spare: [u8; 4],
    /// A GBC CPU step runs (`ControlDevice::step_running`).
    step: bool,
    /// `Lines::owed`.
    owed: u64,
}

impl Control {
    /// `Lines::dma_buses`: the buses of the transfer FF46 last started, and those it blocks now.
    fn dma_buses(&self) -> u8 {
        let class = crate::memory::layout::Dma::during(self.dma.state().0, self.cgb).0;
        class | if self.dma.blocking() { class << 4 } else { 0 }
    }

    /// After the boot ROM, without running one (the labelled convenience of
    /// `Machine::new`): the boot ROM's register state as data, KEY0 the header byte the CGB
    /// boot ROM copies into it.
    pub fn post_boot(cgb: bool, key0: u8) -> Self {
        Control {
            timer: match (cgb, key0 & 0x80 != 0) {
                (true, true) => Timer::post_boot_cgb(),
                (true, false) => Timer::post_boot_compat(),
                (false, _) => Timer::post_boot_dmg(),
            },
            ie: 0,
            iflag: 0x01,
            joypad: if cgb { Joypad::post_boot_cgb() } else { Joypad::post_boot_dmg() },
            serial: if cgb { Serial::post_boot_cgb() } else { Serial::post_boot_dmg() },
            key0,
            lcd: Lcd::post_boot().on_console(cgb),
            // BCPS and OCPS as Nintendo's CGB boot ROM leaves them (measured with `--boot`):
            // compatibility mode reads C8 and D0 (mooneye `misc/boot_hwio-C`), CGB mode C0 and 40.
            palette: PaletteIndex(match (cgb, key0 & 0x80 != 0) {
                (true, true) => [0x80, 0x00],
                (true, false) => [0x88, 0x90],
                (false, _) => [0, 0],
            }),
            ..Self::power_on(cgb)
        }
    }

    /// At power-on, for a boot ROM to run: the counter at 0, the LCD off, no interrupt
    /// flag, KEY0 clear until the boot ROM writes it.
    pub fn power_on(cgb: bool) -> Self {
        Control {
            timer: Timer::power_on(),
            ie: 0,
            iflag: 0,
            joypad: Joypad::post_boot_dmg(),
            serial: Serial::post_boot_dmg(),
            dma: OamDma::post_boot(cgb),
            hdma: Hdma::new(),
            key0: 0,
            key1: 0,
            svbk: 0,
            lcd: Lcd::power_on().on_console(cgb),
            palette: PaletteIndex::default(),
            half: false,
            pause: 0,
            clock: Rtc::default(),
            cgb,
            lengths: Lengths::default(),
            stopped: false,
            blocked: false,
            dropped: false,
            step: false,
            owed: 0,
            waking: false,
            halt: None,
            spare: [0; 4],
        }
    }

    /// Reads each line's mode-3 length from `lengths`, the PPU's board.
    pub fn with_lengths(self, lengths: Lengths) -> Self {
        Control { lengths, ..self }
    }

    pub fn lengths(&self) -> Lengths {
        self.lengths.clone()
    }

    /// The cartridge clock as a save's footer left it.
    pub fn with_clock(self, clock: Rtc) -> Self {
        Control { clock, ..self }
    }

    fn double(&self) -> bool {
        self.key1 & 0x80 != 0
    }

    /// One M-cycle of the PPU's timing: whether it completes a tick. A tick starts
    /// when the LCD turns on.
    fn ticks(&mut self) -> bool {
        if self.double() && self.lcd.state().on {
            self.half = !self.half;
            !self.half
        } else {
            true
        }
    }

    /// The system counter's bit whose falling edges step the APU's frame sequencer: bit
    /// 12 (DIV bit 4), or bit 13 (DIV bit 5) at double speed, 512 Hz either way (Pan
    /// Docs, "Audio details").
    fn sequencer_bit(&self) -> bool {
        let bit = if self.double() { 13 } else { 12 };
        self.timer.counter() >> bit & 1 != 0
    }

    /// M-cycles until the M-cycle where that bit next falls: the next one after a reset
    /// that dropped it, none in a pause (the counter holds), else where the count carries
    /// out of it.
    fn edge_horizon(&self) -> u64 {
        if self.dropped {
            return 1;
        }
        if self.pause > 0 || self.stopped {
            return u64::MAX;
        }
        let period = if self.double() { 4096 } else { 2048 };
        period - (self.timer.counter() as u64 >> 2 & (period - 1))
    }

    /// M-cycles until the M-cycle where that bit next rises by counting: none in a pause.
    /// A reset leaves the counter at 0, so the bit rises half a period on.
    fn rise_horizon(&self) -> u64 {
        if self.pause > 0 || self.stopped {
            return u64::MAX;
        }
        let period = if self.double() { 4096 } else { 2048 };
        let x = self.timer.counter() as u64 >> 2 & (period - 1);
        if x < period / 2 { period / 2 - x } else { period + period / 2 - x }
    }

    /// M-cycles until a speed switch's pause ends, where the CPU runs again.
    fn pause_horizon(&self) -> u64 {
        if self.pause > 0 { self.pause as u64 } else { u64::MAX }
    }

    /// M-cycles until the PPU's timing next needs a block to end.
    fn lcd_horizon(&self) -> u64 {
        match self.lcd.horizon() {
            u64::MAX => u64::MAX,
            h if self.double() => 2 * h - self.half as u64,
            h => h,
        }
    }

    pub fn state(&self) -> ControlState {
        let (j, s) = (&self.joypad, &self.serial);
        ControlState {
            timer: self.timer.state(),
            iflag: self.iflag,
            ie: self.ie,
            joypad: (j.select, j.pressed.0),
            serial: (s.sb, s.sc, s.remaining),
            key0: self.key0,
            key1: self.key1,
            svbk: self.svbk,
            dma: self.dma.state(),
            hdma: self.hdma.state(),
            lcd: self.lcd.state(),
            palette: self.palette,
            half: self.half,
            pause: self.pause,
            clock: self.clock,
            dropped: self.dropped,
            stopped: self.stopped,
            blocked: self.blocked,
            cgb: self.cgb,
            waking: self.waking,
            halt: self.halt,
            spare: self.spare,
        }
    }

    /// While a GBC CPU step runs, the HDMA block the copies from `from` on hold waits for
    /// its end.
    fn hold(&self, s: &mut Signals, from: usize) {
        if self.step {
            s.held.extend(s.copies.drain(from..));
        }
    }

    /// `Lcd::locks`.
    fn locks_now(&self) -> u8 {
        self.lcd.locks()
    }

    fn pending(&self) -> bool {
        self.ie & self.iflag & 0x1F != 0
    }

    /// `waking` after a call that found IE & IF `before`. After M-cycles whose last raised
    /// `late` (bits that rise in an M-cycle's second half), an interrupt that became pending
    /// waits on a CGB, and on a DMG when only late bits are pending. After an event with no
    /// M-cycle, one that became pending waits on a CGB, and one waiting keeps waiting.
    fn settle_wake(&mut self, before: bool, ticked: bool, late: u8) {
        let pending = self.ie & self.iflag & 0x1F;
        self.waking = pending != 0
            && match (ticked, before) {
                (true, true) => false,
                (true, false) => self.cgb || pending & !late == 0,
                (false, true) => self.waking,
                (false, false) => self.cgb,
            };
    }

    /// A joypad interrupt: a selected line fell, which ends ordinary STOP.
    fn joypad_edge(&mut self, edge: u8) {
        self.iflag |= edge;
        if edge != 0 {
            self.stopped = false;
            self.blocked = false;
        }
    }

    /// STAT reads mode 0 and the line's HBlank request is still to come (`Lcd::in_gap`).
    pub fn in_gap(&self) -> bool {
        self.lcd.in_gap(self.half)
    }
}

impl ControlDevice for Control {
    /// A block the step moved counts in HDMA5 as the step ends (`Hdma::count`).
    fn step_running(&mut self, running: bool, _: &mut Vec<Copy>) {
        self.step = running;
        if !running {
            self.hdma.count();
        }
    }

    /// Mode 2 reads two objects' 4 bytes an M-cycle: row 8 (m + 1) at M-cycle m, rows below
    /// 8 and from 0xA0 never corrupted (blargg `oam_bug` 4-scanline_timing).
    fn oam_row(&mut self, _: &mut Vec<Copy>) -> u8 {
        let l = self.lcd.state();
        if !self.cgb && l.on && l.ly < 144 && !l.first && l.m < MODE3_START { 8 * (l.m + 1) } else { 0 }
    }

    /// SameBoy's `halt` samples STAT at the HALT: a block may move at the wake only
    /// when it began outside mode 0. Without an HBlank transfer running nothing is kept,
    /// since none can start while the GBC CPU is halted. Between STAT's mode 0 and the
    /// line's HBlank request (`Lcd::in_gap`) the line's block is still owed unless it moved:
    /// SameBoy's request and mode 0 fall together, so it gives no answer for the gap, and
    /// the rule that a line moves one block is inferred.
    fn halt(&mut self, _: &mut Vec<Copy>) {
        if self.hdma.hblank_active() {
            self.halt = Some(if self.lcd.in_gap(self.half) { !self.hdma.line_done } else { self.lcd.stat_mode(self.half) != 0 });
        }
    }

    /// The block moves before the wake step's M-cycles (SameBoy's `GB_cpu_run` sets
    /// `hdma_on` at the wake).
    fn woke(&mut self, s: &mut Signals) {
        if self.halt.take() == Some(true) && self.lcd.stat_mode(self.half) == 0 {
            let from = s.copies.len();
            self.hdma.line_done |= self.hdma.hblank_active() && self.lcd.in_gap(self.half);
            self.hdma.hblank(self.double(), false, s, &mut self.owed);
            self.hold(s, from);
        }
    }

    fn read(&mut self, addr: u16, _: &mut Vec<Copy>) -> u8 {
        match addr {
            0xFF00 => self.joypad.read(),
            0xFF01 | 0xFF02 => self.serial.read(addr),
            0xFF0F => self.iflag | 0xE0,
            0xFF41 | 0xFF44 | 0xFF45 => self.lcd.read(addr, self.half, self.cgb),
            0xFF68 | 0xFF6A => self.palette.read(addr),
            0xFF46 => Device::read(&self.dma, addr),
            0xFF4C => self.key0,
            OAM_ROW_ADDR => self.oam_row(&mut Vec::new()),
            0xFF51..=0xFF55 => self.hdma.read(addr),
            0xFF4D => 0x7E | self.key1,
            0xFF70 => 0xF8 | self.svbk & 7,
            0xFF72..=0xFF74 => self.spare[addr as usize - 0xFF72],
            0xFF75 => 0x8F | self.spare[3],
            0xFFFF => self.ie,
            0xA000..=0xBFFF => self.clock.read(),
            _ => self.timer.read(addr),
        }
    }

    fn palette(&mut self, addr: u16, _: &mut Vec<Copy>) -> PaletteRead {
        if self.lcd.mode() == 3 { PaletteRead::Byte(0xFF) } else { PaletteRead::At(self.palette.byte(addr)) }
    }

    fn write(&mut self, addr: u16, value: u8, s: &mut Signals) {
        let before = self.pending();
        match addr {
            0xFF00 => {
                let edge = self.joypad.write(value);
                self.joypad_edge(edge);
            }
            0xFF01 | 0xFF02 => self.serial.write(addr, value, self.timer.counter()),
            0xFF0F => self.iflag = value & 0x1F,
            0xFF40 | 0xFF41 | 0xFF44 | 0xFF45 => {
                let (irq, off) = self.lcd.write(addr, value);
                self.iflag |= irq;
                s.blank |= off;
                self.half &= !off;
            }
            0xFF68 | 0xFF6A => self.palette.write(addr, value),
            0xFF69 | 0xFF6B => s.own = self.palette.data_write(addr, self.lcd.mode() == 3).map(|i| (i, value)),
            0xFF46 => Device::write(&mut self.dma, addr, value, s),
            0xFF4C => self.key0 = value,
            0xFF04 => {
                self.dropped |= self.sequencer_bit();
                self.timer.write(addr, value);
            }
            0xFF51..=0xFF55 => {
                let from = s.copies.len();
                let (mode0, gap) = (self.lcd.stat_mode(self.half) == 0, self.lcd.in_gap(self.half));
                self.hdma.write(addr, value, self.double(), mode0, s, &mut self.owed);
                // An HBlank transfer started in mode 0 moved its first block: in the gap, that is
                // this line's.
                self.hdma.line_done |= addr == 0xFF55 && value & 0x80 != 0 && gap;
                self.hold(s, from);
            }
            0xFF4D => self.key1 = self.key1 & 0x80 | value & 1,
            0xFF70 => self.svbk = value,
            0xFF72..=0xFF74 => self.spare[addr as usize - 0xFF72] = value,
            0xFF75 => self.spare[3] = value & 0x70,
            0xFFFF => self.ie = value,
            0x0000..=0xBFFF => self.clock.write(addr, value),
            _ => self.timer.write(addr, value),
        }
        self.settle_wake(before, false, 0);
    }

    fn elapse(&mut self, seconds: u64, _: &mut Vec<Copy>) {
        self.clock.elapse(seconds);
    }

    fn clock(&self) -> Rtc {
        self.clock
    }

    fn stop(&mut self, s: &mut Signals) {
        if self.key1 & 1 == 0 {
            // Ordinary STOP (Pan Docs, "Power Modes"): DIV resets and the GBC CPU waits for a
            // selected joypad line to fall. A DMG's PPU stops, its screen blank; a CGB's
            // runs on but cannot read VRAM unless it stopped in mode 3 (daid's stop_instr).
            self.stopped = true;
            self.dropped |= self.sequencer_bit();
            self.timer.write(0xFF04, 0);
            if self.cgb {
                let was = self.blocked;
                self.blocked = self.lcd.mode() != 3;
                if self.blocked != was {
                    s.stopped = Some(self.blocked);
                }
            } else {
                s.blank = self.lcd.state().on;
            }
            return;
        }
        self.dropped |= self.sequencer_bit();
        self.key1 = (self.key1 ^ 0x80) & 0x80;
        // The mode's edges move with the speed at once, and the STAT line with them.
        self.iflag |= self.lcd.set_double(self.double());
        self.timer.write(0xFF04, 0);
        self.half = false;
        self.pause = SWITCH_MCYCLES;
        self.owed += self.pause as u64;
    }

    /// The renderer's entry points land on the block's last M-cycle, which the horizon
    /// guarantees. In a speed switch's pause only the PPU's timing runs: the HBlanks it
    /// passes move no HDMA block, so an HBlank transfer resumes at the first HBlank after
    /// the pause.
    fn advance(&mut self, mcycles: u64, stopped: bool, s: &mut Signals) {
        let before = self.pending();
        if stopped {
            assert!(mcycles <= self.owed, "{mcycles} M-cycles stopped past the {} owed", self.owed);
            self.owed -= mcycles;
        }
        let edge = self.edge_horizon();
        assert!(mcycles <= edge, "control advanced past the frame sequencer's edge");
        s.edge = mcycles == edge;
        s.reset = s.edge && self.dropped;
        s.rise = mcycles == self.rise_horizon();
        self.dropped &= mcycles == 0;
        let paused = self.pause > 0 || self.stopped;
        let running = if self.stopped {
            0
        } else if paused {
            assert!(mcycles <= self.pause as u64, "control advanced past the speed switch's pause");
            self.pause -= mcycles as u16;
            0
        } else {
            mcycles
        };
        let counted = self.timer.run(running) | self.serial.run(running);
        self.iflag |= counted;
        let mut late = counted;
        let mut hblank = false;
        for i in 0..mcycles {
            let ticked = self.ticks();
            if self.stopped && !self.cgb {
                continue;
            }
            if ticked {
                if self.lcd.wants_length() {
                    self.lcd.lengthen_mode3(self.lengths.latest());
                }
                let t = self.lcd.tick();
                self.iflag |= t.irq;
                if t.late {
                    late |= STAT_IRQ;
                }
                if t.render || t.frame {
                    assert_eq!(i + 1, mcycles, "control advanced past the M-cycle line {} needed the renderer", self.lcd.ly());
                    s.render = t.render.then(|| self.lcd.ly());
                    s.frame = t.frame;
                }
            }
            // The HBlank HDMA request reaches the opcode fetch of the M-cycle after this one:
            // the block moves after the step that fetches, a block held in a step at its end.
            hblank |= self.lcd.requests_at(self.lcd.position(self.half));
        }
        self.dma.advance(running, s);
        // A line whose block moved before its request moves no second one at the request.
        if hblank && !paused && self.halt.is_none() && !self.hdma.line_done {
            let from = s.copies.len();
            self.hdma.hblank(self.double(), self.step, s, &mut self.owed);
            self.hold(s, from);
        }
        self.hdma.line_done &= !hblank;
        self.settle_wake(before, mcycles > 0, late);
    }

    fn raise(&mut self, bits: u8, _: &mut Vec<Copy>) {
        let before = self.pending();
        self.iflag |= bits;
        self.settle_wake(before, false, 0);
    }

    fn ack(&mut self, bits: u8, _: &mut Vec<Copy>) {
        let before = self.pending();
        self.iflag &= !bits;
        self.settle_wake(before, false, 0);
    }

    fn buttons(&mut self, pressed: Buttons, _: &mut Vec<Copy>) -> Option<bool> {
        let before = self.pending();
        let was = self.blocked;
        let edge = self.joypad.set(pressed);
        self.joypad_edge(edge);
        self.settle_wake(before, false, 0);
        (self.blocked != was).then_some(self.blocked)
    }

    fn lines(&self) -> Lines {
        Lines {
            ie: self.ie,
            iflag: self.iflag,
            horizon: self.dma.horizon().min(self.lcd_horizon()).min(self.pause_horizon()).min(self.edge_horizon()).min(self.rise_horizon()).min(if self.waking { 1 } else { u64::MAX }),
            dma_active: self.dma.blocking(),
            dma_buses: self.dma_buses(),
            svbk: self.svbk,
            double: self.double(),
            rate: if self.double() { 1 } else { 2 },
            frame: crate::machine::HALVES_PER_FRAME,
            paused: self.pause > 0 || self.stopped,
            wake: self.pending() && !self.waking,
            owed: self.owed,
            locks: Some(self.locks_now()),
        }
    }

    fn serial_output(&self) -> &[u8] {
        &self.serial.output
    }

    fn locks(&mut self, _: &mut Vec<Copy>) -> u8 {
        self.locks_now()
    }

    /// Whether OAM DMA blocks `mcycles` M-cycles from now, and the PPU's mode on a copy of its timing,
    /// ticked as `advance` ticks it (a tick every second M-cycle at double speed).
    fn ahead(&mut self, mcycles: u8, locks: bool, write: bool) -> u8 {
        let dma = self.dma.blocking_in(mcycles as u64);
        let mut l = 0;
        if locks {
            let (mut lcd, mut half) = (self.lcd.clone(), self.half);
            for _ in 0..mcycles {
                if self.double() && lcd.state().on {
                    half = !half;
                    if half {
                        continue;
                    }
                }
                if self.stopped && !self.cgb {
                    continue;
                }
                if lcd.wants_length() {
                    lcd.lengthen_mode3(self.lengths.latest());
                }
                lcd.tick();
            }
            let shift = if write { crate::step::WRITE_SHIFT } else { 0 };
            l = (lcd.locks() >> shift) & 3;
        }
        l | (dma as u8) << 2
    }

    fn plug(&mut self, link: super::io::Link) -> Result<(), String> {
        self.serial.link = Some(link);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Rng;

    #[derive(Clone, Copy)]
    enum Ev {
        DivWrite,
        Switch,
        Advance(u64),
    }

    /// Runs `script` on the reference and returns, per APU tick, whether a frame-sequencer
    /// step precedes it twice: as the control logic signals it (`Signals::edge`, placed on
    /// the tick that holds its M-cycle, as `Apu::mcycle` places it) and as the APU used
    /// to find it, from the counter bit it saw at each
    /// tick's end. Each advance runs in blocks cut at the horizon, as the machine cuts
    /// them, with the APU's tick phase kept as the machine kept it.
    fn steps_both_ways(script: &[Ev]) -> Vec<(bool, bool)> {
        let mut c = Control::post_boot(true, 0x80).with_lengths(Lengths::constant(44));
        let (mut phase, mut pending, mut seen) = (false, false, false);
        let mut ticks = Vec::new();
        for &ev in script {
            match ev {
                Ev::DivWrite => c.write(0xFF04, 0, &mut Signals::default()),
                Ev::Switch => {
                    c.write(0xFF4D, 1, &mut Signals::default());
                    c.stop(&mut Signals::default());
                    phase = false;
                }
                Ev::Advance(mut left) => {
                    while left > 0 {
                        let (double, m) = (c.double(), left.min(c.lines().horizon));
                        let (counter, held) = (c.timer.counter(), c.pause > 0);
                        let mut s = Signals { fast: double, ..Signals::default() };
                        c.advance(m, false, &mut s);
                        for j in 1..=m {
                            pending |= s.edge && j == m;
                            phase ^= double;
                            if phase {
                                continue;
                            }
                            let b = if double { 13 } else { 12 };
                            let bit = if held { counter } else { counter.wrapping_add(4 * j as u16) } >> b & 1 != 0;
                            ticks.push((std::mem::take(&mut pending), seen && !bit));
                            seen = bit;
                        }
                        left -= m;
                    }
                }
            }
        }
        ticks
    }

    /// Random advances, DIV writes and speed switches, the writes and switches only
    /// where a tick of the APU begins.
    fn random_script(seed: u64, events: usize) -> Vec<Ev> {
        let mut rng = Rng(seed);
        let mut odd = false;
        let mut double = false;
        (0..events)
            .map(|_| {
                let r = rng.draw() % 400;
                if r == 0 && !odd {
                    Ev::DivWrite
                } else if r == 1 && !odd {
                    double = !double;
                    odd = false;
                    Ev::Switch
                } else {
                    let m = 1 + rng.draw() % if rng.draw() % 4 == 0 { 3000 } else { 40 };
                    odd ^= double && m % 2 == 1;
                    Ev::Advance(m)
                }
            })
            .collect()
    }

    #[test]
    fn signalled_steps_are_the_falls_the_apu_saw() {
        let ticks = steps_both_ways(&random_script(0x9E37_79B9_7F4A_7C15, 60_000));
        let steps = ticks.iter().filter(|t| t.0).count();
        assert!(ticks.len() > 1_000_000 && steps > 300, "{} ticks, {steps} steps", ticks.len());
        assert!(ticks.iter().all(|(got, want)| got == want), "a signalled step differs from the counter bit's fall");
    }

    /// The one place the two differ: at double speed a DIV write one M-cycle into a tick,
    /// just after the count set the bit. The bit rose and fell inside the tick, so the
    /// APU, which saw only tick ends, missed the fall; the control logic signals it.
    #[test]
    fn a_reset_inside_the_tick_the_bit_rose_in_steps_once() {
        // Double speed; one M-cycle into a tick, DIV restarts from 0, so bit 13 rises 2,048
        // M-cycles later at a tick's first M-cycle, and a DIV write follows it at once.
        let script = [Ev::Switch, Ev::Advance(crate::machine::control::SWITCH_MCYCLES as u64), Ev::Advance(1), Ev::DivWrite, Ev::Advance(2048), Ev::DivWrite, Ev::Advance(10)];
        let ticks = steps_both_ways(&script);
        let differ: Vec<usize> = (0..ticks.len()).filter(|&i| ticks[i].0 != ticks[i].1).collect();
        assert_eq!(differ.len(), 1, "{differ:?}");
        assert_eq!(ticks[differ[0]], (true, false), "the control logic steps where the APU saw no fall");
    }
}
