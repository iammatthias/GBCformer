//! The PPU's timing: 114 M-cycles per line, 154 lines, mode 3 from M-cycle 20 for the
//! length the renderer gives each line (63 at the least), LY, LYC and the STAT interrupt line. It decides when the renderer draws a
//! line (mode 3's start) and shows a frame (line 144), and when an HBlank HDMA block
//! moves (mode 0's start). Part of the control logic (`machine::control`); the renderer
//! in `machine::ppu` keeps no clock.

pub const VBLANK_IRQ: u8 = 1 << 0;
pub const STAT_IRQ: u8 = 1 << 1;

pub const MCYCLES_PER_LINE: u8 = 114;
pub const MODE3_START: u8 = 20;
pub const MODE0_START: u8 = 63;
/// The line's length is not known yet.
pub const PENDING: u16 = 0;
/// The base of a line's length as the PPU publishes it ([`Lcd::lengthen_mode3`]): the dots
/// from mode 3's start to mode 0's less 128, so the least line, 172 dots, is 44, every
/// length fits a byte and none is [`PENDING`].
pub const LENGTH_BASE: u16 = 128;
/// The most dots to mode 0 a line can publish: mode 0 still falls inside the line.
pub const MAX_DOTS: u16 = 370;
pub const VISIBLE_LINES: u8 = 144;
pub const LINES: u8 = 154;

/// Everything the timing holds, for a checkpoint held to it in lockstep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LcdState {
    /// LCDC bit 7.
    pub on: bool,
    pub ly: u8,
    /// The M-cycle within the line, 0-113: the dot after the latest M-cycle, over 4.
    pub m: u8,
    pub lyc: u8,
    /// STAT bits 3-6, the enabled interrupt sources.
    pub enables: u8,
    /// The OR of the enabled sources; IF bit 1 rises on its rising edge.
    pub line: bool,
    /// STAT's LY=LYC bit. The comparison runs only while the LCD is on; off, the bit keeps
    /// the value it had and alone can hold the line (mooneye `ppu/stat_lyc_onoff`).
    pub coincidence: bool,
    /// Line 0 after the LCD turned on: until mode 3 it reads mode 0, raises no mode
    /// interrupt and leaves OAM open (SameBoy's `GB_display_run` LCD-on path; mooneye
    /// `ppu/stat_lyc_onoff`, gbmicrotest `lcdon_to_stat*`).
    pub first: bool,
}

/// What one M-cycle produced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tick {
    pub irq: u8,
    /// A visible line entered mode 3: draw it now.
    pub render: bool,
    /// Line 144 began: show the frame.
    pub frame: bool,
    /// The mode-2 STAT source, whose rise falls in the M-cycle's second half on a DMG, so
    /// a halted GBC CPU sees it an M-cycle later (`ControlState::waking`): at line 0's
    /// start and in the last M-cycle of lines 0-142 (gbmicrotest `int_oam_halt`, mooneye
    /// `intr_2_*`).
    pub late: bool,
}

#[derive(Clone)]
/// The timing; the dots from the current line's mode-3 start to its mode 0, [`PENDING`] until
/// the line's length is known ([`Lcd::lengthen_mode3`]), which it must be by M-cycle 63
/// ([`Lcd::wants_length`]); whether the console is a CGB; whether the GBC CPU runs at double
/// speed. LY reads the next line from a line's last M-cycle, dot 452 (daid's
/// speed_switch_timing_ly at double speed, blargg oam_bug 1-lcd_sync).
///
/// Positions within a line count 2-dot halves: `p = 2m` at a tick's start, one more after
/// the first M-cycle of a double-speed tick. Mode 0 shows from the half `p` with
/// `2p ≥ D + 77` at normal speed and `2p ≥ D + 82` at double speed, D the dots to mode 0,
/// two dots later on a CGB's line 0 after the LCD turned on. On a DMG D rounds SCX's fine
/// scroll up to 4 (`fifo::mode0_dots`: mooneye's sprite table and `hblank_ly_scx_timing-GS`
/// count whole M-cycles); a CGB's is the dot, so SCX 1 to 3 end mode 3 in the M-cycle SCX 0
/// does (Mealybug `dma/hdma_timing-C`; SameBoy v1.0.5 `display.c`, measured by sweeping a
/// STAT read over every M-cycle after an LCD-on and after a speed switch at SCX 0-2).
pub struct Lcd(LcdState, u16, bool, bool);

impl Lcd {
    /// LCDC 0x91 after either boot ROM: on, at the start of line 0.
    pub fn post_boot() -> Self {
        Lcd(LcdState { on: true, ly: 0, m: 0, lyc: 0, enables: 0, line: false, coincidence: true, first: false }, PENDING, false, false)
    }

    /// At power-on: off, until the boot ROM turns it on.
    pub fn power_on() -> Self {
        Lcd(LcdState { on: false, coincidence: false, ..Self::post_boot().0 }, PENDING, false, false)
    }

    /// The timing of a CGB (`cgb`) or a DMG.
    pub fn on_console(self, cgb: bool) -> Self {
        Lcd(self.0, self.1, cgb, self.3)
    }

    /// The GBC CPU's speed, which places the mode-0 edges and an LCD-on's start. Returns
    /// the interrupt bits the STAT line's rise at the new edges raised.
    pub fn set_double(&mut self, double: bool) -> u8 {
        self.3 = double;
        self.update_line()
    }

    /// The renderer drew this line's mode 3 with `length` the dots to mode 0 less
    /// [`LENGTH_BASE`] (172 dots at the least).
    pub fn lengthen_mode3(&mut self, length: u8) {
        let dots = length as u16 + LENGTH_BASE;
        assert!((172..=MAX_DOTS).contains(&dots), "a mode 3 of {dots} dots");
        self.1 = dots;
    }

    /// Dots a CGB's line 0 after the LCD turned on ends mode 3 later than other lines.
    fn late(&self) -> u16 {
        if self.0.first && self.2 { 2 } else { 0 }
    }

    /// Whether half `p` of the current visible line sees mode 0. With the length not known
    /// yet the line has not reached the least edge.
    fn sees_mode0(&self, p: u16) -> bool {
        if self.1 == PENDING {
            assert!(p / 2 < MODE0_START as u16, "line {} at half {p}: mode 3's length is not known", self.0.ly);
            return false;
        }
        2 * p >= self.1 + 77 + if self.3 { 5 } else { 0 } + self.late()
    }

    /// The half `half` (an M-cycle of a double-speed tick has passed) puts a read at.
    pub fn position(&self, half: bool) -> u16 {
        2 * self.0.m as u16 + (self.3 && half) as u16
    }

    /// The half the HBlank HDMA request reaches the GBC CPU's opcode fetches at: two dots
    /// after STAT's mode 0 at normal speed and one at double speed, `2p ≥ D + 79` or
    /// `D + 83` (Mealybug `dma/hdma_timing-C`; SameBoy v1.0.5's `hdma_on` comes a dot
    /// later and fails its SCX 1 rows), at a tick's start at normal speed.
    fn request_at(&self) -> Option<u16> {
        let s = &self.0;
        if !s.on || s.ly >= VISIBLE_LINES || self.1 == PENDING {
            return None;
        }
        let half = (self.1 + 79 + if self.3 { 4 } else { 0 } + self.late()).div_ceil(2);
        Some(if self.3 { half } else { half.div_ceil(2) * 2 })
    }

    /// STAT reads mode 0 after `half` on a drawn line whose HBlank request is still to come:
    /// the M-cycle (one half at double speed) the two edges fall apart at some fine scrolls.
    pub fn in_gap(&self, half: bool) -> bool {
        let s = &self.0;
        s.m >= MODE3_START && self.stat_mode(half) == 0 && self.request_at().is_some_and(|r| self.position(half) < r)
    }

    /// The HBlank HDMA request reaches the GBC CPU at half `p` of the current line.
    pub fn requests_at(&self, p: u16) -> bool {
        self.request_at() == Some(p)
    }

    /// The half a tick's mode is taken at: at double speed its second M-cycle's.
    fn tick_position(&self, m: u8) -> u16 {
        2 * m as u16 + self.3 as u16
    }

    /// The next tick reaches the least mode-0 start of a line whose length is not known.
    pub fn wants_length(&self) -> bool {
        self.1 == PENDING && self.0.ly < VISIBLE_LINES && self.0.m + 1 == MODE0_START
    }

    pub fn state(&self) -> LcdState {
        self.0
    }

    pub fn ly(&self) -> u8 {
        self.0.ly
    }

    /// The mode a tick holds, for the locks, the STAT interrupt, the palette ports and the
    /// HBlank HDMA block: at double speed as its second M-cycle sees it.
    pub fn mode(&self) -> u8 {
        self.mode_at(self.tick_position(self.0.m))
    }

    /// The mode STAT reads after `half` (an M-cycle of a double-speed tick has passed).
    pub fn stat_mode(&self, half: bool) -> u8 {
        self.mode_at(self.position(half))
    }

    /// The mode at half `p` of the line.
    fn mode_at(&self, p: u16) -> u8 {
        let s = &self.0;
        match (s.on, s.ly, s.m) {
            (false, _, _) => 0,
            (true, VISIBLE_LINES.., _) => 1,
            (true, _, ..MODE3_START) if s.first => 0,
            (true, _, ..MODE3_START) => 2,
            _ if !self.sees_mode0(p) => 3,
            _ => 0,
        }
    }

    /// The GBC CPU's access locks at the current M-cycle: reads' VRAM and OAM in bits 0 and
    /// 1, writes' in bits 2 and 3 (`step::WRITE_SHIFT`). VRAM is locked in mode 3 and OAM in
    /// modes 2 and 3. A DMG's reads and writes differ by an M-cycle at two edges (SameBoy
    /// v1.0.5 `display.c`: `oam_read_blocked` rises 3 dots into a line, before STAT's mode
    /// 2; at OAM scan index 37 `vram_read_blocked` rises and `oam_write_blocked` falls on a
    /// DMG; mooneye `ppu/lcdon_timing-GS` and `lcdon_write_timing-GS`): an OAM read is
    /// locked in the last M-cycle of a line a visible line follows (lines 0-142, and 153
    /// before line 0), and in mode 2's last M-cycle a VRAM read is locked while an OAM write
    /// lands. A CGB's reads and writes see the same locks.
    pub fn locks(&self) -> u8 {
        let s = &self.0;
        if !s.on {
            return 0;
        }
        let mode = self.mode();
        let base = match mode {
            3 => 3,
            2 => 2,
            _ => 0,
        };
        let (mut read, mut write) = (base, base);
        if !self.2 {
            if mode == 2 && s.m == MODE3_START - 1 {
                read |= 1;
                write = 0;
            }
            if s.m == MCYCLES_PER_LINE - 1 && (s.ly < VISIBLE_LINES - 1 || s.ly == LINES - 1) {
                read |= 2;
            }
        }
        read | write << 2
    }

    /// The enabled sources at the current M-cycle.
    fn source(&self) -> bool {
        let s = &self.0;
        // The mode-2 source of lines 1-143 rises in the last M-cycle of the line before, and
        // on a CGB so does line 144's (mooneye `misc/ppu/vblank_stat_intr-C`: its STAT
        // interrupt comes one M-cycle before VBlank's; `vblank_stat_intr-GS` has them
        // together on DMG).
        let last = if self.2 { VISIBLE_LINES } else { VISIBLE_LINES - 1 };
        let mode = if s.m == MCYCLES_PER_LINE - 1 && s.ly < last { 2 } else { self.mode() };
        let opening = s.first && s.m < MODE3_START;
        (s.coincidence && s.enables & 0x40 != 0)
            || s.on && !opening && ((mode == 0 && s.enables & 0x08 != 0)
            || (mode == 1 && s.enables & 0x10 != 0)
            || (mode == 2 && s.enables & 0x20 != 0)
            // Line 144 also raises the mode-2 source as VBlank begins.
            || (s.ly == VISIBLE_LINES && s.m == 0 && s.enables & 0x20 != 0))
    }

    /// The line the LY=LYC comparison sees at this M-cycle: none in a line's last M-cycle,
    /// and on line 153 153 in its first M-cycle, none in its second and 0 from its third to
    /// its last (aappleby's gbmicrotest `line_153_lyc0_stat_timing_*`,
    /// `line_153_lyc153_stat_timing_*`).
    fn compared(&self) -> Option<u8> {
        let s = &self.0;
        match (s.ly, s.m) {
            (153, 0) => Some(153),
            (153, 1) => None,
            (153, _) => Some(0),
            (_, m) if m == MCYCLES_PER_LINE - 1 => None,
            (ly, _) => Some(ly),
        }
    }

    fn update_line(&mut self) -> u8 {
        if self.0.on {
            self.0.coincidence = self.compared() == Some(self.0.lyc);
        }
        let line = self.source();
        let rising = line && !self.0.line;
        self.0.line = line;
        if rising { STAT_IRQ } else { 0 }
    }

    /// Advances one M-cycle.
    pub fn tick(&mut self) -> Tick {
        let s = &mut self.0;
        if !s.on {
            return Tick::default();
        }
        let mut t = Tick::default();
        s.m += 1;
        if s.m == MCYCLES_PER_LINE {
            s.m = 0;
            s.first = false;
            s.ly = (s.ly + 1) % LINES;
            self.1 = PENDING;
            if s.ly == VISIBLE_LINES {
                t.irq |= VBLANK_IRQ;
                t.frame = true;
            }
        }
        let visible = s.ly < VISIBLE_LINES;
        t.render = visible && s.m == MODE3_START;
        t.late = s.enables & 0x20 != 0 && (s.m == MCYCLES_PER_LINE - 1 && s.ly < VISIBLE_LINES - 1 || s.m == 0 && s.ly == 0);
        t.irq |= self.update_line();
        t
    }

    /// M-cycles until the next one the machine must see at a block's end: a visible
    /// line's mode-3 start or line 144's start.
    pub fn horizon(&self) -> u64 {
        let s = &self.0;
        if !s.on {
            return u64::MAX;
        }
        let (ly, m) = (s.ly as u64, s.m as u64);
        let line = MCYCLES_PER_LINE as u64;
        match ly {
            _ if ly < VISIBLE_LINES as u64 && m < MODE3_START as u64 => MODE3_START as u64 - m,
            143 => line - m,
            _ if ly < VISIBLE_LINES as u64 => line - m + MODE3_START as u64,
            _ => (LINES as u64 - ly) * line - m + MODE3_START as u64,
        }
    }

    /// A read after `half` (an M-cycle of a double-speed tick has passed). A read in the
    /// first half of a double-speed tick sees modes 2 and 3 and LY's early increment one
    /// half later than a tick's start: daid's speed_switch_timing_stat and _ly with each
    /// read at its M-cycle. On a CGB's line 0 after the LCD turned on, mode 3 shows one half
    /// later still at double speed (SameBoy v1.0.5, measured as [`Lcd`] says).
    pub fn read(&self, addr: u16, half: bool, cgb: bool) -> u8 {
        let s = &self.0;
        let first_half = self.3 && s.on && !half;
        let early_ly = s.on && s.m == MCYCLES_PER_LINE - 1 && !first_half;
        // Line 153's last M-cycle reads mode 0 (gbmicrotest `line_153_lyc0_stat_timing_f`);
        // its interrupt source stays mode 1's.
        let mode = match self.stat_mode(half) {
            1 if s.ly == LINES - 1 && s.m == MCYCLES_PER_LINE - 1 => 0,
            2 if first_half && s.m == 0 => if s.ly == 0 { 1 } else { 0 },
            3 if self.3 && s.first && s.m == MODE3_START => 0,
            3 if first_half && s.m == MODE3_START => 2,
            mode => mode,
        };
        match addr {
            0xFF41 => 0x80 | s.enables | (s.coincidence as u8) << 2 | mode,
            0xFF44 if early_ly => (s.ly + 1) % LINES,
            // Line 153 reads 0 from its first M-cycle on DMG, its second on CGB
            // (gbmicrotest `line_153_ly_*`).
            0xFF44 if s.ly == LINES - 1 && s.m >= cgb as u8 => 0,
            0xFF44 => s.ly,
            0xFF45 => s.lyc,
            _ => 0xFF,
        }
    }

    /// Writes FF40 (only bit 7 is the timing's), FF41 or FF45. Returns the interrupt
    /// bits the write raised and whether it turned the LCD off.
    pub fn write(&mut self, addr: u16, value: u8) -> (u8, bool) {
        let s = &mut self.0;
        let mut off = false;
        match addr {
            0xFF40 => {
                let on = value & 0x80 != 0;
                if s.on && !on {
                    s.ly = 0;
                    s.m = 0;
                    s.first = false;
                    self.1 = PENDING;
                    off = true;
                }
                // At double speed line 0 starts a tick on: every edge of the lines after it
                // comes 4 dots earlier than at a tick's start (SameBoy v1.0.5, measured as
                // [`Lcd`] says; Mealybug `dma/hdma_timing-C` reads line 1's end).
                if !s.on && on && self.3 {
                    s.m = 1;
                }
                s.first |= !s.on && on;
                s.on = on;
            }
            0xFF41 => s.enables = value & 0x78,
            0xFF45 => s.lyc = value,
            _ => {}
        }
        (self.update_line(), off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One M-cycle, a line's mode 3 lasting the least, 172 dots, unless the test said.
    fn tick(lcd: &mut Lcd) -> Tick {
        if lcd.wants_length() {
            lcd.lengthen_mode3(44);
        }
        lcd.tick()
    }

    fn run(lcd: &mut Lcd, mcycles: usize) -> Tick {
        (0..mcycles).fold(Tick::default(), |a, _| {
            let t = tick(lcd);
            Tick { irq: a.irq | t.irq, render: a.render | t.render, frame: a.frame | t.frame, late: a.late | t.late }
        })
    }

    #[test]
    fn line_and_frame_timing() {
        let mut lcd = Lcd::post_boot();
        assert_eq!((lcd.read(0xFF41, false, false) & 3, lcd.horizon()), (2, 20));
        run(&mut lcd, 19);
        assert!(tick(&mut lcd).render, "mode 3 at M-cycle 20");
        assert_eq!((lcd.read(0xFF41, false, false) & 3, lcd.horizon()), (3, 114));
        run(&mut lcd, 42);
        assert_eq!(lcd.read(0xFF41, false, false) & 3, 3);
        tick(&mut lcd);
        assert!(lcd.requests_at(lcd.position(false)), "the HBlank HDMA request at M-cycle 63");
        assert_eq!(lcd.read(0xFF41, false, false) & 3, 0);
        run(&mut lcd, 51);
        assert_eq!(lcd.read(0xFF44, false, false), 1, "114 M-cycles per line");
        let t = run(&mut lcd, 114 * 143);
        assert!(t.frame && t.irq & VBLANK_IRQ != 0);
        assert_eq!((lcd.read(0xFF44, false, false), lcd.read(0xFF41, false, false) & 3, lcd.horizon()), (144, 1, 10 * 114 + 20));
        assert_eq!(run(&mut lcd, 114 * 10 - 1), Tick::default(), "no render, HBlank or VBlank during VBlank");
        assert_eq!(tick(&mut lcd), Tick::default());
        assert_eq!(lcd.read(0xFF44, false, false), 0, "154 lines per frame");
    }

    /// A line's locks M-cycle by M-cycle, reads' then writes' (`Lcd::locks`): on a DMG an OAM
    /// read is locked in the line before's last M-cycle and a VRAM read in mode 2's last,
    /// where an OAM write lands (mooneye `ppu/lcdon_timing-GS`, `lcdon_write_timing-GS`);
    /// on a CGB both see the modes' locks.
    #[test]
    fn reads_and_writes_lock_by_the_mode_and_a_dmg_s_edges() {
        for cgb in [false, true] {
            let mut lcd = Lcd::post_boot().on_console(cgb);
            run(&mut lcd, 114);
            let seen: Vec<(u8, u8)> = (0..114).map(|_| { let l = (lcd.state().m, lcd.locks()); tick(&mut lcd); l }).collect();
            let at = |m: u8| seen[m as usize].1;
            let (mode2, mode3) = (2 | 2 << 2, 3 | 3 << 2);
            assert_eq!(at(0), mode2, "cgb {cgb}: mode 2 locks OAM");
            assert_eq!(at(19), if cgb { mode2 } else { 3 }, "cgb {cgb}: mode 2's last M-cycle");
            assert_eq!((at(20), at(62)), (mode3, mode3), "cgb {cgb}: mode 3");
            assert_eq!((at(63), at(112)), (0, 0), "cgb {cgb}: mode 0");
            assert_eq!(at(113), if cgb { 0 } else { 2 }, "cgb {cgb}: the line's last M-cycle");
            run(&mut lcd, 114 * 141);
            assert_eq!((lcd.state().ly, lcd.state().m), (143, 0));
            run(&mut lcd, 113);
            assert_eq!(lcd.locks(), 0, "cgb {cgb}: no OAM read lock before VBlank");
            run(&mut lcd, 114 * 10);
            assert_eq!((lcd.state().ly, lcd.state().m, lcd.locks()), (153, 113, if cgb { 0 } else { 2 }), "cgb {cgb}: line 153 before line 0");
        }
    }

    #[test]
    fn lyc_interrupt_fires_once_per_match() {
        let mut lcd = Lcd::post_boot();
        lcd.write(0xFF45, 3);
        lcd.write(0xFF41, 0x40);
        assert_eq!(run(&mut lcd, 114 * 3 - 1).irq & STAT_IRQ, 0);
        assert_eq!(tick(&mut lcd).irq & STAT_IRQ, STAT_IRQ, "LY becomes 3");
        assert_eq!(lcd.read(0xFF41, false, false) & 0x04, 0x04);
        assert_eq!(run(&mut lcd, 112).irq & STAT_IRQ, 0, "level stays high to the line's last M-cycle, no refire");
        assert_eq!(lcd.write(0xFF45, 4).0, 0);
        assert_eq!(lcd.write(0xFF45, 3).0, STAT_IRQ, "a write that makes LY match raises it");
    }

    #[test]
    fn a_longer_mode_3_moves_hblank() {
        let mut lcd = Lcd::post_boot();
        run(&mut lcd, 20);
        lcd.lengthen_mode3((180 - LENGTH_BASE) as u8);
        run(&mut lcd, 44);
        assert_eq!(lcd.read(0xFF41, false, false) & 3, 3, "180 dots end in M-cycle 65");
        assert!(!lcd.requests_at(lcd.position(false)));
        tick(&mut lcd);
        assert_eq!(lcd.read(0xFF41, false, false) & 3, 0);
        assert!(lcd.requests_at(lcd.position(false)), "the request in the same M-cycle on a DMG");
        run(&mut lcd, 114 - 65);
        assert_eq!((lcd.read(0xFF44, false, false), lcd.mode()), (1, 2), "the next line starts at the fixed length again");
        run(&mut lcd, 63 - 0);
        assert_eq!(lcd.mode(), 0);
    }

    /// STAT's mode 0 on a CGB at SCX 0-7 with nothing else on the line (172 + f dots), as
    /// the half of the line it first shows at: SameBoy v1.0.5 measured at SCX 0-7, and
    /// Mealybug `dma/hdma_timing-C` at SCX 1 and 2. A DMG rounds the fine scroll up first.
    #[test]
    fn a_cgb_ends_mode_3_at_the_fine_scroll_s_dot() {
        let first = |cgb: bool, double: bool, f: u16| {
            let mut lcd = Lcd::post_boot().on_console(cgb);
            lcd.set_double(double);
            run(&mut lcd, 114 + 20);
            let dots = if cgb { 172 + f } else { 172 + f.div_ceil(4) * 4 };
            lcd.lengthen_mode3((dots - LENGTH_BASE) as u8);
            // A read lands at an even half at normal speed, at any at double speed.
            (2 * MODE3_START as u16..228).step_by(if double { 1 } else { 2 }).find(|&p| lcd.sees_mode0(p)).unwrap()
        };
        let table = |cgb: bool, double: bool| (0..8).map(|f| first(cgb, double, f)).collect::<Vec<_>>();
        assert_eq!(table(true, false), [126, 126, 126, 126, 128, 128, 128, 128], "a CGB at normal speed: M-cycle 63 to SCX 3");
        assert_eq!(table(true, true), [127, 128, 128, 129, 129, 130, 130, 131], "a CGB at double speed, by halves");
        assert_eq!(table(false, false), [126, 128, 128, 128, 128, 130, 130, 130], "a DMG: 1 to 4 pixels an M-cycle later, 5 to 7 two");
        // The HBlank HDMA request, two dots after STAT's mode 0 at normal speed and one at
        // double speed (Mealybug `dma/hdma_timing-C` at SCX 1 and 2).
        let request = |double: bool, f: u16| {
            let mut lcd = Lcd::post_boot().on_console(true);
            lcd.set_double(double);
            run(&mut lcd, 114 + 20);
            lcd.lengthen_mode3(44 + f as u8);
            lcd.request_at().unwrap()
        };
        assert_eq!((0..8).map(|f| request(false, f)).collect::<Vec<_>>(), [126, 126, 128, 128, 128, 128, 130, 130], "the request at normal speed");
        assert_eq!((0..8).map(|f| request(true, f)).collect::<Vec<_>>(), [128, 128, 129, 129, 130, 130, 131, 131], "the request at double speed");
    }
    #[test]
    fn lcd_off_resets_ly_and_stops() {
        let mut lcd = Lcd::post_boot();
        run(&mut lcd, 114 * 154 + 30);
        assert_eq!(lcd.write(0xFF40, 0x11), (0, true));
        assert_eq!((lcd.read(0xFF44, false, false), lcd.read(0xFF41, false, false) & 3, lcd.horizon()), (0, 0, u64::MAX));
        assert_eq!(run(&mut lcd, 20000), Tick::default(), "nothing while off");
        lcd.write(0xFF41, 0x20);
        assert_eq!(lcd.write(0xFF40, 0x91), (0, false), "on again, line 0 reads mode 0 until mode 3 and raises no mode-2 interrupt");
    }
}
