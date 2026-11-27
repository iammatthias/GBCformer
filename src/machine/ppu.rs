//! PPU: VRAM, OAM, the display registers, the palettes and a scanline renderer that
//! draws each visible line when the control logic's timing (`machine::lcd`) says mode 3
//! begins. CGB mode adds VRAM bank 1 (BG map attributes and tiles), colour palette RAM,
//! and CGB object priority. The PPU keeps no clock: the machine calls
//! [`Ppu::render_line`], [`Ppu::end_frame`] and [`Ppu::lcd_off`] on the control device's
//! signals. The palette index registers (BCPS, OCPS) are the control logic's too; it
//! hands the PPU each palette byte written ([`Ppu::display_write`]).

use super::fifo;
use crate::memory::layout::{display_index, PAL_DMG, PAL_LEN, PAL_REGS};

/// Half-dots from the end of the M-cycle STAT's mode 3 begins in to where a write that
/// ends an M-cycle lands in SameBoy's line (`fifo::Time`); fixed on mealybug's
/// m3_bgp_change.
pub const WRITE_OFFSET: i64 = 8;

/// A line whose mode 3 began: what it started from and the display register writes until it
/// is drawn, at the next line, VBlank, the LCD turned off, STOP or a mode change. A write
/// after mode 3 ends lands past the FIFO's last pixel, so the FIFO's time alone decides what
/// a write reaches.
struct Open {
    line: Line,
    start: u64,
    display: Display,
    window: fifo::Window,
    objects: Vec<u8>,
    double: bool,
    writes: Vec<fifo::Write>,
    /// The line's mode-3 length in dots with no write ([`fifo::judge`]'s measure).
    quiet: u16,
    /// A write [`fifo::judge`] says can change the line.
    marked: bool,
    /// When mode 3 ends with the writes so far.
    end: fifo::Time,
    /// VRAM and OAM as of mode 3, kept when something wrote either before the line ended.
    memory: Option<Box<([u8; 0x4000], [u8; 0xA0])>>,
}

pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 144;

/// 0x00RRGGBB per pixel, row-major.
pub type Frame = [u32; WIDTH * HEIGHT];

/// R, G, B bytes per pixel, row-major: a PPM body, or an image for a screenshot.
pub fn rgb24(frame: &Frame) -> Vec<u8> {
    frame.iter().flat_map(|px| px.to_be_bytes()[1..].to_vec()).collect()
}

/// DMG shades, lightest first, as the grey-green colours the host shows.
pub const DMG_SHADES: [u32; 4] = [0xE0F8D0, 0x88C070, 0x346856, 0x081820];
const CGB_WHITE: u32 = 0xFFFFFF;

/// RGB555 as stored in palette RAM to 0x00RRGGBB, expanding each 5-bit channel as
/// `c << 3 | c >> 2` (the expansion cgb-acid2's reference image uses).
pub fn rgb555_to_rgb(color: u16) -> u32 {
    let expand = |c: u16| ((c & 0x1F) << 3 | (c & 0x1F) >> 2) as u32;
    expand(color) << 16 | expand(color >> 5) << 8 | expand(color >> 10)
}

/// A pixel's final colour as the PPU outputs it: CGB's RGB555, or DMG's 2-bit shade.
pub type Colour = u16;

/// What the screen shows for a [`Colour`]: the presentation, a fixed map with no
/// register in it.
pub fn rgb(cgb: bool, colour: Colour) -> u32 {
    if cgb { rgb555_to_rgb(colour) } else { DMG_SHADES[colour as usize & 3] }
}

/// The palettes a line's pixels are looked up in, as of the line, in the PPU
/// checkpoint's layout (`memory::layout::PAL`): CGB background palette RAM, object
/// palette RAM, then DMG's BGP, OBP0 and OBP1.
pub type Palettes = [u8; PAL_REGS as usize];

/// Every byte a line reads besides VRAM and OAM, in the PPU checkpoint's layout
/// (`memory::layout::PAL`): the [`Palettes`], then LCDC, SCY, SCX, WY and WX.
pub type Display = [u8; PAL_LEN];

/// The palette lookup of a [`Pixel`] drawn in DMG or CGB mode.
pub fn colour(palettes: &Palettes, cgb: bool, px: Pixel) -> Colour {
    colour_in(palettes, if cgb { Mode::Cgb } else { Mode::Dmg }, px)
}

/// How the PPU draws. `Compat` is a CGB in DMG compatibility mode (a DMG cartridge on CGB
/// hardware): it draws as DMG does, then looks each DMG shade up in CGB palette RAM,
/// background palette 0 for BGP and object palettes 0 and 1 for OBP0 and OBP1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Dmg,
    Cgb,
    Compat,
}

impl Mode {
    /// CGB rendering: map attributes, VRAM bank 1, CGB object priority, LCDC.0 as priority.
    pub fn cgb_features(self) -> bool {
        self == Mode::Cgb
    }

    /// CGB hardware: colours are RGB555 from palette RAM.
    pub fn rgb555(self) -> bool {
        self != Mode::Dmg
    }
}

/// The palette lookup of a [`Pixel`] drawn in `mode`.
pub fn colour_in(palettes: &Palettes, mode: Mode, px: Pixel) -> Colour {
    let (color, palette, obj) = (px as usize & 3, px as usize >> 2 & 7, px & 0x20 != 0);
    let entry = |palette: usize, color: usize| {
        let i = (obj as usize) * 64 + palette * 8 + color * 2;
        u16::from_le_bytes([palettes[i], palettes[i + 1]]) & 0x7FFF
    };
    if mode == Mode::Cgb {
        return entry(palette, color);
    }
    let reg = match (obj, palette & 1) {
        (false, _) => 0,
        (true, 0) => 1,
        (true, _) => 2,
    };
    let shade = palettes[PAL_DMG as usize + reg] >> (color * 2) & 3;
    match mode {
        Mode::Compat => entry(reg.saturating_sub(1), shade as usize),
        _ => shade as Colour,
    }
}

/// The visible line a renderer draws at the start of its mode 3. The display registers
/// are display bytes the renderer was handed as written ([`Renderer::display_write`]).
/// `cgb` is CGB hardware; `compat` its DMG compatibility mode.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub ly: u8,
    pub cgb: bool,
    pub compat: bool,
    /// Ordinary STOP keeps a CGB's PPU from VRAM: the line draws black.
    pub blocked: bool,
    /// The GBC CPU runs at double speed: its writes' changes land on half-dots.
    pub double: bool,
}

impl Line {
    pub fn mode(&self) -> Mode {
        match (self.cgb, self.compat) {
            (false, _) => Mode::Dmg,
            (true, false) => Mode::Cgb,
            (true, true) => Mode::Compat,
        }
    }
}

/// The reference renderer's line: each pixel's colour, the window's state after the
/// line and its length (`fifo::length`: the dots to mode 0 less 128), which a renderer
/// under lockstep must equal, and the display registers it drew with (LCDC, SCY, SCX, WY,
/// WX), for the report of a difference.
/// On a line [`fifo::judge`] marks, `trace` is the FIFO's state before the line's first
/// step and after each, which a renderer's stepper must equal pass for pass.
#[derive(Clone, Debug)]
pub struct Drawn {
    pub colours: [Colour; WIDTH],
    pub window: fifo::Window,
    pub length: u8,
    pub registers: [u8; 5],
    pub trace: Option<Vec<fifo::State>>,
}

/// Each drawn line's length (`fifo::length`), from whoever draws it (the reference, or
/// a renderer that draws alone), for the control logic's timing to place mode 0. The PPU
/// issues a line as it hands it over; a reader waits until every issued line is drawn,
/// first asking the renderer to stop queueing (`set_flush`). A renderer whose stepper draws
/// the newest line issues it again for each revision the reader has yet to read
/// ([`Lengths::reissue`]). Clones share one board.
#[derive(Clone, Default)]
pub struct Lengths(std::sync::Arc<LengthsInner>, bool);

#[derive(Default)]
struct Board {
    /// Lines (and revisions) issued and not yet drawn.
    pending: u64,
    /// The newest drawn line's length.
    length: u8,
    /// The reader read the newest line's length.
    read: bool,
}

#[derive(Default)]
struct LengthsInner {
    state: std::sync::Mutex<Board>,
    drawn: std::sync::Condvar,
    flush: std::sync::Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl Lengths {
    /// A board no renderer feeds, for the control logic run alone: every line has `length`, the
    /// dots to mode 0 less `lcd::LENGTH_BASE`.
    pub fn constant(length: u8) -> Lengths {
        let board = Lengths::default();
        board.0.state.lock().expect("the lengths board").length = length;
        board
    }

    /// The same board for a reference that runs beside the model under lockstep: its reads
    /// wait and see the same lengths but do not count as the reader's, so the model's
    /// revisions follow the model's own read alone, as in a run without the reference.
    pub fn follower(&self) -> Lengths {
        Lengths(self.0.clone(), true)
    }

    /// A line was handed to be drawn.
    pub fn issue(&self) {
        let mut s = self.0.state.lock().expect("the lengths board");
        s.pending += 1;
        s.read = false;
    }

    /// The newest line's length will be revised, if the reader has not read it yet: then it
    /// counts as issued again, until the revision is published, and this returns true.
    pub fn reissue(&self) -> bool {
        let mut s = self.0.state.lock().expect("the lengths board");
        if !s.read {
            s.pending += 1;
        }
        !s.read
    }

    /// The reader has not read the newest line's length yet; unlike [`Lengths::reissue`] it
    /// issues nothing.
    pub fn unread(&self) -> bool {
        !self.0.state.lock().expect("the lengths board").read
    }

    /// The oldest issued line (or revision) is drawn, `length` M-cycles of mode 3.
    pub fn publish(&self, length: u8) {
        let mut s = self.0.state.lock().expect("the lengths board");
        assert!(s.pending > 0, "a mode-3 length published for no line issued");
        (s.pending, s.length) = (s.pending - 1, length);
        if s.pending == 0 {
            self.0.drawn.notify_all();
        }
    }

    /// The newest line's length changed after it was published: a display write in its
    /// mode 3 moved where mode 3 ends.
    pub fn revise(&self, length: u8) {
        self.0.state.lock().expect("the lengths board").length = length;
    }

    /// The newest length published or revised so far, without waiting: under lockstep the
    /// reference's, which a renderer's revision is held to.
    pub fn current(&self) -> u8 {
        self.0.state.lock().expect("the lengths board").length
    }

    /// The newest issued line's length, once it is drawn.
    pub fn latest(&self) -> u8 {
        let pending = self.0.state.lock().expect("the lengths board").pending > 0;
        if pending && let Some(flush) = &*self.0.flush.lock().expect("the lengths board's flush") {
            flush();
        }
        let mut s = self.0.state.lock().expect("the lengths board");
        while s.pending > 0 {
            s = self.0.drawn.wait(s).expect("the lengths board");
        }
        s.read |= !self.1;
        s.length
    }

    /// How a reader asks the renderer to hand over the lines it queued.
    pub fn set_flush(&self, flush: Box<dyn Fn() + Send>) {
        *self.0.flush.lock().expect("the lengths board's flush") = Some(flush);
    }
}

/// Draws the visible lines in place of the reference renderer: the PPU checkpoint. It
/// keeps its own display bytes (the palettes and the display registers) and window
/// state; the PPU hands it every visible line at the start of mode 3 and every display
/// byte written, in bus order.
pub trait Renderer: Send {
    /// Starts from `back`, the frame drawn so far.
    fn begin(&mut self, back: &Frame);
    /// Mode 3 of `line` began: the records kept so far are the ones the line draws with,
    /// and any kept after wait until the line is drawn ([`Renderer::line`]).
    fn open(&mut self, line: Line);
    /// Draws `line`; under lockstep `reference` holds the reference renderer's, which the
    /// renderer must equal. Without one, the renderer publishes the line's mode-3 length
    /// to the PPU's [`Lengths`].
    fn line(&mut self, line: Line, reference: Option<Box<Drawn>>);
    /// VBlank began: the frame is every line handed over so far. A renderer on its own
    /// thread need not have drawn them yet; [`Renderer::front`] waits for it.
    fn end_frame(&mut self);
    /// The frame of the last [`Renderer::end_frame`], drawn.
    fn front(&mut self, out: &mut Frame);
    /// A display byte written, at its index in [`Display`]; `open` while a line is open.
    fn display_write(&mut self, index: u8, value: u8, open: Option<LineWrite>);
    /// The display byte at `index` as of every write handed over so far; under lockstep
    /// `reference` holds the host's, which the renderer must equal.
    fn display_read(&mut self, index: u8, reference: Option<u8>) -> u8;
    /// Whether `line` needs the reference.
    fn lockstep(&self) -> bool;
    fn describe(&self) -> Vec<String>;
    fn report(&mut self) -> Vec<String>;
    /// The lines its judge marked, which its stepper drew ([`Midline`]).
    fn stepped(&self) -> u64;
    /// Display reads served from the record queue while a line was open, lines whose
    /// length was revised before the control logic read it, and lines a write the stepper
    /// took reached after that read ([`Midline`]).
    fn rows(&self) -> [u64; 3] {
        [0; 3]
    }
}

/// A display write while a line is open, as a renderer is handed it: the machine's 2-dot
/// halves since the line's mode 3 began, which its length stream judges the write at, and
/// under lockstep the reference's judgement.
#[derive(Clone, Copy, Debug)]
pub struct LineWrite {
    pub halves: u8,
    pub reference: Option<Judged>,
}

/// The reference's judgement of a write ([`fifo::judge`]): its time in the line, in
/// half-dots, and whether it marks the line for the stepper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Judged {
    pub at: fifo::Time,
    pub marks: bool,
}

/// Lines a display write can change, which the stepper draws (`docs/ppu-in-weights.md`
/// §14.2): those the renderer's judge marked, or with no renderer those [`fifo::judge`]
/// marks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Midline {
    pub stepped: u64,
    /// Compatibility mode set while a line was open, which closed it first.
    pub compat_closes: u64,
    /// The renderer's [`Renderer::rows`].
    pub forwarded: u64,
    pub revised_before: u64,
    pub taken_after: u64,
}

pub struct Ppu {
    cgb: bool,
    /// DMG compatibility mode on CGB hardware ([`Mode::Compat`]).
    compat: bool,
    /// In ordinary STOP a CGB's PPU cannot read VRAM and draws black lines.
    stopped: bool,
    /// Bank 0 at 0000-1FFF, bank 1 (CGB only) at 2000-3FFF.
    vram: [u8; 0x4000],
    vbk: u8,
    pub oam: [u8; 0xA0],
    palettes: Palettes,
    lcdc: u8,
    scy: u8,
    scx: u8,
    wy: u8,
    wx: u8,
    window: fifo::Window,
    /// The machine's clock in 2-dot halves ([`Ppu::set_clock`]): when a write lands within
    /// the line being drawn.
    now: u64,
    double: bool,
    /// The visible line whose mode 3 began and that is not drawn yet.
    open: Option<Open>,
    /// The line a renderer drawing alone opened and its start, until it is handed over.
    plain: Option<(Line, u64)>,
    midline: Midline,
    back: Frame,
    front: Frame,
    /// The renderer has a frame `front` has not fetched yet.
    front_pending: bool,
    /// Draws the lines when the PPU is in weights.
    renderer: Option<Box<dyn Renderer>>,
    lengths: Lengths,
}

/// One background or window pixel before palette lookup.
#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct BgPixel {
    color: u8,
    /// CGB map attributes: palette in bits 0-2, priority in bit 7. Zero on DMG.
    attr: u8,
}

/// A rendered pixel before the palette lookup, as one byte: the 2-bit colour index in
/// bits 0-1, the palette in bits 2-4 (CGB: the map or OAM palette; DMG: OBP1 selected
/// in bit 2 for an object, 0 for the background) and bit 5 set when an object won.
pub type Pixel = u8;

impl Ppu {
    pub fn post_boot(cgb: bool) -> Self {
        let blank = if cgb { CGB_WHITE } else { DMG_SHADES[0] };
        let mut palettes: Palettes = [0; PAL_REGS as usize];
        // The CGB boot ROM leaves colour palettes white for CGB games.
        for pair in palettes[..PAL_DMG as usize].chunks_mut(2) {
            pair.copy_from_slice(&0x7FFFu16.to_le_bytes());
        }
        // BGP, then OBP0 and OBP1: uninitialised on hardware; 0x00 after a CGB boot is
        // the common value (SameBoy).
        let obp = if cgb { 0x00 } else { 0xFF };
        palettes[PAL_DMG as usize..].copy_from_slice(&[0xFC, obp, obp]);
        Ppu {
            cgb,
            compat: false,
            stopped: false,
            vram: [0; 0x4000],
            vbk: 0,
            oam: [0; 0xA0],
            palettes,
            lcdc: 0x91,
            scy: 0,
            scx: 0,
            wy: 0,
            wx: 0,
            window: fifo::Window::default(),
            now: 0,
            double: false,
            open: None,
            plain: None,
            midline: Midline::default(),
            back: [blank; WIDTH * HEIGHT],
            front: [blank; WIDTH * HEIGHT],
            front_pending: false,
            renderer: None,
            lengths: Lengths::default(),
        }
    }

    /// The board each drawn line's mode-3 length reaches the control logic by.
    pub fn lengths(&self) -> Lengths {
        self.lengths.clone()
    }

    pub fn post_boot_dmg() -> Self {
        Self::post_boot(false)
    }

    /// A CGB after its boot ROM handed a DMG cartridge over in compatibility mode, with
    /// the palettes the boot ROM loads for a game it does not recognise (background
    /// 7FFF 1BEF 6180 0000, both object palettes 7FFF 421F 1CF2 0000; a recognised
    /// Nintendo title gets one of its own, which only running the boot ROM gives).
    pub fn post_boot_compat() -> Self {
        let mut ppu = Self::post_boot(true);
        ppu.compat = true;
        // As the CGB boot ROM leaves them, measured by running it to its hand-off: the
        // unused background palettes white with colour 0 at FFFF, the unused object
        // palettes and OBP0, OBP1 0.
        for (i, pair) in ppu.palettes[..64].chunks_mut(2).enumerate() {
            pair.copy_from_slice(&if i % 4 == 0 { 0xFFFFu16 } else { 0x7FFF }.to_le_bytes());
        }
        ppu.palettes[64..128].fill(0);
        let bg = [0x7FFF, 0x1BEF, 0x6180, 0x0000u16];
        let obj = [0x7FFF, 0x421F, 0x1CF2, 0x0000u16];
        for (palette, colours) in [(0, bg), (8, obj), (9, obj)] {
            for (i, c) in colours.iter().enumerate() {
                ppu.palettes[palette * 8 + i * 2..][..2].copy_from_slice(&c.to_le_bytes());
            }
        }
        ppu.palettes[PAL_DMG as usize..].copy_from_slice(&[0xFC, 0x00, 0x00]);
        ppu
    }

    /// The VRAM the DMG boot ROM leaves: the header's logo, each bit doubled across and
    /// down, in tiles 1 to 24, the ® mark in tile 25 (both in bit plane 0), and the logo's
    /// two map rows: tiles 1 to 12 and 25 at 9904-9910, 13 to 24 at 9924-992F. Mealybug's
    /// tests draw the ® tile.
    pub fn boot_logo(&mut self, logo: &[u8]) {
        let double = |n: u8| (0..4).fold(0u8, |b, i| b | ((n >> i & 1) * 3) << (2 * i));
        for (k, &byte) in logo.iter().enumerate() {
            for (h, nibble) in [byte >> 4, byte & 15].into_iter().enumerate() {
                let row = 0x10 + (k * 2 + h) * 4;
                self.vram[row] = double(nibble);
                self.vram[row + 2] = double(nibble);
            }
        }
        for (i, b) in [0x3C, 0x42, 0xB9, 0xA5, 0xB9, 0xA5, 0x42, 0x3C].into_iter().enumerate() {
            self.vram[0x190 + i * 2] = b;
        }
        for t in 1..=12u8 {
            self.vram[0x1903 + t as usize] = t;
            self.vram[0x1923 + t as usize] = t + 12;
        }
        self.vram[0x1910] = 0x19;
    }

    /// At power-on, for a boot ROM to run: LCDC 0, the LCD off and blank, the palettes 0.
    pub fn power_on(cgb: bool) -> Self {
        let mut ppu = Self::post_boot(cgb);
        ppu.lcdc = 0;
        ppu.palettes = [0; PAL_REGS as usize];
        ppu
    }

    /// Ordinary STOP began with VRAM out of reach (`true`), or ended.
    pub fn set_stopped(&mut self, blocked: bool) {
        self.close();
        self.stopped = blocked;
    }

    /// Ordinary STOP keeps a CGB's PPU from VRAM now.
    pub fn held(&self) -> bool {
        self.stopped
    }

    /// The frame being drawn and the frame shown.
    pub fn frames(&self) -> (&Frame, &Frame) {
        (&self.back, &self.front)
    }

    /// How the PPU draws now.
    pub fn mode(&self) -> Mode {
        Line { ly: 0, cgb: self.cgb, compat: self.compat, blocked: false, double: false }.mode()
    }

    /// The boot ROM switched the console to DMG compatibility mode (KEY0 bit 2 when it
    /// unmapped itself).
    pub fn set_compat(&mut self, compat: bool) {
        self.midline.compat_closes += (self.plain.is_some() || self.open.is_some()) as u64;
        self.close();
        self.compat = compat;
    }

    /// The last completed frame.
    pub fn vram(&self) -> &[u8] {
        &self.vram
    }

    /// The last finished frame, as of the last [`Ppu::settle`].
    pub fn frame(&self) -> &Frame {
        &self.front
    }

    /// The palettes as of now.
    pub fn palettes(&self) -> &Palettes {
        &self.palettes
    }

    /// The window's state as of now: after the open line, if one is, which a renderer taking
    /// over (`set_renderer`) does not draw.
    pub fn window(&self) -> fifo::Window {
        self.open.as_ref().map_or(self.window, |o| fifo::draw(&self.start(o), &o.writes).window)
    }

    /// A CPU-side VRAM access at 8000-9FFF goes to the bank VBK selects.
    pub fn vram_read(&self, addr: u16) -> u8 {
        self.vram[self.vram_index(addr)]
    }

    pub fn vram_write(&mut self, addr: u16, value: u8) {
        self.before_memory_write();
        let i = self.vram_index(addr);
        self.vram[i] = value;
    }

    fn vram_index(&self, addr: u16) -> usize {
        self.vbk as usize * 0x2000 + (addr as usize & 0x1FFF)
    }

    /// VBK, the VRAM bank the CPU sees at 8000-9FFF.
    pub fn vbk(&self) -> u8 {
        self.vbk
    }

    /// A CPU read of VRAM, OAM or VBK. The display bytes (BGP, OBP0, OBP1 and the display
    /// registers) are read with [`Ppu::display_read`].
    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0x8000..=0x9FFF => self.vram_read(addr),
            0xFE00..=0xFE9F => self.oam[addr as usize - 0xFE00],
            // CGB hardware reads VBK in either mode (mooneye `misc/boot_hwio-C`, SameBoy).
            0xFF4F if self.mode() != Mode::Dmg => 0xFE | self.vbk,
            _ => display_index(addr).map_or(0xFF, |i| self.display_byte(i)),
        }
    }

    /// A CPU write. LCDC goes to the control logic too, whose timing reads bit 7.
    pub fn write(&mut self, addr: u16, value: u8) {
        match addr {
            0x8000..=0x9FFF => self.vram_write(addr, value),
            0xFE00..=0xFE9F => {
                self.before_memory_write();
                self.oam[addr as usize - 0xFE00] = value;
            }
            0xFF4F if self.mode().cgb_features() => self.vbk = value & 1,
            _ => {
                if let Some(i) = display_index(addr) {
                    self.display_write(i, value);
                }
            }
        }
    }

    /// The display byte at `index` in [`Display`], as this PPU holds it.
    fn display_byte(&self, index: u8) -> u8 {
        match (index as u32).checked_sub(PAL_REGS) {
            None => self.palettes[index as usize],
            Some(0) => self.lcdc,
            Some(1) => self.scy,
            Some(2) => self.scx,
            Some(3) => self.wy,
            Some(_) => self.wx,
        }
    }

    /// Every display byte as of now, which a renderer taking over starts from.
    pub fn display(&self) -> Display {
        std::array::from_fn(|i| self.display_byte(i as u8))
    }

    /// A display byte written at its index in [`Display`]: a CGB palette RAM byte the
    /// control logic let through, BGP, OBP0, OBP1 or a display register. A renderer gets
    /// it too, with the halves since the open line's mode 3 began. The reference keeps a
    /// register write in the open line, judges it ([`fifo::judge`]) and revises the line's
    /// published length; palette RAM cannot be written during mode 3, so it never reaches
    /// a line.
    pub fn display_write(&mut self, index: u8, value: u8) {
        self.set_display(index, value);
        let start = self.plain.map(|(_, s)| s).or(self.open.as_ref().map(|o| o.start));
        let halves = start.map(|s| u8::try_from(self.now - s).expect("a display write more than 255 halves into an open line"));
        let mut reference = None;
        if let Some(mut o) = self.open.take() {
            let w = fifo::Write { at: self.line_time(o.start), index, value };
            let marks = fifo::judge(&self.start(&o), o.quiet, w);
            reference = Some(Judged { at: w.at, marks });
            o.marked |= marks;
            if index as u32 >= PAL_DMG {
                o.writes.push(w);
                let fine = o.display[PAL_REGS as usize + 2];
                let before = fifo::length((o.end / 2) as u16, fine, self.mode().rgb555());
                o.end = fifo::draw(&self.start(&o), &o.writes).dots as fifo::Time * 2;
                let after = fifo::length((o.end / 2) as u16, fine, self.mode().rgb555());
                if after != before {
                    self.lengths.revise(after);
                }
            }
            self.open = Some(o);
        }
        if let Some(r) = &mut self.renderer {
            r.display_write(index, value, halves.map(|halves| LineWrite { halves, reference }));
        }
    }

    fn set_display(&mut self, index: u8, value: u8) {
        match (index as u32).checked_sub(PAL_REGS) {
            None => self.palettes[index as usize] = value,
            Some(0) => self.lcdc = value,
            Some(1) => self.scy = value,
            Some(2) => self.scx = value,
            Some(3) => self.wy = value,
            Some(_) => self.wx = value,
        }
    }

    /// A display byte as the bus reads it: the renderer's when one draws the lines, held
    /// to this PPU's own under lockstep.
    pub fn display_read(&mut self, index: u8) -> u8 {
        let own = self.display_byte(index);
        match &mut self.renderer {
            Some(r) => {
                let reference = r.lockstep().then_some(own);
                r.display_read(index, reference)
            }
            None => own,
        }
    }

    /// Line 144 began: the frame drawn so far is the one shown, and the window starts
    /// over for the next. A renderer's frame is fetched when it is first read
    /// ([`Ppu::settle`]), so the machine does not wait for it here.
    pub fn end_frame(&mut self) {
        self.close();
        match &mut self.renderer {
            Some(r) => {
                r.end_frame();
                self.front_pending = true;
            }
            None => self.front = self.back,
        }
        self.window = fifo::Window::default();
    }

    /// Fetches the renderer's frame of the last [`Ppu::end_frame`], if not yet fetched.
    pub fn settle(&mut self) {
        if let (true, Some(r)) = (self.front_pending, &mut self.renderer) {
            r.front(&mut self.front);
            self.front_pending = false;
        }
    }

    /// The LCD turned off: the screen shows blank, and the window starts over.
    pub fn lcd_off(&mut self) {
        self.close();
        // A frame still being drawn was shown before the screen went blank.
        self.settle();
        self.window = fifo::Window::default();
        self.front = [if self.cgb { CGB_WHITE } else { DMG_SHADES[0] }; WIDTH * HEIGHT];
    }

    /// The machine's clock, in 2-dot halves, and the GBC CPU's speed, before anything it
    /// hands the PPU: a display write's time within the line being drawn comes from it.
    pub fn set_clock(&mut self, halves: u64, double: bool) {
        self.now = halves;
        self.double = double;
    }

    /// Half-dots since the open line's mode 3 began, where a write landing now takes effect.
    fn line_time(&self, start: u64) -> fifo::Time {
        ((self.now - start) as i64 * 4 + WRITE_OFFSET).max(0) as fifo::Time
    }

    /// VRAM or OAM is about to change: the open line keeps them as of its start.
    fn before_memory_write(&mut self) {
        if let Some(o) = self.open.as_mut().filter(|o| o.memory.is_none()) {
            o.memory = Some(Box::new((self.vram, self.oam)));
        }
    }

    /// The counts of [`Midline`]: the renderer's when one draws alone.
    pub fn midline(&self) -> Midline {
        match &self.renderer {
            Some(r) if !r.lockstep() => {
                let [forwarded, revised_before, taken_after] = r.rows();
                Midline { stepped: r.stepped(), forwarded, revised_before, taken_after, ..self.midline }
            }
            _ => self.midline,
        }
    }

    /// The FIFO's start for the open line.
    fn start<'a>(&'a self, o: &'a Open) -> fifo::Start<'a> {
        let (vram, oam) = o.memory.as_deref().map_or((&self.vram, &self.oam), |(v, a)| (v, a));
        fifo::Start { ly: o.line.ly, mode: o.line.mode(), double: o.double, blocked: o.line.blocked, display: o.display, window: o.window, vram, oam, objects: &o.objects }
    }

    /// Mode 3 of a visible line began: the line opens with the display bytes, the window's
    /// state and the objects the OAM scan chose, and is drawn when it closes, with every
    /// display register write it saw at its time ([`fifo::draw`]). The reference publishes
    /// the length the line has with the writes so far, and revises it on each write.
    pub fn render_line(&mut self, ly: u8) {
        self.close();
        let line = Line { ly, cgb: self.cgb, compat: self.compat, blocked: self.stopped, double: self.double };
        self.lengths.issue();
        // A renderer without a reference beside it keeps its own display bytes and window
        // state; the host keeps only the line's start, for the writes' times.
        if self.renderer.as_ref().is_some_and(|r| !r.lockstep()) {
            self.plain = Some((line, self.now));
            self.renderer.as_mut().expect("a renderer").open(line);
            return;
        }
        // The window arms when LY equals WY at the line's start. SameBoy also asks for the
        // window enabled, and checks again at an LCDC or WY write; no test in the census
        // tells the two apart (`docs/ppu-in-weights.md` §13.2).
        self.window = self.window.starting(ly, self.wy);
        if let Some(r) = &mut self.renderer {
            r.open(line);
        }
        let mut open = Open { line, start: self.now, display: self.display(), window: self.window, objects: fifo::scan(&self.oam, self.lcdc, ly), double: self.double, writes: Vec::new(), quiet: 0, marked: false, end: 0, memory: None };
        open.quiet = fifo::draw(&self.start(&open), &[]).dots;
        open.end = open.quiet as fifo::Time * 2;
        self.lengths.publish(fifo::length(open.quiet, self.scx, self.mode().rgb555()));
        self.open = Some(open);
    }

    /// Draws the open line, if any, with the writes it saw; a renderer drawing alone is
    /// handed its line.
    fn close(&mut self) {
        if let Some((line, _)) = self.plain.take() {
            self.renderer.as_mut().expect("a renderer drawing alone").line(line, None);
        }
        let Some(o) = self.open.take() else { return };
        let held = self.renderer.as_ref().is_some_and(|r| r.lockstep());
        let (drawn, trace) = {
            let start = self.start(&o);
            let drawn = fifo::draw(&start, &o.writes);
            let trace = (o.marked && held).then(|| {
                let passes = fifo::trace(&start, &o.writes);
                std::iter::once(passes[0].before).chain(passes.iter().map(|p| p.after)).collect()
            });
            (drawn, trace)
        };
        self.midline.stepped += o.marked as u64;
        self.window = drawn.window;
        let [lcdc, scy, scx, wy, wx] = std::array::from_fn(|i| o.display[PAL_REGS as usize + i]);
        let reference = Drawn { colours: drawn.colours, window: drawn.window, length: fifo::length(drawn.dots, scx, self.mode().rgb555()), registers: [lcdc, scy, scx, wy, wx], trace };
        match &mut self.renderer {
            Some(r) => r.line(o.line, Some(Box::new(reference))),
            None => {
                let row = &mut self.back[o.line.ly as usize * WIDTH..][..WIDTH];
                for (out, &c) in row.iter_mut().zip(&reference.colours) {
                    *out = rgb(self.cgb, c);
                }
            }
        }
    }
    /// 2-bit colour index of pixel (x, y) of the 16-byte tile at `base` in `vram`.
    #[cfg(test)]
    fn pixel_at(&self, base: usize, x: u8, y: u8) -> u8 {
        let lo = self.vram[base + y as usize * 2];
        let hi = self.vram[base + y as usize * 2 + 1];
        let bit = 7 - x;
        (hi >> bit & 1) << 1 | (lo >> bit & 1)
    }

    /// Background or window pixel at map coordinates (x, y), honouring LCDC.4
    /// addressing and, on CGB, the map attributes in VRAM bank 1.
    #[cfg(test)]
    fn map_pixel(&self, map_high: bool, x: u8, y: u8) -> BgPixel {
        let entry = if map_high { 0x1C00 } else { 0x1800 } + (y as usize / 8) * 32 + x as usize / 8;
        let tile = self.vram[entry];
        let attr = if self.mode().cgb_features() { self.vram[0x2000 + entry] } else { 0 };
        let base = if self.lcdc & 0x10 == 0 { (0x1000 + (tile as i8 as i32) * 16) as usize } else { tile as usize * 16 };
        let bank = if attr & 0x08 != 0 { 0x2000 } else { 0 };
        let px = if attr & 0x20 != 0 { 7 - x % 8 } else { x % 8 };
        let py = if attr & 0x40 != 0 { 7 - y % 8 } else { y % 8 };
        BgPixel { color: self.pixel_at(bank + base, px, py), attr }
    }
    /// Dots of mode 3 on line `ly` after it was drawn (Pan Docs, "Mode 3 length"): 172, plus
    /// SCX's fine scroll, 6 when the window shows on the line, and per object fetched 6
    /// more, the first object in each background tile also waiting out the rest of that
    /// tile's fetch, `5 - min(5, (X + SCX) mod 8)` (mooneye intr_2_mode0_timing_sprites).
    /// Objects at X 168 and beyond are not fetched.
    pub fn mode3_dots(&self, ly: u8) -> u16 {
        // SCX's fine scroll ends mode 3 at the next M-cycle's boundary: 1-4 pixels one
        // M-cycle later, 5-7 two (mooneye `hblank_ly_scx_timing-GS`).
        let mut dots = 172 + ((self.scx & 7) as u16).div_ceil(4) * 4;
        let cgb = self.mode().cgb_features();
        if (cgb || self.lcdc & 0x01 != 0) && self.lcdc & 0x20 != 0 && self.window.armed && self.wx <= 166 {
            dots += 6;
        }
        if self.lcdc & 0x02 == 0 {
            return dots;
        }
        let height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };
        let mut xs: Vec<u16> = (0..40)
            .filter(|&i| (0..height).contains(&(ly as i16 - (self.oam[i * 4] as i16 - 16))))
            .take(10)
            .map(|i| self.oam[i * 4 + 1] as u16)
            .filter(|&x| x < 168)
            .collect();
        xs.sort_unstable();
        let mut tiles = Vec::new();
        for x in xs {
            let at = x + (self.scx & 7) as u16;
            if !tiles.contains(&(at / 8)) {
                tiles.push(at / 8);
                dots += 5 - (at % 8).min(5);
            }
            dots += 6;
        }
        dots
    }

    /// The reference renderer's pixels for the current line, with the window armed as
    /// of this line and its counter not yet advanced.
    #[cfg(test)]
    fn reference_line(&self, ly: u8) -> [Pixel; WIDTH] {
        let mut bg = [BgPixel::default(); WIDTH];
        if self.mode().cgb_features() || self.lcdc & 0x01 != 0 {
            for (x, px) in bg.iter_mut().enumerate() {
                *px = self.map_pixel(self.lcdc & 0x08 != 0, self.scx.wrapping_add(x as u8), self.scy.wrapping_add(ly));
            }
            let window_x = self.wx as i16 - 7;
            if self.lcdc & 0x20 != 0 && self.window.armed && self.wx <= 166 {
                for x in window_x.max(0)..WIDTH as i16 {
                    bg[x as usize] = self.map_pixel(self.lcdc & 0x40 != 0, (x - window_x) as u8, self.window.rows);
                }
            }
        }
        let sprites = if self.lcdc & 0x02 != 0 { self.sprite_line(ly) } else { [None; WIDTH] };
        std::array::from_fn(|x| self.pick(bg[x], sprites[x]))
    }

    /// Which layer shows at a pixel, as a [`Pixel`]. On DMG the object wins unless it
    /// asks for BG priority over a non-zero BG colour. On CGB the object wins unless
    /// LCDC.0 is set, the BG colour is not 0, and either the map attribute or the
    /// object asks for BG priority.
    #[cfg(test)]
    fn pick(&self, bg: BgPixel, sprite: Option<(u8, u8)>) -> Pixel {
        let obj_wins = match sprite {
            None => false,
            Some((_, attr)) if self.mode().cgb_features() => self.lcdc & 0x01 == 0 || bg.color == 0 || (bg.attr & 0x80 == 0 && attr & 0x80 == 0),
            Some((_, attr)) => !(attr & 0x80 != 0 && bg.color != 0),
        };
        match sprite {
            Some((color, attr)) if obj_wins => color | (if self.mode().cgb_features() { attr & 0x07 } else { attr >> 4 & 1 }) << 2 | 0x20,
            _ => bg.color | (bg.attr & 0x07) << 2,
        }
    }

    /// Hands visible lines to `renderer` from now on; it starts from the frame drawn so far.
    pub fn set_renderer(&mut self, mut renderer: Box<dyn Renderer>) {
        self.close();
        renderer.begin(&self.back);
        self.renderer = Some(renderer);
    }

    pub fn renderer_mut(&mut self) -> Option<&mut (dyn Renderer + 'static)> {
        self.renderer.as_deref_mut()
    }

    pub fn renderer(&self) -> Option<&(dyn Renderer + 'static)> {
        self.renderer.as_deref()
    }


    /// Per pixel, the highest-priority opaque object pixel: (colour index, OAM attributes).
    #[cfg(test)]
    fn sprite_line(&self, ly: u8) -> [Option<(u8, u8)>; WIDTH] {
        let height: i16 = if self.lcdc & 0x04 != 0 { 16 } else { 8 };
        let ly = ly as i16;
        let mut chosen: Vec<usize> = (0..40)
            .filter(|&i| {
                let row = ly - (self.oam[i * 4] as i16 - 16);
                (0..height).contains(&row)
            })
            .take(10)
            .collect();
        // DMG priority: lower X first, then lower OAM index (sort is stable).
        // CGB priority: OAM index alone.
        if !self.mode().cgb_features() {
            chosen.sort_by_key(|&i| self.oam[i * 4 + 1]);
        }
        let mut line = [None; WIDTH];
        for i in chosen {
            let [y, x, tile, attr] = self.oam[i * 4..i * 4 + 4] else { unreachable!() };
            let mut row = (ly - (y as i16 - 16)) as u8;
            if attr & 0x40 != 0 {
                row = height as u8 - 1 - row;
            }
            let tile = if height == 16 { tile & 0xFE } else { tile };
            let bank = if self.mode().cgb_features() && attr & 0x08 != 0 { 0x2000 } else { 0 };
            for col in 0..8u8 {
                let sx = x as i16 - 8 + col as i16;
                if !(0..WIDTH as i16).contains(&sx) || line[sx as usize].is_some() {
                    continue;
                }
                let px = if attr & 0x20 != 0 { 7 - col } else { col };
                let color = self.pixel_at(bank + tile as usize * 16 + (row as usize / 8) * 16, px, row % 8);
                if color != 0 {
                    line[sx as usize] = Some((color, attr));
                }
            }
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OAM holding its own offsets, so each pattern's result reads as the bytes it took.
    fn counting() -> [u8; 0xA0] {
        std::array::from_fn(|i| i as u8)
    }

    #[test]
    fn the_oam_bug_rewrites_the_row_the_scan_reads() {
        let mut oam = counting();
        corrupt_oam(&mut oam, 0x10, OamBug::Write);
        assert_eq!(oam[0x10..0x18], [0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F], "a write takes the row before");
        assert_eq!(oam[..0x10], counting()[..0x10]);
        assert_eq!(oam[0x18..], counting()[0x18..]);
        let mut oam = counting();
        corrupt_oam(&mut oam, 0x28, OamBug::Read);
        assert_eq!(oam[0x20..0x30], [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27]);
        let mut oam = counting();
        corrupt_oam(&mut oam, 0x00, OamBug::Write);
        corrupt_oam(&mut oam, 0xA0, OamBug::Read);
        assert_eq!(oam, counting(), "rows 0 and 0xA0 are never corrupted");
    }

    const S: [u32; 4] = DMG_SHADES;

    /// Tile `n` of VRAM bank `bank` filled with colour index `c` everywhere.
    fn solid_tile_in(ppu: &mut Ppu, bank: usize, n: usize, c: u8) {
        for row in 0..8 {
            ppu.vram[bank * 0x2000 + n * 16 + row * 2] = if c & 1 != 0 { 0xFF } else { 0 };
            ppu.vram[bank * 0x2000 + n * 16 + row * 2 + 1] = if c & 2 != 0 { 0xFF } else { 0 };
        }
    }

    fn solid_tile(ppu: &mut Ppu, n: usize, c: u8) {
        solid_tile_in(ppu, 0, n, c);
    }

    #[test]
    fn lcd_off_blanks() {
        let mut ppu = Ppu::post_boot_dmg();
        ppu.lcd_off();
        assert!(ppu.frame().iter().all(|&p| p == 0xE0F8D0), "blank is the lightest shade");
        let mut cgb = Ppu::post_boot(true);
        cgb.lcd_off();
        assert!(cgb.frame().iter().all(|&p| p == 0xFFFFFF), "CGB blanks white");
    }

    /// Every visible line drawn, then the frame shown, as the control logic's timing
    /// calls them.
    fn render_frame(ppu: &mut Ppu) {
        for ly in 0..HEIGHT as u8 {
            ppu.render_line(ly);
        }
        ppu.end_frame();
    }

    #[test]
    fn background_scroll_and_window() {
        let mut ppu = Ppu::post_boot_dmg();
        ppu.w(0xFF47, 0xE4);
        solid_tile(&mut ppu, 1, 3);
        solid_tile(&mut ppu, 2, 1);
        ppu.vram[0x1800] = 1; // map 0 tile (0,0) black, rest tile 0 (white)
        ppu.w(0xFF43, 4);
        // Window from map 1 (all tile 2), at screen x=80, y=100.
        ppu.vram[0x1C00..0x2000].fill(2);
        ppu.w(0xFF4A, 100);
        ppu.w(0xFF4B, 87);
        ppu.w(0xFF40, 0x80 | 0x40 | 0x20 | 0x10 | 0x01);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!(&f[0..6], &[S[3], S[3], S[3], S[3], S[0], S[0]], "SCX=4 shifts the black tile left");
        assert_eq!(f[8 * WIDTH], S[0]);
        assert_eq!((f[100 * WIDTH + 79], f[100 * WIDTH + 80]), (S[0], S[1]));
        assert_eq!(f[99 * WIDTH + 80], S[0], "window starts at WY");
    }

    #[test]
    fn sprites_priority_flip_and_limit() {
        let mut ppu = Ppu::post_boot_dmg();
        ppu.w(0xFF47, 0xE4);
        ppu.w(0xFF48, 0xE4);
        ppu.w(0xFF49, 0x1B); // inverted: index 1 -> shade 2
        // Tile 4: left column colour 1, rest transparent.
        for row in 0..8 {
            ppu.vram[4 * 16 + row * 2] = 0x80;
        }
        // Sprite 0 at x=20 obp1; sprite 1 at x=20 obp0 (lower index wins tie).
        ppu.oam[0..4].copy_from_slice(&[16, 20, 4, 0x10]);
        ppu.oam[4..8].copy_from_slice(&[16, 20, 4, 0x00]);
        // Sprite 2 at x=40 flipped horizontally: colour lands on screen x 39.
        ppu.oam[8..12].copy_from_slice(&[16, 40, 4, 0x20]);
        // 11 sprites on line 20; the 11th (x=150) is dropped.
        for i in 3..14 {
            let x = if i == 13 { 150 } else { 100 };
            ppu.oam[i * 4..i * 4 + 4].copy_from_slice(&[36, x, 4, 0]);
        }
        ppu.w(0xFF40, 0x80 | 0x10 | 0x02 | 0x01);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!(f[12], S[2], "OAM index 0 wins the X tie, OBP1 maps index 1 to shade 2");
        assert_eq!((f[32], f[39]), (S[0], S[1]), "X flip");
        assert_eq!(f[20 * WIDTH + 92], S[1]);
        assert_eq!(f[20 * WIDTH + 142], S[0], "10-per-line limit");
    }

    #[test]
    fn tall_sprites_and_bg_priority() {
        let mut ppu = Ppu::post_boot_dmg();
        ppu.w(0xFF47, 0xE4);
        ppu.w(0xFF48, 0xE4);
        solid_tile(&mut ppu, 6, 2);
        solid_tile(&mut ppu, 7, 3);
        solid_tile(&mut ppu, 1, 1);
        ppu.vram[0x1800 + 32 * 2 + 2] = 1; // BG tile at (16..24, 16..24) colour 1
        ppu.oam[0..4].copy_from_slice(&[16, 8, 7, 0]); // 8x16: tile 7 & 0xFE = 6 on top
        ppu.oam[4..8].copy_from_slice(&[32, 24, 6, 0xC0]); // behind BG, Y-flipped
        ppu.w(0xFF40, 0x80 | 0x10 | 0x04 | 0x02 | 0x01);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!((f[0], f[8 * WIDTH]), (S[2], S[3]), "top half tile 6, bottom tile 7");
        assert_eq!(f[16 * WIDTH + 16], S[1], "behind non-zero BG");
        assert_eq!(f[24 * WIDTH + 16], S[2], "Y flip puts tile 6 at the bottom, shown over BG colour 0");
    }

    /// Rows of mooneye's intr_2_mode0_timing_sprites: the M-cycles mode 3 grows by with
    /// objects at these X, `(dots - 172) / 4`, SCX 0.
    #[test]
    fn mode_3_grows_as_mooneyes_sprite_table_says() {
        let rows: &[(u16, &[u8])] = &[
            (2, &[0]), (4, &[0, 0]), (16, &[0; 10]), (16, &[1; 10]), (15, &[2; 10]), (16, &[8; 10]), (15, &[167; 10]), (0, &[168; 10]),
            (17, &[0, 0, 0, 0, 0, 160, 160, 160, 160, 160]), (16, &[2, 2, 2, 2, 2, 162, 162, 162, 162, 162]), (15, &[4, 4, 4, 4, 4, 164, 164, 164, 164, 164]),
            (1, &[4]), (2, &[16]), (1, &[167]), (5, &[0, 8]), (4, &[2, 10]), (3, &[4, 12]),
            (27, &[0, 8, 16, 24, 32, 40, 48, 56, 64, 72]), (25, &[1, 9, 17, 25, 33, 41, 49, 57, 65, 73]), (15, &[5, 13, 21, 29, 37, 45, 53, 61, 69, 77]),
            (27, &[72, 64, 56, 48, 40, 32, 24, 16, 8, 0]),
        ];
        for &(extra, xs) in rows {
            let mut ppu = Ppu::post_boot_dmg();
            ppu.w(0xFF40, 0x80 | 0x02);
            for (i, &x) in xs.iter().enumerate() {
                ppu.oam[i * 4..i * 4 + 2].copy_from_slice(&[0x52, x]);
            }
            assert_eq!((ppu.mode3_dots(0x42) - 172) / 4, extra, "objects at {xs:?}");
            for mode in [Mode::Dmg, Mode::Cgb] {
                let objects = fifo::scan(&ppu.oam, ppu.lcdc, 0x42);
                let start = fifo::Start { ly: 0x42, mode, double: false, blocked: false, display: ppu.display(), window: fifo::Window::default(), vram: &ppu.vram, oam: &ppu.oam, objects: &objects };
                assert_eq!((fifo::draw(&start, &[]).dots - 172) / 4, extra, "the FIFO on {mode:?}, objects at {xs:?}");
            }
        }
    }

    #[test]
    fn dmg_ignores_cgb_registers() {
        let mut ppu = Ppu::post_boot_dmg();
        ppu.w(0xFF4F, 1);
        ppu.vram_write(0x8000, 0x12);
        assert_eq!((ppu.r(0xFF4F), ppu.vram[0], ppu.vram[0x2000]), (0xFF, 0x12, 0x00));
    }

    #[test]
    fn vram_bank_select() {
        let mut ppu = Ppu::post_boot(true);
        assert_eq!(ppu.r(0xFF4F), 0xFE);
        ppu.vram_write(0x9800, 0x11);
        ppu.w(0xFF4F, 0xFF);
        assert_eq!(ppu.r(0xFF4F), 0xFF, "only bit 0 is kept; others read 1");
        assert_eq!(ppu.vram_read(0x9800), 0x00);
        ppu.vram_write(0x9800, 0x22);
        ppu.w(0xFF4F, 0x00);
        assert_eq!(ppu.vram_read(0x9800), 0x11);
        assert_eq!((ppu.vram[0x1800], ppu.vram[0x3800]), (0x11, 0x22));
    }

    #[test]
    fn the_palette_lookup_gives_the_final_colour() {
        let mut ppu = Ppu::post_boot(true);
        assert_eq!(colour(ppu.palettes(), true, 0), 0x7FFF, "the CGB boot ROM leaves palettes white");
        ppu.display_write(7 * 8 + 3 * 2, 0x55);
        ppu.display_write(7 * 8 + 3 * 2 + 1, 0x80);
        assert_eq!(colour(ppu.palettes(), true, 3 | 7 << 2), 0x0055, "bit 15 is not a colour bit");
        assert_eq!(rgb(true, 0x0055), 0xAD1000, "0x0055 is red 21, green 2, blue 0");
        let mut dmg = Ppu::post_boot_dmg();
        dmg.w(0xFF47, 0b11_10_01_00);
        dmg.w(0xFF49, 0b00_01_10_11);
        assert_eq!((dmg.r(0xFF47), dmg.r(0xFF48)), (0xE4, 0xFF));
        assert_eq!([2, 2 | 0x20 | 1 << 2].map(|px| colour(dmg.palettes(), false, px)), [2, 1], "BGP for the background, OBP1 for an object asking for it");
    }

    impl Ppu {
        fn w(&mut self, addr: u16, value: u8) {
            self.write(addr, value);
        }

        fn r(&self, addr: u16) -> u8 {
            self.read(addr)
        }
    }

    /// CGB palette `p` colour `c` set to RGB555 `v`, as the control logic hands the
    /// BCPD/OCPD bytes over.
    fn set_color(ppu: &mut Ppu, obj: bool, p: u8, c: u8, v: u16) {
        let i = obj as u8 * 64 + p * 8 + c * 2;
        ppu.display_write(i, v as u8);
        ppu.display_write(i + 1, (v >> 8) as u8);
    }

    #[test]
    fn cgb_bg_attributes_select_palette_bank_and_flip() {
        let mut ppu = Ppu::post_boot(true);
        ppu.w(0xFF40, 0x00);
        set_color(&mut ppu, false, 2, 1, 0x001F); // red
        set_color(&mut ppu, false, 2, 3, 0x03E0); // green
        set_color(&mut ppu, false, 5, 1, 0x7C00); // blue
        // Tile 1 in bank 0: colour 1. Tile 1 in bank 1: top-left pixel colour 3, rest 1.
        solid_tile_in(&mut ppu, 0, 1, 1);
        solid_tile_in(&mut ppu, 1, 1, 1);
        ppu.vram[0x2000 + 16] = 0xFF;
        ppu.vram[0x2000 + 17] = 0x80;
        ppu.vram[0x1800..0x1803].fill(1);
        ppu.vram[0x3800] = 0x05; // palette 5, bank 0
        ppu.vram[0x3801] = 0x0A; // palette 2, bank 1
        ppu.vram[0x3802] = 0x6A; // palette 2, bank 1, X and Y flip
        ppu.w(0xFF40, 0x91);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!(f[0], 0x0000FF, "palette 5 colour 1");
        assert_eq!((f[8], f[9]), (0x00FF00, 0xFF0000), "bank-1 tile, palette 2");
        assert_eq!((f[16], f[7 * WIDTH + 23]), (0xFF0000, 0x00FF00), "flipped: the colour-3 pixel moves to the bottom right");
        assert_eq!(f[24], 0xFFFFFF, "tile 0 colour 0, palette 0 still white");
    }

    #[test]
    fn cgb_object_palette_bank_and_oam_order_priority() {
        let mut ppu = Ppu::post_boot(true);
        ppu.w(0xFF40, 0x00);
        set_color(&mut ppu, true, 3, 1, 0x001F);
        set_color(&mut ppu, true, 6, 1, 0x7C00);
        solid_tile_in(&mut ppu, 1, 4, 1);
        solid_tile_in(&mut ppu, 0, 4, 1);
        // OAM 0 at x=24 palette 3 from bank 1; OAM 1 at x=20 palette 6 overlaps on its left.
        ppu.oam[0..4].copy_from_slice(&[16, 24, 4, 0x0B]);
        ppu.oam[4..8].copy_from_slice(&[16, 20, 4, 0x06]);
        ppu.w(0xFF40, 0x93);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!(f[12], 0x0000FF, "only OAM 1 covers x=12");
        assert_eq!(f[16], 0xFF0000, "OAM 0 wins the overlap despite its larger X");
        ppu.oam[0..4].copy_from_slice(&[16, 24, 4, 0x03]);
        render_frame(&mut ppu);
        assert_eq!(ppu.frame()[16], 0xFF0000, "bank 0 tile 4 is also colour 1");
    }

    #[test]
    fn cgb_bg_priority_and_master_priority() {
        let mut ppu = Ppu::post_boot(true);
        ppu.w(0xFF40, 0x00);
        set_color(&mut ppu, false, 0, 1, 0x03E0); // BG green
        set_color(&mut ppu, true, 0, 1, 0x001F); // OBJ red
        solid_tile(&mut ppu, 1, 1);
        solid_tile(&mut ppu, 2, 1);
        ppu.vram[0x1800] = 1; // x 0-7 BG colour 1, no attr priority
        ppu.vram[0x1801] = 1;
        ppu.vram[0x3801] = 0x80; // x 8-15 BG colour 1 with BG-priority attribute
        // x 16-23 BG colour 0 (tile 0). Line 8 has BG colour 1 at x 0-15.
        ppu.vram[0x1820] = 1;
        ppu.vram[0x1821] = 1;
        ppu.oam[0..4].copy_from_slice(&[16, 8, 2, 0x00]);
        ppu.oam[4..8].copy_from_slice(&[16, 16, 2, 0x00]);
        ppu.oam[8..12].copy_from_slice(&[16, 24, 2, 0x80]);
        ppu.oam[12..16].copy_from_slice(&[24, 8, 2, 0x80]); // line 8: OAM priority over BG colour 1
        ppu.w(0xFF40, 0x93);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!(f[0], 0xFF0000, "object over BG colour 1");
        assert_eq!(f[8], 0x00FF00, "map attribute bit 7 puts BG colour 1 on top");
        assert_eq!(f[16], 0xFF0000, "BG colour 0 never covers an object");
        assert_eq!(f[8 * WIDTH], 0x00FF00, "OAM bit 7 puts BG colour 1 on top");
        ppu.w(0xFF40, 0x92);
        render_frame(&mut ppu);
        let f = ppu.frame();
        assert_eq!((f[8], f[8 * WIDTH]), (0xFF0000, 0xFF0000), "LCDC.0 clear: objects always win");
        assert_eq!(f[8 * WIDTH + 8], 0x00FF00, "and the BG still draws");
    }

    /// The pixel FIFO against the line renderer it replaced, on random lines with no write
    /// during mode 3: the colours, the window's state after the line and the dots to mode 0
    /// (SCX's fine scroll rounded as `fifo::mode0_dots` rounds it).
    #[test]
    fn the_fifo_draws_a_quiet_line_as_the_line_renderer_did() {
        let mut rng = crate::memory::Rng(0x9E37_79B9_7F4A_7C15);
        let mut differ = std::collections::BTreeMap::<String, (u32, String)>::new();
        let cases = 3000;
        for case in 0..cases {
            let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][case % 3];
            let mut ppu = if mode == Mode::Compat { Ppu::post_boot_compat() } else { Ppu::post_boot(mode != Mode::Dmg) };
            let mut byte = || rng.draw() as u8;
            for b in ppu.vram.iter_mut() { *b = byte(); }
            for i in 0..40 {
                ppu.oam[i * 4] = 16 + byte() % 60;
                ppu.oam[i * 4 + 1] = byte() % 176;
                ppu.oam[i * 4 + 2] = byte();
                ppu.oam[i * 4 + 3] = byte();
            }
            for i in 0..PAL_REGS as usize { ppu.palettes[i] = byte(); }
            if mode != Mode::Cgb { ppu.vram[0x2000..].fill(0); }
            ppu.lcdc = 0x80 | byte() & 0x7F;
            ppu.scx = byte();
            ppu.scy = byte();
            ppu.wx = byte() % 176;
            ppu.wy = byte() % 70;
            let ly = ppu.wy.wrapping_add(1 + byte() % 50) % 70;
            ppu.window = fifo::Window { rows: byte() % 40, armed: byte() & 1 != 0, carried: false };
            let px = ppu.reference_line(ly);
            let colours = px.map(|p| colour_in(&ppu.palettes, mode, p));
            let shows = (mode.cgb_features() || ppu.lcdc & 1 != 0) && ppu.lcdc & 0x20 != 0 && ppu.window.armed && ppu.wx <= 166;
            let want = (colours, ppu.window.rows + shows as u8, ppu.mode3_dots(ly));
            let objects = fifo::scan(&ppu.oam, ppu.lcdc, ly);
            let start = fifo::Start { ly, mode, double: false, blocked: false, display: ppu.display(), window: ppu.window, vram: &ppu.vram, oam: &ppu.oam, objects: &objects };
            let got = fifo::draw(&start, &[]);
            // SameBoy behaviours the line renderer lacked, each checked by a test ROM:
            // a DMG's window starts with LCDC.0 clear (white, its rows still counted); a
            // CGB fetches objects with OBJ_EN clear; WX below 7 and at 166 start the window
            // off the pixel grid; a DMG armed with the window off inserts a pixel at WX. With
            // objects under a fine scroll the line renderer's penalty (Pan Docs) and SameBoy's
            // fetcher give different lengths, as they do once the window resets the fetcher's
            // tile grid at WX; mooneye's sprite table, at SCX 0 with no window, holds both.
            let (cgb_hw, win_en, armed) = (mode.rgb555(), ppu.lcdc & 0x20 != 0, ppu.window.armed);
            let window_glitch = armed && (!win_en && !cgb_hw || win_en && (ppu.wx < 7 || ppu.wx == 166) || win_en && !mode.cgb_features() && ppu.lcdc & 1 == 0);
            let length_differs = !objects.is_empty() && (cgb_hw && ppu.lcdc & 2 == 0 || ppu.lcdc & 2 != 0 && (ppu.scx & 7 != 0 || shows));
            if window_glitch {
                continue;
            }
            let state = format!("case {case} {mode:?} ly {ly} lcdc {:02x} scx {} scy {} wx {} wy {} window {:?} objects {:?}", ppu.lcdc, ppu.scx, ppu.scy, ppu.wx, ppu.wy, ppu.window, objects.iter().map(|&i| ppu.oam[i as usize * 4 + 1]).collect::<Vec<_>>());
            let key = format!("{mode:?} window {} objects {}", shows as u8, ppu.lcdc >> 1 & 1);
            let mut note = |what: &'static str, detail: String| { let e = differ.entry(format!("{what}: {key}")).or_insert((0, format!("{state}: {detail}"))); e.0 += 1; };
            if let Some(x) = (0..WIDTH).find(|&x| got.colours[x] != want.0[x]) {
                note("colours", format!("x {x}: {:04x}, want {:04x}", got.colours[x], want.0[x]));
            }
            if got.window.rows != want.1 { note("window rows", format!("{} want {}", got.window.rows, want.1)); }
            // The line renderer rounded SCX's fine scroll up to 4 dots on every console (A3); the
            // FIFO's dots stay unrounded, so they are rounded the same way here.
            let dots = fifo::mode0_dots(got.dots, ppu.scx, false);
            if dots != want.2 && !length_differs { note("length", format!("{dots} dots to mode 0, want {}", want.2)); }
        }
        assert!(differ.is_empty(), "of {cases} lines: {differ:#?}");
    }
}

/// Which access a DMG's OAM corruption bug follows: a write, or an M-cycle that puts a
/// FE page address on the bus without an access (SameBoy `GB_trigger_oam_bug`), or a read
/// (`GB_trigger_oam_bug_read`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OamBug {
    Write,
    Read,
}

/// The DMG's OAM corruption bug on the 8-byte row at byte `row` of OAM, the row the PPU's
/// OAM scan reads this M-cycle (SameBoy v1.0.5 `memory.c`, the DMG patterns; blargg
/// `oam_bug`). Rows below 8 are never corrupted. Words are little-endian pairs of bytes.
pub fn corrupt_oam(oam: &mut [u8; 0xA0], row: usize, bug: OamBug) {
    if row < 8 || row >= 0xA0 {
        return;
    }
    let w = |oam: &[u8; 0xA0], at: usize| u16::from_le_bytes([oam[at], oam[at + 1]]);
    let set = |oam: &mut [u8; 0xA0], at: usize, v: u16| oam[at..at + 2].copy_from_slice(&v.to_le_bytes());
    // The word `k` words from the row's start, k negative for the rows before.
    let at = |k: isize| (row as isize + 2 * k) as usize;
    let copy_row = |oam: &mut [u8; 0xA0], to: usize, from: usize| oam.copy_within(from..from + 8, to);
    match bug {
        OamBug::Write => {
            let (a, b, c) = (w(oam, at(0)), w(oam, at(-4)), w(oam, at(-2)));
            set(oam, at(0), ((a ^ c) & (b ^ c)) ^ c);
            oam.copy_within(row - 6..row, row + 2);
        }
        OamBug::Read => {
            if row & 0x18 == 0x10 {
                if row < 0x98 {
                    let (a, b, c, d) = (w(oam, at(-8)), w(oam, at(-4)), w(oam, at(0)), w(oam, at(-2)));
                    set(oam, at(-4), (b & (a | c | d)) | (a & c & d));
                    copy_row(oam, row - 0x10, row - 0x08);
                }
            } else if row & 0x18 == 0x00 {
                if row < 0x98 {
                    if row == 0x40 {
                        let (c, d, e, f, g, h) = (w(oam, at(-2)), w(oam, at(-3)), w(oam, at(-4)), w(oam, at(-7)), w(oam, at(-8)), w(oam, at(-16)));
                        let b = w(oam, at(0));
                        set(oam, at(-4), (e & (h | g | (!d & f) | c | b)) | (c & g & h));
                    } else {
                        let (a, b, c, d, e) = (w(oam, at(0)), w(oam, at(-2)), w(oam, at(-4)), w(oam, at(-8)), w(oam, at(-16)));
                        let v = match row {
                            0x20 => (c & (a | b | d | e)) | (a & b & d & e),
                            0x60 => (c & (a | b | d | e)) | (b & d & e),
                            _ => c | (a & b & d & e),
                        };
                        set(oam, at(-4), v);
                    }
                    copy_row(oam, row - 0x10, row - 0x08);
                    copy_row(oam, row - 0x20, row - 0x08);
                }
            } else {
                let (a, b, c) = (w(oam, at(0)), w(oam, at(-4)), w(oam, at(-2)));
                let v = b | (a & c);
                set(oam, at(-4), v);
                set(oam, at(0), v);
            }
            copy_row(oam, row, row - 8);
            if row == 0x80 {
                copy_row(oam, 0, row);
            }
        }
    }
}
