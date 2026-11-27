//! One visible line's mode 3 as the hardware draws it: a background and window fetcher
//! feeding an 8-pixel FIFO, objects fetched as the line reaches their X, and every display
//! register read at the dot the hardware reads it. A line is a pure function of its
//! start (the display bytes, the window's state, VRAM, the objects the OAM scan chose) and
//! the display writes the GBC CPU made during it, each at its time ([`Write`]). Its result
//! is the 160 colours, the window's state after the line and mode 3's length.
//!
//! The line is a state machine ([`State`]) stepped one pass at a time ([`State::step`]):
//! a pass is one dot of a wait, one change a pending write makes, or one stretch of the
//! fetcher's code between waits.
//!
//! The model is SameBoy v1.0.5's (`Core/display.c`, `GB_display_run` from `mode_3_start`
//! to the end of mode 3, `advance_fetcher_state_machine`, `render_pixel_if_possible`) and
//! its write timing (`Core/sm83_cpu.c`, the conflict maps of `cycle_write`), for the
//! consoles the census runs: DMG-CPU (SameBoy's DMG family) and CPU CGB E, in CGB mode
//! or in DMG compatibility mode. Pan Docs' "Pixel FIFO" describes the same fetcher.

use super::ppu::{colour_in, Colour, Display, Mode, WIDTH};
use crate::memory::layout::{PAL_DMG, PAL_LEN, PAL_REGS};

#[cfg(test)]
mod oracle;

/// A time within a line, in half-dots from the M-cycle mode 3 begins in (STAT's mode 3):
/// a dot is 2, a single-speed M-cycle 8 and a double-speed one 4. SameBoy's display clock
/// runs at 8 MHz, and its double-speed access conflicts land on half-dots.
pub type Time = u32;

/// A display byte the GBC CPU wrote during mode 3: its index in [`Display`] and the time
/// its write cycle ends ([`Time`]), where an access whose M-cycle ends there lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Write {
    pub at: Time,
    pub index: u8,
    pub value: u8,
}

/// The window's state between lines (SameBoy's `window_y`, `wy_triggered` and the DMG's
/// WX = 166 carry-over).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Window {
    /// Window rows started this frame: the next row the window draws is `rows`.
    pub rows: u8,
    /// LY matched WY at the start of some line this frame.
    pub armed: bool,
    /// A DMG line ended with WX at 166: the next line starts in the window at tile 1.
    pub carried: bool,
}

impl Window {
    /// The window as line `ly` starts it from `self`, the state after the line before: the
    /// frame starts over at line 0, and the window arms where LY equals WY. The PPU's
    /// reference does this at VBlank and at the line's start; the scanline circuit and the
    /// stepper's start pass compute it from LY and their own WY.
    pub fn starting(self, ly: u8, wy: u8) -> Window {
        let w = if ly == 0 { Window::default() } else { self };
        Window { armed: w.armed || ly == wy, ..w }
    }
}

/// What a line starts from. `objects` are the OAM indices the scan chose, at most ten.
pub struct Start<'a> {
    pub ly: u8,
    pub mode: Mode,
    /// The GBC CPU runs at double speed: its accesses' conflicts land on half-dots.
    pub double: bool,
    /// Ordinary STOP keeps a CGB's PPU from VRAM: the line draws black.
    pub blocked: bool,
    pub display: Display,
    pub window: Window,
    pub vram: &'a [u8; 0x4000],
    pub oam: &'a [u8; 0xA0],
    pub objects: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drawn {
    pub colours: [Colour; WIDTH],
    pub window: Window,
    /// Mode 3's length in dots.
    pub dots: u16,
}

const LCDC: u8 = PAL_REGS as u8;
const SCY: u8 = LCDC + 1;
const SCX: u8 = LCDC + 2;
const WY: u8 = LCDC + 3;
const WX: u8 = LCDC + 4;
const BGP: u8 = PAL_DMG as u8;

const BG_EN: u8 = 0x01;
const OBJ_EN: u8 = 0x02;
const OBJ_SIZE: u8 = 0x04;
const BG_MAP: u8 = 0x08;
const TILE_SEL: u8 = 0x10;
const WIN_EN: u8 = 0x20;
const WIN_MAP: u8 = 0x40;

/// The display registers a line keeps ([`State::regs`]): BGP, OBP0, OBP1 and then
/// LCDC, SCY, SCX, WY and WX, the display bytes from `layout::PAL_DMG` on.
pub const REGS: usize = PAL_LEN - PAL_DMG as usize;

/// CGB palette RAM, the display bytes below `PAL_DMG`. The GBC CPU cannot write it during
/// mode 3, so a line reads it as a constant, as it reads VRAM and OAM.
pub type PaletteRam = [u8; PAL_DMG as usize];

/// The most objects the OAM scan chooses for a line.
pub const OBJECTS: usize = 10;

/// The OAM scan of mode 2 (SameBoy `add_object_from_index`): the first ten objects whose
/// rows cover `ly`, in the order the line fetches them, by X and then by OAM index.
pub fn scan(oam: &[u8; 0xA0], lcdc: u8, ly: u8) -> Vec<u8> {
    let height: i16 = if lcdc & OBJ_SIZE != 0 { 16 } else { 8 };
    let mut chosen: Vec<u8> = (0..40u8).filter(|&i| (0..height).contains(&(ly as i16 - (oam[i as usize * 4] as i16 - 16)))).take(OBJECTS).collect();
    chosen.sort_by_key(|&i| oam[i as usize * 4 + 1]);
    chosen
}

/// Draws a line.
pub fn draw(start: &Start, writes: &[Write]) -> Drawn {
    let mut colours = [0; WIDTH];
    let palettes = palettes(&start.display);
    let state = drive(State::new(start), writes, |state, pending| {
        let out = state.step(start.vram, start.oam, palettes, pending);
        if let Some((x, colour)) = out.pixel {
            colours[x as usize] = colour;
        }
        out
    });
    Drawn { colours, window: state.window_after(), dots: state.dots() }
}

/// The CGB palette RAM bytes of the display bytes.
pub fn palettes(display: &Display) -> &PaletteRam {
    display[..PAL_DMG as usize].try_into().expect("palette RAM leads the display bytes")
}

/// One pass of a line: the state before it, the write it was handed, what it did and the
/// state after it.
#[derive(Clone, Copy, Debug)]
pub struct Pass {
    pub before: State,
    pub pending: Option<Pending>,
    pub out: Out,
    pub after: State,
}

/// Every pass of a line.
pub fn trace(start: &Start, writes: &[Write]) -> Vec<Pass> {
    let mut passes = Vec::new();
    let palettes = palettes(&start.display);
    drive(State::new(start), writes, |state, pending| {
        let before = *state;
        let out = state.step(start.vram, start.oam, palettes, pending);
        passes.push(Pass { before, pending, out, after: *state });
        out
    });
    passes
}

/// Steps a line from `state` to its end, `pass` running each pass. The writes are a queue:
/// each pass is handed the head write, which leaves the queue when the pass says its last
/// change applied ([`Out::finished`]), so the driver needs no conflict map. That is time
/// order because the GBC CPU's writes are at least an M-cycle apart, and a write's changes
/// span less than one (`drive_is_time_order` in the tests); the assert holds the spacing.
/// The GBC CPU cannot write palette RAM during mode 3, so a line's writes are to the
/// display registers.
pub fn drive(mut state: State, writes: &[Write], mut pass: impl FnMut(&mut State, Option<Pending>) -> Out) -> State {
    assert!(writes.iter().all(|w| (BGP..BGP + REGS as u8).contains(&w.index)), "a write during mode 3 to a display byte that is not a register: {writes:?}");
    let mcycle = if state.double { 4 } else { 8 };
    assert!(writes.windows(2).all(|p| p[1].at >= p[0].at + mcycle), "display writes less than an M-cycle apart: {writes:?}");
    let (mut head, mut applied) = (0, 0);
    loop {
        let out = pass(&mut state, writes.get(head).map(|&write| Pending { write, applied }));
        if out.applied {
            applied += 1;
            if out.finished {
                (head, applied) = (head + 1, 0);
            }
        }
        if out.done {
            return state;
        }
    }
}

/// Whether display write `w` during the line `start` begins can change what the line draws,
/// its window state or its length, judged against the line's length with no write,
/// `quiet_dots`. A write cannot when it targets WY (read only at the line's start), BGP,
/// OBP0 or OBP1 in CGB mode (which draws from palette RAM), or carries the value the line
/// started with, but for WX on the DMG, where any write marks the WX comparator; or when it
/// lands at or past `2·quiet_dots + REACH`, after the line's last pixel. The PPU
/// checkpoint's length stream computes this per write
/// (`ppu_circuit::circuit::build_length`), and the stepper draws a line any write marks.
pub fn judge(start: &Start, quiet_dots: u16, w: Write) -> bool {
    let late = w.at >= quiet_dots as Time * 2 + REACH;
    let same = w.value == start.display[w.index as usize] && !(start.mode == Mode::Dmg && w.index == WX);
    let unread = w.index == WY || start.mode.cgb_features() && (BGP..BGP + 3).contains(&w.index);
    !(late || same || unread || w.index < BGP)
}

/// Fetcher steps, each one dot (SameBoy `fetcher_step_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Step {
    TileT1,
    TileT2,
    LowT1,
    LowT2,
    HighT1,
    HighT2,
    Push,
}

/// Where the line is in SameBoy's `GB_display_run`: a wait, or the code that runs after one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// `State::dots` dots to wait, then `State::resume`.
    Sleep,
    Init,
    Window,
    /// The window starts after the DMG's extra dot at WX 0 under a fine scroll.
    WindowOn,
    /// The next object, if it is at this X; `first` drops the objects already passed.
    Match { first: bool },
    Render,
    Advance,
    FetchWait,
    FetchA,
    FetchB,
    FetchC,
    FetchD,
    FetchE,
    Done,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Item {
    pub pixel: u8,
    pub palette: u8,
    pub priority: u8,
    pub bg_priority: bool,
}

/// The background FIFO. It only ever holds the tail of one row, or the single pixel a DMG
/// inserts (SameBoy issue 278), so it is the row's two bit planes shifted on each pop,
/// next pixel in bit 7.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BgFifo {
    pub low: u8,
    pub high: u8,
    pub palette: u8,
    pub bg_priority: bool,
    pub size: u8,
}

impl BgFifo {
    fn pop(&mut self) -> Item {
        let pixel = self.low >> 7 | (self.high >> 7) << 1;
        self.low <<= 1;
        self.high <<= 1;
        self.size -= 1;
        Item { pixel, palette: self.palette, priority: 0, bg_priority: self.bg_priority }
    }

    fn push_row(&mut self, low: u8, high: u8, palette: u8, bg_priority: bool, flip: bool) {
        assert_eq!(self.size, 0, "a background row pushed into a FIFO holding pixels");
        let (low, high) = if flip { (low.reverse_bits(), high.reverse_bits()) } else { (low, high) };
        *self = BgFifo { low, high, palette, bg_priority, size: 8 };
    }
}

/// The object FIFO, next pixel first. The slots past the pixels it holds are always
/// [`Item::default`], so a pop from an empty FIFO is a transparent pixel and it needs no size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjFifo {
    pub items: [Item; 8],
}

impl ObjFifo {
    fn pop(&mut self) -> Item {
        let item = self.items[0];
        self.items.copy_within(1.., 0);
        self.items[7] = Item::default();
        item
    }

    fn overlay_row(&mut self, lower: u8, upper: u8, palette: u8, bg_priority: bool, priority: u8, flip: bool) {
        for (i, target) in self.items.iter_mut().enumerate() {
            let bit = if flip { i } else { 7 - i };
            let pixel = (lower >> bit & 1) | (upper >> bit & 1) << 1;
            if pixel != 0 && (target.pixel == 0 || target.priority > priority) {
                *target = Item { pixel, palette, priority, bg_priority };
            }
        }
    }
}

/// A change the write timing makes to the line's registers at a time (SameBoy's
/// `cycle_write` conflict cases).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Set(u8, u8),
    /// A DMG palette write's first half-dot: the new value ORed with the old.
    Or(u8, u8),
    /// A DMG LCDC write's first stage (`GB_CONFLICT_DMG_LCDC`).
    DmgLcdc(u8),
    /// A DMG LCDC write's second stage: the new value.
    DmgLcdcFinal(u8),
    /// A single-speed CGB LCDC write: clearing TILE_SEL raises the tile-select glitch for a
    /// cycle (`GB_CONFLICT_LCDC_CGB`).
    CgbLcdc(u8),
    /// A double-speed CGB LCDC write's first stage (`GB_CONFLICT_LCDC_CGB_DOUBLE`).
    CgbDoubleLcdc(u8),
    WxJustChanged(bool),
    TileSelGlitch(bool),
}

/// The most half-dots before its time a write changes a register: 2 T-cycles at single speed.
pub const EARLIEST: Time = 4;

/// A write landing at or past `2·dots + REACH` changes nothing on a line of `dots`: its
/// earliest change, [`EARLIEST`] before it, comes after the last pixel's code has run
/// (`a_write_past_the_end_changes_nothing`). Up to `2·dots + EARLIEST` it can still reach
/// the last pixel's palette.
pub const REACH: Time = EARLIEST + 1;

/// The dots from mode 3's start to mode 0's on a line of `dots` that began with SCX's fine
/// scroll `fine`. On a DMG the fine scroll ends mode 3 at the next M-cycle's boundary, 1 to
/// 4 pixels an M-cycle later and 5 to 7 two (mooneye `hblank_ly_scx_timing-GS`), so it
/// counts rounded up to 4; the rule is access-timing's on the line renderer's closed form,
/// whose dots a quiet line's FIFO equals. CGB hardware (`cgb`, in either mode) ends mode 3
/// at the dot (`lcd::Lcd`).
pub fn mode0_dots(dots: u16, fine: u8, cgb: bool) -> u16 {
    if cgb {
        return dots;
    }
    let f = (fine & 7) as u16;
    dots - f + f.div_ceil(4) * 4
}

/// The line's length as the PPU publishes it to the control logic: [`mode0_dots`] less
/// `lcd::LENGTH_BASE`.
pub fn length(dots: u16, fine: u8, cgb: bool) -> u8 {
    let d = mode0_dots(dots, fine, cgb);
    u8::try_from(d - super::lcd::LENGTH_BASE).unwrap_or_else(|_| panic!("a mode 3 of {d} dots"))
}

/// A write's change number `n` and when it lands, or none past its last (SameBoy's conflict
/// maps for the DMG, the CGB and the CGB at double speed; a CPU T-cycle is 2 half-dots, or
/// 1 at double speed). A write's changes come in time order.
pub fn act(cgb: bool, double: bool, w: Write, n: u8) -> Option<(Time, Act)> {
    let (i, v) = (w.index, w.value);
    let acts: &[(i64, Act)] = match (cgb, double, i) {
        (false, _, LCDC) => &[(-2, Act::DmgLcdc(v)), (-1, Act::DmgLcdcFinal(v))],
        // SameBoy reads the new SCY a T-cycle early (`GB_CONFLICT_READ_NEW`); Mealybug's notes
        // ("SCY", the-comprehensive-game-boy-ppu-documentation.md) have the DMG take it
        // 2 T-cycles before the CGB does, and m3_scy_change on DMG agrees.
        (false, _, SCY) => &[(-2, Act::Set(i, v))],
        (false, _, BGP..=0x82) => &[(-2, Act::Or(i, v)), (-1, Act::Set(i, v))],
        (false, _, WX) => &[(0, Act::Set(i, v)), (0, Act::WxJustChanged(true)), (1, Act::WxJustChanged(false))],
        (false, _, SCX) => &[(-2, Act::Set(i, v))],
        (true, false, LCDC) => &[(0, Act::CgbLcdc(v)), (1, Act::TileSelGlitch(false))],
        (true, false, BGP..=0x82) => &[(-2, Act::Set(i, v))],
        (true, false, WX) => &[(1, Act::Set(i, v))],
        (true, true, LCDC) => &[(-2, Act::CgbDoubleLcdc(v)), (0, Act::TileSelGlitch(false)), (0, Act::Set(i, v))],
        (true, true, SCX) => &[(-2, Act::Set(i, v))],
        _ => &[(0, Act::Set(i, v))],
    };
    let tc: i64 = if double { 1 } else { 2 };
    acts.get(n as usize).map(|&(cycles, act)| ((w.at as i64 + cycles * tc).max(0) as Time, act))
}

/// A write a pass may apply the next change of, and how many of its changes ([`act`]) the
/// line has applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pending {
    pub write: Write,
    pub applied: u8,
}

/// What one pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Out {
    /// It applied the pending write's next change.
    pub applied: bool,
    /// That change was the write's last.
    pub finished: bool,
    /// A pixel left the FIFO for the screen: its x and colour.
    pub pixel: Option<(u8, Colour)>,
    /// The line's last pixel is out.
    pub done: bool,
}

/// A line between passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub ly: u8,
    pub mode: Mode,
    pub double: bool,
    pub blocked: bool,
    /// The display registers BGP, OBP0, OBP1, LCDC, SCY, SCX, WY and WX, display indices
    /// `PAL_DMG` on.
    pub regs: [u8; REGS],
    pub t: Time,
    pub phase: Phase,
    /// The phase a [`Phase::Sleep`] resumes at.
    pub resume: Phase,
    /// Dots left in a [`Phase::Sleep`].
    pub dots: u8,
    pub lcd_x: u8,
    /// SameBoy's `position_in_line`: -16 at mode 3's start, 0 at the first visible pixel.
    pub pos: u8,
    pub bg: BgFifo,
    pub obj: ObjFifo,
    pub step: Step,
    pub tile: u8,
    pub attr: u8,
    pub data: [u8; 2],
    pub tile_index_address: u16,
    pub tile_data_address: u16,
    pub fetcher_y: u8,
    pub last_tileset: bool,
    pub data_for_sel_glitch: u8,
    pub tile_sel_glitch: bool,
    /// The objects the scan chose that are not yet fetched or passed, in fetch order from
    /// the next: OAM index and X. The slots past `count` are `(0, 0)`.
    pub objects: [(u8, u8); OBJECTS],
    pub count: u8,
    /// The fetched object row's bit planes.
    pub obj_low: u8,
    pub obj_high: u8,
    pub during_object_fetch: bool,
    pub window_y: u8,
    pub window_tile_x: u8,
    pub wy_triggered: bool,
    pub wx_triggered: bool,
    pub window_is_being_fetched: bool,
    pub insert_bg_pixel: bool,
    pub line_has_fractional_scrolling: bool,
    pub cgb_wx_glitch: bool,
    pub wx_just_changed: bool,
    pub disable_window_pixel_insertion_glitch: bool,
    /// SCX's fine scroll as the line began, which places mode 3's end ([`mode0_dots`]).
    pub fine: u8,
}

impl State {
    pub fn new(s: &Start) -> Self {
        assert!(s.objects.len() <= OBJECTS, "the OAM scan chose {} objects", s.objects.len());
        let mut objects = [(0, 0); OBJECTS];
        for (slot, &i) in objects.iter_mut().zip(s.objects) {
            *slot = (i, s.oam[i as usize * 4 + 1]);
        }
        State {
            ly: s.ly,
            mode: s.mode,
            double: s.double,
            blocked: s.blocked,
            regs: s.display[PAL_DMG as usize..].try_into().expect("the display registers end the display bytes"),
            t: 0,
            phase: Phase::Sleep,
            resume: Phase::Init,
            dots: 5,
            lcd_x: 0,
            pos: (-16i8) as u8,
            bg: BgFifo::default(),
            obj: ObjFifo::default(),
            step: Step::TileT1,
            tile: 0,
            attr: 0,
            data: [0; 2],
            tile_index_address: 0,
            tile_data_address: 0,
            fetcher_y: 0,
            last_tileset: false,
            data_for_sel_glitch: 0,
            tile_sel_glitch: false,
            objects,
            count: s.objects.len() as u8,
            obj_low: 0,
            obj_high: 0,
            during_object_fetch: false,
            window_y: s.window.rows.wrapping_sub(1),
            window_tile_x: if s.window.carried { 1 } else { 0 },
            wy_triggered: s.window.armed,
            wx_triggered: s.window.carried,
            window_is_being_fetched: false,
            insert_bg_pixel: false,
            line_has_fractional_scrolling: false,
            cgb_wx_glitch: false,
            wx_just_changed: false,
            disable_window_pixel_insertion_glitch: false,
            fine: s.display[SCX as usize] & 7,
        }
    }

    /// One pass. A sleeping line waits a dot, or first applies the pending write's next
    /// change if it lands by that dot's end: as in SameBoy's state machine, code at a time
    /// runs only once that time has passed, so a write landing on it comes first. A DMG
    /// object fetch the GBC CPU aborts by clearing OBJ_EN ends the wait at the write.
    pub fn step(&mut self, vram: &[u8; 0x4000], oam: &[u8; 0xA0], palettes: &PaletteRam, pending: Option<Pending>) -> Out {
        let mut out = Out::default();
        match self.phase {
            Phase::Sleep => {
                let due = pending.and_then(|p| Some((p, self.act(p.write, p.applied)?))).filter(|(_, (at, _))| *at <= self.t + 2);
                match due {
                    Some((p, (at, act))) => {
                        out.applied = true;
                        out.finished = self.act(p.write, p.applied + 1).is_none();
                        if self.apply(act) {
                            self.t = at;
                            self.dots = 0;
                            self.phase = Phase::Render;
                            return out;
                        }
                    }
                    None => {
                        self.t += 2;
                        self.dots -= 1;
                    }
                }
                if self.dots == 0 {
                    self.phase = self.resume;
                }
            }
            Phase::Init => {
                // Eight pixels of junk, dropped as the fine scroll is applied.
                self.bg.push_row(0, 0, 0, false, false);
                self.phase = Phase::Window;
            }
            Phase::Window => {
                if self.window() {
                    self.sleep(1, Phase::WindowOn);
                } else {
                    self.phase = Phase::Match { first: true };
                }
            }
            Phase::WindowOn => {
                self.start_window();
                self.phase = Phase::Match { first: true };
            }
            Phase::Match { first } => {
                if first {
                    while self.next_object().is_some_and(|(_, x)| x < self.x_for_object_match()) {
                        self.pass_object();
                    }
                    self.during_object_fetch = true;
                }
                let fetch = self.objects_enabled() && self.next_object().is_some_and(|(_, x)| x == self.x_for_object_match());
                self.phase = if fetch { Phase::FetchWait } else { Phase::Render };
            }
            Phase::Render => {
                self.during_object_fetch = false;
                out.pixel = self.render_pixel(palettes);
                self.phase = Phase::Advance;
            }
            Phase::Advance => {
                self.advance_fetcher(vram);
                if self.pos == WIDTH as u8 {
                    self.phase = Phase::Done;
                    out.done = true;
                } else {
                    self.sleep(1, Phase::Window);
                }
            }
            // The object fetch: the background fetcher finishes its tile first.
            Phase::FetchWait => {
                if self.step < Step::HighT2 || self.bg.size == 0 {
                    self.advance_fetcher(vram);
                    self.sleep(1, Phase::FetchWait);
                } else {
                    self.phase = Phase::FetchA;
                }
            }
            Phase::FetchA => {
                self.advance_fetcher(vram);
                self.sleep(1, Phase::FetchB);
            }
            Phase::FetchB => {
                self.advance_fetcher(vram);
                self.sleep(2, Phase::FetchC);
            }
            Phase::FetchC => {
                self.obj_low = self.read(vram, self.object_line_address(oam));
                self.sleep(2, Phase::FetchD);
            }
            Phase::FetchD => {
                self.during_object_fetch = false;
                self.obj_high = self.read(vram, self.object_line_address(oam) + 1);
                self.sleep(1, Phase::FetchE);
            }
            Phase::FetchE => {
                let (index, _) = self.next_object().expect("an object being fetched");
                let flags = oam[index as usize * 4 + 3];
                let palette = if self.cgb_mode() { flags & 7 } else { flags >> 4 & 1 };
                let priority = if self.cgb_mode() { index } else { 0 };
                self.obj.overlay_row(self.obj_low, self.obj_high, palette, flags & 0x80 != 0, priority, flags & 0x20 != 0);
                self.data_for_sel_glitch = self.obj_high;
                self.pass_object();
                self.phase = Phase::Match { first: false };
            }
            Phase::Done => out.done = true,
        }
        out
    }

    fn sleep(&mut self, dots: u8, resume: Phase) {
        self.phase = Phase::Sleep;
        self.dots = dots;
        self.resume = resume;
    }

    fn cgb(&self) -> bool {
        self.mode.rgb555()
    }

    fn cgb_mode(&self) -> bool {
        self.mode.cgb_features()
    }

    fn act(&self, w: Write, n: u8) -> Option<(Time, Act)> {
        act(self.cgb(), self.double, w, n)
    }

    fn reg(&self, i: u8) -> u8 {
        self.regs[(i - BGP) as usize]
    }

    fn lcdc(&self) -> u8 {
        self.reg(LCDC)
    }

    fn read(&self, vram: &[u8; 0x4000], addr: u16) -> u8 {
        if self.blocked { 0xFF } else { vram[addr as usize & 0x3FFF] }
    }

    fn next_object(&self) -> Option<(u8, u8)> {
        (self.count > 0).then_some(self.objects[0])
    }

    fn pass_object(&mut self) {
        self.objects.copy_within(1.., 0);
        self.objects[OBJECTS - 1] = (0, 0);
        self.count -= 1;
    }

    /// Applies a write's change. True when it aborted a DMG object fetch.
    fn apply(&mut self, act: Act) -> bool {
        match act {
            Act::Set(i, v) => self.regs[(i - BGP) as usize] = v,
            Act::Or(i, v) => self.regs[(i - BGP) as usize] |= v,
            Act::DmgLcdc(v) => {
                // SameBoy's first stage keeps the old value but for BG_EN set and OBJ_EN cleared
                // at x 0 or during an object fetch. On the DMG of Mealybug's DMG-blob images the
                // fetcher's address bits take the new value here, as SCY does
                // (m3_lcdc_bg_map_change, _tile_sel_change, _tile_sel_win_change,
                // _win_map_change, _obj_size_change), and so does WIN_EN cleared
                // (m3_lcdc_win_en_change_multiple_wx: the window stops a tile sooner) and BG_EN
                // cleared at x 0 as OBJ_EN is (m3_lcdc_bg_en_change, lines 19 to 22). Any of
                // the three set waits for the second stage (m3_lcdc_bg_en_change: the blob
                // image shows the background a pixel after the DMG-B image does).
                const EARLY: u8 = BG_MAP | TILE_SEL | WIN_MAP | OBJ_SIZE;
                let old = self.lcdc();
                let mut cleared = WIN_EN;
                if self.pos == 0 {
                    cleared |= BG_EN | OBJ_EN;
                }
                if self.during_object_fetch {
                    cleared |= OBJ_EN;
                }
                // The window pixel insertion is disabled as WIN_EN drops. SameBoy disables it at
                // the second stage, which here would let a push between the stages insert a pixel
                // (m3_lcdc_win_en_change_multiple_wx, lines 3 and 31).
                if old & WIN_EN != 0 && v & WIN_EN == 0 && self.window_is_being_fetched {
                    self.disable_window_pixel_insertion_glitch = true;
                }
                return self.write_lcdc((v & EARLY) | (old & !EARLY & (v | !cleared)));
            }
            Act::DmgLcdcFinal(v) => return self.write_lcdc(v),
            Act::CgbLcdc(v) => {
                if !v & self.lcdc() & TILE_SEL != 0 {
                    self.tile_sel_glitch = true;
                }
                return self.write_lcdc(v);
            }
            Act::CgbDoubleLcdc(v) => {
                let old = self.lcdc();
                self.tile_sel_glitch = (v ^ old) & TILE_SEL != 0;
                return self.write_lcdc((v & !(BG_EN | 0x80)) | (old & (BG_EN | 0x80)));
            }
            Act::WxJustChanged(b) => self.wx_just_changed = b,
            Act::TileSelGlitch(b) => self.tile_sel_glitch = b,
        }
        false
    }

    /// An LCDC write as the register sees it (SameBoy `GB_write_memory`, FF40): on a DMG,
    /// clearing OBJ_EN during an object fetch aborts it, which it returns.
    fn write_lcdc(&mut self, v: u8) -> bool {
        let aborted = !self.cgb() && self.lcdc() & OBJ_EN != 0 && v & OBJ_EN == 0 && self.during_object_fetch;
        self.regs[(LCDC - BGP) as usize] = v;
        aborted
    }

    /// The window starts when the line reaches WX (SameBoy, "Handle window"). True when it
    /// starts only after a dot ([`Phase::WindowOn`]).
    fn window(&mut self) -> bool {
        let (wx, scx) = (self.reg(WX), self.reg(SCX));
        let pos = self.pos;
        if !self.wx_triggered && self.wy_triggered && self.lcdc() & WIN_EN != 0 {
            let mut activate = false;
            if wx == 0 {
                activate = pos == (-7i8) as u8 || (pos == (-16i8) as u8 && scx & 7 != 0) || ((-15i8) as u8..=(-8i8) as u8).contains(&pos);
            } else if (wx as u16) < 166 + self.cgb() as u16 {
                if wx == pos.wrapping_add(7) {
                    activate = true;
                } else if !self.cgb() && wx == pos.wrapping_add(6) && !self.wx_just_changed {
                    // SameBoy also sets the LCD a pixel back here, which it finds on some DMGs
                    // only. In m3_lcdc_win_en_change_multiple_wx (lines 16 and 44) the DMG-blob
                    // image draws the window from here; the DMG-B image draws it a pixel sooner,
                    // and both draw all 160 columns, which SameBoy's step back does not.
                    activate = true;
                }
            }
            if activate {
                self.window_y = self.window_y.wrapping_add(1);
                self.window_tile_x = 0;
                self.bg = BgFifo::default();
                if wx == 0 && scx & 7 != 0 && !self.cgb() {
                    return true;
                }
                // The started window is being fetched, so no pixel is inserted below.
                self.start_window();
                return false;
            } else if !self.cgb() && wx == 166 && wx == pos.wrapping_add(7) {
                self.window_y = self.window_y.wrapping_add(1);
            }
        }
        let wx = self.reg(WX);
        if wx == self.pos.wrapping_add(7) && (!self.cgb() || wx == 0) && self.wx_triggered && !self.window_is_being_fetched && self.step == Step::TileT1 && self.bg.size == 8 {
            self.insert_bg_pixel = true;
        }
        false
    }

    fn start_window(&mut self) {
        self.wx_triggered = true;
        self.step = Step::TileT1;
        self.window_is_being_fetched = true;
    }

    fn x_for_object_match(&self) -> u8 {
        let x = self.pos.wrapping_add(8);
        if x > (-16i8) as u8 { 0 } else { x }
    }

    fn objects_enabled(&self) -> bool {
        self.lcdc() & OBJ_EN != 0 || self.cgb()
    }

    fn object_line_address(&self, oam: &[u8; 0xA0]) -> u16 {
        let (index, _) = self.next_object().expect("an object being fetched");
        let [y, _, tile, flags] = oam[index as usize * 4..][..4].try_into().expect("four OAM bytes an object");
        let tall = self.lcdc() & OBJ_SIZE != 0;
        let mask = if tall { 0xF } else { 7 };
        let mut row = self.ly.wrapping_sub(y) & mask;
        if flags & 0x40 != 0 {
            row ^= mask;
        }
        let mut address = (if tall { tile & 0xFE } else { tile }) as u16 * 16 + row as u16 * 2;
        if self.cgb_mode() && flags & 0x08 != 0 {
            address += 0x2000;
        }
        address
    }

    /// SameBoy `render_pixel_if_possible`: the pixel it sends to the screen, if any.
    fn render_pixel(&mut self, palettes: &PaletteRam) -> Option<(u8, Colour)> {
        if self.objects_enabled() && self.next_object().is_some_and(|(_, x)| x == 0) {
            return None;
        }
        if self.bg.size == 0 {
            return None;
        }
        let item = if std::mem::take(&mut self.insert_bg_pixel) { Item::default() } else { self.bg.pop() };
        let mut bg_priority = item.bg_priority;
        let mut sprite = None;
        let o = self.obj.pop();
        if o.pixel != 0 && self.lcdc() & OBJ_EN != 0 {
            sprite = Some(o);
            bg_priority |= o.bg_priority;
        }
        let scx = self.reg(SCX);
        if self.pos.wrapping_add(16) < 8 {
            if self.pos & 7 == scx & 7 {
                self.pos = (-8i8) as u8;
            } else if self.window_is_being_fetched && self.pos & 7 == 6 && scx & 7 == 7 {
                self.pos = (-8i8) as u8;
            } else if self.pos == (-9i8) as u8 {
                self.pos = (-16i8) as u8;
                return None;
            } else {
                self.line_has_fractional_scrolling = true;
            }
        }
        self.window_is_being_fetched = false;
        if self.pos >= WIDTH as u8 {
            self.pos = self.pos.wrapping_add(1);
            return None;
        }
        let mut bg_enabled = true;
        if self.lcdc() & BG_EN == 0 {
            if self.cgb_mode() {
                bg_priority = false;
            } else {
                bg_enabled = false;
            }
        }
        let pixel = if bg_enabled { item.pixel } else { 0 };
        if pixel != 0 && bg_priority {
            sprite = None;
        }
        let px = match sprite {
            Some(o) => o.pixel | o.palette << 2 | 0x20,
            None => pixel | item.palette << 2,
        };
        let colour = if self.blocked {
            0
        } else {
            let mut all = [0; PAL_REGS as usize];
            all[..PAL_DMG as usize].copy_from_slice(palettes);
            all[PAL_DMG as usize..].copy_from_slice(&self.regs[..3]);
            colour_in(&all, self.mode, px)
        };
        let x = self.lcd_x;
        self.pos = self.pos.wrapping_add(1);
        self.lcd_x = self.lcd_x.wrapping_add(1);
        ((x as usize) < WIDTH).then_some((x, colour))
    }

    fn fetcher_y_now(&self) -> u8 {
        if self.wx_triggered { self.window_y } else { self.ly.wrapping_add(self.reg(SCY)) }
    }

    /// SameBoy `GB_update_wx_glitch`.
    fn update_wx_glitch(&mut self) {
        if !self.cgb() {
            return;
        }
        if self.lcdc() & WIN_EN == 0 || !self.wy_triggered {
            self.cgb_wx_glitch = false;
            return;
        }
        let wx = self.reg(WX);
        if wx == 0 {
            self.cgb_wx_glitch = self.pos.wrapping_add(16) <= 8 || (self.pos == (-7i8) as u8 && self.line_has_fractional_scrolling);
            return;
        }
        self.cgb_wx_glitch = self.pos.wrapping_add(7).wrapping_add(self.window_is_being_fetched as u8) == wx;
    }

    /// The tile-select glitch's byte on CPU CGB E (SameBoy `data_for_tile_sel_glitch`, the
    /// branch for every revision but D).
    fn tile_sel_glitch_data(&self) -> Option<u8> {
        if self.last_tileset { (self.tile & 0x80 == 0).then_some(self.tile) } else { Some(self.data_for_sel_glitch) }
    }

    fn tile_address(&self) -> u16 {
        let base = if self.last_tileset { self.tile as u16 * 16 } else { (0x1000 + self.tile as i8 as i32 * 16) as u16 };
        base + if self.attr & 0x08 != 0 { 0x2000 } else { 0 }
    }

    fn data_row(&self) -> u16 {
        (((self.fetcher_y & 7) ^ if self.attr & 0x40 != 0 { 7 } else { 0 }) as u16) * 2
    }

    /// SameBoy `advance_fetcher_state_machine`, one dot.
    fn advance_fetcher(&mut self, vram: &[u8; 0x4000]) {
        match self.step {
            Step::TileT1 => {
                self.update_wx_glitch();
                if self.lcdc() & WIN_EN == 0 {
                    self.wx_triggered = false;
                }
                let lcdc = self.lcdc();
                let map: u16 = if (lcdc & BG_MAP != 0 && !self.wx_triggered) || (lcdc & WIN_MAP != 0 && self.wx_triggered) { 0x1C00 } else { 0x1800 };
                let y = self.fetcher_y_now();
                let x = if self.wx_triggered {
                    self.window_tile_x
                } else if self.pos.wrapping_add(16) < 8 {
                    self.reg(SCX) >> 3
                } else {
                    let sum = self.reg(SCX) as i32 + self.pos as i32 + 8 - (self.cgb() && !self.during_object_fetch) as i32;
                    (sum / 8 & 0x1F) as u8
                };
                // CGB D and later cache the row for the whole fetch (SameBoy: model > CGB C).
                if self.cgb() {
                    self.fetcher_y = y;
                }
                self.tile_index_address = map + x as u16 + y as u16 / 8 * 32;
                self.step = Step::TileT2;
            }
            Step::TileT2 => {
                if !self.cgb_wx_glitch {
                    self.tile = self.read(vram, self.tile_index_address);
                    self.attr = if self.cgb_mode() { self.read(vram, self.tile_index_address + 0x2000) } else { 0 };
                }
                self.step = Step::LowT1;
            }
            Step::LowT1 => {
                self.update_wx_glitch();
                if !self.cgb() {
                    self.fetcher_y = self.fetcher_y_now();
                }
                self.last_tileset = self.lcdc() & TILE_SEL != 0;
                self.tile_data_address = self.tile_address() + self.data_row();
                self.step = Step::LowT2;
            }
            Step::LowT2 => {
                if self.cgb_wx_glitch {
                    self.data[0] = self.data[1];
                } else {
                    let glitched = if self.tile_sel_glitch { self.tile_sel_glitch_data() } else { None };
                    self.data[0] = glitched.unwrap_or_else(|| self.read(vram, self.tile_data_address));
                    if self.last_tileset && self.tile_sel_glitch {
                        self.data_for_sel_glitch = self.read(vram, self.tile_data_address);
                    }
                }
                self.step = Step::HighT1;
            }
            Step::HighT1 => {
                self.update_wx_glitch();
                if !self.cgb() {
                    self.fetcher_y = self.fetcher_y_now();
                }
                self.last_tileset = self.lcdc() & TILE_SEL != 0;
                self.tile_data_address = self.tile_address() + self.data_row() + 1;
                self.step = Step::HighT2;
            }
            Step::HighT2 => {
                if self.cgb_wx_glitch {
                    self.data[1] = self.data[0];
                    if self.wx_triggered {
                        self.window_tile_x = (self.window_tile_x + 1) & 0x1F;
                    }
                    self.step = Step::Push;
                    return;
                } else {
                    let glitched = if self.tile_sel_glitch { self.tile_sel_glitch_data() } else { None };
                    match glitched {
                        Some(byte) => self.data[1] = byte,
                        None => {
                            self.data[1] = self.read(vram, self.tile_data_address);
                            self.data_for_sel_glitch = self.data[1];
                        }
                    }
                    if self.last_tileset && self.tile_sel_glitch {
                        self.data_for_sel_glitch = self.read(vram, self.tile_data_address);
                    }
                }
                if self.wx_triggered {
                    self.window_tile_x = (self.window_tile_x + 1) & 0x1F;
                }
                self.step = Step::Push;
                self.push();
            }
            Step::Push => self.push(),
        }
    }

    fn push(&mut self) {
        if self.bg.size > 0 {
            return;
        }
        if self.wy_triggered && self.lcdc() & WIN_EN == 0 && !self.cgb() && !self.disable_window_pixel_insertion_glitch {
            // SameBoy issue 278: a DMG with the window disabled at WX inserts one pixel.
            let mut logical = self.pos.wrapping_add(7);
            if logical > 167 {
                logical = 0;
            }
            if self.reg(WX) == logical {
                self.bg = BgFifo { size: 1, ..BgFifo::default() };
                return;
            }
        }
        self.bg.push_row(self.data[0], self.data[1], self.attr & 7, self.attr & 0x80 != 0, self.attr & 0x20 != 0);
        self.step = Step::TileT1;
    }

    /// The dots to mode 0 ([`mode0_dots`]) at this point of the line.
    pub fn mode0_dots(&self) -> u16 {
        mode0_dots(self.dots(), self.fine, self.mode.rgb555())
    }

    /// Mode 3's length in dots once the line is done.
    pub fn dots(&self) -> u16 {
        self.t.div_ceil(2) as u16
    }

    /// The window's state after the line (SameBoy, the end of mode 3).
    pub fn window_after(&self) -> Window {
        let mut window_y = self.window_y;
        let mut carried = false;
        if !self.cgb() && self.wy_triggered && self.lcdc() & WIN_EN != 0 && self.reg(WX) == 166 {
            carried = true;
            window_y = window_y.wrapping_add(1);
        }
        // SameBoy resets the rows as line 143 ends; the PPU resets them at VBlank
        // (`Ppu::end_frame`), which no line comes between.
        Window { rows: window_y.wrapping_add(1), armed: self.wy_triggered, carried }
    }
}

#[cfg(test)]
mod length_rule {
    use super::*;
    use crate::machine::ppu::{Display, Mode};

    /// A quiet line's window cases, in closed form (what `ppu_circuit::circuit` computes).
    pub(super) struct Cases {
        /// The window starts at the line's check (SameBoy's "Handle window"): enabled, armed,
        /// not carried, WX up to 165, or 166 on CGB hardware (`166 + cgb`).
        pub start: bool,
        /// The window is fetched from the line's first tile: carried from a DMG line at WX
        /// 166, or started at WX 0 under a fine scroll (at position -16, before the junk row).
        pub from_start: bool,
        /// A pixel inserted at this position: SameBoy issue 278 on a DMG with the window
        /// disabled (`push`), or a window line's `insert_bg_pixel` (`window`).
        pub inserted: Option<i32>,
        /// The inserted pixel is a window line's (`insert_bg_pixel`, the row already pushed), not a
        /// delayed push (issue 278, or the junk row at WX 247).
        pub in_row: bool,
        pub window: Window,
    }

    pub(super) fn cases(mode: Mode, lcdc: u8, scx: u8, wx: u8, window: Window) -> Cases {
        let (f, cgb) = ((scx & 7) as i32, mode.rgb555());
        let (armed, enabled, carried) = (window.armed, lcdc & WIN_EN != 0, window.carried);
        let start = !carried && armed && enabled && (wx <= 165 || wx == 166 && cgb);
        let from_start = carried && enabled || start && wx == 0 && f != 0;
        // Both insertions land where a pushed row meets `pos + 7 == WX`: rows are pushed at
        // positions `-f + 8n` (and the junk row at -16, which a carried line meets at WX 247).
        let on_push = wx <= 166 && (wx as i32 + f) % 8 == 7;
        let inserted = if (mode == Mode::Dmg && armed && !enabled || from_start) && on_push {
            Some(wx as i32 - 7)
        } else if carried && wx == 247 {
            Some(-f)
        } else {
            None
        };
        let count166 = !carried && armed && enabled && wx == 166 && !cgb;
        let carry = !cgb && armed && enabled && wx == 166;
        let rows = window.rows.wrapping_add(start as u8 + count166 as u8 + carry as u8);
        Cases { start, from_start, inserted, in_row: from_start && on_push, window: Window { rows, armed, carried: carry } }
    }

    /// Mode 3's length on a line with no write, in closed form: 172 dots, SCX's fine
    /// scroll, 6 when the window starts (one more at WX 0 under a fine scroll), and per
    /// object fetched (X below 168) 6 dots plus, for the first object of its tile, 5 -
    /// min(5, fine position). An X 0 object counts as fine position 0 of tile 0, whatever
    /// SCX (Pan Docs, "OBJ penalty algorithm"); one at or past the window's start counts in
    /// the window's grid, `X - WX - 1`, and one at or past an inserted pixel in the grid
    /// after it, `X - 9 - P`.
    pub(super) fn closed_form(mode: Mode, lcdc: u8, scx: u8, wx: u8, window: Window, xs: &[u8]) -> i32 {
        let c = cases(mode, lcdc, scx, wx, window);
        let mut sorted: Vec<u8> = xs.iter().copied().filter(|&x| x < 168).collect();
        sorted.sort();
        let mut tiles = Vec::new();
        let f = (scx & 7) as i32;
        let mut dots = 172 + f + if c.start { 6 + (wx == 0 && f != 0) as i32 } else { 0 };
        for x in sorted {
            let x = x as i32;
            let grid = |region: i32, at: i32| (region, at.div_euclid(8), at.rem_euclid(8));
            let (region, tile, fine) = match c.inserted {
                _ if x == 0 => grid(0, 0),
                // The fetcher runs a dot ahead for the row the pixel was inserted before: one
                // tile 9 positions wide from the inserted pixel, then the grid a dot later.
                Some(p) if c.in_row && (0..=8).contains(&(x - 8 - p)) => (2, -1, x - 8 - p),
                Some(p) if x - 8 >= p => grid(2, x - 9 - p),
                _ if c.start && !c.from_start && x - 8 >= wx as i32 - 7 => grid(1, x - wx as i32 - 1),
                _ => grid(0, x + f),
            };
            if !tiles.contains(&(region, tile)) {
                tiles.push((region, tile));
                dots += 5 - fine.min(5);
            }
            dots += 6;
        }
        dots
    }

    #[test]
    fn the_closed_form_is_the_fifos_length() {
        let mut rng = crate::memory::Rng(0x1234_5678_9ABC_DEF1);
        let mut bad = Vec::new();
        let mut took = [0usize; 4];
        let mut wrong = [0usize; 8];
        for case in 0..200_000 {
            let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][case % 3];
            let scx = rng.draw() as u8;
            let wx = match rng.draw() % 4 {
                0 => (rng.draw() % 7) as u8,
                1 => 166,
                2 => rng.draw() as u8,
                _ => 7 + (rng.draw() % 159) as u8,
            };
            let lcdc = 0x83 | (rng.draw() % 2 == 0) as u8 * WIN_EN;
            let window = Window { rows: rng.draw() as u8, armed: rng.draw() % 4 != 0, carried: mode == Mode::Dmg && rng.draw() % 3 == 0 };
            let window = Window { armed: window.armed || window.carried, ..window };
            let k = rng.draw() as usize % 11;
            let xs: Vec<u8> = (0..k).map(|_| if rng.draw() % 5 == 0 { 0 } else { rng.draw() as u8 }).collect();
            let vram = [0u8; 0x4000];
            let mut oam = [0u8; 0xA0];
            for (i, &x) in xs.iter().enumerate() {
                oam[i * 4] = 16;
                oam[i * 4 + 1] = x;
            }
            let mut display: Display = [0; crate::memory::layout::PAL_LEN];
            display[LCDC as usize] = lcdc;
            display[SCX as usize] = scx;
            display[WX as usize] = wx;
            let objects = scan(&oam, lcdc, 0);
            let start = Start { ly: 0, mode, double: false, blocked: false, display, window, vram: &vram, oam: &oam, objects: &objects };
            let drawn = draw(&start, &[]);
            let c = cases(mode, lcdc, scx, wx, window);
            took[0] += (c.start && (wx < 7 || wx == 166) || !c.start && wx == 166 && window.armed && lcdc & WIN_EN != 0) as usize;
            took[1] += (mode == Mode::Dmg && window.armed && lcdc & WIN_EN == 0 && c.inserted.is_some()) as usize;
            took[2] += window.carried as usize;
            took[3] += c.inserted.is_some_and(|p| p >= 0) as usize;
            let (got, want) = ((drawn.dots as i32, drawn.window), (closed_form(mode, lcdc, scx, wx, window, &xs), c.window));
            if got != want {
                wrong[(window.carried as usize) | (c.in_row as usize) << 1 | ((mode == Mode::Dmg && lcdc & WIN_EN == 0 && c.inserted.is_some()) as usize) << 2] += 1;
            }
            if got != want && bad.len() < 12 {
                bad.push(format!("{mode:?} lcdc {lcdc:#04x} scx {} wx {wx} {window:?} xs {xs:?}: fifo {got:?}, closed form {want:?}", scx & 7));
            }
        }
        println!("lines by case: WX below 7 or at 166 {}, inserted with the window off {}, carried {}, a visible inserted pixel {}", took[0], took[1], took[2], took[3]);
        assert!(bad.is_empty(), "wrong by class (carried | in row << 1 | window off << 2): {wrong:?}\n{bad:#?}");
    }

    /// How the CGB's WX glitch (`update_wx_glitch`, SameBoy `GB_update_wx_glitch`) leaves a
    /// fetched tile: its tile index and attributes the stale zeros of `State::new`, and each
    /// bit plane kept, zero, or (`Low`) the low plane copied into the high one.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Plane {
        Keep,
        Zero,
        Low,
    }

    /// The glitched tiles of a quiet line on CGB hardware, with the window enabled and armed.
    /// A T1 step (TileT1, LowT1, HighT1) runs glitched where `pos + 7 + window_is_being_fetched
    /// == WX`, or at WX 0 where `pos <= -8` or `pos == -7` under a fine scroll. With no object at
    /// X 0 the first tile's steps run a dot each from position -15; an object at X 0 runs them
    /// all at -16 (its fetch waits for the fetcher), and one at X 1 runs the rest at -7.
    /// Returns the zeroed tile index and both planes, for window column tile `tile` (`window`)
    /// or the first background tile.
    pub(super) fn glitch(wx: u8, f: i32, x0: bool, x1: bool, window: bool, tile: u8) -> Option<(bool, Plane, Plane)> {
        use Plane::*;
        if window {
            let early = !x0 && !x1;
            return match tile {
                1 => Some((true, if f == 1 && early { Keep } else { Zero }, if f <= 3 && early { Keep } else { Zero })),
                2 if f == 7 && x1 => Some((true, Zero, Zero)),
                _ => None,
            };
        }
        match wx {
            247 if x0 => Some((true, Zero, Zero)),
            248 if !x0 && f >= 1 => Some((true, Keep, Keep)),
            250 if !x0 && f >= 3 => Some((false, Zero, Keep)),
            252 if !x0 && f >= 5 => Some((false, Keep, Low)),
            _ => None,
        }
    }

    /// The background or window pixel a quiet line shows at `x` under `cases`, as a pixel
    /// before the palette lookup: the background at SCX + x, the window at its column, both
    /// a pixel later after an inserted pixel, the inserted pixel itself as colour 0, and the
    /// CGB's glitched tiles (`glitch`). `x0` and `x1`: an object the line fetches at X 0, at X 1.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pixel(mode: Mode, display: &Display, vram: &[u8; 0x4000], ly: u8, window: Window, x0: bool, x1: bool, x: u8) -> u8 {
        let [lcdc, scy, scx, _, wx] = std::array::from_fn(|i| display[LCDC as usize + i]);
        let c = cases(mode, lcdc, scx, wx, window);
        let (x, f) = (x as i32, (scx & 7) as i32);
        if c.inserted == Some(x) {
            return 0;
        }
        let shift = c.inserted.is_some_and(|p| x > p) as i32;
        let glitchy = mode.rgb555() && window.armed && lcdc & WIN_EN != 0;
        let (map, col, row, glitch) = if c.from_start || c.start && x >= wx as i32 - 7 {
            let col = if c.from_start { x + 8 + f - shift } else { x + 7 - wx as i32 } as u8;
            let row = window.rows.wrapping_sub(window.carried as u8);
            let glitch = (glitchy && c.from_start).then(|| glitch(wx, f, x0, x1, true, col >> 3)).flatten();
            (if lcdc & WIN_MAP != 0 { 0x1C00 } else { 0x1800 }, col, row, glitch)
        } else {
            let glitch = (glitchy && x <= 7 - f).then(|| glitch(wx, f, x0, x1, false, 0)).flatten();
            (if lcdc & BG_MAP != 0 { 0x1C00 } else { 0x1800 }, (scx as i32 + x - shift) as u8, ly.wrapping_add(scy), glitch)
        };
        let at = map + (col >> 3) as usize + (row as usize / 8) * 32;
        let stale = glitch.is_some_and(|g| g.0);
        let (tile, attr) = if stale { (0, 0) } else { (vram[at], if mode.cgb_features() { vram[at + 0x2000] } else { 0 }) };
        let base = if lcdc & TILE_SEL != 0 { tile as usize * 16 } else { (0x1000 + tile as i8 as i32 * 16) as usize };
        let line = base + if attr & 0x08 != 0 { 0x2000 } else { 0 } + 2 * ((row & 7) ^ if attr & 0x40 != 0 { 7 } else { 0 }) as usize;
        let (mut low, mut high) = (vram[line], vram[line + 1]);
        if let Some((_, l, h)) = glitch {
            let plane = |p: Plane, own: u8| match p {
                Plane::Keep => own,
                Plane::Zero => 0,
                Plane::Low => low,
            };
            (low, high) = (plane(l, low), plane(h, high));
        }
        let bit = (col & 7) ^ if attr & 0x20 != 0 { 7 } else { 0 };
        let colour = (low >> (7 - bit) & 1) | (high >> (7 - bit) & 1) << 1;
        let shown = if lcdc & BG_EN == 0 && !mode.cgb_features() { 0 } else { colour };
        shown | (attr & 7) << 2
    }

    #[test]
    fn the_pixel_rule_is_the_fifos_background() {
        let mut rng = crate::memory::Rng(0x0BAD_5EED_1E46_0001);
        let mut bad = Vec::new();
        let mut took = [0usize; 4];
        let mut wrong = [0usize; 8];
        for case in 0..60_000 {
            let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][case % 3];
            let mut vram = [0u8; 0x4000];
            for chunk in vram.chunks_mut(8) {
                chunk.copy_from_slice(&rng.draw().to_le_bytes());
            }
            // Objects use tile 0x7F, whose rows are zero: fetched, never seen.
            vram[0x7F0..0x800].fill(0);
            let ly = (rng.draw() % 144) as u8;
            let mut oam = [0u8; 0xA0];
            for e in 0..(rng.draw() % 11) as usize {
                oam[e * 4..e * 4 + 4].copy_from_slice(&[ly + 16 - (rng.draw() % 8) as u8, match rng.draw() % 6 { 0 => 0, 1 => 1, _ => rng.draw() as u8 }, 0x7F, 0]);
            }
            let mut display: Display = [0; crate::memory::layout::PAL_LEN];
            for (i, b) in display.iter_mut().enumerate() {
                *b = i as u8 ^ 0x5A;
            }
            display[BGP as usize] = 0xE4;
            let lcdc = 0x80 | (rng.draw() as u8 & 0x7D) & !OBJ_SIZE;
            display[LCDC as usize] = lcdc | if mode == Mode::Dmg { OBJ_EN } else { 0 };
            display[SCY as usize] = rng.draw() as u8;
            display[SCX as usize] = rng.draw() as u8;
            display[WX as usize] = match rng.draw() % 5 {
                0 => (rng.draw() % 7) as u8,
                1 => 166,
                2 => rng.draw() as u8,
                3 => 247 + (rng.draw() % 8) as u8,
                _ => 7 + (rng.draw() % 159) as u8,
            };
            let carried = mode == Mode::Dmg && rng.draw() % 3 == 0;
            let window = Window { rows: 1 + (rng.draw() % 140) as u8, armed: carried || rng.draw() % 4 != 0, carried };
            let objects = scan(&oam, display[LCDC as usize], ly);
            let start = Start { ly, mode, double: false, blocked: false, display, window, vram: &vram, oam: &oam, objects: &objects };
            let drawn = draw(&start, &[]);
            let palettes: &[u8; PAL_REGS as usize] = display[..PAL_REGS as usize].try_into().unwrap();
            let c = cases(mode, display[LCDC as usize], display[SCX as usize], display[WX as usize], window);
            took[0] += c.inserted.is_some_and(|p| p >= 0) as usize;
            took[1] += c.inserted.is_some_and(|p| p < 0) as usize;
            took[2] += (c.from_start && !carried) as usize;
            took[3] += c.from_start as usize;
            let k0 = objects.iter().filter(|&&i| oam[i as usize * 4 + 1] == 0).count();
            let k1 = objects.iter().filter(|&&i| oam[i as usize * 4 + 1] == 1).count();
            let class = if display[WX as usize] >= 247 && mode.rgb555() { 0 } else if display[WX as usize] == 0 && mode.rgb555() { 1 + k0.min(2) * 2 + k1.min(1) } else { 7 };
            for x in 0..WIDTH as u8 {
                let want = colour_in(palettes, mode, pixel(mode, &display, &vram, ly, window, k0 > 0, k1 > 0, x));
                if drawn.colours[x as usize] != want {
                    wrong[class] += 1;
                    if bad.len() < 12 && (class == 7 || std::env::var_os("ALL").is_some()) {
                        bad.push(format!("{mode:?} ly {ly} lcdc {:#04x} scx {} wx {} {window:?} k0 {k0} k1 {k1} x {x}: fifo {:#06x}, rule {want:#06x}", display[LCDC as usize], display[SCX as usize], display[WX as usize], drawn.colours[x as usize]));
                    }
                    break;
                }
            }
        }
        println!("lines: a visible inserted pixel {}, an inserted pixel off the screen {}, a window from the first tile at WX 0 {}, a window from the first tile {}", took[0], took[1], took[2], took[3]);
        assert!(bad.is_empty() && wrong == [0; 8], "wrong lines by class (CGB WX 247+, CGB WX 0 by X0 objects 0,1,2+ and X1 objects 0,1+, other): {wrong:?}\n{bad:#?}");
    }
}

/// A random line with display writes during it, for the tests that hold the step machine
/// and the circuits that compute it to the reference: random VRAM, OAM with up to ten
/// objects on the line (some at X 0 or past the screen), random display bytes with WX
/// often at its edge cases, a random window, and up to twelve writes to the registers
/// (on the DMG often LCDC, with OBJ_EN cleared half the time), one to eight M-cycles apart,
/// as the GBC CPU's are.
pub struct Sample {
    pub ly: u8,
    pub mode: Mode,
    pub double: bool,
    pub blocked: bool,
    pub display: Display,
    pub window: Window,
    pub vram: Box<[u8; 0x4000]>,
    pub oam: [u8; 0xA0],
    pub objects: Vec<u8>,
    pub writes: Vec<Write>,
}

impl Sample {
    /// Line `case`'s sample: DMG, CGB and compatibility mode in turn.
    pub fn random(rng: &mut crate::memory::Rng, case: usize) -> Sample {
        let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][case % 3];
        let double = mode != Mode::Dmg && rng.draw() % 2 == 0;
        let ly = (rng.draw() % 144) as u8;
        let mut vram = Box::new([0u8; 0x4000]);
        for chunk in vram.chunks_mut(8) {
            chunk.copy_from_slice(&rng.draw().to_le_bytes());
        }
        let mut oam = [0u8; 0xA0];
        for chunk in oam.chunks_mut(8) {
            chunk.copy_from_slice(&rng.draw().to_le_bytes());
        }
        for object in oam.chunks_mut(4) {
            object[0] = 0;
        }
        for _ in 0..rng.draw() % 11 {
            let object = &mut oam[(rng.draw() % 40) as usize * 4..][..2];
            object[0] = ly + 16 - (rng.draw() % 8) as u8;
            object[1] = match rng.draw() % 5 {
                0 => 0,
                1 => 168 + (rng.draw() % 88) as u8,
                _ => 1 + (rng.draw() % 167) as u8,
            };
        }
        let mut display: Display = [0; PAL_LEN];
        for chunk in display.chunks_mut(8) {
            chunk.copy_from_slice(&rng.draw().to_le_bytes());
        }
        if rng.draw() % 4 != 0 {
            display[LCDC as usize] |= OBJ_EN;
        }
        display[WX as usize] = match rng.draw() % 8 {
            0 | 1 => (rng.draw() % 7) as u8,
            2 => 166,
            3 => rng.draw() as u8,
            _ => 7 + (rng.draw() % 159) as u8,
        };
        let window = Window { rows: (rng.draw() % 144) as u8, armed: rng.draw() % 2 == 0, carried: mode == Mode::Dmg && rng.draw() % 8 == 0 }.starting(ly, display[WY as usize]);
        let objects = scan(&oam, display[LCDC as usize], ly);
        let blocked = mode != Mode::Dmg && rng.draw() % 32 == 0;
        let cycle = if double { 4 } else { 8 };
        let mut at = (rng.draw() % 48) as Time;
        let writes: Vec<Write> = (0..rng.draw() % 13)
            .map(|_| {
                let index = if mode == Mode::Dmg && rng.draw() % 3 == 0 { LCDC } else { BGP + (rng.draw() % 8) as u8 };
                let mut value = rng.draw() as u8;
                if index == LCDC && rng.draw() % 2 == 0 {
                    value &= !OBJ_EN;
                }
                let write = Write { at, index, value };
                at += cycle * (1 + (rng.draw() % 8) as Time);
                write
            })
            .collect();
        Sample { ly, mode, double, blocked, display, window, vram, oam, objects, writes }
    }

    pub fn start(&self) -> Start<'_> {
        Start { ly: self.ly, mode: self.mode, double: self.double, blocked: self.blocked, display: self.display, window: self.window, vram: &self.vram, oam: &self.oam, objects: &self.objects }
    }
}

#[cfg(test)]
mod queue {
    use super::*;

    /// The driver the queue replaced: each pass is handed the write whose next change lands
    /// first, the earlier write on a tie, which is SameBoy's time order for any spacing.
    fn drive_by_time(mut state: State, writes: &[Write], vram: &[u8; 0x4000], oam: &[u8; 0xA0], palettes: &PaletteRam) -> Vec<State> {
        let (cgb, double) = (state.mode.rgb555(), state.double);
        let mut applied = vec![0u8; writes.len()];
        let mut states = vec![state];
        loop {
            let next = (0..writes.len()).filter_map(|i| Some((act(cgb, double, writes[i], applied[i])?.0, i))).min().map(|(_, i)| i);
            let out = state.step(vram, oam, palettes, next.map(|i| Pending { write: writes[i], applied: applied[i] }));
            if let (Some(i), true) = (next, out.applied) {
                applied[i] += 1;
            }
            states.push(state);
            if out.done {
                return states;
            }
        }
    }

    #[test]
    fn drive_is_time_order() {
        let mut rng = crate::memory::Rng(0x5EED_0F_0123_4567);
        let mut seen = [[0usize; 2]; 3];
        for case in 0..30_000 {
            let s = Sample::random(&mut rng, case);
            let start = s.start();
            let palettes = palettes(&start.display);
            let want = drive_by_time(State::new(&start), &s.writes, start.vram, start.oam, palettes);
            let mut got = vec![State::new(&start)];
            drive(State::new(&start), &s.writes, |st, p| {
                let out = st.step(start.vram, start.oam, palettes, p);
                got.push(*st);
                out
            });
            assert!(got == want, "line {case}: {:?} double {}: the queue's passes differ from time order; writes {:?}", s.mode, s.double, s.writes);
            seen[case % 3][s.double as usize] += !s.writes.is_empty() as usize;
        }
        assert!(seen.iter().enumerate().all(|(m, n)| n[0] > 0 && (m == 0 || n[1] > 0)), "every mode and speed with writes: {seen:?}");
    }

    /// The writes a marked line hands the stepper run until the next line opens, so some land
    /// after the line's end: those at or past [`REACH`] past its end change nothing.
    #[test]
    fn a_write_past_the_end_changes_nothing() {
        let mut rng = crate::memory::Rng(0x0E4D_0F_11AE_0001);
        for case in 0..30_000 {
            let s = Sample::random(&mut rng, case);
            let start = s.start();
            let drawn = draw(&start, &s.writes);
            let mcycle = if s.double { 4 } else { 8 };
            let end = drawn.dots as Time * 2 + REACH;
            let mut at = s.writes.last().map_or(end, |w| end.max(w.at + mcycle)) + (rng.draw() % 4) as Time;
            let mut writes = s.writes.clone();
            for _ in 0..1 + rng.draw() % 3 {
                writes.push(Write { at, index: BGP + (rng.draw() % REGS as u64) as u8, value: rng.draw() as u8 });
                at += mcycle * (1 + (rng.draw() % 4) as Time);
            }
            assert_eq!(draw(&start, &writes), drawn, "line {case}: {:?} double {}: writes {writes:?}", s.mode, s.double);
        }
    }

    /// [`judge`] is sound: a line whose every write it calls inert draws as with no write.
    /// The writes are drawn to be inert often: WY, a CGB-mode palette register, the value the
    /// line started with (DMG LCDC among them), or a time past the line's end.
    #[test]
    fn a_line_the_judge_passes_draws_as_with_no_write() {
        let mut rng = crate::memory::Rng(0x0A0D_6E_5EED_7777);
        let (mut inert_lines, mut marked, mut marked_changed) = (0, 0, 0);
        let mut kinds = [0usize; 4];
        for case in 0..60_000 {
            let s = Sample::random(&mut rng, case);
            let start = s.start();
            let quiet = draw(&start, &[]);
            let mcycle = if s.double { 4 } else { 8 };
            let mut at = (rng.draw() % 48) as Time;
            let mut writes = Vec::new();
            for _ in 0..1 + rng.draw() % 6 {
                let kind = (rng.draw() % 5) as usize;
                let mut index = BGP + (rng.draw() % REGS as u64) as u8;
                let mut value = rng.draw() as u8;
                match kind {
                    0 => index = WY,
                    1 => index = BGP + (rng.draw() % 3) as u8,
                    2 => value = start.display[index as usize],
                    3 => at = at.max(quiet.dots as Time * 2 + REACH + (rng.draw() % 8) as Time),
                    _ => {}
                }
                if kind < 4 {
                    kinds[kind] += 1;
                }
                writes.push(Write { at, index, value });
                at += mcycle * (1 + (rng.draw() % 8) as Time);
            }
            let any = writes.iter().any(|&w| judge(&start, quiet.dots, w));
            let drawn = draw(&start, &writes);
            if any {
                marked += 1;
                marked_changed += (drawn != quiet) as usize;
            } else {
                inert_lines += 1;
                assert_eq!(drawn, quiet, "line {case}: {:?} double {}: the judge passed every write, but they change the line; writes {writes:?}", s.mode, s.double);
            }
        }
        println!("{inert_lines} lines judged inert, {marked} marked ({marked_changed} of them changed); inert kinds drawn {kinds:?}");
        assert!(inert_lines > 10_000 && marked_changed > 1_000, "both verdicts exercised");
    }
}
