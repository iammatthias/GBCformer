//! The line as it was drawn before [`super::State`]: one loop whose waits (`sleep`) sit
//! inside the object fetch. The state machine must draw every line as it does.

use super::*;
use crate::memory::Rng;

/// Draws a line.
fn draw(start: &Start, writes: &[Write]) -> Drawn {
    let mut line = Line::new(start, writes);
    line.run();
    Drawn { colours: line.out, window: line.window_after(), dots: line.t.div_ceil(2) as u16 }
}

#[derive(Clone, Copy, Default)]
struct Item {
    pixel: u8,
    palette: u8,
    priority: u8,
    bg_priority: bool,
}

#[derive(Clone, Copy, Default)]
struct Fifo {
    items: [Item; 8],
    read: usize,
    size: usize,
}

impl Fifo {
    fn clear(&mut self) {
        self.read = 0;
        self.size = 0;
    }

    fn pop(&mut self) -> Item {
        let item = self.items[self.read];
        self.read = (self.read + 1) & 7;
        self.size -= 1;
        item
    }

    fn push_bg_row(&mut self, lower: u8, upper: u8, palette: u8, bg_priority: bool, flip: bool) {
        assert_eq!(self.size, 0, "a background row pushed into a FIFO holding pixels");
        self.size = 8;
        for i in 0..8 {
            let bit = if flip { i } else { 7 - i };
            self.items[(self.read + i) & 7] = Item { pixel: (lower >> bit & 1) | (upper >> bit & 1) << 1, palette, priority: 0, bg_priority };
        }
    }

    fn overlay_object_row(&mut self, lower: u8, upper: u8, palette: u8, bg_priority: bool, priority: u8, flip: bool) {
        while self.size < 8 {
            self.items[(self.read + self.size) & 7] = Item::default();
            self.size += 1;
        }
        for i in 0..8 {
            let bit = if flip { i } else { 7 - i };
            let pixel = (lower >> bit & 1) | (upper >> bit & 1) << 1;
            let target = &mut self.items[(self.read + i) & 7];
            if pixel != 0 && (target.pixel == 0 || target.priority > priority) {
                *target = Item { pixel, palette, priority, bg_priority };
            }
        }
    }
}

/// Each write as the changes it makes and when (SameBoy's conflict maps for the DMG, the
/// CGB and the CGB at double speed; a CPU T-cycle is 2 half-dots, or 1 at double speed).
fn acts(cgb: bool, double: bool, writes: &[Write]) -> Vec<(Time, Act)> {
    let tc: i64 = if double { 1 } else { 2 };
    let mut acts = Vec::new();
    for w in writes {
        let at = |cycles: i64| (w.at as i64 + cycles * tc).max(0) as Time;
        let (i, v) = (w.index, w.value);
        match (cgb, double, i) {
            (false, _, LCDC) => acts.extend([(at(-2), Act::DmgLcdc(v)), (at(-1), Act::DmgLcdcFinal(v))]),
            // SameBoy reads the new SCY a T-cycle early (`GB_CONFLICT_READ_NEW`); Mealybug's notes
            // ("SCY", the-comprehensive-game-boy-ppu-documentation.md) have the DMG take it
            // 2 T-cycles before the CGB does, and m3_scy_change on DMG agrees.
            (false, _, SCY) => acts.push((at(-2), Act::Set(i, v))),
            (false, _, BGP..=0x82) => acts.extend([(at(-2), Act::Or(i, v)), (at(-1), Act::Set(i, v))]),
            (false, _, WX) => acts.extend([(at(0), Act::Set(i, v)), (at(0), Act::WxJustChanged(true)), (at(1), Act::WxJustChanged(false))]),
            (false, _, SCX) => acts.push((at(-2), Act::Set(i, v))),
            (true, false, LCDC) => acts.push((at(0), Act::CgbLcdc(v))),
            (true, false, BGP..=0x82) => acts.push((at(-2), Act::Set(i, v))),
            (true, false, WX) => acts.push((at(1), Act::Set(i, v))),
            (true, true, LCDC) => acts.extend([(at(-2), Act::CgbDoubleLcdc(v)), (at(0), Act::TileSelGlitch(false)), (at(0), Act::Set(i, v))]),
            (true, true, SCX) => acts.push((at(-2), Act::Set(i, v))),
            _ => acts.push((at(0), Act::Set(i, v))),
        }
        if let (true, false, LCDC) = (cgb, double, i) {
            acts.push((at(1), Act::TileSelGlitch(false)));
        }
    }
    acts.sort_by_key(|&(t, _)| t);
    acts
}

struct Line<'a> {
    ly: u8,
    mode: Mode,
    /// CGB hardware, in either mode: its fetcher and window timing.
    cgb: bool,
    blocked: bool,
    regs: Display,
    vram: &'a [u8; 0x4000],
    oam: &'a [u8; 0xA0],
    acts: Vec<(Time, Act)>,
    next_act: usize,
    t: Time,
    out: [Colour; WIDTH],
    lcd_x: u8,
    /// SameBoy's `position_in_line`: -16 at mode 3's start, 0 at the first visible pixel.
    pos: u8,
    bg: Fifo,
    obj: Fifo,
    step: Step,
    tile: u8,
    attr: u8,
    data: [u8; 2],
    tile_index_address: u16,
    tile_data_address: u16,
    fetcher_y: u8,
    last_tileset: bool,
    data_for_sel_glitch: u8,
    tile_sel_glitch: bool,
    /// The objects not yet fetched, the next last: OAM index, X, Y.
    objects: Vec<(u8, u8, u8)>,
    during_object_fetch: bool,
    object_fetch_aborted: bool,
    window_y: u8,
    window_tile_x: u8,
    wy_triggered: bool,
    wx_triggered: bool,
    window_is_being_fetched: bool,
    insert_bg_pixel: bool,
    line_has_fractional_scrolling: bool,
    cgb_wx_glitch: bool,
    wx_just_changed: bool,
    disable_window_pixel_insertion_glitch: bool,
}

impl<'a> Line<'a> {
    fn new(s: &Start<'a>, writes: &[Write]) -> Self {
        let cgb = s.mode.rgb555();
        let objects = s.objects.iter().rev().map(|&i| (i, s.oam[i as usize * 4 + 1], s.oam[i as usize * 4])).collect();
        Line {
            ly: s.ly,
            mode: s.mode,
            cgb,
            blocked: s.blocked,
            regs: s.display,
            vram: s.vram,
            oam: s.oam,
            acts: acts(cgb, s.double, writes),
            next_act: 0,
            t: 0,
            out: [0; WIDTH],
            lcd_x: 0,
            pos: (-16i8) as u8,
            bg: Fifo::default(),
            obj: Fifo::default(),
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
            during_object_fetch: false,
            object_fetch_aborted: false,
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
        }
    }

    fn reg(&self, i: u8) -> u8 {
        self.regs[i as usize]
    }

    fn lcdc(&self) -> u8 {
        self.reg(LCDC)
    }

    fn cgb_mode(&self) -> bool {
        self.mode.cgb_features()
    }

    /// Lets `dots` dots pass, applying every change due by their end: as in SameBoy's state
    /// machine, code at a time runs only once that time has passed, so a write landing on
    /// it comes first. A DMG object fetch the CPU aborts by clearing OBJ_EN ends the wait
    /// at the write.
    fn sleep(&mut self, dots: u32) {
        let end = self.t + 2 * dots;
        while let Some(&(at, act)) = self.acts.get(self.next_act).filter(|(at, _)| *at <= end) {
            self.next_act += 1;
            self.apply(act);
            if self.object_fetch_aborted {
                self.t = at.max(self.t);
                return;
            }
        }
        self.t = end;
    }

    fn apply(&mut self, act: Act) {
        match act {
            Act::Set(i, v) => self.regs[i as usize] = v,
            Act::Or(i, v) => self.regs[i as usize] |= v,
            Act::DmgLcdc(v) => {
                // As `State::apply`: the address bits and WIN_EN cleared take the new value, and
                // BG_EN and OBJ_EN cleared at x 0 (OBJ_EN also during an object fetch).
                const EARLY: u8 = BG_MAP | TILE_SEL | WIN_MAP | OBJ_SIZE;
                let old = self.lcdc();
                let mut cleared = WIN_EN;
                if self.pos == 0 {
                    cleared |= BG_EN | OBJ_EN;
                }
                if self.during_object_fetch {
                    cleared |= OBJ_EN;
                }
                if old & WIN_EN != 0 && v & WIN_EN == 0 && self.window_is_being_fetched {
                    self.disable_window_pixel_insertion_glitch = true;
                }
                self.write_lcdc((v & EARLY) | (old & !EARLY & (v | !cleared)));
            }
            Act::DmgLcdcFinal(v) => self.write_lcdc(v),
            Act::CgbLcdc(v) => {
                if !v & self.lcdc() & TILE_SEL != 0 {
                    self.tile_sel_glitch = true;
                }
                self.write_lcdc(v);
            }
            Act::CgbDoubleLcdc(v) => {
                let old = self.lcdc();
                self.tile_sel_glitch = (v ^ old) & TILE_SEL != 0;
                self.write_lcdc((v & !(BG_EN | 0x80)) | (old & (BG_EN | 0x80)));
            }
            Act::WxJustChanged(b) => self.wx_just_changed = b,
            Act::TileSelGlitch(b) => self.tile_sel_glitch = b,
        }
    }

    /// An LCDC write as the register sees it (SameBoy `GB_write_memory`, FF40): on a DMG,
    /// clearing OBJ_EN during an object fetch aborts it.
    fn write_lcdc(&mut self, v: u8) {
        if !self.cgb && self.lcdc() & OBJ_EN != 0 && v & OBJ_EN == 0 && self.during_object_fetch {
            self.object_fetch_aborted = true;
        }
        self.regs[LCDC as usize] = v;
    }

    fn vram(&self, addr: u16) -> u8 {
        if self.blocked { 0xFF } else { self.vram[addr as usize & 0x3FFF] }
    }

    fn run(&mut self) {
        self.sleep(5);
        self.bg.clear();
        self.obj.clear();
        // Eight pixels of junk, dropped as the fine scroll is applied.
        self.bg.push_bg_row(0, 0, 0, false, false);
        self.step = Step::TileT1;
        loop {
            self.window();
            if self.objects_and_pixel() {
                break;
            }
            self.sleep(1);
        }
    }

    /// The window starts when the line reaches WX (SameBoy, "Handle window").
    fn window(&mut self) {
        let (wx, scx) = (self.reg(WX), self.reg(SCX));
        let pos = self.pos;
        if !self.wx_triggered && self.wy_triggered && self.lcdc() & WIN_EN != 0 {
            let mut activate = false;
            if wx == 0 {
                activate = pos == (-7i8) as u8 || (pos == (-16i8) as u8 && scx & 7 != 0) || ((-15i8) as u8..=(-8i8) as u8).contains(&pos);
            } else if (wx as u16) < 166 + self.cgb as u16 {
                if wx == pos.wrapping_add(7) {
                    activate = true;
                } else if !self.cgb && wx == pos.wrapping_add(6) && !self.wx_just_changed {
                    activate = true;
                }
            }
            if activate {
                self.window_y = self.window_y.wrapping_add(1);
                self.window_tile_x = 0;
                self.bg.clear();
                if wx == 0 && scx & 7 != 0 && !self.cgb {
                    self.sleep(1);
                }
                self.wx_triggered = true;
                self.step = Step::TileT1;
                self.window_is_being_fetched = true;
            } else if !self.cgb && wx == 166 && wx == pos.wrapping_add(7) {
                self.window_y = self.window_y.wrapping_add(1);
            }
        }
        let wx = self.reg(WX);
        if wx == self.pos.wrapping_add(7) && (!self.cgb || wx == 0) && self.wx_triggered && !self.window_is_being_fetched && self.step == Step::TileT1 && self.bg.size == 8 {
            self.insert_bg_pixel = true;
        }
    }

    fn x_for_object_match(&self) -> u8 {
        let x = self.pos.wrapping_add(8);
        if x > (-16i8) as u8 { 0 } else { x }
    }

    fn objects_enabled(&self) -> bool {
        self.lcdc() & OBJ_EN != 0 || self.cgb
    }

    /// Fetches every object at this X, pushes one pixel and steps the fetcher. True when
    /// the line's last pixel is out.
    fn objects_and_pixel(&mut self) -> bool {
        while self.objects.last().is_some_and(|o| o.1 < self.x_for_object_match()) {
            self.objects.pop();
        }
        self.during_object_fetch = true;
        while let Some(&(index, _, y)) = self.objects.last().filter(|o| self.objects_enabled() && o.1 == self.x_for_object_match()) {
            if !self.fetch_object(index, y) {
                break;
            }
            self.objects.pop();
        }
        self.object_fetch_aborted = false;
        self.during_object_fetch = false;
        self.render_pixel();
        self.advance_fetcher();
        self.pos == WIDTH as u8
    }

    /// One object's fetch, the background fetcher finishing its tile first. False when the
    /// CPU aborted it.
    fn fetch_object(&mut self, index: u8, y: u8) -> bool {
        while self.step < Step::HighT2 || self.bg.size == 0 {
            self.advance_fetcher();
            self.sleep(1);
            if self.object_fetch_aborted {
                return false;
            }
        }
        self.advance_fetcher();
        self.sleep(1);
        if self.object_fetch_aborted {
            return false;
        }
        self.advance_fetcher();
        let tile = self.oam[index as usize * 4 + 2];
        let flags = self.oam[index as usize * 4 + 3];
        self.sleep(2);
        if self.object_fetch_aborted {
            return false;
        }
        let low = self.vram(self.object_line_address(y, tile, flags));
        self.sleep(2);
        if self.object_fetch_aborted {
            return false;
        }
        self.during_object_fetch = false;
        let address = self.object_line_address(y, tile, flags);
        let high = self.vram(address + 1);
        self.sleep(1);
        let palette = if self.cgb_mode() { flags & 7 } else { flags >> 4 & 1 };
        let priority = if self.cgb_mode() { index } else { 0 };
        self.obj.overlay_object_row(low, high, palette, flags & 0x80 != 0, priority, flags & 0x20 != 0);
        self.data_for_sel_glitch = self.vram(address + 1);
        true
    }

    fn object_line_address(&self, y: u8, tile: u8, flags: u8) -> u16 {
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

    /// SameBoy `render_pixel_if_possible`.
    fn render_pixel(&mut self) {
        if self.objects_enabled() && self.objects.last().is_some_and(|o| o.1 == 0) {
            return;
        }
        if self.bg.size == 0 {
            return;
        }
        let item = if std::mem::take(&mut self.insert_bg_pixel) { Item::default() } else { self.bg.pop() };
        let mut bg_priority = item.bg_priority;
        let mut sprite = None;
        if self.obj.size > 0 {
            let o = self.obj.pop();
            if o.pixel != 0 && self.lcdc() & OBJ_EN != 0 {
                sprite = Some(o);
                bg_priority |= o.bg_priority;
            }
        }
        let scx = self.reg(SCX);
        if self.pos.wrapping_add(16) < 8 {
            if self.pos & 7 == scx & 7 {
                self.pos = (-8i8) as u8;
            } else if self.window_is_being_fetched && self.pos & 7 == 6 && scx & 7 == 7 {
                self.pos = (-8i8) as u8;
            } else if self.pos == (-9i8) as u8 {
                self.pos = (-16i8) as u8;
                return;
            } else {
                self.line_has_fractional_scrolling = true;
            }
        }
        self.window_is_being_fetched = false;
        if self.pos >= WIDTH as u8 {
            self.pos = self.pos.wrapping_add(1);
            return;
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
        let palettes: &[u8; PAL_REGS as usize] = self.regs[..PAL_REGS as usize].try_into().expect("the palettes lead the display bytes");
        let colour = if self.blocked { 0 } else { colour_in(palettes, self.mode, px) };
        if let Some(out) = self.out.get_mut(self.lcd_x as usize) {
            *out = colour;
        }
        self.pos = self.pos.wrapping_add(1);
        self.lcd_x = self.lcd_x.wrapping_add(1);
    }

    fn fetcher_y_now(&self) -> u8 {
        if self.wx_triggered { self.window_y } else { self.ly.wrapping_add(self.reg(SCY)) }
    }

    /// SameBoy `GB_update_wx_glitch`.
    fn update_wx_glitch(&mut self) {
        if !self.cgb {
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
    fn advance_fetcher(&mut self) {
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
                    let sum = self.reg(SCX) as i32 + self.pos as i32 + 8 - (self.cgb && !self.during_object_fetch) as i32;
                    (sum / 8 & 0x1F) as u8
                };
                // CGB D and later cache the row for the whole fetch (SameBoy: model > CGB C).
                if self.cgb {
                    self.fetcher_y = y;
                }
                self.tile_index_address = map + x as u16 + y as u16 / 8 * 32;
                self.step = Step::TileT2;
            }
            Step::TileT2 => {
                if !self.cgb_wx_glitch {
                    self.tile = self.vram(self.tile_index_address);
                    self.attr = if self.cgb_mode() { self.vram(self.tile_index_address + 0x2000) } else { 0 };
                }
                self.step = Step::LowT1;
            }
            Step::LowT1 => {
                self.update_wx_glitch();
                if !self.cgb {
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
                    self.data[0] = glitched.unwrap_or_else(|| self.vram(self.tile_data_address));
                    if self.last_tileset && self.tile_sel_glitch {
                        self.data_for_sel_glitch = self.vram(self.tile_data_address);
                    }
                }
                self.step = Step::HighT1;
            }
            Step::HighT1 => {
                self.update_wx_glitch();
                if !self.cgb {
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
                            self.data[1] = self.vram(self.tile_data_address);
                            self.data_for_sel_glitch = self.data[1];
                        }
                    }
                    if self.last_tileset && self.tile_sel_glitch {
                        self.data_for_sel_glitch = self.vram(self.tile_data_address);
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
        if self.wy_triggered && self.lcdc() & WIN_EN == 0 && !self.cgb && !self.disable_window_pixel_insertion_glitch {
            // SameBoy issue 278: a DMG with the window disabled at WX inserts one pixel.
            let mut logical = self.pos.wrapping_add(7);
            if logical > 167 {
                logical = 0;
            }
            if self.reg(WX) == logical {
                self.bg.read = (self.bg.read + 7) & 7;
                self.bg.items[self.bg.read] = Item::default();
                self.bg.size = 1;
                return;
            }
        }
        self.bg.push_bg_row(self.data[0], self.data[1], self.attr & 7, self.attr & 0x80 != 0, self.attr & 0x20 != 0);
        self.step = Step::TileT1;
    }

    /// The window's state after the line (SameBoy, the end of mode 3).
    fn window_after(&self) -> Window {
        let mut window_y = self.window_y;
        let mut carried = false;
        if !self.cgb && self.wy_triggered && self.lcdc() & WIN_EN != 0 && self.reg(WX) == 166 {
            carried = true;
            window_y = window_y.wrapping_add(1);
        }
        // SameBoy resets the rows as line 143 ends; the PPU resets them at VBlank
        // (`Ppu::end_frame`), which no line comes between.
        Window { rows: window_y.wrapping_add(1), armed: self.wy_triggered, carried }
    }
}

#[test]
fn the_state_machine_draws_every_line_as_the_loop_did() {
    let mut rng = Rng(0x0DDB_A11C_0FFE_E123);
    let mut covered = [0; PATHS.len()];
    let (mut passes, mut most) = (0, 0);
    const LINES: usize = 30_000;
    for case in 0..LINES {
        let sample = Sample::random(&mut rng, case);
        let (start, writes) = (sample.start(), &sample.writes);
        let want = draw(&start, writes);
        assert_eq!(super::draw(&start, writes), want, "line {case}: {:?} double {} {:?} objects {:?} writes {writes:?}", start.mode, start.double, start.window, start.objects);
        let trace = trace(&start, writes);
        assert!(trace.last().is_some_and(|p| p.out.done), "line {case} ends done");
        for (n, (_, took)) in covered.iter_mut().zip(PATHS) {
            *n += took(&trace, &want) as usize;
        }
        passes += trace.len();
        most = most.max(trace.len());
    }
    println!("{LINES} lines: passes mean {:.1}, max {most}", passes as f64 / LINES as f64);
    for ((name, _), n) in PATHS.iter().zip(covered) {
        println!("{n:6} lines: {name}");
    }
    assert!(covered.iter().all(|&n| n > 0), "a path no line took: {covered:?}");
}

type Path = (&'static str, fn(&[Pass], &Drawn) -> bool);

fn aborted(trace: &[Pass], resume: Phase) -> bool {
    trace.windows(2).any(|p| p[0].before.phase == Phase::Sleep && p[0].before.resume == resume && p[1].before.phase == Phase::Render)
}

const PATHS: [Path; 12] = [
    ("DMG object fetch aborted waiting on the fetcher", |t, _| aborted(t, Phase::FetchWait)),
    ("DMG object fetch aborted before its tile", |t, _| aborted(t, Phase::FetchB)),
    ("DMG object fetch aborted before its low byte", |t, _| aborted(t, Phase::FetchC)),
    ("DMG object fetch aborted before its high byte", |t, _| aborted(t, Phase::FetchD)),
    ("a second object fetched at one X", |t, _| t.windows(2).any(|p| p[0].before.phase == Phase::Match { first: false } && p[1].before.phase == Phase::FetchWait)),
    ("the window started after the DMG's extra dot at WX 0", |t, _| t.iter().any(|p| p.before.phase == Phase::WindowOn)),
    ("a background pixel inserted at the window's WX", |t, _| t.iter().any(|p| p.before.insert_bg_pixel)),
    ("the CGB's WX glitch", |t, _| t.iter().any(|p| p.before.cgb_wx_glitch)),
    ("the CGB's tile-select glitch", |t, _| t.iter().any(|p| p.before.tile_sel_glitch)),
    ("the window armed at WX below 7 or at 166", |t, _| t[0].before.wy_triggered && (t[0].before.reg(WX) < 7 || t[0].before.reg(WX) == 166)),
    ("a pixel inserted at WX with the window off (issue 278)", |t, _| t.windows(2).any(|p| p[0].before.bg.size == 0 && p[1].before.bg.size == 1)),
    ("a line carried in the window from WX 166", |t, _| t[0].before.wx_triggered),
];
