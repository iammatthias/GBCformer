//! OAM DMA (FF46) and CGB VRAM DMA (HDMA1-5) as devices. Neither touches memory itself:
//! each asks the bus for copies through [`Signals`], and HDMA adds the M-cycles the CPU
//! is stopped for.

use crate::device::{Copy, Device, Signals};

/// OAM DMA: 160 bytes from `page << 8` to OAM, one per M-cycle. A write to FF46 starts the
/// transfer two M-cycles later: the M-cycle after the write still reaches OAM, the next
/// one is the first byte's (mooneye `oam_dma_start`). A write while a transfer blocks keeps
/// it blocking through those two M-cycles, though no byte moves then (SameBoy v1.0.5,
/// `dma_restarting`).
#[derive(Clone)]
pub struct OamDma {
    reg: u8,
    /// A copy in progress: its source page's address (None: a CGB's E000-FFFF, which
    /// writes 0xFF) and the next byte's index, 0xFE and 0xFF for the two M-cycles before
    /// byte 0.
    active: Option<(Option<u16>, u8)>,
    /// The write came while a transfer blocked, so the two M-cycles before byte 0 block too.
    restart: bool,
    cgb: bool,
}

/// The index FF46's write starts a transfer at: two M-cycles before byte 0.
pub const DMA_START: u8 = 0xFE;

impl OamDma {
    pub fn post_boot(cgb: bool) -> Self {
        OamDma { reg: if cgb { 0x00 } else { 0xFF }, active: None, restart: false, cgb }
    }

    /// The transfer still has M-cycles to run, its start included.
    pub fn active(&self) -> bool {
        self.active.is_some()
    }

    /// The transfer blocks the GBC CPU's accesses on its buses (`layout::Dma`): while its
    /// bytes move, and before byte 0 only when it restarted one that blocked.
    pub fn blocking(&self) -> bool {
        self.active.is_some_and(|(_, i)| i < 0xA0 || self.restart)
    }

    /// Whether it blocks `mcycles` M-cycles from now.
    pub fn blocking_in(&self, mcycles: u64) -> bool {
        let Some((_, i)) = self.active else { return false };
        let index = if i >= DMA_START {
            let to_first = 256 - i as u64;
            if mcycles < to_first {
                return self.restart;
            }
            mcycles - to_first
        } else {
            i as u64 + mcycles
        };
        index < 0xA0
    }

    /// FF46, the next byte's index (160 when no copy runs) and whether the start restarted
    /// a blocking transfer.
    pub fn state(&self) -> (u8, u8, bool) {
        (self.reg, self.active.map_or(160, |(_, i)| i), self.restart)
    }
}

impl Device for OamDma {
    fn read(&self, _addr: u16) -> u8 {
        self.reg
    }

    fn write(&mut self, _addr: u16, value: u8, _s: &mut Signals) {
        self.reg = value;
        // Sources in E000-FFFF read the echo of work RAM on DMG; a CGB writes 0xFF
        // (SameBoy's `GB_dma_run`, mooneye `oam_dma/sources-GS` for DMG).
        let source = match value {
            0xE0.. if self.cgb => None,
            0xE0.. => Some(((value - 0x20) as u16) << 8),
            _ => Some((value as u16) << 8),
        };
        self.restart = self.blocking();
        self.active = Some((source, DMA_START));
    }

    /// A byte moved before the block's last M-cycle lands before a line that M-cycle
    /// starts is drawn; the last M-cycle's byte lands after it.
    fn advance(&mut self, mcycles: u64, s: &mut Signals) {
        for k in 0..mcycles {
            let Some((source, i)) = self.active else { return };
            if i >= DMA_START {
                let next = i.wrapping_add(1);
                self.restart &= next != 0;
                self.active = Some((source, next));
                continue;
            }
            let dest = 0xFE00 | i as u16;
            if k + 1 < mcycles { &mut s.early } else { &mut s.copies }.push(match source {
                Some(page) => Copy::Byte { source: page | i as u16, dest },
                None => Copy::Fill { value: 0xFF, dest },
            });
            self.active = (i + 1 < 0xA0).then_some((source, i + 1));
        }
    }

    /// One byte per M-cycle must land before the PPU's next M-cycle can read OAM.
    fn horizon(&self) -> u64 {
        if self.active() { 1 } else { u64::MAX }
    }
}

/// CPU M-cycles one 16-byte HDMA block stops the CPU for: 8 µs, 8 M-cycles at normal
/// speed and 16 at double speed (Pan Docs, "CGB Registers", VRAM DMA).
pub fn block_mcycles(double: bool) -> u64 {
    if double { 16 } else { 8 }
}

/// M-cycles each run of the copier stops the CPU for beside its blocks: one HBlank block, a
/// block at a HALT's wake, or a whole general-purpose transfer is one run. SameBoy v1.0.5
/// `GB_hdma_run` advances 2 T-cycles before the first byte and 2 after the last at normal
/// speed, 4 before at double, so a run of n blocks is 8n + 1 or 16n + 1 M-cycles (Mealybug
/// `dma/hdma_timing-C`'s DIV reads).
pub const RUN_MCYCLES: u64 = 1;

/// CGB VRAM DMA: general-purpose copies run at the HDMA5 write; HBlank copies move one
/// block each time a visible line enters mode 0.
#[derive(Clone)]
pub struct Hdma {
    source: u16,
    /// Offset into VRAM, 0000-1FF0.
    dest: u16,
    /// An HBlank transfer with this many blocks left after the next one.
    hblank_remaining: Option<u8>,
    /// HDMA5 when no HBlank transfer is active: 0xFF after a completed transfer,
    /// 0x80 | the count written by the write that cancelled one.
    idle_status: u8,
    /// An HBlank block moved inside a GBC CPU step and HDMA5 does not count it yet, with
    /// whether that block ran off the end of VRAM: SameBoy v1.0.5 counts a block in
    /// `GB_hdma_run`, after the next opcode fetch, so a read in the step that saw the
    /// request still reads the count before it (Mealybug `dma/hdma_timing-C`). It counts as
    /// the step ends ([`Hdma::count`]). Whether the transfer ends is the block's own fact,
    /// kept from the block (SameBoy decides it in the same run, from the destination the
    /// block left): a write to HDMA3 or HDMA4 later in the step does not change it.
    uncounted: Option<bool>,
    /// This line's block moved before its HBlank request: an HBlank transfer started, or a
    /// HALT woke, while STAT already read mode 0 and the request was still to come. That
    /// request then moves no second block (`Control`, `Lcd::in_gap`).
    pub line_done: bool,
}

impl Hdma {
    pub fn new() -> Self {
        Hdma { source: 0, dest: 0, hblank_remaining: None, idle_status: 0xFF, uncounted: None, line_done: false }
    }

    /// Asks for one 16-byte block, adding the M-cycles it stops the CPU for at the speed
    /// `double` names to `owed`; false if the destination ran off the end of VRAM.
    fn block(&mut self, double: bool, s: &mut Signals, owed: &mut u64) -> bool {
        s.copies.push(Copy::Block { source: self.source, dest: 0x8000 | self.dest, len: 16 });
        self.source = self.source.wrapping_add(16);
        self.dest += 16;
        *owed += block_mcycles(double);
        if self.dest >= 0x2000 {
            self.dest &= 0x1FF0;
            return false;
        }
        true
    }

    /// The source, the destination (offset into VRAM), HDMA5 as it reads, whether a block
    /// waits to be counted, whether this line's block moved before its request, and whether
    /// the block that waits ran off the end of VRAM.
    pub fn state(&self) -> (u16, u16, u8, bool, bool, bool) {
        (self.source, self.dest, self.read(0xFF55), self.uncounted.is_some(), self.line_done, self.uncounted == Some(true))
    }

    /// An HBlank transfer runs.
    pub fn hblank_active(&self) -> bool {
        self.hblank_remaining.is_some()
    }

    /// The HBlank request reached the GBC CPU: an HBlank transfer moves its next block, and
    /// counts it in HDMA5 now or, `held` in a step, as the step ends.
    pub fn hblank(&mut self, double: bool, held: bool, s: &mut Signals, owed: &mut u64) {
        if let Some(left) = self.hblank_remaining {
            *owed += RUN_MCYCLES;
            let ended = !self.block(double, s, owed);
            if held {
                // A step spans one HBlank request at most: its in-step M-cycles are at most 6
                // (CALL, an interrupt dispatch's 5) and two requests are a line apart, 114
                // M-cycles at normal speed and 228 at double. The control circuit's `HP'`
                // counts one block a step on the same premise, which this holds.
                assert!(self.uncounted.is_none(), "refused: a GBC CPU step spans two HBlank requests, and HDMA5 counts one held block a step");
                self.uncounted = Some(ended);
            } else {
                self.counted(left, ended);
            }
        }
    }

    /// Counts a block moved in a step that has ended, ending the transfer if that block ran
    /// off the end of VRAM.
    pub fn count(&mut self) {
        if let Some(ended) = self.uncounted.take()
            && let Some(left) = self.hblank_remaining
        {
            self.counted(left, ended);
        }
    }

    fn counted(&mut self, left: u8, ended: bool) {
        self.hblank_remaining = left.checked_sub(1);
        if ended || left == 0 {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.hblank_remaining = None;
        self.idle_status = 0xFF;
    }

    /// An HBlank transfer started while STAT reads mode 0 (the LCD off, or a visible
    /// line's HBlank) moves its first block at once (SameSuite `dma/hdma_lcd_off`,
    /// `hdma_mode0`; SameBoy's HDMA5 write).
    fn write_control(&mut self, value: u8, double: bool, mode0: bool, s: &mut Signals, owed: &mut u64) {
        let blocks = value & 0x7F;
        match (self.hblank_remaining, value & 0x80 != 0) {
            // A cancel reads back the count it wrote, not the blocks left (SameSuite
            // `dma/hdma_lcd_off`, `hdma_mode0`; SameBoy sets the count before it cancels).
            (Some(_), false) => {
                self.hblank_remaining = None;
                self.idle_status = 0x80 | blocks;
            }
            (_, true) => {
                self.hblank_remaining = Some(blocks);
                if mode0 {
                    self.hblank(double, false, s, owed);
                }
            }
            (None, false) => {
                // Every block the destination has room for, one copy each, in one run.
                *owed += RUN_MCYCLES;
                for _ in 0..=blocks {
                    if !self.block(double, s, owed) {
                        break;
                    }
                }
                self.finish();
            }
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF55 => self.hblank_remaining.unwrap_or(self.idle_status),
            _ => 0xFF,
        }
    }

    /// A write to HDMA1-5; `double` is the CPU's speed, which sets the M-cycles a
    /// general-purpose copy stops it for, added to `owed`, and `mode0` says STAT reads mode 0.
    pub fn write(&mut self, addr: u16, value: u8, double: bool, mode0: bool, s: &mut Signals, owed: &mut u64) {
        match addr {
            0xFF51 => self.source = (value as u16) << 8 | self.source & 0x00F0,
            0xFF52 => self.source = self.source & 0xFF00 | (value & 0xF0) as u16,
            0xFF53 => self.dest = ((value & 0x1F) as u16) << 8 | self.dest & 0x00F0,
            0xFF54 => self.dest = self.dest & 0x1F00 | (value & 0xF0) as u16,
            // The write sets the count: a block the step moved before it is not counted.
            0xFF55 => {
                self.uncounted = None;
                self.write_control(value, double, mode0, s, owed)
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An HBlank transfer of three blocks whose destination is `dest`, with the request
    /// landing inside a GBC CPU step (the block held), then `rewrite` written to HDMA3 in the
    /// same step, then the step's end. Returns HDMA5 after the count.
    fn held_block_then_dest_write(dest: u16, rewrite: u8) -> u8 {
        let (mut h, mut s, mut owed) = (Hdma::new(), Signals::default(), 0);
        h.write(0xFF53, (dest >> 8) as u8, false, false, &mut s, &mut owed);
        h.write(0xFF54, dest as u8, false, false, &mut s, &mut owed);
        h.write(0xFF55, 0x82, false, false, &mut s, &mut owed);
        h.hblank(false, true, &mut s, &mut owed);
        h.write(0xFF53, rewrite, false, false, &mut s, &mut owed);
        h.count();
        h.read(0xFF55)
    }

    /// Review 3, finding 3: a held block that runs off the end of VRAM (offset 0x1FF0 wraps to
    /// 0) ends the transfer at the step's end even when HDMA3 is written in the step after it.
    #[test]
    fn a_held_block_off_the_end_of_vram_ends_the_transfer_after_a_destination_write() {
        assert_eq!(held_block_then_dest_write(0x1FF0, 0x01), 0xFF);
    }

    /// And a held block that did not run off does not end it when the destination is written
    /// to 0 in the step after it: one block counted, one left after the next.
    #[test]
    fn a_held_block_inside_vram_keeps_the_transfer_after_a_destination_write_to_0() {
        assert_eq!(held_block_then_dest_write(0x00F0, 0x00), 0x01);
    }
}
