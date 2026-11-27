//! The newest byte per physical address of the fixed layout: what the model's memory
//! tokens hold, as a byte array. It is the engine's index (the attention result the
//! margin argument proves), the circuit core's memory, and the snapshot a compaction or
//! a mid-run insertion loads from the host.

use super::layout::{self, Banking, BLOCK, FF, HOST, HRAM, HRAM_LEN, IO, IO_BLOCKS, IO_UNMAPPED, LINE, LINES, LINE_BITS, OAM, OAM_LEN, PAL, PAL_LEN, REGS, TAB, TAB_BLOCKS, TOTAL, VOID, VRAM, VRAM_LEN, WRAM, WRAM_LEN};

#[derive(Clone)]
pub struct Image {
    bytes: Vec<u8>,
    /// Whether a token exists at each address: a query must never name an address
    /// without one.
    present: Vec<bool>,
    rom_len: usize,
    ram_len: usize,
    /// The palettes' bytes at `PAL`: none but in the PPU checkpoint's image.
    pal_len: usize,
    /// Bytes overwritten since `mark` or `revert`, oldest first. None until the first
    /// of them. A running game never reverts, so it keeps no log.
    undo: Option<Vec<(u32, u8)>>,
    /// Counts every change to `bytes`, so a reader can tell whether a byte it holds is
    /// still the one at its address.
    generation: u64,
}

impl Image {
    /// An image for a cartridge: the ROM, zero RAM, the marker blocks and the registers.
    pub fn new(rom: &[u8], ram_len: usize, banking: Banking) -> Image {
        let mut i = Image { bytes: vec![0; TOTAL as usize], present: vec![false; TOTAL as usize], rom_len: rom.len(), ram_len, pal_len: 0, undo: None, generation: 0 };
        i.bytes[..rom.len()].copy_from_slice(rom);
        i.present[..rom.len()].fill(true);
        for (base, len) in i.ram_regions() {
            i.present[base as usize..base as usize + len].fill(true);
        }
        i.bytes[FF as usize..(FF + BLOCK) as usize].fill(0xFF);
        for block in [IO + IO_UNMAPPED * BLOCK, layout::BLOCKED] {
            i.bytes[block as usize..(block + BLOCK) as usize].fill(0xFF);
        }
        for bank in 0..TAB_BLOCKS {
            let block = (TAB + bank * BLOCK) as usize;
            i.bytes[block + 0x4F] = 0xFE | (bank & 1) as u8;
            i.bytes[block + 0x70] = 0xF8 | bank as u8;
        }
        for r in 0..layout::N_REGS {
            i.bytes[REGS as usize + r] = banking.0[r];
        }
        i
    }

    /// Every region besides the ROM with its length: cartridge RAM, WRAM, VRAM, OAM, HRAM,
    /// the FF and VOID blocks, the registers, the palettes, the host's fetch bytes, and
    /// the I/O page's blocks and the bank readback table, which hold constants. A
    /// compaction snapshots exactly these.
    pub fn ram_regions(&self) -> [(u32, usize); 13] {
        [(layout::CRAM, self.ram_len), (WRAM, WRAM_LEN), (VRAM, VRAM_LEN), (OAM, OAM_LEN), (HRAM, HRAM_LEN), (FF, BLOCK as usize), (VOID, BLOCK as usize), (REGS, layout::N_REGS), (PAL, self.pal_len), (HOST, BLOCK as usize), (IO, (IO_BLOCKS * BLOCK) as usize), (TAB, (TAB_BLOCKS * BLOCK) as usize), (layout::BOOT, layout::BOOT_LEN)]
    }

    pub fn rom_len(&self) -> usize {
        self.rom_len
    }

    /// The image as data: every byte, whether each address has a token, and the ROM, RAM
    /// and palette lengths (`wiring::MemoryFile`).
    pub fn parts(&self) -> (&[u8], &[bool], [usize; 3]) {
        (&self.bytes, &self.present, [self.rom_len, self.ram_len, self.pal_len])
    }

    /// The image [`Image::parts`] gave.
    pub fn from_parts(bytes: Vec<u8>, present: Vec<bool>, [rom_len, ram_len, pal_len]: [usize; 3]) -> Result<Image, String> {
        if bytes.len() != TOTAL as usize || present.len() != TOTAL as usize {
            return Err(format!("an image of {} bytes and {} flags, not {TOTAL}", bytes.len(), present.len()));
        }
        Ok(Image { bytes, present, rom_len, ram_len, pal_len, undo: None, generation: 0 })
    }

    /// The same image with the ROM region writable, for a harness whose program bytes
    /// live anywhere on the bus (SingleStepTests). Nothing is a ROM prefix then.
    pub fn with_writable_rom(mut self) -> Image {
        self.rom_len = 0;
        self
    }

    /// The same image with tokens for the palettes, the PPU checkpoint's.
    pub fn with_palettes(mut self) -> Image {
        self.pal_len = PAL_LEN;
        self.present[PAL as usize..PAL as usize + PAL_LEN].fill(true);
        self
    }

    pub fn has(&self, p: u32) -> bool {
        (p as usize) < self.present.len() && self.present[p as usize]
    }

    /// The byte at `p`, which must hold a token.
    pub fn read(&self, p: u32) -> u8 {
        assert!(self.has(p), "no memory token at {p:#08x}");
        self.bytes[p as usize]
    }

    /// Sets the byte at `p`, which must hold a token outside the ROM.
    pub fn write(&mut self, p: u32, v: u8) {
        assert!(self.has(p) && p as usize >= self.rom_len, "write to {p:#08x}, which holds no writable token");
        if let Some(undo) = &mut self.undo {
            undo.push((p, self.bytes[p as usize]));
        }
        self.bytes[p as usize] = v;
        self.generation += 1;
    }

    /// Whether line `n` holds a cell, which it does when its first byte holds a token.
    /// Every region starts on a line, so a partly used last line still has its first byte.
    pub fn has_line(&self, n: u32) -> bool {
        self.has(n << LINE_BITS)
    }

    /// Line `n`'s bytes, 0 where a byte of the line holds no token.
    pub fn line(&self, n: u32) -> [u8; LINE] {
        assert!(self.has_line(n), "no memory cell for line {n:#07x} ({:#08x})", n << LINE_BITS);
        let base = (n as usize) << LINE_BITS;
        std::array::from_fn(|k| if self.present[base + k] { self.bytes[base + k] } else { 0 })
    }

    /// Replaces line `n` with a write record's cell. Bytes without a token must stay 0,
    /// and a byte that changes must be writable.
    pub fn write_line(&mut self, n: u32, line: [u8; LINE]) {
        let base = n << LINE_BITS;
        for (k, &v) in line.iter().enumerate() {
            let p = base + k as u32;
            if !self.has(p) {
                assert_eq!(v, 0, "line {n:#07x}: byte {k} holds no token and must stay 0");
            } else if self.bytes[p as usize] != v {
                self.write(p, v);
            }
        }
    }

    /// The number of lines that hold a cell, which is the number of the cache's cells.
    pub fn lines(&self) -> usize {
        (0..LINES).filter(|&n| self.has_line(n)).count()
    }

    /// Changes so far to any byte; equal generations hold equal bytes everywhere.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn banking(&self) -> Banking {
        let mut b = Banking([0; layout::N_REGS]);
        b.0.copy_from_slice(&self.bytes[REGS as usize..REGS as usize + layout::N_REGS]);
        b
    }

    /// Starts a fresh undo log: writes before this point stay, writes after it can be
    /// reverted.
    pub fn mark(&mut self) {
        self.undo = Some(Vec::new());
    }

    /// The writes since `mark`, oldest first, with the byte each replaced.
    pub fn undo_log(&self) -> &[(u32, u8)] {
        self.undo.as_deref().unwrap_or_default()
    }

    /// Undoes every write since `mark` or the last `revert`, and logs from here on.
    pub fn revert(&mut self) {
        for (p, v) in self.undo.take().into_iter().flatten().rev() {
            self.bytes[p as usize] = v;
        }
        self.mark();
        self.generation += 1;
    }

    /// The bytes of one writable region, for snapshots.
    pub fn region(&self, base: u32, len: usize) -> &[u8] {
        &self.bytes[base as usize..base as usize + len]
    }

    /// Copies a region's bytes in from the host, for insertion mid-run.
    pub fn fill(&mut self, base: u32, bytes: &[u8]) {
        assert!(self.present[base as usize..base as usize + bytes.len()].iter().all(|&p| p), "filling {base:#08x} beyond the tokens");
        self.bytes[base as usize..base as usize + bytes.len()].copy_from_slice(bytes);
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_keep_an_undo_log_only_after_a_mark() {
        let mut i = Image::new(&[0; 0x8000], 0, Banking([0; layout::N_REGS]));
        for k in 0..10_000u32 {
            i.write(WRAM + k % 0x100, k as u8);
        }
        assert_eq!(i.undo_log().len(), 0, "a game's writes are never reverted, so none are logged");
        i.mark();
        i.write(WRAM, 0xAA);
        i.write(WRAM, 0xBB);
        assert_eq!(i.undo_log(), [(WRAM, 0x00), (WRAM, 0xAA)]);
        i.revert();
        assert_eq!((i.read(WRAM), i.undo_log().len()), (0x00, 0));
        i.write(WRAM + 1, 0xCC);
        i.revert();
        assert_eq!(i.read(WRAM + 1), 0x01, "a revert keeps logging, so the next one undoes what followed it");
    }
}
