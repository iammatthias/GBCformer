//! The fixed 24-bit physical address space the model's memory tokens live in
//! (`docs/memory-in-weights.md` §4), and the specification of the address translation
//! the GBC CPU circuit performs: bus address plus the banking registers' bytes to a
//! physical address. [`translate`] is plain Rust so it can be checked against the host
//! ([`crate::machine::Machine::physical`]) and so the circuit (`sm83::translate`) can be
//! checked against it.
//!
//! Region bases are constants and bus bits 0..6 pass through unchanged in every read
//! region, so the circuit needs no selection below bit 7. The marker blocks are 128
//! bytes for the same reason: a read of disabled cartridge RAM lands at `FF | (a & 0x7F)`,
//! which holds 0xFF, one of the unusable area in `VOID`, which holds 0, and an access a
//! host device serves in that device's port block of [`IO`], whose high bits name the
//! device ([`port`]) and which holds 0.
//!
//! The banking registers are memory bytes at [`REGS`]. They hold the byte the game
//! wrote, uninterpreted; the translation interprets them the way the mapper does
//! (`max(v & 0x1F, 1)`, `v & 0x0F == 0x0A`), so a register write is an ordinary write
//! record and a compaction snapshots it like any other byte.

pub const ROM: u32 = 0x00_0000;
pub const CRAM: u32 = 0x80_0000;
pub const WRAM: u32 = 0x82_0000;
pub const VRAM: u32 = 0x82_8000;
pub const OAM: u32 = 0x82_C000;
pub const HRAM: u32 = 0x82_C100;
/// 128 bytes of 0xFF: disabled cartridge RAM, unmapped ROM.
pub const FF: u32 = 0x82_C180;
/// 128 bytes of 0: the unusable area's reads, the line an idle read port rests on, and
/// writes that reach nothing (ROM without a register, the unusable area). A write that
/// lands here is not recorded.
pub const VOID: u32 = 0x82_C200;
/// The banking registers, one byte each in [`Reg`] order.
pub const REGS: u32 = 0x82_C280;
/// The PPU checkpoint's display bytes, every byte a line reads besides VRAM and OAM:
/// CGB background palette RAM (64 bytes), object palette RAM (64), BGP, OBP0 and OBP1,
/// then the display registers LCDC, SCY, SCX, WY and WX. Only the PPU's own image holds
/// tokens here (`Image::with_palettes`); no bus address translates to it.
pub const PAL: u32 = 0x82_C300;
pub const PAL_LEN: usize = 136;
/// Where BGP sits in [`PAL`]; OBP0 and OBP1 follow.
pub const PAL_DMG: u32 = 128;
/// Where the display registers start in [`PAL`]: on BGP's line, after OBP1.
pub const PAL_REGS: u32 = 131;

/// The index in [`PAL`] of a display byte's bus address: BGP, OBP0, OBP1 or a display
/// register. The CGB palette RAM bytes have no bus address of their own. It is the
/// display port's code ([`io_code`]) as the host decodes it ([`port`]).
#[cfg(feature = "gbc")]
pub fn display_index(addr: u16) -> Option<u8> {
    let code = (0xFF00..0xFF80).contains(&addr).then(|| io_code(addr as u8, false, false))?;
    (code >= IO_DISPLAY).then_some(PAL_DMG as u8 | (code & 7) as u8)
}
/// 128 bytes the host writes: an opcode byte the CPU fetches from an address the host
/// serves (the I/O page, IE, the unusable area, the MBC3 clock) is kept here as a
/// labelled record at `HOST | (a & 0x7F)` before the step's pass, so the fetch heads
/// always read memory. Only the fetch translation lands here ([`fetch`]).
pub const HOST: u32 = 0x82_C400;
/// The I/O page's 16 blocks, `IO + code·128 + (a & 0x7F)`: block 0 holds 0xFF (unmapped
/// registers), the others are ports, zero bytes the host serves from the device the code
/// names ([`port`]). Codes are the `IO_*` constants.
pub const IO: u32 = 0x82_C800;
pub const IO_BLOCKS: u32 = 16;
pub const IO_UNMAPPED: u32 = 0;
/// The timer, IF, IE (at low bits 0x7F), P1, serial, OAM DMA, STAT, LY, LYC and on CGB
/// KEY0, KEY1, HDMA and the palette index and data ports.
pub const IO_CONTROL: u32 = 1;
pub const IO_APU: u32 = 2;
/// 128 bytes of 0xFF where every access OAM DMA blocks lands (all but HRAM): a read gives
/// 0xFF and a write records nothing. Not a port: the host only cuts the transfer's run
/// when a model-emitted address lands here (`Bus::blocked`).
pub const IO_BLOCKED: u32 = 3;
pub const BLOCKED: u32 = IO + IO_BLOCKED * BLOCK;
/// A000-BFFF while a cartridge with MBC3's clock selects one of its registers.
pub const IO_CLOCK: u32 = 4;
/// The clock's enable and select on a board with it: the second device a RAMG or RAMB
/// write reaches (`sm83::translate::second_port`). No read lands here.
pub const IO_CLOCK_ENABLE: u32 = 6;
pub const IO_CLOCK_SELECT: u32 = 7;
/// 6000-7FFF on a cartridge with the clock: the latch.
pub const IO_LATCH: u32 = 5;
/// `IO_DISPLAY + i` is display byte `PAL_DMG + i`: BGP, OBP0, OBP1, LCDC, SCY, SCX, WY, WX.
pub const IO_DISPLAY: u32 = 8;
/// LCDC's display port, the multi-target one: the host hands its writes to control too.
pub const IO_LCDC: u32 = IO_DISPLAY + 3;
/// VBK's and SVBK's readback: `TAB + (bank << 7) + (a & 0x7F)`, whose byte at 0x4F is
/// `0xFE | (bank & 1)` and at 0x70 `0xF8 | bank`, so a read takes the register's bank bits
/// as address bits and no host byte.
pub const TAB: u32 = 0x82_D000;
pub const TAB_BLOCKS: u32 = 8;
/// ROM bank 0 as the bus shows it while the boot ROM is mapped, `BOOT + a` for a read of
/// `a` in 0000-3FFF: the boot ROM over 0000-00FF, and on CGB hardware 0200-08FF,
/// and the cartridge's bank 0 elsewhere, written at load. Bank 0's window lands here while
/// `Reg::Booting` says the boot ROM is mapped, so the translation reads no address bit for
/// it. Its cells are zeros on a run without a boot ROM.
pub const BOOT: u32 = 0x84_0000;
pub const BOOT_LEN: usize = 0x4000;
pub const TOTAL: u32 = BOOT + BOOT_LEN as u32;
/// Bits of a physical byte address: [`TOTAL`] fits them.
pub const ADDR_BITS: usize = 24;
pub const BLOCK: u32 = 0x80;

/// The port code of an I/O register FF00 + `low7` (`low7` below 0x80), which the
/// translation emits in bits 7..10 of the physical address, on CGB hardware (`hardware`)
/// in CGB mode (`cgb`) or not. VBK and SVBK on CGB read from [`TAB`] and write to their
/// registers, so they have none here. FF72, FF73 and FF75 answer on CGB hardware in
/// either mode, FF74 in CGB mode only (Pan Docs, "Undocumented registers"; mooneye
/// `misc/bits/unused_hwio-C`). BCPS, OCPS, PCM12 and PCM34 answer on CGB hardware in
/// either mode, BCPD and OCPD in CGB mode only (SameBoy v1.0.5 `memory.c`, which gates
/// the first four on `GB_is_cgb` and the data ports on `cgb_mode`; mooneye
/// `misc/boot_hwio-C` reads C8 and D0 at FF68 and FF6A in compatibility mode).
#[cfg(feature = "gbc")]
pub fn io_code(low7: u8, hardware: bool, cgb: bool) -> u32 {
    match low7 {
        0x00..=0x02 | 0x04..=0x07 | 0x0F | 0x41 | 0x44..=0x46 => IO_CONTROL,
        0x68 | 0x6A | 0x72 | 0x73 | 0x75 if hardware => IO_CONTROL,
        0x4C | 0x4D | 0x51..=0x55 | 0x69 | 0x6B | 0x74 if cgb => IO_CONTROL,
        0x10..=0x3F => IO_APU,
        0x76 | 0x77 if hardware => IO_APU,
        0x47..=0x49 => IO_DISPLAY + (low7 - 0x47) as u32,
        0x40 => IO_LCDC,
        0x42 | 0x43 => IO_DISPLAY + 4 + (low7 - 0x42) as u32,
        0x4A | 0x4B => IO_DISPLAY + 6 + (low7 - 0x4A) as u32,
        _ => IO_UNMAPPED,
    }
}

/// The port code of a write to FF00 + `low7`: KEY0 is written only while the boot ROM is
/// mapped, as a register (`Banking::register_at`), so a write there afterwards reaches no
/// device. Every other register's is its [`io_code`].
#[cfg(feature = "gbc")]
pub fn io_write_code(low7: u8, hardware: bool, cgb: bool) -> u32 {
    if low7 == 0x4C { IO_UNMAPPED } else { io_code(low7, hardware, cgb) }
}

/// A host device a physical address the model emitted reaches, decoded from the address
/// alone: a port block's code with the low bits passed on, or a banking register a
/// device also keeps.
#[cfg(feature = "gbc")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    /// A control register at this bus address (FFFF for IE).
    Control(u16),
    Apu(u16),
    /// The PPU's display byte at this index of [`PAL`].
    Display(u8),
    /// The cartridge clock's chip, at the bus address it decodes: 0000 the enable, 4000
    /// the select, 6000 the latch, A000 a register.
    Clock(u16),
}

#[cfg(feature = "gbc")]
impl Port {
    /// The bus address the port's byte sits at on a flat bus (SingleStepTests): the
    /// register's own address, a display byte's register, and A000 or 6000 plus the low
    /// bits for the clock's port and latch (0000 and 4000, RAMG and RAMB, have no byte).
    pub fn bus_addr(self) -> u16 {
        const DISPLAY: [u16; 8] = [0xFF47, 0xFF48, 0xFF49, 0xFF40, 0xFF42, 0xFF43, 0xFF4A, 0xFF4B];
        match self {
            Port::Control(a) | Port::Apu(a) | Port::Clock(a) => a,
            Port::Display(i) => DISPLAY[(i as u32 - PAL_DMG) as usize],
        }
    }
}

/// The device a model-emitted physical address reaches, if any: a port block names it by
/// its code. A write a second device hears names that device in a second address
/// ([`second_port`]), so each address names one device.
#[cfg(feature = "gbc")]
pub fn port(p: u32) -> Option<Port> {
    if is_io(p) {
        let (code, low7) = ((p - IO) >> 7, (p & 0x7F) as u16);
        return match code {
            IO_CONTROL => Some(Port::Control(if low7 == 0x7F { 0xFFFF } else { 0xFF00 | low7 })),
            IO_APU => Some(Port::Apu(0xFF00 | low7)),
            IO_CLOCK => Some(Port::Clock(0xA000 | low7)),
            IO_LATCH => Some(Port::Clock(0x6000 | low7)),
            IO_CLOCK_ENABLE => Some(Port::Clock(0x0000)),
            IO_CLOCK_SELECT => Some(Port::Clock(0x4000)),
            IO_DISPLAY.. => Some(Port::Display(PAL_DMG as u8 | (code & 7) as u8)),
            _ => None,
        };
    }
    None
}

/// The second port a write landing at `p` reaches (the circuit's `WD`): SVBK's record
/// reaches control's FF70, LCDC's display byte control's FF40, and on a board with MBC3's
/// clock RAMG's and RAMB's records the clock's enable and select.
#[cfg(feature = "gbc")]
pub fn second_port(p: u32, has_clock: bool) -> Option<u32> {
    let at = |code: u32, low7: u32| IO + code * BLOCK + low7;
    match p {
        _ if p == REGS + Reg::Svbk as u32 => Some(at(IO_CONTROL, 0x70)),
        _ if p >> 7 == at(IO_LCDC, 0) >> 7 => Some(at(IO_CONTROL, 0x40)),
        _ if has_clock && p == REGS + Reg::Ramg as u32 => Some(at(IO_CLOCK_ENABLE, 0)),
        _ if has_clock && p == REGS + Reg::Ramb as u32 => Some(at(IO_CLOCK_SELECT, 0)),
        _ if p == REGS + Reg::Key0 as u32 => Some(at(IO_CONTROL, 0x4C)),
        _ => None,
    }
}

/// Whether a physical address is in [`IO`].
pub fn is_io(p: u32) -> bool {
    (IO..IO + IO_BLOCKS * BLOCK).contains(&p)
}

/// The byte a constant cell the model serves holds: 0xFF in FF and in the I/O page's
/// unmapped block, 0 in VOID, the readback table's bytes in TAB. None elsewhere, ports
/// included, whose bytes the host serves.
#[cfg(feature = "gbc")]
pub fn constant(p: u32) -> Option<u8> {
    let unmapped = IO + IO_UNMAPPED * BLOCK;
    match p {
        _ if is_ff(p) || is_blocked(p) || (unmapped..unmapped + BLOCK).contains(&p) => Some(0xFF),
        _ if is_void(p) => Some(0),
        _ if (TAB..TAB + TAB_BLOCKS * BLOCK).contains(&p) => {
            let (bank, low7) = ((p - TAB) >> 7, p & 0x7F);
            Some(match low7 {
                0x4F => 0xFE | (bank & 1) as u8,
                0x70 => 0xF8 | bank as u8,
                _ => 0,
            })
        }
        _ => None,
    }
}
/// Bytes per memory line. The cache holds one cell per line, with the line address as
/// its key and the line's 64 bits as its value (`docs/stock-llama.md` §10). Every region
/// starts on a line, so a line never spans two regions.
pub const LINE: usize = 8;
pub const LINE_BITS: usize = 3;
/// The number of lines in the layout; the cells' positions are `0..LINES`.
pub const LINES: u32 = TOTAL >> LINE_BITS;
/// Tokens exist up to an 8 MB ROM, MBC5's 512 banks; `translate` masks banks so no
/// larger address forms.
pub const MAX_ROM: usize = 0x80_0000;
pub const WRAM_LEN: usize = 0x8000;
pub const VRAM_LEN: usize = 0x4000;
pub const OAM_LEN: usize = 0xA0;
pub const HRAM_LEN: usize = 0x7F;

/// The register file. Each is one byte at `REGS + r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    /// The low ROM bank register (MBC5 2000-2FFF, MBC1/MBC3 2000-3FFF, MBC2 with a.8 set).
    Romb0,
    /// MBC5 3000-3FFF (bank bit 8), MBC1 4000-5FFF (bank bits 5-6 and the RAM bank).
    Romb1,
    /// MBC5 and MBC3 4000-5FFF: the RAM bank, or the MBC3 clock register select.
    Ramb,
    /// 0000-1FFF: RAM enable, on when the low nibble is 0x0A.
    Ramg,
    /// MBC1 6000-7FFF, bit 0.
    Mode,
    Svbk,
    Vbk,
    /// Cartridge and hardware facts: bits 0-4 the mapper one-hot, bit 5 the cartridge
    /// has MBC3's clock, bit 6 it has RAM, bit 7 CGB mode. Written at load, never by the
    /// game.
    Mapper,
    /// ROM bank mask, `banks - 1`, bits 0-7 (ROM sizes are powers of two of at most 8 MB).
    Mask0,
    /// Cartridge RAM mask over offset bits 9-16: `(ram_len - 1) >> 9`.
    RamMask,
    /// Bit 0: the run began at power-on with the boot ROM mapped. Written at load only.
    Booting,
    /// FF50's write while the boot ROM is mapped: bit 0 unmaps it.
    Unmapped,
    /// FF50's write while the boot ROM is mapped on CGB hardware whose KEY0 asked for DMG
    /// compatibility (bit 2): bit 0 unmaps it and leaves CGB mode.
    Compat,
    /// KEY0 (FF4C) as the GBC CPU wrote it, which FF50's write consults.
    Key0,
    /// ROM bank mask bit 8 in bit 0: set only for an 8 MB ROM, whose MBC5 bank bit 8
    /// selects. Bit 1 ([`MBC1M`]): an MBC1 multicart's wiring. Written at load, never by
    /// the game.
    Mask1,
}

pub const N_REGS: usize = 15;
pub const REG_NAMES: [&str; N_REGS] = ["ROMB0", "ROMB1", "RAMB", "RAMG", "MODE", "SVBK", "VBK", "MAPPER", "MASK0", "RAMMASK", "BOOTING", "UNMAPPED", "COMPAT", "KEY0", "MASK1"];

/// Bits of the `Mapper` register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mapper {
    Rom = 0,
    Mbc1 = 1,
    Mbc2 = 2,
    Mbc3 = 3,
    Mbc5 = 4,
}
/// The board has MBC3's real-time clock (MBC3 with a timer).
pub const HAS_CLOCK: u8 = 1 << 5;
pub const HAS_RAM: u8 = 1 << 6;
pub const CGB: u8 = 1 << 7;
/// `Mask1` bit 1: an MBC1 multicart, whose bank register's 4 low bits reach the ROM and
/// the high register sits above them.
pub const MBC1M: u8 = 1 << 1;

/// The fifteen register bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Banking(pub [u8; N_REGS]);

impl Banking {
    pub fn get(&self, r: Reg) -> u8 {
        self.0[r as usize]
    }

    pub fn set(&mut self, r: Reg, v: u8) {
        self.0[r as usize] = v;
    }

}

/// The mapper's rules over the registers: the load-time defaults and the translation's
/// specification, which the model is held to.
#[cfg(feature = "gbc")]
impl Banking {
    /// Registers at power-on for a cartridge of `mapper` with `rom_len` bytes of ROM
    /// and `ram_len` of RAM. `rom_len` must be a power of two of at most 8 MB and
    /// `ram_len` zero or a power of two of at least 512.
    pub fn new(mapper: Mapper, rom_len: usize, ram_len: usize, cgb: bool) -> Result<Banking, String> {
        if !rom_len.is_power_of_two() || rom_len > MAX_ROM || rom_len < 0x8000 {
            return Err(format!("ROM of {rom_len} bytes: the model's memory holds power-of-two ROMs from 32 KiB to 8 MiB"));
        }
        if ram_len != 0 && (!ram_len.is_power_of_two() || ram_len < 0x200 || ram_len > CRAM_LEN) {
            return Err(format!("cartridge RAM of {ram_len} bytes: the model's memory holds 0 or a power of two from 512 B to 128 KiB"));
        }
        let banks = (rom_len / 0x4000) as u32 - 1;
        let mut b = Banking([0; N_REGS]);
        b.set(Reg::Romb0, 1);
        b.set(Reg::Mapper, 1 << mapper as u8 | if ram_len > 0 { HAS_RAM } else { 0 } | if cgb { CGB } else { 0 });
        b.set(Reg::Mask0, banks as u8);
        b.set(Reg::Mask1, (banks >> 8) as u8);
        if ram_len > 0 {
            b.set(Reg::RamMask, ((ram_len - 1) >> 9) as u8);
        }
        // ROM+RAM boards have no enable register, so their RAM is always mapped.
        if mapper == Mapper::Rom && ram_len > 0 {
            b.set(Reg::Ramg, 0x0A);
        }
        Ok(b)
    }

    pub fn mapper(&self) -> Mapper {
        let m = self.get(Reg::Mapper);
        [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5].into_iter().find(|&k| m & (1 << k as u8) != 0).expect("the MAPPER byte names a mapper")
    }

    /// CGB mode: CGB hardware (MAPPER bit 7) not left for DMG compatibility.
    pub fn cgb(&self) -> bool {
        self.cgb_hardware() && self.get(Reg::Compat) & 1 == 0
    }

    pub fn cgb_hardware(&self) -> bool {
        self.get(Reg::Mapper) & CGB != 0
    }

    /// The boot ROM is mapped: the run began at power-on and no FF50 write unmapped it.
    pub fn boot_mapped(&self) -> bool {
        self.get(Reg::Booting) & 1 != 0 && (self.get(Reg::Unmapped) | self.get(Reg::Compat)) & 1 == 0
    }

    /// The register FF50's write lands on while the boot ROM is mapped: COMPAT on CGB
    /// hardware whose KEY0 asks for DMG compatibility, else UNMAPPED.
    fn unmap_register(&self) -> Reg {
        if self.cgb_hardware() && self.get(Reg::Key0) & 0x04 != 0 { Reg::Compat } else { Reg::Unmapped }
    }

    fn has_ram(&self) -> bool {
        self.get(Reg::Mapper) & HAS_RAM != 0
    }

    fn ram_enabled(&self) -> bool {
        self.has_ram() && self.get(Reg::Ramg) & 0x0F == 0x0A
    }

    /// The same registers on an MBC1 multicart.
    pub fn multicart(mut self) -> Banking {
        assert_eq!(self.mapper(), Mapper::Mbc1, "only MBC1 has the multicart wiring");
        self.set(Reg::Mask1, self.get(Reg::Mask1) | MBC1M);
        self
    }

    fn is_multicart(&self) -> bool {
        self.get(Reg::Mask1) & MBC1M != 0
    }

    pub fn has_clock(&self) -> bool {
        self.get(Reg::Mapper) & HAS_CLOCK != 0
    }

    /// The same registers on a board with MBC3's clock (MAPPER bit 5).
    pub fn with_clock(mut self) -> Banking {
        assert_eq!(self.mapper(), Mapper::Mbc3, "only MBC3 has the clock");
        self.set(Reg::Mapper, self.get(Reg::Mapper) | HAS_CLOCK);
        self
    }

    /// The ROM bank mapped at `a` (0000-7FFF), masked to the ROM size.
    fn rom_bank(&self, a: u16) -> u32 {
        let (r0, r1) = (self.get(Reg::Romb0) as u32, self.get(Reg::Romb1) as u32);
        let mode = self.get(Reg::Mode) & 1 != 0;
        let bank = match (a < 0x4000, self.mapper()) {
            (true, Mapper::Mbc1) if mode => (r1 & 3) << if self.is_multicart() { 4 } else { 5 },
            (true, _) => 0,
            (false, Mapper::Rom) => 1,
            (false, Mapper::Mbc1) if self.is_multicart() => (r1 & 3) << 4 | (r0 & 0x1F).max(1) & 0x0F,
            (false, Mapper::Mbc1) => (r1 & 3) << 5 | (r0 & 0x1F).max(1),
            (false, Mapper::Mbc2) => (r0 & 0x0F).max(1),
            (false, Mapper::Mbc3) => (r0 & 0x7F).max(1),
            (false, Mapper::Mbc5) => (r1 & 1) << 8 | r0,
        };
        bank & ((self.get(Reg::Mask1) as u32 & 1) << 8 | self.get(Reg::Mask0) as u32)
    }

    /// Where a cartridge RAM access lands: the byte, the FF block (disabled or absent
    /// RAM, an MBC3 select the board does not answer) or the clock's port, which RAMG
    /// enables with or without RAM on a board with the clock and selects 08-0C reach.
    fn cram(&self, a: u16) -> u32 {
        let low7 = (a & 0x7F) as u32;
        let select = self.get(Reg::Ramb);
        if self.has_clock() && self.get(Reg::Ramg) & 0x0F == 0x0A && (0x08..=0x0C).contains(&select) {
            return IO + IO_CLOCK * BLOCK | low7;
        }
        if !self.ram_enabled() {
            return FF | low7;
        }
        let bank = match self.mapper() {
            Mapper::Mbc1 if self.get(Reg::Mode) & 1 != 0 && !self.is_multicart() => (self.get(Reg::Romb1) & 3) as u32,
            Mapper::Mbc3 if select >= 0x08 => return FF | low7,
            Mapper::Mbc3 => (select & 7) as u32,
            Mapper::Mbc5 => (select & 0x0F) as u32,
            _ => 0,
        };
        let mask = (self.get(Reg::RamMask) as u32) << 9 | 0x1FF;
        CRAM + ((bank << 13 | (a as u32 & 0x1FFF)) & mask)
    }

    /// The register a write to `a` lands in, if the mapper has one there.
    fn register_at(&self, a: u16) -> Option<Reg> {
        match (self.mapper(), a) {
            (Mapper::Rom, 0x0000..=0x7FFF) => None,
            (Mapper::Mbc2, 0x0000..=0x3FFF) => Some(if a & 0x100 != 0 { Reg::Romb0 } else { Reg::Ramg }),
            (Mapper::Mbc2, 0x4000..=0x7FFF) => None,
            (_, 0x0000..=0x1FFF) => Some(Reg::Ramg),
            (Mapper::Mbc5, 0x2000..=0x2FFF) => Some(Reg::Romb0),
            (Mapper::Mbc5, 0x3000..=0x3FFF) => Some(Reg::Romb1),
            (_, 0x2000..=0x3FFF) => Some(Reg::Romb0),
            (Mapper::Mbc1, 0x4000..=0x5FFF) => Some(Reg::Romb1),
            (Mapper::Mbc3 | Mapper::Mbc5, 0x4000..=0x5FFF) => Some(Reg::Ramb),
            (Mapper::Mbc1, 0x6000..=0x7FFF) => Some(Reg::Mode),
            (_, 0x6000..=0x7FFF) => None,
            (_, 0xFF4F) if self.cgb() => Some(Reg::Vbk),
            (_, 0xFF70) if self.cgb() => Some(Reg::Svbk),
            (_, 0xFF4C) if self.cgb() && self.boot_mapped() => Some(Reg::Key0),
            (_, 0xFF50) if self.boot_mapped() => Some(self.unmap_register()),
            _ => None,
        }
    }
}

pub const CRAM_LEN: usize = 0x2_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// The CPU accesses a running OAM DMA blocks, by the bus each address sits on (SameBoy
/// v1.0.5 `is_addr_in_dma_use` and `read_high_memory`): OAM always, never the FF page
/// (I/O, HRAM, IE), and below FE00 the addresses on the bus the transfer reads. A
/// source in VRAM takes the VRAM bus; on a CGB a source in C000-DFFF takes only the
/// work RAM bus; any other source (ROM, cartridge RAM, a DMG's work RAM, a CGB's
/// E000-FFFF fill) takes every bus but VRAM's. A blocked read gives 0xFF and a blocked
/// write drops; SameBoy's read of the transfer's own byte on a conflict is not modelled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dma(pub u8);

impl Dma {
    pub const NONE: Dma = Dma(0);
    pub const OAM: u8 = 1;
    pub const VRAM: u8 = 2;
    pub const MAIN: u8 = 4;
    pub const WRAM: u8 = 8;
    /// Every bus below FE00 and OAM: what a running transfer blocked before the buses.
    pub const ALL: Dma = Dma(Self::OAM | Self::VRAM | Self::MAIN | Self::WRAM);

    /// The accesses a transfer from page `source` blocks while it runs, on a CGB or not.
    pub fn during(source: u8, cgb: bool) -> Dma {
        Dma(Self::OAM
            | match source {
                0x80..=0x9F => Self::VRAM,
                0xC0..=0xDF if cgb => Self::WRAM,
                _ => Self::MAIN | Self::WRAM,
            })
    }

    /// Whether the transfer blocks bus address `a`.
    pub fn blocks(self, a: u16) -> bool {
        let bus = match a {
            0x8000..=0x9FFF => Self::VRAM,
            0xC000..=0xFDFF => Self::WRAM,
            0xFE00..=0xFEFF => Self::OAM,
            0xFF00..=0xFFFF => 0,
            _ => Self::MAIN,
        };
        self.0 & bus != 0
    }
}

impl From<bool> for Dma {
    fn from(on: bool) -> Dma {
        if on { Dma::ALL } else { Dma::NONE }
    }
}

/// The physical address of bus address `a` under the registers `b`, with OAM DMA
/// `dma` blocking what it blocks ([`Dma`]). Reads and fetches are `Access::Read`.
#[cfg(feature = "gbc")]
pub fn translate(b: &Banking, a: u16, access: Access, dma: impl Into<Dma>) -> u32 {
    translate_locked(b, a, access, dma, 0)
}

/// [`translate`] under the GBC CPU's access locks (`Lines::locks`: bit 0 VRAM, bit 1
/// OAM): a locked access lands in the FF block, so it reads 0xFF and keeps no record.
#[cfg(feature = "gbc")]
pub fn translate_locked(b: &Banking, a: u16, access: Access, dma: impl Into<Dma>, locks: u8) -> u32 {
    let low7 = (a & 0x7F) as u32;
    if dma.into().blocks(a) {
        return BLOCKED | low7;
    }
    if (locks & 1 != 0 && (0x8000..=0x9FFF).contains(&a)) || (locks & 2 != 0 && (0xFE00..=0xFE9F).contains(&a)) {
        return FF | low7;
    }
    if access == Access::Write {
        if let Some(r) = b.register_at(a) {
            return REGS + r as u32;
        }
        if a < 0x8000 {
            // MBC3's clock latches on a write to 6000-7FFF.
            return if b.has_clock() && a >= 0x6000 { IO + IO_LATCH * BLOCK | low7 } else { VOID | low7 };
        }
    }
    if access == Access::Read && b.boot_mapped() && a < 0x4000 {
        return BOOT + a as u32;
    }
    match a {
        0x0000..=0x7FFF => ROM + b.rom_bank(a) * 0x4000 + (a as u32 & 0x3FFF),
        0x8000..=0x9FFF => {
            let bank = if b.cgb() { (b.get(Reg::Vbk) & 1) as u32 } else { 0 };
            VRAM + bank * 0x2000 + (a as u32 & 0x1FFF)
        }
        0xA000..=0xBFFF => b.cram(a),
        0xC000..=0xFDFF => {
            let off = a as u32 & 0x1FFF;
            if off < 0x1000 { WRAM + off } else { WRAM + ((b.get(Reg::Svbk) & 7).max(1) as u32) * 0x1000 + (off & 0xFFF) }
        }
        0xFE00..=0xFE9F => OAM + (a as u32 & 0xFF),
        0xFF80..=0xFFFE => HRAM + low7,
        0xFEA0..=0xFEFF => VOID | low7,
        0xFF00..=0xFF7F => match (access, a) {
            (Access::Read, 0xFF4F) if b.cgb_hardware() => TAB + (((b.get(Reg::Vbk) & 1) as u32) << 7) | low7,
            (Access::Read, 0xFF70) if b.cgb() => TAB + (((b.get(Reg::Svbk) & 7) as u32) << 7) | low7,
            (Access::Write, _) => IO + io_write_code(low7 as u8, b.cgb_hardware(), b.cgb()) * BLOCK | low7,
            _ => IO + io_code(low7 as u8, b.cgb_hardware(), b.cgb()) * BLOCK | low7,
        },
        0xFFFF => IO + IO_CONTROL * BLOCK | low7,
    }
}

/// The byte a write records: MBC2 RAM is a nibble whose reads return `0xF0 | nibble`,
/// so the token holds `v | 0xF0`. Every other write records `v`.
#[cfg(feature = "gbc")]
pub fn write_value(b: &Banking, a: u16, v: u8) -> u8 {
    if b.mapper() == Mapper::Mbc2 && (0xA000..=0xBFFF).contains(&a) { v | 0xF0 } else { v }
}

/// The physical address an opcode fetch of `a` lands at: the read translation, with a
/// byte the host serves moved to the HOST block at the same low seven bits. The host
/// serves a fetch from a port and, without OAM DMA, from anywhere in the unusable area
/// and the I/O page, constants included (`docs/host-in-weights.md` item 12).
#[cfg(feature = "gbc")]
pub fn fetch(b: &Banking, a: u16, dma: impl Into<Dma>) -> u32 {
    fetch_locked(b, a, dma, 0)
}

/// [`fetch`] under the GBC CPU's access locks, as [`translate_locked`] takes them: an
/// operand fetch takes the locks at its M-cycle as a data read does.
#[cfg(feature = "gbc")]
pub fn fetch_locked(b: &Banking, a: u16, dma: impl Into<Dma>, locks: u8) -> u32 {
    let dma = dma.into();
    let p = translate_locked(b, a, Access::Read, dma, locks);
    if port(p).is_some() || !dma.blocks(a) && (0xFEA0..=0xFF7F).contains(&a) { HOST | (p & 0x7F) } else { p }
}

/// Whether a physical address is in the HOST block: a fetched byte the host serves.
pub fn is_host(p: u32) -> bool {
    (HOST..HOST + BLOCK).contains(&p)
}

/// Whether a physical address is in the FF block.
pub fn is_ff(p: u32) -> bool {
    (FF..FF + BLOCK).contains(&p)
}

/// Whether a physical address is in the block OAM DMA's blocked accesses land in.
pub fn is_blocked(p: u32) -> bool {
    (BLOCKED..BLOCKED + BLOCK).contains(&p)
}

/// Whether a physical address is in the VOID block.
pub fn is_void(p: u32) -> bool {
    (VOID..VOID + BLOCK).contains(&p)
}

/// Whether a write to `p` is kept as a record: anything but the FF and VOID blocks and
/// the I/O page's blocks.
#[cfg(feature = "gbc")]
pub fn recorded(p: u32) -> bool {
    !is_ff(p) && !is_void(p) && !is_io(p)
}

/// The banked windows, each as its first bus address and its length. [`translate`] maps
/// every bus address of a window to the translation of the window's first byte (reads,
/// no OAM DMA) plus the address's offset in the window, under any registers; the echo
/// F000-FDFF follows D000. The memory circuit names each first byte's translation in its
/// `MAP` slots, so a runtime can attend only the mapped banks (`docs/stock-llama.md`).
pub const WINDOWS: [(u16, u32); 4] = [(0x0000, 0x4000), (0x4000, 0x4000), (0xA000, 0x2000), (0xD000, 0x1000)];

/// Where [`translate`] and [`fetch`] map every bus address outside [`WINDOWS`], under
/// any registers: C000-CFFF and its echo to WRAM bank 0, 8000-9FFF to a VRAM bank (both
/// listed), FE00-FE9F to OAM, FF80-FFFE to HRAM, the unusable area to VOID, the I/O page
/// and IE to IO or TAB (all to FF under OAM DMA), a banking register's write to REGS, a
/// write elsewhere below 8000 to VOID or the clock's latch, and a fetch of a byte the
/// host serves to HOST. Each entry is a (base, length) pair.
pub const UNBANKED: [(u32, u32); 10] = [(WRAM, 0x1000), (VRAM, VRAM_LEN as u32), (OAM, OAM_LEN as u32), (HRAM, HRAM_LEN as u32), (FF, BLOCK), (VOID, BLOCK), (REGS, N_REGS as u32), (HOST, BLOCK), (IO, IO_BLOCKS * BLOCK), (TAB, TAB_BLOCKS * BLOCK)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_do_not_overlap_and_fit_24_bits() {
        assert!(TOTAL <= 1 << ADDR_BITS);
        assert_eq!(CRAM, ROM + MAX_ROM as u32);
        assert_eq!(WRAM, CRAM + CRAM_LEN as u32);
        assert_eq!(VRAM, WRAM + WRAM_LEN as u32);
        assert_eq!(OAM, VRAM + VRAM_LEN as u32);
        assert!(OAM + OAM_LEN as u32 <= HRAM);
        assert!(HRAM + HRAM_LEN as u32 <= FF);
        assert_eq!((FF + BLOCK, VOID + BLOCK), (VOID, REGS));
        assert!(REGS + N_REGS as u32 <= PAL);
        assert!(PAL + PAL_LEN as u32 <= HOST && HOST % BLOCK == 0);
        assert!(HOST + BLOCK <= IO && IO + IO_BLOCKS * BLOCK <= TAB && IO % (IO_BLOCKS * BLOCK) == 0 && TAB % (TAB_BLOCKS * BLOCK) == 0);
        assert!(TAB + TAB_BLOCKS * BLOCK <= BOOT && BOOT % BOOT_LEN as u32 == 0 && BOOT_LEN == 0x4000, "BOOT is a bank: fourteen low bits pass");
        assert_eq!(1 << LINE_BITS, LINE);
        for base in [ROM, CRAM, WRAM, VRAM, OAM, HRAM, FF, VOID, REGS, PAL, HOST, IO, TAB, TOTAL] {
            assert_eq!(base % LINE as u32, 0, "{base:#x} starts mid-line");
        }
    }

    #[test]
    fn power_on_registers_describe_the_cartridge() {
        let b = Banking::new(Mapper::Mbc5, 0x10_0000, 0x8000, true).unwrap();
        assert_eq!((b.get(Reg::Romb0), b.get(Reg::Mask0), b.get(Reg::RamMask)), (1, 63, 0x3F));
        assert_eq!(b.get(Reg::Mapper), 1 << 4 | HAS_RAM | CGB);
        assert_eq!(translate(&b, 0x4000, Access::Read, false), 0x4000);
        assert_eq!(translate(&b, 0xA000, Access::Read, false), FF, "RAM disabled at power-on");
        assert_eq!(translate(&b, 0xD000, Access::Read, false), WRAM + 0x1000, "SVBK 0 selects bank 1");
        assert!(Banking::new(Mapper::Mbc5, 0x18_0000, 0, false).is_err(), "1.5 MB is not a power of two");
    }

    /// Every bus address, read, written or fetched, with OAM DMA off and on, lands in a
    /// banked window at its offset from the window's first byte, or in `UNBANKED`, under
    /// random registers of every mapper.
    #[test]
    fn every_address_lands_in_a_window_or_an_unbanked_region() {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        let unbanked = |p: u32| UNBANKED.iter().any(|&(base, len)| (base..base + len).contains(&p));
        for trial in 0..200 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = mappers[trial % 5];
            let ram = if m == Mapper::Mbc2 { 0x200 } else { [0, 0x800, 0x2000, 0x8000][(rng >> 60) as usize % 4] };
            let mut b = Banking::new(m, 0x8000 << (rng >> 3) % 9, ram, rng >> 9 & 1 == 1).unwrap();
            if m == Mapper::Mbc3 && rng >> 40 & 1 == 1 {
                b = b.with_clock();
            }
            for (i, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                b.set(reg, (rng >> (7 * i + 10)) as u8);
            }
            let bases = WINDOWS.map(|(a, len)| (translate(&b, a, Access::Read, false), len));
            for a in 0..=0xFFFFu16 {
                let window = WINDOWS.iter().position(|&(w, len)| (w as u32..w as u32 + len).contains(&(a as u32))).or(((0xF000..0xFE00).contains(&a)).then_some(3));
                for (access, dma, p) in [(Access::Read, false, translate(&b, a, Access::Read, false)), (Access::Read, true, translate(&b, a, Access::Read, true)), (Access::Write, false, translate(&b, a, Access::Write, false)), (Access::Write, true, translate(&b, a, Access::Write, true)), (Access::Read, false, fetch(&b, a, false)), (Access::Read, true, fetch(&b, a, true))] {
                    let in_window = window.is_some_and(|k| (bases[k].0..bases[k].0 + bases[k].1).contains(&p));
                    assert!(in_window || unbanked(p), "{a:#06x} {access:?} dma {dma} under {:?}: {p:#08x} is in no window and no unbanked region", b.0);
                }
            }
        }
    }

    /// A DMA line's placement: the translation of an 8-byte line's first address with the
    /// byte's low three bits concatenated equals the byte's own translation, at every
    /// address outside the I/O page and IE, under random registers of every mapper; and a
    /// VRAM or OAM destination lands as a read does.
    #[test]
    fn a_lines_translation_places_its_bytes() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mappers = [Mapper::Rom, Mapper::Mbc1, Mapper::Mbc2, Mapper::Mbc3, Mapper::Mbc5];
        let mut placed = 0u64;
        for trial in 0..300 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let m = mappers[trial % 5];
            let ram = if m == Mapper::Mbc2 { 0x200 } else { [0, 0x800, 0x2000, 0x8000, 0x2_0000][(rng >> 60) as usize % 5] };
            let mut b = Banking::new(m, 0x8000 << (rng >> 3) % 9, ram, rng >> 9 & 1 == 1).unwrap();
            if m == Mapper::Mbc3 && rng >> 40 & 1 == 1 {
                b = b.with_clock();
            }
            for (i, reg) in [Reg::Romb0, Reg::Romb1, Reg::Ramb, Reg::Ramg, Reg::Mode, Reg::Svbk, Reg::Vbk].into_iter().enumerate() {
                b.set(reg, (rng >> (7 * i + 10)) as u8);
            }
            if rng >> 50 & 1 == 1 {
                b.set(Reg::Ramg, 0x0A);
            }
            if !b.cgb() {
                b.set(Reg::Vbk, 0);
            }
            for a in (0..=0xFFFFu16).filter(|a| !(0xFF00..0xFF80).contains(a) && *a != 0xFFFF) {
                let p = translate(&b, a & !7, Access::Read, false) | (a & 7) as u32;
                assert_eq!(p, translate(&b, a, Access::Read, false), "read {a:#06x} under {:?}", b.0);
                if (0x8000..0xA000).contains(&a) || (0xFE00..0xFEA0).contains(&a) {
                    assert_eq!(p, translate(&b, a, Access::Write, false), "write {a:#06x} under {:?}", b.0);
                }
                placed += 1;
            }
        }
        assert_eq!(placed, 300 * (0x10000 - 0x81));
    }

    #[test]
    fn mbc2_nibbles_read_back_with_the_high_bits_set() {
        let b = Banking::new(Mapper::Mbc2, 0x8000, 0x200, false).unwrap();
        assert_eq!(write_value(&b, 0xA123, 0x25), 0xF5);
        assert_eq!(write_value(&b, 0xC000, 0x25), 0x25);
        assert_eq!(translate(&b, 0xA123, Access::Write, false), FF | 0x23, "disabled until RAMG is written");
        let mut on = b;
        on.set(Reg::Ramg, 0x0A);
        assert_eq!(translate(&on, 0xA323, Access::Write, false), CRAM + 0x123, "512 bytes, mirrored");
    }
}
