//! Cartridge: ROM, mapper registers, and battery RAM. The host owns persistence: it
//! passes saved RAM in and polls for RAM the game finished writing. MBC3's clock is
//! control logic (`machine::rtc`): the cartridge only says when its chip is selected,
//! keeps a saved clock footer for the machine to load, and counts a clock register
//! write toward the next save.

use super::rtc::FOOTER;
use crate::memory::layout::Banking;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperKind {
    Rom,
    Mbc1,
    Mbc2,
    Mbc3,
    Mbc5,
}

/// What header byte 0x147 says is on the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kind {
    pub mapper: MapperKind,
    pub battery: bool,
    pub rtc: bool,
}

impl Kind {
    pub fn from_type(cartridge_type: u8) -> Result<Kind, String> {
        use MapperKind::*;
        let (mapper, battery, rtc) = match cartridge_type {
            0x00 | 0x08 => (Rom, false, false),
            0x09 => (Rom, true, false),
            0x01 | 0x02 => (Mbc1, false, false),
            0x03 => (Mbc1, true, false),
            0x05 => (Mbc2, false, false),
            0x06 => (Mbc2, true, false),
            0x0F | 0x10 => (Mbc3, true, true),
            0x11 | 0x12 => (Mbc3, false, false),
            0x13 => (Mbc3, true, false),
            0x19 | 0x1A | 0x1C | 0x1D => (Mbc5, false, false),
            0x1B | 0x1E => (Mbc5, true, false),
            t => {
                let name = match t {
                    0x0B..=0x0D => "MMM01",
                    0x20 => "MBC6",
                    0x22 => "MBC7",
                    0xFC => "Pocket Camera",
                    0xFD => "Bandai TAMA5",
                    0xFE => "HuC3",
                    0xFF => "HuC1",
                    _ => "unknown",
                };
                return Err(format!("cartridge type {t:#04x} ({name}) is not supported; supported: ROM only, MBC1, MBC2, MBC3, MBC5"));
            }
        };
        Ok(Kind { mapper, battery, rtc })
    }
}

/// External RAM size from header byte 0x149. MBC2's RAM is on the mapper chip.
fn ram_size(mapper: MapperKind, code: u8) -> Result<usize, String> {
    if mapper == MapperKind::Mbc2 {
        return Ok(512);
    }
    Ok(match code {
        0x00 => 0,
        0x01 => 0x800,
        0x02 => 0x2000,
        0x03 => 0x8000,
        0x04 => 0x2_0000,
        0x05 => 0x1_0000,
        c => return Err(format!("RAM size code {c:#04x} is not valid")),
    })
}

#[derive(Debug)]
pub(super) enum Mapper {
    Rom,
    /// `low` is the 5-bit register at 2000, `high` the 2-bit one at 4000. `mode` 1
    /// applies `high` to the 0000 ROM area and to RAM. A multicart (MBC1M) wires only
    /// `low`'s 4 low bits to the ROM, `high` above them, and never banks RAM.
    Mbc1 { low: u8, high: u8, mode: bool, multicart: bool },
    Mbc2 { bank: u8 },
    /// `select` is the 4000 register: RAM bank 0-3 (0-7 on MBC30, with 64 KiB of RAM) or
    /// RTC register 08-0C. Boards without a clock read 0xFF there; on boards with one
    /// the machine routes those accesses to the control logic, which holds the clock.
    Mbc3 { bank: u8, select: u8, rtc: bool },
    Mbc5 { rom_bank: u16, ram_bank: u8 },
}

/// Battery RAM relative to what the host last took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Battery {
    Clean,
    /// Written, and RAM is still enabled: the game may be mid-save.
    Dirty,
    /// Written, then the game disabled RAM: a finished save.
    Committed,
}

/// A cartridge byte's home: an offset into the ROM or into cartridge RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Physical {
    Rom(usize),
    Ram(usize),
}

pub struct Cartridge {
    rom: Vec<u8>,
    pub(super) mapper: Mapper,
    ram: Vec<u8>,
    pub(super) ram_enabled: bool,
    /// ROM banks minus one, rounded up to a power of two.
    rom_bank_mask: usize,
    has_battery: bool,
    battery: Battery,
    /// The clock footer a save ended with, until the machine hands it to the clock.
    footer: Option<Vec<u8>>,
}

/// An MBC1 multicart (MBC1M): a ROM above 256 KiB whose bank 0x10 repeats bank 0's
/// header logo, as SameBoy guesses the wiring (`GB_configure_cart`, mooneye
/// `emulator-only/mbc1/multicart_rom_8Mb`). A logo of one repeated byte is no logo.
fn is_multicart(rom: &[u8]) -> bool {
    let logo = &rom[0x104..0x134];
    rom.len() >= 0x44000 && logo.iter().any(|&b| b != logo[0]) && *logo == rom[0x40104..0x40134]
}

impl Cartridge {
    pub fn rom(&self) -> &[u8] {
        &self.rom
    }

    pub fn new(rom: Vec<u8>, saved_ram: Option<Vec<u8>>) -> Result<Self, String> {
        if rom.len() < 0x8000 || !rom.len().is_multiple_of(0x4000) {
            return Err(format!("ROM size {} is not a multiple of 16 KiB", rom.len()));
        }
        let kind = Kind::from_type(rom[0x147])?;
        let ram_size = ram_size(kind.mapper, rom[0x149])?;
        let (mut ram, footer) = match saved_ram {
            None => (vec![0; ram_size], None),
            Some(bytes) if bytes.len() == ram_size => (bytes, None),
            Some(mut bytes) if kind.rtc && [ram_size + 44, ram_size + FOOTER].contains(&bytes.len()) => {
                let footer = bytes.split_off(ram_size);
                (bytes, Some(footer))
            }
            Some(bytes) => {
                let clock = if kind.rtc { format!(", plus an optional 44- or {FOOTER}-byte clock") } else { String::new() };
                return Err(format!("save is {} bytes, cartridge RAM is {ram_size}{clock}", bytes.len()));
            }
        };
        // MBC2's cells are nibbles that read back as 0xF0 | n. Each is held that way, so a
        // save carries the bytes the bus reads, and a file loads the same whatever its
        // high nibbles hold.
        if kind.mapper == MapperKind::Mbc2 {
            ram.iter_mut().for_each(|b| *b = 0xF0 | *b & 0x0F);
        }
        let mapper = match kind.mapper {
            MapperKind::Rom => Mapper::Rom,
            MapperKind::Mbc1 => Mapper::Mbc1 { low: 1, high: 0, mode: false, multicart: is_multicart(&rom) },
            MapperKind::Mbc2 => Mapper::Mbc2 { bank: 1 },
            MapperKind::Mbc3 => Mapper::Mbc3 { bank: 1, select: 0, rtc: kind.rtc },
            MapperKind::Mbc5 => Mapper::Mbc5 { rom_bank: 1, ram_bank: 0 },
        };
        let banks = rom.len() / 0x4000;
        Ok(Cartridge {
            rom,
            mapper,
            // ROM+RAM boards have no enable register: their RAM is always mapped.
            ram_enabled: kind.mapper == MapperKind::Rom && !ram.is_empty(),
            ram,
            rom_bank_mask: banks.next_power_of_two() - 1,
            has_battery: kind.battery && (ram_size > 0 || kind.rtc),
            battery: Battery::Clean,
            footer,
        })
    }

    /// Whether the board has MBC3's clock.
    pub fn has_battery(&self) -> bool {
        self.has_battery
    }

    /// Writes not yet taken as a save: (written, finished).
    pub fn pending_save(&self) -> (bool, bool) {
        (self.battery != Battery::Clean, self.battery == Battery::Committed)
    }

    pub fn has_clock(&self) -> bool {
        matches!(self.mapper, Mapper::Mbc3 { rtc: true, .. })
    }

    /// The clock footer the save ended with, once.
    pub fn take_footer(&mut self) -> Option<Vec<u8>> {
        self.footer.take()
    }

    /// Whether a write reaches the clock's chip as well as, or instead of, the mapper:
    /// the RAM enable, the select, the latch, and cartridge RAM while the select names a
    /// clock register. Bus decoding only: the clock decides what the write does.
    pub fn clock_hears(&self, addr: u16) -> bool {
        self.has_clock() && matches!(addr, 0x0000..=0x1FFF | 0x4000..=0x7FFF) || self.clock_at(addr)
    }

    /// Whether a cartridge RAM access reaches the clock's chip: A000-BFFF on a board with
    /// a clock, with the select at 08-0C.
    pub fn clock_at(&self, addr: u16) -> bool {
        (0xA000..=0xBFFF).contains(&addr) && matches!(self.mapper, Mapper::Mbc3 { select: 0x08..=0x0C, rtc: true, .. })
    }

    /// RAM the game wrote and then disabled since the last take: a finished save. On boards
    /// with a clock the machine appends the clock footer.
    pub fn take_committed(&mut self) -> Option<Vec<u8>> {
        self.take_if(|b| b == Battery::Committed)
    }

    /// RAM or clock registers written since the last take, finished or not, for shutdown.
    /// A clock that only ticked needs no save: the footer's time plus the time since it
    /// gives the same clock on reload.
    pub fn take_unsaved(&mut self) -> Option<Vec<u8>> {
        self.take_if(|b| b != Battery::Clean)
    }

    fn take_if(&mut self, wanted: impl Fn(Battery) -> bool) -> Option<Vec<u8>> {
        if !self.has_battery || !wanted(self.battery) {
            return None;
        }
        self.battery = Battery::Clean;
        Some(self.ram.clone())
    }

    fn rom_byte(&self, bank: usize, addr: u16) -> u8 {
        let i = (bank & self.rom_bank_mask) * 0x4000 + (addr as usize & 0x3FFF);
        self.rom.get(i).copied().unwrap_or(0xFF)
    }

    /// The ROM bank mapped at `addr` (0000-7FFF) under the current banking registers.
    fn rom_bank_at(&self, addr: u16) -> usize {
        match (addr, &self.mapper) {
            (0x0000..=0x3FFF, Mapper::Mbc1 { high, mode: true, multicart, .. }) => (*high as usize) << if *multicart { 4 } else { 5 },
            (0x0000..=0x3FFF, _) => 0,
            (_, Mapper::Rom) => 1,
            (_, Mapper::Mbc1 { low, high, multicart: true, .. }) => (*high as usize) << 4 | (*low & 0x0F) as usize,
            (_, Mapper::Mbc1 { low, high, .. }) => (*high as usize) << 5 | *low as usize,
            (_, Mapper::Mbc2 { bank } | Mapper::Mbc3 { bank, .. }) => *bank as usize,
            (_, Mapper::Mbc5 { rom_bank, .. }) => *rom_bank as usize,
        }
    }

    /// Where a cartridge access lands under the current banking: an index into the ROM,
    /// or into cartridge RAM. None for unmapped ROM, disabled RAM and the MBC3 clock.
    pub fn physical(&self, addr: u16) -> Option<Physical> {
        match addr {
            0x0000..=0x7FFF => {
                let i = (self.rom_bank_at(addr) & self.rom_bank_mask) * 0x4000 + (addr as usize & 0x3FFF);
                (i < self.rom.len()).then_some(Physical::Rom(i))
            }
            0xA000..=0xBFFF => match (&self.mapper, self.ram_enabled) {
                (Mapper::Mbc3 { select: 0x08..=0x0C, .. }, true) => None,
                (Mapper::Mbc2 { .. }, true) => Some(Physical::Ram(addr as usize & 0x1FF)),
                _ => self.ram_offset(addr).map(Physical::Ram),
            },
            _ => None,
        }
    }

    pub fn ram_len(&self) -> usize {
        self.ram.len()
    }

    pub fn ram(&self) -> &[u8] {
        &self.ram
    }

    /// The mapper and its registers as the model holds them (`memory::layout::Banking`):
    /// the raw register bytes, RAMG as 0x0A when RAM is enabled, and the cartridge
    /// facts, the clock among them. Fails for ROM sizes the fixed layout cannot hold.
    pub fn banking(&self, cgb: bool) -> Result<Banking, String> {
        use crate::memory::layout::{Mapper as M, Reg};
        let kind = match self.mapper {
            Mapper::Rom => M::Rom,
            Mapper::Mbc1 { .. } => M::Mbc1,
            Mapper::Mbc2 { .. } => M::Mbc2,
            Mapper::Mbc3 { .. } => M::Mbc3,
            Mapper::Mbc5 { .. } => M::Mbc5,
        };
        let mut b = Banking::new(kind, self.rom.len(), self.ram.len(), cgb)?;
        if self.has_clock() {
            b = b.with_clock();
        }
        b.set(Reg::Ramg, if self.ram_enabled { 0x0A } else { 0 });
        match self.mapper {
            Mapper::Rom => {}
            Mapper::Mbc1 { low, high, mode, multicart } => {
                if multicart {
                    b = b.multicart();
                }
                b.set(Reg::Romb0, low);
                b.set(Reg::Romb1, high);
                b.set(Reg::Mode, mode as u8);
            }
            Mapper::Mbc2 { bank } => b.set(Reg::Romb0, bank),
            Mapper::Mbc3 { bank, select, .. } => {
                b.set(Reg::Romb0, bank);
                b.set(Reg::Ramb, select);
            }
            Mapper::Mbc5 { rom_bank, ram_bank } => {
                b.set(Reg::Romb0, rom_bank as u8);
                b.set(Reg::Romb1, (rom_bank >> 8) as u8);
                b.set(Reg::Ramb, ram_bank);
            }
        }
        Ok(b)
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x7FFF => self.rom_byte(self.rom_bank_at(addr), addr),
            0xA000..=0xBFFF => match (&self.mapper, self.ram_enabled) {
                (Mapper::Mbc3 { select: 0x08..=0x0C, .. }, true) => 0xFF,
                (Mapper::Mbc2 { .. }, true) => self.ram[addr as usize & 0x1FF],
                _ => self.ram_offset(addr).map_or(0xFF, |i| self.ram[i]),
            },
            _ => 0xFF,
        }
    }

    pub fn write(&mut self, addr: u16, value: u8) {
        match (&mut self.mapper, addr) {
            (Mapper::Rom, 0x0000..=0x7FFF) => {}
            (Mapper::Mbc2 { bank }, 0x0000..=0x3FFF) => {
                if addr & 0x100 == 0 {
                    self.set_ram_enabled(value & 0x0F == 0x0A);
                } else {
                    *bank = (value & 0x0F).max(1);
                }
            }
            (Mapper::Mbc2 { .. }, 0x4000..=0x7FFF) => {}
            (_, 0x0000..=0x1FFF) => self.set_ram_enabled(value & 0x0F == 0x0A),
            (Mapper::Mbc1 { low, .. }, 0x2000..=0x3FFF) => *low = (value & 0x1F).max(1),
            (Mapper::Mbc1 { high, .. }, 0x4000..=0x5FFF) => *high = value & 0x03,
            (Mapper::Mbc1 { mode, .. }, 0x6000..=0x7FFF) => *mode = value & 1 != 0,
            (Mapper::Mbc3 { bank, .. }, 0x2000..=0x3FFF) => *bank = (value & 0x7F).max(1),
            (Mapper::Mbc3 { select, .. }, 0x4000..=0x5FFF) => *select = value,
            (Mapper::Mbc3 { .. }, 0x6000..=0x7FFF) => {}
            (Mapper::Mbc5 { rom_bank, .. }, 0x2000..=0x2FFF) => *rom_bank = (*rom_bank & 0x100) | value as u16,
            (Mapper::Mbc5 { rom_bank, .. }, 0x3000..=0x3FFF) => *rom_bank = (*rom_bank & 0xFF) | ((value as u16 & 1) << 8),
            (Mapper::Mbc5 { ram_bank, .. }, 0x4000..=0x5FFF) => *ram_bank = value & 0x0F,
            (Mapper::Mbc5 { .. }, 0x6000..=0x7FFF) => {}
            // The clock register itself is the control logic's; a write to it, like one
            // to RAM, is what the next save carries.
            (Mapper::Mbc3 { select: 0x08..=0x0C, rtc, .. }, 0xA000..=0xBFFF) => {
                if self.ram_enabled && *rtc {
                    self.battery = Battery::Dirty;
                }
            }
            (Mapper::Mbc2 { .. }, 0xA000..=0xBFFF) => {
                if self.ram_enabled {
                    self.ram[addr as usize & 0x1FF] = 0xF0 | value & 0x0F;
                    self.battery = Battery::Dirty;
                }
            }
            (_, 0xA000..=0xBFFF) => {
                if let Some(i) = self.ram_offset(addr) {
                    self.ram[i] = value;
                    self.battery = Battery::Dirty;
                }
            }
            _ => {}
        }
    }

    fn set_ram_enabled(&mut self, enabled: bool) {
        let was_enabled = std::mem::replace(&mut self.ram_enabled, enabled);
        if was_enabled && !enabled && self.battery == Battery::Dirty {
            self.battery = Battery::Committed;
        }
    }

    /// Index into `ram` for an A000-BFFF access, if RAM is enabled and present.
    fn ram_offset(&self, addr: u16) -> Option<usize> {
        if !self.ram_enabled || self.ram.is_empty() {
            return None;
        }
        let bank = match self.mapper {
            Mapper::Mbc1 { high, mode: true, multicart: false, .. } => high as usize,
            Mapper::Mbc3 { select: 0x08.., .. } => return None,
            Mapper::Mbc3 { select, .. } => select as usize & 0x07,
            Mapper::Mbc5 { ram_bank, .. } => ram_bank as usize,
            _ => 0,
        };
        let i = bank * 0x2000 + (addr as usize - 0xA000);
        Some(i % self.ram.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ROM of `banks` 16 KiB banks whose every bank starts with its bank number
    /// (low byte, then 0xB0 | high bits), with header type and RAM size codes set.
    fn rom(banks: usize, cartridge_type: u8, ram_code: u8) -> Vec<u8> {
        let mut rom = vec![0; banks * 0x4000];
        for bank in 0..banks {
            rom[bank * 0x4000] = bank as u8;
            rom[bank * 0x4000 + 1] = 0xB0 | (bank >> 8) as u8;
        }
        rom[0x147] = cartridge_type;
        rom[0x149] = ram_code;
        rom
    }

    fn mbc5_rom() -> Vec<u8> {
        rom(64, 0x1B, 0x03)
    }

    #[test]
    fn unknown_mapper_is_named_in_the_error() {
        assert_eq!(
            Cartridge::new(rom(2, 0x22, 0), None).err().unwrap(),
            "cartridge type 0x22 (MBC7) is not supported; supported: ROM only, MBC1, MBC2, MBC3, MBC5"
        );
        assert_eq!(Kind::from_type(0x13), Ok(Kind { mapper: MapperKind::Mbc3, battery: true, rtc: false }));
    }

    #[test]
    fn rom_only_ignores_writes() {
        let mut cart = Cartridge::new(rom(2, 0x00, 0), None).unwrap();
        cart.write(0x2000, 0x05);
        cart.write(0xA000, 0x12);
        assert_eq!((cart.read(0x0000), cart.read(0x4000), cart.read(0xA000)), (0, 1, 0xFF));
    }

    #[test]
    fn rom_with_battery_ram_is_always_mapped() {
        let mut cart = Cartridge::new(rom(2, 0x09, 0x02), None).unwrap();
        cart.write(0xA000, 0x5A);
        cart.write(0x0000, 0x00);
        assert_eq!(cart.read(0xA000), 0x5A, "no enable register");
        assert_eq!(cart.take_committed(), None, "no disable edge to commit on");
        assert_eq!(cart.take_unsaved().map(|r| (r.len(), r[0])), Some((0x2000, 0x5A)), "saved at shutdown");
    }

    #[test]
    fn mbc5_rom_banking() {
        let mut cart = Cartridge::new(mbc5_rom(), None).unwrap();
        assert_eq!(cart.read(0x4000), 1, "bank 1 mapped at reset");
        cart.write(0x2000, 0x2A);
        assert_eq!(cart.read(0x4000), 0x2A);
        cart.write(0x2000, 0x00);
        assert_eq!(cart.read(0x4000), 0x00, "MBC5 maps bank 0 at 4000 when selected");
        cart.write(0x2000, 0x45);
        assert_eq!(cart.read(0x4000), 0x05, "bank masked to 64 banks of a 1 MiB ROM");
        cart.write(0x3000, 0x01);
        cart.write(0x2000, 0x03);
        assert_eq!(cart.read(0x4000), 0x03, "bit 8 wraps on a 64-bank ROM");
        assert_eq!(cart.read(0x0000), 0x00, "fixed bank 0 unaffected");
    }

    #[test]
    fn mbc5_ninth_bank_bit() {
        let mut cart = Cartridge::new(rom(512, 0x19, 0), None).unwrap();
        cart.write(0x2000, 0x01);
        cart.write(0x3000, 0x01);
        assert_eq!((cart.read(0x4000), cart.read(0x4001)), (0x01, 0xB1));
    }

    #[test]
    fn mbc5_ram_enable_and_banks() {
        let mut cart = Cartridge::new(mbc5_rom(), None).unwrap();
        cart.write(0xA000, 0x11);
        assert_eq!(cart.read(0xA000), 0xFF, "disabled RAM reads open bus");
        cart.write(0x0000, 0x0A);
        for bank in 0..4u8 {
            cart.write(0x4000, bank);
            cart.write(0xA123, 0x40 + bank);
        }
        for bank in 0..4u8 {
            cart.write(0x4000, bank);
            assert_eq!(cart.read(0xA123), 0x40 + bank);
        }
        cart.write(0x4000, 0x05);
        assert_eq!(cart.read(0xA123), 0x41, "RAM bank masked to 4 banks");
        cart.write(0x0000, 0x00);
        assert_eq!(cart.read(0xA123), 0xFF);
    }

    #[test]
    fn disabling_ram_after_a_write_commits_a_save() {
        let mut cart = Cartridge::new(mbc5_rom(), None).unwrap();
        cart.write(0x0000, 0x0A);
        cart.write(0x0000, 0x00);
        assert_eq!(cart.take_unsaved(), None, "enable/disable without a write");
        cart.write(0x0000, 0x0A);
        cart.write(0x4000, 0x02);
        cart.write(0xA001, 0x5C);
        assert_eq!(cart.take_committed(), None, "RAM still enabled: save may be mid-write");
        cart.write(0x0000, 0x00);
        let saved = cart.take_committed().unwrap().to_vec();
        assert_eq!((saved.len(), saved[0x4001]), (0x8000, 0x5C));
        assert_eq!(cart.take_committed(), None, "taken once");
        let mut reloaded = Cartridge::new(mbc5_rom(), Some(saved)).unwrap();
        reloaded.write(0x0000, 0x0A);
        reloaded.write(0x4000, 0x02);
        assert_eq!(reloaded.read(0xA001), 0x5C, "saved RAM loads on start");
        assert!(Cartridge::new(mbc5_rom(), Some(vec![0; 3])).is_err());
    }

    #[test]
    fn ram_without_battery_never_reports_a_save() {
        let mut cart = Cartridge::new(rom(4, 0x02, 0x02), None).unwrap();
        cart.write(0x0000, 0x0A);
        cart.write(0xA000, 0x33);
        cart.write(0x0000, 0x00);
        assert_eq!(cart.take_unsaved(), None);
        cart.write(0x0000, 0x0A);
        assert_eq!(cart.read(0xA000), 0x33);
    }

    #[test]
    fn mbc1_low_bank_zero_maps_to_one_before_high_bits() {
        let mut cart = Cartridge::new(rom(128, 0x01, 0), None).unwrap();
        cart.write(0x2000, 0x03);
        assert_eq!(cart.read(0x4000), 3);
        cart.write(0x2000, 0x00);
        assert_eq!(cart.read(0x4000), 1);
        cart.write(0x2000, 0x20);
        assert_eq!(cart.read(0x4000), 1, "only 5 bits are kept, and 0 becomes 1");
        cart.write(0x4000, 0x01);
        assert_eq!(cart.read(0x4000), 0x21, "high bits make bank 0x20 unreachable at 4000");
        cart.write(0x4000, 0x03);
        cart.write(0x2000, 0x05);
        assert_eq!(cart.read(0x4000), 0x65);
        assert_eq!(cart.read(0x0000), 0x00, "mode 0: 0000 is bank 0");
        cart.write(0x6000, 0x01);
        assert_eq!(cart.read(0x0000), 0x60, "mode 1: 0000 is bank high<<5");
        cart.write(0x4000, 0x07);
        assert_eq!(cart.read(0x0000), 0x60, "high register is 2 bits");
    }

    #[test]
    fn mbc1_ram_banking_needs_mode_1() {
        let mut cart = Cartridge::new(rom(4, 0x03, 0x03), None).unwrap();
        cart.write(0x0000, 0x0A);
        cart.write(0x4000, 0x02);
        cart.write(0xA010, 0x77);
        cart.write(0x6000, 0x01);
        assert_eq!(cart.read(0xA010), 0x00, "mode 1 maps RAM bank 2, which is empty");
        cart.write(0xA010, 0x88);
        cart.write(0x6000, 0x00);
        assert_eq!(cart.read(0xA010), 0x77, "mode 0 always maps RAM bank 0");
        cart.write(0x6000, 0x01);
        assert_eq!(cart.read(0xA010), 0x88);
        cart.write(0x0000, 0x00);
        assert_eq!(cart.take_committed().unwrap()[0x4010], 0x88);
    }

    #[test]
    fn mbc2_banking_and_nibble_ram() {
        let mut cart = Cartridge::new(rom(16, 0x06, 0), None).unwrap();
        cart.write(0x2100, 0x05);
        assert_eq!(cart.read(0x4000), 5, "address bit 8 set: ROM bank");
        cart.write(0x2100, 0x00);
        assert_eq!(cart.read(0x4000), 1, "bank 0 becomes 1");
        cart.write(0x2100, 0x1F);
        assert_eq!(cart.read(0x4000), 15, "4-bit bank");
        cart.write(0xA000, 0x0C);
        assert_eq!(cart.read(0xA000), 0xFF, "RAM disabled");
        cart.write(0x0000, 0x0A);
        assert_eq!(cart.read(0x4000), 15, "address bit 8 clear: RAM enable, bank unchanged");
        cart.write(0xA000, 0xAC);
        assert_eq!(cart.read(0xA000), 0xFC, "lower nibble stored, upper reads 1s");
        assert_eq!(cart.read(0xA200), 0xFC, "512 nibbles echo through A000-BFFF");
        assert_eq!(cart.read(0xBFFF), 0xF0);
        cart.write(0x0000, 0x00);
        let save = cart.take_committed().unwrap();
        assert_eq!((save.len(), save[0], save[1]), (512, 0xFC, 0xF0), "saved as the bus reads it");
        let loaded = Cartridge::new(rom(16, 0x06, 0), Some([0x05, 0xF7].repeat(256))).unwrap();
        assert_eq!(loaded.ram()[..2], [0xF5, 0xF7], "a save loads the same whatever its high nibbles");
    }

    #[test]
    fn mbc3_banking_and_ram() {
        let mut cart = Cartridge::new(rom(128, 0x13, 0x03), None).unwrap();
        cart.write(0x2000, 0x00);
        assert_eq!(cart.read(0x4000), 1);
        cart.write(0x2000, 0x7F);
        assert_eq!(cart.read(0x4000), 0x7F);
        cart.write(0x2000, 0xC2);
        assert_eq!(cart.read(0x4000), 0x42, "7-bit bank");
        cart.write(0x0000, 0x0A);
        cart.write(0x4000, 0x03);
        cart.write(0xA555, 0x99);
        cart.write(0x4000, 0x00);
        assert_eq!(cart.read(0xA555), 0x00);
        cart.write(0x4000, 0x03);
        assert_eq!(cart.read(0xA555), 0x99);
    }

    #[test]
    fn mbc30_has_eight_ram_banks() {
        let mut cart = Cartridge::new(rom(128, 0x13, 0x05), None).unwrap();
        cart.write(0x0000, 0x0A);
        for bank in 0..8u8 {
            cart.write(0x4000, bank);
            cart.write(0xA010, 0x60 + bank);
        }
        let read: Vec<u8> = (0..8u8).map(|bank| {
            cart.write(0x4000, bank);
            cart.read(0xA010)
        }).collect();
        assert_eq!(read, [0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67]);
    }

    #[test]
    fn mbc3_without_a_clock_reads_ff_at_rtc_selects() {
        let mut cart = Cartridge::new(rom(4, 0x13, 0x03), None).unwrap();
        cart.write(0x0000, 0x0A);
        cart.write(0xA000, 0x42);
        cart.write(0x4000, 0x08);
        cart.write(0xA000, 0x17);
        cart.write(0x6000, 0x00);
        cart.write(0x6000, 0x01);
        assert_eq!(cart.read(0xA000), 0xFF, "no RTC register 08");
        cart.write(0x4000, 0x00);
        assert_eq!(cart.read(0xA000), 0x42, "the write at select 08 did not reach RAM");
    }
}
