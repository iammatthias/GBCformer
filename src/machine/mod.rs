//! Game Boy host: the bus as a range-to-device table ([`route`]), the cartridge, the
//! reference devices (timer, PPU, APU, joypad, serial, OAM DMA, HDMA) behind
//! [`crate::device::Device`], the interrupt registers and, in CGB mode, banked WRAM,
//! KEY0/KEY1. The scheduler in [`Machine::advance`] steps every device once per block
//! after each CPU step.

pub mod apu;
pub mod cart;
pub mod control;
pub mod dma;
pub mod fifo;
pub mod io;
pub mod lcd;
pub mod ppu;
pub mod rtc;
pub mod timer;

use crate::device::{Device, Signals};
use crate::memory::image::Image;
use crate::memory::layout::{self, Access, Banking, Dma, Port, Reg};
use crate::memory::{Addr, Copied};
use crate::rom::Header;
use crate::step::{self, Bus, Core, Cpu};
use cart::Cartridge;
use apu::{Apu, AudioDevice, HighPass, Sample};
use control::{Control, ControlDevice, Lines, PaletteRead};
use io::Buttons;
use ppu::{Frame, OamBug, Ppu};

/// 70224 dots per frame at 4 dots per M-cycle (normal speed). A frame is a span of
/// emulated time, so frame counts keep advancing while the game has the LCD off.
pub const MCYCLES_PER_FRAME: u64 = 17556;

/// Emulated time in half M-cycles of normal speed (2 dots). An M-cycle is 2 of them at
/// normal speed and 1 at CGB double speed.
pub const HALVES_PER_FRAME: u64 = 2 * MCYCLES_PER_FRAME;

/// The console being emulated. CGB hardware runs a DMG-only game in compatibility mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hardware {
    Dmg,
    Cgb,
}

impl Hardware {
    /// The console to emulate for a cartridge: `requested`, or with None, CGB when the game
    /// is CGB-aware. CGB hardware runs a DMG-only game in DMG compatibility mode.
    pub fn resolve(header: &Header, requested: Option<Hardware>) -> Result<Hardware, String> {
        match requested {
            Some(h) => Ok(h),
            None if header.cgb_aware() => Ok(Hardware::Cgb),
            None => Ok(Hardware::Dmg),
        }
    }

    /// The console a boot ROM is for, by its length: 256 bytes DMG, 2,304 CGB.
    pub fn of_boot(boot: &[u8]) -> Option<Hardware> {
        match boot.len() {
            0x100 => Some(Hardware::Dmg),
            0x900 => Some(Hardware::Cgb),
            _ => None,
        }
    }
}

/// Offsets of each region in the flat memory space; ROM starts at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryLayout {
    pub ram: u32,
    pub wram: u32,
    pub vram: u32,
    pub oam: u32,
    pub hram: u32,
    pub total: u32,
}

/// Who answers a bus address. The table is [`route`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dev {
    Cart,
    Ppu,
    Wram,
    Unusable,
    /// The timer, IF, IE, P1, serial, OAM DMA and HDMA control, KEY0, KEY1 and SVBK.
    Control,
    Apu,
    Hram,
    None,
}

/// Who computes the APU: the reference, or the device that replaced it at a hand-off
/// (a compiled checkpoint, alone or in lockstep with the reference).
enum ApuSlot {
    Reference(Apu),
    Replaced(Box<dyn AudioDevice>),
}

impl Device for ApuSlot {
    fn read(&self, addr: u16) -> u8 {
        match self {
            ApuSlot::Reference(a) => Device::read(a, addr),
            ApuSlot::Replaced(d) => d.read(addr),
        }
    }

    fn write(&mut self, addr: u16, value: u8, s: &mut Signals) {
        match self {
            ApuSlot::Reference(a) => Device::write(a, addr, value, s),
            ApuSlot::Replaced(d) => d.write(addr, value, s),
        }
    }

    fn advance(&mut self, mcycles: u64, s: &mut Signals) {
        match self {
            ApuSlot::Reference(a) => a.advance(mcycles, s),
            ApuSlot::Replaced(d) => d.advance(mcycles, s),
        }
    }

    fn horizon(&self) -> u64 {
        match self {
            ApuSlot::Reference(a) => a.horizon(),
            ApuSlot::Replaced(d) => d.horizon(),
        }
    }
}

impl ApuSlot {
    fn device(&mut self) -> &mut dyn AudioDevice {
        match self {
            ApuSlot::Reference(a) => a,
            ApuSlot::Replaced(d) => d.as_mut(),
        }
    }
}

/// The control checkpoint without a lockstep, the common replacement.
pub type CompiledControl = crate::control_circuit::ControlModel<crate::control_circuit::ControlCheckpoint<crate::engine::Engine>>;

/// What replaces the reference timer and interrupt flags at a hand-off.
pub enum ControlReplacement {
    Compiled(Box<CompiledControl>),
    Device(Box<dyn ControlDevice>),
}

/// Who computes the timer and the interrupt flags: the reference, or what replaced it
/// at a hand-off. The reference and the compiled checkpoint are matched inline on every
/// access, so neither calls through a trait object; the halted CPU touches the slot on
/// every M-cycle.
enum ControlSlot {
    Reference(Control),
    Compiled(Box<CompiledControl>),
    Replaced(Box<dyn ControlDevice>),
}

macro_rules! control {
    ($slot:expr, $c:ident => $e:expr) => {
        match $slot {
            ControlSlot::Reference($c) => $e,
            ControlSlot::Compiled($c) => $e,
            ControlSlot::Replaced($c) => $e,
        }
    };
}

/// A call that may change the control device, then the latch of its [`Lines`], in one match.
macro_rules! control_mut {
    ($m:expr, $c:ident => $e:expr) => {{
        let (r, lines) = control!(&mut $m.control, $c => {
            let r = $e;
            (r, $c.lines())
        });
        $m.lines = lines;
        r
    }};
}

/// `control_mut!` for a call that may be a pass moving a copy's bytes, which land at once.
macro_rules! control_landing {
    ($m:expr, $c:ident, $landed:ident => $e:expr) => {{
        let mut $landed = Vec::new();
        let r = control_mut!($m, $c => $e);
        $m.land(&$landed);
        r
    }};
}

/// The boot registers the machine mirrors as the memory model's cells hold them:
/// BOOTING (a power-on run), FF50's byte on UNMAPPED or COMPAT, whichever it landed on,
/// and the KEY0 the boot ROM wrote, which picks compatibility mode at FF50.
#[derive(Clone, Copy, Debug, Default)]
struct BootRegs {
    booting: u8,
    unmapped: u8,
    compat: u8,
    key0: u8,
}

impl BootRegs {
    const ALL: [Reg; 4] = [Reg::Booting, Reg::Unmapped, Reg::Compat, Reg::Key0];

    fn get(&self, r: Reg) -> u8 {
        match r {
            Reg::Booting => self.booting,
            Reg::Unmapped => self.unmapped,
            Reg::Compat => self.compat,
            Reg::Key0 => self.key0,
            _ => unreachable!("{r:?} is not a boot register"),
        }
    }

    fn set(&mut self, r: Reg, v: u8) {
        match r {
            Reg::Booting => self.booting = v,
            Reg::Unmapped => self.unmapped = v,
            Reg::Compat => self.compat = v,
            Reg::Key0 => self.key0 = v,
            _ => unreachable!("{r:?} is not a boot register"),
        }
    }

    /// The boot register at physical address `p`, if it is one.
    fn at(p: u32) -> Option<Reg> {
        BootRegs::ALL.into_iter().find(|&r| p == layout::REGS + r as u32)
    }
}

/// The bus map. CGB-only registers route nowhere on DMG. Inlined so each caller's match
/// on the result fuses with this one; out of line it was 3% of host samples.
#[inline(always)]
fn route(addr: u16, hardware: bool, cgb: bool) -> Dev {
    match addr {
        0x0000..=0x7FFF | 0xA000..=0xBFFF => Dev::Cart,
        0x8000..=0x9FFF | 0xFE00..=0xFE9F => Dev::Ppu,
        0xC000..=0xFDFF => Dev::Wram,
        0xFEA0..=0xFEFF => Dev::Unusable,
        0xFF00..=0xFF02 | 0xFF04..=0xFF07 | 0xFF0F | 0xFFFF => Dev::Control,
        0xFF10..=0xFF3F => Dev::Apu,
        // PCM12 and PCM34, the channels' digital outputs, on CGB hardware in either mode.
        0xFF76 | 0xFF77 if hardware => Dev::Apu,
        // STAT, LY and LYC are the PPU's timing, which the control logic holds; an LCDC
        // write reaches both (`Machine::write_unblocked`).
        0xFF41 | 0xFF44 | 0xFF45 | 0xFF46 => Dev::Control,
        0xFF40..=0xFF4B | 0xFF4F => Dev::Ppu,
        // The palette index registers are control logic; a data port write hands the PPU
        // its byte (`Signals::palette`). CGB hardware keeps the index registers in either
        // mode; the data ports are CGB mode's.
        0xFF68 | 0xFF6A if hardware => Dev::Control,
        0xFF69 | 0xFF6B if cgb => Dev::Control,
        0xFF4C | 0xFF4D | 0xFF70 | 0xFF74 if cgb => Dev::Control,
        // FF72, FF73 and FF75 hold their bytes on CGB hardware in either mode.
        0xFF72 | 0xFF73 | 0xFF75 if hardware => Dev::Control,
        0xFF51..=0xFF55 if cgb => Dev::Control,
        0xFF80..=0xFFFE => Dev::Hram,
        _ => Dev::None,
    }
}

/// Declared in the order the scheduler touches fields, hot first. `repr(C)` keeps the
/// scalars every block reads in two cache lines instead of letting field reordering
/// scatter them around the 32 KB `wram` and the PPU's frame buffer.
#[repr(C)]
pub struct Machine {
    /// The current block's signals; kept to reuse its copy buffer.
    signals: Signals,
    /// The CPU ran STOP; the control logic sees it after the step's M-cycles.
    stop_pending: bool,
    /// Emulated time in half M-cycles of normal speed.
    halves: u64,
    /// The control device's IE, IF, counter and horizon as of its last call.
    lines: Lines,
    /// The timer, IF and IE.
    control: ControlSlot,
    hardware: Hardware,
    /// DMG compatibility mode on CGB hardware: the PPU draws as DMG through CGB palettes,
    /// and the CGB-only registers are gone.
    compat: bool,
    /// ROM bank 0 with the boot ROM over it, while the boot ROM is mapped
    /// (`Machine::power_on`, `layout::BOOT`).
    boot: Option<Box<[u8]>>,
    boot_regs: BootRegs,
    apu: ApuSlot,
    /// The board's output capacitor, applied as presentation to the APU's mix whatever
    /// holds the slot.
    speaker: HighPass,
    cart: Cartridge,
    /// The Unix second the host last handed the cartridge clock, the reference point of
    /// the next elapsed time (wall-clock bookkeeping, a host input).
    clock_time: Option<u64>,
    hram: [u8; 0x7F],
    /// Every byte a DMA copied since `GameBoy::step` last handed them to the core.
    copied: Vec<Copied>,
    /// Whether the host keeps its copy of WRAM, HRAM, VRAM, OAM and VBK as the model's
    /// records change them: only something run beside the model reads it
    /// ([`Machine::drop_shadow`]).
    shadow: bool,
    /// `Counters::dma_blocked`, `Counters::palette_reads` and `Counters::oam_bugs`.
    pub(crate) hits: (u64, u64, u64),
    /// The M-cycles of the current step the devices already ran before its accesses
    /// (`Request::cycles`, `docs/hardware-tests.md` §4.1).
    ticked: u64,
    /// A step's ticks handed DMA copies to the core: its memory is audited after the step.
    tick_copies: bool,
    /// A GBC CPU step is running: the blocks the control logic holds (`Signals::held`)
    /// wait for its end.
    in_step: bool,
    /// The HDMA blocks the control logic held to the step's end.
    held: Vec<crate::device::Copy>,
    /// Bank 0 at C000, banks 1-7 (CGB) at D000.
    wram: [u8; 0x8000],
    pub(crate) ppu: Ppu,
}

impl Machine {
    /// Console state after the boot ROM, without running one: a labelled convenience. The
    /// post-boot registers are a snapshot handed in as data, and the cartridge header's CGB
    /// byte picks CGB mode or DMG compatibility as the boot ROM would have. A run that must
    /// compute all of it starts from [`Machine::power_on`] instead.
    pub fn new(rom: Vec<u8>, saved_ram: Option<Vec<u8>>, hardware: Hardware) -> Result<Self, String> {
        let cgb = hardware == Hardware::Cgb;
        let key0 = rom.get(0x143).copied().unwrap_or(0);
        let compat = cgb && key0 & 0x80 == 0;
        let mut ppu = if compat { Ppu::post_boot_compat() } else { Ppu::post_boot(cgb) };
        if !cgb {
            ppu.boot_logo(rom.get(0x104..0x134).unwrap_or(&[0; 48]));
        }
        Machine::assemble(rom, saved_ram, hardware, ppu, Control::post_boot(cgb, key0), Apu::post_boot(cgb), compat, None)
    }

    /// The console at power-on with `boot` mapped over the cartridge, its boot ROM: 256
    /// bytes at 0000-00FF on DMG, 2,304 on CGB, also at 0200-08FF (the header shows
    /// through between). The GBC CPU starts at 0000 and every device at its power-on state;
    /// nothing is read from the header. The boot ROM writes KEY0 and unmaps itself by
    /// writing FF50, and the memory model's COMPAT cell says then whether the console left
    /// CGB mode. The file is one you supply: it is read, never stored with the repository.
    pub fn power_on(rom: Vec<u8>, saved_ram: Option<Vec<u8>>, hardware: Hardware, boot: Vec<u8>) -> Result<Self, String> {
        let cgb = hardware == Hardware::Cgb;
        let want = if cgb { 0x900 } else { 0x100 };
        if boot.len() != want {
            return Err(format!("a {hardware:?} boot ROM is {want} bytes; this one is {}", boot.len()));
        }
        // Bank 0 as the bus shows it while the boot ROM is mapped (`layout::BOOT`): the
        // boot ROM over the cartridge's bank 0, which shows through 0100-01FF and above.
        let mut bank0 = rom.get(..0x4000).ok_or("a cartridge of at least 16 KB")?.to_vec();
        for (a, &b) in boot.iter().enumerate().filter(|&(a, _)| !(0x100..0x200).contains(&a)) {
            bank0[a] = b;
        }
        Machine::assemble(rom, saved_ram, hardware, Ppu::power_on(cgb), Control::power_on(cgb), Apu::power_on(cgb), false, Some(bank0.into_boxed_slice()))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(rom: Vec<u8>, saved_ram: Option<Vec<u8>>, hardware: Hardware, ppu: Ppu, control: Control, apu: Apu, compat: bool, boot: Option<Box<[u8]>>) -> Result<Self, String> {
        let mut cart = Cartridge::new(rom, saved_ram)?;
        let (clock, clock_time) = cart.take_footer().map_or((rtc::Rtc::default(), None), |f| rtc::Rtc::from_footer(&f));
        let control = control.with_clock(clock).with_lengths(ppu.lengths());
        Ok(Machine {
            hardware,
            cart,
            clock_time,
            wram: [0; 0x8000],
            hram: [0; 0x7F],
            lines: control.lines(),
            control: ControlSlot::Reference(control),
            ppu,
            compat,
            boot_regs: BootRegs { booting: boot.is_some() as u8, compat: compat as u8, ..BootRegs::default() },
            boot,
            apu: ApuSlot::Reference(apu),
            speaker: apu::board(),
            signals: Signals::default(),
            stop_pending: false,
            halves: 0,
            copied: Vec::new(),
            shadow: true,
            hits: (0, 0, 0),
            ticked: 0,
            tick_copies: false,
            in_step: false,
            held: Vec::new(),
        })
    }

    /// Plugs one end of a link cable into the serial port, the control logic's.
    pub fn plug(&mut self, link: io::Link) -> Result<(), String> {
        control_mut!(self, c => c.plug(link))
    }

    /// Ordinary STOP holds the GBC CPU: the scheduler only advances time.
    pub fn stopped(&self) -> bool {
        self.lines.paused
    }

    /// Whether the PPU holds `addr` from the GBC CPU now: VRAM in mode 3, OAM in modes 2
    /// and 3, as the control logic's locks say (`Lines::locks`, a pass when they lapsed).
    #[inline]
    /// Whether the PPU locks `addr` from a read, or with `write` a write (`Lines::locks`).
    fn ppu_locked(&mut self, addr: u16, write: bool) -> bool {
        let shift = if write { step::WRITE_SHIFT } else { 0 };
        let bit = match addr {
            0x8000..=0x9FFF => step::VRAM_LOCKED << shift,
            0xFE00..=0xFE9F => step::OAM_LOCKED << shift,
            _ => return false,
        };
        let locks = match self.lines.locks {
            Some(l) => l,
            None => self.lock_query(),
        };
        locks & bit != 0
    }

    /// The boot ROM is still mapped.
    pub fn booting(&self) -> bool {
        self.boot.is_some()
    }

    /// The boot ROM's length while it is mapped.
    pub fn boot_len(&self) -> Option<usize> {
        self.boot.as_ref().map(|b| b.len())
    }

    /// The boot ROM byte a read of `addr` sees while it is mapped.
    #[inline]
    fn boot_byte(&self, addr: u16) -> Option<u8> {
        self.boot.as_deref()?.get(addr as usize).copied()
    }

    /// The reference machine's FF50 (the GBC CPU on the host with the memory there too):
    /// the boot ROM unmaps for good, and a CGB whose boot ROM wrote KEY0 bit 2 continues in
    /// DMG compatibility mode. With the memory in weights the model's own record decides
    /// (`Bus::write_memory`).
    fn reference_ff50(&mut self, value: u8) {
        if self.boot.is_some() {
            let compat = self.ff50_compat();
            self.boot_regs.set(if compat { Reg::Compat } else { Reg::Unmapped }, value);
            if value & 1 != 0 {
                self.unmap_boot(compat);
            }
        }
    }

    /// The reference's rule for FF50's register: COMPAT on a CGB whose boot ROM wrote KEY0
    /// bit 2, UNMAPPED otherwise.
    fn ff50_compat(&self) -> bool {
        self.hardware == Hardware::Cgb && self.boot_regs.key0 & 0x04 != 0
    }

    /// The boot ROM unmapped, the console left in DMG compatibility mode when `compat`.
    fn unmap_boot(&mut self, compat: bool) {
        self.boot = None;
        if compat {
            self.compat = true;
            self.ppu.set_compat(true);
        }
    }

    /// The console is a CGB, in CGB mode or not.
    pub fn cgb_hardware(&self) -> bool {
        self.hardware == Hardware::Cgb
    }

    pub fn hardware(&self) -> Hardware {
        self.hardware
    }

    /// CGB mode: CGB hardware, not in DMG compatibility mode. Its registers (VBK's bank,
    /// SVBK, KEY1, HDMA, the palette ports) exist only then.
    fn cgb(&self) -> bool {
        self.hardware == Hardware::Cgb && !self.compat
    }

    /// A CGB running a DMG cartridge in DMG compatibility mode.
    pub fn compat(&self) -> bool {
        self.compat
    }

    /// Emulated time so far, in half M-cycles of normal speed.
    pub fn halves(&self) -> u64 {
        self.halves
    }

    /// The last finished frame, waiting for a renderer on its own thread to finish it.
    pub fn frame(&mut self) -> &Frame {
        self.ppu.settle();
        self.ppu.frame()
    }

    /// Bytes the game sent over the serial port.
    pub fn serial_output(&self) -> &[u8] {
        control!(&self.control, c => c.serial_output())
    }

    /// Cartridge RAM as the host holds it, every bank (test ROMs report their results there).
    pub fn cart_ram(&self) -> &[u8] {
        self.cart.ram()
    }

    /// The one flat byte space the memory prototypes hold (`crate::memory`): ROM, then
    /// cartridge RAM, WRAM (all banks), VRAM (both banks), OAM and HRAM.
    pub fn memory_layout(&self) -> MemoryLayout {
        let rom = self.cart.rom().len() as u32;
        let ram = rom + self.cart.ram_len() as u32;
        let wram = ram;
        let vram = wram + 0x8000;
        let oam = vram + 0x4000;
        let hram = oam + 0xA0;
        MemoryLayout { ram, wram, vram, oam, hram, total: hram + 0x7F }
    }

    /// Where a bus address lands in the flat space under the current banking, or None
    /// for I/O registers, IE, unmapped ROM, disabled cartridge RAM and the MBC3 clock.
    pub fn physical(&self, addr: u16) -> Option<u32> {
        let l = self.memory_layout();
        Some(match addr {
            0x0000..=0x7FFF | 0xA000..=0xBFFF => match self.cart.physical(addr)? {
                cart::Physical::Rom(i) => i as u32,
                cart::Physical::Ram(i) => self.cart.rom().len() as u32 + i as u32,
            },
            0x8000..=0x9FFF => {
                let bank = if self.cgb() { self.ppu.vbk() as u32 } else { 0 };
                l.vram + bank * 0x2000 + (addr as u32 & 0x1FFF)
            }
            0xC000..=0xFDFF => l.wram + self.wram_index(addr) as u32,
            0xFE00..=0xFE9F => l.oam + (addr as u32 - 0xFE00),
            0xFF80..=0xFFFE => l.hram + (addr as u32 - 0xFF80),
            _ => return None,
        })
    }

    /// The banking registers as the model holds them (`memory::layout`).
    pub fn banking(&self) -> Result<Banking, String> {
        // MAPPER bit 7 is CGB hardware; the boot registers are as the model's cells hold them.
        let mut b = self.cart.banking(self.hardware == Hardware::Cgb)?;
        for r in BootRegs::ALL {
            b.set(r, self.boot_regs.get(r));
        }
        b.set(Reg::Svbk, self.lines.svbk);
        // VBK as the PPU holds it; DMG and compatibility mode read FF4F as 0xFF, but their
        // VRAM has one bank.
        b.set(Reg::Vbk, if self.cgb() { self.ppu.vbk() } else { 0 });
        Ok(b)
    }

    /// The model's memory image of this machine's state: ROM, every RAM, the markers and
    /// the banking registers, for insertion into a memory core.
    pub fn memory_image(&self) -> Result<Image, String> {
        let banking = self.banking()?;
        let mut i = Image::new(self.cart.rom(), self.cart.ram_len(), banking);
        i.fill(layout::CRAM, self.cart.ram());
        i.fill(layout::WRAM, &self.wram);
        i.fill(layout::VRAM, self.ppu.vram());
        i.fill(layout::OAM, &self.ppu.oam);
        i.fill(layout::HRAM, &self.hram);
        if let Some(boot) = &self.boot {
            i.fill(layout::BOOT, boot);
        }
        Ok(i)
    }

    /// The host's bytes of every writable region the model holds, at fixed-layout bases.
    pub fn ram_regions(&self) -> [(Addr, &[u8]); 5] {
        [(layout::CRAM, self.cart.ram()), (layout::WRAM, &self.wram), (layout::VRAM, self.ppu.vram()), (layout::OAM, &self.ppu.oam), (layout::HRAM, &self.hram)]
    }

    /// VRAM (both banks) and OAM, the regions a DMA writes.
    pub fn video_regions(&self) -> [(Addr, &[u8]); 2] {
        [(layout::VRAM, self.ppu.vram()), (layout::OAM, &self.ppu.oam)]
    }

    /// The 2-dot halves a frame of emulated time lasts, as the control device emits it.
    pub fn frame_halves(&self) -> u64 {
        self.lines.frame
    }

    /// Whether OAM DMA blocks the GBC CPU's accesses on its buses (`layout::Dma`).
    pub fn dma_active(&self) -> bool {
        self.lines.dma_active
    }

    /// Where a bus access lands in the fixed physical layout (`memory::layout`) under
    /// the current banking registers and the bus state `state` (the access locks and OAM
    /// DMA's block, `step::slot_state`): the specification the model's translation is
    /// held to in lockstep.
    pub fn fixed_physical(&self, addr: u16, access: Access, state: u8) -> u32 {
        layout::translate_locked(&self.banking().expect("a loaded cartridge fits the fixed layout"), addr, access, dma_at(&self.lines, state), state)
    }

    /// The bus state now: the control logic's access locks (0 where they lapsed) and
    /// whether OAM DMA blocks.
    pub fn access_state(&self) -> u8 {
        self.lines.locks.unwrap_or(0) & (step::VRAM_LOCKED | step::OAM_LOCKED) | (self.lines.dma_active as u8) << 2
    }

    /// C000-FDFF, echo included, to an index into `wram`.
    fn wram_index(&self, addr: u16) -> usize {
        let a = addr as usize & 0x1FFF;
        if a < 0x1000 { a } else { (self.lines.svbk as usize & 7).max(1) * 0x1000 + (a - 0x1000) }
    }

    pub fn set_buttons(&mut self, pressed: Buttons) {
        if let Some(held) = control_landing!(self, c, landed => c.buttons(pressed, &mut landed)) {
            self.ppu.set_stopped(held);
        }
    }

    /// The reference timer and interrupt flags, while no device has replaced them.
    pub fn control_reference(&self) -> Option<&Control> {
        match &self.control {
            ControlSlot::Reference(c) => Some(c),
            ControlSlot::Compiled(_) | ControlSlot::Replaced(_) => None,
        }
    }

    /// Hands the timer and the interrupt flags to `device` from this M-cycle on.
    pub fn replace_control(&mut self, device: ControlReplacement) {
        self.control = match device {
            ControlReplacement::Compiled(m) => ControlSlot::Compiled(m),
            ControlReplacement::Device(d) => ControlSlot::Replaced(d),
        };
        self.lines = control!(&self.control, c => c.lines());
    }

    pub fn control_describe(&self) -> Vec<String> {
        match &self.control {
            ControlSlot::Reference(_) => Vec::new(),
            ControlSlot::Compiled(m) => m.describe(),
            ControlSlot::Replaced(d) => d.describe(),
        }
    }

    pub fn control_report(&mut self) -> Vec<String> {
        control!(&mut self.control, c => c.report())
    }

    /// Host wall-clock seconds, for a cartridge clock (MBC3). Call at least once per frame.
    /// The unit matters: a host passing milliseconds (`Date.now()` as is) runs the clock
    /// 1000x fast. The control logic gets the seconds elapsed since the last call; the
    /// clock's own rules (the split into its units, halt, carries, the day counter's wrap)
    /// are the control logic's. Time that runs backwards elapses nothing.
    pub fn set_time(&mut self, unix_seconds: u64) {
        if !self.cart.has_clock() {
            return;
        }
        if let Some(last) = self.clock_time.replace(unix_seconds) {
            let seconds = unix_seconds.saturating_sub(last);
            if seconds > 0 {
                control_landing!(self, c, landed => c.elapse(seconds, &mut landed));
            }
        }
    }

    /// Appends the audio generated since the last call: stereo samples at
    /// [`apu::SAMPLE_RATE`], the APU's mix through the board's high-pass filter.
    /// Undrained mixes beyond a few frames are dropped before the filter sees them.
    pub fn take_samples(&mut self, out: &mut Vec<Sample>) {
        let start = out.len();
        self.apu.device().take_samples(out);
        for s in &mut out[start..] {
            *s = self.speaker.apply(*s);
        }
    }

    /// Waits for an APU device on another thread to run everything emulated so far.
    pub fn settle_apu(&mut self) {
        self.apu.device().settle();
    }

    /// The reference APU, while no device has replaced it.
    pub fn apu_reference(&self) -> Option<&Apu> {
        match &self.apu {
            ApuSlot::Reference(a) => Some(a),
            ApuSlot::Replaced(_) => None,
        }
    }

    /// Hands the APU to `device` from this M-cycle on. The reference is dropped; a
    /// device that runs it in lockstep carries its own copy.
    pub fn replace_apu(&mut self, device: Box<dyn AudioDevice>) {
        self.apu = ApuSlot::Replaced(device);
    }

    /// What computes the APU, when a device replaced the reference.
    pub fn apu_describe(&self) -> Vec<String> {
        match &self.apu {
            ApuSlot::Reference(_) => Vec::new(),
            ApuSlot::Replaced(d) => d.describe(),
        }
    }

    pub fn apu_report(&mut self) -> Vec<String> {
        self.apu.device().report()
    }

    /// Battery RAM after the game wrote it and disabled RAM: persist these bytes.
    pub fn take_committed_save(&mut self) -> Option<Vec<u8>> {
        let save = self.cart.take_committed();
        self.with_footer(save)
    }

    /// Battery RAM written since the last take, finished or not. Persist it on shutdown.
    pub fn take_unsaved_save(&mut self) -> Option<Vec<u8>> {
        let save = self.cart.take_unsaved();
        self.with_footer(save)
    }

    /// A save of `ram`, the cartridge RAM a memory core holds, with the footer the board
    /// needs; None on a board without a battery.
    pub fn battery_save(&self, ram: Option<Vec<u8>>) -> Option<Vec<u8>> {
        self.cart.has_battery().then_some(())?;
        self.with_footer(ram)
    }

    /// Stops keeping the host's copy of WRAM, HRAM, VRAM, OAM and VBK, which nothing reads
    /// once memory and the PPU are in weights without a reference beside them
    /// (`docs/host-in-weights.md` §4 item 15). The cartridge's writes still reach the
    /// host's cartridge, whose clock-port decode the opcode fetch and a DMA source read.
    pub fn drop_shadow(&mut self) {
        self.shadow = false;
    }

    /// The host cartridge's writes not yet taken as a save: (written, finished).
    pub fn pending_save(&self) -> (bool, bool) {
        self.cart.pending_save()
    }

    /// The cartridge RAM's length.
    pub fn cart_ram_len(&self) -> usize {
        self.cart.ram_len()
    }

    /// The host's own clocks at a hand-off: the wall time last handed to the cartridge's
    /// clock, and the board filter's charge.
    pub fn host_clocks(&self) -> (Option<u64>, [i32; 2]) {
        (self.clock_time, self.speaker.charge)
    }

    /// Whether the cartridge keeps a battery save, and whether it has a clock.
    pub fn cart_facts(&self) -> (bool, bool) {
        (self.cart.has_battery(), self.cart.has_clock())
    }

    /// A save from a board with a clock ends with the clock's footer, from whichever
    /// device holds the clock, and the wall time it was taken at.
    fn with_footer(&self, save: Option<Vec<u8>>) -> Option<Vec<u8>> {
        let mut save = save?;
        if self.cart.has_clock() {
            save.extend(control!(&self.control, c => c.clock()).footer(self.clock_time));
        }
        Some(save)
    }

    /// Advances every device by `mcycles`, then hands the control logic a STOP the CPU
    /// ran, then advances by the M-cycles the control logic says the CPU stays stopped
    /// (`Lines::owed`), and returns them, HDMA's and the pause's apart. Devices advance in
    /// blocks no longer than the smallest [`Device::horizon`], in a fixed order, at the
    /// speed the control logic gives (`Lines::double`), and what they signal is applied
    /// after the block: the renderer's entry points, the copies.
    pub fn advance(&mut self, mcycles: u64) -> Stall {
        self.advance_handing(mcycles, &mut |_| {})
    }

    /// `advance`, handing each block's copies to `hand_over` before the next block runs, so
    /// a device that follows them (the PPU checkpoint) sees them when the host does.
    pub fn advance_handing(&mut self, mcycles: u64, hand_over: &mut dyn FnMut(&mut Vec<Copied>)) -> Stall {
        let in_step = self.in_step;
        let (hdma, pause) = crate::transport::advance(&mut Handing { m: self, hand_over }, mcycles, in_step);
        Stall { hdma, pause }
    }

    /// An FE page address on the bus while a DMG's PPU reads an OAM row corrupts it
    /// (`ControlDevice::oam_row`).
    fn corrupt_oam(&mut self, bug: OamBug) {
        let row = self.scan_row();
        if row != 0 {
            ppu::corrupt_oam(&mut self.ppu.oam, row as usize, bug);
        }
    }

    /// The OAM row the PPU's scan reads now, from the control logic (`ControlDevice::oam_row`),
    /// asked at every FE page address on the bus. Only the control logic says there is none
    /// (0): on a CGB, outside a visible line's mode 2, on the first line after the LCD
    /// turned on.
    fn scan_row(&mut self) -> u8 {
        let row = control_landing!(self, c, landed => c.oam_row(&mut landed));
        self.hits.2 += (row != 0) as u64;
        row
    }

    /// The GBC CPU model's step left it halted, before the HALT's M-cycle (`ControlDevice::halt`).
    fn halt(&mut self) {
        control_landing!(self, c, landed => c.halt(&mut landed));
    }

    /// The GBC CPU model left HALT, before the wake step's M-cycles (`ControlDevice::woke`).
    fn woke(&mut self) {
        self.signals.reset(self.lines.double);
        control_mut!(self, c => c.woke(&mut self.signals));
        self.apply_signals();
    }

    /// A GBC CPU step starts or ends, for the control logic (`ControlDevice::step_running`)
    /// and for when the blocks it held land.
    fn step_running(&mut self, running: bool) {
        self.in_step = running;
        control_landing!(self, c, landed => c.step_running(running, &mut landed));
    }

    /// The smallest horizon over the devices: how far the next block may run.
    fn horizon(&self) -> u64 {
        self.lines.horizon.min(self.apu.horizon())
    }

    /// Commits the block's signals: the renderer's entry points the control logic's
    /// timing called for, the requested copies. A line is drawn before the block's copies
    /// land, as of the M-cycle its mode 3 began.
    fn apply_signals(&mut self) {
        let mut signals = std::mem::take(&mut self.signals);
        signals.apply(&mut Applying(self));
        self.signals = signals;
    }

    /// Performs the block's early copies, those of its M-cycles before the last, keeping
    /// the rest of its signals for after them.
    fn land_early(&mut self) {
        let mut signals = std::mem::take(&mut self.signals);
        signals.land_early(&mut Applying(self));
        self.signals = signals;
    }

    /// Performs `copies` in order, keeping each byte moved for the core. Without the shadow
    /// machine the host reads no byte through its bus map: the model places and copies each
    /// one from its own memory.
    fn land(&mut self, copies: &[crate::device::Copy]) {
        for c in copies {
            for (source, dest) in c.bytes() {
                let byte = self.shadow.then(|| {
                    let byte = match (source, c) {
                        (Some(source), _) => self.read_synced(source),
                        (None, crate::device::Copy::Fill { value, .. }) => *value,
                        (None, _) => unreachable!("only a fill has no source"),
                    };
                    self.write_unblocked(dest, byte);
                    byte
                });
                self.copied.push(Copied { source, dest, byte });
            }
        }
    }

    /// A bus read that may observe the APU or the control device. The device catches
    /// up to this M-cycle first (`AudioDevice::sync`; a control read is a pass).
    #[inline(always)]
    fn read_synced(&mut self, addr: u16) -> u8 {
        match route(addr, self.cgb_hardware(), self.cgb()) {
            Dev::Control => self.read_control(addr),
            Dev::Cart if self.cart.clock_at(addr) => control_landing!(self, c, landed => c.read(addr, &mut landed)),
            Dev::Apu => self.read_apu(addr),
            Dev::Ppu if let Some(i) = layout::display_index(addr) => self.ppu.display_read(i),
            dev => self.read_dev(dev, addr),
        }
    }

    /// Out of line so the hot read path stays one route; a control read runs a pass. A
    /// palette data port reads the byte its index register points at, held by the PPU.
    #[inline(never)]
    fn read_control(&mut self, addr: u16) -> u8 {
        if matches!(addr, 0xFF69 | 0xFF6B) {
            self.hits.1 += 1;
            return match control_landing!(self, c, landed => c.palette(addr, &mut landed)) {
                PaletteRead::At(i) => self.ppu.display_read(i),
                PaletteRead::Byte(b) => b,
            };
        }
        control_landing!(self, c, landed => c.read(addr, &mut landed))
    }

    #[inline(never)]
    fn read_apu(&mut self, addr: u16) -> u8 {
        self.apu.device().sync(addr);
        Device::read(&self.apu, addr)
    }

    /// A read with no side effect and no device to catch up; control registers are
    /// read only through [`Machine::read_control`].
    fn read_unblocked(&self, addr: u16) -> u8 {
        self.read_dev(route(addr, self.cgb_hardware(), self.cgb()), addr)
    }

    #[inline(always)]
    fn read_dev(&self, dev: Dev, addr: u16) -> u8 {
        match dev {
            Dev::Cart => match self.boot_byte(addr) {
                Some(b) => b,
                None => self.cart.read(addr),
            },
            Dev::Ppu => self.ppu.read(addr),
            Dev::Wram => self.wram[self.wram_index(addr)],
            Dev::Unusable => 0x00,
            Dev::Control => panic!("control register {addr:#06x} read without a pass: control reads go through Machine::read_control"),
            Dev::Apu => Device::read(&self.apu, addr),
            Dev::Hram => self.hram[addr as usize - 0xFF80],
            Dev::None => 0xFF,
        }
    }

    fn write_unblocked(&mut self, addr: u16, value: u8) {
        let dev = route(addr, self.cgb_hardware(), self.cgb());
        self.signals.reset(self.lines.double);
        match dev {
            Dev::Cart => {
                self.cart.write(addr, value);
                if self.cart.clock_hears(addr) {
                    control_mut!(self, c => c.write(addr, value, &mut self.signals));
                }
            }
            Dev::Ppu => {
                self.ppu.write(addr, value);
                if addr == 0xFF40 {
                    control_mut!(self, c => c.write(addr, value, &mut self.signals));
                }
            }
            Dev::Wram => self.wram[self.wram_index(addr)] = value,
            Dev::None if addr == 0xFF50 => self.reference_ff50(value),
            Dev::Unusable | Dev::None => {}
            // KEY0 is written only while the boot ROM is mapped.
            Dev::Control if addr == 0xFF4C && self.boot.is_none() => {}
            Dev::Control => {
                if addr == 0xFF4C {
                    self.boot_regs.key0 = value;
                }
                control_mut!(self, c => c.write(addr, value, &mut self.signals))
            }
            Dev::Apu => Device::write(&mut self.apu, addr, value, &mut self.signals),
            Dev::Hram => self.hram[addr as usize - 0xFF80] = value,
        }
        self.apply_signals();
    }
}

impl Machine {
    /// Work RAM (the bank mapped at D000) or high RAM, without bus rules or side effects;
    /// other addresses read 0xFF.
    pub fn ram_byte(&self, addr: u16) -> u8 {
        match addr {
            0xC000..=0xDFFF => self.wram[self.wram_index(addr)],
            0xFF80..=0xFFFE => self.hram[addr as usize - 0xFF80],
            _ => 0xFF,
        }
    }
}

impl<C: Core> GameBoy<C> {
    /// Work RAM and high RAM as an observer reads them: from the core's own memory when it
    /// holds them, else the host's arrays.
    pub fn ram_byte(&self, addr: u16) -> u8 {
        self.core.peek_ram(addr).unwrap_or_else(|| self.machine.ram_byte(addr))
    }
}

/// What OAM DMA blocks now (`Lines::dma_buses`' high nibble).
fn dma_now(lines: &Lines) -> Dma {
    Dma(lines.dma_buses >> 4)
}

/// What OAM DMA blocks at an access whose bus state is `state`: the transfer's buses when
/// its DMA bit is set.
fn dma_at(lines: &Lines, state: u8) -> Dma {
    if state & step::DMA_BLOCKING != 0 { Dma(lines.dma_buses & 15) } else { Dma::NONE }
}

impl Machine {
    /// A CPU access OAM DMA blocks: the control device lands the transfer's bytes so far.
    #[inline(never)]
    fn dma_blocked(&mut self) {
        self.hits.0 += 1;
        self.signals.reset(self.lines.double);
        control_mut!(self, c => c.cut(&mut self.signals));
        self.apply_signals();
    }
}

impl Bus for Machine {
    fn read(&mut self, addr: u16) -> u8 {
        if dma_now(&self.lines).blocks(addr) {
            self.dma_blocked();
            return 0xFF;
        }
        if addr >> 8 == 0xFE {
            self.corrupt_oam(OamBug::Read);
        }
        if self.ppu_locked(addr, false) {
            return 0xFF;
        }
        self.read_synced(addr)
    }

    fn write(&mut self, addr: u16, value: u8) {
        if dma_now(&self.lines).blocks(addr) {
            self.dma_blocked();
            return;
        }
        if addr >> 8 == 0xFE {
            self.corrupt_oam(OamBug::Write);
        }
        if self.ppu_locked(addr, true) {
            return;
        }
        self.write_unblocked(addr, value);
    }

    /// The device a model-emitted address names. Its code is the dispatch; the bus map
    /// (`route`) is not consulted.
    fn read_port(&mut self, port: Port) -> u8 {
        match port {
            Port::Control(addr) => self.read_control(addr),
            Port::Apu(addr) => self.read_apu(addr),
            Port::Display(i) => self.ppu.display_read(i),
            Port::Clock(addr) => control_landing!(self, c, landed => c.read(addr, &mut landed)),
        }
    }

    /// LCDC reaches the display byte by its port and control by its second port.
    fn write_port(&mut self, port: Port, value: u8) {
        self.signals.reset(self.lines.double);
        match port {
            Port::Control(addr) | Port::Clock(addr) => control_mut!(self, c => c.write(addr, value, &mut self.signals)),
            Port::Apu(addr) => Device::write(&mut self.apu, addr, value, &mut self.signals),
            Port::Display(i) => self.ppu.display_write(i, value),
        }
        self.apply_signals();
    }

    /// The host's copy of a byte the model recorded (the shadow machine,
    /// `docs/host-in-weights.md` item 14): cartridge registers and RAM, VRAM and VBK,
    /// OAM, WRAM and HRAM. SVBK's record reaches control through its port.
    fn write_memory(&mut self, addr: u16, p: u32, value: u8) {
        // The boot registers as the model recorded them: KEY0, and FF50's byte on UNMAPPED
        // or on COMPAT, whichever the model's translation chose. The host only follows the
        // record, and beside a reference holds the choice to the reference's rule.
        if let Some(reg) = BootRegs::at(p) {
            let compat = reg == Reg::Compat;
            if matches!(reg, Reg::Unmapped | Reg::Compat) && self.shadow {
                let rule = self.ff50_compat();
                assert_eq!(compat, rule, "memory lockstep: FF50's write landed on {:?} where the reference leaves CGB mode {rule} (KEY0 {:#04x})", reg, self.boot_regs.key0);
            }
            self.boot_regs.set(reg, value);
            if matches!(reg, Reg::Unmapped | Reg::Compat) && value & 1 != 0 {
                self.unmap_boot(compat);
            }
            return;
        }
        match route(addr, self.cgb_hardware(), self.cgb()) {
            Dev::Cart => self.cart.write(addr, value),
            Dev::Ppu | Dev::Wram | Dev::Hram if !self.shadow => {}
            Dev::Ppu => self.ppu.write(addr, value),
            Dev::Wram => self.wram[self.wram_index(addr)] = value,
            Dev::Hram => self.hram[addr as usize - 0xFF80] = value,
            Dev::Control if addr == 0xFF70 => {}
            dev => panic!("a record of {addr:#06x}, which reaches {dev:?}, not memory"),
        }
    }

    #[inline]
    fn ie(&self) -> u8 {
        self.lines.ie
    }

    #[inline]
    fn iflag(&self) -> u8 {
        self.lines.iflag
    }

    fn dma_active(&self) -> bool {
        self.lines.dma_active
    }

    fn dma_buses(&self) -> u8 {
        self.lines.dma_buses
    }

    fn blocked(&mut self) {
        self.dma_blocked();
    }

    fn locks(&self) -> Option<u8> {
        self.lines.locks
    }

    fn tick(&mut self, mcycles: u8, hand_over: &mut dyn FnMut(&mut Vec<Copied>)) {
        let mut handed = false;
        self.advance_handing(mcycles as u64, &mut |bytes| {
            handed = true;
            hand_over(bytes);
        });
        self.tick_copies |= handed;
        self.ticked += mcycles as u64;
    }

    fn lock_query(&mut self) -> u8 {
        control_landing!(self, c, landed => c.locks(&mut landed))
    }

    fn wake(&self) -> bool {
        self.lines.wake
    }

    fn ahead(&mut self, mcycles: u8, locks: bool, write: bool) -> u8 {
        control_mut!(self, c => c.ahead(mcycles, locks, write))
    }

    fn physical(&self, addr: u16, access: Access, state: u8) -> Option<u32> {
        Some(self.fixed_physical(addr, access, state))
    }

    /// None for the enabled cartridge clock, whose bytes only a pass of the control logic
    /// gives. With RAM disabled the bus reads open bus there, as the memory model does.
    fn peek(&self, addr: u16, state: u8) -> Option<u8> {
        if dma_at(&self.lines, state).blocks(addr) {
            return Some(0xFF);
        }
        let locked = match addr {
            0x8000..=0x9FFF => 1,
            0xFE00..=0xFE9F => 2,
            _ => 0,
        };
        if state & locked != 0 {
            return Some(0xFF);
        }
        // SVBK reads back as control holds it, the one control register the model's
        // memory answers (from its readback table).
        if addr == 0xFF70 && self.cgb() {
            return Some(0xF8 | self.lines.svbk & 7);
        }
        // A control or APU register's byte comes only from a pass: no peek gives it.
        if matches!(route(addr, self.cgb_hardware(), self.cgb()), Dev::Control | Dev::Apu) {
            return None;
        }
        (!(self.cart.clock_at(addr) && self.cart.ram_enabled)).then(|| self.read_unblocked(addr))
    }

    fn banking(&self) -> Option<Banking> {
        Machine::banking(self).ok()
    }

    fn ack(&mut self, bits: u8) {
        if bits != 0 {
            control_landing!(self, c, landed => c.ack(bits, &mut landed));
        }
    }

    fn stop(&mut self) {
        self.stop_pending = true;
    }

    fn oam_bug(&mut self, read: bool) {
        self.corrupt_oam(if read { OamBug::Read } else { OamBug::Write });
    }

    fn oam_row(&mut self) -> u8 {
        self.scan_row()
    }

    /// The host's copy of OAM follows the model's corruption lines, held to the reference's
    /// rule (`ppu::corrupt_oam`) on the copy as it was.
    fn oam_records(&mut self, row: u8, read: bool, records: &[(u32, [u8; 8])]) {
        if !self.shadow {
            return;
        }
        let mut want = self.ppu.oam;
        ppu::corrupt_oam(&mut want, row as usize, if read { OamBug::Read } else { OamBug::Write });
        for &(p, line) in records {
            let at = p.checked_sub(layout::OAM).filter(|&o| o as usize + 8 <= layout::OAM_LEN).unwrap_or_else(|| panic!("an OAM bug record at {p:#08x}, outside OAM"));
            self.ppu.oam[at as usize..at as usize + 8].copy_from_slice(&line);
        }
        assert_eq!(self.ppu.oam, want, "memory lockstep: the OAM bug at row {row:#04x} ({}) wrote other lines than the reference's", if read { "read" } else { "write" });
    }
}


/// Where a block's signals land in the machine (`transport::Signals::apply`): the PPU's
/// entry points, the IF bits back to the control logic, the copies. While a GBC CPU step
/// runs, HDMA blocks wait in `held` for its end, as hardware runs them after an opcode
/// fetch.
struct Applying<'a>(&'a mut Machine);

impl crate::transport::Sink for Applying<'_> {
    fn land(&mut self, copies: &[crate::device::Copy]) {
        self.0.land(copies);
    }

    fn hold(&mut self, copies: &mut Vec<crate::device::Copy>) {
        self.0.held.append(copies);
    }

    fn render(&mut self, line: u8) {
        self.0.ppu.render_line(line);
    }

    fn stopped(&mut self, stopped: bool) {
        self.0.ppu.set_stopped(stopped);
    }

    fn frame(&mut self) {
        self.0.ppu.end_frame();
    }

    fn blank(&mut self) {
        self.0.ppu.lcd_off();
    }

    fn own(&mut self, index: u8, value: u8) {
        self.0.ppu.display_write(index, value);
    }

}

/// The machine as `transport::advance` runs it, each block's copies handed over before
/// the next block runs.
struct Handing<'a> {
    m: &'a mut Machine,
    hand_over: &'a mut dyn FnMut(&mut Vec<Copied>),
}

impl crate::transport::Blocks for Handing<'_> {
    fn horizon(&self) -> u64 {
        self.m.horizon()
    }

    fn block(&mut self, block: u64, stopped: bool) {
        let m = &mut *self.m;
        m.halves += m.lines.rate as u64 * block;
        m.ppu.set_clock(m.halves, m.lines.double);
        m.signals.reset(m.lines.double);
        m.signals.rate = m.lines.rate as u64;
        let Machine { control, lines, apu, signals, .. } = m;
        *lines = control!(control, c => {
            c.advance(block, stopped, signals);
            c.lines()
        });
        apu.advance(block, signals);
        // The bytes of the block's earlier M-cycles reach the core before a line the
        // block's last M-cycle draws.
        m.land_early();
        if !m.copied.is_empty() {
            (self.hand_over)(&mut m.copied);
        }
        m.apply_signals();
        if !m.copied.is_empty() {
            (self.hand_over)(&mut m.copied);
        }
    }

    fn stop(&mut self) {
        let m = &mut *self.m;
        if std::mem::take(&mut m.stop_pending) {
            m.signals.reset(m.lines.double);
            control_mut!(m, c => c.stop(&mut m.signals));
            m.apply_signals();
        }
    }

    fn owed(&self) -> u64 {
        self.m.lines.owed
    }

    fn paused(&self) -> bool {
        self.m.lines.paused
    }

    fn settle(&mut self) {
        let m = &mut *self.m;
        if !m.held.is_empty() {
            let held = std::mem::take(&mut m.held);
            m.land(&held);
            (self.hand_over)(&mut m.copied);
        }
    }
}
/// M-cycles the CPU stayed stopped after a step's own, each at the speed it ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stall {
    /// HDMA blocks: 8 M-cycles each at normal speed, 16 at double speed.
    pub hdma: u64,
    /// A speed switch's pause, in M-cycles of the new speed.
    pub pause: u64,
}

impl Stall {
    pub fn total(self) -> u64 {
        self.hdma + self.pause
    }
}

/// M-cycles spent with the core executing, halted with nothing pending, stopped while
/// HDMA copied, or stopped in a speed switch's pause. Each counts M-cycles at the speed
/// they ran, so at double speed one is half as long.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub steps: u64,
    pub step_mcycles: u64,
    pub halt_mcycles: u64,
    pub hdma_mcycles: u64,
    pub pause_mcycles: u64,
    /// CPU accesses OAM DMA blocked (the memory model's address in `layout::BLOCKED`, or
    /// the reference bus's `Dma::blocks`), and reads of the palette data ports
    /// BCPD and OCPD, counted so a run says whether it met them.
    pub dma_blocked: u64,
    pub palette_reads: u64,
    /// The lines a display write can change, which the stepper draws (`ppu::Midline`).
    pub midline: ppu::Midline,
    /// FE page addresses on the bus while a DMG's PPU read an OAM row: the OAM corruption
    /// bug fired.
    pub oam_bugs: u64,
}

/// A CPU core driving a [`Machine`].
pub struct GameBoy<C: Core> {
    pub(crate) core: C,
    pub(crate) cpu: Cpu,
    pub(crate) machine: Machine,
    pub(crate) counters: Counters,
    /// Where the current frame ends, in the machine's emulated time.
    pub(crate) frame_end: u64,
}

impl<C: Core> GameBoy<C> {
    /// Starts the CPU from the boot ROM hand-off state of the machine's hardware.
    pub fn new(core: C, machine: Machine) -> Self {
        let cpu = match machine.hardware() {
            _ if machine.booting() => Cpu::default(),
            Hardware::Dmg => Cpu::post_boot_dmg(),
            Hardware::Cgb if machine.compat() => Cpu::post_boot_compat(),
            Hardware::Cgb => Cpu::post_boot_cgb(),
        };
        GameBoy { core, cpu, machine, counters: Counters::default(), frame_end: 0 }
    }

    /// The same machine and CPU state continuing on another core, with fresh counters.
    pub fn with_core<D: Core>(self, core: D) -> GameBoy<D> {
        GameBoy { core, cpu: self.cpu, machine: self.machine, counters: Counters::default(), frame_end: self.frame_end }
    }

    /// Runs one CPU step, or one halted M-cycle, plus any HDMA stall or speed switch's
    /// pause it caused, and returns its length.
    pub fn step(&mut self) -> u64 {
        if self.machine.stopped() {
            // Ordinary STOP: time runs on without the GBC CPU until a joypad line falls.
            self.counters.halt_mcycles += 1;
            self.machine.advance(1);
            return 1;
        }
        let was_halted = self.cpu.halted;
        self.machine.step_running(true);
        let stepped = step::step(&mut self.core, &mut self.cpu, &mut self.machine);
        self.machine.step_running(false);
        if was_halted && !self.cpu.halted {
            self.machine.woke();
        }
        // SameBoy samples STAT as HALT runs, before its M-cycle.
        if !was_halted && self.cpu.halted {
            self.machine.halt();
        }
        let ran = match stepped {
            Some(m) => {
                self.counters.steps += 1;
                self.counters.step_mcycles += m as u64;
                m as u64
            }
            None => {
                self.counters.halt_mcycles += 1;
                1
            }
        };
        // Copies a bus write made (general-purpose HDMA) and each block's copies reach the
        // core before the next block runs.
        let (core, machine) = (&mut self.core, &mut self.machine);
        let mut copied = std::mem::take(&mut machine.tick_copies);
        let mut hand_over = |bytes: &mut Vec<Copied>| {
            core.copied(bytes);
            bytes.clear();
            copied = true;
        };
        if !machine.copied.is_empty() {
            hand_over(&mut machine.copied);
        }
        let rest = ran - std::mem::take(&mut machine.ticked);
        let stall = machine.advance_handing(rest, &mut hand_over);
        self.counters.hdma_mcycles += stall.hdma;
        self.counters.pause_mcycles += stall.pause;
        if copied {
            self.core.audit_memory(&self.machine.video_regions());
        }
        ran + stall.total()
    }

    /// The battery save due now, from the core when its memory holds cartridge RAM, else
    /// from the host's cartridge: a finished one (`committed`) or anything unsaved.
    pub fn take_save(&mut self, committed: bool) -> Option<Vec<u8>> {
        match self.core.take_save(committed, self.machine.cart_ram_len()) {
            Some(ram) => self.machine.battery_save(ram),
            None if committed => self.machine.take_committed_save(),
            None => self.machine.take_unsaved_save(),
        }
    }

    /// Runs one frame of emulated time, at either speed.
    pub fn run_frame(&mut self) {
        self.frame_end += self.machine.frame_halves();
        while self.machine.halves < self.frame_end {
            self.step();
        }
        self.core.audit_memory(&self.machine.ram_regions());
    }

    /// `run_frame`, handing `watch` the GBC CPU state and the opcode at PC (None while halted
    /// or when PC is in I/O space) before every step. Test ROMs signal their end with an
    /// opcode (mooneye's `LD B,B`). Work RAM and high RAM come from the core when it holds
    /// them (`Core::peek_ram`), since the host's copy stops following them without a
    /// shadow (`Machine::drop_shadow`).
    pub fn run_frame_watching(&mut self, mut watch: impl FnMut(&Cpu, Option<u8>)) {
        self.frame_end += self.machine.frame_halves();
        while self.machine.halves < self.frame_end {
            let pc = self.cpu.pc;
            let ram = match pc {
                0xC000..=0xDFFF | 0xFF80..=0xFFFE => Some(pc),
                0xE000..=0xFDFF => Some(pc - 0x2000),
                _ => None,
            };
            let op = (!self.cpu.halted && !matches!(route(pc, self.machine.cgb_hardware(), self.machine.cgb()), Dev::Control | Dev::Apu))
                .then(|| ram.and_then(|a| self.core.peek_ram(a)).unwrap_or_else(|| self.machine.read_unblocked(pc)));
            watch(&self.cpu, op);
            self.step();
        }
        self.core.audit_memory(&self.machine.ram_regions());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rom_with(cgb_flag: u8) -> Vec<u8> {
        let mut rom = vec![0; 0x8000];
        rom[0x1234] = 0x42;
        rom[0x143] = cgb_flag;
        for i in 0..0x40 {
            rom[0x4000 + i] = 0xA0 + i as u8;
        }
        rom
    }

    fn plain_machine() -> Machine {
        Machine::new(rom_with(0), None, Hardware::Dmg).unwrap()
    }

    fn cgb_machine() -> Machine {
        Machine::new(rom_with(0x80), None, Hardware::Cgb).unwrap()
    }

    /// A ROM of `banks` 16 KiB banks with header type and RAM size codes set.
    fn cart_rom(banks: usize, cartridge_type: u8, ram_code: u8, cgb_flag: u8) -> Vec<u8> {
        let mut rom = vec![0; banks * 0x4000];
        rom[0x143] = cgb_flag;
        rom[0x147] = cartridge_type;
        rom[0x149] = ram_code;
        rom
    }

    /// The device the reference machine's bus hands a CPU access of `a` to, as the port
    /// the model must name: `route`, LCDC to the PPU and control, the cartridge clock by
    /// the cartridge's select. None where the byte is memory or a constant: SVBK's and
    /// VBK's readback, unmapped I/O, the unusable area, and everything under OAM DMA but
    /// HRAM.
    fn reference_port(m: &Machine, a: u16, access: Access) -> Option<layout::Port> {
        use layout::Port;
        if dma_now(&m.lines).blocks(a) {
            return None;
        }
        let clock_write = access == Access::Write && m.cart.has_clock();
        match route(a, m.cgb_hardware(), m.cgb()) {
            Dev::Control if a == 0xFF70 && access == Access::Read => None,
            Dev::Control if a == 0xFF4C && access == Access::Write && m.boot.is_none() => None,
            Dev::Control => Some(Port::Control(a)),
            Dev::Apu => Some(Port::Apu(a)),
            Dev::Ppu => layout::display_index(a).map(Port::Display),
            Dev::Cart if m.cart.clock_at(a) && m.cart.ram_enabled => Some(Port::Clock(0xA000 | a & 0x7F)),
            Dev::Cart if clock_write && a < 0x2000 => Some(Port::Clock(0x0000)),
            Dev::Cart if clock_write && (0x4000..0x6000).contains(&a) => Some(Port::Clock(0x4000)),
            Dev::Cart if clock_write && (0x6000..0x8000).contains(&a) => Some(Port::Clock(0x6000 | a & 0x7F)),
            _ => None,
        }
    }

    /// The fixed-layout address of a flat `Machine::physical` result.
    fn fixed_of_flat(m: &Machine, flat: u32) -> u32 {
        let l = m.memory_layout();
        let rom_len = m.cart.rom().len() as u32;
        match flat {
            f if f < rom_len => layout::ROM + f,
            f if f < l.ram => layout::CRAM + (f - rom_len),
            f if f < l.vram => layout::WRAM + (f - l.wram),
            f if f < l.oam => layout::VRAM + (f - l.vram),
            f if f < l.hram => layout::OAM + (f - l.oam),
            f => layout::HRAM + (f - l.hram),
        }
    }

    /// Every bus address, read and written, translates the same way through the spec
    /// (from the host's registers and from the raw bytes the game wrote) as through
    /// the host itself, on every mapper, before and after random register traffic.
    #[test]
    fn fixed_layout_translation_matches_the_host_on_every_mapper() {
        let configs = [(2usize, 0x00u8, 0x00u8, 0u8, Hardware::Dmg), (4, 0x09, 0x02, 0, Hardware::Dmg), (64, 0x03, 0x03, 0, Hardware::Dmg), (16, 0x06, 0x00, 0, Hardware::Dmg), (128, 0x10, 0x03, 0x80, Hardware::Cgb), (32, 0x13, 0x05, 0, Hardware::Dmg), (64, 0x1B, 0x04, 0x80, Hardware::Cgb), (128, 0x19, 0x00, 0xC0, Hardware::Cgb), (512, 0x19, 0x00, 0, Hardware::Dmg), (4, 0x0F, 0x00, 0, Hardware::Dmg), (16, 0x10, 0x03, 0, Hardware::Dmg)];
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut draw = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for (banks, kind, ram_code, cgb_flag, hw) in configs {
            let mut m = Machine::new(cart_rom(banks, kind, ram_code, cgb_flag), None, hw).unwrap();
            // The LCD off: no access lock moves a VRAM or OAM address.
            m.write(0xFF40, 0x00);
            let mut raw = m.banking().unwrap();
            let check = |m: &Machine, raw: &Banking, round: usize| {
                let host = m.banking().unwrap();
                let dma = dma_now(&m.lines);
                for a in 0..=0xFFFFu16 {
                    let want = m.fixed_physical(a, Access::Read, m.access_state());
                    assert_eq!(layout::translate(raw, a, Access::Read, dma), want, "raw-written registers, {kind:#04x} round {round} addr {a:#06x}");
                    // Every read lands where the host keeps the byte, in the port of the
                    // device the reference's map routes it to, or on a constant the
                    // reference reads.
                    assert_eq!(layout::port(want), reference_port(m, a, Access::Read), "{kind:#04x} round {round} read {a:#06x} -> {want:#x} (dma {dma:?})");
                    match m.physical(a) {
                        Some(flat) if !dma.blocks(a) => assert_eq!(want, fixed_of_flat(m, flat), "{kind:#04x} round {round} addr {a:#06x}"),
                        _ if layout::port(want).is_some() => {}
                        _ => assert_eq!(layout::constant(want), Bus::peek(m, a, m.access_state()), "{kind:#04x} round {round} read {a:#06x} -> {want:#x} (dma {dma:?})"),
                    }
                    assert_eq!(want & 0x7F, a as u32 & 0x7F, "low bits pass through");
                    let w = layout::translate(raw, a, Access::Write, dma);
                    let w_host = layout::translate(&host, a, Access::Write, dma);
                    assert_eq!(w, w_host, "write translation, {kind:#04x} addr {a:#06x}");
                    // A write reaches the device its address names and the one its second
                    // port names; the reference's LCDC write reaches the PPU and control.
                    let heard: Vec<layout::Port> = [layout::port(w), layout::second_port(w, m.cart.has_clock()).and_then(layout::port)].into_iter().flatten().collect();
                    let want: Vec<layout::Port> = match reference_port(m, a, Access::Write) {
                        Some(p @ layout::Port::Display(_)) if a == 0xFF40 => vec![p, layout::Port::Control(0xFF40)],
                        p => p.into_iter().collect(),
                    };
                    assert_eq!(heard, want, "{kind:#04x} round {round} write {a:#06x} -> {w:#x} (dma {dma:?})");
                    if a < 0x8000 && !dma.blocks(a) {
                        assert!(layout::is_void(w) || layout::port(w).is_some() || (layout::REGS..layout::REGS + layout::N_REGS as u32).contains(&w), "{a:#06x} -> {w:#x}");
                    }
                }
            };
            check(&m, &raw, 0);
            for round in 1..=6 {
                for _ in 0..12 {
                    let r = draw();
                    let a = match r % 6 {
                        0 => (r >> 8) as u16 & 0x7FFF,
                        1 => 0xFF70,
                        2 => 0xFF4F,
                        3 => 0xA000 | (r >> 8) as u16 & 0x1FFF,
                        4 => 0x4000,
                        _ => 0x0000,
                    };
                    // RAM enable and the clock's selects often enough to reach them.
                    let v = match r % 6 {
                        4 => 0x08 + (r >> 32) as u8 % 6,
                        5 => 0x0A,
                        _ => (r >> 32) as u8,
                    };
                    let p = layout::translate(&raw, a, Access::Write, false);
                    m.write(a, v);
                    if (layout::REGS..layout::REGS + layout::N_REGS as u32).contains(&p) {
                        raw.0[(p - layout::REGS) as usize] = v;
                    }
                }
                check(&m, &raw, round);
            }
            // An 8 MiB MBC5 cartridge's bank bit 8 (ROMB1 bit 0): bank 0x123, 4.5 MiB in.
            if banks == 512 {
                for (a, v) in [(0x2000u16, 0x23u8), (0x3000, 0xFF)] {
                    let p = layout::translate(&raw, a, Access::Write, false);
                    m.write(a, v);
                    raw.0[(p - layout::REGS) as usize] = v;
                }
                assert_eq!(layout::translate(&raw, 0x4000, Access::Read, false), layout::ROM + 0x123 * 0x4000, "bank 0x123");
                check(&m, &raw, 8);
            }
            m.write(0xFF46, 0xC0);
            m.advance(2);
            assert!(m.dma_active());
            check(&m, &raw, 7);
        }
    }

    #[test]
    fn oam_dma_blocks_its_source_bus_and_oam_two_m_cycles_after_the_write() {
        let mut m = plain_machine();
        m.write(0xFF40, 0x00);
        for i in 0..0xA0u16 {
            m.write(0xC000 + i, i as u8 ^ 0x5A);
        }
        m.write(0xFF80, 0x99);
        m.write(0x8000, 0x33);
        m.write(0xFF46, 0xC0);
        m.advance(1);
        assert_eq!(m.read(0xFE00), 0x00, "the M-cycle after the write still reaches OAM");
        m.advance(1);
        assert_eq!(m.read(0xFE00), 0xFF, "OAM blocked from the next");
        assert_eq!(m.read(0xC000), 0xFF, "WRAM hidden during DMA");
        assert_eq!(m.read(0x8000), 0x33, "a DMG's VRAM bus stays free under a WRAM source");
        assert_eq!(m.read(0xFF46), 0xC0, "FF46 reads back during DMA");
        assert_eq!(m.read(0x1234), 0xFF, "ROM hidden during DMA");
        assert_eq!(m.read(0xFF80), 0x99, "HRAM visible");
        m.write(0xC000, 0x00);
        m.write(0xFF81, 0x77);
        assert_eq!(m.read(0xFF81), 0x77);
        m.advance(80);
        assert_eq!((m.ppu.oam[79], m.ppu.oam[80]), (79 ^ 0x5A, 0x00), "one byte per M-cycle");
        assert_eq!(m.read(0xFE00), 0xFF);
        m.advance(80);
        assert_eq!(m.read(0xC000), 0x5A, "write during DMA was ignored");
        assert_eq!(m.read(0x1234), 0x42);
        assert_eq!((m.read(0xFE00), m.read(0xFE9F)), (0x5A, 0x9F ^ 0x5A));
    }

    /// FF46 written while a transfer blocks: OAM stays blocked through the new one's two
    /// M-cycles before byte 0, and the new one ends 162 M-cycles after the write.
    #[test]
    fn a_restarted_oam_dma_blocks_through_its_start() {
        let mut m = plain_machine();
        m.write(0xFF40, 0x00);
        m.write(0x8000, 0x01);
        m.write(0xFF46, 0x80);
        m.advance(10);
        m.write(0xFF46, 0x80);
        for _ in 0..161 {
            m.advance(1);
            assert_eq!(m.read(0xFE00), 0xFF, "blocked until the restarted transfer ends");
        }
        m.advance(1);
        assert_eq!(m.read(0xFE00), 0x01, "OAM byte 0 from VRAM once it ends");
    }

    #[test]
    fn io_read_masks_and_echo() {
        let mut m = plain_machine();
        for addr in [0xFF03, 0xFF08, 0xFF15, 0xFF27, 0xFF4C, 0xFF4D, 0xFF4F, 0xFF55, 0xFF68, 0xFF70, 0xFF7F] {
            assert_eq!(m.read(addr), 0xFF, "{addr:#06x} unmapped on DMG");
        }
        assert_eq!(m.read(0xFF0F), 0xE1);
        m.write(0xFF0F, 0x00);
        assert_eq!(m.read(0xFF0F), 0xE0);
        assert_eq!((m.read(0xFF02), m.read(0xFF07), m.read(0xFF41) & 0xF8), (0x7E, 0xF8, 0x80));
        m.write(0xD123, 0xAB);
        assert_eq!(m.read(0xF123), 0xAB);
        m.write(0xE456, 0xCD);
        assert_eq!(m.read(0xC456), 0xCD);
        assert_eq!(m.read(0xFF46), 0xFF);
        m.write(0xFF70, 0x03);
        m.write(0xD000, 0x31);
        assert_eq!((m.ram_byte(0xD000), m.wram[0x1000]), (0x31, 0x31), "DMG ignores SVBK");
    }

    #[test]
    fn timer_interrupt_reaches_if() {
        let mut m = plain_machine();
        m.write(0xFF0F, 0);
        m.write(0xFF04, 0);
        m.write(0xFF05, 0xFF);
        m.write(0xFF07, 0x05);
        m.advance(4);
        assert_eq!(m.iflag() & 0x04, 0);
        m.advance(1);
        assert_eq!(m.iflag() & 0x04, 0x04);
    }

    #[test]
    fn ff72_to_ff75_hold_their_bytes_on_cgb_hardware() {
        let ports = [0xFF72, 0xFF73, 0xFF74, 0xFF75];
        let written = |mut m: Machine| {
            for a in ports {
                m.write(a, 0xA5);
            }
            ports.map(|a| m.read(a))
        };
        assert_eq!(written(cgb_machine()), [0xA5, 0xA5, 0xA5, 0xAF], "CGB mode: FF72-FF74 read back, FF75 keeps bits 4-6");
        let compat = Machine::new(rom_with(0x00), None, Hardware::Cgb).unwrap();
        assert_eq!(written(compat), [0xA5, 0xA5, 0xFF, 0xAF], "compatibility mode: FF74 is CGB mode's alone");
        assert_eq!(written(plain_machine()), [0xFF; 4], "a DMG has none of them");
    }

    #[test]
    fn hardware_follows_the_header_and_cgb_runs_dmg_only_games_in_compatibility_mode() {
        let dmg = crate::rom::identify(&rom_with(0x00)).unwrap();
        let cgb = crate::rom::identify(&rom_with(0x80)).unwrap();
        assert_eq!(Hardware::resolve(&dmg, None), Ok(Hardware::Dmg));
        assert_eq!(Hardware::resolve(&cgb, None), Ok(Hardware::Cgb));
        assert_eq!(Hardware::resolve(&cgb, Some(Hardware::Dmg)), Ok(Hardware::Dmg));
        assert_eq!(Hardware::resolve(&dmg, Some(Hardware::Cgb)), Ok(Hardware::Cgb));
        let mut m = Machine::new(rom_with(0x00), None, Hardware::Cgb).unwrap();
        assert!(m.compat());
        let regs = [0xFF4D, 0xFF55, 0xFF69, 0xFF6B, 0xFF70];
        assert_eq!(regs.map(|a| m.read(a)), [0xFF; 5], "compatibility mode has no CGB-mode registers");
        assert_eq!(m.read(0xFF4F), 0xFE, "CGB hardware reads VBK in compatibility mode");
        assert_eq!([0xFF68, 0xFF6A, 0xFF76, 0xFF77].map(|a| m.read(a)), [0xC8, 0xD0, 0x00, 0x00], "BCPS, OCPS, PCM12 and PCM34 answer on CGB hardware, as the boot ROM leaves them");
        m.write(0xFF68, 0x05);
        assert_eq!(m.read(0xFF68), 0x45, "BCPS keeps its index in compatibility mode");
    }

    #[test]
    fn oam_dma_from_e000_reads_the_echo_on_dmg_and_writes_ff_on_cgb() {
        for (mut m, want) in [(plain_machine(), 0x5A), (cgb_machine(), 0xFF)] {
            m.write(0xFF40, 0x00);
            m.write(0xC010, 0x5A);
            m.write(0xFF46, 0xE0);
            m.advance(162);
            assert_eq!(m.read(0xFE10), want, "OAM byte 0x10 after a transfer from E000 (cgb {})", m.cgb());
        }
    }

    #[test]
    fn the_locks_keep_vram_in_mode_3_and_oam_in_modes_2_and_3_from_the_cpu() {
        let mut m = plain_machine();
        m.write(0x8000, 0x11);
        assert_eq!(m.read(0xFE00), 0xFF, "mode 2 holds OAM");
        m.advance(20);
        m.write(0x8000, 0x22);
        assert_eq!((m.read(0x8000), m.read(0xFE00)), (0xFF, 0xFF), "mode 3 holds both");
        m.advance(43);
        assert_eq!((m.read(0x8000), m.read(0xFE00)), (0x11, 0x00), "HBlank frees them; the mode-3 write dropped");
    }

    #[test]
    fn cgb_post_boot_registers() {
        let mut m = cgb_machine();
        let regs = [0xFF02, 0xFF46, 0xFF4C, 0xFF4D, 0xFF4F, 0xFF51, 0xFF54, 0xFF55, 0xFF70];
        assert_eq!(regs.map(|a| m.read(a)), [0x7E, 0x00, 0x80, 0x7E, 0xFE, 0xFF, 0xFF, 0xFF, 0xF8]);
        let boot = [0xFF00, 0xFF04, 0xFF48, 0xFF49];
        assert_eq!(boot.map(|a| m.read(a)), [0xFF, 0x2F, 0x00, 0x00], "P1, DIV, OBP0, OBP1 after the CGB boot ROM");
        assert_eq!(boot.map(|a| plain_machine().read(a)), [0xCF, 0xAB, 0xFF, 0xFF], "and after the DMG one");
        m.write(0xFF4D, 0x01);
        assert_eq!(m.read(0xFF4D), 0x7F, "a speed switch armed");
        let gb = GameBoy::new(crate::reference::Reference, cgb_machine());
        assert_eq!((gb.cpu.a, gb.cpu.f, gb.cpu.d, gb.cpu.e, gb.cpu.l), (0x11, 0x80, 0xFF, 0x56, 0x0D));
    }

    #[test]
    fn wram_banks_1_to_7() {
        let mut m = cgb_machine();
        for bank in 0..8u8 {
            m.write(0xFF70, bank);
            m.write(0xD010, 0x10 + bank);
        }
        m.write(0xC010, 0xCC);
        for bank in 1..8u8 {
            m.write(0xFF70, bank);
            assert_eq!(m.read(0xD010), 0x10 + bank);
            assert_eq!(m.read(0xF010), 0x10 + bank, "echo follows the bank");
        }
        m.write(0xFF70, 0x00);
        assert_eq!((m.read(0xFF70), m.read(0xD010)), (0xF8, 0x11), "bank 0 reads back as 0 and maps bank 1");
        m.write(0xFF70, 0x0D);
        assert_eq!((m.read(0xFF70), m.ram_byte(0xD010)), (0xFD, 0x15), "3 bits kept; peek sees the mapped bank");
        assert_eq!(m.read(0xC010), 0xCC, "C000 is always bank 0");
    }

    #[test]
    fn vram_bank_through_the_bus() {
        let mut m = cgb_machine();
        m.write(0x8000, 0x01);
        m.write(0xFF4F, 0x01);
        m.write(0x8000, 0x02);
        assert_eq!(m.read(0x8000), 0x02);
        m.write(0xFF4F, 0x00);
        assert_eq!(m.read(0x8000), 0x01);
    }

    fn set_hdma(m: &mut Machine, src: u16, dst: u16) {
        m.write(0xFF51, (src >> 8) as u8);
        m.write(0xFF52, src as u8);
        m.write(0xFF53, (dst >> 8) as u8);
        m.write(0xFF54, dst as u8);
    }

    #[test]
    fn hdma_general_purpose_copies_at_once_and_stalls() {
        let mut m = cgb_machine();
        m.write(0xFF4F, 0x01);
        set_hdma(&mut m, 0x400F, 0x8105);
        m.write(0xFF55, 0x01);
        assert_eq!(m.advance(0), Stall { hdma: 17, pause: 0 }, "two blocks, 8 M-cycles each, and 1 for the run");
        assert_eq!(m.read(0xFF55), 0xFF, "finished");
        assert_eq!((m.read(0x8100), m.read(0x811F), m.read(0x8120)), (0xA0, 0xBF, 0x00), "low nibbles ignored");
        m.write(0xFF4F, 0x00);
        assert_eq!(m.read(0x8100), 0x00, "copied into the bank VBK selected");
        m.write(0xFF4F, 0x01);
        m.write(0xFF55, 0x00);
        assert_eq!(m.read(0x8120), 0xC0, "source and destination continue where they stopped");
    }

    #[test]
    fn hdma_hblank_copies_one_block_per_line() {
        let mut m = cgb_machine();
        set_hdma(&mut m, 0x4000, 0x9000);
        m.write(0xFF55, 0x82);
        assert_eq!(m.read(0xFF55), 0x02, "active, 3 blocks: reads blocks left minus 1");
        assert_eq!(m.read(0x9000), 0x00, "nothing until HBlank");
        assert_eq!(m.advance(63), Stall { hdma: 9, pause: 0 }, "one block, 8 M-cycles of stall and 1 for the run");
        assert_eq!((m.read(0x9000), m.read(0xFF55)), (0xA0, 0x01), "first block at line 0's HBlank");
        assert_eq!((m.read(0x900F), m.read(0x9010)), (0xAF, 0x00));
        m.advance(114);
        assert_eq!((m.read(0x9010), m.read(0xFF55)), (0xB0, 0x00));
        m.write(0xFF55, 0x00);
        assert_eq!(m.read(0xFF55), 0x80, "a cancel reads 0x80 | the count it wrote, 0");
        m.advance(114);
        assert_eq!(m.read(0x9020), 0x00, "no copy after cancel");
        m.write(0xFF55, 0x80);
        m.advance(114);
        assert_eq!((m.read(0x9020), m.read(0xFF55)), (0xC0, 0xFF), "restart resumes; last block finishes");
    }


    /// An HBlank transfer started while STAT already reads mode 0 moves its first block at
    /// once, and that line's HBlank request, which comes up to an M-cycle later at some fine
    /// scrolls, moves no second one: a line moves one block.
    #[test]
    fn a_line_moves_one_hblank_block_when_the_transfer_starts_in_mode_0() {
        for fine in 0..8u8 {
            let mut m = cgb_machine();
            m.write(0xFF43, fine);
            set_hdma(&mut m, 0x4000, 0x9000);
            while m.read(0xFF44) != 2 || m.read(0xFF41) & 3 != 3 {
                m.advance(1);
            }
            let mut n = 0;
            while m.read(0xFF41) & 3 != 0 {
                m.advance(1);
                n += 1;
            }
            m.write(0xFF55, 0x81);
            let first = m.read(0xFF55);
            m.advance(8);
            let (ly, after) = (m.read(0xFF44), m.read(0xFF55));
            eprintln!("fine {fine}: mode 0 {n} M-cycles after mode 3, at once {first:#04x}, 8 later (LY {ly}) {after:#04x}");
            assert_eq!((first, after, ly), (0x00, 0x00, 2), "fine {fine}: one block on line 2");
        }
    }

    /// A HALT entered after STAT reads mode 0 but before the line's HBlank request keeps the
    /// line's block owed, and the wake moves it: a line moves one block, not none.
    #[test]
    fn a_halt_between_mode_0_and_the_hblank_request_keeps_the_line_s_block() {
        for fine in 0..8u8 {
            let mut m = cgb_machine();
            m.write(0xFF43, fine);
            set_hdma(&mut m, 0x4000, 0x9000);
            while m.read(0xFF44) != 2 || m.read(0xFF41) & 3 != 3 {
                m.advance(1);
            }
            m.write(0xFF55, 0x81);
            while m.read(0xFF41) & 3 != 0 {
                m.advance(1);
            }
            m.halt();
            m.advance(4);
            m.woke();
            let (ly, left) = (m.read(0xFF44), m.read(0xFF55));
            eprintln!("fine {fine}: after the wake (LY {ly}) {left:#04x}");
            assert_eq!((left, ly), (0x00, 2), "fine {fine}: line 2's block moved once");
        }
    }
    #[test]
    fn hdma_stall_advances_time_in_gameboy_step() {
        let mut rom = rom_with(0x80);
        // LD A,$01 ; LDH ($55),A ; JR -2 at 0x0100.
        rom[0x100..0x106].copy_from_slice(&[0x3E, 0x01, 0xE0, 0x55, 0x18, 0xFE]);
        let m = Machine::new(rom, None, Hardware::Cgb).unwrap();
        let mut gb = GameBoy::new(crate::reference::Reference, m);
        let lens = [gb.step(), gb.step()];
        assert_eq!(lens, [2, 3 + 17], "the LDH that starts a 2-block GDMA takes 17 more M-cycles");
        assert_eq!(gb.counters.hdma_mcycles, 17);
    }

    /// The switch against the documented timings: STOP with KEY1 armed toggles bit 7,
    /// clears bit 0, resets DIV and pauses the CPU 32,772 M-cycles of the new speed
    /// with DIV held (`control::SWITCH_MCYCLES`); at double speed DIV counts twice as
    /// many per dot, a line takes 228 M-cycles, an HDMA block stops the CPU 16 and the
    /// frame sequencer follows DIV bit 5.
    #[test]
    fn a_speed_switch_follows_the_documented_timings() {
        let pause = control::SWITCH_MCYCLES as u64;
        let mut rom = rom_with(0x80);
        // LD A,$01 ; LDH ($4D),A ; STOP ; JR -2 at 0x0100.
        rom[0x100..0x107].copy_from_slice(&[0x3E, 0x01, 0xE0, 0x4D, 0x10, 0x18, 0xFE]);
        let mut gb = GameBoy::new(crate::reference::Reference, Machine::new(rom, None, Hardware::Cgb).unwrap());
        assert_eq!([gb.step(), gb.step()], [2, 3]);
        assert_eq!(gb.machine.read(0xFF4D), 0x7F, "armed at normal speed");
        assert_eq!(gb.step(), 1 + pause, "STOP, then the pause");
        assert_eq!((gb.counters.hdma_mcycles, gb.counters.pause_mcycles), (0, pause), "the pause counts apart from HDMA");
        let m = &mut gb.machine;
        assert_eq!(m.read(0xFF4D), 0xFE, "double speed, the switch disarmed");
        assert_eq!(m.halves(), 2 * (2 + 3 + 1) + pause, "the pause at 2 dots an M-cycle");
        assert_eq!(m.read(0xFF04), 0, "DIV reset and held through the pause");
        assert_eq!(m.read(0xFF44), 143, "the PPU ran 6 + 16,386 ticks: line 143, tick 90");
        m.advance(64);
        assert_eq!((m.read(0xFF04), m.read(0xFF44)), (1, 144), "DIV counts 64 double-speed M-cycles, the PPU 32 ticks");
        m.advance(211);
        assert_eq!(m.read(0xFF44), 145, "tick 113 of line 144, one M-cycle into it, reads the next line already");
        assert_eq!(m.read(0xFF41) & 3, 1, "still line 144: VBlank");
        m.advance(1);
        assert_eq!(m.read(0xFF44), 145, "two M-cycles a tick, 228 a line");
        set_hdma(m, 0x4000, 0x8000);
        m.write(0xFF55, 0x00);
        assert_eq!(m.advance(0), Stall { hdma: 17, pause: 0 }, "one block stops the CPU for 16 M-cycles at double speed, and the run 1 more");
        // The frame sequencer's steps over 8,192 M-cycles, which the control logic
        // signals from DIV bit 5 at double speed and bit 4 at normal speed.
        let sequencer_steps = |m: &mut Machine| {
            let before = m.apu_reference().expect("the reference APU").div_divider;
            m.advance(8192);
            m.apu_reference().unwrap().div_divider.wrapping_sub(before) & 7
        };
        let double = sequencer_steps(m);
        m.write(0xFF4D, 0x01);
        m.stop();
        assert_eq!((m.advance(0), m.read(0xFF4D)), (Stall { hdma: 0, pause }, 0x7E), "back to normal speed, the same pause in M-cycles");
        assert_eq!((sequencer_steps(m), double), (4, 2), "8192 M-cycles: 4 steps at 512 Hz, 2 at double speed");
    }

    /// A machine whose bus a test drives directly, its control logic the reference or,
    /// with `compiled`, the control circuit in lockstep with it.
    struct Board(Machine);

    fn board(compiled: bool, rom: Vec<u8>, save: Option<Vec<u8>>) -> Result<Board, String> {
        let mut m = Machine::new(rom, save, Hardware::Dmg)?;
        if compiled {
            let reference = m.control_reference().unwrap().clone();
            let model = crate::control_circuit::ControlModel::new(crate::control_circuit::CircuitSpans(crate::control_circuit::build()), &reference.state(), reference.lengths());
            m.replace_control(ControlReplacement::Device(Box::new(crate::control_circuit::ControlLockstep::new(reference, model))));
        }
        Ok(Board(m))
    }

    #[test]
    fn the_bus_says_open_bus_at_a_disabled_clock_and_nothing_at_an_enabled_one() {
        let mut m = Machine::new(cart_rom(4, 0x10, 0x03, 0), None, Hardware::Dmg).unwrap();
        m.write_unblocked(0x4000, 0x08);
        assert_eq!(Bus::peek(&m, 0xA000, m.access_state()), Some(0xFF), "RAM disabled: the memory model serves 0xFF and the audit peeks");
        m.write_unblocked(0x0000, 0x0A);
        assert_eq!(Bus::peek(&m, 0xA000, m.access_state()), None, "only a pass of the control logic gives the clock's byte");
        m.write_unblocked(0x4000, 0x00);
        assert_eq!(Bus::peek(&m, 0xA000, m.access_state()), Some(0x00), "a RAM bank");
    }

    /// Runs `test` with the reference clock, then with the compiled one.
    fn both(test: impl Fn(bool)) {
        test(false);
        test(true);
    }

    impl Board {
        fn write(&mut self, addr: u16, value: u8) {
            self.0.write_unblocked(addr, value);
        }
        fn read(&mut self, addr: u16) -> u8 {
            self.0.read_synced(addr)
        }
        fn set_time(&mut self, now: u64) {
            self.0.set_time(now);
        }
        fn take_committed(&mut self) -> Option<Vec<u8>> {
            self.0.take_committed_save()
        }
        fn take_unsaved(&mut self) -> Option<Vec<u8>> {
            self.0.take_unsaved_save()
        }
    }

    #[test]
    fn mbc3_rtc_counts_host_seconds_and_latches() {
        both(|compiled| {
            let mut cart = board(compiled, cart_rom(4, 0x10, 0x03, 0), None).unwrap();
            cart.set_time(1_000);
            cart.write(0x0000, 0x0A);
            cart.set_time(1_000 + 86400 * 2 + 3600 * 5 + 60 * 7 + 9);
            cart.write(0x4000, 0x08);
            assert_eq!(cart.read(0xA000), 0, "nothing latched yet");
            cart.write(0x6000, 0x00);
            cart.write(0x6000, 0x01);
            let regs: Vec<u8> = (0x08..=0x0C).map(|r| {
                cart.write(0x4000, r);
                cart.read(0xA000)
            }).collect();
            assert_eq!(regs, [9, 7, 5, 2, 0]);
            cart.set_time(1_000 + 86400 * 3);
            cart.write(0x4000, 0x08);
            assert_eq!(cart.read(0xA000), 9, "latched value holds until the next latch");
            cart.write(0x4000, 0x0C);
            cart.write(0xA000, 0x40);
            cart.write(0x4000, 0x08);
            cart.write(0xA000, 30);
            cart.set_time(1_000 + 86400 * 4);
            cart.write(0x6000, 0x00);
            cart.write(0x6000, 0x01);
            assert_eq!(cart.read(0xA000), 30, "halted clock does not advance");
            cart.write(0x4000, 0x0B);
            assert_eq!(cart.read(0xA000), 3);
            cart.write(0x4000, 0x0C);
            assert_eq!(cart.read(0xA000), 0x40);
        });
    }


    /// Reads the latched clock registers 08-0C after latching.
    fn latched_clock(cart: &mut Board) -> Vec<u8> {
        cart.write(0x6000, 0x00);
        cart.write(0x6000, 0x01);
        (0x08..=0x0C)
            .map(|r| {
                cart.write(0x4000, r);
                cart.read(0xA000)
            })
            .collect()
    }

    #[test]
    fn mbc3_save_ends_with_the_bgb_clock_footer() {
        both(|compiled| {
            let mut cart = board(compiled, cart_rom(4, 0x10, 0x03, 0), None).unwrap();
            cart.set_time(1_000);
            cart.write(0x0000, 0x0A);
            cart.set_time(1_000 + 86400 * 2 + 3600 * 5 + 60 * 7 + 9);
            assert_eq!(latched_clock(&mut cart), [9, 7, 5, 2, 0]);
            cart.write(0x4000, 0x00);
            cart.write(0xA000, 0x5A);
            cart.write(0x0000, 0x00);
            let save = cart.take_committed().unwrap();
            assert_eq!((save.len(), save[0]), (0x8000 + 48, 0x5A));
            let time = (1_000u64 + 86400 * 2 + 3600 * 5 + 60 * 7 + 9).to_le_bytes();
            #[rustfmt::skip]
            let footer = [
                9, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                9, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                time[0], time[1], time[2], time[3], time[4], time[5], time[6], time[7],
            ];
            assert_eq!(save[0x8000..], footer);
        });
    }

    #[test]
    fn mbc3_clock_resumes_from_a_footer_and_counts_the_time_away() {
        both(|compiled| {
            let mut save = vec![0x33; 0x8000];
            // 1d 23:59:50, halt and carry clear; latched 0s; saved at Unix time 50_000.
            for v in [50u32, 59, 23, 1, 0, 0, 0, 0, 0, 0] {
                save.extend(v.to_le_bytes());
            }
            save.extend(50_000u64.to_le_bytes());
            let mut cart = board(compiled, cart_rom(4, 0x10, 0x03, 0), Some(save.clone())).unwrap();
            cart.write(0x0000, 0x0A);
            assert_eq!(cart.read(0xA000), 0x33, "RAM loads ahead of the footer");
            cart.set_time(50_000 + 3600 + 15);
            assert_eq!(latched_clock(&mut cart), [5, 0, 1, 2, 0], "an hour and 15 s later");
            save.truncate(0x8000 + 44);
            let mut old = board(compiled, cart_rom(4, 0x10, 0x03, 0), Some(save)).unwrap();
            old.write(0x0000, 0x0A);
            old.set_time(50_000 + 10);
            assert_eq!(latched_clock(&mut old), [0, 0, 0, 2, 0], "the 44-byte footer's u32 time");
            let ram_only = board(compiled, cart_rom(4, 0x10, 0x03, 0), Some(vec![0; 0x8000]));
            assert!(ram_only.is_ok(), "a save without a footer still loads");
            assert_eq!(
                board(compiled, cart_rom(4, 0x10, 0x03, 0), Some(vec![0; 0x8000 + 10])).err().unwrap(),
                "save is 32778 bytes, cartridge RAM is 32768, plus an optional 44- or 48-byte clock"
            );
            assert!(board(compiled, cart_rom(64, 0x1B, 0x03, 0), Some(vec![0; 0x8000 + 48])).is_err(), "no footer without a clock");
        });
    }

    #[test]
    fn mbc3_clock_that_only_ticked_is_not_a_save() {
        both(|compiled| {
            for (cartridge_type, ram_code) in [(0x0F, 0x00), (0x10, 0x03)] {
                let mut cart = board(compiled, cart_rom(4, cartridge_type, ram_code, 0), None).unwrap();
                cart.set_time(7_000);
                cart.write(0x0000, 0x0A);
                cart.set_time(7_000 + 3600);
                assert_eq!(latched_clock(&mut cart), [0, 0, 1, 0, 0]);
                cart.write(0x0000, 0x00);
                assert_eq!(cart.take_committed(), None, "type {cartridge_type:#04x}");
                assert_eq!(cart.take_unsaved(), None, "type {cartridge_type:#04x}: the footer would only restate the time already saved");
            }
        });
    }

    #[test]
    fn mbc3_timer_without_ram_saves_just_the_footer() {
        both(|compiled| {
            let mut cart = board(compiled, cart_rom(4, 0x0F, 0x00, 0), None).unwrap();
            cart.set_time(7_000);
            cart.write(0x0000, 0x0A);
            cart.write(0x4000, 0x08);
            cart.write(0xA000, 0x2A);
            cart.write(0x0000, 0x00);
            let save = cart.take_committed().expect("setting the clock commits a save");
            assert_eq!((save.len(), save[0]), (48, 0x2A));
        });
    }
    #[test]
    fn mbc3_rtc_day_carry() {
        both(|compiled| {
            let mut cart = board(compiled, cart_rom(4, 0x0F, 0x00, 0), None).unwrap();
            cart.set_time(0);
            cart.set_time(86400 * 513 + 1);
            cart.write(0x0000, 0x0A);
            cart.write(0x6000, 0x00);
            cart.write(0x6000, 0x01);
            cart.write(0x4000, 0x0B);
            assert_eq!(cart.read(0xA000), 1);
            cart.write(0x4000, 0x0C);
            assert_eq!(cart.read(0xA000), 0x80, "day counter overflowed past 511");
        });
    }

    #[test]
    fn mbc3_rtc_wraps_on_an_absurd_host_time() {
        both(|compiled| {
            let mut cart = board(compiled, cart_rom(4, 0x0F, 0x00, 0), None).unwrap();
            cart.set_time(10);
            cart.write(0x0000, 0x0A);
            for (reg, value) in [(0x08, 59), (0x09, 59), (0x0A, 23), (0x0B, 0xFF), (0x0C, 0x01)] {
                cart.write(0x4000, reg);
                cart.write(0xA000, value);
            }
            assert_eq!(latched_clock(&mut cart), [59, 59, 23, 0xFF, 0x01], "511d 23:59:59");
            cart.set_time(u64::MAX);
            assert_eq!(latched_clock(&mut cart), [4, 0, 7, 137, 0x80], "u64::MAX - 10 seconds added modulo 512 days, the carry set");
        });
    }
}
