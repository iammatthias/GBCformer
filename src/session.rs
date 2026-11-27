//! A running game: a core driving a machine frame by frame, reporting what the host must
//! act on (a finished battery save) as events.

use crate::machine::apu::Sample;
use crate::machine::io::Buttons;
use crate::machine::ppu::Frame;
use crate::machine::{Counters, GameBoy, Hardware, Machine};
use crate::reference::Reference;
use crate::stack::{AnyCore, Instrumentation, Stack};
use crate::step::Core;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Event {
    /// The game finished writing battery RAM: persist these bytes as the save file.
    Save(Vec<u8>),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Event::Save(bytes) => write!(f, "battery save of {} bytes", bytes.len()),
        }
    }
}

pub struct Session<C: Core> {
    gb: GameBoy<C>,
    frames: u64,
}

impl Session<Reference> {
    /// A game at the boot ROM hand-off, on the reference interpreter until
    /// [`Session::with_stack`] or [`Session::with_core`] picks another core.
    pub fn new(rom: Vec<u8>, saved_ram: Option<Vec<u8>>, hardware: Hardware) -> Result<Self, String> {
        let machine = Machine::new(rom, saved_ram, hardware)?;
        Ok(Session { gb: GameBoy::new(Reference, machine), frames: 0 })
    }

    /// A game at power-on, running the boot ROM `boot` before it (`Machine::power_on`).
    pub fn power_on(rom: Vec<u8>, saved_ram: Option<Vec<u8>>, hardware: Hardware, boot: Vec<u8>) -> Result<Self, String> {
        let machine = Machine::power_on(rom, saved_ram, hardware, boot)?;
        Ok(Session { gb: GameBoy::new(Reference, machine), frames: 0 })
    }

    /// The same game continuing with the components `stack` names in weights, compiled
    /// now; memory in weights starts from this session's memory. Errs when the stack
    /// breaks a dependency rule or names a component not built yet.
    pub fn with_stack(self, stack: Stack, instrumentation: Instrumentation) -> Result<Session<AnyCore>, String> {
        use crate::stack::Component;
        let built = stack.build(&self, instrumentation)?;
        let mut session = self.with_core(built.core);
        let (written, committed) = session.gb.machine.pending_save();
        session.gb.core.seed_battery(written, committed);
        if [Component::Cpu, Component::Memory, Component::Ppu].into_iter().all(|c| stack.get(c) == crate::stack::Backend::Model) {
            session.gb.machine.drop_shadow();
        }
        if let Some(apu) = built.apu {
            session.gb.machine.replace_apu(apu);
        }
        if let Some(control) = built.control {
            session.gb.machine.replace_control(control);
        }
        if let Some(ppu) = built.ppu {
            session.gb.machine.ppu.set_renderer(ppu);
        }
        Ok(session)
    }
}

impl Session<AnyCore> {
    /// What was compiled: the core's checkpoint, the APU's and the control logic's, if any.
    pub fn describe(&self) -> Vec<String> {
        let mut lines = self.gb.core.describe();
        lines.extend(self.gb.machine.apu_describe());
        lines.extend(self.gb.machine.control_describe());
        lines.extend(self.gb.machine.ppu.renderer().map(|r| r.describe()).unwrap_or_default());
        lines
    }

    /// Waits for the cells' rows a worker thread still has to write. A memo check that
    /// failed there stops the run here, before anything reports it.
    pub fn settle(&self) {
        if let Some(memory) = self.gb.core.engine().and_then(|e| e.memory.as_ref()) {
            memory.cells.settle();
        }
    }

    /// After a run of `wall` seconds: the core's report and the devices' counters.
    pub fn report(&mut self, wall: f64, detailed: bool) -> Vec<String> {
        let mut lines = self.gb.core.report(wall, detailed);
        lines.extend(self.gb.machine.apu_report());
        if self.gb.machine.control_reference().is_none() {
            lines.extend(self.gb.machine.control_report());
        }
        lines.extend(self.gb.machine.ppu.renderer_mut().map(|r| r.report()).unwrap_or_default());
        lines
    }
}

impl<C: Core> Session<C> {
    /// The same game continuing on another core, with fresh counters.
    pub fn with_core<D: Core>(self, core: D) -> Session<D> {
        Session { gb: self.gb.with_core(core), frames: self.frames }
    }

    /// Runs one frame with `buttons` held.
    pub fn run_frame(&mut self, buttons: Buttons) -> Vec<Event> {
        self.gb.machine.set_buttons(buttons);
        self.gb.run_frame();
        self.frames += 1;
        self.gb.take_save(true).map(Event::Save).into_iter().collect()
    }

    /// `run_frame`, handing `watch` the GBC CPU state and the opcode at PC before every step
    /// ([`GameBoy::run_frame_watching`]).
    pub fn run_frame_watching(&mut self, buttons: Buttons, watch: impl FnMut(&crate::step::Cpu, Option<u8>)) -> Vec<Event> {
        self.gb.machine.set_buttons(buttons);
        self.gb.run_frame_watching(watch);
        self.frames += 1;
        self.gb.take_save(true).map(Event::Save).into_iter().collect()
    }

    /// The host clock in Unix seconds, for a cartridge clock (MBC3). The session never
    /// reads a clock itself: the host calls this before each frame, or pins one time for
    /// a deterministic replay. Milliseconds (`Date.now()` as is) run the clock 1000x fast.
    pub fn set_time(&mut self, unix_seconds: u64) {
        self.gb.machine.set_time(unix_seconds);
    }

    /// Battery RAM written since the last save, finished or not: persist it on exit.
    pub fn shut_down(&mut self) -> Option<Vec<u8>> {
        self.gb.take_save(false)
    }

    /// The last finished frame. With the PPU checkpoint on its own thread this waits for
    /// it to finish drawing, so reading the frame only when it is shown keeps the
    /// machine's thread from waiting at every VBlank.
    pub fn frame(&mut self) -> &Frame {
        self.gb.machine.frame()
    }

    /// Frames run so far; the frame `run_frame` runs next has this number.
    pub fn frame_number(&self) -> u64 {
        self.frames
    }

    /// Appends the audio mixed since the last drain. With the APU checkpoint on its own
    /// thread that may stop up to about two frames short of the emulated time;
    /// [`Session::finish_audio`] waits for the rest. Undrained audio beyond a few frames
    /// is dropped.
    pub fn drain_audio(&mut self, out: &mut Vec<Sample>) {
        self.gb.machine.take_samples(out);
    }

    /// [`Session::drain_audio`] after an APU on its own thread, which may run up to two
    /// frames behind, has mixed every M-cycle emulated so far.
    pub fn finish_audio(&mut self, out: &mut Vec<Sample>) {
        self.gb.machine.settle_apu();
        self.gb.machine.take_samples(out);
    }

    pub fn counters(&self) -> Counters {
        let (dma_blocked, palette_reads, oam_bugs) = self.gb.machine.hits;
        Counters { dma_blocked, palette_reads, oam_bugs, midline: self.gb.machine.ppu.midline(), ..self.gb.counters }
    }

    /// Bytes the game sent over the serial port (test ROMs print their results there).
    pub fn serial_output(&self) -> &[u8] {
        self.gb.machine.serial_output()
    }

    /// Cartridge RAM, every bank: blargg's newer tests write their status, signature and text at A000.
    pub fn cart_ram(&self) -> &[u8] {
        self.gb.machine.cart_ram()
    }

    /// Work RAM (the bank mapped at D000) or high RAM as the game left it, from the core's
    /// own memory when it holds them; other addresses read 0xFF.
    pub fn ram_byte(&self, addr: u16) -> u8 {
        self.gb.ram_byte(addr)
    }

    /// Connects two games with a link cable (before either hands control to weights).
    pub fn link<D: Core>(&mut self, other: &mut Session<D>) -> Result<(), String> {
        let (a, b) = crate::machine::io::Link::pair();
        self.gb.machine.plug(a)?;
        other.gb.machine.plug(b)
    }

    /// Runs one frame of each of two linked games, stepping whichever is behind in emulated
    /// time, so a transfer one clocks reaches the other within one GBC CPU step.
    pub fn run_linked_frame<D: Core>(&mut self, other: &mut Session<D>, buttons: (Buttons, Buttons)) {
        self.gb.machine.set_buttons(buttons.0);
        other.gb.machine.set_buttons(buttons.1);
        self.gb.frame_end += self.gb.machine.frame_halves();
        other.gb.frame_end += other.gb.machine.frame_halves();
        loop {
            let (a, b) = (self.gb.machine.halves(), other.gb.machine.halves());
            match (a < self.gb.frame_end, b < other.gb.frame_end) {
                (false, false) => break,
                (true, false) => self.gb.step(),
                (false, true) => other.gb.step(),
                (true, true) if a <= b => self.gb.step(),
                (true, true) => other.gb.step(),
            };
        }
        self.gb.core.audit_memory(&self.gb.machine.ram_regions());
        other.gb.core.audit_memory(&other.gb.machine.ram_regions());
        self.frames += 1;
        other.frames += 1;
    }

    /// The boot ROM is still mapped (a session from [`Session::power_on`]).
    pub fn booting(&self) -> bool {
        self.gb.machine.booting()
    }

    /// Runs GBC CPU steps until the boot ROM unmaps itself, at most `max` of them; whether
    /// it did. Frames are not counted: the hand-off lands mid-frame.
    pub fn finish_boot(&mut self, max: u64) -> bool {
        for _ in 0..max {
            if !self.booting() {
                // The next frame starts at the hand-off.
                self.gb.frame_end = self.gb.machine.halves();
                return true;
            }
            self.gb.step();
        }
        !self.booting()
    }

    /// The GBC CPU's registers as the host holds them.
    pub fn cpu(&self) -> crate::step::Cpu {
        self.gb.cpu
    }

    pub fn machine(&self) -> &Machine {
        &self.gb.machine
    }

    /// How far the last step ran past the end of the last frame, in the machine's time.
    pub fn overshoot(&self) -> u64 {
        self.gb.machine.halves() - self.gb.frame_end
    }

    /// A bus read at this M-cycle, for inspection: I/O registers read as the GBC CPU would
    /// read them (a control or APU device catches up first).
    pub fn bus_read(&mut self, addr: u16) -> u8 {
        crate::step::Bus::read(&mut self.gb.machine, addr)
    }

    /// Emulated time so far, in half M-cycles of normal speed.
    pub fn halves(&self) -> u64 {
        self.gb.machine.halves()
    }

    pub fn core(&self) -> &C {
        &self.gb.core
    }

    /// The machine's memory as an image for a memory core taking over now: ROM, every
    /// RAM, the markers and the banking registers.
    pub fn memory_image(&self) -> Result<crate::memory::image::Image, String> {
        self.gb.machine.memory_image()
    }

    /// The reference timer and interrupt flags at this M-cycle, while no device has
    /// replaced them.
    pub fn control_reference(&self) -> Option<&crate::machine::control::Control> {
        self.gb.machine.control_reference()
    }

    /// The PPU's display bytes and window state at this M-cycle, which a renderer taking over
    /// starts from.
    pub fn ppu_state(&self) -> (crate::machine::ppu::Display, crate::machine::fifo::Window) {
        let ppu = &self.gb.machine.ppu;
        (ppu.display(), ppu.window())
    }

    /// The board each drawn line's mode-3 length reaches the control logic by.
    pub fn ppu_lengths(&self) -> crate::machine::ppu::Lengths {
        self.gb.machine.ppu.lengths()
    }

    /// The reference APU at this M-cycle, while no device has replaced it.
    pub fn apu_reference(&self) -> Option<&crate::machine::apu::Apu> {
        self.gb.machine.apu_reference()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An MBC3 clock cartridge whose program latches the clock every loop and copies the
    /// seconds register to C000.
    fn clock_rom() -> Vec<u8> {
        let mut rom = vec![0; 0x8000];
        rom[0x147] = 0x10;
        rom[0x149] = 0x03;
        #[rustfmt::skip]
        let program = [
            0x3E, 0x0A, 0xEA, 0x00, 0x00, // LD A,0A; LD (0000),A   enable RAM
            0x3E, 0x08, 0xEA, 0x00, 0x40, // LD A,08; LD (4000),A   select the seconds register
            0xAF, 0xEA, 0x00, 0x60,       // XOR A; LD (6000),A     arm the latch
            0x3E, 0x01, 0xEA, 0x00, 0x60, // LD A,01; LD (6000),A   latch
            0xFA, 0x00, 0xA0,             // LD A,(A000)
            0xEA, 0x00, 0xC0,             // LD (C000),A
            0x18, 0xEF,                   // JR back to the arm
        ];
        rom[0x100..0x100 + program.len()].copy_from_slice(&program);
        rom
    }

    #[test]
    fn the_cartridge_clock_moves_only_when_the_host_sets_the_time() {
        let mut s = Session::new(clock_rom(), None, Hardware::Dmg).unwrap();
        let seconds = |s: &mut Session<Reference>| s.ram_byte(0xC000);
        s.set_time(1_000);
        s.run_frame(Buttons(0));
        assert_eq!(seconds(&mut s), 0);
        s.set_time(1_045);
        s.run_frame(Buttons(0));
        assert_eq!(seconds(&mut s), 45);
        for _ in 0..3 {
            s.run_frame(Buttons(0));
        }
        assert_eq!(seconds(&mut s), 45, "pinned: no frame reads a clock of its own");
    }
}
