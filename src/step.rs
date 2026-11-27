//! The CPU/host contract. One call pair per CPU step, identical for the reference
//! interpreter and the compiled model, so the two can be compared field by field.
//!
//! A step is either a HALT wake-up, an interrupt dispatch, or one instruction:
//!
//! 1. The host reads the three bytes at `pc` (bus rules apply, no side effects on DMG)
//!    and the IE/IF registers into [`Inputs`].
//! 2. [`Core::request`] names up to two data addresses to read.
//! 3. The host reads them and calls [`Core::execute`], which returns the next CPU
//!    state, up to two memory writes, the step length, and interrupt acknowledgements.
//! 4. The host commits the writes, clears acknowledged IF bits, and advances the
//!    peripherals by `mcycles`. Timing is per step, not per M-cycle.
//!
//! While halted with no pending interrupt the host advances time without calling
//! the core; nothing in the CPU changes in that state. Whether one is pending is the
//! bus's to say ([`Bus::wake`]): the control logic emits it.
//!
//! A core whose memory lives inside the model ([`crate::sm83::CircuitMemCore`] and the
//! memory checkpoint) reads the opcode and data bytes itself. It still receives the
//! bytes the host read, and uses them only for the addresses the host serves
//! ([`StepFacts::served_by_host`]). The host reads each of those from the device the model's
//! physical address names ([`StepFacts::read_ports`], `layout::port`), and hands each write
//! to host memory when the model records it and to the device its physical address
//! names ([`StepFacts::write_targets`]): the model, not the bus map, decides where an I/O
//! access goes. [`Core::audit`] lets it hold its own fetches, reads and those addresses
//! to the bus in lockstep. The bytes a DMA copies on the host reach it after the
//! advance ([`Core::copied`]), and [`Core::audit_memory`] holds its memory image to the
//! host's arrays after each DMA and at every frame end.

use crate::memory::layout::{self, Access, Banking, Port};
use crate::memory::{Addr, Copied};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cpu {
    pub a: u8,
    pub f: u8,
    pub b: u8,
    pub c: u8,
    pub d: u8,
    pub e: u8,
    pub h: u8,
    pub l: u8,
    pub sp: u16,
    pub pc: u16,
    pub ime: bool,
    /// EI was the previous instruction; IME turns on after the current one.
    pub ei_pending: bool,
    pub halted: bool,
    /// HALT ran with IME off and an interrupt pending: the next opcode byte is read twice.
    pub halt_bug: bool,
}

impl Cpu {
    /// DMG register state after the boot ROM hands over at 0x0100.
    pub fn post_boot_dmg() -> Self {
        Cpu { a: 0x01, f: 0xB0, b: 0x00, c: 0x13, d: 0x00, e: 0xD8, h: 0x01, l: 0x4D, sp: 0xFFFE, pc: 0x0100, ..Cpu::default() }
    }

    /// CGB register state after the boot ROM hands a DMG game over in compatibility mode
    /// (Pan Docs, "Power Up Sequence"; B and HL depend on the title, these are for an
    /// unrecognised one).
    pub fn post_boot_compat() -> Self {
        Cpu { a: 0x11, f: 0x80, b: 0x00, c: 0x00, d: 0x00, e: 0x08, h: 0x00, l: 0x7C, sp: 0xFFFE, pc: 0x0100, ..Cpu::default() }
    }

    /// CGB register state after the boot ROM hands a CGB-mode game over at 0x0100.
    pub fn post_boot_cgb() -> Self {
        Cpu { a: 0x11, f: 0x80, b: 0x00, c: 0x00, d: 0xFF, e: 0x56, h: 0x00, l: 0x0D, sp: 0xFFFE, pc: 0x0100, ..Cpu::default() }
    }

    /// The address of fetched byte `k`: PC + k, one byte back after the opcode under the
    /// HALT bug, whose operand 1 is a second read of the opcode's address (Pan Docs, "halt";
    /// SameBoy v1.0.5 `Core/sm83_cpu.c`, the run mode's `pc--` after the opcode's read).
    pub fn fetch_address(&self, k: u16) -> u16 {
        self.pc.wrapping_add(k).wrapping_sub(if k > 0 { self.halt_bug as u16 } else { 0 })
    }

    /// The step's fetched bytes, each read at its fetch address.
    pub fn fetched(&self, mut byte: impl FnMut(u16) -> u8) -> [u8; 3] {
        [0, 1, 2].map(|k| byte(self.fetch_address(k)))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inputs {
    /// The fetched bytes: fetch k at [`Cpu::fetch_address`] (pc, pc+1, pc+2, or under the
    /// HALT bug pc, pc again and pc+1), each read under its own M-cycle's bus state.
    pub op: [u8; 3],
    pub ie: u8,
    pub iflag: u8,
    /// OAM DMA blocks the GBC CPU's accesses on the buses `dma_buses` names.
    pub dma: bool,
    /// `Lines::dma_buses`: the transfer's buses (`layout::Dma` bits) in the low nibble,
    /// those blocked at the step's start in the high.
    pub dma_buses: u8,
    /// The GBC CPU's access locks (bit 0 VRAM in mode 3, bit 1 OAM in modes 2 and 3) as
    /// the bus knows them without asking the control logic, or None.
    pub locks: Option<u8>,
    /// The locks the bus asked the control logic for, when the core's request did.
    pub asked: u8,
    /// Each access's bus state where the host predicted it differs from the step's start
    /// ([`slot_state`]): read 0 and read 1 in the low and high nibbles of byte 0, writes 0
    /// and 1 in byte 1, the fetches of operands 1 and 2 in byte 2. A nibble with bit 3
    /// ([`OVERRIDE`]) set replaces that access's state with its low bits ([`VRAM_LOCKED`],
    /// [`OAM_LOCKED`], [`DMA_BLOCKING`]); a fetch's takes only the DMA bit.
    pub access: [u8; 3],
}

/// The M-cycle of its step each data read and write falls in, counted from the step's
/// start, where the opcode fetch is M-cycle 0 (SingleStepTests' bus log order;
/// `docs/hardware-tests.md` §4.1). Slot i of each is read i or write i of the step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccessCycles {
    pub reads: [u8; 2],
    pub writes: [u8; 2],
    /// The operand bytes the instruction fetches after its opcode, operand k at M-cycle k
    /// (a CB opcode's second byte is one).
    pub fetches: u8,
}

/// A data access's bus state, as [`Inputs::access`] and [`Bus::ahead`] carry it.
pub const VRAM_LOCKED: u8 = 1;
pub const OAM_LOCKED: u8 = 2;
pub const DMA_BLOCKING: u8 = 4;
/// An access nibble that replaces the step-start state.
pub const OVERRIDE: u8 = 8;
/// The bits of one access's nibble in [`Inputs::access`]: two accesses to a byte.
pub const ACCESS_SHIFT: u8 = 4;
/// The write accesses among a step's slots (`transport::SLOTS` order: reads 0 and 1, writes
/// 0 and 1, then the operand fetches), which take the writes' locks: those sit
/// `WRITE_SHIFT` bits above the reads' in `Lines::locks` (`Lcd::locks`).
pub const WRITE_SLOTS: u8 = 0b1100;
pub const WRITE_SHIFT: u8 = 2;

/// The bus state at the step's start as the memory circuit reads it: OAM DMA (CTL bit 4)
/// and the locks it holds (CTL bits 5-6, or the late `LK` the host asked for).
pub fn start_state(inp: &Inputs) -> u8 {
    crate::transport::start_state(STATE_BITS, inp.locks, inp.asked, inp.dma)
}

/// The bus state access `slot` (0 read 0, 1 read 1, 2 write 0, 3 write 1, 4 and 5 the
/// fetches of operands 1 and 2) is translated under: its override nibble when set, else
/// the step's start.
pub fn slot_state(inp: &Inputs, slot: usize) -> u8 {
    let n = inp.access[slot / 2] >> (ACCESS_SHIFT * (slot % 2) as u8) & 15;
    if n & OVERRIDE != 0 { n & 7 } else { start_state(inp) }
}

/// What OAM DMA blocks at data access `slot`: the transfer's buses (`Inputs::dma_buses`)
/// when its state ([`slot_state`]) has the DMA bit.
pub fn slot_dma(inp: &Inputs, slot: usize) -> layout::Dma {
    layout::Dma(if slot_state(inp, slot) & DMA_BLOCKING != 0 { inp.dma_buses & 15 } else { 0 })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Request {
    pub reads: [Option<u16>; 2],
    /// The core needs the access locks the bus could not give (`Inputs::locks` None).
    pub locks: bool,
    /// The M-cycle each read and write of the step falls in, when the core says.
    pub cycles: Option<AccessCycles>,
    /// The step reaches VRAM or the FE page by a data access's bus address, so the access
    /// locks at each access's M-cycle matter (the memory circuit's RC bit 3).
    pub at_lock: bool,
    /// An operand's line reaches VRAM or the FE page, so the access locks at each operand
    /// fetch's M-cycle matter (the memory circuit's WC bit 3).
    pub fetch_lock: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub cpu: Cpu,
    pub writes: [Option<(u16, u8)>; 2],
    pub mcycles: u8,
    /// IF bits the CPU acknowledged by dispatching; the host clears them.
    pub irq_ack: u8,
    /// The step ran STOP. What it does is the control logic's ([`Bus::stop`]): a CGB
    /// speed switch when KEY1 armed one.
    pub stop: bool,
    /// The M-cycle of each M-cycle without an access in which the GBC CPU put an FE page
    /// address on the bus (SameBoy's `cycle_oam_bug` sites: INC rr and DEC rr, JR, PUSH,
    /// CALL and RST before their writes, LD SP,HL, and an interrupt dispatch's PC + 1 and
    /// SP), which a DMG's OAM corruption bug follows ([`Bus::oam_bug`]). Slot 1 is the
    /// dispatch's SP alone.
    pub idle: [Option<u8>; 2],
}

/// What a core says of the step in hand, beside its request and outcome.
#[derive(Clone, Copy, Debug)]
pub struct StepFacts {
    /// Which of the requested reads the host must perform. A core with its own memory
    /// asks only for the bytes outside it (I/O registers, IE, the MBC3 clock).
    pub served_by_host: [bool; 2],
    /// Whether the core takes the bytes at PC and its data bytes from the bus. A core whose
    /// memory lives in the model reads them itself and asks the host only through the
    /// ports it names; the bus then gives a reference beside it (lockstep, the audit) a
    /// side-effect-free peek, so a run with a reference and one without take the same
    /// passes.
    pub reads_the_bus: bool,
    /// The core translates its own data accesses (a memory core), so the host hands it
    /// each access's bus state where it differs from the step's start ([`Inputs::access`]).
    pub timed_accesses: bool,
    /// The device of each read the host serves, when the core's model names it by the
    /// physical address it emitted (`layout::port`); None where the host reads the bus
    /// address.
    pub read_ports: [Option<Port>; 2],
    /// After [`Core::request`]: the device of each fetched byte the host must serve
    /// before the request stands, from the physical address the model emitted for it
    /// (`FP0`..`FP2`). All None when the core needs no fetched byte from the host;
    /// otherwise [`request`] reads them and calls [`Core::serve_fetch`].
    pub fetch_ports: [Option<Port>; 3],
    /// The request's accesses that put an FE page address on the bus, for a core that
    /// serves its own accesses: reads 0 and 1, writes 0 and 1, then the fetches of the
    /// opcode and operands 1 and 2.
    pub fe_accesses: [bool; 7],
    /// The physical address the model emitted for each write of the last outcome, which
    /// places it: host memory when the model records it, the device `layout::port`
    /// names. None where the bus address places the write (a core without memory).
    pub write_targets: [Option<u32>; 2],
    /// The second device each write of the last outcome reaches, by the second port the
    /// model emitted for it (LCDC's control copy, SVBK's, the clock's RAMG and RAMB).
    pub second_ports: [Option<Port>; 2],
    /// The physical address the model emitted for each requested read of the last
    /// outcome; None for a read the bus address places (a core without memory).
    pub read_targets: [Option<u32>; 2],
}

impl StepFacts {
    /// A core without memory of its own: the host serves every byte at its bus address.
    pub const BUS: StepFacts = StepFacts { served_by_host: [true; 2], reads_the_bus: true, timed_accesses: false, read_ports: [None; 2], fetch_ports: [None; 3], fe_accesses: [false; 7], write_targets: [None; 2], second_ports: [None; 2], read_targets: [None; 2] };
}

pub trait Core {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request;
    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome;
    /// [`Core::execute`] in a step on `bus`, after the host served the reads it serves. A
    /// core that observes other bytes beside them (a reference in lockstep) takes them
    /// from the bus's side-effect-free [`Bus::peek`], so observing changes nothing the
    /// devices do.
    fn execute_on(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2], _bus: &impl Bus) -> Outcome {
        self.execute(cpu, inp, data)
    }
    /// After [`Core::execute_on`], at write `i`'s M-cycle: a core that keeps write records
    /// in its own memory keeps write `i`'s there now, so nothing that runs before the
    /// write's M-cycle (a line drawn, a DMA byte copied) sees it.
    fn land_write(&mut self, _i: usize) {}
    /// What the host must know of the step in hand. A core without memory of its own
    /// says [`StepFacts::BUS`].
    fn facts(&self) -> &StepFacts {
        &StepFacts::BUS
    }
    /// Takes the bytes read at [`StepFacts::fetch_ports`] (0 where None) and returns the
    /// step's request.
    fn serve_fetch(&mut self, _cpu: &Cpu, _inp: &Inputs, _bytes: [u8; 3]) -> Request {
        unreachable!("this core asks the host for no fetched byte")
    }
    /// Loads a bus image into a core's own memory (SingleStepTests give one per case).
    /// Errs when the image cannot be held faithfully: two bus addresses that alias one
    /// cell (echo RAM) with different values.
    fn load_bus(&mut self, _ram: &[(u16, u8)]) -> Result<(), String> {
        Ok(())
    }
    /// After a step, holds a core's own fetches, reads and write records to the bus.
    /// Panics at the first disagreement, naming it.
    fn audit(&self, _bus: &impl Bus, _cpu: &Cpu, _inp: &Inputs, _out: &Outcome) {}
    /// After the host advanced its devices: every byte a DMA copied meanwhile, in order.
    /// A core whose memory lives in the model keeps each as records, reading the source
    /// from its own memory.
    fn copied(&mut self, _bytes: &[Copied]) {}
    /// Holds a core's own memory to the host's bytes, region by region (fixed-layout
    /// base and bytes). Panics at the first byte that differs.
    fn audit_memory(&self, _host: &[(Addr, &[u8])]) {}
    /// A core that holds cartridge RAM in its own memory says when a battery save is due
    /// and gives `len` bytes of it: Some(None) when none is, None when the host's
    /// cartridge holds the save.
    fn take_save(&mut self, _committed: bool, _len: usize) -> Option<Option<Vec<u8>>> {
        None
    }
    /// A core that holds work RAM and high RAM in its own memory answers an observer's
    /// read of them ([`crate::machine::GameBoy::ram_byte`]) from it.
    fn peek_ram(&self, _addr: u16) -> Option<u8> {
        None
    }
    /// A DMG's OAM corruption bug at OAM row `row` (`Bus::oam_row`, nonzero) from a read
    /// or not: a core with its own OAM rewrites it as its model emits and returns the lines
    /// it wrote. A core without OAM is never asked.
    fn oam_bug(&mut self, _row: u8, _read: bool) -> Vec<(u32, [u8; 8])> {
        unreachable!("this core holds no OAM")
    }
}

/// Memory as the CPU sees it.
pub trait Bus {
    fn read(&mut self, addr: u16) -> u8;
    fn write(&mut self, addr: u16, value: u8);
    fn ie(&self) -> u8;
    fn iflag(&self) -> u8;
    fn ack(&mut self, bits: u8);
    fn dma_active(&self) -> bool {
        false
    }
    /// `Lines::dma_buses`. A bus without them blocks every bus below FE00 and OAM.
    fn dma_buses(&self) -> u8 {
        let all = crate::memory::layout::Dma::ALL.0;
        all | if self.dma_active() { all << 4 } else { 0 }
    }
    /// The GBC CPU's access locks, when the bus knows them without a pass (`Lines::locks`).
    fn locks(&self) -> Option<u8> {
        Some(0)
    }
    /// The access locks from the control logic, a pass if need be.
    fn lock_query(&mut self) -> u8 {
        0
    }
    /// Runs the devices `mcycles` M-cycles into the step, before its next access, handing
    /// each block's DMA copies to `hand_over` before the next block runs.
    fn tick(&mut self, _mcycles: u8, _hand_over: &mut dyn FnMut(&mut Vec<Copied>)) {}
    /// The bus state ([`VRAM_LOCKED`], [`OAM_LOCKED`], [`DMA_BLOCKING`]) `mcycles` M-cycles
    /// from now, as the control logic emitted it: OAM DMA always, the locks only with
    /// `locks` (else 0), a write's with `write`. A bus without timing stays as it is.
    fn ahead(&mut self, _mcycles: u8, locks: bool, write: bool) -> u8 {
        let shift = if write { WRITE_SHIFT } else { 0 };
        let l = if locks { (self.locks().unwrap_or(0) >> shift) & (VRAM_LOCKED | OAM_LOCKED) } else { 0 };
        l | (self.dma_active() as u8) << 2
    }
    /// A model-emitted address landed in `layout::BLOCKED`: OAM DMA blocked the access,
    /// so a device moving the transfer lands its bytes so far.
    fn blocked(&mut self) {}
    /// A read of the device a model-emitted address names (`layout::port`).
    fn read_port(&mut self, port: Port) -> u8 {
        panic!("this bus has no device behind {port:?}")
    }
    /// A write to the device a model-emitted address names.
    fn write_port(&mut self, port: Port, _value: u8) {
        panic!("this bus has no device behind {port:?}")
    }
    /// A write the model recorded at physical address `p`, kept in the bus's own memory
    /// and reaching no device.
    fn write_memory(&mut self, addr: u16, _p: u32, value: u8) {
        self.write(addr, value);
    }
    /// An interrupt is enabled and requested, so a halted CPU wakes. A bus whose control
    /// logic emits this (`Lines::wake`) says so; others test IE and IF.
    fn wake(&self) -> bool {
        self.ie() & self.iflag() & 0x1F != 0
    }
    /// The fixed-layout address of a bus access (`memory::layout`) under the bus state
    /// `state`, when the bus can say.
    fn physical(&self, _addr: u16, _access: Access, _state: u8) -> Option<u32> {
        None
    }
    /// The byte a read under the bus state `state` would return, without side effects,
    /// when the bus can say.
    fn peek(&self, _addr: u16, _state: u8) -> Option<u8> {
        None
    }
    fn banking(&self) -> Option<Banking> {
        None
    }
    /// The CPU ran STOP; it reaches the bus after the step's writes.
    fn stop(&mut self) {}
    /// The GBC CPU put an FE page address on the bus at this M-cycle (`Outcome::idle`, or a
    /// read or write of FE00-FEFF), `read` for a read: a DMG's OAM corruption bug on the
    /// bus's own OAM.
    fn oam_bug(&mut self, _read: bool) {}
    /// The OAM row the PPU's scan reads at this M-cycle, from the control logic, 0 for
    /// none: what a DMG's OAM corruption bug rewrites in a core's own OAM (`Core::oam_bug`).
    fn oam_row(&mut self) -> u8 {
        0
    }
    /// The lines a core's OAM corruption at `row` wrote (physical address, line), which
    /// the bus's copy of memory follows.
    fn oam_records(&mut self, _row: u8, _read: bool, _records: &[(u32, [u8; 8])]) {}
}

/// A step's request: [`Core::request`], then the fetched bytes the core asks the host
/// for, each read by `read` from the device the model's address names.
pub fn request(core: &mut impl Core, cpu: &Cpu, inp: &Inputs, read: impl FnMut(Port) -> u8) -> Request {
    request_fetching(core, cpu, &mut inp.clone(), read)
}

/// [`request`], leaving each fetched byte the host served in `inp.op`, so the execute,
/// the audit and a reference beside the core see the bytes the model was given.
fn request_fetching(core: &mut impl Core, cpu: &Cpu, inp: &mut Inputs, mut read: impl FnMut(Port) -> u8) -> Request {
    let req = core.request(cpu, inp);
    let ports = core.facts().fetch_ports;
    if ports.iter().all(Option::is_none) {
        return req;
    }
    let bytes = ports.map(|p| p.map_or(0, &mut read));
    for k in 0..3 {
        if ports[k].is_some() {
            inp.op[k] = bytes[k];
        }
    }
    core.serve_fetch(cpu, inp, bytes)
}

/// An FE page address on the bus: a bus with its own memory corrupts it, and a core with
/// its own OAM runs its corruption at the row the control logic names, whose lines the
/// bus's copy follows.
fn oam_event<C: Core, B: Bus>(core: &mut C, bus: &mut B, read: bool) {
    if core.facts().reads_the_bus {
        bus.oam_bug(read);
    } else {
        crate::transport::bus_event(&mut OamRows { core, bus }, read);
    }
}

/// A core with its own OAM and the bus, as the shared bus event sees them.
struct OamRows<'a, C, B> {
    core: &'a mut C,
    bus: &'a mut B,
}

impl<C: Core, B: Bus> crate::transport::Rewrites for OamRows<'_, C, B> {
    fn row(&mut self) -> u8 {
        self.bus.oam_row()
    }

    fn rewrite(&mut self, row: u8, read: bool) {
        let records = self.core.oam_bug(row, read);
        self.bus.oam_records(row, read, &records);
    }
}

/// Runs one step and returns its length in M-cycles. Returns `None` when the CPU is
/// halted with nothing pending, so the caller can fast-forward to the next event.
pub fn step<B: Bus>(core: &mut impl Core, cpu: &mut Cpu, bus: &mut B) -> Option<u8> {
    if cpu.halted && !bus.wake() {
        return None;
    }
    let &StepFacts { reads_the_bus: reads, timed_accesses: timed, .. } = core.facts();
    let held = bus.locks();
    let mut inp = Inputs {
        op: opcode_bytes(bus, cpu, reads && !cpu.halted, crate::transport::start_state(STATE_BITS, held, 0, bus.dma_active())),
        ie: bus.ie(),
        iflag: bus.iflag(),
        dma: bus.dma_active(),
        dma_buses: bus.dma_buses(),
        locks: bus.locks(),
        asked: 0,
        access: [0; 3],
    };
    let (mut req, _, locks_asked) = crate::transport::open_step(&mut StepOpening { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp }, held);
    inp.asked = locks_asked;
    let mut line = crate::transport::Timeline::default();
    // Operand k lands at M-cycle k (mooneye `call_timing` and kin read an operand across
    // OAM DMA's end). A core that takes its bytes from the bus gets each operand read
    // there, and asks again when one differs from the step's start. A core with its own
    // memory fetches under the step's start (`docs/hardware-tests.md` §3.11).
    if !timed
        && let Some(at) = req.cycles
        && at.fetches > 0
    {
        let mut changed = false;
        for k in 1..=at.fetches as u16 {
            line.reach(&mut StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads }, STATE_BITS, k as u8);
            let b = bus.read(cpu.fetch_address(k));
            if b != inp.op[k as usize] {
                inp.op[k as usize] = b;
                changed = true;
            }
        }
        if changed {
            let again = request_fetching(core, cpu, &mut inp, |p| bus.read_port(p));
            assert_eq!(again.cycles.map(|c| c.fetches), Some(at.fetches), "the operand count changed with the operands' bytes");
            req = again;
            if req.locks {
                inp.asked = bus.lock_query();
            }
        }
    }
    // Each access lands at its M-cycle: the devices run up to it first. A core that
    // translates its own accesses gets each one's bus state where it differs from the
    // step's start, and then asks again under them: an access the start's OAM DMA blocked
    // can reach a port the host serves once the transfer ends.
    let mut early: [Option<u8>; 2] = [None; 2];
    if timed
        && let Some(at) = req.cycles
    {
        let start = start_state(&inp);
        let mut h = StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads };
        (line, _, early) = crate::transport::settle_accesses(&mut h, STATE_BITS, asked(&req, at), start);
        req = h.req;
        // Operand k's byte for the audit and a reference beside the core, under the
        // fetch's own state. One the transfer's end unblocks is read once the devices ran
        // to that end (the settled accesses did, as every stream does); a blocked one
        // reads 0xFF whenever it is read.
        for k in 1..=at.fetches as u16 {
            let state = slot_state(&inp, 3 + k as usize);
            if state == start {
                continue;
            }
            assert!(start & DMA_BLOCKING == 0 || state & DMA_BLOCKING != 0 || line.done > 0, "operand {k} unblocked before the devices ran to the transfer's end");
            let a = cpu.fetch_address(k);
            inp.op[k as usize] = if reads { bus.read(a) } else { bus.peek(a, state).unwrap_or(0xFF) };
        }
    }
    let at = req.cycles;
    let &StepFacts { served_by_host: served, read_ports: ports, fe_accesses, .. } = core.facts();
    // A core that serves its own accesses says which put an FE page address on the bus,
    // each an OAM bug event at its M-cycle; a bus that serves them sees them itself.
    let fe = if reads { [false; 7] } else { fe_accesses };
    // The fetches' events first, in M-cycle order: the opcode at the step's start, operand
    // k at M-cycle k, before any data access.
    for k in (0..3).filter(|&k| fe[4 + k]) {
        if let Some(at) = at {
            assert!(line.done <= k as u8 && (k == 0 || k as u8 <= at.fetches), "fetch {k}'s event after the step ran to M-cycle {}", line.done);
            line.reach(&mut StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads }, STATE_BITS, k as u8);
        }
        oam_event(core, bus, true);
    }
    let mut data = [0u8; 2];
    for i in 0..2 {
        if let Some(byte) = early[i] {
            // Served while OAM DMA still blocked the access, which blocks the FE page.
            assert!(!fe[i], "read {i}, served before OAM DMA's end, puts an FE page address on the bus");
            data[i] = byte;
            continue;
        }
        if let (Some(at), Some(_)) = (at, req.reads[i]) {
            line.reach(&mut StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads }, STATE_BITS, at.reads[i]);
        }
        if fe[i] {
            oam_event(core, bus, true);
        }
        data[i] = serve(bus, &req, served, ports, i, reads, slot_state(&inp, i));
    }
    let out = core.execute_on(cpu, &inp, data, bus);
    core.audit(bus, cpu, &inp, &out);
    for p in core.facts().read_targets.into_iter().flatten() {
        if layout::is_blocked(p) {
            bus.blocked();
        }
    }
    for m in out.idle.into_iter().flatten() {
        if at.is_some() {
            line.reach(&mut StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads }, STATE_BITS, m);
        }
        oam_event(core, bus, false);
    }
    let &StepFacts { write_targets: targets, second_ports: second, .. } = core.facts();
    for (i, (addr, value)) in out.writes.into_iter().enumerate().filter_map(|(i, w)| Some((i, w?))) {
        if let Some(at) = at {
            line.reach(&mut StepAccess { core: &mut *core, cpu, bus: &mut *bus, inp: &mut inp, req, reads }, STATE_BITS, at.writes[i]);
        }
        if fe[2 + i] {
            oam_event(core, bus, false);
        }
        core.land_write(i);
        match targets[i] {
            None => bus.write(addr, value),
            Some(p) => {
                if layout::recorded(p) {
                    bus.write_memory(addr, p, value);
                }
                if let Some(port) = layout::port(p) {
                    bus.write_port(port, value);
                } else if layout::is_blocked(p) {
                    bus.blocked();
                }
                if let Some(port) = second[i] {
                    bus.write_port(port, value);
                }
            }
        }
    }
    bus.ack(out.irq_ack);
    if out.stop {
        bus.stop();
    }
    *cpu = out.cpu;
    Some(out.mcycles)
}

/// Read `i` of the request as the host serves it: from the device the core's emitted
/// address names, from the bus address when the core names none (a reference beside a
/// memory core peeks it under `state`, the read's own bus state), or 0 where the core reads
/// its own memory.
fn serve<B: Bus>(bus: &mut B, req: &Request, served: [bool; 2], ports: [Option<Port>; 2], i: usize, reads: bool, state: u8) -> u8 {
    match (req.reads[i], ports[i]) {
        (Some(_), Some(port)) if served[i] => bus.read_port(port),
        (Some(a), None) if served[i] => if reads { bus.read(a) } else { bus.peek(a, state).unwrap_or(0xFF) },
        _ => 0,
    }
}

/// A request as the shared access loop takes it: the data accesses' M-cycles, then operand
/// k's fetch at M-cycle k; each access's mask, the bus state bits it is translated under:
/// OAM DMA's always, the locks where the model's RC bit 3 (data) or WC bit 3 (operands)
/// says its accesses reach the regions they cover.
fn asked(req: &Request, at: AccessCycles) -> crate::transport::Asked {
    let fetch = |k: u8| if at.fetches >= k { k } else { 0 };
    let (data, operand) = (crate::transport::access_mask(STATE_BITS, req.at_lock), crate::transport::access_mask(STATE_BITS, req.fetch_lock));
    crate::transport::Asked { cycles: [at.reads[0], at.reads[1], at.writes[0], at.writes[1], fetch(1), fetch(2)], reads: req.reads.map(|r| r.is_some()), masks: [data, data, data, data, operand, operand] }
}

/// The three bytes at PC, peeked under the step-start state `start`; with `fetch` (a core
/// that reads the bus, on a step that fetches its opcode) the opcode is read, the step's
/// one access at its start. Operands are read at their own M-cycles, so a read here would
/// be a second access to their addresses (an FE page address corrupts OAM once per
/// access). A memory core's fetch reads no host byte: for it they feed only the audit.
fn opcode_bytes<B: Bus>(bus: &mut B, cpu: &Cpu, fetch: bool, start: u8) -> [u8; 3] {
    let mut k = 0;
    cpu.fetched(|a| {
        k += 1;
        if fetch && k == 1 { bus.read(a) } else { bus.peek(a, start).unwrap_or(0xFF) }
    })
}

/// A step's opening as the full program runs it (`transport::open_step`): the core asks
/// under the locks given, the bytes at PC peeked again under them.
struct StepOpening<'a, C: Core, B: Bus> {
    core: &'a mut C,
    cpu: &'a Cpu,
    bus: &'a mut B,
    inp: &'a mut Inputs,
}

impl<C: Core, B: Bus> crate::transport::Opening for StepOpening<'_, C, B> {
    type Request = Request;

    fn request(&mut self, locks: Option<u8>) -> Request {
        if locks != self.inp.locks {
            self.inp.locks = locks;
            self.inp.op = opcode_bytes(self.bus, self.cpu, false, crate::transport::start_state(STATE_BITS, locks, 0, self.inp.dma));
        }
        let bus = &mut *self.bus;
        request_fetching(self.core, self.cpu, self.inp, |p| bus.read_port(p))
    }

    fn fetch_reads_locks(&self, req: &Request) -> bool {
        req.fetch_lock
    }

    fn wants_locks(&self, req: &Request) -> bool {
        req.locks
    }

    fn lock_query(&mut self) -> u8 {
        self.bus.lock_query()
    }
}

/// The bus state's bits for the shared access loop (`transport::settle_accesses`).
pub(crate) const STATE_BITS: crate::transport::StateBits = crate::transport::StateBits { locks: VRAM_LOCKED | OAM_LOCKED, busy: DMA_BLOCKING, replace: OVERRIDE, shift: ACCESS_SHIFT, writes: WRITE_SLOTS, write_shift: WRITE_SHIFT };

/// A step's accesses as the full program runs them: the core asks again through
/// `request_fetching`, the control logic says the bus state ahead, and the devices tick,
/// each block's copies handed to the core.
struct StepAccess<'a, C: Core, B: Bus> {
    core: &'a mut C,
    cpu: &'a Cpu,
    bus: &'a mut B,
    inp: &'a mut Inputs,
    /// The last request made.
    req: Request,
    /// The core reads the bus itself (`StepFacts::reads_the_bus`).
    reads: bool,
}

impl<C: Core, B: Bus> crate::transport::Accesses for StepAccess<'_, C, B> {
    fn request(&mut self, access: &[u8]) -> crate::transport::Asked {
        self.inp.access.copy_from_slice(access);
        let bus = &mut *self.bus;
        self.req = request_fetching(self.core, self.cpu, self.inp, |p| bus.read_port(p));
        asked(&self.req, self.req.cycles.expect("a timed core says each access's cycle"))
    }

    fn serve(&mut self, read: usize) -> u8 {
        let &StepFacts { served_by_host: served, read_ports: ports, .. } = self.core.facts();
        serve(self.bus, &self.req, served, ports, read, self.reads, slot_state(self.inp, read))
    }

    fn ahead(&mut self, cycles: u8, locks: bool, write: bool) -> u8 {
        self.bus.ahead(cycles, locks, write)
    }

    fn tick(&mut self, cycles: u8) {
        let core = &mut *self.core;
        self.bus.tick(cycles, &mut |bytes| {
            core.copied(bytes);
            bytes.clear();
        });
    }

    fn locks(&mut self) -> Option<u8> {
        self.bus.locks()
    }

    fn busy(&mut self) -> bool {
        assert_eq!(self.bus.dma_buses() & 15, self.inp.dma_buses & 15, "an FF46 write within the step moved OAM DMA's buses before an access");
        self.bus.dma_active()
    }

    fn lock_query(&mut self) -> u8 {
        self.bus.lock_query()
    }
}
