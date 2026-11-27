//! The GBC CPU's slot layouts and the cores built on them: the circuit evaluated directly,
//! and the checkpoint run as masked diffusion.

use super::{build, build_mem};
use crate::checkpoint::{CheckpointDevice, SlotAbi};
use crate::circuit::Circuit;
use crate::engine::Engine;
use crate::llama::{tok, Denoiser, MemoryDenoiser, Model};
use crate::memory::image::Image;
use crate::memory::layout::{self, Access, Port, LINE, LINE_BITS};
use crate::sequence::{Positions, Sequence};
use crate::step::{slot_state, start_state, AccessCycles, Bus, Core, Cpu, Inputs, Outcome, Request, StepFacts};
use std::collections::BTreeMap;
use std::ops::Range;

slots! {
    /// Input slots: the CPU state, the three bytes at PC, IE and IF, then the two
    /// data bytes the host fills between passes.
    pub enum In { A, F, B, C, D, E, H, L, SPL, SPH, PCL, PCH, CTL, OP0, OP1, OP2, IE, IF, D0, D1 }
}
slots! {
    /// Input slots of the memory circuit: the CPU state, IE and IF, a DMA line's source and
    /// destination bus addresses (0 on a GBC CPU step), the OAM bug's event byte (`OB`:
    /// the row a DMG's OAM scan reads, bit 0 a read; only the OAM pass decodes what it
    /// reaches), each data access's bus state where it differs from the step's start (`AS0`: reads 0 and 1, `AS1`: writes 0 and 1, a nibble each: bit 0 VRAM locked, bit 1 OAM locked, bit 2 OAM DMA blocking, bit 3 set when the nibble replaces the step-start state), OAM DMA's buses (`DB`, `Lines::dma_buses`: the transfer's in the low nibble, those blocked at the step's start in the high), the fetches of operands 1 and 2 where their state differs from the start (`FS`, nibbles as `AS0`'s, of which only bits 2 and 3 count), then the two data bytes the host
    /// fills only when it serves them (host-served addresses), and the GBC CPU's access
    /// locks when the model asks for them (`LK`, RQ bit 7). An opcode byte the host serves
    /// reaches the model as a record in the HOST block instead.
    pub enum InMem { A, F, B, C, D, E, H, L, SPL, SPH, PCL, PCH, CTL, IE, IF, DSL, DSH, DDL, DDH, OB, AS0, AS1, DB, FS, D0, D1, LK }
}
slots! {
    /// Request slots: two read addresses and RQ, whose bits 0 and 1 enable them. The
    /// memory circuit sets bits 2 and 3 when the host serves a read and bits 4 to 6 when
    /// it serves a fetched byte. Last, the M-cycle each read (`RC`) and write (`WC`) falls
    /// in, slot 0 in bits 0-2 and slot 1 in bits 4-6. In the memory circuit, RC bit 3 says
    /// the step reaches VRAM or the FE page by a data access, and WC bit 3 that an
    /// operand fetch's line does.
    pub enum Rq { RA0L, RA0H, RA1L, RA1H, RQ, RC, WC }
}
slots! {
    /// Result slots: the next CPU state, two writes enabled by WQ's bits 0 and 1, the
    /// M-cycle count and the IF bits acknowledged.
    pub enum Out {
        A = "A'", F = "F'", B = "B'", C = "C'", D = "D'", E = "E'", H = "H'", L = "L'", SPL = "SPL'", SPH = "SPH'", PCL = "PCL'", PCH = "PCH'", CTL = "CTL'",
        WA0L, WA0H, WV0, WA1L, WA1H, WV1, WQ, MC, ACK,
    }
}
slots! {
    /// Result slots of the memory circuit: the host-mode results, then the evidence of
    /// the model's own memory traffic (the bytes it fetched, the physical addresses and
    /// bytes it read) and the write records (the physical byte address, then the whole
    /// line after the write, write 0's byte included when write 1 shares its line),
    /// enabled by WQ's bits 2 and 3; last, the physical address of each banked window's
    /// first byte under this step's registers (`MAP0`, `MAPR`, `MAPS`, `MAPW`: 0000, 4000,
    /// A000 and D000), and the physical address of each fetched byte's bus address
    /// translated as a read (`FP0`..`FP2`), which names the port of a fetched byte the
    /// host serves and which a pass decodes only when the request flags one; last, a DMA
    /// line's source and destination translated as reads (`DP`, `DQ`), which only a DMA
    /// pass decodes; and each write's second port (`WD`), the address of a second device
    /// it reaches, flagged by WQ's bits 4 and 5 and decoded only then; last, the OAM bug's
    /// rows (`OT`, bits 0 to `oam_bug::TARGET_FIRST`), each row's physical address
    /// (`OW0`..`OW4`) and the line they all receive (`OL`), which only the OAM pass decodes.
    pub enum OutMem {
        A = "A'", F = "F'", B = "B'", C = "C'", D = "D'", E = "E'", H = "H'", L = "L'", SPL = "SPL'", SPH = "SPH'", PCL = "PCL'", PCH = "PCH'", CTL = "CTL'",
        WA0L, WA0H, WV0, WA1L, WA1H, WV1, WQ, MC, ACK,
        OP0 = "OP0'", OP1 = "OP1'", OP2 = "OP2'", RP0a, RP0b, RP0c, RP1a, RP1b, RP1c, RV0, RV1,
        WP0a, WP0b, WP0c, WL00, WL01, WL02, WL03, WL04, WL05, WL06, WL07,
        WP1a, WP1b, WP1c, WL10, WL11, WL12, WL13, WL14, WL15, WL16, WL17,
        MAP0a, MAP0b, MAP0c, MAPRa, MAPRb, MAPRc, MAPSa, MAPSb, MAPSc, MAPWa, MAPWb, MAPWc,
        FP0a, FP0b, FP0c, FP1a, FP1b, FP1c, FP2a, FP2b, FP2c,
        DPa, DPb, DPc, DQa, DQb, DQc,
        WD0a, WD0b, WD0c, WD1a, WD1b, WD1c,
        OT, OW0a, OW0b, OW0c, OW1a, OW1b, OW1c, OW2a, OW2b, OW2c, OW3a, OW3b, OW3c, OW4a, OW4b, OW4c,
        OL0, OL1, OL2, OL3, OL4, OL5, OL6, OL7,
    }
}

pub const INPUTS: [&str; 20] = In::NAMES;
pub const INPUTS_MEM: [&str; 27] = InMem::NAMES;
pub const REQUEST: [&str; 7] = Rq::NAMES;
pub const RESULT: [&str; 22] = Out::NAMES;
pub const RESULT_MEM: [&str; 112] = OutMem::NAMES;

/// Input slot bytes in `INPUTS` order.
pub fn encode(cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> [u8; 20] {
    let mut s = [0; In::NAMES.len()];
    s[In::A as usize] = cpu.a;
    s[In::F as usize] = cpu.f;
    s[In::B as usize] = cpu.b;
    s[In::C as usize] = cpu.c;
    s[In::D as usize] = cpu.d;
    s[In::E as usize] = cpu.e;
    s[In::H as usize] = cpu.h;
    s[In::L as usize] = cpu.l;
    s[In::SPL as usize] = cpu.sp as u8;
    s[In::SPH as usize] = (cpu.sp >> 8) as u8;
    s[In::PCL as usize] = cpu.pc as u8;
    s[In::PCH as usize] = (cpu.pc >> 8) as u8;
    s[In::CTL as usize] = ctl_byte(cpu, false);
    s[In::OP0 as usize] = inp.op[0];
    s[In::OP1 as usize] = inp.op[1];
    s[In::OP2 as usize] = inp.op[2];
    s[In::IE as usize] = inp.ie;
    s[In::IF as usize] = inp.iflag;
    s[In::D0 as usize] = data[0];
    s[In::D1 as usize] = data[1];
    s
}

/// The flag bits of the GBC CPU's slots, named once: the decoders here and the wiring
/// table (`gbc_wiring`) read them from these.
pub mod bits {
    /// CTL and CTL': IME, EI pending, halted, the HALT bug (the state carried step to step),
    /// then CTL' bit 4 STOP; CTL bit 4 OAM DMA running, bits 5-6 the access locks and bit 7
    /// that they are present (a memory-mode step).
    pub const CTL_IME: u8 = 1;
    pub const CTL_EI: u8 = 2;
    pub const CTL_HALTED: u8 = 4;
    pub const CTL_HALT_BUG: u8 = 8;
    pub const CTL_CARRIED: u8 = CTL_IME | CTL_EI | CTL_HALTED | CTL_HALT_BUG;
    pub const CTL_STOP: u8 = 16;
    pub const CTL_DMA: u8 = 16;
    pub const CTL_LOCKS_SHIFT: u8 = 5;
    pub const CTL_LOCKS_MASK: u8 = 3;
    pub const CTL_LOCKS_PRESENT: u8 = 0x80;
    /// CTL' bits 5-7, memory mode: the opcode, operand 1 and operand 2 fetches put an FE
    /// page address on the bus unblocked by OAM DMA (`StepFacts::fe_accesses`).
    pub const CTL_FE_FETCH: [u8; 3] = [32, 64, 128];
    /// RQ: reads 0 and 1 made, each served by the host, the three fetched bytes the host
    /// serves, the locks asked for.
    pub const RQ_READ: [u8; 2] = [1, 2];
    pub const RQ_SERVED: [u8; 2] = [4, 8];
    pub const RQ_FETCH: [u8; 3] = [16, 32, 64];
    pub const RQ_LOCKS: u8 = 128;
    /// RC and WC: each access's M-cycle, slot 0 at bit 0 and slot 1 at bit 4, three bits
    /// each; RC bit 3 the locks matter at each data access, WC bit 3 at each operand fetch;
    /// RC bit 7 the step fetches an operand, WC bit 7 a second.
    pub const CYCLE_SHIFT: [u8; 2] = [0, 4];
    pub const CYCLE_MASK: u8 = 7;
    pub const RC_AT_LOCK: u8 = 8;
    pub const WC_FETCH_LOCK: u8 = 8;
    pub const FETCHED: u8 = 128;
    /// WQ: writes 0 and 1 made, kept as records, heard by a second port.
    pub const WQ_WRITE: [u8; 2] = [1, 2];
    pub const WQ_RECORD: [u8; 2] = [4, 8];
    pub const WQ_SECOND: [u8; 2] = [16, 32];
    /// WQ bits 6 and 7, memory mode: writes 0 and 1 put an FE page address on the bus
    /// unblocked by OAM DMA (a DMG's OAM corruption bug, `StepFacts::fe_accesses`).
    pub const WQ_FE: [u8; 2] = [64, 128];
    /// MC: the step's M-cycles in bits 0-2; bits 3 and 4 an FE page address on the bus in an
    /// M-cycle without an access, slot 0 and slot 1 (`Outcome::idle`), bits 5-6 that M-cycle.
    pub const MC_CYCLES: u8 = 7;
    pub const MC_IDLE: [u8; 2] = [8, 16];
    pub const MC_IDLE_SHIFT: u8 = 5;
    pub const MC_IDLE_MASK: u8 = 3;
    /// ACK: the IF bits acknowledged in bits 0-4; bits 5 and 6, memory mode: reads 0 and 1
    /// put an FE page address on the bus unblocked by OAM DMA.
    pub const ACK_IRQ: u8 = 0x1F;
    pub const ACK_FE: [u8; 2] = [32, 64];
}
use bits::*;

fn ctl_byte(cpu: &Cpu, dma: bool) -> u8 {
    CTL_IME * cpu.ime as u8 | CTL_EI * cpu.ei_pending as u8 | CTL_HALTED * cpu.halted as u8 | CTL_HALT_BUG * cpu.halt_bug as u8 | CTL_DMA * dma as u8
}

/// The late bytes of a memory-mode step: the data bytes the host serves, zero where the
/// model reads its own memory, then the access locks when the model asked for them.
pub type Late = [u8; 3];

/// CTL bits 5 to 7 of a memory-mode step: the access locks (VRAM, OAM) while the host's
/// copy of the control model's is current, and bit 7 that it is.
fn ctl_locks(locks: Option<u8>) -> u8 {
    locks.map_or(0, |l| CTL_LOCKS_PRESENT | (l & CTL_LOCKS_MASK) << CTL_LOCKS_SHIFT)
}

/// The late bytes: the data bytes of the reads the host serves, and the locks the host
/// fetched when the model asked.
fn late_bytes(flags: &HostFlags, inp: &Inputs, data: [u8; 2]) -> Late {
    let mut late: Late = [0; 3];
    for i in 0..2 {
        if flags.reads[i] {
            late[i] = data[i];
        }
    }
    if flags.locks {
        late[2] = inp.asked & 3;
    }
    late
}

/// Input slot bytes of the memory circuit in `INPUTS_MEM` order.
pub fn encode_mem(cpu: &Cpu, inp: &Inputs, late: Late) -> [u8; 27] {
    let mut s = [0; InMem::NAMES.len()];
    s[InMem::A as usize] = cpu.a;
    s[InMem::F as usize] = cpu.f;
    s[InMem::B as usize] = cpu.b;
    s[InMem::C as usize] = cpu.c;
    s[InMem::D as usize] = cpu.d;
    s[InMem::E as usize] = cpu.e;
    s[InMem::H as usize] = cpu.h;
    s[InMem::L as usize] = cpu.l;
    s[InMem::SPL as usize] = cpu.sp as u8;
    s[InMem::SPH as usize] = (cpu.sp >> 8) as u8;
    s[InMem::PCL as usize] = cpu.pc as u8;
    s[InMem::PCH as usize] = (cpu.pc >> 8) as u8;
    s[InMem::CTL as usize] = ctl_byte(cpu, inp.dma) | ctl_locks(inp.locks);
    s[InMem::IE as usize] = inp.ie;
    s[InMem::IF as usize] = inp.iflag;
    s[InMem::AS0 as usize] = inp.access[0];
    s[InMem::AS1 as usize] = inp.access[1];
    s[InMem::DB as usize] = inp.dma_buses;
    s[InMem::FS as usize] = inp.access[2];
    s[InMem::D0 as usize..=InMem::LK as usize].copy_from_slice(&late);
    s
}

/// Request slot bytes in `REQUEST` order.
pub fn decode_request(q: &[u8]) -> Request {
    let b = |s: Rq| q[s as usize];
    let w = |l: Rq, h: Rq| b(l) as u16 | (b(h) as u16) << 8;
    let pair = |s: Rq| CYCLE_SHIFT.map(|k| b(s) >> k & CYCLE_MASK);
    let cycles = AccessCycles { reads: pair(Rq::RC), writes: pair(Rq::WC), fetches: (b(Rq::RC) & FETCHED != 0) as u8 + (b(Rq::WC) & FETCHED != 0) as u8 };
    Request { reads: [(b(Rq::RQ) & RQ_READ[0] != 0).then(|| w(Rq::RA0L, Rq::RA0H)), (b(Rq::RQ) & RQ_READ[1] != 0).then(|| w(Rq::RA1L, Rq::RA1H))], locks: b(Rq::RQ) & RQ_LOCKS != 0, cycles: Some(cycles), at_lock: b(Rq::RC) & RC_AT_LOCK != 0, fetch_lock: b(Rq::WC) & WC_FETCH_LOCK != 0 }
}

/// The memory circuit's host flags in the RQ byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostFlags {
    /// The host serves read 0, read 1.
    pub reads: [bool; 2],
    /// The host serves these opcode bytes (fetched from the I/O page, IE, the unusable
    /// area or the MBC3 clock), through the HOST block.
    pub fetch: [bool; 3],
    /// The step reaches VRAM or the FE page and the host's copy of the access locks has
    /// lapsed: the host asks the control model for them.
    pub locks: bool,
}

pub fn decode_host_flags(q: &[u8]) -> HostFlags {
    let rq = q[Rq::RQ as usize];
    HostFlags { reads: RQ_SERVED.map(|m| rq & m != 0), fetch: RQ_FETCH.map(|m| rq & m != 0), locks: rq & RQ_LOCKS != 0 }
}

/// The device of each read the model asks the host for, decoded from the physical
/// address it emitted in the same pass (`layout::port`). A flag on an address that names
/// no device stops the run.
fn read_ports(flags: &HostFlags, result: &[u8]) -> [Option<Port>; 2] {
    let rp = read_addrs(result);
    std::array::from_fn(|i| flags.reads[i].then(|| layout::port(rp[i]).unwrap_or_else(|| panic!("read {i}: the model asks the host for {:#08x}, which names no device", rp[i]))))
}

/// A memory core's facts before its first step: it reads its own memory.
const MEMORY_FACTS: StepFacts = StepFacts { served_by_host: [false; 2], reads_the_bus: false, timed_accesses: true, write_targets: [Some(0); 2], ..StepFacts::BUS };

/// The facts a memory core's request settles: the reads the host serves, their devices
/// and the FE page accesses ([`decode_fe`]), from the pass's flags and result slots.
fn request_facts(facts: &mut StepFacts, flags: &HostFlags, result: &[u8]) {
    facts.served_by_host = flags.reads;
    facts.read_ports = read_ports(flags, result);
    facts.fe_accesses = decode_fe(result);
}

/// The facts a memory core's outcome settles: where the model placed its reads and
/// writes, and the second device a write reaches.
fn outcome_facts(facts: &mut StepFacts, request: &Request, e: &Evidence) {
    facts.write_targets = e.write_addr.map(Some);
    facts.second_ports = e.second.map(|p| p.map(|p| layout::port(p).unwrap_or_else(|| panic!("a second port {p:#08x} that names no device"))));
    facts.read_targets = std::array::from_fn(|i| request.reads[i].map(|_| e.read_addr[i]));
}

/// The CPU outcome in the first 22 result slots, shared by both layouts.
pub fn decode_outcome(r: &[u8]) -> Outcome {
    // One length check for every slot read below.
    let r: &[u8; Out::NAMES.len()] = r[..Out::NAMES.len()].try_into().expect("the outcome's slots");
    let b = |s: Out| r[s as usize];
    let w = |l: Out, h: Out| b(l) as u16 | (b(h) as u16) << 8;
    let (ctl, wq) = (b(Out::CTL), b(Out::WQ));
    Outcome {
        cpu: Cpu {
            a: b(Out::A),
            f: b(Out::F),
            b: b(Out::B),
            c: b(Out::C),
            d: b(Out::D),
            e: b(Out::E),
            h: b(Out::H),
            l: b(Out::L),
            sp: w(Out::SPL, Out::SPH),
            pc: w(Out::PCL, Out::PCH),
            ime: ctl & CTL_IME != 0,
            ei_pending: ctl & CTL_EI != 0,
            halted: ctl & CTL_HALTED != 0,
            halt_bug: ctl & CTL_HALT_BUG != 0,
        },
        writes: [(wq & WQ_WRITE[0] != 0).then(|| (w(Out::WA0L, Out::WA0H), b(Out::WV0))), (wq & WQ_WRITE[1] != 0).then(|| (w(Out::WA1L, Out::WA1H), b(Out::WV1)))],
        mcycles: b(Out::MC) & MC_CYCLES,
        irq_ack: b(Out::ACK) & ACK_IRQ,
        stop: ctl & CTL_STOP != 0,
        idle: MC_IDLE.map(|m| (b(Out::MC) & m != 0).then_some(b(Out::MC) >> MC_IDLE_SHIFT & MC_IDLE_MASK)),
    }
}

/// The accesses that put an FE page address on the bus unblocked by OAM DMA, in the
/// memory circuit's result slots: reads 0 and 1 (ACK bits 5 and 6), then writes 0 and 1
/// (WQ bits 6 and 7). They depend only on the accesses' addresses and bus states, so the
/// request's pass already decodes them.
pub fn decode_fe(r: &[u8]) -> [bool; 7] {
    let (ack, wq, ctl) = (r[OutMem::ACK as usize], r[OutMem::WQ as usize], r[OutMem::CTL as usize]);
    [ack & ACK_FE[0] != 0, ack & ACK_FE[1] != 0, wq & WQ_FE[0] != 0, wq & WQ_FE[1] != 0, ctl & CTL_FE_FETCH[0] != 0, ctl & CTL_FE_FETCH[1] != 0, ctl & CTL_FE_FETCH[2] != 0]
}

/// One write record the model keeps: the physical byte address written, and the whole
/// line holding it after the write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub addr: u32,
    pub line: [u8; LINE],
}

impl Record {
    /// The line's index, the cell the record replaces.
    pub fn line_index(&self) -> u32 {
        self.addr >> LINE_BITS
    }
}

/// What the memory circuit says about its own memory traffic in one step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Evidence {
    pub fetched: [u8; 3],
    pub read_addr: [u32; 2],
    pub read_byte: [u8; 2],
    pub write_addr: [u32; 2],
    pub records: [Option<Record>; 2],
    /// The physical address of each banked window's first byte under the step's registers
    /// before its writes (`layout::WINDOWS` order): the MAP slots.
    pub map: [u32; 4],
    /// Each fetched byte's physical address as a read (`FP`): meaningful only for the
    /// bytes the request flags as the host's.
    pub fetch_ports: [u32; 3],
    /// The second port each write reaches, when WQ flags one.
    pub second: [Option<u32>; 2],
    /// The accesses that put an FE page address on the bus unblocked by OAM DMA
    /// ([`decode_fe`]).
    pub fe: [bool; 7],
}

impl Evidence {
    /// The evidence with the `FP` addresses of the bytes the model fetched itself zeroed:
    /// a pass decodes those slots only when the request flags a byte as the host's.
    fn with_fetch_ports(mut self, flags: &HostFlags) -> Evidence {
        for k in 0..3 {
            if !flags.fetch[k] {
                self.fetch_ports[k] = 0;
            }
        }
        self
    }
}

/// Both reads' physical addresses (`RP`), in the memory circuit's result slots.
fn read_addrs(r: &[u8]) -> [u32; 2] {
    let r: &[u8; OutMem::NAMES.len()] = r.try_into().expect("the memory circuit's result slots");
    let addr = |a: OutMem| r[a as usize] as u32 | (r[a as usize + 1] as u32) << 8 | (r[a as usize + 2] as u32) << 16;
    [addr(OutMem::RP0a), addr(OutMem::RP1a)]
}

pub fn decode_evidence(r: &[u8]) -> Evidence {
    // One length check for every slot read below.
    let r: &[u8; OutMem::NAMES.len()] = r.try_into().expect("the memory circuit's result slots");
    let b = |s: OutMem| r[s as usize];
    let addr = |a: OutMem, m: OutMem, c: OutMem| b(a) as u32 | (b(m) as u32) << 8 | (b(c) as u32) << 16;
    let wq = b(OutMem::WQ);
    let write_addr = [addr(OutMem::WP0a, OutMem::WP0b, OutMem::WP0c), addr(OutMem::WP1a, OutMem::WP1b, OutMem::WP1c)];
    let line = |first: OutMem| -> [u8; LINE] { std::array::from_fn(|k| r[first as usize + k]) };
    Evidence {
        fetched: [b(OutMem::OP0), b(OutMem::OP1), b(OutMem::OP2)],
        read_addr: read_addrs(r),
        read_byte: [b(OutMem::RV0), b(OutMem::RV1)],
        write_addr,
        records: [
            (wq & WQ_RECORD[0] != 0).then(|| Record { addr: write_addr[0], line: line(OutMem::WL00) }),
            (wq & WQ_RECORD[1] != 0).then(|| Record { addr: write_addr[1], line: line(OutMem::WL10) }),
        ],
        map: [addr(OutMem::MAP0a, OutMem::MAP0b, OutMem::MAP0c), addr(OutMem::MAPRa, OutMem::MAPRb, OutMem::MAPRc), addr(OutMem::MAPSa, OutMem::MAPSb, OutMem::MAPSc), addr(OutMem::MAPWa, OutMem::MAPWb, OutMem::MAPWc)],
        fetch_ports: [addr(OutMem::FP0a, OutMem::FP0b, OutMem::FP0c), addr(OutMem::FP1a, OutMem::FP1b, OutMem::FP1c), addr(OutMem::FP2a, OutMem::FP2b, OutMem::FP2c)],
        second: [(wq & WQ_SECOND[0] != 0).then(|| addr(OutMem::WD0a, OutMem::WD0b, OutMem::WD0c)), (wq & WQ_SECOND[1] != 0).then(|| addr(OutMem::WD1a, OutMem::WD1b, OutMem::WD1c))],
        fe: decode_fe(r),
    }
}

/// A battery save's state, read from the model's emitted addresses alone
/// (`docs/host-in-weights.md` §4 rule 10). RAM is enabled while the A000 window the model
/// emitted (`MAPS`) lands in cartridge RAM or the clock's port; a write landed in either
/// makes the save dirty, and the window leaving both finishes it. The host applies no
/// mapper rule: the region is the address's high bits.
pub use crate::transport::Battery;

/// The window whose landing enables the saved regions: the step's last `MAPS`.
pub const SAVE_WINDOW: usize = 2;

/// The regions a battery saves: cartridge RAM and the clock's port block.
pub const SAVED: [(u32, u32); 2] = [(layout::CRAM, layout::WRAM), (layout::IO + layout::IO_CLOCK * layout::BLOCK, layout::IO + (layout::IO_CLOCK + 1) * layout::BLOCK)];

/// One step of the battery tracker: `ev` its evidence, `wrote` which writes the model enabled.
fn battery_step(b: &mut Battery, ev: &Evidence, wrote: [bool; 2]) {
    let within = |p| crate::transport::within(&SAVED, p);
    b.step(within(ev.map[SAVE_WINDOW]), (0..2).any(|i| wrote[i] && within(ev.write_addr[i])));
}

/// The circuit evaluated directly, without the checkpoint.
pub struct CircuitCore(pub Circuit);

impl CircuitCore {
    fn eval(&self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> BTreeMap<String, u8> {
        self.0.eval(&INPUTS.iter().zip(encode(cpu, inp, data)).map(|(n, b)| (n.to_string(), b)).collect())
    }
}

impl Core for CircuitCore {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        let out = self.eval(cpu, inp, [0, 0]);
        decode_request(&REQUEST.map(|n| out[n]))
    }
    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        let out = self.eval(cpu, inp, data);
        decode_outcome(&RESULT.map(|n| out[n]))
    }
}


/// A host-served fetch the request is waiting on: the physical address the model emitted
/// for each flagged byte (`FP`), and the device that address names. A byte at an address
/// that names no device (a constant cell: the unusable area, an unmapped register, VBK's
/// and SVBK's readback) is the model's own cell, which the core copies itself.
#[derive(Clone, Copy, Debug, Default)]
struct PendingFetch {
    at: [Option<u32>; 3],
    ports: [Option<Port>; 3],
}

impl PendingFetch {
    fn new(flags: &HostFlags, e: &Evidence) -> PendingFetch {
        let at: [Option<u32>; 3] = std::array::from_fn(|k| flags.fetch[k].then_some(e.fetch_ports[k]));
        PendingFetch { at, ports: at.map(|p| p.and_then(layout::port)) }
    }
    /// Each flagged byte's HOST record, `HOST | (FP & 0x7F)`, where the fetch translation
    /// lands it: the port's byte the host read, or the model's own cell.
    fn records(&self, bytes: [u8; 3], mut cell: impl FnMut(u32) -> u8) -> Vec<(u32, u8)> {
        (0..3).filter_map(|k| self.at[k].map(|p| (layout::HOST | (p & 0x7F), if self.ports[k].is_some() { bytes[k] } else { cell(p) }))).collect()
    }
}

/// The memory circuit evaluated directly over its own memory image. A step is up to
/// three evaluations: one with every late byte zero; one more after the host keeps the
/// fetched bytes it serves in the HOST block, since the request depends on them; one
/// more with the data bytes the host serves.
pub struct CircuitMemCore {
    pub circuit: Circuit,
    pub image: Image,
    pending: Option<PendingFetch>,
    flags: HostFlags,
    facts: StepFacts,
    request: Request,
    outcome: Option<Outcome>,
    late: Late,
    pub evidence: Evidence,
    /// Steps whose fetch or a read the host served, and steps in all.
    pub host_served_steps: u64,
    pub steps: u64,
}

impl CircuitMemCore {
    pub fn new(image: Image) -> CircuitMemCore {
        CircuitMemCore::with_circuit(build_mem(), image)
    }

    pub fn with_circuit(circuit: Circuit, image: Image) -> CircuitMemCore {
        CircuitMemCore { circuit, image, pending: None, flags: HostFlags::default(), facts: MEMORY_FACTS, request: Request::default(), outcome: None, late: [0; 3], evidence: Evidence::default(), host_served_steps: 0, steps: 0 }
    }

    fn eval(&self, cpu: &Cpu, inp: &Inputs, late: Late) -> BTreeMap<String, u8> {
        let image = &self.image;
        self.circuit.eval_mem(&INPUTS_MEM.iter().zip(encode_mem(cpu, inp, late)).map(|(n, b)| (n.to_string(), b)).collect(), &|n| image.line(n))
    }

    /// Applies the step's write records to the image, a line each, after asserting each
    /// equals the reference: the line as it stands with the write's byte
    /// (`layout::translate`, `layout::write_value`) put in, and write 1's line also
    /// holding write 0's byte when they share it. Every write the reference records has
    /// a record and no other does, except write 0's when write 1's record replaces the
    /// same line.
    fn commit(&mut self, e: &Evidence, out: &Outcome, inp: &Inputs) {
        let b = self.image.banking();
        let mut want: [Option<Record>; 2] = [None; 2];
        for (i, w) in out.writes.iter().enumerate() {
            want[i] = w.and_then(|(a, v)| {
                let s = slot_state(inp, 2 + i);
                let p = layout::translate_locked(&b, a, Access::Write, crate::step::slot_dma(inp, 2 + i), s);
                layout::recorded(p).then(|| {
                    let shared = want[..i].iter().flatten().find(|r| r.line_index() == p >> LINE_BITS);
                    let mut line = shared.map_or_else(|| self.image.line(p >> LINE_BITS), |r| r.line);
                    line[p as usize % LINE] = layout::write_value(&b, a, v);
                    Record { addr: p, line }
                })
            });
        }
        if let [Some(r0), Some(r1)] = want
            && r0.line_index() == r1.line_index()
        {
            want[0] = None;
        }
        for i in 0..2 {
            assert_eq!(e.records[i], want[i], "write {i} {:x?}: record", out.writes[i]);
            if let Some(r) = e.records[i] {
                self.image.write_line(r.line_index(), r.line);
            }
        }
    }
}

impl CircuitMemCore {
    /// The OAM bug at `row`: the circuit's OAM outputs over the image, each row written.
    /// They read only `OB` and the OAM cells, so every other input is 0.
    fn oam_pass(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; LINE])> {
        let mut input: BTreeMap<String, u8> = INPUTS_MEM.iter().map(|n| (n.to_string(), 0)).collect();
        input.insert(InMem::NAMES[InMem::OB as usize].to_string(), oam_event_byte(row, read));
        let image = &self.image;
        let out = self.circuit.eval_mem(&input, &|n| image.line(n));
        let lines = decode_oam(&RESULT_MEM.map(|n| out[n]));
        // The step's write records landed as it ran; one sharing a rewritten line would land
        // before the rewrite here and after it in the checkpoint (`MemCheckpointCore::oam_pass`).
        let records = self.evidence.records.iter().flatten().map(|r| r.addr);
        for &(p, line) in &lines {
            assert!(!crate::transport::staged_in_line(records.clone(), LINE_BITS as u8, p), "a rewrite of {p:#08x}: a write record of the step shares its line");
            self.image.write_line(p >> LINE_BITS, line);
        }
        lines
    }

    fn finish_request(&mut self, out: BTreeMap<String, u8>) -> Request {
        let flags = decode_host_flags(&REQUEST.map(|n| out[n]));
        self.flags = flags;
        request_facts(&mut self.facts, &flags, &RESULT_MEM.map(|n| out[n]));
        let request = decode_request(&REQUEST.map(|n| out[n]));
        self.request = request;
        self.outcome = (!flags.reads[0] && !flags.reads[1] && !flags.locks).then(|| {
            let r = RESULT_MEM.map(|n| out[n]);
            self.evidence = decode_evidence(&r).with_fetch_ports(&flags);
            decode_outcome(&r)
        });
        request
    }
}

impl Core for CircuitMemCore {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        self.late = [0; 3];
        let out = self.eval(cpu, inp, self.late);
        let flags = decode_host_flags(&REQUEST.map(|n| out[n]));
        if flags.fetch.contains(&true) {
            let pending = PendingFetch::new(&flags, &decode_evidence(&RESULT_MEM.map(|n| out[n])));
            self.pending = Some(pending);
            self.facts.fetch_ports = pending.ports;
            if pending.ports.iter().all(Option::is_none) {
                return self.serve_fetch(cpu, inp, [0; 3]);
            }
            return Request::default();
        }
        self.finish_request(out)
    }

    fn serve_fetch(&mut self, cpu: &Cpu, inp: &Inputs, bytes: [u8; 3]) -> Request {
        let pending = self.pending.take().expect("a fetch the host serves is pending");
        self.facts.fetch_ports = [None; 3];
        let image = &self.image;
        for (p, byte) in pending.records(bytes, |p| image.read(p)) {
            self.image.write(p, byte);
        }
        let out = self.eval(cpu, inp, self.late);
        let again = decode_host_flags(&REQUEST.map(|n| out[n]));
        assert_eq!(again.fetch, pending.at.map(|a| a.is_some()), "the fetch addresses depend only on PC and the registers");
        self.finish_request(out)
    }

    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        assert!(self.pending.is_none(), "a step ran before the host served its fetched bytes");
        self.steps += 1;
        if self.flags.fetch.contains(&true) || self.flags.reads[0] || self.flags.reads[1] {
            self.host_served_steps += 1;
        }
        let out = match self.outcome.take() {
            Some(o) => o,
            None => {
                self.late = late_bytes(&self.flags, inp, data);
                let out = self.eval(cpu, inp, self.late);
                let r = RESULT_MEM.map(|n| out[n]);
                self.evidence = decode_evidence(&r).with_fetch_ports(&self.flags);
                decode_outcome(&r)
            }
        };
        let e = self.evidence;
        outcome_facts(&mut self.facts, &self.request, &e);
        self.commit(&e, &out, inp);
        out
    }

    fn facts(&self) -> &StepFacts {
        &self.facts
    }

    fn oam_bug(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; LINE])> {
        self.oam_pass(row, read)
    }

    fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        self.image.revert();
        let b = self.image.banking();
        let mut cells: BTreeMap<u32, (u16, u8)> = BTreeMap::new();
        for &(a, v) in ram {
            let p = layout::translate(&b, a, Access::Read, false);
            if let Some(&(other, w)) = cells.get(&p)
                && w != v
            {
                return Err(format!("{a:#06x}={v:#04x} and {other:#06x}={w:#04x} share the cell {p:#08x}"));
            }
            if let Some(c) = layout::constant(p)
                && c != v
            {
                return Err(format!("{a:#06x}={v:#04x}: the model holds the constant {c:#04x} there ({p:#08x})"));
            }
            cells.insert(p, (a, v));
            if self.image.has(p) && p as usize >= self.image.rom_len() && layout::recorded(p) {
                self.image.write(p, v);
            }
        }
        Ok(())
    }

    fn audit(&self, bus: &impl Bus, cpu: &Cpu, inp: &Inputs, out: &Outcome) {
        audit_evidence(&self.evidence, &self.request, &self.flags, &self.facts.read_ports, bus, cpu, inp, out);
    }
}

/// Holds a memory core's evidence to the bus: the fetched bytes, each requested read's
/// physical address, the device the host served it from (its port) or the byte the model
/// served, and each write's physical address, which places it, and record. A record's
/// line is held to the host's bytes after the step: the bus bytes of the write's 8-byte
/// bus block (0 where no token sits, as at HRAM's end), with this step's writes to the
/// line put in. The registers' line holds raw bytes the
/// host keeps only interpreted, so there only the written bytes and MAPPER are held.
/// Skipped on a bus that cannot say (SingleStepTests).
fn audit_evidence(e: &Evidence, request: &Request, flags: &HostFlags, ports: &[Option<Port>; 2], bus: &impl Bus, cpu: &Cpu, inp: &Inputs, out: &Outcome) {
    let Some(banking) = bus.banking() else { return };
    // A fetched byte the host served came from the port the model named, which the next
    // check holds; the bus peek cannot read a port, so only the model's own bytes are held
    // to it.
    let bus_bytes = inp.op;
    for k in (0..3).filter(|&k| !flags.fetch[k]) {
        assert_eq!(e.fetched[k], bus_bytes[k], "fetched byte {k} at pc {:#06x} (halt bug {}) differs from the bus: {:02x?} against {bus_bytes:02x?}", cpu.pc, cpu.halt_bug, e.fetched);
    }
    for k in (0..3).filter(|&k| flags.fetch[k]) {
        let a = cpu.fetch_address(k as u16);
        assert_eq!(Some(e.fetch_ports[k]), bus.physical(a, Access::Read, start_state(inp)), "fetch {k} of {a:#06x} (halt bug {}): the address the model emitted for the host's byte", cpu.halt_bug);
    }
    // An FE page access the OAM DMA does not block: an OAM bug event (`StepFacts::fe_accesses`).
    let on_fe = |a: u16, access: Access, st: u8| a >> 8 == 0xFE && !bus.physical(a, access, st).is_some_and(layout::is_blocked);
    // The opcode is read on every step but a wake, operand k when the step fetches it.
    let fetches = request.cycles.map_or(0, |c| c.fetches) as u16;
    let fetch_fe = [0u16, 1, 2].map(|k| match k {
        0 => !cpu.halted && on_fe(cpu.pc, Access::Read, start_state(inp)),
        k => k <= fetches && on_fe(cpu.fetch_address(k), Access::Read, slot_state(inp, 3 + k as usize)),
    });
    let want_fe = [0, 1].map(|i| request.reads[i].is_some_and(|a| on_fe(a, Access::Read, slot_state(inp, i)))).into_iter().chain([0, 1].map(|i| out.writes[i].is_some_and(|(a, _)| on_fe(a, Access::Write, slot_state(inp, 2 + i))))).chain(fetch_fe).collect::<Vec<bool>>();
    assert_eq!(e.fe[..], want_fe[..], "the FE page accesses (reads 0 and 1, writes 0 and 1, fetches 0 to 2) the model emitted, at pc {:#06x}", cpu.pc);
    for i in 0..2 {
        if let Some(a) = request.reads[i] {
            let st = slot_state(inp, i);
            assert_eq!(Some(e.read_addr[i]), bus.physical(a, Access::Read, st), "read {i} of {a:#06x}: physical address, under the bus state {st:#x}");
            let want = bus.physical(a, Access::Read, st).and_then(layout::port);
            assert_eq!(ports[i], want, "read {i} of {a:#06x}: the host served it from {:?}, the translation names {want:?}", ports[i]);
            if !flags.reads[i] {
                assert_eq!(Some(e.read_byte[i]), bus.peek(a, st), "read {i} of {a:#06x}: byte");
            }
        }
        let Some((a, v)) = out.writes[i] else {
            assert_eq!(e.records[i], None, "write {i}: record without a write");
            continue;
        };
        let st = slot_state(inp, 2 + i);
        let wp = bus.physical(a, Access::Write, st).unwrap();
        assert_eq!(e.write_addr[i], wp, "write {i} to {a:#06x}: physical address, under the bus state {st:#x}");
        assert_eq!(e.second[i], layout::second_port(wp, banking.has_clock()), "write {i} to {a:#06x}: the second device's port");
        let Some(r) = e.records[i] else {
            // Write 1's record replaces write 0's line when they share it.
            let superseded = i == 0 && out.writes[1].is_some_and(|(a1, _)| bus.physical(a1, Access::Write, slot_state(inp, 3)).is_some_and(|p1| p1 >> LINE_BITS == wp >> LINE_BITS) && e.records[1].is_some());
            assert!(!layout::recorded(wp) || superseded, "write {i} to {a:#06x} = {v:#04x}: no record");
            continue;
        };
        assert!(layout::recorded(wp) && r.addr == wp, "write {i} to {a:#06x} = {v:#04x}: record at {:#08x}", r.addr);
        // A register line's bytes the host knows: the ones written at load only.
        let regs = (wp >> LINE_BITS).checked_sub(layout::REGS >> LINE_BITS).filter(|&l| (l as usize) < layout::N_REGS.div_ceil(LINE));
        let mut want: [Option<u8>; LINE] = if let Some(l) = regs {
            use layout::Reg;
            std::array::from_fn(|k| [Reg::Mapper, Reg::Mask0, Reg::RamMask, Reg::Booting].into_iter().find(|&r| r as usize == l as usize * LINE + k).map(|r| banking.get(r)))
        } else {
            std::array::from_fn(|k| {
                let ak = a & !(LINE as u16 - 1) | k as u16;
                Some(if bus.physical(ak, Access::Write, st) == Some(wp & !(LINE as u32 - 1) | k as u32) { bus.peek(ak, st).unwrap() } else { 0 })
            })
        };
        for (j, (aj, vj)) in out.writes[..=i].iter().enumerate().filter_map(|(j, w)| Some((j, w.as_ref()?))) {
            let pj = bus.physical(*aj, Access::Write, slot_state(inp, 2 + j)).unwrap();
            if pj >> LINE_BITS == wp >> LINE_BITS {
                want[pj as usize % LINE] = Some(layout::write_value(&banking, *aj, *vj));
            }
        }
        for k in 0..LINE {
            if let Some(w) = want[k] {
                assert_eq!(r.line[k], w, "write {i} to {a:#06x} = {v:#04x}: byte {k} of the record's line {:02x?}", r.line);
            }
        }
    }
}

/// The GBC CPU slot ABI: `In` early inputs with D0/D1 late, `Rq` request slots and `Out`
/// result slots.
pub struct Sm83;

impl SlotAbi for Sm83 {
    const INPUTS: &'static [&'static str] = &INPUTS;
    const REQUEST: &'static [&'static str] = &REQUEST;
    const RESULT: &'static [&'static str] = &RESULT;
    const LATE: Range<usize> = In::D0 as usize..In::NAMES.len();
    type State = Cpu;
    type Inputs = Inputs;
    type Late = [u8; 2];
    type Request = Request;
    type Outcome = Outcome;

    fn encode(cpu: &Cpu, inp: &Inputs, slots: &mut [u8]) {
        slots.copy_from_slice(&encode(cpu, inp, [0, 0]));
    }
    fn decode_request(bytes: &[u8]) -> Request {
        decode_request(bytes)
    }
    fn decode_outcome(bytes: &[u8]) -> Outcome {
        decode_outcome(bytes)
    }
    fn needs_late(request: &Request) -> bool {
        request.reads.iter().any(Option::is_some)
    }
}

/// The GBC CPU run as masked diffusion over a checkpoint.
pub type CheckpointCore<M> = CheckpointDevice<Sm83, M>;

impl<M: Denoiser> Core for CheckpointCore<M> {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        CheckpointDevice::request(self, cpu, inp)
    }
    fn execute(&mut self, _cpu: &Cpu, _inp: &Inputs, data: [u8; 2]) -> Outcome {
        CheckpointDevice::execute(self, &data)
    }
}

/// The GBC CPU circuit compiled into its Llama checkpoint and loaded into the fast engine.
pub fn engine() -> Engine {
    Engine::new(Model::compile(&build()))
}

/// The CPU checkpoint as a core.
pub fn checkpoint() -> CheckpointCore<Engine> {
    CheckpointCore::new(engine())
}

/// The memory checkpoint run as masked diffusion over a [`MemoryDenoiser`]: the same
/// protocol as [`CircuitMemCore`], with each evaluation a fill of the masked positions,
/// and the write records kept in the cache after the step.
pub struct MemCheckpointCore<M: MemoryDenoiser> {
    model: M,
    cfg: crate::llama::Config,
    /// The sequence every pass reads: the inputs, and at every output slot its mask
    /// token, but for the request's decoded tokens during a result pass. A pass's decoded
    /// tokens stay in the sequence's store.
    tokens: Sequence,
    /// The positions a request pass, a result pass and a fetch-port pass decode, built
    /// once. Only the last decodes the `FP` slots, which matter only when the request
    /// flags a fetched byte as the host's.
    passes: [Positions; 6],
    pending: Option<PendingFetch>,
    /// The byte each request and result slot (`N_IN` on) last decoded to.
    bytes: Vec<u8>,
    /// The sequence with the decoded tokens of pass `step_pass` in place (`finished`).
    step: Vec<usize>,
    step_pass: usize,
    byte_of: tok::Bytes,
    flags: HostFlags,
    facts: StepFacts,
    request: Request,
    outcome: Option<Outcome>,
    pub evidence: Evidence,
    pub host_served_steps: u64,
    pub steps: u64,
    pub records_kept: u64,
    /// DMA lines placed by a pass of their own (`dma_line`).
    pub dma_passes: u64,
    /// Writes a second device heard, each a pass decoding its `WD` port.
    pub second_passes: u64,
    /// OAM bug events, each a pass decoding the rows it rewrites (`oam_pass`).
    pub oam_passes: u64,
    /// Passes that decode the result after the host's late bytes (`execute`'s second).
    pub result_passes: u64,
    /// Steps asked again under each access's own bus state (`Inputs::access`).
    pub access_overrides: u64,
    /// The step's write records not kept yet, each kept at its write's M-cycle
    /// (`Core::land_write`), and the step's tokens they read when a later pass of the step
    /// overwrote them (empty otherwise).
    staged: [bool; 2],
    staged_tokens: Vec<usize>,
    pub battery: Battery,
}

const N_IN: usize = InMem::NAMES.len();
const N_RQ: usize = Rq::NAMES.len();

/// The result slots only the audit reads: the fetched bytes and both bytes read.
pub const AUDIT_SLOTS: [OutMem; 5] = [OutMem::OP0, OutMem::OP1, OutMem::OP2, OutMem::RV0, OutMem::RV1];

/// The memory core's passes as output slots, counted from the first request slot (`REQUEST`
/// then `RESULT_MEM`): the step pass (request and result), the result pass, the fetch
/// pass, the copy line pass, the second-port pass and the OAM bug's pass. The first two
/// leave out the other four's slots, and with `audit` false also `AUDIT_SLOTS`.
/// `MemCheckpointCore` and the wiring table both take them from here.
pub fn mem_passes(audit: bool) -> [Vec<usize>; 6] {
    let out = |o: usize| N_RQ + o;
    let fp: Vec<usize> = (OutMem::FP0a as usize..=OutMem::FP2c as usize).map(out).collect();
    let dma: Vec<usize> = (OutMem::DPa as usize..=OutMem::DQc as usize).map(out).collect();
    let wd: Vec<usize> = (OutMem::WD0a as usize..=OutMem::WD1c as usize).map(out).collect();
    let oam: Vec<usize> = (OutMem::OT as usize..=OutMem::OL7 as usize).map(out).collect();
    let unread: Vec<usize> = if audit { Vec::new() } else { AUDIT_SLOTS.iter().map(|&o| out(o as usize)).collect() };
    let n = N_RQ + RESULT_MEM.len();
    let keep = |p: &usize| !fp.contains(p) && !dma.contains(p) && !wd.contains(p) && !oam.contains(p) && !unread.contains(p);
    [(0..n).filter(keep).collect(), (N_RQ..n).filter(keep).collect(), fp.clone(), dma.clone(), wd.clone(), oam.clone()]
}

/// The rows an OAM pass's bytes (`RESULT_MEM` order) rewrite: each target `OT` flags, its
/// physical address and the line, in target order, which is the reference's write order
/// (`ppu::corrupt_oam`).
pub fn decode_oam(r: &[u8]) -> Vec<(u32, [u8; LINE])> {
    let addr = |a: usize| r[a] as u32 | (r[a + 1] as u32) << 8 | (r[a + 2] as u32) << 16;
    let line: [u8; LINE] = std::array::from_fn(|k| r[OutMem::OL0 as usize + k]);
    (0..5).filter(|t| r[OutMem::OT as usize] >> t & 1 != 0).map(|t| (addr(OutMem::OW0a as usize + 3 * t), line)).collect()
}

/// The event byte the OAM pass reads (`OB`): the scanned row, bit 0 set for a read.
pub fn oam_event_byte(row: u8, read: bool) -> u8 {
    assert!(row % 8 == 0, "an OAM row {row:#04x} off the 8-byte grid");
    row | read as u8
}

impl<M: MemoryDenoiser> MemCheckpointCore<M> {
    pub fn new(model: M) -> Self {
        let expected: Vec<&str> = INPUTS_MEM.iter().chain(&REQUEST).chain(&RESULT_MEM).copied().collect();
        assert_eq!(model.slots(), expected, "the checkpoint's slots are not the memory ABI");
        let cfg = model.cfg().clone();
        assert_eq!(cfg.seq(), N_IN + N_RQ + RESULT_MEM.len());
        let passes = mem_passes(true).map(|ps| Positions::new(ps.into_iter().map(|p| N_IN + p).collect()));
        let mut tokens = vec![0; cfg.seq()];
        for j in InMem::DSL as usize..=InMem::OB as usize {
            tokens[j] = tok::input(&cfg, j, 0);
        }
        for p in N_IN..cfg.seq() {
            tokens[p] = tok::output_mask(&cfg, p);
        }
        MemCheckpointCore { tokens: Sequence::new(tokens), passes, pending: None, step: Vec::new(), step_pass: 0, bytes: vec![0; cfg.seq() - N_IN], byte_of: tok::Bytes::new(&cfg), cfg, model, flags: HostFlags::default(), facts: MEMORY_FACTS, request: Request::default(), outcome: None, evidence: Evidence::default(), host_served_steps: 0, steps: 0, records_kept: 0, dma_passes: 0, second_passes: 0, oam_passes: 0, result_passes: 0, access_overrides: 0, staged: [false; 2], staged_tokens: Vec::new(), battery: Battery::default() }
    }

    /// Leaves out of every pass the result slots that only the audit reads: the fetched
    /// bytes and both bytes read (`OP0' OP1' OP2' RV0 RV1`), which hold the mask token and
    /// decode to 0 in the evidence. The run keeps every slot the host acts on: the outcome,
    /// the request, both reads' physical addresses (a host-served read's port), the write
    /// records and the MAP slots. A core whose evidence is audited (memory lockstep) or compared
    /// (the dense test) decodes every slot.
    pub fn without_audit_slots(mut self) -> Self {
        let unread = AUDIT_SLOTS.map(|o| N_IN + N_RQ + o as usize);
        for pass in &mut self.passes {
            *pass = Positions::new(pass.iter().copied().filter(|p| !unread.contains(p)).collect());
        }
        self
    }

    /// A write record of the step that is not kept yet falls in the 8-byte line holding
    /// physical address `addr`.
    pub fn staged_in_line(&self, addr: u32) -> bool {
        crate::transport::staged_in_line((0..2).filter(|&i| self.staged[i]).filter_map(|i| self.evidence.records[i].map(|r| r.addr)), LINE_BITS as u8, addr)
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    fn set_inputs(&mut self, cpu: &Cpu, inp: &Inputs, late: Late) {
        let slots = encode_mem(cpu, inp, late);
        // The DMA line inputs keep the last line `dma_line` set, and `OB` the last OAM bug
        // event: they reach only DP and DQ, and the OAM pass's slots
        // (`dma_inputs_reach_only_the_dma_slots`), which a GBC CPU pass does not decode.
        let dma = InMem::DSL as usize..=InMem::OB as usize;
        for j in (0..N_IN).filter(|j| !dma.contains(j)) {
            let is_late = j >= InMem::D0 as usize;
            self.tokens.set(j, if is_late && late[j - InMem::D0 as usize] == 0 { tok::input_mask(&self.cfg, j) } else { tok::input(&self.cfg, j, slots[j]) });
        }
    }

    /// Decodes the positions of pass `k` (0: request and result, 1: result), leaving
    /// their bytes in `self.bytes`. A position the pass did not decode again
    /// (`Sequence::fresh`) keeps the byte of its last decode.
    fn decode(&mut self, k: usize) {
        self.model.fill_seq(&mut self.tokens, &self.passes[k]);
        let (decoded, bytes, byte_of) = (self.tokens.decoded(), &mut self.bytes[..], self.byte_of);
        for &p in self.tokens.fresh() {
            let p = p as usize;
            let Some(v) = byte_of.of(decoded[p]) else { panic!("position {p} ({}) decoded non-byte token {}", self.model.slots()[p], decoded[p]) };
            bytes[p - N_IN] = v;
        }
    }

    /// The result pass. Its sequence holds the request's decoded tokens at their slots, as
    /// the request pass left them; outputs read no output, so they change no token it
    /// decodes.
    fn decode_result(&mut self) {
        for p in N_IN..N_IN + N_RQ {
            self.tokens.set(p, self.tokens.decoded()[p]);
        }
        self.decode(1);
        for p in N_IN..N_IN + N_RQ {
            self.tokens.set(p, tok::output_mask(&self.cfg, p));
        }
    }

    /// Builds in `self.step` the sequence as it stands after pass `k`: the pass's tokens,
    /// with each position it decoded holding its token. A write record's slots are read
    /// from it.
    fn finished(&mut self, k: usize) {
        let (tokens, decoded) = (self.tokens.tokens(), self.tokens.decoded());
        // Every output of the sequence holds its mask here, as `step` does outside the
        // last pass built: only the inputs and the two passes' positions are written.
        if self.step.is_empty() {
            self.step.extend_from_slice(tokens);
        } else {
            self.step[..N_IN].copy_from_slice(&tokens[..N_IN]);
            if self.step_pass != k {
                for &p in self.passes[self.step_pass].iter() {
                    self.step[p] = tokens[p];
                }
            }
        }
        for &p in self.passes[k].iter() {
            self.step[p] = decoded[p];
        }
        self.step_pass = k;
    }

    /// A DMA line's placement, decoded by the model: the physical addresses of the 8-byte
    /// lines holding bus addresses `source` and `dest` (their low three bits dropped),
    /// translated as reads under the registers after the last step. One pass decodes only
    /// the `DP` and `DQ` slots; the DMA inputs keep the line.
    pub fn dma_line(&mut self, source: u16, dest: u16) -> [u32; 2] {
        let lines = [source & !7, dest & !7];
        // A DMA mid-step (a tick before an access) decodes none of the slots the step's
        // records read at `execute`.
        let inputs = [(InMem::DSL, lines[0] as u8), (InMem::DSH, (lines[0] >> 8) as u8), (InMem::DDL, lines[1] as u8), (InMem::DDH, (lines[1] >> 8) as u8)];
        for (slot, byte) in inputs {
            self.tokens.set(slot as usize, tok::input(&self.cfg, slot as usize, byte));
        }
        self.decode(3);
        let b = |s: OutMem| self.bytes[N_RQ + s as usize] as u32;
        let at = [b(OutMem::DPa) | b(OutMem::DPb) << 8 | b(OutMem::DPc) << 16, b(OutMem::DQa) | b(OutMem::DQb) << 8 | b(OutMem::DQc) << 16];
        // The inputs stay at this line: they reach only DP and DQ
        // (`dma_inputs_reach_only_the_dma_slots`), so the passes between re-evaluate nothing.
        self.dma_passes += 1;
        at
    }

    /// A DMG's OAM corruption bug at the scanned `row`: one pass decodes the rows it
    /// rewrites and their line from the OAM cells (`OT`, `OW`, `OL`), each kept as a write
    /// record. It decodes none of the slots the step's own records read.
    fn oam_pass(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; LINE])> {
        // The event byte is the pass's alone: the step's inputs stand as they were after it.
        let ob = self.tokens.get(InMem::OB as usize);
        self.tokens.set(InMem::OB as usize, tok::input(&self.cfg, InMem::OB as usize, oam_event_byte(row, read)));
        self.model.check_next();
        self.decode(5);
        let lines = decode_oam(&self.bytes[N_RQ..]);
        // A write record of the step not landed yet would land over the rewritten line (the
        // runner's `StepStream::rewrite` holds the same).
        for &(p, _) in &lines {
            assert!(!self.staged_in_line(p), "a rewrite of {p:#08x}: a write record of the step not landed yet shares its line");
        }
        let writes: Vec<u8> = (0..5u8).filter(|t| self.bytes[N_RQ + OutMem::OT as usize] >> t & 1 != 0).map(|t| tok::OAM_RECORDS + t).collect();
        self.finished(5);
        self.model.keep(&self.step, &writes);
        self.records_kept += writes.len() as u64;
        self.oam_passes += 1;
        self.tokens.set(InMem::OB as usize, ob);
        lines
    }

    fn settle(&mut self, bytes: &[u8]) -> Outcome {
        self.evidence = decode_evidence(bytes).with_fetch_ports(&self.flags);
        decode_outcome(bytes)
    }
}

impl<M: MemoryDenoiser> MemCheckpointCore<M> {
    /// The request of the pass just decoded, whose fetched bytes are all the model's.
    fn finish_request(&mut self) -> Request {
        let flags = decode_host_flags(&self.bytes[..N_RQ]);
        self.flags = flags;
        request_facts(&mut self.facts, &flags, &self.bytes[N_RQ..]);
        self.request = decode_request(&self.bytes[..N_RQ]);
        self.outcome = (!flags.reads[0] && !flags.reads[1] && !flags.locks).then(|| {
            let bytes = std::mem::take(&mut self.bytes);
            let out = self.settle(&bytes[N_RQ..]);
            self.bytes = bytes;
            out
        });
        self.request
    }
}

impl<M: MemoryDenoiser> MemCheckpointCore<M> {
    /// The step's outcome, its write records staged for [`Core::land_write`].
    fn step_staged(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        assert!(self.pending.is_none(), "a step ran before the host served its fetched bytes");
        self.steps += 1;
        if self.flags.fetch.contains(&true) || self.flags.reads[0] || self.flags.reads[1] {
            self.host_served_steps += 1;
        }
        let out = match self.outcome.take() {
            Some(o) => o,
            None => {
                let late = late_bytes(&self.flags, inp, data);
                self.set_inputs(cpu, inp, late);
                self.decode_result();
                self.result_passes += 1;
                let bytes = std::mem::take(&mut self.bytes);
                let out = self.settle(&bytes[N_RQ..]);
                self.bytes = bytes;
                out
            }
        };
        // The write records wait for their writes' M-cycles, write 0's before write 1's.
        self.staged = self.evidence.records.map(|r| r.is_some());
        // A write a second device hears: one more pass decodes that device's port. The
        // records read the step's tokens as they stand before it.
        if self.evidence.second.iter().any(Option::is_some) {
            if self.staged.contains(&true) {
                self.finished(0);
                self.staged_tokens.clone_from(&self.step);
            }
            self.decode(4);
            self.second_passes += 1;
            self.evidence.second = decode_evidence(&self.bytes[N_RQ..]).second;
        }
        outcome_facts(&mut self.facts, &self.request, &self.evidence);
        battery_step(&mut self.battery, &self.evidence, out.writes.map(|w| w.is_some()));
        out
    }
}

impl<M: MemoryDenoiser> Core for MemCheckpointCore<M> {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        assert_eq!(self.staged, [false; 2], "a step ran before the last one's write records were kept");
        self.access_overrides += (inp.access != [0; 3]) as u64;
        self.set_inputs(cpu, inp, [0; 3]);
        self.decode(0);
        let flags = decode_host_flags(&self.bytes[..N_RQ]);
        if flags.fetch.contains(&true) {
            self.decode(2);
            let pending = PendingFetch::new(&flags, &decode_evidence(&self.bytes[N_RQ..]));
            self.pending = Some(pending);
            self.facts.fetch_ports = pending.ports;
            if pending.ports.iter().all(Option::is_none) {
                return self.serve_fetch(cpu, inp, [0; 3]);
            }
            return Request::default();
        }
        self.finish_request()
    }

    fn serve_fetch(&mut self, _cpu: &Cpu, _inp: &Inputs, bytes: [u8; 3]) -> Request {
        let pending = self.pending.take().expect("a fetch the host serves is pending");
        self.facts.fetch_ports = [None; 3];
        let model = &mut self.model;
        for (p, byte) in pending.records(bytes, |p| model.cell(p)) {
            self.model.keep_host(p, byte);
        }
        self.decode(0);
        let again = decode_host_flags(&self.bytes[..N_RQ]);
        assert_eq!(again.fetch, pending.at.map(|a| a.is_some()), "the fetch addresses depend only on PC and the registers");
        self.finish_request()
    }

    /// Keeps every write record of the step at once, for a caller that runs no bus.
    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        let out = self.step_staged(cpu, inp, data);
        for i in 0..2 {
            self.land_write(i);
        }
        out
    }

    fn execute_on(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2], _bus: &impl Bus) -> Outcome {
        self.step_staged(cpu, inp, data)
    }

    /// Each record replaces its line's cell with the line its WL positions decoded.
    fn land_write(&mut self, i: usize) {
        if !std::mem::take(&mut self.staged[i]) {
            return;
        }
        if self.staged_tokens.is_empty() {
            self.finished(0);
        }
        let tokens = if self.staged_tokens.is_empty() { &self.step } else { &self.staged_tokens };
        self.model.keep(tokens, &[i as u8]);
        self.records_kept += 1;
        if self.staged == [false; 2] {
            self.staged_tokens.clear();
        }
    }

    fn facts(&self) -> &StepFacts {
        &self.facts
    }

    fn oam_bug(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; LINE])> {
        self.oam_pass(row, read)
    }

    fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        self.model.load_bus(ram)
    }

    fn audit(&self, bus: &impl Bus, cpu: &Cpu, inp: &Inputs, out: &Outcome) {
        audit_evidence(&self.evidence, &self.request, &self.flags, &self.facts.read_ports, bus, cpu, inp, out);
    }
}

#[cfg(test)]
mod fetch_port_audit {
    use super::*;
    use crate::memory::layout::{Banking, Mapper};

    struct Flat(Banking);

    impl Bus for Flat {
        fn read(&mut self, _addr: u16) -> u8 {
            0
        }
        fn write(&mut self, _addr: u16, _value: u8) {}
        fn ie(&self) -> u8 {
            0
        }
        fn iflag(&self) -> u8 {
            0
        }
        fn ack(&mut self, _bits: u8) {}
        fn banking(&self) -> Option<Banking> {
            Some(self.0.clone())
        }
        fn physical(&self, addr: u16, access: Access, _state: u8) -> Option<u32> {
            Some(layout::translate(&self.0, addr, access, false))
        }
    }

    /// The audit holds an emitted fetch port to the specification: a port code flipped
    /// from control's (TMA) to the APU's stops the run.
    #[test]
    #[should_panic(expected = "the address the model emitted for the host's byte")]
    fn a_wrong_fetch_port_stops_the_audit() {
        let bus = Flat(Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap());
        let cpu = Cpu { pc: 0xFF06, ..Cpu::default() };
        let inp = Inputs::default();
        let right = layout::translate(&bus.0, 0xFF06, Access::Read, false);
        assert_eq!(layout::port(right), Some(Port::Control(0xFF06)));
        let e = Evidence { fetch_ports: [right ^ (layout::IO_CONTROL ^ layout::IO_APU) << 7, 0, 0], ..Evidence::default() };
        let flags = HostFlags { reads: [false; 2], fetch: [true, false, false], locks: false };
        audit_evidence(&e, &Request::default(), &flags, &[None; 2], &bus, &cpu, &inp, &Outcome::default());
    }
}
