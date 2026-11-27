//! Reference SM83 interpreter: the lockstep oracle for the compiled model.
//!
//! [`Core::request`] returns the data-memory reads a step performs, in bus order,
//! beyond the instruction bytes in `inp.op`. POP/RET read `[sp, sp+1]`,
//! `LD A,(a16)` reads `[a16]`, `INC (HL)` reads `[hl]`, `LDH A,(C)` reads `[0xFF00+c]`.
//! It runs the same code as `execute` with zeroed data and records the addresses,
//! which is exact because no SM83 read address depends on data read in the same step.
//!
//! [`Core::execute`], with `pending = ie & iflag & 0x1F`:
//! 1. Halted (only called with `pending != 0`): wake, no M-cycle of its own, nothing else.
//!    The control logic says when the wake comes (`Lines::wake`), a CGB one M-cycle after
//!    IE & IF, a DMG in the M-cycle it rose unless only its late bits are pending
//!    (`ControlState::waking`, SameBoy `GB_cpu_run`).
//! 2. `ime && pending != 0`: dispatch the lowest pending bit n. Push pc (high byte to
//!    sp-1, then low to sp-2), jump to 0x40 + 8n, clear IME and the EI delay,
//!    acknowledge bit n, 5 M-cycles. With SP 0 the high byte's push writes IE first, and
//!    n is the lowest bit pending under that IE; with none, jump to 0000 and acknowledge
//!    nothing.
//! 3. Otherwise run one instruction from the fetched bytes `op` (0xCB uses `op[1]`).
//!    With the HALT bug set they are read at pc, pc and pc+1 (`Cpu::fetch_address`)
//!    and pc advances one less. EI arms IME for after
//!    the next instruction: IME turns on after the dispatch check of the following step
//!    but before its instruction runs, so DI still wins. RETI sets
//!    IME immediately. HALT with an interrupt pending does not halt: with IME off it sets
//!    the HALT bug, with IME on (EI's delay included) PC stays on the HALT. Otherwise
//!    HALT halts. STOP is one byte. Illegal opcodes change nothing and take
//!    1 M-cycle, since the CPU hangs there.

use crate::step::{AccessCycles, Core, Cpu, Inputs, Outcome, Request};

pub struct Reference;

const FZ: u8 = 0x80;
const FN: u8 = 0x40;
const FH: u8 = 0x20;
const FC: u8 = 0x10;

impl Core for Reference {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        let mut x = Exec::new(cpu, inp, [0, 0]);
        if !cpu.halted && !(cpu.ime && pending(inp) != 0) {
            x.instruction();
        }
        Request { reads: x.reads, locks: false, cycles: Some(run(cpu, inp, [0, 0]).0.at), at_lock: false, fetch_lock: false }
    }

    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        run(cpu, inp, data).1
    }
}

/// One step: the access cycles it ran and its outcome.
fn run(cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> (Exec, Outcome) {
    let mut x = Exec::new(cpu, inp, data);
    let (mut irq_ack, mut stop) = (0, false);
    if cpu.halted {
        x.c.halted = false;
    } else if cpu.ime && pending(inp) != 0 {
        // SameBoy's dispatch puts PC + 1, then SP, on the bus in its second M-cycle.
        x.idle();
        x.idle_on(cpu.pc.wrapping_add(1));
        x.idle[1] = Some((1, cpu.sp));
        x.push(cpu.pc);
        x.idle();
        // The vector is chosen after the high byte's push, which writes IE when SP is 0:
        // the bits it leaves pick the vector, and none leaves 0000 with nothing
        // acknowledged (mooneye `interrupts/ie_push`).
        let ie = if cpu.sp == 0 { (cpu.pc >> 8) as u8 } else { inp.ie };
        let left = ie & inp.iflag & 0x1F;
        x.c.pc = if left == 0 { 0 } else { 0x40 + 8 * left.trailing_zeros() as u16 };
        x.c.ime = false;
        x.c.ei_pending = false;
        irq_ack = left & left.wrapping_neg();
    } else {
        stop = inp.op[0] == 0x10;
        x.instruction();
        // The operand bytes after the opcode, each fetched at its own M-cycle 1, 2.
        x.at.fetches = x.next.saturating_sub(1) as u8;
    }
    let idle = x.idle.map(|e| e.and_then(|(m, addr)| (addr >> 8 == 0xFE).then_some(m)));
    let out = Outcome { cpu: x.c, writes: x.writes, mcycles: x.mcycles, irq_ack, stop, idle };
    (x, out)
}

fn pending(inp: &Inputs) -> u8 {
    inp.ie & inp.iflag & 0x1F
}

fn pair(hi: u8, lo: u8) -> u16 {
    u16::from_be_bytes([hi, lo])
}

/// One step in progress. Every bus access and internal cycle bumps `mcycles`.
struct Exec {
    c: Cpu,
    /// PC at the step's start.
    pc0: u16,
    pending: u8,
    stream: [u8; 3],
    next: usize,
    data: [u8; 2],
    reads: [Option<u16>; 2],
    writes: [Option<(u16, u8)>; 2],
    mcycles: u8,
    /// The M-cycle each read and write fell in.
    at: AccessCycles,
    /// `Outcome::idle`.
    idle: [Option<(u8, u16)>; 2],
}

impl Exec {
    fn new(cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Self {
        Exec { c: *cpu, pc0: cpu.pc, pending: pending(inp), stream: inp.op, next: 0, data, reads: [None; 2], writes: [None; 2], mcycles: 0, at: AccessCycles::default(), idle: [None; 2] }
    }

    fn fetch(&mut self) -> u8 {
        let b = self.stream[self.next];
        self.next += 1;
        self.c.pc = self.c.pc.wrapping_add(1);
        self.mcycles += 1;
        b
    }

    fn fetch16(&mut self) -> u16 {
        let lo = self.fetch();
        pair(self.fetch(), lo)
    }

    fn read(&mut self, addr: u16) -> u8 {
        let i = self.reads.iter().position(Option::is_none).expect("more than two data reads");
        self.reads[i] = Some(addr);
        self.at.reads[i] = self.mcycles;
        self.mcycles += 1;
        self.data[i]
    }

    fn write(&mut self, addr: u16, value: u8) {
        let i = self.writes.iter().position(Option::is_none).expect("more than two writes");
        self.writes[i] = Some((addr, value));
        self.at.writes[i] = self.mcycles;
        self.mcycles += 1;
    }

    fn idle(&mut self) {
        self.mcycles += 1;
    }

    /// An M-cycle without an access that puts `addr` on the bus (`Outcome::idle`).
    fn idle_on(&mut self, addr: u16) {
        self.idle[0] = Some((self.mcycles, addr));
        self.idle();
    }

    fn push(&mut self, v: u16) {
        let [hi, lo] = v.to_be_bytes();
        self.c.sp = self.c.sp.wrapping_sub(1);
        self.write(self.c.sp, hi);
        self.c.sp = self.c.sp.wrapping_sub(1);
        self.write(self.c.sp, lo);
    }

    fn pop(&mut self) -> u16 {
        let lo = self.read(self.c.sp);
        self.c.sp = self.c.sp.wrapping_add(1);
        let hi = self.read(self.c.sp);
        self.c.sp = self.c.sp.wrapping_add(1);
        pair(hi, lo)
    }

    fn hl(&self) -> u16 {
        pair(self.c.h, self.c.l)
    }

    fn set_hl(&mut self, v: u16) {
        [self.c.h, self.c.l] = v.to_be_bytes();
    }

    /// Registers B C D E H L (HL) A.
    fn reg(&mut self, r: u8) -> u8 {
        match r {
            0 => self.c.b,
            1 => self.c.c,
            2 => self.c.d,
            3 => self.c.e,
            4 => self.c.h,
            5 => self.c.l,
            6 => self.read(self.hl()),
            _ => self.c.a,
        }
    }

    fn set_reg(&mut self, r: u8, v: u8) {
        match r {
            0 => self.c.b = v,
            1 => self.c.c = v,
            2 => self.c.d = v,
            3 => self.c.e = v,
            4 => self.c.h = v,
            5 => self.c.l = v,
            6 => self.write(self.hl(), v),
            _ => self.c.a = v,
        }
    }

    /// Register pairs BC DE HL SP.
    fn rp(&self, p: u8) -> u16 {
        match p {
            0 => pair(self.c.b, self.c.c),
            1 => pair(self.c.d, self.c.e),
            2 => self.hl(),
            _ => self.c.sp,
        }
    }

    fn set_rp(&mut self, p: u8, v: u16) {
        let [hi, lo] = v.to_be_bytes();
        match p {
            0 => (self.c.b, self.c.c) = (hi, lo),
            1 => (self.c.d, self.c.e) = (hi, lo),
            2 => (self.c.h, self.c.l) = (hi, lo),
            _ => self.c.sp = v,
        }
    }

    /// Register pairs BC DE HL AF, for PUSH and POP.
    fn rp2(&self, p: u8) -> u16 {
        if p == 3 { pair(self.c.a, self.c.f) } else { self.rp(p) }
    }

    fn set_rp2(&mut self, p: u8, v: u16) {
        if p == 3 {
            [self.c.a, self.c.f] = v.to_be_bytes();
            self.c.f &= 0xF0;
        } else {
            self.set_rp(p, v);
        }
    }

    /// Conditions NZ Z NC C.
    fn cond(&self, y: u8) -> bool {
        let f = self.c.f;
        match y & 3 {
            0 => f & FZ == 0,
            1 => f & FZ != 0,
            2 => f & FC == 0,
            _ => f & FC != 0,
        }
    }

    fn flag(&self, mask: u8) -> bool {
        self.c.f & mask != 0
    }

    fn set_flags(&mut self, z: bool, n: bool, h: bool, c: bool) {
        self.c.f = (z as u8) << 7 | (n as u8) << 6 | (h as u8) << 5 | (c as u8) << 4;
    }

    /// ADD ADC SUB SBC AND XOR OR CP.
    fn alu(&mut self, op: u8, v: u8) {
        let a = self.c.a;
        let carry = (op == 1 || op == 3) && self.flag(FC);
        let cin = carry as u8;
        let r = match op {
            0 | 1 => {
                let r = a.wrapping_add(v).wrapping_add(cin);
                let h = (a & 0xF) + (v & 0xF) + cin > 0xF;
                let c = a as u16 + v as u16 + cin as u16 > 0xFF;
                self.set_flags(r == 0, false, h, c);
                r
            }
            2 | 3 | 7 => {
                let r = a.wrapping_sub(v).wrapping_sub(cin);
                let h = (a & 0xF) < (v & 0xF) + cin;
                let c = (a as u16) < v as u16 + cin as u16;
                self.set_flags(r == 0, true, h, c);
                r
            }
            4 => {
                let r = a & v;
                self.set_flags(r == 0, false, true, false);
                r
            }
            5 => {
                let r = a ^ v;
                self.set_flags(r == 0, false, false, false);
                r
            }
            _ => {
                let r = a | v;
                self.set_flags(r == 0, false, false, false);
                r
            }
        };
        if op != 7 {
            self.c.a = r;
        }
    }

    /// RLC RRC RL RR SLA SRA SWAP SRL. Returns the result and the new carry.
    fn shift(&self, op: u8, v: u8) -> (u8, bool) {
        let cin = self.flag(FC) as u8;
        match op {
            0 => (v.rotate_left(1), v & 0x80 != 0),
            1 => (v.rotate_right(1), v & 1 != 0),
            2 => (v << 1 | cin, v & 0x80 != 0),
            3 => (v >> 1 | cin << 7, v & 1 != 0),
            4 => (v << 1, v & 0x80 != 0),
            5 => (v >> 1 | (v & 0x80), v & 1 != 0),
            6 => (v.rotate_left(4), false),
            _ => (v >> 1, v & 1 != 0),
        }
    }

    /// SP plus a signed immediate, with the flags ADD SP,e and LD HL,SP+e share.
    fn sp_plus_imm(&mut self) -> u16 {
        let e = self.fetch();
        let sp = self.c.sp;
        let h = (sp & 0xF) + (e as u16 & 0xF) > 0xF;
        let c = (sp & 0xFF) + e as u16 > 0xFF;
        self.set_flags(false, false, h, c);
        sp.wrapping_add(e as i8 as u16)
    }

    fn daa(&mut self) {
        let (mut a, n, h) = (self.c.a, self.flag(FN), self.flag(FH));
        let mut c = self.flag(FC);
        if n {
            if c {
                a = a.wrapping_sub(0x60);
            }
            if h {
                a = a.wrapping_sub(0x06);
            }
        } else {
            if c || a > 0x99 {
                a = a.wrapping_add(0x60);
                c = true;
            }
            if h || a & 0xF > 9 {
                a = a.wrapping_add(0x06);
            }
        }
        self.c.a = a;
        self.set_flags(a == 0, n, false, c);
    }

    fn instruction(&mut self) {
        let ei_on_entry = self.c.ei_pending;
        let entry = self.c;
        self.c.ei_pending = false;
        self.c.halt_bug = false;
        let op = self.stream[0];
        if entry.halt_bug {
            self.next = 1;
            self.mcycles = 1;
        } else {
            self.fetch();
        }
        if ei_on_entry {
            self.c.ime = true;
        }
        if !self.decode(op) {
            self.c = entry;
            self.mcycles = 1;
        }
    }

    /// Runs one instruction after its opcode fetch. Returns false for an illegal opcode.
    fn decode(&mut self, op: u8) -> bool {
        let (x, y, z) = (op >> 6, (op >> 3) & 7, op & 7);
        let (p, q) = (y >> 1, y & 1);
        match (x, z) {
            (0, 0) => match y {
                0 => {}
                1 => {
                    let addr = self.fetch16();
                    let [hi, lo] = self.c.sp.to_be_bytes();
                    self.write(addr, lo);
                    self.write(addr.wrapping_add(1), hi);
                }
                2 => {}
                _ => {
                    let e = self.fetch() as i8;
                    if y == 3 || self.cond(y - 4) {
                        let to = self.c.pc.wrapping_add(e as u16);
                        self.idle_on(if y == 3 { self.c.pc } else { to });
                        self.c.pc = to;
                    }
                }
            },
            (0, 1) => {
                if q == 0 {
                    let v = self.fetch16();
                    self.set_rp(p, v);
                } else {
                    let (hl, v) = (self.hl(), self.rp(p));
                    let h = (hl & 0xFFF) + (v & 0xFFF) > 0xFFF;
                    let c = hl as u32 + v as u32 > 0xFFFF;
                    self.set_flags(self.flag(FZ), false, h, c);
                    self.set_hl(hl.wrapping_add(v));
                    self.idle();
                }
            }
            (0, 2) => {
                let addr = match p {
                    0 | 1 => self.rp(p),
                    _ => self.hl(),
                };
                match p {
                    2 => self.set_hl(addr.wrapping_add(1)),
                    3 => self.set_hl(addr.wrapping_sub(1)),
                    _ => {}
                }
                if q == 0 {
                    self.write(addr, self.c.a);
                } else {
                    self.c.a = self.read(addr);
                }
            }
            (0, 3) => {
                let v = self.rp(p);
                self.set_rp(p, if q == 0 { v.wrapping_add(1) } else { v.wrapping_sub(1) });
                self.idle_on(v);
            }
            (0, 4) | (0, 5) => {
                let v = self.reg(y);
                let dec = z == 5;
                let r = if dec { v.wrapping_sub(1) } else { v.wrapping_add(1) };
                let h = if dec { v & 0xF == 0 } else { v & 0xF == 0xF };
                self.set_flags(r == 0, dec, h, self.flag(FC));
                self.set_reg(y, r);
            }
            (0, 6) => {
                let v = self.fetch();
                self.set_reg(y, v);
            }
            (0, 7) => match y {
                0..=3 => {
                    let (r, c) = self.shift(y, self.c.a);
                    self.c.a = r;
                    self.set_flags(false, false, false, c);
                }
                4 => self.daa(),
                5 => {
                    self.c.a = !self.c.a;
                    self.c.f |= FN | FH;
                }
                6 => self.c.f = (self.c.f & FZ) | FC,
                _ => self.c.f = (self.c.f & FZ) | (!self.c.f & FC),
            },
            // HALT with an interrupt pending does not halt: with IME off the next opcode
            // byte is read twice; with IME on (EI's delay included) PC stays on the HALT,
            // so the dispatch returns into it (SameBoy v1.0.5 `halt`; SameSuite
            // `interrupt/ei_delay_halt`).
            (1, 6) if y == 6 => match (self.pending != 0, self.c.ime) {
                (true, false) => self.c.halt_bug = true,
                (true, true) => self.c.pc = self.pc0,
                (false, _) => self.c.halted = true,
            },
            (1, _) => {
                let v = self.reg(z);
                self.set_reg(y, v);
            }
            (2, _) => {
                let v = self.reg(z);
                self.alu(y, v);
            }
            (3, 0) => match y {
                0..=3 => {
                    self.idle();
                    if self.cond(y) {
                        self.c.pc = self.pop();
                        self.idle();
                    }
                }
                4 => {
                    let n = self.fetch();
                    self.write(0xFF00 | n as u16, self.c.a);
                }
                5 => {
                    self.c.sp = self.sp_plus_imm();
                    self.idle();
                    self.idle();
                }
                6 => {
                    let n = self.fetch();
                    self.c.a = self.read(0xFF00 | n as u16);
                }
                _ => {
                    let v = self.sp_plus_imm();
                    self.set_hl(v);
                    self.idle();
                }
            },
            (3, 1) => match (q, p) {
                (0, _) => {
                    let v = self.pop();
                    self.set_rp2(p, v);
                }
                (_, 0) | (_, 1) => {
                    self.c.pc = self.pop();
                    self.idle();
                    if p == 1 {
                        self.c.ime = true;
                    }
                }
                (_, 2) => self.c.pc = self.hl(),
                _ => {
                    self.c.sp = self.hl();
                    self.idle_on(self.c.sp);
                }
            },
            (3, 2) => match y {
                0..=3 => {
                    let addr = self.fetch16();
                    if self.cond(y) {
                        self.c.pc = addr;
                        self.idle();
                    }
                }
                4 => self.write(0xFF00 | self.c.c as u16, self.c.a),
                5 => {
                    let addr = self.fetch16();
                    self.write(addr, self.c.a);
                }
                6 => self.c.a = self.read(0xFF00 | self.c.c as u16),
                _ => {
                    let addr = self.fetch16();
                    self.c.a = self.read(addr);
                }
            },
            (3, 3) => match y {
                0 => {
                    self.c.pc = self.fetch16();
                    self.idle();
                }
                1 => self.prefixed(),
                6 => self.c.ime = false,
                7 => self.c.ei_pending = true,
                _ => return false,
            },
            (3, 4) if y < 4 => {
                let addr = self.fetch16();
                if self.cond(y) {
                    self.call(addr);
                }
            }
            (3, 5) if q == 0 => {
                self.idle_on(self.c.sp);
                self.push(self.rp2(p));
            }
            (3, 5) if y == 1 => {
                let addr = self.fetch16();
                self.call(addr);
            }
            (3, 6) => {
                let v = self.fetch();
                self.alu(y, v);
            }
            (3, 7) => self.call(y as u16 * 8),
            _ => return false,
        }
        true
    }

    fn call(&mut self, addr: u16) {
        self.idle_on(self.c.sp);
        self.push(self.c.pc);
        self.c.pc = addr;
    }

    fn prefixed(&mut self) {
        let op = self.fetch();
        let (x, y, z) = (op >> 6, (op >> 3) & 7, op & 7);
        let v = self.reg(z);
        match x {
            0 => {
                let (r, c) = self.shift(y, v);
                self.set_flags(r == 0, false, false, c);
                self.set_reg(z, r);
            }
            1 => self.set_flags(v & (1 << y) == 0, false, true, self.flag(FC)),
            2 => self.set_reg(z, v & !(1 << y)),
            _ => self.set_reg(z, v | (1 << y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(cpu: Cpu, op: [u8; 3], ie: u8, iflag: u8, data: [u8; 2]) -> (Request, Outcome) {
        let inp = Inputs { op, ie, iflag, dma: false, ..Inputs::default() };
        (Reference.request(&cpu, &inp), Reference.execute(&cpu, &inp, data))
    }

    #[test]
    fn dispatch_pushes_pc_and_jumps_to_lowest_pending_vector() {
        let cpu = Cpu { pc: 0x1234, sp: 0xD000, ime: true, ei_pending: true, ..Cpu::default() };
        let (req, out) = run(cpu, [0x00; 3], 0x1F, 0x06, [0, 0]);
        assert_eq!(req, Request { cycles: Some(AccessCycles { reads: [0, 0], writes: [2, 3], fetches: 0 }), ..Request::default() });
        assert_eq!(out.writes, [Some((0xCFFF, 0x12)), Some((0xCFFE, 0x34))]);
        assert_eq!(out.cpu, Cpu { pc: 0x0048, sp: 0xCFFE, ..Cpu::default() });
        assert_eq!(out.irq_ack, 0x02);
        assert_eq!(out.mcycles, 5);
    }

    #[test]
    fn halted_wakes_without_dispatch_when_ime_off() {
        let cpu = Cpu { pc: 0x0200, halted: true, ..Cpu::default() };
        let (_, out) = run(cpu, [0x3C, 0, 0], 0x01, 0x01, [0, 0]);
        assert_eq!(out.cpu, Cpu { pc: 0x0200, ..Cpu::default() });
        assert_eq!((out.mcycles, out.irq_ack, out.writes), (0, 0, [None; 2]));
    }

    #[test]
    fn ei_enables_ime_after_the_next_instruction() {
        let cpu = Cpu { pc: 0x0100, ..Cpu::default() };
        let (_, ei) = run(cpu, [0xFB, 0, 0], 0x01, 0x01, [0, 0]);
        assert_eq!(ei.cpu, Cpu { pc: 0x0101, ei_pending: true, ..Cpu::default() });
        // The pending interrupt does not preempt the instruction after EI.
        let (_, nop) = run(ei.cpu, [0x00, 0, 0], 0x01, 0x01, [0, 0]);
        assert_eq!(nop.cpu, Cpu { pc: 0x0102, ime: true, ..Cpu::default() });
        assert_eq!(nop.irq_ack, 0);
    }

    #[test]
    fn di_right_after_ei_keeps_ime_off() {
        let cpu = Cpu { pc: 0x0100, ei_pending: true, ..Cpu::default() };
        let (_, out) = run(cpu, [0xF3, 0, 0], 0, 0, [0, 0]);
        assert_eq!(out.cpu, Cpu { pc: 0x0101, ..Cpu::default() });
    }

    #[test]
    fn halt_with_ime_off_and_pending_sets_halt_bug() {
        let cpu = Cpu { pc: 0x0100, ..Cpu::default() };
        let (_, halt) = run(cpu, [0x76, 0, 0], 0x04, 0x04, [0, 0]);
        assert_eq!(halt.cpu, Cpu { pc: 0x0101, halt_bug: true, ..Cpu::default() });
        // LD A,n at 0x0101 reads its own opcode's address again for the immediate (the fetched
        // stream pc, pc, pc + 1) and pc advances by one.
        let (_, ld) = run(halt.cpu, [0x3E, 0x3E, 0x99], 0x04, 0x04, [0, 0]);
        assert_eq!(ld.cpu, Cpu { a: 0x3E, pc: 0x0102, ..Cpu::default() });
        assert_eq!(ld.mcycles, 2);
        // A lock that rose between the two reads: the second read's byte is the immediate.
        let (_, locked) = run(halt.cpu, [0x3E, 0xFF, 0x99], 0x04, 0x04, [0, 0]);
        assert_eq!(locked.cpu, Cpu { a: 0xFF, pc: 0x0102, ..Cpu::default() });
    }

    #[test]
    fn illegal_opcode_hangs_without_changing_state() {
        let cpu = Cpu { pc: 0x0100, ei_pending: true, ..Cpu::default() };
        let (req, out) = run(cpu, [0xD3, 0, 0], 0, 0, [0, 0]);
        assert_eq!(req, Request { cycles: Some(AccessCycles::default()), ..Request::default() });
        assert_eq!((out.cpu, out.writes, out.mcycles), (cpu, [None; 2], 1));
    }

    #[test]
    fn ei_then_halt_with_pending_stays_on_the_halt() {
        let cpu = Cpu { pc: 0x0100, ei_pending: true, ..Cpu::default() };
        let (_, halt) = run(cpu, [0x76, 0, 0], 0x01, 0x01, [0, 0]);
        assert_eq!(halt.cpu, Cpu { pc: 0x0100, ime: true, ..Cpu::default() }, "neither halted nor past it: the dispatch returns into the HALT");
    }

    #[test]
    fn halt_without_pending_halts() {
        let cpu = Cpu { pc: 0x0100, ..Cpu::default() };
        let (_, out) = run(cpu, [0x76, 0, 0], 0x04, 0x00, [0, 0]);
        assert_eq!(out.cpu, Cpu { pc: 0x0101, halted: true, ..Cpu::default() });
    }
}
