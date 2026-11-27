//! Interrupt dispatch, HALT wake-up and the EI/DI/RETI/HALT interplay, fuzzed: the circuit
//! and the checkpoint against the reference interpreter on the step contract, from random
//! single states and from random programs run with interrupts raised at random.
//! Seeded, so a failure reproduces; FUZZ_SCALE multiplies the case counts.

use gbcformer::control_circuit::{self, ControlCheckpoint, ControlLockstep, ControlModel};
use gbcformer::device::Signals;
use gbcformer::machine::control::{Control, ControlDevice};
use gbcformer::reference::Reference;
use gbcformer::sm83::{self, build, CircuitCore};
use gbcformer::step::{self, Bus, Core, Cpu, Inputs, Outcome, Request};

mod common;
use common::Rng;

impl Rng {
    fn chance(&mut self, one_in: u64) -> bool {
        self.next() % one_in == 0
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.next() as usize % items.len()]
    }
}

/// HALT, EI, DI, RETI, NOP, STOP, RET, CALL, RST 38, JR -2, LDH (FF0F),A and LD A,(FF0F)
/// via LDH: the opcodes whose interplay with interrupts the fuzz is after.
const INTERRUPT_OPS: [u8; 12] = [0x76, 0xFB, 0xF3, 0xD9, 0x00, 0x10, 0xC9, 0xCD, 0xFF, 0x18, 0xE0, 0xF0];

/// Runs the candidate beside the reference, counts disagreements, and carries on with
/// the reference's answer so one bug does not derail the rest of a program.
struct Checked<'a, C: Core> {
    candidate: &'a mut C,
    steps: u64,
    /// Steps the reference dispatched an interrupt on.
    dispatches: u64,
    /// Of those, the timer's.
    timer_dispatches: u64,
    mismatches: u64,
    first: Option<String>,
}

impl<C: Core> Checked<'_, C> {
    fn note(&mut self, what: String) {
        self.mismatches += 1;
        self.first.get_or_insert(what);
    }
}

/// What a fuzz run saw: disagreements, interrupt dispatches, and the first disagreement.
struct Fuzzed {
    mismatches: u64,
    dispatches: u64,
    first: Option<String>,
}

impl<C: Core> Core for Checked<'_, C> {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        let (want, got) = (Reference.request(cpu, inp), self.candidate.request(cpu, inp));
        if got != want {
            self.note(format!("request: cpu {cpu:?} inputs {inp:?}\n  want {want:?}\n  got  {got:?}"));
        }
        want
    }

    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        self.steps += 1;
        let (want, got) = (Reference.execute(cpu, inp, data), self.candidate.execute(cpu, inp, data));
        self.dispatches += (want.irq_ack != 0) as u64;
        self.timer_dispatches += (want.irq_ack == 0x04) as u64;
        if got != want {
            self.note(format!("outcome: cpu {cpu:?} inputs {inp:?} data {data:?}\n  want {want:?}\n  got  {got:?}"));
        }
        want
    }
}

/// A random state the host can hand a core: a halted CPU is only stepped with an
/// interrupt pending, and IF holds its 5 real bits. `ei_pending` and `halt_bug` are drawn
/// independently of `halted`, so a wake-up is checked with each of them set too.
fn random_state(rng: &mut Rng) -> (Cpu, Inputs) {
    let halted = rng.chance(4);
    let mut cpu = Cpu {
        a: rng.byte(),
        f: rng.byte() & 0xF0,
        b: rng.byte(),
        c: rng.byte(),
        d: rng.byte(),
        e: rng.byte(),
        h: rng.byte(),
        l: rng.byte(),
        sp: rng.next() as u16,
        pc: rng.next() as u16,
        ime: rng.chance(2),
        ei_pending: rng.chance(3),
        halted,
        halt_bug: rng.chance(4),
    };
    let op0 = if rng.chance(2) { rng.pick(&INTERRUPT_OPS) } else { rng.byte() };
    let mut inp = Inputs { op: [op0, rng.byte(), rng.byte()], ie: rng.byte(), iflag: rng.byte() & 0x1F, dma: false, ..Inputs::default() };
    if rng.chance(3) {
        inp.iflag &= !inp.ie;
    }
    if cpu.halted && inp.ie & inp.iflag & 0x1F == 0 {
        let bit = 1 << (rng.next() % 5);
        (inp.ie, inp.iflag) = (inp.ie | bit, inp.iflag | bit);
    }
    if rng.chance(8) {
        cpu.sp = rng.pick(&[0x0000, 0x0001, 0xFFFE, 0xFFFF]);
    }
    (cpu, inp)
}

fn fuzz_states(core: &mut impl Core, cases: u64, seed: u64) -> Fuzzed {
    let mut rng = Rng(seed);
    let mut checked = Checked { candidate: core, steps: 0, dispatches: 0, timer_dispatches: 0, mismatches: 0, first: None };
    for _ in 0..cases {
        let (cpu, inp) = random_state(&mut rng);
        checked.request(&cpu, &inp);
        checked.execute(&cpu, &inp, [rng.byte(), rng.byte()]);
    }
    Fuzzed { mismatches: checked.mismatches, dispatches: checked.dispatches, first: checked.first }
}

/// Flat 64 KiB with IE at FFFF and IF at FF0F, as the step contract sees memory.
struct Ram {
    bytes: Vec<u8>,
}

impl Bus for Ram {
    fn read(&mut self, addr: u16) -> u8 {
        self.bytes[addr as usize]
    }
    fn write(&mut self, addr: u16, value: u8) {
        self.bytes[addr as usize] = if addr == 0xFF0F { value & 0x1F } else { value };
    }
    fn ie(&self) -> u8 {
        self.bytes[0xFFFF]
    }
    fn iflag(&self) -> u8 {
        self.bytes[0xFF0F]
    }
    fn ack(&mut self, bits: u8) {
        self.bytes[0xFF0F] &= !bits;
    }
}

/// A program dense in interrupt opcodes, with handlers at the vectors that are too.
fn random_program(rng: &mut Rng) -> Ram {
    let mut bytes: Vec<u8> = (0..0x10000).map(|_| if rng.chance(3) { rng.byte() } else { rng.pick(&INTERRUPT_OPS) }).collect();
    for vector in [0x40, 0x48, 0x50, 0x58, 0x60] {
        bytes[vector] = rng.pick(&[0xD9, 0xFB, 0x00, 0x76, 0xC9]);
        bytes[vector + 1] = rng.pick(&[0xD9, 0xC9, 0xFB]);
    }
    bytes[0xFFFF] = rng.byte() | rng.pick(&[0x00, 0x1F]);
    bytes[0xFF0F] = 0;
    Ram { bytes }
}

fn fuzz_programs(core: &mut impl Core, programs: u64, steps: u64, seed: u64) -> Fuzzed {
    let mut rng = Rng(seed);
    let mut checked = Checked { candidate: core, steps: 0, dispatches: 0, timer_dispatches: 0, mismatches: 0, first: None };
    for _ in 0..programs {
        let mut ram = random_program(&mut rng);
        let mut cpu = Cpu { sp: 0xD000, pc: rng.next() as u16 & 0x7FFF, ime: rng.chance(2), ..Cpu::post_boot_dmg() };
        for _ in 0..steps {
            if rng.chance(6) {
                ram.bytes[0xFF0F] |= 1 << (rng.next() % 5);
            }
            // Programs overwrite IE too (LDH (FFFF),A), so keep re-enabling a source now
            // and then or dispatches die out for long stretches.
            if rng.chance(64) {
                ram.bytes[0xFFFF] = rng.byte();
            }
            if rng.chance(16) {
                ram.bytes[0xFFFF] |= 1 << (rng.next() % 5);
            }
            step::step(&mut checked, &mut cpu, &mut ram);
        }
    }
    Fuzzed { mismatches: checked.mismatches, dispatches: checked.dispatches, first: checked.first }
}

/// Flat 64 KiB with the timer, IF and IE in a control device, so interrupts come from
/// TIMA overflows the program sets up and not from random pokes.
struct TimerBus<D: ControlDevice> {
    bytes: Vec<u8>,
    control: D,
}

const CONTROL: [u16; 6] = [0xFF04, 0xFF05, 0xFF06, 0xFF07, 0xFF0F, 0xFFFF];

impl<D: ControlDevice> Bus for TimerBus<D> {
    fn read(&mut self, addr: u16) -> u8 {
        if CONTROL.contains(&addr) {
            return self.control.read(addr, &mut Vec::new());
        }
        self.bytes[addr as usize]
    }
    fn write(&mut self, addr: u16, value: u8) {
        if CONTROL.contains(&addr) {
            self.control.write(addr, value, &mut Signals::default());
        } else {
            self.bytes[addr as usize] = value;
        }
    }
    fn ie(&self) -> u8 {
        self.control.lines().ie
    }
    fn iflag(&self) -> u8 {
        self.control.lines().iflag
    }
    fn ack(&mut self, bits: u8) {
        if bits != 0 {
            self.control.ack(bits, &mut Vec::new());
        }
    }
}

impl<D: ControlDevice> TimerBus<D> {
    /// Advances the timer as the machine's scheduler does: in blocks cut at its horizon.
    fn advance(&mut self, mut mcycles: u64) {
        while mcycles > 0 {
            let block = mcycles.min(self.control.lines().horizon);
            self.control.advance(block, false, &mut Signals::default());
            mcycles -= block;
        }
    }
}

/// A program dense in interrupt opcodes and in writes that program the timer: TAC with
/// the fast rates, TIMA near overflow, TMA, DIV resets, IE with the timer bit, and reads
/// of TIMA and DIV.
fn timer_program(rng: &mut Rng) -> Vec<u8> {
    let mut bytes: Vec<u8> = (0..0x10000).map(|_| if rng.chance(3) { rng.byte() } else { rng.pick(&INTERRUPT_OPS) }).collect();
    for _ in 0..0x1800 {
        let at = rng.next() as usize % 0xFFF0;
        let (reg, value) = match rng.next() % 6 {
            0 => (0x07, 0x04 | rng.byte() & 3),
            1 => (0x05, 0xF0 | rng.byte()),
            2 => {
                let any = rng.byte();
                (0x06, rng.pick(&[0xFF, 0xFE, 0xF0, 0x00, any]))
            }
            3 => (0x04, rng.byte()),
            4 => (0xFF, 0x04 | rng.byte()),
            _ => (0x0F, rng.byte()),
        };
        let load = if rng.chance(4) { [0xF0, reg, 0x00, 0x00] } else { [0x3E, value, 0xE0, reg] };
        bytes[at..at + 4].copy_from_slice(&load);
    }
    for vector in [0x40, 0x48, 0x50, 0x58, 0x60] {
        bytes[vector] = rng.pick(&[0xD9, 0xFB, 0x00, 0x76, 0xC9]);
        bytes[vector + 1] = rng.pick(&[0xD9, 0xC9, 0xFB]);
    }
    bytes
}

/// Runs random timer programs one after another on `control`, whose timer keeps running
/// across them: the candidate CPU core is held to the reference every step, as
/// `fuzz_programs` does.
fn fuzz_timer_programs<D: ControlDevice>(core: &mut impl Core, control: D, programs: u64, steps: u64, seed: u64) -> (Fuzzed, u64) {
    let mut rng = Rng(seed);
    let mut checked = Checked { candidate: core, steps: 0, dispatches: 0, timer_dispatches: 0, mismatches: 0, first: None };
    let mut bus = TimerBus { bytes: Vec::new(), control };
    for _ in 0..programs {
        bus.bytes = timer_program(&mut rng);
        for (reg, value) in [(0xFF07, 0x05), (0xFF06, 0xF0), (0xFF05, 0xF8), (0xFFFF, 0x04 | rng.byte())] {
            bus.write(reg, value);
        }
        let mut cpu = Cpu { sp: 0xD000, pc: rng.next() as u16 & 0x7FFF, ime: rng.chance(2), ..Cpu::post_boot_dmg() };
        for _ in 0..steps {
            let m = step::step(&mut checked, &mut cpu, &mut bus).unwrap_or(1);
            bus.advance(m as u64);
        }
    }
    (Fuzzed { mismatches: checked.mismatches, dispatches: checked.dispatches, first: checked.first }, checked.timer_dispatches)
}

/// The compiled control checkpoint beside the reference timer and flags.
fn compiled_control() -> ControlLockstep<ControlCheckpoint<gbcformer::engine::Engine>> {
    let reference = Control::post_boot(false, 0).with_lengths(gbcformer::machine::ppu::Lengths::constant(44));
    let model = ControlModel::new(ControlCheckpoint::new(control_circuit::engine()), &reference.state(), reference.lengths());
    ControlLockstep::new(reference, model)
}

#[test]
fn timer_driven_interrupts_with_the_compiled_control_logic() {
    for (name, programs, (fuzzed, timer)) in [
        ("circuit", 40, fuzz_timer_programs(&mut CircuitCore(build()), compiled_control(), 40 * scale(), 400, 0xA076_1D64_78BD_642F)),
        ("checkpoint", 200, fuzz_timer_programs(&mut sm83::checkpoint(), compiled_control(), 200 * scale(), 400, 0xE703_7ED1_A0B4_28DB)),
    ] {
        eprintln!("{name}: {} program steps, {} dispatches, {timer} by the timer", programs * scale() * 400, fuzzed.dispatches);
        assert_eq!(fuzzed.mismatches, 0, "{name} disagrees in timer programs; first:\n{}", fuzzed.first.unwrap_or_default());
        assert!(timer >= 200 * scale(), "{name}: only {timer} timer interrupts dispatched");
    }
}

/// FUZZ_SCALE=10 runs ten times the cases.
fn scale() -> u64 {
    std::env::var("FUZZ_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
}

/// The fuzz must also exercise what it is after: at least `min_dispatches` interrupt
/// dispatches in the random states and as many again in the programs.
fn assert_agrees(name: &str, core: &mut impl Core, states: u64, programs: u64, min_dispatches: u64) {
    let t = std::time::Instant::now();
    let s = fuzz_states(core, states * scale(), 0x9E3779B97F4A7C15);
    let p = fuzz_programs(core, programs * scale(), 200, 0xD1B54A32D192ED03);
    eprintln!("{name}: {} random states ({} dispatches), {} program steps ({} dispatches), {:.1}s", states * scale(), s.dispatches, programs * scale() * 200, p.dispatches, t.elapsed().as_secs_f64());
    assert_eq!(s.mismatches, 0, "{name} disagrees on random states; first:\n{}", s.first.unwrap_or_default());
    assert_eq!(p.mismatches, 0, "{name} disagrees in random programs; first:\n{}", p.first.unwrap_or_default());
    assert!(s.dispatches >= min_dispatches * scale(), "{name}: only {} dispatches in random states", s.dispatches);
    assert!(p.dispatches >= min_dispatches * scale(), "{name}: only {} dispatches in random programs", p.dispatches);
}

#[test]
fn circuit_matches_reference_on_interrupts_and_halt() {
    assert_agrees("circuit", &mut CircuitCore(build()), 20_000, 100, 600);
}

#[test]
fn checkpoint_matches_reference_on_interrupts_and_halt() {
    let mut core = sm83::checkpoint();
    assert_agrees("checkpoint", &mut core, 100_000, 500, 3_000);
    let stats = &core.model().stats;
    assert_eq!((stats.fallback_passes, stats.passes == stats.fast_passes), (0, true), "every pass took the engine's fast path");
}

/// The fuzz must catch a core that gets dispatch or wake-up wrong.
#[test]
fn fuzz_rejects_interrupt_mutants() {
    struct Mutant(fn(&Cpu, &Inputs, &mut Outcome));
    impl Core for Mutant {
        fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
            Reference.request(cpu, inp)
        }
        fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
            let mut out = Reference.execute(cpu, inp, data);
            (self.0)(cpu, inp, &mut out);
            out
        }
    }
    let mutants: [(&str, fn(&Cpu, &Inputs, &mut Outcome)); 4] = [
        ("joypad vector off by 8", |_, _, out| {
            if out.irq_ack == 0x10 {
                out.cpu.pc = 0x68;
            }
        }),
        ("dispatch leaves IME on", |cpu, inp, out| {
            if !cpu.halted && cpu.ime && inp.ie & inp.iflag & 0x1F != 0 {
                out.cpu.ime = true;
            }
        }),
        ("wake-up stays halted", |cpu, _, out| {
            if cpu.halted {
                out.cpu.halted = true;
            }
        }),
        ("RETI leaves IME off", |cpu, inp, out| {
            if !cpu.halted && !(cpu.ime && inp.ie & inp.iflag & 0x1F != 0) && inp.op[0] == 0xD9 && !cpu.halt_bug {
                out.cpu.ime = cpu.ime;
            }
        }),
    ];
    for (name, mutate) in mutants {
        let bad_states = fuzz_states(&mut Mutant(mutate), 2_000, 7).mismatches;
        let bad_runs = fuzz_programs(&mut Mutant(mutate), 5, 200, 11).mismatches;
        assert!(bad_states > 0 && bad_runs > 0, "{name}: {bad_states} bad states, {bad_runs} bad program steps");
    }
}
