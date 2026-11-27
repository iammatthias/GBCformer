//! Which parts of the Game Boy run in weights, which on the host, and which run as
//! both with the first divergence stopping the run. A [`Stack`] names a [`Backend`]
//! per [`Component`]. Any stack can be written; [`Stack::resolve`] applies the
//! dependency rules and names the checkpoints to compile ([`Checkpoints`]: the CPU
//! side's [`Weights`], and the APU's own), and [`Stack::build`] compiles them into an
//! [`AnyCore`] and the devices that replace the machine's references. With
//! [`Stack::one_model`] every component in weights runs as a stream of one checkpoint
//! (`crate::one_model`) instead of a checkpoint each; the builders take an [`Engine`]
//! either way.

use crate::apu_circuit::{self, ApuCheckpoint, ApuLockstep, ApuModel};
use crate::control_circuit::{self, ControlCheckpoint, ControlLockstep, ControlModel};
use crate::engine::{Engine, VideoRecords};
use crate::machine::apu::AudioDevice;
use crate::machine::ControlReplacement;
use crate::machine::ppu::Renderer;
use crate::ppu_circuit::{self, PpuCheckpoint, PpuModel};
use crate::llama::Model;
use crate::lockstep::Lockstep;
use crate::memory::attn::Params;
use crate::memory::layout::{self, Access, Port};
use crate::memory::{Addr, CopyFrom, Copied};
use crate::reference::Reference;
use crate::session::Session;
use crate::sm83::{self, CheckpointCore, MemCheckpointCore};
use crate::step::{Bus, Core, Cpu, Inputs, Outcome, Request, StepFacts};
use std::fmt;

/// A part of the Game Boy that can run in weights. `Control` is
/// the timer, interrupt controller, joypad register, DMA control, serial, the PPU's
/// timing and the banking registers; [`Component::still_hosted`] names what a component
/// in weights still leaves on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Component {
    Cpu,
    Memory,
    Ppu,
    Apu,
    Control,
}

/// What the control logic in weights leaves on the host while memory is not in weights.
pub const CONTROL_ON_HOST: &str = "the cartridge banking registers";

impl Component {
    pub const ALL: [Component; 5] = [Component::Cpu, Component::Memory, Component::Ppu, Component::Apu, Component::Control];

    pub fn name(self) -> &'static str {
        match self {
            Component::Cpu => "cpu",
            Component::Memory => "memory",
            Component::Ppu => "ppu",
            Component::Apu => "apu",
            Component::Control => "control",
        }
    }

    pub fn parse(name: &str) -> Option<Component> {
        Component::ALL.into_iter().find(|c| c.name() == name)
    }

    /// The component this one needs in weights too, and why.
    pub fn requires(self) -> Option<(Component, &'static str)> {
        match self {
            Component::Memory => Some((Component::Cpu, "the memory heads live in the CPU checkpoint")),
            Component::Ppu => Some((Component::Memory, "its VRAM and OAM follow the memory checkpoint's write records")),
            Component::Cpu | Component::Apu | Component::Control => None,
        }
    }

    /// The parts of a partly built component that stay on the host when it is in
    /// weights in `stack`, for the run's label.
    pub fn still_hosted(self, stack: Stack) -> Option<&'static str> {
        match self {
            // The memory checkpoint holds the banking registers and translates with them.
            Component::Control if !stack.in_weights(Component::Memory) => Some(CONTROL_ON_HOST),
            _ => None,
        }
    }
}

impl fmt::Display for Component {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Who computes a component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// The reference Rust in `machine` and `reference`.
    #[default]
    Host,
    /// The compiled checkpoint.
    Model,
    /// The checkpoint, with the host's reference run beside it and the first
    /// divergence stopping the run.
    Lockstep,
}

impl Backend {
    pub fn in_weights(self) -> bool {
        self != Backend::Host
    }
}

/// A backend per component, and whether the components in weights share one
/// checkpoint. The default is everything on the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stack {
    backends: [Backend; Component::ALL.len()],
    one_model: bool,
}

/// The CPU-side checkpoints the compiler builds today, one per set of CPU-side
/// components in weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weights {
    None,
    /// `sm83::checkpoint()`: the CPU, with the host serving every byte.
    Cpu,
    /// `sm83::build_mem()` compiled with a KV-cache memory: the CPU and the machine's
    /// memory, with the host serving only the I/O page, IE and the MBC3 clock.
    CpuMemory,
}

/// Everything a stack compiles: the CPU side's checkpoint and, with `apu`, the APU's
/// own (`apu_circuit::engine()`), which needs no memory head and so shares no cache
/// with the CPU side (`docs/apu-in-weights.md` §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoints {
    pub core: Weights,
    pub apu: bool,
    /// The timer and interrupt flags' own checkpoint (`control_circuit::engine()`).
    pub control: bool,
    /// The scanline checkpoint (`ppu_circuit::engine()`), whose VRAM and OAM follow the
    /// CPU side's records.
    pub ppu: bool,
    /// One checkpoint holding every circuit (`one_model::streams`) in place of the
    /// checkpoints above, each part a stream running its own slice of it.
    pub one_model: bool,
}

/// What [`Stack::build`] compiled: the core that steps the CPU, and the device that
/// replaces the machine's reference APU when the APU is in weights.
pub struct Built {
    pub core: AnyCore,
    pub apu: Option<Box<dyn AudioDevice>>,
    pub control: Option<ControlReplacement>,
    /// Draws the lines when the PPU is in weights.
    pub ppu: Option<Box<dyn Renderer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackError {
    Requires { component: Component, requires: Component, because: &'static str },
    /// `--one-model` without memory in weights. The one checkpoint's CPU stream holds
    /// the memory heads, so it has no CPU without its memory.
    OneModelWithoutMemory,
}

impl fmt::Display for StackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StackError::OneModelWithoutMemory => f.write_str("--one-model needs memory in weights (the one checkpoint's CPU stream holds the memory heads): give --weights cpu,memory or more"),
            StackError::Requires { component, requires, because } => write!(f, "{component} in weights requires {requires} in weights ({because})"),
        }
    }
}

impl std::error::Error for StackError {}

/// Knobs the host sets from its environment; the library reads no environment itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Instrumentation {
    /// Time the engine's stages (`GBCFORMER_STATS`).
    pub timed: bool,
    /// Check every Nth memory record against the dense pass; 0 never (`GBCFORMER_ORACLE`).
    pub oracle_every: u64,
    /// How often each engine's fast passes also run through the dense pass
    /// (`GBCFORMER_STEP_ORACLE`).
    pub step_oracle: StepOracle,
    /// Run the APU checkpoint on the CPU's thread instead of its own
    /// (`GBCFORMER_APU_THREAD=0`). On wasm32 it always is.
    pub apu_same_thread: bool,
    /// Run the PPU checkpoint on the CPU's thread instead of its own
    /// (`GBCFORMER_PPU_THREAD=0`). On wasm32 it always is.
    pub ppu_same_thread: bool,
    /// Write the K/V rows of the GBC CPU's write records into their cells on the CPU's
    /// thread instead of a worker's (`GBCFORMER_ROWS_THREAD=0`). On wasm32 it always is.
    pub rows_same_thread: bool,
}

/// Per engine, run every Nth fast pass, the first included, through the dense pass as
/// well, every decoded token compared (`Engine::step_oracle`); 0 never.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepOracle {
    /// The CPU side's engine, its memory included.
    pub cpu: u64,
    pub apu: u64,
    pub control: u64,
    pub ppu: u64,
    /// The CPU side's fast passes whose memory heads read ROM past 2 MiB, on their own
    /// count (`rom=N`), besides `cpu`'s cadence.
    pub rom: u64,
}

impl StepOracle {
    /// `GBCFORMER_STEP_ORACLE`'s value: items separated by commas, each a cadence for
    /// every engine (`5000`) or for one component's (`cpu=150000`; `memory` names the
    /// CPU side's engine too; `rom` the CPU side's passes that read ROM past 2 MiB). A
    /// later item overrides an earlier one.
    pub fn parse(value: &str) -> Result<StepOracle, String> {
        let mut o = StepOracle::default();
        for item in value.split(',').map(str::trim).filter(|i| !i.is_empty()) {
            let (name, count) = item.split_once('=').map_or((None, item), |(a, b)| (Some(a.trim()), b.trim()));
            let n: u64 = count.parse().map_err(|_| format!("GBCFORMER_STEP_ORACLE {item}: {count} is not a count of passes"))?;
            if name == Some("rom") {
                o.rom = n;
                continue;
            }
            match name.map(|x| Component::parse(x).ok_or_else(|| format!("GBCFORMER_STEP_ORACLE {item}: unknown component; the components are {}", list(&Component::ALL, ", ")))).transpose()? {
                None => o = StepOracle { cpu: n, apu: n, control: n, ppu: n, rom: o.rom },
                Some(Component::Cpu | Component::Memory) => o.cpu = n,
                Some(Component::Apu) => o.apu = n,
                Some(Component::Control) => o.control = n,
                Some(Component::Ppu) => o.ppu = n,
            }
        }
        Ok(o)
    }
}

/// The engines a stack runs, one for each checkpoint [`Checkpoints`] names: each
/// compiled from its own checkpoint, or each a stream of the one checkpoint.
struct Engines {
    /// The CPU side's, with its memory under [`Weights::CpuMemory`].
    cpu: Option<Engine>,
    apu: Option<Engine>,
    control: Option<Engine>,
    ppu: Option<Engine>,
    /// The PPU's mode-3 length on its own, and its FIFO stepper, with the PPU.
    length: Option<Engine>,
    stepper: Option<Engine>,
}

impl Engines {
    fn new(checkpoints: Checkpoints, instr: Instrumentation) -> Engines {
        let mut engines = if checkpoints.one_model {
            use crate::one_model::Part;
            let want: Vec<Part> = [(Part::Cpu, true), (Part::Ppu, checkpoints.ppu), (Part::Apu, checkpoints.apu), (Part::Control, checkpoints.control), (Part::Length, checkpoints.ppu), (Part::Stepper, checkpoints.ppu)].into_iter().filter(|p| p.1).map(|p| p.0).collect();
            let s = crate::one_model::streams(&want);
            Engines { cpu: s.cpu, apu: s.apu, control: s.control, ppu: s.ppu, length: s.length, stepper: s.stepper }
        } else {
            let cpu = match checkpoints.core {
                Weights::None => None,
                Weights::Cpu => Some(sm83::engine()),
                Weights::CpuMemory => Some(Engine::new(Model::compile_mem(&sm83::build_mem(), Params::memory()))),
            };
            Engines { cpu, apu: checkpoints.apu.then(apu_circuit::engine), control: checkpoints.control.then(control_circuit::engine), ppu: checkpoints.ppu.then(ppu_circuit::engine), length: checkpoints.ppu.then(ppu_circuit::length_engine), stepper: checkpoints.ppu.then(ppu_circuit::stepper::engine) }
        };
        let o = instr.step_oracle;
        for (engine, every) in [(&mut engines.cpu, o.cpu), (&mut engines.apu, o.apu), (&mut engines.control, o.control), (&mut engines.ppu, o.ppu), (&mut engines.length, o.ppu), (&mut engines.stepper, o.ppu)] {
            if let Some(e) = engine {
                e.timed = instr.timed;
                e.step_oracle = every;
            }
        }
        if let Some(e) = &mut engines.cpu {
            e.rom_oracle = o.rom;
        }
        engines
    }
}

impl Stack {
    pub const HOST: Stack = Stack { backends: [Backend::Host; Component::ALL.len()], one_model: false };

    pub fn get(self, c: Component) -> Backend {
        self.backends[c as usize]
    }

    pub fn with(mut self, c: Component, b: Backend) -> Stack {
        self.backends[c as usize] = b;
        self
    }

    /// `--one-model`: every component in weights as a stream of one checkpoint
    /// (`docs/one-model.md`) instead of a checkpoint per component.
    pub fn with_one_model(mut self, one_model: bool) -> Stack {
        self.one_model = one_model;
        self
    }

    pub fn one_model(self) -> bool {
        self.one_model
    }

    pub fn in_weights(self, c: Component) -> bool {
        self.get(c).in_weights()
    }

    pub fn lockstep(self, c: Component) -> bool {
        self.get(c) == Backend::Lockstep
    }

    fn components(self, keep: impl Fn(Backend) -> bool) -> Vec<Component> {
        Component::ALL.into_iter().filter(|&c| keep(self.get(c))).collect()
    }

    /// The components in weights, in `Component::ALL` order.
    pub fn weights(self) -> Vec<Component> {
        self.components(Backend::in_weights)
    }

    pub fn lockstepped(self) -> Vec<Component> {
        self.components(|b| b == Backend::Lockstep)
    }

    pub fn hosted(self) -> Vec<Component> {
        self.components(|b| b == Backend::Host)
    }

    /// `--weights` and `--lockstep`. `weights` is `none`, `built` (every component the
    /// compiler can put in weights, which is every component), `all`, or component names separated by
    /// commas; `lockstep` is `none`, `all` (every component in weights) or names, each
    /// of which must be in `weights`.
    pub fn parse(weights: &str, lockstep: Option<&str>) -> Result<Stack, String> {
        let mut stack = Stack::HOST;
        let weights = if weights == "built" { Component::ALL.to_vec() } else { named(weights, "--weights", &Component::ALL)? };
        for c in weights {
            stack = stack.with(c, Backend::Model);
        }
        for c in named(lockstep.unwrap_or("none"), "--lockstep", &stack.weights())? {
            if !stack.in_weights(c) {
                return Err(format!("--lockstep {c} needs {c} in --weights"));
            }
            stack = stack.with(c, Backend::Lockstep);
        }
        Ok(stack)
    }

    /// The `--core` names from before per-component selection. `model-mem` and
    /// `lockstep-mem` both hold the model's fetches and reads to the host's bytes every
    /// step, which is memory lockstep, so they map to it.
    pub fn alias(core: &str) -> Option<Stack> {
        let (weights, lockstep) = match core {
            "ref" => ("none", "none"),
            "model" => ("cpu", "none"),
            "lockstep" => ("cpu", "cpu"),
            "model-mem" => ("cpu,memory", "memory"),
            "lockstep-mem" => ("cpu,memory", "all"),
            _ => return None,
        };
        Some(Stack::parse(weights, Some(lockstep)).expect("the aliases are well-formed"))
    }

    /// `--weights X --lockstep Y`, parseable by [`Stack::parse`].
    pub fn flags(self) -> String {
        format!("weights={} lockstep={}", list(&self.weights(), ","), list(&self.lockstepped(), ","))
    }

    /// Applies the dependency rules and names the checkpoints to compile.
    pub fn resolve(self) -> Result<Checkpoints, StackError> {
        for c in self.weights() {
            if let Some((requires, because)) = c.requires()
                && !self.in_weights(requires)
            {
                return Err(StackError::Requires { component: c, requires, because });
            }
        }
        let core = match (self.in_weights(Component::Cpu), self.in_weights(Component::Memory)) {
            (false, false) => Weights::None,
            (true, false) => Weights::Cpu,
            (true, true) => Weights::CpuMemory,
            (false, true) => unreachable!("memory requires cpu"),
        };
        if self.one_model && core != Weights::CpuMemory {
            return Err(StackError::OneModelWithoutMemory);
        }
        Ok(Checkpoints { core, apu: self.in_weights(Component::Apu), control: self.in_weights(Component::Control), ppu: self.in_weights(Component::Ppu), one_model: self.one_model })
    }

    /// Compiles the checkpoints the stack names. Memory in weights starts from `warm`'s
    /// memory image and the APU from `warm`'s APU state, so this runs at the hand-off,
    /// after any warmup frames.
    pub fn build(self, warm: &Session<Reference>, instr: Instrumentation) -> Result<Built, String> {
        let checkpoints = self.resolve().map_err(|e| e.to_string())?;
        let engines = Engines::new(checkpoints, instr);
        let apu = engines.apu.map(|e| self.build_apu(warm, instr, e)).transpose()?;
        let control = engines.control.map(|e| self.build_control(warm, e)).transpose()?;
        let video: Option<VideoRecords> = engines.ppu.is_some().then(VideoRecords::default);
        let memory = checkpoints.core == Weights::CpuMemory;
        let core = engines.cpu.map_or(Ok(AnyCore::Host(Reference)), |e| self.build_core(e, memory, warm, instr, video.clone()))?;
        let (length, stepper) = (engines.length, engines.stepper);
        let ppu = engines.ppu.zip(video).map(|(e, records)| self.build_ppu(warm, instr, records, e, length.expect("the length stream with the PPU"), stepper.expect("the stepper with the PPU"))).transpose()?;
        Ok(Built { core, apu, control, ppu })
    }

    /// The scanline checkpoint, its VRAM and OAM as of this M-cycle, then following the
    /// CPU side's `records`; under `--lockstep ppu` every line is held to the reference
    /// renderer's pixels.
    fn build_ppu(self, warm: &Session<Reference>, instr: Instrumentation, records: VideoRecords, engine: Engine, length: Engine, stepper: Engine) -> Result<Box<dyn Renderer>, String> {
        let (display, window) = warm.ppu_state();
        let image = warm.memory_image()?;
        let mut checkpoint = PpuCheckpoint::new(engine, &image, &display);
        let length = ppu_circuit::LengthModel::new(ppu_circuit::LengthCheckpoint::new(length, &image, &display), window, warm.ppu_lengths());
        let stepper = ppu_circuit::stepper::Stepper::over(stepper, &image, &display);
        checkpoint.set_oracle(instr.oracle_every);
        let model = PpuModel::new(checkpoint, stepper, window, self.lockstep(Component::Ppu), warm.ppu_lengths());
        #[cfg(not(target_arch = "wasm32"))]
        if !instr.ppu_same_thread {
            return Ok(Box::new(ppu_circuit::worker::Worker::spawn(model, records, length)));
        }
        Ok(Box::new(ppu_circuit::Inline::new(model, records, length)))
    }

    /// The control checkpoint, starting from the reference timer and interrupt flags at
    /// this M-cycle; under `--lockstep control` a copy of the reference runs beside it.
    /// It runs on the CPU's thread, because the CPU reads IF at every step.
    fn build_control(self, warm: &Session<Reference>, engine: Engine) -> Result<ControlReplacement, String> {
        let reference = warm.control_reference().ok_or("the control logic was already handed to a device")?.clone();
        let model = ControlModel::new(ControlCheckpoint::new(engine), &reference.state(), reference.lengths());
        Ok(if self.lockstep(Component::Control) { ControlReplacement::Device(Box::new(ControlLockstep::new(reference, model))) } else { ControlReplacement::Compiled(Box::new(model)) })
    }

    /// The APU checkpoint, starting from the reference APU's state at this M-cycle; under
    /// `--lockstep apu` a copy of the reference runs beside it.
    fn build_apu(self, warm: &Session<Reference>, #[cfg_attr(target_arch = "wasm32", allow(unused_variables))] instr: Instrumentation, engine: Engine) -> Result<Box<dyn AudioDevice>, String> {
        let mut reference = warm.apu_reference().ok_or("the APU was already handed to a device")?.clone();
        let model = ApuModel::new(ApuCheckpoint::new(engine), &mut reference);
        let device: Box<dyn AudioDevice> = if self.lockstep(Component::Apu) { Box::new(ApuLockstep::new(reference, model)) } else { Box::new(model) };
        #[cfg(not(target_arch = "wasm32"))]
        if !instr.apu_same_thread {
            return Ok(Box::new(apu_circuit::worker::Worker::spawn(device)));
        }
        Ok(device)
    }

    /// The CPU side's core on `engine`, with the machine's memory in its cache when
    /// `memory`.
    fn build_core(self, mut engine: Engine, memory: bool, warm: &Session<Reference>, instr: Instrumentation, video: Option<VideoRecords>) -> Result<AnyCore, String> {
        let (cpu_lockstep, memory_lockstep) = (self.lockstep(Component::Cpu), self.lockstep(Component::Memory));
        if !memory {
            let core = CheckpointCore::new(engine);
            return Ok(if cpu_lockstep { AnyCore::CpuLockstep(Lockstep::new(Reference, core)) } else { AnyCore::Cpu(core) });
        }
        engine.insert(warm.memory_image()?);
        let cache = engine.memory.as_mut().expect("a memory checkpoint's engine has a memory");
        cache.oracle_every = instr.oracle_every;
        #[cfg(not(target_arch = "wasm32"))]
        if !instr.rows_same_thread {
            cache.cells.defer();
        }
        cache.video = video.map(ppu_circuit::video);
        let core = MemCheckpointCore::new(engine);
        // Without a lockstep nothing reads the evidence slots the audit holds to the bus.
        let core = if cpu_lockstep || memory_lockstep { core } else { core.without_audit_slots() };
        Ok(if cpu_lockstep { AnyCore::CpuLockstepMemory { core: Lockstep::new(Reference, core), audit: memory_lockstep } } else { AnyCore::CpuMemory { core, audit: memory_lockstep } })
    }
}

/// Component names from a flag value: `none`, `all` (the components `all` lists), or
/// names separated by commas.
fn named(value: &str, flag: &str, all: &[Component]) -> Result<Vec<Component>, String> {
    match value {
        "none" | "" => Ok(Vec::new()),
        "all" => Ok(all.to_vec()),
        _ => value.split(',').map(|name| Component::parse(name.trim()).ok_or_else(|| format!("{flag} {name}: unknown component; the components are {}", list(&Component::ALL, ", ")))).collect(),
    }
}

/// Names joined by `sep`, or `none`.
fn list(components: &[Component], sep: &str) -> String {
    if components.is_empty() { "none".into() } else { components.iter().map(|c| c.name()).collect::<Vec<_>>().join(sep) }
}

/// What the host does with every component in weights, besides what
/// `Component::still_hosted` names. Each item is labelled where it happens, in the
/// devices' `describe`.
const HOST_WORK: &str = "; the host besides: presentation (each pixel's colour mapped to the screen's RGB, the board's high-pass filter over the APU's mix, the samples played, the frame pacing, battery saves written), input (the buttons, and the wall-clock time an MBC3 cartridge's clock counts, handed over as the time elapsed), and scheduling (M-cycles counted between passes and down to the instants the control model emitted for each scheduled mode-3 start and frame-sequencer edge, the 48 kHz sample instants and the frame time counted in real time at the halves per M-cycle the control model emits, the speed and the frame-sequencer edges the control model emits handed to the APU, the GBC CPU's passes skipped while it is halted and the control model's wake bit is clear, each data access of the GBC CPU to an I/O register handed to the device whose port the memory model's emitted address names (the model decides the device, the host moves the byte), each DMA byte placed at the address bits the control model emitted and the CPU stopped for the M-cycles it names, and each model's outputs handed to the next pass or the next model), and on the host's own GBC rules still (docs/host-in-weights.md §5): the cartridge's mapper writes (which the clock's decode reads), and without a boot ROM the header's CGB byte and the post-boot state, a snapshot handed over as data";

/// The label every run carries: what ran in weights, what ran on the host, and what
/// ran as both.
impl fmt::Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "in weights: {}; on the host: {}; lockstep: {}", list(&self.weights(), ", "), list(&self.hosted(), ", "), list(&self.lockstepped(), ", "))?;
        for c in self.weights() {
            if let Some(rest) = c.still_hosted(*self) {
                write!(f, "; {c} still on the host: {rest}")?;
            }
        }
        if self.hosted().is_empty() {
            f.write_str(HOST_WORK)?;
        }
        if self.one_model {
            f.write_str("; one checkpoint: every component in weights runs as a stream of the same weights (docs/one-model.md)")?;
        }
        Ok(())
    }
}

/// The cores a [`Stack`] builds, as one type so a session can choose at run time.
/// `audit` is memory lockstep: the model's fetches and reads held to the host's bytes
/// after every step (`Core::audit`).
pub enum AnyCore {
    Host(Reference),
    Cpu(CheckpointCore<Engine>),
    CpuLockstep(Lockstep<Reference, CheckpointCore<Engine>>),
    CpuMemory { core: MemCheckpointCore<Engine>, audit: bool },
    CpuLockstepMemory { core: Lockstep<Reference, MemCheckpointCore<Engine>>, audit: bool },
}

impl AnyCore {
    /// The one compiled checkpoint, when any component is in weights.
    pub fn engine(&self) -> Option<&Engine> {
        match self {
            AnyCore::Host(_) => None,
            AnyCore::Cpu(c) => Some(c.model()),
            AnyCore::CpuLockstep(l) => Some(l.model.model()),
            AnyCore::CpuMemory { core, .. } => Some(core.model()),
            AnyCore::CpuLockstepMemory { core, .. } => Some(core.model.model()),
        }
    }

    /// With memory in weights, the battery save the model's records hold: `len` bytes of
    /// its cartridge RAM when the game finished a save (`committed`) or, at shutdown,
    /// wrote anything since the last one. None when the host's cartridge holds the save.
    /// Carries the host cartridge's unsaved writes over to the model's save tracker.
    pub fn seed_battery(&mut self, written: bool, committed: bool) {
        match self {
            AnyCore::CpuMemory { core, .. } => core.battery = sm83::Battery::seeded(written, committed),
            AnyCore::CpuLockstepMemory { core, .. } => core.model.battery = sm83::Battery::seeded(written, committed),
            _ => {}
        }
    }

    fn own_save(&mut self, committed: bool, len: usize) -> Option<Option<Vec<u8>>> {
        let core = match self {
            AnyCore::CpuMemory { core, .. } => core,
            AnyCore::CpuLockstepMemory { core, .. } => &mut core.model,
            _ => return None,
        };
        let due = if committed { core.battery.take_committed() } else { core.battery.take_unsaved() };
        let image = &core.model().memory.as_ref().expect("a memory checkpoint has a cache").image;
        Some(due.then(|| image.region(layout::CRAM, len).to_vec()))
    }

    fn memory_core(&self) -> Option<&MemCheckpointCore<Engine>> {
        match self {
            AnyCore::CpuMemory { core, .. } => Some(core),
            AnyCore::CpuLockstepMemory { core, .. } => Some(&core.model),
            _ => None,
        }
    }

    fn cpu_core(&self) -> Option<&CheckpointCore<Engine>> {
        match self {
            AnyCore::Cpu(c) => Some(c),
            AnyCore::CpuLockstep(l) => Some(&l.model),
            _ => None,
        }
    }

    /// The compiled checkpoint's shape and, for memory in weights, the cache it starts
    /// from. Empty with everything on the host.
    pub fn describe(&self) -> Vec<String> {
        let Some(engine) = self.engine() else { return Vec::new() };
        let c = &engine.dense().cfg;
        let fast_path = engine.exact().map_or_else(|e| format!("OFF ({e}): every pass runs dense"), |_| "on".into());
        let mut lines = Vec::new();
        match self.memory_core() {
            None => lines.push(format!("checkpoint: {} layers, hidden {}, intermediate {}, {:.1}M parameters ({} nonzero), integer fast path {fast_path}", c.layers, c.hidden, c.intermediate, c.params() as f64 / 1e6, engine.nonzeros())),
            Some(_) => {
                lines.push(format!("memory checkpoint: {} layers, hidden {}, intermediate {}, {} heads, {:.1}M parameters ({} nonzero), integer fast path {fast_path}", c.layers, c.hidden, c.intermediate, c.heads, c.params() as f64 / 1e6, engine.nonzeros()));
                let mem = engine.memory.as_ref().expect("a memory checkpoint has a cache");
                let oracle = if mem.oracle_every > 0 { format!("; dense oracle every {} records", mem.oracle_every) } else { String::new() };
                lines.push(format!("memory: inserted the machine's state as {} cells, one per 8-byte line ({:.0} MB of f32 KV at one KV head of one memory layer; {:.0} MB in a stock runtime's f16 cache){oracle}", mem.kv.len(), mem.kv.bytes() as f64 / 1e6, (mem.kv.len() * c.kv_bytes_per_token() / 2) as f64 / 1e6));
                if mem.cells.deferred() {
                    lines.push("memory: a worker thread writes each write record's K/V rows into its cell, in order, a batch of records at a time (inference management: the rows are the same); what a read needs of them, the key test and the line the value spells, stays on the GBC CPU's thread".into());
                }
            }
        }
        lines
    }

    /// After a run of `wall` seconds: what the host still served, always, and the
    /// engine's stage timings when `detailed`.
    pub fn report(&self, wall: f64, detailed: bool) -> Vec<String> {
        let mut lines = Vec::new();
        // The detailed stats carry the step oracle's count themselves.
        if let Some(e) = self.engine()
            && e.stats.step_checks > 0
            && !detailed
        {
            lines.push(format!("cpu engine: {} passes; {} fast passes checked by the step oracle, every token equal to the dense pass's, {} OAM rewrite passes among them; {}", e.stats.passes, e.stats.step_checks, e.stats.forced_checks, e.stats.step_high_line()));
        }
        if let Some(core) = self.cpu_core()
            && detailed
        {
            let stats = &core.model().stats;
            lines.push(format!("steps: {} one-pass, {} two-pass; {:.2} µs per pass", core.one_pass_steps, core.two_pass_steps, wall * 1e6 / stats.passes.max(1) as f64));
            lines.push(stats.to_string());
        }
        if let Some(core) = self.cpu_core() {
            let e = core.model();
            let census = e.census();
            if !census.is_empty() {
                let top: Vec<String> = census.iter().map(|(n, u)| format!("{n} {u:.1}")).collect();
                lines.push(format!("cpu gate updates per fast pass by node family: {}", top.join(", ")));
            }
            let sources = e.sources();
            if !sources.is_empty() {
                let top: Vec<String> = sources.iter().map(|(n, u)| format!("{n} {u:.1}")).collect();
                lines.push(format!("cpu gate updates per fast pass by changed wire: {}", top.join(", ")));
            }
        }
        if let Some(core) = self.memory_core() {
            let e = core.model();
            let s = &e.stats;
            lines.push(format!("memory: host served {} of {} steps (I/O-page fetches and reads), {} fetched bytes kept as HOST records; {} DMA lines placed by their own pass, {} second-port passes, {} result passes after the host's bytes; {} steps asked again under their accesses' own bus state; {} records kept, {} dense oracle checks ({} copies from ROM past 2 MiB); {} cells in the cache", core.host_served_steps, core.steps, s.host_fetches, core.dma_passes, core.second_passes, core.result_passes, core.access_overrides, s.records, s.oracle_checks, s.oracle_high_rom, e.memory.as_ref().map_or(0, |m| m.kv.len())));
            lines.push(format!("dma: {} bytes kept as copy-mode records, sources read from the model's memory ({} served by the host); which bytes to copy and when comes from the control device, the model's with control in weights", s.copies, s.host_copies));
            if detailed {
                lines.push(format!("steps: {} passes; {:.2} µs per pass", s.passes, wall * 1e6 / s.passes.max(1) as f64));
                lines.push(s.to_string());
            }
            let census = e.census();
            if !census.is_empty() {
                let top: Vec<String> = census.iter().map(|(n, u)| format!("{n} {u:.0}")).collect();
                lines.push(format!("gate updates per fast pass by node family: {}", top.join(", ")));
            }
            let sources = e.sources();
            if !sources.is_empty() {
                let top: Vec<String> = sources.iter().map(|(n, u)| format!("{n} {u:.1}")).collect();
                lines.push(format!("gate updates per fast pass by changed wire: {}", top.join(", ")));
            }
        }
        lines
    }
}

/// One call on whichever core is built.
macro_rules! each {
    ($core:expr, $c:ident => $e:expr) => {
        match $core {
            AnyCore::Host($c) => $e,
            AnyCore::Cpu($c) => $e,
            AnyCore::CpuLockstep($c) => $e,
            AnyCore::CpuMemory { core: $c, .. } => $e,
            AnyCore::CpuLockstepMemory { core: $c, .. } => $e,
        }
    };
}

impl Core for AnyCore {
    fn request(&mut self, cpu: &Cpu, inp: &Inputs) -> Request {
        each!(self, c => Core::request(c, cpu, inp))
    }

    fn execute(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2]) -> Outcome {
        each!(self, c => Core::execute(c, cpu, inp, data))
    }

    fn execute_on(&mut self, cpu: &Cpu, inp: &Inputs, data: [u8; 2], bus: &impl Bus) -> Outcome {
        each!(self, c => Core::execute_on(c, cpu, inp, data, bus))
    }

    fn land_write(&mut self, i: usize) {
        each!(self, c => Core::land_write(c, i))
    }

    fn facts(&self) -> &StepFacts {
        each!(self, c => Core::facts(c))
    }

    fn serve_fetch(&mut self, cpu: &Cpu, inp: &Inputs, bytes: [u8; 3]) -> Request {
        each!(self, c => Core::serve_fetch(c, cpu, inp, bytes))
    }

    fn oam_bug(&mut self, row: u8, read: bool) -> Vec<(u32, [u8; 8])> {
        each!(self, c => Core::oam_bug(c, row, read))
    }

    fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        each!(self, c => Core::load_bus(c, ram))
    }

    fn audit(&self, bus: &impl Bus, cpu: &Cpu, inp: &Inputs, out: &Outcome) {
        match self {
            AnyCore::Host(c) => c.audit(bus, cpu, inp, out),
            AnyCore::Cpu(c) => c.audit(bus, cpu, inp, out),
            AnyCore::CpuLockstep(c) => c.audit(bus, cpu, inp, out),
            AnyCore::CpuMemory { core, audit: true } => core.audit(bus, cpu, inp, out),
            AnyCore::CpuLockstepMemory { core, audit: true } => core.audit(bus, cpu, inp, out),
            AnyCore::CpuMemory { audit: false, .. } | AnyCore::CpuLockstepMemory { audit: false, .. } => {}
        }
    }

    /// Keeps every DMA byte as the model's own copy. Each 8-byte line pair is placed by a
    /// memory pass of its own (`MemCheckpointCore::dma_line`), which translates the line's
    /// source and destination under the registers the model holds: the bus map, VRAM's
    /// bank and the cartridge RAM's mask are the weights'. Byte k of the line is the line's
    /// address with k's bits concatenated. A source in the clock's port takes the host's
    /// byte; one in another port is refused, since its bytes can name different devices.
    /// Under memory lockstep, holds each placement to the specification's translation and
    /// the byte the model read to the byte the host moved.
    fn copied(&mut self, bytes: &[Copied]) {
        let (core, audit) = match self {
            AnyCore::CpuMemory { core, audit } => (core, *audit),
            AnyCore::CpuLockstepMemory { core, audit } => (&mut core.model, *audit),
            AnyCore::Host(_) | AnyCore::Cpu(_) | AnyCore::CpuLockstep(_) => return,
        };
        let spec = audit.then(|| core.model().memory.as_ref().expect("a memory checkpoint has a cache").image.banking());
        let mut placed: Option<([u16; 2], [u32; 2])> = None;
        for c in bytes {
            // A fill (CGB OAM DMA from E000 up) has no source: only its destination is placed,
            // and its byte is the FF block's, the bus driven by nothing.
            let from_bus = c.source.unwrap_or(c.dest);
            let lines = [from_bus & !7, c.dest & !7];
            let at = match placed {
                Some((l, at)) if l == lines => at,
                _ => {
                    let at = core.dma_line(from_bus, c.dest);
                    placed = Some((lines, at));
                    at
                }
            };
            let dest = at[1] | (c.dest & 7) as u32;
            let source = c.source.map_or(layout::FF | (dest & 0x7F), |s| at[0] | (s & 7) as u32);
            if let Some(b) = &spec {
                let want_source = c.source.map_or(source, |s| layout::translate(b, s, Access::Read, false));
                assert_eq!([source, dest], [want_source, layout::translate(b, c.dest, Access::Write, false)], "DMA copy {:?} -> {:#06x}: the model placed it at {source:#08x} -> {dest:#08x}, which the specification does not", c.source, c.dest);
            }
            let from = match c.source.and_then(|_| layout::port(at[0])) {
                None => CopyFrom::Memory(source),
                Some(Port::Clock(_)) => CopyFrom::Host(c.byte.unwrap_or_else(|| panic!("DMA copy {:?} -> {:#06x}: a source in the clock's port, whose byte only the host's machine reads, with that machine dropped; refused", c.source, c.dest))),
                Some(p) => panic!("DMA copy {:?} -> {:#06x}: the source line is {p:?}'s port, whose bytes the line's translation cannot place one by one; refused", c.source, c.dest),
            };
            // A record still waiting for its write's M-cycle would land over this byte, or
            // the copy would read the line before the write: neither can happen while the
            // transfer blocks the write (docs/hardware-tests.md §4.1), and this holds it.
            assert!(!core.staged_in_line(dest) && !matches!(from, CopyFrom::Memory(s) if core.staged_in_line(s)), "DMA copy {:?} -> {:#06x}: a write record of the step not kept yet shares its line", c.source, c.dest);
            let got = core.model_mut().copy(from, dest);
            if audit {
                assert_eq!(Some(got), c.byte, "DMA copy {source:#08x} -> {dest:#08x}: the model read {got:#04x}, the host moved {:?}", c.byte);
            }
        }
    }

    /// Memory lockstep over whole regions: the model's memory image (the newest record
    /// per address, which the index holds) byte for byte against the host's arrays.
    fn take_save(&mut self, committed: bool, len: usize) -> Option<Option<Vec<u8>>> {
        self.own_save(committed, len)
    }

    /// Work RAM and high RAM from the model's memory: C000 and FF80 at their regions'
    /// bases, D000 at the bank window the model emitted on its last step (`MAPW`); the host's
    /// arrays before the first step, which they then equal.
    fn peek_ram(&self, addr: u16) -> Option<u8> {
        let core = self.memory_core().filter(|c| c.steps > 0)?;
        let image = &core.model().memory.as_ref()?.image;
        let at = match addr {
            0xC000..=0xCFFF => layout::WRAM + (addr as u32 & 0xFFF),
            0xD000..=0xDFFF => core.evidence.map[3] + (addr as u32 & 0xFFF),
            0xFF80..=0xFFFE => layout::HRAM + (addr as u32 - 0xFF80),
            _ => return Some(0xFF),
        };
        Some(image.read(at))
    }

    fn audit_memory(&self, host: &[(Addr, &[u8])]) {
        let (AnyCore::CpuMemory { audit: true, .. } | AnyCore::CpuLockstepMemory { audit: true, .. }) = self else { return };
        let engine = self.engine().expect("a memory core has an engine");
        let image = &engine.memory.as_ref().expect("a memory checkpoint has a cache").image;
        for &(base, bytes) in host {
            let model = image.region(base, bytes.len());
            if model == bytes {
                continue;
            }
            if let Some(i) = (0..bytes.len()).find(|&i| model[i] != bytes[i]) {
                let differ = (0..bytes.len()).filter(|&i| model[i] != bytes[i]).count();
                panic!("memory audit: {:#08x} holds {:#04x} in the model's memory and {:#04x} on the host ({differ} of the region's {} bytes at {base:#08x} differ; {} records kept)", base + i as u32, model[i], bytes[i], bytes.len(), engine.stats.records);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Backend::{Host, Lockstep, Model};
    use super::Component::{Apu, Control, Cpu, Memory, Ppu};
    use super::*;

    #[test]
    fn step_oracle_cadence_per_engine() {
        assert_eq!(StepOracle::parse("5000"), Ok(StepOracle { cpu: 5000, apu: 5000, control: 5000, ppu: 5000, rom: 0 }));
        assert_eq!(StepOracle::parse("100, memory=150000 ,ppu=8000"), Ok(StepOracle { cpu: 150000, apu: 100, control: 100, ppu: 8000, rom: 0 }));
        assert_eq!(StepOracle::parse("apu=7"), Ok(StepOracle { cpu: 0, apu: 7, control: 0, ppu: 0, rom: 0 }));
        assert_eq!(StepOracle::parse("rom=16,cpu=10000"), Ok(StepOracle { cpu: 10000, apu: 0, control: 0, ppu: 0, rom: 16 }));
        assert!(StepOracle::parse("gpu=7").unwrap_err().contains("unknown component"));
        assert!(StepOracle::parse("cpu=x").unwrap_err().contains("not a count"));
    }

    /// Every stack: 3 backends for each of 5 components.
    fn every_stack() -> Vec<Stack> {
        (0..3u32.pow(Component::ALL.len() as u32))
            .map(|mut n| {
                let mut s = Stack::HOST;
                for c in Component::ALL {
                    s = s.with(c, [Host, Model, Lockstep][(n % 3) as usize]);
                    n /= 3;
                }
                s
            })
            .collect()
    }

    #[test]
    fn a_stack_resolves_exactly_when_its_weights_are_built_checkpoints() {
        let stacks = every_stack();
        assert_eq!(stacks.len(), 243);
        let mut ok = 0;
        for s in stacks {
            let (apu, control) = (s.in_weights(Apu), s.in_weights(Control));
            let core: Vec<Component> = s.weights().into_iter().filter(|&c| c != Apu && c != Control).collect();
            let want = match core.as_slice() {
                [] => Ok(Checkpoints { core: Weights::None, apu, control, ppu: false, one_model: false }),
                [Cpu] => Ok(Checkpoints { core: Weights::Cpu, apu, control, ppu: false, one_model: false }),
                [Cpu, Memory] => Ok(Checkpoints { core: Weights::CpuMemory, apu, control, ppu: false, one_model: false }),
                [Cpu, Memory, Ppu] => Ok(Checkpoints { core: Weights::CpuMemory, apu, control, ppu: true, one_model: false }),
                w if w.contains(&Memory) => Err(StackError::Requires { component: Memory, requires: Cpu, because: "the memory heads live in the CPU checkpoint" }),
                _ => Err(StackError::Requires { component: Ppu, requires: Memory, because: "its VRAM and OAM follow the memory checkpoint's write records" }),
            };
            assert_eq!(s.resolve(), want, "{s}");
            ok += want.is_ok() as usize;
        }
        assert_eq!(ok, (1 + 2 + 4 + 8) * 3 * 3, "host; cpu model or lockstep; cpu x memory each model or lockstep; cpu x memory x ppu; each with the apu and the control logic on the host, model or lockstep");
    }

    #[test]
    fn every_illegal_combination_names_its_rule() {
        let err = |w: &str| Stack::parse(w, None).unwrap().resolve().unwrap_err().to_string();
        assert_eq!(err("memory"), "memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)");
        assert_eq!(err("memory,apu"), "memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)");
        let ppu = "ppu in weights requires memory in weights (its VRAM and OAM follow the memory checkpoint's write records)";
        assert_eq!(err("ppu"), ppu);
        assert_eq!(err("cpu,ppu"), ppu);
        assert_eq!(err("cpu,ppu,apu,control"), ppu);
        assert_eq!(err("memory,ppu"), "memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)");
        assert_eq!(err("memory,control"), "memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)");
    }

    #[test]
    fn every_legal_combination_resolves() {
        let ok = |w: &str, l: Option<&str>| Stack::parse(w, l).unwrap().resolve().unwrap();
        let cp = |core, apu| Checkpoints { core, apu, control: false, ppu: false, one_model: false };
        let with_control = |core, apu| Checkpoints { core, apu, control: true, ppu: false, one_model: false };
        let everything = Checkpoints { core: Weights::CpuMemory, apu: true, control: true, ppu: true, one_model: false };
        assert_eq!(ok("cpu,memory,ppu", Some("ppu")), Checkpoints { ppu: true, ..cp(Weights::CpuMemory, false) });
        assert_eq!(ok("cpu,memory,apu,ppu", Some("all")), Checkpoints { ppu: true, ..cp(Weights::CpuMemory, true) });
        assert_eq!(ok("all", Some("all")), everything);
        assert_eq!(ok("control", None), with_control(Weights::None, false), "the control checkpoint runs with the CPU on the host");
        assert_eq!(ok("control", Some("control")), with_control(Weights::None, false));
        assert_eq!(ok("cpu,memory,apu,control", Some("all")), with_control(Weights::CpuMemory, true));
        assert_eq!(ok("none", None), cp(Weights::None, false));
        assert_eq!(ok("cpu", None), cp(Weights::Cpu, false));
        assert_eq!(ok("cpu", Some("cpu")), cp(Weights::Cpu, false));
        assert_eq!(ok("cpu,memory", None), cp(Weights::CpuMemory, false));
        assert_eq!(ok("cpu,memory", Some("cpu")), cp(Weights::CpuMemory, false));
        assert_eq!(ok("cpu,memory", Some("memory")), cp(Weights::CpuMemory, false));
        assert_eq!(ok("cpu,memory", Some("all")), cp(Weights::CpuMemory, false));
        assert_eq!(ok("apu", None), cp(Weights::None, true), "the APU checkpoint runs with the CPU on the host");
        assert_eq!(ok("apu", Some("apu")), cp(Weights::None, true));
        assert_eq!(ok("cpu,apu", Some("all")), cp(Weights::Cpu, true));
        assert_eq!(ok("cpu,memory,apu", Some("all")), cp(Weights::CpuMemory, true));
        assert_eq!(ok("built", Some("all")), everything);
    }

    #[test]
    fn the_old_core_names_map_onto_stacks() {
        let stack = |w, l| Stack::parse(w, Some(l)).unwrap();
        assert_eq!(Stack::alias("ref"), Some(Stack::HOST));
        assert_eq!(Stack::alias("model"), Some(stack("cpu", "none")));
        assert_eq!(Stack::alias("lockstep"), Some(stack("cpu", "cpu")));
        assert_eq!(Stack::alias("model-mem"), Some(stack("cpu,memory", "memory")), "model-mem always held the model's memory to the host's bytes");
        assert_eq!(Stack::alias("lockstep-mem"), Some(stack("cpu,memory", "cpu,memory")));
        assert_eq!(Stack::alias("fast"), None);
    }

    #[test]
    fn lockstep_needs_the_component_in_weights() {
        assert_eq!(Stack::parse("cpu", Some("memory")).unwrap_err(), "--lockstep memory needs memory in --weights");
        assert_eq!(Stack::parse("none", Some("cpu")).unwrap_err(), "--lockstep cpu needs cpu in --weights");
        assert_eq!(Stack::parse("cpu", Some("apu")).unwrap_err(), "--lockstep apu needs apu in --weights");
        assert_eq!(Stack::parse("cpu", Some("all")).unwrap().lockstepped(), [Cpu], "all ranges over --weights, not every component");
        assert_eq!(Stack::parse("cpu,memory", Some("none")).unwrap().lockstepped(), []);
    }

    #[test]
    fn unknown_names_are_refused_with_the_list() {
        assert_eq!(Stack::parse("cpu,gpu", None).unwrap_err(), "--weights gpu: unknown component; the components are cpu, memory, ppu, apu, control");
        assert_eq!(Stack::parse("cpu", Some("ram")).unwrap_err(), "--lockstep ram: unknown component; the components are cpu, memory, ppu, apu, control");
    }

    #[test]
    fn the_label_names_every_component_once() {
        let s = Stack::parse("cpu,memory", Some("memory")).unwrap();
        assert_eq!(s.to_string(), "in weights: cpu, memory; on the host: ppu, apu, control; lockstep: memory");
        assert_eq!(s.flags(), "weights=cpu,memory lockstep=memory");
        let s = Stack::parse("cpu,memory,apu", Some("apu")).unwrap();
        assert_eq!(s.to_string(), "in weights: cpu, memory, apu; on the host: ppu, control; lockstep: apu");
        assert_eq!(Stack::HOST.to_string(), "in weights: none; on the host: cpu, memory, ppu, apu, control; lockstep: none");
        assert_eq!(Stack::HOST.flags(), "weights=none lockstep=none");
        assert_eq!(Stack::parse("all", Some("all")).unwrap().hosted(), [], "all is every component, built or not");
        assert_eq!(Stack::parse("built", None).unwrap().weights(), [Cpu, Memory, Ppu, Apu, Control]);
        assert_eq!(Stack::parse("cpu,memory,ppu", None).unwrap().to_string(), "in weights: cpu, memory, ppu; on the host: apu, control; lockstep: none");
        assert_eq!(
            Stack::parse("cpu,control", Some("control")).unwrap().to_string(),
            "in weights: cpu, control; on the host: memory, ppu, apu; lockstep: control; control still on the host: the cartridge banking registers"
        );
        assert_eq!(Stack::parse("cpu,memory,apu,control", None).unwrap().to_string(), "in weights: cpu, memory, apu, control; on the host: ppu; lockstep: none");
        assert_eq!(
            Stack::parse("built", None).unwrap().to_string(),
            "in weights: cpu, memory, ppu, apu, control; on the host: none; lockstep: none; the host besides: presentation (each pixel's colour mapped to the screen's RGB, the board's high-pass filter over the APU's mix, the samples played, the frame pacing, battery saves written), input (the buttons, and the wall-clock time an MBC3 cartridge's clock counts, handed over as the time elapsed), and scheduling (M-cycles counted between passes and down to the instants the control model emitted for each scheduled mode-3 start and frame-sequencer edge, the 48 kHz sample instants and the frame time counted in real time at the halves per M-cycle the control model emits, the speed and the frame-sequencer edges the control model emits handed to the APU, the GBC CPU's passes skipped while it is halted and the control model's wake bit is clear, each data access of the GBC CPU to an I/O register handed to the device whose port the memory model's emitted address names (the model decides the device, the host moves the byte), each DMA byte placed at the address bits the control model emitted and the CPU stopped for the M-cycles it names, and each model's outputs handed to the next pass or the next model), and on the host's own GBC rules still (docs/host-in-weights.md §5): the cartridge's mapper writes (which the clock's decode reads), and without a boot ROM the header's CGB byte and the post-boot state, a snapshot handed over as data"
        );
        assert_eq!(
            Stack::parse("cpu,memory", None).unwrap().with_one_model(true).to_string(),
            "in weights: cpu, memory; on the host: ppu, apu, control; lockstep: none; one checkpoint: every component in weights runs as a stream of the same weights (docs/one-model.md)"
        );
    }

    #[test]
    fn one_model_needs_memory_in_weights() {
        let one = |w: &str| Stack::parse(w, None).unwrap().with_one_model(true).resolve();
        let refused = "--one-model needs memory in weights (the one checkpoint's CPU stream holds the memory heads): give --weights cpu,memory or more";
        assert_eq!(one("none").unwrap_err().to_string(), refused, "with nothing in weights there is no stream to run");
        assert_eq!(one("cpu").unwrap_err().to_string(), refused);
        assert_eq!(one("apu,control").unwrap_err().to_string(), refused);
        assert_eq!(one("memory").unwrap_err().to_string(), "memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)", "the dependency rules come first");
        let all = Checkpoints { core: Weights::CpuMemory, apu: true, control: true, ppu: true, one_model: true };
        assert_eq!(one("all"), Ok(all));
        assert_eq!(one("cpu,memory"), Ok(Checkpoints { core: Weights::CpuMemory, apu: false, control: false, ppu: false, one_model: true }));
    }

    #[test]
    fn one_model_with_nothing_in_weights_is_refused_before_compiling() {
        let session = Session::new(vec![0; 0x8000], None, crate::machine::Hardware::Dmg).unwrap();
        let err = Stack::HOST.with_one_model(true).build(&session, Instrumentation::default()).err();
        assert_eq!(err.as_deref(), Some("--one-model needs memory in weights (the one checkpoint's CPU stream holds the memory heads): give --weights cpu,memory or more"));
    }

    #[test]
    fn everything_on_the_host_builds_the_reference_without_a_checkpoint() {
        let session = Session::new(vec![0; 0x8000], None, crate::machine::Hardware::Dmg).unwrap();
        let built = Stack::HOST.build(&session, Instrumentation::default()).unwrap();
        assert!(matches!(built.core, AnyCore::Host(_)));
        assert!(built.apu.is_none(), "the machine keeps its reference APU");
        assert!(built.control.is_none(), "the machine keeps its reference timer and interrupt flags");
        assert!(built.core.engine().is_none());
        assert_eq!(built.core.describe(), Vec::<String>::new());
        assert_eq!(built.core.report(1.0, true), Vec::<String>::new());
        let mut session = session.with_stack(Stack::HOST, Instrumentation::default()).unwrap();
        assert!(session.apu_reference().is_some());
        assert_eq!((session.describe(), session.report(1.0, true)), (Vec::<String>::new(), Vec::<String>::new()));
    }
}
