use gbcformer::frontend::{self, FPS, PlayOptions};
use gbcformer::machine::Hardware;
use gbcformer::machine::apu::Sample;
use gbcformer::output::{Recorder, frame_sha1, write_ppm, write_wav};
use gbcformer::press::{Press, PressScript};
use gbcformer::save;
use gbcformer::session::{Event, Session};
use gbcformer::stack::{AnyCore, Instrumentation, Stack, StepOracle};
use gbcformer::step::Core;
use gbcformer::transport::Outputs;
use gbcformer::{reference::Reference, rom, sst};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: gbcformer rom <file>
       gbcformer sst [opcode-prefix]
       gbcformer play <rom> [--weights LIST] [--lockstep LIST] [--hardware dmg|cgb|auto] [--frames N] [--record DIR [--record-every N]] [--mute] [--no-save] [--one-model] [--boot FILE]
       gbcformer run <rom> --frames N [--weights LIST] [--lockstep LIST] [--warmup FRAMES] [--press FRAME:BUTTON[:HOLD]]... [--dump out.ppm] [--record DIR [--record-every N]] [--no-save] [--wav out.wav] [--hardware dmg|cgb|auto] [--one-model] [--boot FILE]
hardware: auto (default) = CGB when header byte 0x143 has bit 7 set, else DMG
boot:     a boot ROM file (yours; never part of the repository) run from power-on before the game
weights:  the components the compiled Llama checkpoints compute; the rest run on the host. none (default), cpu, built (every
          component built so far), all, or names separated by commas: cpu, memory, ppu, apu, control. cpu and memory compile
          into one checkpoint, ppu, apu and control into their own. memory needs cpu, ppu needs memory. The run's
          label names what a component in weights still leaves on the host.
lockstep: components in --weights that the host also runs beside the model, the first divergence stopping the run: none
          (default), all (every component in --weights) or names.
core:     the old names, as aliases: ref = --weights none, model = --weights cpu, lockstep = --weights cpu --lockstep cpu,
          model-mem = --weights cpu,memory --lockstep memory, lockstep-mem = --weights cpu,memory --lockstep all
buttons: up down left right a b start select";

fn main() {
    // The machine's thread runs the GBC CPU's passes, the slowest stream.
    gbcformer::qos::interactive();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = gbcformer::sm83::refuse_retired_switches() {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let result = match args.first().map(String::as_str) {
        Some("rom") => rom_info(&args[1..]),
        Some("sst") => {
            let filter = args.get(1).map_or("", String::as_str);
            let mut core = Reference;
            let (cases, failed) = sst::run_all(&mut core, Path::new("vendor/sm83/v1"), filter, true);
            println!("reference: {cases} cases, {failed} failed");
            std::process::exit((failed > 0) as i32);
        }
        Some("play") => Options::parse(Command::Play, &args[1..]).and_then(|o| play(&o)),
        Some("run") => Options::parse(Command::Run, &args[1..]).and_then(|o| run(&o)),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// Compiles the stack's checkpoints and hands `warm` over to them, printing what was
/// built and the label of what runs where.
fn build(stack: Stack, warm: Session<Reference>) -> Result<Session<AnyCore>, String> {
    let instrumentation = Instrumentation {
        timed: std::env::var_os("GBCFORMER_STATS").is_some(),
        oracle_every: std::env::var("GBCFORMER_ORACLE").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
        step_oracle: std::env::var("GBCFORMER_STEP_ORACLE").map_or(Ok(StepOracle::default()), |s| StepOracle::parse(&s))?,
        apu_same_thread: std::env::var("GBCFORMER_APU_THREAD").is_ok_and(|v| v == "0"),
        ppu_same_thread: std::env::var("GBCFORMER_PPU_THREAD").is_ok_and(|v| v == "0"),
        rows_same_thread: std::env::var("GBCFORMER_ROWS_THREAD").is_ok_and(|v| v == "0"),
    };
    let t = Instant::now();
    let frame = warm.frame_number();
    if let Some(dir) = std::env::var_os("GBCFORMER_SNAPSHOT") {
        gbcformer::gbc_wiring::write_snapshot(&warm, std::path::Path::new(&dir))?;
    }
    let s = warm.with_stack(stack, instrumentation)?;
    let lines = s.describe();
    for line in &lines {
        eprintln!("{line}");
    }
    if !lines.is_empty() {
        eprintln!("compiled at frame {frame} in {:.1}s", t.elapsed().as_secs_f64());
    }
    eprintln!("{stack}");
    Ok(s)
}

fn rom_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or(USAGE)?;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let header = rom::identify(&bytes)?;
    println!("{header:?}");
    if header.is_known_yellow() {
        println!("known good: Pokémon Yellow, the release GBCformer is measured against");
    }
    println!("cartridge: {:?}", header.kind()?);
    Ok(())
}

/// The two commands that take `Options`. Some flags belong to one of them only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Play,
    Run,
}

impl Command {
    fn name(self) -> &'static str {
        match self {
            Command::Play => "play",
            Command::Run => "run",
        }
    }
}

struct Options {
    rom: PathBuf,
    frames: Option<u64>,
    presses: PressScript,
    dump: Option<PathBuf>,
    /// Frames run on the selected core, written as frame_NNNNN.ppm (`--record DIR`),
    /// every Nth frame with `--record-every N`.
    record: Option<Recorder>,
    save: bool,
    /// Which components run in weights, on the host, or as both.
    stack: Stack,
    /// Frames run on the reference interpreter before switching to `stack`.
    warmup: u64,
    /// None: CGB when the header's 0x143 bit 7 is set, else DMG.
    hardware: Option<Hardware>,
    /// `run`: write all generated audio here as a WAV file.
    wav: Option<PathBuf>,
    /// `play`: no audio output.
    mute: bool,
    /// A boot ROM file to run first, from power-on (`--boot`); None starts at the hand-off.
    boot: Option<PathBuf>,
}

/// Where battery saves go, and how many the session handed over.
struct Saves {
    path: Option<PathBuf>,
    count: u32,
    outputs: Option<Outputs>,
}

impl Saves {
    /// Save errors are reported, not fatal: the previous save file stays intact.
    /// Persists a save; with `record`, also into the outputs, which hold the model's frames alone.
    fn handle(&mut self, event: Event, record: bool) {
        match event {
            Event::Save(bytes) => {
                self.count += 1;
                if let Some(o) = self.outputs.as_mut().filter(|_| record) {
                    o.save(&bytes).unwrap();
                }
                if let Some(path) = &self.path
                    && let Err(e) = save::write_atomic(path, &bytes)
                {
                    eprintln!("save to {} failed: {e}", path.display());
                }
            }
            other => eprintln!("{other}"),
        }
    }

    fn shut_down<C: Core>(&mut self, session: &mut Session<C>) {
        if let Some(bytes) = session.shut_down() {
            self.handle(Event::Save(bytes), true);
        }
    }

    /// Shuts the session down whatever `result` the window loop ended with, so a window
    /// error never loses unsaved battery RAM.
    fn finish<C: Core>(&mut self, session: &mut Session<C>, result: Result<(), String>) -> Result<(), String> {
        self.shut_down(session);
        result
    }
}

impl Options {
    fn parse(cmd: Command, args: &[String]) -> Result<Self, String> {
        let mut it = args.iter();
        let rom = PathBuf::from(it.next().ok_or(USAGE)?);
        let mut o = Options { rom, frames: None, presses: PressScript::default(), dump: None, record: None, save: true, stack: Stack::HOST, warmup: 0, hardware: None, wav: None, mute: false, boot: None };
        let mut record_every = None;
        let (mut core, mut weights, mut lockstep, mut one_model) = (None, None, None, false);
        while let Some(flag) = it.next() {
            let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
            let only = |owner: Command| if cmd == owner { Ok(()) } else { Err(format!("{} does not take {flag}, a {} flag", cmd.name(), owner.name())) };
            match flag.as_str() {
                "--frames" => o.frames = Some(value()?.parse().map_err(|e| format!("--frames: {e}"))?),
                "--press" => {
                    only(Command::Run)?;
                    o.presses.push(Press::parse(value()?).map_err(|e| format!("--press {e}"))?)
                }
                "--dump" => {
                    only(Command::Run)?;
                    o.dump = Some(PathBuf::from(value()?))
                }
                "--record" => o.record = Some(Recorder { dir: PathBuf::from(value()?), every: 1 }),
                "--record-every" => record_every = Some(value()?.parse().ok().filter(|&n| n > 0).ok_or("--record-every expects a positive number")?),
                "--no-save" => o.save = false,
                "--boot" => o.boot = Some(PathBuf::from(value()?)),
                "--one-model" => one_model = true,
                "--wav" => {
                    only(Command::Run)?;
                    o.wav = Some(PathBuf::from(value()?))
                }
                "--mute" => {
                    only(Command::Play)?;
                    o.mute = true
                }
                "--core" => core = Some(value()?.clone()),
                "--weights" => weights = Some(value()?.clone()),
                "--lockstep" => lockstep = Some(value()?.clone()),
                "--warmup" => {
                    only(Command::Run)?;
                    o.warmup = value()?.parse().map_err(|e| format!("--warmup: {e}"))?
                }
                "--hardware" => {
                    o.hardware = match value()?.as_str() {
                        "dmg" => Some(Hardware::Dmg),
                        "cgb" => Some(Hardware::Cgb),
                        "auto" => None,
                        other => return Err(format!("--hardware {other}: expected dmg, cgb or auto")),
                    }
                }
                _ => return Err(format!("unknown flag {flag}\n{USAGE}")),
            }
        }
        match (&mut o.record, record_every) {
            (Some(r), Some(every)) => r.every = every,
            (None, Some(_)) => return Err("--record-every needs --record DIR".into()),
            _ => {}
        }
        o.stack = match (core, weights, lockstep) {
            (Some(_), Some(_), _) | (Some(_), _, Some(_)) => return Err("--core is an alias for --weights and --lockstep; give one or the other".into()),
            (Some(core), None, None) => Stack::alias(&core).ok_or_else(|| format!("unknown core {core}\n{USAGE}"))?,
            (None, weights, lockstep) => Stack::parse(weights.as_deref().unwrap_or("none"), lockstep.as_deref())?,
        }
        .with_one_model(one_model);
        o.stack.resolve().map_err(|e| e.to_string())?;
        Ok(o)
    }

    /// Loads the ROM and its battery save; refuses cartridges the host cannot emulate.
    fn session(&self) -> Result<(Session<Reference>, Saves), String> {
        let bytes = std::fs::read(&self.rom).map_err(|e| format!("{}: {e}", self.rom.display()))?;
        let header = rom::identify(&bytes)?;
        let kind = header.kind()?;
        let known = if header.is_known_yellow() { " (known good)" } else { "" };
        eprintln!("{:?} sha1 {}{known} {kind:?}", header.title, header.sha1);
        let path = (self.save && kind.battery).then(|| save::path_for(&self.rom));
        let saved_ram = path.as_deref().map(save::load).transpose()?.flatten();
        let boot = self.boot.as_ref().map(|path| std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))).transpose()?;
        // From a boot ROM the console is the one the boot ROM is for, not the header's pick.
        let hardware = match (&boot, self.hardware) {
            (Some(b), None) => Hardware::of_boot(b).ok_or_else(|| format!("a boot ROM is 256 bytes (DMG) or 2,304 (CGB); this one is {}", b.len()))?,
            _ => Hardware::resolve(&header, self.hardware)?,
        };
        eprintln!("hardware: {hardware:?}");
        let session = match boot {
            Some(boot) => {
                eprintln!("boot ROM: {} ({} bytes), run from power-on", self.boot.as_ref().expect("a boot path").display(), boot.len());
                Session::power_on(bytes, saved_ram, hardware, boot)
            }
            None => Session::new(bytes, saved_ram, hardware),
        }
        .map_err(|e| format!("{}: {e}", self.rom.display()))?;
        Ok((session, Saves { path, count: 0, outputs: outputs_from_env() }))
    }
}

/// Runs frames until frame `to` with the scripted presses, draining the audio every frame
/// and keeping it for `--wav`.
fn run_frames<C: Core>(s: &mut Session<C>, saves: &mut Saves, o: &Options, to: u64, audio: &mut Vec<Sample>, record: bool) {
    while s.frame_number() < to {
        let frame = s.frame_number();
        s.set_time(wall_clock(frame));
        for event in s.run_frame(o.presses.buttons_at(frame)) {
            saves.handle(event, record);
        }
        let drained = audio.len();
        s.drain_audio(audio);
        if let Some(out) = saves.outputs.as_mut().filter(|_| record) {
            out.audio(&audio[drained..]).unwrap();
            out.frame(s.frame()).unwrap();
        }
        if o.wav.is_none() {
            audio.clear();
        }
        if let Some(r) = o.record.as_ref().filter(|_| record) {
            r.record(frame, s.frame());
        }
    }
}

fn run(o: &Options) -> Result<(), String> {
    let frames = o.frames.ok_or("run needs --frames N")?;
    let warmup = o.warmup.min(frames);
    let (mut warm, mut saves) = o.session()?;
    let mut audio = Vec::new();
    if warmup > 0 {
        run_frames(&mut warm, &mut saves, o, warmup, &mut audio, false);
        eprintln!("warmup: frames 0..{warmup} ran on the reference interpreter, not the selected stack ({} steps)", warm.counters().steps);
    }
    let mut s = build(o.stack, warm)?;
    let start = Instant::now();
    run_frames(&mut s, &mut saves, o, frames, &mut audio, true);
    // Inside the timed run: an APU on its own thread finishes the last frames here.
    let drained = audio.len();
    s.finish_audio(&mut audio);
    s.settle();
    if let Some(out) = &mut saves.outputs {
        out.audio(&audio[drained..]).unwrap();
    }
    if o.wav.is_none() {
        audio.clear();
    }
    let ran = frames - warmup;
    // Inside the timed run too: the last frame, which waits for a PPU on its own thread.
    let sha = frame_sha1(s.frame());
    let elapsed = start.elapsed().as_secs_f64();
    saves.shut_down(&mut s);
    if !s.serial_output().is_empty() {
        println!("serial: {}", String::from_utf8_lossy(s.serial_output()).trim_end());
    }
    let c = s.counters();
    println!(
        "frames={ran} {} saves={} steps={} step_mcycles={} halt_mcycles={} hdma_mcycles={} pause_mcycles={} dma_blocked={} palette_reads={} stepped_lines={} compat_closes={} forwarded={} revised_before={} taken_after={} oam_bugs={} frame_sha1={}",
        o.stack.flags(),
        saves.count,
        c.steps,
        c.step_mcycles,
        c.halt_mcycles,
        c.hdma_mcycles,
        c.pause_mcycles,
        c.dma_blocked,
        c.palette_reads,
        c.midline.stepped,
        c.midline.compat_closes,
        c.midline.forwarded,
        c.midline.revised_before,
        c.midline.taken_after,
        c.oam_bugs,
        sha
    );
    eprintln!("{:.1} emulated fps ({:.2}x DMG speed) over {elapsed:.2}s", ran as f64 / elapsed, ran as f64 / elapsed / FPS);
    for line in s.report(elapsed, std::env::var_os("GBCFORMER_STATS").is_some()) {
        eprintln!("{line}");
    }
    // With GBCFORMER_UNITS set, the model's changes per fast pass by wire, one
    // "name changes" line each, the record `sm83::changes` reads.
    if let (Some(path), Some(e)) = (std::env::var_os("GBCFORMER_WIRE_CHANGES"), s.core().engine()) {
        let table: String = e.wire_changes().iter().map(|(n, f)| format!("{n} {f:.4}\n")).collect();
        std::fs::write(&path, table).map_err(|e| format!("{}: {e}", path.to_string_lossy()))?;
    }
    if let Some(path) = &o.dump {
        write_ppm(path, s.frame())?;
    }
    if let Some(path) = &o.wav {
        write_wav(path, &audio)?;
    }
    Ok(())
}

fn play(o: &Options) -> Result<(), String> {
    let (s, mut saves) = o.session()?;
    let mut s = build(o.stack, s)?;
    let options = PlayOptions { frames: o.frames, mute: o.mute, record: o.record.clone() };
    let result = frontend::play(&mut s, &options, |e| saves.handle(e, true));
    // As `run` does before it reports: a memo check that failed on the rows' worker stops
    // the process here, before the last save is written.
    s.settle();
    saves.finish(&mut s, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gbcformer::machine::io::Buttons;

    /// An MBC5 battery cartridge whose program enables RAM, writes one byte and spins,
    /// leaving the save unfinished (RAM still enabled).
    fn battery_rom() -> Vec<u8> {
        let mut rom = vec![0; 0x8000];
        rom[0x147] = 0x1B;
        rom[0x149] = 0x02;
        #[rustfmt::skip]
        let program = [
            0x3E, 0x0A, 0xEA, 0x00, 0x00, // LD A,0A; LD (0000),A   enable RAM
            0x3E, 0x5A, 0xEA, 0x00, 0xA0, // LD A,5A; LD (A000),A
            0x18, 0xFE,                   // JR -2
        ];
        rom[0x100..0x100 + program.len()].copy_from_slice(&program);
        rom
    }

    #[test]
    fn a_player_error_still_persists_the_unsaved_ram() {
        let dir = std::env::temp_dir().join(format!("gbcformer-play-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game.sav");
        let mut s = Session::new(battery_rom(), None, Hardware::Dmg).unwrap();
        assert_eq!(s.run_frame(Buttons(0)), [], "RAM stays enabled, so nothing is committed");
        let mut saves = Saves { path: Some(path.clone()), count: 0, outputs: None };
        let result = saves.finish(&mut s, Err("window closed".into()));
        assert_eq!(result, Err("window closed".into()));
        assert_eq!(saves.count, 1);
        let saved = std::fs::read(&path).unwrap();
        assert_eq!((saved.len(), saved[0]), (0x2000, 0x5A));
    }

    fn parse(cmd: Command, args: &[&str]) -> Result<Options, String> {
        Options::parse(cmd, &args.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_flag_of_the_other_command_is_refused_by_name() {
        assert_eq!(parse(Command::Play, &["g.gbc", "--press", "1:a"]).err(), Some("play does not take --press, a run flag".into()));
        assert_eq!(parse(Command::Play, &["g.gbc", "--warmup", "3"]).err(), Some("play does not take --warmup, a run flag".into()));
        assert_eq!(parse(Command::Play, &["g.gbc", "--dump", "x.ppm"]).err(), Some("play does not take --dump, a run flag".into()));
        assert_eq!(parse(Command::Play, &["g.gbc", "--wav", "x.wav"]).err(), Some("play does not take --wav, a run flag".into()));
        assert_eq!(parse(Command::Run, &["g.gbc", "--mute"]).err(), Some("run does not take --mute, a play flag".into()));
        let o = parse(Command::Run, &["g.gbc", "--press", "1:a", "--warmup", "3", "--dump", "x.ppm", "--wav", "x.wav"]).unwrap();
        assert_eq!((o.warmup, o.dump, o.wav), (3, Some(PathBuf::from("x.ppm")), Some(PathBuf::from("x.wav"))));
        assert_eq!(o.presses.buttons_at(1).0, Buttons::A);
    }

    #[test]
    fn record_every_needs_record() {
        assert_eq!(parse(Command::Run, &["g.gbc", "--record-every", "2"]).err(), Some("--record-every needs --record DIR".into()));
        let o = parse(Command::Play, &["g.gbc", "--record", "out", "--record-every", "2"]).unwrap();
        assert_eq!(o.record.map(|r| (r.dir, r.every)), Some((PathBuf::from("out"), 2)));
    }

    #[test]
    fn the_stack_comes_from_weights_and_lockstep_or_from_a_core_alias() {
        let stack = |args: &[&str]| parse(Command::Run, &[&["g.gbc"], args].concat()).map(|o| o.stack);
        assert_eq!(stack(&[]), Ok(Stack::HOST));
        assert_eq!(stack(&["--weights", "cpu,memory", "--lockstep", "cpu"]), Stack::parse("cpu,memory", Some("cpu")));
        assert_eq!(stack(&["--core", "lockstep-mem"]), Stack::parse("cpu,memory", Some("all")));
        assert_eq!(stack(&["--core", "model", "--weights", "cpu"]).err(), Some("--core is an alias for --weights and --lockstep; give one or the other".into()));
        assert_eq!(stack(&["--core", "fast"]).err().map(|e| e.lines().next().unwrap().to_string()), Some("unknown core fast".into()));
        assert_eq!(stack(&["--weights", "ppu"]).err(), Some("ppu in weights requires memory in weights (its VRAM and OAM follow the memory checkpoint's write records)".into()));
        assert_eq!(stack(&["--weights", "memory"]).err(), Some("memory in weights requires cpu in weights (the memory heads live in the CPU checkpoint)".into()));
        assert_eq!(stack(&["--lockstep", "cpu"]).err(), Some("--lockstep cpu needs cpu in --weights".into()));
        assert_eq!(stack(&["--weights", "all", "--one-model"]), Ok(Stack::parse("all", None).unwrap().with_one_model(true)));
        let refused = Some("--one-model needs memory in weights (the one checkpoint's CPU stream holds the memory heads): give --weights cpu,memory or more".to_string());
        assert_eq!(stack(&["--one-model"]).err(), refused, "refused at parse, before the 1.4 B checkpoint compiles");
        assert_eq!(stack(&["--core", "model", "--one-model"]).err(), refused);
    }

    #[test]
    fn play_takes_no_save_and_mute() {
        let o = parse(Command::Play, &["g.gbc", "--no-save", "--mute", "--weights", "all", "--one-model"]).unwrap();
        assert_eq!((o.save, o.mute, o.stack.one_model()), (false, true, true));
        assert!(USAGE.contains("[--mute] [--no-save] [--one-model] [--boot FILE]"), "USAGE lists play --no-save, --one-model and --boot");
    }
}

/// `GBCFORMER_OUTPUTS=<dir>`: the run's outputs as files, the ones the minimal runner writes
/// too (`transport::Outputs`, `docs/minimal-runner.md`). A file that cannot be written
/// stops the run.
fn outputs_from_env() -> Option<Outputs> {
    let dir = PathBuf::from(std::env::var_os("GBCFORMER_OUTPUTS")?);
    Some(Outputs::new(&dir).unwrap_or_else(|e| panic!("{e}")))
}

/// The wall clock an MBC3 cartridge's clock counts: the host's, or with
/// `GBCFORMER_TIME=<unix seconds>[:<frames a second>]` a scripted one that starts there and
/// moves a second every that many frames (default 60), so two runs see the same time.
fn wall_clock(frame: u64) -> u64 {
    match std::env::var("GBCFORMER_TIME") {
        Ok(v) => {
            let (start, per) = v.split_once(':').map_or((v.as_str(), "60"), |(a, b)| (a, b));
            let (start, per): (u64, u64) = (start.parse().expect("GBCFORMER_TIME: unix seconds"), per.parse().expect("GBCFORMER_TIME: frames a second"));
            start + frame / per.max(1)
        }
        Err(_) => frontend::unix_seconds(),
    }
}
