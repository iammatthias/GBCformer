//! Outside test ROMs, each judged by its own pass signal, never by the reference
//! emulator: mooneye's and SameSuite's Fibonacci registers at `LD B,B`, blargg's serial
//! text, and screenshots compared with the suite's own images (Mealybug Tearoom, blargg,
//! acid2, daid). The ROMs live in vendor/hwtests (tools/hwtests/fetch.sh).
//!
//!   cargo run --release --example hwtests -- [--filter TEXT] [--cases FILE] [--shard K/N] [--jobs N] [--weights LIST] [--lockstep LIST] [--one-model] [--boot-dmg FILE] [--boot-cgb FILE]
//!
//! Prints one line per case: `PASS`, `FAIL`, `REFUSED`, or `NA` for a ROM whose own name
//! or documentation leaves the console out (listed, not run); then the suite, the ROM, the
//! console, what decided it, the mark (`APPLIES` or `NA`), SameBoy's verdict
//! (docs/hardware-tests/sameboy.tsv, written by tools/hwtests/sameboy.sh) and the area.
//! `--shard K/N` keeps the cases whose index is K modulo N, so N processes cover the list
//! between them. Each case's verdict also reaches stderr as it finishes.
//! A count per suite follows on stderr. Screenshots compare in the suites' common palette:
//! DMG shades as 000000/555555/AAAAAA/FFFFFF and RGB555 as `c << 3 | c >> 2`, each channel
//! within 4 of the image (blargg's images carry FDFDFD for FFFFFF). Needs ImageMagick to
//! read the PNGs.

use gbcformer::machine::Hardware;
use gbcformer::machine::Hardware::{Cgb, Dmg};
use gbcformer::machine::io::Buttons;
use gbcformer::machine::ppu::{DMG_SHADES, rgb24};
use gbcformer::rom;
use gbcformer::session::Session;
use gbcformer::stack::{Instrumentation, Stack};
use gbcformer::step::Cpu;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const ROOT: &str = "vendor/hwtests";
const SAMEBOY: &str = "docs/hardware-tests/sameboy.tsv";
const FRAMES_PER_SECOND: u64 = 60;

#[derive(Clone, Debug)]
enum End {
    /// After this many frames.
    Frames(u64),
    /// At the first `LD B,B`, the frame it ran in finished, or after this many frames.
    LdBB(u64),
}

#[derive(Clone, Debug)]
enum Judge {
    /// B, C, D, E, H, L = 3, 5, 8, 13, 21, 34 at `LD B,B`.
    Fibonacci { max_frames: u64 },
    /// The screen matches one of these images.
    Screen { images: Vec<PathBuf>, end: End },
    /// blargg: the serial output reads "Passed" or "Failed", or cartridge RAM holds the
    /// signature DE B0 61 at A001 and a finished status at A000 (0 passed), the text at A004.
    Blargg { max_frames: u64 },
}

/// A case's mark: it applies and runs under its judge, or it is not this console's.
#[derive(Clone, Debug)]
enum Mark {
    Applies(Judge),
    Na,
}

/// Why a case has its mark, and where that is written.
#[derive(Clone, Debug)]
struct Why {
    /// An NA case's class, one of the three above.
    class: Option<&'static str>,
    reason: String,
    source: &'static str,
}

#[derive(Clone, Debug)]
struct Case {
    suite: &'static str,
    rom: PathBuf,
    hardware: Hardware,
    mark: Mark,
    why: Why,
}

const NAMING: &str = "mooneye-test-suite README, Test naming";
const MEALYBUG: &str = "Mealybug Tearoom README and its expected images";

/// Mooneye's model suffixes, which SameSuite and Mealybug share: the emulated consoles
/// (DMG-CPU C, CPU CGB E) each covers, and the models it names.
const SUFFIXES: &[(&str, &[Hardware], &str)] = &[
    ("dmgABCmgb", &[Dmg], "DMG ABC and MGB"),
    ("dmgABC", &[Dmg], "DMG ABC"),
    ("GS", &[Dmg], "DMG, MGB, SGB and SGB2"),
    ("C", &[Cgb], "CGB, AGB and AGS"),
    ("cgb", &[Cgb], "CGB"),
    ("cgbABCDE", &[Cgb], "CGB A to E"),
    ("cgbDE", &[Cgb], "CGB D and E"),
    ("dmg0", &[], "DMG 0"),
    ("mgb", &[], "MGB"),
    ("sgb", &[], "SGB"),
    ("sgb2", &[], "SGB2"),
    ("S", &[], "SGB and SGB2"),
    ("A", &[], "AGB and AGS"),
    ("cgb0", &[], "CGB 0"),
    ("cgb0B", &[], "CGB 0 and B"),
    ("cgb0BC", &[], "CGB 0, B and C"),
    ("cgbB", &[], "CGB B"),
];

/// The three kinds of NA case (docs/hardware-tests.md §2.4): the ROM is for other hardware,
/// the ROM itself cannot finish, or no expected result exists for the revision run here.
const WRONG_HARDWARE: &str = "wrong hardware";
const BROKEN_ARTIFACT: &str = "broken artifact";
const MISSING_ORACLE: &str = "missing oracle";

/// ROMs whose documentation leaves a console out: (path fragment, console, class, reason, source).
const NOT_ON: &[(&str, Hardware, &str, &str, &str)] = &[
    ("blargg/oam_bug/", Cgb, WRONG_HARDWARE, "verifies the OAM corruption bug on DMG", "blargg oam_bug/readme.txt"),
    ("blargg/oam_bug/rom_singles/7-timing_effect.gb", Dmg, BROKEN_ARTIFACT, "its text routine stores past its 8 KiB of cartridge RAM into its own code at C000, so it cannot finish (SameBoy v1.0.5 does not either); the bundle oam_bug.gb runs test 7 and reports 07:ok", "the ROM's bytes, docs/hardware-tests.md §3.14"),
    ("blargg/dmg_sound/", Cgb, WRONG_HARDWARE, "the sound ROMs are for DMG or for CGB; these are DMG's", "blargg dmg_sound/readme.txt"),
    ("blargg/cgb_sound/", Dmg, WRONG_HARDWARE, "the sound ROMs are for DMG or for CGB; these are CGB's", "blargg cgb_sound/readme.txt"),
    ("blargg/interrupt_time/", Dmg, WRONG_HARDWARE, "a CGB-only ROM", "c-sp game-boy-test-roms v7.0, blargg howto"),
    ("_change2.gb", Dmg, MISSING_ORACLE, "its only expected image is CPU CGB C's", MEALYBUG),
    ("_change2.gb", Cgb, MISSING_ORACLE, "its only expected image is CPU CGB C's", MEALYBUG),
    ("mealybug-tearoom-tests/ppu/", Cgb, MISSING_ORACLE, "its CGB expected image is CPU CGB D's; Mealybug publishes none for CPU CGB E, the revision run here, and no source (SameBoy's verdicts included) shows CGB E matching it", MEALYBUG),
];

/// Why a case applies, first match on the ROM's path and console: (path fragment,
/// console or any, reason, source).
const APPLIES: &[(&str, Option<Hardware>, &str, &str)] = &[
    ("mooneye-test-suite/emulator-only/", None, "names no model; run on DMG only", NAMING),
    ("mooneye-test-suite/", None, "its Verified results list DMG and CGB", "the test's source header, mooneye-test-suite 8d742b9d55"),
    ("same-suite/apu/", None, "CPU-CGB-E passes all tests", "SameSuite apu README at f71b4b3c37"),
    ("same-suite/", None, "names no console; run on CGB only", "c-sp game-boy-test-roms v7.0, same-suite howto"),
    ("mealybug-tearoom-tests/ppu/", Some(Dmg), "judged by Mealybug's DMG blob image", MEALYBUG),
    ("blargg/", None, "c-sp's tables verify the suite on DMG-CPU C and CPU CGB E", "c-sp game-boy-test-roms v7.0, blargg howto"),
    ("daid/", None, "the Shootout runs it on this console", "GBEmulatorShootout testroms/daid.py"),
    ("acid", None, "the suite publishes an image for this console", "the suite's README"),
];

/// The work areas, first match on the ROM's path (docs/hardware-tests.md §2.3).
const AREAS: &[(&str, &str)] = &[
    ("mealybug-tearoom-tests/ppu/", "PPU mid-scanline"),
    ("ppu_scanline_bgp", "PPU mid-scanline"),
    ("cgb-acid-hell", "PPU mid-scanline"),
    ("acid2", "PPU rendering"),
    ("oam_bug", "OAM bug"),
    ("oam_dma", "OAM DMA"),
    ("mealybug-tearoom-tests/dma/", "DMA (HDMA)"),
    ("same-suite/dma/", "DMA (HDMA)"),
    ("same-suite/apu/", "APU"),
    ("_sound", "APU"),
    ("/timer/", "timer"),
    ("div_timing", "timer"),
    ("/serial/", "serial"),
    ("boot_", "boot state"),
    ("emulator-only/", "cartridge (MBC)"),
    ("/ppu/", "PPU timing"),
    ("interrupt", "interrupts"),
    ("intr_timing", "interrupts"),
    ("di_timing", "interrupts"),
    ("ei_", "interrupts"),
    ("halt_", "interrupts"),
    ("rapid_di_ei", "interrupts"),
    ("if_ie_registers", "interrupts"),
    ("/bits/", "I/O registers"),
    ("", "GBC CPU and timing"),
];

fn console(h: Hardware) -> &'static str {
    if h == Dmg { "dmg" } else { "cgb" }
}

fn area(rom: &Path) -> &'static str {
    let path = rom.to_string_lossy();
    AREAS.iter().find(|(fragment, _)| path.contains(fragment)).map(|&(_, a)| a).unwrap()
}

/// The suites that name models by suffix.
const SUFFIXED: &[&str] = &["mooneye-test-suite/", "same-suite/", "mealybug-tearoom-tests/"];

/// The ruling on a ROM's model suffix, if its suite uses them and it has one: whether it
/// covers `h`, and why.
fn suffix(rom: &Path, h: Hardware) -> Option<(bool, Why)> {
    let path = rom.to_string_lossy();
    if !SUFFIXED.iter().any(|s| path.contains(s)) {
        return None;
    }
    let name = stem(rom);
    let (_, suffix) = name.rsplit_once('-')?;
    let &(_, covers, models) = SUFFIXES.iter().find(|(s, ..)| *s == suffix).unwrap_or_else(|| panic!("{name}: model suffix -{suffix} is not in SUFFIXES"));
    let applies = covers.contains(&h);
    let reason = if applies { format!("suffix -{suffix} names {models}") } else { format!("suffix -{suffix} names {models}, not {}", if h == Dmg { "DMG-CPU C" } else { "CPU CGB E" }) };
    Some((applies, Why { class: (!applies).then_some(WRONG_HARDWARE), reason, source: NAMING }))
}

/// Whether `rom` is not `h`'s, and why.
fn not_on(rom: &Path, h: Hardware) -> Option<Why> {
    let path = rom.to_string_lossy();
    if let Some((false, why)) = suffix(rom, h) {
        return Some(why);
    }
    NOT_ON
        .iter()
        .find(|&&(fragment, on, ..)| on == h && path.contains(fragment))
        .map(|&(_, _, class, reason, source)| Why { class: Some(class), reason: reason.into(), source })
}

fn applies(rom: &Path, h: Hardware) -> Why {
    if let Some((true, why)) = suffix(rom, h) {
        return why;
    }
    let path = rom.to_string_lossy();
    APPLIES
        .iter()
        .find(|&&(fragment, on, ..)| path.contains(fragment) && on.is_none_or(|on| on == h))
        .map(|&(_, _, reason, source)| Why { class: None, reason: reason.into(), source })
        .unwrap_or_else(|| panic!("{}: no APPLIES row", rom.display()))
}

/// The case for `rom` on `h`: NA when its name or documentation leaves `h` out, else run
/// under `judge`.
fn case(suite: &'static str, rom: &Path, hardware: Hardware, judge: Judge) -> Case {
    let (mark, why) = match not_on(rom, hardware) {
        Some(why) => (Mark::Na, why),
        None => (Mark::Applies(judge), applies(rom, hardware)),
    };
    Case { suite, rom: rom.to_path_buf(), hardware, mark, why }
}

fn roms(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for p in paths {
        if p.is_dir() {
            roms(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()), Some("gb" | "gbc")) {
            out.push(p);
        }
    }
}

fn stem(p: &Path) -> String {
    p.file_stem().unwrap().to_string_lossy().into_owned()
}

/// Every case: each ROM on each console the runner considers for its suite, NA where the
/// ROM's name or documentation leaves that console out.
fn cases() -> Vec<Case> {
    let root = Path::new(ROOT);
    let mut cases = Vec::new();
    let gbtr = root.join("gbtr");
    let fib = Judge::Fibonacci { max_frames: 120 * FRAMES_PER_SECOND };

    for sub in ["acceptance", "emulator-only", "misc"] {
        let mut mooneye = Vec::new();
        roms(&gbtr.join("mooneye-test-suite").join(sub), &mut mooneye);
        let consoles: &[Hardware] = if sub == "emulator-only" { &[Dmg] } else { &[Dmg, Cgb] };
        for rom in mooneye {
            for &hardware in consoles {
                cases.push(case("mooneye", &rom, hardware, fib.clone()));
            }
        }
    }

    let mut same = Vec::new();
    for sub in ["apu", "dma", "interrupt", "ppu"] {
        roms(&gbtr.join("same-suite").join(sub), &mut same);
    }
    for rom in same {
        cases.push(case("samesuite", &rom, Cgb, fib.clone()));
    }

    let mut mealybug = Vec::new();
    roms(&gbtr.join("mealybug-tearoom-tests"), &mut mealybug);
    for rom in mealybug {
        let name = stem(&rom);
        if rom.to_string_lossy().contains("/ppu/") {
            for (hardware, tag) in [(Dmg, "dmg_blob"), (Cgb, "cgb_d")] {
                let image = rom.with_file_name(format!("{name}_{tag}.png"));
                // A ROM with no image for this console and no statement leaving it out has no judge.
                if image.exists() || not_on(&rom, hardware).is_some() {
                    cases.push(case("mealybug", &rom, hardware, Judge::Screen { images: vec![image], end: End::LdBB(600) }));
                }
            }
        } else if rom.to_string_lossy().contains("/dma/") {
            for hardware in [Dmg, Cgb] {
                cases.push(case("mealybug", &rom, hardware, fib.clone()));
            }
        }
    }

    // blargg: (directory, seconds on DMG, seconds on CGB), each bundle's screenshot; 0 where
    // the bundle is not that console's.
    for (test, dmg, cgb) in [
        ("cpu_instrs", 55, 31),
        ("instr_timing", 1, 1),
        ("mem_timing", 3, 3),
        ("mem_timing-2", 4, 4),
        ("halt_bug", 2, 2),
        ("oam_bug", 21, 0),
        ("dmg_sound", 36, 0),
        ("cgb_sound", 0, 37),
        ("interrupt_time", 0, 2),
    ] {
        let dir = gbtr.join("blargg").join(test);
        let (dir, rom) = if test == "halt_bug" { (gbtr.join("blargg"), gbtr.join("blargg/halt_bug.gb")) } else { (dir.clone(), dir.join(if test == "mem_timing-2" { "mem_timing.gb".to_string() } else { format!("{test}.gb") })) };
        for (hardware, seconds, tags) in [(Dmg, dmg, ["dmg-cgb", "dmg"]), (Cgb, cgb, ["dmg-cgb", "cgb"])] {
            let prefix = rom.file_stem().unwrap().to_string_lossy().into_owned();
            let images: Vec<PathBuf> = tags.iter().map(|t| dir.join(format!("{prefix}-{t}.png"))).filter(|p| p.exists()).collect();
            let c = case("blargg", &rom, hardware, Judge::Screen { images, end: End::Frames(seconds * FRAMES_PER_SECOND) });
            assert_eq!(seconds == 0, matches!(c.mark, Mark::Na), "{}: {} seconds on {}, but {:?}", rom.display(), seconds, console(hardware), c.why);
            cases.push(c);
        }
    }
    // blargg's single tests, on the console each directory runs on; the other console is
    // listed only where NA.
    for (test, hardware) in [
        ("cpu_instrs/individual", Dmg),
        ("mem_timing/individual", Dmg),
        ("mem_timing-2/rom_singles", Dmg),
        ("dmg_sound/rom_singles", Dmg),
        ("oam_bug/rom_singles", Dmg),
        ("cgb_sound/rom_singles", Cgb),
    ] {
        let mut singles = Vec::new();
        roms(&gbtr.join("blargg").join(test), &mut singles);
        let other = if hardware == Dmg { Cgb } else { Dmg };
        for rom in singles {
            let judge = Judge::Blargg { max_frames: 60 * FRAMES_PER_SECOND };
            cases.push(case("blargg-single", &rom, hardware, judge.clone()));
            if not_on(&rom, other).is_some() {
                cases.push(case("blargg-single", &rom, other, judge));
            }
        }
    }

    let acid = |rom: &str, hardware, image: &str| case("acid", &gbtr.join(rom), hardware, Judge::Screen { images: vec![gbtr.join(image)], end: End::LdBB(600) });
    cases.push(acid("dmg-acid2/dmg-acid2.gb", Dmg, "dmg-acid2/dmg-acid2-dmg.png"));
    cases.push(acid("dmg-acid2/dmg-acid2.gb", Cgb, "dmg-acid2/dmg-acid2-cgb.png"));
    cases.push(acid("cgb-acid2/cgb-acid2.gbc", Cgb, "cgb-acid2/cgb-acid2.png"));
    cases.push(acid("cgb-acid-hell/cgb-acid-hell.gbc", Cgb, "cgb-acid-hell/cgb-acid-hell.png"));

    let daid = root.join("daid");
    let screen = |rom: &str, hardware, images: &[&str]| {
        case("daid", &daid.join(rom), hardware, Judge::Screen { images: images.iter().map(|i| daid.join(i)).collect(), end: End::Frames(60) })
    };
    cases.push(screen("stop_instr.gb", Dmg, &["stop_instr.dmg.png"]));
    cases.push(screen("stop_instr.gb", Cgb, &["stop_instr.gbc.png"]));
    cases.push(screen("stop_instr_gbc_mode3.gb", Cgb, &["stop_instr_gbc_mode3.png"]));
    cases.push(screen("ppu_scanline_bgp.gb", Dmg, &["ppu_scanline_bgp_0.dmg.png", "ppu_scanline_bgp_1.dmg.png", "ppu_scanline_bgp_2.dmg.png"]));
    cases.push(screen("ppu_scanline_bgp.gb", Cgb, &["ppu_scanline_bgp.gbc.png"]));
    for t in ["div", "ly", "stat"] {
        cases.push(screen(&format!("speed_switch_timing_{t}.gbc"), Cgb, &[&format!("speed_switch_timing_{t}.png")]));
    }
    cases
}

/// SameBoy's verdict per (ROM under vendor/hwtests, console), from `SAMEBOY`.
fn sameboy() -> HashMap<(String, String), String> {
    let text = std::fs::read_to_string(SAMEBOY).unwrap_or_else(|e| panic!("{SAMEBOY}: {e}; run tools/hwtests/sameboy.sh"));
    text.lines()
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            ((f[0].to_string(), f[1].to_string()), f[2].to_string())
        })
        .collect()
}

/// Boot ROM files to run each case from power-on with, per console (`--boot-dmg`,
/// `--boot-cgb`); without one a case starts at the hand-off.
#[derive(Default)]
struct Boots {
    dmg: Option<Vec<u8>>,
    cgb: Option<Vec<u8>>,
}

enum Verdict {
    Pass(String),
    Fail(String),
    Refused(String),
}

/// The suites' common palette, row-major RGB.
fn screen(s: &mut Session<impl gbcformer::step::Core>, hardware: Hardware) -> Vec<u8> {
    let frame = s.frame();
    if hardware == Hardware::Cgb {
        return rgb24(frame);
    }
    frame
        .iter()
        .flat_map(|&px| {
            let shade = DMG_SHADES.iter().position(|&c| c == px).unwrap_or(0);
            let grey = [0xFF, 0xAA, 0x55, 0x00][shade];
            [grey, grey, grey]
        })
        .collect()
}

/// The image's pixels as 8-bit RGB, through ImageMagick.
fn load_png(path: &Path) -> Result<Vec<u8>, String> {
    let out = std::process::Command::new("magick")
        .arg(path)
        .args(["-depth", "8", "rgb:-"])
        .output()
        .map_err(|e| format!("magick: {e}"))?;
    if !out.status.success() || out.stdout.len() != 160 * 144 * 3 {
        return Err(format!("{}: not a 160x144 image", path.display()));
    }
    Ok(out.stdout)
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks(3).zip(b.chunks(3)).filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.abs_diff(*q) > 4)).count()
}

fn fibonacci(c: &Cpu) -> bool {
    [c.b, c.c, c.d, c.e, c.h, c.l] == [3, 5, 8, 13, 21, 34]
}

fn run(case: &Case, by: &Judge, stack: Stack, boots: &Boots) -> Verdict {
    let bytes = match std::fs::read(&case.rom) {
        Ok(b) => b,
        Err(e) => return Verdict::Refused(e.to_string()),
    };
    let header = match rom::identify(&bytes) {
        Ok(h) => h,
        Err(e) => return Verdict::Refused(e),
    };
    if let Err(e) = header.kind() {
        return Verdict::Refused(e);
    }
    let hardware = match Hardware::resolve(&header, Some(case.hardware)) {
        Ok(h) => h,
        Err(e) => return Verdict::Refused(e),
    };
    let boot = match hardware {
        Hardware::Dmg => boots.dmg.clone(),
        Hardware::Cgb => boots.cgb.clone(),
    };
    let mut session = match boot.map_or_else(|| Session::new(bytes.clone(), None, hardware), |b| Session::power_on(bytes.clone(), None, hardware, b)) {
        Ok(s) => s,
        Err(e) => return Verdict::Refused(e),
    };
    // A case's time counts from the hand-off. The boot ROM runs on the stack the case
    // names: the memory checkpoint maps it (`tests/boot_overlay.rs`).
    const BOOT_STEPS: u64 = 20_000_000;
    if stack == Stack::HOST {
        if !session.finish_boot(BOOT_STEPS) {
            return Verdict::Fail("the boot ROM did not hand over".into());
        }
        return judge(session, case, by);
    }
    match session.with_stack(stack, Instrumentation::default()) {
        Ok(mut s) => {
            if !s.finish_boot(BOOT_STEPS) {
                return Verdict::Fail("the boot ROM did not hand over".into());
            }
            judge(s, case, by)
        }
        Err(e) => Verdict::Refused(e),
    }
}

/// Runs frames until `LD B,B` (when `stop` asks) or `max` frames; the GBC CPU state at the
/// first `LD B,B`, if one ran.
fn frames<C: gbcformer::step::Core>(s: &mut Session<C>, max: u64, stop: bool) -> Option<Cpu> {
    let mut hit = None;
    while s.frame_number() < max {
        s.run_frame_watching(Buttons(0), |cpu, op| {
            if hit.is_none() && op == Some(0x40) {
                hit = Some(*cpu);
            }
        });
        if stop && hit.is_some() {
            break;
        }
    }
    hit
}

fn judge<C: gbcformer::step::Core>(mut s: Session<C>, case: &Case, by: &Judge) -> Verdict {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match by {
        Judge::Fibonacci { max_frames } => match frames(&mut s, *max_frames, true) {
            Some(cpu) if fibonacci(&cpu) => Verdict::Pass(format!("Fibonacci at frame {}", s.frame_number())),
            Some(cpu) => {
                let dump = std::fs::read(&case.rom).ok().and_then(|rom| samesuite_dump(&mut s, &rom)).unwrap_or_default();
                Verdict::Fail(format!("LD B,B with B={:02X} C={:02X} D={:02X} E={:02X} H={:02X} L={:02X}{dump}", cpu.b, cpu.c, cpu.d, cpu.e, cpu.h, cpu.l))
            }
            None => Verdict::Fail(format!("no LD B,B in {max_frames} frames")),
        },
        Judge::Blargg { max_frames } => {
            while s.frame_number() < *max_frames {
                s.run_frame(Buttons(0));
                let text = String::from_utf8_lossy(s.serial_output()).into_owned();
                if text.contains("Passed") {
                    return Verdict::Pass("serial: Passed".into());
                }
                if text.contains("Failed") {
                    return Verdict::Fail(format!("serial: {}", text.split_whitespace().collect::<Vec<_>>().join(" ")));
                }
                let ram = s.cart_ram();
                if ram.len() > 4 && ram[1..4] == [0xDE, 0xB0, 0x61] && ram[0] != 0x80 {
                    let end = ram[4..].iter().position(|&b| b == 0).map_or(ram.len(), |n| n + 4);
                    let text = String::from_utf8_lossy(&ram[4..end]).split_whitespace().collect::<Vec<_>>().join(" ");
                    return if ram[0] == 0 { Verdict::Pass(format!("A000: {text}")) } else { Verdict::Fail(format!("A000 = {:02X}: {text}", ram[0])) };
                }
            }
            let ram = s.cart_ram();
            let end = ram.iter().skip(4).position(|&b| b == 0).map_or(ram.len(), |n| n + 4);
            let text = String::from_utf8_lossy(ram.get(4..end).unwrap_or(&[])).split_whitespace().collect::<Vec<_>>().join(" ");
            Verdict::Fail(format!("no verdict on serial or at A000 in {max_frames} frames; A004 so far: {text}"))
        }
        Judge::Screen { images, end } => {
            match end {
                End::Frames(n) => {
                    frames(&mut s, *n, false);
                }
                End::LdBB(max) => {
                    frames(&mut s, *max, true);
                }
            }
            let shot = screen(&mut s, case.hardware);
            let mut best = usize::MAX;
            for image in images {
                match load_png(image) {
                    Ok(want) => best = best.min(differing(&shot, &want)),
                    Err(e) => return Verdict::Refused(e),
                }
            }
            if images.is_empty() {
                return Verdict::Refused("no expected image".into());
            }
            if best == 0 { Verdict::Pass(format!("screen equal at frame {}", s.frame_number())) } else { Verdict::Fail(format!("{best} pixels differ at frame {}", s.frame_number())) }
        }
    }));
    outcome.unwrap_or_else(|p| {
        let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
        Verdict::Fail(format!("panicked: {msg}"))
    })
}


/// Per suite: applicable cases, their passes, fails and refusals, NA cases, and SameBoy's
/// passes and unknowns among the applicable.
#[derive(Default)]
struct Tally {
    applicable: usize,
    pass: usize,
    fail: usize,
    refused: usize,
    na: usize,
    sameboy_pass: usize,
    sameboy_unknown: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut filter, mut jobs, mut weights, mut lockstep, mut one_model) = (String::new(), 4usize, None, None, false);
    // `--cases FILE`: only the cases a line of FILE names as "suite<TAB>rom<TAB>console" (the
    // census TSV's columns 2 to 4), a declared sample.
    let mut chosen: Option<std::collections::BTreeSet<String>> = None;
    let mut boots = Boots::default();
    let mut shard = (0usize, 1usize);
    let read = |p: String| std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--filter" => filter = value(),
            "--cases" => {
                let p = value();
                chosen = Some(std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}")).lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect());
            }
            "--shard" => {
                let v = value();
                let (k, n) = v.split_once('/').and_then(|(k, n)| Some((k.parse().ok()?, n.parse().ok()?))).filter(|&(k, n)| k < n).expect("--shard K/N with K < N");
                shard = (k, n);
            }
            "--jobs" => jobs = value().parse().expect("--jobs N"),
            "--weights" => weights = Some(value()),
            "--lockstep" => lockstep = Some(value()),
            "--one-model" => one_model = true,
            "--boot-dmg" => boots.dmg = Some(read(value())),
            "--boot-cgb" => boots.cgb = Some(read(value())),
            other => panic!("unknown flag {other}"),
        }
    }
    let stack = Stack::parse(weights.as_deref().unwrap_or("none"), lockstep.as_deref()).expect("stack").with_one_model(one_model);
    stack.resolve().expect("stack");
    if stack != Stack::HOST {
        // Each case compiles its own checkpoints; one at a time keeps the memory in bounds.
        jobs = 1;
    }
    let all = cases();
    let rom_key = |c: &Case| c.rom.strip_prefix(ROOT).unwrap_or(&c.rom).display().to_string();
    let mut sameboy = sameboy();
    for c in &all {
        if sameboy.contains_key(&(rom_key(c), console(c.hardware).to_string())) {
            assert!(matches!(c.mark, Mark::Applies(_)), "{} {}: SameBoy has a verdict on a case marked NA", rom_key(c), console(c.hardware));
        }
    }
    for (rom, hw) in sameboy.keys().filter(|(rom, hw)| !all.iter().any(|c| &rom_key(c) == rom && console(c.hardware) == hw)) {
        eprintln!("{SAMEBOY}: {rom} {hw} is not a case");
    }
    let cases: Vec<Case> = all.into_iter().filter(|c| format!("{} {} {}", c.suite, c.rom.display(), console(c.hardware)).contains(&filter)).filter(|c| chosen.as_ref().is_none_or(|k| k.contains(&format!("{}\t{}\t{}", c.suite, rom_key(c), console(c.hardware))))).enumerate().filter(|(i, _)| i % shard.1 == shard.0).map(|(_, c)| c).collect();
    if let Some(k) = &chosen {
        assert!(shard.1 > 1 || cases.len() == k.len(), "--cases names {} cases, {} of them in the census", k.len(), cases.len());
    }
    if cases.is_empty() {
        eprintln!("no cases in {ROOT}; run tools/hwtests/fetch.sh");
        std::process::exit(1);
    }
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Verdict>>> = Mutex::new((0..cases.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            // A machine holds two frames and WRAM by value; a boot ROM run builds two.
            std::thread::Builder::new().stack_size(64 << 20).spawn_scoped(scope, || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(case) = cases.get(i) else { break };
                    let Mark::Applies(judge) = &case.mark else { continue };
                    let t = std::time::Instant::now();
                    let verdict = run(case, judge, stack, &boots);
                    eprintln!("{} {} {} {} {:.1} s", match &verdict { Verdict::Pass(_) => "PASS", Verdict::Fail(_) => "FAIL", Verdict::Refused(_) => "REFUSED" }, case.suite, rom_key(case), console(case.hardware), t.elapsed().as_secs_f64());
                    results.lock().unwrap()[i] = Some(verdict);
                }
            }).expect("thread");
        }
    });
    let results = results.into_inner().unwrap();
    let mut tally: Vec<(&str, Tally)> = Vec::new();
    for (case, verdict) in cases.iter().zip(results) {
        let rom = rom_key(case);
        let hw = console(case.hardware);
        let sb = sameboy.remove(&(rom.clone(), hw.to_string())).unwrap_or_else(|| "unknown".into());
        let i = tally.iter().position(|(s, _)| *s == case.suite).unwrap_or_else(|| {
            tally.push((case.suite, Tally::default()));
            tally.len() - 1
        });
        let t = &mut tally[i].1;
        let (tag, mark, detail) = match verdict {
            None => {
                t.na += 1;
                ("NA", "NA", format!("{}: {} ({})", case.why.class.expect("an NA case has a class"), case.why.reason, case.why.source))
            }
            Some(v) => {
                t.applicable += 1;
                t.sameboy_pass += usize::from(sb == "PASS");
                t.sameboy_unknown += usize::from(sb == "unknown");
                let (tag, count, d) = match v {
                    Verdict::Pass(d) => ("PASS", &mut t.pass, d),
                    Verdict::Fail(d) => ("FAIL", &mut t.fail, d),
                    Verdict::Refused(d) => ("REFUSED", &mut t.refused, d),
                };
                *count += 1;
                (tag, "APPLIES", d)
            }
        };
        println!("{tag}\t{}\t{rom}\t{hw}\t{detail}\t{mark}\t{sb}\t{}", case.suite, area(&case.rom));
    }
    for (suite, t) in tally {
        eprintln!(
            "{suite}: {} applicable, {} pass, {} fail, {} refused; {} NA; SameBoy {} pass, {} unknown",
            t.applicable, t.pass, t.fail, t.refused, t.na, t.sameboy_pass, t.sameboy_unknown
        );
    }
}

/// SameSuite's results beside its expected table, when `HW_DUMP` is set: the ROM's
/// `PrintResults` loads `ld bc, RESULTS_START; ld de, CorrectResults; ld hl, $9800`, and
/// its row loop ends at `cp $98 + RESULTS_N_ROWS / 8`. Each differing byte is marked.
fn samesuite_dump<C: gbcformer::step::Core>(s: &mut Session<C>, rom: &[u8]) -> Option<String> {
    std::env::var_os("HW_DUMP")?;
    let at = rom.windows(9).position(|w| w[0] == 0x01 && w[3] == 0x11 && w[6..9] == [0x21, 0x00, 0x98])?;
    let results = u16::from_le_bytes([rom[at + 1], rom[at + 2]]);
    let expected = u16::from_le_bytes([rom[at + 4], rom[at + 5]]) as usize;
    let rows = rom[at..].windows(2).position(|w| w[0] == 0xFE && (0x98..0x9C).contains(&w[1])).map_or(16, |p| (rom[at + p + 1] as usize - 0x98) * 8).max(1);
    let mut out = String::new();
    let mut first = None;
    for r in 0..rows {
        let mut line = format!("\n  row {r:2}:");
        for c in 0..8 {
            let i = r * 8 + c;
            let (got, want) = (s.bus_read(results + i as u16), rom.get(expected + i).copied().unwrap_or(0));
            if got != want && first.is_none() {
                first = Some(i);
            }
            line += &if got == want { format!(" {got:02X}   ") } else { format!(" {got:02X}/{want:02X}") };
        }
        out += &line;
    }
    Some(format!(" first difference at subtest {first:?} (got/want):{out}"))
}
