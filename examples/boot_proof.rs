//! A boot ROM run by the model, held to the reference at the hand-off. The same boot ROM
//! and cartridge run twice from power-on to the FF50 write that unmaps the boot ROM: once
//! on the reference alone, once with every component in weights and every component in
//! lockstep (so each step, segment, span and line is also held as it runs). At the hand-off
//! the model's memory image is held to the host's arrays (`Core::audit_memory`), and then
//! every part of the hand-off state is held to the reference run's: the GBC CPU's registers,
//! the emulated time, every I/O register and IE as the GBC CPU reads them, the display
//! bytes and the window state, cartridge RAM, WRAM, VRAM, OAM and HRAM, and whether the
//! console is in compatibility mode. Exits 1 at any difference.
//!
//!   cargo run --release --example boot_proof -- <boot.bin> <rom> [--hardware dmg|cgb] [--one-model]
//!
//! The console is the boot ROM's (256 bytes DMG, 2,304 CGB); `--hardware` must agree. The
//! boot ROM is a file you supply or SameBoy's; it is read, never copied.

use gbcformer::machine::Hardware;
use gbcformer::step::{Core, Cpu};
use gbcformer::{Instrumentation, Session, Stack};

const BOOT_STEPS: u64 = 50_000_000;

/// Everything the hand-off leaves that a game can observe.
struct HandOff {
    cpu: Cpu,
    halves: u64,
    compat: bool,
    io: Vec<(u16, u8)>,
    display: Vec<u8>,
    window: String,
    regions: Vec<(&'static str, Vec<u8>)>,
}

fn hand_off<C: Core>(s: &mut Session<C>) -> HandOff {
    let io = (0xFF00u16..=0xFF7F).chain([0xFFFF]).map(|a| (a, s.bus_read(a))).collect();
    let (display, window) = s.ppu_state();
    let m = s.machine();
    let names = ["cartridge RAM", "WRAM", "VRAM", "OAM", "HRAM"];
    let regions = names.into_iter().zip(m.ram_regions()).map(|(n, (_, b))| (n, b.to_vec())).collect();
    HandOff { cpu: s.cpu(), halves: s.halves(), compat: m.compat(), io, display: display.to_vec(), window: format!("{window:?}"), regions }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: boot_proof <boot.bin> <rom> [--hardware dmg|cgb] [--one-model]";
    let (Some(boot_path), Some(rom_path)) = (args.first(), args.get(1)) else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let one_model = args.iter().any(|a| a == "--one-model");
    let boot = std::fs::read(boot_path).unwrap_or_else(|e| panic!("{boot_path}: {e}"));
    let rom = std::fs::read(rom_path).unwrap_or_else(|e| panic!("{rom_path}: {e}"));
    let hardware = Hardware::of_boot(&boot).unwrap_or_else(|| panic!("{boot_path}: a boot ROM is 256 bytes (DMG) or 2,304 (CGB), not {}", boot.len()));
    if let Some(i) = args.iter().position(|a| a == "--hardware") {
        let named = match args.get(i + 1).map(String::as_str) {
            Some("dmg") => Hardware::Dmg,
            Some("cgb") => Hardware::Cgb,
            _ => panic!("{usage}"),
        };
        assert_eq!(named, hardware, "--hardware names another console than the boot ROM's");
    }

    let mut reference = Session::power_on(rom.clone(), None, hardware, boot.clone()).expect("power on");
    assert!(reference.finish_boot(BOOT_STEPS), "the reference: the boot ROM did not hand over");
    let want = hand_off(&mut reference);

    let stack = Stack::parse("all", Some("all")).expect("the stack").with_one_model(one_model);
    let mut model = Session::power_on(rom, None, hardware, boot).expect("power on").with_stack(stack, Instrumentation::default()).expect("the stack builds");
    for line in model.describe() {
        println!("{line}");
    }
    assert!(model.finish_boot(BOOT_STEPS), "in weights: the boot ROM did not hand over");
    // The model's memory image, byte for byte, against the host's arrays it recorded into.
    model.core().audit_memory(&model.machine().ram_regions());
    let got = hand_off(&mut model);
    model.settle();

    let mut differ = Vec::new();
    if got.cpu != want.cpu {
        differ.push(format!("GBC CPU: in weights {:?}, reference {:?}", got.cpu, want.cpu));
    }
    if got.halves != want.halves {
        differ.push(format!("time: in weights {} halves, reference {}", got.halves, want.halves));
    }
    if got.compat != want.compat {
        differ.push(format!("compatibility mode: in weights {}, reference {}", got.compat, want.compat));
    }
    for ((a, x), (_, y)) in got.io.iter().zip(&want.io) {
        if x != y {
            differ.push(format!("{a:04X}: in weights {x:02X}, reference {y:02X}"));
        }
    }
    for (i, (x, y)) in got.display.iter().zip(&want.display).enumerate() {
        if x != y {
            differ.push(format!("display byte {i}: in weights {x:02X}, reference {y:02X}"));
        }
    }
    if got.window != want.window {
        differ.push(format!("window: in weights {}, reference {}", got.window, want.window));
    }
    for ((name, x), (_, y)) in got.regions.iter().zip(&want.regions) {
        if let Some(i) = x.iter().zip(y).position(|(a, b)| a != b) {
            differ.push(format!("{name} byte {i:#x}: in weights {:02X}, reference {:02X} ({} bytes differ)", x[i], y[i], x.iter().zip(y).filter(|(a, b)| a != b).count()));
        } else if x.len() != y.len() {
            differ.push(format!("{name}: in weights {} bytes, reference {}", x.len(), y.len()));
        }
    }
    let memory: usize = want.regions.iter().map(|(_, b)| b.len()).sum();
    println!("hand-off at PC {:04X} after {} halves, compatibility mode {}: the GBC CPU state, {} I/O bytes, {} display bytes and {memory} memory bytes compared", want.cpu.pc, want.halves, want.compat, want.io.len(), want.display.len());
    if differ.is_empty() {
        println!("boot proof: the hand-off in weights equals the reference's");
    } else {
        for d in differ.iter().take(40) {
            println!("differs: {d}");
        }
        println!("boot proof: {} differences", differ.len());
        std::process::exit(1);
    }
}
