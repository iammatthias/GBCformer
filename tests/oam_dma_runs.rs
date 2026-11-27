//! The control model moves OAM DMA bytes in runs, a pass a run, and the machine lands a
//! run's bytes at its pass. Nothing may see the difference from one byte an M-cycle: a
//! DMG program starts a transfer at a different point of a line each time, mode 2 and
//! mode 3 included, reads OAM while the transfer blocks it and again after, and moves
//! 40 sprites a frame. A second transfer polls OAM until it reads, so its reads fall on
//! the transfer's end and the PPU's OAM lock inside their own steps. Every frame and the
//! bytes it read must equal the reference machine's.

use gbcformer::machine::io::Buttons;
use gbcformer::step::Core;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

fn program() -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    let mut code = vec![
        0xF3, 0x31, 0xFE, 0xFF, // DI; LD SP,$FFFE (the stack in HRAM)
        0xAF, 0xE0, 0x40, // LCD off
        0x21, 0x10, 0x80, 0x06, 0x10, 0x3E, 0xFF, // tile 1 solid: LD HL,$8010; LD B,16; LD A,$FF
        0x22, 0x05, 0x20, 0xFC, // LD (HL+),A; DEC B; JR NZ
        0x3E, 0xE4, 0xE0, 0x47, 0xE0, 0x48, // BGP = OBP0 = $E4
        0x21, 0x00, 0xC0, 0x06, 0x28, 0x0E, 0x10, 0x16, 0x08, // 40 sprites at $C000: y in C, x in D
        0x71, 0x23, 0x72, 0x23, 0x36, 0x01, 0x23, 0x36, 0x00, 0x23, // y, x, tile 1, attributes 0
        0x79, 0xC6, 0x03, 0x4F, 0x7A, 0xC6, 0x04, 0x57, 0x05, 0x20, 0xEB, // y += 3, x += 4
    ];
    // The transfer's routine in HRAM: start it, read OAM while it blocks the read, wait,
    // read OAM again, return. The CPU sees only HRAM until the transfer ends.
    let routine = [0xE0, 0x46, 0xFA, 0x00, 0xFE, 0xE0, 0xF0, 0x3E, 0x28, 0x3D, 0x20, 0xFD, 0xFA, 0x9F, 0xFE, 0xE0, 0xF1, 0xC9];
    for (i, b) in routine.into_iter().enumerate() {
        code.extend([0x3E, b, 0xE0, 0x80 + i as u8]);
    }
    // A second routine at $FFA0: start the transfer, NOP, POP BC from WRAM while it blocks
    // both reads (SP kept in HL), wait ($FFF3) loops of 4 M-cycles, then read $FE9F until
    // it is not 0xFF and keep it at $FFF2. The NOP puts POP's first read one M-cycle past
    // the end of an OAM line's run, so a cut there runs a pass of its own. The transfer's
    // end and the PPU's OAM lock each fall inside some read's step at some delay, so the
    // read lands on the bus state of its own M-cycle, not its step's start.
    let polling = [0xE0, 0x46, 0x00, 0xF8, 0x00, 0x31, 0x00, 0xC0, 0xC1, 0xF9, 0x21, 0x9F, 0xFE, 0xF0, 0xF3, 0x3D, 0x20, 0xFD, 0x7E, 0xFE, 0xFF, 0x28, 0xFB, 0xE0, 0xF2, 0xC9];
    for (i, b) in polling.into_iter().enumerate() {
        code.extend([0x3E, b, 0xE0, 0xA0 + i as u8]);
    }
    code.extend([0x3E, 0x93, 0xE0, 0x40, 0x1E, 0x00]); // LCD on with sprites; E = 0, the line to start at
    let main = code.len();
    code.extend([
        0xF0, 0x44, 0xBB, 0x20, 0xFB, // wait for LY = E
        0x7B, 0xE6, 0x0F, 0x3C, 0x3D, 0x20, 0xFD, // a delay of E & 15 loops, into mode 2, 3 or 0
        0x3E, 0xC0, 0xCD, 0x80, 0xFF, // the transfer from $C000
        0xF0, 0xF1, 0xEA, 0x00, 0xD0, // keep the byte read after it at $D000
        0x7B, 0xE6, 0x07, 0x3C, 0xE0, 0xF3, 0x21, 0x9F, 0xFE, 0x3E, 0xC0, 0xCD, 0xA0, 0xFF, // the polling transfer, (E & 7) + 1 delay loops
        0x21, 0x00, 0xC0, 0x06, 0x28, 0x34, 0x2C, 0x2C, 0x2C, 0x2C, 0x05, 0x20, 0xF8, // every y + 1
        0x7B, 0xC6, 0x07, 0xFE, 0x90, 0x38, 0x02, 0xD6, 0x90, 0x5F, // E = (E + 7) mod 144
    ]);
    let back = main as i64 - (code.len() as i64 + 2);
    code.extend([0x18, back as i8 as u8]);
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}

fn frames(session: &mut Session<impl Core>, n: usize) -> Vec<(Vec<u32>, [u8; 4])> {
    (0..n)
        .map(|_| {
            session.run_frame(Buttons::default());
            let bytes = [session.ram_byte(0xFFF0), session.ram_byte(0xFFF1), session.ram_byte(0xD000), session.ram_byte(0xFFF2)];
            (session.frame().to_vec(), bytes)
        })
        .collect()
}

#[test]
fn oam_dma_runs_land_as_one_byte_an_m_cycle_would() {
    let n = 30;
    let want = frames(&mut Session::new(program(), None, Hardware::Dmg).unwrap(), n);
    let stack = Stack::parse("cpu,memory,ppu,control", Some("all")).unwrap();
    let session = Session::new(program(), None, Hardware::Dmg).unwrap().with_stack(stack, Instrumentation { ppu_same_thread: true, ..Instrumentation::default() }).unwrap();
    let mut session = session;
    let got = frames(&mut session, n);
    let report = session.report(1.0, false).join("\n");
    let asked: u64 = report.split(" steps asked again under").next().and_then(|s| s.rsplit(' ').next()).and_then(|s| s.parse().ok()).expect("the memory report counts the steps asked again");
    assert!(asked > 0, "no step's access fell past a bus change inside it:\n{report}");
    assert_eq!(want[n - 1].1, [0xFF, want[n - 1].1[1], want[n - 1].1[2], 0x00], "OAM reads 0xFF while the transfer blocks it, and sprite 39's attributes once it ends");
    assert!(want.iter().filter(|(f, _)| f.iter().any(|&p| p != f[0])).count() > n / 2, "the sprites show");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.1 == w.1 && g.0 == w.0, "frame {i}: the bytes read ({:02x?} against {:02x?}) or the pixels differ from the reference machine's", g.1, w.1);
    }
}


/// The control model's passes as a run counts them: its report's first line.
fn control_passes(session: &mut Session<gbcformer::AnyCore>) -> String {
    let report = session.report(1.0, false);
    report.into_iter().find(|l| l.starts_with("control: ") && l.contains(" passes (")).expect("the control model reports its passes")
}

/// The GBC CPU's reference beside the model observes the bytes the model reads itself
/// without touching the bus: a POP BC that OAM DMA blocks reads twice, and a lockstep
/// run must cut the transfer's run where a plain run does, pass for pass.
#[test]
fn lockstep_leaves_the_control_passes_as_a_plain_run_has_them() {
    let n = 12;
    let run = |lockstep: &str| {
        let stack = Stack::parse("cpu,memory,ppu,control", Some(lockstep)).unwrap();
        let mut session = Session::new(program(), None, Hardware::Dmg).unwrap().with_stack(stack, Instrumentation { ppu_same_thread: true, ..Instrumentation::default() }).unwrap();
        let frames = frames(&mut session, n);
        (frames, control_passes(&mut session))
    };
    let (plain, plain_passes) = run("none");
    let (beside, beside_passes) = run("cpu,memory");
    assert!(plain == beside, "the frames and bytes read differ with the GBC CPU in lockstep");
    assert_eq!(beside_passes, plain_passes, "the GBC CPU in lockstep changed the control model's passes");
}
