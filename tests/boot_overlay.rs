//! The boot ROM mapped by the memory model, with a boot ROM of our own: it reads a byte of
//! its 0200-08FF range, copies the header's CGB byte into KEY0 and unmaps itself at 00FE
//! with FF50, as Nintendo's does. The cartridge then stores what it reads at 0000 (its own
//! byte now), at FF4C and at FF70, and spins. A header byte of 0x80 stays in CGB mode; 0x04
//! asks for DMG compatibility, where KEY0 and SVBK read 0xFF. Each runs on the host and
//! with the GBC CPU and memory in weights under memory lockstep, which also holds the
//! model's FF50 register to the reference's rule.

use gbcformer::machine::io::Buttons;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

fn boot() -> Vec<u8> {
    let mut b = vec![0u8; 0x900];
    let code = [
        0xFA, 0x50, 0x02, // LD A,(0250): the boot ROM's own byte
        0xEA, 0x02, 0xC0, // LD (C002),A
        0xFA, 0x43, 0x01, // LD A,(0143): the header shows through
        0xE0, 0x4C, // LDH (4C),A: KEY0
        0x3E, 0x01, // LD A,1
        0xC3, 0xFE, 0x00, // JP 00FE
    ];
    b[..code.len()].copy_from_slice(&code);
    b[0xFE..0x100].copy_from_slice(&[0xE0, 0x50]); // LDH (50),A: unmapped, PC 0100
    b[0x250] = 0x5A;
    b
}

fn rom(cgb: u8) -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0] = 0xA5;
    rom[0x143] = cgb;
    rom[0x147] = 0x00;
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    let code = [
        0xFA, 0x00, 0x00, 0xEA, 0x00, 0xC0, // C000 = (0000)
        0xF0, 0x4C, 0xEA, 0x01, 0xC0, // C001 = KEY0
        0xF0, 0x70, 0xEA, 0x03, 0xC0, // C003 = SVBK
        0x18, 0xFE,
    ];
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}

fn run(cgb: u8, stack: Option<Stack>) -> [u8; 4] {
    let session = Session::power_on(rom(cgb), None, Hardware::Cgb, boot()).unwrap();
    let peek = [0xC000, 0xC001, 0xC002, 0xC003];
    match stack {
        None => {
            let mut s = session;
            for _ in 0..2 {
                s.run_frame(Buttons::default());
            }
            peek.map(|a| s.ram_byte(a))
        }
        Some(stack) => {
            let mut s = session.with_stack(stack, Instrumentation::default()).unwrap();
            for _ in 0..2 {
                s.run_frame(Buttons::default());
            }
            peek.map(|a| s.ram_byte(a))
        }
    }
}

#[test]
fn the_memory_model_maps_the_boot_rom_and_decides_compatibility_at_ff50() {
    for (cgb, want) in [(0x80, [0xA5, 0x80, 0x5A, 0xF8]), (0x04, [0xA5, 0xFF, 0x5A, 0xFF])] {
        assert_eq!(run(cgb, None), want, "header {cgb:#04x} on the host");
        let stack = Stack::parse("cpu,memory", Some("memory")).unwrap();
        assert_eq!(run(cgb, Some(stack)), want, "header {cgb:#04x} with the GBC CPU and memory in weights, in memory lockstep");
    }
}
