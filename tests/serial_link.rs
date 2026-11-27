//! Two machines on a link cable: the side clocking a transfer and the side waiting on
//! the external clock swap bytes and both raise the serial interrupt's flag.

use gbcformer::machine::Hardware;
use gbcformer::machine::io::Buttons;
use gbcformer::session::Session;

/// A ROM that loads SB, starts a transfer with SC = `sc`, waits for SC bit 7 to clear and
/// stores SB at C000, then the IF byte at C001.
fn rom(sb: u8, sc: u8) -> Vec<u8> {
    let mut rom = vec![0; 0x8000];
    #[rustfmt::skip]
    let program = [
        0x3E, sb, 0xE0, 0x01,       // LD A,sb; LDH (01),A
        0x3E, sc, 0xE0, 0x02,       // LD A,sc; LDH (02),A
        0xF0, 0x02, 0xCB, 0x7F,     // wait: LDH A,(02); BIT 7,A
        0x20, 0xFA,                 // JR NZ,wait
        0xF0, 0x01, 0xEA, 0x00, 0xC0, // LDH A,(01); LD (C000),A
        0xF0, 0x0F, 0xEA, 0x01, 0xC0, // LDH A,(0F); LD (C001),A
        0x18, 0xFE,                 // JR -2
    ];
    rom[0x100..0x100 + program.len()].copy_from_slice(&program);
    rom
}

#[test]
fn a_linked_transfer_swaps_the_bytes() {
    let mut master = Session::new(rom(0x42, 0x81), None, Hardware::Dmg).unwrap();
    let mut slave = Session::new(rom(0x99, 0x80), None, Hardware::Dmg).unwrap();
    master.link(&mut slave).unwrap();
    for _ in 0..3 {
        master.run_linked_frame(&mut slave, (Buttons(0), Buttons(0)));
    }
    let got = |s: &mut Session<_>| (s.ram_byte(0xC000), s.ram_byte(0xC001) & 0x08);
    assert_eq!(got(&mut master), (0x99, 0x08), "the clocking side shifted in the other's byte");
    assert_eq!(got(&mut slave), (0x42, 0x08), "the external-clock side completed with the clocking side's byte");
}

#[test]
fn without_a_cable_the_external_clock_never_completes() {
    let mut alone = Session::new(rom(0x99, 0x80), None, Hardware::Dmg).unwrap();
    for _ in 0..3 {
        alone.run_frame(Buttons(0));
    }
    assert_eq!(alone.ram_byte(0xC000), 0x00);
}

#[test]
fn a_linked_transfer_swaps_the_bytes_with_the_control_logic_in_weights() {
    let stack = gbcformer::Stack::parse("control", None).unwrap();
    let build = |sb, sc| Session::new(rom(sb, sc), None, Hardware::Dmg).unwrap().with_stack(stack, gbcformer::Instrumentation::default()).unwrap();
    let (mut master, mut slave) = (build(0x42, 0x81), build(0x99, 0x80));
    master.link(&mut slave).unwrap();
    for _ in 0..3 {
        master.run_linked_frame(&mut slave, (Buttons(0), Buttons(0)));
    }
    let got = |s: &mut Session<_>| (s.ram_byte(0xC000), s.ram_byte(0xC001) & 0x08);
    assert_eq!(got(&mut master), (0x99, 0x08), "the clocking side shifted in the other's byte");
    assert_eq!(got(&mut slave), (0x42, 0x08), "the external-clock side completed with the clocking side's byte");
}
