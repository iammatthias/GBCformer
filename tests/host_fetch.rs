//! Opcode bytes fetched from a port: the memory model emits each one's address (`FP`), and
//! the host reads the device that address names and computes no fetch address itself. The
//! program calls into TMA, which holds RET, and halts at FFFE with IME off and an interrupt
//! pending, so the HALT bug fetches the byte at FFFF, IE (0x04, INC B), twice. It runs on
//! the host and with the GBC CPU and memory in weights under memory lockstep, whose audit
//! holds every emitted fetch address to the specification.

use gbcformer::machine::io::Buttons;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

fn program() -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x147] = 0x00;
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    let code: Vec<u8> = [
        &[0x3E, 0xC9, 0xE0, 0x06][..], // TMA = RET
        &[0xCD, 0x06, 0xFF],           // CALL FF06: RET fetched from control's port
        &[0x3E, 0xAA, 0xEA, 0x00, 0xC0],
        &[0x3E, 0x76, 0xE0, 0xFE], // FFFE = HALT
        &[0x3E, 0x04, 0xE0, 0xFF], // IE = 0x04 (INC B)
        &[0x3E, 0x04, 0xE0, 0x0F], // IF = timer: pending, IME off
        &[0xF3, 0x06, 0x00],       // DI, B = 0
        &[0xC3, 0xFE, 0xFF],       // JP FFFE
    ]
    .concat();
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    // After INC B twice from FFFF, PC wraps to 0000: store B and spin.
    rom[0x0000..0x0006].copy_from_slice(&[0x78, 0xEA, 0x01, 0xC0, 0x18, 0xFE]);
    rom
}

fn run(stack: Option<Stack>) -> [u8; 2] {
    let session = Session::new(program(), None, Hardware::Dmg).unwrap();
    let peek = [0xC000, 0xC001];
    match stack {
        None => {
            let mut s = session;
            for _ in 0..3 {
                s.run_frame(Buttons::default());
            }
            peek.map(|a| s.ram_byte(a))
        }
        Some(stack) => {
            let mut s = session.with_stack(stack, Instrumentation::default()).unwrap();
            for _ in 0..3 {
                s.run_frame(Buttons::default());
            }
            peek.map(|a| s.ram_byte(a))
        }
    }
}

#[test]
fn fetches_from_ports_come_from_the_address_the_model_emits() {
    let host = run(None);
    assert_eq!(host, [0xAA, 0x02], "RET from TMA returned, and INC B ran twice from IE");
    let model = run(Some(Stack::parse("cpu,memory", Some("memory")).unwrap()));
    assert_eq!(model, host, "the same with the GBC CPU and memory in weights, in memory lockstep");
}
