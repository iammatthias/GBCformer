//! Battery saves with memory in weights come from the model's records: the cartridge RAM
//! its cache holds, finished when the A000 window it emits leaves cartridge RAM and the
//! clock's port after a write landed there. Each program runs on the host and with the
//! GBC CPU and memory in weights, and both must hand over the same saves.

use gbcformer::machine::io::Buttons;
use gbcformer::{Event, Hardware, Instrumentation, Session, Stack};

const ENABLE: [u8; 5] = [0x3E, 0x0A, 0xEA, 0x00, 0x00];
const DISABLE: [u8; 4] = [0xAF, 0xEA, 0x00, 0x00];
const LOOP: [u8; 2] = [0x18, 0xFE];

fn rom(cartridge_type: u8, ram_code: u8, code: &[&[u8]]) -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x147] = cartridge_type;
    rom[0x149] = ram_code;
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    let code = code.concat();
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}

fn ld(addr: u16, value: u8) -> Vec<u8> {
    vec![0x3E, value, 0xEA, addr as u8, (addr >> 8) as u8]
}

/// What a run hands over: the save events of each frame after `warmup` host frames, the
/// shutdown save, and work RAM and high RAM as `Session::ram_byte` reads them.
type Saves = (Vec<Vec<Event>>, Option<Vec<u8>>, [u8; 3]);

fn saves(rom: Vec<u8>, stack: Option<Stack>, warmup: usize) -> Saves {
    saves_from(rom, None, stack, warmup)
}

fn saves_from(rom: Vec<u8>, save: Option<Vec<u8>>, stack: Option<Stack>, warmup: usize) -> Saves {
    let mut session = Session::new(rom, save, Hardware::Dmg).unwrap();
    session.set_time(1_000_000);
    for _ in 0..warmup {
        session.run_frame(Buttons::default());
    }
    let mut events = Vec::new();
    let peek = [0xC010, 0xD020, 0xFF90];
    match stack {
        None => {
            let mut s = session;
            for _ in 0..4 {
                events.push(s.run_frame(Buttons::default()));
            }
            let peeks = peek.map(|a| s.ram_byte(a));
            (events, s.shut_down(), peeks)
        }
        Some(stack) => {
            let mut s = session.with_stack(stack, Instrumentation::default()).unwrap();
            for _ in 0..4 {
                events.push(s.run_frame(Buttons::default()));
            }
            let peeks = peek.map(|a| s.ram_byte(a));
            (events, s.shut_down(), peeks)
        }
    }
}

fn same_on_host_and_in_weights(rom: Vec<u8>, warmup: usize) -> Saves {
    let host = saves(rom.clone(), None, warmup);
    let model = saves(rom, Some(Stack::parse("cpu,memory", None).unwrap()), warmup);
    assert_eq!(model, host, "the saves with memory in weights are the host's");
    host
}

fn finished(s: &Saves) -> Vec<&Vec<u8>> {
    s.0.iter()
        .flatten()
        .map(|e| match e {
            Event::Save(b) => b,
            _ => unreachable!("only saves"),
        })
        .collect()
}

#[test]
fn saves_come_from_the_models_records() {
    let written = same_on_host_and_in_weights(rom(0x1B, 0x02, &[&ENABLE, &ld(0xA000, 0x5A), &ld(0xA123, 0xA5), &DISABLE, &ld(0xC010, 0x77), &ld(0xD020, 0x88), &ld(0xFF90, 0x99), &LOOP]), 0);
    assert_eq!(written.2, [0x77, 0x88, 0x99], "an observer reads work RAM and high RAM from the model's memory");
    let first = finished(&written);
    assert_eq!(first.len(), 1, "one finished save");
    assert_eq!((first[0].len(), first[0][0], first[0][0x123]), (0x2000, 0x5A, 0xA5));
    assert_eq!(written.1, None, "nothing written after it");

    let open = same_on_host_and_in_weights(rom(0x1B, 0x02, &[&ENABLE, &ld(0xA000, 0x5A), &LOOP]), 0);
    assert!(open.0.iter().all(|e| e.is_empty()), "RAM still enabled: no finished save");
    assert_eq!(open.1.map(|b| b[0]), Some(0x5A), "saved at shutdown");

    // Written on the host before the hand-off and never finished: still saved at shutdown.
    let handed = same_on_host_and_in_weights(rom(0x1B, 0x02, &[&ENABLE, &ld(0xA000, 0x5A), &LOOP]), 2);
    assert_eq!(handed.1.map(|b| b[0]), Some(0x5A), "the host's unsaved write carried over");

    // MBC3 with the clock: a write to a clock register is a save, as a RAM write is.
    let clock = same_on_host_and_in_weights(rom(0x10, 0x03, &[&ENABLE, &ld(0x4000, 0x08), &ld(0xA000, 0x07), &ld(0x4000, 0x00), &DISABLE, &LOOP]), 0);
    assert_eq!(finished(&clock).len(), 1, "the clock write finished a save");

    // MBC2: 512 nibbles, RAM enabled by a write with address bit 8 clear. Host and model
    // both hold each nibble as the bus reads it back, 0xF0 | n, as SameBoy's and BGB's
    // 512-byte files do, so the saves are equal.
    let mbc2 = same_on_host_and_in_weights(rom(0x06, 0x00, &[&ENABLE, &ld(0xA000, 0x5A), &ld(0xA1FF, 0x03), &DISABLE, &LOOP]), 0);
    let save = finished(&mbc2);
    assert_eq!((save.len(), save[0][0], save[0][0x1FF], save[0][1]), (1, 0xFA, 0xF3, 0xF0), "one save, nibbles as read back, unwritten cells 0xF0");

    // A loaded MBC2 save reads back 0xF0 | n whatever its file's high nibbles hold.
    let read_back = rom(0x06, 0x00, &[&ENABLE, &[0xFA, 0x00, 0xA0, 0xEA, 0x10, 0xC0], &[0xFA, 0x01, 0xA0, 0xEA, 0x20, 0xD0], &LOOP]);
    let file = [0x05, 0xF7].repeat(256);
    let host = saves_from(read_back.clone(), Some(file.clone()), None, 0);
    let model = saves_from(read_back, Some(file), Some(Stack::parse("cpu,memory", None).unwrap()), 0);
    assert_eq!(model, host, "a loaded MBC2 save reads the same with memory in weights");
    assert_eq!(host.2[..2], [0xF5, 0xF7], "A000 and A001 read 0xF0 | n");

    // A clockless MBC3 that selects 08-0C: its A000 window reads 0xFF, as disabled RAM
    // does, so the save finishes there instead of at the RAMG write, with the same bytes.
    let select = same_on_host_and_in_weights(rom(0x13, 0x03, &[&ENABLE, &ld(0xA000, 0x5A), &ld(0x4000, 0x08), &ld(0x4000, 0x00), &DISABLE, &LOOP]), 0);
    assert_eq!(finished(&select).len(), 1, "one save from a clockless select");
}
