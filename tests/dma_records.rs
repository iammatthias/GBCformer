//! DMA copies reach the model's memory: a CGB program fills three WRAM bytes, copies 32
//! of them into VRAM bank 1 by GDMA, then 160 into OAM by OAM DMA from an HRAM routine,
//! all under `model-mem`, whose memory audit holds the model's image to the host's
//! arrays after every DMA and at each frame end.

use gbcformer::machine::io::Buttons;
use gbcformer::memory::layout;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

fn program() -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x143] = 0x80;
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    let mut code = vec![
        0x3E, 0x5A, 0xEA, 0x00, 0xC0, // LD ($C000),$5A
        0x3E, 0xA5, 0xEA, 0x1F, 0xC0, // LD ($C01F),$A5
        0x3E, 0x3C, 0xEA, 0x9F, 0xC0, // LD ($C09F),$3C
        0x3E, 0x01, 0xE0, 0x4F, // VBK = 1
        0x3E, 0xC0, 0xE0, 0x51, 0xAF, 0xE0, 0x52, // source C000
        0x3E, 0x01, 0xE0, 0x53, 0xAF, 0xE0, 0x54, // destination 8100
        0x3E, 0x01, 0xE0, 0x55, // GDMA, two blocks
    ];
    // The OAM DMA routine, copied into HRAM: the CPU sees only HRAM while it runs.
    let routine = [0x3E, 0xC0, 0xE0, 0x46, 0x3E, 0x28, 0x3D, 0x20, 0xFD, 0xC9];
    for (i, b) in routine.into_iter().enumerate() {
        code.extend([0x3E, b, 0xE0, 0x80 + i as u8]);
    }
    code.extend([0xCD, 0x80, 0xFF, 0x18, 0xFE]); // CALL $FF80, then JR to itself
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}

#[test]
fn dma_copies_are_records_in_the_models_memory() {
    let session = Session::new(program(), None, Hardware::Cgb).unwrap();
    let stack = Stack::alias("model-mem").unwrap();
    let mut s = session.with_stack(stack, Instrumentation { oracle_every: 16, ..Instrumentation::default() }).unwrap();
    for _ in 0..3 {
        s.run_frame(Buttons::default());
    }
    let engine = s.core().engine().unwrap();
    let image = &engine.memory.as_ref().unwrap().image;
    let bank1 = layout::VRAM + 0x2000;
    assert_eq!([image.read(bank1 + 0x100), image.read(bank1 + 0x11F), image.read(bank1 + 0x120), image.read(layout::VRAM + 0x100)], [0x5A, 0xA5, 0x00, 0x00], "GDMA into VRAM bank 1");
    assert_eq!([image.read(layout::OAM), image.read(layout::OAM + 0x1F), image.read(layout::OAM + 0x9F)], [0x5A, 0xA5, 0x3C], "OAM DMA");
    assert_eq!((engine.stats.copies, engine.stats.host_copies), (32 + 160, 0));
    assert!(engine.stats.oracle_checks >= 2 * 192 / 16, "every 16th record ran through the dense pass, a copy its reader and its write record: {} checks", engine.stats.oracle_checks);
}
