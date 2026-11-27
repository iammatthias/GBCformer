//! Lines the PPU checkpoint draws while DMA copies land see the copied bytes: the
//! machine hands each block's copies to the memory checkpoint before the next block,
//! so a line drawn later in the same step's advance (or its HDMA stall) has their
//! records, as the reference renderer has the bytes.

use gbcformer::machine::Hardware;
use gbcformer::session::Session;
use gbcformer::stack::{Instrumentation, Stack};

/// A CGB ROM that bumps 256 bytes at $C000 and copies them over tile 0 (every
/// background tile) with a general-purpose HDMA, over and over: the copy lands at the
/// bus write and its stall runs across a line's mode 3.
fn gdma_rom() -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x100..0x104].copy_from_slice(&[0x00, 0xC3, 0x50, 0x01]);
    rom[0x143] = 0x80;
    let main: Vec<u8> = [
        &[0x21, 0x00, 0xC0][..],     // LD HL,$C000
        &[0x34, 0x2C, 0x20, 0xFC],   // INC (HL); INC L; JR NZ: 256 bytes + 1
        &[0x3E, 0xC0, 0xE0, 0x51],   // source $C000
        &[0xAF, 0xE0, 0x52],
        &[0x3E, 0x80, 0xE0, 0x53],   // destination $8000
        &[0xAF, 0xE0, 0x54],
        &[0x3E, 0x0F, 0xE0, 0x55],   // 16 blocks, general purpose
    ]
    .concat();
    let back = -(main.len() as i8 + 2);
    let code: Vec<u8> = [&[0x31, 0xFE, 0xFF][..], &main, &[0x18, back as u8]].concat();
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}

#[test]
fn lines_drawn_while_dma_copies_land_see_the_copied_bytes() {
    let session = Session::new(gdma_rom(), None, Hardware::Cgb).unwrap();
    let stack = Stack::parse("cpu,memory,ppu", Some("ppu")).unwrap();
    let mut session = session.with_stack(stack, Instrumentation { ppu_same_thread: true, ..Instrumentation::default() }).unwrap();
    for _ in 0..12 {
        session.run_frame(Default::default());
    }
    let report = session.report(1.0, false).join("\n");
    assert!(report.contains("1728 lines equal to the reference"), "{report}");
    assert!(report.contains("dma: ") && !report.contains("dma: 0 bytes"), "the ROM copies with HDMA: {report}");
}
