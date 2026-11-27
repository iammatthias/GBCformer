//! Writes a CGB boot ROM of our own (2,304 bytes, no Nintendo code) that switches to DMG
//! compatibility mode while a line is open: it turns the LCD on, waits for line 10's mode 3,
//! writes KEY0 = 4 and unmaps itself at 00FE with FF50, as Nintendo's does, so the COMPAT
//! record the memory model emits lands while the PPU's row for line 10 is open. It also writes
//! a DMG cartridge whose first instruction writes BGP, still in line 10: the runner closes the
//! row at that own-memory write (`Rows::write`), the full program at the record
//! (`Ppu::set_compat`), and the proof counts the close on both sides (`compat_closes`).
//!
//! `cargo run --release --example compat_boot_rom -- <boot.bin> <cartridge.gb>`

fn main() {
    let usage = "usage: compat_boot_rom <boot.bin> <cartridge.gb>";
    let (out, cart) = (std::env::args().nth(1).expect(usage), std::env::args().nth(2).expect(usage));
    let mut b = vec![0u8; 0x900];
    let code = [
        0x31, 0xFE, 0xFF, // LD SP,$FFFE
        0x3E, 0x91, 0xE0, 0x40, // LD A,$91; LDH (LCDC),A
        // $0007: wait for line 10
        0xF0, 0x44, 0xFE, 0x0A, 0x20, 0xFA, // LDH A,(LY); CP 10; JR NZ,$0007
        // $000D: wait for its mode 3
        0xF0, 0x41, 0xE6, 0x03, 0xFE, 0x03, 0x20, 0xF8, // LDH A,(STAT); AND 3; CP 3; JR NZ,$000D
        0x3E, 0x04, 0xE0, 0x4C, // LD A,4; LDH (KEY0),A
        0x3E, 0x01, // LD A,1
        0xC3, 0xFE, 0x00, // JP $00FE
    ];
    b[..code.len()].copy_from_slice(&code);
    b[0xFE..0x100].copy_from_slice(&[0xE0, 0x50]); // LDH (FF50),A: unmapped, PC 0100
    std::fs::write(&out, b).expect("the boot ROM file");
    let mut rom = vec![0u8; 0x8000];
    rom[0x100..0x106].copy_from_slice(&[0x3E, 0xE4, 0xE0, 0x47, 0x18, 0xFE]); // LD A,$E4; LDH (BGP),A; JR -2
    std::fs::write(&cart, rom).expect("the cartridge file");
}
