//! Writes a ROM that reads a display register back in the line it wrote it: on every line it
//! waits for mode 3, writes BGP and reads BGP at once, keeps the byte read at C000 on, and
//! waits for mode 0. The read lands while the line is open, after the write's record was
//! queued and before the line is drawn, so the byte comes from the record queue (the full
//! program's `held_display`, the runner's `Rows::held_record`), which the runner's proof
//! counts on both sides (`forwarded`). Run it with `--hardware dmg`.
//!
//! `cargo run --release --example display_read_rom -- <out.gb>`

fn main() {
    let out = std::env::args().nth(1).expect("usage: display_read_rom <out.gb>");
    let mut rom = vec![0u8; 0x8000];
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]); // JP $0150
    let code = [
        0xF3, // DI
        0x31, 0xFE, 0xFF, // LD SP,$FFFE
        0x21, 0x00, 0xC0, // LD HL,$C000
        0x1E, 0x00, // LD E,0
        // $0159: wait for mode 3
        0xF0, 0x41, 0xE6, 0x03, 0xFE, 0x03, 0x20, 0xF8, // LDH A,(STAT); AND 3; CP 3; JR NZ,$0159
        0x7B, 0xE0, 0x47, // LD A,E; LDH (BGP),A
        0xF0, 0x47, 0x22, 0x1C, // LDH A,(BGP); LD (HL+),A; INC E
        // $0168: wait for mode 0
        0xF0, 0x41, 0xE6, 0x03, 0x20, 0xFA, // LDH A,(STAT); AND 3; JR NZ,$0168
        0x7C, 0xFE, 0xD0, 0x20, 0x02, 0x26, 0xC0, // LD A,H; CP $D0; JR NZ,+2; LD H,$C0
        0x18, 0xE2, // JR $0159
    ];
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    std::fs::write(&out, rom).expect("the ROM file");
}
