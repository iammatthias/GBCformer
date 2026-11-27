//! The fetch-locks program (`tests/fetch_locks.rs`), shared with `examples/fetch_locks_rom.rs`,
//! which writes it as a ROM file for the runner's proof. Each call's A is kept at C000 on.

pub fn program() -> Vec<u8> {
    let mut rom = vec![0u8; 0x8000];
    rom[0x100..0x103].copy_from_slice(&[0xC3, 0x50, 0x01]);
    // RST 38H, which a call runs when an opcode of its routine reads 0xFF: drop the return
    // into the routine and return to the call, with A = $EE when the LD did not run (A is
    // 0 then) and A as the LD left it when only the RET was locked.
    rom[0x38..0x40].copy_from_slice(&[0x33, 0x33, 0xB7, 0x20, 0x02, 0x3E, 0xEE, 0xC9]);
    // The OAM routine sits at FE01, so its row's Y byte (FE00) is 0: on a DMG the OAM
    // corruption bug copies rows into rows, and with every Y 0 no copy is a sprite that
    // lengthens mode 3. After each OAM call the routine is written again at the start of
    // an HBlank (a fetch in mode 2 corrupts it there).
    let mut code = vec![
        0xF3, 0x31, 0xFE, 0xFF, // DI; LD SP,$FFFE (the stack in HRAM)
        0xAF, 0xE0, 0x40, // LCD off
        0x21, 0x00, 0x9F, 0x36, 0x3E, 0x23, 0x36, 0x5A, 0x23, 0x36, 0xC9, // LD A,$5A; RET at $9F00
        0x21, 0x01, 0xFE, 0x36, 0x3E, 0x23, 0x36, 0x5A, 0x23, 0x36, 0xC9, // and at $FE01
        0x3E, 0xE4, 0xE0, 0x47, // BGP = $E4
        0x3E, 0x93, 0xE0, 0x40, // LCD on, sprites on
        0x1E, 0x00, 0x16, 0x00, 0x21, 0x00, 0xC0, // E = 0, the line to start at; D = 0; HL = $C000
    ];
    let main = code.len();
    // Each half waits for LY = E, then (D >> 2) + base loops and D & 3 M-cycles (BIT b,D;
    // JR NZ,+0), D counting 0 to 127 one an iteration: the OAM call sweeps the next line's
    // start (OAM's lock) one M-cycle at a time, and the VRAM call, 5 loops later, its mode
    // 3 (VRAM's).
    for (base, routine) in [(0x13u8, [0x01, 0xFEu8]), (0x18, [0x00, 0x9F])] {
        code.extend([
            0xF0, 0x44, 0xBB, 0x20, 0xFB, // wait for LY = E
            0x7A, 0x0F, 0x0F, 0xE6, 0x1F, 0xC6, base, 0x3D, 0x20, 0xFD,
            0xCB, 0x42, 0x20, 0x00, 0xCB, 0x4A, 0x20, 0x00, 0xCB, 0x4A, 0x20, 0x00,
            0xAF, 0xCD, routine[0], routine[1], 0x22, // XOR A; CALL routine; LD (HL+),A
        ]);
        if routine[1] == 0xFE {
            code.extend([
                0xF0, 0x41, 0xE6, 0x03, 0xFE, 0x03, 0x20, 0xF8, // wait for mode 3
                0xF0, 0x41, 0xE6, 0x03, 0x20, 0xFA, // then for mode 0
                0x01, 0x01, 0xFE, 0x3E, 0x3E, 0x02, 0x0C, 0x3E, 0x5A, 0x02, 0x0C, 0x3E, 0xC9, 0x02, // the routine at $FE01 again
            ]);
        }
        code.extend([
            0x7B, 0xC6, 0x05, 0xFE, 0x90, 0x38, 0x02, 0xD6, 0x90, 0x5F, // E = (E + 5) mod 144
        ]);
    }
    code.extend([
        0x7A, 0x3C, 0xE6, 0x7F, 0x57, // D = (D + 1) & 127
        0x7C, 0xFE, 0xD0, 0x20, 0x02, 0x26, 0xC0, // the bytes kept wrap at $D000
    ]);
    let back = main as i64 - (code.len() as i64 + 2);
    assert!(back >= -128, "the loop is {} bytes, past JR's reach", -back);
    code.extend([0x18, back as i8 as u8]);
    rom[0x150..0x150 + code.len()].copy_from_slice(&code);
    rom
}
