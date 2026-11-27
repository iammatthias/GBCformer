//! Writes `tests/fetch_locks.rs`'s program as a ROM file: it calls `LD A,$5A; RET` in VRAM
//! and in OAM at a different point of a line each time, so the PPU's lock falls between a
//! call's opcode and its operand, and a runner can be held to the full program on operand
//! fetches under the locks.
//!
//! `cargo run --release --example fetch_locks_rom -- <out.gb>`

#[path = "../tests/fetch_locks/program.rs"]
mod program;

fn main() {
    let out = std::env::args().nth(1).expect("usage: fetch_locks_rom <out.gb>");
    std::fs::write(&out, program::program()).expect("the ROM file");
}
