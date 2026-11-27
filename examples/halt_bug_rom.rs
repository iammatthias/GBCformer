//! Writes `tests/halt_bug_locks.rs`'s program as a ROM file: it runs `HALT; LD A,$5A; RET`
//! from OAM on a DMG with an interrupt pending and IME clear, so OAM's read lock rises
//! between the HALT bug's two reads of one opcode in some calls, and the runner can be held to the full program there.
//!
//! `cargo run --release --example halt_bug_rom -- <out.gb>`

#[path = "../tests/halt_bug_locks/program.rs"]
mod program;

fn main() {
    let out = std::env::args().nth(1).expect("usage: halt_bug_rom <out.gb>");
    std::fs::write(&out, program::program()).expect("the ROM file");
}
