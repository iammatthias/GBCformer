//! The HALT bug across a lock edge (docs/hardware-tests.md §5). Under the HALT bug the GBC
//! CPU reads the opcode's address twice, at M-cycles 0 and 1, the second read standing in
//! for operand 1 (Pan Docs, "halt"; SameBoy v1.0.5 `Core/sm83_cpu.c`: the run mode reads
//! the opcode with `cycle_read(gb, gb->pc++)`, then `gb->pc--` under the bug, and the
//! operand's own `cycle_read` reads memory under the bus state of its M-cycle). A program
//! runs `HALT; LD A,$5A; RET` from OAM on a DMG with IME clear and an interrupt pending, at a
//! different point of a line each time, so OAM's read lock rises between the two reads of
//! `LD A,n`'s opcode in some calls, and logs A after each call. The step's inputs carry
//! each fetched byte under its own M-cycle's state, so the log holds the same bytes on the
//! reference, in weights under lockstep, in weights alone (the host's copy of memory
//! dropped) and as one model alone. The runner's passes are held to the one model's
//! (`tools/gate.sh`).

use gbcformer::machine::io::Buttons;
use gbcformer::step::Core;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

#[path = "halt_bug_locks/program.rs"]
mod program;
use program::{LOG, program};

/// The log after 60 frames, as counts of 0x3E (no edge), 0xFF (the lock rose between the
/// two reads) and 0x01 (the opcode's own read locked), and the other bytes.
fn log(session: &mut Session<impl Core>) -> [usize; 4] {
    for _ in 0..60 {
        session.run_frame(Buttons::default());
    }
    // Fetching the shown frame waits for the PPU's worker, so its lockstep is held to the end.
    session.frame();
    let mut n = [0; 4];
    for i in 0..256 {
        n[match session.ram_byte(LOG + i) {
            0x3E => 0,
            0xFF => 1,
            0x01 => 2,
            _ => 3,
        }] += 1;
    }
    n
}

/// The reference's log, the outcome every path must reach.
const WANT: [usize; 4] = [94, 2, 160, 0];

fn in_weights(stack: Stack) -> [usize; 4] {
    let mut session = Session::new(program(), None, Hardware::Dmg).unwrap().with_stack(stack, Instrumentation { ppu_same_thread: true, ..Instrumentation::default() }).unwrap();
    log(&mut session)
}

#[test]
fn the_halt_bug_across_a_lock_edge_on_the_reference() {
    let got = log(&mut Session::new(program(), None, Hardware::Dmg).unwrap());
    assert_eq!(got, WANT, "the reference's log (0x3E, 0xFF, 0x01, other)");
    assert!(got[1] > 0, "a call where the lock rose between the two reads");
}

#[test]
fn the_halt_bug_across_a_lock_edge_in_weights() {
    assert_eq!(in_weights(Stack::parse("cpu,memory,ppu,control", Some("all")).unwrap()), WANT);
}

#[test]
fn the_halt_bug_across_a_lock_edge_in_weights_alone() {
    assert_eq!(in_weights(Stack::parse("all", None).unwrap()), WANT);
}

#[test]
fn the_halt_bug_across_a_lock_edge_as_the_one_model_alone() {
    assert_eq!(in_weights(Stack::parse("all", None).unwrap().with_one_model(true)), WANT);
}
