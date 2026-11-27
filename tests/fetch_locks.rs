//! An operand fetch takes the access locks at its own M-cycle, as a data read does
//! (SameBoy v1.0.5 `cycle_read`: every fetch goes through `GB_read_memory`, so VRAM reads
//! 0xFF in mode 3 and OAM in modes 2 and 3). A program runs `LD A,$5A; RET` from VRAM and
//! from OAM at a different point of a line each time, so the PPU's lock falls between the
//! opcode's fetch and the operand's in some calls, and keeps each A it got. Every frame
//! and every byte kept must equal the reference machine's, in weights with every
//! component held to its reference.

use gbcformer::machine::io::Buttons;
use gbcformer::step::Core;
use gbcformer::{Hardware, Instrumentation, Session, Stack};

/// Where the program keeps the A of each call: one byte per call, OAM's and VRAM's
/// alternating, 4,096 bytes from C000.
const KEPT: u16 = 0xC000;

#[path = "fetch_locks/program.rs"]
mod program;
use program::program;

fn frames(session: &mut Session<impl Core>, n: usize) -> Vec<(Vec<u32>, Vec<u8>)> {
    (0..n)
        .map(|_| {
            session.run_frame(Buttons::default());
            let kept = (0..4096).map(|i| session.ram_byte(KEPT + i)).collect();
            (session.frame().to_vec(), kept)
        })
        .collect()
}

fn holds(hw: Hardware) {
    let n = 24;
    let want = frames(&mut Session::new(program(), None, hw).unwrap(), n);
    // The reference saw the lock fall between a call's opcode and its operand, in OAM
    // (even bytes) and in VRAM (odd bytes).
    for (region, parity) in [("OAM", 0), ("VRAM", 1)] {
        let seen = |v: u8| want.iter().any(|(_, k)| k.iter().skip(parity).step_by(2).any(|&b| b == v));
        assert!(seen(0x5A) && seen(0xFF) && seen(0xEE), "{hw:?} {region}: the reference kept {:02x?}, not each of 5A (unlocked), FF (the operand locked) and EE (the opcode locked)", want[n - 1].1);
    }
    let stack = Stack::parse("cpu,memory,ppu,control", Some("all")).unwrap();
    let mut session = Session::new(program(), None, hw).unwrap().with_stack(stack, Instrumentation { ppu_same_thread: true, ..Instrumentation::default() }).unwrap();
    let got = frames(&mut session, n);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.1 == w.1, "{hw:?} frame {i}: the bytes kept differ from the reference machine's\n got  {:02x?}\n want {:02x?}", g.1, w.1);
        assert!(g.0 == w.0, "{hw:?} frame {i}: the pixels differ from the reference machine's");
    }
}

#[test]
fn an_operand_fetch_takes_the_locks_at_its_m_cycle_dmg() {
    holds(Hardware::Dmg);
}

#[test]
fn an_operand_fetch_takes_the_locks_at_its_m_cycle_cgb() {
    holds(Hardware::Cgb);
}

