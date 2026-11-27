//! The length circuit (`circuit::build_length`) against the pixel FIFO (`machine::fifo`)
//! on random lines with no write during mode 3: mode 3's length in M-cycles and the
//! window's arming and carry after the line, on DMG, CGB and compatibility mode, every window
//! case among them (each counted); and the judge of writes to each line (`fifo::judge`, the
//! time from the write's halves, whether the stepper takes it). `CASES` sets the
//! number of lines (default 3000).

use gbcformer::machine::fifo::{self, Start, Window};
use gbcformer::machine::ppu::{Display, Mode};
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{Banking, Reg, N_REGS, OAM, PAL, PAL_DMG, PAL_LEN, PAL_REGS, VRAM};
use gbcformer::ppu_circuit::circuit;
use std::collections::BTreeMap;

mod common;
use common::Rng;

#[test]
fn the_length_circuit_gives_the_fifos_length() {
    let cases: usize = std::env::var("CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let c = circuit::build_length();
    let mut rng = Rng(0x5EED_1E46_7E00_0001);
    let vram = [0u8; 0x4000];
    let mut bad = Vec::new();
    let mut took = [0usize; 3];
    let mut judged = [0usize; 2];
    for case in 0..cases {
        let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][case % 3];
        let ly = (rng.next() % 144) as u8;
        let mut oam = [0u8; 0xA0];
        for e in 0..40 {
            // Most objects on this line, at any X, X 0 often.
            let y = if rng.next() % 3 == 0 { rng.byte() } else { ly.wrapping_add(16).wrapping_sub((rng.next() % 8) as u8) };
            let x = if rng.next() % 6 == 0 { 0 } else { rng.byte() };
            oam[e * 4..e * 4 + 4].copy_from_slice(&[y, x, rng.byte(), rng.byte()]);
        }
        let mut display: Display = [0; PAL_LEN];
        let regs = PAL_REGS as usize;
        // LCDC on, objects on or off, 8x8 or 8x16, the window on or off.
        display[regs] = 0x80 | rng.byte() & 0x7F;
        display[regs + 1] = rng.byte();
        display[regs + 2] = rng.byte();
        display[regs + 3] = if rng.next() % 4 == 0 { ly } else { rng.byte() };
        display[regs + 4] = match rng.next() % 6 {
            0 => (rng.next() % 7) as u8,
            1 => 166,
            2 => 247 + (rng.next() % 9) as u8,
            3 => rng.byte(),
            _ => 7 + (rng.next() % 159) as u8,
        };
        // Line 0 starts a frame: the window is unarmed and not carried before it. A DMG line
        // carries the window, armed, into the next one in four.
        let carried = mode == Mode::Dmg && ly != 0 && rng.next() % 4 == 0;
        let armed = ly != 0 && (carried || rng.next() % 2 == 0);
        let window = Window { rows: 0, armed: armed || display[regs + 3] == ly, carried };
        took[0] += (window.armed && display[regs] & 0x20 != 0 && (display[regs + 4] < 7 || display[regs + 4] == 166)) as usize;
        took[1] += (mode == Mode::Dmg && window.armed && display[regs] & 0x20 == 0 && display[regs + 4] <= 166 && display[regs + 4].wrapping_add(display[regs + 2]) & 7 == 7) as usize;
        took[2] += carried as usize;
        let objects = fifo::scan(&oam, display[regs], ly);
        let start = Start { ly, mode, double: false, blocked: false, display, window, vram: &vram, oam: &oam, objects: &objects };
        let drawn = fifo::draw(&start, &[]);

        let mut banking = Banking([0; N_REGS]);
        banking.set(Reg::Compat, (mode == Mode::Compat) as u8);
        let mut image = Image::new(&[], 0, banking).with_palettes();
        image.fill(VRAM, &vram);
        image.fill(OAM, &oam);
        image.fill(PAL, &display);
        let inputs: BTreeMap<String, u8> = circuit::LENGTH_INPUTS.iter().map(|n| n.to_string()).zip(circuit::length_inputs(ly, Window { armed, ..window }, mode, None)).collect();
        let out = c.eval_at(&inputs, &|n| image.line(n), None);
        let state = out[circuit::LENGTH_OUTPUTS[1]];
        let (length, armed_after) = (out[circuit::LENGTH_OUTPUTS[0]], (state & 1 != 0, state & circuit::CARRIED_OUT != 0));
        let want = (fifo::length(drawn.dots, display[regs + 2], mode.rgb555()), (drawn.window.armed, drawn.window.carried));
        if (length, armed_after) != want && bad.len() < 8 {
            let xs: Vec<u8> = objects.iter().map(|&i| oam[i as usize * 4 + 1]).collect();
            bad.push(format!("case {case} {mode:?} ly {ly} lcdc {:#04x} scx {} wx {} wy {} {window:?}: objects at {xs:?}: circuit {length} {armed_after:?}, fifo {want:?}", display[regs], display[regs + 2], display[regs + 4], display[regs + 3]));
        }
        // The judge of a write to this line, as the stepper's queue takes it: its time from its
        // halves, whether it marks the line, and whether it is a register's.
        for _ in 0..4 {
            let index = if rng.next() % 4 == 0 { (rng.next() % PAL_LEN as u64) as u8 } else { PAL_DMG as u8 + (rng.next() % 8) as u8 };
            let value = if rng.next() % 2 == 0 { display[index as usize] } else { rng.byte() };
            let halves = (rng.next() % 229) as u8;
            let inputs: BTreeMap<String, u8> = circuit::LENGTH_INPUTS.iter().map(|n| n.to_string()).zip(circuit::length_inputs(ly, Window { armed, ..window }, mode, Some((halves, index, value)))).collect();
            let out = c.eval_at(&inputs, &|n| image.line(n), None);
            let at = halves as fifo::Time * 4 + 8;
            let mj = out[circuit::LENGTH_OUTPUTS[2]];
            let got = (mj & circuit::JUDGE_MARK != 0, mj & circuit::JUDGE_TAKE != 0, u16::from_le_bytes([out[circuit::LENGTH_OUTPUTS[3]], out[circuit::LENGTH_OUTPUTS[4]]]) as fifo::Time);
            let want = (fifo::judge(&start, drawn.dots, fifo::Write { at, index, value }), index as u32 >= PAL_DMG, at);
            judged[want.0 as usize] += 1;
            if got != want && bad.len() < 8 {
                bad.push(format!("case {case} {mode:?} ly {ly}: write {index} = {value:#04x} at {halves} halves: circuit (marks, takes, time) {got:?}, fifo {want:?}"));
            }
        }
    }
    eprintln!("{cases} lines: {} the window armed at WX below 7 or at 166, {} a DMG armed with the window off at a pushed row's WX (issue 278), {} carried in the window from WX 166", took[0], took[1], took[2]);
    assert!(bad.is_empty(), "{bad:#?}");
    assert!(took.iter().all(|&n| n > 0), "every window case came up: {took:?}");
    assert!(judged.iter().all(|&n| n > cases / 4), "writes judged inert and marking: {judged:?}");
}
