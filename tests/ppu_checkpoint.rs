//! The scanline checkpoint: on lines over random VRAM, OAM, palettes and registers, the
//! engine's colour bytes and window state equal the dense pass's (the stock-stack oracle
//! over the KV cache) and the circuit's; and a lockstep divergence on the PPU thread
//! stops the machine's thread.

use gbcformer::llama::{tok, Denoiser};
use gbcformer::engine::{VideoRecords, DEST};
use gbcformer::machine::ppu::{Display, Drawn, Lengths, Line, Ppu, Renderer, HEIGHT, WIDTH};
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{display_index, Banking, Mapper, LINE, LINE_BITS, OAM, OAM_LEN, PAL, PAL_LEN, PAL_REGS, VRAM, VRAM_LEN};
use gbcformer::ppu_circuit::{self, circuit, PpuCheckpoint, PpuModel, Window};
use std::collections::BTreeMap;

mod common;
use common::Rng;

/// Random VRAM, OAM whose objects mostly sit on the screen's rows, and random display bytes.
fn scene(rng: &mut Rng) -> (Image, Display) {
    let mut image = Image::new(&[0; 0x8000], 0x2000, Banking::new(Mapper::Rom, 0x8000, 0x2000, true).unwrap());
    let vram: Vec<u8> = (0..VRAM_LEN).map(|_| rng.byte()).collect();
    let oam: Vec<u8> = (0..OAM_LEN).map(|i| if i % 4 == 0 { 16 + rng.byte() % 150 } else { rng.byte() }).collect();
    image.fill(VRAM, &vram);
    image.fill(OAM, &oam);
    (image, std::array::from_fn(|_| rng.byte()))
}

/// The window's state as `WLN'` and `WAR'` hold it.
fn state(w: Window) -> (u8, u8) {
    (w.rows, w.armed as u8 | if w.carried { circuit::CARRIED_OUT } else { 0 })
}

/// A line, the window's state before it, and its display registers (LCDC, SCY, SCX, WY, WX).
/// The scanline checkpoint under lockstep with its stepper, and the length model, over
/// `image` and `display`, sharing `lengths`.
fn models(image: &Image, display: &Display, window: Window, lengths: Lengths) -> (PpuModel, ppu_circuit::LengthModel) {
    let stepper = ppu_circuit::stepper::Stepper::over(ppu_circuit::stepper::engine(), image, display);
    let model = PpuModel::new(PpuCheckpoint::new(ppu_circuit::engine(), image, display), stepper, window, true, lengths.clone());
    let length = ppu_circuit::LengthModel::new(ppu_circuit::LengthCheckpoint::new(ppu_circuit::length_engine(), image, display), window, lengths);
    (model, length)
}

fn line(rng: &mut Rng) -> (Line, Window, [u8; 5]) {
    // DMG, CGB mode or compatibility mode, a third each.
    let mode = rng.next() % 3;
    // One line in eight drawn black, as ordinary STOP keeps a CGB's PPU from VRAM.
    let l = Line { ly: rng.byte() % HEIGHT as u8, cgb: mode > 0, compat: mode == 2, blocked: rng.next() % 8 == 0, double: false };
    let regs = [0x80 | rng.byte(), rng.byte(), rng.byte(), rng.byte() % 150, if rng.next() % 4 == 0 { [0, 3, 166, 247][rng.next() as usize % 4] } else { rng.byte() % 176 }];
    // A DMG line carried in the window from WX 166 one in four.
    let window = Window { rows: rng.byte() % 144, armed: rng.next() & 1 == 1, carried: mode == 0 && rng.next() % 4 == 0 };
    (l, Window { armed: window.armed || window.carried, ..window }, regs)
}

#[test]
fn the_engine_draws_the_dense_pass_and_the_circuits_colours() {
    let mut rng = Rng(0x5EED_0F0B_1EC7_0001);
    let (image, palettes) = scene(&mut rng);
    let c = circuit::build();
    let mut cp = PpuCheckpoint::new(ppu_circuit::engine(), &image, &palettes);
    let mut memory = image.clone().with_palettes();
    memory.fill(PAL, &palettes);
    let cases: usize = std::env::var("CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    for case in 0..cases {
        // Writes between lines go through the records, as the CPU side's and the
        // palette ports' do.
        for _ in 0..64 {
            let (addr, byte) = match rng.next() % 4 {
                0 => (OAM + rng.next() as u32 % OAM_LEN as u32, rng.byte()),
                1 => (PAL + rng.next() as u32 % PAL_LEN as u32, rng.byte()),
                _ => (VRAM + rng.next() as u32 % VRAM_LEN as u32, rng.byte()),
            };
            cp.write(addr, byte);
            memory.write(addr, byte);
        }
        let (l, window, regs) = line(&mut rng);
        // Compatibility mode reaches the checkpoint as the memory model's COMPAT record.
        cp.write(ppu_circuit::COMPAT, l.compat as u8);
        for (i, &byte) in regs.iter().enumerate() {
            cp.write(PAL + PAL_REGS + i as u32, byte);
            memory.write(PAL + PAL_REGS + i as u32, byte);
        }
        for high in [false, true] {
            let (got, window_after, length) = cp.render(&l, window, high);
            let e = cp.engine();
            assert_eq!(e.stats.fallback_passes, 0, "case {case}: the engine took the dense fallback ({:?})", e.stats.refused);
            let cfg = e.cfg();
            let (tokens, positions) = cp.tokens();
            let mem = e.memory.as_ref().expect("a memory checkpoint");
            let (dense, _) = e.dense().forward_cached(&mem.kv, tokens, mem.kv.len());
            let dense: Vec<u8> = positions.iter().map(|&p| tok::output_value(cfg, dense[p]).expect("a byte token")).collect();
            assert_eq!(&got[..], &dense[..WIDTH], "case {case} high {high}: engine against the dense pass ({l:?} {regs:02x?})");
            assert_eq!(state(window_after), (dense[WIDTH], dense[WIDTH + 1]), "case {case}: the window's state against the dense pass");
            assert_eq!(length, dense[WIDTH + 2], "case {case}: the mode-3 length against the dense pass");
            let inputs: BTreeMap<String, u8> = circuit::INPUTS.iter().map(|n| n.to_string()).zip(circuit::inputs(l.ly, window, l.mode(), high, l.blocked)).collect();
            for x in [0, 7, 8, 81, 159] {
                let want = c.eval_at(&inputs, &|n| memory.line(n), Some(x));
                assert_eq!(got[x], want[&format!("PX{x}")], "case {case} high {high} pixel {x}: engine against the circuit ({l:?} {regs:02x?})");
                assert_eq!(state(window_after), (want[circuit::WINDOW[0]], want[circuit::WINDOW[1]]), "case {case}: the window's state against the circuit");
            }
        }
    }
}

/// A PPU worker handed line `ly` of a random scene with a reference line to hold it to:
/// what the checkpoint itself draws, then altered by `alter`.
fn worker_held_to(seed: u64, ly: u8, alter: impl FnOnce(&mut Drawn)) -> ppu_circuit::worker::Worker {
    let mut rng = Rng(seed);
    let (image, palettes) = scene(&mut rng);
    let mut probe = PpuCheckpoint::new(ppu_circuit::engine(), &image, &palettes);
    let (mut l, window, _) = line(&mut rng);
    l.ly = ly;
    let (low, after, length) = probe.render(&l, window, false);
    let high = if l.cgb { probe.render(&l, window, true).0 } else { [0; WIDTH] };
    let mut wrong = Drawn { colours: std::array::from_fn(|x| u16::from_le_bytes([low[x], high[x]])), window: after, length, registers: [0; 5], trace: None };
    alter(&mut wrong);
    let (model, length) = models(&image, &palettes, window, Default::default());
    let mut w = ppu_circuit::worker::Worker::spawn(model, Default::default(), length);
    w.begin(&[0; WIDTH * HEIGHT]);
    w.open(l);
    w.line(l, Some(Box::new(wrong)));
    w
}

#[test]
#[should_panic(expected = "ppu lockstep: frame 0 line 3 pixel 17")]
fn a_divergence_on_the_ppu_thread_stops_the_caller() {
    let mut w = worker_held_to(0x5EED_0F0B_1EC7_0002, 3, |d| d.colours[17] ^= 1);
    w.end_frame();
    w.front(&mut [0; WIDTH * HEIGHT]);
}

/// A CGB PPU whose lines and palette reads go to the checkpoint over `scene`, with
/// lockstep on; the records queue it shares with the bus's palette writes.
fn ppu_in_weights(seed: u64, on_thread: bool) -> (Ppu, VideoRecords) {
    let mut rng = Rng(seed);
    let (image, _) = scene(&mut rng);
    let ppu = Ppu::post_boot(true);
    let (model, length) = models(&image, &ppu.display(), Window::default(), ppu.lengths());
    let records = VideoRecords::default();
    let renderer: Box<dyn Renderer> = if on_thread { Box::new(ppu_circuit::worker::Worker::spawn(model, records.clone(), length)) } else { Box::new(ppu_circuit::Inline::new(model, records.clone(), length)) };
    let mut ppu = ppu;
    ppu.set_renderer(renderer);
    (ppu, records)
}

#[test]
fn display_reads_come_from_the_checkpoint_and_equal_the_host() {
    for on_thread in [false, true] {
        let (mut ppu, _) = ppu_in_weights(0x5EED_0F0B_1EC7_0003, on_thread);
        ppu.display_write(5, 0x42);
        ppu.write(0xFF49, 0xE4);
        ppu.write(0xFF4B, 0x57);
        let index = |addr| display_index(addr).expect("a display byte");
        assert_eq!(ppu.display_read(5), 0x42, "on thread {on_thread}");
        assert_eq!(ppu.display_read(index(0xFF49)), 0xE4, "on thread {on_thread}: OBP1");
        assert_eq!(ppu.display_read(index(0xFF4B)), 0x57, "on thread {on_thread}: WX, the last display byte");
        assert_eq!((index(0xFF4B), index(0xFF40)), (PAL_LEN as u8 - 1, PAL_REGS as u8));
        let report = ppu.renderer_mut().expect("a renderer").report().join("\n");
        assert!(report.contains("3 display byte reads from the checkpoint's memory index, 0 dense oracle checks; every one equal to the host's byte"), "on thread {on_thread}: {report}");
    }
}

#[test]
#[should_panic(expected = "ppu lockstep: display byte 5: the checkpoint read 0x43, the host 0x42")]
fn a_corrupt_palette_record_stops_the_run_inline() {
    let (mut ppu, records) = ppu_in_weights(0x5EED_0F0B_1EC7_0004, false);
    ppu.display_write(5, 0x42);
    records.lock().unwrap().push((PAL + 5, 0x43));
    ppu.display_read(5);
}

#[test]
#[should_panic(expected = "ppu lockstep: display byte 5: the checkpoint read 0x43, the host 0x42")]
fn a_corrupt_palette_record_stops_the_run_on_the_ppu_thread() {
    let (mut ppu, records) = ppu_in_weights(0x5EED_0F0B_1EC7_0005, true);
    ppu.display_write(5, 0x42);
    records.lock().unwrap().push((PAL + 5, 0x43));
    ppu.display_read(5);
}


#[test]
#[should_panic(expected = "ppu lockstep: frame 0 line 3 pixel 17")]
fn a_divergence_still_batched_when_the_worker_drops_stops_the_caller() {
    drop(worker_held_to(0x5EED_0F0B_1EC7_0002, 3, |d| d.colours[17] ^= 1));
}

/// The palette read's oracle. At a cadence of 1 every read runs the reader record at
/// `PAL + index` through the dense pass over the KV cache, and an index that disagrees
/// with it stops the run. The read comes from the cells' rows, so the oracle must catch
/// rows rebuilt for another byte while the dense cache keeps the old byte.
#[test]
fn the_dense_oracle_checks_every_palette_read_at_cadence_one() {
    let mut rng = Rng(0x5EED_0F0B_1EC7_0006);
    let (image, palettes) = scene(&mut rng);
    let mut cp = PpuCheckpoint::new(ppu_circuit::engine(), &image, &palettes);
    cp.set_oracle(1);
    cp.write(PAL + 9, 0x5A);
    assert_eq!(cp.display(9), 0x5A);
    assert_eq!(cp.display(10), palettes[10]);
    assert_eq!((cp.display_reads(), cp.display_oracle_checks()), (2, 2));
    // Rebuild the cell's rows for another byte and leave the dense oracle's cache as it was.
    let n = (PAL + 9) >> LINE_BITS;
    let mut line = cp.engine_mut().read_line(DEST, n);
    line[(PAL + 9) as usize % LINE] = 0x5B;
    cp.engine_mut().memory.as_mut().expect("a memory checkpoint").cells.put(PAL + 9, line);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cp.display(9))).expect_err("rows that disagree with the dense pass");
    let msg = caught.downcast_ref::<String>().cloned().unwrap_or_default();
    assert_eq!(msg, "assertion `left == right` failed: display oracle: byte 9: the dense pass read 0x5a, the cells' rows 0x5b\n  left: 90\n right: 91");
}

#[test]
#[should_panic(expected = "ppu lockstep: display byte 131: the checkpoint read 0x13, the host 0x91")]
fn a_corrupt_display_register_record_stops_the_run_on_the_ppu_thread() {
    let (mut ppu, records) = ppu_in_weights(0x5EED_0F0B_1EC7_0007, true);
    ppu.write(0xFF40, 0x91);
    records.lock().unwrap().push((PAL + PAL_REGS, 0x13));
    ppu.display_read(PAL_REGS as u8);
}

/// A length model held to a reference whose length is one off stops the caller, and the
/// PPU thread still shuts down on the way out: the model runs outside the queue's lock,
/// so the unwinding `Drop` closes the channel and joins the thread instead of hanging.
#[test]
#[should_panic(expected = "the length model's mode-3 length diverged")]
fn a_length_divergence_stops_the_caller_and_the_ppu_thread() {
    worker_held_to(0x5EED_0F0B_1EC7_0004, 5, |d| d.length += 1);
}

/// The window's edge cases through the machine's renderer seam: a `Ppu` drawing with the
/// checkpoint and the length model on the machine's thread, every line held to the FIFO
/// in lockstep. A DMG frame runs WX 166's carry from line to line, a carried line moved to
/// another WX, the window disabled at a pushed row's WX (issue 278) and WX 0 under every
/// fine scroll; a CGB frame runs WX 0 and WX 247 to 252 under every fine scroll with
/// objects fetched at X 0 and X 1 (the CGB's WX glitch).
#[test]
fn the_window_edge_cases_hold_in_lockstep() {
    for cgb in [false, true] {
        let mut rng = Rng(0x5EED_0F0B_1EC7_0010 + cgb as u64);
        let mut ppu = Ppu::post_boot(cgb);
        for bank in 0..1 + cgb as u8 {
            ppu.write(0xFF4F, bank);
            for a in 0x8000..0xA000u16 {
                ppu.vram_write(a, rng.byte());
            }
        }
        for e in 0..40 {
            // Objects on every row, at X 0 and X 1 among them.
            let x = [0, 1, rng.byte() % 168][e % 3];
            ppu.oam[e * 4..e * 4 + 4].copy_from_slice(&[16 + (e as u8 * 4) % 144, x, rng.byte(), rng.byte()]);
        }
        for i in 0..PAL_REGS as u8 {
            ppu.display_write(i, rng.byte());
        }
        let mut image = Image::new(&[0], 0, Banking::new(Mapper::Rom, 0x8000, 0x2000, true).unwrap()).with_palettes();
        image.fill(VRAM, ppu.vram());
        image.fill(OAM, &ppu.oam);
        let records = VideoRecords::default();
        let lengths = ppu.lengths();
        let display = ppu.display();
        let (model, length) = models(&image, &display, ppu.window(), lengths);
        ppu.set_renderer(Box::new(ppu_circuit::Inline::new(model, records, length)));
        ppu.write(0xFF4A, 0);
        for ly in 0..HEIGHT as u8 {
            let f = ly & 7;
            let (lcdc, wx) = match (cgb, ly / 16) {
                (false, 0) => (0xA3, 166),
                (false, 1) => (0xA3, if ly % 4 == 0 { 166 } else { 7 - f + 8 * (ly % 3) }),
                (false, 2) => (0x83, 7 - f + 8 * (ly % 5)),
                (false, 3) | (true, 0..=1) => (0xA3, 0),
                (true, _) => (0xA3, 247 + ly % 6),
                (false, _) => (if ly % 2 == 0 { 0xA3 } else { 0x83 }, [0, 3, 6, 166, 7 - f][ly as usize % 5]),
            };
            // The registers change in HBlank, after the last line's mode 3 (a line is 228 halves),
            // an M-cycle (2 halves) apart as the GBC CPU writes them.
            ppu.set_clock(ly as u64 * 228 + 200, false);
            ppu.write(0xFF40, lcdc);
            ppu.set_clock(ly as u64 * 228 + 202, false);
            ppu.write(0xFF43, f | (ly & 0x18));
            ppu.set_clock(ly as u64 * 228 + 204, false);
            ppu.write(0xFF4B, wx);
            ppu.set_clock((ly as u64 + 1) * 228, false);
            ppu.render_line(ly);
        }
        // VBlank closes line 143.
        ppu.set_clock(144 * 228, false);
        ppu.end_frame();
        let report = ppu.renderer_mut().expect("a renderer").report().join("\n");
        assert!(report.contains("144 lines equal to the reference, colour for colour"), "cgb {cgb}: {report}");
        assert!(report.contains("ppu length: 144 lines and ") && report.contains("(0 dense fallback)"), "cgb {cgb}: {report}");
    }
}

/// A DMG PPU drawing with the checkpoint inline under lockstep, its length model compiled
/// from `length`, and one line whose BGP write in mode 3 changes it: the judge marks it and
/// the stepper draws it, every pass held to the FIFO's.
fn a_written_line(length: gbcformer::circuit::Circuit) -> String {
    let mut rng = Rng(0x5EED_0F0B_1EC7_0020);
    let mut ppu = Ppu::post_boot(false);
    for a in 0x8000..0xA000u16 {
        ppu.vram_write(a, rng.byte());
    }
    let mut image = Image::new(&[0], 0, Banking::new(Mapper::Rom, 0x8000, 0x2000, true).unwrap()).with_palettes();
    image.fill(VRAM, ppu.vram());
    image.fill(OAM, &ppu.oam);
    let (display, lengths) = (ppu.display(), ppu.lengths());
    let stepper = ppu_circuit::stepper::Stepper::over(ppu_circuit::stepper::engine(), &image, &display);
    let model = PpuModel::new(PpuCheckpoint::new(ppu_circuit::engine(), &image, &display), stepper, ppu.window(), true, lengths.clone());
    let engine = gbcformer::engine::Engine::new(gbcformer::llama::Model::compile_mem(&length, gbcformer::memory::attn::Params::memory()));
    let length = ppu_circuit::LengthModel::new(ppu_circuit::LengthCheckpoint::new(engine, &image, &display), ppu.window(), lengths);
    ppu.set_renderer(Box::new(ppu_circuit::Inline::new(model, VideoRecords::default(), length)));
    ppu.set_clock(0, false);
    ppu.render_line(0);
    // 20 halves into mode 3, 88 half-dots: a pixel well before the line's end reads BGP.
    ppu.set_clock(20, false);
    ppu.write(0xFF47, !ppu.read(0xFF47));
    ppu.set_clock(228, false);
    ppu.render_line(1);
    ppu.set_clock(456, false);
    ppu.end_frame();
    ppu.renderer_mut().expect("a renderer").report().join("\n")
}

#[test]
fn a_line_a_write_changes_is_the_steppers_in_lockstep() {
    let report = a_written_line(circuit::build_length());
    assert!(report.contains("ppu stepper: 1 lines drawn") && report.contains("2 lines equal to the reference"), "{report}");
}

/// The mutant: a length stream whose judge passes every write. Lockstep holds each verdict to
/// the reference's, so the write that changes the line stops the run.
#[test]
#[should_panic(expected = "ppu lockstep: line 0: the length stream's judge of display write")]
fn a_judge_that_passes_a_changing_write_stops_the_run() {
    let mut c = circuit::build_length();
    let mj = c.outputs.iter().position(|(n, _)| n == circuit::LENGTH_OUTPUTS[2]).expect("the verdict");
    c.outputs[mj].1 = std::array::from_fn(|_| gbcformer::circuit::zero());
    a_written_line(c);
}
