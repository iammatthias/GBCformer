//! The scanline circuit against the pixel FIFO (`machine::fifo`) on lines with no write
//! during mode 3. Over random VRAM (both banks on CGB), OAM and palettes, and frames whose
//! registers change between lines, every pixel's colour byte (low and high), the window's
//! state after the line and mode 3's length equal what `fifo::draw` gives, with each head
//! served a whole memory line. Every line in a window case (`CASES`) is evaluated, and the
//! run reports how many lines took each. `CASES` sets the number of lines evaluated
//! (default 600), and `SEED` seeds the scenes.

use gbcformer::machine::fifo::{self, Start, Window};
use gbcformer::machine::ppu::{Mode, Ppu, HEIGHT, WIDTH};
use gbcformer::memory::image::Image;
use gbcformer::memory::layout::{Banking, N_REGS, OAM, PAL, PAL_REGS, VRAM};
use gbcformer::ppu_circuit::circuit;
use std::collections::BTreeMap;

mod common;
use common::Rng;

impl Rng {
    fn one_in(&mut self, n: u64) -> bool {
        self.next() % n == 0
    }
}

/// An object Y, mostly on a few shared rows so that lines often hold more than ten.
fn object_y(rng: &mut Rng) -> u8 {
    match rng.next() % 4 {
        0 => rng.byte(),
        1 => 16 + rng.byte() % 150,
        _ => 40 + 8 * (rng.byte() % 4) + rng.byte() % 4,
    }
}

fn randomise(rng: &mut Rng, ppu: &mut Ppu, cgb: bool, writes: usize) {
    for _ in 0..writes {
        match rng.next() % 4 {
            0 => {
                let e = (rng.next() % 40) as usize * 4;
                let b = (rng.next() % 4) as usize;
                ppu.oam[e + b] = if b == 0 { object_y(rng) } else { rng.byte() };
            }
            1 => ppu.display_write((rng.next() % PAL_REGS as u64) as u8, rng.byte()),
            _ => {
                ppu.write(0xFF4F, cgb as u8 & rng.byte());
                ppu.vram_write(0x8000 + (rng.next() % 0x2000) as u16, rng.byte());
            }
        }
    }
}

fn scene(rng: &mut Rng, cgb: bool) -> Ppu {
    let mut ppu = Ppu::post_boot(cgb);
    for bank in 0..1 + cgb as u8 {
        ppu.write(0xFF4F, bank);
        for a in 0x8000..0xA000u16 {
            ppu.vram_write(a, rng.byte());
        }
    }
    for e in 0..40 {
        // X 0 and X 1 often: the CGB runs its WX glitch's fetcher steps during their fetches.
        let x = if rng.one_in(8) { rng.byte() % 2 } else { rng.byte() };
        ppu.oam[e * 4..e * 4 + 4].copy_from_slice(&[object_y(rng), x, rng.byte(), rng.byte()]);
    }
    for i in 0..PAL_REGS as u8 {
        ppu.display_write(i, rng.byte());
    }
    ppu
}

fn image_of(ppu: &Ppu) -> Image {
    let mut image = Image::new(&[], 0, Banking([0; N_REGS])).with_palettes();
    image.fill(VRAM, ppu.vram());
    image.fill(OAM, &ppu.oam);
    image.fill(PAL, ppu.palettes());
    image
}

/// Changes each of the next line's registers now and then, WX and WY mostly to values
/// where the window shows, and WX often to the window's edge cases: 0 to 6, 166 (which a
/// DMG carries into the next line) and 247 to 252 (the CGB's WX glitch).
fn registers(rng: &mut Rng, ppu: &mut Ppu) {
    for addr in [0xFF40u16, 0xFF42, 0xFF43, 0xFF4A, 0xFF4B] {
        if rng.one_in(6) {
            let v = match addr {
                0xFF4A if !rng.one_in(8) => rng.byte() % 150,
                0xFF4B => match rng.next() % 10 {
                    0 | 1 => 166,
                    2 => rng.byte() % 7,
                    3 => 247 + rng.byte() % 6,
                    4 => rng.byte(),
                    _ => rng.byte() % 168,
                },
                _ => rng.byte(),
            };
            ppu.write(addr, v);
        }
    }
}

/// The window cases `docs/ppu-in-weights.md` §14.1 brought into the circuit, and the CGB's
/// WX glitch, each read from the line's own trace.
const CASES: [&str; 4] = ["the window armed at WX below 7 or at 166", "a pixel inserted at WX with the window off (issue 278)", "a line carried in the window from WX 166", "the CGB's WX glitch"];

fn cases(start: &Start) -> [bool; 4] {
    let trace = fifo::trace(start, &[]);
    let first = &trace[0].before;
    let wx = start.display[PAL_REGS as usize + 4];
    [
        first.wy_triggered && start.display[PAL_REGS as usize] & 0x20 != 0 && (wx < 7 || wx == 166),
        trace.windows(2).any(|p| p[0].before.bg.size == 0 && p[1].before.bg.size == 1),
        start.window.carried,
        trace.iter().any(|p| p.before.cgb_wx_glitch),
    ]
}

#[test]
fn the_circuit_draws_what_the_fifo_draws() {
    let lines: usize = std::env::var("CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    let c = circuit::build();
    let mut rng = Rng(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x5EED_0F0B_11E5_0001));
    let (mut done, mut windows, mut tall, mut crowded, mut wrapped) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut took = [0usize; 4];
    // The CGB WX glitch's lines by WX, fine scroll and an object fetched at X 0 or X 1.
    let mut glitches: BTreeMap<String, usize> = BTreeMap::new();
    let mut scene_no = 0;
    while done < lines {
        // A compatibility-mode scene draws as DMG and looks its shades up in palette RAM.
        let mode = [Mode::Dmg, Mode::Cgb, Mode::Compat][scene_no % 3];
        let cgb = mode == Mode::Cgb;
        scene_no += 1;
        let mut ppu = scene(&mut rng, cgb);
        for _frame in 0..4 {
            randomise(&mut rng, &mut ppu, cgb, 200);
            if mode == Mode::Compat {
                for i in 0..128 {
                    ppu.display_write(i, rng.byte());
                }
            }
            let image = image_of(&ppu);
            let vram: [u8; 0x4000] = ppu.vram().try_into().unwrap();
            let mut window = Window::default();
            for ly in 0..HEIGHT as u8 {
                registers(&mut rng, &mut ppu);
                let display = ppu.display();
                let [lcdc, scy, scx, wy, wx] = std::array::from_fn(|i| display[PAL_REGS as usize + i]);
                if wy == ly {
                    window.armed = true;
                }
                let objects = fifo::scan(&ppu.oam, lcdc, ly);
                let start = Start { ly, mode, double: false, blocked: false, display, window, vram: &vram, oam: &ppu.oam, objects: &objects };
                let drawn = fifo::draw(&start, &[]);
                let before = window;
                window = drawn.window;
                let case = cases(&start);
                // Every line in one of the cases is drawn, and one in twelve of the rest.
                if !case.iter().any(|&c| c) && rng.next() % 12 != 0 || done >= lines {
                    continue;
                }
                done += 1;
                for (n, &c) in took.iter_mut().zip(&case) {
                    *n += c as usize;
                }
                if case[3] {
                    let at = |v: u8| objects.iter().any(|&i| ppu.oam[i as usize * 4 + 1] == v);
                    *glitches.entry(format!("wx {wx} f {} x0 {} x1 {}", scx & 7, at(0) as u8, at(1) as u8)).or_default() += 1;
                }
                let visible = (0..40).filter(|&i| (0..if lcdc & 4 != 0 { 16 } else { 8 }).contains(&(ly as i16 - (ppu.oam[i * 4] as i16 - 16)))).count();
                windows += (drawn.window.rows != before.rows) as usize;
                tall += (lcdc & 4 != 0 && visible > 0) as usize;
                crowded += (visible > 10) as usize;
                wrapped += ((scx >> 3) > 11) as usize;
                let mut image = image.clone();
                image.fill(PAL + PAL_REGS, &[lcdc, scy, scx, wy, wx]);
                let state = (drawn.window.rows, drawn.window.armed as u8 | if drawn.window.carried { circuit::CARRIED_OUT } else { 0 });
                for high in [false, true] {
                    let inputs: BTreeMap<String, u8> = circuit::INPUTS.iter().map(|n| n.to_string()).zip(circuit::inputs(ly, before, mode, high, false)).collect();
                    let got: Vec<(u8, (u8, u8), u8)> = std::thread::scope(|sc| {
                        let jobs: Vec<_> = (0..8)
                            .map(|k| {
                                let (c, inputs, image) = (&c, &inputs, &image);
                                sc.spawn(move || {
                                    (k * 20..k * 20 + 20)
                                        .map(|x| {
                                            let out = c.eval_at(inputs, &|n| image.line(n), Some(x));
                                            (out[&format!("PX{x}")], (out[circuit::WINDOW[0]], out[circuit::WINDOW[1]]), out[circuit::LENGTH])
                                        })
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        jobs.into_iter().flat_map(|j| j.join().unwrap()).collect()
                    });
                    let ctx = format!("scene {scene_no} {mode:?} ly {ly} lcdc {lcdc:#04x} scx {scx} scy {scy} wx {wx} wy {wy} window {before:?} objects {objects:?} high {high}");
                    for x in 0..WIDTH {
                        assert_eq!(got[x].0, drawn.colours[x].to_le_bytes()[high as usize], "{ctx}: pixel {x}");
                        assert_eq!(got[x].1, state, "{ctx}: the window's state after the line");
                        assert_eq!(got[x].2, gbcformer::machine::fifo::length(drawn.dots, scx, mode.rgb555()), "{ctx}: mode 3's length");
                    }
                }
            }
        }
    }
    eprintln!("{done} lines over {scene_no} scenes: {windows} drew the window, {tall} had 8x16 objects, {crowded} more than ten objects, {wrapped} a background wrapping at column 32");
    for (name, n) in CASES.iter().zip(took) {
        eprintln!("{n:5} lines: {name}");
    }
    eprintln!("the CGB's WX glitch by case: {glitches:?}");
    assert!(windows > 0 && tall > 0 && crowded > 0 && wrapped > 0, "every case the circuit picks bytes for came up");
    assert!(took.iter().all(|&n| n > 0), "every window case came up: {took:?}");
}
