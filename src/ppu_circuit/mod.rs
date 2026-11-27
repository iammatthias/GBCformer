//! The PPU's line renderer as a checkpoint (`docs/ppu-in-weights.md`):
//! the scanline circuit (`circuit`) compiled with memory heads over its own VRAM, OAM
//! and display image, one pass per visible line (two on CGB, one per byte of RGB555),
//! each pixel's final colour at 160 output positions that share the per-pixel weights.
//! The image follows the CPU side's write records to VRAM and OAM (CPU writes and DMA
//! copies, `engine::Memory::video`) and the display bytes the bus writes (BGP, OBP0,
//! OBP1, the CGB palette RAM bytes the control logic lets through, and the display
//! registers LCDC, SCY, SCX, WY and WX), in the order the CPU's thread keeps them, so a
//! line sees every record kept before mode 3 of that line and none after, which is what
//! attention over the cache up to that position reads. A line's inputs are only its LY,
//! the window's state and FLG. The window's line counter, arming and carry are the
//! checkpoint's own state, threaded from one line's outputs to the next line's inputs.
//!
//! The host maps each colour to the screen's RGB; `describe` names it.

pub mod circuit;
pub mod stepper;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker;

use crate::engine::{Engine, VideoRecords};
use crate::llama::{tok, Denoiser, Model};
use crate::machine::fifo;
use crate::machine::ppu::{rgb, Colour, Display, Drawn, Frame, Lengths, Line, LineWrite, Mode, Renderer, WIDTH};
use crate::memory::attn::{Params, READ_AT};
use crate::memory::dense::Records;
use crate::memory::image::Image;
use crate::memory::layout::{Reg, LINE, LINE_BITS, OAM, OAM_LEN, PAL, REGS, VRAM, VRAM_LEN};
use crate::sequence::{Positions, Sequence};

/// The memory checkpoint's COMPAT cell, whose records the PPU follows for FLG bit 3.
pub const COMPAT: u32 = REGS + Reg::Compat as u32;

/// The records the PPU stream follows: the COMPAT cell, and VRAM and OAM byte by byte.
pub fn video(records: VideoRecords) -> crate::engine::Video {
    crate::engine::Video { records, cells: vec![COMPAT], ranges: vec![(VRAM, OAM + OAM_LEN as u32)] }
}

/// The compiled scanline checkpoint on the engine.
pub fn engine() -> Engine {
    Engine::new(Model::compile_mem(&circuit::build(), Params::memory()))
}

/// The compiled length checkpoint (`circuit::build_length`) on the engine.
pub fn length_engine() -> Engine {
    Engine::new(Model::compile_mem(&circuit::build_length(), Params::memory()))
}

/// The window's state between lines: its line counter, whether it is armed, and whether a DMG
/// line at WX 166 carried it into the next.
pub use crate::machine::fifo::Window;

/// The scanline checkpoint with its own VRAM, OAM and display bytes.
pub struct PpuCheckpoint {
    engine: Engine,
    /// The inputs, and the mask at every output; a pass's decoded tokens stay in its store.
    tokens: Sequence,
    /// The pixel positions, x order, then the window's two outputs and the mode-3 length.
    px: Positions,
    display_reads: u64,
    display_oracle_checks: u64,
    /// DMG compatibility mode (FLG bit 3): bit 0 of the COMPAT cell as the memory
    /// checkpoint last recorded it, or as the hand-off's image holds it.
    compat: bool,
}

impl PpuCheckpoint {
    /// `engine` holding VRAM and OAM as `memory` (the machine's memory at the hand-off)
    /// has them, and the display bytes `display`.
    pub fn new(mut engine: Engine, memory: &Image, display: &Display) -> PpuCheckpoint {
        // The copy head's query is the dead address 0, so ROM byte 0 is a token too.
        let mut image = Image::new(&[memory.read(0)], 0, memory.banking()).with_palettes();
        image.fill(VRAM, memory.region(VRAM, VRAM_LEN));
        image.fill(OAM, memory.region(OAM, OAM_LEN));
        image.fill(PAL, display);
        engine.insert(image);
        let cfg = engine.cfg();
        let tokens = (0..cfg.seq()).map(|p| if p < cfg.n_inputs { tok::input(cfg, p, 0) } else { tok::output_mask(cfg, p) }).collect();
        let slot = |name: &str| engine.slots().iter().position(|n| *n == name).unwrap_or_else(|| panic!("an output slot {name}"));
        let px: Vec<usize> = (0..WIDTH).map(|x| slot(&format!("PX{x}"))).chain(circuit::WINDOW.map(slot)).chain([slot(circuit::LENGTH)]).collect();
        let compat = memory.banking().get(Reg::Compat) & 1 != 0;
        PpuCheckpoint { engine, tokens: Sequence::new(tokens), px: Positions::new(px), display_reads: 0, display_oracle_checks: 0, compat }
    }

    /// One write record to VRAM, OAM, the display bytes or the COMPAT cell.
    pub fn write(&mut self, addr: u32, byte: u8) {
        if addr == COMPAT {
            self.compat = byte & 1 != 0;
            return;
        }
        self.engine.keep_write(addr, byte);
    }

    /// How a line draws: CGB hardware is the console's, compatibility mode the COMPAT cell's.
    pub fn mode(&self, cgb: bool) -> Mode {
        match (cgb, self.compat) {
            (false, _) => Mode::Dmg,
            (true, false) => Mode::Cgb,
            (true, true) => Mode::Compat,
        }
    }

    /// The display byte at `index`, as a reader record at `PAL + index` reads it. Its copy
    /// head attends over the cells' rows (`Engine::read_line`) by the guarded lookup or by
    /// the full scan. Every `oracle_every`th read also runs that reader record through the
    /// dense pass over the KV cache, as `Engine::copy` samples a DMA source, and stops at a
    /// difference. A dense pass over the whole cache costs seconds, so the oracle only
    /// samples.
    pub fn display(&mut self, index: u8) -> u8 {
        let addr = PAL + index as u32;
        let line = self.engine.read_line(crate::engine::DEST, addr >> LINE_BITS);
        #[cfg(feature = "stock")]
        let line = self.engine.external_read(addr, addr, line);
        let byte = line[addr as usize % LINE];
        let mem = self.engine.memory.as_ref().expect("a memory checkpoint");
        if mem.oracle_every > 0 && self.display_reads % mem.oracle_every == 0 {
            let dense = self.engine.dense();
            let r = Records { cfg: &dense.cfg, slots: &dense.memory.as_ref().expect("a memory checkpoint").slots };
            let (out, _) = dense.forward_cached(&mem.kv, &r.reader(addr, addr), READ_AT);
            let got = r.read_back(&out)[addr as usize % LINE];
            assert_eq!(got, byte, "display oracle: byte {index}: the dense pass read {got:#04x}, the cells' rows {byte:#04x}");
            self.display_oracle_checks += 1;
        }
        self.display_reads += 1;
        byte
    }

    /// Runs the dense oracle on every `every`th display read and the cache's
    /// compactions (0: never), the cadence `GBCFORMER_ORACLE` sets for the memory
    /// checkpoint.
    pub fn set_oracle(&mut self, every: u64) {
        self.engine.memory.as_mut().expect("a memory checkpoint").oracle_every = every;
    }

    pub fn display_reads(&self) -> u64 {
        self.display_reads
    }

    pub fn display_oracle_checks(&self) -> u64 {
        self.display_oracle_checks
    }

    /// One pass: each pixel's colour byte (the high byte of CGB's RGB555 when `high`),
    /// the window's state after the line and its length (`fifo::length`, the dots to mode 0 less 128).
    pub fn render(&mut self, l: &Line, window: Window, high: bool) -> ([u8; WIDTH], Window, u8) {
        let inputs = circuit::inputs(l.ly, window, self.mode(l.cgb), high, l.blocked);
        for (j, b) in inputs.into_iter().enumerate() {
            self.tokens.set(j, tok::input(self.engine.cfg(), j, b));
        }
        self.engine.fill_seq(&mut self.tokens, &self.px);
        let (cfg, out, px) = (self.engine.cfg(), self.tokens.decoded(), &self.px);
        let byte = |i: usize| tok::output_value(cfg, out[px[i]]).unwrap_or_else(|| panic!("output {i} of line {} decoded token {}, not a byte", l.ly, out[px[i]]));
        let state = byte(WIDTH + 1);
        (std::array::from_fn(byte), Window { rows: byte(WIDTH), armed: state & 1 != 0, carried: state & circuit::CARRIED_OUT != 0 }, byte(WIDTH + 2))
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// The tokens of the last pass, and the positions it decoded.
    pub fn tokens(&self) -> (&[usize], &[usize]) {
        (self.tokens.tokens(), &self.px)
    }
}

/// The checkpoint drawing frames: each line's colours, mapped to the screen's RGB, into
/// the back frame, and under lockstep held to the reference's colours and window state,
/// the first difference stopping the run. A line the length stream's judge marked is drawn
/// by the stepper instead (`docs/ppu-in-weights.md` §14.2), which follows the same records.
pub struct PpuModel {
    checkpoint: PpuCheckpoint,
    stepper: stepper::Stepper,
    window: Window,
    back: Box<Frame>,
    lockstep: bool,
    frames: u64,
    lines: u64,
    /// Display byte reads held to the host's byte under lockstep.
    displays_equal: u64,
    passes: u64,
    nanos: u64,
    /// Lines the stepper drew, the lengths it revised, and the nanoseconds in both.
    stepped: (u64, u64, u64),
    /// The newest revision: its line, the writes it drew with and what it drew, which the
    /// line's drawing reuses when no write came since.
    last: Option<(u8, usize, stepper::Stepped)>,
    lengths: Lengths,
}

impl PpuModel {
    /// The checkpoint taking over with the window as `window` (the machine's at the
    /// hand-off); it publishes each line's mode-3 length to `lengths` when no reference drew the line.
    pub fn new(checkpoint: PpuCheckpoint, stepper: stepper::Stepper, window: Window, lockstep: bool, lengths: Lengths) -> PpuModel {
        PpuModel { checkpoint, stepper, window, back: Box::new([0; WIDTH * crate::machine::ppu::HEIGHT]), lockstep, frames: 0, lines: 0, displays_equal: 0, passes: 0, nanos: 0, stepped: (0, 0, 0), last: None, lengths }
    }

    pub fn write(&mut self, records: &[(u32, u8)]) {
        for &(addr, byte) in records {
            self.checkpoint.write(addr, byte);
            if addr != COMPAT {
                self.stepper.write(addr, byte);
            }
        }
    }

    /// The display byte at `index` from the checkpoint; under lockstep held to
    /// `reference`, the host's, the first difference stopping the run.
    pub fn display(&mut self, index: u8, reference: Option<u8>) -> u8 {
        let byte = self.checkpoint.display(index);
        if let Some(want) = reference {
            assert!(byte == want, "ppu lockstep: display byte {index}: the checkpoint read {byte:#04x}, the host {want:#04x} (frame {}, {} display reads equal before it)", self.frames, self.displays_equal);
            self.displays_equal += 1;
        }
        byte
    }

    /// `expected` is the line's length the length model already published: the line's own
    /// must equal it, and the first difference stops the run.
    pub fn draw(&mut self, line: &Line, reference: Option<&Drawn>, expected: Option<u8>) {
        self.check_mode(line, reference);
        let t = std::time::Instant::now();
        let (low, window, length) = self.checkpoint.render(line, self.window, false);
        // Without the length model the low pass publishes the length, so the control logic
        // waiting for it need not wait for the high byte's pass too.
        match expected {
            None if reference.is_none() => self.lengths.publish(length),
            None => {}
            Some(e) => assert_eq!(length, e, "ppu: frame {} line {}: the renderer's mode-3 length against the length model's ({} lines equal before it)", self.frames, line.ly, self.lines),
        }
        let high = if line.cgb { self.checkpoint.render(line, self.window, true).0 } else { [0; WIDTH] };
        self.passes += 1 + line.cgb as u64;
        self.nanos += t.elapsed().as_nanos() as u64;
        let colours: [Colour; WIDTH] = std::array::from_fn(|x| u16::from_le_bytes([low[x], high[x]]));
        self.finish(line, colours, window, length, reference);
    }

    /// The stepper's constants for `line`: compatibility mode from the COMPAT cell's records.
    fn constants(&self, line: &Line) -> stepper::Constants {
        stepper::Constants { ly: line.ly, mode: self.checkpoint.mode(line.cgb), double: line.double, blocked: line.blocked }
    }

    /// The stepper's drawing of `line` with `writes`, the queue the length stream took; the
    /// newest revision's when it drew the same writes.
    fn step(&mut self, line: &Line, writes: &[(fifo::Time, u8, u8)], keep: bool) -> stepper::Stepped {
        if let Some((ly, n, s)) = self.last.take()
            && ly == line.ly
            && n == writes.len()
            && (!keep || !s.states.is_empty())
        {
            return s;
        }
        let t = std::time::Instant::now();
        let c = self.constants(line);
        let s = self.stepper.line(c, self.window, writes, keep);
        self.stepped.2 += t.elapsed().as_nanos() as u64;
        s
    }

    /// The open line's length with the writes so far, which the control logic reads at the
    /// least mode-0 start (`Lengths::latest`); under lockstep held to `reference`, the length
    /// the reference has revised to by then.
    pub fn revise(&mut self, line: &Line, writes: &[(fifo::Time, u8, u8)], reference: Option<u8>) -> u8 {
        let s = self.step(line, writes, self.lockstep);
        let length = stepper::length(s.dots);
        if let Some(want) = reference {
            assert_eq!(length, want, "ppu lockstep: frame {} line {}: the stepper's mode-3 length with {} writes, as the control logic reads it, diverged", self.frames, line.ly, writes.len());
        }
        self.stepped.1 += 1;
        self.last = Some((line.ly, writes.len(), s));
        length
    }

    /// Draws a line the judge marked with the stepper; under lockstep its every state is
    /// held to the reference's trace.
    pub fn draw_stepped(&mut self, line: &Line, writes: &[(fifo::Time, u8, u8)], reference: Option<&Drawn>) {
        self.check_mode(line, reference);
        let s = self.step(line, writes, reference.is_some());
        if let Some(trace) = reference.map(|r| r.trace.as_ref().expect("ppu lockstep: a line the model's judge marked, which the reference's did not")) {
            if let Some(i) = (0..s.states.len().min(trace.len())).find(|&i| s.states[i] != trace[i]) {
                panic!("ppu lockstep: frame {} line {}: the stepper's state after pass {i} diverged\n stepper   {:?}\n reference {:?}", self.frames, line.ly, s.states[i], trace[i]);
            }
            assert_eq!(s.states.len(), trace.len(), "ppu lockstep: frame {} line {}: the stepper's passes against the reference's", self.frames, line.ly);
        }
        self.stepped.0 += 1;
        self.finish(line, s.colours, s.window, stepper::length(s.dots), reference);
    }

    fn check_mode(&self, line: &Line, reference: Option<&Drawn>) {
        if reference.is_some() {
            assert_eq!(self.checkpoint.mode(line.cgb), line.mode(), "ppu lockstep: frame {} line {}: the COMPAT cell and the reference disagree on compatibility mode", self.frames, line.ly);
        }
    }

    /// The window after a drawn line, which the next line starts from.
    pub fn window(&self) -> Window {
        self.window
    }

    /// A line's colours, held to the reference's under lockstep, into the back frame.
    fn finish(&mut self, line: &Line, colours: [Colour; WIDTH], window: Window, length: u8, reference: Option<&Drawn>) {
        if let Some(want) = reference {
            if let Some(x) = (0..WIDTH).find(|&x| colours[x] != want.colours[x]) {
                panic!(
                    "ppu lockstep: frame {} line {} pixel {x}: the checkpoint drew {:#06x}, the reference {:#06x} (the reference drew with LCDC, SCY, SCX, WY, WX {:02x?}; window {:?} cgb {}; {} lines equal before it)\n checkpoint {:04x?}\n reference  {:04x?}",
                    self.frames, line.ly, colours[x], want.colours[x], want.registers, self.window, line.cgb, self.lines, &colours[..], &want.colours[..]
                );
            }
            assert_eq!(window, want.window, "ppu lockstep: frame {} line {}: the window's line counter, arming and carry after the line diverged ({} lines equal before it)", self.frames, line.ly, self.lines);
            assert_eq!(length, want.length, "ppu lockstep: frame {} line {}: the mode-3 length diverged ({} lines equal before it)", self.frames, line.ly, self.lines);
        }
        self.window = window;
        let row = &mut self.back[line.ly as usize * WIDTH..][..WIDTH];
        for (out, &c) in row.iter_mut().zip(&colours) {
            *out = rgb(line.cgb, c);
        }
        self.lines += 1;
    }

    pub fn frame(&mut self, out: &mut Frame) {
        out.copy_from_slice(&self.back[..]);
        self.frames += 1;
    }

    pub fn describe(&self) -> Vec<String> {
        let e = self.checkpoint.engine();
        let c = &e.dense().cfg;
        let fast_path = e.exact().map_or_else(|e| format!("OFF ({e}): every pass runs dense"), |_| "on".into());
        vec![
            format!("ppu checkpoint: {} layers, hidden {}, intermediate {}, {} heads, {:.1}M parameters ({} nonzero), integer fast path {fast_path}; one pass per visible line (two on CGB, one per byte of RGB555), each pixel's final colour at 160 output positions", c.layers, c.hidden, c.intermediate, c.heads, c.params() as f64 / 1e6, e.nonzeros()),
            format!(
                "ppu: VRAM and OAM from the CPU side's write and DMA records, the display bytes (BGP, OBP0, OBP1, CGB palette RAM, and the display registers LCDC, SCY, SCX, WY and WX) from the bus's writes as records, the palette lookup and the window's line counter in the model{}; on the host: each colour mapped to the screen's RGB; when a line is drawn comes from the control logic",
                if self.lockstep { "; every line held to the reference renderer's colours and window state" } else { "" },
            ),
            format!(
                "ppu: a CPU read of BGP, OBP0, OBP1, BCPD, OCPD, LCDC, SCY, SCX, WY or WX is the newest display record at PAL+index, the copy head's attention over the cells' K/V rows (a guarded key lookup, the full scan when the guard fails), sampled by the dense oracle at the memory's oracle cadence{}",
                if self.lockstep { "; every read held to the host's byte" } else { "" }
            ),
            {
                let e = self.stepper.engine();
                let c = &e.dense().cfg;
                format!("ppu stepper: {} layers, hidden {}, intermediate {}, {:.1}M parameters ({} nonzero); one pass per step of the pixel FIFO (a dot, a change a pending display write makes, or the fetcher's code between waits), drawing each line the length stream's judge marked, from the window before it and the queue of display writes the judge took, each at the time the length stream computed; every state slot carried from a pass's outputs to the next pass's inputs{}", c.layers, c.hidden, c.intermediate, c.params() as f64 / 1e6, e.nonzeros(), if self.lockstep { "; every pass's state held to the reference FIFO's" } else { "" })
            },
        ]
    }

    pub fn report(&self) -> Vec<String> {
        let s = &self.checkpoint.engine().stats;
        let equal = if self.lockstep { format!("; {} lines equal to the reference, colour for colour", self.lines) } else { String::new() };
        let cp = &self.checkpoint;
        let displays_equal = if self.lockstep { format!("; every one equal to the host's byte ({})", self.displays_equal) } else { String::new() };
        vec![
            format!("ppu: {} lines in {} passes ({} dense fallback), {:.2} µs per line{equal}", self.lines, self.passes, s.fallback_passes, self.nanos as f64 / self.lines.max(1) as f64 / 1e3),
            format!("ppu: {} display byte reads from the checkpoint's memory index, {} dense oracle checks{displays_equal}", cp.display_reads(), cp.display_oracle_checks()),
            s.to_string(),
            format!("ppu stepper: {} lines drawn, {} lengths revised, in {} passes ({} dense fallback), {:.2} ms{}", self.stepped.0, self.stepped.1, self.stepper.passes, self.stepper.engine().stats.fallback_passes, self.stepped.2 as f64 / 1e6, match self.stepper.engine().stats.step_checks { 0 => String::new(), n => format!("; {n} fast passes checked by the step oracle, every token equal to the dense pass's") }),
        ]
    }
}

/// The records kept since the last call, oldest first.
pub fn take(records: &VideoRecords) -> Vec<(u32, u8)> {
    std::mem::take(&mut *records.lock().expect("the video record queue"))
}

/// A display byte read while a line is open: the newest record at its address among those
/// held behind the line's cut, which is what the memory reads; none when the line made
/// no write to it. Under lockstep the byte is held to the host's.
pub fn held_display(records: &VideoRecords, index: u8, reference: Option<u8>) -> Option<u8> {
    let byte = records.lock().expect("the video record queue").iter().rev().find(|(a, _)| *a == PAL + index as u32).map(|&(_, b)| b)?;
    if let Some(want) = reference {
        assert!(byte == want, "ppu lockstep: display byte {index}, written during the open line: the record holds {byte:#04x}, the host {want:#04x}");
    }
    Some(byte)
}

/// A display byte written, as a record in the queue the CPU side's video records join.
pub fn keep_display(records: &VideoRecords, index: u8, value: u8) {
    records.lock().expect("the video record queue").push((PAL + index as u32, value));
}

/// The checkpoint on the machine's thread. Each line first applies the records kept
/// before it.
pub struct Inline {
    pub model: PpuModel,
    pub records: VideoRecords,
    /// The length model, which answers each line's mode-3 length as it is handed over and
    /// judges each display write while the line is open.
    pub length: LengthModel,
    /// The frame of the last VBlank.
    front: Box<Frame>,
    /// A line opened and not drawn yet: records kept since wait for it.
    cut: bool,
    measured: Option<(u8, Window)>,
    /// The last line drawn was the stepper's: the length stream starts from its window.
    stepped_last: bool,
    /// Display reads served from the record queue while a line was open.
    forwarded: u64,
}

impl Inline {
    pub fn new(model: PpuModel, records: VideoRecords, length: LengthModel) -> Inline {
        Inline { model, records, length, front: Box::new([0; WIDTH * crate::machine::ppu::HEIGHT]), cut: false, measured: None, stepped_last: false, forwarded: 0 }
    }
}

impl Renderer for Inline {
    fn begin(&mut self, back: &Frame) {
        self.model.back.copy_from_slice(back);
    }

    fn open(&mut self, line: Line) {
        let records = take(&self.records);
        self.model.write(&records);
        self.length.write(&records);
        if self.stepped_last {
            self.length.set_window(self.model.window());
        }
        self.measured = Some(self.length.measure(&line, self.model.lockstep));
        self.cut = true;
    }

    fn line(&mut self, line: Line, reference: Option<Box<Drawn>>) {
        self.cut = false;
        let measured = self.measured.take();
        let judging = self.length.close();
        self.stepped_last = judging.as_ref().is_some_and(|j| j.marked);
        match judging {
            Some(j) if j.marked => self.model.draw_stepped(&line, &j.writes, reference.as_deref()),
            _ => {
                let expected = measured.map(|m| LengthModel::check(&line, m, reference.as_deref()));
                self.model.draw(&line, reference.as_deref(), expected);
            }
        }
    }

    fn end_frame(&mut self) {
        self.model.frame(&mut self.front);
    }

    fn front(&mut self, out: &mut Frame) {
        out.copy_from_slice(&self.front[..]);
    }

    fn display_write(&mut self, index: u8, value: u8, open: Option<LineWrite>) {
        keep_display(&self.records, index, value);
        if let Some(w) = open
            && self.length.judge(index, value, w)
        {
            let j = self.length.judging().expect("an open line");
            let reference = self.model.lockstep.then(|| self.model.lengths.current());
            let length = self.model.revise(&j.line, &j.writes, reference);
            j.revise = false;
            self.model.lengths.publish(length);
        }
    }

    fn display_read(&mut self, index: u8, reference: Option<u8>) -> u8 {
        if self.cut
            && let Some(byte) = held_display(&self.records, index, reference)
        {
            self.forwarded += 1;
            return byte;
        }
        let records = if self.cut { Vec::new() } else { take(&self.records) };
        self.model.write(&records);
        self.length.write(&records);
        self.model.display(index, reference)
    }

    fn lockstep(&self) -> bool {
        self.model.lockstep
    }

    fn describe(&self) -> Vec<String> {
        let mut lines = self.model.describe();
        lines.push("ppu: on the machine's thread".into());
        lines.extend(self.length.describe());
        lines
    }

    fn report(&mut self) -> Vec<String> {
        let mut lines = self.model.report();
        lines.extend(self.length.report());
        lines
    }

    fn stepped(&self) -> u64 {
        self.length.marked()
    }

    fn rows(&self) -> [u64; 3] {
        let [before, after] = self.length.rows();
        [self.forwarded, before, after]
    }
}

/// The length circuit's checkpoint (`circuit::build_length`): its own OAM and display
/// image, following the same records as the renderer's, one pass a line.
pub struct LengthCheckpoint {
    engine: Engine,
    tokens: Sequence,
    out_at: Positions,
    compat: bool,
}

impl LengthCheckpoint {
    pub fn new(mut engine: Engine, memory: &Image, display: &Display) -> LengthCheckpoint {
        let mut image = Image::new(&[memory.read(0)], 0, memory.banking()).with_palettes();
        image.fill(OAM, memory.region(OAM, OAM_LEN));
        image.fill(PAL, display);
        engine.insert(image);
        let cfg = engine.cfg();
        let tokens = (0..cfg.seq()).map(|p| if p < cfg.n_inputs { tok::input(cfg, p, 0) } else { tok::output_mask(cfg, p) }).collect();
        let slot = |name: &str| engine.slots().iter().position(|n| *n == name).unwrap_or_else(|| panic!("an output slot {name}"));
        let out_at = circuit::LENGTH_OUTPUTS.map(slot).to_vec();
        let compat = memory.banking().get(Reg::Compat) & 1 != 0;
        LengthCheckpoint { engine, tokens: Sequence::new(tokens), out_at: Positions::new(out_at), compat }
    }

    /// One record the renderer follows too; VRAM's, which no length reads, are left out.
    pub fn write(&mut self, addr: u32, byte: u8) {
        if addr == COMPAT {
            self.compat = byte & 1 != 0;
        } else if !(VRAM..VRAM + VRAM_LEN as u32).contains(&addr) {
            self.engine.keep_write(addr, byte);
        }
    }

    /// The line's mode-3 length and the window's arming and carry after it (its counter is the
    /// renderer's alone and reads 0 here).
    pub fn line(&mut self, l: &Line, window: Window) -> (u8, Window) {
        let (length, window, _) = self.pass(l, window, None);
        (length, window)
    }

    /// The line's pass, judging `write` (halves since mode 3 began, display index, value) when
    /// there is one: whether it marks the line, whether the stepper takes it, and its time in
    /// half-dots.
    pub fn pass(&mut self, l: &Line, window: Window, write: Option<(u8, u8, u8)>) -> (u8, Window, (bool, bool, fifo::Time)) {
        let mode = match (l.cgb, self.compat) {
            (false, _) => Mode::Dmg,
            (true, false) => Mode::Cgb,
            (true, true) => Mode::Compat,
        };
        for (j, b) in circuit::length_inputs(l.ly, window, mode, write).into_iter().enumerate() {
            self.tokens.set(j, tok::input(self.engine.cfg(), j, b));
        }
        self.engine.fill_seq(&mut self.tokens, &self.out_at);
        let (cfg, out, at) = (self.engine.cfg(), self.tokens.decoded(), &self.out_at);
        let byte = |i: usize| tok::output_value(cfg, out[at[i]]).unwrap_or_else(|| panic!("length output {i} of line {} decoded token {}, not a byte", l.ly, out[at[i]]));
        let (state, mj) = (byte(1), byte(2));
        let judged = (mj & circuit::JUDGE_MARK != 0, mj & circuit::JUDGE_TAKE != 0, u16::from_le_bytes([byte(3), byte(4)]) as fifo::Time);
        (byte(0), Window { rows: 0, armed: state & 1 != 0, carried: state & circuit::CARRIED_OUT != 0 }, judged)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }
}

/// The length model on the machine's thread: each line's mode-3 length the moment the line
/// is handed over, published for the control logic, so it never waits for the renderer.
/// The renderer's own length is held to it line by line, and under PPU lockstep it is held
/// to the reference's.
pub struct LengthModel {
    checkpoint: LengthCheckpoint,
    window: Window,
    lengths: Lengths,
    lines: u64,
    nanos: u64,
    /// The open line and what the judge said of its writes.
    open: Option<Judging>,
    /// Writes judged and lines marked.
    judged: (u64, u64),
    /// [`LengthModel::rows`].
    rows: [u64; 2],
}

/// An open line's writes as the length stream judged them: the window the line started
/// from, whether a write marked it, and the queue of writes the stepper takes, each at the
/// time the judge computed (half-dots, display index, value).
pub struct Judging {
    pub line: Line,
    before: Window,
    pub marked: bool,
    pub writes: Vec<(fifo::Time, u8, u8)>,
    /// A revision of the line's length is issued and not yet handed to the stepper.
    pub revise: bool,
    /// A write the stepper took reached the marked line before, and after, the control logic
    /// read its length.
    took_before: bool,
    took_after: bool,
}

impl LengthModel {
    pub fn new(checkpoint: LengthCheckpoint, window: Window, lengths: Lengths) -> LengthModel {
        LengthModel { checkpoint, window: Window { rows: 0, ..window }, lengths, lines: 0, nanos: 0, open: None, judged: (0, 0), rows: [0; 2] }
    }

    pub fn write(&mut self, records: &[(u32, u8)]) {
        for &(addr, byte) in records {
            self.checkpoint.write(addr, byte);
        }
    }

    /// The line's length and the window's arming and carry after it, as its mode 3 begins;
    /// the length is published unless the reference draws the line (under lockstep).
    pub fn measure(&mut self, line: &Line, lockstep: bool) -> (u8, Window) {
        let t = std::time::Instant::now();
        let before = self.window;
        let (length, window) = self.checkpoint.line(line, before);
        self.nanos += t.elapsed().as_nanos() as u64;
        if !lockstep {
            self.lengths.publish(length);
        }
        self.window = window;
        self.lines += 1;
        self.open = Some(Judging { line: *line, before, marked: false, writes: Vec::new(), revise: false, took_before: false, took_after: false });
        (length, window)
    }

    /// A display write to the open line, judged by a pass of the length stream; under
    /// lockstep its time and verdict are held to the reference's. True when a revision of the
    /// line's length is now due: the line is marked, the write is the stepper's, and the
    /// control logic has not read the length yet (`Lengths::reissue`).
    pub fn judge(&mut self, index: u8, value: u8, w: LineWrite) -> bool {
        let Some(j) = self.open.as_mut() else { return false };
        let t = std::time::Instant::now();
        let (_, _, (marks, takes, at)) = self.checkpoint.pass(&j.line, j.before, Some((w.halves, index, value)));
        self.nanos += t.elapsed().as_nanos() as u64;
        if let Some(r) = w.reference {
            assert_eq!((at, marks), (r.at, r.marks), "ppu lockstep: line {}: the length stream's judge of display write {index} = {value:#04x} at {} halves (time, marks) diverged", j.line.ly, w.halves);
        }
        self.judged.0 += 1;
        self.judged.1 += (marks && !j.marked) as u64;
        j.marked |= marks;
        if takes {
            j.writes.push((at, index, value));
        }
        let due = j.marked && takes && !j.revise && self.lengths.reissue();
        j.revise |= due;
        let unread = self.lengths.unread();
        let before = j.marked && takes && !j.took_before && unread;
        j.took_before |= before;
        self.rows[0] += before as u64;
        let after = j.marked && takes && !j.took_after && !unread;
        j.took_after |= after;
        self.rows[1] += after as u64;
        due
    }

    /// The open line's judging so far.
    pub fn judging(&mut self) -> Option<&mut Judging> {
        self.open.as_mut()
    }

    /// The open line, as it is handed over to be drawn.
    pub fn close(&mut self) -> Option<Judging> {
        self.open.take()
    }

    /// The window after a line the stepper drew, which the length stream does not draw:
    /// the next line's pass starts from it.
    pub fn set_window(&mut self, window: Window) {
        self.window = Window { rows: 0, ..window };
    }

    /// The lines the judge marked.
    pub fn marked(&self) -> u64 {
        self.judged.1
    }

    /// Lines whose length was revised before the control logic read it, and lines a write
    /// the stepper took reached after that read.
    pub fn rows(&self) -> [u64; 2] {
        self.rows
    }

    /// The measured length, held to the reference's drawing of the line when there is one.
    pub fn check(line: &Line, (length, window): (u8, Window), reference: Option<&Drawn>) -> u8 {
        if let Some(want) = reference {
            assert_eq!(length, want.length, "ppu lockstep: line {}: the length model's mode-3 length diverged", line.ly);
            assert_eq!((window.armed, window.carried), (want.window.armed, want.window.carried), "ppu lockstep: line {}: the length model's window arming and carry diverged", line.ly);
        }
        length
    }

    pub fn describe(&self) -> Vec<String> {
        let e = self.checkpoint.engine();
        let c = &e.dense().cfg;
        vec![format!("ppu length: {} layers, {:.1}M parameters ({} nonzero); each line's mode-3 length and window arming from the OAM scan, SCX and the window test, one pass a line on the machine's thread as the line is handed over, the renderer's own length held to it line by line; a pass per display write while a line is open judges it (fifo::judge: whether it can change the line, whether the stepper takes it, and its time in half-dots from its halves)", c.layers, c.params() as f64 / 1e6, e.nonzeros())]
    }

    pub fn report(&self) -> Vec<String> {
        let s = &self.checkpoint.engine().stats;
        vec![format!("ppu length: {} lines and {} writes judged, {} lines marked ({} dense fallback), {:.2} µs per pass", self.lines, self.judged.0, self.judged.1, s.fallback_passes, self.nanos as f64 / (self.lines + self.judged.0).max(1) as f64 / 1e3), s.to_string()]
    }
}
