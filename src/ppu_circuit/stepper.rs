//! The pixel FIFO's step as a threshold circuit (`docs/ppu-in-weights.md` §14.2, the FIFO
//! stepper): one pass computes exactly one [`fifo::State::step`], with the line's state in
//! its input slots and the next state in its output slots, as the APU threads its state per
//! segment. VRAM, OAM and CGB palette RAM are memory heads over the checkpoint's own image,
//! as the scanline circuit reads them; the display registers are state. A start pass (FLG
//! bit 4) computes [`fifo::State::new`] instead: the registers from the image, the OAM scan
//! and its fetch order, the window's state from the inputs.
//!
//! Every state field's next value is one mux over the cases that change it, with "keep" as
//! the complement case, as `sm83`'s output fields are built.

use super::circuit::{eq2, k, line_base, lit, nt, oam_query, record_slots, regs, sum, vram_query, w, Cond, Scan, B, W};
use crate::circuit::{not, one, value, zero, Circuit, Lin, Line, Query, Step, LINE};
use crate::machine::fifo::{self, BgFifo, Item, ObjFifo, Out, Pending, Phase, State, Step as Fetch, Window, OBJECTS, REGS};
use crate::machine::ppu::Mode;
use crate::engine::Engine;
use crate::llama::{tok, Denoiser, Model};
use crate::memory::attn::Params;
use crate::memory::image::Image;
use crate::memory::layout::{Banking, Mapper, OAM, PAL, PAL_DMG, VRAM};
use crate::sequence::{Positions, Sequence};
use std::collections::BTreeMap;

/// The line's constants, the start pass's window and the pending write: LY; FLG (bit 0 CGB
/// hardware, 1 CGB mode, 2 double speed, 3 blocked, 4 a start pass, 5 and 6 the start's
/// window armed and carried); the start's window rows; the write's time (two bytes), display
/// index and value; PN (bits 0 and 1 its changes applied, bit 7 a write is pending).
pub const LINE_INPUTS: [&str; 8] = ["LY", "FLG", "SROWS", "PT0", "PT1", "PI", "PV", "PN"];

/// The line's length as the PPU publishes it from the stepper's dots to mode 0
/// (`fifo::length`), as the wiring reads it (`gbc_wiring`).
pub fn length(dots: u16) -> u8 {
    u8::try_from(dots - crate::machine::lcd::LENGTH_BASE).unwrap_or_else(|_| panic!("a mode 3 of {dots} dots"))
}

/// The pass's results besides the state: OUT (bit 0 applied, 1 finished, 2 a pixel
/// emitted, 3 done); the pixel's LCD x and colour bytes (DMG's shade in the low byte), all
/// 0 with no pixel; the window after the line (rows, then bit 0 armed and bit 1 carried)
/// and the dots to mode 0 (`fifo::mode0_dots`, two bytes), which hold once OUT says done.
pub const RESULTS: [&str; 8] = ["OUT", "PXX", "PXL", "PXH", "WROWS", "WFLG", "DOTS0", "DOTS1"];

/// FLG's bits named in [`LINE_INPUTS`] that the driver sets.
pub const FLG_DOUBLE: u8 = 1 << 2;
pub const FLG_BLOCKED: u8 = 1 << 3;
pub const FLG_START: u8 = 1 << 4;
pub const FLG_ARMED: u8 = 1 << 5;
pub const FLG_CARRIED: u8 = 1 << 6;
/// PN: a write is pending.
pub const PN_PENDING: u8 = 0x80;
/// OUT's bits ([`RESULTS`]).
pub const OUT_APPLIED: u8 = 1;
pub const OUT_FINISHED: u8 = 2;
pub const OUT_PIXEL: u8 = 4;
pub const OUT_DONE: u8 = 8;
/// WFLG's bits.
pub const WFLG_ARMED: u8 = 1;
pub const WFLG_CARRIED: u8 = 2;

/// A state slot: its name and its fields, each (field, the field's first bit, width), packed
/// from bit 0 up.
type Slot = (String, Vec<(String, u32, u32)>);

/// The state's slots, in input order; the outputs carry them with a `'`.
pub fn state_slots() -> Vec<Slot> {
    let one = |s: &str, f: &str, width: u32| (s.to_string(), vec![(f.to_string(), 0, width)]);
    let mut v: Vec<Slot> = (0..REGS).map(|r| one(&format!("R{r}"), &format!("r{r}"), 8)).collect();
    let two = |v: &mut Vec<Slot>, s: &str, f: &str| {
        v.push((format!("{s}0"), vec![(f.to_string(), 0, 8)]));
        v.push((format!("{s}1"), vec![(f.to_string(), 8, 8)]));
    };
    two(&mut v, "T", "t");
    v.push(("PH".into(), vec![("phase".into(), 0, 4), ("step".into(), 0, 3)]));
    v.push(("RD".into(), vec![("resume".into(), 0, 4), ("dots".into(), 0, 3)]));
    for (s, f) in [("LX", "lx"), ("POS", "pos"), ("BGL", "bgl"), ("BGH", "bgh")] {
        v.push(one(s, f, 8));
    }
    v.push(("BGF".into(), vec![("bgpal".into(), 0, 3), ("bgprio".into(), 0, 1), ("bgsize".into(), 0, 4)]));
    for i in 0..8 {
        v.push((format!("OB{i}"), vec![(format!("pix{i}"), 0, 2), (format!("pal{i}"), 0, 3), (format!("bgp{i}"), 0, 1)]));
        v.push(one(&format!("OP{i}"), &format!("prio{i}"), 8));
    }
    for (s, f) in [("TILE", "tile"), ("ATTR", "attr"), ("D0", "d0"), ("D1", "d1")] {
        v.push(one(s, f, 8));
    }
    two(&mut v, "TIA", "tia");
    two(&mut v, "TDA", "tda");
    for (s, f) in [("FY", "fy"), ("SEL", "sel")] {
        v.push(one(s, f, 8));
    }
    for j in 0..OBJECTS {
        v.push(one(&format!("OI{j}"), &format!("oi{j}"), 8));
        v.push(one(&format!("OX{j}"), &format!("ox{j}"), 8));
    }
    v.push(("OV0".into(), vec![("ov".into(), 0, 8)]));
    v.push(("OV1".into(), vec![("ov".into(), 8, 2)]));
    for (s, f) in [("OL", "ol"), ("OH", "oh"), ("WYC", "wyc"), ("WTX", "wtx")] {
        v.push(one(s, f, 8));
    }
    let flags = |names: &[&str]| names.iter().map(|n| (n.to_string(), 0, 1)).collect::<Vec<_>>();
    v.push(("F1".into(), flags(&["lt", "tsg", "dof", "wyt", "wxt", "wibf", "ibp", "lhfs"])));
    let mut f2 = flags(&["cwg", "wjc", "dis"]);
    f2.push(("fine".into(), 0, 3));
    v.push(("F2".into(), f2));
    v
}

/// `state_slots`, made once.
fn slots() -> &'static [Slot] {
    static SLOTS: std::sync::OnceLock<Vec<Slot>> = std::sync::OnceLock::new();
    SLOTS.get_or_init(state_slots)
}

/// Every field and its width.
fn widths() -> BTreeMap<String, u32> {
    let mut m = BTreeMap::new();
    for (_, parts) in state_slots() {
        for (f, lo, width) in parts {
            *m.entry(f).or_insert(0) = (lo + width).max(*m.get(&f).unwrap_or(&0));
        }
    }
    m
}

const PHASES: [Phase; 15] = [
    Phase::Sleep,
    Phase::Init,
    Phase::Window,
    Phase::WindowOn,
    Phase::Match { first: true },
    Phase::Match { first: false },
    Phase::Render,
    Phase::Advance,
    Phase::FetchWait,
    Phase::FetchA,
    Phase::FetchB,
    Phase::FetchC,
    Phase::FetchD,
    Phase::FetchE,
    Phase::Done,
];
const STEPS: [Fetch; 7] = [Fetch::TileT1, Fetch::TileT2, Fetch::LowT1, Fetch::LowT2, Fetch::HighT1, Fetch::HighT2, Fetch::Push];

fn code(p: Phase) -> usize {
    PHASES.iter().position(|&q| q == p).expect("a phase")
}
fn step_code(s: Fetch) -> usize {
    STEPS.iter().position(|&q| q == s).expect("a fetcher step")
}

/// A state's fields by name.
/// The circuit computes a step only on states a line reaches, and these two facts of them
/// it relies on are checked here, where every pass's state is encoded: the position is
/// -16..-1 or 0..160, and the object FIFO holds priority 0 outside CGB mode.
pub fn fields(s: &State) -> BTreeMap<String, u32> {
    assert!(s.pos <= 160 || s.pos >= 240, "a line at position {}, which no line reaches", s.pos);
    assert!(s.mode.cgb_features() || s.obj.items.iter().all(|i| i.priority == 0), "an object priority outside CGB mode: {:?}", s.obj);
    let mut m: BTreeMap<String, u32> = BTreeMap::new();
    let mut put = |n: &str, v: u32| {
        m.insert(n.to_string(), v);
    };
    for (r, &b) in s.regs.iter().enumerate() {
        put(&format!("r{r}"), b as u32);
    }
    assert!(s.t < 1 << 16, "a line's time {} past 16 bits", s.t);
    put("t", s.t);
    put("phase", code(s.phase) as u32);
    put("step", step_code(s.step) as u32);
    put("resume", code(s.resume) as u32);
    put("dots", s.dots as u32);
    put("fine", s.fine as u32);
    put("lx", s.lcd_x as u32);
    put("pos", s.pos as u32);
    put("bgl", s.bg.low as u32);
    put("bgh", s.bg.high as u32);
    put("bgpal", s.bg.palette as u32);
    put("bgprio", s.bg.bg_priority as u32);
    put("bgsize", s.bg.size as u32);
    for (i, it) in s.obj.items.iter().enumerate() {
        put(&format!("pix{i}"), it.pixel as u32);
        put(&format!("pal{i}"), it.palette as u32);
        put(&format!("bgp{i}"), it.bg_priority as u32);
        put(&format!("prio{i}"), it.priority as u32);
    }
    put("tile", s.tile as u32);
    put("attr", s.attr as u32);
    put("d0", s.data[0] as u32);
    put("d1", s.data[1] as u32);
    put("tia", s.tile_index_address as u32);
    put("tda", s.tile_data_address as u32);
    put("fy", s.fetcher_y as u32);
    put("sel", s.data_for_sel_glitch as u32);
    for (j, &(i, x)) in s.objects.iter().enumerate() {
        put(&format!("oi{j}"), i as u32);
        put(&format!("ox{j}"), x as u32);
    }
    put("ov", (1u32 << s.count) - 1);
    put("ol", s.obj_low as u32);
    put("oh", s.obj_high as u32);
    put("wyc", s.window_y as u32);
    put("wtx", s.window_tile_x as u32);
    for (n, b) in [
        ("lt", s.last_tileset),
        ("tsg", s.tile_sel_glitch),
        ("dof", s.during_object_fetch),
        ("wyt", s.wy_triggered),
        ("wxt", s.wx_triggered),
        ("wibf", s.window_is_being_fetched),
        ("ibp", s.insert_bg_pixel),
        ("lhfs", s.line_has_fractional_scrolling),
        ("cwg", s.cgb_wx_glitch),
        ("wjc", s.wx_just_changed),
        ("dis", s.disable_window_pixel_insertion_glitch),
    ] {
        put(n, b as u32);
    }
    m
}

/// The line's constants a state carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constants {
    pub ly: u8,
    pub mode: Mode,
    pub double: bool,
    pub blocked: bool,
}

impl Constants {
    pub fn of(s: &State) -> Constants {
        Constants { ly: s.ly, mode: s.mode, double: s.double, blocked: s.blocked }
    }
}

/// The state whose fields are `f`, on the line `c`.
fn state(f: &BTreeMap<String, u32>, c: Constants) -> State {
    let g = |n: &str| *f.get(n).unwrap_or_else(|| panic!("no field {n}"));
    let b = |n: &str| g(n) != 0;
    let phase = |v: u32| *PHASES.get(v as usize).unwrap_or_else(|| panic!("phase code {v}"));
    let ov = g("ov");
    let count = ov.trailing_ones();
    assert_eq!(ov, (1 << count) - 1, "the objects' valid bits {ov:#b} are not a thermometer");
    State {
        ly: c.ly,
        mode: c.mode,
        double: c.double,
        blocked: c.blocked,
        regs: std::array::from_fn(|r| g(&format!("r{r}")) as u8),
        t: g("t"),
        phase: phase(g("phase")),
        resume: phase(g("resume")),
        dots: g("dots") as u8,
        lcd_x: g("lx") as u8,
        pos: g("pos") as u8,
        bg: BgFifo { low: g("bgl") as u8, high: g("bgh") as u8, palette: g("bgpal") as u8, bg_priority: b("bgprio"), size: g("bgsize") as u8 },
        obj: ObjFifo { items: std::array::from_fn(|i| Item { pixel: g(&format!("pix{i}")) as u8, palette: g(&format!("pal{i}")) as u8, priority: g(&format!("prio{i}")) as u8, bg_priority: b(&format!("bgp{i}")) }) },
        step: *STEPS.get(g("step") as usize).unwrap_or_else(|| panic!("step code {}", g("step"))),
        tile: g("tile") as u8,
        attr: g("attr") as u8,
        data: [g("d0") as u8, g("d1") as u8],
        tile_index_address: g("tia") as u16,
        tile_data_address: g("tda") as u16,
        fetcher_y: g("fy") as u8,
        last_tileset: b("lt"),
        data_for_sel_glitch: g("sel") as u8,
        tile_sel_glitch: b("tsg"),
        objects: std::array::from_fn(|j| (g(&format!("oi{j}")) as u8, g(&format!("ox{j}")) as u8)),
        count: count as u8,
        obj_low: g("ol") as u8,
        obj_high: g("oh") as u8,
        during_object_fetch: b("dof"),
        window_y: g("wyc") as u8,
        window_tile_x: g("wtx") as u8,
        wy_triggered: b("wyt"),
        wx_triggered: b("wxt"),
        window_is_being_fetched: b("wibf"),
        insert_bg_pixel: b("ibp"),
        line_has_fractional_scrolling: b("lhfs"),
        cgb_wx_glitch: b("cwg"),
        wx_just_changed: b("wjc"),
        disable_window_pixel_insertion_glitch: b("dis"),
        fine: g("fine") as u8,
    }
}

fn pack(f: &BTreeMap<String, u32>) -> Vec<u8> {
    slots().iter().map(|(_, parts)| parts.iter().scan(0, |at, (n, lo, width)| {
        let bits = ((f[n] >> lo) & ((1 << width) - 1)) << *at;
        *at += width;
        Some(bits as u8)
    }).fold(0, |a, b| a | b)).collect()
}

fn unpack(bytes: &[u8]) -> BTreeMap<String, u32> {
    let mut f: BTreeMap<String, u32> = BTreeMap::new();
    for ((_, parts), &byte) in slots().iter().zip(bytes) {
        let mut at = 0;
        for (n, lo, width) in parts {
            *f.entry(n.clone()).or_insert(0) |= ((byte as u32 >> at) & ((1 << width) - 1)) << lo;
            at += width;
        }
    }
    f
}

/// The flag byte of a line's passes.
pub fn flags(c: Constants) -> u8 {
    c.mode.rgb555() as u8 | (c.mode.cgb_features() as u8) << 1 | if c.double { FLG_DOUBLE } else { 0 } | if c.blocked { FLG_BLOCKED } else { 0 }
}

/// A step pass's input bytes, in the circuit's input order.
pub fn step_inputs(s: &State, pending: Option<Pending>) -> Vec<u8> {
    let c = Constants::of(s);
    let mut v = vec![s.ly, flags(c), 0];
    match pending {
        Some(p) => {
            assert!(p.write.at < 1 << 16 && (PAL_DMG as u8..PAL_DMG as u8 + REGS as u8).contains(&p.write.index) && p.applied < 4, "a pending write the stepper does not take: {p:?}");
            v.extend([p.write.at as u8, (p.write.at >> 8) as u8, p.write.index, p.write.value, p.applied | PN_PENDING]);
        }
        None => v.extend([0; 5]),
    }
    v.extend(pack(&fields(s)));
    v
}

/// A start pass's input bytes: the line's constants and the window before it.
pub fn start_inputs(c: Constants, window: Window) -> Vec<u8> {
    let mut v = vec![c.ly, flags(c) | FLG_START | if window.armed { FLG_ARMED } else { 0 } | if window.carried { FLG_CARRIED } else { 0 }, window.rows];
    v.extend([0; 5]);
    v.extend(vec![0; slots().len()]);
    v
}

/// What a pass decoded: the next state, what it did, and the window and length that hold
/// once it is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub state: State,
    pub out: Out,
    pub window: Window,
    pub dots: u16,
}

/// The output bytes by slot name (`out(name)`) as a pass's results on the line `c`.
pub fn decode(c: Constants, out: &dyn Fn(&str) -> u8) -> Decoded {
    let bytes: Vec<u8> = slots().iter().map(|(s, _)| out(&format!("{s}'"))).collect();
    let state = state(&unpack(&bytes), c);
    let o = out("OUT");
    let pixel = (o & OUT_PIXEL != 0).then(|| (out("PXX"), u16::from_le_bytes([out("PXL"), out("PXH")])));
    assert!(o & OUT_PIXEL != 0 || (out("PXX"), out("PXL"), out("PXH")) == (0, 0, 0), "no pixel, yet x and colour are not 0");
    let wf = out("WFLG");
    Decoded {
        state,
        out: Out { applied: o & OUT_APPLIED != 0, finished: o & OUT_FINISHED != 0, pixel, done: o & OUT_DONE != 0 },
        window: Window { rows: out("WROWS"), armed: wf & WFLG_ARMED != 0, carried: wf & WFLG_CARRIED != 0 },
        dots: u16::from_le_bytes([out("DOTS0"), out("DOTS1")]),
    }
}

/// Both conditions.
fn both(a: &Cond, b: &Cond) -> Cond {
    Cond { lin: a.lin.clone() + b.lin.clone(), n: a.n + b.n }
}

/// `w` reversed bit for bit.
fn rev(w: &[Lin]) -> W {
    w.iter().rev().cloned().collect()
}

/// `w` shifted toward bit 7 by one, bit 0 cleared (a FIFO's pop).
fn shl(w: &[Lin]) -> W {
    std::iter::once(zero()).chain(w[..7].iter().cloned()).collect()
}

/// The wires every pass shares.
struct X {
    ly: W,
    cgb: Lin,
    cgbm: Lin,
    blk: Lin,
    lcdc: W,
    scx: W,
    wx: W,
    pos: W,
    /// The position is -16..-9: `pos + 16 < 8`.
    neg8: Cond,
    neg8n: Lin,
    wx0: Lin,
    /// WX is `pos + 7` modulo 256.
    e7: Lin,
    size0: Cond,
    size0n: Lin,
    /// LY + SCY, the background's row.
    bgy: W,
}

/// The circuit's builder: the scanline circuit's helpers, every field's bits as the pass
/// reads them, the cases that change each, and the steps of the results.
struct S {
    b: B,
    cur: BTreeMap<String, W>,
    cases: BTreeMap<String, Vec<(Cond, W)>>,
    start: Lin,
    /// The pass kind the cases being added belong to, and per kind the deepest level a
    /// case's condition or value reaches before its field's mux.
    kind: &'static str,
    levels: BTreeMap<&'static str, u32>,
    applied: Vec<Step>,
    finished: Vec<Step>,
    done: Vec<Step>,
}

impl S {
    fn f(&self, n: &str) -> W {
        self.cur.get(n).unwrap_or_else(|| panic!("no field {n}")).clone()
    }
    fn bit(&self, n: &str) -> Lin {
        self.cur[n][0].clone()
    }
    /// Field `n` becomes `v` (missing high bits 0) when `cond` holds.
    fn set(&mut self, cond: &Cond, n: &str, v: W) {
        let width = self.cur.get(n).unwrap_or_else(|| panic!("no field {n}")).len();
        assert!(v.len() <= width, "{n}: {} bits for a {width}-bit field", v.len());
        let mut v = v;
        v.resize(width, zero());
        let level = self.b.c.level_of(&[cond.lin.clone()]).max(self.b.c.level_of(&v));
        let deepest = self.levels.entry(self.kind).or_insert(0);
        *deepest = (*deepest).max(level);
        self.cases.entry(n.to_string()).or_default().push((cond.clone(), v));
    }
    fn set_k(&mut self, cond: &Cond, n: &str, v: u64) {
        let width = self.cur[n].len();
        self.set(cond, n, (0..width).map(|i| k((v >> i & 1) as i64)).collect());
    }
    /// Phase `p` in a step pass.
    fn at(&self, p: Phase) -> Cond {
        Cond::spells(&self.cur["phase"][..4], code(p)).and(&nt(&self.start))
    }
    fn st(&self, s: Fetch) -> Cond {
        Cond::spells(&self.cur["step"][..3], step_code(s))
    }
    fn node(&mut self, name: &str, steps: Vec<Step>) -> Lin {
        self.b.c.node(name, steps)
    }
    fn when(&mut self, name: &str, c: &Cond) -> Lin {
        self.node(name, vec![c.step()])
    }
    /// The low `n` bits of `x + add` modulo 2^n.
    fn plus(&mut self, name: &str, x: &[Lin], n: usize, add: i64) -> W {
        let add = add.rem_euclid(1 << n);
        self.b.c.sum_bits(name, n, |j| value(x, j) + (add & ((1 << j) - 1)))
    }
    fn sleep(&mut self, cond: &Cond, dots: u64, resume: Phase) {
        self.set_k(cond, "phase", code(Phase::Sleep) as u64);
        self.set_k(cond, "dots", dots);
        self.set_k(cond, "resume", code(resume) as u64);
    }
    fn go(&mut self, cond: &Cond, p: Phase) {
        self.set_k(cond, "phase", code(p) as u64);
    }
    /// SameBoy's `start_window`.
    fn start_window(&mut self, cond: &Cond) {
        self.set_k(cond, "wxt", 1);
        self.set_k(cond, "step", step_code(Fetch::TileT1) as u64);
        self.set_k(cond, "wibf", 1);
    }
    /// The byte VRAM holds at `off` (13 bits) in `bank`, as `fifo` reads it: 0xFF while
    /// the line is blocked.
    fn read(&mut self, name: &str, bank: Lin, off: &[Lin], blk: &Lin) -> W {
        let line = self.b.head(name, 1, vram_query(bank, &off[..13]));
        self.pick(name, &line, &off[..3], blk)
    }
    fn pick(&mut self, name: &str, line: &Line, low: &[Lin], blk: &Lin) -> W {
        (0..8)
            .map(|i| {
                let mut steps: Vec<Step> = (0..LINE).map(|kk| Cond::spells(low, kk).and(&nt(blk)).and(&Lin::from(line[kk][i])).step()).collect();
                steps.push(Cond::all(&[blk.clone()]).step());
                self.node(&format!("{name}.{i}"), steps)
            })
            .collect()
    }
    /// Every field's next value: one node per bit over the cases that change it and the
    /// keep case, which holds when none of them does.
    fn finish(&mut self) -> BTreeMap<String, W> {
        let mut next = BTreeMap::new();
        let names: Vec<String> = self.cur.keys().cloned().collect();
        for n in names {
            let cur = self.f(&n);
            let cases = self.cases.remove(&n).unwrap_or_default();
            if cases.is_empty() {
                next.insert(n, cur);
                continue;
            }
            let any = self.node(&format!("{n}?"), cases.iter().map(|(c, _)| c.step()).collect());
            let bits = (0..cur.len())
                .map(|i| {
                    let mut steps: Vec<Step> = cases.iter().map(|(c, v)| c.clone().and(&v[i]).step()).collect();
                    steps.push(Cond::all(&[nt(&any), cur[i].clone()]).step());
                    self.node(&format!("{n}'{i}"), steps)
                })
                .collect();
            next.insert(n, bits);
        }
        assert!(self.cases.is_empty(), "cases for fields that do not exist: {:?}", self.cases.keys());
        next
    }
}

/// The stepper circuit: inputs `LINE_INPUTS` then the state slots, outputs the state slots
/// (each with a `'`), `RESULTS`, then the write-record slots the memory compiler expects.
pub fn build() -> Circuit {
    build_with_levels().0
}

/// `build`, and per pass kind the deepest level its cases reach before the fields' muxes.
pub fn build_with_levels() -> (Circuit, BTreeMap<&'static str, u32>) {
    let mut c = Circuit::default();
    let line: Vec<W> = LINE_INPUTS.iter().map(|n| w(c.input(n))).collect();
    let [ly, flg, srows, pt0, pt1, pi, pv, pn]: [W; 8] = line.try_into().expect("eight line inputs");
    let slots = state_slots();
    let ins: Vec<W> = slots.iter().map(|(s, _)| w(c.input(s))).collect();
    let mut cur: BTreeMap<String, W> = widths().into_iter().map(|(n, width)| (n, vec![zero(); width as usize])).collect();
    for ((_, parts), bits) in slots.iter().zip(&ins) {
        let mut at = 0;
        for (n, lo, width) in parts {
            for i in 0..*width {
                cur.get_mut(n).unwrap()[(lo + i) as usize] = bits[(at + i) as usize].clone();
            }
            at += width;
        }
    }
    // The display registers as the start pass reads them: the newest records on BGP's line.
    let image_regs = regs(&mut c);
    let b = B { c, lcdc: image_regs[3].clone(), cgb: zero(), pals: [image_regs[0].clone(), image_regs[1].clone(), image_regs[2].clone()], compat: zero() };
    let mut s = S { b, cur, cases: BTreeMap::new(), start: flg[4].clone(), kind: "Init, WindowOn", levels: BTreeMap::new(), applied: Vec::new(), finished: Vec::new(), done: Vec::new() };
    let pos = s.f("pos");
    let wx = s.f("r7");
    let size0 = Cond::spells(&s.f("bgsize"), 0);
    let neg8 = Cond::spells(&pos[3..], 0b11110);
    let scy = s.f("r4");
    let x = X {
        cgb: flg[0].clone(),
        cgbm: flg[1].clone(),
        blk: flg[3].clone(),
        lcdc: s.f("r3"),
        scx: s.f("r5"),
        neg8n: s.when("neg8", &neg8),
        neg8,
        wx0: s.b.c.and("wx0", &not(&wx)),
        e7: s.node("wx=pos+7", mod256(value(&wx, 8) - value(&pos, 8) - 7)),
        size0n: s.when("bg empty", &size0),
        size0,
        bgy: s.b.c.sum_bits("bgy", 8, |j| value(&ly, j) + value(&scy, j)),
        ly,
        wx,
        pos,
    };

    let init = s.at(Phase::Init);
    for f in ["bgl", "bgh", "bgpal", "bgprio"] {
        s.set_k(&init, f, 0);
    }
    s.set_k(&init, "bgsize", 8);
    s.go(&init, Phase::Window);
    let on = s.at(Phase::WindowOn);
    s.start_window(&on);
    s.go(&on, Phase::Match { first: true });
    let done = s.at(Phase::Done);
    s.done.push(done.step());

    sleep_pass(&mut s, &x, &[pt0, pt1].concat(), &pi, &pv, &pn, &flg[2]);
    window_pass(&mut s, &x);
    match_pass(&mut s, &x);
    let (emitted, colour) = render_pass(&mut s, &x);
    fetcher(&mut s, &x);
    object_fetch(&mut s, &x);
    start_pass(&mut s, &x, &srows, &flg, &image_regs);

    // The window after the line and mode 3's length, from the pass's own state: the pass
    // that ends a line changes neither.
    let wyt = s.bit("wyt");
    let carried = s.when("carried", &both(&Cond::all(&[nt(&x.cgb), wyt.clone(), x.lcdc[5].clone()]), &Cond::spells(&x.wx, 166)));
    let wyc = s.f("wyc");
    let rows = s.b.c.sum_bits("rows", 8, |j| value(&wyc, j) + if j > 0 { one() + carried.clone() } else { zero() });
    let t = s.f("t");
    // The dots to mode 0 (`fifo::mode0_dots`): the time's dots, and on a DMG (−fine) mod 4,
    // fine bit 0 plus twice bit 0 xor bit 1; a CGB's are the time's.
    let fine = s.f("fine");
    let x01 = s.b.c.eq("fine xor", fine[0].clone() + fine[1].clone(), 1);
    let f0 = s.b.c.and("fine.0 dmg", &[fine[0].clone(), nt(&x.cgb)]);
    let x01 = s.b.c.and("fine xor dmg", &[x01, nt(&x.cgb)]);
    let up = f0.clone() + x01 * 2;
    let dots = s.b.c.sum_bits("dots", 17, |j| value(&t, j) + (j > 0) as i64 + if j >= 3 { up.clone() * 2 } else if j == 2 { f0.clone() * 2 } else { zero() });
    let lx = s.f("lx");
    let pxx: W = (0..8).map(|i| s.b.c.and(&format!("pxx.{i}"), &[emitted.clone(), lx[i].clone()])).collect();
    let (applied, finished, done) = (std::mem::take(&mut s.applied), std::mem::take(&mut s.finished), std::mem::take(&mut s.done));
    let applied = s.node("applied", applied);
    let finished = s.node("finished", finished);
    let done = s.node("done", done);

    let next = s.finish();
    for (slot, parts) in &slots {
        let mut bits: Vec<Lin> = Vec::new();
        for (n, lo, width) in parts {
            bits.extend(next[n][*lo as usize..(*lo + *width) as usize].iter().cloned());
        }
        bits.resize(8, zero());
        s.b.c.output(&format!("{slot}'"), bits.try_into().unwrap());
    }
    let byte = |v: &[Lin]| -> [Lin; 8] { std::array::from_fn(|i| v.get(i).cloned().unwrap_or_else(zero)) };
    s.b.c.output("OUT", byte(&[applied, finished, emitted, done]));
    s.b.c.output("PXX", byte(&pxx));
    s.b.c.output("PXL", byte(&colour[..8]));
    s.b.c.output("PXH", byte(&colour[8..]));
    s.b.c.output("WROWS", byte(&rows));
    s.b.c.output("WFLG", byte(&[wyt, carried]));
    s.b.c.output("DOTS0", byte(&dots[1..9]));
    s.b.c.output("DOTS1", byte(&dots[9..17]));
    record_slots(&mut s.b.c);
    s.b.c.schedule(crate::llama::UNIT_QUANTUM);
    (s.b.c, s.levels)
}

/// `[f ≡ 0 mod 256]` for a form in (−512, 256).
fn mod256(f: Lin) -> Vec<Step> {
    [eq2(f.clone(), 0), eq2(f, -256)].concat()
}

/// A sleeping line: a dot passes, or the pending write's next change applies first if it
/// lands by that dot's end. The changes, their times and which is a write's last come from
/// `fifo::act`, tabled over the console, the speed, the register and the change number.
fn sleep_pass(s: &mut S, x: &X, at: &[Lin], pi: &[Lin], pv: &[Lin], pn: &[Lin], dbl: &Lin) {
    s.kind = "Sleep";
    let sl = s.at(Phase::Sleep);
    let t = s.f("t");
    let lcdc = x.lcdc.clone();
    let (dof, wibf) = (s.bit("dof"), s.bit("wibf"));
    let mut cmp: BTreeMap<i64, Lin> = BTreeMap::new();
    let mut dues: Vec<Lin> = Vec::new();
    let pos0 = s.when("pos=0", &Cond::spells(&x.pos, 0));
    let mut abort_times: BTreeMap<i64, (W, Lin)> = BTreeMap::new();
    for cg in 0..2 {
        for db in 0..2 {
            for r in 0..REGS {
                for n in 0..4u8 {
                    let w = fifo::Write { at: 1000, index: PAL_DMG as u8 + r as u8, value: 0 };
                    let Some((time, act)) = fifo::act(cg == 1, db == 1, w, n) else { continue };
                    let off = time as i64 - 1000;
                    let last = fifo::act(cg == 1, db == 1, w, n + 1).is_none();
                    let cmp = cmp.entry(off).or_insert_with(|| s.b.c.ge(&format!("due@{off}"), value(&t, 16) + 2 - value(at, 16) - off, 0)).clone();
                    let which = both(&both(&Cond::all(&[pn[7].clone(), lit(&x.cgb, cg), lit(dbl, db)]), &Cond::spells(&pi[..3], r)), &Cond::spells(&pn[..2], n as usize));
                    let due = s.when(&format!("due{cg}{db}{r}{n}"), &which.and(&cmp));
                    dues.push(due.clone());
                    let dc = sl.clone().and(&due);
                    s.applied.push(dc.step());
                    if last {
                        s.finished.push(dc.step());
                    }
                    let reg = format!("r{r}");
                    let old = s.f(&reg);
                    let mut abort = None;
                    match act {
                        fifo::Act::Set(..) => s.set(&dc, &reg, pv.to_vec()),
                        fifo::Act::Or(..) => {
                            let v = (0..8).map(|i| s.b.c.or(&format!("or{r}.{i}"), &[old[i].clone(), pv[i].clone()])).collect();
                            s.set(&dc, &reg, v);
                        }
                        fifo::Act::DmgLcdc(_) => {
                            // The address bits take the new value; WIN_EN, and BG_EN at x 0, only clear;
                            // OBJ_EN clears at x 0 or during an object fetch (`Act::DmgLcdc`).
                            let b0 = s.node("dmg lcdc.0", vec![Cond::all(&[old[0].clone(), pv[0].clone()]).step(), Cond::all(&[old[0].clone(), nt(&pv[0]), nt(&pos0)]).step()]);
                            let b5 = s.b.c.and("dmg lcdc.5", &[old[5].clone(), pv[5].clone()]);
                            let b1 = s.node("dmg lcdc.1", vec![Cond::all(&[old[1].clone(), pv[1].clone()]).step(), Cond::all(&[old[1].clone(), nt(&pv[1]), nt(&pos0), nt(&dof)]).step()]);
                            let v = vec![b0, b1, pv[2].clone(), pv[3].clone(), pv[4].clone(), b5, pv[6].clone(), old[7].clone()];
                            s.set(&dc, &reg, v);
                            s.set_k(&dc.clone().and(&lcdc[5]).and(&nt(&pv[5])).and(&wibf), "dis", 1);
                            abort = Some(off);
                        }
                        fifo::Act::DmgLcdcFinal(_) => {
                            s.set(&dc, &reg, pv.to_vec());
                            abort = Some(off);
                        }
                        fifo::Act::CgbLcdc(_) => {
                            s.set(&dc, &reg, pv.to_vec());
                            s.set_k(&dc.clone().and(&nt(&pv[4])).and(&lcdc[4]), "tsg", 1);
                        }
                        fifo::Act::CgbDoubleLcdc(_) => {
                            let x4 = s.node("tile sel changed", vec![Cond::all(&[pv[4].clone(), nt(&lcdc[4])]).step(), Cond::all(&[nt(&pv[4]), lcdc[4].clone()]).step()]);
                            s.set(&dc, "tsg", vec![x4]);
                            let v = (0..8).map(|i| if i == 0 || i == 7 { old[i].clone() } else { pv[i].clone() }).collect();
                            s.set(&dc, &reg, v);
                        }
                        fifo::Act::WxJustChanged(b) => s.set_k(&dc, "wjc", b as u64),
                        fifo::Act::TileSelGlitch(b) => s.set_k(&dc, "tsg", b as u64),
                    }
                    // A DMG LCDC write clearing OBJ_EN during an object fetch aborts it: the
                    // wait ends at the write, and the line renders next.
                    if let Some(off) = abort {
                        let ac = dc.clone().and(&lcdc[1]).and(&dof).and(&nt(&pv[1]));
                        let (time, nonneg) = abort_times.entry(off).or_insert_with(|| {
                            let time = s.plus(&format!("at{off}"), at, 16, off);
                            let nonneg = s.b.c.ge(&format!("at{off}>=0"), value(at, 16) + off, 0);
                            (time, nonneg)
                        }).clone();
                        s.set(&ac.clone().and(&nonneg), "t", time);
                        s.set_k(&ac.clone().and(&nt(&nonneg)), "t", 0);
                        s.set_k(&ac, "dots", 0);
                        s.go(&ac, Phase::Render);
                    }
                }
            }
        }
    }
    let tick = sl.and(&(one() - sum(dues)));
    let t2 = s.plus("t+2", &t, 16, 2);
    s.set(&tick, "t", t2);
    let dots = s.f("dots");
    let d1 = s.plus("dots-1", &dots, 3, -1);
    s.set(&tick, "dots", d1);
    let resume = s.f("resume");
    s.set(&both(&tick, &Cond::spells(&dots, 1)), "phase", resume);
}

/// The window starts when the line reaches WX (`State::window`), or a pixel is inserted.
fn window_pass(s: &mut S, x: &X) {
    s.kind = "Window";
    let wn = s.at(Phase::Window);
    let (wxt, wyt, wibf, wjc) = (s.bit("wxt"), s.bit("wyt"), s.bit("wibf"), s.bit("wjc"));
    let (pos, scx, wx) = (&x.pos, &x.scx, &x.wx);
    let a = wn.clone().and(&nt(&wxt)).and(&wyt).and(&x.lcdc[5]);
    let p = |v: usize| Cond::spells(pos, v);
    let scx7 = value(scx, 3);
    let act0 = s.node(
        "window at wx 0",
        vec![x.neg8.ge(value(pos, 3), 1, 7), p(248).step(), p(249).step(), p(240).ge(scx7.clone(), 1, 7)],
    );
    let lt166 = s.b.c.ge("wx<166+cgb", k(165) + x.cgb.clone() - value(wx, 8), 0);
    let e6 = s.node("wx=pos+6", mod256(value(wx, 8) - value(pos, 8) - 6));
    let scxnz = s.b.c.ge("scx&7", scx7, 1);
    let wyc = s.f("wyc");
    let wyinc = s.plus("window_y+1", &wyc, 8, 1);
    let a1 = a.clone().and(&x.wx0).and(&act0);
    let a1s = a1.clone().and(&nt(&x.cgb)).and(&scxnz);
    let a2 = a.clone().and(&nt(&x.wx0)).and(&lt166).and(&x.e7);
    let a3 = a.clone().and(&nt(&x.wx0)).and(&lt166).and(&nt(&x.cgb)).and(&e6).and(&nt(&wjc));
    let starts = [a1.clone().and(&x.cgb), a1.clone().and(&nt(&x.cgb)).and(&nt(&scxnz)), a2.clone(), a3.clone()];
    for c in [a1s.clone()].iter().chain(&starts) {
        s.set(c, "wyc", wyinc.clone());
        s.set_k(c, "wtx", 0);
        for f in ["bgl", "bgh", "bgpal", "bgprio", "bgsize"] {
            s.set_k(c, f, 0);
        }
    }
    for c in &starts {
        s.start_window(c);
    }
    let w166 = both(&a.clone().and(&nt(&x.cgb)).and(&x.e7), &Cond::spells(wx, 166));
    s.set(&w166, "wyc", wyinc);
    let ins = both(&wn.clone().and(&x.e7).and(&wxt).and(&nt(&wibf)), &both(&s.st(Fetch::TileT1), &Cond::spells(&s.f("bgsize"), 8)));
    s.set_k(&ins.clone().and(&nt(&x.cgb)), "ibp", 1);
    s.set_k(&ins.and(&x.cgb).and(&x.wx0), "ibp", 1);
    let a1sn = s.when("window waits a dot", &a1s);
    s.sleep(&a1s, 1, Phase::WindowOn);
    s.go(&wn.and(&nt(&a1sn)), Phase::Match { first: true });
}

/// The next object, if it is at this X (`Phase::Match`): on the first match of a dot the
/// objects already passed leave the queue first.
fn match_pass(s: &mut S, x: &X) {
    s.kind = "Match";
    let (mf, mn) = (s.at(Phase::Match { first: true }), s.at(Phase::Match { first: false }));
    let ov = s.f("ov");
    let xs: Vec<W> = (0..OBJECTS).map(|j| s.f(&format!("ox{j}"))).collect();
    // ok_j: object j is valid and left of the match point max(0, pos + 8), pos signed: a
    // position of 240 or more (`hi`) is negative, -16..-1.
    let hi = Cond::spells(&x.pos[4..], 0b1111);
    let pv = value(&x.pos, 8);
    let ok: W = (0..OBJECTS)
        .map(|j| {
            let (v, xj) = (Cond::all(&[ov[j].clone()]), value(&xs[j], 8));
            let mut minus = both(&v, &hi).ge(pv.clone() + 8 - xj.clone(), 1, 263);
            minus.coef = -1;
            let steps = vec![v.ge(pv.clone() + 8 - xj.clone(), 1, 263), minus, both(&v, &hi).ge(pv.clone() - 248 - xj, 1, 7)];
            s.node(&format!("passed{j}"), steps)
        })
        .collect();
    // eq_j: object j's X is the match point.
    let eq: W = (0..OBJECTS)
        .map(|j| {
            let xj = value(&xs[j], 8);
            let mut steps = eq2(xj.clone() - pv.clone() - 8, 0);
            steps.extend(hi.eq(xj.clone() - pv.clone() - 8, 0, 247).map(|st| Step { coef: -st.coef, ..st }));
            steps.extend(hi.eq(xj.clone() - pv.clone() + 248, 0, 503));
            steps.push(hi.clone().and(&nt(&x.pos[3])).ge(-sum(xs[j].iter().cloned()), 0, 0));
            s.node(&format!("at object {j}"), steps)
        })
        .collect();
    // thr[m]: the first m objects are all passed, so the queue drops at least m.
    let mut thr: W = vec![one()];
    for m in 1..=OBJECTS {
        thr.push(s.b.c.ge(&format!("drops {m}"), sum(ok[..m].iter().cloned()), m as i64));
    }
    thr.push(zero());
    let objen = s.b.c.or("objects enabled", &[x.lcdc[1].clone(), x.cgb.clone()]);
    for m in 1..=OBJECTS {
        let dm = mf.clone().and(&(thr[m].clone() - thr[m + 1].clone()));
        shift_objects(s, &dm, m);
    }
    s.set_k(&mf, "dof", 1);
    let mut steps: Vec<Step> = (0..OBJECTS).map(|m| mf.clone().and(&(thr[m].clone() - thr[m + 1].clone())).and(&objen).and(&ov[m]).and(&eq[m]).step()).collect();
    steps.push(mn.and(&objen).and(&ov[0]).and(&eq[0]).step());
    let fetch = s.node("fetch object", steps);
    let either = Cond::spells(&s.f("phase")[1..4], 0b010).and(&nt(&s.start));
    s.go(&either.clone().and(&fetch), Phase::FetchWait);
    s.go(&either.and(&nt(&fetch)), Phase::Render);
}

/// The object queue less its first `m` objects.
fn shift_objects(s: &mut S, cond: &Cond, m: usize) {
    for j in 0..OBJECTS {
        for f in ["oi", "ox"] {
            let v = if j + m < OBJECTS { s.f(&format!("{f}{}", j + m)) } else { vec![] };
            s.set(cond, &format!("{f}{j}"), v);
        }
    }
    let ov = s.f("ov");
    s.set(cond, "ov", ov[m..].to_vec());
}

/// `render_pixel_if_possible`: the pixel, its colour, and the fine scroll. Returns whether
/// a pixel leaves for the screen and its colour's 16 bits (0 when none does).
fn render_pass(s: &mut S, x: &X) -> (Lin, W) {
    s.kind = "Render";
    let r = s.at(Phase::Render);
    let ov = s.f("ov");
    let x0z = Cond::spells(&s.f("ox0"), 0);
    let (lcdc, cgb, cgbm, blk) = (&x.lcdc, &x.cgb, &x.cgbm, &x.blk);
    let waits = s.node("object at 0", vec![both(&Cond::all(&[lcdc[1].clone(), ov[0].clone()]), &x0z).step(), both(&Cond::all(&[nt(&lcdc[1]), cgb.clone(), ov[0].clone()]), &x0z).step()]);
    s.set_k(&r, "dof", 0);
    s.go(&r, Phase::Advance);
    let go = r.and(&nt(&waits)).and(&nt(&x.size0n));
    let ibp = s.bit("ibp");
    s.set_k(&go, "ibp", 0);
    let pop = go.clone().and(&nt(&ibp));
    let (bgl, bgh, size) = (s.f("bgl"), s.f("bgh"), s.f("bgsize"));
    s.set(&pop, "bgl", shl(&bgl));
    s.set(&pop, "bgh", shl(&bgh));
    let size1 = s.plus("bg size-1", &size, 4, -1);
    s.set(&pop, "bgsize", size1);
    for i in 0..8 {
        for f in ["pix", "pal", "bgp", "prio"] {
            let v = if i < 7 { s.f(&format!("{f}{}", i + 1)) } else { vec![] };
            s.set(&go, &format!("{f}{i}"), v);
        }
    }
    let (pos, scx) = (&x.pos, &x.scx);
    let wibf = s.bit("wibf");
    let fe = s.node("pos&7=scx&7", eq2(value(pos, 3) - value(scx, 3), 0));
    let f2 = both(&Cond::all(&[wibf.clone()]), &both(&Cond::spells(&pos[..3], 6), &Cond::spells(&scx[..3], 7)));
    let jump = s.node("fine scroll done", vec![x.neg8.clone().and(&fe).step(), both(&x.neg8, &f2).step()]);
    let p247 = s.when("pos=-9", &Cond::spells(pos, 247));
    let pos1 = s.plus("pos+1", pos, 8, 1);
    let to = |v: u64| (0..8).map(|i| k((v >> i & 1) as i64)).collect::<W>();
    let j = go.clone().and(&jump);
    s.set(&j, "pos", to(249));
    s.set_k(&j, "wibf", 0);
    s.set(&both(&go.clone().and(&nt(&fe)), &Cond::spells(pos, 247)), "pos", to(240));
    let frac = both(&go, &x.neg8).and(&nt(&jump)).and(&nt(&p247));
    s.set_k(&frac, "lhfs", 1);
    s.set_k(&frac, "wibf", 0);
    s.set(&frac, "pos", pos1.clone());
    let off = both(&go, &Cond::spells(&pos[3..], 0b11111));
    s.set_k(&off, "wibf", 0);
    s.set(&off, "pos", pos1.clone());
    let negative = s.when("pos<0", &Cond::spells(&pos[4..], 0b1111));
    let px = go.clone().and(&nt(&negative));
    s.set_k(&px, "wibf", 0);
    s.set(&px, "pos", pos1);
    let lx = s.f("lx");
    let lx1 = s.plus("lx+1", &lx, 8, 1);
    s.set(&px, "lx", lx1);
    let lt160 = s.b.c.ge("lx<160", k(159) - value(&lx, 8), 0);
    let emitted = s.when("emitted", &px.clone().and(&lt160));

    // The pixel that shows (`render_pixel`): the background's, unless an opaque object
    // pixel with OBJ_EN set wins over it.
    let (o, opal, obgp) = (s.f("pix0"), s.f("pal0"), s.bit("bgp0"));
    let (bgpal, bgprio) = (s.f("bgpal"), s.bit("bgprio"));
    let row7 = [bgl[7].clone(), bgh[7].clone()];
    let spr = s.b.c.ge("sprite", lcdc[1].clone() * 2 + o[0].clone() + o[1].clone(), 3);
    let bgen = [Cond::all(&[lcdc[0].clone(), nt(&ibp)]), Cond::all(&[nt(&lcdc[0]), cgbm.clone(), nt(&ibp)])];
    let pz = s.node("bg opaque", bgen.iter().map(|c| c.ge(row7[0].clone() + row7[1].clone(), 1, 2)).collect());
    let bce: W = (0..2).map(|b| s.node(&format!("bg colour.{b}"), bgen.iter().map(|c| c.clone().and(&row7[b]).step()).collect())).collect();
    let mut bp = Vec::new();
    for e in [Cond::all(&[nt(cgbm)]), Cond::all(&[cgbm.clone(), lcdc[0].clone()])] {
        bp.push(e.clone().and(&obgp).step());
        bp.push(e.and(&nt(&obgp)).and(&nt(&ibp)).and(&bgprio).step());
    }
    let bpt = s.node("bg priority", bp);
    let shows = s.node("object shows", vec![Cond::all(&[spr.clone(), nt(&pz)]).step(), Cond::all(&[spr, pz, nt(&bpt)]).step()]);
    let col: W = (0..2).map(|b| s.node(&format!("colour.{b}"), vec![Cond::all(&[shows.clone(), o[b].clone()]).step(), Cond::all(&[nt(&shows), bce[b].clone()]).step()])).collect();
    let pal: W = (0..3).map(|b| s.node(&format!("palette.{b}"), vec![Cond::all(&[shows.clone(), opal[b].clone()]).step(), Cond::all(&[nt(&shows), nt(&ibp), bgpal[b].clone()]).step()])).collect();
    // CGB mode reads palette RAM at obj·64 + palette·8 + colour·2.
    let query = |bits: &[Lin]| {
        let mut q = line_base();
        q.extend(bits.iter().cloned());
        for b in q.len()..crate::circuit::ADDR_BITS {
            q.push(k(((PAL >> b) & 1) as i64));
        }
        Query::Wires(q)
    };
    let cline = s.b.head("pal", 1, query(&[pal[0].clone(), pal[1].clone(), pal[2].clone(), shows.clone()]));
    // DMG rendering: the shade BGP, OBP0 or OBP1 gives the colour. Compatibility mode looks
    // it up in palette RAM: background palette 0, object palette 0 or 1.
    let regs = [s.f("r0"), s.f("r1"), s.f("r2")];
    let sel = [Cond::all(&[nt(&shows)]), Cond::all(&[shows.clone(), nt(&opal[0])]), Cond::all(&[shows.clone(), opal[0].clone()])];
    let shade: W = (0..2)
        .map(|jj| {
            let steps = (0..4).flat_map(|c| sel.iter().enumerate().map(move |(r, sc)| (c, r, sc.clone()))).map(|(c, r, sc)| both(&sc, &Cond::spells(&col, c)).and(&regs[r][2 * c + jj]).step()).collect();
            s.node(&format!("shade.{jj}"), steps)
        })
        .collect();
    let obp1 = s.b.c.and("compat obp1", &[shows.clone(), opal[0].clone()]);
    let dline = s.b.head("compat pal", 1, query(&[obp1, zero(), zero(), shows.clone()]));
    let on = Cond::all(&[emitted.clone(), nt(blk)]);
    let colour: W = (0..16)
        .map(|j| {
            let (byte, bit) = (j / 8, j % 8);
            let mut steps = Vec::new();
            if j < 15 {
                for c in 0..4 {
                    steps.push(both(&on, &Cond::spells(&col, c)).and(cgbm).and(&Lin::from(cline[2 * c + byte][bit])).step());
                    steps.push(both(&on, &Cond::spells(&shade, c)).and(cgb).and(&nt(cgbm)).and(&Lin::from(dline[2 * c + byte][bit])).step());
                }
            }
            if j < 2 {
                steps.push(on.clone().and(&nt(cgb)).and(&shade[j]).step());
            }
            s.node(&format!("colour {j}"), steps)
        })
        .collect();
    (emitted, colour)
}

/// SameBoy's `GB_update_wx_glitch` under `cond`, CGB hardware only.
fn wx_glitch(s: &mut S, x: &X, cond: &Cond, near: &Lin, at: &Lin) {
    let c = cond.clone().and(&x.cgb);
    let wyt = s.bit("wyt");
    s.set_k(&c.clone().and(&nt(&x.lcdc[5])), "cwg", 0);
    s.set_k(&c.clone().and(&x.lcdc[5]).and(&nt(&wyt)), "cwg", 0);
    s.set(&c.clone().and(&x.lcdc[5]).and(&wyt).and(&x.wx0), "cwg", vec![near.clone()]);
    s.set(&c.and(&x.lcdc[5]).and(&wyt).and(&nt(&x.wx0)), "cwg", vec![at.clone()]);
}

/// SameBoy's `advance_fetcher_state_machine`, one dot, in the phases that run it, and the
/// push into the background FIFO.
fn fetcher(s: &mut S, x: &X) {
    s.kind = "fetcher (Advance, FetchWait, FetchA, FetchB)";
    let lo = [Phase::Advance, Phase::FetchA, Phase::FetchB, Phase::FetchWait].map(|q| s.at(q).step()).to_vec();
    let fa_lo = s.node("fetcher runs", lo);
    let mut hi: Vec<Step> = [Phase::Advance, Phase::FetchA, Phase::FetchB].map(|q| s.at(q).step()).to_vec();
    hi.push(both(&s.at(Phase::FetchWait), &x.size0).step());
    let fa_hi = s.node("fetcher runs late", hi);
    let (lcdc, cgb, cgbm, pos) = (&x.lcdc, &x.cgb, &x.cgbm, &x.pos);
    let (wxt, wibf, lhfs, dof, wyt) = (s.bit("wxt"), s.bit("wibf"), s.bit("lhfs"), s.bit("dof"), s.bit("wyt"));
    let near = s.node("wx glitch at wx 0", vec![x.neg8.step(), Cond::spells(pos, 248).step(), Cond::spells(pos, 249).and(&lhfs).step()]);
    let atwx = s.node("wx glitch", mod256(value(&x.wx, 8) - value(pos, 8) - 7 - wibf.clone()));
    let step_is = |s: &S, en: &Lin, st: Fetch| both(&Cond::all(&[en.clone()]), &s.st(st));
    let (wtx, wyc, fy, tile, attr) = (s.f("wtx"), s.f("wyc"), s.f("fy"), s.f("tile"), s.f("attr"));

    // TileT1: the map address.
    let c0 = step_is(s, &fa_lo, Fetch::TileT1);
    wx_glitch(s, x, &c0, &near, &atwx);
    s.set_k(&c0.clone().and(&nt(&lcdc[5])), "wxt", 0);
    let tia = |xb: &[Lin], y: &[Lin], map: &Lin| -> W { [xb.to_vec(), y[3..8].to_vec(), vec![map.clone(), one(), one()]].concat() };
    let w1 = c0.clone().and(&wxt).and(&lcdc[5]);
    s.set(&w1, "tia", tia(&wtx[..5], &wyc, &lcdc[6]));
    s.set(&w1.clone().and(cgb), "fy", wyc.clone());
    let scx = &x.scx;
    let sx8 = s.b.c.sum_bits("map x", 8, |j| value(scx, j) + value(pos, j) + (8 & ((1 << j) - 1)));
    let sx7 = s.b.c.sum_bits("map x cgb", 8, |j| value(scx, j) + value(pos, j) + (7 & ((1 << j) - 1)));
    for w0 in [c0.clone().and(&nt(&wxt)), c0.clone().and(&wxt).and(&nt(&lcdc[5]))] {
        s.set(&both(&w0, &x.neg8), "tia", tia(&scx[3..8], &x.bgy, &lcdc[3]));
        let fine = w0.clone().and(&nt(&x.neg8n));
        s.set(&fine.clone().and(cgb).and(&nt(&dof)), "tia", tia(&sx7[3..8], &x.bgy, &lcdc[3]));
        s.set(&fine.clone().and(&nt(cgb)), "tia", tia(&sx8[3..8], &x.bgy, &lcdc[3]));
        s.set(&fine.and(cgb).and(&dof), "tia", tia(&sx8[3..8], &x.bgy, &lcdc[3]));
        s.set(&w0.and(cgb), "fy", x.bgy.clone());
    }
    s.set_k(&c0, "step", step_code(Fetch::TileT2) as u64);

    // TileT2: the tile and its attributes.
    let c1 = step_is(s, &fa_lo, Fetch::TileT2).and(&nt(&s.bit("cwg")));
    let ta = s.f("tia");
    let rt = s.read("tile", zero(), &ta, &x.blk);
    let ra = s.read("attr", one(), &ta, &x.blk);
    s.set(&c1, "tile", rt);
    s.set(&c1.clone().and(cgbm), "attr", ra);
    s.set_k(&c1.clone().and(&nt(cgbm)), "attr", 0);
    s.set_k(&step_is(s, &fa_lo, Fetch::TileT2), "step", step_code(Fetch::LowT1) as u64);

    // LowT1 and HighT1: the data address, its row from the row the fetcher reads.
    let flip: Vec<W> = [("fy", &fy), ("wyc", &wyc), ("bgy", &x.bgy)]
        .iter()
        .map(|(n, src)| (0..3).map(|b| s.node(&format!("row {n}.{b}"), vec![Cond::all(&[src[b].clone(), nt(&attr[6])]).step(), Cond::all(&[nt(&src[b]), attr[6].clone()]).step()])).collect())
        .collect();
    let b12 = s.b.c.and("tile data b12", &[nt(&lcdc[4]), nt(&tile[7])]);
    for (st, plane, next) in [(Fetch::LowT1, 0, Fetch::LowT2), (Fetch::HighT1, 1, Fetch::HighT2)] {
        let cc = step_is(s, &fa_lo, st);
        wx_glitch(s, x, &cc, &near, &atwx);
        s.set(&cc.clone().and(&nt(cgb)).and(&wxt), "fy", wyc.clone());
        s.set(&cc.clone().and(&nt(cgb)).and(&nt(&wxt)), "fy", x.bgy.clone());
        s.set(&cc, "lt", vec![lcdc[4].clone()]);
        let tda = |row: &W| -> W { [vec![k(plane)], row.clone(), tile[..7].to_vec(), vec![tile[7].clone(), b12.clone(), attr[3].clone()]].concat() };
        s.set(&cc.clone().and(cgb), "tda", tda(&flip[0]));
        s.set(&cc.clone().and(&nt(cgb)).and(&wxt), "tda", tda(&flip[1]));
        s.set(&cc.clone().and(&nt(cgb)).and(&nt(&wxt)), "tda", tda(&flip[2]));
        s.set_k(&cc, "step", step_code(next) as u64);
    }

    // LowT2 and HighT2: the data bytes, or the CGB's glitched ones.
    let (cwg, tsg, lt) = (s.bit("cwg"), s.bit("tsg"), s.bit("lt"));
    let (d0, d1, sel) = (s.f("d0"), s.f("d1"), s.f("sel"));
    let da = s.f("tda");
    let rd = s.read("data", da[13].clone(), &da, &x.blk);
    // (condition, the byte read, whether the glitch byte takes the read too): with the
    // tile-select glitch from the read, and on HighT2 always.
    let sources = |g: &Cond, high: bool| {
        vec![
            (g.clone().and(&tsg).and(&lt).and(&nt(&tile[7])), tile.clone(), true),
            (g.clone().and(&tsg).and(&lt).and(&tile[7]), rd.clone(), true),
            (g.clone().and(&tsg).and(&nt(&lt)), sel.clone(), false),
            (g.clone().and(&nt(&tsg)), rd.clone(), high),
        ]
    };
    let c3 = step_is(s, &fa_lo, Fetch::LowT2);
    s.set(&c3.clone().and(&cwg), "d0", d1.clone());
    for (g, byte, takes) in sources(&c3.clone().and(&nt(&cwg)), false) {
        s.set(&g, "d0", byte);
        if takes {
            s.set(&g, "sel", rd.clone());
        }
    }
    s.set_k(&c3, "step", step_code(Fetch::HighT1) as u64);
    let c5 = step_is(s, &fa_hi, Fetch::HighT2);
    let wtx1 = s.plus("window tile+1", &wtx[..5], 5, 1);
    s.set(&c5.clone().and(&wxt), "wtx", wtx1);
    s.set(&c5.clone().and(&cwg), "d1", d0.clone());
    s.set_k(&c5.clone().and(&cwg), "step", step_code(Fetch::Push) as u64);

    // The push (`push`): with the FIFO empty a row goes in, or a DMG armed with the window
    // off inserts one pixel at WX (SameBoy issue 278).
    let dis = s.bit("dis");
    let gl = Cond::all(&[wyt.clone(), nt(&lcdc[5]), nt(cgb), nt(&dis)]);
    let early = s.node("pos -16..-8", vec![x.neg8.step(), Cond::spells(pos, 248).step()]);
    let pg = s.node("pixel at wx", vec![gl.clone().and(&x.e7).and(&nt(&early)).step(), gl.and(&x.wx0).and(&early).step()]);
    let push = |s: &mut S, cond: &Cond, high: &W, late: bool| {
        if late {
            s.set_k(&cond.clone().and(&nt(&x.size0n)), "step", step_code(Fetch::Push) as u64);
        }
        let e = both(cond, &x.size0);
        let one_px = e.clone().and(&pg);
        for f in ["bgl", "bgh", "bgpal", "bgprio"] {
            s.set_k(&one_px, f, 0);
        }
        s.set_k(&one_px, "bgsize", 1);
        if late {
            s.set_k(&one_px, "step", step_code(Fetch::Push) as u64);
        }
        let row = e.and(&nt(&pg));
        for flip in 0..2 {
            let fc = row.clone().and(&lit(&attr[5], flip));
            let (l, h) = if flip == 1 { (rev(&d0), rev(high)) } else { (d0.clone(), high.clone()) };
            s.set(&fc, "bgl", l);
            s.set(&fc, "bgh", h);
        }
        s.set(&row, "bgpal", attr[..3].to_vec());
        s.set(&row, "bgprio", vec![attr[7].clone()]);
        s.set_k(&row, "bgsize", 8);
        s.set_k(&row, "step", step_code(Fetch::TileT1) as u64);
    };
    for (g, byte, takes) in sources(&c5.clone().and(&nt(&cwg)), true) {
        s.set(&g, "d1", byte.clone());
        if takes {
            s.set(&g, "sel", rd.clone());
        }
        push(s, &g, &byte, true);
    }
    let c6 = step_is(s, &fa_hi, Fetch::Push);
    push(s, &c6, &d1, false);

    // The phases that ran the fetcher go on.
    let adv = s.at(Phase::Advance);
    let end = both(&adv, &Cond::spells(pos, 160));
    s.go(&end, Phase::Done);
    s.done.push(end.step());
    let endn = s.when("pos=160", &end);
    s.sleep(&adv.and(&nt(&endn)), 1, Phase::Window);
    let fw = s.at(Phase::FetchWait);
    let stp = s.f("step");
    let early_step = s.node("step<HighT2", vec![Cond::all(&[nt(&stp[2])]).step(), Cond::all(&[stp[2].clone(), nt(&stp[1]), nt(&stp[0])]).step()]);
    let wait = s.node("fetch waits", vec![fw.clone().and(&early_step).step(), fw.clone().and(&nt(&early_step)).and(&x.size0n).step()]);
    s.sleep(&Cond::all(&[wait.clone()]), 1, Phase::FetchWait);
    s.go(&fw.and(&nt(&wait)), Phase::FetchA);
    let fa = s.at(Phase::FetchA);
    s.sleep(&fa, 1, Phase::FetchB);
    let fb = s.at(Phase::FetchB);
    s.sleep(&fb, 2, Phase::FetchC);
}

/// The object fetch's reads (FetchC, FetchD) and its overlay into the object FIFO (FetchE).
fn object_fetch(s: &mut S, x: &X) {
    s.kind = "object fetch (FetchC, FetchD, FetchE)";
    let oi = s.f("oi0");
    let e = oi[0].clone();
    let mut off = line_base();
    off.extend(oi[1..6].iter().cloned());
    let line = s.b.head("oam", 1, oam_query(&off));
    let byte = |en: usize, b: usize| w(line[4 * en + b]);
    let ly = x.ly.clone();
    let rows: Vec<W> = (0..2)
        .map(|en| {
            let y = byte(en, 0);
            s.b.c.sum_bits(&format!("object row{en}"), 4, |j| value(&ly, j) + value(&not(&y), j) + (j > 0) as i64)
        })
        .collect();
    let tall = x.lcdc[2].clone();
    let xor = |s: &mut S, name: &str, r: usize, gate: &[Lin]| -> Lin {
        let steps = (0..2)
            .flat_map(|en| {
                let f = byte(en, 3)[6].clone();
                let base = [vec![lit(&e, en)], gate.to_vec()].concat();
                [Cond::all(&[base.clone(), vec![rows[en][r].clone(), nt(&f)]].concat()).step(), Cond::all(&[base, vec![nt(&rows[en][r]), f]].concat()).step()]
            })
            .collect();
        s.node(name, steps)
    };
    let mut a: W = vec![zero()];
    for r in 0..3 {
        a.push(xor(s, &format!("object addr.{}", r + 1), r, &[]));
    }
    let tall_row = xor(s, "object addr.4 tall", 3, std::slice::from_ref(&tall));
    let a4 = s.node("object addr.4", vec![Cond::all(&[tall_row.clone()]).step(), Cond::all(&[lit(&e, 0), nt(&tall), byte(0, 2)[0].clone()]).step(), Cond::all(&[lit(&e, 1), nt(&tall), byte(1, 2)[0].clone()]).step()]);
    a.push(a4);
    for b in 5..12 {
        a.push(s.node(&format!("object addr.{b}"), (0..2).map(|en| Cond::all(&[lit(&e, en), byte(en, 2)[b - 4].clone()]).step()).collect()));
    }
    a.push(zero());
    let bank = s.node("object bank", (0..2).map(|en| Cond::all(&[lit(&e, en), x.cgbm.clone(), byte(en, 3)[3].clone()]).step()).collect());
    let data = s.b.head("object row", 1, vram_query(bank, &a));
    let ol = s.pick("object low", &data, &[zero(), a[1].clone(), a[2].clone()], &x.blk);
    let oh = s.pick("object high", &data, &[one(), a[1].clone(), a[2].clone()], &x.blk);
    let fc = s.at(Phase::FetchC);
    s.set(&fc, "ol", ol);
    s.sleep(&fc, 2, Phase::FetchD);
    let fd = s.at(Phase::FetchD);
    s.set_k(&fd, "dof", 0);
    s.set(&fd, "oh", oh);
    s.sleep(&fd, 1, Phase::FetchE);

    let fe = s.at(Phase::FetchE);
    let fl: W = (0..8).map(|i| s.node(&format!("object flags.{i}"), (0..2).map(|en| Cond::all(&[lit(&e, en), byte(en, 3)[i].clone()]).step()).collect())).collect();
    let (rl, rh) = (s.f("ol"), s.f("oh"));
    let cgbm = x.cgbm.clone();
    for i in 0..8 {
        let pix = s.f(&format!("pix{i}"));
        let gt = s.b.c.ge(&format!("prio{i}>index"), value(&s.f(&format!("prio{i}")), 8) - value(&oi, 8), 1);
        for flip in 0..2 {
            let b = if flip == 1 { i } else { 7 - i };
            let (lo, hi) = (rl[b].clone(), rh[b].clone());
            let opaque = lo.clone() + hi.clone();
            let rep = s.node(
                &format!("overlay{i}.{flip}"),
                vec![Cond::all(&[nt(&pix[0]), nt(&pix[1])]).ge(opaque.clone(), 1, 2), Cond::all(&[pix[0].clone(), gt.clone()]).ge(opaque.clone(), 1, 2), Cond::all(&[nt(&pix[0]), pix[1].clone(), gt.clone()]).ge(opaque, 1, 2)],
            );
            let c = fe.clone().and(&lit(&fl[5], flip)).and(&rep);
            s.set(&c, &format!("pix{i}"), vec![lo, hi]);
            s.set(&c, &format!("bgp{i}"), vec![fl[7].clone()]);
            s.set(&c.clone().and(&cgbm), &format!("pal{i}"), fl[..3].to_vec());
            s.set(&c.clone().and(&nt(&cgbm)), &format!("pal{i}"), vec![fl[4].clone()]);
            s.set(&c.clone().and(&cgbm), &format!("prio{i}"), oi.clone());
            s.set_k(&c.and(&nt(&cgbm)), &format!("prio{i}"), 0);
        }
    }
    shift_objects(s, &fe, 1);
    s.set(&fe, "sel", rh);
    s.go(&fe, Phase::Match { first: false });
}

/// `State::new`: the registers from the image, the OAM scan in fetch order (by X, then
/// OAM index), the window from the inputs, every other field its start.
fn start_pass(s: &mut S, x: &X, srows: &[Lin], flg: &[Lin], image_regs: &[W; 8]) {
    s.kind = "start";
    let st = Cond::all(&[flg[4].clone()]);
    // The window before the line, as the scanline circuit takes it: the frame starts over at
    // line 0, and the window arms where LY equals WY (`fifo::Window::starting`).
    let first = s.b.c.and("start LY=0", &(0..8).map(|i| nt(&x.ly[i])).collect::<Vec<_>>());
    let lyeqwy = s.b.c.node("start LY=WY", eq2(value(&x.ly, 8) - value(&image_regs[6], 8), 0));
    let armed = s.b.c.ge("start armed", flg[5].clone() - first.clone() + lyeqwy * 2, 1);
    let carried = s.b.c.ge("start carried", flg[6].clone() - first.clone(), 1);
    let srows: W = (0..8).map(|i| s.b.c.ge(&format!("start rows.{i}"), srows[i].clone() - first.clone(), 1)).collect();
    let srows = &srows[..];
    let scan = s.b.scan(&x.ly, Some(one()), false);
    let rank = s.b.ranks(&scan, &zero());
    let Scan { valid, index, xsel, .. } = scan;
    let mut special: BTreeMap<String, W> = BTreeMap::new();
    for r in 0..REGS {
        special.insert(format!("r{r}"), image_regs[r].clone());
    }
    let konst = |v: u64, n: usize| (0..n).map(|i| k((v >> i & 1) as i64)).collect::<W>();
    special.insert("resume".into(), konst(code(Phase::Init) as u64, 4));
    special.insert("dots".into(), konst(5, 3));
    special.insert("pos".into(), konst(240, 8));
    for r in 0..OBJECTS {
        let pick = |s: &mut S, name: String, field: &[W], b: usize| {
            let steps = (0..OBJECTS).flat_map(|sl| Cond::all(&[valid[sl].clone(), field[sl][b].clone()]).eq(rank[sl].clone(), r as i64, OBJECTS as i64 - 1)).collect();
            s.node(&name, steps)
        };
        let oi: W = (0..6).map(|b| pick(s, format!("sorted{r}.{b}"), &index, b)).collect();
        let ox: W = (0..8).map(|b| pick(s, format!("sorted x{r}.{b}"), &xsel, b)).collect();
        special.insert(format!("oi{r}"), oi);
        special.insert(format!("ox{r}"), ox);
    }
    special.insert("ov".into(), valid.clone());
    let wyc = s.plus("rows-1", srows, 8, -1);
    special.insert("wyc".into(), wyc);
    special.insert("wtx".into(), vec![carried.clone()]);
    special.insert("wyt".into(), vec![armed]);
    special.insert("wxt".into(), vec![carried]);
    special.insert("fine".into(), image_regs[5][..3].to_vec());
    let names: Vec<String> = s.cur.keys().cloned().collect();
    for n in names {
        match special.remove(&n) {
            Some(v) => s.set(&st, &n, v),
            None => s.set_k(&st, &n, 0),
        }
    }
    assert!(special.is_empty(), "start values for fields that do not exist: {:?}", special.keys());
}

/// The stepper's memory: VRAM, OAM and the display bytes, as `PpuCheckpoint` holds them.
pub fn image(vram: &[u8; 0x4000], oam: &[u8; 0xA0], display: &crate::machine::ppu::Display) -> Image {
    let banking = Banking::new(Mapper::Rom, 0x8000, 0, true).expect("a 32 KiB ROM's banking");
    let mut image = Image::new(&[0], 0, banking).with_palettes();
    image.fill(VRAM, vram);
    image.fill(OAM, oam);
    image.fill(PAL, display);
    image
}

/// The circuit evaluated directly on a pass's input bytes over `memory`.
pub fn eval(c: &Circuit, inputs: &[u8], memory: &Image, constants: Constants) -> Decoded {
    let map: BTreeMap<String, u8> = c.inputs.iter().map(|(n, _)| n.clone()).zip(inputs.iter().copied()).collect();
    let out = c.eval_mem(&map, &|n| memory.line(n));
    decode(constants, &|n| out[n])
}


/// The compiled stepper on the engine.
pub fn engine() -> Engine {
    Engine::new(Model::compile_mem(&build(), Params::memory()))
}

/// The stepper checkpoint over one line's VRAM, OAM and display bytes.
pub struct Stepper {
    engine: Engine,
    tokens: Sequence,
    /// The output positions, in output order, and their slot names.
    positions: Positions,
    names: Vec<String>,
    pub passes: u64,
    /// Nanoseconds in the engine's passes alone, without the slots' encoding and decoding.
    pub engine_ns: u128,
}

impl Stepper {
    pub fn new(engine: Engine, vram: &[u8; 0x4000], oam: &[u8; 0xA0], display: &crate::machine::ppu::Display) -> Stepper {
        Stepper::with_image(engine, image(vram, oam, display))
    }

    fn with_image(mut engine: Engine, image: Image) -> Stepper {
        engine.insert(image);
        let cfg = engine.cfg();
        let tokens = (0..cfg.seq()).map(|p| if p < cfg.n_inputs { tok::input(cfg, p, 0) } else { tok::output_mask(cfg, p) }).collect();
        // Every output but the write-record slots, which a stepper pass leaves empty.
        let positions: Vec<usize> = (cfg.n_inputs..cfg.seq()).filter(|&p| !engine.slots()[p].starts_with("WP") && !engine.slots()[p].starts_with("WL")).collect();
        let names = positions.iter().map(|&p| engine.slots()[p].clone()).collect();
        Stepper { engine, tokens: Sequence::new(tokens), positions: Positions::new(positions), names, passes: 0, engine_ns: 0 }
    }

    /// A new line's memory, the engine kept.
    pub fn load(&mut self, vram: &[u8; 0x4000], oam: &[u8; 0xA0], display: &crate::machine::ppu::Display) {
        self.engine.insert(image(vram, oam, display));
    }

    /// One pass over the input bytes `inputs`; its output bytes by slot name.
    pub fn run(&mut self, inputs: &[u8]) -> BTreeMap<String, u8> {
        let cfg = self.engine.cfg();
        assert_eq!(inputs.len(), cfg.n_inputs, "the stepper's input slots");
        for (j, &b) in inputs.iter().enumerate() {
            self.tokens.set(j, tok::input(cfg, j, b));
        }
        let t = std::time::Instant::now();
        self.engine.fill_seq(&mut self.tokens, &self.positions);
        self.engine_ns += t.elapsed().as_nanos();
        self.passes += 1;
        let (cfg, out) = (self.engine.cfg(), self.tokens.decoded());
        self.names.iter().zip(self.positions.iter()).map(|(n, &p)| (n.clone(), tok::output_value(cfg, out[p]).unwrap_or_else(|| panic!("output {n} decoded token {}, not a byte", out[p])))).collect()
    }

    pub fn pass(&mut self, inputs: &[u8], c: Constants) -> Decoded {
        let out = self.run(inputs);
        decode(c, &|n| out[n])
    }

    /// A whole line drawn by the checkpoint alone (its dots to mode 0, `fifo::mode0_dots`): a
    /// start pass, then step passes fed the
    /// pending writes as `fifo::drive` feeds them.
    pub fn draw(&mut self, start: &fifo::Start, writes: &[fifo::Write]) -> fifo::Drawn {
        let c = Constants { ly: start.ly, mode: start.mode, double: start.double, blocked: start.blocked };
        let first = self.pass(&start_inputs(c, start.window), c).state;
        let mut colours = [0; crate::machine::ppu::WIDTH];
        let mut last = None;
        fifo::drive(first, writes, |st, pending| {
            let d = self.pass(&step_inputs(st, pending), c);
            *st = d.state;
            if let Some((x, colour)) = d.out.pixel {
                colours[x as usize] = colour;
            }
            last = Some(d);
            d.out
        });
        let last = last.expect("a line takes passes");
        fifo::Drawn { colours, window: last.window, dots: last.dots }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The tokens of the last pass and the positions it decoded.
    pub fn tokens(&self) -> (&[usize], &[usize]) {
        (self.tokens.tokens(), &self.positions)
    }
}


/// A line the stepper drew: each pixel's colour, the window after it, mode 3's length in
/// dots, and the state after each pass (the start pass first).
pub struct Stepped {
    pub colours: [crate::machine::ppu::Colour; crate::machine::ppu::WIDTH],
    pub window: Window,
    pub dots: u16,
    pub states: Vec<State>,
}

impl Stepper {
    /// The stepper over the hand-off's VRAM, OAM and display bytes, as the scanline
    /// checkpoint takes them over (`PpuCheckpoint::new`); after this it follows the same
    /// records ([`Stepper::write`]).
    pub fn over(engine: Engine, memory: &Image, display: &crate::machine::ppu::Display) -> Stepper {
        let mut image = Image::new(&[memory.read(0)], 0, memory.banking()).with_palettes();
        image.fill(VRAM, memory.region(VRAM, crate::memory::layout::VRAM_LEN));
        image.fill(OAM, memory.region(OAM, crate::memory::layout::OAM_LEN));
        image.fill(PAL, display);
        Stepper::with_image(engine, image)
    }

    /// One record to VRAM, OAM or the display bytes.
    pub fn write(&mut self, addr: u32, byte: u8) {
        self.engine.keep_write(addr, byte);
    }

    /// Draws a line from `window`, the window's state before it, with the queue of writes the
    /// length stream took, each (time in half-dots, display index, value). Every state slot
    /// is carried from a pass's outputs to the next pass's inputs byte for byte, and the queue
    /// moves only as the passes say: the head write is offered until a pass says its last
    /// change applied. `keep` asks for each pass's decoded state.
    pub fn line(&mut self, c: Constants, window: Window, writes: &[(fifo::Time, u8, u8)], keep: bool) -> Stepped {
        let names: Vec<String> = slots().iter().map(|(s, _)| format!("{s}'")).collect();
        let mut out = self.run(&start_inputs(c, window));
        let mut states = Vec::new();
        let mut colours = [0; crate::machine::ppu::WIDTH];
        let (mut head, mut applied) = (0, 0u8);
        loop {
            let d = decode(c, &|n| out[n]);
            if keep {
                states.push(d.state);
            }
            if let Some((x, colour)) = d.out.pixel {
                colours[x as usize] = colour;
            }
            if d.out.applied {
                applied += 1;
                if d.out.finished {
                    (head, applied) = (head + 1, 0);
                }
            }
            if d.out.done {
                return Stepped { colours, window: d.window, dots: d.dots, states };
            }
            let pending = match writes.get(head) {
                Some(&(at, index, value)) => [at as u8, (at >> 8) as u8, index, value, applied | PN_PENDING],
                None => [0; 5],
            };
            let mut inputs = vec![c.ly, flags(c), 0];
            inputs.extend(pending);
            inputs.extend(names.iter().map(|n| out[n]));
            out = self.run(&inputs);
        }
    }
}
