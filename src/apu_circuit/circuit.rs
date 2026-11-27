//! One APU segment as a threshold circuit (`docs/apu-in-weights.md`), the rules of
//! `machine::apu` (SameBoy's CPU CGB E APU) in three phases over the carried state
//! (`fields::FIELDS`):
//!
//! 1. the register write that opens the segment (`write.rs`), at most one;
//! 2. the frame-sequencer events at its first M-cycle (`events.rs`): a fall of the
//!    frame sequencer's DIV bit, a rise, or the envelope steps a fall at double speed left
//!    waiting, which the host cuts into segments of one M-cycle;
//! 3. `n` APU cycles of 2 dots (`run.rs`): `D` at double speed, `2·D` at normal speed,
//!    at most [`MAX_CYCLES`].
//!
//! The events are `WA WV EV D RA`. `EV` bits: 0 a write, 1 a fall, 2 a sample ends the
//! segment, 3 double speed, 4 the fall came from a reset of the counter, 5 a rise. `RA` is
//! the low byte of the CPU read that ends the segment (0 for none), and `RV` the byte it
//! returns from the state after the segment.

use super::build::*;
use super::fields::placement;
use crate::circuit::{Circuit, Lin, Step};
use std::collections::BTreeMap;

pub const EVENTS: [&str; 5] = ["WA", "WV", "EV", "D", "RA"];
pub const SAMPLES: [&str; 4] = ["SL0", "SL1", "SR0", "SR1"];
pub const READ: &str = "RV";
/// The halves (2-dot units) from the end of the segment to the next sample.
pub const TO_SAMPLE: &str = "SD";
/// The M-cycles of the next segment after which it ends, 0 for no end before its sample.
pub const TO_END: &str = "SE";
/// The M-cycles of the segment its pass left: D when it opens with a frame-sequencer event.
pub const LEFT: &str = "DL";
/// The most M-cycles one segment covers at normal speed (twice as many at double).
pub const MAX_DELTA: i64 = 22;
/// The most APU cycles one segment runs.
pub const MAX_CYCLES: i64 = 44;

pub struct Built {
    pub c: Circuit,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// The carried fields by name, as bits.
pub(super) type St = BTreeMap<&'static str, W>;

/// The segment's event wires.
pub(super) struct Ev {
    pub wp: Lin,
    pub wa: W,
    pub wv: W,
    pub fall: Lin,
    pub reset: Lin,
    pub rise: Lin,
    pub dbl: Lin,
    /// D ≥ 1: an M-cycle starts, so the events and the waiting envelope steps run.
    pub dpos: Lin,
    /// The APU cycles as bits (at most 44).
    pub n: W,
}

pub(super) fn lit(x: &Lin, set: bool) -> Lin {
    if set { x.clone() } else { nt(x) }
}

pub(super) fn bits_w(bits: &[Lin]) -> Vec<(Lin, i64)> {
    bits.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).collect()
}

pub(super) fn neg_w(bits: &[Lin]) -> Vec<(Lin, i64)> {
    bits.iter().enumerate().map(|(b, x)| (x.clone(), -(1i64 << b))).collect()
}

pub(super) fn with(base: &[Lin], extra: &[Lin]) -> Vec<Lin> {
    base.iter().chain(extra).cloned().collect()
}

/// Literals for `x == v` over the bits of `x`.
pub(super) fn is(x: &[Lin], v: u64) -> Vec<Lin> {
    x.iter().enumerate().map(|(b, l)| lit(l, v >> b & 1 == 1)).collect()
}

pub(super) fn eqk(c: &mut Circuit, name: &str, x: &[Lin], v: u64) -> Lin {
    all(c, name, &is(x, v))
}

/// `[x ≠ 0]`.
pub(super) fn nz(c: &mut Circuit, name: &str, x: &[Lin]) -> Lin {
    c.ge(name, sum(x.iter().cloned()), 1)
}

/// `(x + k) mod 2^width`.
pub(super) fn addk(c: &mut Circuit, name: &str, x: &[Lin], k: i64, width: usize) -> W {
    sum_mod(c, name, &bits_w(x), k, width, None)
}

/// `x[idx]`, one step per index.
pub(super) fn pick(c: &mut Circuit, name: &str, x: &[Lin], idx: &[Lin]) -> Lin {
    let steps = (0..1u64 << idx.len())
        .filter(|&i| (i as usize) < x.len())
        .map(|i| {
            let lits = with(&is(idx, i), &[x[i as usize].clone()]);
            Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }
        })
        .collect();
    c.node(name, steps)
}

/// `x >> sh` over `width` bits.
pub(super) fn shr(c: &mut Circuit, name: &str, x: &[Lin], sh: &[Lin], width: usize) -> W {
    (0..width)
        .map(|i| {
            let steps = (0..1usize << sh.len())
                .filter(|s| i + s < x.len())
                .map(|s| {
                    let lits = with(&is(sh, s as u64), &[x[i + s].clone()]);
                    Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }
                })
                .collect();
            c.node(&format!("{name}.{i}"), steps)
        })
        .collect()
}

/// `DUTIES[idx + 8·duty] ? vol : 0`, each bit one step per high (duty, index).
pub(super) fn duty_level(c: &mut Circuit, name: &str, duty: &[Lin], idx: &[Lin], vol: &[Lin]) -> W {
    // The duty bit once, then a bit of the volume each, so a step re-evaluates one node
    // rather than every level bit's table.
    let mut steps = Vec::new();
    for d in 0..4u64 {
        for i in 0..8u64 {
            if crate::machine::apu::DUTIES[(i + 8 * d) as usize] != 0 {
                let lits = with(&is(duty, d), &is(idx, i));
                steps.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
            }
        }
    }
    let high = c.node(&format!("{name} duty"), steps);
    (0..4).map(|b| all(c, &format!("{name}.{b}"), &[high.clone(), vol[b].clone()])).collect()
}

/// `bits ∧ gate` per bit: a copy of a fast-changing field that a write-only value reads,
/// so it holds still while no write reads it.
pub(super) fn gated(c: &mut Circuit, name: &str, bits: &[Lin], gate: &Lin) -> W {
    bits.iter().enumerate().map(|(b, x)| all(c, &format!("{name}.{b}"), &[x.clone(), gate.clone()])).collect()
}

/// Per field, the cases a phase loads it with: (conjunction of 0/1 literals, value),
/// mutually exclusive per field. `finish` muxes each field that has any.
pub(super) struct Cases(BTreeMap<&'static str, Vec<(Vec<Lin>, W)>>);

impl Cases {
    pub fn new() -> Self {
        Cases(BTreeMap::new())
    }

    pub fn put(&mut self, field: &'static str, gates: Vec<Lin>, value: W) {
        self.0.entry(field).or_default().push((gates, value));
    }

    pub fn finish(self, c: &mut Circuit, tag: &str, s: &St) -> St {
        let mut out = s.clone();
        for (field, cases) in self.0 {
            let keep = &s[field];
            for (_, v) in &cases {
                assert_eq!(v.len(), keep.len(), "{field}: a case of {} bits for {} bits", v.len(), keep.len());
            }
            out.insert(field, sel(c, &format!("{field}{tag}"), keep, &cases));
        }
        out
    }
}

pub fn build() -> Built {
    let mut c = Circuit::default();
    let (place, nbytes) = placement();
    let inputs: Vec<String> = (0..nbytes).map(|i| format!("S{i}")).chain(EVENTS.iter().map(|s| s.to_string())).collect();
    let in_bytes: Vec<W> = inputs.iter().map(|n| w(c.input(n))).collect();
    let (wa, wv, evb, d, ra) = (in_bytes[nbytes].clone(), in_bytes[nbytes + 1].clone(), in_bytes[nbytes + 2].clone(), in_bytes[nbytes + 3].clone(), in_bytes[nbytes + 4].clone());
    let s0: St = place.iter().map(|&(n, at, width)| (n, (0..width).map(|b| in_bytes[(at + b) / 8][(at + b) % 8].clone()).collect())).collect();

    let dbl = evb[3].clone();
    let dm: W = d[..6].to_vec();
    // A segment that opens with a frame-sequencer event runs its write and its event only
    // and leaves its D M-cycles (`DL`) for the segments after it, as `Apu::mcycle` runs
    // the events at an M-cycle's start: every reader of D, and the sample that would end
    // the segment, sees 0 here. The fall and the rise never share a segment.
    let ev = evb[1].clone() + evb[5].clone();
    // The APU cycles: D at double speed, 2·D at normal speed.
    let n: W = (0..6)
        .map(|b| {
            let mut steps = vec![Step { coef: 1, lin: dbl.clone() + dm[b].clone() - ev.clone(), theta: 2 }];
            if b > 0 {
                steps.push(Step { coef: 1, lin: nt(&dbl) + dm[b - 1].clone() - ev.clone(), theta: 2 });
            }
            c.node(&format!("n.{b}"), steps)
        })
        .collect();
    let dpos = c.ge("D>0", sum(dm.iter().cloned()) - ev.clone() * 6, 1);
    let dg: W = dm.iter().enumerate().map(|(b, x)| c.ge(&format!("D'.{b}"), x.clone() - ev.clone(), 1)).collect();
    let sg = c.ge("EV.2'", evb[2].clone() - ev.clone(), 1);
    let left: W = dm.iter().enumerate().map(|(b, x)| c.ge(&format!("DL.{b}"), x.clone() + ev.clone(), 2)).collect();
    let e = Ev { wp: evb[0].clone(), wa, wv, fall: evb[1].clone(), reset: evb[4].clone(), rise: evb[5].clone(), dbl: dbl.clone(), dpos, n };

    let s1 = super::write::phase(&mut c, &s0, &e);
    // A segment with a frame-sequencer event runs no cycles (D is 0 to them), so the
    // events and the cycles read the post-write state side by side; the waiting envelope
    // steps come first.
    let s2 = super::events::phase(&mut c, &s1, &e);
    let s1w = super::events::waiting(&mut c, &s1, &e);
    let run = super::run::phase(&mut c, &s1w, &e);
    let event = c.ge("event", e.fall.clone() + e.rise.clone(), 1);
    let mut s3 = St::new();
    for (f, after) in run {
        let by_event = &s2[f];
        let v = match (by_event != &s1[f], after != s1[f]) {
            (true, true) => sel(&mut c, &format!("{f}_m"), &after, &[(vec![event.clone()], by_event.clone())]),
            (true, false) => by_event.clone(),
            _ => after,
        };
        s3.insert(f, v);
    }
    // A channel's latched sample is visible only while it plays, and every start
    // reloads it, so the carried form is 0 while the channel is off.
    for (act, smp) in [("act1", "smp1"), ("act2", "smp2"), ("act3", "smp3"), ("act4", "smp4")] {
        let a = s3[act][0].clone();
        let v: W = s3[smp].iter().enumerate().map(|(b, x)| all(&mut c, &format!("{smp}'.{b}"), &[a.clone(), x.clone()])).collect();
        s3.insert(smp, v);
    }

    let samples = mix(&mut c, &s3, &evb[2]);
    let (sq, sr, sd) = sample_clock(&mut c, &s0["sq"], &s0["sr"], &dg, &sg, &dbl);
    s3.insert("sq", sq);
    s3.insert("sr", sr);

    let mut outputs = Vec::new();
    for i in 0..nbytes {
        let bits: Vec<Lin> = (0..8)
            .map(|b| {
                let at = 8 * i + b;
                place.iter().find(|(_, s, wd)| (*s..s + wd).contains(&at)).map_or(k(0), |(n, s, _)| s3[n][at - s].clone())
            })
            .collect();
        let name = format!("S{i}'");
        c.output(&name, byte_of(&bits));
        outputs.push(name);
    }
    for (side, bits) in samples.iter().enumerate() {
        for h in 0..2 {
            let name = SAMPLES[2 * side + h];
            c.output(name, byte_of(&bits[8 * h..8 * h + 8]));
            outputs.push(name.to_string());
        }
    }
    let rv = readback(&mut c, &ra, &s3);
    c.output(READ, byte_of(&rv));
    outputs.push(READ.to_string());
    c.output(TO_SAMPLE, byte_of(&sd));
    outputs.push(TO_SAMPLE.to_string());
    // `pend` is 2 only after a counted fall at double speed, whose steps the next
    // M-cycle's start runs: the next segment ends after that one M-cycle.
    c.output(TO_END, byte_of(&s3["pend"][1..]));
    outputs.push(TO_END.to_string());
    c.output(LEFT, byte_of(&left));
    outputs.push(LEFT.to_string());
    Built { c, inputs, outputs }
}

/// The DAC of channel `ch` (0-3) after the segment.
pub(super) fn dac(c: &mut Circuit, s: &St, ch: usize) -> Lin {
    match ch {
        0 => c.ge("dac1", sum(s["nr12"][3..].iter().cloned()), 1),
        1 => c.ge("dac2", sum(s["nr22"][3..].iter().cloned()), 1),
        2 => s["en3"][0].clone(),
        _ => c.ge("dac4", sum(s["nr42"][3..].iter().cloned()), 1),
    }
}

/// The mix per side, the chip's digital output: per routed channel with its DAC on,
/// `2·out − 15`, times NR50's volume + 1. Every wire is 0 in a segment without a sample.
/// The board's high-pass filter after it is the host's presentation.
fn mix(c: &mut Circuit, s: &St, sample: &Lin) -> Vec<W> {
    let (nr50, nr51) = (s["nr50"].clone(), s["nr51"].clone());
    let dacs: Vec<Lin> = (0..4).map(|ch| dac(c, s, ch)).collect();
    let mut out = Vec::new();
    for side in 0..2 {
        let shift = [4, 0][side];
        let tag = ["L", "R"][side];
        let mut a_terms = Vec::new();
        let mut b_terms = Vec::new();
        for (vl, mult) in [(None, 1i64), (Some(nr50[shift].clone()), 1), (Some(nr50[shift + 1].clone()), 2), (Some(nr50[shift + 2].clone()), 4)] {
            for ch in 0..4 {
                let base: Vec<Lin> = [nr51[shift + ch].clone(), sample.clone()].into_iter().chain(vl.clone()).collect();
                let smp = s[["smp1", "smp2", "smp3", "smp4"][ch]].clone();
                for (b, x) in smp.iter().enumerate() {
                    let wire = all(c, &format!("mix{tag}{ch}.{b}.{mult}"), &with(&base, &[dacs[ch].clone(), x.clone()]));
                    a_terms.push((wire, mult << b));
                }
                let dw = all(c, &format!("mixdac{tag}{ch}.{mult}"), &with(&base, &[dacs[ch].clone()]));
                b_terms.push((dw, mult));
            }
        }
        // The mix is 2A − 15B: A the level wires' sum (at most 480), B the DAC wires' (at
        // most 32).
        let a = sum_mod(c, &format!("A{tag}"), &a_terms, 0, 9, None);
        let b = sum_mod(c, &format!("B{tag}"), &b_terms, 0, 6, None);
        let mut q: Vec<(Lin, i64)> = a.iter().enumerate().map(|(i, x)| (x.clone(), 2 << i)).collect();
        q.extend(b.iter().enumerate().flat_map(|(i, x)| [(x.clone(), 1 << i), (x.clone(), -(16 << i))]));
        // -480 to 480 fits 10 bits of two's complement; the i16's top six repeat bit 9.
        let mut side_mix = sum_blocked(c, &format!("S{tag}"), &q, 0, 10, None);
        side_mix.extend(std::iter::repeat_n(side_mix[9].clone(), 6));
        out.push(side_mix);
    }
    out
}

/// The 48 kHz sample clock, which counts real time: 48,000 a second against 2^22
/// T-cycles is 750 units of 2^-14 of a sample's period per M-cycle at normal speed and
/// 375 at double, a sample every 16,384 units. The count since the last sample is
/// `750·q + r`, r below 750. A segment of D M-cycles adds D·750 or D·375, and one that
/// ends in a sample (`EV.2`) takes 16,384 = 21·750 + 634 away, which leaves q at 0
/// (the sample falls in the M-cycle that crosses it). So r moves by 0, 116, 375 or 491
/// modulo 750, by whether the segment sampled and whether it ran an odd number of
/// M-cycles at double speed, and q by D, by D/2 plus r's carry, or to 0. A half (2 dots,
/// one M-cycle at double speed) adds 375 at either speed, so `SD` is the least h ≥ 1 with
/// `750·q + r + 375·h ≥ 16,384`, `42 − 2q + [r < 634] + [r < 259]`, the halves to the
/// next sample, which the host counts down by each M-cycle's halves (the control logic's
/// rate). Returns (q', r', SD's six bits).
fn sample_clock(c: &mut Circuit, q: &W, r: &W, d: &W, sample: &Lin, dbl: &Lin) -> (W, W, W) {
    let odd = all(c, "smp.odd", &[dbl.clone(), d[0].clone()]);
    // r + step modulo 750: r + step below 750, or r + step − 750.
    let moves = [(116, vec![sample.clone(), nt(&odd)]), (375, vec![nt(sample), odd.clone()]), (491, vec![sample.clone(), odd.clone()])];
    let mut cases = Vec::new();
    let mut wraps = Vec::new();
    for (step, gates) in moves {
        let wrap = c.ge(&format!("smp.r≥{}", 750 - step), val(r), 750 - step);
        cases.push((with(&gates, &[nt(&wrap)]), sum_mod(c, &format!("smp.r+{step}"), &bits_w(r), step, 10, None)));
        cases.push((with(&gates, &[wrap.clone()]), sum_mod(c, &format!("smp.r-{}", 750 - step), &bits_w(r), step - 750, 10, None)));
        wraps.push(wrap);
    }
    let r1 = sel(c, "sr'", r, &cases);
    // At double speed an odd segment's half step carries into q when r wraps.
    let carry = all(c, "smp.carry", &[odd, wraps[1].clone()]);
    let normal: Vec<(Lin, i64)> = bits_w(q).into_iter().chain(bits_w(d)).collect();
    let double: Vec<(Lin, i64)> = bits_w(q).into_iter().chain(bits_w(&d[1..])).chain([(carry, 1)]).collect();
    let qn = sum_mod(c, "smp.q+D", &normal, 0, 5, None);
    let qd = sum_mod(c, "smp.q+D/2", &double, 0, 5, None);
    let q1 = sel(c, "sq'", q, &[(vec![nt(sample), nt(dbl)], qn), (vec![nt(sample), dbl.clone()], qd), (vec![sample.clone()], kbits(0, 5))]);
    let below = |c: &mut Circuit, t: i64| c.ge(&format!("smp.r'<{t}"), k(t - 1) - val(&r1), 0);
    let (l634, l259) = (below(c, 634), below(c, 259));
    let minus_q = |n: i64| bits_w(&q1).into_iter().map(move |(x, w)| (x, -n * w));
    let sd: W = sum_mod(c, "SD", &minus_q(2).chain([(l634, 1), (l259, 1)]).collect::<Vec<_>>(), 42, 6, None).into_iter().chain([k(0), k(0)]).collect();
    (q1, r1, sd)
}

/// The byte a CPU read of `FF00 | RA` returns from the state after the segment, as
/// `Apu::read` computes it: the stored bits under each register's read mask, NR52's
/// status, wave RAM through channel 3's position while it plays, PCM12 and PCM34 on CGB,
/// and 0xFF from the unmapped addresses. One level over the post-segment fields, because
/// the cases are exclusive.
fn readback(c: &mut Circuit, ra: &W, s: &St) -> W {
    let o = |n: &str| s[n].clone();
    let shifted = |n: &str, at: usize| -> W { (0..8).map(|b| if b >= at && b - at < s[n].len() { s[n][b - at].clone() } else { k(0) }).collect() };
    let or = |parts: &[W]| -> W { (0..8).map(|b| sum(parts.iter().map(|p| p[b].clone()))).collect() };
    let none = || kbits(0, 8);
    let cgb = s["cgb"][0].clone();
    let nib2 = |lo: &str, hi: &str| -> W { s[lo].iter().chain(s[hi].iter()).cloned().collect() };
    let regs: Vec<(u8, W, u8)> = vec![
        (0x10, o("nr10").into_iter().chain([k(0)]).collect(), 0x80),
        (0x11, shifted("duty1", 6), 0x3F),
        (0x12, o("nr12"), 0x00),
        (0x13, none(), 0xFF),
        (0x14, shifted("x4b1", 6), 0xBF),
        (0x15, none(), 0xFF),
        (0x16, shifted("duty2", 6), 0x3F),
        (0x17, o("nr22"), 0x00),
        (0x18, none(), 0xFF),
        (0x19, shifted("x4b2", 6), 0xBF),
        (0x1A, shifted("en3", 7), 0x7F),
        (0x1B, none(), 0xFF),
        (0x1C, shifted("lvl3", 5), 0x9F),
        (0x1D, none(), 0xFF),
        (0x1E, shifted("x4b3", 6), 0xBF),
        (0x1F, none(), 0xFF),
        (0x20, none(), 0xFF),
        (0x21, o("nr42"), 0x00),
        (0x22, o("nr43"), 0x00),
        (0x23, shifted("x4b4", 6), 0xBF),
        (0x24, o("nr50"), 0x00),
        (0x25, o("nr51"), 0x00),
        (0x26, or(&[shifted("act1", 0), shifted("act2", 1), shifted("act3", 2), shifted("act4", 3), shifted("glob", 7)]), 0x70),
    ];
    let unmapped = (0x27..=0x2F).map(|a| (a, kbits(0, 8), 0xFF));
    let hots: Vec<(Lin, W, u8)> = regs.into_iter().chain(unmapped).map(|(a, v, m)| (all(c, &format!("RA={a:02x}"), &is(ra, a as u64)), v, m)).collect();
    // PCM12 and PCM34 read the channels' outputs on CGB, 0xFF on DMG.
    let pcm: Vec<(Lin, W)> = [(0x76u64, nib2("smp1", "smp2")), (0x77, nib2("smp3", "smp4"))].into_iter().map(|(a, v)| (all(c, &format!("RA={a:02x}"), &is(ra, a)), v)).collect();
    let wave = all(c, "RA=3x", &[ra[4].clone(), ra[5].clone(), nt(&ra[6]), nt(&ra[7])]);
    let (act3, pos3, jr) = (s["act3"][0].clone(), o("pos3"), s["jr3"][0].clone());
    // On DMG a read while channel 3 plays reaches its byte only in the cycle it read it.
    let locked = [wave.clone(), act3.clone(), nt(&cgb), nt(&jr)];
    (0..8)
        .map(|b| {
            let mut steps = vec![Step { coef: 1, lin: sum(locked.iter().cloned()), theta: 4 }];
            for (h, v, mask) in &hots {
                if mask >> b & 1 == 1 {
                    steps.push(Step { coef: 1, lin: h.clone(), theta: 1 });
                } else {
                    steps.push(Step { coef: 1, lin: h.clone() + v[b].clone(), theta: 2 });
                }
            }
            for (h, v) in &pcm {
                steps.push(Step { coef: 1, lin: h.clone() + nt(&cgb), theta: 2 });
                steps.push(Step { coef: 1, lin: h.clone() + cgb.clone() + v[b].clone(), theta: 3 });
            }
            for i in 0..16u64 {
                let ram = s[format!("ram{i}").as_str()][b].clone();
                for (playing, index) in [(vec![nt(&act3)], &ra[..4]), (vec![act3.clone(), cgb.clone()], &pos3[1..5]), (vec![act3.clone(), nt(&cgb), jr.clone()], &pos3[1..5])] {
                    let lits = with(&is(index, i), &with(&with(&[wave.clone()], &playing), &[ram.clone()]));
                    steps.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
                }
            }
            c.node(&format!("RV.{b}"), steps)
        })
        .collect()
}
