//! The register write that opens a segment (`Apu::write`): every field it changes, as
//! cases muxed into the post-write state. One write per segment, so the registers' cases
//! are exclusive by address. A second stage sets the values that read post-write fields
//! (a re-latched sample after an envelope glitch, a trigger's byte after a corruption).

use super::build::*;
use super::circuit::*;
use crate::circuit::{Circuit, Lin, Step};
use std::collections::BTreeMap;

/// Fields that survive power-off: wave RAM, the sample clock, the console and the DIV
/// bit's level.
const KEPT_OFF: [&str; 20] = ["ram0", "ram1", "ram2", "ram3", "ram4", "ram5", "ram6", "ram7", "ram8", "ram9", "ram10", "ram11", "ram12", "ram13", "ram14", "ram15", "sq", "sr", "cgb", "divb"];

/// SameBoy's trigger test `!(((x) / 2) & 0x400)` with C's division, over a form `x` in
/// [−7, 8191]: x = −1, or x in [0, 2047] ∪ [4096, 6143].
fn clear10(c: &mut Circuit, name: &str, x: Lin) -> Lin {
    c.node(name, vec![Step { coef: 1, lin: x.clone(), theta: -1 }, Step { coef: -1, lin: x.clone(), theta: 2048 }, Step { coef: 1, lin: x.clone(), theta: 4096 }, Step { coef: -1, lin: x, theta: 6144 }])
}

/// 4 bits of `nibble(byte, pos0) >> [4, 0, 1, 2][lvl]`, the wave channel's sample.
pub(super) fn wave_level(c: &mut Circuit, name: &str, byte: &[Lin], pos0: &Lin, lvl: &[Lin]) -> W {
    (0..4)
        .map(|b| {
            let mut steps = Vec::new();
            for (l, sh) in [(1u64, 0usize), (2, 1), (3, 2)] {
                for odd in [false, true] {
                    if b + sh < 4 {
                        let bit = if odd { byte[b + sh].clone() } else { byte[4 + b + sh].clone() };
                        let lits = with(&with(&is(lvl, l), &[lit(pos0, odd)]), &[bit]);
                        steps.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
                    }
                }
            }
            c.node(&format!("{name}.{b}"), steps)
        })
        .collect()
}

/// Wave RAM's byte at the 4-bit index `idx`.
pub(super) fn ram_at(c: &mut Circuit, name: &str, s: &St, idx: &[Lin]) -> W {
    (0..8)
        .map(|b| {
            let col: W = (0..16).map(|i| s[ram(i)][b].clone()).collect();
            pick(c, &format!("{name}.{b}"), &col, idx)
        })
        .collect()
}

pub(super) fn ram(i: usize) -> &'static str {
    super::fields::FIELDS[super::fields::FIELDS.iter().position(|(n, _)| *n == "ram0").unwrap() + i].0
}

/// The state's field for channel `n` ("1" to "4") by role.
pub(super) fn key(n: &str, role: &str) -> &'static str {
    let name = match (role, n) {
        ("nrx2", "1") => "nr12".to_string(),
        ("nrx2", "2") => "nr22".to_string(),
        ("nrx2", "4") => "nr42".to_string(),
        _ => format!("{role}{n}"),
    };
    super::fields::FIELDS.iter().find(|(f, _)| *f == name).unwrap_or_else(|| panic!("no field {name}")).0
}

/// An NRx2 write while the channel plays (`_nrx2_glitch`, CPU CGB E): the cases for its
/// volume, envelope countdown and envelope clock under `g`.
fn nrx2_glitch(c: &mut Circuit, n: &str, s: &St, v: &[Lin], g: &[Lin], cs: &mut Cases) {
    let (vol, old, env) = (s[key(n, "vol")].clone(), s[key(n, "nrx2")].clone(), s[key(n, "env")].clone());
    let (lk, ck, sl) = (env[0].clone(), env[1].clone(), env[2].clone());
    let inv = c.eq(&format!("{n} nrx2 invert"), v[3].clone() + old[3].clone(), 1);
    let o7z = all(c, &format!("{n} nrx2 old&7=0"), &is(&old[..3], 0));
    let v7nz = nz(c, &format!("{n} nrx2 v&7"), &v[..3]);
    let st0 = all(c, &format!("{n} nrx2 tick"), &[v7nz, o7z.clone(), nt(&lk)]);
    let eight = all(c, &format!("{n} nrx2 8/8"), &with(&with(&is(&v[..4], 8), &is(&old[..4], 8)), &[nt(&lk)]));
    let fresh = all(c, &format!("{n} nrx2 fresh"), &[o7z, nt(&lk)]);
    let tick_d = st0.clone() + eight.clone();
    let flip: W = vol.iter().map(nt).collect();
    let from14 = sum_mod(c, &format!("{n} 14-vol"), &neg_w(&vol), 14, 4, None);
    let from16 = sum_mod(c, &format!("{n} 16-vol"), &neg_w(&vol), 16, 4, None);
    let up = addk(c, &format!("{n} vol+1"), &vol, 1, 4);
    let down = addk(c, &format!("{n} vol-1"), &vol, -1, 4);
    let gv = |extra: &[Lin]| with(g, extra);
    cs.put(key(n, "vol"), gv(&[inv.clone(), v[3].clone(), fresh.clone()]), flip.clone());
    cs.put(key(n, "vol"), gv(&[inv.clone(), v[3].clone(), nt(&fresh)]), from14);
    cs.put(key(n, "vol"), gv(&[inv.clone(), nt(&v[3]), st0.clone()]), flip);
    cs.put(key(n, "vol"), gv(&[inv.clone(), nt(&v[3]), nt(&st0)]), from16);
    cs.put(key(n, "vol"), gv(&[nt(&inv), v[3].clone(), tick_d.clone()]), up);
    cs.put(key(n, "vol"), gv(&[nt(&inv), nt(&v[3]), tick_d]), down);
    cs.put(key(n, "vcd"), gv(&[ck.clone()]), v[..3].to_vec());
    // The clock clears, locking when it should, unless the write ticked; with v&7 = 0
    // only the 8/8 case ticks.
    let ticked = all(c, &format!("{n} nrx2 8/8 tick"), &[nt(&inv), eight]);
    let locks = c.ge(&format!("{n} nrx2 lock"), lk + sl.clone(), 1);
    cs.put(key(n, "env"), gv(&with(&is(&v[..3], 0), &[ck, nt(&ticked)])), vec![locks, k(0), sl]);
}

/// `64 − (v & 63)` or `256 − v`, a length load.
fn length_load(c: &mut Circuit, name: &str, v: &[Lin], max: i64) -> W {
    let bits = max.trailing_zeros() as usize;
    sum_mod(c, name, &neg_w(&v[..bits]), max, bits + 1, None)
}

/// An NRx4 write's length: a trigger reloads an expired length (disabling it), then
/// enabling length while the DIV divider's low bit is 1 clocks it once, which a trigger
/// takes from 0 to `max − 1`. Cases for `pl`, `le`, `x4b` and the expiry's `act` under
/// `g`, the trigger bit `v7`.
fn nrx4_length(c: &mut Circuit, n: &str, s: &St, v: &[Lin], g: &Lin, max: i64, cs: &mut Cases) {
    let (pl, le, div0) = (s[key(n, "pl")].clone(), s[key(n, "le")][0].clone(), s["div"][0].clone());
    let bits = pl.len();
    let t = v[7].clone();
    let plz = eqk(c, &format!("{n} pl=0"), &pl, 0);
    let pl1 = eqk(c, &format!("{n} pl=1"), &pl, 1);
    let plge2 = k(1) - plz.clone() - pl1.clone();
    let clocks = all(c, &format!("{n} length glitch"), &[g.clone(), v[6].clone(), div0]);
    let minus = addk(c, &format!("{n} pl-1"), &pl, -1, bits);
    let g = g.clone();
    cs.put(key(n, "pl"), vec![g.clone(), t.clone(), plz.clone(), clocks.clone()], kbits(max as u64 - 1, bits));
    cs.put(key(n, "pl"), vec![g.clone(), t.clone(), plz.clone(), nt(&clocks)], kbits(max as u64, bits));
    cs.put(key(n, "pl"), vec![t.clone(), pl1.clone(), clocks.clone(), nt(&le)], kbits(max as u64 - 1, bits));
    cs.put(key(n, "pl"), vec![t.clone(), plge2, clocks.clone(), nt(&le)], minus.clone());
    cs.put(key(n, "pl"), vec![nt(&t), nt(&plz), clocks.clone(), nt(&le)], minus);
    cs.put(key(n, "le"), vec![g.clone()], vec![v[6].clone()]);
    cs.put(key(n, "x4b"), vec![g], vec![v[6].clone()]);
    cs.put(key(n, "act"), vec![nt(&t), clocks, nt(&le), pl1], kbits(0, 1));
}

/// An NRx3 or NRx4 frequency's countdown reload, `4095 − 2·len + extra − lf_div·lf`.
fn reload(c: &mut Circuit, name: &str, len: &[Lin], konst: i64, lf: Option<&Lin>) -> W {
    let mut terms: Vec<(Lin, i64)> = neg_w(len).into_iter().map(|(x, w)| (x, 2 * w)).collect();
    if let Some(lf) = lf {
        terms.push((lf.clone(), -1));
    }
    sum_blocked(c, name, &terms, konst, 13, None)
}

/// Square `n`'s writes. Returns, for the second stage, the gates under which the
/// channel re-latches its sample from the post-write index and volume.
fn square(c: &mut Circuit, n: &str, s: &St, v: &[Lin], w: [&Lin; 5], cs: &mut Cases) -> Vec<Lin> {
    let [w1, w1off, w2, w3, w4] = w;
    let g = |f: &str| s[key(n, f)].clone();
    let g1 = |f: &str| s[key(n, f)][0].clone();
    let (cgb, lf) = (s["cgb"][0].clone(), s["lfdiv"][0].clone());
    let (act, sup, nrx2, len, cnt, jr, dly, env) = (g1("act"), g1("sup"), g("nrx2"), g("len"), g("c"), g1("jr"), g("dly"), g("env"));
    let dacv = nz(c, &format!("{n} dac(WV)"), &v[3..]);
    // NRx1: the length, and the duty while the APU is on.
    let load = length_load(c, &format!("{n} NRx1 load"), v, 64);
    cs.put(key(n, "pl"), vec![w1.clone()], load.clone());
    cs.put(key(n, "pl"), vec![w1off.clone()], load);
    cs.put(key(n, "duty"), vec![w1.clone()], v[6..8].to_vec());
    // NRx2: turning the DAC off stops the channel; a write while it plays glitches the
    // envelope and re-latches the sample.
    cs.put(key(n, "nrx2"), vec![w2.clone()], v.to_vec());
    cs.put(key(n, "act"), vec![w2.clone(), nt(&dacv)], kbits(0, 1));
    let playing = vec![w2.clone(), dacv, act.clone()];
    nrx2_glitch(c, n, s, v, &playing, cs);
    let mut relatch = vec![all(c, &format!("{n} NRx2 relatch"), &with(&playing, &[nt(&sup)]))];
    // NRx3: the frequency's low byte; a countdown that just reloaded reloads again.
    let len3: W = v.iter().cloned().chain(len[8..].iter().cloned()).collect();
    cs.put(key(n, "len"), vec![w3.clone()], len3.clone());
    let r3 = reload(c, &format!("{n} NRx3 reload"), &len3, 4095, None);
    cs.put(key(n, "c"), vec![w3.clone(), jr.clone()], r3);
    // NRx4.
    let t = v[7].clone();
    let len4: W = len[..8].iter().cloned().chain(v[..3].iter().cloned()).collect();
    cs.put(key(n, "len"), vec![w4.clone()], len4.clone());
    let v77 = all(c, &format!("{n} WV&7=7"), &v[..3].to_vec());
    cs.put(key(n, "x4l"), vec![w4.clone()], vec![v77.clone()]);
    // Leaving frequency bits 0x700 steps the index back when the countdown sits at a
    // whole period (SameBoy's write-timing compensation).
    // The countdown changes nearly every pass; the NRx4 tests read a copy that holds
    // still without the write.
    let cw = gated(c, &format!("{n} c@NRx4"), &cnt, w4);
    let odd = c.ge(&format!("{n} cgb|c odd"), cgb.clone() + cw[0].clone(), 1);
    let half_eq = c.eq(&format!("{n} c>>1=len^7FF"), val(&cw[1..]) + val(&len), 2047);
    let h = all(c, &format!("{n} NRx4 step back"), &[w4.clone(), nt(&t), act.clone(), g1("x4l"), nt(&v77), odd, g1("dt"), half_eq]);
    let tr = all(c, &format!("{n} trig"), &[w4.clone(), t.clone()]);
    let tri = all(c, &format!("{n} trig idle"), &[tr.clone(), nt(&act)]);
    let tra = all(c, &format!("{n} trig playing"), &[tr.clone(), act.clone()]);
    // A stopped channel's start advances the index when bit 10 of (countdown − delay) / 2
    // is clear, the countdown read after a just-reloaded channel's reload.
    let dl = val(&dly);
    let c_now = clear10(c, &format!("{n} trig clear10(c-d)"), val(&cw) - dl.clone());
    let r_lin = Lin::konst(4095) - val(&len4) * 2;
    let c_rel = clear10(c, &format!("{n} trig clear10(r-d)"), r_lin - dl.clone());
    let f = c.node(&format!("{n} trig forward"), vec![Step { coef: 1, lin: cgb.clone() + nt(&jr) + c_now + nt(&v[2]), theta: 4 }, Step { coef: 1, lin: cgb.clone() + jr.clone() + c_rel + nt(&v[2]), theta: 4 }]);
    let fwd = all(c, &format!("{n} trig idle advance"), &[tri.clone(), f.clone()]);
    // A playing channel's start advances it when it did not just reload; otherwise a move
    // to frequency 0x7FF while suppressed waits 2 cycles more.
    let c_act = clear10(c, &format!("{n} trig clear10(c-1-d)"), val(&cw) - 1 - dl);
    let ca = all(c, &format!("{n} trig advance"), &[tra.clone(), cgb.clone(), nt(&jr), nt(&v[2]), c_act]);
    let was7ff = eqk(c, &format!("{n} len=7FF"), &len, 0x7FF);
    let cb = all(c, &format!("{n} trig extra"), &with(&with(&[tra.clone(), cgb.clone(), nt(&ca)], &v[..3].to_vec()), &with(&is(&len[..8], 0xFF), &[nt(&was7ff), sup.clone()])));
    let idx = g("idx");
    let idx_up = addk(c, &format!("{n} idx+1"), &idx, 1, 3);
    let idx_dn = addk(c, &format!("{n} idx-1"), &idx, -1, 3);
    cs.put(key(n, "idx"), vec![h.clone()], idx_dn);
    cs.put(key(n, "idx"), vec![fwd], idx_up.clone());
    cs.put(key(n, "idx"), vec![ca.clone()], idx_up);
    let dac2 = nz(c, &format!("{n} dac(NRx2)"), &nrx2[3..]);
    cs.put(key(n, "sup"), vec![h], kbits(0, 1));
    cs.put(key(n, "sup"), vec![ca.clone()], kbits(0, 1));
    cs.put(key(n, "sup"), vec![tri.clone(), dac2.clone()], vec![nt(&f)]);
    cs.put(key(n, "dt"), vec![tr.clone()], kbits(0, 1));
    cs.put(key(n, "env"), vec![tr.clone()], vec![k(0), k(0), env[2].clone()]);
    // The start delay on top of the period: 6 − lf_div idle or after the extra wait,
    // else 4 − lf_div.
    let long = c.node(&format!("{n} trig long"), vec![Step { coef: 1, lin: tri.clone() + cb.clone(), theta: 1 }]);
    cs.put(key(n, "dly"), vec![long.clone()], vec![lf.clone(), nt(&lf), k(1)]);
    cs.put(key(n, "dly"), vec![tra.clone(), nt(&cb)], vec![lf.clone(), lf.clone(), nt(&lf)]);
    let c_long = reload(c, &format!("{n} trig c long"), &len4, 4100, Some(&lf));
    let c_short = reload(c, &format!("{n} trig c short"), &len4, 4098, Some(&lf));
    cs.put(key(n, "c"), vec![long], c_long);
    cs.put(key(n, "c"), vec![tra.clone(), nt(&cb)], c_short);
    let r4 = reload(c, &format!("{n} NRx4 reload"), &len4, 4095, None);
    cs.put(key(n, "c"), vec![w4.clone(), nt(&t), jr], r4);
    cs.put(key(n, "vol"), vec![tr.clone()], nrx2[4..].to_vec());
    cs.put(key(n, "vcd"), vec![tr.clone()], nrx2[..3].to_vec());
    nrx4_length(c, n, s, v, w4, 64, cs);
    cs.put(key(n, "act"), vec![tri.clone(), dac2], kbits(1, 1));
    cs.put(key(n, "smp"), vec![tri], kbits(0, 4));
    relatch.push(tra);
    if n == "1" {
        // Channel 1's start restarts the sweep.
        let nr10 = s["nr10"].clone();
        let shift = nr10[..3].to_vec();
        let snz = nz(c, "trig NR10&7", &shift);
        cs.put("inst", vec![tr.clone()], kbits(0, 1));
        cs.put("shd", vec![tr.clone()], kbits(0, 11));
        cs.put("cmp", vec![tr.clone()], kbits(0, 11));
        cs.put("calc", vec![tr.clone(), snz.clone()], shift.clone());
        cs.put("rtm", vec![tr.clone(), snz.clone()], vec![nt(&act), k(1)]);
        cs.put("unsh", vec![tr.clone(), snz.clone()], kbits(0, 1));
        let shifted = shr(c, "trig add", &len4, &shift, 11);
        cs.put("add", vec![tr.clone(), snz.clone()], shifted);
        cs.put("add", vec![tr.clone(), nt(&snz)], kbits(0, 11));
        let h1 = c.eq("trig hold.1", lf.clone() - cgb.clone(), 0);
        let h2 = all(c, "trig hold.2", &[cgb.clone(), nt(&lf)]);
        cs.put("hold", vec![tr.clone()], vec![lf.clone(), h1, h2]);
        cs.put("swcd", vec![tr], nr10[4..7].iter().map(nt).collect());
    }
    relatch
}

/// NR10: the glitch on a running calculation, the overflow check against the old
/// negate bit, and the sweep calculation a period of 7 starts.
fn nr10(c: &mut Circuit, s: &St, v: &[Lin], w10: &Lin, cs: &mut Cases) {
    let g = |f: &str| s[f].clone();
    let lf = s["lfdiv"][0].clone();
    let (old, calc, rtm, swcd, hold) = (g("nr10"), g("calc"), g("rtm"), g("swcd"), g("hold"));
    let v7z = is(&v[..3], 0);
    let v7nz = nz(c, "NR10 WV&7", &v[..3]);
    let pace = c.ge("NR10 pace", v[4].clone() + v[5].clone() + v[6].clone(), 1);
    let ts = all(c, "NR10 sweep calc", &with(&[w10.clone(), pace], &is(&swcd, 7)));
    let r2 = eqk(c, "rtm=2", &rtm, 2);
    let o7z = is(&old[..3], 0);
    let vge2 = c.ge("WV&7>=2", v[1].clone() + v[2].clone(), 1);
    let dec2 = all(c, "NR10 dec2", &with(&o7z, &[nt(&lf), vge2]));
    let cge2 = c.ge("calc>=2", calc[1].clone() + calc[2].clone(), 1);
    let dec1 = all(c, "NR10 dec1", &with(&o7z, &[nt(&lf), v7nz.clone(), cge2]));
    let vm1 = addk(c, "WV&7-1", &v[..3], -1, 3);
    let cm1 = addk(c, "calc-1", &calc, -1, 3);
    cs.put("calc", vec![ts.clone()], v[..3].to_vec());
    cs.put("calc", vec![w10.clone(), nt(&ts), r2.clone(), dec2.clone()], vm1);
    cs.put("calc", vec![w10.clone(), nt(&ts), r2.clone(), nt(&dec2)], v[..3].to_vec());
    cs.put("calc", vec![w10.clone(), nt(&ts), nt(&r2), dec1], cm1);
    cs.put("rtm", vec![ts.clone()], vec![nt(&lf), lf.clone()]);
    cs.put("rtm", with(&[w10.clone(), nt(&ts), r2], &v7z), kbits(0, 2));
    cs.put("nr10", vec![w10.clone()], v[..7].to_vec());
    let ovf = c.ge("NR10 overflow", val(&g("shd")) + val(&g("cmp")) + old[3].clone(), 2048);
    cs.put("act1", vec![w10.clone(), ovf, nt(&v[3])], kbits(0, 1));
    let terms: Vec<(Lin, i64)> = bits_w(&g("add")).into_iter().chain(bits_w(&g("shd"))).chain([(v[3].clone(), 1)]).collect();
    let sum11 = sum_blocked(c, "NR10 len", &terms, 0, 11, None);
    cs.put("len1", vec![ts.clone(), v7nz.clone()], sum11.clone());
    let hold0 = is(&hold, 0);
    let shifted = shr(c, "NR10 add", &sum11, &v[..3], 11);
    cs.put("add", with(&[ts.clone(), v7nz.clone()], &hold0), shifted);
    cs.put("add", with(&with(&[ts.clone()], &v7z), &hold0), g("len1"));
    cs.put("unsh", vec![ts.clone()], vec![nt(&v7nz)]);
    cs.put("swcd", vec![ts.clone()], v[4..7].iter().map(nt).collect());
    cs.put("inst", with(&[ts], &v7z), kbits(1, 1));
}

/// DMG's wave RAM corruption at a trigger in the cycle before the channel reads: offset
/// ⌈pos/2⌉ mod 16; byte 0 takes the offset's byte for the first four, else the block of
/// four holding it is copied to the start, its first byte the majority of itself, byte 0
/// and the read byte.
fn corrupt(c: &mut Circuit, s: &St, bad: &Lin, cs: &mut Cases) {
    let pos = s["pos3"].clone();
    let r: Vec<W> = (0..16).map(|i| s[ram(i)].clone()).collect();
    let offs: Vec<(usize, Lin)> = (1..16usize)
        .map(|o| {
            let steps = [2 * o - 1, 2 * o].map(|p| {
                let lits = with(&[bad.clone()], &is(&pos, (p % 32) as u64));
                Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }
            });
            (o, c.node(&format!("w3off{o}"), steps.to_vec()))
        })
        .collect();
    let majority: BTreeMap<usize, W> = (4..16)
        .map(|o| {
            let block = o & !3;
            (o, (0..8).map(|b| c.ge(&format!("w3maj{o}.{b}"), r[0][b].clone() + r[block][b].clone() + r[o][b].clone(), 2)).collect())
        })
        .collect();
    for i in 0..16usize {
        for (o, gate) in &offs {
            let (o, block) = (*o, *o & !3);
            let value = match (i, o) {
                (0, 1..=3) => r[o].clone(),
                (0, 4..) => majority[&o].clone(),
                (1..=3, 4..) => r[block + i].clone(),
                _ if o >= 4 && i == block => majority[&o].clone(),
                _ => continue,
            };
            cs.put(ram(i), vec![gate.clone()], value);
        }
    }
}

/// The wave channel's writes. Returns the gate of a trigger whose channel reads this
/// cycle (its byte is wave RAM's first after any corruption) and the start's gate.
fn wave(c: &mut Circuit, s: &St, e: &Ev, w: [&Lin; 6], wram: &Lin, cs: &mut Cases) -> (Lin, Lin) {
    let [w1a, w1b, w1boff, w1c, w1d, w1e] = w;
    let v = &e.wv;
    let (act, en, pos, cnt, bug, byte, len, cgb, jr) = (s["act3"][0].clone(), s["en3"][0].clone(), s["pos3"].clone(), s["c3"].clone(), s["bug3"].clone(), s["byte3"].clone(), s["len3"].clone(), s["cgb"][0].clone(), s["jr3"][0].clone());
    cs.put("en3", vec![w1a.clone()], vec![v[7].clone()]);
    let stop = all(c, "NR30 off", &[w1a.clone(), nt(&v[7])]);
    cs.put("pul3", vec![stop.clone()], kbits(0, 1));
    cs.put("act3", vec![stop.clone()], kbits(0, 1));
    let c0 = vec![eqk(c, "c3=0", &cnt, 0)];
    let here = ram_at(c, "wave bus", s, &pos[1..]);
    cs.put("byte3", with(&[stop, act.clone()], &c0), here);
    let load = length_load(c, "NR31 load", v, 256);
    cs.put("pl3", vec![w1b.clone()], load.clone());
    cs.put("pl3", vec![w1boff.clone()], load);
    cs.put("lvl3", vec![w1c.clone()], v[5..7].to_vec());
    let relevel = wave_level(c, "NR32 smp", &byte, &pos[0], &v[5..7]);
    cs.put("smp3", vec![w1c.clone(), act.clone()], relevel);
    let len3: W = v.iter().cloned().chain(len[8..].iter().cloned()).collect();
    cs.put("len3", vec![w1d.clone()], len3.clone());
    let r3 = sum_mod(c, "NR33 reload", &neg_w(&len3), 2047, 12, None);
    cs.put("c3", with(&[w1d.clone()], &is(&bug, 1)), r3);
    let tr = all(c, "NR34 trig", &[w1e.clone(), v[7].clone()]);
    let len4: W = len[..8].iter().cloned().chain(v[..3].iter().cloned()).collect();
    cs.put("len3", vec![w1e.clone()], len4.clone());
    cs.put("pul3", vec![tr.clone()], kbits(1, 1));
    cs.put("pos3", vec![tr.clone()], kbits(0, 5));
    let start = sum_mod(c, "NR34 c", &neg_w(&len4), 2050, 12, None);
    cs.put("c3", vec![tr.clone()], start);
    nrx4_length(c, "3", s, v, w1e, 256, cs);
    cs.put("act3", vec![tr.clone(), en.clone()], kbits(1, 1));
    let bad = all(c, "NR34 corrupt", &with(&[tr.clone(), nt(&cgb), act.clone()], &c0));
    corrupt(c, s, &bad, cs);
    // A CPU write to wave RAM: by address while the channel is off; while it plays, to
    // the byte it plays, on DMG only in the cycle it read it.
    for i in 0..16u64 {
        cs.put(ram(i as usize), with(&[wram.clone(), nt(&act)], &is(&e.wa[..4], i)), v.clone());
        cs.put(ram(i as usize), with(&[wram.clone(), act.clone(), cgb.clone()], &is(&pos[1..], i)), v.clone());
        cs.put(ram(i as usize), with(&[wram.clone(), act.clone(), nt(&cgb), jr.clone()], &is(&pos[1..], i)), v.clone());
    }
    let reread = all(c, "NR34 reread", &with(&[tr.clone(), act], &c0));
    let starts = all(c, "NR34 starts", &[tr, en]);
    (reread, starts)
}

/// NR44's trigger (`prepare_noise_start`, CPU CGB E): the counter's restart phase by the
/// divisor, the alignment and the background counter, and the LFSR's start.
fn noise_trigger(c: &mut Circuit, s: &St, v: &[Lin], w23: &Lin, cs: &mut Cases) {
    let g = |f: &str| s[f].clone();
    let g1 = |f: &str| s[f][0].clone();
    let (act, nr42, nr43) = (g1("act4"), g("nr42"), g("nr43"));
    let tr = all(c, "NR44 trig", &[w23.clone(), v[7].clone()]);
    // The counter, its countdown and the alignment change nearly every pass; the
    // trigger reads copies that hold still without it.
    let (cc, al, ctr) = (gated(c, "cc4@NR44", &g("cc4"), &tr), gated(c, "align@NR44", &g("align"), &tr), gated(c, "ctr4@NR44", &g("ctr4"), &tr));
    let dac = nz(c, "dac(NR42)", &nr42[3..]);
    cs.put("cact", vec![tr.clone()], vec![dac.clone()]);
    cs.put("ddis", vec![tr.clone()], vec![nt(&dac)]);
    cs.put("bact", vec![tr.clone()], kbits(1, 1));
    cs.put("env4", vec![tr.clone()], vec![k(0), k(0), s["env4"][2].clone()]);
    let div = nr43[..3].to_vec();
    let dz = eqk(c, "NR43 div=0", &div, 0);
    let d1 = eqk(c, "NR43 div=1", &div, 1);
    let dgt1 = k(1) - dz.clone() - d1.clone();
    let cc1 = eqk(c, "cc4=1", &cc, 1);
    let cc2 = eqk(c, "cc4=2", &cc, 2);
    let (a0, a1) = (al[0].clone(), al[1].clone());
    let b1 = all(c, "NR44 count", &[tr.clone(), dgt1.clone(), cc1]);
    let b2 = all(c, "NR44 phase", &[tr.clone(), cc2, nt(&a0), nt(&a1), act.clone()]);
    let d8 = all(c, "NR44 div 8", &[b2.clone(), dz.clone()]);
    let dzr = all(c, "NR44 div 0", &[tr.clone(), dz.clone(), nt(&b2)]);
    let d1r = all(c, "NR44 div 1", &[tr.clone(), d1.clone()]);
    let dgr = all(c, "NR44 div n", &[tr.clone(), dgt1]);
    let big = d8.clone() + dgr.clone();
    let nonzero = big.clone() + d1r.clone();
    let bact = g1("bact");
    let hi0 = is(&nr43[4..], 0);
    let mut terms: Vec<(Lin, i64)> = vec![(dzr.clone(), 6), (d8.clone(), 38), (d1r.clone(), 10), (dgr.clone(), 6)];
    for (b, x) in div.iter().enumerate() {
        terms.push((all(c, &format!("NR44 4div.{b}"), &[dgr.clone(), x.clone()]), 4 << b));
    }
    let mut add = |c: &mut Circuit, name: &str, lits: &[Lin], w: i64| terms.push((all(c, name, lits), w));
    add(c, "NR44 odd 0 bg", &[a0.clone(), dzr.clone(), bact.clone()], -1);
    add(c, "NR44 odd 0", &[a0.clone(), dzr.clone(), nt(&bact)], 1);
    add(c, "NR44 3 1 idle", &[a0.clone(), a1.clone(), d1r.clone(), nt(&act)], 1);
    add(c, "NR44 3 1", &[a0.clone(), a1.clone(), d1r.clone(), act.clone()], -3);
    add(c, "NR44 3 n", &[a0.clone(), a1.clone(), big.clone()], -3);
    add(c, "NR44 1", &[a0.clone(), nt(&a1), nonzero.clone()], -1);
    add(c, "NR44 1 1", &[a0.clone(), nt(&a1), d1r.clone(), act.clone()], -4);
    add(c, "NR44 2", &[nt(&a0), a1.clone(), nonzero], -2);
    add(c, "NR44 0 n", &[nt(&a0), nt(&a1), big.clone()], -4);
    add(c, "NR44 0 1", &with(&[nt(&a0), nt(&a1), d1r.clone(), act.clone()], &hi0), -4);
    add(c, "NR44 bg n", &[big, nt(&dac), nt(&a0), nt(&a1)], 4);
    add(c, "NR44 bg 0", &[dzr.clone(), bact.clone(), nt(&act), nt(&a0), nt(&a1), g1("ddis")], 28);
    add(c, "NR44 bg 1", &[d1r.clone(), bact, nt(&act), nt(&a0), nt(&a1)], -4);
    add(c, "NR44 div1 glitch", &[b2.clone(), d1.clone(), nt(&g1("dst4"))], -4);
    let cd = sum_mod(c, "NR44 cc", &terms, 0, 6, None);
    cs.put("cc4", vec![tr.clone()], cd);
    // The counter counts once when its countdown was about to.
    let b2d1 = all(c, "NR44 phase 1", &[b2, d1]);
    let up: W = addk(c, "NR44 ctr+1", &ctr[..14], 1, 14).into_iter().chain([k(0)]).collect();
    cs.put("ctr4", vec![b1], up.clone());
    cs.put("ctr4", vec![b2d1.clone()], up);
    // That count steps the LFSR at once when it raises the selected bit.
    let rise_steps: Vec<Step> = (0..14u64)
        .map(|sh| {
            let mut lits = with(&with(&is(&nr43[4..], sh), &[b2d1.clone()]), &[nt(&ctr[sh as usize])]);
            lits.extend(ctr[..sh as usize].iter().cloned());
            Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }
        })
        .collect();
    let instant = c.node("NR44 instant step", rise_steps);
    let x55 = all(c, "NR44 lfsr 55", &[dzr, act, a0.clone(), a1.clone()]);
    let narrow = nr43[3].clone();
    cs.put("lfsr", vec![x55.clone()], kbits(0x55, 15));
    cs.put("lfsr", vec![instant.clone()], (0..15).map(|i| if i == 14 { k(1) } else if i == 6 { narrow.clone() } else { k(0) }).collect());
    cs.put("lfsr", vec![tr.clone(), nt(&x55), nt(&instant)], kbits(0, 15));
    cs.put("b7", vec![instant.clone()], kbits(0, 1));
    cs.put("sin", vec![instant], vec![narrow]);
    cs.put("vol4", vec![tr.clone()], nr42[4..].to_vec());
    cs.put("cur4", vec![tr.clone()], kbits(0, 1));
    cs.put("vcd4", vec![tr.clone()], nr42[..3].to_vec());
    let al2 = all(c, "align=2", &[nt(&a0), a1.clone()]);
    cs.put("dst4", vec![tr.clone()], vec![al2]);
    nrx4_length(c, "4", s, v, w23, 64, cs);
    cs.put("act4", vec![tr.clone(), dac], kbits(1, 1));
    cs.put("smp4", vec![tr], kbits(0, 4));
}

pub(super) fn phase(c: &mut Circuit, s: &St, e: &Ev) -> St {
    let g = |n: &str| s[n].clone();
    let g1 = |n: &str| s[n][0].clone();
    let (glob, cgb) = (g1("glob"), g1("cgb"));
    let v = e.wv.clone();
    let at = |a: u8| with(&is(&e.wa, a as u64), &[e.wp.clone()]);
    let mut on: BTreeMap<u8, Lin> = BTreeMap::new();
    for a in (0x10..=0x25u8).filter(|a| ![0x15, 0x1F].contains(a)) {
        on.insert(a, all(c, &format!("w{a:02x}"), &with(&at(a), &[glob.clone()])));
    }
    let off: BTreeMap<u8, Lin> = [0x11u8, 0x16, 0x1B, 0x20].iter().map(|&a| (a, all(c, &format!("w{a:02x}off"), &with(&at(a), &[nt(&glob), nt(&cgb)])))).collect();
    let pon = all(c, "w26 on", &with(&at(0x26), &[v[7].clone(), nt(&glob)]));
    let poff = all(c, "w26 off", &with(&at(0x26), &[nt(&v[7]), glob.clone()]));
    let wram = all(c, "w3x", &[e.wa[4].clone(), e.wa[5].clone(), nt(&e.wa[6]), nt(&e.wa[7]), e.wp.clone()]);
    let mut cs = Cases::new();

    // Power-off clears every field but those it keeps; power-on (`GB_apu_init`) finds
    // them cleared and sets the few it starts nonzero.
    for f in s.keys() {
        if !KEPT_OFF.contains(f) {
            cs.put(f, vec![poff.clone()], kbits(0, s[f].len()));
        }
    }
    cs.put("glob", vec![pon.clone()], kbits(1, 1));
    cs.put("lfdiv", vec![pon.clone()], kbits(1, 1));
    // The 1 MHz phase and the alignment count on while the APU is off.
    cs.put("align", vec![pon.clone()], kbits(0, 2));
    cs.put("c1", vec![pon.clone()], kbits(0x1FFF, 13));
    cs.put("c2", vec![pon.clone()], kbits(0x1FFF, 13));
    let divb = g1("divb");
    cs.put("skip", vec![pon.clone()], vec![k(0), divb.clone()]);
    cs.put("div", vec![pon], vec![divb, k(0), k(0)]);
    cs.put("nr50", vec![on[&0x24].clone()], v.clone());
    cs.put("nr51", vec![on[&0x25].clone()], v.clone());
    nr10(c, s, &v, &on[&0x10], &mut cs);
    let relatch1 = square(c, "1", s, &v, [&on[&0x11], &off[&0x11], &on[&0x12], &on[&0x13], &on[&0x14]], &mut cs);
    let relatch2 = square(c, "2", s, &v, [&on[&0x16], &off[&0x16], &on[&0x17], &on[&0x18], &on[&0x19]], &mut cs);
    let (reread, starts) = wave(c, s, e, [&on[&0x1A], &on[&0x1B], &off[&0x1B], &on[&0x1C], &on[&0x1D], &on[&0x1E]], &wram, &mut cs);
    // The noise channel.
    let load = length_load(c, "NR41 load", &v, 64);
    cs.put("pl4", vec![on[&0x20].clone()], load.clone());
    cs.put("pl4", vec![off[&0x20].clone()], load);
    let w21 = on[&0x21].clone();
    let dacv = nz(c, "NR42 dac(WV)", &v[3..]);
    let act4 = g1("act4");
    cs.put("nr42", vec![w21.clone()], v.clone());
    cs.put("act4", vec![w21.clone(), nt(&dacv)], kbits(0, 1));
    cs.put("cact", vec![w21.clone(), nt(&dacv)], kbits(0, 1));
    let nr43nz = nz(c, "NR43&7", &g("nr43")[..3]);
    cs.put("bact", vec![w21.clone(), nt(&dacv), nt(&cgb)], kbits(0, 1));
    cs.put("bact", vec![w21.clone(), nt(&dacv), cgb.clone(), act4.clone(), nr43nz.clone()], kbits(0, 1));
    let (ccw, ctrw) = (gated(c, "cc4@NR42", &g("cc4"), &w21), gated(c, "ctr4@NR42", &g("ctr4"), &w21));
    let cc_le2 = c.ge("cc4<=2", k(2) - val(&ccw), 0);
    let ctr_up = addk(c, "NR42 ctr+1", &ctrw, 1, 15);
    cs.put("ctr4", vec![w21.clone(), nt(&dacv), act4.clone(), nr43nz, cc_le2], ctr_up);
    let playing4 = vec![w21, dacv, act4];
    nrx2_glitch(c, "4", s, &v, &playing4, &mut cs);
    let relatch4 = all(c, "NR42 relatch", &playing4);
    let w22 = on[&0x22].clone();
    cs.put("nr43", vec![w22.clone()], v.clone());
    let rld = g1("rld4");
    let v7z = is(&v[..3], 0);
    let v7nz = nz(c, "NR43 WV&7", &v[..3]);
    cs.put("cc4", with(&[w22.clone(), rld.clone()], &v7z), kbits(2, 6));
    let al22 = gated(c, "align@NR43", &g("align"), &w22);
    for (a, off_a) in [(0u64, 2u64), (1, 1), (2, 0), (3, 3)] {
        cs.put("cc4", with(&[w22.clone(), rld.clone(), v7nz.clone()], &is(&al22, a)), vec![k((off_a & 1) as i64), k((off_a >> 1) as i64), v[0].clone(), v[1].clone(), v[2].clone(), k(0)]);
    }
    super::lfsr::nr43_write(c, s, &v, &w22, &mut cs);
    noise_trigger(c, s, &v, &on[&0x23], &mut cs);

    let s1 = cs.finish(c, "'", s);
    // Second stage: samples that read post-write fields.
    let mut cs = Cases::new();
    for (n, gates) in [("1", relatch1), ("2", relatch2)] {
        let lvl = duty_level(c, &format!("{n} relatch smp"), &s1[key(n, "duty")], &s1[key(n, "idx")], &s1[key(n, "vol")]);
        // A playing channel's trigger re-latches only once its index left suppression.
        let fires: Vec<Lin> = gates.iter().enumerate().map(|(i, g)| if i == 0 { g.clone() } else { all(c, &format!("{n} trig relatch"), &[g.clone(), nt(&s1[key(n, "sup")][0])]) }).collect();
        let any = c.node(&format!("{n} relatches"), vec![Step { coef: 1, lin: sum(fires.into_iter()), theta: 1 }]);
        cs.put(key(n, "smp"), vec![any], lvl);
    }
    let byte1 = s1["ram0"].clone();
    cs.put("byte3", vec![reread.clone()], byte1.clone());
    let byte_t: W = (0..8).map(|b| c.node(&format!("NR34 byte.{b}"), vec![Step { coef: 1, lin: reread.clone() + byte1[b].clone(), theta: 2 }, Step { coef: 1, lin: nt(&reread) + s1["byte3"][b].clone(), theta: 2 }])).collect();
    let smp_t = wave_level(c, "NR34 smp", &byte_t, &k(0), &s1["lvl3"]);
    cs.put("smp3", vec![starts], smp_t);
    let cur = s1["cur4"][0].clone();
    let smp4: W = (0..4).map(|b| all(c, &format!("NR42 relatch smp.{b}"), &[cur.clone(), s1["vol4"][b].clone()])).collect();
    cs.put("smp4", vec![relatch4], smp4);
    cs.finish(c, "''", &s1)
}
