//! The frame sequencer's events at a segment's first M-cycle, in `Apu::mcycle`'s order:
//! a fall from a reset of the counter, the envelope steps a fall at double speed left
//! waiting, a counted fall, a rise. SameBoy's `GB_apu_div_event`,
//! `GB_apu_delayed_envelope_tick` and `GB_apu_div_secondary_event`. Only one of the falls
//! and rises comes per segment: the host gives each its own one-M-cycle segment.

use super::build::*;
use super::circuit::*;
use super::write::key;
use crate::circuit::{Circuit, Lin, Step};
use std::collections::BTreeMap;

/// `[x]` for one of several exclusive conjunctions.
fn any(c: &mut Circuit, name: &str, sets: &[Vec<Lin>]) -> Lin {
    let steps = sets.iter().map(|lits| Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 }).collect();
    c.node(name, steps)
}

pub(super) fn phase(c: &mut Circuit, s: &St, e: &Ev) -> St {
    let g = |n: &str| s[n].clone();
    let g1 = |n: &str| s[n][0].clone();
    let (glob, cgb, lf) = (g1("glob"), g1("cgb"), g1("lfdiv"));
    let mut cs = Cases::new();
    let (skip, div) = (g("skip"), g("div"));
    let fr = all(c, "fall on", &[e.fall.clone(), glob.clone()]);
    let skip2 = is(&skip, 2);
    let (skip0, skip1) = (is(&skip, 0), is(&skip, 1));
    cs.put("skip", with(&[fr.clone()], &skip2), kbits(1, 1).into_iter().chain([k(0)]).collect());
    cs.put("skip", with(&[fr.clone()], &skip1), kbits(0, 2));
    let not2 = c.ge("skip!=2", k(1) - skip[1].clone() + skip[0].clone(), 1);
    let run = all(c, "fall runs", &[fr.clone(), not2]);
    let inc = addk(c, "div+1", &div, 1, 3);
    cs.put("div", with(&[run.clone()], &skip0), inc);
    // The divider after the event: incremented unless the skip just ended.
    let d7 = any(c, "div'=7", &[with(&with(&[run.clone()], &skip0), &is(&div, 6)), with(&with(&[run.clone()], &skip1), &is(&div, 7))]);
    let d1 = any(c, "div' odd", &[with(&[run.clone(), nt(&div[0])], &skip0), with(&[run.clone(), div[0].clone()], &skip1)]);
    let d3 = any(c, "div'&3=3", &[with(&with(&[run.clone()], &skip0), &is(&div[..2], 2)), with(&with(&[run.clone()], &skip1), &is(&div[..2], 3))]);
    let reset = all(c, "fall reset", &[e.fall.clone(), e.reset.clone()]);
    // The envelope steps: now for a reset's fall or any fall at normal speed or on DMG;
    // a counted fall at double speed on CGB leaves them for the next M-cycle.
    let deferred = all(c, "deferred", &[e.dbl.clone(), cgb.clone()]);
    let now = c.node("ticks now", vec![Step { coef: 1, lin: run.clone() + reset.clone(), theta: 2 }, Step { coef: 1, lin: run.clone() + nt(&reset) + nt(&deferred), theta: 3 }]);
    // A counted fall at double speed on CGB leaves its steps for the next M-cycle's start
    // (pend 2 while its own M-cycle runs); a reset's fall there runs them now, with any
    // left waiting.
    let pset = all(c, "pend set", &[run.clone(), nt(&reset), deferred.clone()]);
    cs.put("pend", vec![pset], vec![k(0), k(1)]);
    let merged = all(c, "pend merged", &[run.clone(), reset.clone(), deferred]);
    cs.put("pend", vec![merged], kbits(0, 2));
    let locks = steps(c, s, &now, "fs", &mut cs);
    for n in ["1", "2", "4"] {
        let (env, vol, vcd, nrx2, act) = (g(key(n, "env")), g(key(n, "vol")), g(key(n, "vcd")), g(key(n, "nrx2")), g1(key(n, "act")));
        let (lk, ck, sl) = (env[0].clone(), env[1].clone(), env[2].clone());
        let vm1 = addk(c, &format!("{n} vcd-1"), &vcd, -1, 3);
        cs.put(key(n, "vcd"), vec![d7.clone(), nt(&ck)], vm1);
        let locks = locks[n].clone();
        // The rise: an envelope whose countdown ran out reloads it and arms its clock.
        let rs = all(c, &format!("{n} env rise"), &with(&[e.rise.clone(), glob.clone(), act], &is(&vcd, 0)));
        cs.put(key(n, "vcd"), vec![rs.clone()], nrx2[..3].to_vec());
        let paced = nz(c, &format!("{n} rise pace"), &nrx2[..3]);
        let should = any(c, &format!("{n} rise should lock"), &[with(&is(&vol, 15), &[nrx2[3].clone()]), with(&is(&vol, 0), &[nt(&nrx2[3])])]);
        cs.put(key(n, "env"), vec![rs.clone(), paced.clone(), nt(&ck)], vec![lk.clone(), k(1), should]);
        cs.put(key(n, "env"), vec![rs, nt(&paced), ck], vec![locks, k(0), sl]);
    }
    // Length clocks at odd divider steps.
    for n in ["1", "2", "3", "4"] {
        let (pl, le) = (g(key(n, "pl")), g1(key(n, "le")));
        let nzp = nz(c, &format!("{n} fs pl>0"), &pl);
        let one = eqk(c, &format!("{n} fs pl=1"), &pl, 1);
        let m1 = addk(c, &format!("{n} fs pl-1"), &pl, -1, pl.len());
        cs.put(key(n, "pl"), vec![d1.clone(), le.clone(), nzp], m1);
        cs.put(key(n, "act"), vec![d1.clone(), le, one], kbits(0, 1));
    }
    // The sweep's period at every fourth step; at 7 with a pace set, a calculation starts.
    {
        let (nr10, swcd, hold) = (g("nr10"), g("swcd"), g("hold"));
        let pace = c.ge("fs pace", nr10[4].clone() + nr10[5].clone() + nr10[6].clone(), 1);
        let tsc = all(c, "fs sweep calc", &with(&[d3.clone(), pace], &is(&swcd, 6)));
        let up = addk(c, "swcd+1", &swcd, 1, 3);
        cs.put("swcd", vec![d3.clone(), nt(&tsc)], up);
        cs.put("swcd", vec![tsc.clone()], nr10[4..7].iter().map(nt).collect());
        let shift = nr10[..3].to_vec();
        let snz = nz(c, "fs NR10&7", &shift);
        let terms: Vec<(Lin, i64)> = bits_w(&g("add")).into_iter().chain(bits_w(&g("shd"))).chain([(nr10[3].clone(), 1)]).collect();
        let sum11 = sum_blocked(c, "fs len", &terms, 0, 11, None);
        cs.put("len1", vec![tsc.clone(), snz.clone()], sum11.clone());
        let hold0 = is(&hold, 0);
        let shifted = shr(c, "fs add", &sum11, &shift, 11);
        cs.put("add", with(&[tsc.clone(), snz.clone()], &hold0), shifted);
        cs.put("add", with(&[tsc.clone(), nt(&snz)], &hold0), g("len1"));
        cs.put("calc", vec![tsc.clone()], shift.clone());
        // A DIV write's event at normal speed reloads 1, else 1 + lf_div.
        let dw = all(c, "fs div write", &[reset, nt(&e.dbl)]);
        cs.put("rtm", vec![tsc.clone(), dw.clone()], vec![k(1), k(0)]);
        cs.put("rtm", vec![tsc.clone(), nt(&dw)], vec![nt(&lf), lf]);
        cs.put("unsh", vec![tsc.clone()], vec![nt(&snz)]);
        cs.put("inst", with(&[tsc], &is(&shift, 0)), kbits(1, 1));
    }
    cs.put("divb", vec![e.fall.clone()], kbits(0, 1));
    cs.put("divb", vec![e.rise.clone()], kbits(1, 1));
    cs.finish(c, "_e", s)
}

/// The envelope steps under `gate` (`tick_square_envelope`, `tick_noise_envelope`): a
/// clock that is set clears, locking when it should; unlocked and paced, the volume moves
/// and a playing channel re-latches its sample. Returns each channel's `locked ∨
/// should_lock` wire for the clock's other cases.
fn steps(c: &mut Circuit, s: &St, gate: &Lin, tag: &str, cs: &mut Cases) -> BTreeMap<&'static str, Lin> {
    let g = |n: &str| s[n].clone();
    let g1 = |n: &str| s[n][0].clone();
    let mut out = BTreeMap::new();
    for n in ["1", "2", "4"] {
        let (env, vol, nrx2) = (g(key(n, "env")), g(key(n, "vol")), g(key(n, "nrx2")));
        let (lk, ck, sl) = (env[0].clone(), env[1].clone(), env[2].clone());
        let tk = all(c, &format!("{n} {tag} env tick"), &[gate.clone(), ck]);
        let locks = c.ge(&format!("{n} {tag} env locks"), lk.clone() + sl.clone(), 1);
        cs.put(key(n, "env"), vec![tk.clone()], vec![locks.clone(), k(0), sl.clone()]);
        let pace = nz(c, &format!("{n} {tag} env pace"), &nrx2[..3]);
        let up = addk(c, &format!("{n} {tag} env vol+1"), &vol, 1, 4);
        let down = addk(c, &format!("{n} {tag} env vol-1"), &vol, -1, 4);
        let moves = [tk, nt(&lk), nt(&sl), pace];
        for (dir, v) in [(true, up), (false, down)] {
            let gates = with(&moves, &[lit(&nrx2[3], dir)]);
            let smp = if n == "4" {
                let bit = g("lfsr")[0].clone();
                (0..4).map(|b| all(c, &format!("4 {tag} env smp{dir}.{b}"), &[bit.clone(), v[b].clone()])).collect()
            } else {
                duty_level(c, &format!("{n} {tag} env smp{dir}"), &g(key(n, "duty")), &g(key(n, "idx")), &v)
            };
            let smp_gates = if n == "4" { gates.clone() } else { with(&gates, &[nt(&g1(key(n, "sup")))]) };
            cs.put(key(n, "vol"), gates, v);
            cs.put(key(n, "smp"), smp_gates, smp);
        }
        out.insert(key(n, "vol"), locks);
    }
    out.into_iter().map(|(k, v)| (&k[3..], v)).collect()
}

/// The envelope steps a fall at double speed left waiting, at the first M-cycle of the
/// segment after it (`GB_apu_delayed_envelope_tick`), and the waiting count's move: 2,
/// the fall's own M-cycle, becomes 1 once a cycle runs.
pub(super) fn waiting(c: &mut Circuit, s: &St, e: &Ev) -> St {
    let pend = s["pend"].clone();
    let mut cs = Cases::new();
    let due = all(c, "pend due", &with(&[e.dpos.clone()], &is(&pend, 1)));
    let fires = all(c, "pend fires", &[due.clone(), s["glob"][0].clone()]);
    steps(c, s, &fires, "dl", &mut cs);
    cs.put("pend", vec![due], kbits(0, 2));
    cs.put("pend", with(&[e.dpos.clone()], &is(&pend, 2)), kbits(1, 2));
    cs.finish(c, "_w", s)
}
