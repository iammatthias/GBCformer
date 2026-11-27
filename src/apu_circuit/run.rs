//! `n` APU cycles of a segment (`Apu::run`, SameBoy's `GB_apu_run`), at most 44, as
//! closed forms: each countdown's wraps are threshold tests on `n`, and the state after
//! them sums the reloads.

use super::build::*;
use super::circuit::*;
use super::lfsr::{gated_parity, Affine};
use super::write::{key, ram_at, wave_level};
use crate::circuit::{Circuit, Lin, Step};

/// The most `n − countdown − 1` reaches.
const DMAX: i64 = MAX_CYCLES - 1;

/// A countdown `c` that reloads `R = P − 1` with `P = per·(2048 − len)`: it wraps at the
/// cycles `c + 1 + j·P ≤ n`. Returns the first wrap and the later ones. Wrap j ≥ 1 needs
/// `j·P ≤ DMAX`, so `2048 − len ≤ q = DMAX / (j·per)`; then len's bits from `nb` (the
/// least with 2^nb ≥ q) up are all 1 and `2048 − len = 2^nb − (len mod 2^nb)`, which
/// keeps every weight small. A clear high bit of len, or a set bit of c from 6 up,
/// subtracts enough to make the test fail, as it does exactly.
fn wraps(c: &mut Circuit, name: &str, cnt: &[Lin], len: &[Lin], per: i64, n: &[Lin], gate: &Lin) -> (Lin, Vec<Lin>) {
    const BIG: i64 = 64;
    let nl = val(n);
    let mut l0 = nl.clone() - 1 + (gate.clone() - 1) * BIG;
    for (b, x) in cnt.iter().enumerate() {
        l0 = l0 - x.clone() * if b < 6 { 1i64 << b } else { BIG };
    }
    let w0 = c.ge(&format!("{name} wrap0"), l0, 0);
    let mut later = Vec::new();
    for j in 1..=DMAX / per {
        let q = DMAX / (j * per);
        let nb = (0..12).find(|&b| 1i64 << b >= q).unwrap();
        let mut lin = nl.clone() - 1 + (gate.clone() - 1) * BIG - j * per * (1i64 << nb);
        for (b, x) in cnt.iter().enumerate() {
            lin = lin - x.clone() * if b < 6 { 1i64 << b } else { BIG };
        }
        for (b, x) in len.iter().enumerate() {
            lin = if b < nb { lin + x.clone() * (j * per << b) } else { lin - nt(x) * BIG };
        }
        later.push(c.ge(&format!("{name} wrap{j}"), lin, 0));
    }
    (w0, later)
}

/// The countdown after the cycles, `c − n + k·P` while `gate`, where the first wrap adds
/// P (one AND per period bit) and the m later wraps add `m·P`, a narrow product since
/// they need `P ≤ DMAX`. Also returns `c' − P + 1`, zero exactly when the last wrap fell
/// on the last cycle (the countdown just reloaded).
fn after(c: &mut Circuit, name: &str, cnt: &[Lin], n: &[Lin], gate: &Lin, w0: &Lin, later: &[Lin], period: &[Lin]) -> (W, Lin) {
    let w0p: W = period.iter().enumerate().map(|(b, x)| all(c, &format!("{name} w0P{b}"), &[w0.clone(), x.clone()])).collect();
    let small = 6;
    let m = sum_mod(c, &format!("{name} m"), &later.iter().map(|x| (x.clone(), 1)).collect::<Vec<_>>(), 0, small, None);
    let mut mp_terms = Vec::new();
    for (a, mb) in m.iter().enumerate() {
        for (b, p) in period[..small].iter().enumerate() {
            if a + b < small {
                mp_terms.push((all(c, &format!("{name} m{a}P{b}"), &[mb.clone(), p.clone()]), 1i64 << (a + b)));
            }
        }
    }
    let mp = sum_mod(c, &format!("{name} mP"), &mp_terms, 0, small, None);
    let ng: W = n.iter().enumerate().map(|(b, x)| all(c, &format!("{name} n.{b}"), &[x.clone(), gate.clone()])).collect();
    let mut terms: Vec<(Lin, i64)> = bits_w(cnt);
    terms.extend(neg_w(&ng));
    terms.extend(bits_w(&w0p));
    terms.extend(bits_w(&mp));
    let past = terms.iter().fold(Lin::konst(1), |a, (x, w)| a + x.clone() * *w) - val(period);
    (sum_blocked(c, &format!("{name} c'"), &terms, 0, cnt.len(), None), past)
}

/// `per·(2048 − len)` as bits.
fn period(c: &mut Circuit, name: &str, len: &[Lin], per: i64, bits: usize) -> W {
    sum_mod(c, name, &neg_w(len).into_iter().map(|(x, w)| (x, per * w)).collect::<Vec<_>>(), 2048 * per, bits, None)
}

pub(super) fn phase(c: &mut Circuit, s: &St, e: &Ev) -> St {
    let g = |n: &str| s[n].clone();
    let g1 = |n: &str| s[n][0].clone();
    let n = e.n.clone();
    let lf = g1("lfdiv");
    let mut cs = Cases::new();

    // The 1 MHz phase and the noise channel's 2-bit alignment.
    let lf1 = c.eq("lf'", lf.clone() + n[0].clone(), 1);
    cs.put("lfdiv", vec![e.dpos.clone()], vec![lf1]);
    let al: W = sum_mod(c, "align+n", &bits_w(&g("align")).into_iter().chain(bits_w(&n[..2])).collect::<Vec<_>>(), 0, 2, None);
    cs.put("align", vec![e.dpos.clone()], al);

    // A wave read the channel left pending lands after 1 or 2 cycles.
    let (bug, pos, act3) = (g("bug3"), g("pos3"), g1("act3"));
    // The countdowns below are idle most passes; each reads a copy of the cycles gated by
    // its own being nonzero, so it holds still while idle.
    let bnz = nz(c, "bug3>0", &bug);
    let nb = gated(c, "n@bug", &n, &bnz);
    let n2 = c.ge("n>=2", val(&nb), 2);
    let fires = c.node("bugged read", vec![Step { coef: 1, lin: e.dpos.clone() + bug[0].clone() + nt(&bug[1]), theta: 3 }, Step { coef: 1, lin: n2.clone() + nt(&bug[0]) + bug[1].clone(), theta: 3 }]);
    let bus = ram_at(c, "run wave bus", s, &pos[1..]);
    let byte_b: W = (0..8).map(|b| c.node(&format!("byte after bug.{b}"), vec![Step { coef: 1, lin: fires.clone() + bus[b].clone(), theta: 2 }, Step { coef: 1, lin: nt(&fires) + s["byte3"][b].clone(), theta: 2 }])).collect();
    let smp_b = wave_level(c, "smp after bug", &byte_b, &pos[0], &g("lvl3"));
    let smp3_b: W = (0..4).map(|b| c.node(&format!("smp3 after bug.{b}"), vec![Step { coef: 1, lin: fires.clone() + act3.clone() + smp_b[b].clone(), theta: 3 }, Step { coef: 1, lin: nt(&fires) + s["smp3"][b].clone(), theta: 2 }, Step { coef: 1, lin: fires.clone() + nt(&act3) + s["smp3"][b].clone(), theta: 3 }])).collect();
    let waits = all(c, "bugged read waits", &[e.dpos.clone(), nt(&n2), nt(&bug[0]), bug[1].clone()]);
    let bug_b: W = vec![c.node("bug after.0", vec![Step { coef: 1, lin: waits.clone(), theta: 1 }, Step { coef: 1, lin: nt(&fires) + nt(&waits) + bug[0].clone(), theta: 3 }]), all(c, "bug after.1", &[nt(&fires), nt(&waits), bug[1].clone()])];

    // The sweep: its reload timer, then its calculation countdown, at 1 MHz.
    let busy = c.ge("sweep busy", sum(g("rtm").into_iter().chain(g("calc"))), 1);
    let ns = gated(c, "n@sweep", &n, &busy);
    let sc = ns[1..].iter().enumerate().fold(Lin::konst(0), |a, (b, x)| a + x.clone() * (1i64 << b)) + all(c, "sweep odd", &[ns[0].clone(), lf.clone()]);
    let (r, cl, nr10) = (g("rtm"), g("calc"), g("nr10"));
    let ca = c.ge("sweep r>sc", val(&r) - sc.clone(), 1);
    let r_left = sum_mod(c, "sweep r-sc", &bits_w(&r).into_iter().chain(neg_terms(&sc)).collect::<Vec<_>>(), 0, 2, None);
    cs.put("rtm", vec![e.dpos.clone(), ca.clone()], r_left);
    cs.put("rtm", vec![e.dpos.clone(), nt(&ca)], kbits(0, 2));
    cs.put("inst", vec![e.dpos.clone(), nt(&ca)], kbits(0, 1));
    let rnz = nz(c, "rtm>0", &r);
    let fire0 = all(c, "sweep fire0", &with(&[e.dpos.clone(), nt(&ca), rnz, g1("inst")], &is(&cl, 0)));
    let shifted = c.ge("sweep shifted", sum(nr10[..3].iter().cloned()) + g1("unsh"), 1);
    let cnz = nz(c, "calc>0", &cl);
    let counting = all(c, "sweep counts", &[e.dpos.clone(), cnz, shifted]);
    let gt = c.ge("sweep calc>sc'", val(&cl) + val(&r) - sc.clone(), 1);
    let cl_left = sum_mod(c, "sweep calc-sc'", &bits_w(&cl).into_iter().chain(bits_w(&r)).chain(neg_terms(&sc)).collect::<Vec<_>>(), 0, 3, None);
    cs.put("calc", vec![counting.clone(), nt(&ca), gt.clone()], cl_left);
    cs.put("calc", vec![counting.clone(), nt(&ca), nt(&gt)], kbits(0, 3));
    let fire1 = all(c, "sweep fire1", &[counting, nt(&ca), nt(&gt)]);
    let fire = c.ge("sweep done", fire0 + fire1, 1);
    // `sweep_calculation_done`: the shadow takes the frequency (unless a restart holds
    // it), a negate inverts the addend, and an overflow without negate stops channel 1.
    // SameBoy runs the APU every M-cycle while the sweep counts down, so the calculation
    // lands at the M-cycle of its tick T = rtm + calc: T at normal speed, 2T − lf_div at
    // double (a tick every second M-cycle). The restart hold and channel 1's countdown
    // see the APU cycles before it, `n1 = 2T − 2 + dbl·(1 − lf_div)`.
    let hold = g("hold");
    let dlf = all(c, "dbl lf", &[e.dbl.clone(), lf.clone()]);
    let n1_terms: Vec<(Lin, i64)> = bits_w(&r).into_iter().chain(bits_w(&cl)).map(|(x, w)| (x, 2 * w)).chain([(e.dbl.clone(), 1), (dlf, -1)]).collect();
    let n1 = sum_mod(c, "sweep n1", &n1_terms, -2, 6, None);
    let hz = c.ge("hold at the calculation", val(&n1) - val(&hold), 0);
    let n3 = nr10[3].clone();
    let (add, shd, len1) = (g("add"), g("shd"), g("len1"));
    cs.put("shd", vec![fire.clone(), hz.clone()], len1.clone());
    let inv: W = add.iter().map(nt).collect();
    cs.put("add", vec![fire.clone(), n3.clone()], inv.clone());
    cs.put("cmp", vec![fire.clone(), n3.clone()], inv);
    cs.put("cmp", vec![fire.clone(), nt(&n3)], add.clone());
    let ovf = c.node("sweep overflow", vec![Step { coef: 1, lin: (val(&len1) + val(&add)) * 1 + (fire.clone() + nt(&n3) + hz.clone() - 3) * 4096, theta: 2048 }, Step { coef: 1, lin: (val(&shd) + val(&add)) * 1 + (fire.clone() + nt(&n3) + nt(&hz) - 3) * 4096, theta: 2048 }]);
    cs.put("act1", vec![ovf.clone()], kbits(0, 1));
    // The restart hold counts down.
    let hnz = nz(c, "hold>0", &hold);
    let nh = gated(c, "n@hold", &n, &hnz);
    let hold_more = c.ge("hold>n", val(&hold) - val(&nh), 1);
    let hold_left = sum_mod(c, "hold-n", &bits_w(&hold).into_iter().chain(neg_w(&nh)).collect::<Vec<_>>(), 0, 3, None);
    cs.put("hold", vec![hold_more.clone()], hold_left);
    cs.put("hold", vec![e.dpos.clone(), nt(&hold_more)], kbits(0, 3));

    // The squares, each counting while it plays (channel 1 unless the sweep just
    // stopped it).
    for sq in ["1", "2"] {
        // Channel 1 stopped by the sweep in the segment's first M-cycle never ran.
        let ran = if sq == "1" { nt(&all(c, "sq1 stopped at once", &with(&[ovf.clone()], &is(&n1, 0)))) } else { k(1) };
        let gate = all(c, &format!("{sq} runs"), &[g1(key(sq, "act")), e.dpos.clone(), ran]);
        // Channel 1 stopped by the sweep ran only the cycles before the calculation.
        let n: W = if sq == "1" { (0..6).map(|b| c.node(&format!("sq1 n.{b}"), vec![Step { coef: 1, lin: ovf.clone() + n1[b].clone(), theta: 2 }, Step { coef: 1, lin: nt(&ovf) + n[b].clone(), theta: 2 }])).collect() } else { n.clone() };
        let (cnt, len, dly) = (g(key(sq, "c")), g(key(sq, "len")), g(key(sq, "dly")));
        let dnz = nz(c, &format!("{sq} dly>0"), &dly);
        let nd = gated(c, &format!("{sq} n@dly"), &n, &dnz);
        let more = c.ge(&format!("{sq} dly>n"), val(&dly) - val(&nd), 1);
        let dly_left = sum_mod(c, &format!("{sq} dly-n"), &bits_w(&dly).into_iter().chain(neg_w(&nd)).collect::<Vec<_>>(), 0, 3, None);
        cs.put(key(sq, "dly"), vec![gate.clone(), more.clone()], dly_left);
        cs.put(key(sq, "dly"), vec![gate.clone(), nt(&more)], kbits(0, 3));
        let (w0, later) = wraps(c, &format!("sq{sq}"), &cnt, &len, 2, &n, &gate);
        let p = period(c, &format!("sq{sq} P"), &len, 2, 13);
        let (c1, past) = after(c, &format!("sq{sq}"), &cnt, &n, &gate, &w0, &later, &p);
        cs.put(key(sq, "c"), vec![gate.clone()], c1);
        let ks: Vec<(Lin, i64)> = std::iter::once((w0.clone(), 1)).chain(later.iter().map(|x| (x.clone(), 1))).collect();
        let idx1 = sum_mod(c, &format!("sq{sq} idx'"), &bits_w(&g(key(sq, "idx"))).into_iter().chain(ks).collect::<Vec<_>>(), 0, 3, None);
        cs.put(key(sq, "idx"), vec![w0.clone()], idx1.clone());
        cs.put(key(sq, "sup"), vec![w0.clone()], kbits(0, 1));
        cs.put(key(sq, "dt"), vec![w0.clone()], kbits(1, 1));
        let reloaded = c.eq(&format!("sq{sq} just reloaded"), past, 0);
        let jr = all(c, &format!("sq{sq} jr'"), &[w0.clone(), reloaded]);
        cs.put(key(sq, "jr"), vec![gate], vec![jr]);
        let smp = duty_level(c, &format!("sq{sq} smp'"), &g(key(sq, "duty")), &idx1, &g(key(sq, "vol")));
        cs.put(key(sq, "smp"), vec![w0], smp);
    }

    // The wave channel: playing, it steps its position and reads wave RAM; stopped by its
    // length with the DAC on, it still counts and reads the bus.
    {
        let (en, pul, cnt, len) = (g1("en3"), g1("pul3"), g("c3"), g("len3"));
        let ghost = all(c, "wave ghost", &[nt(&act3), en, pul, e.dpos.clone()]);
        let playing = all(c, "wave plays", &[act3.clone(), e.dpos.clone()]);
        let gate = c.ge("wave counts", playing.clone() + ghost.clone(), 1);
        let (w0, later) = wraps(c, "wave", &cnt, &len, 1, &n, &gate);
        let p = period(c, "wave P", &len, 1, 12);
        let (c1, past) = after(c, "wave", &cnt, &n, &gate, &w0, &later, &p);
        cs.put("c3", vec![gate.clone()], c1.clone());
        let reloaded = c.eq("wave just reloaded", past, 0);
        let ks: Vec<(Lin, i64)> = std::iter::once((w0.clone(), 1)).chain(later.iter().map(|x| (x.clone(), 1))).collect();
        let pos1 = sum_mod(c, "pos3'", &bits_w(&pos).into_iter().chain(ks).collect::<Vec<_>>(), 0, 5, None);
        let steps = all(c, "wave steps", &[playing.clone(), w0.clone()]);
        cs.put("pos3", vec![steps.clone()], pos1.clone());
        let byte1 = ram_at(c, "wave byte'", s, &pos1[1..]);
        let smp1 = wave_level(c, "wave smp'", &byte1, &pos1[0], &g("lvl3"));
        cs.put("byte3", vec![steps.clone()], byte1);
        cs.put("smp3", vec![steps.clone()], smp1);
        cs.put("byte3", vec![nt(&steps), fires.clone()], byte_b.clone());
        cs.put("smp3", vec![nt(&steps), fires.clone()], smp3_b);
        let jr = all(c, "jr3'", &[playing, w0.clone(), reloaded.clone()]);
        cs.put("jr3", vec![e.dpos.clone()], vec![jr]);
        // The ghost's reads: every wrap but one landing on the last cycle reads the bus;
        // that one leaves a read 1 cycle on, and a countdown at 0 one 2 cycles on.
        let reads = c.node("ghost reads", vec![Step { coef: 1, lin: ghost.clone() + later.first().cloned().unwrap_or(k(0)), theta: 2 }, Step { coef: 1, lin: ghost.clone() + w0.clone() + nt(&reloaded) + later.first().map(nt).unwrap_or(k(1)), theta: 4 }]);
        cs.put("byte3", vec![reads, nt(&fires)], bus.clone());
        let zero = is(&c1, 0);
        let at0 = all(c, "ghost at 0", &with(&[ghost.clone()], &zero));
        let late = all(c, "ghost late", &[ghost.clone(), w0, reloaded, nt(&at0)]);
        cs.put("bug3", vec![at0.clone()], vec![k(0), k(1)]);
        cs.put("bug3", vec![late.clone()], vec![k(1), k(0)]);
        let settled = c.node("bug settled", vec![Step { coef: 1, lin: e.dpos.clone() - at0 - late, theta: 1 }]);
        cs.put("bug3", vec![settled], bug_b);
    }

    noise(c, s, e, &mut cs);
    cs.finish(c, "_r", s)
}

/// `−sc` as weighted terms for a sum: `sc` is a form over 0/1 literals with positive
/// weights.
fn neg_terms(sc: &Lin) -> Vec<(Lin, i64)> {
    sc.terms.iter().map(|(&x, &w)| (Lin::from(x), -w)).collect::<Vec<_>>().into_iter().chain((sc.c != 0).then(|| (Lin::konst(1), -sc.c))).collect()
}

/// The noise channel: its counter counts every `d` cycles while either counter runs, and
/// each rise of the bit NR43 selects steps the LFSR while the channel plays.
fn noise(c: &mut Circuit, s: &St, e: &Ev, cs: &mut Cases) {
    let g = |n: &str| s[n].clone();
    let g1 = |n: &str| s[n][0].clone();
    let n = e.n.clone();
    let nl = val(&n);
    let (cc, ctr, nr43) = (g("cc4"), g("ctr4"), g("nr43"));
    let counting = c.node("noise counts", vec![Step { coef: 1, lin: e.dpos.clone() + g1("cact"), theta: 2 }, Step { coef: 1, lin: e.dpos.clone() + nt(&g1("cact")) + g1("bact"), theta: 3 }]);
    let div = nr43[..3].to_vec();
    let dz = eqk(c, "noise div=0", &div, 0);
    // d = 4·div, or 2 for div 0.
    let d_lin = val(&div) * 4 + dz.clone() * 2;
    let w0 = c.ge("noise wrap0", nl.clone() - val(&cc) + (counting.clone() - 1) * 128, 0);
    let later: Vec<Lin> = (1..=DMAX / 2).map(|j| c.ge(&format!("noise wrap{j}"), nl.clone() - val(&cc) - d_lin.clone() * j + (counting.clone() - 1) * 1024, 0)).collect();
    let kl = later.iter().fold(w0.clone(), |a, x| a + x.clone());
    let dbits: W = vec![k(0), dz.clone(), div[0].clone(), div[1].clone(), div[2].clone()];
    let w0d: W = dbits.iter().enumerate().map(|(b, x)| all(c, &format!("noise w0d{b}"), &[w0.clone(), x.clone()])).collect();
    let m = sum_mod(c, "noise m", &later.iter().map(|x| (x.clone(), 1)).collect::<Vec<_>>(), 0, 5, None);
    let mut md = Vec::new();
    for (a, mb) in m.iter().enumerate() {
        for (b, db) in dbits.iter().enumerate() {
            if a + b < 6 && *db != k(0) {
                md.push((all(c, &format!("noise m{a}d{b}"), &[mb.clone(), db.clone()]), 1i64 << (a + b)));
            }
        }
    }
    let mdv = sum_mod(c, "noise md", &md, 0, 6, None);
    let ng: W = n.iter().enumerate().map(|(b, x)| all(c, &format!("noise n.{b}"), &[x.clone(), counting.clone()])).collect();
    let mut terms: Vec<(Lin, i64)> = bits_w(&cc);
    terms.extend(neg_w(&ng));
    terms.extend(bits_w(&w0d));
    terms.extend(bits_w(&mdv));
    let at_d = terms.iter().fold(Lin::konst(0), |a, (x, w)| a + x.clone() * *w) - d_lin.clone();
    let cc1 = sum_mod(c, "cc4'", &terms, 0, 6, None);
    cs.put("cc4", vec![counting.clone()], cc1);
    let reloaded = c.eq("noise reloaded", at_d, 0);
    let rld = all(c, "rld4'", &[w0.clone(), reloaded]);
    cs.put("rld4", vec![counting.clone()], vec![rld]);
    cs.put("dst4", vec![w0.clone()], kbits(1, 1));
    let ks: Vec<(Lin, i64)> = std::iter::once((w0.clone(), 1)).chain(later.iter().map(|x| (x.clone(), 1))).collect();
    let ctr1: W = sum_blocked(c, "ctr4'", &bits_w(&ctr[..14]).into_iter().chain(ks).collect::<Vec<_>>(), 0, 14, None).into_iter().chain([k(0)]).collect();
    cs.put("ctr4", vec![w0.clone()], ctr1);
    // The LFSR's steps: the rises of counter bit `sh` among the k counts from `ctr`. For
    // sh ≥ 5 a count of at most 22 rises it at most once, when bits 5 to sh − 1 are set,
    // bit sh clear and the low five bits carry; for sh ≤ 4, `⌊(L + M/2 + k)/M⌋ − bit sh`
    // with M = 2^(sh+1) and L the counter's low bits below M.
    let steps_on = all(c, "lfsr steps", &[counting, g1("act4")]);
    let sh = nr43[4..].to_vec();
    let carry5 = c.ge("ctr low5 carries", val(&ctr[..5]) + kl.clone(), 32);
    let maxv = (DMAX / 2 + 1) as usize / 2 + 1;
    let hot: Vec<Lin> = (1..=maxv)
        .map(|v| {
            let mut st = Vec::new();
            let base = with(&[steps_on.clone()], &[]);
            for shv in 0..14usize {
                let mut lits = with(&base, &is(&sh, shv as u64));
                if shv >= 5 {
                    if v != 1 {
                        continue;
                    }
                    lits.push(nt(&ctr[shv]));
                    lits.extend(ctr[5..shv].iter().cloned());
                    lits.push(carry5.clone());
                    st.push(Step { coef: 1, lin: sum(lits.iter().cloned()), theta: lits.len() as i64 });
                    continue;
                }
                let mm = 1i64 << (shv + 1);
                let x = val(&ctr[..=shv]) + mm / 2 + kl.clone();
                for set in [false, true] {
                    let mut l2 = lits.clone();
                    l2.push(lit(&ctr[shv], set));
                    let lo = (v as i64 + set as i64) * mm;
                    let gsum = sum(l2.iter().cloned());
                    let gn = l2.len() as i64;
                    let big = 256;
                    st.push(Step { coef: 1, lin: x.clone() + (gsum.clone() - gn) * big, theta: lo });
                    st.push(Step { coef: -1, lin: x.clone() + (gsum - gn) * big, theta: lo + mm });
                }
            }
            c.node(&format!("lfsr steps={v}"), st)
        })
        .collect();
    let narrow = nr43[3].clone();
    let l = g("lfsr");
    let powers: Vec<[Affine; 2]> = (0..=maxv).map(|v| [Affine::power(false, v), Affine::power(true, v)]).collect();
    let any_step = c.ge("lfsr stepped", sum(hot.iter().cloned()), 1);
    let bit_of = |c: &mut Circuit, name: &str, i: usize, minus: usize| -> Lin {
        let mut st = Vec::new();
        for (vi, hv) in hot.iter().enumerate() {
            let v = vi + 1 - minus;
            for (mode, ml) in [nt(&narrow), narrow.clone()].iter().enumerate() {
                let a = &powers[v][mode];
                let bits: Vec<Lin> = (0..15).filter(|j| a.mask[i] >> j & 1 == 1).map(|j| l[j].clone()).collect();
                st.extend(gated_parity(&bits, a.konst[i], &[hv.clone(), ml.clone()]));
            }
        }
        c.node(name, st)
    };
    let l1: W = (0..15).map(|i| bit_of(c, &format!("lfsr'.{i}"), i, 0)).collect();
    cs.put("lfsr", vec![any_step.clone()], l1.clone());
    let b7 = bit_of(c, "b7'", 7, 1);
    cs.put("b7", vec![any_step.clone()], vec![b7]);
    cs.put("sin", vec![any_step.clone()], vec![narrow]);
    cs.put("cur4", vec![any_step.clone()], vec![l1[0].clone()]);
    let vol = g("vol4");
    let smp: W = (0..4).map(|b| all(c, &format!("smp4'.{b}"), &[l1[0].clone(), vol[b].clone()])).collect();
    cs.put("smp4", vec![any_step], smp);
}
