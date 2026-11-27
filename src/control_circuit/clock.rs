//! MBC3's real-time clock as part of the control circuit (`docs/control-in-weights.md`
//! §20), held to `machine::rtc::Rtc`. Its events run in place: the clock reads nothing
//! the span's ticks move. An event is `EV` bit 6 with a code in `EA`'s low three bits:
//! a write the clock's chip sees (the RAM enable, the select, the latch, a register),
//! a read of the latched register, or the wall time elapsed. The wall time comes one
//! binary digit of its seconds a pass: `EX` = k for 2^k seconds. The circuit splits 2^k
//! seconds into days (the day counter's 9 bits, and whether they pass 511), hours,
//! minutes and seconds by a table of its own and adds them into the registers with their
//! carries.

use super::circuit::is;
use crate::apu_circuit::build::*;
use crate::circuit::{Circuit, Lin};

/// `EV` bit 6: a cartridge clock event, its kind in `EA`.
pub const EV_CLOCK: u8 = 64;
/// `EA` codes of a clock event; a write's code is its address's top three bits.
pub const CLOCK_ENABLE: u8 = 0;
pub const CLOCK_SELECT: u8 = 2;
pub const CLOCK_LATCH: u8 = 3;
pub const CLOCK_WRITE: u8 = 5;
pub const CLOCK_READ: u8 = 6;
pub const CLOCK_TIME: u8 = 7;

/// The clock's state slots, packed so the input slots fill whole layer-0 heads
/// (`llama::GROUP`): `CK0` S with halt (bit 6) and the carry (bit 7), `CK1` M with the
/// day counter's bit 8 (bit 6) and RAM enabled (bit 7), `CK2` H with the select's code
/// (bits 5-7: 0 for no clock register, 1 to 5 for 08 to 0C), `CK3` DL; `CKL0` to
/// `CKL3` the latched copy in the same layout, with `CKL1` bit 7 the latch armed.
pub const STATE: [&str; 8] = ["CK0", "CK1", "CK2", "CK3", "CKL0", "CKL1", "CKL2", "CKL3"];
pub const NEXT: [&str; 8] = ["CK0'", "CK1'", "CK2'", "CK3'", "CKL0'", "CKL1'", "CKL2'", "CKL3'"];
/// The binary digits of the seconds one elapse event's `EX` names.
pub const DIGITS: u32 = 64;

pub struct Clock {
    /// The next state bytes, in `NEXT` order.
    pub next: Vec<W>,
    /// The read byte's cases for `RV`'s mux.
    pub reads: Vec<(Vec<Lin>, W)>,
}

fn terms(bits: &[Lin]) -> Vec<(Lin, i64)> {
    bits.iter().enumerate().map(|(b, x)| (x.clone(), 1i64 << b)).collect()
}

fn pad(bits: W) -> W {
    let n = bits.len();
    [bits, kbits(0, 8 - n)].concat()
}

pub fn build(c: &mut Circuit, st: &[W], ev: &W, ea: &W, ex: &W) -> Clock {
    let [k0, k1, k2, k3, l0, l1, l2, l3]: [W; 8] = st.to_vec().try_into().unwrap();
    let (s, halt, carry) = (k0[..6].to_vec(), k0[6].clone(), k0[7].clone());
    let (m, d8, en) = (k1[..6].to_vec(), k1[6].clone(), k1[7].clone());
    let (h, select) = (k2[..5].to_vec(), k2[5..].to_vec());
    let (dl, armed) = (k3.clone(), l1[7].clone());
    // The latched registers as RTC S, M, H, DL and DH read them.
    let latched: [W; 5] = [pad(l0[..6].to_vec()), pad(l1[..6].to_vec()), pad(l2[..5].to_vec()), l3.clone(), [vec![l1[6].clone()], kbits(0, 5), l0[6..].to_vec()].concat()];
    let code = |c: &mut Circuit, name: &str, k: u8, more: &[Lin]| all(c, name, &[vec![ev[6].clone()], is(&ea[..3], k as u64), more.to_vec()].concat());
    let w_en = code(c, "clock enable", CLOCK_ENABLE, &[]);
    let w_sel = code(c, "clock select", CLOCK_SELECT, &[]);
    let w_latch = code(c, "clock latch", CLOCK_LATCH, &[]);
    let w_reg = code(c, "clock write", CLOCK_WRITE, &[en.clone()]);
    let r_clk = code(c, "clock read", CLOCK_READ, &[]);
    let t_add = code(c, "clock elapse", CLOCK_TIME, &[nt(&halt)]);
    let reg: Vec<Lin> = (0..5).map(|i| all(c, &format!("clock register {i}"), &is(&select, i as u64 + 1))).collect();
    // A select write's code: the register 08 to 0C it names, from 1, or 0.
    let named: Vec<Lin> = (0..5).map(|i| all(c, &format!("clock names {i}"), &is(ex, 8 + i as u64))).collect();
    let code_n: W = (0..3).map(|b| sum((0..5).filter(|i| (i + 1) >> b & 1 == 1).map(|i| named[i].clone()))).collect();
    let ex0 = all(c, "clock ex=0", &is(ex, 0));
    let ex1 = all(c, "clock ex=1", &is(ex, 1));
    let en_v = all(c, "clock enable value", &is(&ex[..4], 0x0A));

    // A register write reduces S and M modulo 60 and H modulo 24.
    let reduce = |c: &mut Circuit, name: &str, modulus: i64, n: usize| {
        let mut t = terms(ex);
        for j in 1..=255 / modulus {
            t.push((c.ge(&format!("{name}>={j}"), val(ex), modulus * j), -modulus));
        }
        sum_mod(c, name, &t, 0, n, None)
    };
    let w60 = reduce(c, "clock ex%60", 60, 6);
    let w24 = reduce(c, "clock ex%24", 24, 5);

    // 2^k seconds for the k in `EX`, split by a table: each bit of its seconds, minutes,
    // hours and days below 512 the OR of the k it is set for, and whether the days pass 511.
    let digit: Vec<Lin> = (0..DIGITS as u64).map(|k| all(c, &format!("clock 2^{k} s"), &is(&ex[..6], k))).collect();
    let split = |k: u32| {
        let secs = 1u128 << k;
        [secs % 60, secs / 60 % 60, secs / 3600 % 24, secs / 86400 % 512, (secs / 86400 >= 512) as u128]
    };
    let part = |c: &mut Circuit, name: &str, i: usize, bits: usize| -> W {
        (0..bits).map(|b| {
            let set: Vec<Lin> = (0..DIGITS).filter(|&k| split(k)[i] >> b & 1 == 1).map(|k| digit[k as usize].clone()).collect();
            if set.is_empty() { k(0) } else { c.or(&format!("{name}.{b}"), &set) }
        }).collect()
    };
    let (es, em, eh, ed) = (part(c, "clock 2^k S", 0, 6), part(c, "clock 2^k M", 1, 6), part(c, "clock 2^k H", 2, 5), part(c, "clock 2^k D", 3, 9));
    let far = part(c, "clock 2^k past 511", 4, 1).remove(0);

    // The elapsed time added in with the carries, one register a level.
    let c1 = c.ge("clock s carry", val(&s) + val(&es), 60);
    let s_t = sum_mod(c, "clock S+", &[terms(&s), terms(&es), vec![(c1.clone(), -60)]].concat(), 0, 6, None);
    let c2 = c.ge("clock m carry", val(&m) + val(&em) + c1.clone(), 60);
    let m_t = sum_mod(c, "clock M+", &[terms(&m), terms(&em), vec![(c1, 1), (c2.clone(), -60)]].concat(), 0, 6, None);
    let c3 = c.ge("clock h carry", val(&h) + val(&eh) + c2.clone(), 24);
    let h_t = sum_mod(c, "clock H+", &[terms(&h), terms(&eh), vec![(c2, 1), (c3.clone(), -24)]].concat(), 0, 5, None);
    let d9: W = [dl.clone(), vec![d8.clone()]].concat();
    let d_terms = [terms(&d9), terms(&ed), vec![(c3.clone(), 1)]].concat();
    let d_t = sum_mod(c, "clock D+", &d_terms, 0, 9, None);
    let wrap = c.ge("clock day wrap", val(&d9) + val(&ed) + c3, 512);

    let wr = |i: usize| vec![w_reg.clone(), reg[i].clone()];
    let tick = vec![t_add.clone()];
    let s_n = sel(c, "CK0'", &s, &[(wr(0), w60.clone()), (tick.clone(), s_t)]);
    let halt_n = sel(c, "CK0'.6", &[halt], &[(wr(4), vec![ex[6].clone()])]);
    let one = vec![k(1)];
    let carry_n = sel(c, "CK0'.7", &[carry], &[(wr(4), vec![ex[7].clone()]), (vec![t_add.clone(), far.clone()], one.clone()), (vec![t_add, nt(&far), wrap], one)]);
    let m_n = sel(c, "CK1'", &m, &[(wr(1), w60), (tick.clone(), m_t)]);
    let d8_n = sel(c, "CK1'.6", &[d8], &[(wr(4), vec![ex[0].clone()]), (tick.clone(), vec![d_t[8].clone()])]);
    let en_n = sel(c, "CK1'.7", &[en.clone()], &[(vec![w_en], vec![en_v])]);
    let h_n = sel(c, "CK2'", &h, &[(wr(2), w24), (tick.clone(), h_t)]);
    let select_n = sel(c, "CK2'.code", &select, &[(vec![w_sel], code_n)]);
    let dl_n = sel(c, "CK3'", &dl, &[(wr(3), ex.clone()), (tick, d_t[..8].to_vec())]);
    // A latch copies the live registers, which share the latched copy's layout.
    let latch = vec![w_latch.clone(), armed.clone(), ex1];
    let armed_n = sel(c, "CKL1'.7", &[armed], &[(vec![w_latch], vec![ex0])]);
    let next = vec![
        [s_n, halt_n, carry_n].concat(),
        [m_n, d8_n, en_n].concat(),
        [h_n, select_n].concat(),
        dl_n,
        sel(c, "CKL0'", &l0, &[(latch.clone(), k0.clone())]),
        [sel(c, "CKL1'", &l1[..7], &[(latch.clone(), k1[..7].to_vec())]), armed_n].concat(),
        pad(sel(c, "CKL2'", &l2[..5], &[(latch.clone(), k2[..5].to_vec())])),
        sel(c, "CKL3'", &l3, &[(latch, k3)]),
    ];

    // A read sees the latched register the select names while enabled, else open bus.
    let mut reads: Vec<(Vec<Lin>, W)> = (0..5).map(|i| (vec![r_clk.clone(), en.clone(), reg[i].clone()], latched[i].clone())).collect();
    let any = sum(reg.iter().cloned());
    reads.push((vec![r_clk.clone(), nt(&en)], kbits(0xFF, 8)));
    reads.push((vec![r_clk, en, nt(&any)], kbits(0xFF, 8)));
    Clock { next, reads }
}
