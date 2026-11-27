//! The segment's carried state as named bit fields packed into bytes, one table shared
//! by the circuit (which slices its input bytes by it) and the host side (which packs
//! the reference APU into it and names the field that differs). The fields are
//! `machine::apu::Apu`'s, SameBoy's APU state, minus what it derives.

use crate::machine::apu::{Apu, EnvelopeClock, NOISE, SQUARE_1, SQUARE_2, WAVE};
use std::collections::BTreeMap;

/// Every carried field and its width, packed end to end: a field may straddle bytes, so
/// the state is as few input slots as its bits allow.
pub const FIELDS: &[(&str, usize)] = &[
    // The fields most passes change, together, so a pass changes few state bytes.
    ("c1", 13),
    ("c2", 13),
    ("c3", 12),
    ("cc4", 6),
    ("ctr4", 15),
    ("idx1", 3),
    ("idx2", 3),
    ("pos3", 5),
    ("lfsr", 15),
    ("lfdiv", 1),
    ("align", 2),
    // The sample clock: real time since the last 48 kHz sample, 750·sq + sr in units of
    // 2^-14 of a sample's period (`circuit::sample_clock`).
    ("sq", 5),
    ("sr", 10),
    ("smp1", 4),
    ("smp2", 4),
    ("smp3", 4),
    ("smp4", 4),
    ("byte3", 8),
    ("jr1", 1),
    ("jr2", 1),
    ("jr3", 1),
    ("dly1", 3),
    ("dly2", 3),
    ("dt1", 1),
    ("dt2", 1),
    ("sup1", 1),
    ("sup2", 1),
    ("rld4", 1),
    ("dst4", 1),
    ("cur4", 1),
    ("b7", 1),
    ("sin", 1),
    ("bug3", 2),
    ("hold", 3),
    ("calc", 3),
    ("rtm", 2),
    ("inst", 1),
    // The rest, globals first, then by channel.
    ("glob", 1),
    ("cgb", 1),
    ("divb", 1),
    ("div", 3),
    ("skip", 2),
    ("pend", 2),
    ("nr50", 8),
    ("nr51", 8),
    ("act1", 1),
    ("act2", 1),
    ("act3", 1),
    ("act4", 1),
    ("nr10", 7),
    ("swcd", 3),
    ("add", 11),
    ("shd", 11),
    ("unsh", 1),
    ("cmp", 11),
    ("pl1", 7),
    ("vol1", 4),
    ("vcd1", 3),
    ("len1", 11),
    ("le1", 1),
    ("env1", 3),
    ("duty1", 2),
    ("nr12", 8),
    ("x4b1", 1),
    ("x4l1", 1),
    ("pl2", 7),
    ("vol2", 4),
    ("vcd2", 3),
    ("len2", 11),
    ("le2", 1),
    ("env2", 3),
    ("duty2", 2),
    ("nr22", 8),
    ("x4b2", 1),
    ("x4l2", 1),
    ("en3", 1),
    ("pl3", 9),
    ("lvl3", 2),
    ("len3", 11),
    ("le3", 1),
    ("pul3", 1),
    ("x4b3", 1),
    ("ram0", 8),
    ("ram1", 8),
    ("ram2", 8),
    ("ram3", 8),
    ("ram4", 8),
    ("ram5", 8),
    ("ram6", 8),
    ("ram7", 8),
    ("ram8", 8),
    ("ram9", 8),
    ("ram10", 8),
    ("ram11", 8),
    ("ram12", 8),
    ("ram13", 8),
    ("ram14", 8),
    ("ram15", 8),
    ("pl4", 7),
    ("vol4", 4),
    ("vcd4", 3),
    ("le4", 1),
    ("env4", 3),
    ("nr42", 8),
    ("nr43", 8),
    ("x4b4", 1),
    ("cact", 1),
    ("bact", 1),
    ("ddis", 1),
];

/// (field, first bit, width) with bit = 8·byte + offset.
pub fn placement() -> (Vec<(&'static str, usize, usize)>, usize) {
    let mut out = Vec::new();
    let mut bit = 0usize;
    for &(name, width) in FIELDS {
        out.push((name, bit, width));
        bit += width;
    }
    (out, bit.div_ceil(8))
}

/// The carried state's bytes, `placement().1` of them.
pub const BYTES: usize = 63;
pub type State = [u8; BYTES];

pub fn pack(values: &BTreeMap<&str, u64>) -> State {
    let (place, n) = placement();
    assert_eq!(n, BYTES);
    let mut bytes = [0u8; BYTES];
    for (name, at, width) in place {
        let v = values[name];
        assert!(v < 1 << width, "{name} = {v} does not fit {width} bits");
        for b in 0..width {
            if v >> b & 1 == 1 {
                bytes[(at + b) / 8] |= 1 << ((at + b) % 8);
            }
        }
    }
    bytes
}

pub fn unpack(bytes: &[u8]) -> BTreeMap<&'static str, u64> {
    let (place, _) = placement();
    place.into_iter().map(|(name, at, width)| (name, (0..width).fold(0u64, |v, b| v | ((bytes[(at + b) / 8] >> ((at + b) % 8) & 1) as u64) << b))).collect()
}

/// An envelope clock as bits: locked, clock, should_lock.
pub fn env_bits(e: &EnvelopeClock) -> u64 {
    e.locked as u64 | (e.clock as u64) << 1 | (e.should_lock as u64) << 2
}

/// A square's countdown: SameBoy's 0xFFFF after power-on, which only the trigger's bit-10
/// test reads, is 0x1FFF here, which that test reads the same way.
fn countdown(c: u16) -> u64 {
    if c == 0xFFFF { 0x1FFF } else { c as u64 }
}

/// The reference APU as field values.
pub fn of(a: &Apu) -> BTreeMap<&'static str, u64> {
    let mut m: BTreeMap<&'static str, u64> = BTreeMap::new();
    let b = |x: bool| x as u64;
    // The reference counts SAMPLE_RATE per T-cycle against CPU_HZ: 2^8 of its units are
    // one of the model's, an M-cycle 750 of them at normal speed and 375 at double.
    assert!(a.sample_phase % 256 == 0, "sample phase {} is not a whole unit", a.sample_phase);
    let p = (a.sample_phase / 256) as u64;
    m.insert("sq", p / 750);
    m.insert("sr", p % 750);
    let r = |i: usize| a.regs[i] as u64;
    m.insert("glob", b(a.global_enable));
    m.insert("cgb", b(a.cgb));
    m.insert("divb", b(a.div_bit));
    m.insert("div", a.div_divider as u64 & 7);
    m.insert("skip", a.skip_div_event as u64);
    m.insert("pend", a.pending_envelope_tick as u64);
    m.insert("nr50", r(0x14));
    m.insert("nr51", r(0x15));
    m.insert("lfdiv", b(a.lf_div));
    m.insert("align", a.noise.alignment as u64 & 3);
    for (ch, k) in [(SQUARE_1, "1"), (SQUARE_2, "2"), (WAVE, "3"), (NOISE, "4")] {
        m.insert(key("act", k), b(a.is_active[ch]));
        m.insert(key("smp", k), if a.is_active[ch] { a.samples[ch] as u64 } else { 0 });
    }
    m.insert("nr10", r(0x00) & 0x7F);
    m.insert("swcd", a.square_sweep_countdown as u64);
    m.insert("calc", a.square_sweep_calculate_countdown as u64);
    m.insert("rtm", a.square_sweep_calculate_countdown_reload_timer as u64);
    m.insert("add", a.sweep_length_addend as u64);
    m.insert("shd", a.shadow_sweep_sample_length as u64);
    m.insert("unsh", b(a.unshifted_sweep));
    m.insert("inst", b(a.square_sweep_instant_calculation_done));
    m.insert("hold", a.channel_1_restart_hold as u64);
    m.insert("cmp", a.channel1_completed_addend as u64);
    for (i, k, nrx1, nrx2, nrx4) in [(SQUARE_1, "1", 0x01, 0x02, 0x04), (SQUARE_2, "2", 0x06, 0x07, 0x09)] {
        let s = &a.square[i];
        m.insert(key("pl", k), s.pulse_length as u64);
        m.insert(key("vol", k), s.current_volume as u64);
        m.insert(key("vcd", k), s.volume_countdown as u64);
        m.insert(key("idx", k), s.current_sample_index as u64);
        m.insert(key("sup", k), b(s.sample_surpressed));
        m.insert(key("c", k), countdown(s.sample_countdown));
        m.insert(key("len", k), s.sample_length as u64);
        m.insert(key("le", k), b(s.length_enabled));
        m.insert(key("env", k), env_bits(&s.envelope_clock));
        m.insert(key("dly", k), s.delay as u64);
        m.insert(key("dt", k), b(s.did_tick));
        m.insert(key("jr", k), b(s.just_reloaded));
        m.insert(key("duty", k), r(nrx1) >> 6);
        m.insert(if i == SQUARE_1 { "nr12" } else { "nr22" }, r(nrx2));
        m.insert(key("x4b", k), r(nrx4) >> 6 & 1);
        m.insert(key("x4l", k), b(r(nrx4) & 7 == 7));
    }
    let w = &a.wave;
    m.insert("en3", b(w.enable));
    m.insert("pl3", w.pulse_length as u64);
    m.insert("lvl3", r(0x0C) >> 5 & 3);
    m.insert("len3", w.sample_length as u64);
    m.insert("le3", b(w.length_enabled));
    m.insert("c3", w.sample_countdown as u64);
    m.insert("pos3", w.current_sample_index as u64);
    m.insert("byte3", w.current_sample_byte as u64);
    m.insert("jr3", b(w.wave_form_just_read));
    m.insert("pul3", b(w.pulsed));
    m.insert("bug3", w.bugged_read_countdown as u64);
    m.insert("x4b3", r(0x0E) >> 6 & 1);
    for i in 0..16 {
        m.insert(RAM_KEYS[i], a.regs[0x20 + i] as u64);
    }
    let n = &a.noise;
    m.insert("pl4", n.pulse_length as u64);
    m.insert("vol4", n.current_volume as u64);
    m.insert("vcd4", n.volume_countdown as u64);
    m.insert("lfsr", n.lfsr as u64);
    m.insert("cc4", n.counter_countdown as u64);
    m.insert("ctr4", n.counter as u64);
    m.insert("le4", b(n.length_enabled));
    m.insert("cur4", b(n.current_lfsr_sample));
    m.insert("dst4", b(n.did_step_counter));
    m.insert("rld4", b(n.countdown_reloaded));
    m.insert("env4", env_bits(&n.envelope_clock));
    m.insert("nr42", r(0x11));
    m.insert("nr43", r(0x12));
    m.insert("x4b4", r(0x13) >> 6 & 1);
    m.insert("cact", b(a.noise_counter_active));
    m.insert("bact", b(a.noise_background_counter_active));
    m.insert("ddis", b(a.noise_started_with_dac_disabled));
    m.insert("b7", b(a.lfsr_bit_7_before_step));
    m.insert("sin", b(a.lfsr_stepped_in_narrow));
    m
}

const RAM_KEYS: [&str; 16] = ["ram0", "ram1", "ram2", "ram3", "ram4", "ram5", "ram6", "ram7", "ram8", "ram9", "ram10", "ram11", "ram12", "ram13", "ram14", "ram15"];

fn key(base: &str, ch: &str) -> &'static str {
    let s = format!("{base}{ch}");
    FIELDS.iter().find(|(n, _)| *n == s).unwrap_or_else(|| panic!("no field {s}")).0
}

/// The reference APU's carried state bytes.
pub fn state_of(a: &Apu) -> State {
    pack(&of(a))
}

/// The fields whose values differ, as "name model/reference".
pub fn diff(got: &[u8], want: &[u8]) -> String {
    let (got, want) = (unpack(got), unpack(want));
    FIELDS.iter().filter(|(n, _)| got[n] != want[n]).map(|(n, _)| format!("{n} {}/{}", got[n], want[n])).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_fields_fill_the_state_bytes() {
        assert_eq!(super::placement().1, super::BYTES);
    }
}
