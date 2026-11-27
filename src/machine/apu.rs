// Ported from SameBoy v1.0.5 Core/apu.c (commit ea667ad79aa28409f034b1fd6da7304e1844750d),
// Copyright (c) 2015-2026 Lior Halphon, under the Expat licence: licenses/SameBoy-Expat.txt.

//! Audio processing unit (FF10-FF3F, and on CGB the channel outputs at FF76 and FF77):
//! two square channels (the first with a frequency sweep), the wave channel and the noise
//! channel, mixed to 48 kHz stereo samples. Behaviour follows SameBoy v1.0.5's
//! `Core/apu.c` (commit ea667ad) for CPU CGB E, the console SameSuite was written against,
//! and its DMG rules where the DMG differs (wave RAM access and corruption, length
//! counters while off). Time counts in SameBoy's APU cycles, 2 dots each: two an M-cycle at
//! normal speed, one at double speed. The chip's output is the digital mix ([`Mix`]); the
//! board's output capacitor after it is presentation ([`HighPass`]).

pub const CPU_HZ: u32 = 4_194_304;
pub const SAMPLE_RATE: u32 = 48_000;
/// Samples held between drains; more are dropped, so an undrained APU never grows.
pub const SAMPLE_CAPACITY: usize = 4096;

/// A stereo sample pair for the speakers: left, right.
pub type Sample = [i16; 2];

/// The APU chip's output for one sample instant, left then right: per side, the DACs'
/// sum over the channels NR51 routes there (each DAC -15 to 15, 0 when off) times NR50's
/// volume (1 to 8), so -480 to 480. A DAC converts its channel's digital output, the
/// value PCM12 and PCM34 read. [`HighPass`] turns it into a [`Sample`].
pub type Mix = [i16; 2];

/// The console's output capacitor, a high-pass filter after the APU chip that removes
/// the mix's DC offset. It is presentation and not part of the chip, so the host applies
/// it to every [`Mix`] whatever computes the APU, as it maps the PPU's colours to the
/// screen's RGB.
pub use crate::transport::HighPass;

/// The board's filter: a mix is at most 480, and x32 leaves the filter room to overshoot;
/// the capacitor's charge is scaled by 256.
pub const GAIN: i32 = 32;
pub const SHIFT: u8 = 8;

/// The board's filter, uncharged.
pub fn board() -> HighPass {
    HighPass { charge: [0; 2], gain: GAIN, shift: SHIFT }
}

/// Whatever computes the APU, as the machine sees it: the bus contract, the mix it
/// produced, and a catch-up before each GBC CPU read of its registers so the read sees the
/// state at its M-cycle. The reference is [`Apu`]; a compiled checkpoint implements it too.
pub trait AudioDevice: crate::device::Device + Send {
    /// Called before every bus read of `addr` (FF10-FF3F, FF76, FF77); the next
    /// `read(addr)`, before any write or advance, returns the byte the read sees.
    fn sync(&mut self, addr: u16);
    /// Appends the mixes produced since the last call. A device on another thread may
    /// still be mixing the latest M-cycles; [`AudioDevice::settle`] waits for them.
    fn take_samples(&mut self, out: &mut Vec<Mix>);
    /// Returns once every event the machine sent has run.
    fn settle(&mut self) {}
    /// What computes the samples, for the run's label.
    fn describe(&self) -> Vec<String> {
        Vec::new()
    }
    /// Counters after a run.
    fn report(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// SameBoy's duty table, `DUTIES[index + 8·duty]`.
pub const DUTIES: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 1, //
    1, 0, 0, 0, 0, 0, 0, 1, //
    1, 0, 0, 0, 0, 1, 1, 1, //
    0, 1, 1, 1, 1, 1, 1, 0,
];

pub const SQUARE_1: usize = 0;
pub const SQUARE_2: usize = 1;
pub const WAVE: usize = 2;
pub const NOISE: usize = 3;

/// The envelope's clock (`GB_envelope_clock_t`): the volume steps at a frame-sequencer
/// event while `clock` is set, and `locked` stops it once it reached 0 or 15.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvelopeClock {
    pub locked: bool,
    pub clock: bool,
    pub should_lock: bool,
}

impl EnvelopeClock {
    fn set(&mut self, value: bool, direction: bool, volume: u8) {
        if self.clock == value {
            return;
        }
        if value {
            self.clock = true;
            self.should_lock = (volume == 0xF && direction) || (volume == 0 && !direction);
        } else {
            self.clock = false;
            self.locked |= self.should_lock;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Square {
    /// Length clocks left, reloaded from NRx1 as 64 − n.
    pub pulse_length: u8,
    pub current_volume: u8,
    pub volume_countdown: u8,
    pub current_sample_index: u8,
    /// No duty step since the channel started: its output stays 0.
    pub sample_surpressed: bool,
    /// APU cycles to the next duty step, reloaded with `(sample_length ^ 0x7FF)·2 + 1`.
    pub sample_countdown: u16,
    /// NRx3 and NRx4's frequency bits.
    pub sample_length: u16,
    pub length_enabled: bool,
    pub envelope_clock: EnvelopeClock,
    /// SameBoy's start delay, in APU cycles (`delay`).
    pub delay: u8,
    pub did_tick: bool,
    /// The countdown reloaded in the last APU cycle run.
    pub just_reloaded: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Wave {
    /// NR30's DAC.
    pub enable: bool,
    pub pulse_length: u16,
    /// NR32 as a right shift: 4 (mute), 0, 1, 2.
    pub shift: u8,
    pub sample_length: u16,
    pub length_enabled: bool,
    pub sample_countdown: u16,
    pub current_sample_index: u8,
    pub current_sample_byte: u8,
    /// The channel read wave RAM in the last APU cycle run.
    pub wave_form_just_read: bool,
    /// Triggered since its DAC came on.
    pub pulsed: bool,
    pub bugged_read_countdown: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Noise {
    pub pulse_length: u8,
    pub current_volume: u8,
    pub volume_countdown: u8,
    pub lfsr: u16,
    pub narrow: bool,
    /// APU cycles to the next count of `counter`, reloaded from NR43's divisor.
    pub counter_countdown: u8,
    /// A 14-bit counter whose bit NR43 bits 4-7 select steps the LFSR when it rises.
    pub counter: u16,
    pub length_enabled: bool,
    /// APU cycles run, modulo 4.
    pub alignment: u8,
    pub current_lfsr_sample: bool,
    pub did_step_counter: bool,
    pub countdown_reloaded: bool,
    pub envelope_clock: EnvelopeClock,
}

/// `skip_div_event`: turning the APU on while the frame sequencer's DIV bit is set skips
/// the first event.
pub const SKIP_INACTIVE: u8 = 0;
pub const SKIP_SKIPPED: u8 = 1;
pub const SKIP_SKIP: u8 = 2;

/// Every register's read mask from FF10, SameBoy's `read_mask`.
const READ_MASK: [u8; 0x20] = [
    0x80, 0x3F, 0x00, 0xFF, 0xBF, 0xFF, 0x3F, 0x00, 0xFF, 0xBF, 0x7F, 0xFF, 0x9F, 0xFF, 0xBF, 0xFF, //
    0xFF, 0x00, 0x00, 0xBF, 0x00, 0x00, 0x70, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

const NR10: usize = 0x00;
const NR11: usize = 0x01;
const NR12: usize = 0x02;
const NR14: usize = 0x04;
const NR21: usize = 0x06;
const NR22: usize = 0x07;
const NR24: usize = 0x09;
const NR30: usize = 0x0A;
const NR42: usize = 0x11;
const NR43: usize = 0x12;
const NR44: usize = 0x13;
const NR50: usize = 0x14;
const NR51: usize = 0x15;
const NR52: usize = 0x16;
const WAV: usize = 0x20;

#[derive(Clone)]
pub struct Apu {
    pub cgb: bool,
    /// FF10-FF3F as last written (SameBoy's `io_registers`), wave RAM from index 0x20.
    pub regs: [u8; 0x30],
    pub global_enable: bool,
    /// Each channel's latched digital output (`samples`), which PCM12 and PCM34 read
    /// while the channel is active.
    pub samples: [u8; 4],
    pub is_active: [bool; 4],
    pub div_divider: u8,
    /// The APU runs at 2 MHz and the squares at 1 MHz; this is the 1 MHz phase.
    pub lf_div: bool,
    pub square_sweep_countdown: u8,
    pub square_sweep_calculate_countdown: u8,
    pub square_sweep_calculate_countdown_reload_timer: u8,
    pub sweep_length_addend: u16,
    pub shadow_sweep_sample_length: u16,
    pub unshifted_sweep: bool,
    pub square_sweep_instant_calculation_done: bool,
    pub channel_1_restart_hold: u8,
    pub channel1_completed_addend: u16,
    pub square: [Square; 2],
    pub wave: Wave,
    pub noise: Noise,
    pub skip_div_event: u8,
    /// At double speed a frame-sequencer event's envelope steps wait for the next
    /// M-cycle's start: 1 when they run there, 2 while the M-cycle of a counted fall
    /// still runs.
    pub pending_envelope_tick: u8,
    pub noise_counter_active: bool,
    pub noise_background_counter_active: bool,
    pub lfsr_stepped_in_narrow: bool,
    pub lfsr_bit_7_before_step: bool,
    pub noise_started_with_dac_disabled: bool,
    /// The speed of the last M-cycle advanced.
    pub double: bool,
    /// The frame sequencer's DIV bit (bit 4, bit 5 at double speed) as its edges left it.
    pub div_bit: bool,
    /// Sample clock: SAMPLE_RATE per T-cycle, a sample per CPU_HZ, counted per M-cycle
    /// in real time (two dots of it at double speed).
    pub sample_phase: u32,
    samples_out: Vec<Mix>,
}

/// One M-cycle's frame-sequencer inputs, as the control logic signals them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Div {
    /// The frame sequencer's DIV bit fell at this M-cycle.
    pub fall: bool,
    /// It fell because a DIV write or a STOP reset the counter (SameBoy's
    /// `during_div_write`).
    pub reset: bool,
    /// It rose at this M-cycle.
    pub rise: bool,
}

impl Div {
    /// The events a block's signals give its last M-cycle.
    pub fn of(s: &crate::device::Signals) -> Div {
        Div { fall: s.edge, reset: s.edge && s.reset, rise: s.rise }
    }
}

impl Apu {
    /// At power-on: NR52 off, until the boot ROM turns it on.
    pub fn power_on(cgb: bool) -> Self {
        Apu {
            cgb,
            regs: [0; 0x30],
            global_enable: false,
            samples: [0; 4],
            is_active: [false; 4],
            div_divider: 0,
            lf_div: false,
            square_sweep_countdown: 0,
            square_sweep_calculate_countdown: 0,
            square_sweep_calculate_countdown_reload_timer: 0,
            sweep_length_addend: 0,
            shadow_sweep_sample_length: 0,
            unshifted_sweep: false,
            square_sweep_instant_calculation_done: false,
            channel_1_restart_hold: 0,
            channel1_completed_addend: 0,
            square: [Square::default(); 2],
            wave: Wave::default(),
            noise: Noise::default(),
            skip_div_event: SKIP_INACTIVE,
            pending_envelope_tick: 0,
            noise_counter_active: false,
            noise_background_counter_active: false,
            lfsr_stepped_in_narrow: false,
            lfsr_bit_7_before_step: false,
            noise_started_with_dac_disabled: false,
            double: false,
            div_bit: false,
            sample_phase: 0,
            samples_out: Vec::with_capacity(SAMPLE_CAPACITY),
        }
    }

    /// As the boot ROM leaves it: channel 1 still on after the start-up chime, its
    /// envelope decayed to silence (NR52 reads 0xF1).
    pub fn post_boot(cgb: bool) -> Self {
        let mut apu = Self::power_on(cgb);
        for (addr, value) in [(0xFF26, 0x80), (0xFF11, 0x80), (0xFF12, 0xF3), (0xFF25, 0xF3), (0xFF24, 0x77), (0xFF13, 0xC1), (0xFF14, 0x87)] {
            apu.write(addr, value);
        }
        apu.regs[NR14] = 0x07;
        apu.square[SQUARE_1].current_volume = 0;
        apu.square[SQUARE_1].envelope_clock = EnvelopeClock::default();
        apu.square[SQUARE_1].sample_surpressed = false;
        apu.samples[SQUARE_1] = 0;
        apu
    }

    fn reg(&self, r: usize) -> u8 {
        self.regs[r]
    }

    pub fn dac_enabled(&self, ch: usize) -> bool {
        match ch {
            SQUARE_1 => self.reg(NR12) & 0xF8 != 0,
            SQUARE_2 => self.reg(NR22) & 0xF8 != 0,
            WAVE => self.wave.enable,
            _ => self.reg(NR42) & 0xF8 != 0,
        }
    }

    /// `update_sample`: a disabled DAC keeps its latched sample.
    fn update_sample(&mut self, ch: usize, value: u8) {
        if self.dac_enabled(ch) {
            self.samples[ch] = value;
        }
    }

    fn update_square_sample(&mut self, i: usize) {
        if self.square[i].sample_surpressed {
            return;
        }
        let duty = self.reg(if i == SQUARE_1 { NR11 } else { NR21 }) >> 6;
        let out = DUTIES[(self.square[i].current_sample_index + duty * 8) as usize] != 0;
        self.update_sample(i, if out { self.square[i].current_volume } else { 0 });
    }

    fn update_wave_sample(&mut self) {
        let w = &self.wave;
        let nibble = if w.current_sample_index & 1 != 0 { w.current_sample_byte & 0xF } else { w.current_sample_byte >> 4 };
        let v = nibble >> w.shift;
        self.update_sample(WAVE, v);
    }

    /// The channel's digital output as PCM12 and PCM34 read it.
    pub fn output(&self, ch: usize) -> u8 {
        if self.is_active[ch] { self.samples[ch] } else { 0 }
    }

    fn tick_square_envelope(&mut self, i: usize) {
        self.square[i].envelope_clock.set(false, false, 0);
        if self.square[i].envelope_clock.locked {
            return;
        }
        let nrx2 = self.reg(if i == SQUARE_1 { NR12 } else { NR22 });
        if nrx2 & 7 == 0 {
            return;
        }
        let s = &mut self.square[i];
        s.current_volume = if nrx2 & 8 != 0 { s.current_volume.wrapping_add(1) } else { s.current_volume.wrapping_sub(1) } & 0xF;
        if self.is_active[i] {
            self.update_square_sample(i);
        }
    }

    fn tick_noise_envelope(&mut self) {
        self.noise.envelope_clock.set(false, false, 0);
        if self.noise.envelope_clock.locked {
            return;
        }
        let nr42 = self.reg(NR42);
        if nr42 & 7 == 0 {
            return;
        }
        let n = &mut self.noise;
        n.current_volume = if nr42 & 8 != 0 { n.current_volume.wrapping_add(1) } else { n.current_volume.wrapping_sub(1) } & 0xF;
        if self.is_active[NOISE] {
            let v = if self.noise.lfsr & 1 != 0 { self.noise.current_volume } else { 0 };
            self.update_sample(NOISE, v);
        }
    }

    /// The sweep's delayed calculation and overflow check (`sweep_calculation_done`).
    fn sweep_calculation_done(&mut self) {
        if self.channel_1_restart_hold == 0 {
            self.shadow_sweep_sample_length = self.square[SQUARE_1].sample_length;
        }
        if self.reg(NR10) & 8 != 0 {
            self.sweep_length_addend ^= 0x7FF;
        }
        if self.shadow_sweep_sample_length + self.sweep_length_addend > 0x7FF && self.reg(NR10) & 8 == 0 {
            self.is_active[SQUARE_1] = false;
            self.update_sample(SQUARE_1, 0);
        }
        self.channel1_completed_addend = self.sweep_length_addend;
    }

    fn trigger_sweep_calculation(&mut self, during_div_write: bool) {
        let nr10 = self.reg(NR10);
        if nr10 & 0x70 != 0 && self.square_sweep_countdown == 7 {
            if nr10 & 7 != 0 {
                let len = self.sweep_length_addend + self.shadow_sweep_sample_length + (nr10 & 8 != 0) as u16;
                self.square[SQUARE_1].sample_length = len & 0x7FF;
            }
            if self.channel_1_restart_hold == 0 {
                self.sweep_length_addend = self.square[SQUARE_1].sample_length >> (nr10 & 7);
            }
            self.square_sweep_calculate_countdown = nr10 & 7;
            self.square_sweep_calculate_countdown_reload_timer = 1 + self.lf_div as u8;
            if !self.double && during_div_write {
                self.square_sweep_calculate_countdown_reload_timer = 1;
            }
            self.unshifted_sweep = nr10 & 7 == 0;
            self.square_sweep_countdown = ((nr10 >> 4) & 7) ^ 7;
            if self.square_sweep_calculate_countdown == 0 {
                self.square_sweep_instant_calculation_done = true;
            }
        }
    }

    fn delayed_envelope_tick(&mut self) {
        self.pending_envelope_tick = 0;
        if !self.global_enable {
            return;
        }
        self.envelope_ticks();
    }

    fn envelope_ticks(&mut self) {
        for i in [SQUARE_1, SQUARE_2] {
            if self.square[i].envelope_clock.clock {
                self.tick_square_envelope(i);
            }
        }
        if self.noise.envelope_clock.clock {
            self.tick_noise_envelope();
        }
    }

    /// The frame sequencer's DIV bit fell (`GB_apu_div_event`).
    fn div_event(&mut self, during_div_write: bool) {
        if !self.global_enable {
            return;
        }
        if self.skip_div_event == SKIP_SKIP {
            self.skip_div_event = SKIP_SKIPPED;
            return;
        }
        if self.skip_div_event == SKIP_SKIPPED {
            self.skip_div_event = SKIP_INACTIVE;
        } else {
            self.div_divider = self.div_divider.wrapping_add(1);
        }
        if self.div_divider & 7 == 7 {
            for i in [SQUARE_1, SQUARE_2] {
                if !self.square[i].envelope_clock.clock {
                    self.square[i].volume_countdown = self.square[i].volume_countdown.wrapping_sub(1) & 7;
                }
            }
            if !self.noise.envelope_clock.clock {
                self.noise.volume_countdown = self.noise.volume_countdown.wrapping_sub(1) & 7;
            }
        }
        if self.double && self.cgb {
            self.pending_envelope_tick = if during_div_write { 1 } else { 2 };
        } else {
            self.envelope_ticks();
        }
        if self.div_divider & 1 == 1 {
            for i in [SQUARE_1, SQUARE_2] {
                let s = &mut self.square[i];
                if s.length_enabled && s.pulse_length != 0 {
                    s.pulse_length -= 1;
                    if s.pulse_length == 0 {
                        self.is_active[i] = false;
                        self.update_sample(i, 0);
                    }
                }
            }
            let w = &mut self.wave;
            if w.length_enabled && w.pulse_length != 0 {
                w.pulse_length -= 1;
                if w.pulse_length == 0 {
                    self.is_active[WAVE] = false;
                    self.update_sample(WAVE, 0);
                }
            }
            let n = &mut self.noise;
            if n.length_enabled && n.pulse_length != 0 {
                n.pulse_length -= 1;
                if n.pulse_length == 0 {
                    self.is_active[NOISE] = false;
                    self.update_sample(NOISE, 0);
                }
            }
        }
        if self.div_divider & 3 == 3 {
            self.square_sweep_countdown = (self.square_sweep_countdown + 1) & 7;
            self.trigger_sweep_calculation(during_div_write);
        }
    }

    /// The frame sequencer's DIV bit rose (`GB_apu_div_secondary_event`): an envelope
    /// whose countdown ran out reloads it and arms its clock.
    fn div_secondary_event(&mut self) {
        if !self.global_enable {
            return;
        }
        for i in [SQUARE_1, SQUARE_2] {
            let nrx2 = self.reg(if i == SQUARE_1 { NR12 } else { NR22 });
            let s = &mut self.square[i];
            if self.is_active[i] && s.volume_countdown == 0 {
                s.volume_countdown = nrx2 & 7;
                s.envelope_clock.set(s.volume_countdown != 0, nrx2 & 8 != 0, s.current_volume);
            }
        }
        let nr42 = self.reg(NR42);
        let n = &mut self.noise;
        if self.is_active[NOISE] && n.volume_countdown == 0 {
            n.volume_countdown = nr42 & 7;
            n.envelope_clock.set(n.volume_countdown != 0, nr42 & 8 != 0, n.current_volume);
        }
    }

    fn update_lfsr(&mut self) {
        self.noise.current_lfsr_sample = self.noise.lfsr & 1 != 0;
        if self.is_active[NOISE] {
            let v = if self.noise.current_lfsr_sample { self.noise.current_volume } else { 0 };
            self.update_sample(NOISE, v);
        }
    }

    fn lfsr_feedback(&self) -> bool {
        (self.noise.lfsr ^ (self.noise.lfsr >> 1) ^ 1) & 1 != 0
    }

    fn step_lfsr(&mut self) {
        let high = self.lfsr_feedback();
        self.lfsr_bit_7_before_step = self.noise.lfsr & 0x80 != 0;
        let mask = if self.noise.narrow { 0x4040 } else { 0x4000 };
        self.noise.lfsr >>= 1;
        if high {
            self.noise.lfsr |= mask;
        } else {
            self.noise.lfsr &= !mask;
        }
        self.update_lfsr();
        self.lfsr_stepped_in_narrow = self.noise.narrow;
    }

    /// `cycles` APU cycles (`GB_apu_run`'s body).
    fn run(&mut self, cycles: u16) {
        if cycles == 0 {
            return;
        }
        if self.wave.bugged_read_countdown != 0 {
            for _ in 0..cycles {
                self.wave.bugged_read_countdown -= 1;
                if self.wave.bugged_read_countdown == 0 {
                    self.wave.current_sample_byte = self.wave_sample_bus();
                    if self.is_active[WAVE] {
                        self.update_wave_sample();
                    }
                    break;
                }
            }
        }
        self.lf_div ^= cycles & 1 != 0;
        self.noise.alignment = self.noise.alignment.wrapping_add(cycles as u8);
        let mut sweep_cycles = cycles / 2;
        if cycles & 1 != 0 && !self.lf_div {
            sweep_cycles += 1;
        }
        if self.square_sweep_calculate_countdown_reload_timer as u16 > sweep_cycles {
            self.square_sweep_calculate_countdown_reload_timer -= sweep_cycles as u8;
            sweep_cycles = 0;
        } else {
            if self.square_sweep_calculate_countdown_reload_timer != 0 && self.square_sweep_calculate_countdown == 0 && self.square_sweep_instant_calculation_done {
                self.sweep_calculation_done();
            }
            self.square_sweep_instant_calculation_done = false;
            sweep_cycles -= self.square_sweep_calculate_countdown_reload_timer as u16;
            self.square_sweep_calculate_countdown_reload_timer = 0;
        }
        if self.square_sweep_calculate_countdown != 0 && (self.reg(NR10) & 7 != 0 || self.unshifted_sweep) {
            if self.square_sweep_calculate_countdown as u16 > sweep_cycles {
                self.square_sweep_calculate_countdown -= sweep_cycles as u8;
            } else {
                self.square_sweep_calculate_countdown = 0;
                self.sweep_calculation_done();
            }
        }
        if self.channel_1_restart_hold != 0 {
            self.channel_1_restart_hold = self.channel_1_restart_hold.saturating_sub(cycles.min(255) as u8);
        }
        for i in [SQUARE_1, SQUARE_2] {
            if !self.is_active[i] {
                continue;
            }
            let mut left = cycles;
            let s = &mut self.square[i];
            if s.delay != 0 {
                s.delay = if (s.delay as u16) < left { 0 } else { s.delay - left as u8 };
            }
            while left > self.square[i].sample_countdown {
                let s = &mut self.square[i];
                left -= s.sample_countdown + 1;
                s.sample_countdown = (s.sample_length ^ 0x7FF) * 2 + 1;
                s.current_sample_index = (s.current_sample_index + 1) & 7;
                s.sample_surpressed = false;
                s.did_tick = true;
                self.update_square_sample(i);
            }
            let s = &mut self.square[i];
            s.just_reloaded = left == 0;
            if left != 0 {
                s.sample_countdown -= left;
            }
        }
        self.wave.wave_form_just_read = false;
        if self.is_active[WAVE] {
            let mut left = cycles;
            while left > self.wave.sample_countdown {
                let w = &mut self.wave;
                left -= w.sample_countdown + 1;
                w.sample_countdown = w.sample_length ^ 0x7FF;
                w.current_sample_index = (w.current_sample_index + 1) & 0x1F;
                w.current_sample_byte = self.regs[WAV + (w.current_sample_index as usize >> 1)];
                self.update_wave_sample();
                self.wave.wave_form_just_read = true;
            }
            if left != 0 {
                self.wave.sample_countdown -= left;
                self.wave.wave_form_just_read = false;
            }
        } else if self.wave.enable && self.wave.pulsed {
            let mut left = cycles;
            while left > self.wave.sample_countdown {
                left -= self.wave.sample_countdown + 1;
                self.wave.sample_countdown = self.wave.sample_length ^ 0x7FF;
                if left != 0 {
                    self.wave.current_sample_byte = self.wave_sample_bus();
                } else {
                    self.wave.bugged_read_countdown = 1;
                }
            }
            if left != 0 {
                self.wave.sample_countdown -= left;
            }
            if self.wave.sample_countdown == 0 {
                self.wave.bugged_read_countdown = 2;
            }
        }
        if self.noise_counter_active || self.noise_background_counter_active {
            let mut left = cycles;
            let divisor = match (self.reg(NR43) & 7) << 2 {
                0 => 2,
                d => d,
            };
            if self.noise.counter_countdown == 0 {
                self.noise.counter_countdown = divisor;
            }
            while left >= self.noise.counter_countdown as u16 {
                left -= self.noise.counter_countdown as u16;
                self.noise.counter_countdown = divisor;
                let mask = 1u32 << (self.reg(NR43) >> 4);
                let old_bit = self.noise.counter as u32 & mask != 0;
                self.noise.counter = (self.noise.counter + 1) & 0x3FFF;
                self.noise.did_step_counter = true;
                let new_bit = self.noise.counter as u32 & mask != 0;
                if new_bit && !old_bit && self.is_active[NOISE] {
                    self.step_lfsr();
                }
            }
            if left != 0 {
                self.noise.counter_countdown -= left as u8;
                self.noise.countdown_reloaded = false;
            } else {
                self.noise.countdown_reloaded = true;
            }
        }
    }

    /// The byte channel 3's read sees: the wave RAM byte its position selects. SameBoy
    /// reads it at the address bus's low nibble, which is the position's byte when the
    /// channel drives the bus.
    fn wave_sample_bus(&self) -> u8 {
        self.regs[WAV + (self.wave.current_sample_index as usize >> 1)]
    }

    /// One M-cycle: the frame sequencer's events in SameBoy's order (a reset's fall at
    /// the access before the M-cycle, then the envelope steps an event at double speed
    /// left waiting, then the counted fall or rise), the M-cycle's APU cycles, then the
    /// sample clock.
    pub fn mcycle(&mut self, double: bool, div: Div) {
        self.double = double;
        if div.fall && div.reset {
            self.div_bit = false;
            self.div_event(true);
        }
        if self.pending_envelope_tick == 1 {
            self.delayed_envelope_tick();
        }
        if div.fall && !div.reset {
            self.div_bit = false;
            self.div_event(false);
        }
        if div.rise {
            self.div_bit = true;
            self.div_secondary_event();
        }
        self.run(if double { 1 } else { 2 });
        if self.pending_envelope_tick == 2 {
            self.pending_envelope_tick = 1;
        }
        self.sample_phase += if double { 2 } else { 4 } * SAMPLE_RATE;
        if self.sample_phase >= CPU_HZ {
            self.sample_phase -= CPU_HZ;
            self.push_sample();
        }
    }

    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF76 => return if self.cgb { self.output(SQUARE_2) << 4 | self.output(SQUARE_1) } else { 0xFF },
            0xFF77 => return if self.cgb { self.output(NOISE) << 4 | self.output(WAVE) } else { 0xFF },
            0xFF26 => {
                let on = (0..4).rev().fold(0u8, |bits, ch| bits << 1 | self.is_active[ch] as u8);
                return (self.global_enable as u8) << 7 | 0x70 | on;
            }
            0xFF10..=0xFF2F => self.regs[addr as usize - 0xFF10] | READ_MASK[addr as usize - 0xFF10],
            0xFF30..=0xFF3F => match self.wave_index(addr) {
                Some(i) => self.regs[WAV + i],
                None => 0xFF,
            },
            _ => 0xFF,
        }
    }

    /// The wave RAM byte a GBC CPU access at `addr` reaches. While channel 3 plays, every
    /// address reaches the byte it is playing; on DMG only in the APU cycle channel 3 read
    /// it, and otherwise a read sees 0xFF and a write is dropped.
    fn wave_index(&self, addr: u16) -> Option<usize> {
        if !self.is_active[WAVE] {
            return Some(addr as usize & 0xF);
        }
        if !self.cgb && !self.wave.wave_form_just_read {
            return None;
        }
        Some(self.wave.current_sample_index as usize / 2)
    }

    pub fn write(&mut self, addr: u16, value: u8) {
        if !(0xFF10..=0xFF3F).contains(&addr) {
            return;
        }
        let mut reg = addr as usize - 0xFF10;
        if !self.global_enable && reg != NR52 && reg < WAV && (self.cgb || ![NR11, NR21, 0x0B, 0x10].contains(&reg)) {
            return;
        }
        let mut value = value;
        if reg >= WAV {
            match self.wave_index(addr) {
                Some(i) => reg = WAV + i,
                None => return,
            }
        }
        match reg {
            NR50 | NR51 => {}
            NR52 => {
                let old_pulse = [self.square[0].pulse_length, self.square[1].pulse_length];
                let (old_wave, old_noise) = (self.wave.pulse_length, self.noise.pulse_length);
                if value & 0x80 != 0 && !self.global_enable {
                    self.apu_init();
                    self.global_enable = true;
                } else if value & 0x80 == 0 && self.global_enable {
                    for ch in 0..4 {
                        self.update_sample(ch, 0);
                    }
                    self.power_off();
                }
                if !self.cgb && value & 0x80 != 0 {
                    self.square[0].pulse_length = old_pulse[0];
                    self.square[1].pulse_length = old_pulse[1];
                    self.wave.pulse_length = old_wave;
                    self.noise.pulse_length = old_noise;
                }
            }
            NR10 => {
                if self.square_sweep_calculate_countdown != 0 || self.square_sweep_calculate_countdown_reload_timer != 0 {
                    self.nr10_write_glitch(value);
                }
                let old_negate = self.reg(NR10) & 8 != 0;
                self.regs[NR10] = value;
                if self.shadow_sweep_sample_length + self.channel1_completed_addend + old_negate as u16 > 0x7FF && value & 8 == 0 {
                    self.is_active[SQUARE_1] = false;
                    self.update_sample(SQUARE_1, 0);
                }
                self.trigger_sweep_calculation(false);
            }
            NR11 | NR21 => {
                let i = if reg == NR21 { SQUARE_2 } else { SQUARE_1 };
                self.square[i].pulse_length = 0x40 - (value & 0x3F);
                if !self.global_enable {
                    value &= 0x3F;
                }
            }
            NR12 | NR22 => {
                let i = if reg == NR22 { SQUARE_2 } else { SQUARE_1 };
                if value & 0xF8 == 0 {
                    self.regs[reg] = value;
                    self.is_active[i] = false;
                    self.update_sample(i, 0);
                } else if self.is_active[i] {
                    let old = self.regs[reg];
                    let s = &mut self.square[i];
                    nrx2_glitch(&mut s.current_volume, value, old, &mut s.volume_countdown, &mut s.envelope_clock);
                    self.regs[reg] = value;
                    self.update_square_sample(i);
                }
            }
            0x03 | 0x08 => {
                let i = if reg == 0x08 { SQUARE_2 } else { SQUARE_1 };
                let s = &mut self.square[i];
                s.sample_length = s.sample_length & !0xFF | value as u16;
                if s.just_reloaded {
                    s.sample_countdown = (s.sample_length ^ 0x7FF) * 2 + 1;
                }
            }
            NR14 | NR24 => self.write_nrx4(reg, value),
            NR30 => {
                self.wave.enable = value & 0x80 != 0;
                if !self.wave.enable {
                    self.wave.pulsed = false;
                    if self.is_active[WAVE] && self.wave.sample_countdown == 0 {
                        // SameBoy reads the byte at the program counter's low nibble here;
                        // the byte at the position is the bus's value when the channel drives it.
                        self.wave.current_sample_byte = self.wave_sample_bus();
                    }
                    self.is_active[WAVE] = false;
                    self.update_sample(WAVE, 0);
                }
            }
            0x0B => self.wave.pulse_length = 0x100 - value as u16,
            0x0C => {
                self.wave.shift = [4, 0, 1, 2][(value as usize >> 5) & 3];
                if self.is_active[WAVE] {
                    self.update_wave_sample();
                }
            }
            0x0D => {
                self.wave.sample_length = self.wave.sample_length & !0xFF | value as u16;
                if self.wave.bugged_read_countdown == 1 {
                    self.wave.sample_countdown = self.wave.sample_length ^ 0x7FF;
                }
            }
            0x0E => self.write_nr34(value),
            0x10 => self.noise.pulse_length = 0x40 - (value & 0x3F),
            NR42 => {
                if value & 0xF8 == 0 {
                    if self.is_active[NOISE] && self.reg(NR43) & 7 != 0 {
                        if self.noise.counter_countdown <= 2 {
                            self.noise.counter = self.noise.counter.wrapping_add(1);
                        }
                        self.noise_background_counter_active = false;
                    }
                    if !self.cgb {
                        self.noise_background_counter_active = false;
                    }
                    self.regs[NR42] = value;
                    self.is_active[NOISE] = false;
                    self.update_sample(NOISE, 0);
                    self.noise_counter_active = false;
                } else if self.is_active[NOISE] {
                    let old = self.regs[NR42];
                    let n = &mut self.noise;
                    nrx2_glitch(&mut n.current_volume, value, old, &mut n.volume_countdown, &mut n.envelope_clock);
                    self.regs[NR42] = value;
                    let v = if self.noise.current_lfsr_sample { self.noise.current_volume } else { 0 };
                    self.update_sample(NOISE, v);
                }
            }
            NR43 => {
                if self.noise.countdown_reloaded {
                    let divisor = match (value & 7) << 2 {
                        0 => 2,
                        d => d,
                    };
                    self.noise.counter_countdown = divisor + if divisor == 2 { 0 } else { [2, 1, 0, 3][(self.noise.alignment & 3) as usize] };
                }
                self.nr43_write(value);
            }
            NR44 => self.write_nr44(value),
            _ => {}
        }
        self.regs[reg] = value;
    }

    /// `GB_apu_init`, at NR52's power-on.
    fn apu_init(&mut self) {
        let regs = self.regs;
        self.power_off();
        self.regs = regs;
        self.lf_div = true;
        self.wave.shift = 4;
        if self.div_bit {
            self.skip_div_event = SKIP_SKIP;
            self.div_divider = 1;
        }
        self.square[SQUARE_1].sample_countdown = 0xFFFF;
        self.square[SQUARE_2].sample_countdown = 0xFFFF;
    }

    /// Power-off clears every register and the channels' state; wave RAM survives.
    fn power_off(&mut self) {
        let (cgb, double, div_bit, sample_phase, wave_ram) = (self.cgb, self.double, self.div_bit, self.sample_phase, <[u8; 16]>::try_from(&self.regs[WAV..]).unwrap());
        let out = std::mem::take(&mut self.samples_out);
        *self = Apu { double, div_bit, sample_phase, samples_out: out, ..Apu::power_on(cgb) };
        self.regs[WAV..].copy_from_slice(&wave_ram);
    }

    fn nr10_write_glitch(&mut self, value: u8) {
        if self.square_sweep_calculate_countdown_reload_timer == 2 {
            self.square_sweep_calculate_countdown = value & 7;
            if self.square_sweep_calculate_countdown == 0 {
                self.square_sweep_calculate_countdown_reload_timer = 0;
            }
        }
        if value & 7 != 0 && self.reg(NR10) & 7 == 0 && !self.lf_div && self.square_sweep_calculate_countdown > 1 {
            self.square_sweep_calculate_countdown -= 1;
            if self.square_sweep_calculate_countdown == 0 {
                self.sweep_calculation_done();
            }
        }
    }

    fn write_nrx4(&mut self, reg: usize, value: u8) {
        let i = if reg == NR24 { SQUARE_2 } else { SQUARE_1 };
        let nrx2 = self.reg(if i == SQUARE_1 { NR12 } else { NR22 });
        let was_active = self.is_active[i];
        if value & 0x80 == 0 && self.is_active[i] && self.regs[reg] & 7 == 7 && value & 7 != 7 {
            let s = &mut self.square[i];
            if (self.cgb || s.sample_countdown & 1 != 0) && s.did_tick && s.sample_countdown >> 1 == (s.sample_length ^ 0x7FF) {
                s.current_sample_index = s.current_sample_index.wrapping_sub(1) & 7;
                s.sample_surpressed = false;
            }
        }
        let old_sample_length = self.square[i].sample_length;
        {
            let s = &mut self.square[i];
            s.sample_length = s.sample_length & 0xFF | (value as u16 & 7) << 8;
            if s.just_reloaded {
                s.sample_countdown = (s.sample_length ^ 0x7FF) * 2 + 1;
            }
        }
        if value & 0x80 != 0 {
            let (lf_div, cgb) = (self.lf_div as i32, self.cgb);
            let s = &mut self.square[i];
            s.envelope_clock.locked = false;
            s.envelope_clock.clock = false;
            s.did_tick = false;
            let mut force_unsurpressed = false;
            if !self.is_active[i] {
                if cgb && (s.sample_countdown as i32 - s.delay as i32) / 2 & 0x400 == 0 && value & 4 == 0 {
                    s.current_sample_index = (s.current_sample_index + 1) & 7;
                    force_unsurpressed = true;
                }
                s.delay = (6 - lf_div) as u8;
                s.sample_countdown = (s.sample_length ^ 0x7FF) * 2 + s.delay as u16;
            } else {
                let mut extra_delay = 0;
                if !cgb {
                } else if !s.just_reloaded && value & 4 == 0 && (s.sample_countdown as i32 - 1 - s.delay as i32) / 2 & 0x400 == 0 {
                    s.current_sample_index = (s.current_sample_index + 1) & 7;
                    s.sample_surpressed = false;
                } else if s.sample_length == 0x7FF && old_sample_length != 0x7FF && s.sample_surpressed {
                    extra_delay += 2;
                }
                s.delay = (4 - lf_div + extra_delay) as u8;
                s.sample_countdown = (s.sample_length ^ 0x7FF) * 2 + s.delay as u16;
            }
            s.current_volume = nrx2 >> 4;
            if self.is_active[i] {
                self.update_square_sample(i);
            }
            self.square[i].volume_countdown = nrx2 & 7;
            if nrx2 & 0xF8 != 0 && !self.is_active[i] {
                self.is_active[i] = true;
                self.update_sample(i, 0);
                self.square[i].sample_surpressed = !force_unsurpressed;
            }
            if self.square[i].pulse_length == 0 {
                self.square[i].pulse_length = 0x40;
                self.square[i].length_enabled = false;
            }
            if i == SQUARE_1 {
                let nr10 = self.reg(NR10);
                self.square_sweep_instant_calculation_done = false;
                self.shadow_sweep_sample_length = 0;
                self.channel1_completed_addend = 0;
                if nr10 & 7 != 0 {
                    self.square_sweep_calculate_countdown = nr10 & 7;
                    self.square_sweep_calculate_countdown_reload_timer = 2;
                    self.unshifted_sweep = false;
                    if !was_active {
                        self.square_sweep_calculate_countdown_reload_timer += 1;
                    }
                    self.sweep_length_addend = self.square[SQUARE_1].sample_length >> (nr10 & 7);
                } else {
                    self.sweep_length_addend = 0;
                }
                self.channel_1_restart_hold = 2 - self.lf_div as u8 + self.cgb as u8 * 2;
                self.square_sweep_countdown = ((nr10 >> 4) & 7) ^ 7;
            }
        }
        // Enabling length while the DIV divider's low bit is 1 clocks it once.
        let s = &mut self.square[i];
        if value & 0x40 != 0 && !s.length_enabled && self.div_divider & 1 != 0 && s.pulse_length != 0 {
            s.pulse_length -= 1;
            if s.pulse_length == 0 {
                if value & 0x80 != 0 {
                    s.pulse_length = 0x3F;
                } else {
                    self.is_active[i] = false;
                    self.update_sample(i, 0);
                }
            }
        }
        self.square[i].length_enabled = value & 0x40 != 0;
    }

    fn write_nr34(&mut self, value: u8) {
        self.wave.sample_length = self.wave.sample_length & 0xFF | (value as u16 & 7) << 8;
        if value & 0x80 != 0 {
            self.wave.pulsed = true;
            // DMG: a trigger in the APU cycle before the channel reads wave RAM corrupts it.
            if !self.cgb && self.is_active[WAVE] && self.wave.sample_countdown == 0 {
                let offset = ((self.wave.current_sample_index as usize + 1) >> 1) & 0xF;
                let ram = &mut self.regs[WAV..];
                if offset < 4 {
                    ram[0] = ram[offset];
                } else {
                    let block = offset & !3;
                    let (a, b, c) = (ram[0], ram[block], ram[offset]);
                    ram[block] = a & b | a & c | b & c;
                    ram.copy_within(block..block + 4, 0);
                }
            }
            self.wave.current_sample_index = 0;
            if self.is_active[WAVE] && self.wave.sample_countdown == 0 {
                self.wave.current_sample_byte = self.regs[WAV];
            }
            if self.wave.enable {
                self.is_active[WAVE] = true;
                let v = (self.wave.current_sample_byte >> 4) >> self.wave.shift;
                self.update_sample(WAVE, v);
            }
            self.wave.sample_countdown = (self.wave.sample_length ^ 0x7FF) + 3;
            if self.wave.pulse_length == 0 {
                self.wave.pulse_length = 0x100;
                self.wave.length_enabled = false;
            }
        }
        let w = &mut self.wave;
        if value & 0x40 != 0 && !w.length_enabled && self.div_divider & 1 != 0 && w.pulse_length != 0 {
            w.pulse_length -= 1;
            if w.pulse_length == 0 {
                if value & 0x80 != 0 {
                    w.pulse_length = 0xFF;
                } else {
                    self.is_active[WAVE] = false;
                    self.update_sample(WAVE, 0);
                }
            }
        }
        self.wave.length_enabled = value & 0x40 != 0;
    }

    fn prepare_noise_start(&mut self) {
        self.noise_counter_active = self.reg(NR42) & 0xF8 != 0;
        let was_started_with_dac_disabled = self.noise_started_with_dac_disabled;
        self.noise_started_with_dac_disabled = !self.noise_counter_active;
        let mut divisor = self.reg(NR43) & 7;
        let was_background_counting = self.noise_background_counter_active;
        self.noise_background_counter_active = true;
        let mut instant_step = false;
        let mut div_1_glitch = false;
        let active = self.is_active[NOISE];
        let n = &mut self.noise;
        if divisor > 1 && n.counter_countdown == 1 {
            n.counter = (n.counter + 1) & 0x3FFF;
        } else if n.counter_countdown == 2 && n.alignment & 3 == 0 && active {
            if divisor == 0 {
                divisor = 8;
            } else if divisor == 1 {
                if !n.did_step_counter {
                    div_1_glitch = true;
                }
                let mask = 1u32 << (self.regs[NR43] >> 4);
                let old_bit = n.counter as u32 & mask != 0;
                n.counter = (n.counter + 1) & 0x3FFF;
                let new_bit = n.counter as u32 & mask != 0;
                if new_bit && !old_bit {
                    instant_step = true;
                }
            }
        }
        let mut cd: i32 = if divisor == 0 { 6 } else { divisor as i32 * 4 + 6 };
        let align = n.alignment;
        if align & 1 != 0 {
            if divisor == 0 {
                if was_background_counting {
                    cd -= 1;
                } else {
                    cd += 1;
                }
            } else if align & 2 != 0 {
                if divisor == 1 && !active {
                    cd += 1;
                } else {
                    cd -= 3;
                }
            } else {
                cd -= 1;
                if divisor == 1 && active {
                    cd -= 4;
                }
            }
        } else if divisor != 0 {
            if align & 2 != 0 {
                cd -= 2;
            } else if divisor > 1 {
                cd -= 4;
            } else if divisor == 1 && active && self.regs[NR43] & 0xF0 == 0 {
                cd -= 4;
            }
        }
        if divisor > 1 {
            if !self.noise_counter_active && align & 3 == 0 {
                cd += 4;
            }
        } else if was_background_counting && !active && align & 3 == 0 {
            if divisor == 0 {
                if was_started_with_dac_disabled {
                    cd += 28;
                }
            } else {
                cd -= 4;
            }
        }
        if div_1_glitch {
            cd -= 4;
        }
        n.counter_countdown = cd as u8;
        n.lfsr = if divisor == 0 && active && align & 3 == 3 { 0x0055 } else { 0 };
        if instant_step {
            self.step_lfsr();
        }
    }

    /// NR43 writes (`nr43_write`, CPU CGB E): a change of the selected counter bit can
    /// step the LFSR, and some changes glitch it.
    fn nr43_write(&mut self, new: u8) {
        let old_narrow = self.noise.narrow;
        self.noise.narrow = new & 8 != 0;
        let old = self.regs[NR43];
        self.regs[NR43] = new;
        if old & 0xF0 == new & 0xF0 {
            return;
        }
        let counter = self.noise.counter as u32;
        let old_bit = (counter >> (old >> 4)) & 1 != 0;
        let glitch_value = (old & 0x7F) | (new & 0x80);
        let glitch_bit = (counter >> (glitch_value >> 4)) & 1 != 0;
        let new_bit = (counter >> (new >> 4)) & 1 != 0;
        if old_bit == new_bit && new_bit != glitch_bit {
            if new_bit {
                if new & 0x80 == 0 {
                    self.step_lfsr();
                } else {
                    let t1 = (old >> 4) & 7;
                    let t2 = (new >> 4) & 7;
                    if (t1 ^ 7) + t2 > 7 || (t1 ^ 7) & t2 != 0 {
                        let l = &mut self.noise.lfsr;
                        *l &= !0x80;
                        *l |= (*l >> 1) & 0x80;
                        if (t1 == 0 || t1 == 4) && t2 == 3 {
                            *l &= (*l >> 1) | 0x545;
                            self.update_lfsr();
                        } else if t1 == 2 && t2 == 3 {
                            let mut mask = 0x555;
                            if *l & 0xC == 0xC {
                                mask |= 8;
                            }
                            if *l & 0xC00 == 0xC00 {
                                mask |= 0x800;
                            }
                            *l &= (*l >> 1) | mask;
                            self.update_lfsr();
                        }
                        if !self.noise.narrow && old_narrow && self.lfsr_stepped_in_narrow {
                            if self.lfsr_bit_7_before_step {
                                self.noise.lfsr |= 0x40;
                            } else {
                                self.noise.lfsr &= !0x40;
                            }
                        }
                        self.noise.lfsr |= if self.noise.narrow { 0x4040 } else { 0x4000 };
                        self.lfsr_stepped_in_narrow = self.noise.narrow;
                    }
                }
            } else {
                const MAP: [u8; 64] = {
                    let mut m = [0u8; 64];
                    let set = [(0o02, 4), (0o03, 2), (0o04, 2), (0o05, 2), (0o12, 2), (0o13, 4), (0o14, 2), (0o15, 2), (0o20, 1), (0o21, 2), (0o23, 1), (0o24, 5), (0o25, 3), (0o34, 2), (0o35, 2), (0o41, 2), (0o42, 2), (0o43, 2), (0o50, 6), (0o52, 2), (0o53, 2)];
                    let mut k = 0;
                    while k < set.len() {
                        m[set[k].0] = set[k].1;
                        k += 1;
                    }
                    m
                };
                let glitch = if new & 0x80 != 0 { MAP[(((old & 0x70) >> 1) | ((new & 0x70) >> 4)) as usize] } else { 0 };
                let narrow = self.noise.narrow;
                match glitch {
                    1 | 6 => {
                        self.step_lfsr();
                        let l = &mut self.noise.lfsr;
                        if glitch == 6 {
                            if (narrow && *l & 0x71 == 0x20) || *l & 0x71 == 0x61 {
                                *l &= !0x20;
                            }
                            if *l & 0x7001 == 0x2000 || *l & 0x7001 == 0x6001 {
                                *l &= !0x2000;
                            }
                        }
                        if *l & 3 == 2 {
                            *l &= !2;
                        }
                    }
                    2 => {
                        let prev = self.noise.lfsr;
                        self.step_lfsr();
                        self.noise.lfsr &= prev | 1;
                    }
                    3 | 5 => {
                        let l = &mut self.noise.lfsr;
                        if glitch == 5 {
                            if *l & 3 == 2 {
                                *l &= if narrow { !0x4040 } else { !0x4000 };
                            }
                            if *l & 0x19 == 8 {
                                *l &= !8;
                            }
                        }
                        *l &= !1;
                        *l |= (*l >> 1) & 1;
                        self.update_lfsr();
                        self.lfsr_stepped_in_narrow = narrow;
                    }
                    4 => {
                        let prev = self.noise.lfsr;
                        self.step_lfsr();
                        self.noise.lfsr &= prev | if narrow { !0x2022 } else { !0x2002 };
                    }
                    _ => self.step_lfsr(),
                }
            }
        } else if !old_bit && new_bit {
            self.step_lfsr();
        }
    }

    fn write_nr44(&mut self, value: u8) {
        if value & 0x80 != 0 {
            self.noise.envelope_clock.locked = false;
            self.noise.envelope_clock.clock = false;
            self.noise.lfsr = 0;
            self.prepare_noise_start();
            let nr42 = self.reg(NR42);
            let n = &mut self.noise;
            n.current_volume = nr42 >> 4;
            n.current_lfsr_sample = false;
            n.volume_countdown = nr42 & 7;
            n.did_step_counter = n.alignment & 3 == 2;
            if nr42 & 0xF8 != 0 {
                self.is_active[NOISE] = true;
                self.update_sample(NOISE, 0);
            }
            if self.noise.pulse_length == 0 {
                self.noise.pulse_length = 0x40;
                self.noise.length_enabled = false;
            }
        }
        let n = &mut self.noise;
        if value & 0x40 != 0 && !n.length_enabled && self.div_divider & 1 != 0 && n.pulse_length != 0 {
            n.pulse_length -= 1;
            if n.pulse_length == 0 {
                if value & 0x80 != 0 {
                    n.pulse_length = 0x3F;
                } else {
                    self.is_active[NOISE] = false;
                    self.update_sample(NOISE, 0);
                }
            }
        }
        self.noise.length_enabled = value & 0x40 != 0;
    }

    /// Each DAC maps its channel's digital output, 0-15, to -15..15 (0 when the DAC is
    /// off); NR51 routes channels to each side and NR50 scales a side by 1-8 ([`Mix`]).
    pub fn mix(&self) -> Mix {
        let channels: [i16; 4] = std::array::from_fn(|ch| if self.dac_enabled(ch) { 2 * self.output(ch) as i16 - 15 } else { 0 });
        let (nr50, nr51) = (self.reg(NR50), self.reg(NR51));
        let mut out: Mix = [0; 2];
        for (side, shift) in [(0, 4), (1, 0)] {
            let routed = (nr51 >> shift) & 0x0F;
            let mix: i16 = (0..4).filter(|ch| routed >> ch & 1 != 0).map(|ch| channels[ch]).sum();
            out[side] = mix * ((nr50 >> shift & 7) as i16 + 1);
        }
        out
    }

    fn push_sample(&mut self) {
        let out = self.mix();
        if self.samples_out.len() < SAMPLE_CAPACITY {
            self.samples_out.push(out);
        }
    }

    /// Appends the mixes generated since the last call to `out`.
    pub fn take_samples(&mut self, out: &mut Vec<Mix>) {
        out.extend_from_slice(&self.samples_out);
        self.samples_out.clear();
    }

    /// The M-cycle just run left envelope steps waiting for the next M-cycle's start (a
    /// counted fall at double speed on CGB with the APU on), so a segment of closed forms
    /// must end after it.
    pub fn steps_wait(&self) -> bool {
        self.pending_envelope_tick == 1
    }
}

/// An NRx2 write while the channel plays ("zombie mode", `_nrx2_glitch`, CPU CGB E).
fn nrx2_glitch(volume: &mut u8, value: u8, old: u8, countdown: &mut u8, clock: &mut EnvelopeClock) {
    if clock.clock {
        *countdown = value & 7;
    }
    let mut should_tick = value & 7 != 0 && old & 7 == 0 && !clock.locked;
    let should_invert = (value & 8) ^ (old & 8) != 0;
    if value & 0xF == 8 && old & 0xF == 8 && !clock.locked {
        should_tick = true;
    }
    if should_invert {
        if value & 8 != 0 {
            if old & 7 == 0 && !clock.locked {
                *volume ^= 0xF;
            } else {
                *volume = 0xEu8.wrapping_sub(*volume) & 0xF;
            }
            should_tick = false;
        } else {
            *volume = 0x10u8.wrapping_sub(*volume) & 0xF;
        }
    }
    if should_tick {
        *volume = if value & 8 != 0 { volume.wrapping_add(1) } else { volume.wrapping_sub(1) } & 0xF;
    } else if value & 7 == 0 && clock.clock {
        clock.set(false, false, 0);
    }
}

impl crate::device::Device for Apu {
    fn read(&self, addr: u16) -> u8 {
        Apu::read(self, addr)
    }

    fn write(&mut self, addr: u16, value: u8, _s: &mut crate::device::Signals) {
        Apu::write(self, addr, value);
    }

    /// `mcycles` M-cycles at the block's speed, with the frame-sequencer events the
    /// control logic signals at the last.
    fn advance(&mut self, mcycles: u64, s: &mut crate::device::Signals) {
        for k in 1..=mcycles {
            let div = if k == mcycles { Div::of(s) } else { Div::default() };
            self.mcycle(s.fast, div);
        }
    }
}

impl AudioDevice for Apu {
    /// The reference reads its state as it is, because the machine has already ticked
    /// every M-cycle.
    fn sync(&mut self, _addr: u16) {}

    fn take_samples(&mut self, out: &mut Vec<Mix>) {
        Apu::take_samples(self, out);
    }
}
