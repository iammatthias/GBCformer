//! DIV/TIMA/TMA/TAC. TIMA counts falling edges of one bit of the 16-bit system
//! counter (ANDed with the enable bit), so DIV and TAC writes can tick it too.

pub const IRQ: u8 = 1 << 2;

/// Everything the timer holds, for a checkpoint held to it in lockstep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimerState {
    pub counter: u16,
    pub tima: u8,
    pub tma: u8,
    pub tac: u8,
    pub reload_pending: bool,
    pub reloaded: bool,
}

#[derive(Clone)]
pub struct Timer {
    /// System counter in T-cycles; DIV is its high byte.
    counter: u16,
    tima: u8,
    tma: u8,
    tac: u8,
    /// TIMA overflowed during the last M-cycle; it reads 0 until the TMA reload.
    reload_pending: bool,
    /// The last M-cycle reloaded TIMA from TMA: a TIMA write in it is lost, and a TMA
    /// write reaches TIMA too (mooneye-gb `tima_write_cycle`, `tma_write_cycle`;
    /// mooneye `tima_write_reloading`, `tma_write_reloading`).
    reloaded: bool,
}

impl Timer {
    /// At power-on, the system counter at 0.
    pub fn power_on() -> Self {
        Timer { counter: 0, ..Self::post_boot_dmg() }
    }

    pub fn post_boot_dmg() -> Self {
        Timer { counter: 0xABCC, tima: 0, tma: 0, tac: 0xF8, reload_pending: false, reloaded: false }
    }

    /// After the CGB boot ROM hands a CGB-mode game over. The counter is mGBA's CGB-header
    /// boot skip (internalDiv 0x2F0, 12 T-cycles to its next tick), converted the way its
    /// DMG pair (0xABC, 4) gives the 0xABCC above.
    pub fn post_boot_cgb() -> Self {
        Timer { counter: 0x2F04, ..Self::post_boot_dmg() }
    }

    /// After the CGB boot ROM hands a DMG game over in compatibility mode. mooneye
    /// `misc/boot_div-cgbABCDE` (verified on CGB A-E) reads DIV 0x27 just after it ticks
    /// at its read 34 M-cycles after the hand-off (0100's NOP and JP, 27 NOPs, LDH's
    /// third M-cycle), so the counter starts at 0x2700 − 4·34.
    pub fn post_boot_compat() -> Self {
        Timer { counter: 0x2678, ..Self::post_boot_dmg() }
    }

    fn input(&self) -> bool {
        const BIT: [u16; 4] = [9, 3, 5, 7];
        self.tac & 0x04 != 0 && self.counter >> BIT[(self.tac & 3) as usize] & 1 != 0
    }

    fn set_counter_or_tac(&mut self, change: impl FnOnce(&mut Self)) {
        let before = self.input();
        change(self);
        if before && !self.input() {
            self.increment();
        }
    }

    fn increment(&mut self) {
        let (v, overflow) = self.tima.overflowing_add(1);
        self.tima = v;
        self.reload_pending = overflow;
    }

    /// Advances one M-cycle and returns the interrupt bits it raised.
    fn tick(&mut self) -> u8 {
        let mut irq = 0;
        self.reloaded = self.reload_pending;
        if self.reload_pending {
            self.reload_pending = false;
            self.tima = self.tma;
            irq = IRQ;
        }
        self.set_counter_or_tac(|t| t.counter = t.counter.wrapping_add(4));
        irq
    }

    /// The 16-bit system counter, whose high byte is DIV.
    pub fn counter(&self) -> u16 {
        self.counter
    }

    /// Advances `mcycles` M-cycles and returns the interrupt bits they raised.
    pub fn run(&mut self, mcycles: u64) -> u8 {
        (0..mcycles).fold(0, |irq, _| irq | self.tick())
    }

    pub fn state(&self) -> TimerState {
        TimerState { counter: self.counter, tima: self.tima, tma: self.tma, tac: self.tac & 7, reload_pending: self.reload_pending, reloaded: self.reloaded }
    }
}

/// The registers. The timer runs inside the control logic, which takes the interrupt bits
/// `Timer::run` returns.
impl Timer {
    pub fn read(&self, addr: u16) -> u8 {
        match addr {
            0xFF04 => (self.counter >> 8) as u8,
            0xFF05 => self.tima,
            0xFF06 => self.tma,
            _ => self.tac | 0xF8,
        }
    }

    pub fn write(&mut self, addr: u16, value: u8) {
        match addr {
            0xFF04 => self.set_counter_or_tac(|t| t.counter = 0),
            0xFF05 if self.reloaded => {}
            0xFF05 => {
                self.tima = value;
                self.reload_pending = false;
            }
            0xFF06 => {
                self.tma = value;
                if self.reloaded {
                    self.tima = value;
                }
            }
            _ => self.set_counter_or_tac(|t| t.tac = value & 0x07),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticks(t: &mut Timer, n: usize) -> u8 {
        (0..n).fold(0, |irq, _| irq | t.tick())
    }

    #[test]
    fn overflow_reads_zero_for_one_cycle_then_reloads() {
        let mut t = Timer::post_boot_dmg();
        t.write(0xFF04, 0);
        t.write(0xFF06, 0x23);
        t.write(0xFF05, 0xFF);
        t.write(0xFF07, 0x05);
        assert_eq!(ticks(&mut t, 3), 0);
        assert_eq!(t.read(0xFF05), 0xFF, "16 T-cycle period: no edge after 12");
        assert_eq!(ticks(&mut t, 1), 0);
        assert_eq!(t.read(0xFF05), 0x00, "overflow on the 4th M-cycle, IF not yet set");
        assert_eq!(ticks(&mut t, 1), IRQ);
        assert_eq!(t.read(0xFF05), 0x23, "TMA loaded one M-cycle later");
    }

    #[test]
    fn tima_write_during_overflow_cancels_reload() {
        let mut t = Timer::post_boot_dmg();
        t.write(0xFF04, 0);
        t.write(0xFF06, 0x23);
        t.write(0xFF05, 0xFF);
        t.write(0xFF07, 0x05);
        ticks(&mut t, 4);
        t.write(0xFF05, 0x80);
        assert_eq!(ticks(&mut t, 1), 0);
        assert_eq!(t.read(0xFF05), 0x80);
    }

    /// In the M-cycle the reload ran, a TIMA write is lost and a TMA write lands in TIMA
    /// too; one M-cycle later both are plain writes.
    #[test]
    fn writes_in_the_reload_cycle() {
        let start = || {
            let mut t = Timer::post_boot_dmg();
            t.write(0xFF04, 0);
            t.write(0xFF06, 0x23);
            t.write(0xFF05, 0xFF);
            t.write(0xFF07, 0x05);
            ticks(&mut t, 5);
            t
        };
        let mut t = start();
        t.write(0xFF05, 0x80);
        assert_eq!(t.read(0xFF05), 0x23, "the TIMA write in the reload cycle is lost");
        let mut t = start();
        t.write(0xFF06, 0x44);
        assert_eq!((t.read(0xFF05), t.read(0xFF06)), (0x44, 0x44), "the TMA write reaches TIMA");
        let mut t = start();
        ticks(&mut t, 1);
        t.write(0xFF05, 0x80);
        t.write(0xFF06, 0x44);
        assert_eq!((t.read(0xFF05), t.read(0xFF06)), (0x80, 0x44), "a cycle later both are plain writes");
    }

    #[test]
    fn rates_and_div() {
        let mut t = Timer::post_boot_dmg();
        t.write(0xFF04, 0);
        t.write(0xFF05, 0);
        t.write(0xFF07, 0x04);
        ticks(&mut t, 1024);
        assert_eq!(t.read(0xFF05), 4, "4096 Hz: one tick per 256 M-cycles");
        assert_eq!(t.read(0xFF04), 16, "DIV ticks every 64 M-cycles");
        t.write(0xFF04, 0x55);
        assert_eq!(t.read(0xFF04), 0, "any DIV write clears it");
    }

    #[test]
    fn div_reset_on_high_bit_ticks_tima() {
        let mut t = Timer::post_boot_dmg();
        t.write(0xFF04, 0);
        t.write(0xFF05, 0x10);
        t.write(0xFF07, 0x05);
        ticks(&mut t, 2);
        assert_eq!(t.read(0xFF05), 0x10, "bit 3 high, no edge yet");
        t.write(0xFF04, 0);
        assert_eq!(t.read(0xFF05), 0x11, "falling edge from the DIV reset");
    }

    #[test]
    fn read_masks() {
        let t = Timer::post_boot_dmg();
        assert_eq!(t.read(0xFF07), 0xF8);
        assert_eq!(t.read(0xFF04), 0xAB);
    }
}
