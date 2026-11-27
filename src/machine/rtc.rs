//! MBC3's real-time clock, the reference the control checkpoint is held to
//! (`docs/control-in-weights.md` §20). The cartridge decodes which chip a cartridge RAM
//! access reaches; this module holds the clock's registers and its own copies of the
//! select (4000-5FFF) and RAM enable (0000-1FFF) writes, which decide the register an
//! access reaches. Wall-clock time is a host input: the host hands over the seconds
//! elapsed since its last reading and applies no clock rule of its own.

/// Seconds the day counter's 9 bits hold before it wraps and sets the carry.
const DAY_LIMIT: u64 = 512 * 86400;
pub const FOOTER: usize = 48;

/// The clock's state, the same bytes a checkpoint holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rtc {
    /// RTC S, M, H, DL and DH (day bit 8, halt in bit 6, the day counter's carry in bit
    /// 7), always within range: S and M below 60, H below 24, DH's other bits 0.
    pub regs: [u8; 5],
    /// The copy the last latch took, which reads see.
    pub latched: [u8; 5],
    /// The clock register the last 4000-5FFF write named, 0 to 4 for 08 to 0C; None for
    /// a RAM bank or any other value, which the cartridge's own select keeps.
    pub select: Option<u8>,
    /// RAM and the clock enabled (0x0A written to 0000-1FFF).
    pub enabled: bool,
    /// 0x00 was the last byte written to 6000-7FFF, so a 0x01 latches.
    pub armed: bool,
}

impl Rtc {
    fn halted(&self) -> bool {
        self.regs[4] & 0x40 != 0
    }

    fn seconds(&self) -> u64 {
        let r = &self.regs;
        let days = (r[4] as u64 & 1) << 8 | r[3] as u64;
        days * 86400 + r[2] as u64 * 3600 + r[1] as u64 * 60 + r[0] as u64
    }

    /// Sets the count from seconds below 512 days, keeping halt and the carry.
    fn set_seconds(&mut self, s: u64) {
        let days = s / 86400;
        let flags = self.regs[4] & 0xC0;
        self.regs = [(s % 60) as u8, (s / 60 % 60) as u8, (s / 3600 % 24) as u8, days as u8, (days >> 8) as u8 & 1 | flags];
    }

    /// Registers as written, each reduced into its range the way this clock counts.
    fn set_registers(&mut self, r: [u8; 5]) {
        self.regs = [r[0] % 60, r[1] % 60, r[2] % 24, r[3], r[4] & 0xC1];
    }

    /// Counts the wall time elapsed unless halted. Past day 511 the counter wraps and
    /// the carry sets, however far past.
    pub fn elapse(&mut self, seconds: u64) {
        if self.halted() {
            return;
        }
        let sum = self.seconds() as u128 + seconds as u128;
        if sum >= DAY_LIMIT as u128 {
            self.regs[4] |= 0x80;
        }
        self.set_seconds((sum % DAY_LIMIT as u128) as u64);
    }

    /// The register a cartridge RAM access reaches, when the clock is enabled and selected.
    fn register(&self) -> Option<usize> {
        self.select.filter(|_| self.enabled).map(usize::from)
    }

    /// A read of A000-BFFF with the clock's chip selected: the latched register, or open
    /// bus while disabled.
    pub fn read(&self) -> u8 {
        self.register().map_or(0xFF, |i| self.latched[i])
    }

    /// A write the clock sees: the RAM enable, the select, the latch, or a register.
    pub fn write(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..=0x1FFF => self.enabled = value & 0x0F == 0x0A,
            0x4000..=0x5FFF => self.select = (0x08..=0x0C).contains(&value).then(|| value - 0x08),
            0x6000..=0x7FFF => {
                if self.armed && value == 0x01 {
                    self.latched = self.regs;
                }
                self.armed = value == 0x00;
            }
            0xA000..=0xBFFF => {
                if let Some(i) = self.register() {
                    let mut r = self.regs;
                    r[i] = value;
                    self.set_registers(r);
                }
            }
            _ => panic!("the clock sees no write to {addr:#06x}"),
        }
    }

    /// The clock as BGB and VBA-M append it to a save: the live and the latched registers
    /// as little-endian u32s, then the Unix time it was taken at as a u64.
    pub fn footer(&self, time: Option<u64>) -> [u8; FOOTER] {
        let mut out = [0; FOOTER];
        for (field, value) in out.chunks_exact_mut(4).zip(self.regs.into_iter().chain(self.latched)) {
            field.copy_from_slice(&(value as u32).to_le_bytes());
        }
        out[40..].copy_from_slice(&time.unwrap_or(0).to_le_bytes());
        out
    }

    /// A clock and the Unix time it was saved at from a 48-byte footer, or the 44-byte
    /// one older writers use with a u32 time. A footer written before the host ever set
    /// the time has no reference point.
    pub fn from_footer(bytes: &[u8]) -> (Rtc, Option<u64>) {
        let field = |i: usize| bytes[i * 4];
        let mut rtc = Rtc { latched: std::array::from_fn(|i| field(5 + i)), ..Rtc::default() };
        rtc.set_registers(std::array::from_fn(field));
        let mut time = [0; 8];
        time[..bytes.len() - 40].copy_from_slice(&bytes[40..]);
        (rtc, Some(u64::from_le_bytes(time)).filter(|&t| t != 0))
    }
}
