//! Joypad, and serial with or without a link partner ([`Cable`]).

pub const SERIAL_IRQ: u8 = 1 << 3;
pub const JOYPAD_IRQ: u8 = 1 << 4;

/// Pressed buttons as a bit set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Buttons(pub u8);

impl std::ops::BitOr for Buttons {
    type Output = Buttons;

    fn bitor(self, other: Buttons) -> Buttons {
        Buttons(self.0 | other.0)
    }
}

impl Buttons {
    pub const RIGHT: u8 = 1 << 0;
    pub const LEFT: u8 = 1 << 1;
    pub const UP: u8 = 1 << 2;
    pub const DOWN: u8 = 1 << 3;
    pub const A: u8 = 1 << 4;
    pub const B: u8 = 1 << 5;
    pub const SELECT: u8 = 1 << 6;
    pub const START: u8 = 1 << 7;

    pub fn parse(name: &str) -> Option<u8> {
        Some(match name.to_ascii_lowercase().as_str() {
            "right" => Self::RIGHT,
            "left" => Self::LEFT,
            "up" => Self::UP,
            "down" => Self::DOWN,
            "a" => Self::A,
            "b" => Self::B,
            "select" => Self::SELECT,
            "start" => Self::START,
            _ => return None,
        })
    }
}
#[derive(Clone)]
pub struct Joypad {
    /// P14/P15 as last written (bits 4-5, active low).
    pub(super) select: u8,
    pub(super) pressed: Buttons,
}

impl Joypad {
    pub fn post_boot_dmg() -> Self {
        Joypad { select: 0x00, pressed: Buttons(0) }
    }

    /// The CGB boot ROM ends by writing 0xFF to P1, deselecting both groups (SameBoy's
    /// cgb_boot.asm), so P1 reads 0xFF.
    pub fn post_boot_cgb() -> Self {
        Joypad { select: 0x30, ..Self::post_boot_dmg() }
    }

    /// Active-low input lines P10-P13 for the selected groups.
    fn lines(&self) -> u8 {
        let mut low = 0;
        if self.select & 0x10 == 0 {
            low |= self.pressed.0 & 0x0F;
        }
        if self.select & 0x20 == 0 {
            low |= self.pressed.0 >> 4;
        }
        !low & 0x0F
    }

    pub fn read(&self) -> u8 {
        0xC0 | self.select | self.lines()
    }

    /// Writes P1's select bits; returns the joypad interrupt if a selected line fell.
    pub fn write(&mut self, value: u8) -> u8 {
        let before = self.lines();
        self.select = value & 0x30;
        self.falling_edges(before)
    }

    /// Sets the pressed buttons; returns the joypad interrupt if a selected line fell.
    pub fn set(&mut self, pressed: Buttons) -> u8 {
        let before = self.lines();
        self.pressed = pressed;
        self.falling_edges(before)
    }

    fn falling_edges(&self, before: u8) -> u8 {
        if before & !self.lines() != 0 { JOYPAD_IRQ } else { 0 }
    }
}

/// A link cable between two machines' serial ports. Each end shows the other its SB as
/// written and receives the byte the clocking side shifted out. The machine clocking a
/// transfer (internal clock) swaps the two bytes when its 8 bits are done; the other side
/// takes its byte at its next advance and, waiting with the external clock, completes.
#[derive(Default)]
pub struct Cable {
    ends: [End; 2],
}

#[derive(Default)]
struct End {
    sb: u8,
    inbox: Option<u8>,
}

/// One machine's end of a [`Cable`].
#[derive(Clone)]
pub struct Link {
    cable: std::sync::Arc<std::sync::Mutex<Cable>>,
    side: usize,
}

impl Link {
    /// Both ends of a new cable.
    pub fn pair() -> (Link, Link) {
        let cable = std::sync::Arc::new(std::sync::Mutex::new(Cable::default()));
        (Link { cable: cable.clone(), side: 0 }, Link { cable, side: 1 })
    }

    /// The byte the other end shows.
    pub fn their_sb(&self) -> u8 {
        self.with(|_, other| other.sb)
    }

    /// This end shows `sb`.
    pub fn show(&self, sb: u8) {
        self.with(|me, _| me.sb = sb);
    }

    /// This end's transfer ended: `byte` reaches the other end.
    pub fn send(&self, byte: u8) {
        self.with(|_, other| other.inbox = Some(byte));
    }

    /// The byte the other end clocked across, not yet taken.
    pub fn waiting(&self) -> Option<u8> {
        self.with(|me, _| me.inbox)
    }

    /// Takes the byte the other end clocked across.
    pub fn take(&self) {
        self.with(|me, _| me.inbox = None);
    }

    fn with<R>(&self, f: impl FnOnce(&mut End, &mut End) -> R) -> R {
        let mut c = self.cable.lock().unwrap();
        let [a, b] = &mut c.ends;
        if self.side == 0 { f(a, b) } else { f(b, a) }
    }
}

/// Serial port. With nothing attached an internal-clock transfer shifts in 1s and an
/// external-clock one never ends; with a [`Link`] the bytes cross the cable.
#[derive(Clone)]
pub struct Serial {
    /// The other machine, when a cable is plugged in.
    pub link: Option<Link>,
    pub(super) sb: u8,
    pub(super) sc: u8,
    /// M-cycles left in the current transfer.
    pub(super) remaining: u16,
    /// Bytes the game sent, for blargg test ROM output.
    pub output: Vec<u8>,
}

/// 8 bits at 8192 Hz.
pub const TRANSFER_MCYCLES: u16 = 8 * 128;

impl Serial {
    pub fn post_boot_dmg() -> Self {
        Serial { sb: 0, sc: 0, remaining: 0, output: Vec::new(), link: None }
    }

    /// SC reads 0x7E after the CGB boot ROM, in CGB mode and in compatibility mode alike
    /// (Nintendo's CGB boot ROM run to its hand-off; mooneye
    /// `misc/boot_hwio-C`).
    pub fn post_boot_cgb() -> Self {
        Self::post_boot_dmg()
    }

    pub fn read(&self, addr: u16) -> u8 {
        if addr == 0xFF01 { self.sb } else { self.sc | 0x7E }
    }

    /// A write to SB or SC with the system counter at `counter` (T-cycles). The internal
    /// clock shifts on the falling edges of the counter's bit 8, so a transfer ends at the
    /// eighth edge after its start, not 1,024 M-cycles after it (mooneye
    /// `serial/boot_sclk_align`: "clock edges align based on the reset time").
    pub fn write(&mut self, addr: u16, value: u8, counter: u16) {
        if addr == 0xFF01 {
            self.sb = value;
            if let Some(l) = &self.link {
                l.with(|me, _| me.sb = value);
            }
            return;
        }
        self.sc = value & 0x81;
        if self.sc == 0x81 {
            self.output.push(self.sb);
            self.remaining = TRANSFER_MCYCLES - (counter >> 2 & 127);
        }
    }

    /// Advances `mcycles` M-cycles and returns the interrupt bits they raised.
    pub fn run(&mut self, mcycles: u64) -> u8 {
        if self.remaining == 0 {
            return if self.link.is_some() { self.receive() } else { 0 };
        }
        let ran = mcycles.min(self.remaining as u64) as u16;
        self.remaining -= ran;
        if self.remaining > 0 {
            return 0;
        }
        self.sb = match &self.link {
            Some(l) => {
                let out = self.sb;
                l.with(|me, other| {
                    other.inbox = Some(out);
                    me.sb = other.sb;
                    other.sb
                })
            }
            None => 0xFF,
        };
        self.sc &= 0x7F;
        SERIAL_IRQ
    }

    /// A byte the other side clocked across: it replaces SB, and a transfer waiting on
    /// the external clock completes.
    fn receive(&mut self) -> u8 {
        let Some(byte) = self.link.as_ref().and_then(|l| l.with(|me, _| me.inbox.take().inspect(|&b| me.sb = b))) else { return 0 };
        self.sb = byte;
        if self.sc & 0x81 == 0x80 {
            self.sc &= 0x7F;
            return SERIAL_IRQ;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joypad_groups_and_press_interrupt() {
        let mut j = Joypad::post_boot_dmg();
        assert_eq!(j.read(), 0xCF);
        j.write(0x20);
        assert_eq!(j.set(Buttons(Buttons::START)), 0, "Start is not in the selected group");
        assert_eq!(j.read(), 0xEF);
        assert_eq!(j.set(Buttons(Buttons::START | Buttons::DOWN)), JOYPAD_IRQ);
        assert_eq!(j.read(), 0xE7);
        j.write(0x10);
        assert_eq!(j.read(), 0xD7, "buttons group shows Start");
        j.write(0x00);
        assert_eq!(j.read(), 0xC7, "both groups AND together");
    }

    #[test]
    fn serial_without_partner_completes_with_ff() {
        let mut s = Serial::post_boot_dmg();
        s.write(0xFF01, b'P', 0);
        s.write(0xFF02, 0x81, 0x0108);
        assert_eq!((s.run(1023 - 0x42), s.read(0xFF02)), (0, 0xFF));
        assert_eq!(s.run(1), SERIAL_IRQ);
        assert_eq!((s.read(0xFF01), s.read(0xFF02)), (0xFF, 0x7F));
        assert_eq!(s.output, b"P");
    }
}
