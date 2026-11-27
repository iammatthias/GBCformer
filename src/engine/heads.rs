//! A memory head's line from its query's bit forms, and a bus's byte from its columns'.
//!
//! A query's address bit is the sign of its form, `[form ≥ 1]`, while the form lies in
//! [−1, 2] (`circuit::address_bit`). A column select's form is its row's form plus
//! `(2·off − 1)·(1 − gate)` (`circuit::Query::gated`), so with the gate 0 or 1 and every
//! row form 0 or 1 its line is the row's or the off line, with no bit formed. Columns
//! of one row share the row's forms as one unit.

/// An address unit as its bit forms stand: the line their signs spell and where they lie.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Unit {
    line: u32,
    forms: Forms,
}

/// Where a unit's forms lie: all 0 or 1, all in [−1, 2] (each still spells its bit by
/// its sign), or one outside (no address).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
enum Forms {
    #[default]
    Bits,
    Signs,
    Out,
}

impl Unit {
    pub(super) fn form(bits: &[i32]) -> Unit {
        // Forms that are all 0 or 1 are the line's bits themselves, and they are exactly
        // those whose OR, as unsigned, is at most 1.
        let (mut any, mut line) = (0u32, 0u32);
        for (i, &b) in bits.iter().enumerate() {
            any |= b as u32;
            line |= (b as u32) << i;
        }
        if any <= 1 { Unit { line, forms: Forms::Bits } } else { Unit::form_by_sign(bits) }
    }

    #[cold]
    fn form_by_sign(bits: &[i32]) -> Unit {
        let line = bits.iter().enumerate().fold(0, |line, (i, &b)| line | ((b >= 1) as u32) << i);
        let forms = if bits.iter().all(|b| (0..=1).contains(b)) {
            Forms::Bits
        } else if bits.iter().all(|b| (-1..=2).contains(b)) {
            Forms::Signs
        } else {
            Forms::Out
        };
        Unit { line, forms }
    }

    /// The line of a head that reads the unit alone; None when a form is outside [−1, 2].
    pub(super) fn line(self) -> Option<u32> {
        (self.forms != Forms::Out).then_some(self.line)
    }

    /// Whether every form is 0 or 1, as a column select's row must be for its gate to
    /// pick between the row's line and the off line.
    pub(super) fn is_row(self) -> bool {
        self.forms == Forms::Bits
    }

    /// The line of a column select on this row with its gate at `gate`, and whether the
    /// line depends on the row's forms. None when a form of the column is outside [−1, 2].
    pub(super) fn column(self, bits: &[i32], gate: i32, off: u32) -> Option<(u32, bool)> {
        match gate {
            0 if self.is_row() => Some((off, false)),
            1 if self.is_row() => Some((self.line, true)),
            _ => column_by_bit(bits, gate, off).map(|line| (line, true)),
        }
    }
}

/// A column select's line formed bit by bit from its row's forms: the rule `Unit::column`
/// shortens.
#[cold]
fn column_by_bit(bits: &[i32], gate: i32, off: u32) -> Option<u32> {
    let mut line = 0;
    for (i, &b) in bits.iter().enumerate() {
        let form = b + (2 * (off >> i & 1) as i32 - 1) * (1 - gate);
        if !(-1..=2).contains(&form) {
            return None;
        }
        line |= ((form >= 1) as u32) << i;
    }
    Some(line)
}

/// A bus's byte, the sum of its columns' lane bytes, which must not overlap: each channel
/// is 0 or 1. While a layer's columns move, `rest` is what the columns that have not moved
/// drive and `new` what the moved ones drive now.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Bus {
    byte: u8,
    rest: u8,
    new: u8,
    open: bool,
    clash: bool,
}

impl Bus {
    /// One column's lane byte moved from `old` to `new`; a column moves at most once
    /// before its bus settles. True on the bus's first move since it settled.
    pub(super) fn drive(&mut self, old: u8, new: u8) -> bool {
        let first = !self.open;
        if first {
            *self = Bus { byte: self.byte, rest: self.byte, new: 0, open: true, clash: false };
        }
        // The settled byte held `old` from this column alone.
        self.rest &= !old;
        self.clash |= self.new & new != 0;
        self.new |= new;
        first
    }

    /// The bus's byte after the moves and the bits that changed; None when two columns
    /// drive one bit.
    pub(super) fn settle(&mut self) -> Option<(u8, u8)> {
        self.open = false;
        if self.clash || self.rest & self.new != 0 {
            return None;
        }
        let flips = (self.rest | self.new) ^ self.byte;
        self.byte ^= flips;
        Some((self.byte, flips))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    /// The address rule itself: every form of the column in [−1, 2], each bit its sign.
    fn by_definition(bits: &[i32], gate: i32, off: u32) -> Option<u32> {
        let forms: Vec<i32> = bits.iter().enumerate().map(|(i, &b)| b + if off >> i & 1 == 1 { 1 - gate } else { gate - 1 }).collect();
        forms.iter().all(|f| (-1..=2).contains(f)).then(|| forms.iter().enumerate().fold(0, |a, (i, &f)| a | ((f >= 1) as u32) << i))
    }

    /// A column's line by `Unit::column` is the line its forms' signs spell, for 0/1 rows
    /// and for rows and gates outside them, and it follows the row exactly when the row's
    /// forms can move it.
    #[test]
    fn a_column_reads_the_line_its_forms_spell() {
        let mut next = rng(0x9E37_79B9_7F4A_7C15);
        let (mut short, mut long, mut refused) = (0, 0, 0);
        for round in 0..200_000 {
            let wide = round % 4 == 0;
            let bits: Vec<i32> = (0..21).map(|_| if wide && next() % 8 == 0 { (next() % 6) as i32 - 2 } else { (next() & 1) as i32 }).collect();
            let gate = if round % 16 == 5 { (next() % 5) as i32 - 2 } else { (next() & 1) as i32 };
            let off = next() as u32 & 0x1F_FFFF;
            let unit = Unit::form(&bits);
            let want = by_definition(&bits, gate, off);
            let got = unit.column(&bits, gate, off);
            assert_eq!(got.map(|g| g.0), want, "bits {bits:?} gate {gate} off {off:#x}");
            match got {
                Some((line, false)) => {
                    assert_eq!((line, gate), (off, 0));
                    short += 1;
                }
                Some((_, true)) if unit.is_row() && gate == 1 => short += 1,
                Some(_) => long += 1,
                None => refused += 1,
            }
            assert_eq!(unit.line(), by_definition(&bits, 1, 0), "an ungated head's line is its forms' signs");
            assert_eq!(unit, Unit::form_by_sign(&bits), "the short form is the form by sign");
        }
        assert!(short > 100_000 && long > 100 && refused > 1_000, "{short} short, {long} bit by bit, {refused} refused");
    }

    /// A bus's settled byte is the OR of its columns' bytes, refused exactly when two of
    /// them share a bit, whatever order the columns move in.
    #[test]
    fn a_bus_settles_to_the_sum_of_its_columns() {
        let mut next = rng(0xD1B5_4A32_D192_ED03);
        let (mut settled, mut refused) = (0, 0);
        for _ in 0..50_000 {
            let mut cols = [0u8; 10];
            let mut bus = Bus::default();
            for _ in 0..6 {
                let mut moved = false;
                // A column moves at most once before its bus settles.
                let first = (next() % 10) as usize;
                for c in (first..10).take(1 + (next() % 4) as usize) {
                    let new = if next() % 3 == 0 { 0 } else { next() as u8 & next() as u8 & next() as u8 };
                    if new != cols[c] {
                        assert_eq!(bus.drive(cols[c], new), !moved);
                        cols[c] = new;
                        moved = true;
                    }
                }
                if !moved {
                    continue;
                }
                let clash = (0..10).any(|a| (0..a).any(|b| cols[a] & cols[b] != 0));
                let sum = cols.iter().fold(0, |s, &c| s | c);
                let before = bus.byte;
                match bus.settle() {
                    Some((byte, flips)) => {
                        assert!(!clash, "columns {cols:02x?} overlap");
                        assert_eq!((byte, flips), (sum, sum ^ before));
                        settled += 1;
                    }
                    None => {
                        assert!(clash, "columns {cols:02x?} do not overlap");
                        refused += 1;
                        break;
                    }
                }
            }
        }
        assert!(settled > 10_000 && refused > 10_000, "{settled} settled, {refused} refused");
    }
}
