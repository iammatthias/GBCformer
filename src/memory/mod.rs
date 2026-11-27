//! Memory held by the model instead of host byte arrays.
//!
//! The ROM, cartridge RAM, WRAM, VRAM, OAM and HRAM form one flat byte space
//! ([`crate::machine::Machine::memory_layout`]); bank switching is address translation
//! that the control logic performs before an address reaches the memory, so the memory
//! itself only sees flat addresses.
//!
//! Two prototypes were benchmarked (`docs/perf-log.md`, "Memory prototypes") and A was
//! chosen. [`attn`] is prototype A, the memory as a Llama attention cache: the cartridge
//! is prefilled as cells, a write replaces its line's cell, and a read is one attention
//! head whose query matches the address. Prototype B, a DeltaNet-style recurrent layer
//! (NESFormer's mechanism, outside the Llama layer set), was removed with its benchmark
//! `examples/membench` (`docs/perf-log.md`, "Memory prototypes", has the numbers).
//!
//! [`attn::AttnMemory`] is the dense form of A, the arithmetic a stock inference stack
//! would do. [`Bytes`] is the reference, and [`Lockstep`] stops at the first read that
//! differs.

pub mod attn;
pub mod dense;
pub mod image;
pub mod layout;

/// A flat physical address into the memory the model holds.
pub type Addr = u32;

pub trait Memory {
    fn read(&mut self, addr: Addr) -> u8;
    fn write(&mut self, addr: Addr, value: u8);
}

/// The reference: a byte array.
pub struct Bytes(pub Vec<u8>);

impl Memory for Bytes {
    fn read(&mut self, addr: Addr) -> u8 {
        self.0[addr as usize]
    }
    fn write(&mut self, addr: Addr, value: u8) {
        self.0[addr as usize] = value;
    }
}

/// One byte a DMA moved on the host (OAM DMA, CGB HDMA or GDMA): the bus addresses it
/// moved between and the byte the host moved. A memory core places both addresses in
/// the fixed layout from the banks its model emitted ([`layout::place`]), and keeps the
/// copy as a copy-mode pair of records: a reader at the source, a write at the
/// destination. The host applies no mapper rule to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copied {
    /// None for a fill (`device::Copy::Fill`), whose byte comes from no source.
    pub source: Option<u16>,
    pub dest: u16,
    /// The byte the host's bus read, while the host keeps its shadow machine.
    pub byte: Option<u8>,
}

/// Where a copy's byte comes from: the model's own memory, or the host for a source it
/// serves (an HDMA source in the I/O page lands in VOID), which a run reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyFrom {
    Memory(Addr),
    Host(u8),
}

/// One CPU memory access, as recorded from a run: a read, or a write of `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub addr: Addr,
    pub write: Option<u8>,
}

/// Runs a candidate beside the reference and panics at the first read that differs,
/// naming the access that caused it.
pub struct Lockstep<R: Memory, M: Memory> {
    pub reference: R,
    pub model: M,
    pub accesses: u64,
}

impl<R: Memory, M: Memory> Lockstep<R, M> {
    pub fn new(reference: R, model: M) -> Self {
        Lockstep { reference, model, accesses: 0 }
    }
}

impl<R: Memory, M: Memory> Memory for Lockstep<R, M> {
    fn read(&mut self, addr: Addr) -> u8 {
        let (want, got) = (self.reference.read(addr), self.model.read(addr));
        assert_eq!(got, want, "read of {addr:#x} diverged at access {}", self.accesses);
        self.accesses += 1;
        got
    }
    fn write(&mut self, addr: Addr, value: u8) {
        self.reference.write(addr, value);
        self.model.write(addr, value);
        self.accesses += 1;
    }
}

/// A deterministic xorshift stream for the random read/write tests.
pub struct Rng(pub u64);

impl Rng {
    pub fn draw(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `n` random accesses over `rom_len` read-only bytes followed by `ram_len` writable
/// ones: reads anywhere, writes only into the RAM.
pub fn random_stream(rng: &mut Rng, rom_len: u32, ram_len: u32, n: usize, write_share: u32) -> Vec<Access> {
    (0..n)
        .map(|_| {
            let r = rng.draw();
            if ((r % 100) as u32) < write_share {
                Access { addr: rom_len + (r >> 8) as u32 % ram_len, write: Some((r >> 40) as u8) }
            } else {
                Access { addr: (r >> 8) as u32 % (rom_len + ram_len), write: None }
            }
        })
        .collect()
}
