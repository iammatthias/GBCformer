//! The peripheral contract. A [`Device`] owns some bus addresses and advances in blocks
//! of M-cycles; everything another device or the bus must see leaves through
//! [`Signals`]. The scheduler in `machine` cuts each block at the smallest
//! [`Device::horizon`], so a device that must interleave with the others at M-cycle
//! resolution (OAM DMA while it copies) gets blocks of one, and the control logic,
//! which holds the PPU's timing, ends a block at every M-cycle the renderer must act
//! on and says so through [`Signals`]. Within a block no device observes another. That
//! is what lets a compiled checkpoint run one block per pass instead of one pass per
//! M-cycle.
//!
//! The reference devices are `machine::{apu, dma}` and `machine::control::Control`, which
//! runs the timer and `io` inside it. Each checkpoint device
//! has its own lockstep beside its reference (`ApuLockstep`, `ControlLockstep`, and
//! `PpuModel` under `--lockstep ppu`), since each compares state at its own pass
//! boundaries.

/// The copies and the block's signals are the transport's (`crate::transport`), which the
/// minimal runner shares: `double` is its `fast`, an LCD turned off its `blank`, a palette
/// byte its `own`; `held` holds the HDMA blocks the control logic keeps to the end of the
/// GBC CPU step that runs.
pub use crate::transport::{Copy, Signals};

pub trait Device {
    fn read(&self, addr: u16) -> u8;
    /// Writes a register. Interrupt bits and copies the write causes go into `s`.
    fn write(&mut self, addr: u16, value: u8, s: &mut Signals);
    /// Advances `mcycles` M-cycles, at most [`Device::horizon`] of them.
    fn advance(&mut self, mcycles: u64, s: &mut Signals);
    /// M-cycles this device can advance before another device must see what it did.
    /// The scheduler never advances any device past the smallest horizon.
    fn horizon(&self) -> u64 {
        u64::MAX
    }
}
