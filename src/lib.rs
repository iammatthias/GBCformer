#[macro_use]
#[cfg_attr(not(feature = "gbc"), allow(unused_macros))]
mod slots;
// The inference core: the checkpoint format, the dense pass, the fast engine, the KV
// cache's memory cells and the GGUF reader. It holds the layout of the model's physical
// address space (region bases, the line size); the address translation's specification,
// the port decode and the mapper's defaults are behind the `gbc` feature, as is all
// that follows.
pub mod checkpoint;
pub mod circuit;
pub mod engine;
pub mod f16acc;
pub mod gguf;
pub mod level;
pub mod llama;
pub mod memory;
pub mod qos;
pub mod sequence;
pub mod transport;
pub mod wiring;
// The Game Boy: the circuits the compiler writes into weights, the reference devices,
// the host machine and its scheduler, sessions and tooling.
#[cfg(feature = "gbc")]
pub mod apu_circuit;
#[cfg(feature = "gbc")]
pub mod control_circuit;
#[cfg(feature = "gbc")]
pub mod device;
#[cfg(feature = "gbc")]
pub mod gbc_wiring;
#[cfg(feature = "gbc")]
pub mod lockstep;
#[cfg(feature = "gbc")]
pub mod machine;
#[cfg(feature = "gbc")]
pub mod one_model;
#[cfg(all(feature = "gbc", not(target_arch = "wasm32")))]
pub mod output;
#[cfg(feature = "gbc")]
pub mod ppu_circuit;
#[cfg(feature = "gbc")]
pub mod press;
#[cfg(feature = "gbc")]
pub mod reference;
#[cfg(feature = "gbc")]
pub mod rom;
#[cfg(all(feature = "gbc", not(target_arch = "wasm32")))]
pub mod save;
#[cfg(feature = "gbc")]
pub mod session;
#[cfg(feature = "gbc")]
pub mod sm83;
#[cfg(feature = "gbc")]
pub mod sst;
#[cfg(feature = "gbc")]
pub mod stack;
#[cfg(feature = "gbc")]
pub mod step;
#[cfg(all(feature = "frontend", not(target_arch = "wasm32")))]
pub mod frontend;

#[cfg(feature = "gbc")]
pub use machine::Hardware;
#[cfg(feature = "gbc")]
pub use session::{Event, Session};
#[cfg(feature = "gbc")]
pub use stack::{AnyCore, Backend, Component, Instrumentation, Stack};
