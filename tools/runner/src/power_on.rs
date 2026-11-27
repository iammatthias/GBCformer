//! A run from power-on, built from the ROM and the boot ROM alone: the console the boot
//! ROM's length names, the board the ROM's header bytes and length name, both tables of
//! the wiring (`wiring::PowerOn`).

use gbcformer::memory::image::Image;
use gbcformer::wiring::{Board, PowerOn, Snapshot, Sparse};

fn image(s: &Sparse, lengths: [usize; 3]) -> (Vec<u8>, Vec<bool>, [usize; 3]) {
    let mut bytes = vec![0u8; s.len];
    let mut present = vec![false; s.len];
    for &(p, b) in &s.bytes {
        bytes[p as usize] = b;
    }
    for &(a, b) in &s.present {
        present[a as usize..b as usize].fill(true);
    }
    (bytes, present, lengths)
}

/// The snapshot, the memory, the lines stream's own memory and the two frames at power-on.
pub fn build(w: &PowerOn, rom: &[u8], boot: &[u8]) -> Result<(Snapshot, Image, Image, Image, Vec<u32>), String> {
    let console = w.consoles.iter().find(|c| c.boot_len == boot.len()).ok_or(format!("a boot ROM of {} bytes names no console", boot.len()))?;
    let byte = |at: usize| rom.get(at).copied().ok_or(format!("a ROM of {} bytes has no header", rom.len()));
    let (kind, ram) = (byte(w.header_type)?, byte(w.header_ram)?);
    let board: &Board = console.boards.iter().find(|b| b.kind == kind && b.ram == ram && b.rom_len == rom.len()).ok_or(format!("no board for header bytes {kind}, {ram} and a ROM of {} bytes", rom.len()))?;
    let mut snap = console.snapshot.clone();
    let regs = w.regs as usize;
    let (mut bytes, mut present, mut lengths) = image(&console.memory, console.memory.lengths);
    let rom_at = snap.rom.0 as usize;
    bytes[rom_at..rom_at + rom.len()].copy_from_slice(rom);
    present[rom_at..rom_at + rom.len()].fill(true);
    bytes[regs..regs + board.regs.len()].copy_from_slice(&board.regs);
    let ram_at = w.ram as usize;
    bytes[ram_at..ram_at + board.ram_len].fill(board.ram_fill);
    present[ram_at..ram_at + board.ram_len].fill(true);
    let (boot_at, boot_len) = snap.boot.ok_or("the console's snapshot maps no boot ROM")?;
    bytes[boot_at as usize..boot_at as usize + boot_len].copy_from_slice(&boot_image(w, rom, boot, boot_len)?);
    lengths[0] = rom.len();
    lengths[1] = board.ram_len;
    let memory = Image::from_parts(bytes, present, lengths)?;
    let (mut obytes, opresent, olengths) = image(&console.own, console.own.lengths);
    let own_at = snap.own_rom.0 as usize;
    obytes[own_at..own_at + snap.own_rom.1].copy_from_slice(&rom[..snap.own_rom.1]);
    obytes[regs..regs + board.regs.len()].copy_from_slice(&board.regs);
    let own = Image::from_parts(obytes, opresent, olengths)?;
    let (mut lbytes, lpresent, llengths) = image(&console.len, console.len.lengths);
    lbytes[own_at..own_at + snap.own_rom.1].copy_from_slice(&rom[..snap.own_rom.1]);
    lbytes[regs..regs + board.regs.len()].copy_from_slice(&board.regs);
    let len = Image::from_parts(lbytes, lpresent, llengths)?;
    snap.rom = (snap.rom.0, rom.len());
    snap.save_len = board.save_len;
    snap.clock = board.clock;
    snap.memory.lengths = lengths;
    Ok((snap, memory, own, len, console.frames.clone()))
}

/// The boot image the memory holds while the boot ROM is mapped: ROM bank 0, `len` bytes,
/// with the boot ROM over it in the ranges the console's wiring names.
pub fn boot_image(w: &PowerOn, rom: &[u8], boot: &[u8], len: usize) -> Result<Vec<u8>, String> {
    let console = w.consoles.iter().find(|c| c.boot_len == boot.len()).ok_or(format!("a boot ROM of {} bytes names no console", boot.len()))?;
    let mut image = rom.get(..len).ok_or(format!("a ROM of {} bytes is shorter than its boot image", rom.len()))?.to_vec();
    for &(a, b) in &console.boot_overlay {
        image[a..b].copy_from_slice(&boot[a..b]);
    }
    Ok(image)
}
