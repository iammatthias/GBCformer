//! The memory checkpoint's records: the index and the KV cache the dense oracle reads,
//! cartridge insertion, write and copy records, and their oracles. The memory holds one
//! cell per 8-byte line (`docs/stock-llama.md` §4, §10), and a write record replaces its
//! line's cell with the whole line after the write.

use super::cells::{Cells, RowBuilder};
use super::Engine;
use crate::llama::{line_of_value, tok, KvCache, MemoryDenoiser};
use crate::memory::attn::{key, Kind as RecKind, READ_AT};
use crate::memory::dense::Records;
use crate::memory::image::Image;
use crate::memory::layout::{self as mem_layout, LINE, LINE_BITS};
use crate::memory::CopyFrom;
#[cfg(feature = "gbc")]
use std::collections::HashMap;

/// Write records another stream follows, oldest first, as the engine holding the memory
/// keeps them, one per byte of a kept line that changed.
pub type VideoRecords = std::sync::Arc<std::sync::Mutex<Vec<(u32, u8)>>>;

/// What a follower takes: the byte at each of `cells` when it changed, then every changed
/// byte of a line in one of `ranges`. The caller names both; the engine holds no address.
#[derive(Clone, Debug)]
pub struct Video {
    pub records: VideoRecords,
    pub cells: Vec<u32>,
    pub ranges: Vec<(u32, u32)>,
}

/// A reader record's copy heads: the one at the destination's gathered address, and the
/// one at the source's (`MemorySpec::copy_heads`).
pub const DEST: usize = 0;
pub const SOURCE: usize = 1;

/// The memory the engine holds for a memory checkpoint: the index (the bytes, which the
/// margin argument proves are every memory head's attention result, a line at a time)
/// and the KV cache the dense oracle reads, the same lines as cells.
pub struct Memory {
    pub image: Image,
    pub kv: KvCache,
    /// The cells as the runtime's cache holds them. `cells::Cells::read` serves every read
    /// the model makes from these rows.
    pub cells: Cells,
    /// Run the dense oracle on every `n`th step and kept record (0: never).
    pub oracle_every: u64,
    /// Where write records to VRAM and OAM also go, in order: the PPU checkpoint's
    /// image follows the cache through them.
    pub video: Option<Video>,
}

impl Engine {
    /// Inserts a cartridge: every line of the image becomes a cell. `Params::check` checks
    /// the margins for every cell position the layout can hold.
    pub fn insert(&mut self, image: Image) {
        let spec = self.dense.memory.as_ref().expect("a memory checkpoint");
        spec.params.check();
        let old = self.memory.take();
        let builder = match &old {
            Some(m) => m.cells.builder.clone(),
            None => std::sync::Arc::new(RowBuilder::new(&self.dense).unwrap_or_else(|e| panic!("the cells' rows: {e}"))),
        };
        let video = old.and_then(|m| m.video);
        let mut kv = KvCache::new(spec.heads.clone());
        kv.fill(&image);
        let t = std::time::Instant::now();
        let mut cells = Cells::new(builder);
        for n in (0..mem_layout::LINES).filter(|&n| image.has_line(n)) {
            cells.fill(n << LINE_BITS, image.line(n));
        }
        self.stats.cell_rows += cells.len() as u64;
        self.stats.cell_insert_ns += t.elapsed().as_nanos() as u64;
        #[cfg(feature = "record")]
        if let Some(e) = self.external.as_mut() {
            e.insert(&image);
        }
        self.memory = Some(Memory { image, kv, cells, oracle_every: 0, video });
        self.state.valid = false;
        self.replay = replay_recorder(self);
    }

    /// Keeps one write record: the line of byte address `addr` (which the record's
    /// address tokens spell) in the index and its cell.
    pub(super) fn write_record(&mut self, addr: u32, line: [u8; LINE]) {
        if let Some(r) = self.replay.as_mut() {
            r.write(addr, line);
        }
        let n = addr >> LINE_BITS;
        let mem = self.memory.as_mut().unwrap();
        let before = mem.image.generation();
        let old = mem.image.line(n);
        mem.image.write_line(n, line);
        // The heads reading line `n` read again next pass. Only while the state saw every
        // change before this one; otherwise `memory_reads` finds the generations apart
        // and every head reads.
        let s = &mut self.state;
        if s.seen_gen == before {
            s.seen_gen = mem.image.generation();
            for (hi, &a) in s.head_addr.iter().enumerate() {
                if a == n {
                    s.head_hit[hi / 64] |= 1 << (hi % 64);
                }
            }
        }
        let base = n << LINE_BITS;
        if let Some(v) = &mem.video {
            let cells = v.cells.iter().filter(|&&c| c & !(LINE as u32 - 1) == base && old[c as usize % LINE] != line[c as usize % LINE]).map(|&c| (c, line[c as usize % LINE]));
            let bytes = v.ranges.iter().any(|&(a, b)| (a..b).contains(&base)).then(|| (0..LINE).filter(|&k| old[k] != line[k]).map(|k| (base + k as u32, line[k]))).into_iter().flatten();
            let new: Vec<(u32, u8)> = cells.chain(bytes).collect();
            if !new.is_empty() {
                v.records.lock().expect("the video record queue").extend(new);
            }
        }
        mem.kv.set(n, line);
        self.stats.cell_rows += mem.cells.put(addr, line) as u64;
        (self.stats.cell_memo, self.stats.cell_memo_checked, self.stats.cell_memo_len) = (mem.cells.memo_hits, mem.cells.memo_checked, mem.cells.memo_len());
        self.stats.records += 1;
        #[cfg(feature = "record")]
        if let Some(e) = self.external.as_mut() {
            e.write(addr, line);
        }
    }

    /// Keeps `byte` at `addr` as a write record of its line. A reader of another
    /// checkpoint's records (the PPU's image) follows them this way, a byte at a time.
    pub fn keep_write(&mut self, addr: u32, byte: u8) {
        let mut line = self.read_line(DEST, addr >> LINE_BITS);
        line[addr as usize % LINE] = byte;
        self.write_record(addr, line);
    }

    /// Line `n` as a reader record's copy head `head` ([`DEST`] or [`SOURCE`]) reads it,
    /// by the attention of its lattice query over the cells' rows (`Cells::read`).
    pub fn read_line(&mut self, head: usize, n: u32) -> [u8; LINE] {
        let mut layer = self.dense.memory.as_ref().expect("a memory checkpoint").copy_heads[head].0;
        let cells = &self.memory.as_ref().expect("a memory checkpoint").cells;
        // A stream of the one model shallower than the GBC CPU's copy layer (the stepper's)
        // reads the line at its first memory layer: every memory layer's rows hold the same
        // line, and `Cells::read` still requires one row's softmax.
        if layer >= self.dense.cfg.layers {
            layer = *cells.builder.memory_layers.first().expect("a stream with memory heads");
        }
        let (line, served) = cells.read(layer, n).unwrap_or_else(|| panic!("line {n:#07x}: the softmax over the rows at layer {layer} is not one row"));
        self.stats.count_read(served, cells.scan_all);
        line
    }

    fn records(&self) -> Records<'_> {
        Records { cfg: &self.dense.cfg, slots: &self.dense.memory.as_ref().unwrap().slots }
    }

    /// One byte a DMA copied, in copy mode: a reader record `REC WP0a..c(dest)
    /// WP1a..c(source) RD0..RD7`, whose reader tokens decode the destination's line with
    /// the source's byte over it (served from the index), then the write record of that
    /// line, like a GBC CPU write. A host-served source is first kept as a host record in
    /// the HOST block, then read from there. Returns the byte. When `keep` samples the
    /// write record for the oracle, the reader record runs through the dense pass too.
    pub fn copy(&mut self, from: CopyFrom, dest: u32) -> u8 {
        let source = match from {
            CopyFrom::Memory(source) => source,
            CopyFrom::Host(byte) => {
                self.stats.host_copies += 1;
                let at = mem_layout::HOST | (dest & 0x7F);
                self.keep_host(at, byte);
                at
            }
        };
        let byte = self.read_line(SOURCE, source >> LINE_BITS)[source as usize % LINE];
        let mut line = self.read_line(DEST, dest >> LINE_BITS);
        line[dest as usize % LINE] = byte;
        // The runtime's reader record, checked against the index.
        #[cfg(feature = "record")]
        self.external_read(dest, source, line);
        let mem = self.memory.as_ref().expect("a memory checkpoint");
        if mem.oracle_every > 0 && self.stats.records % mem.oracle_every == 0 {
            let r = self.records();
            let (out, _) = self.dense.forward_cached(&mem.kv, &r.reader(dest, source), READ_AT);
            assert_eq!(r.read_back(&out), line, "copy oracle: reader at {source:#08x} into {dest:#08x}");
            self.stats.oracle_checks += 1;
            self.stats.oracle_high_rom += (0x20_0000..mem_layout::CRAM).contains(&source) as u64;
        }
        let rec = tok::Record { write: 0, addr: dest, line };
        if self.oracle_due() {
            self.oracle(&[rec]);
        }
        self.write_record(dest, line);
        self.stats.copies += 1;
        byte
    }

    /// Loads a bus image (SingleStepTests) as write records over the inserted image,
    /// first undoing the previous case's writes the same way.
    #[cfg(feature = "gbc")]
    pub fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        let mem = self.memory.as_mut().ok_or("no cartridge inserted")?;
        let undo: Vec<(u32, u8)> = mem.image.undo_log().iter().rev().copied().collect();
        mem.image.mark();
        for (p, old) in undo {
            self.keep_write(p, old);
        }
        let mem = self.memory.as_mut().unwrap();
        mem.image.mark();
        let b = mem.image.banking();
        let mut cells: HashMap<u32, (u16, u8)> = HashMap::new();
        for &(a, v) in ram {
            let p = mem_layout::translate(&b, a, mem_layout::Access::Read, false);
            if let Some(&(other, w)) = cells.get(&p)
                && w != v
            {
                return Err(format!("{a:#06x}={v:#04x} and {other:#06x}={w:#04x} share the cell {p:#08x}"));
            }
            if let Some(c) = mem_layout::constant(p)
                && c != v
            {
                return Err(format!("{a:#06x}={v:#04x}: the model holds the constant {c:#04x} there ({p:#08x})"));
            }
            cells.insert(p, (a, v));
            let mem = self.memory.as_ref().unwrap();
            if mem.image.has(p) && p as usize >= mem.image.rom_len() && mem_layout::recorded(p) {
                self.keep_write(p, v);
            }
        }
        Ok(())
    }

    /// The record slots in the stream's own sequence and vocabulary.
    fn stream_slots(&self) -> (crate::llama::Config, tok::RecordSlots) {
        match self.view.as_ref() {
            Some(v) => (v.cfg.clone(), self.stream_records.expect("a memory stream has every record slot")),
            None => (self.dense.cfg.clone(), self.dense.memory.as_ref().unwrap().slots),
        }
    }
}

impl MemoryDenoiser for Engine {
    fn check_next(&mut self) {
        self.check_next = self.step_oracle > 0;
    }

    fn load_bus(&mut self, ram: &[(u16, u8)]) -> Result<(), String> {
        #[cfg(feature = "gbc")]
        return Engine::load_bus(self, ram);
        #[cfg(not(feature = "gbc"))]
        return Err(format!("{} bus bytes: a bus image needs the address translation, in the gbc feature", ram.len()));
    }

    fn cell(&mut self, addr: u32) -> u8 {
        self.read_line(DEST, addr >> LINE_BITS)[addr as usize % LINE]
    }

    /// Keeps a record in the host's own block. The rest of the line holds its earlier
    /// host records.
    fn keep_host(&mut self, addr: u32, byte: u8) {
        assert!(mem_layout::is_host(addr), "a host record outside the HOST block: {addr:#08x}");
        let mut line = self.read_line(DEST, addr >> LINE_BITS);
        line[addr as usize % LINE] = byte;
        let rec = tok::Record { write: 0, addr, line };
        if self.oracle_due() {
            self.oracle(&[rec]);
        }
        self.write_record(addr, line);
        self.stats.host_fetches += 1;
    }

    /// Keeps the write records of a finished step: each kept write's line, at the line
    /// its address tokens spell. Every `oracle_every`th one runs its record group
    /// through the dense pass over the cache and compares the cell's key and value
    /// with the formula's bit for bit.
    fn keep(&mut self, tokens: &[usize], writes: &[u8]) {
        let (cfg, slots) = self.stream_slots();
        let kept = tok::records(&cfg, &slots, tokens, writes);
        #[cfg(feature = "record")]
        if self.recorder.is_some() {
            // The kept positions: each kept write's address and line slots.
            let positions: Vec<usize> = writes.iter().flat_map(|&w| { let (addr, line) = slots.of(w); addr.into_iter().chain(line) }).collect();
            let own: Vec<usize> = positions.iter().map(|&p| tokens[p]).collect();
            let v = self.view.as_mut().expect("only a view records");
            v.place_tokens(tokens);
            let (merged, at): (Vec<usize>, Vec<usize>) = (v.tokens.clone(), positions.iter().map(|&p| v.at_of(p)).collect());
            self.record("keep", &merged, &at, &own);
        }
        if self.oracle_due() {
            self.oracle(&kept);
        }
        for r in &kept {
            self.write_record(r.addr, r.line);
        }
    }
}

impl Engine {
    fn oracle_due(&self) -> bool {
        self.memory.as_ref().is_some_and(|m| m.oracle_every > 0 && self.stats.records % m.oracle_every == 0)
    }

    /// The record oracle: each kept line as the write record a stock runtime decodes,
    /// its cell token's key and value against the formula.
    fn oracle(&mut self, kept: &[tok::Record]) {
        let mem = self.memory.as_mut().unwrap();
        let r = Records { cfg: &self.dense.cfg, slots: &self.dense.memory.as_ref().unwrap().slots };
        for (i, rec) in kept.iter().enumerate() {
            // The runtime evicts the line's old cell before it decodes the record, so the
            // record's cell is the only cell for its line.
            let n = rec.line_index();
            let old = mem.kv.get(n);
            mem.kv.remove(n);
            let (_, g) = self.dense.forward_cached(&mem.kv, &r.write(rec.addr, rec.line), READ_AT);
            if let Some(b) = old {
                mem.kv.set(n, b);
            }
            let (k, v) = g.head(&self.dense.cfg, mem.kv.heads[0], Records::CELL);
            assert_eq!(k.map(f32::to_bits), key(RecKind::Cell, n, READ_AT).map(f32::to_bits), "record oracle: key of record {i} ({:#08x})", rec.addr);
            assert_eq!(line_of_value(&v), rec.line, "record oracle: line of record {i}");
        }
        self.stats.oracle_checks += 1;
    }
}

#[cfg(feature = "record")]
impl Engine {
    /// The line a reader record `REC WP0a..c(dest) WP1a..c(source) RD0..RD7` reads. It is
    /// the external runtime's when one is installed, which must equal `index`, the
    /// engine's own (the first difference stops the run, as a lockstep does).
    pub fn external_read(&mut self, dest: u32, source: u32, index: [u8; LINE]) -> [u8; LINE] {
        let Some(got) = self.external.as_mut().and_then(|e| e.read(dest, source)) else { return index };
        assert_eq!(got, index, "the external runtime's reader record of {source:#08x} into {dest:#08x} decoded {got:02x?}, the engine's index holds {index:02x?}");
        got
    }
}

/// A replay recorder for `e`'s stream, when `GBCFORMER_REPLAY_OUT` asks for it.
fn replay_recorder(e: &Engine) -> Option<Box<super::replay::Recorder>> {
    super::replay::Recorder::from_env(e.view.as_ref().map_or(&e.dense.slots, |v| &v.slots))
}
