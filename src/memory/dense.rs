//! The dense oracle for a memory checkpoint: the model run by `forward_cached` over a
//! KV cache of one cell per 8-byte line (`docs/stock-llama.md` §4, §10). A write is a
//! record group decoded through the model, `REC WP0a..c WP1a..c WL00..WL07 CELL`, whose
//! cell token's key and value replace the line's cell. Before storing them, `keep`
//! checks that every memory head agrees on them and that they are the formula's. A read is a reader
//! group, `REC WP0a..c WP1a..c (WL0 masks) RD0..RD7`, whose reader tokens decode the
//! destination's line with the source's byte written over it.

use super::attn::{key, Kind, READ_AT};
use super::layout::{LINE, LINE_BITS, VOID};
use crate::llama::{line_of_value, tok, Config, Denoiser, Fam, KvCache, MemoryDenoiser, Model};

pub struct DenseMemory {
    pub model: Model,
    pub cache: KvCache,
}

/// The tokens of a write record, `REC WP0a..c WP1a..c WL0..WL7 CELL`.
pub const RECORD: usize = 8 + LINE;

/// The tokens of the record groups, for a checkpoint's record slots.
pub struct Records<'a> {
    pub cfg: &'a Config,
    pub slots: &'a tok::RecordSlots,
}

impl Records<'_> {
    fn addr(&self, w: u8, addr: u32) -> [usize; 3] {
        std::array::from_fn(|b| tok::family(self.cfg, Fam::Addr { write: w, byte: b as u8 }, (addr >> (8 * b)) as u8))
    }

    /// A write record that puts `line` at the line holding byte address `addr`. Its last
    /// token, the cell, is the one the cache keeps.
    pub fn write(&self, addr: u32, line: [u8; LINE]) -> Vec<usize> {
        self.write_tokens(addr, line).to_vec()
    }

    /// `write` as an array.
    pub fn write_tokens(&self, addr: u32, line: [u8; LINE]) -> [usize; RECORD] {
        let (a, b) = (self.addr(0, addr), self.addr(1, 0));
        std::array::from_fn(|j| match j {
            0 => tok::rec(self.cfg),
            1..=3 => a[j - 1],
            4..=6 => b[j - 4],
            Self::CELL => tok::cell(self.cfg),
            _ => tok::family(self.cfg, Fam::Line { byte: (j - 7) as u8 }, line[j - 7]),
        })
    }

    /// The position of a write record's cell.
    pub const CELL: usize = 7 + LINE;

    /// A reader record: its last `LINE` tokens decode to the line at `dest` with the byte
    /// at `source` written over byte `dest mod LINE`.
    pub fn reader(&self, dest: u32, source: u32) -> Vec<usize> {
        let mut t = vec![tok::rec(self.cfg)];
        t.extend(self.addr(0, dest));
        t.extend(self.addr(1, source));
        t.extend(self.slots.line[0].iter().map(|&p| tok::output_mask(self.cfg, p)));
        t.extend((0..LINE).map(|k| tok::rd(self.cfg, k)));
        t
    }

    /// The line a decoded reader record spells.
    pub fn read_back(&self, out: &[usize]) -> [u8; LINE] {
        std::array::from_fn(|k| match tok::family_value(self.cfg, out[7 + LINE + k]) {
            Some((Fam::Line { byte }, v)) if byte as usize == k => v,
            other => panic!("reader token {k} decoded {other:?} (token {})", out[7 + LINE + k]),
        })
    }
}

impl DenseMemory {
    pub fn new(model: Model) -> DenseMemory {
        let spec = model.memory.as_ref().expect("a memory checkpoint");
        let heads = spec.heads.clone();
        let mut dm = DenseMemory { model, cache: KvCache::new(heads) };
        // VOID's first line goes in before any other record, because an off column reads
        // it (`docs/hardware-shaped.md` §20), at every record's own positions too.
        dm.push(VOID, [0; LINE]);
        dm
    }

    pub fn records(&self) -> Records<'_> {
        Records { cfg: &self.model.cfg, slots: &self.model.memory.as_ref().unwrap().slots }
    }

    /// Keeps `line` at `addr`'s line through the model: one pass over its write record,
    /// whose cell token every memory head must agree on and the formula must give.
    pub fn push(&mut self, addr: u32, line: [u8; LINE]) {
        let toks = self.records().write(addr, line);
        let n = addr >> LINE_BITS;
        assert!(n == VOID >> LINE_BITS || self.cache.get(VOID >> LINE_BITS).is_some(), "the record of {addr:#08x} decodes before VOID's cell: its off columns would read another line");
        // The runtime's eviction. The record's cell is then the line's only cell, so no
        // head finds two.
        self.cache.remove(n);
        let (_, group) = self.model.forward_cached(&self.cache, &toks, READ_AT);
        let cfg = &self.model.cfg;
        let heads = &self.cache.heads;
        let (k, v) = group.head(cfg, heads[0], Records::CELL);
        for &h in &heads[1..] {
            let (k2, v2) = group.head(cfg, h, Records::CELL);
            assert_eq!((k.map(f32::to_bits), v.map(f32::to_bits)), (k2.map(f32::to_bits), v2.map(f32::to_bits)), "memory heads {:?} and {h:?} disagree on the record of {addr:#08x}", heads[0]);
        }
        assert_eq!(k.map(f32::to_bits), key(Kind::Cell, n, READ_AT).map(f32::to_bits), "the record of {addr:#08x} keys away from the formula");
        assert_eq!(line_of_value(&v), line, "the record of {addr:#08x} holds another line");
        self.cache.set(n, line);
    }

    /// The line at `dest` with the byte at `source` over byte `dest mod LINE`, read
    /// through the model by a reader record.
    pub fn reader(&self, dest: u32, source: u32) -> [u8; LINE] {
        let r = self.records();
        let (out, _) = self.model.forward_cached(&self.cache, &r.reader(dest, source), READ_AT);
        r.read_back(&out)
    }

    /// Reads `addr` through the model, by a reader record whose source is its destination.
    pub fn read(&self, addr: u32) -> u8 {
        self.reader(addr, addr)[addr as usize % LINE]
    }
}

impl Denoiser for DenseMemory {
    fn cfg(&self) -> &Config {
        &self.model.cfg
    }
    fn slots(&self) -> &[String] {
        &self.model.slots
    }
    fn fill_into(&mut self, tokens: &[usize], positions: &[usize], out: &mut Vec<usize>) {
        let (all, _) = self.model.forward_cached(&self.cache, tokens, READ_AT);
        out.clear();
        out.extend(positions.iter().map(|&p| all[p]));
    }
}

impl MemoryDenoiser for DenseMemory {
    fn cell(&mut self, addr: u32) -> u8 {
        self.cache.get(addr >> LINE_BITS).expect("every line has a cell")[addr as usize % LINE]
    }

    fn keep_host(&mut self, addr: u32, byte: u8) {
        assert!(super::layout::is_host(addr), "a host record outside the HOST block: {addr:#08x}");
        // In the host's own block, the rest of the line holds the host's earlier records.
        let mut line = self.cache.get(addr >> LINE_BITS).expect("the HOST block has cells");
        line[addr as usize % LINE] = byte;
        self.push(addr, line);
    }

    /// Each kept write of the finished group `tokens` becomes its record group, decoded
    /// through the model.
    fn keep(&mut self, tokens: &[usize], writes: &[u8]) {
        let cfg = self.model.cfg.clone();
        let slots = self.model.memory.as_ref().unwrap().slots;
        for r in tok::records(&cfg, &slots, tokens, writes) {
            self.push(r.addr, r.line);
        }
    }
}
