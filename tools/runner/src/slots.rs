//! A stream's sequence: its token per position and the byte each position holds, with
//! the wiring's slot names resolved to positions once. The tokens sit in a
//! `gbcformer::sequence::Sequence`, so a pass places the positions set since the last one
//! and writes back the positions it decoded again.

use gbcformer::engine::Engine;
use gbcformer::llama::{tok, Config, Denoiser};
pub use gbcformer::sequence::Positions;
use gbcformer::sequence::Sequence;
use gbcformer::wiring::{Field, Flag};

/// A field resolved to positions.
#[derive(Clone, Debug)]
pub struct At {
    positions: Vec<usize>,
    shift: u8,
    mask: u64,
    bias: i64,
}

/// A flag resolved to its position.
#[derive(Clone, Copy, Debug)]
pub struct Bit {
    position: usize,
    mask: u8,
}

/// One stream: its engine, the tokens of its sequence and the byte at each position. Every
/// output holds its mask token but while `show` places its decoded token; `bytes` keeps an
/// output's last decoded byte.
pub struct Seq {
    pub engine: Engine,
    pub cfg: Config,
    tokens: Sequence,
    pub bytes: Vec<u8>,
    names: Vec<String>,
    byte_of: tok::Bytes,
}

impl Bit {
    pub fn position(self) -> usize {
        self.position
    }

    pub fn mask(self) -> u8 {
        self.mask
    }
}

impl Seq {
    pub fn new(engine: Engine) -> Seq {
        let cfg = engine.cfg().clone();
        let names = engine.slots().to_vec();
        let tokens = (0..cfg.seq()).map(|p| if p < cfg.n_inputs { tok::input(&cfg, p, 0) } else { tok::output_mask(&cfg, p) }).collect();
        Seq { tokens: Sequence::new(tokens), bytes: vec![0; cfg.seq()], byte_of: tok::Bytes::new(&cfg), names, engine, cfg }
    }

    pub fn n_inputs(&self) -> usize {
        self.cfg.n_inputs
    }

    /// The position of input slot `name`.
    pub fn input(&self, name: &str) -> usize {
        self.names[..self.cfg.n_inputs].iter().position(|n| n == name).unwrap_or_else(|| panic!("the wiring names an input {name} the stream lacks"))
    }

    /// The position of output slot `name`.
    pub fn output(&self, name: &str) -> usize {
        self.cfg.n_inputs + self.names[self.cfg.n_inputs..].iter().position(|n| n == name).unwrap_or_else(|| panic!("the wiring names an output {name} the stream lacks"))
    }

    pub fn outputs(&self, names: &[String]) -> Vec<usize> {
        names.iter().map(|n| self.output(n)).collect()
    }

    /// The output slots `names` as the list a pass decodes.
    pub fn pass(&self, names: &[String]) -> Positions {
        Positions::new(self.outputs(names))
    }

    pub fn field(&self, f: &Field) -> At {
        At { positions: f.slots.iter().map(|s| self.output(s)).collect(), shift: f.shift, mask: f.mask, bias: f.bias }
    }

    pub fn flag(&self, f: &Flag) -> Bit {
        Bit { position: self.output(&f.slot), mask: f.mask }
    }

    /// A flag on an input slot.
    pub fn input_flag(&self, f: &Flag) -> Bit {
        Bit { position: self.input(&f.slot), mask: f.mask }
    }

    pub fn get(&self, at: &At) -> u64 {
        let raw = at.positions.iter().rev().fold(0u64, |v, &p| v << 8 | self.bytes[p] as u64);
        ((raw >> at.shift & at.mask) as i64 + at.bias) as u64
    }

    pub fn is(&self, b: Bit) -> bool {
        self.bytes[b.position] & b.mask != 0
    }

    pub fn set_input(&mut self, p: usize, v: u8) {
        self.bytes[p] = v;
        self.tokens.set(p, tok::input(&self.cfg, p, v));
    }

    pub fn mask_input(&mut self, p: usize) {
        self.bytes[p] = 0;
        self.tokens.set(p, tok::input_mask(&self.cfg, p));
    }

    /// One pass: decodes `positions`, leaving each one's byte in `bytes`. A position the
    /// pass did not decode again keeps the byte of its last decode.
    pub fn decode(&mut self, positions: &Positions) {
        self.engine.fill_seq(&mut self.tokens, positions);
        let decoded = self.tokens.decoded();
        for &p in self.tokens.fresh() {
            let p = p as usize;
            let Some(v) = self.byte_of.of(decoded[p]) else { panic!("position {p} ({}) decoded the token {}, not a byte", self.names[p], decoded[p]) };
            self.bytes[p] = v;
        }
    }

    /// Places at each of `positions` the token it last decoded to, for a pass that reads
    /// an earlier pass's outputs at their slots.
    pub fn show(&mut self, positions: &[usize]) {
        for &p in positions {
            self.tokens.set(p, self.tokens.decoded()[p]);
        }
    }

    /// The mask token back at each of `positions`.
    pub fn hide(&mut self, positions: &[usize]) {
        for &p in positions {
            self.tokens.set(p, tok::output_mask(&self.cfg, p));
        }
    }

    /// Writes into `out` the sequence as it stands after the pass over `positions`: its
    /// tokens, each of `positions` holding the token it decoded to.
    pub fn finished(&self, positions: &[usize], out: &mut Vec<usize>) {
        out.clear();
        out.extend_from_slice(self.tokens.tokens());
        for &p in positions {
            out[p] = self.tokens.decoded()[p];
        }
    }
}
