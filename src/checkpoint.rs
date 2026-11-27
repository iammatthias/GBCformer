//! A compiled checkpoint run as a device: the slot ABI names its token positions, and
//! [`CheckpointDevice`] owns the masked-diffusion protocol over them.
//!
//! One step is up to two passes over the same sequence. Pass 1 holds the early inputs,
//! masks the late ones and every output, and decodes every output slot. The request
//! slots never depend on the late inputs, so they are final; when the request needs no
//! late data the result slots are final too. Otherwise the result slots are masked
//! again, the host fills the late inputs, and pass 2 decodes the result slots.

use crate::llama::{tok, Config, Denoiser};
use crate::sequence::{Positions, Sequence};
use std::marker::PhantomData;
use std::ops::Range;

/// The slot layout of one checkpoint and the typed view of its bytes. Slot order is
/// the token order: `INPUTS`, then `REQUEST`, then `RESULT`.
pub trait SlotAbi {
    const INPUTS: &'static [&'static str];
    const REQUEST: &'static [&'static str];
    const RESULT: &'static [&'static str];
    /// Input slots the host fills between passes. Every request slot must be
    /// independent of them.
    const LATE: Range<usize>;
    type State;
    type Inputs;
    /// The bytes of the late slots, in `LATE` order.
    type Late: AsRef<[u8]>;
    type Request;
    type Outcome;
    /// Writes the early input slots into `slots` (`INPUTS.len()` bytes). The late
    /// slots are ignored.
    fn encode(state: &Self::State, inputs: &Self::Inputs, slots: &mut [u8]);
    fn decode_request(bytes: &[u8]) -> Self::Request;
    fn decode_outcome(bytes: &[u8]) -> Self::Outcome;
    /// Whether the outcome depends on the late inputs this request asks for.
    fn needs_late(request: &Self::Request) -> bool;
}

/// A checkpoint with slot layout `A`, stepped by two-pass masked diffusion.
pub struct CheckpointDevice<A: SlotAbi, M: Denoiser> {
    model: M,
    cfg: Config,
    /// The sequence every pass reads: the inputs, and at every output slot its mask
    /// token, but for the request's decoded tokens during pass 2.
    tokens: Sequence,
    /// The positions pass 1 decodes (every output) and pass 2 (the result slots).
    passes: [Positions; 2],
    /// The byte each output slot last decoded to, from the first output on.
    bytes: Vec<u8>,
    slots: Vec<u8>,
    /// The outcome pass 1 settled, when the request needed no late data.
    settled: Option<A::Outcome>,
    pub one_pass_steps: u64,
    pub two_pass_steps: u64,
    abi: PhantomData<A>,
}

impl<A: SlotAbi, M: Denoiser> CheckpointDevice<A, M> {
    pub fn new(model: M) -> Self {
        let expected: Vec<&str> = A::INPUTS.iter().chain(A::REQUEST).chain(A::RESULT).copied().collect();
        assert_eq!(model.slots(), expected, "the checkpoint's slots are not this ABI");
        let cfg = model.cfg().clone();
        assert_eq!(cfg.n_inputs, A::INPUTS.len());
        assert!(A::LATE.end <= cfg.n_inputs, "late slots {:?} beyond the {} inputs", A::LATE, cfg.n_inputs);
        let tokens = (0..cfg.seq()).map(|p| if p < cfg.n_inputs { 0 } else { tok::output_mask(&cfg, p) }).collect();
        CheckpointDevice {
            tokens: Sequence::new(tokens),
            passes: [Positions::new((cfg.n_inputs..cfg.seq()).collect()), Positions::new((cfg.n_inputs + A::REQUEST.len()..cfg.seq()).collect())],
            bytes: vec![0; cfg.n_outputs],
            slots: vec![0; cfg.n_inputs],
            settled: None,
            one_pass_steps: 0,
            two_pass_steps: 0,
            abi: PhantomData,
            cfg,
            model,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    fn request_positions(&self) -> Range<usize> {
        self.cfg.n_inputs..self.cfg.n_inputs + A::REQUEST.len()
    }

    /// Pass 1: decodes the request, and the outcome too when it needs no late data.
    pub fn request(&mut self, state: &A::State, inputs: &A::Inputs) -> A::Request {
        A::encode(state, inputs, &mut self.slots);
        for (j, &b) in self.slots.iter().enumerate() {
            self.tokens.set(j, if A::LATE.contains(&j) { tok::input_mask(&self.cfg, j) } else { tok::input(&self.cfg, j, b) });
        }
        self.decode(0);
        let request = A::decode_request(&self.bytes[..A::REQUEST.len()]);
        self.settled = (!A::needs_late(&request)).then(|| A::decode_outcome(&self.bytes[A::REQUEST.len()..]));
        request
    }

    /// Pass 2, or the outcome pass 1 settled.
    pub fn execute(&mut self, late: &A::Late) -> A::Outcome {
        if let Some(out) = self.settled.take() {
            self.one_pass_steps += 1;
            return out;
        }
        self.two_pass_steps += 1;
        for (j, &b) in A::LATE.zip(late.as_ref()) {
            self.tokens.set(j, tok::input(&self.cfg, j, b));
        }
        // Pass 2's sequence holds the request's decoded tokens at their slots, as pass 1
        // left them; outputs read no output, so they change no token it decodes.
        for p in self.request_positions() {
            self.tokens.set(p, self.tokens.decoded()[p]);
        }
        self.decode(1);
        for p in self.request_positions() {
            self.tokens.set(p, tok::output_mask(&self.cfg, p));
        }
        A::decode_outcome(&self.bytes[A::REQUEST.len()..])
    }

    /// Runs pass `k` and leaves its bytes in `self.bytes`. A position the pass did not
    /// decode again (`Sequence::fresh`) keeps the byte of its last decode.
    fn decode(&mut self, k: usize) {
        self.model.fill_seq(&mut self.tokens, &self.passes[k]);
        let decoded = self.tokens.decoded();
        for &p in self.tokens.fresh() {
            let p = p as usize;
            let v = tok::output_value(&self.cfg, decoded[p]).unwrap_or_else(|| panic!("position {p} ({}) decoded non-byte token {}", self.model.slots()[p], decoded[p]));
            self.bytes[p - self.cfg.n_inputs] = v;
        }
    }
}
