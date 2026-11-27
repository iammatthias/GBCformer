//! A stream's view of a checkpoint that holds several: its own slots and tokens.

use crate::llama::{tok, Config, Fam};
use crate::sequence::Stamp;

/// One stream's slots in a checkpoint that holds several: its own slot names, in its
/// own token order (inputs first), and per slot the checkpoint's position for it, or
/// None for a slot the checkpoint left out (never placed and never decoded).
/// `families` is the per-slot token families its tokens use (0 without memory).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamSpec {
    pub slots: Vec<String>,
    pub at: Vec<Option<usize>>,
    pub n_inputs: usize,
    pub families: usize,
    /// The part of an overlay checkpoint the stream runs (`Model::stream`).
    pub part: usize,
}

/// A stream's view of the checkpoint: its own `Config` and slots, its tokens mapped to
/// the checkpoint's and back, and the checkpoint's sequence with every position the
/// stream does not own holding a fixed token (byte 0 at an input, the mask at an
/// output), so every layer-0 head finds its slots. A slot the stream shares with other
/// parts holds the stream's byte for every holder: the checkpoint gathers it once.
pub(super) struct View {
    pub(super) cfg: Config,
    pub(super) slots: Vec<String>,
    pub(super) at: Vec<Option<u32>>,
    /// Per stream token, the checkpoint's token (u32::MAX: none), and the reverse.
    pub(super) to: Vec<u32>,
    pub(super) from: Vec<u32>,
    pub(super) tokens: Vec<usize>,
    pub(super) positions: Vec<usize>,
    /// The stream's tokens and positions as last placed (usize::MAX before the first
    /// place), so a place writes only what changed.
    pub(super) placed: Vec<usize>,
    pub(super) placed_positions: Vec<usize>,
    /// The id of the list `placed_positions` holds (`Positions::id`), or 0 for a list given
    /// without one.
    pub(super) placed_key: u64,
    /// The `positions_id` each named list was mapped under, by its id.
    ids: Vec<(u64, u64)>,
    /// The sequence state `placed` holds (`Sequence::stamp`), or None when the tokens came
    /// from a slice: a sequence at this stamp differs from `placed` only where it lists a
    /// change.
    pub(super) placed_stamp: Option<Stamp>,
    /// Per checkpoint position, the stream's position there (u32::MAX: none).
    pub(super) stream_of: Vec<u32>,
    /// The checkpoint positions the stream holds, in the stream's order.
    pub(super) own: Vec<u32>,
    /// An id for `positions`' content. `place` takes a new id each time it maps a new list,
    /// so a reader that saw this id has an equal list (0 before the first).
    pub(super) positions_id: u64,
    /// The checkpoint positions `place` rewrote since the last layer-0 diff, once each:
    /// every position whose token differs from the last fast pass's is among them.
    pub(super) dirty: Vec<usize>,
    pub(super) is_dirty: Vec<bool>,
}

impl View {
    pub(super) fn new(m: &Config, s: &StreamSpec) -> View {
        let n_out = s.slots.len() - s.n_inputs;
        let vocab = s.n_inputs * 257 + 256 + n_out + if s.families > 0 { 256 * s.families + tok::SPECIAL } else { 0 };
        assert!(s.families == 0 || s.families == m.families, "a stream with memory needs the checkpoint's token families");
        let cfg = Config { n_inputs: s.n_inputs, n_outputs: n_out, families: s.families, vocab, ..m.clone() };
        let at: Vec<Option<u32>> = s.at.iter().map(|q| q.map(|q| q as u32)).collect();
        let mut to = vec![u32::MAX; vocab];
        for (j, q) in at[..s.n_inputs].iter().enumerate() {
            let Some(q) = q.map(|q| q as usize) else { continue };
            for v in 0..=255u8 {
                to[tok::input(&cfg, j, v)] = tok::input(m, q, v) as u32;
            }
            to[tok::input_mask(&cfg, j)] = tok::input_mask(m, q) as u32;
        }
        for v in 0..=255u8 {
            to[tok::output(&cfg, v)] = tok::output(m, v) as u32;
        }
        for (p, q) in at.iter().enumerate().skip(s.n_inputs) {
            if let Some(q) = q {
                to[tok::output_mask(&cfg, p)] = tok::output_mask(m, *q as usize) as u32;
            }
        }
        if s.families > 0 {
            for f in Fam::ALL {
                for v in 0..=255u8 {
                    to[tok::family(&cfg, f, v)] = tok::family(m, f, v) as u32;
                }
            }
            for i in 0..tok::SPECIAL {
                to[tok::rec(&cfg) + i] = (tok::rec(m) + i) as u32;
            }
        }
        let mut from = vec![u32::MAX; m.vocab];
        for (t, &g) in to.iter().enumerate().filter(|(_, g)| **g != u32::MAX) {
            from[g as usize] = t as u32;
        }
        let tokens = (0..m.seq()).map(|q| if q < m.n_inputs { tok::input(m, q, 0) } else { tok::output_mask(m, q) }).collect();
        let seq = m.seq();
        let mut stream_of = vec![u32::MAX; seq];
        for (p, q) in at.iter().enumerate() {
            if let Some(q) = q {
                stream_of[*q as usize] = p as u32;
            }
        }
        let own = at.iter().flatten().copied().collect();
        View { own, cfg, slots: s.slots.clone(), placed: vec![usize::MAX; at.len()], at, to, from, tokens, positions: Vec::new(), placed_positions: Vec::new(), placed_key: 0, ids: Vec::new(), placed_stamp: None, stream_of, positions_id: 0, dirty: Vec::new(), is_dirty: vec![false; seq] }
    }

    /// Maps `positions`, the list `key` names (0: a list with no id), unless it is the
    /// list mapped last. A named list keeps one `positions_id` however often the passes
    /// come back to it.
    #[inline(always)]
    pub(super) fn place_positions(&mut self, positions: &[usize], key: u64) {
        if key != 0 && key == self.placed_key {
            return;
        }
        self.placed_key = key;
        let known = self.ids.iter().find(|&&(k, _)| k == key && key != 0).map(|&(_, id)| id);
        if self.placed_positions != positions {
            let mut mapped = std::mem::take(&mut self.positions);
            mapped.clear();
            mapped.extend(positions.iter().map(|&p| self.at_of(p)));
            self.positions = mapped;
            self.placed_positions.clear();
            self.placed_positions.extend_from_slice(positions);
            self.positions_id = known.unwrap_or_else(super::fresh_positions_id);
        }
        if key != 0 && known.is_none() {
            // A device names a handful of lists. Past that every id is forgotten, and a
            // list mapped again takes a new one.
            if self.ids.len() >= 64 {
                self.ids.clear();
            }
            self.ids.push((key, self.positions_id));
        }
    }

    /// The checkpoint position of stream position `p`.
    pub(super) fn at_of(&self, p: usize) -> usize {
        self.at[p].unwrap_or_else(|| panic!("stream position {p} ({}) is not in the checkpoint", self.slots[p])) as usize
    }

    /// Writes into the checkpoint's sequence the tokens of a sequence that differs from the
    /// last placement only at `changed`. With `verify`, the whole sequence is compared
    /// too, and a difference elsewhere stops the run.
    #[inline(always)]
    pub(super) fn place_changed(&mut self, tokens: &[usize], changed: &[u32], verify: bool) {
        assert_eq!(tokens.len(), self.placed.len(), "a stream's sequence keeps its length");
        for &p in changed {
            self.place_one(p as usize, tokens[p as usize]);
        }
        if verify && let Some(p) = (0..tokens.len()).find(|&p| self.placed[p] != tokens[p]) {
            panic!("stream position {p} ({}) holds token {}, placed as {}, and the sequence lists no change there", self.slots[p], tokens[p], self.placed[p]);
        }
    }

    /// Writes the stream's tokens into the checkpoint's sequence. Every token is compared
    /// with the last placement, only those that changed are mapped, and the checkpoint
    /// positions they rewrite join `dirty`. The placement then names no sequence's stamp.
    pub(super) fn place_tokens(&mut self, tokens: &[usize]) {
        assert_eq!(tokens.len(), self.placed.len(), "a stream's sequence keeps its length");
        self.placed_stamp = None;
        // Most tokens repeat, so groups of four are compared whole first.
        const GROUP: usize = 4;
        let whole = tokens.len() / GROUP * GROUP;
        for p in (0..whole).step_by(GROUP) {
            let (new, old): (&[usize; GROUP], &[usize; GROUP]) = (tokens[p..p + GROUP].try_into().unwrap(), self.placed[p..p + GROUP].try_into().unwrap());
            if new != old {
                for q in p..p + GROUP {
                    self.place_one(q, tokens[q]);
                }
            }
        }
        for q in whole..tokens.len() {
            self.place_one(q, tokens[q]);
        }
    }

    fn place_one(&mut self, p: usize, t: usize) {
        if self.placed[p] == t {
            return;
        }
        self.placed[p] = t;
        if let Some(q) = self.at[p] {
            let (q, g) = (q as usize, self.to[t] as usize);
            if self.tokens[q] != g {
                self.tokens[q] = g;
                if !std::mem::replace(&mut self.is_dirty[q], true) {
                    self.dirty.push(q);
                }
            }
        }
    }

    /// Empties `dirty`.
    #[inline]
    pub(super) fn clean(&mut self) {
        for &q in &self.dirty {
            self.is_dirty[q] = false;
        }
        self.dirty.clear();
    }

    pub(super) fn back(&self, t: usize) -> usize {
        match self.from[t] {
            u32::MAX => usize::MAX,
            l => l as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No checkpoint holds a token above 65,535 since the parts shared the input slots
    /// (27,511 tokens), so no run exercises one. A checkpoint of 300 input slots does: its
    /// last input slots' bytes, its output tokens, its families and its record tokens all
    /// sit above 65,535, and a stream's view maps each of its tokens there and back.
    #[test]
    fn a_views_tokens_map_above_65535_and_back() {
        let families = Fam::ALL.len();
        let (n_inputs, n_outputs) = (300, 12);
        let vocab = n_inputs * 257 + 256 + n_outputs + 256 * families + tok::SPECIAL;
        let m = Config { hidden: 64, intermediate: 64, layers: 1, heads: 1, kv_heads: 1, vocab, n_inputs, n_outputs, n_const: 2, families };
        // The stream holds the checkpoint's last 4 input slots and last 3 output slots.
        let at: Vec<Option<usize>> = (296..300).chain(309..312).map(Some).collect();
        let s = StreamSpec { slots: (0..at.len()).map(|i| format!("s{i}")).collect(), at, n_inputs: 4, families, part: 0 };
        let v = View::new(&m, &s);
        assert_eq!(v.cfg.vocab, 4 * 257 + 256 + 3 + 256 * families + tok::SPECIAL);
        assert_eq!(v.to[tok::input(&v.cfg, 0, 7)], 296 * 256 + 7);
        assert_eq!(v.to[tok::input_mask(&v.cfg, 3)] as usize, 300 * 256 + 299);
        assert_eq!(v.to[tok::output(&v.cfg, 255)] as usize, 300 * 257 + 255);
        assert_eq!(v.to[tok::output_mask(&v.cfg, 6)] as usize, 300 * 257 + 256 + 11);
        assert_eq!(v.to[tok::rec(&v.cfg)] as usize, 300 * 257 + 256 + 12 + 256 * families);
        assert_eq!(v.to[tok::rec(&v.cfg) + tok::SPECIAL - 1] as usize, vocab - 1);
        for (t, &g) in v.to.iter().enumerate() {
            assert!(g as usize > 65_535 && (g as usize) < vocab, "stream token {t} maps to {g}");
            assert_eq!(v.from[g as usize] as usize, t, "checkpoint token {g} maps back");
        }
        assert_eq!(v.from.iter().filter(|&&t| t != u32::MAX).count(), v.cfg.vocab);
    }
}
