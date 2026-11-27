//! A device's token sequence as a handle, so a pass works on what changed.
//!
//! [`Sequence`] holds the tokens a device's passes read and the tokens they decoded. Its
//! tokens are private and [`Sequence::set`] is the only write, so the positions it lists
//! as changed are every position that differs from the last pass's tokens. A denoiser that
//! kept what it placed last time places those alone, and writes back only the positions
//! whose decode it ran again. The sequence's stamp names its state at the last pass: a
//! denoiser that holds another stamp places and decodes the whole sequence.

use std::sync::atomic::{AtomicU64, Ordering};

/// A new nonzero id, unique in the process.
pub(crate) fn fresh_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The positions a pass decodes, fixed at construction: equal ids mean equal lists.
#[derive(Clone, Debug)]
pub struct Positions {
    list: Box<[usize]>,
    id: u64,
}

impl Positions {
    pub fn new(list: Vec<usize>) -> Positions {
        Positions { list: list.into(), id: fresh_id() }
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

impl std::ops::Deref for Positions {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.list
    }
}

/// A sequence's state at a pass: which sequence, and how many passes it has had.
pub type Stamp = (u64, u64);

pub struct Sequence {
    tokens: Vec<usize>,
    /// The positions `set` gave another token since the last pass began, once each.
    changed: Vec<u32>,
    is_changed: Vec<bool>,
    /// The changed positions of the pass under way.
    placing: Vec<u32>,
    /// By position, the token its last decode gave.
    decoded: Vec<usize>,
    /// The positions the last pass wrote in `decoded`.
    fresh: Vec<u32>,
    /// A positions-ordered list for a denoiser that fills one (`whole`).
    list: Vec<usize>,
    id: u64,
    passes: u64,
}

/// One pass's side of a sequence: its tokens, the positions that changed since the last
/// pass began, and the decoded store.
pub struct Pass<'a> {
    /// The sequence's stamp from this pass on.
    pub stamp: Stamp,
    pub tokens: &'a [usize],
    pub changed: &'a [u32],
    pub decoded: Decoded<'a>,
}

/// A sequence's decoded tokens, as a pass writes them.
pub struct Decoded<'a> {
    at: &'a mut [usize],
    fresh: &'a mut Vec<u32>,
}

impl Decoded<'_> {
    /// Position `p` decoded to `t` in this pass.
    #[inline(always)]
    pub fn put(&mut self, p: usize, t: usize) {
        self.at[p] = t;
        self.fresh.push(p as u32);
    }

    /// Every position of `positions` decoded in this pass, to `tokens` in their order.
    pub fn put_all(&mut self, positions: &[usize], tokens: impl Iterator<Item = usize>) {
        self.fresh.extend(positions.iter().map(|&p| p as u32));
        let mut written = 0;
        for (&p, t) in positions.iter().zip(tokens) {
            self.at[p] = t;
            written += 1;
        }
        // A pass delegated to an external runtime may return a short list; the positions past
        // it would keep their old tokens while listed as decoded.
        assert!(written == positions.len(), "{written} tokens for {} positions", positions.len());
    }

    /// The token position `p` last decoded to.
    pub fn at(&self, p: usize) -> usize {
        self.at[p]
    }

    /// Test hook: another token at `p`, with no record of the write.
    #[cfg(feature = "testing")]
    pub fn corrupt(&mut self, p: usize, x: usize) {
        self.at[p] ^= x;
    }
}

impl Sequence {
    pub fn new(tokens: Vec<usize>) -> Sequence {
        let n = tokens.len();
        Sequence { tokens, changed: Vec::new(), is_changed: vec![false; n], placing: Vec::new(), decoded: vec![0; n], fresh: Vec::new(), list: Vec::new(), id: fresh_id(), passes: 0 }
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    #[inline(always)]
    pub fn get(&self, p: usize) -> usize {
        self.tokens[p]
    }

    /// Token `t` at position `p`.
    #[inline(always)]
    pub fn set(&mut self, p: usize, t: usize) {
        if self.tokens[p] != t {
            self.tokens[p] = t;
            if !std::mem::replace(&mut self.is_changed[p], true) {
                self.changed.push(p as u32);
            }
        }
    }

    /// Test hook: token `t` at `p` with no record of the change, which `set` never does.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_unrecorded(&mut self, p: usize, t: usize) {
        self.tokens[p] = t;
    }

    /// Test hook: another token in the decoded store at `p`, as no pass leaves it.
    #[cfg(any(test, feature = "testing"))]
    pub fn corrupt_decoded(&mut self, p: usize, x: usize) {
        self.decoded[p] ^= x;
    }

    /// By position, the token its last decode gave: after a pass, the pass's token at each
    /// of its positions.
    pub fn decoded(&self) -> &[usize] {
        &self.decoded
    }

    /// The positions the last pass wrote in `decoded`. A position of the pass that is not
    /// among them holds the token it held before the pass.
    pub fn fresh(&self) -> &[u32] {
        &self.fresh
    }

    pub fn stamp(&self) -> Stamp {
        (self.id, self.passes)
    }

    /// Begins a pass: the stamp moves on, and the changes recorded so far are the pass's.
    pub fn begin(&mut self) -> Pass<'_> {
        self.passes += 1;
        std::mem::swap(&mut self.changed, &mut self.placing);
        self.changed.clear();
        for &p in &self.placing {
            self.is_changed[p as usize] = false;
        }
        self.fresh.clear();
        Pass { stamp: (self.id, self.passes), tokens: &self.tokens, changed: &self.placing, decoded: Decoded { at: &mut self.decoded, fresh: &mut self.fresh } }
    }

    /// A pass by a denoiser that fills a positions-ordered list from the whole sequence:
    /// every position of `positions` is written.
    pub fn whole(&mut self, positions: &[usize], fill: impl FnOnce(&[usize], &mut Vec<usize>)) {
        let mut list = std::mem::take(&mut self.list);
        let mut pass = self.begin();
        fill(pass.tokens, &mut list);
        assert_eq!(list.len(), positions.len(), "a pass decodes each of its positions");
        pass.decoded.put_all(positions, list.iter().copied());
        self.list = list;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_takes_the_positions_set_since_the_last() {
        let mut s = Sequence::new(vec![1, 2, 3, 4]);
        let before = s.stamp();
        s.set(2, 3);
        s.set(1, 9);
        s.set(1, 8);
        s.set(3, 5);
        s.set(3, 4);
        let pass = s.begin();
        // A position set back to its token stays listed: the list is a superset of the differences.
        assert_eq!((pass.tokens, pass.changed), (&[1, 8, 3, 4][..], &[1, 3][..]));
        assert_ne!(s.stamp(), before);
        assert!(s.begin().changed.is_empty());
        s.set(1, 2);
        assert_eq!(s.begin().changed, [1]);
    }

    #[test]
    fn a_whole_pass_writes_every_position() {
        let mut s = Sequence::new(vec![0; 5]);
        s.whole(&[4, 2], |tokens, out| {
            assert_eq!(tokens.len(), 5);
            out.clear();
            out.extend([7, 6]);
        });
        assert_eq!((s.fresh(), s.decoded()[4], s.decoded()[2]), (&[4, 2][..], 7, 6));
        s.whole(&[2], |_, out| *out = vec![5]);
        assert_eq!((s.fresh(), s.decoded()[4], s.decoded()[2]), (&[2][..], 7, 5));
    }
}
