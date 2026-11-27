//! Brackets for the replay bench's cost model: one stage of one pass between two
//! readings of the thread's counters, or a pass's shape counted. A pass runs its
//! bracketed copy (`Engine::fill_stream`, `P` true) only while the probe is armed, so
//! the pass every other caller runs holds no reading and no test of one.

use super::counters::{self, Counts};

/// What an armed pass brackets or counts, a stage by its place in the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The view's placement of the stream's tokens and positions.
    View,
    /// Layer 0: the moved positions and the attention diff.
    Layer0,
    /// The layer-0 changes applied to their wires and routed to their readers.
    Scatter,
    /// Inside a circuit layer's memory heads: the head bucket's drain into the address
    /// bits, the dirty and hit heads read, and the buses their columns moved.
    HeadBucket(u8),
    Heads(u8),
    Buses(u8),
    /// A circuit layer's memory heads (the three above and the rest of the call), its gate
    /// walk, and the drain of its writes.
    Reads(u8),
    Walk(u8),
    Drain(u8),
    /// The routing layer's walk, then the drain of the route totals.
    Route,
    RouteDrain,
    Positional,
    /// The head marks' reset and the last checks.
    Tail,
    /// Before the decode loop: the stale marks of moved positions and the wanted positions.
    DecodeSetup,
    Decode,
    /// The decoded tokens copied out.
    Out,
    /// No bracket: the pass's shape (`SHAPE`) counted over its layers.
    Shape,
}

/// What `Stage::Shape` counts over a pass: the block vectors and lone gate entries its
/// walks update, the channels its drains find touched, and the memory heads whose query
/// channels or line changed.
pub const SHAPE: [&str; 4] = ["vectors", "lone entries", "touched channels", "dirty heads"];

impl Stage {
    /// Every bracketed stage of a pass over `layers` circuit layers, in the order the pass
    /// ends them. The three parts of a layer's memory heads come before the whole.
    pub fn all(layers: usize) -> Vec<Stage> {
        let mut v = vec![Stage::View, Stage::Layer0, Stage::Scatter];
        for l in 0..layers as u8 {
            v.extend([Stage::HeadBucket(l), Stage::Heads(l), Stage::Buses(l), Stage::Reads(l), Stage::Walk(l), Stage::Drain(l)]);
        }
        v.extend([Stage::Route, Stage::RouteDrain, Stage::Positional, Stage::Tail, Stage::DecodeSetup, Stage::Decode, Stage::Out]);
        v
    }
}

/// One armed pass's reading.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    /// The counters between the stage's start and its end, the bracket's own cost included.
    pub counts: Counts,
    /// What the stage worked on, by stage (`Engine::advance` names them at each bracket),
    /// or the pass's shape.
    pub items: [u32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct Probe {
    armed: Option<Stage>,
    open: Option<Counts>,
    got: Option<Reading>,
    shape: [u32; 4],
}

impl Probe {
    /// The next pass brackets `stage`.
    pub fn arm(&mut self, stage: Stage) {
        *self = Probe { armed: Some(stage), ..Probe::default() };
    }

    /// Disarms, with the armed stage's reading when the pass ran it.
    pub fn take(&mut self) -> Option<Reading> {
        match self.armed.take() {
            Some(Stage::Shape) => Some(Reading { counts: Counts::default(), items: self.shape }),
            _ => self.got.take(),
        }
    }

    #[inline(always)]
    pub(super) fn is_armed(&self) -> bool {
        self.armed.is_some()
    }

    /// Whether this pass needs `stage`'s items: it brackets the stage or counts the shape.
    #[inline(always)]
    pub(super) fn wants<const P: bool>(&self, stage: Stage) -> bool {
        P && (self.armed == Some(stage) || self.armed == Some(Stage::Shape))
    }

    /// Adds to the pass's shape, when it is counted.
    #[inline(always)]
    pub(super) fn shape<const P: bool>(&mut self, add: [u32; 4]) {
        if P && self.armed == Some(Stage::Shape) {
            for (s, a) in self.shape.iter_mut().zip(add) {
                *s += a;
            }
        }
    }

    #[inline(always)]
    pub(super) fn enter<const P: bool>(&mut self, stage: Stage) {
        if P && self.armed == Some(stage) {
            self.open = counters::cpi();
        }
    }

    /// Ends the bracket `enter` opened.
    #[inline(always)]
    pub(super) fn leave<const P: bool>(&mut self, stage: Stage, items: [u32; 4]) {
        if P && self.armed == Some(stage) {
            let end = counters::cpi();
            if let (Some(a), Some(b)) = (self.open.take(), end) {
                self.got = Some(Reading { counts: b.since(a), items });
            }
        }
    }
}
