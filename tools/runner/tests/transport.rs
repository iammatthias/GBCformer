//! The transport acts on each number and bit a pass emits, with values no model of today
//! emits: a host that kept a rule of its own (a dropped edge one cycle on, the pass after a
//! schedule a period on, stalls summed, a run's last byte after the line) fails here.

use gbcformer::transport::*;

/// A stream of spans whose passes emit the asks, and the further pass, the test lists.
struct Fake {
    t: Spans,
    outs: Vec<(Asks, Option<u8>)>,
    events: Vec<u8>,
}

impl Fake {
    fn new(outs: Vec<(Asks, Option<u8>)>) -> Fake {
        Fake { t: Spans::new(100), outs, events: Vec::new() }
    }
}

impl Passes for Fake {
    type Event = u8;
    type Out = (Asks, Option<u8>);
    type Cx<'a> = ();
    fn once(&mut self, event: u8, _: &mut ()) -> Self::Out {
        self.events.push(event);
        self.t.close(&Cut { horizon: 50, ..Cut::default() });
        self.outs.remove(0)
    }
    fn asks(out: &Self::Out) -> Asks {
        out.0
    }
    fn follow(out: &Self::Out) -> Option<u8> {
        out.1
    }
    fn own(&self, event: Own) -> u8 {
        match event {
            Own::Tick => 100,
            Own::Stop => 101,
        }
    }
    fn countdown(&mut self) -> &mut Spans {
        &mut self.t
    }
    fn run_bits(&self) -> u8 {
        3
    }
    fn held(&self) -> bool {
        false
    }
    fn line(&self) -> u8 {
        7
    }
}

#[test]
fn a_dropped_edge_falls_where_the_pass_says() {
    for drop in [1, 2, 5] {
        let mut t = Spans::new(100);
        t.close(&Cut { horizon: 50, drop, ..Cut::default() });
        assert_eq!(t.until_edge(), drop, "the edge {drop} cycles on");
        assert_eq!(t.advance(drop, false), (Edges { fall: true, reset: true, rise: false }, Due::Nothing));
    }
    let mut t = Spans::new(100);
    t.close(&Cut { horizon: 50, ..Cut::default() });
    assert_eq!(t.until_edge(), u64::MAX, "no drop, no edge");
}

#[test]
fn the_rise_and_the_counted_fall_are_cut_where_the_pass_says() {
    let mut t = Spans::new(100);
    t.close(&Cut { horizon: 50, edge: Some(8), rise: Some(3), ..Cut::default() });
    assert_eq!(t.horizon(), 3, "the block ends at the rise");
    assert_eq!(t.advance(3, false), (Edges { fall: false, reset: false, rise: true }, Due::Nothing));
    assert_eq!(t.horizon(), 5, "then at the fall");
    assert_eq!(t.advance(5, false), (Edges { fall: true, reset: false, rise: false }, Due::Nothing));
}

#[test]
fn the_pass_after_a_schedule_falls_at_its_tail() {
    let mut t = Spans::new(200);
    let mut lines = [0; MAX_STARTS];
    lines[..2].copy_from_slice(&[4, 5]);
    t.close(&Cut { horizon: 10, schedule: Schedule { count: 2, first: 10, period: 7, tail: 3, phase: 9, lines }, ..Cut::default() });
    assert_eq!(t.advance(10, false), (Edges::default(), Due::Start(4)));
    assert_eq!(t.horizon(), 7, "the next start a period on");
    assert_eq!(t.advance(7, false), (Edges::default(), Due::Start(5)));
    assert_eq!(t.horizon(), 3, "the next pass at the tail, not a period on");
    assert_eq!(t.advance(3, false), (Edges::default(), Due::Pass));
    assert_eq!(t.open().position, Some((9, 5)));
}

#[test]
fn further_passes_run_at_the_events_the_passes_name() {
    let block = |source| Asks { block: Some((source, 0x8000, 16, false)), ..Asks::default() };
    let mut f = Fake::new(vec![(block(0x100), Some(9)), (block(0x110), Some(4)), (block(0x120), None)]);
    let mut s = Signals::default();
    f.pass_with(7, &mut s, &mut (), None);
    assert_eq!(f.events, [7, 9, 4]);
    assert_eq!(s.copies, [0x100, 0x110, 0x120].map(|source| Copy::Block { source, dest: 0x8000, len: 16 }));
}

#[test]
fn a_block_goes_where_its_held_bit_says() {
    for held in [true, false] {
        let mut f = Fake::new(vec![(Asks { block: Some((0x100, 0x8000, 16, held)), ..Asks::default() }, None)]);
        let mut s = Signals::default();
        f.pass_with(7, &mut s, &mut (), None);
        assert_eq!((s.held.len(), s.copies.len()), if held { (1, 0) } else { (0, 1) });
    }
}

#[test]
fn the_stopped_cycles_are_the_total_the_last_pass_emitted() {
    let mut t = Spans::new(100);
    t.close(&Cut { horizon: 50, owed: 30, ..Cut::default() });
    t.advance(10, false);
    assert_eq!(t.owed(), 30, "cycles the steps stream runs count nothing down");
    t.advance(10, true);
    assert_eq!(t.owed(), 20);
    t.close(&Cut { horizon: 50, owed: 5, ..Cut::default() });
    assert_eq!(t.owed(), 5, "a pass's total replaces the count, it does not add to it");
}

/// A host whose passes, one a block, emit the totals listed, and the pause flag.
struct Host {
    t: Spans,
    totals: Vec<(u64, bool)>,
    paused: bool,
    ran: Vec<(u64, bool)>,
}

impl Blocks for Host {
    fn horizon(&self) -> u64 {
        self.t.horizon()
    }
    fn block(&mut self, cycles: u64, stopped: bool) {
        self.t.advance(cycles, stopped);
        self.ran.push((cycles, stopped));
        let (owed, paused) = if self.totals.is_empty() { (self.t.owed(), self.paused) } else { self.totals.remove(0) };
        self.t.close(&Cut { horizon: 4, owed, ..Cut::default() });
        self.paused = paused;
    }
    fn stop(&mut self) {}
    fn owed(&self) -> u64 {
        self.t.owed()
    }
    fn paused(&self) -> bool {
        self.paused
    }
    fn settle(&mut self) {}
}

#[test]
fn a_step_ends_stopped_for_the_cycles_the_passes_emit() {
    // The step's last block emits 10 owed, a pause; the stopped blocks' passes keep the
    // count, then one raises it to 9 left and ends the pause.
    let mut h = Host { t: Spans::new(100), totals: vec![(10, true), (6, true), (9, false)], paused: false, ran: Vec::new() };
    h.t.close(&Cut { horizon: 4, ..Cut::default() });
    let (stalled, paused) = advance(&mut h, 4, false);
    assert_eq!(h.ran, [(4, false), (4, true), (4, true), (4, true), (4, true), (1, true)]);
    assert_eq!((stalled, paused), (9, 8));
    assert_eq!(h.t.owed(), 0);
}

#[test]
fn a_run_s_last_byte_lands_where_its_own_bit_says() {
    for last_early in [false, true] {
        let run = Run { page: 0xC1, line: 0x18, before: 0b011, last: 0b100, last_early, dest: 0xFE, fill: 0 };
        let (mut early, mut late) = (Vec::new(), Vec::new());
        run.bytes(3, &mut early, &mut late);
        let byte = |j: u16| Copy::Byte { source: 0xC118 | j, dest: 0xFE18 | j };
        let mut want = vec![byte(0), byte(1)];
        if last_early { want.push(byte(2)) }
        assert_eq!(early, want);
        assert_eq!(late, if last_early { vec![] } else { vec![byte(2)] });
    }
}

#[test]
fn a_cut_is_a_pass_at_any_cycle() {
    let mut f = Fake::new(vec![(Asks::default(), None)]);
    f.t.close(&Cut { horizon: 50, ..Cut::default() });
    f.cut(&mut Signals::default(), &mut ());
    assert_eq!(f.events, [100], "a cut at the cycle of the last pass is a pass");
}

#[test]
fn a_pass_alone_hands_its_bytes_over_at_once() {
    let run = Run { page: 0xC0, line: 0, before: 0b1, last: 0b10, last_early: false, dest: 0xFE, fill: 0 };
    let mut f = Fake::new(vec![(Asks { run: Some(run), ..Asks::default() }, None)]);
    let mut landed = Vec::new();
    f.pass_alone(5, &mut landed, &mut ());
    assert_eq!(landed, [0, 1].map(|j| Copy::Byte { source: 0xC000 | j, dest: 0xFE00 | j }));
}

/// Closes for `cycles` cycles of `rate` halves from a pass that emitted `to_sample` and `to_end`.
fn closes(to_sample: u64, to_end: u64, rate: u64, cycles: u64) -> Vec<Close> {
    let mut c = SegmentClock::emitted(to_sample, to_end);
    (0..cycles).map(|_| c.cycle(rate)).collect()
}

#[test]
fn a_segment_ends_only_where_the_pass_says() {
    use Close::*;
    assert_eq!(closes(4, 0, 1, 4), [Open, Open, Open, Sample], "no end emitted: the sample closes it");
    assert_eq!(closes(5, 0, 2, 3), [Open, Open, Sample], "the cycle whose halves reach or pass the sample's");
    assert_eq!(closes(4, 1, 1, 1), [End], "an end after one cycle");
    assert_eq!(closes(8, 3, 1, 3), [Open, Open, End], "an end after three cycles");
    assert_eq!(closes(2, 2, 1, 2), [Open, Sample], "the end and the sample at one cycle: the sample's pass");
    assert!(SegmentClock::emitted(5, 1).ends_next() && !SegmentClock::emitted(5, 0).ends_next() && !SegmentClock::emitted(5, 2).ends_next());
}

/// A segments model whose passes follow the numbers the test gives it: a sample every
/// `period` halves, an end after one cycle where a fall's pass ran, and every cycle left by
/// a pass whose segment opens with events. It logs each segment it runs and each it keeps.
struct Model {
    period: u64,
    since: u64,
    end_after_fall: bool,
    ran: Vec<Segment>,
    kept: Vec<Segment>,
}

impl SegmentPass for Model {
    fn pass(&mut self, seg: &Segment) -> Emitted {
        self.ran.push(*seg);
        let event = seg.events != Edges::default();
        if !event {
            self.since = if seg.sample { 0 } else { self.since + seg.cycles * seg.rate };
        }
        Emitted { to_sample: self.period - self.since, to_end: (event && seg.events.fall && self.end_after_fall) as u64, rest: if event { seg.cycles } else { 0 } }
    }
    fn keep(&mut self, seg: &Segment) {
        self.kept.push(*seg);
    }
}

fn fake(period: u64, end_after_fall: bool) -> Model {
    Model { period, since: 0, end_after_fall, ran: Vec::new(), kept: Vec::new() }
}

fn seg(events: Edges, cycles: u64, sample: bool, read: Option<u8>) -> Segment {
    Segment { events, cycles, rate: if cycles > 0 { 1 } else { 0 }, sample, read, ..Segment::default() }
}

#[test]
fn the_cycles_a_pass_leaves_are_counted_again() {
    let fall = Edges { fall: true, ..Edges::default() };
    let mut p = fake(20, true);
    let mut c = Cutter::new(&mut p);
    for _ in 0..2 {
        c.count(&mut p, false, 1);
    }
    assert!(c.events(&mut p, fall, false), "the event closes the cycles before it");
    for _ in 0..4 {
        assert_eq!(c.count(&mut p, false, 1), Close::Open, "no close the host decides");
    }
    c.read(&mut p, 3);
    let none = Edges::default();
    assert_eq!(p.ran, [seg(none, 0, false, None), seg(none, 2, false, None), seg(fall, 4, false, Some(3)), seg(none, 1, false, None), seg(none, 3, false, Some(3))]);
    assert_eq!(p.kept.last(), Some(&seg(none, 3, false, Some(3))), "the read kept once, from the pass that left no cycles");
    assert_eq!(p.kept.iter().filter(|s| s.read.is_some()).count(), 1);
    assert_eq!(c.ended, [2], "the end after the event's own cycle");
    assert_eq!(c.settled(), 6);
}

#[test]
fn a_left_sample_is_reached_again() {
    let rise = Edges { rise: true, ..Edges::default() };
    let mut p = fake(6, false);
    let mut c = Cutter::new(&mut p);
    c.count(&mut p, false, 1);
    c.events(&mut p, rise, false);
    let closes: Vec<Close> = (0..5).map(|_| c.count(&mut p, false, 1)).collect();
    assert_eq!(closes, [Close::Open, Close::Open, Close::Open, Close::Open, Close::Sample]);
    let none = Edges::default();
    assert_eq!(p.ran, [seg(none, 0, false, None), seg(none, 1, false, None), seg(rise, 5, true, None), seg(none, 5, true, None)]);
    assert_eq!(p.kept.iter().filter(|s| s.sample).count(), 1, "the sample kept once");
    assert!(c.ended.is_empty());
}

/// A step's opening whose requests flag the first fetched byte's locks and the ask for
/// them as the test lists, one entry per request; it logs the locks each request ran under.
struct Opens {
    flags: Vec<(bool, bool)>,
    ran: Vec<Option<u8>>,
    queries: u8,
}

impl Opening for Opens {
    type Request = (bool, bool);
    fn request(&mut self, locks: Option<u8>) -> (bool, bool) {
        self.ran.push(locks);
        self.flags.remove(0)
    }
    fn fetch_reads_locks(&self, req: &(bool, bool)) -> bool {
        req.0
    }
    fn wants_locks(&self, req: &(bool, bool)) -> bool {
        req.1
    }
    fn lock_query(&mut self) -> u8 {
        self.queries += 1;
        self.queries * 3
    }
}

fn open(held: Option<u8>, flags: &[(bool, bool)]) -> (Option<u8>, u8, Vec<Option<u8>>) {
    let mut h = Opens { flags: flags.to_vec(), ran: Vec::new(), queries: 0 };
    let (_, locks, asked) = open_step(&mut h, held);
    assert!(h.flags.is_empty(), "requests left unasked: {:?}", h.flags);
    (locks, asked, h.ran)
}

#[test]
fn a_step_asks_for_the_lapsed_locks_only_where_its_fetch_reads_them() {
    assert_eq!(open(None, &[(true, false), (true, false)]), (Some(3), 0, vec![None, Some(3)]), "lapsed and read: a pass, then the step again under its locks");
    assert_eq!(open(Some(1), &[(true, false)]), (Some(1), 0, vec![Some(1)]), "held: no pass");
    assert_eq!(open(None, &[(false, false)]), (None, 0, vec![None]), "not read: no pass");
    assert_eq!(open(None, &[(true, false), (false, true)]), (Some(3), 6, vec![None, Some(3)]), "the second request's own ask is a second pass");
    assert_eq!(open(None, &[(false, true)]), (None, 3, vec![None]), "an ask alone is one pass, the start's locks still lapsed");
}

#[test]
fn an_access_takes_the_busy_bit_always_and_the_locks_by_its_flag() {
    let bits = StateBits { locks: 3, busy: 4, replace: 8, shift: 4, writes: 0, write_shift: 2 };
    assert_eq!(access_mask(bits, false), 4);
    assert_eq!(access_mask(bits, true), 7);
    assert_eq!(start_state(bits, None, 2, true), 6, "the asked locks where the held ones lapsed");
    assert_eq!(start_state(bits, Some(1), 2, false), 1, "the held locks over the asked");
}

/// A DMG's reads and writes see different lock pairs at two M-cycles (docs/hardware-tests.md
/// §3.17): a write slot takes the pair `write_shift` above the reads', a read slot the reads'.
#[test]
fn a_write_slot_takes_the_writes_lock_pair() {
    // Writes 0 and 1 are slots 2 and 3, their pair 2 bits above the reads', as the GBC
    // wiring has them (`step::WRITE_SLOTS`, `WRITE_SHIFT`; the runner links no machine).
    let bits = StateBits { locks: 3, busy: 4, replace: 8, shift: 4, writes: 0b1100, write_shift: 2 };
    // Reads see OAM locked (2), writes VRAM (1), as at M-cycle 19 of a DMG's mode 2 with the
    // pairs swapped, so each slot's pair is told apart.
    let locks = 2 | 1 << 2;
    assert_eq!([0, 1, 2, 3, 4, 5].map(|s| bits.slot_locks(locks, s)), [2, 2, 1, 1, 2, 2], "reads 0-1 and the fetches take the reads' pair, writes 0-1 the writes'");
    let held = [(locks, 10u64)];
    assert_eq!(bus_state(&held, 0..0, 3, true, false, bits, || panic!("no look-ahead")), 2, "a read inside the held span");
    assert_eq!(bus_state(&held, 0..0, 3, true, true, bits, || panic!("no look-ahead")), 1, "a write inside the held span");
    assert_eq!(bus_state(&held, 2..5, 12, true, true, bits, || 1 << 2), 1, "a write past the span takes the look-ahead's writes' pair");
    assert_eq!(bus_state(&held, 2..5, 4, false, true, bits, || panic!("no locks asked")), 4, "the busy bit alone");
}
