//! The cells' K/V rows, apart from what a lookup reads of them (`cells::Facts`).
//!
//! `Cells::put` decides on the calling thread what the next read needs: the line's row,
//! the facts of the record's rows, and on a memo miss the rows themselves, built from the
//! weights. What is left is copying: the rows into the cell and into the memo, and the
//! sampled memo check's second build. [`Store`] does that at once, or, after
//! [`Store::defer`], on a worker thread that takes the records in order, a batch at a
//! time. Anything that reads the rows waits for the worker first, so it sees the rows
//! every record so far left.
//!
//! Deferring is inference management: it moves when and on which core the rows are
//! written, and changes no row. A memo check that fails on the worker stops the run when
//! a later batch is handed over, at [`Store::settle`], or when the cells are dropped. The
//! worker may be `BEHIND` batches behind, so that is fewer than (`BEHIND` + 2) × `BATCH`
//! = 3,072 records after the write: the rest of the check's batch, the `BEHIND` batches
//! accepted before the worker has run it, and the batch whose hand-over raises it.

use super::cells::{build_rows, turn, Facts, Key, Pair, Row, RowBuilder, NO_PAIR};
use crate::memory::attn::READ_AT;
use crate::memory::layout::LINE_BITS;
use std::sync::Arc;

/// Per group and row a key and a value, and per memo entry the rows built for its
/// record.
pub(super) struct Rows {
    builder: Arc<RowBuilder>,
    pub k: Vec<Vec<Row>>,
    pub v: Vec<Vec<Row>>,
    pub memo: Vec<Pair>,
}

impl Rows {
    fn new(builder: Arc<RowBuilder>) -> Rows {
        let groups = builder.groups();
        Rows { builder, k: vec![Vec::new(); groups], v: vec![Vec::new(); groups], memo: Vec::new() }
    }

    /// Row `r` takes `pair`. A new row is the next one.
    pub fn place(&mut self, r: usize, (ks, vs): &Pair) {
        for g in 0..self.k.len() {
            if r == self.k[g].len() {
                self.k[g].push(ks[g]);
                self.v[g].push(vs[g]);
            } else {
                self.k[g][r] = ks[g];
                self.v[g][r] = vs[g];
            }
        }
    }

    fn hit(&mut self, r: u32, at: u32) {
        let pair = self.memo[at as usize];
        self.place(r as usize, &pair);
    }

    fn miss(&mut self, r: u32, at: u32, pair: &Pair) {
        self.memo[at as usize] = *pair;
        self.place(r as usize, pair);
    }

    fn check(&mut self, r: u32, at: u32, key: Key, facts: &Facts) {
        let n = key.0 >> LINE_BITS;
        let (pair, again) = build_rows(&self.builder, key, &turn(n as i64 - (READ_AT + 1) as i64));
        assert!(self.memo[at as usize] == pair && *facts == again, "line {n:#07x}: the memo's rows for the record of {:#08x} differ from the rows built again", key.0);
        self.place(r as usize, &pair);
    }
}

/// The rows, written at once or by a worker thread.
pub(super) enum Store {
    Inline(Rows),
    #[cfg(not(target_arch = "wasm32"))]
    Deferred(worker::Deferred),
}

/// The rows with every record so far applied.
pub(super) enum Settled<'a> {
    Here(&'a Rows),
    #[cfg(not(target_arch = "wasm32"))]
    Locked(std::sync::MutexGuard<'a, Rows>),
}

impl std::ops::Deref for Settled<'_> {
    type Target = Rows;

    fn deref(&self) -> &Rows {
        match self {
            Settled::Here(rows) => rows,
            #[cfg(not(target_arch = "wasm32"))]
            Settled::Locked(rows) => rows,
        }
    }
}

impl Store {
    pub fn new(builder: Arc<RowBuilder>) -> Store {
        Store::Inline(Rows::new(builder))
    }

    /// Moves the rows behind a worker thread. A second call changes nothing.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn defer(&mut self) {
        if let Store::Inline(rows) = self {
            let rows = std::mem::replace(rows, Rows::new(rows.builder.clone()));
            *self = Store::Deferred(worker::Deferred::spawn(rows));
        }
    }

    pub fn deferred(&self) -> bool {
        !matches!(self, Store::Inline(_))
    }

    /// Waits until the worker has applied every record so far, and raises its panic.
    pub fn settle(&self) {
        match self {
            Store::Inline(_) => {}
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => d.settle(),
        }
    }

    pub fn rows(&self) -> Settled<'_> {
        match self {
            Store::Inline(rows) => Settled::Here(rows),
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => Settled::Locked(d.rows()),
        }
    }

    pub fn with<T>(&mut self, f: impl FnOnce(&mut Rows) -> T) -> T {
        match self {
            Store::Inline(rows) => f(rows),
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => f(&mut d.rows()),
        }
    }

    /// Row `r` takes the rows memo entry `at` holds.
    #[inline]
    pub fn hit(&mut self, r: u32, at: u32) {
        match self {
            Store::Inline(rows) => rows.hit(r, at),
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => d.push(worker::Op::Hit { r, at }, None),
        }
    }

    /// Memo entry `at` takes `pair`, built for a record it did not hold, and row `r` with it.
    #[inline]
    pub fn miss(&mut self, r: u32, at: u32, pair: &Pair) {
        match self {
            Store::Inline(rows) => rows.miss(r, at, pair),
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => d.push(worker::Op::Miss { r, at }, Some(pair)),
        }
    }

    /// [`Store::hit`] with the sampled check: the rows of `key` are built again and must
    /// equal the entry's, and their facts the `facts` the lookup was given.
    #[inline]
    pub fn check(&mut self, r: u32, at: u32, key: Key, facts: &Facts) {
        match self {
            Store::Inline(rows) => rows.check(r, at, key, facts),
            #[cfg(not(target_arch = "wasm32"))]
            Store::Deferred(d) => d.push(worker::Op::Check { r, at, key, facts: *facts }, None),
        }
    }

    /// An empty memo of `len` entries.
    pub fn clear_memo(&mut self, len: usize) {
        self.with(|rows| rows.memo = vec![NO_PAIR; len]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod worker {
    use super::{Facts, Key, Pair, Rows};
    use std::collections::VecDeque;
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};

    /// What one record leaves for the rows, as `Store`'s methods name it.
    pub enum Op {
        Hit { r: u32, at: u32 },
        /// Its rows are the batch's next `built`.
        Miss { r: u32, at: u32 },
        Check { r: u32, at: u32, key: Key, facts: Facts },
    }

    #[derive(Default)]
    struct Batch {
        ops: Vec<Op>,
        built: Vec<Pair>,
    }

    /// The records handed over at once.
    const BATCH: usize = 512;
    /// The batches the worker may be behind before `push` waits for it. It bounds how
    /// late a failed check stops the run (the module's doc). The machine's thread never
    /// waited at 64 and the queue was never deeper than one batch (`docs/perf-log.md`,
    /// "Round 5b"), so 4 keeps that and stops the run within a sixteenth of the lag.
    const BEHIND: usize = 4;

    #[derive(Default)]
    struct Queue {
        batches: VecDeque<Batch>,
        /// The worker holds a batch it took.
        busy: bool,
        stop: bool,
        /// The worker stopped at a panic (a failed check), raised once from `panic`.
        failed: bool,
        panic: Option<Box<dyn std::any::Any + Send>>,
    }

    struct Shared {
        rows: Mutex<Rows>,
        queue: Mutex<Queue>,
        /// The queue changed: a batch came, one was taken or one was finished.
        moved: Condvar,
    }

    pub struct Deferred {
        shared: Arc<Shared>,
        /// The records since the last hand-over. `push` owns it; a reader of the rows on
        /// another thread hands it over through the lock.
        pending: Mutex<Batch>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Deferred {
        pub fn spawn(rows: Rows) -> Deferred {
            let shared = Arc::new(Shared { rows: Mutex::new(rows), queue: Mutex::default(), moved: Condvar::new() });
            let theirs = shared.clone();
            let thread = std::thread::Builder::new().name("cell rows".into()).spawn(move || theirs.work()).expect("the cell rows' thread");
            Deferred { shared, pending: Mutex::default(), thread: Some(thread) }
        }

        pub fn push(&mut self, op: Op, built: Option<&Pair>) {
            let batch = self.pending.get_mut().unwrap();
            batch.ops.push(op);
            batch.built.extend(built);
            if batch.ops.len() >= BATCH {
                let batch = std::mem::replace(batch, Batch { ops: Vec::with_capacity(BATCH), built: Vec::new() });
                self.send(batch);
            }
        }

        /// Hands `batch` over once the worker is within `BEHIND` batches, unless the
        /// worker has failed.
        fn queue(&self, batch: Batch) -> MutexGuard<'_, Queue> {
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            while q.batches.len() >= BEHIND && !q.failed {
                q = self.shared.moved.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            if !q.failed {
                q.batches.push_back(batch);
                self.shared.moved.notify_all();
            }
            q
        }

        fn send(&self, batch: Batch) {
            raise(&mut self.queue(batch));
        }

        /// Hands the pending records over and waits until the worker has applied every
        /// record or failed.
        fn idle(&self) -> MutexGuard<'_, Queue> {
            let batch = std::mem::take(&mut *self.pending.lock().unwrap_or_else(|e| e.into_inner()));
            if !batch.ops.is_empty() {
                drop(self.queue(batch));
            }
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            while (!q.batches.is_empty() || q.busy) && !q.failed {
                q = self.shared.moved.wait(q).unwrap_or_else(|e| e.into_inner());
            }
            q
        }

        pub fn settle(&self) {
            raise(&mut self.idle());
        }

        pub fn rows(&self) -> MutexGuard<'_, Rows> {
            self.settle();
            self.shared.rows.lock().unwrap()
        }
    }

    /// Raises the worker's panic on the calling thread.
    fn raise(q: &mut MutexGuard<'_, Queue>) {
        if let Some(panic) = q.panic.take() {
            std::panic::resume_unwind(panic);
        }
        assert!(!q.failed, "the cells' rows stopped at a failed memo check");
    }

    impl Shared {
        fn work(&self) {
            loop {
                let batch = {
                    let mut q = self.queue.lock().unwrap();
                    loop {
                        if let Some(batch) = q.batches.pop_front() {
                            q.busy = true;
                            break batch;
                        }
                        if q.stop {
                            return;
                        }
                        q = self.moved.wait(q).unwrap();
                    }
                };
                self.moved.notify_all();
                let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.rows.lock().unwrap().apply(&batch)));
                let mut q = self.queue.lock().unwrap();
                q.busy = false;
                if let Err(panic) = ran {
                    // Nothing after a failed check is applied: the run stops at the next hand-over.
                    q.batches.clear();
                    (q.stop, q.failed, q.panic) = (true, true, Some(panic));
                }
                self.moved.notify_all();
            }
        }
    }

    impl Rows {
        fn apply(&mut self, batch: &Batch) {
            let mut built = batch.built.iter();
            for op in &batch.ops {
                match *op {
                    Op::Hit { r, at } => self.hit(r, at),
                    Op::Miss { r, at } => self.miss(r, at, built.next().expect("a miss carries its rows")),
                    Op::Check { r, at, key, ref facts } => self.check(r, at, key, facts),
                }
            }
        }
    }

    impl Drop for Deferred {
        fn drop(&mut self) {
            // A check still queued runs before the cells go, and a check that failed with
            // no hand-over after it is raised here: a late check is not a lost one.
            let panic = if std::thread::panicking() { None } else { self.idle().panic.take() };
            self.shared.queue.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
            self.shared.moved.notify_all();
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
            if let Some(panic) = panic {
                std::panic::resume_unwind(panic);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::{Facts, Rows, NO_PAIR};
        use super::{Deferred, Op, BATCH};
        use crate::engine::Engine;
        use crate::llama::Model;
        use crate::memory::attn::Params;
        use crate::memory::image::Image;
        use crate::memory::layout::{Banking, Mapper, LINE};

        /// A worker whose first record is a memo check that fails: the entry is empty.
        fn failing() -> Deferred {
            let mut engine = Engine::new(Model::compile_mem(&crate::sm83::build_mem(), Params::memory()));
            engine.insert(Image::new(&[0; 0x8000], 0x2000, Banking::new(Mapper::Rom, 0x8000, 0x2000, false).unwrap()));
            let mut rows = Rows::new(engine.memory.as_ref().unwrap().cells.builder.clone());
            rows.memo = vec![NO_PAIR; 1];
            let mut d = Deferred::spawn(rows);
            d.push(Op::Check { r: 0, at: 0, key: (0x4000, [0x5A; LINE]), facts: Facts::NONE }, None);
            d
        }

        fn message(panic: Box<dyn std::any::Any + Send>) -> String {
            panic.downcast_ref::<String>().cloned().unwrap_or_default()
        }

        #[test]
        fn a_check_that_failed_with_no_hand_over_after_it_stops_the_drop() {
            let mut d = failing();
            for _ in 1..BATCH {
                d.push(Op::Hit { r: 0, at: 0 }, None);
            }
            // The batch is with the worker, and nothing is handed over after it fails.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while !d.shared.queue.lock().unwrap().failed {
                assert!(std::time::Instant::now() < deadline, "the worker ran the failing batch");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(d)));
            let said = message(dropped.expect_err("the drop raises the failed check"));
            assert!(said.contains("differ from the rows built again"), "{said}");
        }

        #[test]
        fn a_failed_check_still_pending_stops_the_drop() {
            let d = failing();
            let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(d)));
            let said = message(dropped.expect_err("the drop runs the pending check"));
            assert!(said.contains("differ from the rows built again"), "{said}");
        }
    }
}
