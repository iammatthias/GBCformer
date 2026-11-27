//! Helper threads that run the parts of one pass side by side. `Pool::each` hands part
//! `i` of a slice to helper `i - 1` and runs part 0 on the calling thread, then waits for
//! all of them. A helper spins on the next job for a while (passes come in bursts) and
//! then parks until the next `each` wakes it.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// How long an idle helper spins before it parks.
const SPIN: Duration = Duration::from_micros(50);

/// One job: a type-erased call of the closure on item `i`.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

struct Shared {
    /// Every `each` bumps it, and a helper runs the job once per value.
    epoch: AtomicU64,
    job: UnsafeCell<Option<Job>>,
    /// Helpers still running the current job.
    running: AtomicUsize,
    panicked: AtomicBool,
    stop: AtomicBool,
}

// SAFETY: `job` is written only by `each` while no helper reads it (every helper has
// finished the last job, `running` is 0, and the epoch is not yet bumped), and read only
// by helpers after the epoch's Acquire load.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub(super) struct Pool {
    shared: Arc<Shared>,
    helpers: Vec<(Thread, JoinHandle<()>)>,
}

impl Pool {
    /// A pool of `helpers` threads beside the caller's.
    pub(super) fn new(helpers: usize) -> Pool {
        let shared = Arc::new(Shared { epoch: AtomicU64::new(0), job: UnsafeCell::new(None), running: AtomicUsize::new(0), panicked: AtomicBool::new(false), stop: AtomicBool::new(false) });
        let helpers = (0..helpers)
            .map(|h| {
                let s = shared.clone();
                let handle = thread::Builder::new().name(format!("part {}", h + 1)).spawn(move || {
                    crate::qos::interactive();
                    helper(&s, h + 1)
                }).expect("a helper thread");
                (handle.thread().clone(), handle)
            })
            .collect();
        Pool { shared, helpers }
    }

    /// The number of parts `each` runs at once.
    pub(super) fn parts(&self) -> usize {
        self.helpers.len() + 1
    }

    /// Runs `f` on every item, item `i` on helper `i - 1` and item 0 here, and returns
    /// when all are done. `items` holds exactly `parts()` items.
    pub(super) fn each<T: Send, F: Fn(&mut T) + Sync>(&self, items: &mut [T], f: &F) {
        assert_eq!(items.len(), self.parts());
        struct Ctx<T, F> {
            items: *mut T,
            f: *const F,
        }
        unsafe fn call<T, F: Fn(&mut T)>(data: *const (), i: usize) {
            // SAFETY: `each` keeps the context, the items and `f` alive until every
            // helper is done, and each index is handed to one thread only.
            unsafe {
                let ctx = &*(data as *const Ctx<T, F>);
                (*ctx.f)(&mut *ctx.items.add(i));
            }
        }
        let ctx = Ctx { items: items.as_mut_ptr(), f: f as *const F };
        let s = &*self.shared;
        // SAFETY: no helper runs (the last `each` waited for `running` to reach 0), so
        // none reads `job` until the epoch below moves.
        unsafe { *s.job.get() = Some(Job { data: &ctx as *const Ctx<T, F> as *const (), call: call::<T, F> }) };
        s.running.store(self.helpers.len(), Ordering::Relaxed);
        s.epoch.fetch_add(1, Ordering::Release);
        for (t, _) in &self.helpers {
            t.unpark();
        }
        // The helpers read `ctx` and the items until `running` reaches 0, so this frame
        // must not end before then, even when item 0 panics.
        struct Join<'a>(&'a Shared);
        impl Drop for Join<'_> {
            fn drop(&mut self) {
                while self.0.running.load(Ordering::Acquire) != 0 {
                    std::hint::spin_loop();
                }
            }
        }
        let join = Join(s);
        // SAFETY: item 0 is this thread's.
        unsafe { call::<T, F>(&ctx as *const Ctx<T, F> as *const (), 0) };
        drop(join);
        assert!(!s.panicked.load(Ordering::Relaxed), "a helper of the positional stage panicked");
    }
}

fn helper(s: &Shared, i: usize) {
    let mut seen = 0u64;
    loop {
        let start = Instant::now();
        let mut spins = 0u32;
        let epoch = loop {
            let e = s.epoch.load(Ordering::Acquire);
            if e != seen || s.stop.load(Ordering::Relaxed) {
                break e;
            }
            spins += 1;
            if spins % 64 == 0 && start.elapsed() > SPIN {
                thread::park_timeout(Duration::from_millis(50));
            } else {
                std::hint::spin_loop();
            }
        };
        if s.stop.load(Ordering::Relaxed) {
            return;
        }
        seen = epoch;
        // SAFETY: the Acquire load of the new epoch orders this read after `each` wrote
        // the job, and `each` does not return (so the job stays valid) until `running`
        // is 0.
        let job = unsafe { (*s.job.get()).expect("a job with the epoch") };
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (job.call)(job.data, i) }));
        if ran.is_err() {
            s.panicked.store(true, Ordering::Relaxed);
        }
        s.running.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        for (t, h) in self.helpers.drain(..) {
            t.unpark();
            let _ = h.join();
        }
    }
}
