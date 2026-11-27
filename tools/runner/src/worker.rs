//! A stream on its own thread: every call reaches it in the order the scheduler made it,
//! so its passes are the ones it would run inline. `post` queues a call, handed over with
//! the others in batches, since each hand-over wakes the thread; `call` hands over what is
//! queued and its own call, and waits for its result.

use std::sync::mpsc::{channel, Sender};

type Job<T> = Box<dyn FnOnce(&mut T) + Send>;

/// Calls handed over a wake.
const BATCH: usize = 8;

pub enum Worker<T> {
    Thread { jobs: Option<Sender<Vec<Job<T>>>>, queued: Vec<Job<T>>, handle: Option<std::thread::JoinHandle<T>> },
    Inline(Box<T>),
}

impl<T: Send + 'static> Worker<T> {
    pub fn new(t: T, thread: bool) -> Worker<T> {
        if !thread {
            return Worker::Inline(Box::new(t));
        }
        let (tx, rx) = channel::<Vec<Job<T>>>();
        let handle = std::thread::spawn(move || {
            gbcformer::qos::interactive();
            let mut t = t;
            for batch in rx {
                for job in batch {
                    job(&mut t);
                }
            }
            t
        });
        Worker::Thread { jobs: Some(tx), queued: Vec::with_capacity(BATCH), handle: Some(handle) }
    }

    fn hand_over(jobs: &Option<Sender<Vec<Job<T>>>>, queued: &mut Vec<Job<T>>) {
        if !queued.is_empty() {
            let batch = std::mem::replace(queued, Vec::with_capacity(BATCH));
            jobs.as_ref().expect("a running worker").send(batch).expect("the worker runs");
        }
    }

    pub fn post(&mut self, f: impl FnOnce(&mut T) + Send + 'static) {
        match self {
            Worker::Thread { jobs, queued, .. } => {
                queued.push(Box::new(f));
                if queued.len() >= BATCH {
                    Self::hand_over(jobs, queued);
                }
            }
            Worker::Inline(t) => f(t),
        }
    }

    /// `post`, handed over at once with everything queued before it: for a call that is
    /// itself a batch.
    pub fn send(&mut self, f: impl FnOnce(&mut T) + Send + 'static) {
        self.post(f);
        if let Worker::Thread { jobs, queued, .. } = self {
            Self::hand_over(jobs, queued);
        }
    }

    pub fn call<R: Send + 'static>(&mut self, f: impl FnOnce(&mut T) -> R + Send + 'static) -> R {
        match self {
            Worker::Thread { jobs, queued, .. } => {
                let (tx, rx) = channel();
                queued.push(Box::new(move |t| tx.send(f(t)).expect("the caller waits")));
                Self::hand_over(jobs, queued);
                rx.recv().expect("the worker answers")
            }
            Worker::Inline(t) => f(t),
        }
    }

    /// The stream back, after every queued call ran.
    pub fn finish(self) -> T {
        match self {
            Worker::Thread { mut jobs, mut queued, mut handle } => {
                Self::hand_over(&jobs, &mut queued);
                drop(jobs.take());
                handle.take().expect("a running worker").join().expect("the worker ran to the end")
            }
            Worker::Inline(t) => *t,
        }
    }
}
