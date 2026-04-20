//! Cross-thread reindex notification.
//!
//! The recall path stats a file before reading its stub range and detects
//! staleness from an mtime mismatch. When a store is attached to a
//! `ReindexQueue`, staleness is reported two ways: the `stale` flag is
//! persisted (source of truth, survives restarts) and the path is pushed
//! onto the queue (wake-up signal for idle workers).
//!
//! SQLite has no cross-connection notification primitive — the per-connection
//! `update_hook` fires only on the connection that did the write — so we use
//! the DB for durability and an in-process channel for signaling. Workers
//! that crash lose pending channel messages; a startup sweep over
//! `stale = 1` rows recovers them.
//!
//! The default `NoopReindexQueue` exists so library-mode callers (embedded
//! use, no background worker) get a loud `StaleStub` error on mismatch
//! without having to wire anything up.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

/// Sink for "this path needs reingestion" signals. Implementations must be
/// cheap — recall-path code calls `enqueue` while holding no locks it cares
/// about and continues immediately.
pub trait ReindexQueue: Send + Sync {
    fn enqueue(&self, path: &str);
}

/// Default no-op queue. Returned by `NoopReindexQueue::default()`. Use when
/// the application has no reindex worker attached — staleness still surfaces
/// as `CawError::StaleStub` on the failing recall, but nothing gets scheduled.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopReindexQueue;

impl ReindexQueue for NoopReindexQueue {
    fn enqueue(&self, _path: &str) {}
}

/// Multi-producer, multi-consumer path queue backed by a `Mutex<VecDeque>`
/// plus a `Condvar` for wakeups, with a `HashSet` deduping paths that are
/// still pending. Pending dedup matters because a single edited file can
/// produce many stale recalls in rapid succession (one per chunk); we don't
/// want the worker to reingest it N times.
#[derive(Clone)]
pub struct ChannelReindexQueue {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    cv: Condvar,
}

struct State {
    deque: VecDeque<String>,
    pending: HashSet<String>,
    closed: bool,
}

impl ChannelReindexQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    deque: VecDeque::new(),
                    pending: HashSet::new(),
                    closed: false,
                }),
                cv: Condvar::new(),
            }),
        }
    }

    /// Clone a receiver handle. Multiple receivers share the same queue;
    /// each `recv()` call returns exactly one path.
    pub fn receiver(&self) -> ReindexReceiver {
        ReindexReceiver {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Signal all waiting receivers to exit. After close, `recv` returns
    /// `None` once the deque drains. Subsequent `enqueue` calls are dropped.
    pub fn close(&self) {
        let mut state = self.inner.state.lock().expect("poisoned");
        state.closed = true;
        self.inner.cv.notify_all();
    }
}

impl Default for ChannelReindexQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl ReindexQueue for ChannelReindexQueue {
    fn enqueue(&self, path: &str) {
        let mut state = self.inner.state.lock().expect("poisoned");
        if state.closed {
            return;
        }
        if state.pending.insert(path.to_string()) {
            state.deque.push_back(path.to_string());
            self.inner.cv.notify_one();
        }
    }
}

pub struct ReindexReceiver {
    inner: Arc<Inner>,
}

impl ReindexReceiver {
    /// Block until a path is available. Returns `None` if the queue is
    /// closed and drained.
    pub fn recv(&self) -> Option<String> {
        let mut state = self.inner.state.lock().expect("poisoned");
        loop {
            if let Some(path) = state.deque.pop_front() {
                state.pending.remove(&path);
                return Some(path);
            }
            if state.closed {
                return None;
            }
            state = self.inner.cv.wait(state).expect("poisoned");
        }
    }
}

/// Drain `receiver` until the queue is closed, calling `work` on each path.
/// Errors from `work` are swallowed — the application is expected to log or
/// surface them itself, because the worker must keep pulling regardless of
/// whether any single reingest succeeded. Pair with `ChannelReindexQueue::close`
/// to shut down cleanly.
///
/// Typical spawn pattern:
/// ```ignore
/// let queue = ChannelReindexQueue::new();
/// let store = SqliteStubStore::new("index.db", 384)?
///     .with_reindex_queue(Arc::new(queue.clone()));
/// for _ in 0..num_workers {
///     let rx = queue.receiver();
///     let mut worker_store = store.open_another()?;
///     std::thread::spawn(move || {
///         run_worker(rx, |path| reingest_one(path, &mut worker_store))
///     });
/// }
/// ```
pub fn run_worker<F>(receiver: ReindexReceiver, mut work: F)
where
    F: FnMut(&str),
{
    while let Some(path) = receiver.recv() {
        work(&path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_dedups_pending_paths() {
        let q = ChannelReindexQueue::new();
        q.enqueue("a.md");
        q.enqueue("a.md");
        q.enqueue("a.md");
        q.close();
        let rx = q.receiver();
        assert_eq!(rx.recv(), Some("a.md".to_string()));
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn reenqueue_after_recv_is_allowed() {
        let q = ChannelReindexQueue::new();
        q.enqueue("a.md");
        let rx = q.receiver();
        assert_eq!(rx.recv(), Some("a.md".to_string()));
        q.enqueue("a.md");
        q.close();
        assert_eq!(rx.recv(), Some("a.md".to_string()));
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn multiple_receivers_split_work() {
        let q = ChannelReindexQueue::new();
        for i in 0..10 {
            q.enqueue(&format!("f{i}.md"));
        }
        q.close();
        let rx1 = q.receiver();
        let rx2 = q.receiver();
        let t1 = std::thread::spawn(move || {
            let mut got = Vec::new();
            while let Some(p) = rx1.recv() {
                got.push(p);
            }
            got
        });
        let t2 = std::thread::spawn(move || {
            let mut got = Vec::new();
            while let Some(p) = rx2.recv() {
                got.push(p);
            }
            got
        });
        let mut all = t1.join().unwrap();
        all.extend(t2.join().unwrap());
        all.sort();
        let expected: Vec<String> = (0..10).map(|i| format!("f{i}.md")).collect();
        assert_eq!(all, expected);
    }
}
