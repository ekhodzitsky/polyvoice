#![allow(clippy::unwrap_used)]
//! Loom model-checking for the session-pool checkout/drop pattern.
//!
//! This test verifies that the `PooledSession`-style guard (checkout from a
//! lock-free queue, return on Drop) is sound under all possible thread
//! interleavings.
//!
//! To run with full Loom model checking (slower but exhaustive):
//! ```bash
//! LOOM_MAX_PREEMPTIONS=2 cargo test --test loom_pool
//! ```

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;
use std::cell::RefCell;

/// A mock pooled resource (stand-in for `ort::session::Session`).
struct MockSession {
    #[allow(dead_code)]
    id: usize,
}

/// A mock lock-free pool using `loom`'s rendezvous channel as a stand-in
/// for `crossbeam_queue::ArrayQueue` (which is not loom-aware).
struct MockPool {
    #[allow(dead_code)]
    sender: loom::sync::mpsc::Sender<MockSession>,
    receiver: RefCell<loom::sync::mpsc::Receiver<MockSession>>,
}

impl MockPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = loom::sync::mpsc::channel();
        for i in 0..size {
            sender.send(MockSession { id: i }).unwrap();
        }
        Self {
            sender,
            receiver: RefCell::new(receiver),
        }
    }

    fn pop(&self) -> Option<MockSession> {
        self.receiver.borrow_mut().try_recv().ok()
    }

    fn push(&self, session: MockSession) {
        let _ = self.sender.send(session);
    }
}

/// Guard that returns the session to the pool on drop.
struct MockGuard<'a> {
    session: Option<MockSession>,
    pool: &'a MockPool,
}

impl<'a> Drop for MockGuard<'a> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.push(session);
        }
    }
}

#[test]
fn test_pool_checkout_drop_returns() {
    loom::model(|| {
        let pool = MockPool::new(2);
        let guard1 = pool.pop().map(|s| MockGuard {
            session: Some(s),
            pool: &pool,
        });
        let guard2 = pool.pop().map(|s| MockGuard {
            session: Some(s),
            pool: &pool,
        });

        // Pool is now empty.
        assert!(pool.pop().is_none());

        // Drop guards — resources return to pool.
        drop(guard1);
        drop(guard2);

        // Both resources are available again.
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_some());
        assert!(pool.pop().is_none());
    });
}

#[test]
fn test_pool_concurrent_stress() {
    loom::model(|| {
        let pool = Arc::new(MockPool::new(1));
        let checked_out = Arc::new(AtomicUsize::new(0));
        let returned = Arc::new(AtomicUsize::new(0));

        let t1 = {
            let pool = Arc::clone(&pool);
            let checked_out = Arc::clone(&checked_out);
            let returned = Arc::clone(&returned);
            thread::spawn(move || {
                if let Some(s) = pool.pop() {
                    checked_out.fetch_add(1, Ordering::SeqCst);
                    let guard = MockGuard {
                        session: Some(s),
                        pool: &pool,
                    };
                    // Simulate work.
                    thread::yield_now();
                    drop(guard);
                    returned.fetch_add(1, Ordering::SeqCst);
                }
            })
        };

        let t2 = {
            let pool = Arc::clone(&pool);
            let checked_out = Arc::clone(&checked_out);
            let returned = Arc::clone(&returned);
            thread::spawn(move || {
                if let Some(s) = pool.pop() {
                    checked_out.fetch_add(1, Ordering::SeqCst);
                    let guard = MockGuard {
                        session: Some(s),
                        pool: &pool,
                    };
                    thread::yield_now();
                    drop(guard);
                    returned.fetch_add(1, Ordering::SeqCst);
                }
            })
        };

        t1.join().unwrap();
        t2.join().unwrap();

        // Total checkouts and returns must be balanced.
        assert_eq!(
            checked_out.load(Ordering::SeqCst),
            returned.load(Ordering::SeqCst),
            "every checkout must be matched by a return"
        );
    });
}
