//! Bounded activation scratch (gigastt-style: 1 recycled slot, soft cap).
//!
//! Hot paths take a `Vec<f32>`, grow it for this forward, then put it back.
//! Concurrent forwards allocate extra vecs; at most one spare is kept, and
//! [`reclaim`] drops it after each file so the process high-water mark is
//! not pinned for the rest of the run.

use std::sync::{Mutex, OnceLock};

/// One recycled buffer. Extra puts are dropped.
const POOL_SLOTS: usize = 1;
/// ~8 MiB of f32 — a typical 10 s packed LSTM `xw` (seq×batch×4H). Larger
/// spikes shrink back to this on return.
const SOFT_CAP_ELEMS: usize = 2 * 1024 * 1024;

struct ScratchPool {
    free: Vec<Vec<f32>>,
}

fn pool() -> &'static Mutex<ScratchPool> {
    static POOL: OnceLock<Mutex<ScratchPool>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(ScratchPool { free: Vec::new() }))
}

fn lock() -> std::sync::MutexGuard<'static, ScratchPool> {
    pool().lock().unwrap_or_else(|e| e.into_inner())
}

/// Take a buffer with at least `need` elements (zero-filled on growth).
pub fn take_f32(need: usize) -> Vec<f32> {
    let mut v = {
        let mut g = lock();
        g.free.pop().unwrap_or_default()
    };
    if v.len() < need {
        v.resize(need, 0.0);
    } else {
        v.truncate(need);
        v.fill(0.0);
    }
    v
}

/// Return a buffer to the pool. Capacity above [`SOFT_CAP_ELEMS`] is released.
pub fn put_f32(mut v: Vec<f32>) {
    v.clear();
    if v.capacity() > SOFT_CAP_ELEMS {
        v.shrink_to(SOFT_CAP_ELEMS);
    }
    let mut g = lock();
    if g.free.len() < POOL_SLOTS {
        g.free.push(v);
    }
}

/// Drop the recycled slot after a file so activations are not pinned.
pub fn reclaim() {
    lock().free.clear();
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pool_keeps_one_slot_and_shrinks() {
        reclaim();
        let a = take_f32(SOFT_CAP_ELEMS + 1024);
        assert_eq!(a.len(), SOFT_CAP_ELEMS + 1024);
        put_f32(a);
        let b = take_f32(8);
        assert_eq!(b.len(), 8);
        assert!(b.capacity() <= SOFT_CAP_ELEMS);
        put_f32(b);
        reclaim();
        let c = take_f32(4);
        assert_eq!(c.capacity(), 4);
        put_f32(c);
        reclaim();
    }

    #[test]
    fn extra_puts_are_dropped() {
        reclaim();
        put_f32(vec![1.0; 16]);
        put_f32(vec![2.0; 32]);
        {
            let g = lock();
            assert_eq!(g.free.len(), POOL_SLOTS);
        }
        reclaim();
    }
}
