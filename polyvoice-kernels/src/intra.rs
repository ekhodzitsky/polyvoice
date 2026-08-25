//! Parked intra-op workers.
//!
//! `thread::scope` per convolution costs more than a 32-channel 3×3 at T=400.
//! Workers sleep on a condvar and claim job indices from an atomic counter.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, Once, OnceLock};

struct Pool {
    batch: AtomicUsize,
    next: AtomicUsize,
    n: AtomicUsize,
    func: AtomicUsize,
    data: AtomicUsize,
    left: AtomicUsize,
    mu: Mutex<()>,
    start: Condvar,
    done: Condvar,
    workers: usize,
}

static POOL: OnceLock<Pool> = OnceLock::new();
static SPAWN: Once = Once::new();

fn pool() -> &'static Pool {
    let p = POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(0)
            .min(2);
        Pool {
            batch: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            n: AtomicUsize::new(0),
            func: AtomicUsize::new(0),
            data: AtomicUsize::new(0),
            left: AtomicUsize::new(0),
            mu: Mutex::new(()),
            start: Condvar::new(),
            done: Condvar::new(),
            workers,
        }
    });
    SPAWN.call_once(|| {
        for _ in 0..p.workers {
            std::thread::spawn(worker_loop);
        }
    });
    p
}

fn worker_loop() {
    let p = POOL.get().expect("intra pool");
    let mut seen = 0usize;
    loop {
        {
            let mut g = p.mu.lock().unwrap();
            while p.batch.load(Ordering::Acquire) == seen {
                g = p.start.wait(g).unwrap();
            }
            seen = p.batch.load(Ordering::Acquire);
        }
        let n = p.n.load(Ordering::Relaxed);
        let func: fn(*const (), usize) =
            unsafe { std::mem::transmute(p.func.load(Ordering::Relaxed)) };
        let data = p.data.load(Ordering::Relaxed) as *const ();
        loop {
            let i = p.next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            func(data, i);
        }
        if p.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = p.mu.lock().unwrap();
            p.done.notify_one();
        }
    }
}

/// Run `f(0..n)` on the caller plus parked workers. `f` must be safe to call
/// concurrently for distinct indices.
///
/// The pool is a single batch slot: callers are serialized so two ResNets
/// cannot publish overlapping jobs (that deadlocks `left`).
pub fn run<F: Fn(usize) + Sync>(n: usize, f: F) {
    let n = n.max(1);
    let p = pool();
    if n == 1 || p.workers == 0 {
        f(0);
        return;
    }
    static RUN: Mutex<()> = Mutex::new(());
    let _serial = match RUN.try_lock() {
        Ok(g) => g,
        Err(_) => {
            for i in 0..n {
                f(i);
            }
            return;
        }
    };
    struct Env<F>(F);
    let env = Env(f);
    fn call<F: Fn(usize)>(ptr: *const (), i: usize) {
        // SAFETY: dispatch waits for every worker before `env` drops.
        let env = unsafe { &*ptr.cast::<Env<F>>() };
        (env.0)(i);
    }
    dispatch(p, n, call::<F>, (&raw const env).cast::<()>());
}

fn dispatch(p: &Pool, n: usize, func: fn(*const (), usize), data: *const ()) {
    p.func.store(func as usize, Ordering::Relaxed);
    p.data.store(data as usize, Ordering::Relaxed);
    p.n.store(n, Ordering::Relaxed);
    p.next.store(0, Ordering::Relaxed);
    p.left.store(p.workers, Ordering::SeqCst);
    {
        let _g = p.mu.lock().unwrap();
        p.batch.fetch_add(1, Ordering::Release);
        p.start.notify_all();
    }
    loop {
        let i = p.next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        func(data, i);
    }
    let mut g = p.mu.lock().unwrap();
    while p.left.load(Ordering::Acquire) != 0 {
        g = p.done.wait(g).unwrap();
    }
}
