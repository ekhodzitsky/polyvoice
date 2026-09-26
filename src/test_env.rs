//! Serialised process-environment mutation for unit tests.
//!
//! `std::env::set_var` changes process-global state, so tests that set or
//! read `POLYVOICE_*` overrides race under the default multi-threaded test
//! harness (nextest isolates processes; `cargo test` does not). Every such
//! test takes [`lock`] first. The guard records the prior value of each
//! variable it touches and restores it on drop, so a caller's pre-existing
//! environment survives the test run.

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive access to the process environment plus undo on drop.
pub(crate) struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(OsString, Option<OsString>)>,
}

/// Take the environment lock. Hold the guard for the whole test, including
/// any call that reads the variables under test.
pub(crate) fn lock() -> EnvGuard {
    EnvGuard {
        _lock: ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner()),
        saved: Vec::new(),
    }
}

impl EnvGuard {
    fn save(&mut self, key: &OsStr) {
        if !self.saved.iter().any(|(k, _)| k == key) {
            self.saved.push((key.to_owned(), std::env::var_os(key)));
        }
    }

    pub(crate) fn set(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.save(key);
        // SAFETY: the global lock serialises every test that mutates or reads
        // these overrides, so no other test observes a torn environment.
        unsafe { std::env::set_var(key, value) };
    }

    pub(crate) fn remove(&mut self, key: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.save(key);
        // SAFETY: see `set`.
        unsafe { std::env::remove_var(key) };
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prior) in self.saved.drain(..).rev() {
            // SAFETY: still under the lock; restores the caller's environment.
            unsafe {
                match prior {
                    Some(value) => std::env::set_var(&key, value),
                    None => std::env::remove_var(&key),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn guard_restores_prior_value_and_absence_on_drop() {
        const PRESENT: &str = "POLYVOICE_TEST_ENV_PRESENT";
        const ABSENT: &str = "POLYVOICE_TEST_ENV_ABSENT";
        let mut guard = super::lock();
        // SAFETY: under the lock; seeds the pre-existing state the guard
        // must hand back.
        unsafe {
            std::env::set_var(PRESENT, "prior");
            std::env::remove_var(ABSENT);
        }
        guard.set(PRESENT, "during");
        guard.set(ABSENT, "during");
        guard.remove(PRESENT);
        assert!(std::env::var_os(PRESENT).is_none());
        assert_eq!(std::env::var(ABSENT).as_deref(), Ok("during"));
        drop(guard);
        assert_eq!(std::env::var(PRESENT).as_deref(), Ok("prior"));
        assert!(std::env::var_os(ABSENT).is_none());

        let _guard = super::lock();
        // SAFETY: under the lock; leaves nothing behind for other tests.
        unsafe { std::env::remove_var(PRESENT) };
    }
}
