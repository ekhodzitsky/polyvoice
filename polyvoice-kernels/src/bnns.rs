//! Apple BNNS convolution — Winograd / implicit GEMM, no im2col.
//!
//! Filters are pooled by weight identity + input spatial size + fused ReLU.
//! BNNS is not thread-safe on a shared filter, so apply checks a filter out
//! of the pool. `VECLIB_MAXIMUM_THREADS=1` is set at model load so Accelerate
//! GEMM does not spawn extra workers; BNNS uses 2 threads on large feature
//! maps and 1 otherwise. Override with `POLYVOICE_BNNS_THREADS`.

#![cfg(target_vendor = "apple")]

use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

unsafe extern "C" {
    fn pv_bnns_conv_create(
        ic: i32,
        ih: i32,
        iw: i32,
        oc: i32,
        k: i32,
        stride: i32,
        pad: i32,
        relu: i32,
        n_threads: i32,
        weight: *const f32,
        bias: *const f32,
    ) -> *mut c_void;
    fn pv_bnns_conv_apply(filter: *mut c_void, input: *const f32, output: *mut f32) -> i32;
    fn pv_bnns_conv_apply_n(
        filter: *mut c_void,
        n: usize,
        input: *const f32,
        in_stride: usize,
        output: *mut f32,
        out_stride: usize,
    ) -> i32;
    fn pv_bnns_conv_destroy(filter: *mut c_void);
}

struct Filter {
    raw: NonNull<c_void>,
}

// SAFETY: a filter is only applied by the thread that currently owns it.
unsafe impl Send for Filter {}

impl Drop for Filter {
    fn drop(&mut self) {
        unsafe { pv_bnns_conv_destroy(self.raw.as_ptr()) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    wptr: usize,
    ih: u16,
    iw: u16,
    ic: u16,
    oc: u16,
    k: u8,
    stride: u8,
    pad: u8,
    relu: bool,
    nth: u8,
}

struct Pool {
    ready: HashMap<Key, Vec<Filter>>,
    failed: HashSet<Key>,
}

/// Same-T windowed embedding reuses these across clips. Unique-T turns
/// (Vox-3) never hit; 24 is one ResNet minus the evicted tail.
const MAX_KEYS: usize = 24;

static CREATE_COUNT: AtomicU64 = AtomicU64::new(0);
static HIT_COUNT: AtomicU64 = AtomicU64::new(0);
static CREATE_NS: AtomicU64 = AtomicU64::new(0);

/// `(creates, cache hits, create-nanoseconds)` since process start.
pub fn prof() -> (u64, u64, u64) {
    (
        CREATE_COUNT.load(Ordering::Relaxed),
        HIT_COUNT.load(Ordering::Relaxed),
        CREATE_NS.load(Ordering::Relaxed),
    )
}

fn pool() -> &'static Mutex<Pool> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        Mutex::new(Pool {
            ready: HashMap::new(),
            failed: HashSet::new(),
        })
    })
}

fn enabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    !*OFF.get_or_init(|| std::env::var_os("POLYVOICE_NO_BNNS").is_some())
}

fn bnns_threads_override() -> Option<i32> {
    static N: OnceLock<Option<i32>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("POLYVOICE_BNNS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &i32| n >= 0)
    })
}

/// Isolated 3×3 apply is ~1.5–2× faster at 2 threads than at 1, even on the
/// first call. Auto (`0`) oversubscribes next to the embed pool and bloats
/// RSS, so we only spend a second worker on large spatial maps.
fn bnns_threads_for(ih: usize, iw: usize) -> i32 {
    if let Some(n) = bnns_threads_override() {
        return n;
    }
    if ih.saturating_mul(iw) >= 8_000 { 2 } else { 1 }
}

/// Fill `y` with `conv(x)`. False → caller must use the im2col path.
#[allow(clippy::too_many_arguments)]
pub fn try_conv2d(
    weight: &[f32],
    bias: &[f32],
    oc: usize,
    ic: usize,
    k: usize,
    stride: usize,
    pad: usize,
    relu: bool,
    weight_id: usize,
    x: &Tensor,
    y: &mut Tensor,
) -> bool {
    if !enabled() || x.n == 0 {
        return false;
    }
    let (Ok(ih), Ok(iw), Ok(icu), Ok(ocu), Ok(ku), Ok(su), Ok(pu)) = (
        u16::try_from(x.h),
        u16::try_from(x.w),
        u16::try_from(ic),
        u16::try_from(oc),
        u8::try_from(k),
        u8::try_from(stride),
        u8::try_from(pad),
    ) else {
        return false;
    };
    let nth = bnns_threads_for(x.h, x.w);
    let key = Key {
        wptr: weight_id,
        ih,
        iw,
        ic: icu,
        oc: ocu,
        k: ku,
        stride: su,
        pad: pu,
        relu,
        nth: u8::try_from(nth).unwrap_or(2),
    };
    let mut guard = match pool().lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if guard.failed.contains(&key) {
        return false;
    }
    let filter = guard.ready.entry(key).or_default().pop();
    let filter = match filter {
        Some(f) => {
            HIT_COUNT.fetch_add(1, Ordering::Relaxed);
            f
        }
        None => {
            evict_old_keys(&mut guard, key);
            let t0 = Instant::now();
            let ptr = unsafe {
                pv_bnns_conv_create(
                    ic as i32,
                    x.h as i32,
                    x.w as i32,
                    oc as i32,
                    k as i32,
                    stride as i32,
                    pad as i32,
                    i32::from(relu),
                    nth,
                    weight.as_ptr(),
                    bias.as_ptr(),
                )
            };
            CREATE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CREATE_COUNT.fetch_add(1, Ordering::Relaxed);
            match NonNull::new(ptr) {
                Some(raw) => Filter { raw },
                None => {
                    guard.failed.insert(key);
                    return false;
                }
            }
        }
    };
    drop(guard);
    let rc = if x.n == 1 {
        unsafe { pv_bnns_conv_apply(filter.raw.as_ptr(), x.data.as_ptr(), y.data.as_mut_ptr()) }
    } else {
        let in_stride = ic.saturating_mul(x.h).saturating_mul(x.w);
        let out_stride = oc.saturating_mul(y.h).saturating_mul(y.w);
        unsafe {
            pv_bnns_conv_apply_n(
                filter.raw.as_ptr(),
                x.n,
                x.data.as_ptr(),
                in_stride,
                y.data.as_mut_ptr(),
                out_stride,
            )
        }
    };
    if let Ok(mut g) = pool().lock() {
        let bucket = g.ready.entry(key).or_default();
        if bucket.is_empty() {
            bucket.push(filter);
        }
    }
    rc == 0
}

fn evict_old_keys(pool: &mut Pool, keep: Key) {
    while pool.ready.len() >= MAX_KEYS {
        let victim = pool.ready.keys().copied().find(|k| *k != keep);
        match victim {
            Some(v) => {
                pool.ready.remove(&v);
            }
            None => break,
        }
    }
}
