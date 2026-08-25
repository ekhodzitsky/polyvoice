//! System OpenBLAS `cblas_sgemm` on Linux (same role as Accelerate on Apple).
//!
//! Linked only when `build.rs` finds OpenBLAS via pkg-config. Callers pin
//! `OPENBLAS_NUM_THREADS=1` so the BLAS pool does not fight window/embed workers.

#![cfg(linux_cblas)]

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

unsafe extern "C" {
    fn openblas_set_num_threads(n: i32);
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `C = alpha * A[m,k] @ B[k,n] + beta * C`.
///
/// # Safety
/// `a`, `b`, `c` must be valid for `m*k`, `k*n`, `m*n` elements; `m,n,k` fit `i32`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_rowmajor(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a,
            k as i32,
            b,
            n as i32,
            beta,
            c,
            n as i32,
        );
    }
}

/// Stop OpenBLAS from spawning its own pool. Safe to call many times.
pub fn pin_to_one_thread() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: called from model load, before the embed/window pools start.
        unsafe {
            std::env::set_var("OPENBLAS_NUM_THREADS", "1");
            std::env::set_var("OMP_NUM_THREADS", "1");
            openblas_set_num_threads(1);
        }
    });
}
