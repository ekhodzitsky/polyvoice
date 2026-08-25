//! Row-major GEMM used by LSTM / Linear: `C[m,n] = A[m,k] @ B[k,n]`.
//!
//! `B` is stored **transposed from ONNX** (`[k, n]` instead of `[n, k]`) so
//! the inner kernel is a saxpy over a contiguous `n`-vector. The 8-wide
//! unroll is written so LLVM autovectorizes on aarch64 and x86_64.

#[inline]
pub fn gemm_bias(a: &[f32], b: &[f32], bias: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));
    debug_assert_eq!(bias.len(), n);
    debug_assert_eq!(c.len(), m.saturating_mul(n));
    #[cfg(not(target_vendor = "apple"))]
    if rten_worth_it(m, n, k) && crate::rten_matmul::gemm_colbias(a, b, bias, c, m, n, k) {
        return;
    }
    if try_accel_colbias(a, b, bias, c, m, n, k) {
        return;
    }
    for mi in 0..m {
        let crow = &mut c[mi * n..mi * n + n];
        crow.copy_from_slice(bias);
        let arow = &a[mi * k..mi * k + k];
        for (kk, &av) in arow.iter().enumerate() {
            saxpy(av, &b[kk * n..kk * n + n], crow);
        }
    }
}

/// `C[m,n] = A[m,k] @ B[k,n] + bias[m]` (bias broadcast across each row).
#[inline]
pub fn gemm_bias_row(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));
    debug_assert_eq!(bias.len(), m);
    debug_assert_eq!(c.len(), m.saturating_mul(n));
    // Small N (LSTM-like) stays on the long saxpy; conv spatial maps use a
    // register-blocked kernel so B's N-panel stays in registers across M.
    #[cfg(not(target_vendor = "apple"))]
    if rten_worth_it(m, n, k) && crate::rten_matmul::gemm_rowbias(a, b, bias, c, m, n, k) {
        return;
    }
    if try_accel_rowbias(a, b, bias, c, m, n, k) {
        return;
    }
    if n < 8 || m < 2 {
        for mi in 0..m {
            let crow = &mut c[mi * n..mi * n + n];
            crow.fill(bias[mi]);
            let arow = &a[mi * k..mi * k + k];
            for (kk, &av) in arow.iter().enumerate() {
                saxpy(av, &b[kk * n..kk * n + n], crow);
            }
        }
        return;
    }
    gemm_rowbias_blocked(a, b, bias, c, m, n, k);
}

/// FLOPs below this stay on the in-crate kernel (cblas launch cost).
const ACCEL_MIN_FLOPS: usize = 32 * 64 * 32;

/// rten-gemm wins on fat N×K even when M is the LSTM batch (8).
#[cfg(not(target_vendor = "apple"))]
fn rten_worth_it(m: usize, n: usize, k: usize) -> bool {
    n >= 64 && k >= 32 && m.saturating_mul(n).saturating_mul(k) >= 8 * 64 * 32
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn accel_worth_it(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(n).saturating_mul(k) >= ACCEL_MIN_FLOPS
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn sgemm(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    #[cfg(target_vendor = "apple")]
    unsafe {
        crate::accelerate::sgemm_rowmajor(a, b, c, m, n, k, alpha, beta);
    }
    #[cfg(linux_cblas)]
    unsafe {
        crate::linux_cblas::sgemm_rowmajor(a, b, c, m, n, k, alpha, beta);
    }
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn pin_blas() {
    #[cfg(target_vendor = "apple")]
    crate::accelerate::pin_to_one_thread();
    #[cfg(linux_cblas)]
    crate::linux_cblas::pin_to_one_thread();
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn try_accel_rowbias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    if !accel_worth_it(m, n, k) {
        return false;
    }
    pin_blas();
    for mi in 0..m {
        c[mi * n..mi * n + n].fill(bias[mi]);
    }
    // SAFETY: slices match m,n,k; C was just written.
    sgemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, 1.0, 1.0);
    true
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn try_accel_colbias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    if !accel_worth_it(m, n, k) {
        return false;
    }
    pin_blas();
    for mi in 0..m {
        c[mi * n..mi * n + n].copy_from_slice(bias);
    }
    sgemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, 1.0, 1.0);
    true
}

#[cfg(any(target_vendor = "apple", linux_cblas))]
fn try_accel_add(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> bool {
    if !accel_worth_it(m, n, k) {
        return false;
    }
    // Recurrent LSTM: M = batch, N = 4H, K = H. OpenBLAS re-packs B every
    // timestep; the in-crate kernel wins until M is large enough to amortize.
    #[cfg(linux_cblas)]
    if m < 128 {
        return false;
    }
    pin_blas();
    sgemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, 1.0, 1.0);
    true
}

#[cfg(not(any(target_vendor = "apple", linux_cblas)))]
fn try_accel_rowbias(
    _: &[f32],
    _: &[f32],
    _: &[f32],
    _: &mut [f32],
    _: usize,
    _: usize,
    _: usize,
) -> bool {
    false
}

#[cfg(not(any(target_vendor = "apple", linux_cblas)))]
fn try_accel_colbias(
    _: &[f32],
    _: &[f32],
    _: &[f32],
    _: &mut [f32],
    _: usize,
    _: usize,
    _: usize,
) -> bool {
    false
}

#[cfg(not(any(target_vendor = "apple", linux_cblas)))]
fn try_accel_add(_: &[f32], _: &[f32], _: &mut [f32], _: usize, _: usize, _: usize) -> bool {
    false
}

fn gemm_rowbias_blocked(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let mut m0 = 0;
    while m0 + 8 <= m {
        let mut n0 = 0;
        while n0 + 8 <= n {
            #[cfg(target_arch = "aarch64")]
            neon_rowbias_8x8(a, b, c, m0, n0, n, k, bias);
            #[cfg(not(target_arch = "aarch64"))]
            kernel_rowbias::<8, 8>(a, b, c, m0, n0, n, k, bias);
            n0 += 8;
        }
        if n0 < n {
            kernel_rowbias_tail(a, b, c, m0, 8, n0, n, n, k, bias);
        }
        m0 += 8;
    }
    while m0 + 4 <= m {
        let mut n0 = 0;
        while n0 + 8 <= n {
            #[cfg(target_arch = "aarch64")]
            neon_rowbias_4x8(a, b, c, m0, n0, n, k, bias);
            #[cfg(not(target_arch = "aarch64"))]
            kernel_rowbias::<4, 8>(a, b, c, m0, n0, n, k, bias);
            n0 += 8;
        }
        if n0 < n {
            kernel_rowbias_tail(a, b, c, m0, 4, n0, n, n, k, bias);
        }
        m0 += 4;
    }
    if m0 < m {
        kernel_rowbias_tail(a, b, c, m0, m - m0, 0, n, n, k, bias);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel_rowbias<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    n0: usize,
    n: usize,
    k: usize,
    bias: &[f32],
) {
    let mut acc = [[0.0f32; NR]; MR];
    for i in 0..MR {
        let bv = bias[m0 + i];
        for j in 0..NR {
            acc[i][j] = bv;
        }
    }
    for p in 0..k {
        let brow = &b[p * n + n0..p * n + n0 + NR];
        for i in 0..MR {
            let av = a[(m0 + i) * k + p];
            for j in 0..NR {
                acc[i][j] += av * brow[j];
            }
        }
    }
    for i in 0..MR {
        c[(m0 + i) * n + n0..(m0 + i) * n + n0 + NR].copy_from_slice(&acc[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn neon_rowbias_4x8(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    n0: usize,
    n: usize,
    k: usize,
    bias: &[f32],
) {
    use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_n_f32, vld1q_f32, vst1q_f32};
    // SAFETY: caller guarantees m0+3 < m and n0+7 < n, so every load/store
    // stays inside the a/b/c slices. f32 NEON unaligned loads are defined.
    unsafe {
        let mut c00: float32x4_t = vdupq_n_f32(bias[m0]);
        let mut c01: float32x4_t = vdupq_n_f32(bias[m0]);
        let mut c10: float32x4_t = vdupq_n_f32(bias[m0 + 1]);
        let mut c11: float32x4_t = vdupq_n_f32(bias[m0 + 1]);
        let mut c20: float32x4_t = vdupq_n_f32(bias[m0 + 2]);
        let mut c21: float32x4_t = vdupq_n_f32(bias[m0 + 2]);
        let mut c30: float32x4_t = vdupq_n_f32(bias[m0 + 3]);
        let mut c31: float32x4_t = vdupq_n_f32(bias[m0 + 3]);
        let a0 = a.as_ptr().add(m0 * k);
        let a1 = a.as_ptr().add((m0 + 1) * k);
        let a2 = a.as_ptr().add((m0 + 2) * k);
        let a3 = a.as_ptr().add((m0 + 3) * k);
        let bp = b.as_ptr();
        for p in 0..k {
            let bptr = bp.add(p * n + n0);
            let b0 = vld1q_f32(bptr);
            let b1 = vld1q_f32(bptr.add(4));
            c00 = vfmaq_n_f32(c00, b0, *a0.add(p));
            c01 = vfmaq_n_f32(c01, b1, *a0.add(p));
            c10 = vfmaq_n_f32(c10, b0, *a1.add(p));
            c11 = vfmaq_n_f32(c11, b1, *a1.add(p));
            c20 = vfmaq_n_f32(c20, b0, *a2.add(p));
            c21 = vfmaq_n_f32(c21, b1, *a2.add(p));
            c30 = vfmaq_n_f32(c30, b0, *a3.add(p));
            c31 = vfmaq_n_f32(c31, b1, *a3.add(p));
        }
        let cp = c.as_mut_ptr();
        vst1q_f32(cp.add(m0 * n + n0), c00);
        vst1q_f32(cp.add(m0 * n + n0 + 4), c01);
        vst1q_f32(cp.add((m0 + 1) * n + n0), c10);
        vst1q_f32(cp.add((m0 + 1) * n + n0 + 4), c11);
        vst1q_f32(cp.add((m0 + 2) * n + n0), c20);
        vst1q_f32(cp.add((m0 + 2) * n + n0 + 4), c21);
        vst1q_f32(cp.add((m0 + 3) * n + n0), c30);
        vst1q_f32(cp.add((m0 + 3) * n + n0 + 4), c31);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn neon_rowbias_8x8(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    n0: usize,
    n: usize,
    k: usize,
    bias: &[f32],
) {
    neon_rowbias_4x8(a, b, c, m0, n0, n, k, bias);
    neon_rowbias_4x8(a, b, c, m0 + 4, n0, n, k, bias);
}

#[allow(clippy::too_many_arguments)]
fn kernel_rowbias_tail(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    mr: usize,
    n0: usize,
    n1: usize,
    n: usize,
    k: usize,
    bias: &[f32],
) {
    for i in 0..mr {
        let dest = &mut c[(m0 + i) * n + n0..(m0 + i) * n + n1];
        dest.fill(bias[m0 + i]);
        for p in 0..k {
            let av = a[(m0 + i) * k + p];
            let brow = &b[p * n + n0..p * n + n1];
            for (d, &bv) in dest.iter_mut().zip(brow.iter()) {
                *d += av * bv;
            }
        }
    }
}

/// `C += A @ B`.
#[inline]
pub fn gemm_add(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    debug_assert_eq!(a.len(), m.saturating_mul(k));
    debug_assert_eq!(b.len(), k.saturating_mul(n));
    debug_assert_eq!(c.len(), m.saturating_mul(n));
    #[cfg(not(target_vendor = "apple"))]
    if rten_worth_it(m, n, k) && crate::rten_matmul::gemm_add(a, b, c, m, n, k) {
        return;
    }
    if try_accel_add(a, b, c, m, n, k) {
        return;
    }
    if n >= 8 && m >= 2 {
        gemm_add_blocked(a, b, c, m, n, k);
        return;
    }
    for mi in 0..m {
        let crow = &mut c[mi * n..mi * n + n];
        let arow = &a[mi * k..mi * k + k];
        for (kk, &av) in arow.iter().enumerate() {
            saxpy(av, &b[kk * n..kk * n + n], crow);
        }
    }
}

fn gemm_add_blocked(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut m0 = 0;
    while m0 + 8 <= m {
        let mut n0 = 0;
        while n0 + 8 <= n {
            #[cfg(target_arch = "aarch64")]
            neon_add_8x8(a, b, c, m0, n0, n, k);
            #[cfg(not(target_arch = "aarch64"))]
            kernel_add::<8, 8>(a, b, c, m0, n0, n, k);
            n0 += 8;
        }
        if n0 < n {
            kernel_add_tail(a, b, c, m0, 8, n0, n, n, k);
        }
        m0 += 8;
    }
    while m0 + 4 <= m {
        let mut n0 = 0;
        while n0 + 8 <= n {
            #[cfg(target_arch = "aarch64")]
            neon_add_4x8(a, b, c, m0, n0, n, k);
            #[cfg(not(target_arch = "aarch64"))]
            kernel_add::<4, 8>(a, b, c, m0, n0, n, k);
            n0 += 8;
        }
        if n0 < n {
            kernel_add_tail(a, b, c, m0, 4, n0, n, n, k);
        }
        m0 += 4;
    }
    if m0 < m {
        kernel_add_tail(a, b, c, m0, m - m0, 0, n, n, k);
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kernel_add<const MR: usize, const NR: usize>(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    n0: usize,
    n: usize,
    k: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for i in 0..MR {
        acc[i].copy_from_slice(&c[(m0 + i) * n + n0..(m0 + i) * n + n0 + NR]);
    }
    for p in 0..k {
        let brow = &b[p * n + n0..p * n + n0 + NR];
        for i in 0..MR {
            let av = a[(m0 + i) * k + p];
            for j in 0..NR {
                acc[i][j] += av * brow[j];
            }
        }
    }
    for i in 0..MR {
        c[(m0 + i) * n + n0..(m0 + i) * n + n0 + NR].copy_from_slice(&acc[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_add_4x8(a: &[f32], b: &[f32], c: &mut [f32], m0: usize, n0: usize, n: usize, k: usize) {
    use std::arch::aarch64::{float32x4_t, vfmaq_n_f32, vld1q_f32, vst1q_f32};
    // SAFETY: caller guarantees m0+3 < m and n0+7 < n.
    unsafe {
        let cp = c.as_mut_ptr();
        let mut c00: float32x4_t = vld1q_f32(cp.add(m0 * n + n0));
        let mut c01: float32x4_t = vld1q_f32(cp.add(m0 * n + n0 + 4));
        let mut c10: float32x4_t = vld1q_f32(cp.add((m0 + 1) * n + n0));
        let mut c11: float32x4_t = vld1q_f32(cp.add((m0 + 1) * n + n0 + 4));
        let mut c20: float32x4_t = vld1q_f32(cp.add((m0 + 2) * n + n0));
        let mut c21: float32x4_t = vld1q_f32(cp.add((m0 + 2) * n + n0 + 4));
        let mut c30: float32x4_t = vld1q_f32(cp.add((m0 + 3) * n + n0));
        let mut c31: float32x4_t = vld1q_f32(cp.add((m0 + 3) * n + n0 + 4));
        let a0 = a.as_ptr().add(m0 * k);
        let a1 = a.as_ptr().add((m0 + 1) * k);
        let a2 = a.as_ptr().add((m0 + 2) * k);
        let a3 = a.as_ptr().add((m0 + 3) * k);
        let bp = b.as_ptr();
        for p in 0..k {
            let bptr = bp.add(p * n + n0);
            let b0 = vld1q_f32(bptr);
            let b1 = vld1q_f32(bptr.add(4));
            c00 = vfmaq_n_f32(c00, b0, *a0.add(p));
            c01 = vfmaq_n_f32(c01, b1, *a0.add(p));
            c10 = vfmaq_n_f32(c10, b0, *a1.add(p));
            c11 = vfmaq_n_f32(c11, b1, *a1.add(p));
            c20 = vfmaq_n_f32(c20, b0, *a2.add(p));
            c21 = vfmaq_n_f32(c21, b1, *a2.add(p));
            c30 = vfmaq_n_f32(c30, b0, *a3.add(p));
            c31 = vfmaq_n_f32(c31, b1, *a3.add(p));
        }
        vst1q_f32(cp.add(m0 * n + n0), c00);
        vst1q_f32(cp.add(m0 * n + n0 + 4), c01);
        vst1q_f32(cp.add((m0 + 1) * n + n0), c10);
        vst1q_f32(cp.add((m0 + 1) * n + n0 + 4), c11);
        vst1q_f32(cp.add((m0 + 2) * n + n0), c20);
        vst1q_f32(cp.add((m0 + 2) * n + n0 + 4), c21);
        vst1q_f32(cp.add((m0 + 3) * n + n0), c30);
        vst1q_f32(cp.add((m0 + 3) * n + n0 + 4), c31);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_add_8x8(a: &[f32], b: &[f32], c: &mut [f32], m0: usize, n0: usize, n: usize, k: usize) {
    neon_add_4x8(a, b, c, m0, n0, n, k);
    neon_add_4x8(a, b, c, m0 + 4, n0, n, k);
}

#[allow(clippy::too_many_arguments)]
fn kernel_add_tail(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m0: usize,
    mr: usize,
    n0: usize,
    n1: usize,
    n: usize,
    k: usize,
) {
    for i in 0..mr {
        let dest = &mut c[(m0 + i) * n + n0..(m0 + i) * n + n1];
        for p in 0..k {
            let av = a[(m0 + i) * k + p];
            let brow = &b[p * n + n0..p * n + n1];
            for (d, &bv) in dest.iter_mut().zip(brow.iter()) {
                *d += av * bv;
            }
        }
    }
}

/// `y[i] += alpha * x[i]`.
#[inline]
pub fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let n = y.len();
    let mut i = 0;
    while i + 8 <= n {
        y[i] += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        y[i + 4] += alpha * x[i + 4];
        y[i + 5] += alpha * x[i + 5];
        y[i + 6] += alpha * x[i + 6];
        y[i + 7] += alpha * x[i + 7];
        i += 8;
    }
    while i < n {
        y[i] += alpha * x[i];
        i += 1;
    }
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut acc = acc0 + acc1 + acc2 + acc3;
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// Transpose `[rows, cols]` row-major → `[cols, rows]`.
pub fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows.saturating_mul(cols));
    let mut dst = vec![0.0f32; rows.saturating_mul(cols)];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn gemm_bias_row_tiled_matches_naive_wide() {
        let m = 4;
        let n = 130;
        let k = 7;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let bias: Vec<f32> = (0..m).map(|i| i as f32 * 0.3).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_bias_row(&a, &b, &bias, &mut c, m, n, k);
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = bias[mi];
                for kk in 0..k {
                    acc += a[mi * k + kk] * b[kk * n + ni];
                }
                let got = c[mi * n + ni];
                assert!((got - acc).abs() < 1e-4, "m={mi} n={ni} {got} vs {acc}");
            }
        }
    }

    #[test]
    fn gemm_bias_matches_naive() {
        let m = 3;
        let n = 5;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b_kn: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_bias(&a, &b_kn, &bias, &mut c, m, n, k);
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = bias[ni];
                for kk in 0..k {
                    acc += a[mi * k + kk] * b_kn[kk * n + ni];
                }
                let got = c[mi * n + ni];
                assert!((got - acc).abs() < 1e-5, "m={mi} n={ni} {got} vs {acc}");
            }
        }
    }

    #[test]
    fn gemm_add_blocked_matches_naive() {
        let m = 7;
        let n = 20;
        let k = 11;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let mut c: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01).collect();
        let mut want = c.clone();
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = want[mi * n + ni];
                for kk in 0..k {
                    acc += a[mi * k + kk] * b[kk * n + ni];
                }
                want[mi * n + ni] = acc;
            }
        }
        gemm_add(&a, &b, &mut c, m, n, k);
        for i in 0..c.len() {
            assert!(
                (c[i] - want[i]).abs() < 1e-4,
                "i={i} {} vs {}",
                c[i],
                want[i]
            );
        }
    }

    #[test]
    fn gemm_add_lstm_shape_matches_naive() {
        // Recurrent LSTM step: M=batch, N=4H, K=H — rten on Linux.
        let m = 8;
        let n = 128;
        let k = 64;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.007 - 0.2).collect();
        let mut c: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.002).collect();
        let mut want = c.clone();
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = want[mi * n + ni];
                for kk in 0..k {
                    acc += a[mi * k + kk] * b[kk * n + ni];
                }
                want[mi * n + ni] = acc;
            }
        }
        gemm_add(&a, &b, &mut c, m, n, k);
        let mut max = 0.0f32;
        for i in 0..c.len() {
            max = max.max((c[i] - want[i]).abs());
        }
        assert!(max < 5e-3, "lstm-shaped gemm_add maxabs={max}");
    }

    #[test]
    fn gemm_bias_wide_matches_naive() {
        let m = 16;
        let n = 128;
        let k = 60;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.008 - 0.15).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.05).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_bias(&a, &b, &bias, &mut c, m, n, k);
        let mut max = 0.0f32;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = bias[ni];
                for kk in 0..k {
                    acc += a[mi * k + kk] * b[kk * n + ni];
                }
                max = max.max((c[mi * n + ni] - acc).abs());
            }
        }
        assert!(max < 5e-3, "wide gemm_bias maxabs={max}");
    }

    #[test]
    fn saxpy_and_dot_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];
        saxpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
        assert!((dot(&x, &[1.0, 0.0, 1.0, 0.0]) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn transpose_roundtrip_shape() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&src, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
