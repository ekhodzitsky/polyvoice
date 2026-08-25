//! Packed GEMM via `rten-gemm` (pure Rust, INT8 + FP32).
//!
//! Used off-Apple: no AMX/BNNS, and system OpenBLAS in the Desktop Linux VM
//! is ~10 GFLOP/s. rten-gemm has NEON/dotprod/i8mm (and AVX/VNNI on x86).
//! Apple keeps BNNS / Accelerate. Rayon is pinned to 1 thread so this does
//! not fight window/embed workers.

#[cfg(not(target_vendor = "apple"))]
use crate::conv::Conv2d;
#[cfg(not(target_vendor = "apple"))]
use crate::tensor::Tensor;
#[cfg(not(target_vendor = "apple"))]
use rten_gemm::{
    BiasVector, ColOffsets, GemmExecutor, GemmInputA, GemmInputB, GemmOptions, Im2Col,
    PackedAMatrix, PackedBMatrix, QuantParams, RowOffsets,
};
#[cfg(not(target_vendor = "apple"))]
use rten_tensor::NdTensorView;
#[cfg(not(target_vendor = "apple"))]
use std::cell::RefCell;

pub fn pin_parallelism() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: model load, before worker pools start.
        unsafe {
            if std::env::var_os("RAYON_NUM_THREADS").is_none() {
                std::env::set_var("RAYON_NUM_THREADS", "1");
            }
        }
        #[cfg(linux_cblas)]
        crate::linux_cblas::pin_to_one_thread();
        #[cfg(target_vendor = "apple")]
        crate::accelerate::pin_to_one_thread();
    });
}

#[cfg(not(target_vendor = "apple"))]
struct I8Scratch {
    xq: Vec<i8>,
    padded: Vec<i8>,
    wu8: Vec<u8>,
    acc: Vec<i32>,
    a_zp: Vec<u8>,
    b_zp: Vec<i8>,
    row_c: Vec<i32>,
    row_y: Vec<i32>,
    row_x: Vec<i32>,
    col_y: Vec<i32>,
    col_x: Vec<i32>,
}

#[cfg(not(target_vendor = "apple"))]
thread_local! {
    static F32_EXEC: GemmExecutor<f32, f32, f32> = {
        pin_parallelism();
        GemmExecutor::new()
    };
    static I8_EXEC: GemmExecutor<u8, i8, i32> = {
        pin_parallelism();
        GemmExecutor::new()
    };
    static I8_SCRATCH: RefCell<I8Scratch> = const { RefCell::new(I8Scratch {
        xq: Vec::new(),
        padded: Vec::new(),
        wu8: Vec::new(),
        acc: Vec::new(),
        a_zp: Vec::new(),
        b_zp: Vec::new(),
        row_c: Vec::new(),
        row_y: Vec::new(),
        row_x: Vec::new(),
        col_y: Vec::new(),
        col_x: Vec::new(),
    }) };
    static PACKED_F32_B: RefCell<Vec<(usize, usize, usize, PackedBMatrix<f32>)>> =
        const { RefCell::new(Vec::new()) };
    static PACKED_F32_A: RefCell<Vec<(usize, usize, usize, PackedAMatrix<f32>)>> =
        const { RefCell::new(Vec::new()) };
    static PACKED_I8_B: RefCell<Vec<(usize, usize, usize, PackedBMatrix<i8>)>> =
        const { RefCell::new(Vec::new()) };
    static A_U8: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static KN_T: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

#[cfg(not(target_vendor = "apple"))]
const PACK_CAP: usize = 32;

/// `C[m,n] = A[m,k] @ B[k,n] + bias[m]`.
#[cfg(not(target_vendor = "apple"))]
pub fn gemm_rowbias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    if m == 0 || n == 0 || k == 0 {
        return false;
    }
    F32_EXEC.with(|exec| {
        PACKED_F32_A.with(|cell| {
            let mut cache = cell.borrow_mut();
            let key = a.as_ptr() as usize;
            if !cache
                .iter()
                .any(|(p, mm, kk, _)| *p == key && *mm == m && *kk == k)
                && cache.len() < PACK_CAP
            {
                let a_m = NdTensorView::from_data([m, k], a);
                cache.push((key, m, k, exec.prepack_a(a_m)));
            }
            let a_in = match cache
                .iter()
                .find(|(p, mm, kk, _)| *p == key && *mm == m && *kk == k)
            {
                Some((_, _, _, packed)) => GemmInputA::Packed(packed),
                None => GemmInputA::Unpacked(NdTensorView::from_data([m, k], a)),
            };
            exec.gemm(
                c,
                a_in,
                GemmInputB::Unpacked(NdTensorView::from_data([k, n], b)),
                GemmOptions {
                    alpha: 1.0,
                    beta: 0.0,
                    bias: Some(BiasVector::Column(bias)),
                    ..GemmOptions::default()
                },
            )
            .is_ok()
        })
    })
}

/// `C[m,n] = A[m,k] @ B[k,n] + bias[n]`.
#[cfg(not(target_vendor = "apple"))]
pub fn gemm_colbias(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    if m == 0 || n == 0 || k == 0 {
        return false;
    }
    let a_m = NdTensorView::from_data([m, k], a);
    F32_EXEC.with(|exec| {
        PACKED_F32_B.with(|cell| {
            let mut cache = cell.borrow_mut();
            let key = b.as_ptr() as usize;
            if !cache
                .iter()
                .any(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k)
                && cache.len() < PACK_CAP
            {
                let b_m = NdTensorView::from_data([k, n], b);
                cache.push((key, n, k, exec.prepack_b(b_m)));
            }
            let b_in = match cache
                .iter()
                .find(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k)
            {
                Some((_, _, _, packed)) => GemmInputB::Packed(packed),
                None => GemmInputB::Unpacked(NdTensorView::from_data([k, n], b)),
            };
            exec.gemm(
                c,
                GemmInputA::Unpacked(a_m),
                b_in,
                GemmOptions {
                    alpha: 1.0,
                    beta: 0.0,
                    bias: Some(BiasVector::Row(bias)),
                    ..GemmOptions::default()
                },
            )
            .is_ok()
        })
    })
}

/// `C[m,n] += A[m,k] @ B[k,n]`.
#[cfg(not(target_vendor = "apple"))]
pub fn gemm_add(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> bool {
    if m == 0 || n == 0 || k == 0 {
        return false;
    }
    let a_m = NdTensorView::from_data([m, k], a);
    F32_EXEC.with(|exec| {
        PACKED_F32_B.with(|cell| {
            let mut cache = cell.borrow_mut();
            let key = b.as_ptr() as usize;
            let hit = cache
                .iter()
                .position(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k);
            if hit.is_none() && cache.len() < PACK_CAP {
                let b_m = NdTensorView::from_data([k, n], b);
                cache.push((key, n, k, exec.prepack_b(b_m)));
            }
            let b_in = match cache
                .iter()
                .find(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k)
            {
                Some((_, _, _, packed)) => GemmInputB::Packed(packed),
                None => GemmInputB::Unpacked(NdTensorView::from_data([k, n], b)),
            };
            exec.gemm(
                c,
                GemmInputA::Unpacked(a_m),
                b_in,
                GemmOptions {
                    alpha: 1.0,
                    beta: 1.0,
                    bias: None,
                    ..GemmOptions::default()
                },
            )
            .is_ok()
        })
    })
}

/// `C = Q(A; a_scale, a_zp) @ (B - b_zp) * a_scale * b_scale [+ bias]`.
///
/// Activation scale is a **fixed per-op** value (not min/max of `A`).
/// `B` is INT8 `[k, n]` with per-column or scalar scale/zp (zp is 0 for
/// the shipping signed LSTM weights).
#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8_static(
    a: &[f32],
    a_scale: f32,
    a_zp: u8,
    b: &[i8],
    b_scale: &[f32],
    b_zp: &[i8],
    bias: Option<&[f32]>,
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    accumulate: bool,
) -> bool {
    if m == 0 || n == 0 || k == 0 || a_scale.abs() < 1e-12 {
        return false;
    }
    if a.len() != m.saturating_mul(k)
        || b.len() != k.saturating_mul(n)
        || c.len() != m.saturating_mul(n)
    {
        return false;
    }
    if let Some(bias) = bias
        && bias.len() != n
    {
        return false;
    }
    I8_EXEC.with(|exec| {
        I8_SCRATCH.with(|cell| {
            A_U8.with(|aq| {
                PACKED_I8_B.with(|cache| {
                    let mut scratch = cell.borrow_mut();
                    let mut xq = aq.borrow_mut();
                    let need_a = m.saturating_mul(k);
                    if xq.len() < need_a {
                        xq.resize(need_a, 0);
                    }
                    quant_u8_static(a, a_scale, a_zp, &mut xq[..need_a]);
                    let need = m.saturating_mul(n);
                    if scratch.acc.len() < need {
                        scratch.acc.resize(need, 0);
                    }
                    scratch.a_zp.resize(m, a_zp);
                    scratch.a_zp.fill(a_zp);
                    let bz0 = b_zp.first().copied().unwrap_or(0);
                    scratch.b_zp.resize(n, bz0);
                    if b_zp.len() == n {
                        scratch.b_zp.copy_from_slice(b_zp);
                    } else {
                        scratch.b_zp.fill(bz0);
                    }
                    let a_zp_local = scratch.a_zp.clone();
                    let b_zp_local = scratch.b_zp.clone();
                    let mut packed = cache.borrow_mut();
                    let key = b.as_ptr() as usize;
                    if !packed
                        .iter()
                        .any(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k)
                        && packed.len() < PACK_CAP
                    {
                        let b_m = NdTensorView::from_data([k, n], b);
                        packed.push((key, n, k, exec.prepack_b(b_m)));
                    }
                    let b_in = match packed
                        .iter()
                        .find(|(p, nn, kk, _)| *p == key && *nn == n && *kk == k)
                    {
                        Some((_, _, _, p)) => GemmInputB::Packed(p),
                        None => GemmInputB::Unpacked(NdTensorView::from_data([k, n], b)),
                    };
                    let a_q = QuantParams {
                        zero_point: a_zp_local.as_slice(),
                    };
                    let b_q = QuantParams {
                        zero_point: b_zp_local.as_slice(),
                    };
                    let acc = &mut scratch.acc[..need];
                    if exec
                        .gemm(
                            acc,
                            GemmInputA::Unpacked(NdTensorView::from_data([m, k], &xq[..need_a])),
                            b_in,
                            GemmOptions {
                                alpha: 1.0,
                                beta: 0,
                                bias: None,
                                a_quant: Some(a_q),
                                b_quant: Some(b_q),
                            },
                        )
                        .is_err()
                    {
                        return false;
                    }
                    dequant_acc(acc, a_scale, b_scale, bias, c, m, n, accumulate);
                    true
                })
            })
        })
    })
}

/// `C = ((A_u8 - a_zp) @ (B_[n,k]^T - b_zp)) * a_scale * b_scale + bias`.
///
/// `B` is qlinear `[n, k]`. Transposed to `[k, n]` and run Unpacked: rten's
/// prepacked B path does not apply per-column zp (LSTM weights are zp 0).
#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_u8i8_nk(
    a_u8: &[u8],
    a_zp: u8,
    b_nk: &[i8],
    b_scale: &[f32],
    b_zp: &[i8],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    a_scale: f32,
) -> bool {
    if m == 0 || n == 0 || k == 0 || a_scale.abs() < 1e-12 {
        return false;
    }
    if a_u8.len() != m.saturating_mul(k)
        || b_nk.len() != n.saturating_mul(k)
        || c.len() != m.saturating_mul(n)
        || bias.len() != n
    {
        return false;
    }
    I8_EXEC.with(|exec| {
        I8_SCRATCH.with(|cell| {
            KN_T.with(|kt| {
                let mut scratch = cell.borrow_mut();
                let mut kn = kt.borrow_mut();
                let nkk = k.saturating_mul(n);
                if kn.len() < nkk {
                    kn.resize(nkk, 0);
                }
                for ni in 0..n {
                    let src = &b_nk[ni * k..ni * k + k];
                    for ki in 0..k {
                        kn[ki * n + ni] = src[ki];
                    }
                }
                let need = m.saturating_mul(n);
                if scratch.acc.len() < need {
                    scratch.acc.resize(need, 0);
                }
                scratch.a_zp.resize(m, a_zp);
                scratch.a_zp.fill(a_zp);
                let bz0 = b_zp.first().copied().unwrap_or(0);
                scratch.b_zp.resize(n, bz0);
                if b_zp.len() == n {
                    scratch.b_zp.copy_from_slice(b_zp);
                } else {
                    scratch.b_zp.fill(bz0);
                }
                let a_zp_local = scratch.a_zp.clone();
                let b_zp_local = scratch.b_zp.clone();
                let a_q = QuantParams {
                    zero_point: a_zp_local.as_slice(),
                };
                let b_q = QuantParams {
                    zero_point: b_zp_local.as_slice(),
                };
                let acc = &mut scratch.acc[..need];
                if exec
                    .gemm(
                        acc,
                        GemmInputA::Unpacked(NdTensorView::from_data([m, k], a_u8)),
                        GemmInputB::Unpacked(NdTensorView::from_data([k, n], &kn[..nkk])),
                        GemmOptions {
                            alpha: 1.0,
                            beta: 0,
                            bias: None,
                            a_quant: Some(a_q),
                            b_quant: Some(b_q),
                        },
                    )
                    .is_err()
                {
                    return false;
                }
                dequant_acc(acc, a_scale, b_scale, Some(bias), c, m, n, false);
                true
            })
        })
    })
}

#[cfg(not(target_vendor = "apple"))]
fn dequant_acc(
    acc: &[i32],
    a_scale: f32,
    b_scale: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    m: usize,
    n: usize,
    accumulate: bool,
) {
    #[cfg(target_arch = "aarch64")]
    if n.is_multiple_of(4) && n > 0 {
        dequant_acc_neon(acc, a_scale, b_scale, bias, c, m, n, accumulate);
        return;
    }
    let s0 = b_scale.first().copied().unwrap_or(1.0);
    for mi in 0..m {
        let row = mi * n;
        for ni in 0..n {
            let sw = if b_scale.len() == n { b_scale[ni] } else { s0 };
            let v = acc[row + ni] as f32 * a_scale * sw;
            if accumulate {
                c[row + ni] += v;
            } else if let Some(bias) = bias {
                c[row + ni] = v + bias[ni];
            } else {
                c[row + ni] = v;
            }
        }
    }
}

#[cfg(all(not(target_vendor = "apple"), target_arch = "aarch64"))]
fn dequant_acc_neon(
    acc: &[i32],
    a_scale: f32,
    b_scale: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    m: usize,
    n: usize,
    accumulate: bool,
) {
    use std::arch::aarch64::{
        vaddq_f32, vcvtq_f32_s32, vdupq_n_f32, vld1q_f32, vld1q_s32, vmulq_f32, vst1q_f32,
    };
    let sa = unsafe { vdupq_n_f32(a_scale) };
    let s0 = b_scale.first().copied().unwrap_or(1.0);
    let per_col = b_scale.len() == n;
    let s0v = unsafe { vdupq_n_f32(s0) };
    for mi in 0..m {
        let row = mi * n;
        let mut ni = 0usize;
        while ni < n {
            unsafe {
                let mut v = vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc.as_ptr().add(row + ni))), sa);
                let sw = if per_col {
                    vld1q_f32(b_scale.as_ptr().add(ni))
                } else {
                    s0v
                };
                v = vmulq_f32(v, sw);
                if accumulate {
                    v = vaddq_f32(v, vld1q_f32(c.as_ptr().add(row + ni)));
                } else if let Some(b) = bias {
                    v = vaddq_f32(v, vld1q_f32(b.as_ptr().add(ni)));
                }
                vst1q_f32(c.as_mut_ptr().add(row + ni), v);
            }
            ni += 4;
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
fn quant_u8_static(src: &[f32], scale: f32, zp: u8, dst: &mut [u8]) {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    let n = src.len().min(dst.len());
    #[cfg(target_arch = "aarch64")]
    {
        quant_u8_neon(src, s, z, dst, n);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        let q = src[i] / s + z;
        dst[i] = q.round().clamp(0.0, 255.0) as u8;
    }
}

/// Same `vdiv` + `vrndaq` + clamp as the scalar loop (Rust `f32::round`).
#[cfg(all(not(target_vendor = "apple"), target_arch = "aarch64"))]
fn quant_u8_neon(src: &[f32], s: f32, z: f32, dst: &mut [u8], n: usize) {
    use std::arch::aarch64::{
        vaddq_f32, vcombine_u8, vcombine_u16, vcvtq_u32_f32, vdivq_f32, vld1q_f32, vmaxq_f32,
        vminq_f32, vmovq_n_f32, vqmovn_u16, vqmovn_u32, vrndaq_f32, vst1q_u8,
    };
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(0.0) };
    let hi = unsafe { vmovq_n_f32(255.0) };
    let mut i = 0usize;
    while i + 16 <= n {
        unsafe {
            let q0 = vcvtq_u32_f32(vmaxq_f32(
                vminq_f32(
                    vrndaq_f32(vaddq_f32(vdivq_f32(vld1q_f32(src.as_ptr().add(i)), vs), vz)),
                    hi,
                ),
                lo,
            ));
            let q1 = vcvtq_u32_f32(vmaxq_f32(
                vminq_f32(
                    vrndaq_f32(vaddq_f32(
                        vdivq_f32(vld1q_f32(src.as_ptr().add(i + 4)), vs),
                        vz,
                    )),
                    hi,
                ),
                lo,
            ));
            let q2 = vcvtq_u32_f32(vmaxq_f32(
                vminq_f32(
                    vrndaq_f32(vaddq_f32(
                        vdivq_f32(vld1q_f32(src.as_ptr().add(i + 8)), vs),
                        vz,
                    )),
                    hi,
                ),
                lo,
            ));
            let q3 = vcvtq_u32_f32(vmaxq_f32(
                vminq_f32(
                    vrndaq_f32(vaddq_f32(
                        vdivq_f32(vld1q_f32(src.as_ptr().add(i + 12)), vs),
                        vz,
                    )),
                    hi,
                ),
                lo,
            ));
            let a = vcombine_u16(vqmovn_u32(q0), vqmovn_u32(q1));
            let b = vcombine_u16(vqmovn_u32(q2), vqmovn_u32(q3));
            vst1q_u8(
                dst.as_mut_ptr().add(i),
                vcombine_u8(vqmovn_u16(a), vqmovn_u16(b)),
            );
        }
        i += 16;
    }
    for j in i..n {
        let q = src[j] / s + z;
        dst[j] = q.round().clamp(0.0, 255.0) as u8;
    }
}

/// Virtual-im2col INT8 conv. A is weights `[oc, k]` as u8 (zp 128); B is a
/// padded CHW activation image so rten packs im2col blocks instead of a full
/// column buffer. Pad pixels are `act_zp`.
#[cfg(not(target_vendor = "apple"))]
pub fn try_conv_i8(conv: &Conv2d, x: &Tensor, y: &mut Tensor, relu: bool) -> bool {
    if conv.q_w.is_empty() || conv.out_scale.is_empty() || x.n == 0 {
        return false;
    }
    let k_raw = conv.ic.saturating_mul(conv.k).saturating_mul(conv.k);
    if k_raw == 0 || conv.q_w.len() != conv.oc.saturating_mul(k_raw) {
        return false;
    }
    pin_parallelism();
    let oh = (x.h + 2 * conv.pad - conv.k) / conv.stride + 1;
    let ow = (x.w + 2 * conv.pad - conv.k) / conv.stride + 1;
    if y.h != oh || y.w != ow || y.c != conv.oc || y.n != x.n {
        return false;
    }
    let spatial = oh.saturating_mul(ow);
    let zp = conv.act_zp;
    let scale = conv.act_scale.unwrap_or(1.0);
    let pad = conv.pad;
    let hp = x.h + 2 * pad;
    let wp = x.w + 2 * pad;
    let plane = hp * wp;

    I8_EXEC.with(|exec| {
        let col_step = exec.im2col_col_count_step().max(1);
        let row_step = exec.im2col_row_count_step().max(1);
        let n_rows_pad = k_raw.next_multiple_of(row_step);
        let n_cols_pad = spatial.next_multiple_of(col_step);
        I8_SCRATCH.with(|cell| {
            run_i8_conv(
                conv,
                x,
                y,
                relu,
                k_raw,
                oh,
                ow,
                spatial,
                zp,
                scale,
                pad,
                hp,
                wp,
                plane,
                n_rows_pad,
                n_cols_pad,
                exec,
                &mut cell.borrow_mut(),
            )
        })
    })
}

#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::too_many_arguments)]
fn run_i8_conv(
    conv: &Conv2d,
    x: &Tensor,
    y: &mut Tensor,
    relu: bool,
    k_raw: usize,
    oh: usize,
    ow: usize,
    spatial: usize,
    zp: i8,
    scale: f32,
    pad: usize,
    hp: usize,
    wp: usize,
    plane: usize,
    n_rows_pad: usize,
    n_cols_pad: usize,
    exec: &GemmExecutor<u8, i8, i32>,
    scratch: &mut I8Scratch,
) -> bool {
    let I8Scratch {
        xq,
        padded,
        wu8,
        acc,
        a_zp,
        b_zp,
        row_c,
        row_y,
        row_x,
        col_y,
        col_x,
    } = scratch;
    if xq.len() < x.data.len() {
        xq.resize(x.data.len(), 0);
    }
    quantize_i8(&x.data, scale, zp, &mut xq[..x.data.len()]);
    if wu8.len() < conv.oc * k_raw {
        wu8.resize(conv.oc * k_raw, 0);
    }
    for (d, &w) in wu8.iter_mut().zip(conv.q_w.iter()) {
        *d = (i32::from(w) + 128) as u8;
    }
    a_zp.resize(conv.oc, 128);
    a_zp.fill(128);
    b_zp.resize(spatial, zp);
    b_zp.fill(zp);
    if acc.len() < conv.oc * spatial {
        acc.resize(conv.oc * spatial, 0);
    }
    if padded.len() < conv.ic * plane {
        padded.resize(conv.ic * plane, zp);
    }

    fill_im2col_offsets(
        conv.ic,
        conv.k,
        conv.stride,
        hp,
        wp,
        oh,
        ow,
        n_rows_pad,
        n_cols_pad,
        row_c,
        row_y,
        row_x,
        col_y,
        col_x,
    );

    let weights = NdTensorView::from_data([conv.oc, k_raw], &wu8[..conv.oc * k_raw]);
    let a_quant = QuantParams {
        zero_point: &a_zp[..conv.oc],
    };
    let b_quant = QuantParams {
        zero_point: &b_zp[..spatial],
    };

    for ni in 0..x.n {
        padded.fill(zp);
        copy_nchw_pad(
            &xq[ni * x.c * x.h * x.w..(ni + 1) * x.c * x.h * x.w],
            x.c,
            x.h,
            x.w,
            pad,
            hp,
            wp,
            &mut padded[..conv.ic * plane],
        );
        let image = NdTensorView::from_data([conv.ic, hp, wp], &padded[..conv.ic * plane]);
        let im2col = Im2Col {
            image,
            row_offsets: RowOffsets {
                chan: row_c.clone(),
                y: row_y.clone(),
                x: row_x.clone(),
            },
            col_offsets: ColOffsets {
                y: col_y.clone(),
                x: col_x.clone(),
            },
            n_cols: spatial,
            n_rows: k_raw,
            max_y_offset: ((hp - 1) * wp) as i32,
            max_x_offset: (wp - 1) as i32,
        };
        let out = &mut acc[..conv.oc * spatial];
        match exec.gemm(
            out,
            GemmInputA::Unpacked(weights),
            GemmInputB::Im2Col(&im2col),
            GemmOptions {
                alpha: 1.0,
                beta: 0,
                bias: None,
                a_quant: Some(a_quant),
                b_quant: Some(b_quant),
            },
        ) {
            Ok(()) => dequant_nchw(out, conv, y, ni, spatial, relu),
            Err(_) => return false,
        }
    }
    true
}

#[cfg(not(target_vendor = "apple"))]
fn quantize_i8(src: &[f32], scale: f32, zp: i8, dst: &mut [i8]) {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let q = (v / s).round() + z;
        *d = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::too_many_arguments)]
fn copy_nchw_pad(
    src: &[i8],
    c: usize,
    h: usize,
    w: usize,
    pad: usize,
    hp: usize,
    wp: usize,
    dst: &mut [i8],
) {
    for ch in 0..c {
        let splane = &src[ch * h * w..ch * h * w + h * w];
        let dplane = &mut dst[ch * hp * wp..ch * hp * wp + hp * wp];
        for y in 0..h {
            let drow = (y + pad) * wp + pad;
            dplane[drow..drow + w].copy_from_slice(&splane[y * w..y * w + w]);
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
#[allow(clippy::too_many_arguments)]
fn fill_im2col_offsets(
    ic: usize,
    k: usize,
    stride: usize,
    hp: usize,
    wp: usize,
    oh: usize,
    ow: usize,
    n_rows_pad: usize,
    n_cols_pad: usize,
    row_c: &mut Vec<i32>,
    row_y: &mut Vec<i32>,
    row_x: &mut Vec<i32>,
    col_y: &mut Vec<i32>,
    col_x: &mut Vec<i32>,
) {
    let chan_stride = (hp * wp) as i32;
    let h_stride = wp as i32;
    row_c.clear();
    row_y.clear();
    row_x.clear();
    for c in 0..ic {
        for ky in 0..k {
            for kx in 0..k {
                row_c.push(c as i32 * chan_stride);
                row_y.push(ky as i32 * h_stride);
                row_x.push(kx as i32);
            }
        }
    }
    while row_c.len() < n_rows_pad {
        row_c.push(i32::MAX);
        row_y.push(i32::MAX);
        row_x.push(i32::MAX);
    }
    col_y.clear();
    col_x.clear();
    for oy in 0..oh {
        let in_y = oy * stride;
        for ox in 0..ow {
            let in_x = ox * stride;
            col_y.push(in_y as i32 * h_stride);
            col_x.push(in_x as i32);
        }
    }
    while col_y.len() < n_cols_pad {
        col_y.push(i32::MAX);
        col_x.push(i32::MAX);
    }
}

#[cfg(not(target_vendor = "apple"))]
fn dequant_nchw(acc: &[i32], conv: &Conv2d, y: &mut Tensor, ni: usize, spatial: usize, relu: bool) {
    for o in 0..conv.oc {
        let s = conv.out_scale.get(o).copied().unwrap_or(1.0);
        let b = conv.bias.get(o).copied().unwrap_or(0.0);
        let src = o * spatial;
        let dst = ni * conv.oc * spatial + src;
        for hw in 0..spatial {
            let mut v = acc[src + hw] as f32 * s + b;
            if relu && v < 0.0 {
                v = 0.0;
            }
            y.data[dst + hw] = v;
        }
    }
}

#[cfg(all(test, not(target_vendor = "apple")))]
mod tests {
    use super::try_conv_i8;
    use crate::conv::Conv2d;
    use crate::tensor::Tensor;

    fn quantized_fixture(oc: usize, ic: usize, k: usize, h: usize, w: usize) -> (Conv2d, Tensor) {
        let k_raw = ic * k * k;
        let mut x = Tensor::zeros(1, ic, h, w);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * k_raw];
        let mut q_scale = vec![0.0f32; oc];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..k_raw {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                m = m.max(v.abs());
                q_w[o * k_raw + i] = 0;
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..k_raw {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                q_w[o * k_raw + i] = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let bias: Vec<f32> = (0..oc).map(|o| (o as f32) * 0.05 - 0.1).collect();
        let conv = Conv2d::quantized(oc, ic, k, 1, q_w, q_scale, bias).with_input_quant(0.04, -128);
        (conv, x)
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn implicit_fwd(conv: &Conv2d, x: &Tensor) -> Tensor {
        crate::conv_i8::force_i8(true);
        let got = conv.forward(x);
        crate::conv_i8::force_i8(false);
        got
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dequant_acc_neon_matches_scalar() {
        let m = 3usize;
        let n = 8usize;
        let acc: Vec<i32> = (0..m * n).map(|i| (i as i32) * 13 - 40).collect();
        let b_scale: Vec<f32> = (0..n).map(|i| 0.01 + (i as f32) * 0.001).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let a_scale = 0.07f32;
        let mut got = vec![1.5f32; m * n];
        super::dequant_acc_neon(&acc, a_scale, &b_scale, Some(&bias), &mut got, m, n, false);
        let mut want = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let v = acc[mi * n + ni] as f32 * a_scale * b_scale[ni];
                want[mi * n + ni] = v + bias[ni];
            }
        }
        assert_eq!(got, want);
        let mut g2 = vec![0.25f32; m * n];
        let mut w2 = g2.clone();
        super::dequant_acc_neon(&acc, a_scale, &b_scale, None, &mut g2, m, n, true);
        for mi in 0..m {
            for ni in 0..n {
                w2[mi * n + ni] += acc[mi * n + ni] as f32 * a_scale * b_scale[ni];
            }
        }
        assert_eq!(g2, w2);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn quant_u8_neon_matches_scalar() {
        let src: Vec<f32> = (0..67)
            .map(|i| (i as f32) * 0.37 - 8.0)
            .chain([-100.0, 100.0, 0.0, -0.02, 0.02])
            .collect();
        let n = src.len();
        let mut got = vec![0u8; n];
        let mut want = vec![0u8; n];
        let s = 0.11f32;
        let z = 128u8;
        super::quant_u8_static(&src, s, z, &mut got);
        for i in 0..n {
            let q = src[i] / s + f32::from(z);
            want[i] = q.round().clamp(0.0, 255.0) as u8;
        }
        assert_eq!(got, want);
    }

    #[test]
    fn virtual_im2col_tracks_implicit_3x3() {
        let (conv, x) = quantized_fixture(8, 16, 3, 6, 10);
        let want = implicit_fwd(&conv, &x);
        let mut got = Tensor::zeros(x.n, conv.oc, x.h, x.w);
        assert!(try_conv_i8(&conv, &x, &mut got, false));
        let d = maxabs(&got.data, &want.data);
        assert!(d < 0.05, "3x3 rten vs implicit maxabs={d}");
    }

    #[test]
    fn virtual_im2col_tracks_implicit_1x1() {
        let (conv, x) = quantized_fixture(8, 16, 1, 5, 9);
        let want = implicit_fwd(&conv, &x);
        let mut got = Tensor::zeros(x.n, conv.oc, x.h, x.w);
        assert!(try_conv_i8(&conv, &x, &mut got, false));
        let d = maxabs(&got.data, &want.data);
        assert!(d < 0.05, "1x1 rten vs implicit maxabs={d}");
    }
}
