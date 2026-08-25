//! Implicit INT8 GEMM convolution — no full im2col.
//!
//! Matches ONNX QDQ: `y[oc] = act_scale * w_scale[oc] * Σ (w_i8 - w_zp)(x_i8 - x_zp) + bias`.
//! Activations are quantized from the incoming f32 map; 3×3 / 1×1 patches are
//! packed a few output pixels at a time so the working set stays in L1.
//! On Apple Silicon the inner product is `sdot` (not the unstable `vdotq_s32`).

use crate::conv::Conv2d;
use crate::tensor::Tensor;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};

static INTRA_THREADS: AtomicUsize = AtomicUsize::new(1);

/// Intra-op workers for INT8 3x3 s1. Embedder sets this to ncpu when it
/// runs a single ResNet so we share one activation, like MLAS.
pub fn set_intra_threads(n: usize) {
    INTRA_THREADS.store(n.max(1), Ordering::Relaxed);
}

fn intra_threads() -> usize {
    std::env::var("POLYVOICE_CONV_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| INTRA_THREADS.load(Ordering::Relaxed))
        .max(1)
}

#[cfg(test)]
#[test]
fn quantize_matches_scalar_round_clip() {
    let src: Vec<f32> = (0..64)
        .map(|i| (i as f32) * 0.13 - 3.1)
        .chain([-100.0, 100.0, 0.0, -0.02, 0.02])
        .collect();
    let mut got = vec![0i8; src.len()];
    quantize(&src, 0.04, -128, &mut got);
    for (i, &v) in src.iter().enumerate() {
        let q = (v / 0.04).round() + -128.0;
        let want = q.clamp(-128.0, 127.0) as i8;
        assert_eq!(got[i], want, "i={i} v={v} got={} want={want}", got[i]);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn sdot_signed_ones() {
    if !has_dotprod() {
        return;
    }
    let a = [1i8; 16];
    let b = [1i8; 16];
    let got = unsafe { dot_i8_sdot(&a, &b) };
    assert_eq!(got, 16, "sdot 1·1");
    let a = [-1i8; 16];
    let got = unsafe { dot_i8_sdot(&a, &b) };
    assert_eq!(got, -16, "sdot (-1)·1");
}

#[cfg(test)]
thread_local! {
    static FORCE_I8: Cell<bool> = const { Cell::new(false) };
}

/// Test hook: run the integer kernel even when `POLYVOICE_I8_CONV` is unset.
#[cfg(test)]
pub fn force_i8(on: bool) {
    FORCE_I8.with(|c| c.set(on));
}

#[cfg(test)]
fn i8_forced() -> bool {
    FORCE_I8.with(Cell::get)
}

#[cfg(not(test))]
fn i8_forced() -> bool {
    false
}

const NR: usize = 8;
const NR16: usize = 16;
const MR: usize = 4;

/// Write this conv's relu(f32) as the *next* layer's QDQ i8 (scale folded
/// into a multiply, not a div in the K-loop).
#[derive(Clone, Copy)]
struct I8Dest {
    p: usize,
    inv_scale: f32,
    zp: f32,
}

impl I8Dest {
    #[inline(always)]
    fn store(self, idx: usize, v: f32) {
        let q = (v * self.inv_scale).round() + self.zp;
        // SAFETY: try_conv_to_i8 sizes the dest to N·OC·OH·OW; idx is the
        // same NCHW address the f32 store would have used.
        unsafe {
            *(self.p as *mut i8).add(idx) = q.clamp(-128.0, 127.0) as i8;
        }
    }
}
/// Output pixels in one implicit panel. Fat enough for GEMM, small enough
/// that the pack stays in L1 (~ k_pad × PN bytes).
const PN: usize = 32;
/// Pixels in one s1/s2 zip WAVE (`WAVE=32` × `PN`). Serial scans overwrite
/// the slab each wave, so TLS need not cover a long row.
const ZIP_WAVE_PX: usize = 32 * PN;

thread_local! {
    static XQ: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
    static XQ_SEED: Cell<Option<(u32, i8, usize)>> = const { Cell::new(None) };
    static PANEL_KN: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
    static PANEL_NK: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
    static ROW3: RefCell<Vec<i8>> = const { RefCell::new(Vec::new()) };
}

fn take_xq_seed(scale: f32, zp: i8, len: usize) -> bool {
    XQ_SEED.with(|c| {
        let hit = c.get() == Some((scale.to_bits(), zp, len));
        c.set(None);
        hit
    })
}

/// `a = relu(a)` and leave TLS XQ ready for the next conv at `scale`.
pub(crate) fn seed_xq_relu(a: &mut Tensor, scale: f32, zp: i8) {
    XQ.with(|cell| {
        let mut xq = cell.borrow_mut();
        let n = a.data.len();
        if xq.len() < n {
            xq.resize(n, 0);
        }
        crate::tensor::relu_quantize_inplace(a, scale, zp, &mut xq[..n]);
        XQ_SEED.with(|c| c.set(Some((scale.to_bits(), zp, n))));
    });
}

/// `a = relu(a+b)` and leave TLS XQ ready for the next conv at `scale`.
pub(crate) fn seed_xq_add_relu(a: &mut Tensor, b: &Tensor, scale: f32, zp: i8) {
    XQ.with(|cell| {
        let mut xq = cell.borrow_mut();
        let n = a.data.len();
        if xq.len() < n {
            xq.resize(n, 0);
        }
        crate::tensor::add_relu_quantize_inplace(a, b, scale, zp, &mut xq[..n]);
        XQ_SEED.with(|c| c.set(Some((scale.to_bits(), zp, n))));
    });
}

pub(crate) fn i8_conv_on() -> bool {
    i8_forced() || {
        static USE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *USE.get_or_init(|| {
            if std::env::var_os("POLYVOICE_NO_I8_CONV").is_some() {
                return false;
            }
            if std::env::var_os("POLYVOICE_I8_CONV").is_some() {
                return true;
            }
            cfg!(all(target_os = "linux", target_arch = "aarch64"))
        })
    }
}

/// True if this layer ran the integer kernel (caller must not also run BNNS).
pub fn try_conv(conv: &Conv2d, x: &Tensor, y: &mut Tensor, relu: bool) -> bool {
    if conv.q_w_pad.is_empty() || conv.out_scale.is_empty() || x.n == 0 {
        return false;
    }
    // Apple: BNNS Winograd is faster; keep integer GEMM opt-in.
    // Linux aarch64: no BNNS, OpenBLAS FP32 is ~10 GFLOP/s in the Desktop
    // VM — SDOT integer GEMM is the faster default. POLYVOICE_NO_I8_CONV=1
    // forces the float path; POLYVOICE_I8_CONV=1 forces integer everywhere.
    if !i8_conv_on() {
        return false;
    }
    // Full-map rten im2col is slower than the in-crate 4×16 SDOT kernel
    // (ResNet T=400: ~520 ms vs ~230 ms). Opt in with POLYVOICE_RTEN_CONV=1.
    #[cfg(not(target_vendor = "apple"))]
    if std::env::var_os("POLYVOICE_RTEN_CONV").is_some()
        && crate::rten_matmul::try_conv_i8(conv, x, y, relu)
    {
        return true;
    }
    let xq_len = x.data.len();
    let scale = conv.act_scale.unwrap_or(1.0);
    let zp = conv.act_zp;
    XQ.with(|cell| {
        let mut xq = cell.borrow_mut();
        if xq.len() < xq_len {
            xq.resize(xq_len, 0);
        }
        if !take_xq_seed(scale, zp, xq_len) {
            quantize(&x.data, scale, zp, &mut xq[..xq_len]);
        }
        run_from_xq(conv, x.n, x.h, x.w, &xq[..xq_len], y, relu, None)
    })
}

/// Like [`try_conv`], but the output is the next layer's quantized input.
pub fn try_conv_to_i8(
    conv: &Conv2d,
    x: &Tensor,
    yq: &mut [i8],
    relu: bool,
    next_scale: f32,
    next_zp: i8,
) -> bool {
    if conv.q_w_pad.is_empty() || conv.out_scale.is_empty() || x.n == 0 {
        return false;
    }
    if !i8_conv_on() || next_scale.abs() < 1e-12 {
        return false;
    }
    if conv.k != 3 || conv.pad != 1 || conv.stride > 1 {
        return false;
    }
    let (oh, ow) = conv.out_hw_dims(x.h, x.w);
    let need =
        x.n.saturating_mul(conv.oc)
            .saturating_mul(oh)
            .saturating_mul(ow);
    if yq.len() < need {
        return false;
    }
    let dest = I8Dest {
        p: yq.as_mut_ptr() as usize,
        inv_scale: 1.0 / next_scale,
        zp: f32::from(next_zp),
    };
    let xq_len = x.data.len();
    let scale = conv.act_scale.unwrap_or(1.0);
    let zp = conv.act_zp;
    XQ.with(|cell| {
        let mut xq = cell.borrow_mut();
        if xq.len() < xq_len {
            xq.resize(xq_len, 0);
        }
        if !take_xq_seed(scale, zp, xq_len) {
            quantize(&x.data, scale, zp, &mut xq[..xq_len]);
        }
        // i8 dest is written directly; keep shape only (no f32 map).
        let mut dummy = Tensor {
            n: x.n,
            c: conv.oc,
            h: oh,
            w: ow,
            data: Vec::new(),
        };
        run_from_xq(
            conv,
            x.n,
            x.h,
            x.w,
            &xq[..xq_len],
            &mut dummy,
            relu,
            Some(dest),
        )
    })
}

/// Integer conv from an already-quantized NCHW map (skips the f32 quantize).
pub fn try_from_i8(
    conv: &Conv2d,
    xq: &[i8],
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    relu: bool,
) -> bool {
    if conv.q_w_pad.is_empty() || conv.out_scale.is_empty() || n == 0 {
        return false;
    }
    if !i8_conv_on() {
        return false;
    }
    if xq.len()
        != n.saturating_mul(conv.ic)
            .saturating_mul(h)
            .saturating_mul(w)
    {
        return false;
    }
    run_from_xq(conv, n, h, w, xq, y, relu, None)
}

fn run_from_xq(
    conv: &Conv2d,
    n: usize,
    ih: usize,
    iw: usize,
    xq: &[i8],
    y: &mut Tensor,
    relu: bool,
    i8d: Option<I8Dest>,
) -> bool {
    let k_raw = conv.ic.saturating_mul(conv.k).saturating_mul(conv.k);
    if k_raw == 0 || conv.k_pad < k_raw {
        return false;
    }
    let kn_len = conv.k_pad.saturating_mul(PN);
    PANEL_KN.with(|knc| {
        PANEL_NK.with(|nkc| {
            let mut kn = knc.borrow_mut();
            let mut nk = nkc.borrow_mut();
            if kn.len() < kn_len {
                kn.resize(kn_len, 0);
            }
            if nk.len() < kn_len {
                nk.resize(kn_len, 0);
            }
            if conv.k == 3 && conv.pad == 1 && conv.stride <= 1 {
                // TLS is taken inside conv3x3_s1_rows so the parked pool
                // can reuse the same slots on the caller thread.
                drop(kn);
                drop(nk);
                conv3x3_s1_rows(conv, n, ih, iw, y, xq, relu, i8d);
            } else if conv.k == 3 && conv.pad == 1 && conv.stride == 2 {
                drop(kn);
                drop(nk);
                conv3x3_s2_rows(conv, n, ih, iw, y, xq, relu, i8d);
            } else if conv.k == 3 && conv.pad == 1 {
                conv3x3(conv, n, ih, iw, y, xq, &mut kn, &mut nk, relu);
            } else if conv.k == 1 && conv.pad == 0 {
                let need = conv.k_pad.saturating_mul(y.w.max(PN));
                if nk.len() < need {
                    nk.resize(need, 0);
                }
                conv1x1(conv, n, ih, iw, y, xq, &mut kn, &mut nk, relu);
            } else {
                conv_gather(conv, n, ih, iw, y, xq, &mut nk[..conv.k_pad], relu);
            }
        });
    });
    true
}

fn quantize(src: &[f32], scale: f32, zp: i8, dst: &mut [i8]) {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    #[cfg(target_arch = "aarch64")]
    quantize_neon(src, s, z, dst);
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let q = (v / s).round() + z;
        *d = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn quantize_neon(src: &[f32], s: f32, z: f32, dst: &mut [i8]) {
    use std::arch::aarch64::{vgetq_lane_s32, vmovq_n_f32};
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(-128.0) };
    let hi = unsafe { vmovq_n_f32(127.0) };
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 16 <= n {
        unsafe {
            let q0 = quant4(src.as_ptr().add(i), vs, vz, lo, hi);
            let q1 = quant4(src.as_ptr().add(i + 4), vs, vz, lo, hi);
            let q2 = quant4(src.as_ptr().add(i + 8), vs, vz, lo, hi);
            let q3 = quant4(src.as_ptr().add(i + 12), vs, vz, lo, hi);
            store16_i8(dst.as_mut_ptr().add(i), q0, q1, q2, q3);
        }
        i += 16;
    }
    while i + 4 <= n {
        unsafe {
            let qi = quant4(src.as_ptr().add(i), vs, vz, lo, hi);
            dst[i] = vgetq_lane_s32::<0>(qi) as i8;
            dst[i + 1] = vgetq_lane_s32::<1>(qi) as i8;
            dst[i + 2] = vgetq_lane_s32::<2>(qi) as i8;
            dst[i + 3] = vgetq_lane_s32::<3>(qi) as i8;
        }
        i += 4;
    }
    for j in i..n {
        let q = (src[j] / s).round() + z;
        dst[j] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn quant4(
    src: *const f32,
    vs: std::arch::aarch64::float32x4_t,
    vz: std::arch::aarch64::float32x4_t,
    lo: std::arch::aarch64::float32x4_t,
    hi: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::{
        vaddq_f32, vcvtq_s32_f32, vdivq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vrndaq_f32,
    };
    unsafe {
        let x = vld1q_f32(src);
        vcvtq_s32_f32(vmaxq_f32(
            vminq_f32(vaddq_f32(vrndaq_f32(vdivq_f32(x, vs)), vz), hi),
            lo,
        ))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store16_i8(
    dst: *mut i8,
    q0: std::arch::aarch64::int32x4_t,
    q1: std::arch::aarch64::int32x4_t,
    q2: std::arch::aarch64::int32x4_t,
    q3: std::arch::aarch64::int32x4_t,
) {
    use std::arch::aarch64::{vcombine_s8, vcombine_s16, vqmovn_s16, vqmovn_s32, vst1q_s8};
    unsafe {
        let a = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        let b = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(dst, vcombine_s8(vqmovn_s16(a), vqmovn_s16(b)));
    }
}

/// 3×3 s1: keep three padded input rows packed as `[3][ic][w+2]` so the
/// channel stride is `w+2` instead of `h*w`.
///
/// Intra-op: late layers (oc ≥ 64) split *output channels* so each worker
/// keeps a small weight working set. Early 32-ch maps keep the existing
/// output-row split (tiny weights, huge spatial). Never spawn inside a tile.
#[allow(clippy::too_many_arguments)]
fn conv3x3_s1_rows(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    relu: bool,
    i8d: Option<I8Dest>,
) {
    let (ic, oc, zp) = (conv.ic, conv.oc, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    let oh = y.h;
    let ow = y.w;
    let kn_len = conv.k_pad.saturating_mul(PN);
    // Serial/oy-split: one WAVE slab (overwritten). Shared-OC still zips the
    // whole row so workers can stream W without re-gather.
    // +NR16: shared-OC leftover 1..15 is one overlapping 4x16 (same as serial).
    let zip_row = conv.k_pad.saturating_mul(ow.max(PN).saturating_add(NR16));
    let zip_len = conv.k_pad.saturating_mul(ow.max(PN).min(ZIP_WAVE_PX));
    let threads = intra_threads();
    // MAC estimate: skip the pool when spawn-equivalent work is tiny (stem).
    let macs = (oc as u64) * (ic as u64) * 9 * (oh as u64) * (ow as u64);
    if threads > 1 && macs >= 20_000_000 {
        let y_ptr = y.data.as_mut_ptr() as usize;
        let y_len = y.data.len();
        // Wide OC: zip the row once, then split only the SDOT stream so
        // workers do not re-gather. Narrow OC / fat maps: split output rows.
        let split_oc = oc >= 128 && oc / MR >= 2;
        let tcount = if split_oc {
            threads.min(oc / MR).max(1)
        } else {
            threads.min(oh.max(1))
        };
        if tcount > 1 && split_oc {
            ROW3.with(|rc| {
                PANEL_KN.with(|knc| {
                    PANEL_NK.with(|nkc| {
                        let mut rows = rc.borrow_mut();
                        let mut kn = knc.borrow_mut();
                        let mut nk = nkc.borrow_mut();
                        if kn.len() < kn_len {
                            kn.resize(kn_len, 0);
                        }
                        let map_len = zip_row.saturating_mul(oh.max(1));
                        if nk.len() < map_len {
                            nk.resize(map_len, 0);
                        }
                        if rows.len() < 3 * plane {
                            rows.resize(3 * plane, zp);
                        }
                        for ni in 0..n {
                            let xbase = ni * ic * h * w;
                            let ybase = ni * oc * oh * ow;
                            let ximg = &xq[xbase..xbase + ic * h * w];
                            s1_scan(
                                conv,
                                ximg,
                                h,
                                w,
                                &mut y.data,
                                ybase,
                                oh,
                                ow,
                                &mut rows,
                                &mut kn,
                                &mut nk,
                                relu,
                                i8d,
                                0,
                                oc,
                                0,
                                oh,
                                tcount,
                            );
                        }
                    });
                });
            });
            return;
        }
        if tcount > 1 {
            for ni in 0..n {
                let xbase = ni * ic * h * w;
                let ybase = ni * oc * oh * ow;
                let ximg = &xq[xbase..xbase + ic * h * w];
                crate::intra::run(tcount, |t| {
                    let chunk = oh.div_ceil(tcount);
                    let oy0 = t * chunk;
                    let oy1 = (oy0 + chunk).min(oh);
                    if oy0 >= oy1 {
                        return;
                    }
                    // SAFETY: oy-split writes disjoint row ranges. The pool
                    // joins before return.
                    let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                    ROW3.with(|rc| {
                        PANEL_KN.with(|knc| {
                            PANEL_NK.with(|nkc| {
                                let mut rows = rc.borrow_mut();
                                let mut kn = knc.borrow_mut();
                                let mut nk = nkc.borrow_mut();
                                if kn.len() < kn_len {
                                    kn.resize(kn_len, 0);
                                }
                                if nk.len() < zip_len {
                                    nk.resize(zip_len, 0);
                                }
                                if rows.len() < 3 * plane {
                                    rows.resize(3 * plane, zp);
                                }
                                s1_scan(
                                    conv, ximg, h, w, yd, ybase, oh, ow, &mut rows, &mut kn,
                                    &mut nk, relu, i8d, 0, oc, oy0, oy1, 0,
                                );
                            });
                        });
                    });
                });
            }
            return;
        }
    }
    ROW3.with(|rc| {
        PANEL_KN.with(|knc| {
            PANEL_NK.with(|nkc| {
                let mut rows = rc.borrow_mut();
                let mut kn = knc.borrow_mut();
                let mut nk = nkc.borrow_mut();
                if kn.len() < kn_len {
                    kn.resize(kn_len, 0);
                }
                if nk.len() < zip_len {
                    nk.resize(zip_len, 0);
                }
                if rows.len() < 3 * plane {
                    rows.resize(3 * plane, zp);
                }
                for ni in 0..n {
                    let xbase = ni * ic * h * w;
                    let ybase = ni * oc * oh * ow;
                    let ximg = &xq[xbase..xbase + ic * h * w];
                    s1_scan(
                        conv,
                        ximg,
                        h,
                        w,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        &mut rows,
                        &mut kn,
                        &mut nk,
                        relu,
                        i8d,
                        0,
                        oc,
                        0,
                        oh,
                        0,
                    );
                }
            });
        });
    });
}

fn oc_bounds(oc: usize, t: usize, n: usize) -> (usize, usize) {
    let groups = oc / MR;
    let base = groups / n;
    let extra = groups % n;
    let g0 = t * base + t.min(extra);
    let g1 = g0 + base + if t < extra { 1 } else { 0 };
    let lo = g0 * MR;
    let hi = if t + 1 == n { oc } else { g1 * MR };
    (lo, hi)
}

#[allow(clippy::too_many_arguments)]
fn s1_scan(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    rows: &mut [i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
    oy0: usize,
    oy1: usize,
    oc_par: usize,
) {
    let (ic, zp) = (conv.ic, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    let k_raw = ic * 9;
    for kh in 0..2 {
        let iy = oy0.wrapping_add(kh).wrapping_sub(1);
        let slot = (oy0 + kh) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            iy,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
    }
    #[cfg(target_arch = "aarch64")]
    let row_zip = has_dotprod() && conv.k_pad.is_multiple_of(4);
    #[cfg(not(target_arch = "aarch64"))]
    let row_zip = false;
    if oc_par > 1 && row_zip {
        #[cfg(target_arch = "aarch64")]
        {
            s1_scan_shared_oc(
                conv, ximg, h, w, yd, ybase, oh, ow, rows, kn, nk, relu, i8d, oy0, oy1, oc_par,
                plane, wp, zp,
            );
            return;
        }
    }
    for oy in oy0..oy1 {
        let slot = (oy + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            oy + 1,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
        // A pn-wide tile at `ox` reads padded x in [ox, ox+pn+1]. The row
        // is `w+2` wide, so ox+pn <= w. For 3x3 s1 pad=1, ow == w.
        if row_zip {
            #[cfg(target_arch = "aarch64")]
            s1_scan_row_zip(
                conv, rows, plane, wp, yd, ybase, oh, ow, oy, kn, nk, relu, i8d, oc0, oc1,
            );
        } else {
            let mut ox = 0usize;
            while ox + PN <= ow {
                implicit_tile_from_rows(
                    conv, rows, plane, wp, yd, ybase, oh, ow, oy, ox, PN, kn, nk, relu, i8d, oc0,
                    oc1,
                );
                ox += PN;
            }
            while ox + NR16 <= ow {
                implicit_tile_from_rows(
                    conv, rows, plane, wp, yd, ybase, oh, ow, oy, ox, NR16, kn, nk, relu, i8d, oc0,
                    oc1,
                );
                ox += NR16;
            }
            while ox < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, ox, nk);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, nk, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
        }
    }
}

/// Gather every in-row tile, zip once, then stream OC across tiles so the
/// 4-row weight panel stays hot. Bound is `ox+pn <= ow` (right halo fits).
/// Zip every output row once, then one intra-op OC stream. Per-row pool
/// wakeups cost more than a T=400 layer-3 row; one dispatch amortizes.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn s1_scan_shared_oc(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    rows: &mut [i8],
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oy0: usize,
    oy1: usize,
    oc_par: usize,
    plane: usize,
    wp: usize,
    zp: i8,
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let row_bytes = k_pad.saturating_mul(ow.max(PN).saturating_add(NR16));
    let nrows = oy1.saturating_sub(oy0);
    let need = row_bytes.saturating_mul(nrows.max(1));
    if zip.len() < need {
        // Caller should have reserved the map; fall back to serial waves.
        for oy in oy0..oy1 {
            let slot = (oy + 2) % 3;
            pack_pad_row(
                ximg,
                ic,
                h,
                w,
                wp,
                zp,
                oy + 1,
                &mut rows[slot * plane..(slot + 1) * plane],
            );
            s1_scan_row_zip(
                conv, rows, plane, wp, yd, ybase, oh, ow, oy, kn, zip, relu, i8d, 0, conv.oc,
            );
        }
        return;
    }
    const MAX_TILES: usize = 256;
    let mut oxs = [0usize; MAX_TILES];
    let mut pns = [0usize; MAX_TILES];
    let mut zoff = [0usize; MAX_TILES];
    let mut ntiles = 0usize;
    let mut acc = 0usize;
    let mut ox = 0usize;
    while ox < ow && ntiles < MAX_TILES {
        let pn = if ox + PN <= ow {
            PN
        } else if ox + NR16 <= ow {
            NR16
        } else {
            break;
        };
        oxs[ntiles] = ox;
        pns[ntiles] = pn;
        zoff[ntiles] = acc;
        acc += k_pad * pn;
        ntiles += 1;
        ox += pn;
    }
    // Leftover 1..15: one overlapping 4x16 so intra matches serial waves.
    if ow >= NR16 && ox < ow && ntiles < MAX_TILES {
        let ox0 = ow - NR16;
        oxs[ntiles] = ox0;
        pns[ntiles] = NR16;
        zoff[ntiles] = acc;
        acc += k_pad * NR16;
        ntiles += 1;
        ox = ow;
    }
    let tail_ox = ox;
    if ntiles == 0 {
        for oy in oy0..oy1 {
            let slot = (oy + 2) % 3;
            pack_pad_row(
                ximg,
                ic,
                h,
                w,
                wp,
                zp,
                oy + 1,
                &mut rows[slot * plane..(slot + 1) * plane],
            );
            let mut x = tail_ox;
            while x < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, x, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, x, kn, k_raw, relu, i8d, 0, conv.oc,
                );
                x += 1;
            }
        }
        return;
    }
    for kh in 0..2 {
        let iy = oy0.wrapping_add(kh).wrapping_sub(1);
        let slot = (oy0 + kh) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            iy,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
    }
    for oy in oy0..oy1 {
        let slot = (oy + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            oy + 1,
            &mut rows[slot * plane..(slot + 1) * plane],
        );
        let dest = &mut zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
        for t in 0..ntiles {
            let pn = pns[t];
            gather_kn_from_rows(conv, rows, plane, wp, oy, oxs[t], pn, kn);
            pack_kn_sdot16(kn, &mut dest[zoff[t]..zoff[t] + k_pad * pn], k_pad, pn);
        }
        let mut x = tail_ox;
        while x < ow {
            gather_one_from_rows(conv, rows, plane, wp, oy, x, kn);
            store_col(
                conv, yd, ybase, oh, ow, oy, x, kn, k_raw, relu, i8d, 0, conv.oc,
            );
            x += 1;
        }
    }
    let y_ptr = yd.as_mut_ptr() as usize;
    let y_len = yd.len();
    let zip_ptr = zip.as_ptr() as usize;
    crate::intra::run(oc_par, |t| {
        let (a, b) = oc_bounds(conv.oc, t, oc_par);
        if a >= b {
            return;
        }
        let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
        let zip = unsafe { std::slice::from_raw_parts(zip_ptr as *const i8, need) };
        for oy in oy0..oy1 {
            let rowz = &zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
            let mut mo = a;
            while mo < b {
                let mr = (conv.oc - mo).min(MR);
                if mr != MR {
                    mo += mr;
                    continue;
                }
                for ti in 0..ntiles {
                    let pn = pns[ti];
                    let z = &rowz[zoff[ti]..zoff[ti] + k_pad * pn];
                    let mut no = 0usize;
                    while no + NR16 <= pn {
                        unsafe {
                            kernel_4x16_zip_store(
                                conv,
                                yd,
                                ybase,
                                oh,
                                ow,
                                oy,
                                oxs[ti] + no,
                                mo,
                                z,
                                pn,
                                no,
                                relu,
                                i8d,
                            );
                        }
                        no += NR16;
                    }
                }
                mo += mr;
            }
        }
    });
    // Leftover OC groups (oc not multiple of 4) stay serial.
    let rem = conv.oc % MR;
    if rem != 0 {
        let mo0 = conv.oc - rem;
        for oy in oy0..oy1 {
            let rowz = &zip[(oy - oy0) * row_bytes..(oy - oy0) * row_bytes + acc];
            for ti in 0..ntiles {
                let pn = pns[ti];
                let ox0 = oxs[ti];
                gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                for mi in 0..rem {
                    let wr = &conv.q_w_pad[(mo0 + mi) * k_pad..(mo0 + mi + 1) * k_pad];
                    for tcol in 0..pn {
                        let mut accu = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            accu += i32::from(wv) * i32::from(kn[kk * pn + tcol]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            ox0 + tcol,
                            mo0 + mi,
                            accu,
                            relu,
                            i8d,
                        );
                    }
                }
                let _ = rowz;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn s1_scan_row_zip(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let k_pad = conv.k_pad;
    let k_raw = conv.ic * 9;
    // Waves of tiles so a long row (T=2000/4100) stays on SDOT; 32 tiles
    // is 1024 px and keeps the zip panel around one L2-sized slab.
    const WAVE: usize = 32;
    let mut oxs = [0usize; WAVE];
    let mut pns = [0usize; WAVE];
    let mut zoff = [0usize; WAVE];
    let mut ox = 0usize;
    while ox < ow {
        let mut ntiles = 0usize;
        let mut acc = 0usize;
        while ox < ow && ntiles < WAVE {
            let pn = if ox + PN <= ow {
                PN
            } else if ox + NR16 <= ow {
                NR16
            } else {
                break;
            };
            oxs[ntiles] = ox;
            pns[ntiles] = pn;
            zoff[ntiles] = acc;
            acc += k_pad * pn;
            ntiles += 1;
            ox += pn;
        }
        if ntiles == 0 {
            // Leftover 1..15: one overlapping 4x16 tile (reuses zip[0..16*k_pad]).
            // 8..15 already kept; this extends to the 4-wide l3 tail (ow=100).
            if ow >= NR16 && ox < ow && zip.len() >= k_pad * NR16 {
                let ox0 = ow - NR16;
                let pn = NR16;
                gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                pack_kn_sdot16(kn, &mut zip[..k_pad * pn], k_pad, pn);
                let z = &zip[..k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
                                    let mut a = 0i32;
                                    for (kk, &wv) in wr.iter().enumerate() {
                                        a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                                    }
                                    write_out_sl(
                                        conv,
                                        yd,
                                        ybase,
                                        oh,
                                        ow,
                                        oy,
                                        ox0 + ti,
                                        mo + mi,
                                        a,
                                        relu,
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                    mo += mr;
                }
                ox = ow;
            }
            while ox < ow {
                gather_one_from_rows(conv, rows, plane, wp, oy, ox, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
            break;
        }
        debug_assert!(zip.len() >= acc);
        // Fat K (layer 3/4): gather+pack+OC one tile so the 36–70 KiB A slab
        // stays in L1. Packing the whole WAVE first evicts it (WAVE=32 of
        // k_pad=1152 is ~1 MiB). WAVE stays 32; this is not fat-K WAVE=1.
        // Narrow K still zips the wave, then streams W-hot OC-outer.
        let tile_outer = k_pad >= 1024;
        if tile_outer {
            for t in 0..ntiles {
                let pn = pns[t];
                gather_kn_from_rows(conv, rows, plane, wp, oy, oxs[t], pn, kn);
                pack_kn_sdot16(kn, &mut zip[zoff[t]..zoff[t] + k_pad * pn], k_pad, pn);
                let ox0 = oxs[t];
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
                                    let mut a = 0i32;
                                    for (kk, &wv) in wr.iter().enumerate() {
                                        a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                                    }
                                    write_out_sl(
                                        conv,
                                        yd,
                                        ybase,
                                        oh,
                                        ow,
                                        oy,
                                        ox0 + ti,
                                        mo + mi,
                                        a,
                                        relu,
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                    mo += mr;
                }
            }
        } else {
            for t in 0..ntiles {
                let pn = pns[t];
                gather_kn_from_rows(conv, rows, plane, wp, oy, oxs[t], pn, kn);
                pack_kn_sdot16(kn, &mut zip[zoff[t]..zoff[t] + k_pad * pn], k_pad, pn);
            }
            let mut mo = oc0;
            while mo < oc1 {
                let mr = (conv.oc - mo).min(MR);
                for t in 0..ntiles {
                    let pn = pns[t];
                    let ox0 = oxs[t];
                    let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_from_rows(conv, rows, plane, wp, oy, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
                                    let mut a = 0i32;
                                    for (kk, &wv) in wr.iter().enumerate() {
                                        a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                                    }
                                    write_out_sl(
                                        conv,
                                        yd,
                                        ybase,
                                        oh,
                                        ow,
                                        oy,
                                        ox0 + ti,
                                        mo + mi,
                                        a,
                                        relu,
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                }
                mo += mr;
            }
        }
    }
}

fn pack_pad_row(
    ximg: &[i8],
    ic: usize,
    h: usize,
    w: usize,
    wp: usize,
    zp: i8,
    iy: usize,
    dst: &mut [i8],
) {
    if iy >= h {
        dst.fill(zp);
        return;
    }
    for c in 0..ic {
        let d = &mut dst[c * wp..c * wp + wp];
        d[0] = zp;
        d[1..1 + w].copy_from_slice(&ximg[(c * h + iy) * w..(c * h + iy) * w + w]);
        d[1 + w] = zp;
    }
}

fn row_slot(oy: usize, kh: usize) -> usize {
    // rows packed: before loop, slot 0 = iy=-1, slot 1 = iy=0;
    // each oy packs iy=oy+1 into slot (oy+2)%3.
    // iy = oy + kh - 1 lives in slot (oy + kh) % 3?
    // oy=0, kh=0, iy=-1 → slot 0. (0+0)%3=0.
    // oy=0, kh=1, iy=0 → slot 1.
    // oy=0, kh=2, iy=1 → packed this iter into slot 2.
    // oy=1, kh=0, iy=0 → slot 1. (1+0)%3=1.
    // oy=1, kh=1, iy=1 → slot 2.
    // oy=1, kh=2, iy=2 → packed into slot (1+2)%3=0. Yes.
    (oy + kh) % 3
}

fn gather_one_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    tile: &mut [i8],
) {
    let (ic, k_pad, zp) = (conv.ic, conv.k_pad, conv.act_zp);
    let k_raw = ic * 9;
    tile[..k_raw].fill(zp);
    tile[k_raw..k_pad].fill(0);
    for c in 0..ic {
        for kh in 0..3 {
            let rs = row_slot(oy, kh) * plane + c * wp + ox;
            for kw in 0..3 {
                tile[c * 9 + kh * 3 + kw] = rows[rs + kw];
            }
        }
    }
}

fn gather_kn_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let nkk = k_pad * pn;
    // k_raw taps are overwritten; only the 16-wide K tail needs zeros.
    if k_pad > k_raw {
        kn[k_raw * pn..nkk].fill(0);
    }
    for c in 0..ic {
        for kh in 0..3 {
            let base = row_slot(oy, kh) * plane + c * wp + ox;
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                kn[kidx * pn..kidx * pn + pn].copy_from_slice(&rows[base + kw..base + kw + pn]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn implicit_tile_from_rows(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    gather_kn_from_rows(conv, rows, plane, wp, oy, ox, pn, kn);
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() && pn.is_multiple_of(NR16) {
        gemm_panel_kn16(
            conv, yd, ybase, oh, ow, oy, ox, pn, kn, nk, relu, i8d, oc0, oc1,
        );
        return;
    }
    let k_pad = conv.k_pad;
    let nkk = k_pad * pn;
    transpose_kn_nk(&kn[..nkk], &mut nk[..nkk], k_pad, pn);
    gemm_panel(conv, yd, ybase, oh, ow, oy, ox, pn, nk, relu, i8d, oc0, oc1);
}

/// 3×3 s2 pad=1: three padded input rows, advance by two rows per output
/// row (one row is reused). Gather is a stride-2 pull from `[ic][w+2]`.
fn conv3x3_s2_rows(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    relu: bool,
    i8d: Option<I8Dest>,
) {
    let (ic, oc, zp) = (conv.ic, conv.oc, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    let oh = y.h;
    let ow = y.w;
    let kn_len = conv.k_pad.saturating_mul(PN);
    let zip_len = conv.k_pad.saturating_mul(ow.max(PN).min(ZIP_WAVE_PX));
    let threads = intra_threads();
    let macs = (oc as u64) * (ic as u64) * 9 * (oh as u64) * (ow as u64);
    // l3/l4 downsample (oc>=128) was serial even with set_intra(2). OC-split
    // re-gathers per worker; two extra intra::run per ResNet, not per-row.
    if threads > 1 && macs >= 20_000_000 && oc >= 128 && oc / MR >= 2 {
        let tcount = threads.min(oc / MR).max(1);
        if tcount > 1 {
            let y_ptr = y.data.as_mut_ptr() as usize;
            let y_len = y.data.len();
            for ni in 0..n {
                let xbase = ni * ic * h * w;
                let ybase = ni * oc * oh * ow;
                let ximg = &xq[xbase..xbase + ic * h * w];
                crate::intra::run(tcount, |t| {
                    let (a, b) = oc_bounds(oc, t, tcount);
                    if a >= b {
                        return;
                    }
                    let yd = unsafe { std::slice::from_raw_parts_mut(y_ptr as *mut f32, y_len) };
                    ROW3.with(|rc| {
                        PANEL_KN.with(|knc| {
                            PANEL_NK.with(|nkc| {
                                let mut rows = rc.borrow_mut();
                                let mut kn = knc.borrow_mut();
                                let mut nk = nkc.borrow_mut();
                                if kn.len() < kn_len {
                                    kn.resize(kn_len, 0);
                                }
                                if nk.len() < zip_len {
                                    nk.resize(zip_len, 0);
                                }
                                if rows.len() < 3 * plane {
                                    rows.resize(3 * plane, zp);
                                }
                                s2_scan(
                                    conv, ximg, h, w, yd, ybase, oh, ow, &mut rows, &mut kn,
                                    &mut nk, relu, i8d, a, b,
                                );
                            });
                        });
                    });
                });
            }
            return;
        }
    }
    ROW3.with(|rc| {
        PANEL_KN.with(|knc| {
            PANEL_NK.with(|nkc| {
                let mut rows = rc.borrow_mut();
                let mut kn = knc.borrow_mut();
                let mut nk = nkc.borrow_mut();
                if kn.len() < kn_len {
                    kn.resize(kn_len, 0);
                }
                if nk.len() < zip_len {
                    nk.resize(zip_len, 0);
                }
                if rows.len() < 3 * plane {
                    rows.resize(3 * plane, zp);
                }
                for ni in 0..n {
                    let xbase = ni * ic * h * w;
                    let ybase = ni * oc * oh * ow;
                    let ximg = &xq[xbase..xbase + ic * h * w];
                    s2_scan(
                        conv,
                        ximg,
                        h,
                        w,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        &mut rows,
                        &mut kn,
                        &mut nk,
                        relu,
                        i8d,
                        0,
                        oc,
                    );
                }
            });
        });
    });
}

#[allow(clippy::too_many_arguments)]
fn s2_scan(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    rows: &mut [i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let (ic, zp) = (conv.ic, conv.act_zp);
    let wp = w + 2;
    let plane = ic * wp;
    // iy = -1 is out of range → pack_pad_row fills zp.
    pack_pad_row(ximg, ic, h, w, wp, zp, usize::MAX, &mut rows[0..plane]);
    pack_pad_row(ximg, ic, h, w, wp, zp, 0, &mut rows[plane..2 * plane]);
    pack_pad_row(ximg, ic, h, w, wp, zp, 1, &mut rows[2 * plane..3 * plane]);
    let mut top = 0usize;
    let mut mid = 1usize;
    let mut bot = 2usize;
    for oy in 0..oh {
        s2_scan_row_zip(
            conv,
            rows,
            plane,
            wp,
            [top, mid, bot],
            yd,
            ybase,
            oh,
            ow,
            oy,
            kn,
            nk,
            relu,
            i8d,
            oc0,
            oc1,
        );
        let next_mid = 2 * (oy + 1);
        let next_bot = next_mid + 1;
        top = bot;
        mid = (bot + 1) % 3;
        bot = (bot + 2) % 3;
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            next_mid,
            &mut rows[mid * plane..(mid + 1) * plane],
        );
        pack_pad_row(
            ximg,
            ic,
            h,
            w,
            wp,
            zp,
            next_bot,
            &mut rows[bot * plane..(bot + 1) * plane],
        );
    }
}

fn gather_kn_s2(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    slots: [usize; 3],
    ox: usize,
    pn: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    let k_raw = ic * 9;
    let nkk = k_pad * pn;
    if k_pad > k_raw {
        kn[k_raw * pn..nkk].fill(0);
    }
    for c in 0..ic {
        for kh in 0..3 {
            let row_base = slots[kh] * plane + c * wp;
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                let dst = &mut kn[kidx * pn..kidx * pn + pn];
                copy_strided2(dst, &rows[row_base + ox * 2 + kw..]);
            }
        }
    }
}

/// Dest[i] = src[2*i]. 16-wide unzip on aarch64; scalar tail.
pub(crate) fn copy_strided2(dst: &mut [i8], src: &[i8]) {
    let n = dst.len();
    debug_assert!(src.len() >= n.saturating_mul(2).saturating_sub(1).max(n));
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::{vld1q_s8, vst1q_s8, vuzp1q_s8};
        // Two 16-wide vuzp1 (not vld2q — that lost on product). Need 64 src
        // bytes; skip when the padded row is a tight 2n-1 tail.
        while i + 32 <= n && 2 * i + 64 <= src.len() {
            unsafe {
                let p = src.as_ptr().add(2 * i);
                let d = dst.as_mut_ptr().add(i);
                vst1q_s8(d, vuzp1q_s8(vld1q_s8(p), vld1q_s8(p.add(16))));
                vst1q_s8(
                    d.add(16),
                    vuzp1q_s8(vld1q_s8(p.add(32)), vld1q_s8(p.add(48))),
                );
            }
            i += 32;
        }
        while i + 16 <= n && 2 * i + 32 <= src.len() {
            unsafe {
                let a = vld1q_s8(src.as_ptr().add(2 * i));
                let b = vld1q_s8(src.as_ptr().add(2 * i + 16));
                vst1q_s8(dst.as_mut_ptr().add(i), vuzp1q_s8(a, b));
            }
            i += 16;
        }
    }
    while i < n {
        dst[i] = src[2 * i];
        i += 1;
    }
}

/// Same zip-then-OC schedule as s1, with stride-2 gather from padded rows.
#[allow(clippy::too_many_arguments)]
fn s2_scan_row_zip(
    conv: &Conv2d,
    rows: &[i8],
    plane: usize,
    wp: usize,
    slots: [usize; 3],
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let k_pad = conv.k_pad;
    let k_raw = conv.ic * 9;
    const WAVE: usize = 32;
    let mut oxs = [0usize; WAVE];
    let mut pns = [0usize; WAVE];
    let mut zoff = [0usize; WAVE];
    let mut ox = 0usize;
    let use_zip = {
        #[cfg(target_arch = "aarch64")]
        {
            has_dotprod() && k_pad.is_multiple_of(4)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    };
    if !use_zip {
        while ox < ow {
            let pn = if ox + PN <= ow {
                PN
            } else if ox + NR16 <= ow {
                NR16
            } else {
                break;
            };
            gather_kn_s2(conv, rows, plane, wp, slots, ox, pn, kn);
            let nkk = k_pad * pn;
            transpose_kn_nk(&kn[..nkk], &mut zip[..nkk], k_pad, pn);
            gemm_panel(
                conv, yd, ybase, oh, ow, oy, ox, pn, zip, relu, i8d, oc0, oc1,
            );
            ox += pn;
        }
        while ox < ow {
            // Scalar tail: one output pixel, NK tile in kn[0..k_pad].
            gather_kn_s2(conv, rows, plane, wp, slots, ox, 1, kn);
            store_col(
                conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
            );
            ox += 1;
        }
        return;
    }
    while ox < ow {
        let mut ntiles = 0usize;
        let mut acc = 0usize;
        while ox < ow && ntiles < WAVE {
            let pn = if ox + PN <= ow {
                PN
            } else if ox + NR16 <= ow {
                NR16
            } else {
                break;
            };
            oxs[ntiles] = ox;
            pns[ntiles] = pn;
            zoff[ntiles] = acc;
            acc += k_pad * pn;
            ntiles += 1;
            ox += pn;
        }
        if ntiles == 0 {
            while ox < ow {
                gather_kn_s2(conv, rows, plane, wp, slots, ox, 1, kn);
                store_col(
                    conv, yd, ybase, oh, ow, oy, ox, kn, k_raw, relu, i8d, oc0, oc1,
                );
                ox += 1;
            }
            break;
        }
        debug_assert!(zip.len() >= acc);
        for t in 0..ntiles {
            let pn = pns[t];
            gather_kn_s2(conv, rows, plane, wp, slots, oxs[t], pn, kn);
            #[cfg(target_arch = "aarch64")]
            pack_kn_sdot16(kn, &mut zip[zoff[t]..zoff[t] + k_pad * pn], k_pad, pn);
        }
        // Fat-K s2 only (layer-4 3x3 downsample, k_pad=1152). Narrower s2
        // stays W-hot OC-outer; k_pad>=512 on s2 already lost on product.
        let tile_outer = k_pad >= 1024;
        if tile_outer {
            for t in 0..ntiles {
                let pn = pns[t];
                let ox0 = oxs[t];
                #[cfg(target_arch = "aarch64")]
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut mo = oc0;
                while mo < oc1 {
                    let mr = (conv.oc - mo).min(MR);
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_s2(conv, rows, plane, wp, slots, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
                                    let mut a = 0i32;
                                    for (kk, &wv) in wr.iter().enumerate() {
                                        a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                                    }
                                    write_out_sl(
                                        conv,
                                        yd,
                                        ybase,
                                        oh,
                                        ow,
                                        oy,
                                        ox0 + ti,
                                        mo + mi,
                                        a,
                                        relu,
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                    mo += mr;
                }
            }
        } else {
            let mut mo = oc0;
            while mo < oc1 {
                let mr = (conv.oc - mo).min(MR);
                for t in 0..ntiles {
                    let pn = pns[t];
                    let ox0 = oxs[t];
                    #[cfg(target_arch = "aarch64")]
                    let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                    let mut no = 0usize;
                    while no < pn {
                        if mr == MR && no + NR16 <= pn {
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                kernel_4x16_zip_store(
                                    conv,
                                    yd,
                                    ybase,
                                    oh,
                                    ow,
                                    oy,
                                    ox0 + no,
                                    mo,
                                    z,
                                    pn,
                                    no,
                                    relu,
                                    i8d,
                                );
                            }
                            no += NR16;
                        } else {
                            gather_kn_s2(conv, rows, plane, wp, slots, ox0, pn, kn);
                            for mi in 0..mr {
                                let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                                for ti in no..pn {
                                    let mut a = 0i32;
                                    for (kk, &wv) in wr.iter().enumerate() {
                                        a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                                    }
                                    write_out_sl(
                                        conv,
                                        yd,
                                        ybase,
                                        oh,
                                        ow,
                                        oy,
                                        ox0 + ti,
                                        mo + mi,
                                        a,
                                        relu,
                                        i8d,
                                    );
                                }
                            }
                            no = pn;
                        }
                    }
                }
                mo += mr;
            }
        }
    }
}

fn conv3x3(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
) {
    let (ic, oc) = (conv.ic, conv.oc);
    let (oh, ow, sx) = (y.h, y.w, conv.stride.max(1));
    let k_raw = ic * 9;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * oc * oh * ow;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..oh {
            let iy_mid = oy * sx;
            let interior_y = iy_mid >= 1 && iy_mid + 1 < h;
            let mut ox = 0usize;
            if interior_y {
                while ox < 1 && ox < ow {
                    gather_one_3x3(conv, ximg, h, w, oy, ox, nk);
                    store_col(
                        conv,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        oy,
                        ox,
                        nk,
                        k_raw,
                        relu,
                        None,
                        0,
                        conv.oc,
                    );
                    ox += 1;
                }
                while ox + PN <= ow && last_ix(ox, PN, sx) < w {
                    implicit_tile_3x3(conv, ximg, y, ybase, h, w, oh, ow, oy, ox, PN, kn, nk, relu);
                    ox += PN;
                }
                while ox + NR16 <= ow && last_ix(ox, NR16, sx) < w {
                    implicit_tile_3x3(
                        conv, ximg, y, ybase, h, w, oh, ow, oy, ox, NR16, kn, nk, relu,
                    );
                    ox += NR16;
                }
                while ox + NR <= ow && last_ix(ox, NR, sx) < w {
                    implicit_tile_3x3(conv, ximg, y, ybase, h, w, oh, ow, oy, ox, NR, kn, nk, relu);
                    ox += NR;
                }
            }
            while ox < ow {
                gather_one_3x3(conv, ximg, h, w, oy, ox, nk);
                store_col(
                    conv,
                    &mut y.data,
                    ybase,
                    oh,
                    ow,
                    oy,
                    ox,
                    nk,
                    k_raw,
                    relu,
                    None,
                    0,
                    conv.oc,
                );
                ox += 1;
            }
        }
    }
}

/// Last input-x of a `pn`-wide 3×3 tile starting at output `ox` (pad=1).
fn last_ix(ox: usize, pn: usize, sx: usize) -> usize {
    (ox + pn - 1) * sx + 1
}

fn gather_one_3x3(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    oy: usize,
    ox: usize,
    tile: &mut [i8],
) {
    let (ic, k_pad, zp, sx, pad) = (
        conv.ic,
        conv.k_pad,
        conv.act_zp,
        conv.stride.max(1),
        conv.pad,
    );
    let k_raw = ic * 9;
    tile[..k_raw].fill(zp);
    tile[k_raw..k_pad].fill(0);
    let hh = h as isize;
    let ww = w as isize;
    let iy0 = oy as isize * sx as isize - pad as isize;
    let ix0 = ox as isize * sx as isize - pad as isize;
    for c in 0..ic {
        for kh in 0..3 {
            let iy = iy0 + kh as isize;
            for kw in 0..3 {
                let ix = ix0 + kw as isize;
                if iy >= 0 && iy < hh && ix >= 0 && ix < ww {
                    tile[c * 9 + kh * 3 + kw] = ximg[(c * h + iy as usize) * w + ix as usize];
                }
            }
        }
    }
}

/// Interior 3×3 tile: gather into KN (contiguous pixels per tap) then
/// transpose to NK for the SDOT kernel. `pn` is 8 or 32.
#[allow(clippy::too_many_arguments)]
fn implicit_tile_3x3(
    conv: &Conv2d,
    ximg: &[i8],
    y: &mut Tensor,
    ybase: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
) {
    let (ic, k_pad, sx) = (conv.ic, conv.k_pad, conv.stride.max(1));
    let nkk = k_pad * pn;
    debug_assert!(nkk <= kn.len() && nkk <= nk.len());
    kn[..nkk].fill(0);
    for c in 0..ic {
        for kh in 0..3 {
            let iy = oy * sx + kh - 1;
            let src = &ximg[(c * h + iy) * w..];
            for kw in 0..3 {
                let kidx = c * 9 + kh * 3 + kw;
                let ix0 = ox * sx + kw - 1;
                let dst = &mut kn[kidx * pn..kidx * pn + pn];
                if sx == 1 {
                    dst.copy_from_slice(&src[ix0..ix0 + pn]);
                } else {
                    for (t, d) in dst.iter_mut().enumerate() {
                        *d = src[ix0 + t * sx];
                    }
                }
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() && pn.is_multiple_of(NR16) {
        gemm_panel_kn16(
            conv,
            &mut y.data,
            ybase,
            oh,
            ow,
            oy,
            ox,
            pn,
            kn,
            nk,
            relu,
            None,
            0,
            conv.oc,
        );
        return;
    }
    transpose_kn_nk(&kn[..nkk], &mut nk[..nkk], k_pad, pn);
    gemm_panel(
        conv,
        &mut y.data,
        ybase,
        oh,
        ow,
        oy,
        ox,
        pn,
        nk,
        relu,
        None,
        0,
        conv.oc,
    );
}

fn transpose_kn_nk(kn: &[i8], nk: &mut [i8], k: usize, n: usize) {
    for ki in 0..k {
        let src = &kn[ki * n..ki * n + n];
        for ni in 0..n {
            nk[ni * k + ki] = src[ni];
        }
    }
}

fn conv1x1(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    kn: &mut [i8],
    nk: &mut [i8],
    relu: bool,
) {
    let (ic, k_pad, sx) = (conv.ic, conv.k_pad, conv.stride.max(1));
    let (oh, ow, oc) = (y.h, y.w, conv.oc);
    #[cfg(target_arch = "aarch64")]
    let row_zip = has_dotprod() && k_pad.is_multiple_of(4);
    #[cfg(not(target_arch = "aarch64"))]
    let row_zip = false;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * oc * oh * ow;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..oh {
            let iy = oy * sx;
            if row_zip {
                #[cfg(target_arch = "aarch64")]
                conv1x1_row_zip(
                    conv,
                    ximg,
                    h,
                    w,
                    iy,
                    sx,
                    &mut y.data,
                    ybase,
                    oh,
                    ow,
                    oy,
                    kn,
                    nk,
                    relu,
                    k_pad,
                    oc,
                );
            } else {
                let mut ox = 0usize;
                while ox < ow {
                    let pn = conv1x1_pn(sx, ow, w, ox);
                    if pn == 0 {
                        break;
                    }
                    gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
                    transpose_kn_nk(kn, nk, k_pad, pn);
                    gemm_panel(
                        conv,
                        &mut y.data,
                        ybase,
                        oh,
                        ow,
                        oy,
                        ox,
                        pn,
                        nk,
                        relu,
                        None,
                        0,
                        oc,
                    );
                    ox += pn;
                }
            }
        }
    }
}

fn conv1x1_pn(sx: usize, ow: usize, w: usize, ox: usize) -> usize {
    let max_pn = if sx == 1 {
        ow - ox
    } else {
        let last = w.saturating_sub(1) / sx;
        last.saturating_add(1).saturating_sub(ox)
    };
    if max_pn >= PN {
        PN
    } else if max_pn >= NR16 {
        NR16
    } else {
        max_pn.min(PN)
    }
}

fn gather_1x1(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    iy: usize,
    ox: usize,
    pn: usize,
    sx: usize,
    kn: &mut [i8],
) {
    let (ic, k_pad) = (conv.ic, conv.k_pad);
    if k_pad > ic {
        kn[ic * pn..k_pad * pn].fill(0);
    }
    for c in 0..ic {
        let dst = &mut kn[c * pn..c * pn + pn];
        if sx == 1 {
            let src0 = (c * h + iy) * w + ox;
            dst.copy_from_slice(&ximg[src0..src0 + pn]);
        } else if sx == 2 {
            let src0 = (c * h + iy) * w + ox * 2;
            copy_strided2(dst, &ximg[src0..]);
        } else {
            for t in 0..pn {
                dst[t] = ximg[(c * h + iy) * w + (ox + t) * sx];
            }
        }
    }
}

/// Gather every in-row 1×1 tile, zip once, then stream OC so the 4-row
/// weight panel stays hot (same schedule as 3×3 s1).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn conv1x1_row_zip(
    conv: &Conv2d,
    ximg: &[i8],
    h: usize,
    w: usize,
    iy: usize,
    sx: usize,
    yd: &mut [f32],
    ybase: usize,
    oh: usize,
    ow: usize,
    oy: usize,
    kn: &mut [i8],
    zip: &mut [i8],
    relu: bool,
    k_pad: usize,
    oc: usize,
) {
    const MAX_TILES: usize = 256;
    let mut oxs = [0usize; MAX_TILES];
    let mut pns = [0usize; MAX_TILES];
    let mut zoff = [0usize; MAX_TILES];
    let mut ntiles = 0usize;
    let mut acc = 0usize;
    let mut ox = 0usize;
    while ox < ow && ntiles < MAX_TILES {
        let pn = conv1x1_pn(sx, ow, w, ox);
        if pn == 0 || !pn.is_multiple_of(NR16) {
            break;
        }
        gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
        pack_kn_sdot16(kn, &mut zip[acc..acc + k_pad * pn], k_pad, pn);
        oxs[ntiles] = ox;
        pns[ntiles] = pn;
        zoff[ntiles] = acc;
        acc += k_pad * pn;
        ntiles += 1;
        ox += pn;
    }
    let mut mo = 0usize;
    while mo < oc {
        let mr = (oc - mo).min(MR);
        if mr == MR {
            for t in 0..ntiles {
                let pn = pns[t];
                let z = &zip[zoff[t]..zoff[t] + k_pad * pn];
                let mut no = 0usize;
                while no + NR16 <= pn {
                    unsafe {
                        kernel_4x16_zip_store(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            oxs[t] + no,
                            mo,
                            z,
                            pn,
                            no,
                            relu,
                            None,
                        );
                    }
                    no += NR16;
                }
            }
        } else {
            for t in 0..ntiles {
                gather_1x1(conv, ximg, h, w, iy, oxs[t], pns[t], sx, kn);
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                    for ti in 0..pns[t] {
                        let mut a = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            a += i32::from(wv) * i32::from(kn[kk * pns[t] + ti]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            oxs[t] + ti,
                            mo + mi,
                            a,
                            relu,
                            None,
                        );
                    }
                }
            }
        }
        mo += mr;
    }
    // Leftover 1..15: one overlapping 4x16 so the 1x1 tail stays on SDOT.
    // gemm_panel 4x8 on this tail is the old spilled mxn kernel.
    if ow >= NR16 && ox < ow && zip.len() >= k_pad * NR16 {
        let ox0 = ow - NR16;
        let pn = NR16;
        gather_1x1(conv, ximg, h, w, iy, ox0, pn, sx, kn);
        pack_kn_sdot16(kn, &mut zip[..k_pad * pn], k_pad, pn);
        let z = &zip[..k_pad * pn];
        let mut mo = 0usize;
        while mo < oc {
            let mr = (oc - mo).min(MR);
            if mr == MR {
                unsafe {
                    kernel_4x16_zip_store(
                        conv, yd, ybase, oh, ow, oy, ox0, mo, z, pn, 0, relu, None,
                    );
                }
            } else {
                gather_1x1(conv, ximg, h, w, iy, ox0, pn, sx, kn);
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * k_pad..(mo + mi + 1) * k_pad];
                    for ti in 0..pn {
                        let mut a = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            a += i32::from(wv) * i32::from(kn[kk * pn + ti]);
                        }
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            oh,
                            ow,
                            oy,
                            ox0 + ti,
                            mo + mi,
                            a,
                            relu,
                            None,
                        );
                    }
                }
            }
            mo += mr;
        }
        ox = ow;
    }
    while ox < ow {
        let pn = conv1x1_pn(sx, ow, w, ox);
        if pn == 0 {
            break;
        }
        gather_1x1(conv, ximg, h, w, iy, ox, pn, sx, kn);
        let nkk = k_pad * pn;
        transpose_kn_nk(&kn[..nkk], &mut zip[..nkk], k_pad, pn);
        gemm_panel(conv, yd, ybase, oh, ow, oy, ox, pn, zip, relu, None, 0, oc);
        ox += pn;
    }
}

fn conv_gather(
    conv: &Conv2d,
    n: usize,
    h: usize,
    w: usize,
    y: &mut Tensor,
    xq: &[i8],
    tile: &mut [i8],
    relu: bool,
) {
    let (ic, k, stride, pad, k_pad) = (conv.ic, conv.k, conv.stride, conv.pad, conv.k_pad);
    let k_raw = ic * k * k;
    for ni in 0..n {
        let xbase = ni * ic * h * w;
        let ybase = ni * conv.oc * y.h * y.w;
        let ximg = &xq[xbase..xbase + ic * h * w];
        for oy in 0..y.h {
            for ox in 0..y.w {
                tile[..k_raw].fill(conv.act_zp);
                tile[k_raw..k_pad].fill(0);
                let iy0 = oy as isize * stride as isize - pad as isize;
                let ix0 = ox as isize * stride as isize - pad as isize;
                let mut kk = 0usize;
                for c in 0..ic {
                    for kh in 0..k {
                        let iy = iy0 + kh as isize;
                        for kw in 0..k {
                            let ix = ix0 + kw as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                tile[kk] = ximg[(c * h + iy as usize) * w + ix as usize];
                            }
                            kk += 1;
                        }
                    }
                }
                debug_assert_eq!(kk, k_raw);
                store_col(
                    conv,
                    &mut y.data,
                    ybase,
                    y.h,
                    y.w,
                    oy,
                    ox,
                    tile,
                    k_raw,
                    relu,
                    None,
                    0,
                    conv.oc,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_panel(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    nk: &[i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let mut mo = oc0;
    while mo < oc1 {
        let mr = (conv.oc - mo).min(MR);
        let mut no = 0usize;
        while no < pn {
            let nr = (pn - no).min(NR);
            if mr == MR && nr == NR {
                let mut tmp = [[0i32; NR]; MR];
                kernel_mxn(conv, mo, &nk[no * conv.k_pad..], &mut tmp);
                for (mi, row) in tmp.iter().enumerate() {
                    for (t, &acc) in row.iter().enumerate() {
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no + t,
                            mo + mi,
                            acc,
                            relu,
                            i8d,
                        );
                    }
                }
            } else {
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
                    for t in 0..nr {
                        let xr = &nk[(no + t) * conv.k_pad..(no + t + 1) * conv.k_pad];
                        write_out_sl(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no + t,
                            mo + mi,
                            dot_i8(wr, xr),
                            relu,
                            i8d,
                        );
                    }
                }
            }
            no += nr;
        }
        mo += mr;
    }
}

#[allow(clippy::too_many_arguments)]
fn store_col(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    tile: &[i8],
    _k_raw: usize,
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    for oc in oc0..oc1 {
        let wr = &conv.q_w_pad[oc * conv.k_pad..oc * conv.k_pad + conv.k_pad];
        let a = dot_i8(wr, &tile[..conv.k_pad]);
        write_out_sl(conv, yd, ybase, yh, yw, oy, ox, oc, a, relu, i8d);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn write_out_sl(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    oc: usize,
    acc: i32,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    let mut v = acc as f32 * conv.out_scale[oc] + conv.eff_bias[oc];
    if relu && v < 0.0 {
        v = 0.0;
    }
    let idx = ybase + (oc * yh + oy) * yw + ox;
    if let Some(d) = i8d {
        d.store(idx, v);
    } else {
        yd[idx] = v;
    }
}

#[cfg(target_arch = "aarch64")]
fn has_dotprod() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        true
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
}

fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        return unsafe { dot_i8_sdot(a, b) };
    }
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += i32::from(x) * i32::from(y);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe)]
unsafe fn dot_i8_sdot(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::{int8x16_t, int32x4_t, vaddvq_s32, vdupq_n_s32, vld1q_s8};
    debug_assert!(a.len().is_multiple_of(16));
    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut k = 0;
    while k < a.len() {
        unsafe {
            let av: int8x16_t = vld1q_s8(a.as_ptr().add(k));
            let bv: int8x16_t = vld1q_s8(b.as_ptr().add(k));
            acc = sdot_dotprod(acc, av, bv);
        }
        k += 16;
    }
    let _: int32x4_t = acc;
    unsafe { vaddvq_s32(acc) }
}

fn kernel_mxn(conv: &Conv2d, mo: usize, tile: &[i8], out: &mut [[i32; NR]; MR]) {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        unsafe {
            kernel_mxn_sdot(conv, mo, tile, out);
        }
        return;
    }
    for mi in 0..MR {
        let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
        for t in 0..NR {
            out[mi][t] = dot_i8(wr, &tile[t * conv.k_pad..(t + 1) * conv.k_pad]);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe)]
unsafe fn kernel_mxn_sdot(conv: &Conv2d, mo: usize, tile: &[i8], out: &mut [[i32; NR]; MR]) {
    use std::arch::aarch64::{int8x16_t, vaddvq_s32, vdupq_n_s32, vld1q_s8};
    let k_pad = conv.k_pad;
    let mut acc = [[unsafe { vdupq_n_s32(0) }; NR]; MR];
    let wp = conv.q_w_pad.as_ptr();
    let tp = tile.as_ptr();
    let mut k = 0;
    while k < k_pad {
        unsafe {
            let mut xs = [vld1q_s8(tp.add(k)); NR];
            for (t, xv) in xs.iter_mut().enumerate() {
                *xv = vld1q_s8(tp.add(t * k_pad + k));
            }
            for (row, acc_row) in acc.iter_mut().enumerate() {
                let w: int8x16_t = vld1q_s8(wp.add((mo + row) * k_pad + k));
                for (t, xv) in xs.iter().enumerate() {
                    acc_row[t] = sdot_dotprod(acc_row[t], w, *xv);
                }
            }
        }
        k += 16;
    }
    for (mi, row) in out.iter_mut().enumerate() {
        for (t, slot) in row.iter_mut().enumerate() {
            *slot = unsafe { vaddvq_s32(acc[mi][t]) };
        }
    }
}

/// KN-panel GEMM: each `q`-reg holds 4 adjacent output pixels (not a K-reduction).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn gemm_panel_kn16(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    pn: usize,
    kn: &[i8],
    zip: &mut [i8],
    relu: bool,
    i8d: Option<I8Dest>,
    oc0: usize,
    oc1: usize,
) {
    let zip_len = conv.k_pad.saturating_mul(pn);
    let use_zip = zip.len() >= zip_len && pn.is_multiple_of(NR16) && conv.k_pad.is_multiple_of(4);
    if use_zip {
        pack_kn_sdot16(kn, &mut zip[..zip_len], conv.k_pad, pn);
    }
    let mut mo = oc0;
    while mo < oc1 {
        let mr = (conv.oc - mo).min(MR);
        let mut no = 0usize;
        while no < pn {
            if mr == MR && no + NR16 <= pn {
                unsafe {
                    if use_zip {
                        kernel_4x16_zip_store(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no,
                            mo,
                            zip,
                            pn,
                            no,
                            relu,
                            i8d,
                        );
                    } else {
                        kernel_4x16_kn_store(
                            conv,
                            yd,
                            ybase,
                            yh,
                            yw,
                            oy,
                            ox + no,
                            mo,
                            kn,
                            pn,
                            no,
                            relu,
                            i8d,
                        );
                    }
                }
                no += NR16;
            } else {
                for mi in 0..mr {
                    let wr = &conv.q_w_pad[(mo + mi) * conv.k_pad..(mo + mi + 1) * conv.k_pad];
                    for t in no..pn {
                        let mut acc = 0i32;
                        for (kk, &wv) in wr.iter().enumerate() {
                            acc += i32::from(wv) * i32::from(kn[kk * pn + t]);
                        }
                        write_out_sl(conv, yd, ybase, yh, yw, oy, ox + t, mo + mi, acc, relu, i8d);
                    }
                }
                no = pn;
            }
        }
        mo += mr;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store4_i8(dst: *mut i8, v: std::arch::aarch64::float32x4_t, inv: f32, zp: f32) {
    use std::arch::aarch64::{
        vaddq_f32, vcombine_s16, vcvtq_s32_f32, vdupq_n_f32, vget_lane_s32, vmaxq_f32, vminq_f32,
        vmulq_f32, vqmovn_s16, vqmovn_s32, vreinterpret_s32_s8, vrndaq_f32,
    };
    unsafe {
        let q = vaddq_f32(vrndaq_f32(vmulq_f32(v, vdupq_n_f32(inv))), vdupq_n_f32(zp));
        let qi = vcvtq_s32_f32(vmaxq_f32(
            vminq_f32(q, vdupq_n_f32(127.0)),
            vdupq_n_f32(-128.0),
        ));
        let i16 = vqmovn_s32(qi);
        let i8x8 = vqmovn_s16(vcombine_s16(i16, i16));
        let bits = vget_lane_s32::<0>(vreinterpret_s32_s8(i8x8)) as u32;
        core::ptr::write_unaligned(dst as *mut u32, bits);
    }
}

#[cfg(target_arch = "aarch64")]
fn pack_kn_sdot16(kn: &[i8], zip: &mut [i8], k_pad: usize, pn: usize) {
    use std::arch::aarch64::{
        vld1q_s8, vreinterpretq_s8_s16, vreinterpretq_s16_s8, vst1q_s8, vzip1q_s8, vzip1q_s16,
        vzip2q_s8, vzip2q_s16,
    };
    let mut o = 0usize;
    let mut k = 0usize;
    while k + 4 <= k_pad {
        let mut n0 = 0usize;
        while n0 + 16 <= pn {
            unsafe {
                let b0 = vld1q_s8(kn.as_ptr().add(k * pn + n0));
                let b1 = vld1q_s8(kn.as_ptr().add((k + 1) * pn + n0));
                let b2 = vld1q_s8(kn.as_ptr().add((k + 2) * pn + n0));
                let b3 = vld1q_s8(kn.as_ptr().add((k + 3) * pn + n0));
                let z01l = vzip1q_s8(b0, b1);
                let z01h = vzip2q_s8(b0, b1);
                let z23l = vzip1q_s8(b2, b3);
                let z23h = vzip2q_s8(b2, b3);
                let zp = zip.as_mut_ptr().add(o);
                vst1q_s8(
                    zp,
                    vreinterpretq_s8_s16(vzip1q_s16(
                        vreinterpretq_s16_s8(z01l),
                        vreinterpretq_s16_s8(z23l),
                    )),
                );
                vst1q_s8(
                    zp.add(16),
                    vreinterpretq_s8_s16(vzip2q_s16(
                        vreinterpretq_s16_s8(z01l),
                        vreinterpretq_s16_s8(z23l),
                    )),
                );
                vst1q_s8(
                    zp.add(32),
                    vreinterpretq_s8_s16(vzip1q_s16(
                        vreinterpretq_s16_s8(z01h),
                        vreinterpretq_s16_s8(z23h),
                    )),
                );
                vst1q_s8(
                    zp.add(48),
                    vreinterpretq_s8_s16(vzip2q_s16(
                        vreinterpretq_s16_s8(z01h),
                        vreinterpretq_s16_s8(z23h),
                    )),
                );
            }
            o += 64;
            n0 += 16;
        }
        k += 4;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe, clippy::too_many_arguments)]
unsafe fn kernel_4x16_zip_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    zip: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::aarch64::{
        int8x16_t, int32x4_t, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_s8,
        vmaxq_f32, vmovq_n_f32, vst1q_f32,
    };
    let k_pad = conv.k_pad;
    let tiles_n = pn / NR16;
    let tile = n0 / NR16;
    let zp = zip.as_ptr();
    let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; MR];
    let mut k = 0usize;
    while k + 4 <= k_pad {
        unsafe {
            let off = (k / 4) * tiles_n * 64 + tile * 64;
            let bt = [
                vld1q_s8(zp.add(off)),
                vld1q_s8(zp.add(off + 16)),
                vld1q_s8(zp.add(off + 32)),
                vld1q_s8(zp.add(off + 48)),
            ];
            let a_tile: int8x16_t = if !conv.q_w_4x4.is_empty() {
                let w4 = conv.q_w_4x4.as_ptr();
                let wrow = (mo / 4) * (k_pad / 4) * 16;
                vld1q_s8(w4.add(wrow + (k / 4) * 16))
            } else {
                let wp = conv.q_w_pad.as_ptr();
                let mut abuf = [0i8; 16];
                core::ptr::copy_nonoverlapping(wp.add(mo * k_pad + k), abuf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 1) * k_pad + k),
                    abuf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 2) * k_pad + k),
                    abuf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 3) * k_pad + k),
                    abuf.as_mut_ptr().add(12),
                    4,
                );
                vld1q_s8(abuf.as_ptr())
            };
            for i in 0..4 {
                acc[0][i] = sdot_lane::<0>(acc[0][i], bt[i], a_tile);
                acc[1][i] = sdot_lane::<1>(acc[1][i], bt[i], a_tile);
                acc[2][i] = sdot_lane::<2>(acc[2][i], bt[i], a_tile);
                acc[3][i] = sdot_lane::<3>(acc[3][i], bt[i], a_tile);
            }
        }
        k += 4;
    }
    let zero = unsafe { vmovq_n_f32(0.0) };
    for r in 0..MR {
        let scale = unsafe { vdupq_n_f32(conv.out_scale[mo + r]) };
        let bias = unsafe { vdupq_n_f32(conv.eff_bias[mo + r]) };
        let dst0 = ybase + ((mo + r) * yh + oy) * yw + ox;
        for i in 0..4 {
            unsafe {
                let mut v = vfmaq_f32(bias, vcvtq_f32_s32(acc[r][i]), scale);
                if relu {
                    v = vmaxq_f32(v, zero);
                }
                if let Some(d) = i8d {
                    store4_i8((d.p as *mut i8).add(dst0 + i * 4), v, d.inv_scale, d.zp);
                } else {
                    vst1q_f32(yd[dst0 + i * 4..].as_mut_ptr(), v);
                }
            }
        }
    }
    let _: int32x4_t = acc[0][0];
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unused_unsafe, clippy::too_many_arguments)]
unsafe fn kernel_4x16_kn_store(
    conv: &Conv2d,
    yd: &mut [f32],
    ybase: usize,
    yh: usize,
    yw: usize,
    oy: usize,
    ox: usize,
    mo: usize,
    kn: &[i8],
    pn: usize,
    n0: usize,
    relu: bool,
    i8d: Option<I8Dest>,
) {
    use std::arch::aarch64::{
        int8x16_t, int32x4_t, vcvtq_f32_s32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32, vld1q_s8,
        vmaxq_f32, vmovq_n_f32, vreinterpretq_s8_s16, vreinterpretq_s16_s8, vst1q_f32, vzip1q_s8,
        vzip1q_s16, vzip2q_s8, vzip2q_s16,
    };
    let k_pad = conv.k_pad;
    let wp = conv.q_w_pad.as_ptr();
    let kp = kn.as_ptr();
    let mut acc = [[unsafe { vdupq_n_s32(0) }; 4]; MR];
    let mut k = 0usize;
    while k + 4 <= k_pad {
        unsafe {
            if k + 8 <= k_pad {
                // Next K-group of activations + the next 4×4 weight tile.
                core::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) kp.add((k + 4) * pn + n0),
                    options(readonly, nostack),
                );
                if !conv.q_w_4x4.is_empty() {
                    let tiles_k = k_pad / 4;
                    let noff = (mo / 4) * tiles_k * 16 + (k / 4 + 1) * 16;
                    core::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) conv.q_w_4x4.as_ptr().add(noff),
                        options(readonly, nostack),
                    );
                }
            }
            let b0 = vld1q_s8(kp.add(k * pn + n0));
            let b1 = vld1q_s8(kp.add((k + 1) * pn + n0));
            let b2 = vld1q_s8(kp.add((k + 2) * pn + n0));
            let b3 = vld1q_s8(kp.add((k + 3) * pn + n0));
            let z01l = vzip1q_s8(b0, b1);
            let z01h = vzip2q_s8(b0, b1);
            let z23l = vzip1q_s8(b2, b3);
            let z23h = vzip2q_s8(b2, b3);
            let bt = [
                vreinterpretq_s8_s16(vzip1q_s16(
                    vreinterpretq_s16_s8(z01l),
                    vreinterpretq_s16_s8(z23l),
                )),
                vreinterpretq_s8_s16(vzip2q_s16(
                    vreinterpretq_s16_s8(z01l),
                    vreinterpretq_s16_s8(z23l),
                )),
                vreinterpretq_s8_s16(vzip1q_s16(
                    vreinterpretq_s16_s8(z01h),
                    vreinterpretq_s16_s8(z23h),
                )),
                vreinterpretq_s8_s16(vzip2q_s16(
                    vreinterpretq_s16_s8(z01h),
                    vreinterpretq_s16_s8(z23h),
                )),
            ];
            let a_tile: int8x16_t = if !conv.q_w_4x4.is_empty() {
                let tiles_k = k_pad / 4;
                let off = (mo / 4) * tiles_k * 16 + (k / 4) * 16;
                vld1q_s8(conv.q_w_4x4.as_ptr().add(off))
            } else {
                let mut abuf = [0i8; 16];
                core::ptr::copy_nonoverlapping(wp.add(mo * k_pad + k), abuf.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 1) * k_pad + k),
                    abuf.as_mut_ptr().add(4),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 2) * k_pad + k),
                    abuf.as_mut_ptr().add(8),
                    4,
                );
                core::ptr::copy_nonoverlapping(
                    wp.add((mo + 3) * k_pad + k),
                    abuf.as_mut_ptr().add(12),
                    4,
                );
                vld1q_s8(abuf.as_ptr())
            };
            for i in 0..4 {
                acc[0][i] = sdot_lane::<0>(acc[0][i], bt[i], a_tile);
                acc[1][i] = sdot_lane::<1>(acc[1][i], bt[i], a_tile);
                acc[2][i] = sdot_lane::<2>(acc[2][i], bt[i], a_tile);
                acc[3][i] = sdot_lane::<3>(acc[3][i], bt[i], a_tile);
            }
        }
        k += 4;
    }
    let zero = unsafe { vmovq_n_f32(0.0) };
    for r in 0..MR {
        let scale = unsafe { vdupq_n_f32(conv.out_scale[mo + r]) };
        let bias = unsafe { vdupq_n_f32(conv.eff_bias[mo + r]) };
        let dst0 = ybase + ((mo + r) * yh + oy) * yw + ox;
        for i in 0..4 {
            unsafe {
                let mut v = vfmaq_f32(bias, vcvtq_f32_s32(acc[r][i]), scale);
                if relu {
                    v = vmaxq_f32(v, zero);
                }
                if let Some(d) = i8d {
                    store4_i8((d.p as *mut i8).add(dst0 + i * 4), v, d.inv_scale, d.zp);
                } else {
                    vst1q_f32(yd[dst0 + i * 4..].as_mut_ptr(), v);
                }
            }
        }
    }
    let _: int32x4_t = acc[0][0];
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
#[allow(unused_unsafe)]
unsafe fn sdot_lane<const LANE: i32>(
    mut acc: std::arch::aarch64::int32x4_t,
    b: std::arch::aarch64::int8x16_t,
    a: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SDOT Vd.4S, Vn.16B, Vm.4B[lane] — four columns × 4-K against one A row.
    unsafe {
        match LANE {
            0 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[0]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            1 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[1]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            2 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[2]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            3 => core::arch::asm!(
                "sdot {acc:v}.4s, {b:v}.16b, {a:v}.4b[3]",
                acc = inout(vreg) acc,
                b = in(vreg) b,
                a = in(vreg) a,
                options(pure, nomem, nostack),
            ),
            _ => {}
        }
    }
    acc
}

/// Must live in a `+dotprod` function — Linux rustc will not assemble `sdot`
/// on the generic aarch64 target. Keep it `#[inline]` so the helper does not
/// take SIMD args across a non-dotprod ABI boundary.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
#[allow(unused_unsafe)]
unsafe fn sdot_dotprod(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SDOT Vd.4S, Vn.16B, Vm.16B — four 4-wide i8 dots into the four lanes.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

#[cfg(test)]
mod strided2_tests {
    #[test]
    fn copy_strided2_matches_scalar() {
        for n in [1usize, 8, 16, 17, 32, 33, 48, 64] {
            let src: Vec<i8> = (0..n * 2 + 8).map(|i| (i as i8).wrapping_mul(3)).collect();
            let mut got = vec![0i8; n];
            let mut want = vec![0i8; n];
            super::copy_strided2(&mut got, &src);
            for i in 0..n {
                want[i] = src[2 * i];
            }
            assert_eq!(got, want, "n={n}");
        }
    }
}
