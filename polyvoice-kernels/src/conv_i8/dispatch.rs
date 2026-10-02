
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

static FILE_PARALLELISM: AtomicUsize = AtomicUsize::new(1);

/// How many files the caller diarizes concurrently. Front doors that fan out
/// internally (segmentation windows, embed pool) divide their fan-out by this
/// so jobs × workers stays near core count. 1 (default) = single-file.
pub fn set_file_parallelism(n: usize) {
    FILE_PARALLELISM.store(n.max(1), Ordering::Relaxed);
}

pub fn file_parallelism() -> usize {
    FILE_PARALLELISM.load(Ordering::Relaxed).max(1)
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

#[cfg(all(test, target_arch = "x86_64"))]
#[test]
fn vnni_dot_matches_scalar() {
    if !has_avx512_vnni() {
        return;
    }
    for n in [16usize, 23, 64, 80, 144, 288, 1152] {
        let a: Vec<i8> = (0..n)
            .map(|i| (i as i8).wrapping_mul(37).wrapping_add(11))
            .collect();
        let b: Vec<i8> = (0..n)
            .map(|i| (i as i8).wrapping_mul(-19).wrapping_add(7))
            .collect();
        let want: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| i32::from(x) * i32::from(y))
            .sum();
        assert_eq!(unsafe { dot_i8_vnni(&a, &b) }, want, "n={n}");
    }
    // Full-range extremes: VPMADDUBSW would saturate; VPDPBUSD stays exact.
    let lo = vec![-128i8; 64];
    assert_eq!(unsafe { dot_i8_vnni(&lo, &lo) }, 64 * 16384);
    let hi = vec![127i8; 64];
    assert_eq!(unsafe { dot_i8_vnni(&hi, &lo) }, 64 * -16256);
}

#[cfg(all(test, target_arch = "x86_64"))]
#[test]
fn vnni_zip_kernel_matches_scalar() {
    if !has_avx512_vnni() {
        return;
    }
    let (oc, ic) = (8usize, 4usize);
    let k_raw = ic * 9;
    let k_pad = k_raw.div_ceil(16) * 16;
    let q_w: Vec<i8> = (0..oc * k_raw)
        .map(|i| (i as i8).wrapping_mul(29).wrapping_sub(40))
        .collect();
    let q_scale = vec![0.003f32; oc];
    let bias: Vec<f32> = (0..oc).map(|o| o as f32 * 0.01 - 0.03).collect();
    let conv = crate::conv::Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, bias)
        .with_input_quant(0.04, -128);
    let pn = 32usize;
    let kn: Vec<i8> = (0..k_pad * pn)
        .map(|i| (i as i8).wrapping_mul(13).wrapping_add(5))
        .collect();
    let mut zip = vec![0i8; k_pad * pn];
    pack_kn_zip16(&kn, &mut zip, k_pad, pn);
    let (yh, yw) = (1usize, 64usize);
    let mut got = vec![0f32; oc * yh * yw];
    for mo in (0..oc).step_by(MR) {
        // Zip variant, second 16-px tile, no relu.
        unsafe {
            kernel_4x16_zip_store(
                &conv, &mut got, 0, yh, yw, 0, 16, mo, &zip, pn, 16, false, None,
            );
        }
        // KN register-interleave variant, first tile, relu on.
        unsafe {
            kernel_4x16_kn_store(&conv, &mut got, 0, yh, yw, 0, 0, mo, &kn, pn, 0, true, None);
        }
    }
    for o in 0..oc {
        let wr = &conv.q_w_pad[o * k_pad..(o + 1) * k_pad];
        for t in 0..pn {
            let mut acc = 0i32;
            for (kk, &wv) in wr.iter().enumerate() {
                acc += i32::from(wv) * i32::from(kn[kk * pn + t]);
            }
            let mut want = acc as f32 * conv.out_scale[o] + conv.eff_bias[o];
            if t < 16 {
                want = want.max(0.0);
            }
            let got_v = got[o * yw + t];
            assert!(
                (got_v - want).abs() < 1e-3,
                "oc={o} t={t} got={got_v} want={want}"
            );
        }
    }
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
    scale: f32,
    zp: f32,
}

impl I8Dest {
    #[inline(always)]
    fn store(self, idx: usize, v: f32) {
        let q = (v / self.scale).round() + self.zp;
        // SAFETY: try_conv_to_i8 sizes the dest to N·OC·OH·OW; idx is the
        // same NCHW address the f32 store would have used.
        unsafe {
            *(self.p as *mut i8).add(idx) = q.clamp(-128.0, 127.0) as i8;
        }
    }
}

/// Requantize a dequantized conv output onto its QDQ pre-add lattice
/// (ONNX `QuantizeLinear`/`DequantizeLinear`: round, clamp, dequant).
#[inline(always)]
fn requant_f32(v: f32, scale: f32, zp: i8) -> f32 {
    let z = f32::from(zp);
    let q = (v / scale).round() + z;
    (q.clamp(-128.0, 127.0) - z) * scale
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
            // x86_64: exact only with AVX-512 VNNI (VPDPBUSD); VPMADDUBSW
            // kernels saturate on full-range i8, so scalar stays the default.
            #[cfg(target_arch = "x86_64")]
            if has_avx512_vnni() {
                return true;
            }
            cfg!(all(
                target_arch = "aarch64",
                any(
                    target_os = "linux",
                    all(target_vendor = "apple", not(apple_accelerate))
                )
            ))
        })
    }
}

/// True if this layer ran the integer kernel (caller must not also run BNNS).
pub fn try_conv(conv: &Conv2d, x: &Tensor, y: &mut Tensor, relu: bool) -> bool {
    if conv.q_w_pad.is_empty() || conv.out_scale.is_empty() || x.n == 0 {
        return false;
    }
    // Apple with Accelerate: BNNS Winograd is faster; keep integer GEMM opt-in.
    // Linux aarch64 (SDOT) and x86_64 with AVX-512 VNNI: no BNNS — the
    // exact integer GEMM is the faster default. POLYVOICE_NO_I8_CONV=1
    // forces the float path; POLYVOICE_I8_CONV=1 forces integer everywhere.
    if !i8_conv_on() {
        return false;
    }
    // Full-map rten im2col is slower than the in-crate 4×16 SDOT kernel
    // (ResNet T=400: ~520 ms vs ~230 ms). Opt in with POLYVOICE_RTEN_CONV=1.
    #[cfg(not(apple_accelerate))]
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
        scale: next_scale,
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

