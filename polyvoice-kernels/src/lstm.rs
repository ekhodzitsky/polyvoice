//! ONNX bidirectional LSTM (no peephole, input_forget=0).
//!
//! Gate order is ONNX's `i, o, f, c`. ONNX stores `W [2, 4H, I]`,
//! `R [2, 4H, H]`, `B [2, 8H]`. We keep **transposed** `W [2, I, 4H]` and
//! `R [2, H, 4H]` plus fused `Wb+Rb [2, 4H]` so each step is a GEMM + gates.
//! Sequence layout is `[seq, batch, feat]`.

use crate::gemm::{gemm_add, gemm_bias, transpose};
use std::cell::RefCell;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-40.0, 40.0)).exp())
}

struct LstmScratch {
    xw: Vec<f32>,
    ht: Vec<f32>,
    ct: Vec<f32>,
}

thread_local! {
    static LSTM_SCRATCH: RefCell<LstmScratch> = const {
        RefCell::new(LstmScratch {
            xw: Vec::new(),
            ht: Vec::new(),
            ct: Vec::new(),
        })
    };
}

/// ONNX gates `i,o,f,c` then `h = o * tanh(f*c + i*tanh(c̃))`.
///
/// On aarch64 the four gate planes (`i`, `o`, `f`, `c̃` are contiguous
/// `h`-runs) are processed four lanes at a time with NEON. The vector
/// transcendentals are minimax approximations accurate to a few ulp against
/// scalar libm, so logits can drift by ~1e-7 relative; the scalar path below
/// remains the reference for other targets.
#[inline]
fn apply_gates(gates: &[f32], ht: &mut [f32], ct: &mut [f32], y: &mut [f32], h: usize) {
    debug_assert_eq!(gates.len(), 4 * h);
    debug_assert_eq!(ht.len(), h);
    debug_assert_eq!(ct.len(), h);
    debug_assert_eq!(y.len(), h);
    #[cfg(target_arch = "aarch64")]
    let mut hh = neon_gates::apply_gates_vec(gates, ht, ct, y, h);
    #[cfg(not(target_arch = "aarch64"))]
    let mut hh = 0;
    while hh < h {
        let it = sigmoid(gates[hh]);
        let ot = sigmoid(gates[h + hh]);
        let ft = sigmoid(gates[2 * h + hh]);
        let c_tilde = gates[3 * h + hh].tanh();
        let c = ft * ct[hh] + it * c_tilde;
        ct[hh] = c;
        let hv = ot * c.tanh();
        ht[hh] = hv;
        y[hh] = hv;
        hh += 1;
    }
}

/// NEON gate evaluation. `exp` is a Cephes-style degree-5 minimax on the
/// Cody–Waite reduced range (≤ ~2 ulp vs libm on the clamped domain);
/// sigmoid and tanh reuse it with an IEEE `fdiv`, landing within ~4 ulp of
/// the scalar path. Inputs are clamped exactly like the scalar sigmoid, and
/// the tanh form `1 − 2/(e^{2|x|}+1)` saturates to ±1 for large `|x|` just
/// like libm. NEON is baseline on aarch64, hence the safe wrappers.
#[cfg(target_arch = "aarch64")]
mod neon_gates {
    use core::arch::aarch64::*;

    /// exp(x) for finite x, clamped to [-87, 87] (beyond that f32 overflows
    /// or flushes toward 0 anyway). ~1–2 ulp.
    #[inline(always)]
    fn exp4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let x = vminq_f32(x, vdupq_n_f32(87.0));
            let x = vmaxq_f32(x, vdupq_n_f32(-87.0));
            // n = rint(x * log2(e)); r = x − n·ln2 via Cody–Waite split.
            let nf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.442_695_040_888_963_4)));
            let ni = vcvtq_s32_f32(nf);
            let r = vfmsq_f32(x, nf, vdupq_n_f32(0.693_359_375));
            let r = vfmsq_f32(r, nf, vdupq_n_f32(-2.121_944_40e-4));
            let z = vmulq_f32(r, r);
            let mut p = vdupq_n_f32(1.987_569_150_0e-4);
            p = vfmaq_f32(vdupq_n_f32(1.398_199_950_7e-3), p, r);
            p = vfmaq_f32(vdupq_n_f32(8.333_451_907_3e-3), p, r);
            p = vfmaq_f32(vdupq_n_f32(4.166_579_589_4e-2), p, r);
            p = vfmaq_f32(vdupq_n_f32(1.666_666_545_9e-1), p, r);
            p = vfmaq_f32(vdupq_n_f32(5.000_000_120_1e-1), p, r);
            let y = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0)), p, z);
            let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, vdupq_n_s32(127))));
            vmulq_f32(y, pow2n)
        }
    }

    /// Same clamp-then-`1/(1+e^-x)` shape as the scalar sigmoid.
    #[inline(always)]
    pub(super) fn sigmoid4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let x = vmaxq_f32(x, vdupq_n_f32(-40.0));
            let x = vminq_f32(x, vdupq_n_f32(40.0));
            let e = exp4(vnegq_f32(x));
            vdivq_f32(vdupq_n_f32(1.0), vaddq_f32(vdupq_n_f32(1.0), e))
        }
    }

    /// Cephes-style tanhf: for |x| < 0.625 the odd rational form
    /// `x + x·z·P(z)/Q(z)`, z = x² (avoids the `1 − 2/(e^{2x}+1)`
    /// cancellation near zero); larger |x| take the exp form, which
    /// saturates to ±1 like libm. ~2–4 ulp overall.
    #[inline(always)]
    pub(super) fn tanh4(x: float32x4_t) -> float32x4_t {
        unsafe {
            // Small-|x| branch: polynomial in z = x², odd in x.
            let z = vmulq_f32(x, x);
            let mut p = vdupq_n_f32(-9.643_991_794_250_522_4e-1);
            p = vfmaq_f32(vdupq_n_f32(-9.928_772_310_019_185_9e1), p, z);
            p = vfmaq_f32(vdupq_n_f32(-1.614_687_684_417_084_5e3), p, z);
            let mut q = vaddq_f32(z, vdupq_n_f32(1.128_116_784_916_329_3e2));
            q = vfmaq_f32(vdupq_n_f32(2.235_488_390_601_004_5e3), q, z);
            q = vfmaq_f32(vdupq_n_f32(4.844_063_053_251_254_9e3), q, z);
            let small = vfmaq_f32(x, vmulq_f32(x, z), vdivq_f32(p, q));
            // Large-|x| branch: sign(x)·(1 − 2/(e^{2|x|} + 1)).
            let ax = vabsq_f32(x);
            let e = exp4(vaddq_f32(ax, ax));
            let m = vsubq_f32(
                vdupq_n_f32(1.0),
                vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(e, vdupq_n_f32(1.0))),
            );
            let large = vbslq_f32(vdupq_n_u32(0x8000_0000), x, m);
            vbslq_f32(vcaltq_f32(x, vdupq_n_f32(0.625)), small, large)
        }
    }

    /// Vector body of [`super::apply_gates`]; returns the first index not yet
    /// processed (the scalar tail continues from there). Linear terms keep
    /// the scalar operand order (two multiplies, one add — no FMA) so only
    /// the transcendentals differ from the scalar path.
    pub(super) fn apply_gates_vec(
        gates: &[f32],
        ht: &mut [f32],
        ct: &mut [f32],
        y: &mut [f32],
        h: usize,
    ) -> usize {
        let mut hh = 0;
        unsafe {
            while hh + 4 <= h {
                let it = sigmoid4(vld1q_f32(gates[hh..].as_ptr()));
                let ot = sigmoid4(vld1q_f32(gates[h + hh..].as_ptr()));
                let ft = sigmoid4(vld1q_f32(gates[2 * h + hh..].as_ptr()));
                let c_tilde = tanh4(vld1q_f32(gates[3 * h + hh..].as_ptr()));
                let c = vaddq_f32(
                    vmulq_f32(ft, vld1q_f32(ct[hh..].as_ptr())),
                    vmulq_f32(it, c_tilde),
                );
                vst1q_f32(ct[hh..].as_mut_ptr(), c);
                let hv = vmulq_f32(ot, tanh4(c));
                vst1q_f32(ht[hh..].as_mut_ptr(), hv);
                vst1q_f32(y[hh..].as_mut_ptr(), hv);
                hh += 4;
            }
        }
        hh
    }
}

/// One bidirectional LSTM layer. `input_size` is I, `hidden` is H.
// The INT8 weight path below is wired up only on non-Apple targets; on Apple
// the fields/methods exist but stay unread, so silence the lint there only.
#[cfg_attr(target_vendor = "apple", allow(dead_code))]
pub struct BiLstm {
    pub hidden: usize,
    pub input: usize,
    /// [2, I, 4H] — ONNX W transposed per direction
    w_t: Vec<f32>,
    /// [2, H, 4H] — ONNX R transposed per direction
    r_t: Vec<f32>,
    /// [2, 4H] — Wb + Rb fused
    bias: Vec<f32>,
    /// Shipping QDQ `W` as `[2, I, 4H]` (same layout as `w_t`). Empty = FP32.
    w_i8: Vec<i8>,
    w_scale: Vec<f32>,
    w_zp: Vec<i8>,
    r_i8: Vec<i8>,
    r_scale: Vec<f32>,
    r_zp: Vec<i8>,
    /// Fixed activation scales (not min/max of the current tensor).
    /// Hidden is tanh, so `|h| ≤ 1`. Input uses a conservative bound.
    x_scale: f32,
    h_scale: f32,
}

impl BiLstm {
    /// Build from ONNX initializers: `w [2, 4H, I]`, `r [2, 4H, H]`, `b [2, 8H]`.
    pub fn from_onnx(w: Vec<f32>, r: Vec<f32>, b: Vec<f32>, hidden: usize, input: usize) -> Self {
        let four_h = 4 * hidden;
        let mut w_t = Vec::with_capacity(2 * input * four_h);
        let mut r_t = Vec::with_capacity(2 * hidden * four_h);
        let mut bias = vec![0.0f32; 2 * four_h];
        for dir in 0..2 {
            let w_off = dir * four_h * input;
            w_t.extend(transpose(&w[w_off..w_off + four_h * input], four_h, input));
            let r_off = dir * four_h * hidden;
            r_t.extend(transpose(
                &r[r_off..r_off + four_h * hidden],
                four_h,
                hidden,
            ));
            let b_off = dir * 8 * hidden;
            for g in 0..4 {
                for hh in 0..hidden {
                    bias[dir * four_h + g * hidden + hh] =
                        b[b_off + g * hidden + hh] + b[b_off + (4 + g) * hidden + hh];
                }
            }
        }
        Self {
            hidden,
            input,
            w_t,
            r_t,
            bias,
            w_i8: Vec::new(),
            w_scale: Vec::new(),
            w_zp: Vec::new(),
            r_i8: Vec::new(),
            r_scale: Vec::new(),
            r_zp: Vec::new(),
            x_scale: 8.0 / 127.0,
            h_scale: 1.0 / 127.0,
        }
    }

    /// Attach shipping QDQ weights. `w_zp`/`r_zp` must be all-zero (signed INT8).
    #[cfg_attr(target_vendor = "apple", allow(dead_code))]
    pub fn with_i8(
        mut self,
        w_i8: Vec<i8>,
        w_scale: Vec<f32>,
        w_zp: Vec<i8>,
        r_i8: Vec<i8>,
        r_scale: Vec<f32>,
        r_zp: Vec<i8>,
    ) -> Self {
        if w_zp.iter().any(|&z| z != 0) || r_zp.iter().any(|&z| z != 0) {
            return self;
        }
        self.w_i8 = w_i8;
        self.w_scale = w_scale;
        self.w_zp = w_zp;
        self.r_i8 = r_i8;
        self.r_scale = r_scale;
        self.r_zp = r_zp;
        self
    }

    #[cfg(test)]
    pub fn has_i8_weights(&self) -> bool {
        !self.w_i8.is_empty()
    }

    /// `x` is `[seq, batch, input]`. Returns `[seq, batch, 2*hidden]`
    /// (forward ‖ reverse at each t).
    #[cfg(test)]
    pub fn forward(&self, x: &[f32], seq: usize, batch: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), seq * batch * self.input);
        let h = self.hidden;
        let mut y = vec![0.0f32; seq * batch * 2 * h];
        self.forward_into(x, seq, batch, &mut y);
        y
    }

    /// Write `[seq, batch, 2*hidden]` into `y` (resized).
    pub fn forward_into(&self, x: &[f32], seq: usize, batch: usize, y: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), seq * batch * self.input);
        let h = self.hidden;
        y.clear();
        y.resize(seq * batch * 2 * h, 0.0);
        self.run_direction(x, seq, batch, 0, false, y);
        self.run_direction(x, seq, batch, 1, true, y);
    }

    fn run_direction(
        &self,
        x: &[f32],
        seq: usize,
        batch: usize,
        dir: usize,
        reverse: bool,
        y: &mut [f32],
    ) {
        let h = self.hidden;
        let i_sz = self.input;
        let four_h = 4 * h;
        let wt = &self.w_t[dir * i_sz * four_h..(dir + 1) * i_sz * four_h];
        let rt = &self.r_t[dir * h * four_h..(dir + 1) * h * four_h];
        let bias = &self.bias[dir * four_h..(dir + 1) * four_h];

        LSTM_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let mut xw = std::mem::take(&mut scratch.xw);
            let mut ht = std::mem::take(&mut scratch.ht);
            let mut ct = std::mem::take(&mut scratch.ct);
            xw.resize(seq * batch * four_h, 0.0);
            ht.clear();
            ht.resize(batch * h, 0.0);
            ct.clear();
            ct.resize(batch * h, 0.0);
            let used_i8_w = try_i8_w(self, x, wt, bias, &mut xw, seq, batch, four_h, i_sz, dir);
            if !used_i8_w {
                gemm_bias(x, wt, bias, &mut xw, seq * batch, four_h, i_sz);
            }

            let mut remaining = seq;
            let mut t = if reverse { seq } else { 0 };
            while remaining > 0 {
                if reverse {
                    t -= 1;
                }
                let g0 = t * batch * four_h;
                let gates = &mut xw[g0..g0 + batch * four_h];
                let used_i8_r = try_i8_r(self, &ht, rt, gates, batch, four_h, h, dir);
                if !used_i8_r {
                    gemm_add(&ht, rt, gates, batch, four_h, h);
                }
                for b_i in 0..batch {
                    let base = b_i * four_h;
                    let h_base = b_i * h;
                    let y_off = (t * batch + b_i) * 2 * h + dir * h;
                    apply_gates(
                        &gates[base..base + four_h],
                        &mut ht[h_base..h_base + h],
                        &mut ct[h_base..h_base + h],
                        &mut y[y_off..y_off + h],
                        h,
                    );
                }
                if !reverse {
                    t += 1;
                }
                remaining -= 1;
            }
            scratch.xw = xw;
            scratch.ht = ht;
            scratch.ct = ct;
        });
    }
}

#[cfg_attr(target_vendor = "apple", allow(dead_code))]
const A_ZP: u8 = 128;

fn try_i8_w(
    layer: &BiLstm,
    x: &[f32],
    _wt: &[f32],
    bias: &[f32],
    xw: &mut [f32],
    seq: usize,
    batch: usize,
    four_h: usize,
    i_sz: usize,
    dir: usize,
) -> bool {
    #[cfg(not(target_vendor = "apple"))]
    {
        if layer.w_i8.is_empty() {
            return false;
        }
        let woff = dir * i_sz * four_h;
        let (ws, wz) = dir_affine(&layer.w_scale, &layer.w_zp, dir, four_h);
        return crate::rten_matmul::gemm_i8_static(
            x,
            layer.x_scale,
            A_ZP,
            &layer.w_i8[woff..woff + i_sz * four_h],
            ws,
            wz,
            Some(bias),
            xw,
            seq * batch,
            four_h,
            i_sz,
            false,
        );
    }
    #[cfg(target_vendor = "apple")]
    {
        let _ = (layer, x, bias, xw, seq, batch, four_h, i_sz, dir);
        false
    }
}

fn try_i8_r(
    layer: &BiLstm,
    ht: &[f32],
    _rt: &[f32],
    gates: &mut [f32],
    batch: usize,
    four_h: usize,
    h: usize,
    dir: usize,
) -> bool {
    #[cfg(not(target_vendor = "apple"))]
    {
        if layer.r_i8.is_empty() {
            return false;
        }
        let roff = dir * h * four_h;
        let (rs, rz) = dir_affine(&layer.r_scale, &layer.r_zp, dir, four_h);
        return crate::rten_matmul::gemm_i8_static(
            ht,
            layer.h_scale,
            A_ZP,
            &layer.r_i8[roff..roff + h * four_h],
            rs,
            rz,
            None,
            gates,
            batch,
            four_h,
            h,
            true,
        );
    }
    #[cfg(target_vendor = "apple")]
    {
        let _ = (layer, ht, gates, batch, four_h, h, dir);
        false
    }
}

#[cfg_attr(target_vendor = "apple", allow(dead_code))]
fn dir_affine<'a>(scale: &'a [f32], zp: &'a [i8], dir: usize, n: usize) -> (&'a [f32], &'a [i8]) {
    let s = if scale.len() == 2 * n {
        &scale[dir * n..(dir + 1) * n]
    } else if scale.len() == 2 {
        &scale[dir..dir + 1]
    } else {
        scale
    };
    let z = if zp.len() == 2 * n {
        &zp[dir * n..(dir + 1) * n]
    } else if zp.len() == 2 {
        &zp[dir..dir + 1]
    } else {
        zp
    };
    (s, z)
}

/// Log-softmax on the last axis of `[n, t, c]`.
pub fn log_softmax_last(x: &mut [f32], n: usize, t: usize, c: usize) {
    debug_assert_eq!(x.len(), n * t * c);
    for nt in 0..(n * t) {
        let row = &mut x[nt * c..nt * c + c];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row.iter() {
            sum += (*v - max).exp();
        }
        let lse = max + sum.ln();
        for v in row.iter_mut() {
            *v -= lse;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(50.0) > 0.99);
        assert!(sigmoid(-50.0) < 0.01);
    }

    /// Monotonic bit map so `|a − b|` in this space counts ulps.
    #[cfg(target_arch = "aarch64")]
    fn ordered_bits(f: f32) -> i64 {
        let i = f.to_bits() as i32 as i64;
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_gates_within_a_few_ulp_of_scalar() {
        use core::arch::aarch64::*;
        // Dense grid over the clamp domain plus sparse large magnitudes for
        // the un-clamped tanh-on-c path.
        let mut vals: Vec<f32> = Vec::new();
        let mut v = -45.0f32;
        while v <= 45.0 {
            vals.push(v);
            v += 0.0007;
        }
        for k in 1..200 {
            vals.push(45.0 * 1.1f32.powi(k));
            vals.push(-45.0 * 1.1f32.powi(k));
        }
        let mut max_sig = 0i64;
        let mut max_tanh = 0i64;
        let mut chunk = vals.chunks_exact(4);
        for c4 in &mut chunk {
            unsafe {
                let x = vld1q_f32(c4.as_ptr());
                let s = neon_gates::sigmoid4(x);
                let t = neon_gates::tanh4(x);
                let mut sv = [0.0f32; 4];
                let mut tv = [0.0f32; 4];
                vst1q_f32(sv.as_mut_ptr(), s);
                vst1q_f32(tv.as_mut_ptr(), t);
                for (lane, &xv) in c4.iter().enumerate() {
                    let ds = (ordered_bits(sv[lane]) - ordered_bits(sigmoid(xv))).abs();
                    let dt = (ordered_bits(tv[lane]) - ordered_bits(xv.tanh())).abs();
                    max_sig = max_sig.max(ds);
                    max_tanh = max_tanh.max(dt);
                }
            }
        }
        for &xv in chunk.remainder() {
            let _ = (sigmoid(xv), xv.tanh());
        }
        eprintln!("max ulp: sigmoid={max_sig} tanh={max_tanh}");
        assert!(max_sig <= 6, "sigmoid ulp {max_sig}");
        assert!(max_tanh <= 6, "tanh ulp {max_tanh}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn apply_gates_vec_close_to_scalar() {
        // h=9 exercises the vector body twice plus the scalar tail.
        let h = 9usize;
        let gates: Vec<f32> = (0..4 * h).map(|i| ((i % 23) as f32) * 0.31 - 3.1).collect();
        let ct0: Vec<f32> = (0..h).map(|i| ((i % 7) as f32) * 0.17 - 0.5).collect();
        let run = |ct_init: &[f32], vec: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut ht = vec![0.0f32; h];
            let mut ct = ct_init.to_vec();
            let mut y = vec![0.0f32; h];
            if vec {
                apply_gates(&gates, &mut ht, &mut ct, &mut y, h);
            } else {
                for hh in 0..h {
                    let it = sigmoid(gates[hh]);
                    let ot = sigmoid(gates[h + hh]);
                    let ft = sigmoid(gates[2 * h + hh]);
                    let c_tilde = gates[3 * h + hh].tanh();
                    let c = ft * ct[hh] + it * c_tilde;
                    ct[hh] = c;
                    let hv = ot * c.tanh();
                    ht[hh] = hv;
                    y[hh] = hv;
                }
            }
            (ht, ct, y)
        };
        let (hv, cv, yv) = run(&ct0, true);
        let (hs, cs, ys) = run(&ct0, false);
        for i in 0..h {
            assert!((hv[i] - hs[i]).abs() < 1e-6, "ht[{i}] {} vs {}", hv[i], hs[i]);
            assert!((cv[i] - cs[i]).abs() < 1e-6, "ct[{i}] {} vs {}", cv[i], cs[i]);
            assert!((yv[i] - ys[i]).abs() < 1e-6, "y[{i}] {} vs {}", yv[i], ys[i]);
        }
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        log_softmax_last(&mut x, 1, 1, 3);
        let p: f32 = x.iter().map(|v| v.exp()).sum();
        assert!((p - 1.0).abs() < 1e-5);
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_direction(
        x: &[f32],
        w: &[f32],
        r: &[f32],
        b: &[f32],
        seq: usize,
        batch: usize,
        h: usize,
        i_sz: usize,
        reverse: bool,
        dir: usize,
        y: &mut [f32],
    ) {
        let mut ht = vec![0.0f32; batch * h];
        let mut ct = vec![0.0f32; batch * h];
        let mut t = if reverse { seq } else { 0 };
        let mut left = seq;
        while left > 0 {
            if reverse {
                t -= 1;
            }
            for b_i in 0..batch {
                let xt = &x[(t * batch + b_i) * i_sz..(t * batch + b_i) * i_sz + i_sz];
                let hh = &ht[b_i * h..b_i * h + h];
                let mut gates = vec![0.0f32; 4 * h];
                for g in 0..4 {
                    for hh_i in 0..h {
                        let mut acc = b[g * h + hh_i] + b[(4 + g) * h + hh_i];
                        let wrow = &w[(g * h + hh_i) * i_sz..(g * h + hh_i) * i_sz + i_sz];
                        let rrow = &r[(g * h + hh_i) * h..(g * h + hh_i) * h + h];
                        for i in 0..i_sz {
                            acc += xt[i] * wrow[i];
                        }
                        for i in 0..h {
                            acc += hh[i] * rrow[i];
                        }
                        gates[g * h + hh_i] = acc;
                    }
                }
                for hh_i in 0..h {
                    let it = sigmoid(gates[hh_i]);
                    let ot = sigmoid(gates[h + hh_i]);
                    let ft = sigmoid(gates[2 * h + hh_i]);
                    let c_tilde = gates[3 * h + hh_i].tanh();
                    let c = ft * ct[b_i * h + hh_i] + it * c_tilde;
                    ct[b_i * h + hh_i] = c;
                    let hv = ot * c.tanh();
                    ht[b_i * h + hh_i] = hv;
                    y[(t * batch + b_i) * 2 * h + dir * h + hh_i] = hv;
                }
            }
            if !reverse {
                t += 1;
            }
            left -= 1;
        }
    }

    #[test]
    fn gemm_lstm_matches_naive() {
        lstm_matches_naive_case(3, 4, 5, 2, 1e-5);
    }

    /// h=8 routes `apply_gates` through the NEON body on aarch64; ulp-level
    /// transcendental drift compounds through the recurrence, hence 1e-4.
    #[test]
    fn gemm_lstm_matches_naive_h8() {
        lstm_matches_naive_case(8, 4, 6, 2, 1e-4);
    }

    fn lstm_matches_naive_case(h: usize, i_sz: usize, seq: usize, batch: usize, tol: f32) {
        let four_h = 4 * h;
        let mut w = vec![0.0f32; 2 * four_h * i_sz];
        let mut r = vec![0.0f32; 2 * four_h * h];
        let mut b = vec![0.0f32; 2 * 8 * h];
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.03 - 0.2;
        }
        for (i, v) in r.iter_mut().enumerate() {
            *v = ((i % 13) as f32) * 0.02 - 0.1;
        }
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.01;
        }
        let x: Vec<f32> = (0..seq * batch * i_sz)
            .map(|i| ((i % 11) as f32) * 0.05 - 0.25)
            .collect();
        let layer = BiLstm::from_onnx(w.clone(), r.clone(), b.clone(), h, i_sz);
        let got = layer.forward(&x, seq, batch);
        let mut want = vec![0.0f32; seq * batch * 2 * h];
        for dir in 0..2 {
            let w_off = dir * four_h * i_sz;
            let r_off = dir * four_h * h;
            let b_off = dir * 8 * h;
            naive_direction(
                &x,
                &w[w_off..w_off + four_h * i_sz],
                &r[r_off..r_off + four_h * h],
                &b[b_off..b_off + 8 * h],
                seq,
                batch,
                h,
                i_sz,
                dir == 1,
                dir,
                &mut want,
            );
        }
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < tol, "i={i} {g} vs {w}");
        }
    }
}
