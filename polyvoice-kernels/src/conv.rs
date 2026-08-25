//! NCHW Conv2d for the shipping WeSpeaker ResNet34 (k=1 or 3, groups=1).

use crate::tensor::Tensor;
use std::cell::RefCell;
use std::sync::OnceLock;

fn skip_qdq() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    // Activation fake-quant does not move Vox-3 DER and costs a full map
    // write per layer. Keep it behind POLYVOICE_QDQ=1 for bit-exact QDQ.
    *S.get_or_init(|| std::env::var_os("POLYVOICE_QDQ").is_none())
}

thread_local! {
    static COL_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    pub oc: usize,
    pub ic: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    pub weight: Vec<f32>, // [oc, ic, k, k] — empty when the INT8 path owns the layer
    pub q_w: Vec<i8>,     // unpadded OIHW, for BNNS dequant fallback
    pub q_scale: Vec<f32>,
    pub bias: Vec<f32>,
    /// Static QDQ activation scale. `None` = full-precision input (stem).
    pub act_scale: Option<f32>,
    pub act_zp: i8,
    pub(crate) q_w_pad: Vec<i8>,
    /// `q_w_pad` retiled as `[oc/4][k_pad/4][4×4]` for the 4×16 SDOT kernel.
    /// Read only by the aarch64 zip kernels.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) q_w_4x4: Vec<i8>,
    pub(crate) w_sum: Vec<i32>,
    pub(crate) k_pad: usize,
    pub(crate) out_scale: Vec<f32>,
    pub(crate) eff_bias: Vec<f32>,
}

impl Conv2d {
    pub fn new(
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        weight: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let pad = if k == 1 { 0 } else { 1 };
        Self {
            oc,
            ic,
            k,
            stride,
            pad,
            weight,
            q_w: Vec::new(),
            q_scale: Vec::new(),
            bias,
            act_scale: None,
            act_zp: 0,
            q_w_pad: Vec::new(),
            q_w_4x4: Vec::new(),
            w_sum: Vec::new(),
            k_pad: 0,
            out_scale: Vec::new(),
            eff_bias: Vec::new(),
        }
    }

    /// Keep INT8 weights packed for the implicit GEMM; BNNS fallback dequants
    /// into a TLS buffer from the unpadded `q_w`.
    pub fn quantized(
        oc: usize,
        ic: usize,
        k: usize,
        stride: usize,
        q_w: Vec<i8>,
        q_scale: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let pad = if k == 1 { 0 } else { 1 };
        let k_raw = ic.saturating_mul(k).saturating_mul(k);
        let k_pad = k_raw.saturating_add(15) & !15;
        let mut q_w_pad = vec![0i8; oc.saturating_mul(k_pad)];
        let mut w_sum = vec![0i32; oc];
        if k_raw > 0 {
            for o in 0..oc {
                let src = &q_w[o * k_raw..o * k_raw + k_raw];
                q_w_pad[o * k_pad..o * k_pad + k_raw].copy_from_slice(src);
                w_sum[o] = src.iter().map(|&v| i32::from(v)).sum();
            }
        }
        let mut weight = Vec::new();
        dequant_oc(&q_w, &q_scale, oc, &mut weight);
        let q_w_4x4 = pack_w_4x4(&q_w_pad, oc, k_pad);
        Self {
            oc,
            ic,
            k,
            stride,
            pad,
            weight,
            q_w,
            q_scale,
            bias,
            act_scale: None,
            act_zp: 0,
            q_w_pad,
            q_w_4x4,
            w_sum,
            k_pad,
            out_scale: Vec::new(),
            eff_bias: Vec::new(),
        }
    }

    pub fn with_input_quant(mut self, scale: f32, zp: i8) -> Self {
        self.act_scale = Some(scale);
        self.act_zp = zp;
        let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
        let z = f32::from(zp);
        self.out_scale = self.q_scale.iter().map(|&ws| s * ws).collect();
        self.eff_bias = (0..self.oc)
            .map(|o| {
                let corr = self.out_scale.get(o).copied().unwrap_or(s)
                    * z
                    * (self.w_sum.get(o).copied().unwrap_or(0) as f32);
                self.bias.get(o).copied().unwrap_or(0.0) - corr
            })
            .collect();
        self
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_act(x, false)
    }

    /// Conv + ReLU. On Apple, BNNS fuses the activation into the filter.
    #[cfg(test)]
    pub fn forward_relu(&self, x: &Tensor) -> Tensor {
        self.forward_act(x, true)
    }

    fn forward_act(&self, x: &Tensor, relu: bool) -> Tensor {
        let (oh, ow) = self.out_hw(x);
        let mut y = Tensor::uninit(x.n, self.oc, oh, ow);
        self.forward_into(x, &mut y, relu);
        y
    }

    pub(crate) fn out_hw_dims(&self, h: usize, w: usize) -> (usize, usize) {
        let oh = (h + 2 * self.pad - self.k) / self.stride + 1;
        let ow = (w + 2 * self.pad - self.k) / self.stride + 1;
        (oh, ow)
    }

    fn out_hw(&self, x: &Tensor) -> (usize, usize) {
        self.out_hw_dims(x.h, x.w)
    }

    /// Write `conv(x)` into `y`, growing `y` if needed.
    pub(crate) fn forward_into(&self, x: &Tensor, y: &mut Tensor, relu: bool) {
        let (oh, ow) = self.out_hw(x);
        y.reuse_uninit(x.n, self.oc, oh, ow);
        if crate::conv_i8::try_conv(self, x, y, relu) {
            return;
        }
        let qbuf;
        let x = if let Some(scale) = self.act_scale {
            if skip_qdq() {
                x
            } else {
                qbuf = fake_quant_tensor(x, scale, self.act_zp);
                &qbuf
            }
        } else {
            x
        };
        self.with_weight_f32(|w| self.forward_with_weight(x, w, y, oh, ow, relu));
    }

    /// Like `forward_into`, but fake-quant may overwrite `x`.
    pub(crate) fn forward_into_owned(&self, x: &mut Tensor, y: &mut Tensor, relu: bool) {
        let (oh, ow) = self.out_hw(x);
        y.reuse_uninit(x.n, self.oc, oh, ow);
        if crate::conv_i8::try_conv(self, x, y, relu) {
            return;
        }
        if let Some(scale) = self.act_scale
            && !skip_qdq()
        {
            crate::tensor::fake_quant_inplace(x, scale, self.act_zp);
        }
        self.with_weight_f32(|w| self.forward_with_weight(x, w, y, oh, ow, relu));
    }

    /// Identity of the live weight buffer — BNNS filter cache key (Apple only).
    #[cfg(target_vendor = "apple")]
    fn weight_id(&self) -> usize {
        if !self.q_w.is_empty() {
            self.q_w.as_ptr() as usize
        } else {
            self.weight.as_ptr() as usize
        }
    }

    fn with_weight_f32(&self, f: impl FnOnce(&[f32])) {
        if !self.weight.is_empty() {
            f(&self.weight);
            return;
        }
        W_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            dequant_oc(&self.q_w, &self.q_scale, self.oc, &mut buf);
            f(&buf);
        });
    }

    fn forward_with_weight(
        &self,
        x: &Tensor,
        w: &[f32],
        y: &mut Tensor,
        oh: usize,
        ow: usize,
        relu: bool,
    ) {
        #[cfg(target_vendor = "apple")]
        {
            if crate::bnns::try_conv2d(
                w,
                &self.bias,
                self.oc,
                self.ic,
                self.k,
                self.stride,
                self.pad,
                relu,
                self.weight_id(),
                x,
                y,
            ) {
                return;
            }
        }
        let spatial = oh * ow;
        let k_col = self.ic * self.k * self.k;
        if x.n == 1 {
            let yrow = &mut y.data[..self.oc * spatial];
            if self.k == 1 && self.pad == 0 && self.stride == 1 {
                crate::gemm::gemm_bias_row(
                    w,
                    &x.data[..k_col * spatial],
                    &self.bias,
                    yrow,
                    self.oc,
                    spatial,
                    k_col,
                );
            } else {
                with_col_buf(k_col * spatial, |col| {
                    if self.k == 3 && self.pad == 1 && self.stride == 1 {
                        im2col_3x3_s1_p1(x, 0, col, spatial, 0);
                    } else {
                        col.fill(0.0);
                        im2col_into(x, 0, self.k, self.stride, self.pad, oh, ow, col, spatial, 0);
                    }
                    crate::gemm::gemm_bias_row(w, col, &self.bias, yrow, self.oc, spatial, k_col);
                });
            }
            if relu {
                crate::tensor::relu_inplace(y);
            }
            return;
        }
        let n_img = x.n;
        let pack = n_img * spatial;
        with_col_buf(k_col * pack + self.oc * pack, |buf| {
            let (col, gemm_out) = buf.split_at_mut(k_col * pack);
            if self.k == 1 && self.pad == 0 && self.stride == 1 {
                pack_nchw_to_ckn(x, col, spatial);
            } else {
                col.fill(0.0);
                for ni in 0..n_img {
                    let off = ni * spatial;
                    if self.k == 3 && self.pad == 1 && self.stride == 1 {
                        im2col_3x3_s1_p1(x, ni, col, pack, off);
                    } else {
                        im2col_into(x, ni, self.k, self.stride, self.pad, oh, ow, col, pack, off);
                    }
                }
            }
            crate::gemm::gemm_bias_row(w, col, &self.bias, gemm_out, self.oc, pack, k_col);
            unpack_ckn_to_nchw(gemm_out, y, spatial);
        });
        if relu {
            crate::tensor::relu_inplace(y);
        }
    }
}

thread_local! {
    static W_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn pack_w_4x4(q_w_pad: &[i8], oc: usize, k_pad: usize) -> Vec<i8> {
    if oc < 4 || !oc.is_multiple_of(4) || !k_pad.is_multiple_of(4) || q_w_pad.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0i8; oc * k_pad];
    let tiles_m = oc / 4;
    let tiles_k = k_pad / 4;
    for tm in 0..tiles_m {
        for tk in 0..tiles_k {
            let dst = (tm * tiles_k + tk) * 16;
            for r in 0..4 {
                let src = (tm * 4 + r) * k_pad + tk * 4;
                out[dst + r * 4..dst + r * 4 + 4].copy_from_slice(&q_w_pad[src..src + 4]);
            }
        }
    }
    out
}

fn dequant_oc(q: &[i8], scale: &[f32], oc: usize, dst: &mut Vec<f32>) {
    dst.resize(q.len(), 0.0);
    if q.is_empty() {
        return;
    }
    let inner = q.len() / oc.max(1);
    for o in 0..oc {
        let s = scale.get(o).copied().unwrap_or(1.0);
        let off = o * inner;
        for i in 0..inner {
            dst[off + i] = f32::from(q[off + i]) * s;
        }
    }
}

/// NCHW `[n,c,h,w]` → `[c, n*h*w]` for 1×1 conv.
fn pack_nchw_to_ckn(x: &Tensor, dst: &mut [f32], spatial: usize) {
    let pack = x.n * spatial;
    debug_assert_eq!(dst.len(), x.c * pack);
    for ni in 0..x.n {
        for ic in 0..x.c {
            let src = ni * x.c * spatial + ic * spatial;
            let off = ic * pack + ni * spatial;
            dst[off..off + spatial].copy_from_slice(&x.data[src..src + spatial]);
        }
    }
}

/// `[c, n*h*w]` → NCHW `[n,c,h,w]`.
fn unpack_ckn_to_nchw(src: &[f32], y: &mut Tensor, spatial: usize) {
    let pack = y.n * spatial;
    debug_assert_eq!(src.len(), y.c * pack);
    for ni in 0..y.n {
        for oc in 0..y.c {
            let off = oc * pack + ni * spatial;
            let dst = ni * y.c * spatial + oc * spatial;
            y.data[dst..dst + spatial].copy_from_slice(&src[off..off + spatial]);
        }
    }
}

fn fake_quant_tensor(x: &Tensor, scale: f32, zp: i8) -> Tensor {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    let mut y = Tensor::uninit(x.n, x.c, x.h, x.w);
    for (dst, &src) in y.data.iter_mut().zip(x.data.iter()) {
        let q = (src / s).round() + z;
        let q = q.clamp(-128.0, 127.0);
        *dst = (q - z) * s;
    }
    y
}

fn with_col_buf(len: usize, f: impl FnOnce(&mut [f32])) {
    COL_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
        f(&mut buf[..len]);
    });
}

/// Identity 3×3, pad=1, stride=1: row copies, no per-pixel bounds checks.
/// `col` is `[k_col, pack]`; this image occupies columns `[off, off+spatial)`.
fn im2col_3x3_s1_p1(x: &Tensor, n: usize, col: &mut [f32], pack: usize, off: usize) {
    let h = x.h;
    let w = x.w;
    let spatial = h.saturating_mul(w);
    debug_assert!(off + spatial <= pack);
    debug_assert_eq!(col.len(), x.c.saturating_mul(9).saturating_mul(pack));
    if h == 0 || w == 0 {
        return;
    }
    let base = n * x.c * h * w;
    for ic in 0..x.c {
        let plane = &x.data[base + ic * h * w..base + (ic + 1) * h * w];
        for kh in 0..3 {
            for kw in 0..3 {
                let row = (ic * 9 + kh * 3 + kw) * pack + off;
                let dst = &mut col[row..row + spatial];
                for oy in 0..h {
                    let drow = &mut dst[oy * w..oy * w + w];
                    let iy = oy as isize + kh as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        drow.fill(0.0);
                        continue;
                    }
                    let srow = &plane[(iy as usize) * w..(iy as usize) * w + w];
                    match kw {
                        0 => {
                            drow[0] = 0.0;
                            drow[1..].copy_from_slice(&srow[..w - 1]);
                        }
                        1 => drow.copy_from_slice(srow),
                        _ => {
                            drow[..w - 1].copy_from_slice(&srow[1..]);
                            drow[w - 1] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// `Col[ic*k*k, oh*ow]` so `Y[oc, hw] = W[oc, ic*k*k] @ Col`.
#[allow(clippy::too_many_arguments)]
fn im2col_into(
    x: &Tensor,
    n: usize,
    k: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    col: &mut [f32],
    pack: usize,
    off: usize,
) {
    let spatial = oh * ow;
    let inner = k * k;
    debug_assert!(off + spatial <= pack);
    debug_assert_eq!(col.len(), x.c * inner * pack);
    let p = pad as isize;
    let s = stride as isize;
    let base = n * x.c * x.h * x.w;
    let (xh, xw) = (x.h, x.w);
    for ic in 0..x.c {
        let plane = &x.data[base + ic * xh * xw..base + (ic + 1) * xh * xw];
        for kh in 0..k {
            for kw in 0..k {
                let row = (ic * inner + kh * k + kw) * pack + off;
                let dst = &mut col[row..row + spatial];
                let ih0 = kh as isize - p;
                let iw0 = kw as isize - p;
                let mut idx = 0;
                for oh_i in 0..oh {
                    let ih = ih0 + oh_i as isize * s;
                    if ih < 0 || ih >= xh as isize {
                        idx += ow;
                        continue;
                    }
                    let src_row = &plane[(ih as usize) * xw..(ih as usize) * xw + xw];
                    for ow_i in 0..ow {
                        let iw = iw0 + ow_i as isize * s;
                        if iw >= 0 && iw < xw as isize {
                            dst[idx] = src_row[iw as usize];
                        }
                        idx += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn conv1x1_is_per_pixel_gemm() {
        let mut x = Tensor::zeros(1, 2, 1, 1);
        x.set(0, 0, 0, 0, 1.0);
        x.set(0, 1, 0, 0, 2.0);
        // oc=1, ic=2, k=1: w = [3, 4]
        let conv = Conv2d::new(1, 2, 1, 1, vec![3.0, 4.0], vec![0.5]);
        let y = conv.forward(&x);
        assert_eq!(y.c, 1);
        assert_eq!(y.h, 1);
        assert_eq!(y.w, 1);
        assert!((y.get(0, 0, 0, 0) - 11.5).abs() < 1e-6);
    }

    #[test]
    fn conv3x3_s1_p1_preserves_spatial() {
        let x = Tensor::zeros(1, 1, 4, 5);
        let w = vec![0.0; 9];
        let conv = Conv2d::new(1, 1, 3, 1, w, vec![1.0]);
        let y = conv.forward(&x);
        assert_eq!((y.h, y.w), (4, 5));
        assert!(y.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn conv3x3_s2_p1_halves_spatial() {
        let x = Tensor::zeros(1, 1, 8, 10);
        let conv = Conv2d::new(1, 1, 3, 2, vec![0.0; 9], vec![0.0]);
        let y = conv.forward(&x);
        assert_eq!((y.h, y.w), (4, 5));
    }

    fn naive_conv(conv: &Conv2d, x: &Tensor) -> Tensor {
        let oh = (x.h + 2 * conv.pad - conv.k) / conv.stride + 1;
        let ow = (x.w + 2 * conv.pad - conv.k) / conv.stride + 1;
        let mut y = Tensor::zeros(x.n, conv.oc, oh, ow);
        let p = conv.pad as isize;
        for n in 0..x.n {
            for oc in 0..conv.oc {
                let b = conv.bias[oc];
                for oh_i in 0..oh {
                    let ih0 = oh_i as isize * conv.stride as isize - p;
                    for ow_i in 0..ow {
                        let iw0 = ow_i as isize * conv.stride as isize - p;
                        let mut acc = b;
                        for ic in 0..conv.ic {
                            for kh in 0..conv.k {
                                let ih = ih0 + kh as isize;
                                if ih < 0 || ih >= x.h as isize {
                                    continue;
                                }
                                for kw in 0..conv.k {
                                    let iw = iw0 + kw as isize;
                                    if iw < 0 || iw >= x.w as isize {
                                        continue;
                                    }
                                    let wi = ((oc * conv.ic + ic) * conv.k + kh) * conv.k + kw;
                                    acc += conv.weight[wi] * x.get(n, ic, ih as usize, iw as usize);
                                }
                            }
                        }
                        y.set(n, oc, oh_i, ow_i, acc);
                    }
                }
            }
        }
        y
    }

    fn assert_close(got: &Tensor, want: &Tensor) {
        assert_eq!(got.data.len(), want.data.len());
        for (i, (g, w)) in got.data.iter().zip(want.data.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "i={i} {g} vs {w}");
        }
    }

    fn assert_close_tile(got: &Tensor, want: &Tensor) {
        assert_eq!(got.data.len(), want.data.len());
        let mut max = 0.0f32;
        for (g, w) in got.data.iter().zip(want.data.iter()) {
            max = max.max((g - w).abs());
        }
        assert!(max < 2e-3, "tiled conv maxabs={max}");
    }

    #[test]
    fn implicit_i8_3x3_tracks_fakequant_float() {
        let oc = 4;
        let ic = 32;
        let mut x = Tensor::zeros(1, ic, 6, 20);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic * 9];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                w_f[o * ic * 9 + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic * 9 {
                let q = (w_f[o * ic * 9 + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic * 9 + i] = q as i8;
                w_f[o * ic * 9 + i] = q * s;
            }
        }
        let bias = vec![0.05, -0.1, 0.0, 0.2];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, bias.clone())
            .with_input_quant(0.04, -128);
        let mut refc = Conv2d::new(oc, ic, 3, 1, w_f, bias);
        refc = refc.with_input_quant(0.04, -128);
        // Reference: fake-quant then float conv (QDQ semantics).
        let mut xq = x.clone();
        crate::tensor::fake_quant_inplace(&mut xq, 0.04, -128);
        refc.act_scale = None;
        let want = naive_conv(&refc, &xq);
        let got = conv.forward(&x);
        assert_eq!(got.data.len(), want.data.len());
        let mut max = 0.0f32;
        let mut arg = 0usize;
        for (i, (g, w)) in got.data.iter().zip(want.data.iter()).enumerate() {
            let d = (g - w).abs();
            if d > max {
                max = d;
                arg = i;
            }
        }
        assert!(max < 0.05, "i8 vs fakequant-float maxabs={max} at {arg}");
        crate::set_intra_threads(4);
        let got_p = conv.forward(&x);
        crate::set_intra_threads(1);
        for (i, (g, q)) in got.data.iter().zip(got_p.data.iter()).enumerate() {
            assert!((g - q).abs() < 1e-5, "intra-op mismatch i={i} {g} vs {q}");
        }
        let b = conv.forward(&x);
        let n2 = {
            let mut xx = Tensor::zeros(2, ic, 6, 20);
            xx.data[..x.data.len()].copy_from_slice(&x.data);
            xx.data[x.data.len()..].copy_from_slice(&x.data);
            conv.forward(&xx)
        };
        for i in 0..b.data.len() {
            let d = (n2.data[i] - b.data[i]).abs();
            assert!(d < 1e-5, "n=2[0] vs n=1 {d}");
            let d = (n2.data[b.data.len() + i] - b.data[i]).abs();
            assert!(d < 1e-5, "n=2[1] vs n=1 {d}");
        }
        crate::conv_i8::force_i8(false);
    }

    #[test]
    fn implicit_i8_3x3_wide_row_tracks_fakequant_float() {
        // WAVE=32 tiles * PN=32 = 1024 px; leftover 4 must not drop later waves.
        let oc = 4;
        let ic = 8;
        let h = 3;
        let w = 1100;
        let mut x = Tensor::zeros(1, ic, h, w);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic * 9];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                w_f[o * ic * 9 + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic * 9 {
                let q = (w_f[o * ic * 9 + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic * 9 + i] = q as i8;
                w_f[o * ic * 9 + i] = q * s;
            }
        }
        let bias = vec![0.05, -0.1, 0.0, 0.2];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, bias.clone())
            .with_input_quant(0.04, -128);
        let mut refc = Conv2d::new(oc, ic, 3, 1, w_f, bias);
        refc = refc.with_input_quant(0.04, -128);
        let mut xq = x.clone();
        crate::tensor::fake_quant_inplace(&mut xq, 0.04, -128);
        refc.act_scale = None;
        let want = naive_conv(&refc, &xq);
        let got = conv.forward(&x);
        assert_eq!(got.data.len(), want.data.len());
        let mut max = 0.0f32;
        let mut arg = 0usize;
        for (i, (g, wv)) in got.data.iter().zip(want.data.iter()).enumerate() {
            let d = (g - wv).abs();
            if d > max {
                max = d;
                arg = i;
            }
        }
        assert!(
            max < 0.05,
            "wide-row i8 vs fakequant-float maxabs={max} at {arg}"
        );
        crate::conv_i8::force_i8(false);
    }

    #[test]
    fn implicit_i8_3x3_to_i8_intra_matches_serial() {
        // Identity conv1 writes i8 dest. f32 intra tests do not cover this.
        let oc = 128;
        let ic = 16;
        let h = 8;
        let w = 1100;
        let mut x = Tensor::zeros(1, ic, h, w);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic * 9 {
                let q = (((o * 17 + i) as f32) * 0.02 - 0.3) / s;
                q_w[o * ic * 9 + i] = q.round().clamp(-127.0, 127.0) as i8;
            }
        }
        let bias = vec![0.01f32; oc];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, bias).with_input_quant(0.04, -128);
        let (oh, ow) = conv.out_hw_dims(h, w);
        let need = oc * oh * ow;
        let mut serial = vec![0i8; need];
        let mut par = vec![0i8; need];
        crate::set_intra_threads(1);
        assert!(crate::conv_i8::try_conv_to_i8(
            &conv,
            &x,
            &mut serial,
            true,
            0.05,
            -128
        ));
        crate::set_intra_threads(4);
        assert!(crate::conv_i8::try_conv_to_i8(
            &conv, &x, &mut par, true, 0.05, -128
        ));
        crate::set_intra_threads(1);
        crate::conv_i8::force_i8(false);
        let mut nmis = 0usize;
        let mut arg = 0usize;
        for (i, (&s, &q)) in serial.iter().zip(par.iter()).enumerate() {
            if s != q {
                nmis += 1;
                if nmis == 1 {
                    arg = i;
                }
            }
        }
        assert!(
            nmis == 0,
            "to_i8 intra mismatch nmis={nmis} first={arg} serial={} par={}",
            serial[arg],
            par[arg]
        );
    }

    #[test]
    fn implicit_i8_3x3_oc_split_matches_serial() {
        // oc≥64 takes the layer-level output-channel split.
        let oc = 64;
        let ic = 8;
        let mut x = Tensor::zeros(1, ic, 8, 40);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic * 9];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 17 + i) as f32) * 0.02 - 0.3;
                w_f[o * ic * 9 + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic * 9 {
                let q = (w_f[o * ic * 9 + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic * 9 + i] = q as i8;
                w_f[o * ic * 9 + i] = q * s;
            }
        }
        let bias = vec![0.01f32; oc];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, bias).with_input_quant(0.04, -128);
        crate::set_intra_threads(1);
        let serial = conv.forward(&x);
        crate::set_intra_threads(4);
        let par = conv.forward(&x);
        crate::set_intra_threads(1);
        crate::conv_i8::force_i8(false);
        assert_eq!(serial.data.len(), par.data.len());
        for (i, (s, q)) in serial.data.iter().zip(par.data.iter()).enumerate() {
            assert!((s - q).abs() < 1e-5, "oc-split mismatch i={i} {s} vs {q}");
        }
    }

    #[test]
    fn implicit_i8_3x3_s2_tracks_fakequant_float() {
        let oc = 4;
        let ic = 16;
        let mut x = Tensor::zeros(1, ic, 12, 20);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.35;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic * 9];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 11 + i) as f32) * 0.02 - 0.25;
                w_f[o * ic * 9 + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic * 9 {
                let q = (w_f[o * ic * 9 + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic * 9 + i] = q as i8;
                w_f[o * ic * 9 + i] = q * s;
            }
        }
        let bias = vec![0.04, -0.08, 0.0, 0.15];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 2, q_w, q_scale, bias.clone())
            .with_input_quant(0.04, -128);
        let mut refc = Conv2d::new(oc, ic, 3, 2, w_f, bias);
        refc = refc.with_input_quant(0.04, -128);
        let mut xq = x.clone();
        crate::tensor::fake_quant_inplace(&mut xq, 0.04, -128);
        refc.act_scale = None;
        let want = naive_conv(&refc, &xq);
        let got = conv.forward(&x);
        let mut max = 0.0f32;
        for (g, w) in got.data.iter().zip(want.data.iter()) {
            max = max.max((g - w).abs());
        }
        assert!(max < 0.05, "s2 i8 vs fakequant-float maxabs={max}");
        crate::conv_i8::force_i8(false);
    }

    #[test]
    fn implicit_i8_1x1_tracks_fakequant_float() {
        let oc = 4;
        let ic = 32;
        let mut x = Tensor::zeros(1, ic, 5, 9);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let mut q_w = vec![0i8; oc * ic];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic {
                let v = ((o * 13 + i) as f32) * 0.03 - 0.2;
                w_f[o * ic + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic {
                let q = (w_f[o * ic + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic + i] = q as i8;
                w_f[o * ic + i] = q * s;
            }
        }
        let bias = vec![0.05, -0.1, 0.0, 0.2];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 1, 1, q_w, q_scale, bias.clone())
            .with_input_quant(0.04, -128);
        let mut refc = Conv2d::new(oc, ic, 1, 1, w_f, bias);
        refc = refc.with_input_quant(0.04, -128);
        let mut xq = x.clone();
        crate::tensor::fake_quant_inplace(&mut xq, 0.04, -128);
        refc.act_scale = None;
        let want = naive_conv(&refc, &xq);
        let got = conv.forward(&x);
        let mut max = 0.0f32;
        for (g, w) in got.data.iter().zip(want.data.iter()) {
            max = max.max((g - w).abs());
        }
        assert!(max < 0.05, "1x1 i8 vs fakequant-float maxabs={max}");
        let mut x40 = Tensor::zeros(1, ic, 5, 40);
        for i in 0..x40.data.len() {
            x40.data[i] = (i as f32) * 0.01 - 0.4;
        }
        let b = conv.forward(&x40);
        let n2 = {
            let mut xx = Tensor::zeros(2, ic, 5, 40);
            xx.data[..x40.data.len()].copy_from_slice(&x40.data);
            xx.data[x40.data.len()..].copy_from_slice(&x40.data);
            conv.forward(&xx)
        };
        for i in 0..b.data.len() {
            let d0 = (n2.data[i] - b.data[i]).abs();
            let d1 = (n2.data[b.data.len() + i] - b.data[i]).abs();
            assert!(d0 < 1e-5, "1x1 n=2[0] vs n=1 {d0}");
            assert!(d1 < 1e-5, "1x1 n=2[1] vs n=1 {d1}");
        }
        crate::conv_i8::force_i8(false);
    }

    #[test]
    fn implicit_i8_1x1_s2_tracks_fakequant_float() {
        let oc = 4;
        let ic = 32;
        let mut x = Tensor::zeros(1, ic, 8, 40);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.35;
        }
        let mut q_w = vec![0i8; oc * ic];
        let mut q_scale = vec![0.0f32; oc];
        let mut w_f = vec![0.0f32; oc * ic];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic {
                let v = ((o * 13 + i) as f32) * 0.03 - 0.2;
                w_f[o * ic + i] = v;
                m = m.max(v.abs());
            }
            let s = (m / 127.0).max(1e-8);
            q_scale[o] = s;
            for i in 0..ic {
                let q = (w_f[o * ic + i] / s).round().clamp(-127.0, 127.0);
                q_w[o * ic + i] = q as i8;
                w_f[o * ic + i] = q * s;
            }
        }
        let bias = vec![0.05, -0.1, 0.0, 0.2];
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 1, 2, q_w, q_scale, bias.clone())
            .with_input_quant(0.04, -128);
        let mut refc = Conv2d::new(oc, ic, 1, 2, w_f, bias);
        refc = refc.with_input_quant(0.04, -128);
        let mut xq = x.clone();
        crate::tensor::fake_quant_inplace(&mut xq, 0.04, -128);
        refc.act_scale = None;
        let want = naive_conv(&refc, &xq);
        let got = conv.forward(&x);
        let mut max = 0.0f32;
        for (g, w) in got.data.iter().zip(want.data.iter()) {
            max = max.max((g - w).abs());
        }
        assert!(max < 0.05, "1x1 s2 i8 vs fakequant-float maxabs={max}");
        crate::conv_i8::force_i8(false);
    }

    #[test]
    fn seeded_try_conv_matches_quantize() {
        let oc = 4;
        let ic = 16;
        let mut x = Tensor::zeros(1, ic, 6, 12);
        let mut z = Tensor::zeros(1, ic, 6, 12);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.02 - 0.9;
            z.data[i] = (i as f32) * -0.01 + 0.2;
        }
        let mut q_w = vec![0i8; oc * ic * 9];
        let mut q_scale = vec![0.0f32; oc];
        for o in 0..oc {
            let mut m = 0.0f32;
            for i in 0..ic * 9 {
                let v = ((o * 7 + i) as f32) * 0.02 - 0.2;
                m = m.max(v.abs());
                q_w[o * ic * 9 + i] = (v / 0.01).round().clamp(-127.0, 127.0) as i8;
            }
            q_scale[o] = (m / 127.0).max(1e-8);
        }
        crate::conv_i8::force_i8(true);
        let conv = Conv2d::quantized(oc, ic, 3, 1, q_w, q_scale, vec![0.0; oc])
            .with_input_quant(0.04, -128);
        let mut x_ref = x.clone();
        crate::tensor::add_relu_inplace(&mut x_ref, &z);
        let want = conv.forward(&x_ref);
        crate::conv_i8::seed_xq_add_relu(&mut x, &z, 0.04, -128);
        let got = conv.forward(&x);
        crate::conv_i8::force_i8(false);
        assert_eq!(got.data.len(), want.data.len());
        for (i, (g, w)) in got.data.iter().zip(want.data.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "seeded mismatch i={i} {g} vs {w}");
        }
    }

    #[test]
    fn gemm_conv_matches_naive_3x3() {
        let mut x = Tensor::zeros(1, 2, 5, 6);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.07 - 0.4;
        }
        let w: Vec<f32> = (0..3 * 2 * 9).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let conv = Conv2d::new(3, 2, 3, 1, w, vec![0.1, -0.2, 0.05]);
        assert_close(&conv.forward(&x), &naive_conv(&conv, &x));
        let mut want_relu = naive_conv(&conv, &x);
        crate::tensor::relu_inplace(&mut want_relu);
        assert_close(&conv.forward_relu(&x), &want_relu);
    }

    #[test]
    fn conv3x3_wide_matches_naive() {
        // Wider than the BNNS time tile (128) so Apple takes the tiled path.
        let mut x = Tensor::zeros(1, 2, 8, 200);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.01 - 0.3;
        }
        let w: Vec<f32> = (0..3 * 2 * 9).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let conv = Conv2d::new(3, 2, 3, 1, w, vec![0.05, -0.1, 0.02]);
        assert_close_tile(&conv.forward(&x), &naive_conv(&conv, &x));
        let mut x2 = Tensor::zeros(2, 2, 8, 200);
        for i in 0..x2.data.len() {
            x2.data[i] = (i as f32) * 0.01 - 0.25;
        }
        assert_close_tile(&conv.forward(&x2), &naive_conv(&conv, &x2));
    }

    #[test]
    fn conv3x3_s2_and_n2_match_naive() {
        let mut x = Tensor::zeros(2, 2, 8, 10);
        for i in 0..x.data.len() {
            x.data[i] = (i as f32) * 0.05 - 0.3;
        }
        let w: Vec<f32> = (0..4 * 2 * 9).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let conv = Conv2d::new(4, 2, 3, 2, w, vec![0.0, 0.1, -0.1, 0.2]);
        assert_close(&conv.forward(&x), &naive_conv(&conv, &x));
    }
}
