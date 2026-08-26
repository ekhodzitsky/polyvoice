//! `[N, C, L]` f32 sequence used by SincNet / powerset.

use std::cell::RefCell;

thread_local! {
    static IM2COL: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Debug)]
pub struct Seq1d {
    pub n: usize,
    pub c: usize,
    pub l: usize,
    pub data: Vec<f32>,
}

impl Seq1d {
    pub fn zeros(n: usize, c: usize, l: usize) -> Self {
        Self {
            n,
            c,
            l,
            data: vec![0.0; n.saturating_mul(c).saturating_mul(l)],
        }
    }

    #[inline]
    #[cfg(test)]
    pub fn idx(&self, n: usize, c: usize, l: usize) -> usize {
        (n * self.c + c) * self.l + l
    }

    #[inline]
    #[cfg(test)]
    pub fn get(&self, n: usize, c: usize, l: usize) -> f32 {
        self.data[self.idx(n, c, l)]
    }

    #[inline]
    #[cfg(test)]
    pub fn set(&mut self, n: usize, c: usize, l: usize, v: f32) {
        let i = self.idx(n, c, l);
        self.data[i] = v;
    }
}

pub fn leaky_relu_inplace(x: &mut Seq1d, alpha: f32) {
    leaky_relu_slice(&mut x.data, alpha);
}

/// LeakyReLU over a flat slice (Linear after LSTM uses this).
pub fn leaky_relu_slice(x: &mut [f32], alpha: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        neon_leaky(x, alpha);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for v in x {
        if *v < 0.0 {
            *v *= alpha;
        }
    }
}

/// InstanceNorm over the last axis (ONNX `axes=[2]`, population variance).
pub fn instance_norm_inplace(x: &mut Seq1d, scale: &[f32], bias: &[f32], eps: f32) {
    debug_assert_eq!(scale.len(), x.c);
    debug_assert_eq!(bias.len(), x.c);
    let lf = x.l.max(1) as f32;
    let span = x.l;
    for n in 0..x.n {
        for c in 0..x.c {
            let off = (n * x.c + c) * span;
            let sl = &mut x.data[off..off + span];
            #[cfg(target_arch = "aarch64")]
            neon_inorm(sl, lf, scale[c], bias[c], eps);
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut mean = 0.0;
                for &v in sl.iter() {
                    mean += v;
                }
                mean /= lf;
                let mut var = 0.0;
                for &v in sl.iter() {
                    let d = v - mean;
                    var += d * d;
                }
                var /= lf;
                let inv = 1.0 / (var + eps).sqrt();
                let s = scale[c];
                let b = bias[c];
                for v in sl.iter_mut() {
                    *v = (*v - mean) * inv * s + b;
                }
            }
        }
    }
}

/// Conv1d, NCL, no groups. `pads` are valid (0). Optional bias.
pub fn conv1d(
    x: &Seq1d,
    weight: &[f32], // [oc, ic, k]
    bias: Option<&[f32]>,
    oc: usize,
    k: usize,
    stride: usize,
) -> Seq1d {
    if x.c == 1 {
        return conv1d_ic1(x, weight, bias, oc, k, stride);
    }
    if k == 5 {
        return conv1d_k5(x, weight, bias, oc, stride);
    }
    conv1d_generic(x, weight, bias, oc, k, stride)
}

fn conv1d_generic(
    x: &Seq1d,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc: usize,
    k: usize,
    stride: usize,
) -> Seq1d {
    let ol = (x.l - k) / stride + 1;
    let mut y = Seq1d::zeros(x.n, oc, ol);
    for n in 0..x.n {
        for o in 0..oc {
            let b = bias.map(|bb| bb[o]).unwrap_or(0.0);
            let wbase = o * x.c * k;
            let ybase = (n * oc + o) * ol;
            for ol_i in 0..ol {
                let i0 = ol_i * stride;
                let mut acc = b;
                for ic in 0..x.c {
                    let xbase = (n * x.c + ic) * x.l + i0;
                    let ww = wbase + ic * k;
                    acc += crate::gemm::dot(&weight[ww..ww + k], &x.data[xbase..xbase + k]);
                }
                y.data[ybase + ol_i] = acc;
            }
        }
    }
    y
}

/// SincNet first conv: 80 filters, ic=1, k=251, stride=10.
/// Im2col + GEMM so Apple AMX / the blocked kernel see a fat matmul.
fn conv1d_ic1(
    x: &Seq1d,
    weight: &[f32],
    bias: Option<&[f32]>,
    oc: usize,
    k: usize,
    stride: usize,
) -> Seq1d {
    debug_assert_eq!(x.c, 1);
    let ol = (x.l - k) / stride + 1;
    let mut y = Seq1d::zeros(x.n, oc, ol);
    let zeros;
    let bias_row: &[f32] = match bias {
        Some(b) => b,
        None => {
            zeros = vec![0.0f32; oc];
            &zeros
        }
    };
    IM2COL.with(|cell| {
        let mut col = cell.borrow_mut();
        col.resize(k * ol, 0.0);
        for n in 0..x.n {
            let xrow = &x.data[n * x.l..n * x.l + x.l];
            for kk in 0..k {
                let dst = &mut col[kk * ol..(kk + 1) * ol];
                let mut i0 = kk;
                for slot in dst.iter_mut() {
                    *slot = xrow[i0];
                    i0 += stride;
                }
            }
            let yoff = n * oc * ol;
            crate::gemm::gemm_bias_row(
                weight,
                &col,
                bias_row,
                &mut y.data[yoff..yoff + oc * ol],
                oc,
                ol,
                k,
            );
        }
    });
    y
}

/// Post-sinc 5-tap convs (ic=80/60). Im2col + GEMM.
fn conv1d_k5(x: &Seq1d, weight: &[f32], bias: Option<&[f32]>, oc: usize, stride: usize) -> Seq1d {
    const K: usize = 5;
    let ol = (x.l - K) / stride + 1;
    let mut y = Seq1d::zeros(x.n, oc, ol);
    let ic_n = x.c;
    let k_col = ic_n * K;
    let zeros;
    let bias_row: &[f32] = match bias {
        Some(b) => b,
        None => {
            zeros = vec![0.0f32; oc];
            &zeros
        }
    };
    IM2COL.with(|cell| {
        let mut col = cell.borrow_mut();
        col.resize(k_col * ol, 0.0);
        for n in 0..x.n {
            for ic in 0..ic_n {
                let src = &x.data[(n * ic_n + ic) * x.l..(n * ic_n + ic) * x.l + x.l];
                for kk in 0..K {
                    let dst = &mut col[(ic * K + kk) * ol..(ic * K + kk + 1) * ol];
                    let mut i0 = kk;
                    for slot in dst.iter_mut() {
                        *slot = src[i0];
                        i0 += stride;
                    }
                }
            }
            let yoff = n * oc * ol;
            crate::gemm::gemm_bias_row(
                weight,
                &col,
                bias_row,
                &mut y.data[yoff..yoff + oc * ol],
                oc,
                ol,
                k_col,
            );
        }
    });
    y
}

/// MaxPool1d, valid, ceil_mode=0.
pub fn max_pool1d(x: &Seq1d, k: usize, stride: usize) -> Seq1d {
    let ol = (x.l - k) / stride + 1;
    let mut y = Seq1d::zeros(x.n, x.c, ol);
    let span = x.l;
    for n in 0..x.n {
        for c in 0..x.c {
            let src = &x.data[(n * x.c + c) * span..(n * x.c + c) * span + span];
            let dst = &mut y.data[(n * x.c + c) * ol..(n * x.c + c) * ol + ol];
            if k == 3 && stride == 3 {
                let mut ol_i = 0;
                while ol_i + 4 <= ol {
                    let i0 = ol_i * 3;
                    dst[ol_i] = src[i0].max(src[i0 + 1]).max(src[i0 + 2]);
                    dst[ol_i + 1] = src[i0 + 3].max(src[i0 + 4]).max(src[i0 + 5]);
                    dst[ol_i + 2] = src[i0 + 6].max(src[i0 + 7]).max(src[i0 + 8]);
                    dst[ol_i + 3] = src[i0 + 9].max(src[i0 + 10]).max(src[i0 + 11]);
                    ol_i += 4;
                }
                while ol_i < ol {
                    let i0 = ol_i * 3;
                    dst[ol_i] = src[i0].max(src[i0 + 1]).max(src[i0 + 2]);
                    ol_i += 1;
                }
            } else {
                for (ol_i, slot) in dst.iter_mut().enumerate() {
                    let i0 = ol_i * stride;
                    let mut m = src[i0];
                    for kk in 1..k {
                        let v = src[i0 + kk];
                        if v > m {
                            m = v;
                        }
                    }
                    *slot = m;
                }
            }
        }
    }
    y
}

pub fn abs_inplace(x: &mut Seq1d) {
    #[cfg(target_arch = "aarch64")]
    {
        neon_abs(&mut x.data);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for v in &mut x.data {
        *v = v.abs();
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_leaky(data: &mut [f32], alpha: f32) {
    use std::arch::aarch64::{vbslq_f32, vcleq_f32, vld1q_f32, vmovq_n_f32, vmulq_f32, vst1q_f32};
    let zero = unsafe { vmovq_n_f32(0.0) };
    let va = unsafe { vmovq_n_f32(alpha) };
    let mut chunks = data.chunks_exact_mut(4);
    for c in chunks.by_ref() {
        unsafe {
            let x = vld1q_f32(c.as_ptr());
            let neg = vmulq_f32(x, va);
            let y = vbslq_f32(vcleq_f32(x, zero), neg, x);
            vst1q_f32(c.as_mut_ptr(), y);
        }
    }
    for v in chunks.into_remainder() {
        if *v < 0.0 {
            *v *= alpha;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_abs(data: &mut [f32]) {
    use std::arch::aarch64::{vabsq_f32, vld1q_f32, vst1q_f32};
    let mut chunks = data.chunks_exact_mut(4);
    for c in chunks.by_ref() {
        unsafe {
            vst1q_f32(c.as_mut_ptr(), vabsq_f32(vld1q_f32(c.as_ptr())));
        }
    }
    for v in chunks.into_remainder() {
        *v = v.abs();
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_inorm(sl: &mut [f32], lf: f32, scale: f32, bias: f32, eps: f32) {
    use std::arch::aarch64::{
        vaddvq_f32, vfmaq_f32, vld1q_f32, vmovq_n_f32, vmulq_f32, vst1q_f32, vsubq_f32,
    };
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= sl.len() {
        unsafe {
            sum += vaddvq_f32(vld1q_f32(sl[i..].as_ptr()));
        }
        i += 4;
    }
    for &v in &sl[i..] {
        sum += v;
    }
    let mean = sum / lf;
    let vm = unsafe { vmovq_n_f32(mean) };
    let mut var = 0.0f32;
    i = 0;
    while i + 4 <= sl.len() {
        unsafe {
            let d = vsubq_f32(vld1q_f32(sl[i..].as_ptr()), vm);
            var += vaddvq_f32(vmulq_f32(d, d));
        }
        i += 4;
    }
    for &v in &sl[i..] {
        let d = v - mean;
        var += d * d;
    }
    var /= lf;
    let inv = 1.0 / (var + eps).sqrt();
    let vs = unsafe { vmovq_n_f32(inv * scale) };
    let vb = unsafe { vmovq_n_f32(bias) };
    i = 0;
    while i + 4 <= sl.len() {
        unsafe {
            let d = vsubq_f32(vld1q_f32(sl[i..].as_ptr()), vm);
            vst1q_f32(sl[i..].as_mut_ptr(), vfmaq_f32(vb, d, vs));
        }
        i += 4;
    }
    let g = inv * scale;
    for v in &mut sl[i..] {
        *v = (*v - mean) * g + bias;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_valid_length() {
        let x = Seq1d::zeros(1, 1, 20);
        let w = vec![1.0; 5];
        let y = conv1d(&x, &w, None, 1, 5, 2);
        assert_eq!(y.l, 8); // (20-5)/2+1
    }

    #[test]
    fn conv1d_ic1_dot() {
        let mut x = Seq1d::zeros(1, 1, 8);
        for i in 0..8 {
            x.set(0, 0, i, i as f32);
        }
        let w = vec![1.0, 0.0, -1.0];
        let y = conv1d(&x, &w, Some(&[0.5]), 1, 3, 1);
        assert_eq!(y.l, 6);
        assert!((y.get(0, 0, 0) - (0.5 + 0.0 - 2.0)).abs() < 1e-6);
        assert!((y.get(0, 0, 1) - (0.5 + 1.0 - 3.0)).abs() < 1e-6);
    }

    #[test]
    fn max_pool_valid() {
        let mut x = Seq1d::zeros(1, 1, 9);
        for i in 0..9 {
            x.set(0, 0, i, i as f32);
        }
        let y = max_pool1d(&x, 3, 3);
        assert_eq!(y.l, 3);
        assert_eq!(y.get(0, 0, 0), 2.0);
        assert_eq!(y.get(0, 0, 1), 5.0);
        assert_eq!(y.get(0, 0, 2), 8.0);
    }

    #[test]
    fn instance_norm_zero_mean_unit_scale() {
        let mut x = Seq1d::zeros(1, 1, 4);
        for i in 0..4 {
            x.set(0, 0, i, i as f32);
        }
        instance_norm_inplace(&mut x, &[1.0], &[0.0], 0.0);
        let mean: f32 = (0..4).map(|i| x.get(0, 0, i)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }
}
