//! NCHW f32 tensor used by the ResNet kernels.

#[derive(Clone, Debug)]
pub struct Tensor {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            n,
            c,
            h,
            w,
            data: vec![0.0; n.saturating_mul(c).saturating_mul(h).saturating_mul(w)],
        }
    }

    /// Allocate without zeroing. Caller must write every element before any read.
    #[allow(clippy::uninit_vec)]
    pub fn uninit(n: usize, c: usize, h: usize, w: usize) -> Self {
        let mut t = Self {
            n: 0,
            c: 0,
            h: 0,
            w: 0,
            data: Vec::new(),
        };
        t.reuse_uninit(n, c, h, w);
        t
    }

    /// Grow-in-place uninit so ResNet can ping-pong two activation buffers.
    #[allow(clippy::uninit_vec)]
    pub fn reuse_uninit(&mut self, n: usize, c: usize, h: usize, w: usize) {
        let len = n.saturating_mul(c).saturating_mul(h).saturating_mul(w);
        if self.data.capacity() < len {
            self.data = Vec::with_capacity(len);
        }
        // SAFETY: Conv2d / BNNS / GEMM overwrite the whole buffer before it is read.
        unsafe {
            self.data.set_len(len);
        }
        self.n = n;
        self.c = c;
        self.h = h;
        self.w = w;
    }

    #[inline]
    pub fn idx(&self, n: usize, c: usize, h: usize, w: usize) -> usize {
        ((n * self.c + c) * self.h + h) * self.w + w
    }

    #[inline]
    #[cfg(test)]
    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> f32 {
        self.data[self.idx(n, c, h, w)]
    }

    #[inline]
    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, v: f32) {
        let i = self.idx(n, c, h, w);
        self.data[i] = v;
    }
}

/// y = relu(x) in place.
pub fn relu_inplace(x: &mut Tensor) {
    #[cfg(target_arch = "aarch64")]
    neon_relu(&mut x.data);
    #[cfg(not(target_arch = "aarch64"))]
    for v in &mut x.data {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Fake-quant `x` in place (ONNX QDQ: round, clip, dequant).
pub fn fake_quant_inplace(x: &mut Tensor, scale: f32, zp: i8) {
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    #[cfg(target_arch = "aarch64")]
    neon_fake_quant(&mut x.data, s, z);
    #[cfg(not(target_arch = "aarch64"))]
    for v in &mut x.data {
        let q = (*v / s).round() + z;
        let q = q.clamp(-128.0, 127.0);
        *v = (q - z) * s;
    }
}

/// `a = relu(a + b)` in one pass.
pub fn add_relu_inplace(a: &mut Tensor, b: &Tensor) {
    debug_assert_eq!(a.data.len(), b.data.len());
    #[cfg(target_arch = "aarch64")]
    neon_add_relu(&mut a.data, &b.data);
    #[cfg(not(target_arch = "aarch64"))]
    for (y, &x) in a.data.iter_mut().zip(b.data.iter()) {
        let s = *y + x;
        *y = if s > 0.0 { s } else { 0.0 };
    }
}

/// `a = relu(a)` and write the QDQ i8 of the result (vdiv).
pub fn relu_quantize_inplace(a: &mut Tensor, scale: f32, zp: i8, dst: &mut [i8]) {
    let n = a.data.len().min(dst.len());
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    #[cfg(target_arch = "aarch64")]
    neon_relu_quantize(&mut a.data[..n], s, z, &mut dst[..n]);
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        let v = a.data[i];
        let v = if v > 0.0 { v } else { 0.0 };
        a.data[i] = v;
        let q = (v / s).round() + z;
        dst[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

/// `a = relu(a + b)` and write the QDQ i8 of the result (vdiv, same as
/// standalone activation quantize). One map pass instead of add then quantize.
pub fn add_relu_quantize_inplace(a: &mut Tensor, b: &Tensor, scale: f32, zp: i8, dst: &mut [i8]) {
    debug_assert_eq!(a.data.len(), b.data.len());
    let n = a.data.len().min(dst.len());
    let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
    let z = f32::from(zp);
    #[cfg(target_arch = "aarch64")]
    neon_add_relu_quantize(&mut a.data[..n], &b.data[..n], s, z, &mut dst[..n]);
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..n {
        let v = a.data[i] + b.data[i];
        let v = if v > 0.0 { v } else { 0.0 };
        a.data[i] = v;
        let q = (v / s).round() + z;
        dst[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_relu(data: &mut [f32]) {
    use std::arch::aarch64::{vld1q_f32, vmaxq_f32, vmovq_n_f32, vst1q_f32};
    let z = unsafe { vmovq_n_f32(0.0) };
    let mut chunks = data.chunks_exact_mut(4);
    for c in chunks.by_ref() {
        unsafe {
            let v = vld1q_f32(c.as_ptr());
            vst1q_f32(c.as_mut_ptr(), vmaxq_f32(v, z));
        }
    }
    for v in chunks.into_remainder() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_relu_quantize(dst: &mut [f32], s: f32, z: f32, qdst: &mut [i8]) {
    use std::arch::aarch64::{
        vaddq_f32, vcvtq_s32_f32, vdivq_f32, vgetq_lane_s32, vld1q_f32, vmaxq_f32, vminq_f32,
        vmovq_n_f32, vrndaq_f32, vst1q_f32,
    };
    let n = dst.len().min(qdst.len());
    let z0 = unsafe { vmovq_n_f32(0.0) };
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(-128.0) };
    let hi = unsafe { vmovq_n_f32(127.0) };
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let r = vmaxq_f32(vld1q_f32(dst[i..].as_ptr()), z0);
            vst1q_f32(dst[i..].as_mut_ptr(), r);
            let q = vcvtq_s32_f32(vmaxq_f32(
                vminq_f32(vaddq_f32(vrndaq_f32(vdivq_f32(r, vs)), vz), hi),
                lo,
            ));
            qdst[i] = vgetq_lane_s32::<0>(q) as i8;
            qdst[i + 1] = vgetq_lane_s32::<1>(q) as i8;
            qdst[i + 2] = vgetq_lane_s32::<2>(q) as i8;
            qdst[i + 3] = vgetq_lane_s32::<3>(q) as i8;
        }
        i += 4;
    }
    for j in i..n {
        let v = if dst[j] > 0.0 { dst[j] } else { 0.0 };
        dst[j] = v;
        let q = (v / s).round() + z;
        qdst[j] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_add_relu(dst: &mut [f32], src: &[f32]) {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmaxq_f32, vmovq_n_f32, vst1q_f32};
    let n = dst.len().min(src.len());
    let z = unsafe { vmovq_n_f32(0.0) };
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let a = vld1q_f32(dst[i..].as_ptr());
            let b = vld1q_f32(src[i..].as_ptr());
            vst1q_f32(dst[i..].as_mut_ptr(), vmaxq_f32(vaddq_f32(a, b), z));
        }
        i += 4;
    }
    for j in i..n {
        let s = dst[j] + src[j];
        dst[j] = if s > 0.0 { s } else { 0.0 };
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_add_relu_quantize(dst: &mut [f32], src: &[f32], s: f32, z: f32, qdst: &mut [i8]) {
    use std::arch::aarch64::{
        vaddq_f32, vcvtq_s32_f32, vdivq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmovq_n_f32,
        vrndaq_f32, vst1q_f32,
    };
    let n = dst.len().min(src.len()).min(qdst.len());
    let z0 = unsafe { vmovq_n_f32(0.0) };
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(-128.0) };
    let hi = unsafe { vmovq_n_f32(127.0) };
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let a = vld1q_f32(dst[i..].as_ptr());
            let b = vld1q_f32(src[i..].as_ptr());
            let r = vmaxq_f32(vaddq_f32(a, b), z0);
            vst1q_f32(dst[i..].as_mut_ptr(), r);
            let q = vcvtq_s32_f32(vmaxq_f32(
                vminq_f32(vaddq_f32(vrndaq_f32(vdivq_f32(r, vs)), vz), hi),
                lo,
            ));
            qdst[i] = {
                use std::arch::aarch64::vgetq_lane_s32;
                vgetq_lane_s32::<0>(q) as i8
            };
            qdst[i + 1] = {
                use std::arch::aarch64::vgetq_lane_s32;
                vgetq_lane_s32::<1>(q) as i8
            };
            qdst[i + 2] = {
                use std::arch::aarch64::vgetq_lane_s32;
                vgetq_lane_s32::<2>(q) as i8
            };
            qdst[i + 3] = {
                use std::arch::aarch64::vgetq_lane_s32;
                vgetq_lane_s32::<3>(q) as i8
            };
        }
        i += 4;
    }
    for j in i..n {
        let v = dst[j] + src[j];
        let v = if v > 0.0 { v } else { 0.0 };
        dst[j] = v;
        let q = (v / s).round() + z;
        qdst[j] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_fake_quant(data: &mut [f32], s: f32, z: f32) {
    use std::arch::aarch64::{
        vaddq_f32, vdivq_f32, vld1q_f32, vmaxq_f32, vminq_f32, vmovq_n_f32, vmulq_f32, vrndaq_f32,
        vst1q_f32, vsubq_f32,
    };
    let vs = unsafe { vmovq_n_f32(s) };
    let vz = unsafe { vmovq_n_f32(z) };
    let lo = unsafe { vmovq_n_f32(-128.0) };
    let hi = unsafe { vmovq_n_f32(127.0) };
    let mut chunks = data.chunks_exact_mut(4);
    for c in chunks.by_ref() {
        unsafe {
            let x = vld1q_f32(c.as_ptr());
            let q = vaddq_f32(vrndaq_f32(vdivq_f32(x, vs)), vz);
            let q = vmaxq_f32(vminq_f32(q, hi), lo);
            vst1q_f32(c.as_mut_ptr(), vmulq_f32(vsubq_f32(q, vz), vs));
        }
    }
    for v in chunks.into_remainder() {
        let q = (*v / s).round() + z;
        let q = q.clamp(-128.0, 127.0);
        *v = (q - z) * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_quantize_matches_two_step() {
        let mut a = Tensor::zeros(1, 3, 5, 7);
        for i in 0..a.data.len() {
            a.data[i] = (i as f32) * 0.07 - 1.8;
        }
        let mut two = a.clone();
        relu_inplace(&mut two);
        let scale = 0.04;
        let zp: i8 = -128;
        let mut want = vec![0i8; two.data.len()];
        let s = scale;
        let z = f32::from(zp);
        for (d, &v) in want.iter_mut().zip(two.data.iter()) {
            let q = (v / s).round() + z;
            *d = q.clamp(-128.0, 127.0) as i8;
        }
        let mut got = vec![0i8; a.data.len()];
        relu_quantize_inplace(&mut a, scale, zp, &mut got);
        for i in 0..a.data.len() {
            assert!((a.data[i] - two.data[i]).abs() < 1e-6, "f32 i={i}");
            assert_eq!(got[i], want[i], "i8 i={i}");
        }
    }

    #[test]
    fn add_relu_quantize_matches_two_step() {
        let mut a = Tensor::zeros(1, 3, 5, 7);
        let mut b = Tensor::zeros(1, 3, 5, 7);
        for i in 0..a.data.len() {
            a.data[i] = (i as f32) * 0.07 - 1.8;
            b.data[i] = (i as f32) * -0.03 + 0.4;
        }
        let mut two = a.clone();
        add_relu_inplace(&mut two, &b);
        let mut want = vec![0i8; two.data.len()];
        let scale = 0.04;
        let zp: i8 = -128;
        let s = scale;
        let z = f32::from(zp);
        for (d, &v) in want.iter_mut().zip(two.data.iter()) {
            let q = (v / s).round() + z;
            *d = q.clamp(-128.0, 127.0) as i8;
        }
        let mut got = vec![0i8; a.data.len()];
        add_relu_quantize_inplace(&mut a, &b, scale, zp, &mut got);
        assert_eq!(a.data.len(), two.data.len());
        for i in 0..a.data.len() {
            assert!((a.data[i] - two.data[i]).abs() < 1e-6, "f32 i={i}");
            assert_eq!(got[i], want[i], "i8 i={i} got={} want={}", got[i], want[i]);
        }
    }
}
