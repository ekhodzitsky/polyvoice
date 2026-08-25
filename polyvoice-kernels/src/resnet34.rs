//! WeSpeaker ResNet34 matching shipping `resnet34_int8` (QDQ) or FP32.
//!
//! Input fbank `[T, 80]` → transpose/unsqueeze to NCHW `[1,1,80,T]` → stem +
//! layers `[3,4,6,3]` (32/64/128/256) → unbiased temporal std-pool → GEMM
//! 5120→256 → subtract `mean_vec`. BatchNorm is already fused into Conv.

use crate::conv::Conv2d;
use crate::error::KernelError;
use crate::onnx_init::{OnnxTensor, load_initializers, take_f32, take_i8_quant};
use crate::tensor::{Tensor, add_relu_inplace, relu_inplace};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const EMBED_DIM: usize = 256;
pub const N_MELS: usize = 80;
const FC_IN: usize = 5120;
const STD_EPS: f32 = 1.0e-7;

/// `(weight, bias, oc, ic, k, stride, qdq_input)` — `qdq_input` is the
/// static activation scale prefix (`input.3` → `input.3_scale`), `None` for stem.
#[allow(clippy::type_complexity)]
const CONVS: &[(&str, &str, usize, usize, usize, usize, Option<&str>)] = &[
    ("onnx::Conv_367", "onnx::Conv_368", 32, 1, 3, 1, None),
    (
        "onnx::Conv_370",
        "onnx::Conv_371",
        32,
        32,
        3,
        1,
        Some("input.3"),
    ),
    (
        "onnx::Conv_373",
        "onnx::Conv_374",
        32,
        32,
        3,
        1,
        Some("input.11"),
    ),
    (
        "onnx::Conv_376",
        "onnx::Conv_377",
        32,
        32,
        3,
        1,
        Some("input.19"),
    ),
    (
        "onnx::Conv_379",
        "onnx::Conv_380",
        32,
        32,
        3,
        1,
        Some("input.27"),
    ),
    (
        "onnx::Conv_382",
        "onnx::Conv_383",
        32,
        32,
        3,
        1,
        Some("input.35"),
    ),
    (
        "onnx::Conv_385",
        "onnx::Conv_386",
        32,
        32,
        3,
        1,
        Some("input.43"),
    ),
    (
        "onnx::Conv_388",
        "onnx::Conv_389",
        64,
        32,
        3,
        2,
        Some("input.51"),
    ),
    (
        "onnx::Conv_391",
        "onnx::Conv_392",
        64,
        64,
        3,
        1,
        Some("input.59"),
    ),
    (
        "onnx::Conv_394",
        "onnx::Conv_395",
        64,
        32,
        1,
        2,
        Some("input.51"),
    ),
    (
        "onnx::Conv_397",
        "onnx::Conv_398",
        64,
        64,
        3,
        1,
        Some("input.71"),
    ),
    (
        "onnx::Conv_400",
        "onnx::Conv_401",
        64,
        64,
        3,
        1,
        Some("input.79"),
    ),
    (
        "onnx::Conv_403",
        "onnx::Conv_404",
        64,
        64,
        3,
        1,
        Some("input.87"),
    ),
    (
        "onnx::Conv_406",
        "onnx::Conv_407",
        64,
        64,
        3,
        1,
        Some("input.95"),
    ),
    (
        "onnx::Conv_409",
        "onnx::Conv_410",
        64,
        64,
        3,
        1,
        Some("input.103"),
    ),
    (
        "onnx::Conv_412",
        "onnx::Conv_413",
        64,
        64,
        3,
        1,
        Some("input.111"),
    ),
    (
        "onnx::Conv_415",
        "onnx::Conv_416",
        128,
        64,
        3,
        2,
        Some("input.119"),
    ),
    (
        "onnx::Conv_418",
        "onnx::Conv_419",
        128,
        128,
        3,
        1,
        Some("input.127"),
    ),
    (
        "onnx::Conv_421",
        "onnx::Conv_422",
        128,
        64,
        1,
        2,
        Some("input.119"),
    ),
    (
        "onnx::Conv_424",
        "onnx::Conv_425",
        128,
        128,
        3,
        1,
        Some("input.139"),
    ),
    (
        "onnx::Conv_427",
        "onnx::Conv_428",
        128,
        128,
        3,
        1,
        Some("input.147"),
    ),
    (
        "onnx::Conv_430",
        "onnx::Conv_431",
        128,
        128,
        3,
        1,
        Some("input.155"),
    ),
    (
        "onnx::Conv_433",
        "onnx::Conv_434",
        128,
        128,
        3,
        1,
        Some("input.163"),
    ),
    (
        "onnx::Conv_436",
        "onnx::Conv_437",
        128,
        128,
        3,
        1,
        Some("input.171"),
    ),
    (
        "onnx::Conv_439",
        "onnx::Conv_440",
        128,
        128,
        3,
        1,
        Some("input.179"),
    ),
    (
        "onnx::Conv_442",
        "onnx::Conv_443",
        128,
        128,
        3,
        1,
        Some("input.187"),
    ),
    (
        "onnx::Conv_445",
        "onnx::Conv_446",
        128,
        128,
        3,
        1,
        Some("input.195"),
    ),
    (
        "onnx::Conv_448",
        "onnx::Conv_449",
        128,
        128,
        3,
        1,
        Some("input.203"),
    ),
    (
        "onnx::Conv_451",
        "onnx::Conv_452",
        128,
        128,
        3,
        1,
        Some("input.211"),
    ),
    (
        "onnx::Conv_454",
        "onnx::Conv_455",
        256,
        128,
        3,
        2,
        Some("input.219"),
    ),
    (
        "onnx::Conv_457",
        "onnx::Conv_458",
        256,
        256,
        3,
        1,
        Some("input.227"),
    ),
    (
        "onnx::Conv_460",
        "onnx::Conv_461",
        256,
        128,
        1,
        2,
        Some("input.219"),
    ),
    (
        "onnx::Conv_463",
        "onnx::Conv_464",
        256,
        256,
        3,
        1,
        Some("input.239"),
    ),
    (
        "onnx::Conv_466",
        "onnx::Conv_467",
        256,
        256,
        3,
        1,
        Some("input.247"),
    ),
    (
        "onnx::Conv_469",
        "onnx::Conv_470",
        256,
        256,
        3,
        1,
        Some("input.255"),
    ),
    (
        "onnx::Conv_472",
        "onnx::Conv_473",
        256,
        256,
        3,
        1,
        Some("input.263"),
    ),
];

struct Block {
    conv1: Conv2d,
    conv2: Conv2d,
    down: Option<Conv2d>,
}

/// Hand-written WeSpeaker ResNet34 (shipping ONNX weights, no graph runtime).
pub struct ResNet34 {
    stem: Conv2d,
    layer1: Vec<Block>,
    layer2: Vec<Block>,
    layer3: Vec<Block>,
    layer4: Vec<Block>,
    fc_w: Vec<f32>, // [256, 5120]
    fc_b: Vec<f32>,
    mean_vec: Vec<f32>,
    fc_act_scale: Option<f32>,
    fc_act_zp: i8,
    onnx_path: PathBuf,
    /// Compiled graph is the only conv path; layer weights were not loaded.
    graph_only: bool,
}

impl ResNet34 {
    /// Load weights from shipping `resnet34_int8.onnx` or the FP32 file
    /// (initializers only; QDQ weights are dequantized).
    pub fn from_onnx_path(path: &Path) -> Result<Self, KernelError> {
        crate::rten_matmul::pin_parallelism();
        #[cfg(target_vendor = "apple")]
        let graph_ready = crate::bnns_graph::warmup(path);
        #[cfg(not(target_vendor = "apple"))]
        let graph_ready = false;
        let init = load_initializers(path)?;
        // When the compiled graph is live, skip dequantizing 36 convs into
        // FP32 — that copy was in the peak RSS together with the graph.
        let skip_convs = graph_ready && !cfg!(test);
        let mut net = Self::from_initializers(&init, skip_convs)?;
        net.onnx_path = path.to_path_buf();
        net.graph_only = skip_convs;
        Ok(net)
    }

    fn from_initializers(
        init: &HashMap<String, OnnxTensor>,
        skip_convs: bool,
    ) -> Result<Self, KernelError> {
        let mut convs = Vec::with_capacity(CONVS.len());
        for &(wn, bn, oc, ic, k, stride, qin) in CONVS {
            if skip_convs {
                convs.push(Conv2d::new(oc, ic, k, stride, Vec::new(), vec![0.0; oc]));
            } else {
                convs.push(take_conv(init, wn, bn, oc, ic, k, stride, qin)?);
            }
        }
        let mut it = convs.into_iter();
        let mut next = || {
            it.next().ok_or_else(|| KernelError::Model {
                detail: "internal: ran out of convs".into(),
            })
        };
        let stem = next()?;
        let layer1 = identity_blocks(&mut next, 3)?;
        let layer2 = down_then_identity(&mut next, 3)?;
        let layer3 = down_then_identity(&mut next, 5)?;
        let layer4 = down_then_identity(&mut next, 2)?;
        let fc_w = take_f32(init, "model.seg_1.weight", &[EMBED_DIM, FC_IN])?;
        let fc_b = take_f32(init, "model.seg_1.bias", &[EMBED_DIM])?;
        let mean_vec = take_f32(init, "mean_vec", &[EMBED_DIM])?;
        let (fc_act_scale, fc_act_zp) = match take_act(init, Some("onnx::Gemm_363")) {
            Some((s, z)) => (Some(s), z),
            None => (None, 0),
        };
        Ok(Self {
            stem,
            layer1,
            layer2,
            layer3,
            layer4,
            fc_w,
            fc_b,
            mean_vec,
            fc_act_scale,
            fc_act_zp,
            onnx_path: PathBuf::new(),
            graph_only: false,
        })
    }

    /// Run on CMVN fbank frames, row-major `T × 80`. Returns raw 256-d embedding
    /// (not L2-normalized — the caller matches the ONNX `embs` output).
    pub fn embed_fbank(&self, frames: &[f32], n_frames: usize) -> Result<Vec<f32>, KernelError> {
        if n_frames == 0 {
            return Err(KernelError::EmptyFbank);
        }
        if frames.len() != n_frames * N_MELS {
            return Err(KernelError::FbankShape {
                n_frames,
                n_mels: frames.len() / n_frames,
                expected_mels: N_MELS,
            });
        }
        // feats [T, 80] → NCHW [1, 1, 80, T]
        let mut x = Tensor::zeros(1, 1, N_MELS, n_frames);
        for t in 0..n_frames {
            for m in 0..N_MELS {
                x.set(0, 0, m, t, frames[t * N_MELS + m]);
            }
        }
        #[cfg(target_vendor = "apple")]
        if let Some(y) = crate::bnns_graph::try_forward(&self.onnx_path, &x)? {
            x = y;
        } else if self.graph_only {
            return Err(KernelError::Model {
                detail: "compiled ResNet graph failed to run".into(),
            });
        } else {
            x = self.stem.forward(&x);
            after_stem(&mut x, &self.layer1);
            x = run_layers(&self.layer1, &self.layer2, &self.layer3, &self.layer4, x);
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            x = self.stem.forward(&x);
            after_stem(&mut x, &self.layer1);
            x = run_layers(&self.layer1, &self.layer2, &self.layer3, &self.layer4, x);
        }
        let mut pooled = stats_pool_n(&x, 0, x.w);
        fake_quant_slice(&mut pooled, self.fc_act_scale, self.fc_act_zp);
        Ok(gemm_sub(&pooled, &self.fc_w, &self.fc_b, &self.mean_vec))
    }

    /// Batch of CMVN fbank sequences. Every item must have the same `T`
    /// (padding a short clip next to a long one leaks through the residual
    /// stack's receptive field).
    pub fn embed_fbank_batch(
        &self,
        items: &[&[f32]],
        n_frames: &[usize],
    ) -> Result<Vec<Vec<f32>>, KernelError> {
        if items.len() != n_frames.len() {
            return Err(KernelError::FbankShape {
                n_frames: items.len(),
                n_mels: n_frames.len(),
                expected_mels: items.len(),
            });
        }
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if items.len() == 1 {
            return Ok(vec![self.embed_fbank(items[0], n_frames[0])?]);
        }
        let t = n_frames[0];
        if t == 0 || n_frames.iter().any(|&ti| ti != t) {
            return Err(KernelError::EmptyFbank);
        }
        for flat in items {
            if flat.len() != t * N_MELS {
                return Err(KernelError::FbankShape {
                    n_frames: t,
                    n_mels: flat.len() / t.max(1),
                    expected_mels: N_MELS,
                });
            }
        }
        let n_img = items.len();
        #[cfg(target_vendor = "apple")]
        if n_img > 1 && crate::bnns_graph::resolve_path(&self.onnx_path).is_some() {
            let mut out = Vec::with_capacity(n_img);
            for (flat, &ti) in items.iter().zip(n_frames.iter()) {
                out.push(self.embed_fbank(flat, ti)?);
            }
            return Ok(out);
        }
        let mut x = Tensor::zeros(n_img, 1, N_MELS, t);
        for (ni, flat) in items.iter().enumerate() {
            for ti in 0..t {
                for m in 0..N_MELS {
                    x.set(ni, 0, m, ti, flat[ti * N_MELS + m]);
                }
            }
        }
        x = self.stem.forward(&x);
        after_stem(&mut x, &self.layer1);
        x = run_layers(&self.layer1, &self.layer2, &self.layer3, &self.layer4, x);
        let mut out = Vec::with_capacity(n_img);
        for ni in 0..n_img {
            let mut pooled = stats_pool_n(&x, ni, x.w);
            fake_quant_slice(&mut pooled, self.fc_act_scale, self.fc_act_zp);
            out.push(gemm_sub(&pooled, &self.fc_w, &self.fc_b, &self.mean_vec));
        }
        Ok(out)
    }
}

fn identity_blocks(
    next: &mut impl FnMut() -> Result<Conv2d, KernelError>,
    n: usize,
) -> Result<Vec<Block>, KernelError> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(Block {
            conv1: next()?,
            conv2: next()?,
            down: None,
        });
    }
    Ok(out)
}

fn down_then_identity(
    next: &mut impl FnMut() -> Result<Conv2d, KernelError>,
    n_id: usize,
) -> Result<Vec<Block>, KernelError> {
    let mut out = Vec::with_capacity(n_id + 1);
    out.push(Block {
        conv1: next()?,
        conv2: next()?,
        down: Some(next()?),
    });
    out.extend(identity_blocks(next, n_id)?);
    Ok(out)
}

fn in_q(b: Option<&Block>) -> Option<(f32, i8)> {
    b.and_then(|b| b.conv1.act_scale.map(|s| (s, b.conv1.act_zp)))
}

fn after_stem(x: &mut Tensor, layer1: &[Block]) {
    if let Some((s, zp)) = in_q(layer1.first()) {
        crate::conv_i8::seed_xq_relu(x, s, zp);
    } else {
        relu_inplace(x);
    }
}

fn run_layers(l1: &[Block], l2: &[Block], l3: &[Block], l4: &[Block], x: Tensor) -> Tensor {
    let x = run_layer(l1, x, in_q(l2.first()));
    let x = run_layer(l2, x, in_q(l3.first()));
    let x = run_layer(l3, x, in_q(l4.first()));
    run_layer(l4, x, None)
}

fn run_layer(blocks: &[Block], mut x: Tensor, after: Option<(f32, i8)>) -> Tensor {
    for (i, b) in blocks.iter().enumerate() {
        let next_q = if i + 1 < blocks.len() {
            in_q(Some(&blocks[i + 1]))
        } else {
            after
        };
        x = run_block(b, x, next_q);
    }
    x
}

thread_local! {
    static BLOCK_YQ: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
    static BLOCK_Z: std::cell::RefCell<Tensor> = const {
        std::cell::RefCell::new(Tensor {
            n: 0,
            c: 0,
            h: 0,
            w: 0,
            data: Vec::new(),
        })
    };
}

fn finish_add(x: &mut Tensor, z: &Tensor, next_q: Option<(f32, i8)>) {
    if let Some((s, zp)) = next_q {
        crate::conv_i8::seed_xq_add_relu(x, z, s, zp);
    } else {
        add_relu_inplace(x, z);
    }
}

fn run_block(b: &Block, mut x: Tensor, next_q: Option<(f32, i8)>) -> Tensor {
    if b.down.is_none() && try_i8_identity(b, &mut x, next_q) {
        return x;
    }
    let mut y = Tensor::uninit(0, 0, 0, 0);
    b.conv1.forward_into(&x, &mut y, true);
    let mut z = Tensor::uninit(0, 0, 0, 0);
    b.conv2.forward_into_owned(&mut y, &mut z, false);
    if let Some(d) = &b.down {
        d.forward_into(&x, &mut y, false);
        finish_add(&mut y, &z, next_q);
        y
    } else {
        finish_add(&mut x, &z, next_q);
        x
    }
}

/// Identity block: conv1 writes the next layer's i8 (no extra map quantize).
fn try_i8_identity(b: &Block, x: &mut Tensor, next_q: Option<(f32, i8)>) -> bool {
    if !crate::conv_i8::i8_conv_on() {
        return false;
    }
    if b.conv1.k != 3 || b.conv1.stride != 1 || b.conv1.pad != 1 {
        return false;
    }
    if b.conv1.q_w_pad.is_empty() || b.conv2.q_w_pad.is_empty() {
        return false;
    }
    let Some(next_s) = b.conv2.act_scale else {
        return false;
    };
    let (oh, ow) = b.conv1.out_hw_dims(x.h, x.w);
    let yq_len =
        x.n.saturating_mul(b.conv1.oc)
            .saturating_mul(oh)
            .saturating_mul(ow);
    let ok = BLOCK_YQ.with(|yc| {
        BLOCK_Z.with(|zc| {
            let mut yq = yc.borrow_mut();
            let mut z = zc.borrow_mut();
            if yq.len() < yq_len {
                yq.resize(yq_len, 0);
            }
            z.reuse_uninit(x.n, b.conv2.oc, oh, ow);
            if !crate::conv_i8::try_conv_to_i8(
                &b.conv1,
                x,
                &mut yq[..yq_len],
                true,
                next_s,
                b.conv2.act_zp,
            ) {
                return false;
            }
            if !crate::conv_i8::try_from_i8(&b.conv2, &yq[..yq_len], x.n, oh, ow, &mut z, false) {
                return false;
            }
            finish_add(x, &z, next_q);
            true
        })
    });
    ok
}

/// Unbiased std-pool over the last (time) axis of image `n`, first `t` steps.
fn stats_pool_n(x: &Tensor, n: usize, t: usize) -> Vec<f32> {
    let t = t.max(1).min(x.w);
    let spatial = x.c * x.h;
    let mut mean = vec![0.0f32; spatial];
    let mut var = vec![0.0f32; spatial];
    stats_pool_fill(x, n, t, &mut mean, &mut var);
    let mut out = Vec::with_capacity(spatial * 2);
    out.extend_from_slice(&mean);
    out.extend(var.into_iter().map(f32::sqrt));
    out
}

fn stats_pool_fill(x: &Tensor, n: usize, t: usize, mean: &mut [f32], var: &mut [f32]) {
    let t_f = t as f32;
    let inv = 1.0 / t_f;
    let unb = if t > 1 { t_f / (t_f - 1.0) } else { 0.0 };
    for c in 0..x.c {
        for h in 0..x.h {
            let base = x.idx(n, c, h, 0);
            let row = &x.data[base..base + t];
            let (m, v) = stats_row(row, inv, unb);
            let i = c * x.h + h;
            mean[i] = m;
            var[i] = v;
        }
    }
}

fn stats_row(row: &[f32], inv: f32, unb: f32) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        return stats_row_neon(row, inv, unb);
    }
    #[cfg(not(target_arch = "aarch64"))]
    stats_row_scalar(row, inv, unb)
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn stats_row_scalar(row: &[f32], inv: f32, unb: f32) -> (f32, f32) {
    let mut s = 0.0f32;
    for &v in row {
        s += v;
    }
    let m = s * inv;
    let mut q = 0.0f32;
    for &v in row {
        let d = v - m;
        q += d * d;
    }
    (m, q * inv * unb + STD_EPS)
}

#[cfg(target_arch = "aarch64")]
fn stats_row_neon(row: &[f32], inv: f32, unb: f32) -> (f32, f32) {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};
    let n = row.len();
    let p = row.as_ptr();
    let mut i = 0usize;
    let mut sum;
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut s0 = z;
        let mut s1 = z;
        let mut s2 = z;
        let mut s3 = z;
        while i + 32 <= n {
            s0 = vaddq_f32(s0, vld1q_f32(p.add(i)));
            s1 = vaddq_f32(s1, vld1q_f32(p.add(i + 4)));
            s2 = vaddq_f32(s2, vld1q_f32(p.add(i + 8)));
            s3 = vaddq_f32(s3, vld1q_f32(p.add(i + 12)));
            s0 = vaddq_f32(s0, vld1q_f32(p.add(i + 16)));
            s1 = vaddq_f32(s1, vld1q_f32(p.add(i + 20)));
            s2 = vaddq_f32(s2, vld1q_f32(p.add(i + 24)));
            s3 = vaddq_f32(s3, vld1q_f32(p.add(i + 28)));
            i += 32;
        }
        while i + 16 <= n {
            s0 = vaddq_f32(s0, vld1q_f32(p.add(i)));
            s1 = vaddq_f32(s1, vld1q_f32(p.add(i + 4)));
            s2 = vaddq_f32(s2, vld1q_f32(p.add(i + 8)));
            s3 = vaddq_f32(s3, vld1q_f32(p.add(i + 12)));
            i += 16;
        }
        sum = vaddvq_f32(s0) + vaddvq_f32(s1) + vaddvq_f32(s2) + vaddvq_f32(s3);
        while i + 4 <= n {
            sum += vaddvq_f32(vld1q_f32(p.add(i)));
            i += 4;
        }
    }
    while i < n {
        sum += row[i];
        i += 1;
    }
    let m = sum * inv;
    let vm = unsafe { vdupq_n_f32(m) };
    i = 0;
    let mut q;
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut a0 = z;
        let mut a1 = z;
        let mut a2 = z;
        let mut a3 = z;
        while i + 32 <= n {
            let d0 = vsubq_f32(vld1q_f32(p.add(i)), vm);
            let d1 = vsubq_f32(vld1q_f32(p.add(i + 4)), vm);
            let d2 = vsubq_f32(vld1q_f32(p.add(i + 8)), vm);
            let d3 = vsubq_f32(vld1q_f32(p.add(i + 12)), vm);
            let d4 = vsubq_f32(vld1q_f32(p.add(i + 16)), vm);
            let d5 = vsubq_f32(vld1q_f32(p.add(i + 20)), vm);
            let d6 = vsubq_f32(vld1q_f32(p.add(i + 24)), vm);
            let d7 = vsubq_f32(vld1q_f32(p.add(i + 28)), vm);
            a0 = vfmaq_f32(a0, d0, d0);
            a1 = vfmaq_f32(a1, d1, d1);
            a2 = vfmaq_f32(a2, d2, d2);
            a3 = vfmaq_f32(a3, d3, d3);
            a0 = vfmaq_f32(a0, d4, d4);
            a1 = vfmaq_f32(a1, d5, d5);
            a2 = vfmaq_f32(a2, d6, d6);
            a3 = vfmaq_f32(a3, d7, d7);
            i += 32;
        }
        while i + 16 <= n {
            let d0 = vsubq_f32(vld1q_f32(p.add(i)), vm);
            let d1 = vsubq_f32(vld1q_f32(p.add(i + 4)), vm);
            let d2 = vsubq_f32(vld1q_f32(p.add(i + 8)), vm);
            let d3 = vsubq_f32(vld1q_f32(p.add(i + 12)), vm);
            a0 = vfmaq_f32(a0, d0, d0);
            a1 = vfmaq_f32(a1, d1, d1);
            a2 = vfmaq_f32(a2, d2, d2);
            a3 = vfmaq_f32(a3, d3, d3);
            i += 16;
        }
        q = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3);
        while i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(p.add(i)), vm);
            q += vaddvq_f32(vfmaq_f32(z, d, d));
            i += 4;
        }
    }
    while i < n {
        let d = row[i] - m;
        q += d * d;
        i += 1;
    }
    (m, q * inv * unb + STD_EPS)
}

fn gemm_sub(x: &[f32], w: &[f32], b: &[f32], mean: &[f32]) -> Vec<f32> {
    // y = x @ W^T + b - mean_vec; W is [256, 5120]
    let mut y = vec![0.0f32; EMBED_DIM];
    for oc in 0..EMBED_DIM {
        let row = &w[oc * FC_IN..oc * FC_IN + FC_IN];
        let mut acc = b[oc];
        for (a, &wt) in x.iter().zip(row.iter()) {
            acc += a * wt;
        }
        y[oc] = acc - mean[oc];
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn take_conv(
    init: &HashMap<String, OnnxTensor>,
    w_name: &str,
    b_name: &str,
    oc: usize,
    ic: usize,
    k: usize,
    stride: usize,
    qin: Option<&str>,
) -> Result<Conv2d, KernelError> {
    let b = take_f32(init, b_name, &[oc])?;
    let mut conv = if let Ok((q_w, q_scale, _zp)) = take_i8_quant(init, w_name, &[oc, ic, k, k]) {
        Conv2d::quantized(oc, ic, k, stride, q_w, q_scale, b)
    } else {
        Conv2d::new(
            oc,
            ic,
            k,
            stride,
            take_f32(init, w_name, &[oc, ic, k, k])?,
            b,
        )
    };
    if let Some((scale, zp)) = take_act(init, qin) {
        conv = conv.with_input_quant(scale, zp);
    }
    Ok(conv)
}

fn take_act(init: &HashMap<String, OnnxTensor>, prefix: Option<&str>) -> Option<(f32, i8)> {
    let prefix = prefix?;
    let scale = take_f32(init, &format!("{prefix}_scale"), &[]).ok()?;
    if scale.is_empty() {
        return None;
    }
    let zp_name = format!("{prefix}_zero_point");
    let zp = init.get(&zp_name).map(|t| match &t.payload {
        crate::onnx_init::OnnxPayload::I8(v) => v.first().copied().unwrap_or(-128),
        crate::onnx_init::OnnxPayload::I32(v) => v.first().map(|&x| x as i8).unwrap_or(-128),
        crate::onnx_init::OnnxPayload::F32(_) => -128,
    })?;
    Some((scale[0], zp))
}

fn fake_quant_slice(x: &mut [f32], scale: Option<f32>, zp: i8) {
    let Some(s) = scale else {
        return;
    };
    let s = if s.abs() < 1e-12 { 1.0 } else { s };
    let z = f32::from(zp);
    for v in x {
        let q = (*v / s).round() + z;
        let q = q.clamp(-128.0, 127.0);
        *v = (q - z) * s;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> Option<PathBuf> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models");
        [
            root.join("int8/resnet34_int8.onnx"),
            root.join("resnet34_int8.onnx"),
        ]
        .into_iter()
        .find(|p| p.is_file())
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    fn layer_embed(net: &ResNet34, frames: &[f32], t: usize) -> Vec<f32> {
        let mut x = Tensor::zeros(1, 1, N_MELS, t);
        for ti in 0..t {
            for m in 0..N_MELS {
                x.set(0, 0, m, ti, frames[ti * N_MELS + m]);
            }
        }
        x = net.stem.forward(&x);
        after_stem(&mut x, &net.layer1);
        x = run_layers(&net.layer1, &net.layer2, &net.layer3, &net.layer4, x);
        let mut pooled = stats_pool_n(&x, 0, x.w);
        fake_quant_slice(&mut pooled, net.fc_act_scale, net.fc_act_zp);
        gemm_sub(&pooled, &net.fc_w, &net.fc_b, &net.mean_vec)
    }

    #[test]
    fn bnns_graph_tracks_layer_path() {
        let Some(path) = model_path() else {
            return;
        };
        #[cfg(target_vendor = "apple")]
        if crate::bnns_graph::resolve_path(&path).is_none() {
            eprintln!("skip: resnet34_bnns.mlmodelc missing");
            return;
        }
        let net = ResNet34::from_onnx_path(&path).expect("load");
        eprintln!(
            "rust stem w00={:?} b0={:?}",
            &net.stem.weight[..4],
            &net.stem.bias[..4]
        );
        let c1 = &net.layer1[0].conv1;
        eprintln!(
            "rust l1c1 w00={:?} b0={:?} wsum={} bsum={}",
            &c1.weight[..4],
            &c1.bias[..4],
            c1.weight.iter().sum::<f32>(),
            c1.bias.iter().sum::<f32>()
        );
        for t in [80usize, 150, 400] {
            let mut frames = vec![0.0f32; t * N_MELS];
            for (i, v) in frames.iter_mut().enumerate() {
                *v = ((i % 17) as f32) * 0.02 - 0.16;
            }
            let g = net.embed_fbank(&frames, t).expect("graph");
            let l = layer_embed(&net, &frames, t);
            let c = cosine(&g, &l);
            eprintln!("T={t} cosine={c:.6}");
            assert!(g.iter().all(|v| v.is_finite()));
            // Layer BNNS and the compiled graph are both skip-QDQ FP32 but
            // not bit-identical (Winograd vs fused). DER is the gate.
        }
    }

    #[test]
    fn embed_intra_matches_serial_long_t() {
        let Some(path) = model_path() else {
            return;
        };
        let net = ResNet34::from_onnx_path(&path).expect("load");
        for t in [400usize, 2000, 4100] {
            let mut frames = vec![0.0f32; t * N_MELS];
            for (i, v) in frames.iter_mut().enumerate() {
                *v = ((i % 17) as f32) * 0.02 - 0.16;
            }
            crate::set_intra_threads(1);
            let a = net.embed_fbank(&frames, t).expect("serial");
            crate::set_intra_threads(4);
            let b = net.embed_fbank(&frames, t).expect("intra");
            crate::set_intra_threads(1);
            let mut nmis = 0usize;
            for (&x, &y) in a.iter().zip(b.iter()) {
                if (x - y).abs() > 1e-5 {
                    nmis += 1;
                }
            }
            assert_eq!(nmis, 0, "T={t} intra vs serial nmis={nmis}");
        }
    }

    #[test]
    #[ignore = "manual timing"]
    fn bench_embed_layers() {
        let Some(path) = model_path() else {
            return;
        };
        let net = ResNet34::from_onnx_path(&path).expect("load");
        let t = 400usize;
        let mut frames = vec![0.0f32; t * N_MELS];
        for (i, v) in frames.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.02 - 0.16;
        }
        let _ = net.embed_fbank(&frames, t).expect("warm");
        let mut x = Tensor::zeros(1, 1, N_MELS, t);
        for ti in 0..t {
            for m in 0..N_MELS {
                x.set(0, 0, m, ti, frames[ti * N_MELS + m]);
            }
        }
        let t0 = std::time::Instant::now();
        x = net.stem.forward(&x);
        after_stem(&mut x, &net.layer1);
        let stem = t0.elapsed();
        let t0 = std::time::Instant::now();
        x = run_layer(&net.layer1, x, in_q(net.layer2.first()));
        let l1 = t0.elapsed();
        let t0 = std::time::Instant::now();
        x = run_layer(&net.layer2, x, in_q(net.layer3.first()));
        let l2 = t0.elapsed();
        let t0 = std::time::Instant::now();
        x = run_layer(&net.layer3, x, in_q(net.layer4.first()));
        let l3 = t0.elapsed();
        let t0 = std::time::Instant::now();
        x = run_layer(&net.layer4, x, None);
        let l4 = t0.elapsed();
        eprintln!(
            "T=400 stem={:.1}ms l1={:.1}ms l2={:.1}ms l3={:.1}ms l4={:.1}ms",
            stem.as_secs_f64() * 1e3,
            l1.as_secs_f64() * 1e3,
            l2.as_secs_f64() * 1e3,
            l3.as_secs_f64() * 1e3,
            l4.as_secs_f64() * 1e3
        );
        let _ = x;
    }

    #[test]
    #[ignore = "manual timing"]
    fn bench_embed_typical_t() {
        let Some(path) = model_path() else {
            return;
        };
        let net = ResNet34::from_onnx_path(&path).expect("load");
        for t in [150usize, 400, 800, 2000, 4100] {
            let mut frames = vec![0.0f32; t * N_MELS];
            for (i, v) in frames.iter_mut().enumerate() {
                *v = ((i % 17) as f32) * 0.02 - 0.16;
            }
            let _ = net.embed_fbank(&frames, t).expect("warm");
            let n = if t >= 2000 { 8 } else { 20 };
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                let _ = net.embed_fbank(&frames, t).expect("fwd");
            }
            eprintln!(
                "native T={t} avg={:.2}ms",
                t0.elapsed().as_secs_f64() * 1e3 / n as f64
            );
        }
    }

    #[test]
    fn loads_shipping_onnx_and_embeds() {
        let Some(path) = model_path() else {
            eprintln!("skip: models/wespeaker_resnet34.onnx missing");
            return;
        };
        let net = ResNet34::from_onnx_path(&path).expect("load");
        // 80 frames × 80 mels of small noise — enough time after 8× downsample.
        let n_frames = 80;
        let mut frames = vec![0.0f32; n_frames * N_MELS];
        for (i, v) in frames.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.01 - 0.08;
        }
        let emb = net.embed_fbank(&frames, n_frames).expect("forward");
        assert_eq!(emb.len(), EMBED_DIM);
        assert!(emb.iter().all(|v| v.is_finite()));
        // mean_vec centering: values should not all be ~0 or explode.
        let max = emb.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max > 1e-3 && max < 100.0, "suspicious range max={max}");
        assert!(net.stem.act_scale.is_none(), "stem must stay float");
        assert!(
            net.layer1[0].conv1.act_scale.is_some(),
            "int8 graph must carry activation scales"
        );
        let c = &net.layer1[0].conv1;
        eprintln!(
            "l1c1 scale={:?} zp={} out0={} wsum0={} k_pad={} q_w={}",
            c.act_scale,
            c.act_zp,
            c.out_scale.first().copied().unwrap_or(0.0),
            c.w_sum.first().copied().unwrap_or(0),
            c.k_pad,
            c.q_w.len()
        );
    }

    #[test]
    fn batch_matches_sequential_mixed_lengths() {
        let Some(path) = model_path() else {
            return;
        };
        let net = ResNet34::from_onnx_path(&path).unwrap();
        let mk = |t: usize, seed: u32| {
            let mut frames = vec![0.0f32; t * N_MELS];
            for (i, v) in frames.iter_mut().enumerate() {
                *v = ((i as u32).wrapping_mul(seed) % 17) as f32 * 0.01 - 0.08;
            }
            frames
        };
        let a = mk(96, 3);
        let b = mk(96, 7);
        let c = mk(96, 11);
        let seq = [
            net.embed_fbank(&a, 96).unwrap(),
            net.embed_fbank(&b, 96).unwrap(),
            net.embed_fbank(&c, 96).unwrap(),
        ];
        let batch = net.embed_fbank_batch(&[&a, &b, &c], &[96, 96, 96]).unwrap();
        assert_eq!(batch.len(), 3);
        for (i, (s, t)) in seq.iter().zip(batch.iter()).enumerate() {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (&x, &y) in s.iter().zip(t.iter()) {
                dot += f64::from(x) * f64::from(y);
                na += f64::from(x) * f64::from(x);
                nb += f64::from(y) * f64::from(y);
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
            assert!(cos > 0.999, "item {i} cosine={cos}");
        }
    }

    #[test]
    fn stats_pool_tracks_scalar_rows() {
        // Contiguous time axis of a fat last-layer map (256×10×T).
        for t in [1usize, 3, 8, 16, 50, 51] {
            let mut row = vec![0.0f32; t];
            for (i, v) in row.iter_mut().enumerate() {
                *v = ((i % 13) as f32) * 0.07 - 0.4;
            }
            let inv = 1.0 / t as f32;
            let unb = if t > 1 {
                t as f32 / (t as f32 - 1.0)
            } else {
                0.0
            };
            let (mn, vv) = stats_row(&row, inv, unb);
            let (ms, vs) = stats_row_scalar(&row, inv, unb);
            let dm = (mn - ms).abs();
            let dv = (vv - vs).abs();
            assert!(
                dm < 2e-4 && dv < 2e-4,
                "t={t} dm={dm} dv={dv} {mn}/{ms} {vv}/{vs}"
            );
        }
    }

    #[test]
    fn rejects_wrong_mel_count() {
        let Some(path) = model_path() else {
            return;
        };
        let net = ResNet34::from_onnx_path(&path).unwrap();
        let err = net.embed_fbank(&[0.0; 40], 2).unwrap_err();
        assert!(matches!(err, KernelError::FbankShape { .. }));
    }
}
