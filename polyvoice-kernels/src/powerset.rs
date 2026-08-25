//! Pyannote powerset-3.0 matching shipping `powerset_int8` (or FP32).
//!
//! Waveform `[N,1,T]` → SincNet (IN + conv/pool/LeakyReLU) → 4× biLSTM-128
//! → Linear 256→128→128→7 → LogSoftmax. The shipping `If` is a no-op (both
//! branches were identical); InstanceNorm is the rewrite's explicit mean/var.

use crate::error::KernelError;
use crate::lstm::{BiLstm, log_softmax_last};
use crate::onnx_init::{OnnxTensor, load_initializers, take_f32, take_i8_quant};
use crate::qlinear;
use crate::seq1d::{
    Seq1d, abs_inplace, conv1d, instance_norm_inplace, leaky_relu_inplace, leaky_relu_slice,
    max_pool1d,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

pub const N_CLASSES: usize = 7;
pub const LEAKY_ALPHA: f32 = 0.01;
const IN_EPS: f32 = 1.0e-5;
const HIDDEN: usize = 128;

struct PsScratch {
    seq: Vec<f32>,
    lstm_a: Vec<f32>,
    lstm_b: Vec<f32>,
    ntf: Vec<f32>,
}

thread_local! {
    static PS_SCRATCH: RefCell<PsScratch> = const {
        RefCell::new(PsScratch {
            seq: Vec::new(),
            lstm_a: Vec::new(),
            lstm_b: Vec::new(),
            ntf: Vec::new(),
        })
    };
}

/// Minimum T so SincNet still emits one frame (valid conv/pool stack).
pub const MIN_SAMPLES: usize = 1251;

struct Linear {
    /// FP32 `[in, out]` when the file has no QDQ (tests / FP32).
    w: Vec<f32>,
    /// INT8 `[in, out]` + per-column scale/zp for `MatMulInteger`.
    w_i8: Vec<i8>,
    w_scale: Vec<f32>,
    w_zp: Vec<i8>,
    b: Vec<f32>,
    din: usize,
    dout: usize,
}

/// Hand-written powerset-3.0 (shipping ONNX weights, no graph runtime).
pub struct Powerset {
    wav_scale: Vec<f32>,
    wav_bias: Vec<f32>,
    sinc_w: Vec<f32>, // [80, 1, 251]
    n0_s: Vec<f32>,
    n0_b: Vec<f32>,
    c1_w: Vec<f32>, // [60, 80, 5]
    c1_b: Vec<f32>,
    n1_s: Vec<f32>,
    n1_b: Vec<f32>,
    c2_w: Vec<f32>, // [60, 60, 5]
    c2_b: Vec<f32>,
    n2_s: Vec<f32>,
    n2_b: Vec<f32>,
    lstm: [BiLstm; 4],
    lin0: Linear,
    lin1: Linear,
    clf: Linear,
}

impl Powerset {
    pub fn from_onnx_path(path: &Path) -> Result<Self, KernelError> {
        crate::rten_matmul::pin_parallelism();
        let init = load_initializers(path)?;
        Self::from_initializers(&init)
    }

    fn from_initializers(init: &HashMap<String, OnnxTensor>) -> Result<Self, KernelError> {
        let lstm = [
            take_lstm(
                init,
                "onnx::LSTM_784",
                "onnx::LSTM_785",
                "onnx::LSTM_783",
                60,
            )?,
            take_lstm(
                init,
                "onnx::LSTM_827",
                "onnx::LSTM_828",
                "onnx::LSTM_826",
                256,
            )?,
            take_lstm(
                init,
                "onnx::LSTM_870",
                "onnx::LSTM_871",
                "onnx::LSTM_869",
                256,
            )?,
            take_lstm(
                init,
                "onnx::LSTM_913",
                "onnx::LSTM_914",
                "onnx::LSTM_912",
                256,
            )?,
        ];
        Ok(Self {
            wav_scale: take_f32(init, "ortshared_1_1_1_1_token_110", &[1])?,
            wav_bias: take_f32(init, "ortshared_1_1_1_0_token_107", &[1])?,
            sinc_w: take_f32(init, "/sincnet/conv1d.0/Concat_2_output_0", &[80, 1, 251])?,
            n0_s: take_f32(init, "sincnet.norm1d.0.weight", &[80])?,
            n0_b: take_f32(init, "sincnet.norm1d.0.bias", &[80])?,
            c1_w: take_f32(init, "sincnet.conv1d.1.weight", &[60, 80, 5])?,
            c1_b: take_f32(init, "sincnet.conv1d.1.bias", &[60])?,
            n1_s: take_f32(init, "sincnet.norm1d.1.weight", &[60])?,
            n1_b: take_f32(init, "sincnet.norm1d.1.bias", &[60])?,
            c2_w: take_f32(init, "sincnet.conv1d.2.weight", &[60, 60, 5])?,
            c2_b: take_f32(init, "sincnet.conv1d.2.bias", &[60])?,
            n2_s: take_f32(init, "sincnet.norm1d.2.weight", &[60])?,
            n2_b: take_f32(init, "sincnet.norm1d.2.bias", &[60])?,
            lstm,
            lin0: take_linear(init, "onnx::MatMul_915", "linear.0.bias", 256, 128)?,
            lin1: take_linear(init, "onnx::MatMul_916", "linear.1.bias", 128, 128)?,
            clf: take_linear(
                init,
                "onnx::MatMul_917",
                "ortshared_1_1_7_0_token_109",
                128,
                N_CLASSES,
            )?,
        })
    }

    /// `waveforms` is packed `[N, T]` (channel already squeezed). Returns
    /// packed log-softmax `[N, F, 7]` and `F`.
    pub fn forward(
        &self,
        waveforms: &[f32],
        n: usize,
        t: usize,
    ) -> Result<(Vec<f32>, usize), KernelError> {
        if n == 0 || t == 0 || waveforms.len() != n * t {
            return Err(KernelError::WaveformShape {
                n,
                t,
                len: waveforms.len(),
            });
        }
        if t < MIN_SAMPLES {
            return Err(KernelError::WaveformTooShort {
                t,
                min_t: MIN_SAMPLES,
            });
        }
        let mut x = Seq1d::zeros(n, 1, t);
        x.data.copy_from_slice(waveforms);
        instance_norm_inplace(&mut x, &self.wav_scale, &self.wav_bias, IN_EPS);
        x = conv1d(&x, &self.sinc_w, None, 80, 251, 10);
        abs_inplace(&mut x);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &self.n0_s, &self.n0_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);
        x = conv1d(&x, &self.c1_w, Some(&self.c1_b), 60, 5, 1);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &self.n1_s, &self.n1_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);
        x = conv1d(&x, &self.c2_w, Some(&self.c2_b), 60, 5, 1);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &self.n2_s, &self.n2_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);

        // [N, 60, F] → [F, N, 60]
        let frames = x.l;
        PS_SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let mut seq = std::mem::take(&mut s.seq);
            let mut lstm_a = std::mem::take(&mut s.lstm_a);
            let mut lstm_b = std::mem::take(&mut s.lstm_b);
            let mut ntf = std::mem::take(&mut s.ntf);
            seq.resize(frames * n * 60, 0.0);
            for ni in 0..n {
                for c in 0..60 {
                    let src = &x.data[(ni * 60 + c) * frames..(ni * 60 + c) * frames + frames];
                    for f in 0..frames {
                        seq[(f * n + ni) * 60 + c] = src[f];
                    }
                }
            }
            debug_assert_eq!(self.lstm[0].input, 60);
            self.lstm[0].forward_into(&seq, frames, n, &mut lstm_a);
            let mut use_a = true;
            for layer in &self.lstm[1..] {
                debug_assert_eq!(layer.input, 256);
                if use_a {
                    layer.forward_into(&lstm_a, frames, n, &mut lstm_b);
                } else {
                    layer.forward_into(&lstm_b, frames, n, &mut lstm_a);
                }
                use_a = !use_a;
            }
            ntf.resize(n * frames * 256, 0.0);
            let cur = if use_a { &lstm_a } else { &lstm_b };
            for f in 0..frames {
                for ni in 0..n {
                    let src_i = (f * n + ni) * 256;
                    let dst = (ni * frames + f) * 256;
                    ntf[dst..dst + 256].copy_from_slice(&cur[src_i..src_i + 256]);
                }
            }
            let mut h = apply_linear(&ntf, n, frames, &self.lin0);
            leaky_relu_slice(&mut h, LEAKY_ALPHA);
            h = apply_linear(&h, n, frames, &self.lin1);
            leaky_relu_slice(&mut h, LEAKY_ALPHA);
            let mut logits = apply_linear(&h, n, frames, &self.clf);
            log_softmax_last(&mut logits, n, frames, N_CLASSES);
            s.seq = seq;
            s.lstm_a = lstm_a;
            s.lstm_b = lstm_b;
            s.ntf = ntf;
            Ok((logits, frames))
        })
    }
}

fn apply_linear(x: &[f32], n: usize, frames: usize, lin: &Linear) -> Vec<f32> {
    let rows = n.saturating_mul(frames);
    if !lin.w_i8.is_empty() {
        // One DynamicQuantizeLinear scale per item so N>1 matches N sequential
        // runs (shipping INT8 is not batch-invariant if the scale is shared).
        let mut y = vec![0.0f32; rows * lin.dout];
        let span = frames * lin.din;
        let ospan = frames * lin.dout;
        for ni in 0..n {
            let yi = qlinear::dyn_matmul(
                &x[ni * span..ni * span + span],
                &lin.w_i8,
                &lin.w_scale,
                &lin.w_zp,
                &lin.b,
                frames,
                lin.dout,
                lin.din,
            );
            y[ni * ospan..ni * ospan + ospan].copy_from_slice(&yi);
        }
        return y;
    }
    let mut y = vec![0.0f32; rows * lin.dout];
    crate::gemm::gemm_bias(x, &lin.w, &lin.b, &mut y, rows, lin.dout, lin.din);
    y
}

fn take_lstm(
    init: &HashMap<String, OnnxTensor>,
    w: &str,
    r: &str,
    b: &str,
    input: usize,
) -> Result<BiLstm, KernelError> {
    let layer = BiLstm::from_onnx(
        take_f32(init, w, &[2, 4 * HIDDEN, input])?,
        take_f32(init, r, &[2, 4 * HIDDEN, HIDDEN])?,
        take_f32(init, b, &[2, 8 * HIDDEN])?,
        HIDDEN,
        input,
    );
    #[cfg(not(target_vendor = "apple"))]
    {
        if let (Ok(wq), Ok(rq)) = (take_lstm_i8(init, w, input), take_lstm_i8(init, r, HIDDEN)) {
            return Ok(layer.with_i8(wq.0, wq.1, wq.2, rq.0, rq.1, rq.2));
        }
    }
    Ok(layer)
}

/// LSTM QDQ payload as `[2, K, 4H]` (GEMM `B` layout). The INT8 file stores
/// last-two axes swapped vs the FP32 `[2, 4H, K]` layout.
#[cfg_attr(target_vendor = "apple", allow(dead_code))]
fn take_lstm_i8(
    init: &HashMap<String, OnnxTensor>,
    name: &str,
    k_dim: usize,
) -> Result<(Vec<i8>, Vec<f32>, Vec<i8>), KernelError> {
    let four = 4 * HIDDEN;
    if let Ok(q) = take_i8_quant(init, name, &[2, k_dim, four]) {
        return Ok(q);
    }
    let (raw, scale, zp) = take_i8_quant(init, name, &[2, four, k_dim])?;
    Ok((transpose_i8_last2(&raw, 2, four, k_dim), scale, zp))
}

#[cfg_attr(target_vendor = "apple", allow(dead_code))]
fn transpose_i8_last2(data: &[i8], d0: usize, d1: usize, d2: usize) -> Vec<i8> {
    let mut out = vec![0i8; d0.saturating_mul(d1).saturating_mul(d2)];
    for a in 0..d0 {
        for i in 0..d1 {
            for j in 0..d2 {
                out[a * d2 * d1 + j * d1 + i] = data[a * d1 * d2 + i * d2 + j];
            }
        }
    }
    out
}

fn take_linear(
    init: &HashMap<String, OnnxTensor>,
    w: &str,
    b: &str,
    din: usize,
    dout: usize,
) -> Result<Linear, KernelError> {
    let bias = take_f32(init, b, &[dout])?;
    if let Ok((w_i8, w_scale, w_zp)) = take_i8_quant(init, w, &[din, dout]) {
        return Ok(Linear {
            w: Vec::new(),
            w_i8: qlinear::transpose_kn(&w_i8, din, dout),
            w_scale,
            w_zp,
            b: bias,
            din,
            dout,
        });
    }
    Ok(Linear {
        w: take_f32(init, w, &[din, dout])?,
        w_i8: Vec::new(),
        w_scale: Vec::new(),
        w_zp: Vec::new(),
        b: bias,
        din,
        dout,
    })
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
        for name in ["int8/powerset_int8.onnx", "powerset_int8.onnx"] {
            let p = root.join(name);
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    fn int8_logits_track_fp32() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models");
        let q = root.join("int8/powerset_int8.onnx");
        let f = root.join("powerset_fp32.onnx");
        if !q.is_file() || !f.is_file() {
            return;
        }
        let a = Powerset::from_onnx_path(&q).unwrap();
        let b = Powerset::from_onnx_path(&f).unwrap();
        let t = 160_000;
        let wav: Vec<f32> = (0..t).map(|i| 0.05 * ((i as f32) * 0.02).sin()).collect();
        let (ya, fa) = a.forward(&wav, 1, t).unwrap();
        let (yb, fb) = b.forward(&wav, 1, t).unwrap();
        assert_eq!(fa, fb);
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in ya.iter().zip(yb.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        let mut agree = 0usize;
        for f in 0..fa {
            let ia = (0..7)
                .max_by(|i, j| ya[f * 7 + i].partial_cmp(&ya[f * 7 + j]).unwrap())
                .unwrap();
            let ib = (0..7)
                .max_by(|i, j| yb[f * 7 + i].partial_cmp(&yb[f * 7 + j]).unwrap())
                .unwrap();
            agree += usize::from(ia == ib);
        }
        eprintln!(
            "powerset int8↔fp32 logits cosine={cos:.6} frames={fa} argmax={agree}/{fa} lstm_i8={}",
            a.lstm.iter().filter(|l| l.has_i8_weights()).count()
        );
        assert!(cos > 0.98, "powerset int8 vs fp32 cosine={cos}");
    }

    fn load_f32(path: &str) -> Vec<f32> {
        let b = std::fs::read(path).unwrap();
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn report_pair(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label} len {} vs {}", a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut maxabs = 0.0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
            maxabs = maxabs.max((x - y).abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        eprintln!(
            "{label}: cosine={cos:.6} maxabs={maxabs:.5} sum_a={:.3} sum_b={:.3}",
            a.iter().sum::<f32>(),
            b.iter().sum::<f32>()
        );
    }

    #[test]
    fn int8_stage_vs_ort() {
        if !PathBuf::from("/tmp/fuzfh10s.f32").is_file() {
            return;
        }
        let model =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/int8/powerset_int8.onnx");
        let net = Powerset::from_onnx_path(&model).unwrap();
        let samples = load_f32("/tmp/fuzfh10s.f32");
        let n = 1;
        let t = samples.len();
        let mut x = Seq1d::zeros(n, 1, t);
        x.data.copy_from_slice(&samples);
        instance_norm_inplace(&mut x, &net.wav_scale, &net.wav_bias, IN_EPS);
        x = conv1d(&x, &net.sinc_w, None, 80, 251, 10);
        abs_inplace(&mut x);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &net.n0_s, &net.n0_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);
        x = conv1d(&x, &net.c1_w, Some(&net.c1_b), 60, 5, 1);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &net.n1_s, &net.n1_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);
        x = conv1d(&x, &net.c2_w, Some(&net.c2_b), 60, 5, 1);
        x = max_pool1d(&x, 3, 3);
        instance_norm_inplace(&mut x, &net.n2_s, &net.n2_b, IN_EPS);
        leaky_relu_inplace(&mut x, LEAKY_ALPHA);
        report_pair(
            "sinc",
            &x.data,
            &load_f32("/tmp/ort__sincnet_LeakyRelu_2_output_0.f32"),
        );

        let frames = x.l;
        let mut seq = vec![0.0f32; frames * n * 60];
        for ni in 0..n {
            for c in 0..60 {
                for f in 0..frames {
                    seq[(f * n + ni) * 60 + c] = x.get(ni, c, f);
                }
            }
        }
        let files = [
            "/tmp/ort__lstm_Reshape_output_0.f32",
            "/tmp/ort__lstm_Reshape_1_output_0.f32",
            "/tmp/ort__lstm_Reshape_2_output_0.f32",
            "/tmp/ort__lstm_Reshape_3_output_0.f32",
        ];
        let mut cur = seq;
        for (i, layer) in net.lstm.iter().enumerate() {
            cur = layer.forward(&cur, frames, n);
            report_pair(&format!("lstm{i}"), &cur, &load_f32(files[i]));
        }
        let mut ntf = vec![0.0f32; n * frames * 256];
        for f in 0..frames {
            for ni in 0..n {
                let src = (f * n + ni) * 256;
                let dst = (ni * frames + f) * 256;
                ntf[dst..dst + 256].copy_from_slice(&cur[src..src + 256]);
            }
        }
        report_pair(
            "pre_lin",
            &ntf,
            &load_f32("/tmp/ort__lstm_Transpose_5_output_0.f32"),
        );
        let mut h = apply_linear(&ntf, n, frames, &net.lin0);
        leaky_relu_slice(&mut h, LEAKY_ALPHA);
        report_pair("lin0", &h, &load_f32("/tmp/ort__LeakyRelu_output_0.f32"));
        h = apply_linear(&h, n, frames, &net.lin1);
        leaky_relu_slice(&mut h, LEAKY_ALPHA);
        report_pair("lin1", &h, &load_f32("/tmp/ort__LeakyRelu_1_output_0.f32"));
    }

    #[test]
    fn int8_matches_ort_dumped_window() {
        let wav = PathBuf::from("/tmp/fuzfh10s.f32");
        let ort = PathBuf::from("/tmp/ort_powerset.f32");
        let model =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/int8/powerset_int8.onnx");
        if !wav.is_file() || !ort.is_file() || !model.is_file() {
            return;
        }
        let pcm = std::fs::read(&wav).unwrap();
        let mut samples = vec![0.0f32; pcm.len() / 4];
        for (i, c) in pcm.chunks_exact(4).enumerate() {
            samples[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        let yort_b = std::fs::read(&ort).unwrap();
        let mut yort = vec![0.0f32; yort_b.len() / 4];
        for (i, c) in yort_b.chunks_exact(4).enumerate() {
            yort[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        let net = Powerset::from_onnx_path(&model).unwrap();
        let (y, f) = net.forward(&samples, 1, samples.len()).unwrap();
        let fp_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/powerset_fp32.onnx");
        if fp_path.is_file() {
            let nfp = Powerset::from_onnx_path(&fp_path).unwrap();
            let (yf, _) = nfp.forward(&samples, 1, samples.len()).unwrap();
            let mut ag = 0usize;
            for fi in 0..f {
                let ia = (0..7)
                    .max_by(|i, j| y[fi * 7 + i].partial_cmp(&y[fi * 7 + j]).unwrap())
                    .unwrap();
                let ib = (0..7)
                    .max_by(|i, j| yf[fi * 7 + i].partial_cmp(&yf[fi * 7 + j]).unwrap())
                    .unwrap();
                ag += usize::from(ia == ib);
            }
            eprintln!("native int8↔native fp32 argmax={ag}/{f}");
        }
        assert_eq!(y.len(), yort.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut agree = 0usize;
        for fi in 0..f {
            let ia = (0..7)
                .max_by(|i, j| y[fi * 7 + i].partial_cmp(&y[fi * 7 + j]).unwrap())
                .unwrap();
            let ib = (0..7)
                .max_by(|i, j| yort[fi * 7 + i].partial_cmp(&yort[fi * 7 + j]).unwrap())
                .unwrap();
            agree += usize::from(ia == ib);
        }
        for (&x, &z) in y.iter().zip(yort.iter()) {
            dot += f64::from(x) * f64::from(z);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(z) * f64::from(z);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        eprintln!("native int8↔ort int8 cosine={cos:.6} argmax={agree}/{f}");
        assert!(cos > 0.999, "native vs ort powerset cosine={cos}");
        assert!(agree + 2 >= f, "native vs ort argmax={agree}/{f}");
    }

    #[test]
    fn loads_and_runs_short_window() {
        let Some(path) = model_path() else {
            eprintln!("skip: powerset onnx missing");
            return;
        };
        let net = Powerset::from_onnx_path(&path).expect("load");
        let t = 16_000; // 1 s
        let mut wav = vec![0.0f32; t];
        for (i, v) in wav.iter_mut().enumerate() {
            *v = 0.05 * ((i as f32) * 0.02).sin();
        }
        let (out, frames) = net.forward(&wav, 1, t).expect("forward");
        assert_eq!(out.len(), frames * N_CLASSES);
        assert!(frames >= 1);
        assert!(out.iter().all(|v| v.is_finite()));
        // each frame's softmax sums to 1
        for f in 0..frames {
            let p: f32 = out[f * 7..f * 7 + 7].iter().map(|v| v.exp()).sum();
            assert!((p - 1.0).abs() < 1e-4, "frame {f} softmax={p}");
        }
    }

    #[test]
    #[ignore = "manual timing"]
    fn bench_powerset_10s() {
        let Some(path) = model_path() else {
            return;
        };
        let net = Powerset::from_onnx_path(&path).expect("load");
        for n in [1usize, 4, 8] {
            let t = 160_000;
            let wav = vec![0.01f32; n * t];
            let _ = net.forward(&wav, n, t).expect("warm");
            let reps = if n >= 8 { 2 } else { 4 };
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = net.forward(&wav, n, t).expect("fwd");
            }
            eprintln!(
                "powerset N={n} 10s avg={:.3}s",
                t0.elapsed().as_secs_f64() / reps as f64
            );
        }
    }

    #[test]
    fn batch_matches_sequential() {
        let Some(path) = model_path() else {
            return;
        };
        let net = Powerset::from_onnx_path(&path).expect("load");
        let t = 16_000;
        let mut a = vec![0.0f32; t];
        let mut b = vec![0.0f32; t];
        for i in 0..t {
            a[i] = 0.04 * ((i as f32) * 0.01).sin();
            b[i] = 0.03 * ((i as f32) * 0.03).cos();
        }
        let (ya, fa) = net.forward(&a, 1, t).unwrap();
        let (yb, fb) = net.forward(&b, 1, t).unwrap();
        let mut packed = a;
        packed.extend_from_slice(&b);
        let (y2, f2) = net.forward(&packed, 2, t).unwrap();
        assert_eq!(fa, f2);
        assert_eq!(fb, f2);
        let row = f2 * 7;
        for i in 0..row {
            assert!((y2[i] - ya[i]).abs() < 1e-5);
            assert!((y2[row + i] - yb[i]).abs() < 1e-5);
        }
    }
}
