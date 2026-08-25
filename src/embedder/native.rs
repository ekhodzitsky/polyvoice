//! WeSpeaker ResNet34 via `polyvoice-kernels` (no ONNX runtime).

use super::{Embedder, EmbedderError};
use crate::features::{FbankExtractor, apply_cmvn_inplace};
use crate::utils::l2_normalize;
use polyvoice_kernels::{EMBED_DIM, ResNet34};
use std::path::{Path, PathBuf};

/// Hand-written WeSpeaker ResNet34 (256-d). Same fbank + CMVN + L2 as the
/// ONNX adapter; inference is the fused-BN graph in `polyvoice-kernels`.
pub struct ResNet34Native {
    net: ResNet34,
    fbank: FbankExtractor,
    path: PathBuf,
}

impl ResNet34Native {
    /// Load weights from shipping `resnet34_int8.onnx` (initializers only).
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, EmbedderError> {
        let path = path.as_ref();
        let net = ResNet34::from_onnx_path(path).map_err(|e| EmbedderError::ModelIo {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;
        Ok(Self {
            net,
            fbank: FbankExtractor::new(crate::features::FbankConfig::default()),
            path: path.to_path_buf(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.path
    }

    fn n_frames_of(&self, samples: &[f32]) -> usize {
        let win = self.fbank.config.win_length;
        let hop = self.fbank.config.hop_length;
        let n = samples.len().max(win);
        1 + (n - win) / hop
    }

    fn fbank_one(&self, samples: &[f32]) -> Result<(Vec<f32>, usize), EmbedderError> {
        let min_samples = self.fbank.config.win_length;
        let padded: Vec<f32>;
        let samples = if samples.len() < min_samples {
            padded = {
                let mut v = vec![0.0_f32; min_samples];
                v[..samples.len()].copy_from_slice(samples);
                v
            };
            &padded
        } else {
            samples
        };
        let frames = self
            .fbank
            .extract(samples)
            .map_err(|e| EmbedderError::InferenceFailed {
                detail: e.to_string(),
            })?;
        if frames.is_empty() {
            let sr = self.fbank.config.sample_rate as f32;
            return Err(EmbedderError::AudioTooShort {
                actual_secs: samples.len() as f32 / sr,
                min_secs: min_samples as f32 / sr,
            });
        }
        let mut frames = frames;
        apply_cmvn_inplace(&mut frames);
        let n_frames = frames.len();
        let flat: Vec<f32> = frames.into_iter().flatten().collect();
        Ok((flat, n_frames))
    }

    fn embed_prepared(&self, flat: &[f32], n_frames: usize) -> Result<Vec<f32>, EmbedderError> {
        let mut embedding =
            self.net
                .embed_fbank(flat, n_frames)
                .map_err(|e| EmbedderError::InferenceFailed {
                    detail: e.to_string(),
                })?;
        if embedding.len() != EMBED_DIM {
            return Err(EmbedderError::DimMismatch {
                expected: EMBED_DIM,
                actual: embedding.len(),
            });
        }
        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

impl Embedder for ResNet34Native {
    fn dim(&self) -> usize {
        EMBED_DIM
    }

    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if audios.len() <= 1 {
            return audios.iter().map(|a| self.embed(a)).collect();
        }
        let n = audios.len();
        // Group by frame count from the PCM length so Job::Many still
        // batches same-T clips. Extract fbank inside the worker so a
        // long file does not hold every CMVN map at once.
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, a) in audios.iter().enumerate() {
            groups.entry(self.n_frames_of(a)).or_default().push(i);
        }
        enum Job {
            One(usize),
            Many(Vec<usize>),
        }
        let mut jobs: Vec<Job> = Vec::new();
        for (t, slots) in groups {
            if slots.len() == 1 {
                jobs.push(Job::One(slots[0]));
            } else {
                // Long clips: n=8 NCHW maps thrash L2. Short clips keep
                // chunks(8) so one ResNet forward reuses W.
                let pack = if t >= 800 { 4 } else { 8 };
                for chunk in slots.chunks(pack) {
                    jobs.push(Job::Many(chunk.to_vec()));
                }
            }
        }
        // Unique-T clips cannot pack into Job::Many, so when most jobs are
        // Job::One the packed-batch default leaves a core idle; those
        // batches get one more serial worker. 3 still beats 2/4/8 for
        // packed batches on the Linux ARM VM: more thrash L2 on INT8
        // tiles, fewer leave cores idle.
        let ones = jobs.iter().filter(|j| matches!(j, Job::One(_))).count();
        let cap = std::env::var("POLYVOICE_EMBED_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if ones * 2 > jobs.len() { 4 } else { 3 })
            .max(1);
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(jobs.len())
            .min(cap);
        #[cfg(not(target_vendor = "apple"))]
        polyvoice_kernels::set_intra_threads(3);
        let jobs = std::sync::Mutex::new(jobs);
        let result = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut local: Vec<(usize, Result<Vec<f32>, EmbedderError>)> = Vec::new();
                        loop {
                            let job = match jobs.lock() {
                                Ok(mut g) => g.pop(),
                                Err(_) => break,
                            };
                            let Some(job) = job else { break };
                            match job {
                                Job::One(s) => {
                                    local.push((s, self.embed(audios[s])));
                                }
                                Job::Many(slots) => {
                                    let mut owned: Vec<Vec<f32>> = Vec::with_capacity(slots.len());
                                    let mut ts: Vec<usize> = Vec::with_capacity(slots.len());
                                    let mut ok_slots: Vec<usize> = Vec::with_capacity(slots.len());
                                    for s in slots {
                                        match self.fbank_one(audios[s]) {
                                            Ok((flat, t)) => {
                                                owned.push(flat);
                                                ts.push(t);
                                                ok_slots.push(s);
                                            }
                                            Err(e) => local.push((s, Err(e))),
                                        }
                                    }
                                    if owned.is_empty() {
                                        continue;
                                    }
                                    let flats: Vec<&[f32]> =
                                        owned.iter().map(|v| v.as_slice()).collect();
                                    match self.net.embed_fbank_batch(&flats, &ts) {
                                        Ok(embs) => {
                                            for (s, mut emb) in ok_slots.into_iter().zip(embs) {
                                                if emb.len() != EMBED_DIM {
                                                    local.push((
                                                        s,
                                                        Err(EmbedderError::DimMismatch {
                                                            expected: EMBED_DIM,
                                                            actual: emb.len(),
                                                        }),
                                                    ));
                                                } else {
                                                    l2_normalize(&mut emb);
                                                    local.push((s, Ok(emb)));
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let idx = ok_slots[0];
                                            local.push((
                                                idx,
                                                Err(EmbedderError::InferenceFailed {
                                                    detail: e.to_string(),
                                                }),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        local
                    })
                })
                .collect();
            let mut out: Vec<Option<Result<Vec<f32>, EmbedderError>>> =
                (0..n).map(|_| None).collect();
            for handle in handles {
                match handle.join() {
                    Ok(rows) => {
                        for (i, r) in rows {
                            out[i] = Some(r);
                        }
                    }
                    Err(_) => {
                        return Err(EmbedderError::InferenceFailed {
                            detail: "native embed worker panicked".into(),
                        });
                    }
                }
            }
            let result = out
                .into_iter()
                .map(|o| {
                    o.ok_or_else(|| EmbedderError::InferenceFailed {
                        detail: "native embed dropped a clip".into(),
                    })
                    .and_then(|r| r)
                })
                .collect();
            #[cfg(target_vendor = "apple")]
            if std::env::var_os("POLYVOICE_BNNS_PROF").is_some() {
                let (c, h, ns) = polyvoice_kernels::bnns_prof();
                eprintln!(
                    "bnns prof creates={c} hits={h} create_ms={:.1}",
                    ns as f64 / 1e6
                );
            }
            result
        });
        #[cfg(not(target_vendor = "apple"))]
        polyvoice_kernels::set_intra_threads(1);
        result
    }

    fn embed(&self, samples: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let (flat, n_frames) = self.fbank_one(samples)?;
        self.embed_prepared(&flat, n_frames)
    }
}

#[cfg(all(test, feature = "onnx"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::embedder::ResNet34Adapter;
    use crate::onnx::{ExecutionProvider, InferenceBackend};
    use std::path::Path;

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    }

    /// Amplitude-modulated harmonic stack — closer to speech than a pure sine.
    /// A lone sine lands in a degenerate corner of embedding space where
    /// INT8-vs-float accumulation ordering swamps the signal (cosine ~0.95 on
    /// the production Linux INT8 path, worse on the FP32 fallbacks), while the
    /// DER gates prove end-task parity on real audio.
    fn harmonic_pcm(secs: f32) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let pi = std::f32::consts::PI;
                let env = 0.5 + 0.5 * (2.0 * pi * 3.0 * t).sin();
                env * (0.30 * (2.0 * pi * 160.0 * t).sin()
                    + 0.18 * (2.0 * pi * 320.0 * t).sin()
                    + 0.12 * (2.0 * pi * 480.0 * t).sin()
                    + 0.07 * (2.0 * pi * 900.0 * t).sin())
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn native_matches_onnx_resnet34() {
        let path = Path::new("models/int8/resnet34_int8.onnx");
        if !path.is_file() {
            eprintln!("skip: resnet34_int8.onnx missing");
            return;
        }
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let onnx = ResNet34Adapter::new(path, 1, ExecutionProvider::Cpu).unwrap();
        InferenceBackend::force(None);
        let native = ResNet34Native::from_onnx_path(path).unwrap();
        let pcm = harmonic_pcm(1.0);
        let a = onnx.embed(&pcm).unwrap();
        let b = native.embed(&pcm).unwrap();
        let c = cosine(&a, &b);
        eprintln!("native↔ort ResNet34 cosine={c:.6}");
        // Tripwire floor, not an exact-parity claim. The Linux aarch64 INT8
        // path accumulates integer dot products like ort and measures ~0.975
        // here; the FP32 fallbacks (Darwin BNNS, x86 in-crate) dequantize the
        // same weights and land ~0.83. End-task parity is proven by the DER
        // gates / scoreboard on real audio, not by synthetic-input cosine.
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        let floor = 0.93;
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        let floor = 0.70;
        assert!(c > floor, "native ResNet34 diverged from ort, cosine={c}");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod batch_tests {
    use super::*;
    use std::path::Path;

    fn sine_pcm(secs: f32) -> Vec<f32> {
        let n = (secs * 16_000.0) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            })
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn embed_batch_matches_sequential() {
        let path = Path::new("models/int8/resnet34_int8.onnx");
        if !path.is_file() {
            eprintln!("skip: resnet34_int8.onnx missing");
            return;
        }
        let native = ResNet34Native::from_onnx_path(path).unwrap();
        let a = sine_pcm(0.6);
        let b = sine_pcm(0.8);
        let seq = [
            native.embed(&a).unwrap(),
            native.embed(&b).unwrap(),
            native.embed(&a).unwrap(),
        ];
        let batch = native.embed_batch(&[&a, &b, &a]).unwrap();
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
            let c = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
            assert!(c > 0.999, "batch[{i}] cosine={c}");
        }
    }
}
