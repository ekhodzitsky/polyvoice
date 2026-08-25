//! Pyannote powerset-3.0 via `polyvoice-kernels` (no ONNX runtime).

use crate::segmentation::aggregator::{AggregationConfig, Aggregator, WindowOutput};
use crate::segmentation::{MIN_AUDIO_SAMPLES, RawSegment, SegmentationError, Segmenter};
use polyvoice_kernels::{N_CLASSES, Powerset};
use std::path::{Path, PathBuf};

/// Hand-written powerset-3.0. Same 10 s / 2 s geometry as the shipping ONNX
/// adapter; inference is SincNet + 4× biLSTM in `polyvoice-kernels`.
pub struct PowersetNative {
    net: Powerset,
    path: PathBuf,
    window_secs: f32,
    hop_secs: f32,
    sample_rate: u32,
    aggregation: AggregationConfig,
}

impl PowersetNative {
    pub fn from_onnx_path(path: impl AsRef<Path>) -> Result<Self, SegmentationError> {
        let path = path.as_ref();
        let net = Powerset::from_onnx_path(path).map_err(|e| SegmentationError::ModelIo {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })?;
        Ok(Self {
            net,
            path: path.to_path_buf(),
            window_secs: 10.0,
            hop_secs: 2.0,
            sample_rate: 16_000,
            aggregation: AggregationConfig::default(),
        })
    }

    pub fn model_path(&self) -> &Path {
        &self.path
    }

    /// Packed `[N, T]` waveforms → packed log-softmax `[N, F, 7]` and `F`.
    pub fn infer_packed(
        &self,
        waveforms: &[f32],
        n: usize,
        t: usize,
    ) -> Result<(Vec<f32>, usize), SegmentationError> {
        self.net
            .forward(waveforms, n, t)
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx: 0,
                detail: e.to_string(),
            })
    }

    fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate as f32) as usize
    }

    fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate as f32) as usize
    }

    /// Packed inference over `specs`, split across cores when there is more
    /// than one 10 s window. Each worker keeps a packed batch so LSTM GEMM
    /// still sees N>1 when a chunk has several windows.
    fn infer_windows(
        &self,
        audio: &[f32],
        specs: &[(usize, usize)],
        win: usize,
    ) -> Result<(Vec<Vec<f32>>, usize), SegmentationError> {
        let n = specs.len();
        if n == 0 {
            return Ok((Vec::new(), 0));
        }
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
            .min(n);
        if threads <= 1 {
            return self.infer_chunk(audio, specs, win);
        }
        // Cap packed N so a long AMI file does not hold ~n/ncpu 10s
        // waveforms per worker. Vox-3 stays on the keep static
        // schedule (chunk = n/ncpu <= 16). Larger files use a
        // work-queue of 4 so the packed LSTM working set stays smaller
        // than 8-window packs.
        let static_chunk = n.div_ceil(threads);
        let chunk = if static_chunk <= 16 { static_chunk } else { 4 };
        if n.div_ceil(chunk) <= threads {
            std::thread::scope(|scope| {
                let handles: Vec<_> = specs
                    .chunks(chunk)
                    .map(|ch| scope.spawn(|| self.infer_chunk(audio, ch, win)))
                    .collect();
                let mut all = Vec::with_capacity(n);
                let mut frames = 0usize;
                for (i, handle) in handles.into_iter().enumerate() {
                    match handle.join() {
                        Ok(Ok((rows, f))) => {
                            if i == 0 {
                                frames = f;
                            } else if f != frames {
                                return Err(SegmentationError::InvalidOutputShape {
                                    actual_shape: vec![rows.len(), f, N_CLASSES],
                                });
                            }
                            all.extend(rows);
                        }
                        Ok(Err(e)) => return Err(e),
                        Err(_) => {
                            return Err(SegmentationError::InferenceFailed {
                                window_idx: i.saturating_mul(chunk),
                                detail: "native window worker panicked".into(),
                            });
                        }
                    }
                }
                Ok((all, frames))
            })
        } else {
            let packs: Vec<(usize, &[(usize, usize)])> = {
                let mut v = Vec::new();
                let mut off = 0usize;
                for ch in specs.chunks(chunk) {
                    v.push((off, ch));
                    off += ch.len();
                }
                v
            };
            let jobs = std::sync::Mutex::new(packs);
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut local: Vec<(usize, Vec<Vec<f32>>, usize)> = Vec::new();
                            loop {
                                let job = match jobs.lock() {
                                    Ok(mut g) => {
                                        if g.is_empty() {
                                            None
                                        } else {
                                            Some(g.remove(0))
                                        }
                                    }
                                    Err(_) => break,
                                };
                                let Some((off, ch)) = job else { break };
                                match self.infer_chunk(audio, ch, win) {
                                    Ok((rows, f)) => local.push((off, rows, f)),
                                    Err(e) => return Err(e),
                                }
                            }
                            Ok(local)
                        })
                    })
                    .collect();
                let mut slots: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
                let mut frames = 0usize;
                for handle in handles {
                    match handle.join() {
                        Ok(Ok(rows)) => {
                            for (off, rs, f) in rows {
                                if frames == 0 {
                                    frames = f;
                                } else if f != frames {
                                    return Err(SegmentationError::InvalidOutputShape {
                                        actual_shape: vec![rs.len(), f, N_CLASSES],
                                    });
                                }
                                for (i, row) in rs.into_iter().enumerate() {
                                    slots[off + i] = Some(row);
                                }
                            }
                        }
                        Ok(Err(e)) => return Err(e),
                        Err(_) => {
                            return Err(SegmentationError::InferenceFailed {
                                window_idx: 0,
                                detail: "native window worker panicked".into(),
                            });
                        }
                    }
                }
                let mut all = Vec::with_capacity(n);
                for (i, s) in slots.into_iter().enumerate() {
                    match s {
                        Some(r) => all.push(r),
                        None => {
                            return Err(SegmentationError::InferenceFailed {
                                window_idx: i,
                                detail: "native window worker dropped a pack".into(),
                            });
                        }
                    }
                }
                Ok((all, frames))
            })
        }
    }

    fn infer_chunk(
        &self,
        audio: &[f32],
        specs: &[(usize, usize)],
        win: usize,
    ) -> Result<(Vec<Vec<f32>>, usize), SegmentationError> {
        let n = specs.len();
        let mut packed = vec![0.0f32; n.saturating_mul(win)];
        for (i, &(_idx, start)) in specs.iter().enumerate() {
            let sl = &audio[start..(start + win).min(audio.len())];
            packed[i * win..i * win + sl.len()].copy_from_slice(sl);
        }
        let (logits, frames) = self.infer_packed(&packed, n, win)?;
        let row = frames.saturating_mul(N_CLASSES);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(logits[i * row..i * row + row].to_vec());
        }
        Ok((out, frames))
    }
}

impl Segmenter for PowersetNative {
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        if audio.len() < MIN_AUDIO_SAMPLES {
            return Err(SegmentationError::AudioTooShort {
                actual_secs: audio.len() as f32 / self.sample_rate as f32,
                min_secs: MIN_AUDIO_SAMPLES as f32 / self.sample_rate as f32,
            });
        }
        let win = self.window_samples();
        let hop = self.hop_samples();
        let specs: Vec<(usize, usize)> = crate::window::WindowIter::new(audio.len(), win, hop)
            .include_partial()
            .enumerate()
            .map(|(i, (start, _))| (i, start))
            .collect();
        let n = specs.len();
        let (logits_by_window, frames) = self.infer_windows(audio, &specs, win)?;
        if frames == 0 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: vec![n, frames, N_CLASSES],
            });
        }
        let mut windows = Vec::with_capacity(n);
        for (i, &(_idx, start)) in specs.iter().enumerate() {
            let start_t = start as f32 / self.sample_rate as f32;
            let end_t = (start + win) as f32 / self.sample_rate as f32;
            windows.push(WindowOutput::new(
                start_t,
                end_t,
                logits_by_window[i].clone(),
                frames,
            )?);
        }
        Aggregator::new(self.aggregation.clone()).stitch(&windows)
    }

    fn max_local_speakers(&self) -> usize {
        3
    }

    fn supports_overlap(&self) -> bool {
        true
    }
}

#[cfg(all(test, feature = "onnx"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::onnx::{
        ExecutionProvider, InferenceBackend, InferenceRuntime, InferenceTensor, NamedTensor,
        build_session_with_ep,
    };
    use std::path::Path;

    fn model_path() -> Option<&'static Path> {
        for p in [
            Path::new("models/int8/powerset_int8.onnx"),
            Path::new("models/powerset_int8.onnx"),
        ] {
            if p.is_file() {
                return Some(p);
            }
        }
        None
    }

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

    #[test]
    #[cfg_attr(miri, ignore)]
    fn native_matches_ort_one_second() {
        let Some(path) = model_path() else {
            eprintln!("skip: powerset onnx missing");
            return;
        };
        let t = 16_000usize;
        let wav: Vec<f32> = (0..t).map(|i| 0.05 * ((i as f32) * 0.02).sin()).collect();

        let native = PowersetNative::from_onnx_path(path).unwrap();
        let (nlog, nf) = native.infer_packed(&wav, 1, t).unwrap();

        InferenceBackend::force(Some(InferenceBackend::Ort));
        let mut sess = build_session_with_ep(path, ExecutionProvider::Cpu, Some(1)).unwrap();
        let input = InferenceTensor::f32(vec![1, 1, t], wav);
        let name = sess.primary_input_name().unwrap_or("x").to_owned();
        let out = sess.run(&[NamedTensor::new(&name, &input)]).unwrap();
        InferenceBackend::force(None);
        let first = out.into_iter().next().unwrap();
        let shape = first.shape.clone();
        let olog = first.into_f32().unwrap();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[2], 7);
        assert_eq!(shape[1], nf);
        assert_eq!(olog.len(), nlog.len());

        let c = cosine(&nlog, &olog);
        let mut max_abs = 0.0f32;
        let mut argmax_eq = 0usize;
        let frames = nf;
        for f in 0..frames {
            let a = &nlog[f * 7..f * 7 + 7];
            let b = &olog[f * 7..f * 7 + 7];
            for i in 0..7 {
                max_abs = max_abs.max((a[i] - b[i]).abs());
            }
            let ia = a
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0;
            let ib = b
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0;
            if ia == ib {
                argmax_eq += 1;
            }
        }
        eprintln!(
            "native↔ort powerset cosine={c:.6} max_abs={max_abs:.5} argmax={argmax_eq}/{frames}"
        );
        assert!(c > 0.99, "log-softmax cosine {c}");
        assert!(
            argmax_eq * 100 >= frames * 95,
            "argmax agreement {argmax_eq}/{frames}"
        );
    }
}
