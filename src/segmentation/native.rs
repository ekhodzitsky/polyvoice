//! Pyannote powerset-3.0 via `polyvoice-kernels` (no ONNX runtime).

use crate::segmentation::aggregator::{AggregationConfig, Aggregator, WindowOutput};
use crate::segmentation::{
    BinarizationConfig, MIN_AUDIO_SAMPLES, RawSegment, SegmentationError, Segmenter,
};
use polyvoice_kernels::{N_CLASSES, Powerset};
use std::path::{Path, PathBuf};

/// Hand-written powerset-3.0. Same 10 s / 2 s geometry as the shipping ONNX
/// adapter; inference is SincNet + 4× biLSTM in `polyvoice-kernels`.
pub struct PowersetNative {
    #[cfg(not(test))]
    net: Powerset,
    /// `None` only in aggregation tests that must not load an ONNX file.
    #[cfg(test)]
    net: Option<Powerset>,
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
            #[cfg(not(test))]
            net,
            #[cfg(test)]
            net: Some(net),
            path: path.to_path_buf(),
            window_secs: 10.0,
            hop_secs: 2.0,
            sample_rate: 16_000,
            aggregation: AggregationConfig::default(),
        })
    }

    /// Calibrated binarization for the aggregator. `None` keeps the default
    /// argmax path (`AggregationConfig::default`).
    pub(crate) fn with_binarization(mut self, binarization: Option<BinarizationConfig>) -> Self {
        self.aggregation.binarization = binarization;
        self
    }

    /// Aggregation-only stand-in. Does not load a net.
    #[cfg(test)]
    fn without_weights() -> Self {
        Self {
            net: None,
            path: PathBuf::new(),
            window_secs: 10.0,
            hop_secs: 2.0,
            sample_rate: 16_000,
            aggregation: AggregationConfig::default(),
        }
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
        #[cfg(not(test))]
        let net = &self.net;
        #[cfg(test)]
        let net = self
            .net
            .as_ref()
            .ok_or_else(|| SegmentationError::InferenceFailed {
                window_idx: 0,
                detail: "native powerset weights are not loaded".into(),
            })?;
        net.forward(waveforms, n, t)
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
            // Several files in flight: each gets a fair share of the cores.
            .div_ceil(polyvoice_kernels::file_parallelism())
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
                                    Err(_) => {
                                        return Err(SegmentationError::InferenceFailed {
                                            window_idx: 0,
                                            detail: "native window worker lock poisoned".into(),
                                        });
                                    }
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

impl PowersetNative {
    fn infer_as_windows(&self, audio: &[f32]) -> Result<Vec<WindowOutput>, SegmentationError> {
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
        Ok(windows)
    }
}

impl Segmenter for PowersetNative {
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        let windows = self.infer_as_windows(audio)?;
        Aggregator::new(self.aggregation.clone()).stitch(&windows)
    }

    fn windows(&self, audio: &[f32]) -> Result<Option<Vec<WindowOutput>>, SegmentationError> {
        Ok(Some(self.infer_as_windows(audio)?))
    }

    fn max_local_speakers(&self) -> usize {
        3
    }

    fn supports_overlap(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::BinarizationConfig;

    #[test]
    fn default_binarization_is_none_and_some_is_stored() {
        let plain = PowersetNative::without_weights();
        assert!(plain.aggregation.binarization.is_none());
        assert_eq!(plain.aggregation.min_segment_secs, 0.0);
        assert_eq!(plain.aggregation.max_local_speakers, 3);

        let bin = BinarizationConfig {
            onset: 0.64,
            offset: 0.42,
            min_duration_on: 0.15,
            min_duration_off: 0.05,
        };
        let stored = plain.with_binarization(Some(bin));
        assert_eq!(stored.aggregation.binarization, Some(bin));
        assert_eq!(stored.aggregation.min_segment_secs, 0.0);
        assert_eq!(stored.aggregation.max_local_speakers, 3);

        let cleared = stored.with_binarization(None);
        assert!(cleared.aggregation.binarization.is_none());
    }
}
