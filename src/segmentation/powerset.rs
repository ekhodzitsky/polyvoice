//! `PowersetSegmenter` — ONNX-backed `Segmenter` wrapping
//! `sherpa-onnx-pyannote-segmentation-3-0`.
//!
//! Slides a 10-second window across the audio with a 1.0s hop (90% overlap),
//! runs ONNX inference per window, and feeds outputs into `Aggregator`.
//! Inference goes through [`crate::onnx::InferenceRuntime`]; this module does
//! not import `ort::`.

use crate::onnx::{InferenceRuntime, InferenceTensor, NamedTensor, RuntimeSession};
use crate::segmentation::aggregator::{AggregationConfig, Aggregator, WindowOutput};
use crate::segmentation::{MIN_AUDIO_SAMPLES, RawSegment, SegmentationError, Segmenter};
use std::path::{Path, PathBuf};

/// Tunable parameters for `PowersetSegmenter`.
#[derive(Debug, Clone)]
pub struct PowersetConfig {
    /// Window duration in seconds.
    pub window_secs: f32,
    /// Hop size between windows in seconds.
    pub hop_secs: f32,
    /// Sample rate the model expects (16000 for sherpa-onnx-pyannote-segmentation-3-0).
    pub sample_rate: u32,
    /// Forwarded to the inner `Aggregator`.
    pub aggregation: AggregationConfig,
    /// Number of pooled inference sessions; windows fan out across them.
    /// `0` is treated as 1. Default: `clamp(available_parallelism, 1, 4)`.
    pub pool_size: usize,
}

/// Default session-pool size: a few parallel windows without oversubscribing
/// the machine (each session still gets a fair share of intra-op threads).
fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
}

impl Default for PowersetConfig {
    fn default() -> Self {
        // Hard-coded fallbacks used when the ONNX has no metadata_props and
        // the manifest entry lacks geometry fields. Prefer loading via
        // `models::metadata::load_model_config` + `with_model_meta` so
        // self-describing models win when present.
        Self {
            window_secs: 10.0,
            hop_secs: 1.0,
            sample_rate: 16000,
            aggregation: AggregationConfig::default(),
            pool_size: default_pool_size(),
        }
    }
}

impl PowersetConfig {
    /// Overlay fields from a [`crate::models::ModelConfigMeta`] onto this
    /// config. Only non-`None` meta fields replace the current values — stage
    /// defaults stay for anything the model/manifest did not carry.
    ///
    /// Window geometry is overlaid as a pair and only when the result stays
    /// valid (`0 < hop_secs <= window_secs`, positive sample rate, at least
    /// one sample per window/hop): inconsistent model metadata is ignored in
    /// favor of the current values so it cannot turn into a panic inside
    /// `segment()`. Geometry written directly onto the public fields is
    /// re-validated by `segment()` and reported as
    /// [`SegmentationError::InvalidGeometry`].
    ///
    /// Available when the `download` feature (models module) is enabled.
    #[cfg(feature = "download")]
    pub fn with_model_meta(mut self, meta: &crate::models::ModelConfigMeta) -> Self {
        if let Some(sr) = meta.sample_rate
            && sr > 0
        {
            self.sample_rate = sr;
        }
        let candidate = PowersetConfig {
            window_secs: meta.window_secs.unwrap_or(self.window_secs),
            hop_secs: meta.hop_secs.unwrap_or(self.hop_secs),
            sample_rate: self.sample_rate,
            aggregation: self.aggregation.clone(),
            pool_size: self.pool_size,
        };
        if candidate.validate_geometry().is_ok() {
            self.window_secs = candidate.window_secs;
            self.hop_secs = candidate.hop_secs;
        }
        self
    }

    /// Validate the window geometry against the contract of
    /// [`crate::window::WindowIter`] (used by `segment`): positive finite
    /// durations, hop not larger than the window, and a sample rate that
    /// turns both into at least one sample.
    fn validate_geometry(&self) -> Result<(), SegmentationError> {
        if self.sample_rate == 0 {
            return Err(SegmentationError::InvalidGeometry {
                detail: "sample_rate must be > 0".to_string(),
            });
        }
        let window_secs = self.window_secs;
        let hop_secs = self.hop_secs;
        if !window_secs.is_finite() || window_secs <= 0.0 {
            return Err(SegmentationError::InvalidGeometry {
                detail: format!("window_secs must be finite and > 0, got {window_secs}"),
            });
        }
        if !hop_secs.is_finite() || hop_secs <= 0.0 {
            return Err(SegmentationError::InvalidGeometry {
                detail: format!("hop_secs must be finite and > 0, got {hop_secs}"),
            });
        }
        if hop_secs > window_secs {
            return Err(SegmentationError::InvalidGeometry {
                detail: format!("hop_secs ({hop_secs}) must be <= window_secs ({window_secs})"),
            });
        }
        if self.window_samples() == 0 || self.hop_samples() == 0 {
            return Err(SegmentationError::InvalidGeometry {
                detail: format!(
                    "window_secs ({window_secs}) / hop_secs ({hop_secs}) must each yield at \
                     least one sample at sample_rate {}",
                    self.sample_rate
                ),
            });
        }
        Ok(())
    }

    fn window_samples(&self) -> usize {
        (self.window_secs * self.sample_rate as f32) as usize
    }

    fn hop_samples(&self) -> usize {
        (self.hop_secs * self.sample_rate as f32) as usize
    }
}

/// ONNX-backed powerset speaker segmenter.
pub struct PowersetSegmenter {
    pool: crate::utils::ObjectPool<RuntimeSession>,
    input_name: String,
    config: PowersetConfig,
    model_path: PathBuf,
}

impl PowersetSegmenter {
    /// { true }
    /// `pub fn new(model_path: impl AsRef<Path>) -> Result<Self, SegmentationError>`
    /// { true }
    /// Load the ONNX model from `model_path` with the target's default
    /// execution provider (today's behavior: CoreML on Apple Silicon).
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, SegmentationError> {
        Self::with_config(
            model_path,
            PowersetConfig::default(),
            crate::onnx::ExecutionProvider::auto(),
        )
    }

    /// { true }
    /// `pub fn with_config( model_path: impl AsRef<Path>, config: PowersetConfig, ep: ExecutionProvider, ) -> Result<Self, SegmentationError>`
    /// { true }
    /// Load with explicit configuration and execution provider.
    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: PowersetConfig,
        ep: crate::onnx::ExecutionProvider,
    ) -> Result<Self, SegmentationError> {
        let path = model_path.as_ref().to_path_buf();
        let pool_size = config.pool_size.max(1);
        // Each pool session gets a fair share of the machine's cores so a
        // loaded pool does not oversubscribe (same policy as the embedder).
        let intra = std::thread::available_parallelism()
            .map(|n| (n.get() / pool_size).max(1))
            .unwrap_or(1);
        let mut sessions = Vec::with_capacity(pool_size);
        let mut input_name = None;
        for _ in 0..pool_size {
            let session =
                crate::onnx::build_session_with_ep(&path, ep, Some(intra)).map_err(|e| {
                    SegmentationError::ModelIo {
                        path: path.clone(),
                        detail: e.to_string(),
                    }
                })?;
            if input_name.is_none() {
                input_name = Some(
                    session
                        .primary_input_name()
                        .unwrap_or("waveform")
                        .to_owned(),
                );
            }
            sessions.push(session);
        }
        let input_name = input_name.unwrap_or_else(|| "waveform".to_owned());
        Ok(Self {
            pool: crate::utils::ObjectPool::new(sessions),
            input_name,
            config,
            model_path: path,
        })
    }

    /// { true }
    /// pub fn config(&self) -> &PowersetConfig
    /// { ret == &self.config }
    pub fn config(&self) -> &PowersetConfig {
        &self.config
    }

    /// { true }
    /// pub fn model_path(&self) -> &Path
    /// { ret == self.model_path.as_path() }
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    fn window_samples(&self) -> usize {
        self.config.window_samples()
    }

    fn hop_samples(&self) -> usize {
        self.config.hop_samples()
    }

    /// Run inference on a single 10-second window.
    /// Returns (logits_flat_row_major, num_frames).
    fn infer_window(
        &self,
        session: &mut RuntimeSession,
        window: &[f32],
        window_idx: usize,
    ) -> Result<(Vec<f32>, usize), SegmentationError> {
        let win_samples = self.window_samples();
        // Zero-pad short audio to the full window length.
        let mut buf = vec![0.0_f32; win_samples];
        let n = window.len().min(win_samples);
        buf[..n].copy_from_slice(&window[..n]);

        // Shape [1, 1, win_samples] matching the model's "waveform" input.
        let input_tensor = InferenceTensor::f32(vec![1, 1, win_samples], buf);

        let outputs = session
            .run(&[NamedTensor::new(self.input_name.as_str(), &input_tensor)])
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("session.run: {e}"),
            })?;

        let first =
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| SegmentationError::InferenceFailed {
                    window_idx,
                    detail: "model produced no outputs".to_string(),
                })?;

        let shape_vec = first.shape.clone();
        let data = first
            .into_f32()
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("extract f32: {e}"),
            })?;

        // Expected shape: [1, num_frames, 7].
        if shape_vec.len() != 3 || shape_vec[0] != 1 || shape_vec[2] != 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: shape_vec,
            });
        }
        let num_frames = shape_vec[1];
        Ok((data, num_frames))
    }
}

impl Segmenter for PowersetSegmenter {
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        self.config.validate_geometry()?;
        if audio.len() < MIN_AUDIO_SAMPLES {
            return Err(SegmentationError::AudioTooShort {
                actual_secs: audio.len() as f32 / self.config.sample_rate as f32,
                min_secs: MIN_AUDIO_SAMPLES as f32 / self.config.sample_rate as f32,
            });
        }

        let win_samples = self.window_samples();
        let hop_samples = self.hop_samples();

        // Window starts are computed up front so worker threads only borrow
        // `audio` immutably. Each window is independent (the segmenter keeps
        // no state across windows), so chunks fan out across scoped threads
        // that check out pooled sessions; results are flattened in window
        // order, keeping the aggregated output bit-identical to the
        // sequential version.
        let specs: Vec<(usize, usize)> =
            crate::window::WindowIter::new(audio.len(), win_samples, hop_samples)
                .include_partial()
                .enumerate()
                .map(|(window_idx, (start_sample, _end_sample))| (window_idx, start_sample))
                .collect();

        let n = specs.len();
        let num_threads = self.config.pool_size.max(1).min(n).max(1);
        let chunk_size = n.div_ceil(num_threads);

        let windows: Vec<WindowOutput> = std::thread::scope(|s| {
            let handles: Vec<_> = specs
                .chunks(chunk_size.max(1))
                .map(|chunk| {
                    s.spawn(move || {
                        let mut session = self.pool.checkout();
                        chunk
                            .iter()
                            .map(|&(window_idx, start_sample)| {
                                let slice = &audio
                                    [start_sample..(start_sample + win_samples).min(audio.len())];
                                let (logits, num_frames) =
                                    self.infer_window(&mut session, slice, window_idx)?;
                                let start_t = start_sample as f32 / self.config.sample_rate as f32;
                                let end_t = (start_sample + win_samples) as f32
                                    / self.config.sample_rate as f32;
                                WindowOutput::new(start_t, end_t, logits, num_frames)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            let mut windows = Vec::with_capacity(n);
            for h in handles {
                // A panicking worker panics here too, as in the sequential version.
                let chunk_results = h.join().unwrap_or_else(|e| std::panic::resume_unwind(e));
                windows.extend(chunk_results);
            }
            windows.into_iter().collect::<Result<Vec<_>, _>>()
        })?;

        let agg = Aggregator::new(self.config.aggregation.clone());
        agg.stitch(&windows)
    }

    fn max_local_speakers(&self) -> usize {
        3
    }

    fn supports_overlap(&self) -> bool {
        true
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_geometry_accepts_default() {
        assert!(PowersetConfig::default().validate_geometry().is_ok());
    }

    #[test]
    fn validate_geometry_rejects_zero_window() {
        let config = PowersetConfig {
            window_secs: 0.0,
            ..Default::default()
        };
        let err = config.validate_geometry().unwrap_err();
        assert!(matches!(err, SegmentationError::InvalidGeometry { .. }));
    }

    #[test]
    fn validate_geometry_rejects_hop_larger_than_window() {
        let config = PowersetConfig {
            window_secs: 1.0,
            hop_secs: 2.0,
            ..Default::default()
        };
        let err = config.validate_geometry().unwrap_err();
        match err {
            SegmentationError::InvalidGeometry { detail } => {
                assert!(detail.contains("hop_secs"), "got: {detail}");
            }
            other => panic!("expected InvalidGeometry, got {other:?}"),
        }
    }

    #[test]
    fn validate_geometry_rejects_sub_sample_window() {
        // Positive but truncates to zero samples at 16 kHz.
        let config = PowersetConfig {
            window_secs: 1e-9,
            hop_secs: 1e-9,
            ..Default::default()
        };
        let err = config.validate_geometry().unwrap_err();
        assert!(matches!(err, SegmentationError::InvalidGeometry { .. }));
    }

    #[test]
    fn validate_geometry_rejects_zero_sample_rate() {
        let config = PowersetConfig {
            sample_rate: 0,
            ..Default::default()
        };
        let err = config.validate_geometry().unwrap_err();
        assert!(matches!(err, SegmentationError::InvalidGeometry { .. }));
    }

    #[cfg(feature = "download")]
    #[test]
    fn with_model_meta_applies_valid_geometry() {
        let meta = crate::models::ModelConfigMeta {
            sample_rate: Some(16000),
            window_secs: Some(5.0),
            hop_secs: Some(0.5),
            ..Default::default()
        };
        let config = PowersetConfig::default().with_model_meta(&meta);
        assert!((config.window_secs - 5.0).abs() < 1e-6);
        assert!((config.hop_secs - 0.5).abs() < 1e-6);
    }

    #[cfg(feature = "download")]
    #[test]
    fn with_model_meta_ignores_invalid_geometry() {
        let default = PowersetConfig::default();
        // hop > window after overlay: the pair must be rejected wholesale.
        let meta = crate::models::ModelConfigMeta {
            window_secs: Some(1.0),
            hop_secs: Some(2.0),
            ..Default::default()
        };
        let config = PowersetConfig::default().with_model_meta(&meta);
        assert!((config.window_secs - default.window_secs).abs() < 1e-6);
        assert!((config.hop_secs - default.hop_secs).abs() < 1e-6);

        // Non-positive window: rejected as well.
        let meta = crate::models::ModelConfigMeta {
            window_secs: Some(0.0),
            ..Default::default()
        };
        let config = PowersetConfig::default().with_model_meta(&meta);
        assert!((config.window_secs - default.window_secs).abs() < 1e-6);

        // Zero sample rate: rejected, valid hop still applies.
        let meta = crate::models::ModelConfigMeta {
            sample_rate: Some(0),
            hop_secs: Some(0.5),
            ..Default::default()
        };
        let config = PowersetConfig::default().with_model_meta(&meta);
        assert_eq!(config.sample_rate, default.sample_rate);
        assert!((config.hop_secs - 0.5).abs() < 1e-6);
    }
}
