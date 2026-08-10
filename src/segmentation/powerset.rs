//! `PowersetSegmenter` — ONNX-backed `Segmenter` wrapping
//! `sherpa-onnx-pyannote-segmentation-3-0`.
//!
//! Slides a 10-second window across the audio with a 2.0s hop (80% overlap),
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
    /// How many sliding windows to pack into one ONNX `run` (`[N, 1, T]`).
    /// The powerset graph is dynamic-batch; N>1 is bit-identical to N×1 on
    /// CPU EP and typically faster. `0` is treated as 1. Default: 8.
    /// Override at runtime with `POLYVOICE_POWERSET_BATCH_SIZE`.
    pub batch_size: usize,
}

/// Default session-pool size: a few parallel windows without oversubscribing
/// the machine (each session still gets a fair share of intra-op threads).
fn default_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 4)
}

/// Default ONNX micro-batch size for multi-window `run`s.
///
/// **1** is the shipping default: production `powerset_int8` (models-int8-v2)
/// is not batch-invariant (N>1 changes logits vs N×1). Full AMI-16 CPU gate:
/// N=8 is ~20–30% faster on segmentation but +0.13 pp DER₀ — rejected under
/// the no-regression policy. Override with `POLYVOICE_POWERSET_BATCH_SIZE`
/// only for experiments that re-run the DER gate.
fn default_batch_size() -> usize {
    1
}

/// Resolve batch size: `POLYVOICE_POWERSET_BATCH_SIZE` env (if >0) → config → 1.
fn resolve_batch_size(configured: usize) -> usize {
    std::env::var("POLYVOICE_POWERSET_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(configured.max(1))
        .max(1)
}

impl Default for PowersetConfig {
    fn default() -> Self {
        // Hard-coded fallbacks used when the ONNX has no metadata_props and
        // the manifest entry lacks geometry fields. Prefer loading via
        // `models::metadata::load_model_config` + `with_model_meta` so
        // self-describing models win when present.
        Self {
            window_secs: 10.0,
            hop_secs: 2.0,
            sample_rate: 16000,
            aggregation: AggregationConfig::default(),
            pool_size: default_pool_size(),
            batch_size: default_batch_size(),
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
            batch_size: self.batch_size,
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
        let pool_size = crate::onnx::resolve_session_pool_size(config.pool_size);
        // Each pool session gets a fair share of the machine's cores so a
        // loaded pool does not oversubscribe (same policy as the embedder).
        // Overridable via POLYVOICE_INTRA_THREADS.
        let intra = crate::onnx::resolve_intra_threads(pool_size);
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

    /// Run inference on `windows.len()` sliding windows in one ONNX call.
    ///
    /// Input layout is `[N, 1, T]` (dynamic batch on the shipped powerset
    /// graph). Output is `[N, num_frames, 7]`, split into N row-major logit
    /// buffers. Order matches `windows`. Partial (short) windows are
    /// zero-padded to `T = window_samples()`.
    ///
    /// Packs windows into one `run`. On some ONNX graphs (e.g. older local
    /// FP32 / certain INT8 exports) N-batch is bit-identical to N×1; the
    /// shipping `powerset_int8` (models-int8-v2) is **not** — treat N as a
    /// measured accuracy/speed knob, not a pure scheduling flag.
    fn infer_windows_batch(
        &self,
        session: &mut RuntimeSession,
        windows: &[&[f32]],
        first_window_idx: usize,
    ) -> Result<Vec<(Vec<f32>, usize)>, SegmentationError> {
        let batch = windows.len();
        if batch == 0 {
            return Ok(Vec::new());
        }
        let win_samples = self.window_samples();
        // Pack N zero-padded windows into a contiguous [N, 1, T] buffer.
        let mut buf = vec![0.0_f32; batch * win_samples];
        for (i, window) in windows.iter().enumerate() {
            let n = window.len().min(win_samples);
            let start = i * win_samples;
            buf[start..start + n].copy_from_slice(&window[..n]);
        }

        let input_tensor = InferenceTensor::f32(vec![batch, 1, win_samples], buf);

        let outputs = session
            .run(&[NamedTensor::new(self.input_name.as_str(), &input_tensor)])
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx: first_window_idx,
                detail: format!("session.run (batch={batch}): {e}"),
            })?;

        let first =
            outputs
                .into_iter()
                .next()
                .ok_or_else(|| SegmentationError::InferenceFailed {
                    window_idx: first_window_idx,
                    detail: "model produced no outputs".to_string(),
                })?;

        let shape_vec = first.shape.clone();
        let data = first
            .into_f32()
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx: first_window_idx,
                detail: format!("extract f32: {e}"),
            })?;

        // Expected shape: [N, num_frames, 7].
        if shape_vec.len() != 3 || shape_vec[0] != batch || shape_vec[2] != 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: shape_vec,
            });
        }
        let num_frames = shape_vec[1];
        let row = num_frames
            .checked_mul(7)
            .ok_or_else(|| SegmentationError::InferenceFailed {
                window_idx: first_window_idx,
                detail: format!("num_frames*7 overflow: frames={num_frames}"),
            })?;
        if data.len() != batch * row {
            return Err(SegmentationError::InferenceFailed {
                window_idx: first_window_idx,
                detail: format!(
                    "output len {} != batch ({batch}) * frames*7 ({row})",
                    data.len()
                ),
            });
        }

        let mut out = Vec::with_capacity(batch);
        for i in 0..batch {
            let start = i * row;
            out.push((data[start..start + row].to_vec(), num_frames));
        }
        Ok(out)
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
        // no state across windows), so work fans out across scoped threads
        // that check out pooled sessions. Within a worker, windows are packed
        // into ONNX micro-batches of `batch_size` (`[N,1,T]`) — bit-identical
        // to N sequential runs on CPU EP, faster on pure CPU.
        let specs: Vec<(usize, usize)> =
            crate::window::WindowIter::new(audio.len(), win_samples, hop_samples)
                .include_partial()
                .enumerate()
                .map(|(window_idx, (start_sample, _end_sample))| (window_idx, start_sample))
                .collect();

        let n = specs.len();
        // Use the same env-resolved pool size as construction for fan-out.
        let pool = crate::onnx::resolve_session_pool_size(self.config.pool_size);
        let num_threads = pool.max(1).min(n).max(1);
        let chunk_size = n.div_ceil(num_threads);
        let batch_size = resolve_batch_size(self.config.batch_size);

        let windows: Vec<WindowOutput> = std::thread::scope(|s| {
            let handles: Vec<_> = specs
                .chunks(chunk_size.max(1))
                .map(|chunk| {
                    s.spawn(move || -> Result<Vec<WindowOutput>, SegmentationError> {
                        let mut session = self.pool.checkout();
                        let mut results = Vec::with_capacity(chunk.len());
                        for sub in chunk.chunks(batch_size) {
                            let slices: Vec<&[f32]> = sub
                                .iter()
                                .map(|&(_window_idx, start_sample)| {
                                    &audio[start_sample
                                        ..(start_sample + win_samples).min(audio.len())]
                                })
                                .collect();
                            let first_idx = sub[0].0;
                            let batch_out =
                                self.infer_windows_batch(&mut session, &slices, first_idx)?;
                            for (i, (logits, num_frames)) in batch_out.into_iter().enumerate() {
                                let (_window_idx, start_sample) = sub[i];
                                let start_t =
                                    start_sample as f32 / self.config.sample_rate as f32;
                                let end_t = (start_sample + win_samples) as f32
                                    / self.config.sample_rate as f32;
                                results.push(WindowOutput::new(
                                    start_t, end_t, logits, num_frames,
                                )?);
                            }
                        }
                        Ok(results)
                    })
                })
                .collect();

            let mut windows = Vec::with_capacity(n);
            for h in handles {
                // A panicking worker panics here too, as in the sequential version.
                let chunk_results = h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))?;
                windows.extend(chunk_results);
            }
            Ok::<Vec<WindowOutput>, SegmentationError>(windows)
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

    /// Path to a local powerset model (INT8 preferred; FP32 fallback for quant trees).
    fn local_model_path() -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let int8 = root.join("int8/powerset_int8.onnx");
        if int8.is_file() {
            return int8;
        }
        root.join("powerset_fp32.onnx")
    }

    fn sine_audio(secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect()
    }

    /// Small-window segmenter so inference stays fast; `pool_size` fans
    /// windows out across pooled sessions. `None` (test skips) when the
    /// gitignored model blob is not present locally.
    fn load_test_segmenter(pool_size: usize) -> Option<PowersetSegmenter> {
        let path = local_model_path();
        if !path.exists() {
            eprintln!("skip: local powerset ONNX missing");
            return None;
        }
        let config = PowersetConfig {
            window_secs: 2.0,
            hop_secs: 1.0,
            pool_size,
            ..Default::default()
        };
        Some(
            PowersetSegmenter::with_config(path, config, crate::onnx::ExecutionProvider::Cpu)
                .expect("local powerset model loads"),
        )
    }

    #[test]
    fn with_config_missing_model_reports_model_io() {
        let err = PowersetSegmenter::with_config(
            "/nonexistent/powerset.onnx",
            PowersetConfig::default(),
            crate::onnx::ExecutionProvider::Cpu,
        )
        .err()
        .expect("missing model must fail");
        match err {
            SegmentationError::ModelIo { path, .. } => {
                assert!(path.ends_with("powerset.onnx"), "got {path:?}");
            }
            other => panic!("expected ModelIo, got {other:?}"),
        }
    }

    #[test]
    fn new_loads_local_model_and_exposes_accessors() {
        let path = local_model_path();
        if !path.exists() {
            eprintln!("skip: local powerset ONNX missing");
            return;
        }
        let seg = PowersetSegmenter::new(&path).expect("local powerset model loads");
        assert!(
            seg.model_path().ends_with("powerset_int8.onnx")
                || seg.model_path().ends_with("powerset_fp32.onnx")
        );
        let cfg = seg.config();
        assert!((cfg.window_secs - 10.0).abs() < 1e-6);
        assert!((cfg.hop_secs - 2.0).abs() < 1e-6);
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(seg.window_samples(), 160_000);
        assert_eq!(seg.hop_samples(), 32_000);
        assert_eq!(seg.max_local_speakers(), 3);
        assert!(seg.supports_overlap());
    }

    #[test]
    fn segment_rejects_too_short_audio() {
        let Some(seg) = load_test_segmenter(1) else {
            return;
        };
        let err = seg.segment(&vec![0.0_f32; 100]).unwrap_err();
        match err {
            SegmentationError::AudioTooShort {
                actual_secs,
                min_secs,
            } => {
                assert!(actual_secs < min_secs);
                assert!((min_secs - 0.1).abs() < 1e-6);
            }
            other => panic!("expected AudioTooShort, got {other:?}"),
        }
    }

    #[test]
    fn segment_rejects_invalid_geometry_before_inference() {
        // Geometry is only validated in `segment()`, not at load time.
        if !local_model_path().exists() {
            eprintln!("skip: models/powerset_fp32.onnx missing");
            return;
        }
        let config = PowersetConfig {
            window_secs: 1.0,
            hop_secs: 2.0,
            ..Default::default()
        };
        let seg = PowersetSegmenter::with_config(
            local_model_path(),
            config,
            crate::onnx::ExecutionProvider::Cpu,
        )
        .expect("load does not validate geometry");
        let err = seg.segment(&vec![0.0_f32; 16_000]).unwrap_err();
        assert!(matches!(err, SegmentationError::InvalidGeometry { .. }));
    }

    #[test]
    fn segment_runs_pooled_windows_and_returns_well_formed_segments() {
        // pool_size 2 exercises the scoped-thread fan-out; 5s of audio with a
        // 2s window / 1s hop yields 5 windows, the last one partial.
        let Some(seg) = load_test_segmenter(2) else {
            return;
        };
        let audio = sine_audio(5.0, 16_000);
        let total_secs = audio.len() as f64 / 16_000.0;
        let segments = seg.segment(&audio).expect("segment runs");
        for w in segments.windows(2) {
            assert!(
                w[0].time.start <= w[1].time.start,
                "segments must be sorted by start"
            );
        }
        for s in &segments {
            assert!(s.time.start >= 0.0, "start in bounds: {s:?}");
            assert!(
                s.time.end <= total_secs + 1e-3,
                "end in bounds: {s:?} vs {total_secs}"
            );
            assert!(s.time.end >= s.time.start, "non-decreasing time: {s:?}");
            assert!(s.local_speaker_idx < 3, "local speaker bound: {s:?}");
            assert!(
                (0.0..=1.0).contains(&s.confidence.get()),
                "confidence in range: {s:?}"
            );
        }
    }

    #[test]
    fn segment_single_window_pool_size_zero_treated_as_one() {
        // pool_size 0 must not panic or spawn zero workers.
        let Some(seg) = load_test_segmenter(0) else {
            return;
        };
        let audio = sine_audio(2.0, 16_000);
        let segments = seg.segment(&audio).expect("segment runs");
        for s in &segments {
            assert!(s.local_speaker_idx < 3);
        }
    }

    #[test]
    fn resolve_batch_size_is_at_least_one() {
        assert!(resolve_batch_size(0) >= 1);
        assert!(resolve_batch_size(8) >= 1);
    }

    /// Batched and sequential logits must match on CPU (bit-identical on
    /// powerset_int8/fp32). This is the DER-safety contract for the batch path.
    #[test]
    fn infer_batch_matches_sequential_on_cpu() {
        let path = local_model_path();
        if !path.exists() {
            eprintln!("skip: local powerset ONNX missing");
            return;
        }
        let config = PowersetConfig {
            window_secs: 2.0,
            hop_secs: 1.0,
            pool_size: 1,
            batch_size: 4,
            ..Default::default()
        };
        let seg = PowersetSegmenter::with_config(
            path,
            config,
            crate::onnx::ExecutionProvider::Cpu,
        )
        .expect("load");
        let win = seg.window_samples();
        // Three synthetic windows (last one short → zero-pad path).
        let w0: Vec<f32> = (0..win).map(|i| (i as f32 * 0.001).sin()).collect();
        let w1: Vec<f32> = (0..win).map(|i| (i as f32 * 0.002).cos()).collect();
        let w2: Vec<f32> = (0..win / 2).map(|i| (i as f32 * 0.003).sin()).collect();
        let mut session = seg.pool.checkout();
        let seq: Vec<(Vec<f32>, usize)> = [&w0[..], &w1[..], &w2[..]]
            .iter()
            .enumerate()
            .map(|(i, w)| {
                seg.infer_windows_batch(&mut session, &[*w], i)
                    .expect("seq")
                    .into_iter()
                    .next()
                    .expect("N=1 row")
            })
            .collect();
        let batch = seg
            .infer_windows_batch(&mut session, &[&w0, &w1, &w2], 0)
            .expect("batch");
        assert_eq!(seq.len(), batch.len());
        for (i, ((s_logits, s_nf), (b_logits, b_nf))) in seq.iter().zip(batch.iter()).enumerate() {
            assert_eq!(s_nf, b_nf, "window {i} frame count");
            assert_eq!(
                s_logits, b_logits,
                "window {i} logits must be bit-identical batch vs sequential"
            );
        }
    }
}
