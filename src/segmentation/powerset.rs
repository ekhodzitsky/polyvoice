//! `PowersetSegmenter` — ONNX-backed `Segmenter` wrapping
//! `sherpa-onnx-pyannote-segmentation-3-0`.
//!
//! Slides a 10-second window across the audio with a 500ms hop (95% overlap),
//! runs ONNX inference per window, and feeds outputs into `Aggregator`.

use crate::segmentation::aggregator::{AggregationConfig, Aggregator, WindowOutput};
use crate::segmentation::{RawSegment, SegmentationError, Segmenter, MIN_AUDIO_SAMPLES};
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
}

impl Default for PowersetConfig {
    fn default() -> Self {
        Self {
            window_secs: 10.0,
            hop_secs: 0.5,
            sample_rate: 16000,
            aggregation: AggregationConfig::default(),
        }
    }
}

/// ONNX-backed powerset speaker segmenter.
pub struct PowersetSegmenter {
    session: Mutex<Session>,
    config: PowersetConfig,
    model_path: PathBuf,
}

impl PowersetSegmenter {
    /// Load the ONNX model from `model_path`.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, SegmentationError> {
        Self::with_config(model_path, PowersetConfig::default())
    }

    /// Load with explicit configuration.
    pub fn with_config(
        model_path: impl AsRef<Path>,
        config: PowersetConfig,
    ) -> Result<Self, SegmentationError> {
        let path = model_path.as_ref().to_path_buf();
        let session = Session::builder()
            .map_err(|e| SegmentationError::ModelIo {
                path: path.clone(),
                detail: format!("session builder failed: {e}"),
            })?
            .commit_from_file(&path)
            .map_err(|e| SegmentationError::ModelIo {
                path: path.clone(),
                detail: format!("commit_from_file failed: {e}"),
            })?;
        Ok(Self {
            session: Mutex::new(session),
            config,
            model_path: path,
        })
    }

    pub fn config(&self) -> &PowersetConfig {
        &self.config
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    fn window_samples(&self) -> usize {
        (self.config.window_secs * self.config.sample_rate as f32) as usize
    }

    fn hop_samples(&self) -> usize {
        (self.config.hop_secs * self.config.sample_rate as f32) as usize
    }

    /// Run inference on a single 10-second window.
    /// Returns (logits_flat_row_major, num_frames).
    fn infer_window(
        &self,
        window: &[f32],
        window_idx: usize,
    ) -> Result<(Vec<f32>, usize), SegmentationError> {
        let win_samples = self.window_samples();
        // Zero-pad short audio to the full window length.
        let mut buf = vec![0.0_f32; win_samples];
        let n = window.len().min(win_samples);
        buf[..n].copy_from_slice(&window[..n]);

        // Build input tensor with shape [1, 1, win_samples] matching the model's
        // "waveform" input. Uses the same TensorRef::from_array_view pattern as
        // silero_vad.rs (flat slice with explicit shape tuple).
        let input_tensor =
            ort::value::TensorRef::from_array_view(([1_usize, 1_usize, win_samples], buf.as_slice()))
                .map_err(|e| SegmentationError::InferenceFailed {
                    window_idx,
                    detail: format!("input tensor: {e}"),
                })?;

        let mut guard = self
            .session
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let outputs = guard
            .run(ort::inputs!["waveform" => input_tensor])
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("session.run: {e}"),
            })?;

        // Extract first output by index (robust to any output name).
        // try_extract_tensor returns (shape_slice, data_slice) matching ecapa.rs pattern.
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SegmentationError::InferenceFailed {
                window_idx,
                detail: format!("try_extract_tensor: {e}"),
            })?;

        // Expected shape: [1, num_frames, 7].
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if shape_vec.len() != 3 || shape_vec[0] != 1 || shape_vec[2] != 7 {
            return Err(SegmentationError::InvalidOutputShape {
                actual_shape: shape_vec,
            });
        }
        let num_frames = shape_vec[1];
        Ok((data.to_vec(), num_frames))
    }
}

impl Segmenter for PowersetSegmenter {
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        if audio.len() < MIN_AUDIO_SAMPLES {
            return Err(SegmentationError::AudioTooShort {
                actual_secs: audio.len() as f32 / self.config.sample_rate as f32,
                min_secs: MIN_AUDIO_SAMPLES as f32 / self.config.sample_rate as f32,
            });
        }

        let win_samples = self.window_samples();
        let hop_samples = self.hop_samples();
        let total_samples = audio.len();
        let mut windows: Vec<WindowOutput> = Vec::new();
        let mut window_idx = 0_usize;
        let mut start_sample = 0_usize;
        loop {
            let end_sample = (start_sample + win_samples).min(total_samples);
            let slice = &audio[start_sample..end_sample];
            let (logits, num_frames) = self.infer_window(slice, window_idx)?;
            let start_t = start_sample as f32 / self.config.sample_rate as f32;
            let end_t = (start_sample + win_samples) as f32 / self.config.sample_rate as f32;
            let w = WindowOutput::new(start_t, end_t, logits, num_frames)?;
            windows.push(w);
            window_idx += 1;
            if start_sample + win_samples >= total_samples {
                break;
            }
            start_sample += hop_samples;
        }

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
