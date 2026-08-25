//! Silero VAD (v6-generation) ONNX integration.
//!
//! Implements `VoiceActivityDetector` using the shipped Silero VAD ONNX model
//! (v6-generation weights; upstream v6.0 replaced the master file 2025-08-25,
//! releases through v6.2.1 keep the same architecture). The pinned SHA-256 in
//! `src/models/manifest.toml` (`1a153a22…`) is the source of truth for the
//! file we download — the upstream URL still tracks `master`, so a force-push
//! can break fresh installs (hash check fails closed). See
//! `scripts/mirror-silero-vad.md` for the release-asset mirror procedure.
//!
//! The model is stateful (LSTM) — hidden state is carried between calls
//! to `process()` and reset via `reset()`. Inference goes through
//! [`crate::onnx::InferenceRuntime`]; this module does not import `ort::`.

use crate::onnx::{InferenceRuntime, InferenceTensor, NamedTensor, RuntimeSession};
use crate::vad::{VadError, VoiceActivityDetector};

/// Errors from [`SileroVad`] construction.
///
/// Load-time failures are kept separate from the runtime [`VadError`] surface:
/// a model that never loads is a deployment problem, not a per-frame one.
#[derive(thiserror::Error, Debug)]
pub enum SileroVadError {
    /// `chunk_size` was 0 — the model scores fixed-size frames.
    #[error("SileroVad: chunk_size must be > 0")]
    ZeroChunkSize,

    /// The inference session failed to load (missing/invalid model file or
    /// backend build failure).
    #[error("failed to load Silero VAD model: {0}")]
    Session(#[from] crate::onnx::OnnxError),
}

pub struct SileroVad {
    session: RuntimeSession,
    state: Vec<f32>,
    context: Vec<f32>,
    chunk_size: usize,
    context_size: usize,
}

impl SileroVad {
    const STATE_SIZE: usize = 2 * 128;
    /// Silero VAD weights are trained for 16 kHz mono audio only.
    const SAMPLE_RATE: u32 = 16_000;

    /// { true }
    /// `pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, SileroVadError>`
    /// { true }
    /// Load with the historical default (no execution provider — plain CPU).
    pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, SileroVadError> {
        Self::with_ep(model_path, chunk_size, crate::onnx::ExecutionProvider::Cpu)
    }

    /// { true }
    /// `pub fn with_ep(model_path: &std::path::Path, chunk_size: usize, ep: ExecutionProvider) -> Result<Self, SileroVadError>`
    /// { true }
    /// Load with an explicit execution provider. The session goes through the
    /// shared EP-aware builder, so header validation and warn-and-CPU-fallback
    /// for unwired providers match every other ONNX session in the crate.
    pub fn with_ep(
        model_path: &std::path::Path,
        chunk_size: usize,
        ep: crate::onnx::ExecutionProvider,
    ) -> Result<Self, SileroVadError> {
        if chunk_size == 0 {
            return Err(SileroVadError::ZeroChunkSize);
        }
        let session = crate::onnx::build_session_with_ep(model_path, ep, None)?;

        let context_size = if chunk_size >= 512 { 64 } else { 32 };

        Ok(Self {
            session,
            state: vec![0.0f32; Self::STATE_SIZE],
            context: vec![0.0f32; context_size],
            chunk_size,
            context_size,
        })
    }

    fn run_chunk(&mut self, chunk: &[f32]) -> Result<f32, VadError> {
        let mut input = Vec::with_capacity(self.context_size + chunk.len());
        input.extend_from_slice(&self.context);
        input.extend_from_slice(chunk);

        let input_len = input.len();
        let input_tensor = InferenceTensor::f32(vec![1, input_len], input);
        let sr_tensor = InferenceTensor::i64_scalar(Self::SAMPLE_RATE as i64);
        let state_tensor = InferenceTensor::f32(vec![2, 1, 128], self.state.clone());

        let outputs = self
            .session
            .run(&[
                NamedTensor::new("input", &input_tensor),
                NamedTensor::new("state", &state_tensor),
                NamedTensor::new("sr", &sr_tensor),
            ])
            .map_err(|e| VadError::Model(e.to_string()))?;

        if outputs.len() < 2 {
            return Err(VadError::Model(
                "Silero VAD model produced fewer than 2 outputs".to_string(),
            ));
        }

        let prob_data = outputs[0]
            .as_f32_slice()
            .map_err(|e| VadError::Model(e.to_string()))?;
        let new_state = outputs[1]
            .as_f32_slice()
            .map_err(|e| VadError::Model(e.to_string()))?;

        let prob = prob_data
            .first()
            .copied()
            .ok_or_else(|| VadError::Model("empty probability output".to_string()))?;

        self.state = new_state.to_vec();
        if chunk.len() >= self.context_size {
            self.context
                .copy_from_slice(&chunk[chunk.len() - self.context_size..]);
        }

        Ok(prob)
    }
}

impl VoiceActivityDetector for SileroVad {
    fn reset(&mut self) {
        self.state = vec![0.0f32; Self::STATE_SIZE];
        self.context.fill(0.0);
    }

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        if !samples.len().is_multiple_of(self.chunk_size) {
            return Err(VadError::InvalidChunkSize {
                expected: self.chunk_size,
                got: samples.len(),
            });
        }

        let mut probs = Vec::with_capacity(samples.len() / self.chunk_size);
        for chunk in samples.chunks(self.chunk_size) {
            let prob = self.run_chunk(chunk)?;
            probs.push(prob);
        }
        Ok(probs)
    }

    fn sample_rate(&self) -> u32 {
        Self::SAMPLE_RATE
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "onnx")]
    use crate::onnx::{ExecutionProvider, InferenceBackend};
    use std::path::Path;
    #[cfg(feature = "onnx")]
    use std::path::PathBuf;

    #[cfg(feature = "onnx")]
    const SILERO: &str = "models/silero_vad.onnx";

    #[cfg(feature = "onnx")]
    fn silero_path() -> Option<PathBuf> {
        let p = Path::new(SILERO);
        if p.is_file() {
            Some(p.to_path_buf())
        } else {
            None
        }
    }

    /// `n` samples of a 300 Hz sine at 16 kHz, amplitude 0.3.
    #[cfg(feature = "onnx")]
    fn sine_samples(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    /// Extract the construction error without requiring `Debug` on the VAD
    /// itself (it holds an ONNX session, so it does not derive it).
    fn build_err(r: Result<SileroVad, SileroVadError>) -> SileroVadError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected construction to fail"),
        }
    }

    #[test]
    fn new_rejects_zero_chunk_size() {
        let err = build_err(SileroVad::new(Path::new("models/__missing__.onnx"), 0));
        assert!(matches!(err, SileroVadError::ZeroChunkSize));
        assert_eq!(err.to_string(), "SileroVad: chunk_size must be > 0");
    }

    #[test]
    fn new_fails_on_garbage_model() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xAB; 64]).unwrap();
        let err = build_err(SileroVad::new(tmp.path(), 512));
        assert!(matches!(err, SileroVadError::Session(_)));
        assert!(
            err.to_string().contains("failed to load Silero VAD model"),
            "unexpected: {err}"
        );
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn with_ep_sets_context_size_from_chunk() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        // Pin ort: silero does not load on the tract backend today.
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let vad = SileroVad::new(&path, 512).unwrap();
        assert_eq!(vad.context_size, 64);
        assert_eq!(vad.context.len(), 64);
        assert_eq!(vad.state.len(), SileroVad::STATE_SIZE);
        assert!(vad.state.iter().all(|v| *v == 0.0));
        let vad = SileroVad::new(&path, 256).unwrap();
        assert_eq!(vad.context_size, 32);
        assert_eq!(vad.context.len(), 32);
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn with_ep_accepts_unwired_providers() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        // Unwired providers warn and fall back to CPU — never panic or error.
        assert!(SileroVad::with_ep(&path, 512, ExecutionProvider::Cuda).is_ok());
        assert!(SileroVad::with_ep(&path, 512, ExecutionProvider::auto()).is_ok());
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn process_rejects_partial_chunk() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let mut vad = SileroVad::new(&path, 512).unwrap();
        let err = vad.process(&vec![0.0f32; 500]).unwrap_err();
        assert!(matches!(
            err,
            VadError::InvalidChunkSize {
                expected: 512,
                got: 500
            }
        ));
        let err = vad.process(&vec![0.0f32; 512 + 256]).unwrap_err();
        assert!(matches!(
            err,
            VadError::InvalidChunkSize {
                expected: 512,
                got: 768
            }
        ));
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn process_returns_probs_in_unit_range() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let mut vad = SileroVad::new(&path, 512).unwrap();
        assert_eq!(vad.sample_rate(), 16_000);
        let probs = vad.process(&sine_samples(512 * 4)).unwrap();
        assert_eq!(probs.len(), 4);
        assert!(
            probs
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "probs out of range: {probs:?}"
        );
        // LSTM state and trailing context are carried between chunks.
        assert!(vad.state.iter().any(|v| *v != 0.0));
        assert!(vad.context.iter().any(|v| *v != 0.0));
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn silence_scores_low() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let mut vad = SileroVad::new(&path, 512).unwrap();
        let probs = vad.process(&vec![0.0f32; 512 * 2]).unwrap();
        assert_eq!(probs.len(), 2);
        assert!(
            probs.iter().all(|p| *p < 0.5),
            "silence scored high: {probs:?}"
        );
        InferenceBackend::force(None);
    }

    #[test]
    #[cfg(feature = "onnx")]
    #[cfg_attr(miri, ignore)]
    fn reset_restores_fresh_state() {
        let Some(path) = silero_path() else {
            eprintln!("skip: {SILERO} missing");
            return;
        };
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let mut vad = SileroVad::new(&path, 512).unwrap();
        let chunk = sine_samples(512);
        let first = vad.process(&chunk).unwrap();
        vad.reset();
        assert!(vad.state.iter().all(|v| *v == 0.0));
        assert!(vad.context.iter().all(|v| *v == 0.0));
        let second = vad.process(&chunk).unwrap();
        assert!(
            (first[0] - second[0]).abs() < 1e-6,
            "reset did not restore determinism: {} vs {}",
            first[0],
            second[0]
        );
        InferenceBackend::force(None);
    }
}
