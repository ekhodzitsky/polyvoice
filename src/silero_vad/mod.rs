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

#[cfg(feature = "onnx")]
use crate::onnx::{InferenceRuntime, InferenceTensor, NamedTensor, RuntimeSession};
#[cfg(feature = "onnx")]
use crate::vad::{VadError, VoiceActivityDetector};

#[cfg(feature = "onnx")]
pub struct SileroVad {
    session: RuntimeSession,
    state: Vec<f32>,
    context: Vec<f32>,
    sample_rate: u32,
    chunk_size: usize,
    context_size: usize,
}

#[cfg(feature = "onnx")]
impl SileroVad {
    const STATE_SIZE: usize = 2 * 128;

    /// { true }
    /// `pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, anyhow::Error>`
    /// { true }
    /// Load with the historical default (no execution provider — plain CPU).
    pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, anyhow::Error> {
        Self::with_ep(model_path, chunk_size, crate::onnx::ExecutionProvider::Cpu)
    }

    /// { true }
    /// `pub fn with_ep(model_path: &std::path::Path, chunk_size: usize, ep: ExecutionProvider) -> Result<Self, anyhow::Error>`
    /// { true }
    /// Load with an explicit execution provider. The session goes through the
    /// shared EP-aware builder, so header validation and warn-and-CPU-fallback
    /// for unwired providers match every other ONNX session in the crate.
    pub fn with_ep(
        model_path: &std::path::Path,
        chunk_size: usize,
        ep: crate::onnx::ExecutionProvider,
    ) -> Result<Self, anyhow::Error> {
        if chunk_size == 0 {
            anyhow::bail!("SileroVad::new: chunk_size must be > 0");
        }
        let session = crate::onnx::build_session_with_ep(model_path, ep, None)?;

        let context_size = if chunk_size >= 512 { 64 } else { 32 };

        Ok(Self {
            session,
            state: vec![0.0f32; Self::STATE_SIZE],
            context: vec![0.0f32; context_size],
            sample_rate: 16000,
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
        let sr_tensor = InferenceTensor::i64_scalar(self.sample_rate as i64);
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

#[cfg(feature = "onnx")]
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
        self.sample_rate
    }
}

/// Stub when the `onnx` feature is disabled.
#[cfg(not(feature = "onnx"))]
pub struct SileroVad;

#[cfg(not(feature = "onnx"))]
impl SileroVad {
    /// { true }
    /// `pub fn new(_model_path: &std::path::Path, _chunk_size: usize) -> Result<Self, anyhow::Error>`
    /// { true }
    pub fn new(_model_path: &std::path::Path, _chunk_size: usize) -> Result<Self, anyhow::Error> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    #[test]
    fn test_silero_vad_stub_without_onnx() {
        #[cfg(not(feature = "onnx"))]
        {
            let result = super::SileroVad::new(std::path::Path::new("model.onnx"), 512);
            assert!(result.is_err());
        }
    }
}
