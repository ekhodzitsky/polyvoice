//! Silero VAD v5 ONNX integration.
//!
//! Implements `VoiceActivityDetector` using the Silero VAD v5 ONNX model.
//! The model is stateful (LSTM) — hidden state is carried between calls
//! to `process()` and reset via `reset()`.

#[cfg(feature = "onnx")]
use crate::vad::{VadError, VoiceActivityDetector};

#[cfg(feature = "onnx")]
pub struct SileroVad {
    session: ort::session::Session,
    state: Vec<f32>,
    context: Vec<f32>,
    sample_rate: u32,
    chunk_size: usize,
    context_size: usize,
}

#[cfg(feature = "onnx")]
impl SileroVad {
    const STATE_SIZE: usize = 2 * 128;

    pub fn new(model_path: &std::path::Path, chunk_size: usize) -> Result<Self, anyhow::Error> {
        crate::onnx::validate_onnx_header(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = ort::session::Session::builder()
            .map_err(|e| anyhow::anyhow!("session builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("load model: {e}"))?;

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

        let input_tensor =
            ort::value::TensorRef::from_array_view(([1_usize, input.len()], input.as_slice()))
                .map_err(|e| VadError::Model(e.to_string()))?;

        let sr_array = ndarray::arr0(self.sample_rate as i64);
        let sr_tensor = ort::value::TensorRef::from_array_view(&sr_array)
            .map_err(|e| VadError::Model(e.to_string()))?;

        let state_array = ndarray::Array3::from_shape_vec((2, 1, 128), self.state.clone())
            .map_err(|e| VadError::Model(e.to_string()))?;
        let state_tensor = ort::value::TensorRef::from_array_view(&state_array)
            .map_err(|e| VadError::Model(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input_tensor, "state" => state_tensor, "sr" => sr_tensor])
            .map_err(|e| VadError::Model(e.to_string()))?;

        let (_, prob_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Model(e.to_string()))?;

        let (_, new_state) = outputs[1]
            .try_extract_tensor::<f32>()
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
    pub fn new(_model_path: &std::path::Path, _chunk_size: usize) -> Result<Self, anyhow::Error> {
        anyhow::bail!("the `onnx` feature is not enabled")
    }
}

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
