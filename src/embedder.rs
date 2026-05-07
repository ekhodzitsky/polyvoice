//! v1.0 `Embedder` trait + concrete extractors (CAM++, ResNet34) + pool +
//! overlap-mask helper.
//!
//! Added in v0.6 (M2). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1, §5.3.

/// Speaker embedding extractor — turns a slice of 16 kHz mono audio into a
/// fixed-dimension embedding vector. Implementations are expected to L2-normalize
/// their output so cosine similarity is a meaningful metric downstream.
///
/// In v1.0 (M2) the polyvoice crate introduces `Embedder` as the canonical
/// trait. The legacy `EmbeddingExtractor` trait and its implementations
/// (`FbankOnnxExtractor`, `OnnxEmbeddingExtractor`, `DummyExtractor`) remain
/// available unchanged — M6 will deprecate them.
pub trait Embedder: Send + Sync {
    /// Output dimension of this embedder. Constant per instance.
    fn dim(&self) -> usize;

    /// Compute an embedding for one audio segment.
    ///
    /// **Requires:** `audio` is 16 kHz mono PCM.
    /// **Guarantees on Ok:** `result.len() == self.dim()` and the vector is L2-normalized
    /// (`|sum(x²)¹ᐟ² − 1.0| < 1e-3`).
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError>;

    /// Compute embeddings for a batch of audio segments. Default implementation
    /// is sequential; impls may override with a true batched ONNX call.
    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        audios.iter().map(|a| self.embed(a)).collect()
    }
}

/// Errors from `Embedder` implementations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("audio too short for this embedder: {actual_secs:.3}s < {min_secs:.3}s")]
    AudioTooShort { actual_secs: f32, min_secs: f32 },

    #[error("ONNX inference failed: {detail}")]
    InferenceFailed { detail: String },

    #[error("expected embedding dim {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("model file io error on {path}: {detail}")]
    ModelIo {
        path: std::path::PathBuf,
        detail: String,
    },

    #[error("legacy adapter error: {0}")]
    Legacy(String),
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    /// In-memory dummy used by trait tests.
    struct ConstantEmbedder {
        values: Vec<f32>,
    }

    impl Embedder for ConstantEmbedder {
        fn dim(&self) -> usize {
            self.values.len()
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Ok(self.values.clone())
        }
    }

    #[test]
    fn embedder_trait_object_is_dyn_compatible() {
        let e = ConstantEmbedder {
            values: vec![0.1, 0.2, 0.3],
        };
        let _b: Box<dyn Embedder> = Box::new(e);
    }

    #[test]
    fn embedder_default_batch_is_serial() {
        let e = ConstantEmbedder {
            values: vec![0.5; 4],
        };
        let inputs: Vec<&[f32]> = vec![&[][..], &[][..], &[][..]];
        let out = e.embed_batch(&inputs).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.len() == 4 && v[0] == 0.5));
    }

    #[test]
    fn embedder_dim_matches_output() {
        let e = ConstantEmbedder {
            values: vec![1.0; 192],
        };
        assert_eq!(e.dim(), 192);
        assert_eq!(e.embed(&[]).unwrap().len(), 192);
    }

    #[test]
    fn embedder_error_audio_too_short_displays() {
        let err = EmbedderError::AudioTooShort {
            actual_secs: 0.05,
            min_secs: 0.25,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0.05"));
        assert!(msg.contains("0.25"));
    }
}
