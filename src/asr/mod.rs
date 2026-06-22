//! ASR (speech-to-text) trait — the stable interface the opt-in `polyvoice-asr`
//! companion crate implements and the word→speaker join targets.
//!
//! **Trait-only.** No backend, no `ort`/model dependency. Pure-Rust and
//! wasm-clean: the default build carries only this interface, never an ASR engine
//! (the actual model, e.g. Parakeet TDT via `parakeet-rs` at ~600 MB, lives in the
//! opt-in `polyvoice-asr` crate).

use crate::types::{SampleRate, Word};

/// A speech-to-text backend producing word-level timestamps.
///
/// Implemented by the opt-in `polyvoice-asr` crate. Object-safe — store as
/// `Box<dyn Asr>` and pass into the cascaded diarize→transcribe→join pipeline.
pub trait Asr: Send + Sync {
    /// Transcribe mono `audio` sampled at `sample_rate` into time-stamped words.
    fn transcribe(&self, audio: &[f32], sample_rate: SampleRate) -> Result<Vec<Word>, AsrError>;
}

/// Errors an [`Asr`] backend can return.
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("unsupported sample rate: {actual} Hz")]
    UnsupportedSampleRate { actual: u32 },

    #[error("ASR inference failed: {detail}")]
    InferenceFailed { detail: String },

    #[error("model file io error on {path}: {detail}")]
    ModelIo {
        path: std::path::PathBuf,
        detail: String,
    },

    #[error("ASR backend error: {0}")]
    Backend(String),
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeRange;

    struct MockAsr;
    impl Asr for MockAsr {
        fn transcribe(
            &self,
            _audio: &[f32],
            _sample_rate: SampleRate,
        ) -> Result<Vec<Word>, AsrError> {
            Ok(vec![Word {
                word: "hi".to_owned(),
                time: TimeRange {
                    start: 0.0,
                    end: 0.5,
                },
                confidence: 0.9,
            }])
        }
    }

    #[test]
    fn asr_is_object_safe() {
        let boxed: Box<dyn Asr> = Box::new(MockAsr);
        let sr = SampleRate::new(16000).unwrap();
        let words = boxed.transcribe(&[0.0_f32; 16], sr).unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hi");
    }

    #[test]
    fn asr_error_displays() {
        assert!(
            AsrError::UnsupportedSampleRate { actual: 8000 }
                .to_string()
                .contains("8000")
        );
    }
}
