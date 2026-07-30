//! Optional pure-Rust VAD backend via [earshot](https://crates.io/crates/earshot).
//!
//! # Feature gate
//!
//! Enabled only with `--features vad-earshot`. Core builds without this feature
//! never reference earshot (verify with `cargo tree -e normal | rg earshot`).
//!
//! # Default VAD
//!
//! **Silero remains the production default** and the DER-parity reference.
//! This adapter is opt-in for experimentation and embedded/no-runtime paths.
//! A default switch is out of scope until a measured parity gate passes (see
//! `benchmarks/results/earshot-vad-notes.md`).
//!
//! # Frame contract
//!
//! earshot scores mono PCM at 16 kHz in fixed **256-sample** windows (16 ms).
//! [`EarshotVad`] follows the crate-wide
//! [frame contract](crate::vad::VoiceActivityDetector#frame-contract):
//! [`VoiceActivityDetector::process`] accepts only multiples of
//! [`FRAME_SIZE`](crate::earshot_vad::FRAME_SIZE) samples and emits exactly
//! one speech probability in `[0, 1]` per complete frame. Partial chunks are
//! rejected with [`VadError::InvalidChunkSize`] — there is no hidden buffering
//! inside `process`, so frame indices always line up with the caller's input.
//! Callers with arbitrary chunk sizes must accumulate samples themselves
//! (both [`crate::vad::segment_speech`] and the streaming pipeline already
//! frame audio before calling `process`).
//!
//! The upstream score is continuous in `[0, 1]` (not a hard binary label);
//! thresholding is left to the caller (e.g. [`crate::vad::VadConfig::threshold`]).
//! Set [`crate::vad::VadConfig::frame_size`] = 256 when pairing with
//! [`crate::vad::segment_speech`] or a streaming pipeline so pipeline frame
//! indices align with model frames.
//!
//! # Watch list (not implemented here)
//!
//! - **TEN-VAD** — forbidden (Apache-looking license with a non-compete rider).
//! - **FireRedVAD** — candidate only when an ONNX export exists.

use crate::vad::{VadError, VoiceActivityDetector};

/// Adapter type id for [`crate::models::AdapterRegistry`] (`AdapterStage::Vad`).
pub const ADAPTER_TYPE: &str = "earshot";

/// Native analysis frame length expected by earshot (samples @ 16 kHz).
pub const FRAME_SIZE: usize = 256;

/// Sample rate required by earshot.
pub const SAMPLE_RATE: u32 = 16_000;

/// Voice activity detector backed by pure-Rust earshot.
///
/// Construct with [`EarshotVad::new`]. The inner detector is heap-allocated
/// (~8 KiB state) so construction does not blow small stacks.
pub struct EarshotVad {
    detector: Box<earshot::Detector>,
}

impl EarshotVad {
    /// Create a detector with a fresh earshot state.
    pub fn new() -> Self {
        Self {
            // Prefer heap construction: Detector is ~8 KiB of state.
            detector: earshot::Detector::default_boxed(),
        }
    }

    /// Native frame length in samples (always 256).
    pub fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    fn score_frame(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        debug_assert_eq!(frame.len(), FRAME_SIZE);
        let score = self.detector.predict_f32(frame);
        // earshot returns -1.0 only when the frame length is wrong (debug path).
        if !(0.0..=1.0).contains(&score) {
            return Err(VadError::Model(format!(
                "earshot returned out-of-range score {score}"
            )));
        }
        Ok(score)
    }
}

impl Default for EarshotVad {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceActivityDetector for EarshotVad {
    fn reset(&mut self) {
        self.detector.reset();
    }

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        // Same reject contract as EnergyVad / SileroVad: input must be a
        // multiple of the native 256-sample frame; incomplete chunks are an
        // error, never buffered across calls.
        if !samples.len().is_multiple_of(FRAME_SIZE) {
            return Err(VadError::InvalidChunkSize {
                expected: FRAME_SIZE,
                got: samples.len(),
            });
        }
        let mut probs = Vec::with_capacity(samples.len() / FRAME_SIZE);
        for frame in samples.chunks_exact(FRAME_SIZE) {
            probs.push(self.score_frame(frame)?);
        }
        Ok(probs)
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}

/// Register the earshot adapter type with an [`crate::models::AdapterRegistry`].
///
/// Name-marker only (same pattern as Sortformer): concrete construction is
/// [`EarshotVad::new`]. Safe to call once; returns
/// [`crate::models::AdapterError::AlreadyRegistered`] if the id is already present.
#[cfg(feature = "download")]
pub fn register_with(
    registry: &mut crate::models::AdapterRegistry,
) -> Result<(), crate::models::AdapterError> {
    use crate::models::{AdapterFactory, AdapterStage, BuiltinAdapter};
    use std::sync::Arc;

    let factory: AdapterFactory = Arc::new(|| {
        Box::new(BuiltinAdapter {
            stage: AdapterStage::Vad,
            id: ADAPTER_TYPE.to_owned(),
        })
    });
    registry.register(AdapterStage::Vad, ADAPTER_TYPE, factory)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::vad::VoiceActivityDetector;

    /// Synthetic "speech-ish" signal: multi-formant tone with AM, peak |x| ≤ 1.
    fn speechish(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                let f0 = 180.0;
                let carrier = (2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * (2.0 * f0) * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * (3.0 * f0) * t).sin();
                let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
                (carrier * envelope * 0.35).clamp(-1.0, 1.0)
            })
            .collect()
    }

    #[test]
    fn construct_and_sample_rate() {
        let vad = EarshotVad::new();
        assert_eq!(vad.sample_rate(), 16_000);
        assert_eq!(vad.frame_size(), 256);
        assert_eq!(ADAPTER_TYPE, "earshot");
    }

    #[test]
    fn silence_scores_low() {
        let mut vad = EarshotVad::new();
        // ~0.5 s of zeros → 31 full frames (31 * 256 = 7936)
        let silence = vec![0.0f32; FRAME_SIZE * 31];
        let probs = vad.process(&silence).expect("silence process");
        assert_eq!(probs.len(), 31);
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        let mean = probs.iter().sum::<f32>() / probs.len() as f32;
        assert!(
            mean < 0.5,
            "expected silence mean score < 0.5, got {mean} ({probs:?})"
        );
    }

    #[test]
    fn speech_scores_higher_than_silence() {
        let mut vad = EarshotVad::new();
        let silence = vec![0.0f32; FRAME_SIZE * 40];
        let speech = speechish(FRAME_SIZE * 40);

        let silence_probs = vad.process(&silence).expect("silence");
        vad.reset();
        let speech_probs = vad.process(&speech).expect("speech");

        assert_eq!(silence_probs.len(), speech_probs.len());
        let silence_mean = silence_probs.iter().sum::<f32>() / silence_probs.len() as f32;
        let speech_mean = speech_probs.iter().sum::<f32>() / speech_probs.len() as f32;
        // Absolute thresholds are model-dependent; relative ordering is the contract.
        assert!(
            speech_mean > silence_mean,
            "speech mean ({speech_mean}) should exceed silence mean ({silence_mean})"
        );
        assert!(speech_probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut vad = EarshotVad::new();
        // Partial chunks are rejected outright; nothing is buffered.
        let odd = vec![0.1f32; 100];
        let err = vad
            .process(&odd)
            .expect_err("partial chunk must be rejected");
        assert!(matches!(
            err,
            VadError::InvalidChunkSize {
                expected: FRAME_SIZE,
                got: 100
            }
        ));

        // After reset, the same input must score exactly as on a fresh
        // detector (internal model state is gone).
        let silence = vec![0.0f32; FRAME_SIZE * 8];
        let first = vad.process(&silence).expect("silence");
        vad.reset();
        let second = vad.process(&silence).expect("post-reset silence");
        assert_eq!(first, second, "reset must restore initial detector state");
    }

    #[test]
    fn partial_chunks_are_rejected() {
        let mut vad = EarshotVad::new();
        // Any length that is not a multiple of FRAME_SIZE is an error, and a
        // rejected chunk leaves no residue that could affect the next call.
        for &n in &[1usize, 17, 100, 255, 257, 511, 1000] {
            let chunk = speechish(n);
            let err = vad.process(&chunk).expect_err("partial chunk");
            assert!(matches!(
                err,
                VadError::InvalidChunkSize {
                    expected: FRAME_SIZE,
                    got
                } if got == n
            ));
        }
        let full = speechish(FRAME_SIZE * 3);
        let probs = vad.process(&full).expect("aligned chunk");
        assert_eq!(probs.len(), 3);
    }

    #[test]
    fn aligned_input_yields_one_prob_per_frame() {
        let mut vad = EarshotVad::new();
        for frames in [1usize, 2, 5] {
            let audio = speechish(FRAME_SIZE * frames);
            let probs = vad.process(&audio).expect("aligned");
            assert_eq!(probs.len(), frames);
            assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let mut vad = EarshotVad::new();
        let probs = vad.process(&[]).expect("empty");
        assert!(probs.is_empty());
    }

    #[test]
    fn multi_chunk_equals_single_pass() {
        let audio = speechish(FRAME_SIZE * 10);
        let mut a = EarshotVad::new();
        let mut b = EarshotVad::new();

        let once = a.process(&audio).expect("once");

        // Split on a frame boundary: chunked scoring must match single-pass.
        let mid = FRAME_SIZE * 4;
        let mut streamed = b.process(&audio[..mid]).expect("part1");
        streamed.extend(b.process(&audio[mid..]).expect("part2"));

        assert_eq!(once.len(), streamed.len());
        for (i, (x, y)) in once.iter().zip(streamed.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "frame {i}: single-pass {x} vs streamed {y}"
            );
        }
    }

    #[cfg(feature = "download")]
    #[test]
    fn register_with_empty_registry() {
        let mut reg = crate::models::AdapterRegistry::new();
        register_with(&mut reg).unwrap();
        assert!(reg.contains(crate::models::AdapterStage::Vad, ADAPTER_TYPE));
        let err = register_with(&mut reg).expect_err("duplicate");
        assert!(matches!(
            err,
            crate::models::AdapterError::AlreadyRegistered { .. }
        ));
    }
}
