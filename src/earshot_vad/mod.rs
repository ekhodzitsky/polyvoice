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
//! [`EarshotVad`] buffers arbitrary caller chunk sizes and emits one speech
//! probability in `[0, 1]` per complete frame. Leftover samples are held until
//! the next [`VoiceActivityDetector::process`] call or cleared by [`reset`](VoiceActivityDetector::reset).
//!
//! The upstream score is continuous in `[0, 1]` (not a hard binary label);
//! thresholding is left to the caller (e.g. [`crate::vad::VadConfig::threshold`]).
//! Prefer [`crate::vad::VadConfig::frame_size`] = 256 when pairing with
//! [`crate::vad::segment_speech`] so sample indices align with model frames.
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
    /// Samples not yet filled to a complete [`FRAME_SIZE`] window.
    pending: Vec<f32>,
}

impl EarshotVad {
    /// Create a detector with a fresh earshot state and empty frame buffer.
    pub fn new() -> Self {
        Self {
            // Prefer heap construction: Detector is ~8 KiB of state.
            detector: earshot::Detector::default_boxed(),
            pending: Vec::with_capacity(FRAME_SIZE),
        }
    }

    /// Native frame length in samples (always 256).
    pub fn frame_size(&self) -> usize {
        FRAME_SIZE
    }

    /// Number of samples buffered and not yet scored.
    pub fn pending_samples(&self) -> usize {
        self.pending.len()
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
        self.pending.clear();
    }

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        // Buffer arbitrary chunk sizes into fixed 256-sample frames.
        // Unlike SileroVad / EnergyVad, incomplete tails are retained rather
        // than rejected with InvalidChunkSize — streaming-friendly.
        self.pending.extend_from_slice(samples);
        let n_frames = self.pending.len() / FRAME_SIZE;
        let mut probs = Vec::with_capacity(n_frames);

        let mut consumed = 0usize;
        while consumed + FRAME_SIZE <= self.pending.len() {
            let mut frame = [0.0f32; FRAME_SIZE];
            frame.copy_from_slice(&self.pending[consumed..consumed + FRAME_SIZE]);
            probs.push(self.score_frame(&frame)?);
            consumed += FRAME_SIZE;
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
        }
        debug_assert!(self.pending.len() < FRAME_SIZE);

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
        assert_eq!(vad.pending_samples(), 0);
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
    fn reset_clears_pending_and_state() {
        let mut vad = EarshotVad::new();
        // Partial frame left in the buffer.
        let odd = vec![0.1f32; 100];
        let probs = vad.process(&odd).expect("partial");
        assert!(probs.is_empty());
        assert_eq!(vad.pending_samples(), 100);

        vad.reset();
        assert_eq!(vad.pending_samples(), 0);

        // After reset, a fresh silence sequence must score low again (state gone).
        let silence = vec![0.0f32; FRAME_SIZE * 8];
        let probs = vad.process(&silence).expect("post-reset silence");
        assert_eq!(probs.len(), 8);
        let mean = probs.iter().sum::<f32>() / probs.len() as f32;
        assert!(mean < 0.5, "post-reset silence mean {mean}");
    }

    #[test]
    fn odd_chunk_sizes_do_not_crash() {
        let mut vad = EarshotVad::new();
        // Mix of sizes that are not multiples of 256, including 1 and empty.
        let sizes = [0usize, 1, 17, 100, 255, 256, 257, 511, 512, 1000, 3];
        let mut total_in = 0usize;
        let mut total_out = 0usize;
        for &n in &sizes {
            let chunk = speechish(n);
            total_in += n;
            let probs = vad.process(&chunk).expect("odd chunk");
            total_out += probs.len();
            assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
        let expected_frames = total_in / FRAME_SIZE;
        // Pending holds the remainder; completed frames must match floor division.
        assert_eq!(total_out, expected_frames);
        assert_eq!(vad.pending_samples(), total_in % FRAME_SIZE);
    }

    #[test]
    fn empty_input_returns_empty() {
        let mut vad = EarshotVad::new();
        let probs = vad.process(&[]).expect("empty");
        assert!(probs.is_empty());
    }

    #[test]
    fn multi_chunk_equals_single_pass() {
        let audio = speechish(FRAME_SIZE * 10 + 50);
        let mut a = EarshotVad::new();
        let mut b = EarshotVad::new();

        let once = a.process(&audio).expect("once");

        let mid = audio.len() / 3;
        let mut streamed = b.process(&audio[..mid]).expect("part1");
        streamed.extend(b.process(&audio[mid..]).expect("part2"));

        assert_eq!(once.len(), streamed.len());
        for (i, (x, y)) in once.iter().zip(streamed.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "frame {i}: single-pass {x} vs streamed {y}"
            );
        }
        assert_eq!(a.pending_samples(), b.pending_samples());
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
