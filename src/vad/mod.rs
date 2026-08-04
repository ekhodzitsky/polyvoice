//! Voice Activity Detection (VAD) trait and utilities.
//!
//! Use this module to detect speech regions in audio before embedding or
//! diarization. See [`VoiceActivityDetector`] for the trait and
//! [`segment_speech`] for the high-level helper.

pub(crate) mod hysteresis;

use crate::types::DiarizationConfig;
use hysteresis::{RegionEvent, RegionTracker, TailPolicy};

/// Trait for voice activity detectors.
///
/// Implementations are expected to be stateful and process audio in small
/// fixed-size windows (e.g. 512 samples for Silero VAD).
///
/// # Frame contract
///
/// Every implementation scores audio in fixed native frames of `F` samples
/// and follows the same input/output ratio in
/// [`process`](VoiceActivityDetector::process):
///
/// - `samples.len()` must be a multiple of `F` (empty input is allowed and
///   yields an empty vector). Anything else is rejected with
///   [`VadError::InvalidChunkSize`] — partial frames are never buffered or
///   silently dropped inside `process`. Callers that receive arbitrary chunk
///   sizes must accumulate samples up to a multiple of `F` themselves.
/// - The returned vector holds exactly `samples.len() / F` probabilities in
///   `[0, 1]`, one per native frame in input order: probability `i` covers
///   input samples `[i * F, (i + 1) * F)`. Pipelines that number frames to
///   derive timestamps rely on this ratio.
///
/// Native frame size `F` per implementation shipped in this crate:
///
/// - [`EnergyVad`] — the `frame_size` passed to [`EnergyVad::new`].
/// - `SileroVad` (feature `onnx`) — the `chunk_size` passed to
///   `SileroVad::new`.
/// - `EarshotVad` (feature `vad-earshot`) — 256 samples
///   (`earshot_vad::FRAME_SIZE`).
pub trait VoiceActivityDetector: Send {
    /// Reset internal state (LSTM buffers, etc.) for a new audio stream.
    fn reset(&mut self);

    /// Process a chunk of audio and return speech probability for each frame.
    ///
    /// See the trait-level [frame contract](VoiceActivityDetector#frame-contract)
    /// for the input-size requirement and the exact
    /// `probs.len() == samples.len() / F` ratio.
    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError>;

    /// Expected input sample rate.
    fn sample_rate(&self) -> u32;
}

#[derive(thiserror::Error, Debug)]
pub enum VadError {
    #[error("model error: {0}")]
    Model(String),
    #[error("invalid chunk size: expected multiple of {expected}, got {got}")]
    InvalidChunkSize { expected: usize, got: usize },
}

/// Configuration for voice activity detection.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Frame size in samples.
    pub frame_size: usize,
    /// Speech probability threshold.
    pub threshold: f32,
    /// Minimum silence duration to split segments (ms).
    pub min_silence_ms: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_size: 512,
            threshold: 0.5,
            min_silence_ms: 300.0,
        }
    }
}

/// Frame-level VAD geometry derived from the sample rate: frame duration
/// plus the duration-based limits expressed in whole frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrameGeometry {
    /// Duration of one frame in milliseconds.
    pub ms_per_frame: f32,
    /// [`VadConfig::min_silence_ms`] in whole frames (ceil).
    pub min_silence_frames: usize,
    /// Minimum speech duration (`min_speech_secs`) in whole frames (ceil).
    pub min_speech_frames: usize,
}

impl VadConfig {
    /// { true }
    /// `pub fn frame_geometry(&self, sample_rate: u32, min_speech_secs: f32) -> Result<VadFrameGeometry, VadError>`
    /// { ret.is_ok() == (self.frame_size > 0) }
    /// Derive the frame geometry at `sample_rate`: how long one frame is and
    /// how the configured durations translate to whole frames.
    ///
    /// `min_speech_secs` is the minimum speech duration to keep a region
    /// (see `SpeechFilterConfig::min_speech_secs`). Rejects `frame_size == 0`
    /// with [`VadError::InvalidChunkSize`].
    pub fn frame_geometry(
        &self,
        sample_rate: u32,
        min_speech_secs: f32,
    ) -> Result<VadFrameGeometry, VadError> {
        if self.frame_size == 0 {
            return Err(VadError::InvalidChunkSize {
                expected: 1,
                got: 0,
            });
        }
        let ms_per_frame = (self.frame_size as f32 / sample_rate as f32) * 1000.0;
        Ok(VadFrameGeometry {
            ms_per_frame,
            min_silence_frames: (self.min_silence_ms / ms_per_frame).ceil() as usize,
            min_speech_frames: ((min_speech_secs * 1000.0) / ms_per_frame).ceil() as usize,
        })
    }
}

/// A simple energy-based VAD for tests and fallback scenarios.
pub struct EnergyVad {
    threshold: f32,
    sample_rate: u32,
    frame_size: usize,
}

impl EnergyVad {
    /// { frame_size > 0 }
    /// pub fn new(threshold_db: f32, sample_rate: u32, frame_size: usize) -> Self
    /// { true }
    /// Create an energy-based voice activity detector.
    ///
    /// `threshold_db` is the energy threshold in dB (converted internally to linear).
    /// `frame_size` must be a positive multiple of the expected chunk size.
    ///
    /// ```rust
    /// use polyvoice::{EnergyVad, VoiceActivityDetector};
    /// let vad = EnergyVad::new(-40.0, 16000, 512);
    /// assert_eq!(vad.sample_rate(), 16000);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `frame_size == 0`.
    /// Use [`try_new`](Self::try_new) for a fallible alternative.
    #[allow(clippy::panic)] // Documented convenience over `try_new`.
    pub fn new(threshold_db: f32, sample_rate: u32, frame_size: usize) -> Self {
        match Self::try_new(threshold_db, sample_rate, frame_size) {
            Ok(vad) => vad,
            Err(_) => panic!("EnergyVad::new: frame_size must be > 0"),
        }
    }

    /// { true }
    /// `pub fn try_new(threshold_db: f32, sample_rate: u32, frame_size: usize) -> Result<Self, VadError>`
    /// { ret.is_ok() == (frame_size > 0) }
    /// Fallible constructor: rejects `frame_size == 0` with
    /// [`VadError::InvalidChunkSize`] instead of panicking.
    pub fn try_new(
        threshold_db: f32,
        sample_rate: u32,
        frame_size: usize,
    ) -> Result<Self, VadError> {
        if frame_size == 0 {
            return Err(VadError::InvalidChunkSize {
                expected: 1,
                got: 0,
            });
        }
        Ok(Self {
            threshold: 10f32.powf(threshold_db / 20.0),
            sample_rate,
            frame_size,
        })
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn reset(&mut self) {}

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        if !samples.len().is_multiple_of(self.frame_size) {
            return Err(VadError::InvalidChunkSize {
                expected: self.frame_size,
                got: samples.len(),
            });
        }
        let mut probs = Vec::with_capacity(samples.len() / self.frame_size);
        for chunk in samples.chunks(self.frame_size) {
            let energy: f32 = chunk.iter().map(|s| s * s).sum::<f32>().sqrt();
            let prob = (energy / self.threshold).min(1.0);
            probs.push(prob);
        }
        Ok(probs)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Event emitted by [`VadStateMachine`] when the speech state changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// A speech region started at the given frame index.
    SpeechStart { start_frame: usize },
    /// A speech region ended. `end_frame` is exclusive.
    SpeechEnd {
        start_frame: usize,
        end_frame: usize,
    },
}

/// Incremental speech-region detector.
///
/// Maintains the same state machine as [`segment_speech`] but operates
/// frame-by-frame. Useful for both batch and streaming pipelines.
///
/// Built on the shared scalar hysteresis core in `hysteresis`: a region
/// stays open through silence shorter than `min_silence_frames` and closes
/// *after* the closing silence run. Events always alternate
/// [`VadEvent::SpeechStart`]/[`VadEvent::SpeechEnd`]; short-region
/// suppression is applied by callers via
/// [`meets_min_speech_duration`](VadStateMachine::meets_min_speech_duration).
#[derive(Debug, Clone)]
pub struct VadStateMachine {
    threshold: f32,
    tracker: RegionTracker,
}

impl VadStateMachine {
    /// { true }
    /// `pub fn new(threshold: f32, min_silence_frames: usize, min_speech_frames: usize) -> Self`
    /// { true }
    /// Create a new state machine.
    pub fn new(threshold: f32, min_silence_frames: usize, min_speech_frames: usize) -> Self {
        Self {
            threshold,
            tracker: RegionTracker::new(min_silence_frames, min_speech_frames, TailPolicy::Keep),
        }
    }

    /// { true }
    /// `pub fn advance(&mut self, prob: f32, frame: usize) -> Option<VadEvent>`
    /// { true }
    /// Advance by one frame probability.
    ///
    /// Returns [`VadEvent::SpeechStart`] when speech begins and
    /// [`VadEvent::SpeechEnd`] when a speech region completes (silence
    /// exceeded `min_silence_frames`).
    pub fn advance(&mut self, prob: f32, frame: usize) -> Option<VadEvent> {
        // Historical threshold semantics: inside a region only a frame
        // strictly below the threshold counts toward silence, so a NaN
        // probability reads as speech inside a region and as silence outside.
        let active = if self.tracker.in_region() {
            prob >= self.threshold || prob.is_nan()
        } else {
            prob >= self.threshold
        };
        match self.tracker.advance(active, frame) {
            Some(RegionEvent::Start { start_frame }) => Some(VadEvent::SpeechStart { start_frame }),
            Some(RegionEvent::End {
                start_frame,
                end_frame,
            }) => Some(VadEvent::SpeechEnd {
                start_frame,
                end_frame,
            }),
            None => None,
        }
    }

    /// { true }
    /// `pub fn flush(&mut self, frame: usize) -> Option<VadEvent>`
    /// { !self.in_speech() }
    /// Finalize any in-flight speech region.
    ///
    /// Returns [`VadEvent::SpeechEnd`] if a region was active.
    pub fn flush(&mut self, frame: usize) -> Option<VadEvent> {
        match self.tracker.flush(frame) {
            Some(RegionEvent::End {
                start_frame,
                end_frame,
            }) => Some(VadEvent::SpeechEnd {
                start_frame,
                end_frame,
            }),
            Some(RegionEvent::Start { .. }) | None => None,
        }
    }

    /// { true }
    /// `pub fn in_speech(&self) -> bool`
    /// { ret == self.in_speech() }
    /// Whether the detector is currently inside a speech region.
    pub fn in_speech(&self) -> bool {
        self.tracker.in_region()
    }

    /// { true }
    /// `pub fn min_speech_frames(&self) -> usize`
    /// { ret == self.min_speech_frames() }
    /// Minimum speech frames required for a region to be emitted.
    pub fn min_speech_frames(&self) -> usize {
        self.tracker.min_on()
    }

    /// { true }
    /// `pub fn meets_min_speech_duration(&self, start_frame: usize, end_frame: usize) -> bool`
    /// { ret == (end_frame - start_frame >= self.min_speech_frames()) }
    /// Whether a region spanning `[start_frame, end_frame)` survives the
    /// minimum speech-duration filter.
    ///
    /// Single point for the short-region suppression rule shared by
    /// [`segment_speech`] and the streaming pipeline. The machine always
    /// emits `SpeechEnd` for a closed region — swallowing the event would
    /// break the start/end alternation — so callers apply this predicate to
    /// decide whether the region produces output.
    pub fn meets_min_speech_duration(&self, start_frame: usize, end_frame: usize) -> bool {
        self.tracker.keeps(start_frame, end_frame)
    }
}

/// { true }
/// `pub fn segment_speech<V: VoiceActivityDetector>( vad: &mut V, samples: &[f32], config: &DiarizationConfig, vad_config: &VadConfig, ) -> Result<Vec<(usize, usize)>, VadError>`
/// { ret.as_ref().map_or(true, |v| v.iter().all(|(s, e)| s < e)) }
/// Segment speech regions using a voice activity detector.
///
/// Returns a list of `(start_sample, end_sample)` pairs where speech was detected.
///
/// Only complete frames are scored: the trailing `samples.len() % frame_size`
/// samples (a partial tail) are dropped, not padded. Pad upstream or choose a
/// dividing `frame_size` when the tail matters.
///
/// `vad_config.frame_size` should equal the detector's native frame size (see
/// the [frame contract](VoiceActivityDetector#frame-contract)) so every frame
/// yields exactly one probability and segment sample indices stay aligned
/// with the input.
///
/// Frame durations are derived from the detector's own
/// [`sample_rate`](VoiceActivityDetector::sample_rate) — the samples being
/// framed are consumed by the detector, so they are at its rate. `config`
/// supplies only the speech-filter duration (`min_speech_secs`).
///
/// ```rust
/// use polyvoice::{EnergyVad, segment_speech, DiarizationConfig, VadConfig};
/// let mut vad = EnergyVad::new(-40.0, 16000, 512);
/// let samples = vec![0.5f32; 16000]; // 1 second of "loud" audio
/// let config = DiarizationConfig::default();
/// let vad_config = VadConfig::default();
/// let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();
/// assert!(!segs.is_empty());
/// assert!(segs.iter().all(|(s, e)| s < e));
/// ```
pub fn segment_speech<V: VoiceActivityDetector>(
    vad: &mut V,
    samples: &[f32],
    config: &DiarizationConfig,
    vad_config: &VadConfig,
) -> Result<Vec<(usize, usize)>, VadError> {
    vad.reset();
    let frame_size = vad_config.frame_size;
    let geometry =
        vad_config.frame_geometry(vad.sample_rate(), config.speech_filter.min_speech_secs)?;
    let num_frames = samples.len() / frame_size;
    let mut probs = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let chunk = &samples[i * frame_size..(i + 1) * frame_size];
        let frame_probs = vad.process(chunk)?;
        probs.extend(frame_probs);
    }

    let mut sm = VadStateMachine::new(
        vad_config.threshold,
        geometry.min_silence_frames,
        geometry.min_speech_frames,
    );
    let mut segments = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
        if let Some(VadEvent::SpeechEnd {
            start_frame,
            end_frame,
        }) = sm.advance(prob, i)
            && sm.meets_min_speech_duration(start_frame, end_frame)
        {
            segments.push((start_frame * frame_size, end_frame * frame_size));
        }
    }

    if let Some(VadEvent::SpeechEnd {
        start_frame,
        end_frame,
    }) = sm.flush(num_frames)
        && sm.meets_min_speech_duration(start_frame, end_frame)
    {
        segments.push((start_frame * frame_size, end_frame * frame_size));
    }

    Ok(segments)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_vad_process_high_energy() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![0.5f32; 512];
        let probs = vad.process(&samples).unwrap();
        assert_eq!(probs.len(), 1);
        assert!(
            probs[0] > 0.9,
            "high energy should give prob > 0.9, got {}",
            probs[0]
        );
    }

    #[test]
    fn energy_vad_process_low_energy() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        // threshold = 10^(-40/20) = 0.01
        // energy = sqrt(512 * amplitude^2) must be < 0.001 for prob < 0.1
        // amplitude < 0.001 / sqrt(512) ≈ 4.4e-5
        let samples = vec![1e-5f32; 512];
        let probs = vad.process(&samples).unwrap();
        assert_eq!(probs.len(), 1);
        assert!(
            probs[0] < 0.1,
            "low energy should give prob < 0.1, got {}",
            probs[0]
        );
    }

    #[test]
    fn energy_vad_invalid_chunk_size() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![0.5f32; 256]; // not a multiple of 512
        let err = vad.process(&samples).unwrap_err();
        match err {
            VadError::InvalidChunkSize {
                expected: 512,
                got: 256,
            } => {}
            other => panic!("expected InvalidChunkSize(512, 256), got {:?}", other),
        }
    }

    #[test]
    fn energy_vad_multiple_chunks() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![0.5f32; 512 * 4];
        let probs = vad.process(&samples).unwrap();
        assert_eq!(probs.len(), 4);
        assert!(probs.iter().all(|&p| p > 0.9));
    }

    #[test]
    fn vad_state_machine_advance_speech_start() {
        let mut sm = VadStateMachine::new(0.5, 3, 1);
        assert!(!sm.in_speech());
        let event = sm.advance(0.6, 0);
        assert_eq!(event, Some(VadEvent::SpeechStart { start_frame: 0 }));
        assert!(sm.in_speech());
    }

    #[test]
    fn vad_state_machine_advance_speech_end_after_silence() {
        let mut sm = VadStateMachine::new(0.5, 3, 1);
        sm.advance(0.6, 0); // SpeechStart
        sm.advance(0.6, 1);
        sm.advance(0.6, 2);
        // silence frames
        sm.advance(0.1, 3);
        sm.advance(0.1, 4);
        let event = sm.advance(0.1, 5); // 3rd silence frame → SpeechEnd
        assert_eq!(
            event,
            Some(VadEvent::SpeechEnd {
                start_frame: 0,
                end_frame: 6,
            })
        );
        assert!(!sm.in_speech());
    }

    #[test]
    fn vad_state_machine_silence_count_resets_on_speech() {
        let mut sm = VadStateMachine::new(0.5, 3, 1);
        sm.advance(0.6, 0); // SpeechStart
        sm.advance(0.1, 1); // silence 1
        sm.advance(0.1, 2); // silence 2
        sm.advance(0.6, 3); // back to speech → reset silence_count
        sm.advance(0.1, 4); // silence 1
        sm.advance(0.1, 5); // silence 2
        let event = sm.advance(0.1, 6); // silence 3 → SpeechEnd
        assert_eq!(
            event,
            Some(VadEvent::SpeechEnd {
                start_frame: 0,
                end_frame: 7,
            })
        );
    }

    #[test]
    fn vad_state_machine_flush_during_speech() {
        let mut sm = VadStateMachine::new(0.5, 3, 1);
        sm.advance(0.6, 0); // SpeechStart
        let event = sm.flush(5);
        assert_eq!(
            event,
            Some(VadEvent::SpeechEnd {
                start_frame: 0,
                end_frame: 5,
            })
        );
        assert!(!sm.in_speech());
    }

    #[test]
    fn vad_state_machine_flush_when_silent() {
        let mut sm = VadStateMachine::new(0.5, 3, 1);
        let event = sm.flush(10);
        assert_eq!(event, None);
        assert!(!sm.in_speech());
    }

    #[test]
    fn vad_state_machine_exposes_min_speech_frames() {
        let sm = VadStateMachine::new(0.5, 3, 7);
        assert_eq!(sm.min_speech_frames(), 7);
        assert!(sm.meets_min_speech_duration(0, 7));
        assert!(!sm.meets_min_speech_duration(0, 6));
    }

    #[test]
    fn segment_speech_empty_samples() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples: Vec<f32> = vec![];
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn segment_speech_all_silence() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![1e-5f32; 16000]; // 1 second of very low energy
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn segment_speech_sustained_loud() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![0.5f32; 16000 * 3]; // 3 seconds of loud audio
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();
        assert!(!segs.is_empty());
        assert!(segs.iter().all(|(s, e)| s < e));
    }

    #[test]
    fn segment_speech_ignores_partial_trailing_chunk() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        // 768 = 512 + 256 — trailing 256 samples are ignored by segment_speech
        let samples = vec![0.5f32; 768];
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let segs = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap();
        // Only the first 512 samples are processed, which is 1 frame → may or may not be
        // enough for a segment depending on min_speech_frames. The key point is that
        // it does NOT error and the trailing partial chunk is silently ignored.
        assert!(segs.iter().all(|(s, e)| s < e));
    }

    #[test]
    fn segment_speech_rejects_zero_frame_size() {
        let mut vad = EnergyVad::new(-40.0, 16000, 512);
        let samples = vec![0.5f32; 512];
        let config = DiarizationConfig::default();
        let vad_config = VadConfig {
            frame_size: 0,
            ..Default::default()
        };
        let err = segment_speech(&mut vad, &samples, &config, &vad_config).unwrap_err();
        assert!(matches!(err, VadError::InvalidChunkSize { got: 0, .. }));
    }

    #[test]
    #[should_panic(expected = "EnergyVad::new: frame_size must be > 0")]
    fn energy_vad_rejects_zero_frame_size() {
        let _ = EnergyVad::new(-40.0, 16000, 0);
    }

    #[test]
    fn energy_vad_try_new_rejects_zero_frame_size() {
        let res = EnergyVad::try_new(-40.0, 16000, 0);
        assert!(matches!(
            res,
            Err(VadError::InvalidChunkSize { got: 0, .. })
        ));
        assert!(EnergyVad::try_new(-40.0, 16000, 512).is_ok());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    /// Generates valid sample vectors whose length is a multiple of `frame_size`.
    fn valid_samples(frame_size: usize) -> impl Strategy<Value = Vec<f32>> {
        (0usize..=64usize)
            .prop_map(move |n| n * frame_size)
            .prop_flat_map(move |len| prop::collection::vec(-1.0f32..=1.0f32, len))
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// EnergyVad::process never panics on valid chunk-sized input and
        /// returns probabilities in [0, 1].
        #[test]
        fn energy_vad_process_never_panics(
            samples in valid_samples(512),
        ) {
            let mut vad = EnergyVad::new(-40.0, 16000, 512);
            let result = vad.process(&samples);
            if let Ok(probs) = result {
                prop_assert_eq!(probs.len(), samples.len() / 512);
                prop_assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)),
                    "probabilities must be in [0, 1]");
            }
        }

        /// segment_speech never panics and returns valid segments.
        #[test]
        fn segment_speech_never_panics_and_segments_valid(
            samples in prop::collection::vec(-1.0f32..=1.0f32, 0..=16000),
        ) {
            let mut vad = EnergyVad::new(-40.0, 16000, 512);
            let config = DiarizationConfig::default();
            let vad_config = VadConfig::default();

            let result = segment_speech(&mut vad, &samples, &config, &vad_config);

            match result {
                Ok(segs) => {
                    prop_assert!(
                        segs.iter().all(|(s, e)| s < e),
                        "all segments must have start < end"
                    );
                }
                Err(_) => {
                    // Err is acceptable (e.g. downstream VAD may reject chunk size),
                    // but we must never panic.
                }
            }
        }

        /// VadStateMachine maintains invariants across random parameters and
        /// probability sequences.
        #[test]
        fn vad_state_machine_invariants(
            threshold in 0.0f32..=1.0f32,
            min_silence_frames in 0usize..=10usize,
            min_speech_frames in 0usize..=10usize,
            probs in prop::collection::vec(0.0f32..=1.0f32, 0..=128usize),
        ) {
            let mut sm = VadStateMachine::new(threshold, min_silence_frames, min_speech_frames);
            let mut in_speech_after_flush = false;

            for (i, &prob) in probs.iter().enumerate() {
                if let Some(event) = sm.advance(prob, i) {
                    match event {
                        VadEvent::SpeechStart { start_frame } => {
                            prop_assert!(
                                !in_speech_after_flush,
                                "SpeechStart without preceding SpeechEnd at frame {}", start_frame
                            );
                            in_speech_after_flush = true;
                        }
                        VadEvent::SpeechEnd { start_frame, end_frame } => {
                            prop_assert!(
                                in_speech_after_flush,
                                "SpeechEnd without preceding SpeechStart"
                            );
                            prop_assert!(
                                start_frame < end_frame,
                                "SpeechEnd: start_frame {} must be < end_frame {}",
                                start_frame, end_frame
                            );
                            in_speech_after_flush = false;
                        }
                    }
                }
            }

            if let Some(VadEvent::SpeechEnd { start_frame, end_frame }) = sm.flush(probs.len()) {
                prop_assert!(
                    start_frame < end_frame,
                    "flush SpeechEnd: start_frame {} must be < end_frame {}",
                    start_frame, end_frame
                );
            }
            prop_assert!(
                !sm.in_speech(),
                "after flush in_speech must be false"
            );
        }
    }
}
