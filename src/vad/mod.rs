//! Voice Activity Detection (VAD) trait and utilities.
//!
//! Use this module to detect speech regions in audio before embedding or
//! diarization. See [`VoiceActivityDetector`] for the trait and
//! [`segment_speech`] for the high-level helper.

use crate::types::DiarizationConfig;

/// Trait for voice activity detectors.
///
/// Implementations are expected to be stateful and process audio in small
/// fixed-size windows (e.g. 512 samples for Silero VAD).
pub trait VoiceActivityDetector: Send {
    /// Reset internal state (LSTM buffers, etc.) for a new audio stream.
    fn reset(&mut self);

    /// Process a chunk of audio and return speech probability for each frame.
    ///
    /// The returned vector has one probability per analysis frame within the chunk.
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
    pub fn new(threshold_db: f32, sample_rate: u32, frame_size: usize) -> Self {
        Self {
            threshold: 10f32.powf(threshold_db / 20.0),
            sample_rate,
            frame_size,
        }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn reset(&mut self) {}

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        if samples.len() % self.frame_size != 0 {
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
#[derive(Debug, Clone)]
pub struct VadStateMachine {
    threshold: f32,
    min_silence_frames: usize,
    min_speech_frames: usize,
    in_speech: bool,
    seg_start_frame: usize,
    silence_count: usize,
}

impl VadStateMachine {
    /// { true }
    /// `pub fn new(threshold: f32, min_silence_frames: usize, min_speech_frames: usize) -> Self`
    /// { true }
    /// Create a new state machine.
    pub fn new(threshold: f32, min_silence_frames: usize, min_speech_frames: usize) -> Self {
        Self {
            threshold,
            min_silence_frames,
            min_speech_frames,
            in_speech: false,
            seg_start_frame: 0,
            silence_count: 0,
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
        if self.in_speech {
            if prob < self.threshold {
                self.silence_count += 1;
                if self.silence_count >= self.min_silence_frames {
                    let event = VadEvent::SpeechEnd {
                        start_frame: self.seg_start_frame,
                        end_frame: frame + 1,
                    };
                    self.in_speech = false;
                    self.silence_count = 0;
                    return Some(event);
                }
            } else {
                self.silence_count = 0;
            }
        } else if prob >= self.threshold {
            self.in_speech = true;
            self.seg_start_frame = frame;
            self.silence_count = 0;
            return Some(VadEvent::SpeechStart { start_frame: frame });
        }
        None
    }

    /// { true }
    /// `pub fn flush(&mut self, frame: usize) -> Option<VadEvent>`
    /// { !self.in_speech }
    /// Finalize any in-flight speech region.
    ///
    /// Returns [`VadEvent::SpeechEnd`] if a region was active.
    pub fn flush(&mut self, frame: usize) -> Option<VadEvent> {
        if self.in_speech {
            let event = VadEvent::SpeechEnd {
                start_frame: self.seg_start_frame,
                end_frame: frame,
            };
            self.in_speech = false;
            self.silence_count = 0;
            return Some(event);
        }
        None
    }

    /// { true }
    /// `pub fn in_speech(&self) -> bool`
    /// { ret == self.in_speech }
    /// Whether the detector is currently inside a speech region.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// { true }
    /// `pub fn min_speech_frames(&self) -> usize`
    /// { ret == self.min_speech_frames }
    /// Minimum speech frames required for a region to be emitted.
    pub fn min_speech_frames(&self) -> usize {
        self.min_speech_frames
    }
}

/// { true }
/// `pub fn segment_speech<V: VoiceActivityDetector>( vad: &mut V, samples: &[f32], config: &DiarizationConfig, vad_config: &VadConfig, ) -> Result<Vec<(usize, usize)>, VadError>`
/// { ret.as_ref().map_or(true, |v| v.iter().all(|(s, e)| s < e)) }
/// Segment speech regions using a voice activity detector.
///
/// Returns a list of `(start_sample, end_sample)` pairs where speech was detected.
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
    let num_frames = samples.len() / frame_size;
    let mut probs = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let chunk = &samples[i * frame_size..(i + 1) * frame_size];
        let frame_probs = vad.process(chunk)?;
        probs.extend(frame_probs);
    }

    let sr = config.window.sample_rate.get() as f32;
    let ms_per_frame = (frame_size as f32 / sr) * 1000.0;
    let min_speech_frames =
        ((config.speech_filter.min_speech_secs * 1000.0) / ms_per_frame).ceil() as usize;
    let threshold = vad_config.threshold;
    let min_silence_frames = (vad_config.min_silence_ms / ms_per_frame).ceil() as usize;

    let mut sm = VadStateMachine::new(threshold, min_silence_frames, min_speech_frames);
    let mut segments = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
        if let Some(VadEvent::SpeechEnd {
            start_frame,
            end_frame,
        }) = sm.advance(prob, i)
        {
            let duration_frames = end_frame - start_frame;
            if duration_frames >= min_speech_frames {
                segments.push((start_frame * frame_size, end_frame * frame_size));
            }
        }
    }

    if let Some(VadEvent::SpeechEnd {
        start_frame,
        end_frame,
    }) = sm.flush(num_frames)
    {
        let duration_frames = end_frame - start_frame;
        if duration_frames >= min_speech_frames {
            segments.push((start_frame * frame_size, end_frame * frame_size));
        }
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
