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
/// { TODO: precondition }
/// pub fn new(threshold_db: f32, sample_rate: u32, frame_size: usize) -> Self
/// { TODO: postcondition }
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
            return Some(VadEvent::SpeechStart {
                start_frame: frame,
            });
        }
        None
    }

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

    /// Whether the detector is currently inside a speech region.
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Minimum speech frames required for a region to be emitted.
    pub fn min_speech_frames(&self) -> usize {
        self.min_speech_frames
    }
}

/// { TODO: precondition }
/// `pub fn segment_speech<V: VoiceActivityDetector>( vad: &mut V, samples: &[f32], config: &DiarizationConfig, vad_config: &VadConfig, ) -> Result<Vec<(usize, usize)>, VadError>`
/// { TODO: postcondition }
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
    let min_speech_frames = ((config.speech_filter.min_speech_secs * 1000.0) / ms_per_frame).ceil() as usize;
    let threshold = vad_config.threshold;
    let min_silence_frames = (vad_config.min_silence_ms / ms_per_frame).ceil() as usize;

    let mut sm = VadStateMachine::new(threshold, min_silence_frames, min_speech_frames);
    let mut segments = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
        if let Some(VadEvent::SpeechEnd { start_frame, end_frame }) = sm.advance(prob, i) {
            let duration_frames = end_frame - start_frame;
            if duration_frames >= min_speech_frames {
                segments.push((start_frame * frame_size, end_frame * frame_size));
            }
        }
    }

    if let Some(VadEvent::SpeechEnd { start_frame, end_frame }) = sm.flush(num_frames) {
        let duration_frames = end_frame - start_frame;
        if duration_frames >= min_speech_frames {
            segments.push((start_frame * frame_size, end_frame * frame_size));
        }
    }

    Ok(segments)
}
