//! Voice Activity Detection trait and utilities.

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

/// A simple energy-based VAD for tests and fallback scenarios.
pub struct EnergyVad {
    threshold: f32,
    sample_rate: u32,
}

impl EnergyVad {
    /// { sample_rate >= 8000 }
    /// `fn new(threshold_db: f32, sample_rate: u32) -> Self`
    /// { ret.sample_rate == sample_rate }
    pub fn new(threshold_db: f32, sample_rate: u32) -> Self {
        Self {
            threshold: 10f32.powf(threshold_db / 20.0),
            sample_rate,
        }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn reset(&mut self) {}

    fn process(&mut self, samples: &[f32]) -> Result<Vec<f32>, VadError> {
        // Frame-level energy: use 512-sample windows.
        let frame_size = 512usize;
        if !samples.len().is_multiple_of(frame_size) {
            return Err(VadError::InvalidChunkSize {
                expected: frame_size,
                got: samples.len(),
            });
        }
        let mut probs = Vec::with_capacity(samples.len() / frame_size);
        for chunk in samples.chunks(frame_size) {
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

/// { samples.len() >= 512 }
/// `fn segment_speech<V: VoiceActivityDetector>(vad: &mut V, samples: &[f32], config: &DiarizationConfig) -> Result<Vec<(usize, usize)>, VadError>`
/// { ret.iter().all(|(s, e)| s < e) }
pub fn segment_speech<V: VoiceActivityDetector>(
    vad: &mut V,
    samples: &[f32],
    config: &DiarizationConfig,
) -> Result<Vec<(usize, usize)>, VadError> {
    vad.reset();
    let frame_size = 512usize;
    let num_frames = samples.len() / frame_size;
    let mut probs = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let chunk = &samples[i * frame_size..(i + 1) * frame_size];
        let frame_probs = vad.process(chunk)?;
        probs.extend(frame_probs);
    }

    let sr = config.sample_rate as f32;
    let ms_per_frame = (frame_size as f32 / sr) * 1000.0;
    let min_speech_frames =
        ((config.min_speech_secs * 1000.0) / ms_per_frame).ceil() as usize;
    let threshold = 0.5f32; // Default speech probability threshold.

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut seg_start = 0usize;
    let mut silence_count = 0usize;
    let min_silence_frames =
        ((300.0f32) / ms_per_frame).ceil() as usize; // 300 ms default min silence.

    for (i, &prob) in probs.iter().enumerate() {
        if in_speech {
            if prob < threshold {
                silence_count += 1;
                if silence_count >= min_silence_frames {
                    let seg_end = (i + 1) * frame_size;
                    let duration_frames = i + 1 - seg_start / frame_size;
                    if duration_frames >= min_speech_frames {
                        segments.push((seg_start, seg_end));
                    }
                    in_speech = false;
                    silence_count = 0;
                }
            } else {
                silence_count = 0;
            }
        } else if prob >= threshold {
            seg_start = i * frame_size;
            in_speech = true;
            silence_count = 0;
        }
    }

    if in_speech {
        let seg_end = num_frames * frame_size;
        let duration_frames = num_frames - seg_start / frame_size;
        if duration_frames >= min_speech_frames {
            segments.push((seg_start, seg_end));
        }
    }

    Ok(segments)
}
