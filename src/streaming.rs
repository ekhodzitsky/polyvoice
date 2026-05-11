//! Real-time streaming diarization pipeline.
//!
//! Processes audio incrementally chunk-by-chunk with bounded latency.
//! Unlike the offline [`Pipeline`](crate::pipeline::Pipeline), `StreamingPipeline`
//! emits [`SpeakerTurn`]s as soon as each embedding window is processed.
//!
//! Latency is bounded by `DiarizationConfig::window_secs` plus VAD look-ahead.
//!
//! # Example
//! ```rust,no_run
//! use polyvoice::streaming::StreamingPipeline;
//! use polyvoice::{EnergyVad, DummyExtractor, DiarizationConfig, VadConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let vad = EnergyVad::new(-40.0, 16000, 512);
//!     let extractor = DummyExtractor::new(256);
//!     let mut pipeline = StreamingPipeline::new(
//!         vad,
//!         extractor,
//!         DiarizationConfig::default(),
//!         VadConfig::default(),
//!     )?;
//!     let chunk = vec![0.0f32; 16000];
//!     let _turns = pipeline.feed(&chunk)?;
//!     Ok(())
//! }
//! ```

use crate::cluster::{ClusterConfig, SpeakerCluster};
use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::types::{DiarizationConfig, SpeakerTurn, TimeRange};
use crate::vad::{VadError, VoiceActivityDetector};
use crate::VadConfig;

/// Errors from streaming pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

/// Stateful streaming diarization pipeline.
///
/// Generic over a [`VoiceActivityDetector`] `V` and an [`EmbeddingExtractor`] `E`.
pub struct StreamingPipeline<V, E> {
    vad: V,
    extractor: E,
    cluster: SpeakerCluster,
    config: DiarizationConfig,
    frame_size: usize,
    sample_rate: u32,
    // VAD buffering
    vad_buffer: Vec<f32>,
    // Speech detection state
    in_speech: bool,
    seg_start_frame: usize,
    silence_count: usize,
    min_silence_frames: usize,
    min_speech_frames: usize,
    threshold: f32,
    // Embedding state (active speech region)
    window_buffer: Vec<f32>,
    next_window_start_sample: usize,
    // Output
    turns: Vec<SpeakerTurn>,
    total_frames: usize,
}

impl<V, E> StreamingPipeline<V, E>
where
    V: VoiceActivityDetector,
    E: EmbeddingExtractor,
{
    /// Create a new streaming pipeline.
    ///
    /// # Errors
    /// Returns `VadError::InvalidChunkSize` if the VAD `frame_size` is zero.
    pub fn new(
        vad: V,
        extractor: E,
        config: DiarizationConfig,
        vad_config: VadConfig,
    ) -> Result<Self, StreamingError> {
        let frame_size = vad_config.frame_size;
        if frame_size == 0 {
            return Err(VadError::InvalidChunkSize {
                expected: 1,
                got: 0,
            }
            .into());
        }
        let sample_rate = config.sample_rate.get();
        let sr_f = sample_rate as f32;
        let ms_per_frame = (frame_size as f32 / sr_f) * 1000.0;
        let min_silence_frames = (vad_config.min_silence_ms / ms_per_frame).ceil() as usize;
        let min_speech_frames =
            ((config.min_speech_secs * 1000.0) / ms_per_frame).ceil() as usize;

        let cluster = SpeakerCluster::new(ClusterConfig {
            threshold: config.threshold,
            max_speakers: config.max_speakers,
        });

        Ok(Self {
            vad,
            extractor,
            cluster,
            config,
            frame_size,
            sample_rate,
            vad_buffer: Vec::new(),
            in_speech: false,
            seg_start_frame: 0,
            silence_count: 0,
            min_silence_frames,
            min_speech_frames,
            threshold: vad_config.threshold,
            window_buffer: Vec::new(),
            next_window_start_sample: 0,
            turns: Vec::new(),
            total_frames: 0,
        })
    }

    /// Feed a chunk of audio samples and return any newly finalized speaker turns.
    ///
    /// The pipeline internally buffers samples until a full VAD frame is available,
    /// then runs the frame through VAD, speech detection, and — during active speech —
    /// extracts embeddings and assigns speakers incrementally.
    ///
    /// Callers should feed chunks as they arrive from the audio source (e.g. microphone).
    /// There is no minimum chunk size; sub-frame chunks are buffered transparently.
    pub fn feed(&mut self, samples: &[f32]) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut new_turns = Vec::new();
        self.vad_buffer.extend_from_slice(samples);

        let frame_size = self.frame_size;
        while self.vad_buffer.len() >= frame_size {
            let frame: Vec<f32> = self.vad_buffer.drain(..frame_size).collect();
            let probs = self.vad.process(&frame)?;

            // Each call to process on exactly frame_size samples is expected to
            // return at least one probability. If the VAD returns more than one
            // (e.g. sub-frame resolution), we treat them as successive frames.
            for &prob in &probs {
                let current_frame = self.total_frames;
                self.total_frames += 1;

                if self.in_speech {
                    if prob < self.threshold {
                        self.silence_count += 1;
                        if self.silence_count >= self.min_silence_frames {
                            // Speech region ended.
                            let seg_end_frame = current_frame;
                            let seg_end_sample = seg_end_frame * frame_size;
                            let duration_frames = seg_end_frame - self.seg_start_frame;
                            if duration_frames >= self.min_speech_frames {
                                new_turns.extend(self.flush_window_buffer(seg_end_sample)?);
                            } else {
                                self.window_buffer.clear();
                            }
                            self.in_speech = false;
                            self.silence_count = 0;
                        }
                    } else {
                        self.silence_count = 0;
                    }

                    if self.in_speech {
                        self.window_buffer.extend_from_slice(&frame);
                        new_turns.extend(self.try_extract_windows()?);
                    }
                } else if prob >= self.threshold {
                    // Speech region started.
                    self.in_speech = true;
                    self.seg_start_frame = current_frame;
                    self.next_window_start_sample = self.seg_start_frame * frame_size;
                    self.silence_count = 0;
                    self.window_buffer.clear();
                }
            }
        }

        Ok(new_turns)
    }

    /// Flush any pending audio and return final speaker turns.
    ///
    /// This finalizes an in-flight speech region (if any), extracts the last
    /// embedding window, and clears all internal buffers. After `flush` the
    /// pipeline is ready to process a new stream (or the same stream after a
    /// gap) via subsequent `feed` calls.
    pub fn flush(&mut self) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut new_turns = Vec::new();

        // Discard any trailing sub-frame samples.
        self.vad_buffer.clear();

        if self.in_speech {
            let seg_end_sample = self.total_frames * self.frame_size;
            let duration_frames = self.total_frames - self.seg_start_frame;
            if duration_frames >= self.min_speech_frames {
                new_turns.extend(self.flush_window_buffer(seg_end_sample)?);
            } else {
                self.window_buffer.clear();
            }
            self.in_speech = false;
            self.silence_count = 0;
        }

        Ok(new_turns)
    }

    /// { self.cluster.num_speakers() >= 0 }
    /// `pub fn num_speakers(&self) -> usize`
    /// { ret == self.cluster.num_speakers() }
    /// Return the number of distinct speakers observed so far.
    pub fn num_speakers(&self) -> usize {
        self.cluster.num_speakers()
    }

    /// { true }
    /// `pub fn turns(&self) -> &[SpeakerTurn]`
    /// { ret.iter().all(|t| t.time.start <= t.time.end) }
    /// Return all turns emitted so far (including those from prior `feed` calls).
    pub fn turns(&self) -> &[SpeakerTurn] {
        &self.turns
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Extract as many full windows as possible from `window_buffer`.
    fn try_extract_windows(&mut self) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut turns = Vec::new();
        let window_samples = self.config.window_samples();
        let hop_samples = self.config.hop_samples();
        let sr_f = self.sample_rate as f64;

        while self.window_buffer.len() >= window_samples {
            let embedding = self
                .extractor
                .extract(&self.window_buffer[..window_samples], &self.config)?;
            let (speaker, _conf) = self.cluster.assign(&embedding);

            let start = self.next_window_start_sample;
            let end = start + window_samples;
            turns.push(SpeakerTurn {
                speaker,
                time: TimeRange {
                    start: start as f64 / sr_f,
                    end: end as f64 / sr_f,
                },
                text: None,
            });

            self.next_window_start_sample += hop_samples;
            self.window_buffer.drain(..hop_samples);
        }

        Ok(turns)
    }

    /// Zero-pad the trailing `window_buffer` to `window_samples`, extract one
    /// final embedding, and clear the buffer.
    fn flush_window_buffer(
        &mut self,
        seg_end_sample: usize,
    ) -> Result<Vec<SpeakerTurn>, StreamingError> {
        let mut turns = Vec::new();
        let window_samples = self.config.window_samples();
        let sr_f = self.sample_rate as f64;

        if !self.window_buffer.is_empty() {
            if self.window_buffer.len() < window_samples {
                self.window_buffer
                    .resize(window_samples, 0.0f32);
            }
            let embedding = self
                .extractor
                .extract(&self.window_buffer[..window_samples], &self.config)?;
            let (speaker, _conf) = self.cluster.assign(&embedding);

            let start = self.next_window_start_sample;
            let end = seg_end_sample.min(start + window_samples);
            turns.push(SpeakerTurn {
                speaker,
                time: TimeRange {
                    start: start as f64 / sr_f,
                    end: end as f64 / sr_f,
                },
                text: None,
            });

            self.window_buffer.clear();
        }

        Ok(turns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::DummyExtractor;
    use crate::{EnergyVad, VadConfig};

    fn default_config() -> DiarizationConfig {
        DiarizationConfig::default()
    }

    fn default_vad_config() -> VadConfig {
        VadConfig::default()
    }

    fn pipeline() -> StreamingPipeline<EnergyVad, DummyExtractor> {
        let vad = EnergyVad::new(-40.0, 16000, 512);
        let extractor = DummyExtractor::new(256);
        StreamingPipeline::new(vad, extractor, default_config(), default_vad_config()).unwrap()
    }

    /// Loud samples that should trigger speech detection.
    fn loud_samples(seconds: f32) -> Vec<f32> {
        let n = (seconds * 16000.0) as usize;
        vec![0.5f32; n]
    }

    /// Silent samples that should not trigger speech.
    fn silent_samples(seconds: f32) -> Vec<f32> {
        let n = (seconds * 16000.0) as usize;
        vec![0.0f32; n]
    }

    #[test]
    fn streaming_pipeline_new_is_empty() {
        let p = pipeline();
        assert_eq!(p.num_speakers(), 0);
        assert!(p.turns().is_empty());
    }

    #[test]
    fn feed_silence_returns_no_turns() {
        let mut p = pipeline();
        let turns = p.feed(&silent_samples(2.0)).unwrap();
        assert!(turns.is_empty());
        assert!(p.turns().is_empty());
    }

    #[test]
    fn feed_loud_audio_returns_at_least_one_turn() {
        let mut p = pipeline();
        // 5 seconds of loud audio guarantees at least one full window (1.5 s)
        let turns = p.feed(&loud_samples(5.0)).unwrap();
        assert!(!turns.is_empty(), "expected at least one turn for 5 s of speech");
    }

    #[test]
    fn flush_after_speech_emits_remaining_turn() {
        let mut p = pipeline();
        // Feed just under one window — no turn emitted yet.
        let _ = p.feed(&loud_samples(1.0)).unwrap();
        let turns = p.flush().unwrap();
        assert!(!turns.is_empty(), "flush should emit the trailing partial window");
    }

    #[test]
    fn turns_are_monotonically_ordered() {
        let mut p = pipeline();
        let _ = p.feed(&loud_samples(5.0)).unwrap();
        let _ = p.flush().unwrap();
        let turns = p.turns();
        for i in 1..turns.len() {
            assert!(
                turns[i].time.start >= turns[i - 1].time.start,
                "turns must be monotonically ordered"
            );
        }
    }
}
