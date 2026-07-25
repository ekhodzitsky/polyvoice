//! High-level offline diarization pipeline.
//!
//! Wires together VAD, embedding extraction, and AHC clustering into a
//! single `run()` call that takes audio and returns `DiarizationResult`.
//!
//! # Bring-your-own embedder
//!
//! `Pipeline` is generic over [`crate::Embedder`]. Implement that trait on an
//! external encoder (Candle, tract, custom) — no `onnx` feature required:
//!
//! ```rust
//! use polyvoice::{
//!     DiarizationConfig, Embedder, EmbedderError, EnergyVad, Pipeline, VadConfig,
//! };
//!
//! struct FixedEmbedder;
//!
//! impl Embedder for FixedEmbedder {
//!     fn dim(&self) -> usize { 4 }
//!     fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
//!         Ok(vec![1.0, 0.0, 0.0, 0.0])
//!     }
//! }
//!
//! let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());
//! let mut vad = EnergyVad::new(-40.0, 16_000, 512);
//! // Silence → no speech; exercise the type bounds only.
//! let _ = pipeline.run(&vec![0.0f32; 16_000], &FixedEmbedder, &mut vad);
//! ```

use crate::ahc::agglomerative_cluster;
use crate::embedder::{Embedder, EmbedderError};
use crate::types::{
    DiarizationConfig, DiarizationResult, Segment, SpeakerId, SpeakerTurn, TimeRange,
};
use crate::vad::{VadConfig, VadError, VoiceActivityDetector, segment_speech};
use crate::wav;
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("VAD error: {0}")]
    Vad(#[from] VadError),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbedderError),
    #[error("WAV error: {0}")]
    Wav(#[from] wav::WavError),
    #[error("unsupported WAV sample rate: {actual}, expected: {expected}")]
    UnsupportedSampleRate { expected: u32, actual: u32 },
    #[error("no speech detected in audio")]
    NoSpeech,
    #[error("audio too long: {actual_secs:.1}s > max {max_secs:.1}s")]
    AudioTooLong { actual_secs: f32, max_secs: f32 },
}

pub struct Pipeline {
    config: DiarizationConfig,
    vad_config: VadConfig,
}

impl Pipeline {
    /// { true }
    /// pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self
    /// { true }
    pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self {
        Self { config, vad_config }
    }

    /// { true }
    /// `pub fn run<E: Embedder, V: VoiceActivityDetector>( &self, samples: &[f32], extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, PipelineError>`
    /// { ret.as_ref().map_or(true, |r| r.num_speakers <= r.segments.len()) }
    /// Run the full diarization pipeline on raw f32 samples.
    ///
    /// `extractor` must implement [`Embedder`] (the supported BYO surface).
    /// Legacy [`crate::embedding::EmbeddingExtractor`] types work through an
    /// automatic bridge.
    ///
    /// Returns [`PipelineError::AudioTooLong`] if the input exceeds
    /// `config.max_duration_secs` (default 1 hour).
    pub fn run<E: Embedder, V: VoiceActivityDetector>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let actual_secs = samples.len() as f32 / self.config.window.sample_rate.get() as f32;
        if actual_secs > self.config.max_duration_secs {
            return Err(PipelineError::AudioTooLong {
                actual_secs,
                max_secs: self.config.max_duration_secs,
            });
        }
        let speech_regions = segment_speech(vad, samples, &self.config, &self.vad_config)?;

        if speech_regions.is_empty() {
            return Ok(DiarizationResult::new(Vec::new(), Vec::new(), 0));
        }

        let sr = self.config.window.sample_rate.get() as f64;
        let window = self.config.window_samples();
        let hop = self.config.hop_samples();
        let mut embeddings = Vec::new();
        let mut time_ranges = Vec::new();

        for &(start, end) in &speech_regions {
            let region = &samples[start..end];

            if region.len() < window {
                let mut padded = vec![0.0f32; window];
                padded[..region.len()].copy_from_slice(region);
                let emb = extractor.embed(&padded)?;
                embeddings.push(emb);
                time_ranges.push(TimeRange {
                    start: start as f64 / sr,
                    end: end as f64 / sr,
                });
            } else {
                for (offset, offset_end) in
                    crate::window::WindowIter::new(region.len(), window, hop)
                {
                    let chunk = &region[offset..offset_end];
                    let emb = extractor.embed(chunk)?;
                    embeddings.push(emb);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr,
                        end: (start + offset_end) as f64 / sr,
                    });
                }
            }
        }

        if embeddings.is_empty() {
            return Ok(DiarizationResult::new(Vec::new(), Vec::new(), 0));
        }

        let labels = agglomerative_cluster(&embeddings, self.config.cluster.threshold);
        // Dissolve spurious tiny clusters into the nearest large speaker — the
        // over-clustering fix. Duration pruning (length-invariant) takes
        // precedence when configured; otherwise the member-count rule applies.
        let labels = if self.config.cluster.min_cluster_secs > 0.0 {
            crate::ahc::prune_small_clusters_by_duration(
                &time_ranges,
                &embeddings,
                labels,
                self.config.cluster.min_cluster_secs,
            )
        } else {
            crate::ahc::prune_small_clusters(
                &embeddings,
                labels,
                self.config.cluster.min_cluster_size,
            )
        };
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

        // Per-window confidence from cosine similarity to the speaker centroid
        // (logistic map — ranking score, not a calibrated probability).
        let speaker_ids: Vec<SpeakerId> = labels.iter().map(|&l| SpeakerId(l as u32)).collect();
        let confidences =
            crate::types::segment_confidences_from_embeddings(&speaker_ids, &embeddings);

        let mut segments: Vec<Segment> = labels
            .iter()
            .zip(time_ranges.iter())
            .enumerate()
            .map(|(i, (&label, &time))| Segment {
                time,
                speaker: Some(SpeakerId(label as u32)),
                confidence: confidences.get(i).copied(),
            })
            .collect();

        segments =
            crate::utils::merge_segments(segments, self.config.speech_filter.max_gap_secs as f64);
        segments.retain(|s| s.time.duration() >= self.config.speech_filter.min_speech_secs as f64);

        let turns: Vec<SpeakerTurn> = segments
            .iter()
            .filter_map(|s| {
                s.speaker.map(|spk| SpeakerTurn {
                    speaker: spk,
                    time: s.time,
                    text: None,
                    stable: true,
                })
            })
            .collect();

        let sr_hz = self.config.window.sample_rate.get();
        Ok(DiarizationResult::new(segments, turns, num_speakers)
            .with_audio(samples.len() as f64 / sr_hz as f64, sr_hz))
    }

    /// { true }
    /// `pub fn run_from_wav<E: Embedder, V: VoiceActivityDetector>( &self, path: &Path, extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, PipelineError>`
    /// { ret.as_ref().map_or(true, |r| r.num_speakers <= r.segments.len()) }
    /// Run the pipeline from a WAV file path.
    ///
    /// Returns [`PipelineError::UnsupportedSampleRate`] if the WAV sample rate
    /// does not match [`crate::types::WindowConfig::sample_rate`] in
    /// [`DiarizationConfig::window`].
    pub fn run_from_wav<E: Embedder, V: VoiceActivityDetector>(
        &self,
        path: &Path,
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let (samples, sample_rate) = wav::read_wav(path)?;
        let expected = self.config.window.sample_rate.get();
        if sample_rate != expected {
            return Err(PipelineError::UnsupportedSampleRate {
                expected,
                actual: sample_rate,
            });
        }
        self.run(&samples, extractor, vad)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Embedder;
    use std::io::Cursor;
    use std::sync::Mutex;

    #[test]
    fn pipeline_new_with_defaults() {
        let config = DiarizationConfig::default();
        let vad_config = VadConfig::default();
        let pipeline = Pipeline::new(config, vad_config);
        // Pipeline exists; basic sanity check via debug print would require
        // accessing private fields, so we just verify construction succeeds.
        assert!(std::mem::size_of_val(&pipeline) > 0);
    }

    #[test]
    fn audio_too_long_error() {
        let config = DiarizationConfig {
            max_duration_secs: 1.0,
            ..Default::default()
        };
        let vad_config = VadConfig::default();
        let pipeline = Pipeline::new(config, vad_config);

        // Create 2 seconds of silence at 16kHz
        let samples = vec![0.0f32; 32000];
        let extractor = crate::embedding::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16000, 512);
        let result = pipeline.run(&samples, &extractor, &mut vad);
        assert!(
            matches!(result, Err(PipelineError::AudioTooLong { .. })),
            "expected AudioTooLong error, got {:?}",
            result
        );
    }

    #[test]
    fn wav_sample_rate_mismatch_error() {
        // Create a 1-second mono WAV at 22050 Hz while the pipeline expects 16000 Hz.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = hound::WavWriter::new(cursor, spec).unwrap();
            for i in 0..22050 {
                let sample = ((i as f32 / 22050.0) * std::f32::consts::TAU * 440.0).sin();
                writer.write_sample((sample * 32767.0) as i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &buf).unwrap();

        let config = DiarizationConfig::default();
        let pipeline = Pipeline::new(config, VadConfig::default());
        let extractor = crate::embedding::DummyExtractor::new(256);
        let mut vad = crate::vad::EnergyVad::new(-40.0, 16000, 512);
        let result = pipeline.run_from_wav(tmp.path(), &extractor, &mut vad);
        assert!(
            matches!(
                result,
                Err(PipelineError::UnsupportedSampleRate {
                    expected: 16000,
                    actual: 22050,
                })
            ),
            "expected UnsupportedSampleRate error, got {:?}",
            result
        );
    }

    /// Deterministic two-prototype embedder: high zero-crossing rate → speaker A,
    /// low ZCR → speaker B. Orthogonal unit vectors so AHC must yield ≥2 clusters.
    struct TwoSpeakerEmbedder {
        /// Call count — unused, held for Send+Sync realism (shared encoder pattern).
        _calls: Mutex<usize>,
    }

    impl TwoSpeakerEmbedder {
        fn new() -> Self {
            Self {
                _calls: Mutex::new(0),
            }
        }
    }

    impl Embedder for TwoSpeakerEmbedder {
        fn dim(&self) -> usize {
            4
        }

        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            if let Ok(mut c) = self._calls.lock() {
                *c += 1;
            }
            let mut zcr = 0usize;
            for w in audio.windows(2) {
                if w[0].signum() != w[1].signum() {
                    zcr += 1;
                }
            }
            let rate = zcr as f32 / audio.len().max(1) as f32;
            // 300 Hz @ 16 kHz ≈ 0.0375 ZCR; 800 Hz ≈ 0.1 ZCR.
            let mut v = if rate > 0.06 {
                vec![1.0, 0.0, 0.0, 0.0]
            } else {
                vec![0.0, 1.0, 0.0, 0.0]
            };
            crate::utils::l2_normalize(&mut v);
            Ok(v)
        }
    }

    fn sine_wave(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
        let n = (duration_secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn custom_embedder_two_speakers_without_onnx() {
        let sr = 16_000u32;
        let mut samples = sine_wave(300.0, 2.0, sr);
        samples.extend(std::iter::repeat_n(0.0, sr as usize)); // 1 s silence
        samples.extend(sine_wave(800.0, 2.0, sr));

        let mut config = DiarizationConfig::default();
        // Orthogonal prototypes must never merge.
        config.cluster.threshold = 0.9;
        config.cluster.min_cluster_size = 1;
        config.cluster.min_cluster_secs = 0.0;

        let pipeline = Pipeline::new(config, VadConfig::default());
        let embedder = TwoSpeakerEmbedder::new();
        let mut vad = crate::vad::EnergyVad::new(-40.0, sr, 512);
        let result = pipeline.run(&samples, &embedder, &mut vad).unwrap();

        assert!(
            result.num_speakers >= 2,
            "expected ≥2 speakers from deterministic two-prototype embedder, got {}",
            result.num_speakers
        );
        assert!(!result.turns.is_empty());
    }
}
