//! High-level diarization pipeline.
//!
//! Wires together VAD, embedding extraction, and AHC clustering into a
//! single `run()` call that takes audio and returns `DiarizationResult`.

use crate::ahc::agglomerative_cluster;
use crate::embedding::EmbeddingExtractor;
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
    Embedding(#[from] crate::embedding::EmbeddingError),
    #[error("WAV error: {0}")]
    Wav(#[from] wav::WavError),
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
    /// { TODO: precondition }
    /// pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self
    /// { TODO: postcondition }
    pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self {
        Self { config, vad_config }
    }

    /// { TODO: precondition }
    /// `pub fn run<E: EmbeddingExtractor, V: VoiceActivityDetector>( &self, samples: &[f32], extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, PipelineError>`
    /// { TODO: postcondition }
    /// Run the full diarization pipeline on raw f32 samples.
    ///
    /// Returns [`PipelineError::AudioTooLong`] if the input exceeds
    /// `config.max_duration_secs` (default 1 hour).
    pub fn run<E: EmbeddingExtractor, V: VoiceActivityDetector>(
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
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
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
                let emb = extractor.extract(&padded, &self.config)?;
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
                    let emb = extractor.extract(chunk, &self.config)?;
                    embeddings.push(emb);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr,
                        end: (start + offset_end) as f64 / sr,
                    });
                }
            }
        }

        if embeddings.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let labels = agglomerative_cluster(&embeddings, self.config.cluster.threshold);
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

        let mut segments: Vec<Segment> = labels
            .iter()
            .zip(time_ranges.iter())
            .map(|(&label, &time)| Segment {
                time,
                speaker: Some(SpeakerId(label as u32)),
                confidence: None,
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
                })
            })
            .collect();

        Ok(DiarizationResult {
            segments,
            turns,
            num_speakers,
        })
    }

    /// { TODO: precondition }
    /// `pub fn run_from_wav<E: EmbeddingExtractor, V: VoiceActivityDetector>( &self, path: &Path, extractor: &E, vad: &mut V, ) -> Result<DiarizationResult, PipelineError>`
    /// { TODO: postcondition }
    /// Run the pipeline from a WAV file path.
    pub fn run_from_wav<E: EmbeddingExtractor, V: VoiceActivityDetector>(
        &self,
        path: &Path,
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let (samples, _sample_rate) = wav::read_wav(path)?;
        self.run(&samples, extractor, vad)
    }
}
