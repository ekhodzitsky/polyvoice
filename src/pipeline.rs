//! High-level diarization pipeline.
//!
//! Wires together VAD, embedding extraction, and AHC clustering into a
//! single `run()` call that takes audio and returns `DiarizationResult`.

use crate::ahc::{agglomerative_cluster, agglomerative_cluster_auto};
use crate::embedding::EmbeddingExtractor;
use crate::kmeans::kmeans_auto_k;
use crate::spectral::spectral_cluster;
use crate::types::ClusteringBackend;
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
}

pub struct Pipeline {
    config: DiarizationConfig,
    vad_config: VadConfig,
}

impl Pipeline {
    pub fn new(config: DiarizationConfig, vad_config: VadConfig) -> Self {
        Self { config, vad_config }
    }

    /// Run the full diarization pipeline on raw f32 samples.
    pub fn run<E: EmbeddingExtractor, V: VoiceActivityDetector>(
        &self,
        samples: &[f32],
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let speech_regions = segment_speech(vad, samples, &self.config, &self.vad_config)?;

        if speech_regions.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let sr = self.config.sample_rate.get() as f64;
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
                let mut offset = 0;
                while offset + window <= region.len() {
                    let chunk = &region[offset..offset + window];
                    let emb = extractor.extract(chunk, &self.config)?;
                    embeddings.push(emb);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr,
                        end: (start + offset + window) as f64 / sr,
                    });
                    offset += hop;
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

        let labels = match self.config.clustering_backend {
            ClusteringBackend::Ahc => agglomerative_cluster(&embeddings, self.config.threshold),
            ClusteringBackend::KMeans => kmeans_auto_k(&embeddings, self.config.max_speakers, 100),
            ClusteringBackend::Spectral => spectral_cluster(&embeddings, self.config.max_speakers),
            ClusteringBackend::Auto => agglomerative_cluster_auto(&embeddings).0,
        };
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

        segments = crate::utils::merge_segments(segments, self.config.max_gap_secs as f64);
        segments.retain(|s| s.time.duration() >= self.config.min_speech_secs as f64);
        if self.config.min_turn_duration_secs > 0.0 {
            segments.retain(|s| s.time.duration() >= self.config.min_turn_duration_secs as f64);
        }

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

    /// Run the pipeline from a WAV file path.
    pub fn run_from_wav<E: EmbeddingExtractor, V: VoiceActivityDetector>(
        &self,
        path: &Path,
        extractor: &E,
        vad: &mut V,
    ) -> Result<DiarizationResult, PipelineError> {
        let (samples, sample_rate) = wav::read_wav(path)?;
        if sample_rate != self.config.sample_rate.get() {
            tracing::warn!(
                "WAV sample rate {} Hz does not match config {} Hz",
                sample_rate,
                self.config.sample_rate.get()
            );
        }
        self.run(&samples, extractor, vad)
    }
}


