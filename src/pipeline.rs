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
use crate::utils::{cosine_similarity, l2_normalize};
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

        // Multi-scale embedding averaging: smooth embeddings temporally before clustering.
        // Currently disabled — experiments on AMI showed degradation (DER increased).
        // let embeddings = smooth_embeddings(&embeddings, 1);

        let raw_labels = match self.config.clustering_backend {
            ClusteringBackend::Ahc => agglomerative_cluster(&embeddings, self.config.threshold),
            ClusteringBackend::KMeans => kmeans_auto_k(&embeddings, self.config.max_speakers, 100),
            ClusteringBackend::Spectral => spectral_cluster(&embeddings, self.config.max_speakers),
            ClusteringBackend::Auto => agglomerative_cluster_auto(&embeddings).0,
        };

        // Post-process: merge speakers with very few embeddings (outliers).
        let labels = merge_small_speakers(&embeddings, &raw_labels, self.config.min_embeddings_per_speaker);

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



/// Temporal smoothing of cluster labels using centroid similarities.
///
/// Computes centroids for each speaker, then for each embedding computes
/// cosine similarity to all centroids.  A moving average (window size =
/// `window_frames * 2 + 1`) is applied along the time axis, and each
/// embedding is reassigned to the speaker with the highest smoothed score.
#[allow(dead_code)]
fn temporal_smooth(embeddings: &[Vec<f32>], labels: &[usize], window_frames: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let k = labels.iter().copied().max().unwrap_or(0) + 1;
    if k == 0 {
        return Vec::new();
    }
    let dim = embeddings[0].len();

    // Compute centroids.
    let mut centroids = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (i, emb) in embeddings.iter().enumerate() {
        let c = labels[i];
        for (centroid, &v) in centroids[c].iter_mut().zip(emb.iter()) {
            *centroid += v;
        }
        counts[c] += 1;
    }
    for (c, centroid) in centroids.iter_mut().enumerate() {
        if counts[c] > 0 {
            for v in centroid.iter_mut() {
                *v /= counts[c] as f32;
            }
            l2_normalize(centroid);
        }
    }

    // Compute raw similarity matrix [n × k].
    let mut sims = vec![vec![0.0f32; k]; n];
    for (i, emb) in embeddings.iter().enumerate() {
        for (c_idx, centroid) in centroids.iter().enumerate() {
            sims[i][c_idx] = cosine_similarity(emb, centroid);
        }
    }

    // Apply temporal smoothing (moving average).
    let mut smoothed = vec![vec![0.0f32; k]; n];
    let w = window_frames as isize;
    for (i, smoothed_row) in smoothed.iter_mut().enumerate() {
        for (c, smoothed_val) in smoothed_row.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for j in (i as isize - w)..=(i as isize + w) {
                if j >= 0 && j < n as isize {
                    sum += sims[j as usize][c];
                    count += 1;
                }
            }
            *smoothed_val = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    // Reassign labels.
    let mut new_labels = vec![0usize; n];
    for (i, smoothed_row) in smoothed.iter().enumerate() {
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for (c, &sim) in smoothed_row.iter().enumerate() {
            if sim > best_sim {
                best_sim = sim;
                best = c;
            }
        }
        new_labels[i] = best;
    }

    new_labels
}

/// Smooth embeddings by averaging each embedding with its temporal neighbours.
#[allow(dead_code)]
fn smooth_embeddings(embeddings: &[Vec<f32>], window: usize) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let mut avg = vec![0.0f32; dim];
        let mut count = 0usize;
        for emb in embeddings.iter().take((i + window).min(n - 1) + 1).skip(i.saturating_sub(window)) {
            for (a, &v) in avg.iter_mut().zip(emb.iter()) {
                *a += v;
            }
            count += 1;
        }
        for a in avg.iter_mut() {
            *a /= count as f32;
        }
        smoothed.push(avg);
    }
    smoothed
}

/// Merge speakers that have fewer than `min_embeddings` assignments.
///
/// Reassigns all embeddings of a "small" speaker to the nearest other speaker
/// centroid.  This removes spurious outlier clusters created by AHC.
fn merge_small_speakers(embeddings: &[Vec<f32>], labels: &[usize], min_embeddings: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let num_speakers = labels.iter().copied().max().unwrap_or(0) + 1;
    if num_speakers <= 1 {
        return labels.to_vec();
    }
    let dim = embeddings[0].len();

    // Count embeddings per speaker.
    let mut counts = vec![0usize; num_speakers];
    for &l in labels {
        counts[l] += 1;
    }

    // Identify small speakers.
    let small: Vec<bool> = counts.iter().map(|&c| c < min_embeddings).collect();
    if !small.iter().any(|&s| s) {
        return labels.to_vec();
    }

    // Compute centroids for non-small speakers.
    let mut centroids = vec![vec![0.0f32; dim]; num_speakers];
    let mut centroid_counts = vec![0usize; num_speakers];
    for (i, emb) in embeddings.iter().enumerate() {
        let l = labels[i];
        for (c, &v) in centroids[l].iter_mut().zip(emb.iter()) {
            *c += v;
        }
        centroid_counts[l] += 1;
    }
    for (l, centroid) in centroids.iter_mut().enumerate() {
        if centroid_counts[l] > 0 {
            for v in centroid.iter_mut() {
                *v /= centroid_counts[l] as f32;
            }
            l2_normalize(centroid);
        }
    }

    // Reassign embeddings of small speakers to nearest non-small speaker.
    let mut new_labels = labels.to_vec();
    for (i, emb) in embeddings.iter().enumerate() {
        let l = labels[i];
        if small[l] {
            let mut best = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (other, centroid) in centroids.iter().enumerate() {
                if small[other] || other == l {
                    continue;
                }
                let sim = cosine_similarity(emb, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best = other;
                }
            }
            new_labels[i] = best;
        }
    }

    // Remap labels to contiguous indices.
    let mut remap = std::collections::HashMap::new();
    let mut next = 0usize;
    for l in &mut new_labels {
        let entry = remap.entry(*l).or_insert_with(|| {
            let nl = next;
            next += 1;
            nl
        });
        *l = *entry;
    }

    new_labels
}
