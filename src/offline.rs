//! Offline (file-based) speaker diarization.

use crate::cluster::SpeakerCluster;
use crate::embedding::EmbeddingExtractor;
use crate::types::{
    DiarizationConfig, DiarizationResult, Segment, SpeakerId, SpeakerTurn, TimeRange,
};

/// Offline diarizer that processes an entire audio file at once.
pub struct OfflineDiarizer {
    config: DiarizationConfig,
}

impl OfflineDiarizer {
    /// Create an offline diarizer with the given configuration.
    ///
    /// ```rust
    /// use polyvoice::{OfflineDiarizer, DiarizationConfig};
    /// let diarizer = OfflineDiarizer::new(DiarizationConfig::default());
    /// ```
    pub fn new(config: DiarizationConfig) -> Self {
        Self { config }
    }

    /// Run diarization on an entire audio buffer.
    ///
    /// Returns speaker segments and turns. If the audio is shorter than the
    /// analysis window, the result will contain no segments.
    ///
    /// ```rust
    /// use polyvoice::{OfflineDiarizer, DiarizationConfig, DummyExtractor};
    /// let diarizer = OfflineDiarizer::new(DiarizationConfig::default());
    /// let extractor = DummyExtractor::new(256);
    /// let samples = vec![0.0f32; 16000 * 10]; // 10 seconds
    /// let result = diarizer.run(&samples, &extractor).unwrap();
    /// // Result structure is valid even if segments may be empty for silence.
    /// ```
    pub fn run<E: EmbeddingExtractor>(
        &self,
        samples: &[f32],
        extractor: &E,
    ) -> Result<DiarizationResult, crate::embedding::EmbeddingError> {
        let window = self.config.window_samples();
        let hop = self.config.hop_samples();
        let sr = self.config.sample_rate.get() as f64;

        if samples.len() < window {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        // --- Pass 1: extract embeddings and initial online clustering ---
        let mut cluster = SpeakerCluster::new(self.config);
        let mut windows = Vec::new(); // (start_sample, end_sample, embedding, speaker, confidence)

        let mut start = 0usize;
        while start + window <= samples.len() {
            let end = start + window;
            let chunk = &samples[start..end];
            let embedding = extractor.extract(chunk, &self.config)?;
            let (speaker, confidence) = cluster.assign(&embedding);
            windows.push((start, end, embedding, speaker, confidence));
            start += hop;
        }

        // Handle trailing audio: pad with zeros.
        if start < samples.len() {
            let end = samples.len();
            let mut padded = vec![0.0f32; window];
            let copy_len = end - start;
            padded[..copy_len].copy_from_slice(&samples[start..end]);
            let embedding = extractor.extract(&padded, &self.config)?;
            let (speaker, confidence) = cluster.assign(&embedding);
            windows.push((start, start + window, embedding, speaker, confidence));
        }

        // --- Pass 2: post-processing ---
        let segments = self.post_process(&windows, samples.len(), sr);
        let turns = self.segments_to_turns(&segments);

        Ok(DiarizationResult {
            num_speakers: cluster.num_speakers(),
            segments,
            turns,
        })
    }

    fn post_process(
        &self,
        windows: &[(usize, usize, Vec<f32>, SpeakerId, f32)],
        total_samples: usize,
        sr: f64,
    ) -> Vec<Segment> {
        let mut segments = Vec::new();

        // Convert windows to segments.
        for (start, end, _emb, speaker, confidence) in windows {
            segments.push(Segment {
                time: TimeRange {
                    start: *start as f64 / sr,
                    end: (*end as f64 / sr).min(total_samples as f64 / sr),
                },
                speaker: Some(*speaker),
                confidence: Some(*confidence),
            });
        }

        // Merge adjacent segments with the same speaker and small gaps.
        segments = crate::utils::merge_segments(segments, self.config.max_gap_secs as f64);

        // Remove very short segments (< min_speech_secs).
        segments.retain(|s| s.time.duration() >= self.config.min_speech_secs as f64);

        segments
    }

    fn segments_to_turns(&self, segments: &[Segment]) -> Vec<SpeakerTurn> {
        segments
            .iter()
            .filter_map(|s| {
                s.speaker.map(|spk| SpeakerTurn {
                    speaker: spk,
                    time: s.time,
                    text: None,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::DummyExtractor;

    #[test]
    fn test_offline_empty() {
        let config = DiarizationConfig::default();
        let diarizer = OfflineDiarizer::new(config);
        let extractor = DummyExtractor::new(256);
        let result = diarizer.run(&[], &extractor).unwrap();
        assert_eq!(result.num_speakers, 0);
        assert!(result.segments.is_empty());
    }

    #[test]
    fn test_offline_short_audio() {
        let config = DiarizationConfig::default();
        let diarizer = OfflineDiarizer::new(config);
        let extractor = DummyExtractor::new(256);
        // 1 second of audio — less than default 1.5s window.
        let samples = vec![0.0f32; 16000];
        let result = diarizer.run(&samples, &extractor).unwrap();
        assert_eq!(result.num_speakers, 0);
        assert!(result.segments.is_empty());
    }

    #[test]
    fn test_offline_basic() {
        let config = DiarizationConfig {
            window_secs: 0.5,
            hop_secs: 0.25,
            ..Default::default()
        };
        let diarizer = OfflineDiarizer::new(config);
        let extractor = DummyExtractor::new(256);
        // 5 seconds of audio.
        let samples = vec![0.1f32; 16000 * 5];
        let result = diarizer.run(&samples, &extractor).unwrap();
        // Should produce some segments.
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_merge_segments() {
        let segs = vec![
            Segment {
                time: TimeRange {
                    start: 0.0,
                    end: 1.0,
                },
                speaker: Some(SpeakerId(0)),
                confidence: Some(0.8),
            },
            Segment {
                time: TimeRange {
                    start: 1.2,
                    end: 2.0,
                },
                speaker: Some(SpeakerId(0)),
                confidence: Some(0.9),
            },
            Segment {
                time: TimeRange {
                    start: 2.5,
                    end: 3.0,
                },
                speaker: Some(SpeakerId(1)),
                confidence: Some(0.7),
            },
        ];
        let merged = crate::utils::merge_segments(segs, 0.5);
        assert_eq!(merged.len(), 2);
        assert!((merged[0].time.end - 2.0).abs() < 1e-5);
    }
}
