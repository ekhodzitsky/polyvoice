//! Hybrid pipeline: PowersetSegmenter as a VAD → sliding-window embeddings → AHC.
//!
//! PowersetSegmenter is used only for speech-region detection (it handles overlap
//! better than SileroVAD), but its `local_speaker_idx` labels are ignored.
//! Speaker identity is resolved globally by clustering ResNet34 embeddings,
//! exactly as the legacy v0.5 pipeline does.  This removes the 3-speaker ceiling
//! of the powerset model while keeping its superior segmentation quality.

use crate::clusterer::Clusterer;
use crate::embedder::Embedder;
use crate::pipeline_v2::PipelineError;
use crate::segmentation::{RawSegment, Segmenter};
use crate::types::{DiarizationResult, SampleRate, Segment, SpeakerId, SpeakerTurn, TimeRange};
use crate::utils::merge_segments;
use crate::window::WindowIter;

pub struct HybridPipeline {
    segmenter: Box<dyn Segmenter>,
    embedder: Box<dyn Embedder>,
    clusterer: Box<dyn Clusterer>,
    window_samples: usize,
    hop_samples: usize,
    sample_rate: u32,
    min_speech_secs: f64,
    max_gap_secs: f64,
    include_partial_chunks: bool,
    exclude_overlap: bool,
}

impl HybridPipeline {
    pub fn new(
        segmenter: Box<dyn Segmenter>,
        embedder: Box<dyn Embedder>,
        clusterer: Box<dyn Clusterer>,
    ) -> Self {
        Self {
            segmenter,
            embedder,
            clusterer,
            window_samples: 2 * 16000, // 2 seconds
            hop_samples: 16000 + 8000, // 1.5 seconds
            sample_rate: 16000,
            min_speech_secs: 0.25,
            max_gap_secs: 0.5,
            include_partial_chunks: true,
            exclude_overlap: false,
        }
    }

    /// Whether to include the final partial window when a speech region
    /// does not divide evenly into `window_samples`. Defaults to `true`.
    /// When `false`, only full windows are embedded, which avoids zero-padded
    /// partial chunks that can produce misleading embeddings.
    pub fn with_include_partial_chunks(mut self, include: bool) -> Self {
        self.include_partial_chunks = include;
        self
    }

    /// Whether to exclude overlap segments from speech regions. Defaults to `false`.
    /// When `true`, only non-overlap segments are used for embedding extraction.
    pub fn with_exclude_overlap(mut self, exclude: bool) -> Self {
        self.exclude_overlap = exclude;
        self
    }

    /// Set the window size in samples. Default is 2 seconds (32000 @ 16 kHz).
    pub fn with_window_samples(mut self, samples: usize) -> Self {
        self.window_samples = samples;
        self
    }

    /// Set the hop size in samples. Default is 1.5 seconds (24000 @ 16 kHz).
    pub fn with_hop_samples(mut self, samples: usize) -> Self {
        self.hop_samples = samples;
        self
    }

    pub fn run(&self, samples: &[f32], sr: SampleRate) -> Result<DiarizationResult, PipelineError> {
        if sr.get() != self.sample_rate {
            return Err(PipelineError::UnsupportedSampleRate { actual: sr.get() });
        }

        let raw_segments = self.segmenter.segment(samples)?;
        if raw_segments.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let speech_regions = if self.exclude_overlap {
            extract_speech_regions_filtered(&raw_segments, |s| !s.is_overlap)
        } else {
            extract_speech_regions(&raw_segments)
        };
        if speech_regions.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let sr_f = self.sample_rate as f64;
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        let mut time_ranges: Vec<TimeRange> = Vec::new();

        for &(start_sec, end_sec) in &speech_regions {
            let start = (start_sec * sr_f) as usize;
            let end = (end_sec * sr_f) as usize;
            let region = &samples[start..end.min(samples.len())];

            if region.len() < self.window_samples {
                let mut padded = vec![0.0_f32; self.window_samples];
                padded[..region.len()].copy_from_slice(region);
                chunks.push(padded);
                time_ranges.push(TimeRange {
                    start: start_sec,
                    end: end_sec,
                });
            } else {
                for (offset, offset_end) in {
                    let iter = WindowIter::new(region.len(), self.window_samples, self.hop_samples);
                    if self.include_partial_chunks {
                        iter.include_partial()
                    } else {
                        iter
                    }
                } {
                    let chunk_len = offset_end - offset;
                    let chunk = if chunk_len < self.window_samples {
                        let mut padded = vec![0.0_f32; self.window_samples];
                        padded[..chunk_len].copy_from_slice(&region[offset..offset_end]);
                        padded
                    } else {
                        region[offset..offset_end].to_vec()
                    };
                    chunks.push(chunk);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr_f,
                        end: (start + offset_end) as f64 / sr_f,
                    });
                }
            }
        }

        let chunk_refs: Vec<&[f32]> = chunks.iter().map(|c| c.as_slice()).collect();
        let embeddings = self.embedder.embed_batch(&chunk_refs)?;

        if embeddings.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                turns: Vec::new(),
                num_speakers: 0,
            });
        }

        let labels = self.clusterer.cluster(&embeddings)?;
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

        segments = merge_segments(segments, self.max_gap_secs);
        segments.retain(|s| s.time.duration() >= self.min_speech_secs);

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

    /// Run the pipeline and return raw embeddings + timing for diagnostics.
    /// Does not apply post-processing (merge, filter), so the caller can
    /// experiment with different clustering parameters.
    pub fn run_diagnostics(
        &self,
        samples: &[f32],
        sr: SampleRate,
    ) -> Result<HybridDiagnostics, PipelineError> {
        if sr.get() != self.sample_rate {
            return Err(PipelineError::UnsupportedSampleRate { actual: sr.get() });
        }

        let raw_segments = self.segmenter.segment(samples)?;
        if raw_segments.is_empty() {
            return Ok(HybridDiagnostics {
                embeddings: Vec::new(),
                time_ranges: Vec::new(),
                raw_chunk_lengths: Vec::new(),
                labels: Vec::new(),
                num_speakers: 0,
            });
        }

        let speech_regions = if self.exclude_overlap {
            extract_speech_regions_filtered(&raw_segments, |s| !s.is_overlap)
        } else {
            extract_speech_regions(&raw_segments)
        };
        if speech_regions.is_empty() {
            return Ok(HybridDiagnostics {
                embeddings: Vec::new(),
                time_ranges: Vec::new(),
                raw_chunk_lengths: Vec::new(),
                labels: Vec::new(),
                num_speakers: 0,
            });
        }

        let sr_f = self.sample_rate as f64;
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        let mut time_ranges: Vec<TimeRange> = Vec::new();
        let mut raw_chunk_lengths: Vec<usize> = Vec::new();

        for &(start_sec, end_sec) in &speech_regions {
            let start = (start_sec * sr_f) as usize;
            let end = (end_sec * sr_f) as usize;
            let region = &samples[start..end.min(samples.len())];

            if region.len() < self.window_samples {
                let mut padded = vec![0.0_f32; self.window_samples];
                padded[..region.len()].copy_from_slice(region);
                chunks.push(padded);
                raw_chunk_lengths.push(region.len());
                time_ranges.push(TimeRange {
                    start: start_sec,
                    end: end_sec,
                });
            } else {
                for (offset, offset_end) in {
                    let iter = WindowIter::new(region.len(), self.window_samples, self.hop_samples);
                    if self.include_partial_chunks {
                        iter.include_partial()
                    } else {
                        iter
                    }
                } {
                    let chunk_len = offset_end - offset;
                    let chunk = if chunk_len < self.window_samples {
                        let mut padded = vec![0.0_f32; self.window_samples];
                        padded[..chunk_len].copy_from_slice(&region[offset..offset_end]);
                        padded
                    } else {
                        region[offset..offset_end].to_vec()
                    };
                    chunks.push(chunk);
                    raw_chunk_lengths.push(chunk_len);
                    time_ranges.push(TimeRange {
                        start: (start + offset) as f64 / sr_f,
                        end: (start + offset_end) as f64 / sr_f,
                    });
                }
            }
        }

        let chunk_refs: Vec<&[f32]> = chunks.iter().map(|c| c.as_slice()).collect();
        let embeddings = self.embedder.embed_batch(&chunk_refs)?;

        if embeddings.is_empty() {
            return Ok(HybridDiagnostics {
                embeddings: Vec::new(),
                time_ranges: Vec::new(),
                raw_chunk_lengths: Vec::new(),
                labels: Vec::new(),
                num_speakers: 0,
            });
        }

        let labels = self.clusterer.cluster(&embeddings)?;
        let num_speakers = labels.iter().copied().max().map_or(0, |m| m + 1);

        Ok(HybridDiagnostics {
            embeddings,
            time_ranges,
            raw_chunk_lengths,
            labels,
            num_speakers,
        })
    }
}

/// Raw output of the hybrid pipeline before post-processing.
pub struct HybridDiagnostics {
    pub embeddings: Vec<Vec<f32>>,
    pub time_ranges: Vec<TimeRange>,
    /// Length of each audio chunk **before** zero-padding.
    pub raw_chunk_lengths: Vec<usize>,
    pub labels: Vec<usize>,
    pub num_speakers: usize,
}

/// Build speech regions as the union of all segment time ranges,
/// ignoring speaker labels and overlap flags.
fn extract_speech_regions(segments: &[RawSegment]) -> Vec<(f64, f64)> {
    extract_speech_regions_filtered(segments, |_| true)
}

/// Build speech regions as the union of segment time ranges,
/// optionally excluding overlap segments.
fn extract_speech_regions_filtered(
    segments: &[RawSegment],
    include: impl Fn(&RawSegment) -> bool,
) -> Vec<(f64, f64)> {
    if segments.is_empty() {
        return Vec::new();
    }
    let mut intervals: Vec<(f64, f64)> = segments
        .iter()
        .filter(|s| include(s))
        .map(|s| (s.time.start, s.time.end))
        .collect();
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for &(start, end) in &intervals {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
            continue;
        }
        merged.push((start, end));
    }
    merged
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_speech_regions_merges_overlapping() {
        let segs = vec![
            RawSegment {
                time: TimeRange {
                    start: 0.0,
                    end: 1.0,
                },
                local_speaker_idx: 0,
                is_overlap: false,
                confidence: crate::types::Confidence::new(0.9).unwrap(),
            },
            RawSegment {
                time: TimeRange {
                    start: 0.5,
                    end: 2.0,
                },
                local_speaker_idx: 1,
                is_overlap: true,
                confidence: crate::types::Confidence::new(0.9).unwrap(),
            },
            RawSegment {
                time: TimeRange {
                    start: 3.0,
                    end: 4.0,
                },
                local_speaker_idx: 0,
                is_overlap: false,
                confidence: crate::types::Confidence::new(0.9).unwrap(),
            },
        ];
        let regions = extract_speech_regions(&segs);
        assert_eq!(regions, vec![(0.0, 2.0), (3.0, 4.0)]);
    }

    #[test]
    fn extract_speech_regions_empty() {
        let regions = extract_speech_regions(&[]);
        assert!(regions.is_empty());
    }
}
