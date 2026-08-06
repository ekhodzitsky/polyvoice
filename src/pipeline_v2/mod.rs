//! `polyvoice::pipeline_v2` — trait-wired production ONNX diarization pipeline.

#[cfg(not(all(
    feature = "onnx",
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
)))]
compile_error!(
    "pipeline_v2 requires onnx + download + segmentation + embedder + clusterer + resegmentation features"
);

pub mod builder;
pub mod config;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
pub mod mocks;

use crate::clusterer::{Clusterer, ClustererError};
use crate::embedder::{Embedder, EmbedderError, apply_overlap_mask};
use crate::models::RegistryError;
use crate::resegmentation::{
    OverlapRegionInput, ResegmentError, ResegmentInputs, Resegmenter, SpeakerCentroid,
    compute_centroids, extract_overlap_time_ranges,
};
use crate::segmentation::{SegmentationError, Segmenter};
use crate::types::{DiarizationResult, SampleRate, Segment, SpeakerId, SpeakerTurn, TimeRange};
use crate::utils::{l2_normalize, merge_segments};

pub use builder::{ConfigError, PipelineBuilder};
pub use config::{ClustererKind, ExecutionProvider, PipelineConfig};

/// Wall-clock seconds per pipeline stage, from [`Pipeline::run_with_timings`].
/// In pipeline v2 voice-activity detection is part of the powerset segmenter,
/// so there is no separate VAD stage.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct StageTimings {
    pub segmentation_secs: f64,
    pub embedding_secs: f64,
    pub clustering_secs: f64,
    pub resegmentation_secs: f64,
}

/// Minimum segment length (seconds) accepted for embedding. WeSpeaker ResNet34
/// downsamples the time axis ~8×, so a segment shorter than ~0.11s (≈10 fbank
/// frames at win=400/hop=160) leaves ≤1 frame after downsampling and the
/// temporal statistics-pooling std collapses to sqrt(≈0) → NaN. Empirically the
/// clean boundary is 0.119s; 0.20s keeps a safe margin while still feeding the
/// clusterer every segment below the 0.25s min_speech output filter.
const MIN_EMBED_SECS: f64 = 0.20;

/// Expand primary segments into embedding units. With `window = None` each
/// segment is one unit (sparse, one embedding per segment). With `Some(w)` each
/// segment longer than `w` is split into `w`-second sub-windows hopped by `w/2`
/// (dense, legacy-style), every sub-window inheriting the segment's local speaker
/// index; sub-`w` segments stay whole. The returned units are owned clones so
/// downstream zips by sub-window time.
fn expand_embed_units(
    segs: &[crate::segmentation::RawSegment],
    window: Option<f32>,
) -> Vec<crate::segmentation::RawSegment> {
    let w = match window {
        Some(w) if w > 0.0 => w as f64,
        _ => return segs.to_vec(),
    };
    let hop = (w / 2.0).max(0.05);
    let mut out = Vec::with_capacity(segs.len());
    for seg in segs {
        if seg.time.end - seg.time.start <= w {
            out.push(seg.clone());
            continue;
        }
        let mut t = seg.time.start;
        loop {
            let end = (t + w).min(seg.time.end);
            let mut sub = seg.clone();
            sub.time = TimeRange { start: t, end };
            out.push(sub);
            if end >= seg.time.end {
                break;
            }
            t += hop;
        }
    }
    out
}

/// Zip each surviving source segment with its cluster label into a turn.
/// Sources, not primary segments: dropped segments produced no embedding and
/// have no label, so the zip must stay parallel to the embeddings.
fn primary_turns_from_labels(
    sources: &[crate::segmentation::RawSegment],
    labels: &[usize],
) -> Vec<SpeakerTurn> {
    sources
        .iter()
        .zip(labels.iter())
        .map(|(seg, &lbl)| SpeakerTurn {
            speaker: SpeakerId(lbl as u32),
            time: seg.time,
            text: None,
            stable: true,
        })
        .collect()
}

/// Sum and count of the window confidences contributing to one turn: windows
/// labeled as the turn's speaker whose midpoint falls in
/// `[turn.start, turn.end)`. Contributions accumulate in ascending window
/// order, so the f32 sum is identical whether the candidate range is located
/// by binary search (`mids_sorted = true`) or by scanning every window.
fn window_confidence_sum(
    turn: &SpeakerTurn,
    speaker_ids: &[SpeakerId],
    window_conf: &[f32],
    mids: &[f64],
    mids_sorted: bool,
) -> (f32, u32) {
    let candidates = if mids_sorted {
        // Non-decreasing midpoints: the windows that can match the midpoint
        // predicate form one contiguous range — binary-search its bounds
        // instead of scanning every window. The per-window predicate below is
        // still evaluated, keeping one code path for both modes.
        let lo = mids.partition_point(|&m| m < turn.time.start);
        let hi = mids.partition_point(|&m| m < turn.time.end);
        lo..hi
    } else {
        0..mids.len()
    };
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for i in candidates {
        if speaker_ids.get(i).copied() != Some(turn.speaker) {
            continue;
        }
        if mids[i] >= turn.time.start
            && mids[i] < turn.time.end
            && let Some(&c) = window_conf.get(i)
        {
            sum += c;
            n += 1;
        }
    }
    (sum, n)
}

/// Hard cap on PCM length accepted by [`Pipeline::run`] / [`Pipeline::run_with_timings`].
/// Matches the C FFI (`MAX_SAMPLES`) and the WAV loader's ~1-hour policy so library
/// and Python callers cannot unbounded-allocate on untrusted buffers.
pub const MAX_AUDIO_SAMPLES: usize = 16_000 * 3_600;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("audio sample rate {actual} unsupported, expected 16000")]
    UnsupportedSampleRate { actual: u32 },
    #[error(
        "audio too long: {actual_samples} samples exceeds max {max_samples} (~1 hour at 16 kHz)"
    )]
    AudioTooLong {
        actual_samples: usize,
        max_samples: usize,
    },
    #[error("segmentation failed: {0}")]
    Segmentation(#[from] SegmentationError),
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbedderError),
    #[error("clustering failed: {0}")]
    Clustering(#[from] ClustererError),
    #[error("resegmentation failed: {0}")]
    Resegment(#[from] ResegmentError),
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
    #[error("model registry error: {0}")]
    Registry(#[from] RegistryError),
}

pub struct Pipeline {
    config: PipelineConfig,
    segmenter: Box<dyn Segmenter>,
    embedder: Box<dyn Embedder>,
    clusterer: Box<dyn Clusterer>,
    resegmenter: Box<dyn Resegmenter>,
}

impl Pipeline {
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }

    pub(crate) fn from_components(
        config: PipelineConfig,
        segmenter: Box<dyn Segmenter>,
        embedder: Box<dyn Embedder>,
        clusterer: Box<dyn Clusterer>,
        resegmenter: Box<dyn Resegmenter>,
    ) -> Self {
        Self {
            config,
            segmenter,
            embedder,
            clusterer,
            resegmenter,
        }
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    pub fn run(&self, samples: &[f32], sr: SampleRate) -> Result<DiarizationResult, PipelineError> {
        self.run_with_timings(samples, sr).map(|(result, _)| result)
    }

    /// Like [`Pipeline::run`], but also returns wall-clock seconds spent in each
    /// pipeline stage. Benchmark entry point (RTFx breakdown per backend); the
    /// instrumentation is four `Instant` reads, so `run` simply delegates here.
    pub fn run_with_timings(
        &self,
        samples: &[f32],
        sr: SampleRate,
    ) -> Result<(DiarizationResult, StageTimings), PipelineError> {
        if sr.get() != self.config.sample_rate.get() {
            return Err(PipelineError::UnsupportedSampleRate { actual: sr.get() });
        }
        if samples.len() > MAX_AUDIO_SAMPLES {
            return Err(PipelineError::AudioTooLong {
                actual_samples: samples.len(),
                max_samples: MAX_AUDIO_SAMPLES,
            });
        }
        let mut timings = StageTimings::default();

        let t = std::time::Instant::now();
        let raw_segments = self.segmenter.segment(samples)?;
        timings.segmentation_secs = t.elapsed().as_secs_f64();
        if raw_segments.is_empty() {
            return Ok((DiarizationResult::new(Vec::new(), Vec::new(), 0), timings));
        }

        let overlap_ranges = extract_overlap_time_ranges(&raw_segments);
        let primary_segments: Vec<_> = raw_segments
            .iter()
            .filter(|s| !s.is_overlap)
            .cloned()
            .collect();

        let t = std::time::Instant::now();
        let (embeddings, sources) =
            self.embed_primary_segments(&primary_segments, &overlap_ranges, samples)?;
        timings.embedding_secs = t.elapsed().as_secs_f64();
        if embeddings.is_empty() {
            return Ok((DiarizationResult::new(Vec::new(), Vec::new(), 0), timings));
        }

        let t = std::time::Instant::now();
        let labels = self.cluster_embeddings(&embeddings, &sources)?;
        timings.clustering_secs = t.elapsed().as_secs_f64();

        let primary_turns = primary_turns_from_labels(&sources, &labels);
        let centroids: Vec<SpeakerCentroid> = compute_centroids(&embeddings, &labels);

        // Bridge the powerset segmenter's file-consistent local speaker indices
        // to global clusters via Hungarian assignment on co-occurrence duration,
        // with cannot-link constraints from overlap pairs (two locals that share
        // an overlap must not collapse onto one global).
        let cannot_link: Vec<(u8, u8)> = overlap_ranges
            .iter()
            .map(|(_, lo, hi)| (*lo.min(hi), *lo.max(hi)))
            .collect();
        let local_to_global = self.map_local_to_global(&sources, &labels, &cannot_link);

        let t = std::time::Instant::now();
        let mut all_turns = self.resegment_turns(
            &overlap_ranges,
            &centroids,
            &primary_turns,
            &local_to_global,
            samples,
        )?;
        timings.resegmentation_secs = t.elapsed().as_secs_f64();

        // Guarantee sorted-by-start output regardless of which Resegmenter impl ran.
        all_turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));

        let min_secs = self.config.min_speech_secs as f64;
        all_turns.retain(|t| t.time.duration() >= min_secs);

        let (merged_segments, merged_turns) =
            self.merge_with_confidence(&all_turns, &sources, &labels, &embeddings);

        let num_speakers = merged_turns
            .iter()
            .map(|t| t.speaker.0)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let result = DiarizationResult::new(merged_segments, merged_turns, num_speakers)
            .with_audio(samples.len() as f64 / sr.get() as f64, sr.get())
            .with_provenance(crate::types::Provenance {
                profile: self.config.profile.manifest_id().to_owned(),
                ..Default::default()
            });
        Ok((result, timings))
    }

    /// Embedding stage: expand primary segments into embedding units, mask
    /// overlap audio out of each unit, and embed. Units that are empty, below
    /// MIN_EMBED_SECS, or yield a non-finite embedding are dropped, so the
    /// returned source segments (parallel to the embeddings) may be fewer than
    /// `primary_segments`.
    fn embed_primary_segments(
        &self,
        primary_segments: &[crate::segmentation::RawSegment],
        overlap_ranges: &[(TimeRange, u8, u8)],
        samples: &[f32],
    ) -> Result<(Vec<Vec<f32>>, Vec<crate::segmentation::RawSegment>), PipelineError> {
        let sample_rate = self.config.sample_rate.get() as f64;
        // Optional dense embedding: split each primary segment into overlapping
        // sub-windows so a speaker run yields several embeddings (legacy-style
        // dense windows) for more robust centroids. `None` keeps one embedding
        // per segment. Each unit carries its parent's local speaker index.
        let embed_units = expand_embed_units(primary_segments, self.config.embed_window_secs);
        // PLDA backends (VBx) need the raw embedding scale for mean-centering;
        // cosine backends are scale-invariant and get the L2-normalized vectors.
        let raw_embeddings = self.clusterer.wants_raw_embeddings();
        // First pass: slice each unit out of the waveform and zero-fill its
        // overlap regions. Masking is cheap and pure, so doing it up front lets
        // the embed stage below consume ready-made chunks in one batch while
        // `kept` preserves the original unit order for result pairing.
        let mut masked_chunks: Vec<Vec<f32>> = Vec::with_capacity(embed_units.len());
        let mut kept: Vec<crate::segmentation::RawSegment> = Vec::with_capacity(embed_units.len());
        for seg in embed_units {
            let start_idx = (seg.time.start * sample_rate) as usize;
            let end_idx = ((seg.time.end * sample_rate) as usize).min(samples.len());
            if end_idx <= start_idx {
                continue;
            }
            // Skip segments too short to embed without NaN (see MIN_EMBED_SECS).
            if (end_idx - start_idx) as f64 / sample_rate < MIN_EMBED_SECS {
                continue;
            }
            let chunk = &samples[start_idx..end_idx];
            // Zero-fill any overlap regions inside this primary chunk before
            // embedding, so two-speaker audio cannot bias the embedding.
            let seg_start = seg.time.start;
            let seg_end = seg.time.end;
            let local_overlaps: Vec<(f32, f32)> = overlap_ranges
                .iter()
                .filter_map(|(ot, _, _)| {
                    let lo = ot.start.max(seg_start);
                    let hi = ot.end.min(seg_end);
                    if hi > lo {
                        Some(((lo - seg_start) as f32, (hi - seg_start) as f32))
                    } else {
                        None
                    }
                })
                .collect();
            let masked = apply_overlap_mask(chunk, &local_overlaps, self.config.sample_rate.get());
            masked_chunks.push(masked);
            kept.push(seg);
        }
        // Embed all units as one batch: ONNX-backed embedders fan this out
        // across threads over their internal session pool, so units no longer
        // serialize on a single session. `embed_batch` preserves input order,
        // so the zip below restores the per-unit pairing deterministically and
        // surfaces the first error in unit order, as the sequential loop did.
        let chunk_refs: Vec<&[f32]> = masked_chunks.iter().map(Vec::as_slice).collect();
        let batch = self.embedder.embed_batch(&chunk_refs)?;
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch.len());
        let mut sources: Vec<crate::segmentation::RawSegment> = Vec::with_capacity(batch.len());
        for (seg, mut emb) in kept.into_iter().zip(batch) {
            // Defense in depth: never let a non-finite embedding reach the clusterer.
            if !emb.iter().all(|v| v.is_finite()) {
                tracing::warn!(
                    "skipping non-finite embedding for segment {:.3}-{:.3}s",
                    seg.time.start,
                    seg.time.end
                );
                continue;
            }
            if !raw_embeddings {
                l2_normalize(&mut emb);
            }
            embeddings.push(emb);
            sources.push(seg);
        }
        Ok((embeddings, sources))
    }

    /// Clustering stage. Per-embedding durations enable cVBx short-segment
    /// filtering inside clusterers that opt in (VBx); others ignore the slice.
    fn cluster_embeddings(
        &self,
        embeddings: &[Vec<f32>],
        sources: &[crate::segmentation::RawSegment],
    ) -> Result<Vec<usize>, PipelineError> {
        let durations: Vec<f64> = sources.iter().map(|s| s.time.duration()).collect();
        Ok(self
            .clusterer
            .cluster_with_durations(embeddings, &durations)?)
    }

    /// Resegmentation stage: when overlap resegmentation is enabled and there
    /// are overlaps plus at least two centroids, reassign overlap regions to
    /// two speakers; otherwise the primary turns pass through, sorted by start.
    fn resegment_turns(
        &self,
        overlap_ranges: &[(TimeRange, u8, u8)],
        centroids: &[SpeakerCentroid],
        primary_turns: &[SpeakerTurn],
        local_to_global: &std::collections::HashMap<u8, SpeakerId>,
        samples: &[f32],
    ) -> Result<Vec<SpeakerTurn>, PipelineError> {
        if self.config.resegment_overlap && !overlap_ranges.is_empty() && centroids.len() >= 2 {
            let overlap_inputs =
                self.build_overlap_inputs(overlap_ranges, primary_turns, local_to_global, samples)?;
            Ok(self.resegmenter.resegment(ResegmentInputs {
                primary_turns,
                speaker_centroids: centroids,
                overlap_regions: &overlap_inputs,
            })?)
        } else {
            let mut turns = primary_turns.to_vec();
            turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            Ok(turns)
        }
    }

    /// Confidence + merge stage: score each turn by averaging the window
    /// confidences (embedding↔centroid cosine, logistic map) whose midpoint
    /// falls inside the turn (fallback: None), then merge adjacent same-speaker
    /// segments within `max_gap`. After merge_segments the per-run confidence
    /// is the mean of present values.
    fn merge_with_confidence(
        &self,
        turns: &[SpeakerTurn],
        sources: &[crate::segmentation::RawSegment],
        labels: &[usize],
        embeddings: &[Vec<f32>],
    ) -> (Vec<Segment>, Vec<SpeakerTurn>) {
        let max_gap = self.config.max_gap_secs as f64;
        let speaker_ids: Vec<SpeakerId> = labels.iter().map(|&l| SpeakerId(l as u32)).collect();
        let window_conf =
            crate::types::segment_confidences_from_embeddings(&speaker_ids, embeddings);
        // Window midpoints. Primary segments are disjoint and time-ordered in
        // production, so `mids` is non-decreasing and the per-turn confidence
        // lookup can binary-search its candidate range instead of scanning
        // every window (O(turns × windows) → O(turns × log windows));
        // unordered (mock/adversarial) input takes the exact full-scan path.
        let mids: Vec<f64> = sources
            .iter()
            .map(|s| (s.time.start + s.time.end) / 2.0)
            .collect();
        let mids_sorted = mids.windows(2).all(|w| w[0] <= w[1]);
        let merged_segments: Vec<Segment> = turns
            .iter()
            .map(|t| {
                let (sum, n) =
                    window_confidence_sum(t, &speaker_ids, &window_conf, &mids, mids_sorted);
                Segment {
                    time: t.time,
                    speaker: Some(t.speaker),
                    confidence: if n > 0 { Some(sum / n as f32) } else { None },
                }
            })
            .collect();
        let merged_segments = merge_segments(merged_segments, max_gap);
        let merged_turns: Vec<SpeakerTurn> = merged_segments
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
        (merged_segments, merged_turns)
    }

    /// Hungarian co-occurrence map from each file-consistent local speaker
    /// index to a global cluster. Only locals that appear as primary segments
    /// participate (inactive locals are never invented — the pyannote-style
    /// "inactive speakers in the similarity matrix" anti-pattern). Cannot-link
    /// pairs from overlap regions are forced onto distinct globals.
    fn map_local_to_global(
        &self,
        sources: &[crate::segmentation::RawSegment],
        labels: &[usize],
        cannot_link: &[(u8, u8)],
    ) -> std::collections::HashMap<u8, SpeakerId> {
        // Ablation toggle (PipelineConfig::disable_seg_overlap): return an
        // empty map so every overlap region takes the mixed-embedding fallback.
        // Lets the segmentation-derived overlap path be A/B-measured against
        // the legacy path in one binary.
        if self.config.disable_seg_overlap {
            return std::collections::HashMap::new();
        }
        let local_idx: Vec<u8> = sources.iter().map(|s| s.local_speaker_idx).collect();
        let durations: Vec<f64> = sources.iter().map(|s| s.time.duration()).collect();
        let cooc = crate::clusterer::build_cooccurrence(&local_idx, labels, &durations);
        // Ablation (PipelineConfig::majority_local_map): majority vote instead
        // of Hungarian.
        if self.config.majority_local_map {
            return crate::clusterer::majority_local_to_global(&cooc);
        }
        crate::clusterer::hungarian_local_to_global(&cooc, cannot_link)
    }

    fn build_overlap_inputs(
        &self,
        overlap_ranges: &[(TimeRange, u8, u8)],
        primary_turns: &[SpeakerTurn],
        local_to_global: &std::collections::HashMap<u8, SpeakerId>,
        samples: &[f32],
    ) -> Result<Vec<OverlapRegionInput>, PipelineError> {
        let sample_rate = self.config.sample_rate.get() as f64;
        let mut out = Vec::with_capacity(overlap_ranges.len());
        for (time, lo, hi) in overlap_ranges {
            let g_lo = local_to_global.get(lo).copied();
            let g_hi = local_to_global.get(hi).copied();

            // Both local speakers of the overlap map to a global cluster: take
            // the segmenter's own two-speaker assignment directly and skip the
            // mixed-voice embedding entirely (the overlap-accuracy win).
            if let (Some(a), Some(b)) = (g_lo, g_hi) {
                out.push(OverlapRegionInput {
                    time: *time,
                    primary_speaker: a,
                    secondary_speaker: Some(b),
                    embedding: Vec::new(),
                });
                continue;
            }

            // At least one local index never appeared as a solo segment, so its
            // global identity is unknown. Anchor on whichever local did map (or
            // the nearest primary turn by midpoint) and recover the other speaker
            // downstream from a mixed-region embedding.
            let primary = g_lo.or(g_hi).unwrap_or_else(|| {
                primary_turns
                    .iter()
                    .find(|t| t.time.start <= time.start && time.end <= t.time.end)
                    .map(|t| t.speaker)
                    .unwrap_or_else(|| {
                        let mid = (time.start + time.end) / 2.0;
                        let tmid = |t: &SpeakerTurn| (t.time.start + t.time.end) / 2.0;
                        primary_turns
                            .iter()
                            .min_by(|a, b| (tmid(a) - mid).abs().total_cmp(&(tmid(b) - mid).abs()))
                            .map(|t| t.speaker)
                            .unwrap_or(SpeakerId(0))
                    })
            });
            let start_idx = (time.start * sample_rate) as usize;
            let end_idx = ((time.end * sample_rate) as usize).min(samples.len());
            if end_idx <= start_idx {
                continue;
            }
            // Skip overlap chunks too short to embed without NaN (see MIN_EMBED_SECS).
            if (end_idx - start_idx) as f64 / sample_rate < MIN_EMBED_SECS {
                continue;
            }
            let chunk = &samples[start_idx..end_idx];
            let mut emb = self.embedder.embed(chunk)?;
            if !emb.iter().all(|v| v.is_finite()) {
                continue;
            }
            l2_normalize(&mut emb);
            out.push(OverlapRegionInput {
                time: *time,
                primary_speaker: primary,
                secondary_speaker: None,
                embedding: emb,
            });
        }
        Ok(out)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_v2::mocks::{MockClusterer, MockEmbedder, MockSegmenter, raw_segment};
    use crate::resegmentation::OverlapResegmenter;
    use crate::types::Profile;
    use proptest::prelude::*;

    #[test]
    fn expand_embed_units_none_is_identity() {
        let segs = vec![
            raw_segment(0.0, 5.0, 0, false),
            raw_segment(6.0, 7.0, 1, false),
        ];
        let out = expand_embed_units(&segs, None);
        assert_eq!(out, segs);
    }

    #[test]
    fn expand_embed_units_splits_long_keeps_short() {
        // 5s segment with a 1.5s window (0.75 hop) → several sub-windows, each
        // inheriting the parent's local speaker index and ending at the segment
        // boundary; a 1s segment (< window) stays whole.
        let segs = vec![
            raw_segment(0.0, 5.0, 2, false),
            raw_segment(6.0, 7.0, 1, false),
        ];
        let out = expand_embed_units(&segs, Some(1.5));
        let long: Vec<_> = out.iter().filter(|s| s.local_speaker_idx == 2).collect();
        assert!(
            long.len() >= 4,
            "5s/1.5s should yield >=4 sub-windows, got {}",
            long.len()
        );
        assert!(
            long.iter()
                .all(|s| s.time.start >= 0.0 && s.time.end <= 5.0 + 1e-9)
        );
        assert!(
            long.iter()
                .all(|s| (s.time.end - s.time.start) <= 1.5 + 1e-9)
        );
        assert_eq!(
            long.last().unwrap().time.end,
            5.0,
            "last sub-window ends at the segment boundary"
        );
        // short segment untouched
        let short: Vec<_> = out.iter().filter(|s| s.local_speaker_idx == 1).collect();
        assert_eq!(short.len(), 1);
        assert_eq!(
            short[0].time,
            TimeRange {
                start: 6.0,
                end: 7.0
            }
        );
    }

    fn pipeline_with_segments(segs: Vec<crate::segmentation::RawSegment>) -> Pipeline {
        let cfg = PipelineConfig {
            profile: Profile::Custom,
            resegment_overlap: false,
            min_speech_secs: 0.0,
            max_gap_secs: 0.0,
            ..PipelineConfig::default()
        };
        Pipeline::from_components(
            cfg,
            Box::new(MockSegmenter { segments: segs }),
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        )
    }

    #[test]
    fn pipeline_run_unsupported_sample_rate_returns_err() {
        let p = pipeline_with_segments(vec![raw_segment(0.0, 1.0, 0, false)]);
        let bad = SampleRate::new(8000).unwrap();
        let err = p.run(&vec![0.0_f32; 8000], bad).unwrap_err();
        assert!(matches!(
            err,
            PipelineError::UnsupportedSampleRate { actual: 8000 }
        ));
    }

    #[test]
    fn max_audio_samples_matches_one_hour_at_16khz() {
        // Parity with FFI (`MAX_SAMPLES`) and the WAV ~1 h policy.
        assert_eq!(MAX_AUDIO_SAMPLES, 16_000 * 3_600);
    }

    #[test]
    fn pipeline_run_silence_returns_empty() {
        let p = pipeline_with_segments(Vec::new());
        let result = p
            .run(&vec![0.0_f32; 16000], SampleRate::new(16000).unwrap())
            .unwrap();
        assert!(result.turns.is_empty());
        assert_eq!(result.num_speakers, 0);
    }

    #[test]
    fn pipeline_run_two_segments_one_cluster() {
        let segs = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(1.5, 2.5, 0, false),
        ];
        let p = pipeline_with_segments(segs);
        let result = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        assert_eq!(result.num_speakers, 1);
        assert!(!result.turns.is_empty());
    }

    #[test]
    fn pipeline_resegment_overlap_disabled_path_used() {
        let segs = vec![
            raw_segment(0.0, 1.0, 0, true),
            raw_segment(0.0, 1.0, 1, true),
            raw_segment(1.5, 2.5, 0, false),
        ];
        let p = pipeline_with_segments(segs);
        let result = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        assert!(result.num_speakers <= 1);
    }

    fn pipeline_with_embedder(
        segs: Vec<crate::segmentation::RawSegment>,
        embedder: Box<dyn Embedder>,
    ) -> Pipeline {
        let cfg = PipelineConfig {
            profile: Profile::Custom,
            resegment_overlap: false,
            min_speech_secs: 0.0,
            max_gap_secs: 0.0,
            ..PipelineConfig::default()
        };
        Pipeline::from_components(
            cfg,
            Box::new(MockSegmenter { segments: segs }),
            embedder,
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        )
    }

    /// Records the shortest audio slice the embedder was asked to embed.
    struct RecordingEmbedder {
        min_samples_seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Embedder for RecordingEmbedder {
        fn dim(&self) -> usize {
            192
        }
        fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            self.min_samples_seen
                .fetch_min(audio.len(), std::sync::atomic::Ordering::SeqCst);
            let mut v = vec![0.0_f32; 192];
            v[0] = 1.0;
            Ok(v)
        }
    }

    /// Always emits a non-finite embedding.
    struct NanEmbedder;

    impl Embedder for NanEmbedder {
        fn dim(&self) -> usize {
            192
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            let mut v = vec![0.1_f32; 192];
            v[0] = f32::NAN;
            Ok(v)
        }
    }

    #[test]
    fn pipeline_skips_segments_below_min_embed_secs() {
        // One sub-threshold segment (0.05s) and one well above it (1.0s).
        let segs = vec![
            raw_segment(0.0, 0.05, 0, false),
            raw_segment(1.0, 2.0, 0, false),
        ];
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let embedder = Box::new(RecordingEmbedder {
            min_samples_seen: counter.clone(),
        });
        let p = pipeline_with_embedder(segs, embedder);
        let _ = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        let min_seen = counter.load(std::sync::atomic::Ordering::SeqCst);
        // The 0.05s (800-sample) segment must never have reached the embedder.
        assert!(
            min_seen as f64 / 16000.0 >= MIN_EMBED_SECS,
            "shortest embedded slice was {min_seen} samples, below MIN_EMBED_SECS floor"
        );
    }

    #[test]
    fn pipeline_skips_non_finite_embeddings() {
        // Two long-enough segments, but the embedder emits NaN for both, so
        // every embedding is dropped and nothing poisons the clusterer.
        let segs = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(2.0, 3.0, 0, false),
        ];
        let p = pipeline_with_embedder(segs, Box::new(NanEmbedder));
        let result = p
            .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
            .unwrap();
        assert!(result.turns.is_empty());
        assert_eq!(result.num_speakers, 0);
    }

    /// The binary-search fast path (sorted mids) and the full-scan fallback
    /// must both agree with a naive per-turn scan — same contributing windows,
    /// same accumulation order, bit-identical sums.
    #[test]
    fn window_confidence_sum_matches_naive_scan() {
        fn naive(
            turn: &SpeakerTurn,
            speaker_ids: &[SpeakerId],
            window_conf: &[f32],
            mids: &[f64],
        ) -> (f32, u32) {
            let mut sum = 0.0f32;
            let mut n = 0u32;
            for (i, &mid) in mids.iter().enumerate() {
                if speaker_ids.get(i).copied() != Some(turn.speaker) {
                    continue;
                }
                if mid >= turn.time.start
                    && mid < turn.time.end
                    && let Some(&c) = window_conf.get(i)
                {
                    sum += c;
                    n += 1;
                }
            }
            (sum, n)
        }

        let speaker_ids = vec![
            SpeakerId(0),
            SpeakerId(1),
            SpeakerId(0),
            SpeakerId(0),
            SpeakerId(1),
        ];
        let window_conf = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let sorted_mids = vec![0.5, 1.0, 1.5, 2.5, 3.5];
        // Same values, shuffled: not non-decreasing, fallback path only.
        let unsorted_mids = vec![1.5, 0.5, 3.5, 1.0, 2.5];

        let cases = [
            // Boundaries: mid == start is included, mid == end is excluded.
            SpeakerTurn {
                speaker: SpeakerId(0),
                time: TimeRange {
                    start: 0.5,
                    end: 2.5,
                },
                text: None,
                stable: true,
            },
            // No window of the right speaker in range → (0.0, 0).
            SpeakerTurn {
                speaker: SpeakerId(1),
                time: TimeRange {
                    start: 1.6,
                    end: 2.4,
                },
                text: None,
                stable: true,
            },
            // Range outside every midpoint.
            SpeakerTurn {
                speaker: SpeakerId(0),
                time: TimeRange {
                    start: 10.0,
                    end: 11.0,
                },
                text: None,
                stable: true,
            },
        ];

        for turn in &cases {
            let expected = naive(turn, &speaker_ids, &window_conf, &sorted_mids);
            assert_eq!(
                window_confidence_sum(turn, &speaker_ids, &window_conf, &sorted_mids, true),
                expected,
                "sorted fast path disagrees with naive scan"
            );
            assert_eq!(
                window_confidence_sum(turn, &speaker_ids, &window_conf, &sorted_mids, false),
                expected,
                "full-scan path disagrees with naive scan"
            );
            let expected_unsorted = naive(turn, &speaker_ids, &window_conf, &unsorted_mids);
            assert_eq!(
                window_confidence_sum(turn, &speaker_ids, &window_conf, &unsorted_mids, false),
                expected_unsorted,
                "unsorted full-scan path disagrees with naive scan"
            );
        }
    }

    // --- Error propagation, toggles, and overlap-input construction ---

    fn custom_cfg() -> PipelineConfig {
        PipelineConfig {
            profile: Profile::Custom,
            resegment_overlap: false,
            min_speech_secs: 0.0,
            max_gap_secs: 0.0,
            ..PipelineConfig::default()
        }
    }

    fn pipeline_custom(
        cfg: PipelineConfig,
        segs: Vec<crate::segmentation::RawSegment>,
        embedder: Box<dyn Embedder>,
        clusterer: Box<dyn Clusterer>,
        resegmenter: Box<dyn Resegmenter>,
    ) -> Pipeline {
        Pipeline::from_components(
            cfg,
            Box::new(MockSegmenter { segments: segs }),
            embedder,
            clusterer,
            resegmenter,
        )
    }

    fn turn(spk: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        }
    }

    /// Segmenter that always fails.
    struct FailingSegmenter;

    impl Segmenter for FailingSegmenter {
        fn segment(
            &self,
            _audio: &[f32],
        ) -> Result<Vec<crate::segmentation::RawSegment>, SegmentationError> {
            Err(SegmentationError::AudioTooShort {
                actual_secs: 0.0,
                min_secs: 0.1,
            })
        }
        fn max_local_speakers(&self) -> usize {
            3
        }
        fn supports_overlap(&self) -> bool {
            true
        }
    }

    /// Embedder that always fails.
    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn dim(&self) -> usize {
            192
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Err(EmbedderError::InferenceFailed {
                detail: "boom".to_owned(),
            })
        }
    }

    /// Resegmenter that always fails.
    struct FailingResegmenter;

    impl Resegmenter for FailingResegmenter {
        fn resegment(
            &self,
            _inputs: ResegmentInputs<'_>,
        ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
            Err(ResegmentError::MissingPrimaryCentroid {
                index: 0,
                primary: SpeakerId(0),
            })
        }
    }

    /// Two solo segments (locals `solo.0`, `solo.1`) plus an overlap pair
    /// (locals `overlap.0`, `overlap.1`) at 1.5–2.0s; the clusterer maps the
    /// two solo embeddings to distinct clusters.
    fn overlap_pipeline(
        resegmenter: Box<dyn Resegmenter>,
        solo: (u8, u8),
        overlap: (u8, u8),
    ) -> Pipeline {
        let segs = vec![
            raw_segment(0.0, 1.0, solo.0, false),
            raw_segment(1.5, 2.0, overlap.0, true),
            raw_segment(1.5, 2.0, overlap.1, true),
            raw_segment(2.5, 3.5, solo.1, false),
        ];
        let mut cfg = custom_cfg();
        cfg.resegment_overlap = true;
        pipeline_custom(
            cfg,
            segs,
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer { labels: vec![0, 1] }),
            resegmenter,
        )
    }

    #[test]
    fn expand_embed_units_nonpositive_window_is_identity() {
        let segs = vec![raw_segment(0.0, 5.0, 0, false)];
        assert_eq!(expand_embed_units(&segs, Some(0.0)), segs);
        assert_eq!(expand_embed_units(&segs, Some(-1.0)), segs);
    }

    #[test]
    fn pipeline_builder_and_config_accessors() {
        let _builder = Pipeline::builder();
        let p = pipeline_with_segments(Vec::new());
        assert_eq!(p.config().profile, Profile::Custom);
    }

    #[test]
    fn stage_timings_serialize_and_run_reports_them() {
        let t = StageTimings {
            segmentation_secs: 1.0,
            embedding_secs: 2.0,
            clustering_secs: 3.0,
            resegmentation_secs: 4.0,
        };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("segmentation_secs"));
        assert!(json.contains("resegmentation_secs"));

        let p = pipeline_with_segments(vec![raw_segment(0.0, 1.0, 0, false)]);
        let (_result, timings) = p
            .run_with_timings(&vec![0.0_f32; 16000 * 2], SampleRate::new(16000).unwrap())
            .unwrap();
        assert!(timings.segmentation_secs >= 0.0);
        assert!(timings.embedding_secs >= 0.0);
        assert!(timings.clustering_secs >= 0.0);
        assert!(timings.resegmentation_secs >= 0.0);
    }

    #[test]
    fn pipeline_run_segmentation_error_propagates() {
        let p = Pipeline::from_components(
            custom_cfg(),
            Box::new(FailingSegmenter),
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        );
        let err = p
            .run(&vec![0.0_f32; 16000], SampleRate::new(16000).unwrap())
            .unwrap_err();
        assert!(matches!(err, PipelineError::Segmentation(_)));
    }

    #[test]
    fn pipeline_run_embedding_error_propagates() {
        let p = pipeline_custom(
            custom_cfg(),
            vec![raw_segment(0.0, 1.0, 0, false)],
            Box::new(FailingEmbedder),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        );
        let err = p
            .run(&vec![0.0_f32; 16000 * 2], SampleRate::new(16000).unwrap())
            .unwrap_err();
        assert!(matches!(err, PipelineError::Embedding(_)));
    }

    #[test]
    fn pipeline_run_clustering_error_propagates() {
        // Two embeddings but only one canned label → length-mismatch error.
        let p = pipeline_custom(
            custom_cfg(),
            vec![
                raw_segment(0.0, 1.0, 0, false),
                raw_segment(2.0, 3.0, 0, false),
            ],
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer { labels: vec![0] }),
            Box::new(OverlapResegmenter::default()),
        );
        let err = p
            .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
            .unwrap_err();
        assert!(matches!(err, PipelineError::Clustering(_)));
    }

    #[test]
    fn pipeline_run_resegment_error_propagates() {
        let p = overlap_pipeline(Box::new(FailingResegmenter), (0, 1), (0, 1));
        let err = p
            .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
            .unwrap_err();
        assert!(matches!(err, PipelineError::Resegment(_)));
    }

    #[test]
    fn pipeline_run_overlap_direct_assignment_when_both_locals_map() {
        let p = overlap_pipeline(Box::new(OverlapResegmenter::default()), (0, 1), (0, 1));
        let result = p
            .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
            .unwrap();
        assert_eq!(result.num_speakers, 2);
        // The overlap region 1.5–2.0s is emitted for both speakers.
        for spk in [0u32, 1] {
            assert!(
                result.turns.iter().any(|t| t.speaker.0 == spk
                    && t.time.start <= 1.5 + 1e-9
                    && t.time.end >= 2.0 - 1e-9),
                "speaker {spk} missing a turn covering the overlap region"
            );
        }
    }

    #[test]
    fn pipeline_run_overlap_mixed_embedding_fallback_when_local_unmapped() {
        // Overlap locals are (0, 2) but local 2 never appears solo, so its
        // global identity is unknown and the mixed-region embedding fallback
        // recovers the second speaker from the nearest non-primary centroid.
        let p = overlap_pipeline(Box::new(OverlapResegmenter::default()), (0, 1), (0, 2));
        let result = p
            .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
            .unwrap();
        assert_eq!(result.num_speakers, 2);
    }

    #[test]
    fn pipeline_run_dense_embed_window_splits_long_segments() {
        let mut cfg = custom_cfg();
        cfg.embed_window_secs = Some(0.5);
        let p = pipeline_custom(
            cfg,
            vec![raw_segment(0.0, 2.0, 0, false)],
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        );
        let result = p
            .run(&vec![0.0_f32; 16000 * 3], SampleRate::new(16000).unwrap())
            .unwrap();
        assert_eq!(result.num_speakers, 1);
        assert!(!result.turns.is_empty());
    }

    fn overlap_inputs_pipeline(embedder: Box<dyn Embedder>) -> Pipeline {
        pipeline_custom(
            custom_cfg(),
            Vec::new(),
            embedder,
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        )
    }

    #[test]
    fn build_overlap_inputs_both_mapped_uses_direct_assignment() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let overlaps = vec![(
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            0u8,
            1u8,
        )];
        let map: std::collections::HashMap<u8, SpeakerId> =
            [(0u8, SpeakerId(3)), (1u8, SpeakerId(4))]
                .into_iter()
                .collect();
        let out = p
            .build_overlap_inputs(&overlaps, &[], &map, &vec![0.0_f32; 16000])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].primary_speaker, SpeakerId(3));
        assert_eq!(out[0].secondary_speaker, Some(SpeakerId(4)));
        assert!(
            out[0].embedding.is_empty(),
            "direct assignment skips the mixed embedding"
        );
    }

    #[test]
    fn build_overlap_inputs_one_mapped_anchors_on_mapped_local() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let overlaps = vec![(
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            0u8,
            2u8,
        )];
        let map: std::collections::HashMap<u8, SpeakerId> =
            [(2u8, SpeakerId(5))].into_iter().collect();
        let out = p
            .build_overlap_inputs(&overlaps, &[], &map, &vec![0.0_f32; 16000])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].primary_speaker, SpeakerId(5));
        assert_eq!(out[0].secondary_speaker, None);
        assert_eq!(
            out[0].embedding.len(),
            192,
            "unmapped side recovered from a mixed-region embedding"
        );
    }

    #[test]
    fn build_overlap_inputs_unmapped_anchors_on_containing_turn() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let overlaps = vec![(
            TimeRange {
                start: 1.0,
                end: 2.0,
            },
            0u8,
            1u8,
        )];
        let turns = vec![turn(7, 0.0, 5.0)];
        let out = p
            .build_overlap_inputs(
                &overlaps,
                &turns,
                &std::collections::HashMap::new(),
                &vec![0.0_f32; 16000 * 5],
            )
            .unwrap();
        assert_eq!(out[0].primary_speaker, SpeakerId(7));
    }

    #[test]
    fn build_overlap_inputs_unmapped_falls_back_to_nearest_turn_midpoint() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let overlaps = vec![(
            TimeRange {
                start: 1.0,
                end: 2.0,
            },
            0u8,
            1u8,
        )];
        // Neither turn contains the overlap (midpoint 1.5): nearest midpoint
        // is the 0–0.8s turn (mid 0.4), not the 10–11s turn (mid 10.5).
        let turns = vec![turn(3, 0.0, 0.8), turn(4, 10.0, 11.0)];
        let out = p
            .build_overlap_inputs(
                &overlaps,
                &turns,
                &std::collections::HashMap::new(),
                &vec![0.0_f32; 16000 * 12],
            )
            .unwrap();
        assert_eq!(out[0].primary_speaker, SpeakerId(3));
    }

    #[test]
    fn build_overlap_inputs_unmapped_without_turns_defaults_to_zero() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let overlaps = vec![(
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            0u8,
            1u8,
        )];
        let out = p
            .build_overlap_inputs(
                &overlaps,
                &[],
                &std::collections::HashMap::new(),
                &vec![0.0_f32; 16000],
            )
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].primary_speaker, SpeakerId(0));
    }

    #[test]
    fn build_overlap_inputs_skips_out_of_audio_and_too_short_ranges() {
        let p = overlap_inputs_pipeline(Box::new(MockEmbedder::default()));
        let map: std::collections::HashMap<u8, SpeakerId> =
            [(0u8, SpeakerId(0))].into_iter().collect();
        let overlaps = vec![
            (
                TimeRange {
                    start: 5.0,
                    end: 6.0,
                },
                0u8,
                1u8,
            ), // beyond the 2s of audio
            (
                TimeRange {
                    start: 0.0,
                    end: 0.1,
                },
                0u8,
                1u8,
            ), // below MIN_EMBED_SECS
        ];
        let out = p
            .build_overlap_inputs(&overlaps, &[], &map, &vec![0.0_f32; 16000 * 2])
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn build_overlap_inputs_skips_non_finite_overlap_embedding() {
        let p = overlap_inputs_pipeline(Box::new(NanEmbedder));
        let overlaps = vec![(
            TimeRange {
                start: 0.0,
                end: 1.0,
            },
            0u8,
            1u8,
        )];
        let out = p
            .build_overlap_inputs(
                &overlaps,
                &[],
                &std::collections::HashMap::new(),
                &vec![0.0_f32; 16000],
            )
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn map_local_to_global_disable_toggle_returns_empty_map() {
        let mut cfg = custom_cfg();
        cfg.disable_seg_overlap = true;
        let p = pipeline_custom(
            cfg,
            Vec::new(),
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        );
        let sources = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(2.0, 3.0, 1, false),
        ];
        let map = p.map_local_to_global(&sources, &[0, 1], &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn map_local_to_global_majority_toggle_maps_by_vote() {
        let mut cfg = custom_cfg();
        cfg.majority_local_map = true;
        let p = pipeline_custom(
            cfg,
            Vec::new(),
            Box::new(MockEmbedder::default()),
            Box::new(MockClusterer::default()),
            Box::new(OverlapResegmenter::default()),
        );
        let sources = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(1.0, 2.0, 0, false),
            raw_segment(2.0, 3.0, 1, false),
        ];
        // Local 0 co-occurs with cluster 1 for 2s and never with cluster 0;
        // local 1 co-occurs with cluster 0.
        let map = p.map_local_to_global(&sources, &[1, 1, 0], &[]);
        assert_eq!(map.get(&0), Some(&SpeakerId(1)));
        assert_eq!(map.get(&1), Some(&SpeakerId(0)));
    }

    /// Records the L2 norms of the embeddings the clusterer receives, and
    /// opts into raw (non-L2-normalized) embeddings like a PLDA backend.
    struct NormRecordingClusterer {
        norms: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }

    impl Clusterer for NormRecordingClusterer {
        fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
            let mut g = self.norms.lock().unwrap();
            for e in embeddings {
                g.push(e.iter().map(|v| v * v).sum::<f32>().sqrt());
            }
            Ok(vec![0; embeddings.len()])
        }
        fn max_clusters(&self) -> usize {
            16
        }
        fn wants_raw_embeddings(&self) -> bool {
            true
        }
    }

    #[test]
    fn pipeline_raw_embeddings_clusterer_receives_unnormalized_vectors() {
        let norms = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let clusterer = NormRecordingClusterer {
            norms: norms.clone(),
        };
        // Non-unit embedding: magnitude 2 along axis 0.
        let mut v = vec![0.0_f32; 192];
        v[0] = 2.0;
        let p = pipeline_custom(
            custom_cfg(),
            vec![raw_segment(0.0, 1.0, 0, false)],
            Box::new(MockEmbedder { embedding: v }),
            Box::new(clusterer),
            Box::new(OverlapResegmenter::default()),
        );
        let _ = p
            .run(&vec![0.0_f32; 16000 * 2], SampleRate::new(16000).unwrap())
            .unwrap();
        let norms = norms.lock().unwrap();
        assert_eq!(norms.len(), 1);
        assert!(
            (norms[0] - 2.0).abs() < 1e-4,
            "raw embedding scale must be preserved, got norm {}",
            norms[0]
        );
    }

    #[test]
    fn merge_with_confidence_handles_unsorted_sources() {
        // Sources out of time order force the full-scan confidence path.
        let p = pipeline_with_segments(Vec::new());
        let sources = vec![
            raw_segment(2.0, 3.0, 0, false),
            raw_segment(0.0, 1.0, 0, false),
        ];
        let labels = vec![0usize, 1];
        let embeddings = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let turns = vec![turn(0, 2.0, 3.0), turn(1, 0.0, 1.0)];
        let (segments, merged_turns) =
            p.merge_with_confidence(&turns, &sources, &labels, &embeddings);
        assert_eq!(segments.len(), 2);
        assert_eq!(merged_turns.len(), 2);
        assert!(segments.iter().all(|s| s.confidence.is_some()));
    }

    #[test]
    fn pipeline_error_display_covers_all_variants() {
        let cases: Vec<(PipelineError, &str)> = vec![
            (
                PipelineError::UnsupportedSampleRate { actual: 8000 },
                "8000",
            ),
            (
                PipelineError::AudioTooLong {
                    actual_samples: MAX_AUDIO_SAMPLES + 1,
                    max_samples: MAX_AUDIO_SAMPLES,
                },
                "too long",
            ),
            (
                PipelineError::Segmentation(SegmentationError::AudioTooShort {
                    actual_secs: 0.0,
                    min_secs: 0.1,
                }),
                "segmentation failed",
            ),
            (
                PipelineError::Embedding(EmbedderError::InferenceFailed {
                    detail: "x".to_owned(),
                }),
                "embedding failed",
            ),
            (
                PipelineError::Clustering(ClustererError::TooFewEmbeddings { actual: 1, min: 2 }),
                "clustering failed",
            ),
            (
                PipelineError::Resegment(ResegmentError::MissingPrimaryCentroid {
                    index: 0,
                    primary: SpeakerId(0),
                }),
                "resegmentation failed",
            ),
            (
                PipelineError::Config(ConfigError::RegistryInCustomProfile),
                "config error",
            ),
            (
                PipelineError::Registry(RegistryError::ModelNotFound {
                    model_id: "m".to_owned(),
                }),
                "model registry error",
            ),
        ];
        for (err, needle) in cases {
            assert!(
                err.to_string().contains(needle),
                "`{err}` missing `{needle}`"
            );
        }
    }

    // Pipeline output turns must be monotonically ordered by start time
    // regardless of input segment order.
    proptest! {
        #[test]
        fn pipeline_turns_are_monotonically_ordered(
            segments in prop::collection::vec(
                (0.0f64..=10.0, 0.0f64..=10.0, 0u8..=2u8, prop::bool::ANY),
                0..=20usize,
            ),
        ) {
            let segs: Vec<_> = segments
                .into_iter()
                .map(|(s, e, spk, overlap)| {
                    let (start, end) = if s < e { (s, e) } else { (e, s) };
                    raw_segment(start, end, spk, overlap)
                })
                .collect();
            let p = pipeline_with_segments(segs);
            let result = p
                .run(&vec![0.0_f32; 16000 * 10], SampleRate::new(16000).unwrap())
                .unwrap();

            for i in 1..result.turns.len() {
                assert!(
                    result.turns[i - 1].time.start <= result.turns[i].time.start,
                    "turns must be monotonically ordered by start time"
                );
            }
        }
    }
}
