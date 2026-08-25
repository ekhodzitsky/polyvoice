//! `polyvoice::pipeline_v2` — trait-wired production ONNX diarization pipeline.

#[cfg(not(all(
    any(
        feature = "infer",
        all(feature = "segmenter-native", feature = "embedder-native")
    ),
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
)))]
compile_error!(
    "pipeline_v2 requires download + segmentation + embedder + clusterer + resegmentation \
     and an engine (`onnx`, `backend-tract`, or `pipeline-native`)"
);

pub mod builder;
mod clusterer_factory;
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
#[path = "run_tests.rs"]
mod tests;
