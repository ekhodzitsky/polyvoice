//! M6a — additive `polyvoice::pipeline_v2` module.

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
pub mod hybrid;

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

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("audio sample rate {actual} unsupported, expected 16000")]
    UnsupportedSampleRate { actual: u32 },
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
    #[error("model load error: {detail}")]
    ModelLoad { detail: String },
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

        let sample_rate = self.config.sample_rate.get() as f64;
        // Optional dense embedding: split each primary segment into overlapping
        // sub-windows so a speaker run yields several embeddings (legacy-style
        // dense windows) for more robust centroids. `None` keeps one embedding
        // per segment. Each unit carries its parent's local speaker index.
        let embed_units = expand_embed_units(&primary_segments, self.config.embed_window_secs);
        // PLDA backends (VBx) need the raw embedding scale for mean-centering;
        // cosine backends are scale-invariant and get the L2-normalized vectors.
        let raw_embeddings = self.clusterer.wants_raw_embeddings();
        let t = std::time::Instant::now();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(embed_units.len());
        let mut valid_segments: Vec<&crate::segmentation::RawSegment> =
            Vec::with_capacity(embed_units.len());
        for seg in &embed_units {
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
            // Fix 2: zero-fill any overlap regions inside this primary chunk before embedding.
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
            let mut emb = self.embedder.embed(&masked)?;
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
            valid_segments.push(seg);
        }

        timings.embedding_secs = t.elapsed().as_secs_f64();

        if embeddings.is_empty() {
            return Ok((DiarizationResult::new(Vec::new(), Vec::new(), 0), timings));
        }

        let t = std::time::Instant::now();
        // Per-embedding durations enable cVBx short-segment filtering inside
        // clusterers that opt in (VBx); others ignore the slice.
        let durations: Vec<f64> = valid_segments.iter().map(|s| s.time.duration()).collect();
        let labels = self
            .clusterer
            .cluster_with_durations(&embeddings, &durations)?;
        timings.clustering_secs = t.elapsed().as_secs_f64();

        // Fix 1: zip valid_segments (survivors only) with labels — not primary_segments.
        let mut primary_turns: Vec<SpeakerTurn> = valid_segments
            .iter()
            .zip(labels.iter())
            .map(|(seg, &lbl)| SpeakerTurn {
                speaker: SpeakerId(lbl as u32),
                time: seg.time,
                text: None,
                stable: true,
            })
            .collect();

        let centroids: Vec<SpeakerCentroid> = compute_centroids(&embeddings, &labels);

        // Bridge the powerset segmenter's file-consistent local speaker indices
        // to global clusters via Hungarian assignment on co-occurrence duration,
        // with cannot-link constraints from overlap pairs (two locals that share
        // an overlap must not collapse onto one global).
        let cannot_link: Vec<(u8, u8)> = overlap_ranges
            .iter()
            .map(|(_, lo, hi)| (*lo.min(hi), *lo.max(hi)))
            .collect();
        let local_to_global = Self::map_local_to_global(&valid_segments, &labels, &cannot_link);

        let t = std::time::Instant::now();
        let mut all_turns: Vec<SpeakerTurn> = if self.config.resegment_overlap
            && !overlap_ranges.is_empty()
            && centroids.len() >= 2
        {
            let overlap_inputs = self.build_overlap_inputs(
                &overlap_ranges,
                &primary_turns,
                &local_to_global,
                samples,
            )?;
            self.resegmenter.resegment(ResegmentInputs {
                primary_turns: &primary_turns,
                speaker_centroids: &centroids,
                overlap_regions: &overlap_inputs,
            })?
        } else {
            primary_turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            primary_turns
        };

        timings.resegmentation_secs = t.elapsed().as_secs_f64();

        // Fix 4: guarantee sorted order regardless of which Resegmenter impl ran.
        all_turns.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));

        let min_secs = self.config.min_speech_secs as f64;
        all_turns.retain(|t| t.time.duration() >= min_secs);

        let max_gap = self.config.max_gap_secs as f64;
        // Window-level confidence from embedding↔centroid cosine (logistic map).
        // After merge_segments the per-run confidence is the mean of present values.
        let speaker_ids: Vec<SpeakerId> = labels.iter().map(|&l| SpeakerId(l as u32)).collect();
        let window_conf =
            crate::types::segment_confidences_from_embeddings(&speaker_ids, &embeddings);
        // Map each merged turn's speaker/time back to a confidence by averaging
        // window confidences whose midpoint falls inside the turn (fallback: None).
        let merged_segments: Vec<Segment> = all_turns
            .iter()
            .map(|t| {
                let mut sum = 0.0f32;
                let mut n = 0u32;
                for (i, seg) in valid_segments.iter().enumerate() {
                    if speaker_ids.get(i).copied() != Some(t.speaker) {
                        continue;
                    }
                    let mid = (seg.time.start + seg.time.end) / 2.0;
                    if mid >= t.time.start
                        && mid < t.time.end
                        && let Some(&c) = window_conf.get(i)
                    {
                        sum += c;
                        n += 1;
                    }
                }
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

    /// Hungarian co-occurrence map from each file-consistent local speaker
    /// index to a global cluster. Only locals that appear as primary segments
    /// participate (inactive locals are never invented — the pyannote-style
    /// "inactive speakers in the similarity matrix" anti-pattern). Cannot-link
    /// pairs from overlap regions are forced onto distinct globals.
    fn map_local_to_global(
        valid_segments: &[&crate::segmentation::RawSegment],
        labels: &[usize],
        cannot_link: &[(u8, u8)],
    ) -> std::collections::HashMap<u8, SpeakerId> {
        // Ablation toggle: when set, return an empty map so every overlap region
        // takes the mixed-embedding fallback. Lets the segmentation-derived
        // overlap path be A/B-measured against the legacy path in one binary.
        if std::env::var_os("POLYVOICE_V2_DISABLE_SEG_OVERLAP").is_some() {
            return std::collections::HashMap::new();
        }
        let local_idx: Vec<u8> = valid_segments.iter().map(|s| s.local_speaker_idx).collect();
        let durations: Vec<f64> = valid_segments.iter().map(|s| s.time.duration()).collect();
        let cooc = crate::clusterer::build_cooccurrence(&local_idx, labels, &durations);
        // Ablation: majority vote instead of Hungarian.
        if std::env::var_os("POLYVOICE_V2_MAJORITY_LOCAL_MAP").is_some() {
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
