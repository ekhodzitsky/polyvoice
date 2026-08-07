//! v1.0 OverlapResegmenter — overlap-aware post-clustering pass.
//!
//! Pure Rust, wasm32-clean. Operates on already-computed speaker centroids and
//! overlap-region embeddings supplied by the caller: `pipeline_v2::Pipeline`
//! computes them with its own `Embedder` (masking overlap spans via
//! `crate::embedder::apply_overlap_mask`) and passes them in as
//! [`OverlapRegionInput`]s.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

/// Time-range equality tolerance (seconds) used by `extract_overlap_time_ranges`
/// when matching pairs of `RawSegment`s that should occupy the same span.
#[cfg(feature = "segmentation")]
const TIME_RANGE_EPS_SECS: f64 = 1e-6;

/// Speaker resegmenter — given primary single-speaker turns, cluster centroids,
/// and per-overlap-region embeddings, returns a (possibly overlap-aware) flat
/// list of `SpeakerTurn`s where overlap regions may produce two turns over the
/// same time range with different speakers.
///
/// In v1.0 the polyvoice crate introduces `Resegmenter` as the canonical
/// trait. The legacy `crate::overlap::detect_overlaps` remains as an
/// interval-only helper unrelated to this pass.
pub trait Resegmenter: Send + Sync {
    /// Run the pass.
    ///
    /// **Requires:** all centroid vectors and all overlap embeddings have the
    /// same dimension and are approximately L2-normalized.
    /// **Guarantees on Ok:** every turn in `inputs.primary_turns` is preserved
    /// verbatim; secondary turns (if any) carry an existing `SpeakerId` from
    /// `inputs.speaker_centroids` and never repeat the primary speaker for the
    /// same region; output is sorted by `time.start`.
    ///
    /// **Validation order:** structural checks (centroid dimensions, overlap
    /// embedding dimensions, `primary_speaker` presence) run before duration
    /// filtering. A short overlap region with an invalid primary speaker
    /// returns `MissingPrimaryCentroid`, not silent success.
    ///
    /// **Fast path:** when `inputs.speaker_centroids.len() < 2` or
    /// `inputs.overlap_regions` is empty, `inputs.primary_turns` is returned
    /// sorted without further validation; no error is produced even if a
    /// would-be overlap region had an invalid primary or dim.
    fn resegment(&self, inputs: ResegmentInputs<'_>) -> Result<Vec<SpeakerTurn>, ResegmentError>;
}

/// All inputs needed by `Resegmenter::resegment`.
#[derive(Debug, Clone)]
pub struct ResegmentInputs<'a> {
    pub primary_turns: &'a [SpeakerTurn],
    pub speaker_centroids: &'a [SpeakerCentroid],
    pub overlap_regions: &'a [OverlapRegionInput],
}

/// L2-normalized centroid for one speaker cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerCentroid {
    pub speaker: SpeakerId,
    pub embedding: Vec<f32>,
}

/// One overlap region with its caller-supplied embedding.
///
/// `embedding` is expected to be L2-normalized; this struct does not enforce
/// it (`OverlapResegmenter` returns `OverlapDimMismatch` only on dimension
/// mismatches, not on norm drift).
///
/// When `secondary_speaker` is `Some`, the powerset/EEND segmenter already
/// identified both speakers active in this region (its file-consistent local
/// indices were mapped to global clusters upstream). The resegmenter then trusts
/// that assignment, emits both speakers, and ignores `embedding` — avoiding the
/// degraded nearest-centroid guess on a mixed-voice embedding. When `None`, it
/// falls back to that centroid match on `embedding`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapRegionInput {
    pub time: TimeRange,
    pub primary_speaker: SpeakerId,
    pub secondary_speaker: Option<SpeakerId>,
    pub embedding: Vec<f32>,
}

/// Errors from `Resegmenter` implementations.
#[derive(Debug, thiserror::Error)]
pub enum ResegmentError {
    #[error("centroid dim mismatch at index {index}: expected {expected}, got {actual}")]
    CentroidDimMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("overlap embedding dim mismatch at index {index}: expected {expected}, got {actual}")]
    OverlapDimMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },

    #[error("primary speaker {primary} for overlap region {index} not present in centroids")]
    MissingPrimaryCentroid { index: usize, primary: SpeakerId },
}

/// { true }
/// `pub fn compute_centroids(embeddings: &[Vec<f32>], labels: &[usize]) -> Vec<SpeakerCentroid>`
/// { ret.iter().all(|c| c.embedding.len() == embeddings.first().map_or(0, |e| e.len())) }
/// Compute per-cluster L2-normalized centroids from clustered embeddings.
///
/// `labels[i]` is the cluster label of `embeddings[i]`. The cluster id stored
/// in the resulting `SpeakerCentroid` is the raw `labels[i]` cast to `SpeakerId`.
/// Empty clusters yield no entry. Output is sorted by `SpeakerId.0` ascending.
///
/// Returns an empty `Vec` if `embeddings.len() != labels.len()` or both are
/// empty — never panics.
///
/// **Pure Rust, wasm32-clean.**
pub fn compute_centroids(embeddings: &[Vec<f32>], labels: &[usize]) -> Vec<SpeakerCentroid> {
    if embeddings.len() != labels.len() || embeddings.is_empty() {
        return Vec::new();
    }
    // Bucket by label.
    let mut buckets: std::collections::BTreeMap<usize, Vec<&Vec<f32>>> =
        std::collections::BTreeMap::new();
    for (emb, &lbl) in embeddings.iter().zip(labels.iter()) {
        buckets.entry(lbl).or_default().push(emb);
    }
    let mut out = Vec::with_capacity(buckets.len());
    for (lbl, members) in buckets {
        let owned: Vec<Vec<f32>> = members.iter().map(|e| (*e).clone()).collect();
        if let Some(mut mean) = crate::utils::mean_vector(&owned) {
            crate::utils::l2_normalize(&mut mean);
            // Truncating cast; cluster labels are well within u32 range in practice.
            let id = SpeakerId(lbl as u32);
            out.push(SpeakerCentroid {
                speaker: id,
                embedding: mean,
            });
        }
    }
    // BTreeMap iterates in label order, but cast to SpeakerId may reorder if
    // u32 truncation happened. Sort explicitly.
    out.sort_by_key(|c| c.speaker.0);
    out
}

/// { true }
/// `pub fn extract_overlap_time_ranges( segments: &[crate::segmentation::RawSegment], ) -> Vec<(TimeRange, u8, u8)>`
/// { ret.iter().all(|(_, lo, hi)| lo < hi) }
/// Find pairs of `RawSegment`s that share a time range, are flagged
/// `is_overlap = true`, and carry two distinct `local_speaker_idx`.
/// Returns `(time_range, lo_local_idx, hi_local_idx)` per detected pair.
///
/// "Same time range" uses an `f64` tolerance of `1e-6`.
///
/// `lo_local_idx < hi_local_idx`. Caller is responsible for the local→global
/// `SpeakerId` mapping (typically from the same clustering pipeline).
///
/// **Pure Rust, wasm32-clean.** Gated `segmentation` because `RawSegment`
/// lives in the segmentation module.
#[cfg(feature = "segmentation")]
pub fn extract_overlap_time_ranges(
    segments: &[crate::segmentation::RawSegment],
) -> Vec<(TimeRange, u8, u8)> {
    let mut pairs: Vec<(TimeRange, u8, u8)> = Vec::new();
    for (i, a) in segments.iter().enumerate() {
        if !a.is_overlap {
            continue;
        }
        for b in segments.iter().skip(i + 1) {
            if !b.is_overlap {
                continue;
            }
            if a.local_speaker_idx == b.local_speaker_idx {
                continue;
            }
            if (a.time.start - b.time.start).abs() > TIME_RANGE_EPS_SECS
                || (a.time.end - b.time.end).abs() > TIME_RANGE_EPS_SECS
            {
                continue;
            }
            let (lo, hi) = if a.local_speaker_idx < b.local_speaker_idx {
                (a.local_speaker_idx, b.local_speaker_idx)
            } else {
                (b.local_speaker_idx, a.local_speaker_idx)
            };
            pairs.push((a.time, lo, hi));
        }
    }
    pairs
}

/// Default-constructible overlap-aware resegmenter.
///
/// For each overlap region it prefers the segmenter's own speaker assignment
/// (`OverlapRegionInput::secondary_speaker`), emitting both active speakers
/// directly. When that is absent it falls back to picking the nearest non-primary
/// cluster centroid (by cosine similarity) above a configurable threshold and
/// minimum duration.
///
/// Typical usage (from `Pipeline`):
///
/// ```rust,ignore
/// let r = OverlapResegmenter::default();
/// let out = r.resegment(ResegmentInputs {
///     primary_turns: &turns,
///     speaker_centroids: &centroids,
///     overlap_regions: &regions,
/// })?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OverlapResegmenter {
    threshold: f32,
    min_overlap_secs: f32,
}

impl OverlapResegmenter {
    /// { true }
    /// pub fn new(threshold: f32, min_overlap_secs: f32) -> Self
    /// { ret.min_overlap_secs >= 0.0 }
    /// `threshold` — minimum cosine similarity required to attach a secondary
    /// speaker to an overlap region. Default `0.0` (always attach the nearest
    /// non-primary cluster).
    /// `min_overlap_secs` — overlap regions shorter than this are skipped.
    /// Default `0.1`.
    pub fn new(threshold: f32, min_overlap_secs: f32) -> Self {
        Self {
            threshold,
            min_overlap_secs: min_overlap_secs.max(0.0),
        }
    }

    /// { true }
    /// pub fn threshold(&self) -> f32
    /// { ret == self.threshold }
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// { true }
    /// pub fn min_overlap_secs(&self) -> f32
    /// { ret == self.min_overlap_secs }
    pub fn min_overlap_secs(&self) -> f32 {
        self.min_overlap_secs
    }
}

impl Default for OverlapResegmenter {
    fn default() -> Self {
        Self::new(0.0, 0.1)
    }
}

impl Resegmenter for OverlapResegmenter {
    fn resegment(&self, inputs: ResegmentInputs<'_>) -> Result<Vec<SpeakerTurn>, ResegmentError> {
        let mut out: Vec<SpeakerTurn> = inputs.primary_turns.to_vec();

        // Fast paths.
        if inputs.speaker_centroids.len() < 2 || inputs.overlap_regions.is_empty() {
            out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
            return Ok(out);
        }

        // Validate centroid dimensionality first (single-pass).
        let expected_dim = inputs.speaker_centroids[0].embedding.len();
        for (i, c) in inputs.speaker_centroids.iter().enumerate() {
            if c.embedding.len() != expected_dim {
                return Err(ResegmentError::CentroidDimMismatch {
                    index: i,
                    expected: expected_dim,
                    actual: c.embedding.len(),
                });
            }
        }

        for (i, region) in inputs.overlap_regions.iter().enumerate() {
            match region.secondary_speaker {
                // Segmentation-derived path: the powerset/EEND segmenter already
                // identified both speakers in this region (its file-consistent
                // local indices were mapped to global clusters upstream). Trust
                // that assignment and emit both speakers over the span, ignoring
                // the mixed-voice embedding. The aggregator splits the primary's
                // run at the overlap boundary, so `primary_turns` does NOT cover
                // this span — emit the primary here too, or it is missed.
                Some(secondary) => {
                    if region.time.duration() < f64::from(self.min_overlap_secs) {
                        continue;
                    }
                    out.push(SpeakerTurn {
                        speaker: region.primary_speaker,
                        time: region.time,
                        text: None,
                        stable: true,
                    });
                    if secondary != region.primary_speaker {
                        out.push(SpeakerTurn {
                            speaker: secondary,
                            time: region.time,
                            text: None,
                            stable: true,
                        });
                    }
                }
                // Fallback path: the second speaker was not resolved from the
                // segmentation (a local index never appeared as a solo segment),
                // so recover it from the nearest non-primary centroid on the
                // region's (mixed) embedding. Structural checks run before the
                // duration filter — an invalid region errors even when short.
                None => {
                    if region.embedding.len() != expected_dim {
                        return Err(ResegmentError::OverlapDimMismatch {
                            index: i,
                            expected: expected_dim,
                            actual: region.embedding.len(),
                        });
                    }
                    if !inputs
                        .speaker_centroids
                        .iter()
                        .any(|c| c.speaker == region.primary_speaker)
                    {
                        return Err(ResegmentError::MissingPrimaryCentroid {
                            index: i,
                            primary: region.primary_speaker,
                        });
                    }
                    if region.time.duration() < f64::from(self.min_overlap_secs) {
                        continue;
                    }
                    let mut best: Option<(SpeakerId, f32)> = None;
                    for c in inputs.speaker_centroids.iter() {
                        if c.speaker == region.primary_speaker {
                            continue;
                        }
                        let s = crate::utils::cosine_similarity(&region.embedding, &c.embedding);
                        let take = match best {
                            None => true,
                            Some((_, b)) => s > b,
                        };
                        if take {
                            best = Some((c.speaker, s));
                        }
                    }
                    if let Some((id, score)) = best
                        && score > self.threshold
                    {
                        out.push(SpeakerTurn {
                            speaker: id,
                            time: region.time,
                            text: None,
                            stable: true,
                        });
                    }
                }
            }
        }

        out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
        Ok(out)
    }
}
#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "trait_tests.rs"]
mod trait_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "centroid_tests.rs"]
mod centroid_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "segmentation")]
#[path = "overlap_extract_tests.rs"]
mod overlap_extract_tests;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[path = "resegmenter_tests.rs"]
mod resegmenter_tests;
