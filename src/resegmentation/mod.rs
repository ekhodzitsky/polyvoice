//! v1.0 OverlapResegmenter — overlap-aware post-clustering pass.
//!
//! Added in v0.6 (M4).
//!
//! Pure Rust, wasm32-clean. Operates on already-computed speaker centroids and
//! overlap-region embeddings supplied by the caller. M6 (`Pipeline`) wires the
//! `EmbedderPool` and `apply_overlap_mask` into this.

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
/// In v1.0 (M4) the polyvoice crate introduces `Resegmenter` as the canonical
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
/// Typical usage (from `Pipeline` in M6):
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
mod trait_tests {
    use super::*;

    /// In-memory dummy used by trait conformance tests.
    struct ConstantResegmenter {
        out: Vec<SpeakerTurn>,
    }

    impl Resegmenter for ConstantResegmenter {
        fn resegment(
            &self,
            _inputs: ResegmentInputs<'_>,
        ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
            Ok(self.out.clone())
        }
    }

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        }
    }

    #[test]
    fn resegmenter_trait_object_is_dyn_compatible() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0)],
        };
        let _b: Box<dyn Resegmenter> = Box::new(r);
    }

    #[test]
    fn resegmenter_returns_owned_turns() {
        let r = ConstantResegmenter {
            out: vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)],
        };
        let inputs = ResegmentInputs {
            primary_turns: &[],
            speaker_centroids: &[],
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
    }

    #[test]
    fn error_centroid_dim_mismatch_displays() {
        let err = ResegmentError::CentroidDimMismatch {
            index: 1,
            expected: 192,
            actual: 256,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("256"));
        assert!(msg.contains("index 1"));
    }

    #[test]
    fn error_overlap_dim_mismatch_displays() {
        let err = ResegmentError::OverlapDimMismatch {
            index: 0,
            expected: 192,
            actual: 64,
        };
        let msg = format!("{err}");
        assert!(msg.contains("192"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn error_missing_primary_centroid_displays() {
        let err = ResegmentError::MissingPrimaryCentroid {
            index: 2,
            primary: SpeakerId(7),
        };
        let msg = format!("{err}");
        assert!(msg.contains('2'));
        assert!(msg.contains('7'));
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod centroid_tests {
    use super::*;

    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        v
    }

    #[test]
    fn compute_centroids_l2_normalized() {
        let embeddings = vec![unit(3, 0), unit(3, 0), unit(3, 1), unit(3, 1)];
        let labels = vec![0, 0, 1, 1];
        let centroids = compute_centroids(&embeddings, &labels);
        assert_eq!(centroids.len(), 2);
        for c in &centroids {
            let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (n - 1.0).abs() < 1e-3,
                "centroid not L2-normalized: norm={n}"
            );
        }
    }

    #[test]
    fn compute_centroids_drops_empty_clusters() {
        // Labels skip from 0 to 2; cluster 1 has no members.
        let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 1)];
        let labels = vec![0, 2, 2];
        let centroids = compute_centroids(&embeddings, &labels);
        assert_eq!(centroids.len(), 2);
        let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
        assert_eq!(speakers, vec![0, 2]);
    }

    #[test]
    fn compute_centroids_sorted_by_speaker_id() {
        let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 2)];
        let labels = vec![5, 1, 3];
        let centroids = compute_centroids(&embeddings, &labels);
        let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
        assert_eq!(speakers, vec![1, 3, 5]);
    }

    #[test]
    fn compute_centroids_empty_input_returns_empty() {
        let centroids = compute_centroids(&[], &[]);
        assert!(centroids.is_empty());
    }

    #[test]
    fn compute_centroids_label_mismatch_returns_empty() {
        // Mismatched lengths: caller bug, conservative empty return rather than panic.
        let centroids = compute_centroids(&[unit(3, 0)], &[0, 1]);
        assert!(centroids.is_empty());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
#[cfg(feature = "segmentation")]
mod overlap_extract_tests {
    use super::*;
    use crate::segmentation::RawSegment;
    use crate::types::Confidence;

    fn raw(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
        RawSegment {
            time: TimeRange { start, end },
            local_speaker_idx: spk,
            is_overlap: overlap,
            confidence: Confidence::new(0.9).expect("0.9 is valid confidence"),
        }
    }

    #[test]
    fn extract_returns_pairs_for_simultaneous_overlap_segments() {
        // Two RawSegments with the same time range and is_overlap = true:
        // aggregator's canonical overlap output.
        let segs = vec![raw(0.0, 1.0, 0, true), raw(0.0, 1.0, 1, true)];
        let pairs = extract_overlap_time_ranges(&segs);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].0.start - 0.0).abs() < 1e-6);
        assert!((pairs[0].0.end - 1.0).abs() < 1e-6);
        // local pair is (lo, hi) where lo < hi.
        assert_eq!(pairs[0].1, 0);
        assert_eq!(pairs[0].2, 1);
    }

    #[test]
    fn extract_ignores_non_overlap_segments() {
        let segs = vec![raw(0.0, 1.0, 0, false), raw(0.0, 1.0, 1, false)];
        let pairs = extract_overlap_time_ranges(&segs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn extract_ignores_overlap_flag_without_pair() {
        // is_overlap=true but only one local speaker present at this range.
        let segs = vec![raw(0.0, 1.0, 0, true)];
        let pairs = extract_overlap_time_ranges(&segs);
        assert!(pairs.is_empty());
    }

    #[test]
    fn extract_handles_multiple_overlap_regions() {
        let segs = vec![
            raw(0.0, 1.0, 0, true),
            raw(0.0, 1.0, 1, true),
            raw(2.0, 3.0, 1, true),
            raw(2.0, 3.0, 2, true),
        ];
        let pairs = extract_overlap_time_ranges(&segs);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].1, 0);
        assert_eq!(pairs[0].2, 1);
        assert_eq!(pairs[1].1, 1);
        assert_eq!(pairs[1].2, 2);
    }

    #[test]
    fn extract_three_way_overlap_emits_all_three_pairs() {
        // Three RawSegments at the same time range with distinct local indices.
        // The O(N²) loop should emit all (0,1), (0,2), (1,2) pairs.
        let segs = vec![
            raw(0.0, 1.0, 0, true),
            raw(0.0, 1.0, 1, true),
            raw(0.0, 1.0, 2, true),
        ];
        let pairs = extract_overlap_time_ranges(&segs);
        assert_eq!(pairs.len(), 3);
        let local_pairs: std::collections::HashSet<(u8, u8)> =
            pairs.iter().map(|p| (p.1, p.2)).collect();
        assert!(local_pairs.contains(&(0, 1)));
        assert!(local_pairs.contains(&(0, 2)));
        assert!(local_pairs.contains(&(1, 2)));
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod resegmenter_tests {
    use super::*;
    use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[axis] = 1.0;
        v
    }

    fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
            stable: true,
        }
    }

    fn centroid(spk: u32, dim: usize, axis: usize) -> SpeakerCentroid {
        SpeakerCentroid {
            speaker: SpeakerId(spk),
            embedding: unit(dim, axis),
        }
    }

    fn region(start: f64, end: f64, primary: u32, dim: usize, axis: usize) -> OverlapRegionInput {
        OverlapRegionInput {
            time: TimeRange { start, end },
            primary_speaker: SpeakerId(primary),
            secondary_speaker: None,
            embedding: unit(dim, axis),
        }
    }

    #[test]
    fn no_overlap_passes_primary_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0), turn(2.0, 3.0, 1)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn single_cluster_passes_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0)];
        let regions = vec![region(0.5, 0.9, 0, 3, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn picks_secondary_excluding_primary() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1), centroid(2, 3, 2)];
        // Overlap region embedding lies along axis 1 → nearest to centroid id=1.
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out.len(), 2);
        // Both turns cover (0.0, 1.0); one is primary (id=0), other is secondary (id=1).
        let speakers: Vec<u32> = out.iter().map(|t| t.speaker.0).collect();
        assert!(speakers.contains(&0));
        assert!(speakers.contains(&1));
        assert!(!speakers.contains(&2));
    }

    #[test]
    fn threshold_blocks_low_cosine() {
        // Threshold 0.99 — only near-perfect matches allowed.
        let r = OverlapResegmenter::new(0.99, 0.0);
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        // Overlap embedding along axis 0 (matches primary); cosine to centroid 1 = 0.
        let regions = vec![region(0.0, 1.0, 0, 3, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary, "no secondary should be appended");
    }

    #[test]
    fn min_duration_blocks_short_region() {
        // Region duration 0.05s < default 0.1s → skipped.
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![region(0.10, 0.15, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn output_is_sorted_by_start() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(2.0, 3.0, 0), turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![region(2.0, 3.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let out = r.resegment(inputs).unwrap();
        for w in out.windows(2) {
            assert!(w[0].time.start <= w[1].time.start);
        }
    }

    #[test]
    fn missing_primary_centroid_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(1, 3, 1), centroid(2, 3, 2)];
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("missing primary must error");
        assert!(matches!(
            err,
            ResegmentError::MissingPrimaryCentroid {
                primary: SpeakerId(0),
                ..
            }
        ));
    }

    #[test]
    fn centroid_dim_mismatch_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![
            centroid(0, 3, 0),
            SpeakerCentroid {
                speaker: SpeakerId(1),
                embedding: vec![1.0, 0.0], // dim 2, not 3
            },
        ];
        let regions = vec![region(0.0, 1.0, 0, 3, 1)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("dim mismatch must error");
        assert!(matches!(err, ResegmentError::CentroidDimMismatch { .. }));
    }

    #[test]
    fn overlap_dim_mismatch_errors() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![OverlapRegionInput {
            time: TimeRange {
                start: 0.0,
                end: 1.0,
            },
            primary_speaker: SpeakerId(0),
            secondary_speaker: None,
            embedding: vec![1.0, 0.0], // dim 2, not 3
        }];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        };
        let err = r.resegment(inputs).expect_err("dim mismatch must error");
        assert!(matches!(err, ResegmentError::OverlapDimMismatch { .. }));
    }

    #[test]
    fn empty_centroids_passes_through() {
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 1.0, 0)];
        let inputs = ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &[],
            overlap_regions: &[],
        };
        let out = r.resegment(inputs).unwrap();
        assert_eq!(out, primary);
    }

    #[test]
    fn segmentation_derived_secondary_emits_both_speakers() {
        // When the segmenter resolves the second speaker, both speakers must be
        // emitted over the overlap span, and the embedding must be ignored (here
        // deliberately empty to prove it is never inspected on this path).
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 2.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![OverlapRegionInput {
            time: TimeRange {
                start: 2.0,
                end: 3.0,
            },
            primary_speaker: SpeakerId(0),
            secondary_speaker: Some(SpeakerId(1)),
            embedding: Vec::new(),
        }];
        let out = r
            .resegment(ResegmentInputs {
                primary_turns: &primary,
                speaker_centroids: &centroids,
                overlap_regions: &regions,
            })
            .unwrap();
        let over_span = |spk: u32| {
            out.iter().any(|t| {
                t.speaker == SpeakerId(spk)
                    && (t.time.start - 2.0).abs() < 1e-9
                    && (t.time.end - 3.0).abs() < 1e-9
            })
        };
        assert!(
            over_span(0),
            "primary must be emitted over the overlap span"
        );
        assert!(
            over_span(1),
            "segmentation-identified secondary must be emitted"
        );
    }

    #[test]
    fn segmentation_derived_secondary_equal_to_primary_emits_one() {
        // Degenerate overlap where both local indices map to the same global
        // cluster: emit a single speaker, never a spurious second.
        let r = OverlapResegmenter::default();
        let primary = vec![turn(0.0, 2.0, 0)];
        let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
        let regions = vec![OverlapRegionInput {
            time: TimeRange {
                start: 2.0,
                end: 3.0,
            },
            primary_speaker: SpeakerId(0),
            secondary_speaker: Some(SpeakerId(0)),
            embedding: Vec::new(),
        }];
        let out = r
            .resegment(ResegmentInputs {
                primary_turns: &primary,
                speaker_centroids: &centroids,
                overlap_regions: &regions,
            })
            .unwrap();
        let over_span: Vec<_> = out
            .iter()
            .filter(|t| (t.time.start - 2.0).abs() < 1e-9 && (t.time.end - 3.0).abs() < 1e-9)
            .collect();
        assert_eq!(
            over_span.len(),
            1,
            "a single-speaker overlap must emit exactly one turn for the span"
        );
        assert_eq!(over_span[0].speaker, SpeakerId(0));
    }
}
