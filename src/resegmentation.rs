//! v1.0 OverlapResegmenter — overlap-aware post-clustering pass.
//!
//! Added in v0.6 (M4). See `docs/superpowers/specs/2026-05-07-m4-overlap-resegmenter-design.md`
//! and `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §3.1.
//!
//! Pure Rust, wasm32-clean. Operates on already-computed speaker centroids and
//! overlap-region embeddings supplied by the caller. M6 (`Pipeline`) wires the
//! `EmbedderPool` and `apply_overlap_mask` into this.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange};

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
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapRegionInput {
    pub time: TimeRange,
    pub primary_speaker: SpeakerId,
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
pub fn compute_centroids(
    embeddings: &[Vec<f32>],
    labels: &[usize],
) -> Vec<SpeakerCentroid> {
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
            // SpeakerId is u32; clamp to its range conservatively.
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
            if (a.time.start - b.time.start).abs() > 1e-6
                || (a.time.end - b.time.end).abs() > 1e-6
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
        let embeddings = vec![
            unit(3, 0),
            unit(3, 0),
            unit(3, 1),
            unit(3, 1),
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = compute_centroids(&embeddings, &labels);
        assert_eq!(centroids.len(), 2);
        for c in &centroids {
            let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-3, "centroid not L2-normalized: norm={n}");
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

#[cfg(all(test, feature = "segmentation"))]
mod overlap_extract_tests {
    use super::*;
    use crate::segmentation::RawSegment;
    use crate::types::Confidence;

    fn raw(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
        RawSegment {
            time: TimeRange { start, end },
            local_speaker_idx: spk,
            is_overlap: overlap,
            confidence: Confidence::new(0.9).unwrap(),
        }
    }

    #[test]
    fn extract_returns_pairs_for_simultaneous_overlap_segments() {
        // Two RawSegments with the same time range and is_overlap = true:
        // aggregator's canonical overlap output.
        let segs = vec![
            raw(0.0, 1.0, 0, true),
            raw(0.0, 1.0, 1, true),
        ];
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
        let segs = vec![
            raw(0.0, 1.0, 0, false),
            raw(0.0, 1.0, 1, false),
        ];
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
}
