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
