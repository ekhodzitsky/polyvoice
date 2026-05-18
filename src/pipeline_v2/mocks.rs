//! M6a — test-only `Mock{Segmenter,Embedder,Clusterer}` for the
//! `pipeline_v2` builder validation tests and the synthetic integration
//! test in `tests/pipeline_v2_synthetic_test.rs`.

use crate::clusterer::{Clusterer, ClustererError};
use crate::embedder::{Embedder, EmbedderError};
use crate::resegmentation::{Resegmenter, ResegmentError, ResegmentInputs};
use crate::segmentation::{RawSegment, SegmentationError, Segmenter};
use crate::types::{Confidence, SpeakerTurn, TimeRange};

/// Constant-output `Segmenter` for builder tests.
#[derive(Default)]
pub struct MockSegmenter {
    pub segments: Vec<RawSegment>,
}

impl Segmenter for MockSegmenter {
    fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        Ok(self.segments.clone())
    }

    fn max_local_speakers(&self) -> usize {
        3
    }

    fn supports_overlap(&self) -> bool {
        true
    }
}

/// Constant-output `Embedder` for builder tests.
pub struct MockEmbedder {
    pub embedding: Vec<f32>,
}

impl Default for MockEmbedder {
    fn default() -> Self {
        // 192-d unit vector along axis 0; matches CAM++ output dim used
        // throughout the spec.
        let mut v = vec![0.0_f32; 192];
        v[0] = 1.0;
        Self { embedding: v }
    }
}

impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.embedding.len()
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embedding.clone())
    }
}

/// Constant-label `Clusterer` for builder tests.
#[derive(Default)]
pub struct MockClusterer {
    pub labels: Vec<usize>,
}

impl Clusterer for MockClusterer {
    fn cluster(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<usize>, ClustererError> {
        if self.labels.is_empty() {
            // Default: single cluster.
            return Ok(vec![0; embeddings.len()]);
        }
        if self.labels.len() != embeddings.len() {
            return Err(ClustererError::AlgorithmFailed {
                detail: "MockClusterer labels length mismatch".to_owned(),
            });
        }
        Ok(self.labels.clone())
    }

    fn max_clusters(&self) -> usize {
        16
    }
}

/// Pass-through `Resegmenter` (returns input primary turns sorted, no
/// secondary speakers added).
#[derive(Default)]
pub struct PassThroughResegmenter;

impl Resegmenter for PassThroughResegmenter {
    fn resegment(
        &self,
        inputs: ResegmentInputs<'_>,
    ) -> Result<Vec<SpeakerTurn>, ResegmentError> {
        let mut out: Vec<SpeakerTurn> = inputs.primary_turns.to_vec();
        out.sort_by(|a, b| a.time.start.total_cmp(&b.time.start));
        Ok(out)
    }
}

/// Convenience constructor for a single `RawSegment` used in tests.
pub fn raw_segment(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
    RawSegment {
        time: TimeRange { start, end },
        local_speaker_idx: spk,
        is_overlap: overlap,
        confidence: Confidence::new(0.9).unwrap(),
    }
}
