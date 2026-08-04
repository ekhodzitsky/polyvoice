//! Test-only `Mock{Segmenter,Embedder,Clusterer}` implementations used by the
//! `pipeline_v2` builder validation and pipeline unit tests. They return fixed
//! canned outputs so pipeline wiring can be exercised without ONNX models.

use crate::clusterer::{Clusterer, ClustererError};
use crate::embedder::{Embedder, EmbedderError};
use crate::resegmentation::{ResegmentError, ResegmentInputs, Resegmenter};
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
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
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
    fn resegment(&self, inputs: ResegmentInputs<'_>) -> Result<Vec<SpeakerTurn>, ResegmentError> {
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
        confidence: Confidence::new(0.9).expect("0.9 is within valid confidence range"),
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeakerId;

    #[test]
    fn mock_segmenter_returns_canned_segments() {
        let segs = vec![
            raw_segment(0.0, 1.0, 0, false),
            raw_segment(1.5, 2.5, 1, true),
        ];
        let s = MockSegmenter {
            segments: segs.clone(),
        };
        assert_eq!(s.segment(&[0.0_f32; 1600]).unwrap(), segs);
        assert_eq!(s.max_local_speakers(), 3);
        assert!(s.supports_overlap());
    }

    #[test]
    fn mock_segmenter_default_is_empty() {
        let s = MockSegmenter::default();
        assert!(s.segment(&[0.0_f32; 1600]).unwrap().is_empty());
    }

    #[test]
    fn mock_embedder_default_is_192d_unit_vector() {
        let e = MockEmbedder::default();
        assert_eq!(e.dim(), 192);
        let emb = e.embed(&[0.0_f32; 1600]).unwrap();
        assert_eq!(emb.len(), 192);
        assert_eq!(emb[0], 1.0);
        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mock_embedder_returns_configured_embedding() {
        let e = MockEmbedder {
            embedding: vec![0.5, -0.5],
        };
        assert_eq!(e.dim(), 2);
        assert_eq!(e.embed(&[]).unwrap(), vec![0.5, -0.5]);
    }

    #[test]
    fn mock_clusterer_default_assigns_single_cluster() {
        let c = MockClusterer::default();
        let labels = c
            .cluster(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]])
            .unwrap();
        assert_eq!(labels, vec![0, 0, 0]);
        assert_eq!(c.max_clusters(), 16);
    }

    #[test]
    fn mock_clusterer_returns_configured_labels() {
        let c = MockClusterer { labels: vec![2, 1] };
        let labels = c.cluster(&[vec![1.0], vec![2.0]]).unwrap();
        assert_eq!(labels, vec![2, 1]);
    }

    #[test]
    fn mock_clusterer_label_length_mismatch_errors() {
        let c = MockClusterer {
            labels: vec![0, 1, 2],
        };
        let err = c.cluster(&[vec![1.0], vec![2.0]]).unwrap_err();
        assert!(matches!(err, ClustererError::AlgorithmFailed { .. }));
    }

    #[test]
    fn passthrough_resegmenter_sorts_primary_turns_by_start() {
        let turns = vec![
            SpeakerTurn {
                speaker: SpeakerId(1),
                time: TimeRange {
                    start: 2.0,
                    end: 3.0,
                },
                text: None,
                stable: true,
            },
            SpeakerTurn {
                speaker: SpeakerId(0),
                time: TimeRange {
                    start: 0.0,
                    end: 1.0,
                },
                text: None,
                stable: true,
            },
        ];
        let out = PassThroughResegmenter
            .resegment(ResegmentInputs {
                primary_turns: &turns,
                speaker_centroids: &[],
                overlap_regions: &[],
            })
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].speaker, SpeakerId(0));
        assert_eq!(out[0].time.start, 0.0);
        assert_eq!(out[1].speaker, SpeakerId(1));
        assert_eq!(out[1].time.start, 2.0);
    }

    #[test]
    fn raw_segment_sets_all_fields() {
        let seg = raw_segment(1.0, 2.5, 3, true);
        assert_eq!(
            seg.time,
            TimeRange {
                start: 1.0,
                end: 2.5
            }
        );
        assert_eq!(seg.local_speaker_idx, 3);
        assert!(seg.is_overlap);
        assert!((seg.confidence.get() - 0.9).abs() < 1e-6);
    }
}
