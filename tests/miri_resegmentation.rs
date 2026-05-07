//! Miri-friendly subset of M4 resegmenter tests. Covers no-overlap pass-through,
//! single-overlap cosine matching, and centroid math. ONNX-free, deterministic.

#![cfg(feature = "resegmentation")]

use polyvoice::resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentInputs, Resegmenter, SpeakerCentroid,
    compute_centroids,
};
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

fn unit(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    v
}

#[test]
fn miri_resegment_no_overlap() {
    let primary = vec![SpeakerTurn {
        speaker: SpeakerId(0),
        time: TimeRange {
            start: 0.0,
            end: 1.0,
        },
        text: None,
    }];
    let centroids = vec![SpeakerCentroid {
        speaker: SpeakerId(0),
        embedding: unit(4, 0),
    }];
    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &[],
        })
        .unwrap();
    assert_eq!(out, primary);
}

#[test]
fn miri_resegment_single_overlap() {
    let primary = vec![SpeakerTurn {
        speaker: SpeakerId(0),
        time: TimeRange {
            start: 0.0,
            end: 1.0,
        },
        text: None,
    }];
    let centroids = vec![
        SpeakerCentroid {
            speaker: SpeakerId(0),
            embedding: unit(4, 0),
        },
        SpeakerCentroid {
            speaker: SpeakerId(1),
            embedding: unit(4, 1),
        },
    ];
    let regions = vec![OverlapRegionInput {
        time: TimeRange {
            start: 0.0,
            end: 1.0,
        },
        primary_speaker: SpeakerId(0),
        embedding: unit(4, 1),
    }];
    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    assert_eq!(out.len(), 2);
    // Secondary turn must be SpeakerId(1) — its centroid is the unit vector
    // along axis 1, which exactly matches the overlap embedding.
    assert!(
        out.iter().any(|t| t.speaker == SpeakerId(1)),
        "expected secondary SpeakerId(1) appended, got speakers {:?}",
        out.iter().map(|t| t.speaker.0).collect::<Vec<_>>()
    );
}

#[test]
fn miri_compute_centroids() {
    let embeddings = vec![unit(4, 0), unit(4, 0), unit(4, 1), unit(4, 1)];
    let labels = vec![0, 0, 1, 1];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);
    for c in &centroids {
        let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3);
    }
}
