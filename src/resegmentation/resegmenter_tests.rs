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
fn constructor_getters_and_clamping() {
    let r = OverlapResegmenter::new(0.7, 0.25);
    assert_eq!(r.threshold(), 0.7);
    assert_eq!(r.min_overlap_secs(), 0.25);
    // Negative min_overlap_secs clamps to zero.
    let clamped = OverlapResegmenter::new(0.0, -1.0);
    assert_eq!(clamped.min_overlap_secs(), 0.0);
    let d = OverlapResegmenter::default();
    assert_eq!(d.threshold(), 0.0);
    assert_eq!(d.min_overlap_secs(), 0.1);
}

#[test]
fn segmentation_derived_short_region_is_skipped() {
    // A resolved secondary speaker does not rescue a region shorter than
    // min_overlap_secs — the overlap span is dropped entirely.
    let r = OverlapResegmenter::default(); // min_overlap_secs = 0.1
    let primary = vec![turn(0.0, 2.0, 0)];
    let centroids = vec![centroid(0, 3, 0), centroid(1, 3, 1)];
    let regions = vec![OverlapRegionInput {
        time: TimeRange {
            start: 2.0,
            end: 2.05,
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
    assert_eq!(out, primary, "short overlap region must be skipped");
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
