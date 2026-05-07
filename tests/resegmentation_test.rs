//! Integration test for the M4 OverlapResegmenter on synthetic data.
//! Pure-CPU; runs in normal `cargo test` (no model required).

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

fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
    SpeakerTurn {
        speaker: SpeakerId(spk),
        time: TimeRange { start, end },
        text: None,
    }
}

#[test]
fn end_to_end_synthetic_two_speakers_overlap() {
    // Two speakers, one overlap region. Embeddings are 8-d unit vectors.
    let dim = 8;
    let embeddings = vec![
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 1),
        unit(dim, 1),
        unit(dim, 1),
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);

    let primary = vec![turn(0.0, 5.0, 0), turn(5.0, 10.0, 1)];
    // Overlap at 4.5–5.5: primary spk=0, embedding aligned with axis 1 (i.e. spk=1).
    let regions = vec![OverlapRegionInput {
        time: TimeRange {
            start: 4.5,
            end: 5.5,
        },
        primary_speaker: SpeakerId(0),
        embedding: unit(dim, 1),
    }];

    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    assert_eq!(out.len(), 3, "primary 2 + secondary 1");
    let secondary = out
        .iter()
        .find(|t| (t.time.start - 4.5).abs() < 1e-6 && (t.time.end - 5.5).abs() < 1e-6)
        .expect("secondary turn at 4.5..5.5 missing");
    assert_eq!(secondary.speaker, SpeakerId(1));
}

#[test]
fn end_to_end_three_speakers_two_pairs() {
    let dim = 8;
    let embeddings = vec![
        unit(dim, 0),
        unit(dim, 0),
        unit(dim, 1),
        unit(dim, 1),
        unit(dim, 2),
        unit(dim, 2),
    ];
    let labels = vec![0, 0, 1, 1, 2, 2];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 3);

    let primary = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1), turn(4.0, 6.0, 2)];
    let regions = vec![
        // 1.0..2.0: primary 0, secondary best should be 1.
        OverlapRegionInput {
            time: TimeRange {
                start: 1.0,
                end: 2.0,
            },
            primary_speaker: SpeakerId(0),
            embedding: unit(dim, 1),
        },
        // 4.0..5.0: primary 2, secondary best should be 1.
        OverlapRegionInput {
            time: TimeRange {
                start: 4.0,
                end: 5.0,
            },
            primary_speaker: SpeakerId(2),
            embedding: unit(dim, 1),
        },
    ];

    let r = OverlapResegmenter::default();
    let out = r
        .resegment(ResegmentInputs {
            primary_turns: &primary,
            speaker_centroids: &centroids,
            overlap_regions: &regions,
        })
        .unwrap();
    // Two secondaries appended.
    assert_eq!(out.len(), 5);
    let n_spk1 = out.iter().filter(|t| t.speaker == SpeakerId(1)).count();
    assert!(n_spk1 >= 2, "expected ≥2 turns for speaker 1, got {n_spk1}");
    // Sorted by start.
    for w in out.windows(2) {
        assert!(w[0].time.start <= w[1].time.start);
    }
}

#[test]
fn rttm_round_trip_preserves_overlap_turns() {
    use polyvoice::rttm::write_rttm;

    let dim = 4;
    let centroids = vec![
        SpeakerCentroid {
            speaker: SpeakerId(0),
            embedding: unit(dim, 0),
        },
        SpeakerCentroid {
            speaker: SpeakerId(1),
            embedding: unit(dim, 1),
        },
    ];
    let primary = vec![turn(0.0, 1.0, 0)];
    let regions = vec![OverlapRegionInput {
        time: TimeRange {
            start: 0.2,
            end: 0.8,
        },
        primary_speaker: SpeakerId(0),
        embedding: unit(dim, 1),
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

    // RTTM writer must accept overlapping spans without error or data loss.
    let mut buf = Vec::new();
    write_rttm(&mut buf, "test", &out).expect("rttm write");
    let s = String::from_utf8(buf).unwrap();
    let n_lines = s.lines().filter(|l| l.starts_with("SPEAKER")).count();
    assert_eq!(n_lines, 2, "expected 2 SPEAKER lines, got {n_lines}: {s}");
    assert!(s.contains("SPEAKER_00"));
    assert!(s.contains("SPEAKER_01"));
}
