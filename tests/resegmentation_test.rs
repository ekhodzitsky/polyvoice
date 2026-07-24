#![allow(clippy::unwrap_used)]
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
        stable: true,
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
        secondary_speaker: None,
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
            secondary_speaker: None,
            embedding: unit(dim, 1),
        },
        // 4.0..5.0: primary 2, secondary best should be 1.
        OverlapRegionInput {
            time: TimeRange {
                start: 4.0,
                end: 5.0,
            },
            primary_speaker: SpeakerId(2),
            secondary_speaker: None,
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
        secondary_speaker: None,
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

mod proptests {
    use super::{turn, unit};
    use polyvoice::resegmentation::{
        OverlapRegionInput, OverlapResegmenter, ResegmentInputs, Resegmenter, SpeakerCentroid,
        compute_centroids,
    };
    use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};
    use proptest::prelude::*;

    /// Build an L2-normalized embedding from raw f32 components.
    /// Falls back to a unit-axis vector if the input norm is too small to
    /// normalize stably (mirrors `crate::utils::l2_normalize`).
    fn normalize(mut v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 1e-3 {
            for x in &mut v {
                *x /= n;
            }
        } else {
            for x in v.iter_mut() {
                *x = 0.0;
            }
            if !v.is_empty() {
                v[0] = 1.0;
            }
        }
        v
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            .. ProptestConfig::default()
        })]

        /// Invariant 1: every primary turn appears verbatim in the output
        /// (set inclusion on `(start, end, speaker.0)`).
        #[test]
        fn primary_turns_are_preserved_in_output(
            primary_count in 1usize..6,
            num_centroids in 2usize..6,
            embedding_seed_a in -1.0_f32..1.0_f32,
            embedding_seed_b in -1.0_f32..1.0_f32,
            overlap_count in 0usize..5,
        ) {
            let dim = 4;
            // Build deterministic centroids by varying axis offsets.
            let centroids: Vec<SpeakerCentroid> = (0..num_centroids)
                .map(|i| SpeakerCentroid {
                    speaker: SpeakerId(i as u32),
                    embedding: unit(dim, i % dim),
                })
                .collect();
            // Build primary turns over distinct, non-overlapping intervals so the
            // set comparison is unambiguous.
            let primary: Vec<SpeakerTurn> = (0..primary_count)
                .map(|i| turn(i as f64 * 2.0, i as f64 * 2.0 + 1.5, (i % num_centroids) as u32))
                .collect();
            let regions: Vec<OverlapRegionInput> = (0..overlap_count)
                .map(|i| OverlapRegionInput {
                    time: TimeRange { start: i as f64 * 0.7, end: i as f64 * 0.7 + 0.5 },
                    primary_speaker: SpeakerId((i % num_centroids) as u32),
                    secondary_speaker: None,
                    embedding: normalize(vec![embedding_seed_a, embedding_seed_b, 0.0, 0.0]),
                })
                .collect();

            let r = OverlapResegmenter::default();
            let out = r.resegment(ResegmentInputs {
                primary_turns: &primary,
                speaker_centroids: &centroids,
                overlap_regions: &regions,
            }).unwrap();

            // Set inclusion: every primary turn appears with the same triple in the output.
            for p in &primary {
                let key = (p.time.start, p.time.end, p.speaker.0);
                let found = out.iter().any(|t| {
                    (t.time.start - p.time.start).abs() < 1e-9
                        && (t.time.end - p.time.end).abs() < 1e-9
                        && t.speaker.0 == key.2
                });
                prop_assert!(found, "primary turn {:?} missing from output", key);
            }
        }

        /// Invariant 2: output is sorted by `time.start`.
        #[test]
        fn output_is_sorted_by_start(
            primary_count in 0usize..6,
            num_centroids in 2usize..5,
            overlap_count in 0usize..5,
        ) {
            let dim = 4;
            let centroids: Vec<SpeakerCentroid> = (0..num_centroids)
                .map(|i| SpeakerCentroid {
                    speaker: SpeakerId(i as u32),
                    embedding: unit(dim, i % dim),
                })
                .collect();
            // Deliberately unsorted primary turns — the resegmenter must still
            // produce a sorted output.
            let primary: Vec<SpeakerTurn> = (0..primary_count)
                .map(|i| {
                    let start = ((primary_count - i) as f64) * 1.3;
                    turn(start, start + 0.5, (i % num_centroids) as u32)
                })
                .collect();
            let regions: Vec<OverlapRegionInput> = (0..overlap_count)
                .map(|i| OverlapRegionInput {
                    time: TimeRange { start: i as f64 * 0.9, end: i as f64 * 0.9 + 0.4 },
                    primary_speaker: SpeakerId(0),
                    secondary_speaker: None,
                    embedding: unit(dim, (i + 1) % dim),
                })
                .collect();

            let r = OverlapResegmenter::default();
            let out = r.resegment(ResegmentInputs {
                primary_turns: &primary,
                speaker_centroids: &centroids,
                overlap_regions: &regions,
            }).unwrap();

            for w in out.windows(2) {
                prop_assert!(
                    w[0].time.start <= w[1].time.start,
                    "output not sorted: {} > {}",
                    w[0].time.start,
                    w[1].time.start
                );
            }
        }

        /// Invariant 3: `compute_centroids` returns L2-normalized centroids.
        #[test]
        fn compute_centroids_outputs_are_l2_normalized(
            num_clusters in 1usize..5,
            members_per_cluster in 1usize..6,
        ) {
            let dim = 4;
            let total = num_clusters * members_per_cluster;
            let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(total);
            let mut labels: Vec<usize> = Vec::with_capacity(total);
            for cluster in 0..num_clusters {
                for _ in 0..members_per_cluster {
                    embeddings.push(unit(dim, cluster % dim));
                    labels.push(cluster);
                }
            }
            let centroids = compute_centroids(&embeddings, &labels);
            prop_assert_eq!(centroids.len(), num_clusters);
            for c in &centroids {
                let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!(
                    (n - 1.0).abs() < 1e-3,
                    "centroid not L2-normalized: norm={}",
                    n
                );
            }
        }
    }
}
