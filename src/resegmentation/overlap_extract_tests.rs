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
fn extract_normalizes_descending_local_indices() {
    // Higher local index first in the input: the emitted pair must still be
    // ordered (lo, hi).
    let segs = vec![raw(0.0, 1.0, 2, true), raw(0.0, 1.0, 0, true)];
    let pairs = extract_overlap_time_ranges(&segs);
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].1, 0);
    assert_eq!(pairs[0].2, 2);
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
