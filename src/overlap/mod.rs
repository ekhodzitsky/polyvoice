//! Overlap detection: identify frames where multiple speakers may be active.

use crate::types::{Segment, SpeakerId, TimeRange};

/// { true }
/// `pub fn detect_overlaps(segments: &[Segment]) -> Vec<OverlapRegion>`
/// { ret.iter().all(|o| o.speakers.len() >= 2) }
/// Detect regions where multiple speakers overlap.
///
/// Zero-length segments and segments without a speaker label are ignored.
///
/// ```rust
/// use polyvoice::{detect_overlaps, Segment, SpeakerId, TimeRange};
/// let segments = vec![
///     Segment { time: TimeRange { start: 0.0, end: 3.0 }, speaker: Some(SpeakerId(0)), confidence: None },
///     Segment { time: TimeRange { start: 2.0, end: 5.0 }, speaker: Some(SpeakerId(1)), confidence: None },
/// ];
/// let overlaps = detect_overlaps(&segments);
/// assert_eq!(overlaps.len(), 1);
/// assert_eq!(overlaps[0].speakers.len(), 2);
/// ```
pub fn detect_overlaps(segments: &[Segment]) -> Vec<OverlapRegion> {
    if segments.len() < 2 {
        return Vec::new();
    }

    // Collect all unique event points (start / end of every segment).
    // Filter out zero-length segments and segments without a speaker label
    // to avoid phantom overlaps.
    let mut events: Vec<(f64, bool, SpeakerId)> = Vec::new(); // (time, is_start, speaker)
    for seg in segments {
        if let Some(spk) = seg.speaker
            && seg.time.start < seg.time.end
        {
            events.push((seg.time.start, true, spk));
            events.push((seg.time.end, false, spk));
        }
    }

    events.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut active: Vec<SpeakerId> = Vec::new();
    let mut overlaps = Vec::new();
    let mut last_time = events.first().map(|e| e.0).unwrap_or(0.0);

    for (time, is_start, speaker) in events {
        if time > last_time && active.len() > 1 {
            overlaps.push(OverlapRegion {
                time: TimeRange {
                    start: last_time,
                    end: time,
                },
                speakers: active.clone(),
            });
        }
        if is_start {
            if !active.contains(&speaker) {
                active.push(speaker);
            }
        } else {
            active.retain(|&s| s != speaker);
        }
        last_time = time;
    }

    // Merge adjacent overlaps with the same speaker set.
    merge_adjacent_overlaps(overlaps)
}

/// A region where multiple speakers overlap.
///
/// ```rust
/// use polyvoice::{OverlapRegion, SpeakerId, TimeRange};
/// let region = OverlapRegion {
///     time: TimeRange { start: 1.0, end: 2.0 },
///     speakers: vec![SpeakerId(0), SpeakerId(1)],
/// };
/// assert_eq!(region.speakers.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapRegion {
    pub time: TimeRange,
    pub speakers: Vec<SpeakerId>,
}

fn merge_adjacent_overlaps(mut regions: Vec<OverlapRegion>) -> Vec<OverlapRegion> {
    if regions.is_empty() {
        return regions;
    }
    let mut merged = Vec::new();
    let mut current = regions.remove(0);

    for next in regions {
        if current.speakers == next.speakers && (next.time.start - current.time.end).abs() < 1e-3 {
            current.time.end = next.time.end;
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, speaker: u32) -> Segment {
        Segment {
            time: TimeRange { start, end },
            speaker: Some(SpeakerId(speaker)),
            confidence: None,
        }
    }

    #[test]
    fn test_no_overlap() {
        let segs = vec![seg(0.0, 1.0, 0), seg(1.5, 2.5, 1)];
        let overlaps = detect_overlaps(&segs);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_simple_overlap() {
        let segs = vec![seg(0.0, 2.0, 0), seg(1.0, 3.0, 1)];
        let overlaps = detect_overlaps(&segs);
        assert_eq!(overlaps.len(), 1);
        assert!((overlaps[0].time.start - 1.0).abs() < 1e-5);
        assert!((overlaps[0].time.end - 2.0).abs() < 1e-5);
        assert_eq!(overlaps[0].speakers.len(), 2);
    }

    #[test]
    fn test_three_way_overlap() {
        let segs = vec![seg(0.0, 3.0, 0), seg(1.0, 3.0, 1), seg(2.0, 4.0, 2)];
        let overlaps = detect_overlaps(&segs);
        // 1.0-2.0: spk0+spk1
        // 2.0-3.0: spk0+spk1+spk2
        assert_eq!(overlaps.len(), 2);
        assert_eq!(overlaps[1].speakers.len(), 3);
    }

    #[test]
    fn test_zero_length_no_phantom_overlap() {
        // Zero-length segments at start, middle, and end should not produce overlaps.
        let segs = vec![
            seg(0.0, 0.0, 0), // zero-length at start
            seg(0.0, 2.0, 1),
            seg(1.0, 1.0, 2), // zero-length in middle
            seg(1.5, 3.5, 3),
            seg(3.5, 3.5, 0), // zero-length at end
        ];
        let overlaps = detect_overlaps(&segs);
        // Only overlap is between speaker 1 (0.0-2.0) and speaker 3 (1.5-3.5): 1.5-2.0
        assert_eq!(overlaps.len(), 1);
        assert!((overlaps[0].time.start - 1.5).abs() < 1e-5);
        assert!((overlaps[0].time.end - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_unlabeled_segments_ignored() {
        let segs = vec![
            Segment {
                time: TimeRange {
                    start: 0.0,
                    end: 2.0,
                },
                speaker: Some(SpeakerId(0)),
                confidence: None,
            },
            Segment {
                time: TimeRange {
                    start: 1.0,
                    end: 3.0,
                },
                speaker: None,
                confidence: None,
            },
        ];
        let overlaps = detect_overlaps(&segs);
        // The unlabeled segment should be ignored; no overlap detected.
        assert!(overlaps.is_empty());
    }
}
