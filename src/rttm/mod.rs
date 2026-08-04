//! RTTM (Rich Transcription Time Marked) parser and writer.

use crate::types::{SpeakerId, SpeakerTurn, TimeRange};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum RttmError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid RTTM line {line}: {reason}")]
    Parse { line: usize, reason: String },
}

#[derive(Debug, Clone)]
pub struct RttmSegment {
    pub file_id: String,
    pub start: f64,
    pub duration: f64,
    pub speaker: String,
}

impl RttmSegment {
    /// { true }
    /// pub fn end(&self) -> f64
    /// { ret == self.start + self.duration }
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }
}

/// { true }
/// `pub fn parse_rttm<R: BufRead>(reader: R) -> Result<Vec<RttmSegment>, RttmError>`
/// { ret.as_ref().map_or(true, |v| v.iter().all(|s| s.start >= 0.0 && s.duration >= 0.0)) }
/// Parse RTTM content from a reader, returning segments grouped by file_id.
pub fn parse_rttm<R: BufRead>(reader: R) -> Result<Vec<RttmSegment>, RttmError> {
    let mut segments = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 9 {
            return Err(RttmError::Parse {
                line: idx + 1,
                reason: format!("expected >= 9 fields, got {}", fields.len()),
            });
        }

        if fields[0] != "SPEAKER" {
            continue;
        }

        let start: f64 = fields[3].parse().map_err(|_| RttmError::Parse {
            line: idx + 1,
            reason: format!("invalid start time: {}", fields[3]),
        })?;
        if !start.is_finite() || start < 0.0 {
            return Err(RttmError::Parse {
                line: idx + 1,
                reason: format!("invalid start time: {}", start),
            });
        }

        let duration: f64 = fields[4].parse().map_err(|_| RttmError::Parse {
            line: idx + 1,
            reason: format!("invalid duration: {}", fields[4]),
        })?;
        if !duration.is_finite() || duration < 0.0 {
            return Err(RttmError::Parse {
                line: idx + 1,
                reason: format!("invalid duration: {}", duration),
            });
        }
        let end = start + duration;
        if !end.is_finite() {
            return Err(RttmError::Parse {
                line: idx + 1,
                reason: format!("non-finite segment end time: {}", end),
            });
        }

        segments.push(RttmSegment {
            file_id: fields[1].to_string(),
            start,
            duration,
            speaker: fields[7].to_string(),
        });
    }

    Ok(segments)
}

/// { true }
/// `pub fn parse_rttm_file(path: &Path) -> Result<Vec<RttmSegment>, RttmError>`
/// { true }
/// Parse an RTTM file from disk.
pub fn parse_rttm_file(path: &Path) -> Result<Vec<RttmSegment>, RttmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    parse_rttm(reader)
}

/// { true }
/// `pub fn group_by_file(segments: &[RttmSegment]) -> HashMap<&str, Vec<&RttmSegment>>`
/// { ret.len() <= segments.len() }
/// Group RTTM segments by file_id.
pub fn group_by_file(segments: &[RttmSegment]) -> HashMap<&str, Vec<&RttmSegment>> {
    let mut groups: HashMap<&str, Vec<&RttmSegment>> = HashMap::new();
    for seg in segments {
        groups.entry(&seg.file_id).or_default().push(seg);
    }
    groups
}

/// { true }
/// `pub fn to_speaker_turns( segments: &[RttmSegment], ) -> (Vec<SpeakerTurn>, HashMap<String, SpeakerId>)`
/// { ret.0.len() == segments.len() }
/// Convert RTTM segments to SpeakerTurns with string→SpeakerId mapping.
pub fn to_speaker_turns(
    segments: &[RttmSegment],
) -> (Vec<SpeakerTurn>, HashMap<String, SpeakerId>) {
    let mut speaker_map: HashMap<String, SpeakerId> = HashMap::new();
    let mut next_id = 0u32;

    let turns = segments
        .iter()
        .map(|seg| {
            let id = *speaker_map.entry(seg.speaker.clone()).or_insert_with(|| {
                let id = SpeakerId(next_id);
                next_id += 1;
                id
            });
            SpeakerTurn {
                speaker: id,
                time: TimeRange {
                    start: seg.start,
                    end: seg.end(),
                },
                text: None,
                stable: true,
            }
        })
        .collect();

    (turns, speaker_map)
}

/// { true }
/// `pub fn write_rttm<W: Write>( writer: &mut W, file_id: &str, turns: &[SpeakerTurn], ) -> Result<(), RttmError>`
/// { true }
/// Write speaker turns as RTTM to a writer.
pub fn write_rttm<W: Write>(
    writer: &mut W,
    file_id: &str,
    turns: &[SpeakerTurn],
) -> Result<(), RttmError> {
    for turn in turns {
        writeln!(
            writer,
            "SPEAKER {} 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
            file_id,
            turn.time.start,
            turn.time.duration(),
            turn.speaker,
        )?;
    }
    Ok(())
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_rttm() {
        let input = "\
SPEAKER file1 1 0.500 2.300 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER file1 1 3.000 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER file1 1 5.000 3.000 <NA> <NA> SPEAKER_00 <NA> <NA>
";
        let segments = parse_rttm(input.as_bytes()).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].file_id, "file1");
        assert!((segments[0].start - 0.5).abs() < 1e-6);
        assert!((segments[0].duration - 2.3).abs() < 1e-6);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[1].end() - 4.5).abs() < 1e-6);
    }

    #[test]
    fn skip_comments_and_empty() {
        let input = "\
; This is a comment
SPEAKER file1 1 0.0 1.0 <NA> <NA> A <NA> <NA>

SPEAKER file1 1 2.0 1.0 <NA> <NA> B <NA> <NA>
";
        let segments = parse_rttm(input.as_bytes()).unwrap();
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn roundtrip_rttm() {
        let turns = vec![
            SpeakerTurn {
                speaker: SpeakerId(0),
                time: TimeRange {
                    start: 0.5,
                    end: 2.8,
                },
                text: None,
                stable: true,
            },
            SpeakerTurn {
                speaker: SpeakerId(1),
                time: TimeRange {
                    start: 3.0,
                    end: 4.5,
                },
                text: None,
                stable: true,
            },
        ];
        let mut buf = Vec::new();
        write_rttm(&mut buf, "test", &turns).unwrap();
        let parsed = parse_rttm(buf.as_slice()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0].start - 0.5).abs() < 1e-3);
        assert!((parsed[0].duration - 2.3).abs() < 1e-3);
        assert_eq!(parsed[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn to_speaker_turns_mapping() {
        let segments = vec![
            RttmSegment {
                file_id: "f".into(),
                start: 0.0,
                duration: 1.0,
                speaker: "Alice".into(),
            },
            RttmSegment {
                file_id: "f".into(),
                start: 1.5,
                duration: 2.0,
                speaker: "Bob".into(),
            },
            RttmSegment {
                file_id: "f".into(),
                start: 4.0,
                duration: 1.0,
                speaker: "Alice".into(),
            },
        ];
        let (turns, map) = to_speaker_turns(&segments);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].speaker, turns[2].speaker);
        assert_ne!(turns[0].speaker, turns[1].speaker);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn reject_non_finite_end_time() {
        let input = format!(
            "SPEAKER f 1 {:.3} {:.3} <NA> <NA> A <NA> <NA>\n",
            f64::MAX,
            f64::MAX
        );
        let result = parse_rttm(input.as_bytes());
        assert!(result.is_err());
    }

    fn parse_error(input: &str) -> (usize, String) {
        match parse_rttm(input.as_bytes()) {
            Err(RttmError::Parse { line, reason }) => (line, reason),
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[test]
    fn reject_too_few_fields_with_line_number() {
        // First line is valid; the malformed line reports its 1-based index.
        let input = "\
SPEAKER f 1 0.0 1.0 <NA> <NA> A <NA> <NA>
; comment lines do not count as content but do count as lines
SPEAKER f 1 0.5 2.3 <NA> <NA> SPEAKER_00
";
        let (line, reason) = parse_error(input);
        assert_eq!(line, 3);
        assert!(reason.contains("expected >= 9 fields, got 8"), "{reason}");
    }

    #[test]
    fn skip_non_speaker_record_types() {
        // RTTM files may carry SPKR-INFO etc.; only SPEAKER lines are segments.
        let input = "\
SPKR-INFO file1 1 <NA> <NA> <NA> unknown SPEAKER_00 <NA>
SPEAKER file1 1 0.0 1.0 <NA> <NA> A <NA> <NA>
";
        let segments = parse_rttm(input.as_bytes()).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "A");
    }

    #[test]
    fn reject_non_numeric_start() {
        let (_, reason) = parse_error("SPEAKER f 1 abc 1.0 <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid start time: abc"), "{reason}");
    }

    #[test]
    fn reject_negative_or_non_finite_start() {
        let (_, reason) = parse_error("SPEAKER f 1 -0.5 1.0 <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid start time: -0.5"), "{reason}");
        // "NaN"/"inf" parse as f64 but fail the finite/non-negative check.
        let (_, reason) = parse_error("SPEAKER f 1 NaN 1.0 <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid start time"), "{reason}");
        let (_, reason) = parse_error("SPEAKER f 1 inf 1.0 <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid start time"), "{reason}");
    }

    #[test]
    fn reject_bad_duration() {
        let (_, reason) = parse_error("SPEAKER f 1 0.0 xyz <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid duration: xyz"), "{reason}");
        let (_, reason) = parse_error("SPEAKER f 1 0.0 -1.0 <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid duration: -1"), "{reason}");
        let (_, reason) = parse_error("SPEAKER f 1 0.0 NaN <NA> <NA> A <NA> <NA>\n");
        assert!(reason.contains("invalid duration"), "{reason}");
    }

    #[test]
    fn zero_duration_segments_are_accepted() {
        let input = "SPEAKER f 1 1.0 0.0 <NA> <NA> A <NA> <NA>\n";
        let segments = parse_rttm(input.as_bytes()).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].end(), 1.0);
    }

    #[test]
    fn parse_rttm_file_reads_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ref.rttm");
        std::fs::write(&path, "SPEAKER f 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n").unwrap();
        let segments = parse_rttm_file(&path).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].file_id, "f");
    }

    #[test]
    fn parse_rttm_file_missing_path_is_io_error() {
        let err = parse_rttm_file(Path::new("definitely/not/a/real/file.rttm")).unwrap_err();
        assert!(matches!(err, RttmError::Io(_)));
    }

    #[test]
    fn error_display_includes_line_and_reason() {
        let e = RttmError::Parse {
            line: 7,
            reason: "boom".into(),
        };
        assert_eq!(e.to_string(), "invalid RTTM line 7: boom");
        let io = RttmError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "nope"));
        assert!(io.to_string().starts_with("I/O error:"), "{io}");
    }

    #[test]
    fn group_by_file_collects_per_file() {
        let mk = |file: &str, start: f64| RttmSegment {
            file_id: file.into(),
            start,
            duration: 1.0,
            speaker: "A".into(),
        };
        let segments = vec![mk("a", 0.0), mk("b", 0.0), mk("a", 2.0)];
        let groups = group_by_file(&segments);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["a"].len(), 2);
        assert_eq!(groups["b"].len(), 1);
        // Grouping keeps the original order within each file.
        assert!(groups["a"][0].start < groups["a"][1].start);
    }

    #[test]
    fn to_speaker_turns_empty_input() {
        let (turns, map) = to_speaker_turns(&[]);
        assert!(turns.is_empty());
        assert!(map.is_empty());
    }

    #[test]
    fn to_speaker_turns_preserves_overlap_and_times() {
        // Overlapping reference segments stay overlapping turns (no merging).
        let segments = vec![
            RttmSegment {
                file_id: "f".into(),
                start: 0.0,
                duration: 2.0,
                speaker: "A".into(),
            },
            RttmSegment {
                file_id: "f".into(),
                start: 1.0,
                duration: 2.0,
                speaker: "B".into(),
            },
        ];
        let (turns, map) = to_speaker_turns(&segments);
        assert_eq!(turns.len(), 2);
        assert!((turns[0].time.end - 2.0).abs() < 1e-9);
        assert!((turns[1].time.start - 1.0).abs() < 1e-9);
        assert!(turns[0].stable);
        assert!(turns[0].text.is_none());
        assert_eq!(map["A"], turns[0].speaker);
        assert_eq!(map["B"], turns[1].speaker);
    }

    #[test]
    fn write_rttm_empty_turns_produces_no_lines() {
        let mut buf = Vec::new();
        write_rttm(&mut buf, "f", &[]).unwrap();
        assert!(buf.is_empty());
    }

    #[test]
    fn write_rttm_propagates_io_errors() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "closed",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let turns = vec![SpeakerTurn {
            speaker: SpeakerId(0),
            time: TimeRange {
                start: 0.0,
                end: 1.0,
            },
            text: None,
            stable: true,
        }];
        let mut w = FailingWriter;
        assert!(matches!(
            write_rttm(&mut w, "f", &turns),
            Err(RttmError::Io(_))
        ));
    }
}
