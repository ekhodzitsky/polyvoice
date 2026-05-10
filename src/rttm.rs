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
/// { TODO: precondition }
/// pub fn end(&self) -> f64
/// { TODO: postcondition }
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }
}

/// { TODO: precondition }
/// pub fn parse_rttm<R: BufRead>(reader: R) -> Result<Vec<RttmSegment>, RttmError>
/// { TODO: postcondition }
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

        segments.push(RttmSegment {
            file_id: fields[1].to_string(),
            start,
            duration,
            speaker: fields[7].to_string(),
        });
    }

    Ok(segments)
}

/// { TODO: precondition }
/// pub fn parse_rttm_file(path: &Path) -> Result<Vec<RttmSegment>, RttmError>
/// { TODO: postcondition }
/// Parse an RTTM file from disk.
pub fn parse_rttm_file(path: &Path) -> Result<Vec<RttmSegment>, RttmError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    parse_rttm(reader)
}

/// { TODO: precondition }
/// pub fn group_by_file(segments: &[RttmSegment]) -> HashMap<&str, Vec<&RttmSegment>>
/// { TODO: postcondition }
/// Group RTTM segments by file_id.
pub fn group_by_file(segments: &[RttmSegment]) -> HashMap<&str, Vec<&RttmSegment>> {
    let mut groups: HashMap<&str, Vec<&RttmSegment>> = HashMap::new();
    for seg in segments {
        groups.entry(&seg.file_id).or_default().push(seg);
    }
    groups
}

/// { TODO: precondition }
/// pub fn to_speaker_turns( segments: &[RttmSegment], ) -> (Vec<SpeakerTurn>, HashMap<String, SpeakerId>)
/// { TODO: postcondition }
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
            }
        })
        .collect();

    (turns, speaker_map)
}

/// { TODO: precondition }
/// pub fn write_rttm<W: Write>( writer: &mut W, file_id: &str, turns: &[SpeakerTurn], ) -> Result<(), RttmError>
/// { TODO: postcondition }
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
            },
            SpeakerTurn {
                speaker: SpeakerId(1),
                time: TimeRange {
                    start: 3.0,
                    end: 4.5,
                },
                text: None,
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
}
