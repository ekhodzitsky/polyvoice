//! Subtitle / plain-text projections of a diarization result (SRT, WebVTT, TXT).
//!
//! Each writer takes an `impl Write` and the speaker turns, projecting **one turn
//! to exactly one block** (lossless). Pure-Rust and wasm-clean — no filesystem
//! I/O inside the writers. Timecodes are rounded to the nearest millisecond; SRT
//! uses a comma decimal separator (`HH:MM:SS,mmm`), WebVTT a dot
//! (`HH:MM:SS.mmm`). Speakers render via the canonical `SPEAKER_NN` label, and
//! the cue carries `SpeakerTurn.text` when an ASR pass has populated it.

use crate::types::SpeakerTurn;
use std::io::{self, Write};

/// Format `secs` as `HH:MM:SS<sep>mmm`, rounding to the nearest millisecond.
/// Non-finite or negative inputs clamp to `00:00:00<sep>000`.
fn timecode(secs: f64, sep: char) -> String {
    let total_ms = if secs.is_finite() && secs > 0.0 {
        (secs * 1000.0).round() as u64
    } else {
        0
    };
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

/// Normalize transcript text for a single-line cue payload: newlines collapse
/// to spaces (one turn = one block is this module's contract) and a literal
/// `-->` is broken up so no payload line can parse as a timing line.
fn sanitize_cue_text(text: &str) -> String {
    text.replace("\r\n", " ")
        .replace(['\r', '\n'], " ")
        .replace("-->", "-- >")
}

/// Escape WebVTT cue text: `&` and `<` are the two characters the spec
/// requires escaping inside cue payloads.
fn escape_vtt_text(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

/// Cue line: `SPEAKER_NN: text` when transcript text is present, else `SPEAKER_NN`.
fn cue_label(turn: &SpeakerTurn) -> String {
    match &turn.text {
        Some(t) if !t.is_empty() => format!("{}: {}", turn.speaker, sanitize_cue_text(t)),
        _ => turn.speaker.to_string(),
    }
}

/// WebVTT cue payload: the standard voice span `<v SPEAKER_NN>text</v>` when
/// transcript text is present (text sanitized and spec-escaped), else the bare
/// `SPEAKER_NN` label (a voice span with an empty payload is not meaningful WebVTT).
fn vtt_cue_payload(turn: &SpeakerTurn) -> String {
    match &turn.text {
        Some(t) if !t.is_empty() => format!(
            "<v {}>{}</v>",
            turn.speaker,
            escape_vtt_text(&sanitize_cue_text(t))
        ),
        _ => turn.speaker.to_string(),
    }
}

/// { true }
/// `pub fn write_srt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()>`
/// { true }
/// Write speaker turns as SubRip (SRT): numbered blocks, `,`-separated milliseconds.
pub fn write_srt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()> {
    for (i, turn) in turns.iter().enumerate() {
        writeln!(writer, "{}", i + 1)?;
        writeln!(
            writer,
            "{} --> {}",
            timecode(turn.time.start, ','),
            timecode(turn.time.end, ',')
        )?;
        writeln!(writer, "{}", cue_label(turn))?;
        writeln!(writer)?;
    }
    Ok(())
}

/// { true }
/// `pub fn write_vtt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()>`
/// { true }
/// Write speaker turns as WebVTT: `WEBVTT` header, `.`-separated milliseconds.
/// Transcribed cues use the standard voice span `<v SPEAKER_NN>text</v>`.
pub fn write_vtt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()> {
    writeln!(writer, "WEBVTT")?;
    writeln!(writer)?;
    for turn in turns {
        writeln!(
            writer,
            "{} --> {}",
            timecode(turn.time.start, '.'),
            timecode(turn.time.end, '.')
        )?;
        writeln!(writer, "{}", vtt_cue_payload(turn))?;
        writeln!(writer)?;
    }
    Ok(())
}

/// { true }
/// `pub fn write_txt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()>`
/// { true }
/// Write speaker turns as a readable transcript: `[start - end] SPEAKER_NN: text`,
/// one line per turn, in order.
pub fn write_txt<W: Write>(writer: &mut W, turns: &[SpeakerTurn]) -> io::Result<()> {
    for turn in turns {
        writeln!(
            writer,
            "[{:.2} - {:.2}] {}",
            turn.time.start.max(0.0),
            turn.time.end.max(0.0),
            cue_label(turn)
        )?;
    }
    Ok(())
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SpeakerId, TimeRange};

    fn turn(id: u32, start: f64, end: f64, text: Option<&str>) -> SpeakerTurn {
        SpeakerTurn {
            speaker: SpeakerId(id),
            time: TimeRange { start, end },
            text: text.map(|s| s.to_owned()),
            stable: true,
        }
    }

    fn render<F>(f: F) -> String
    where
        F: Fn(&mut Vec<u8>) -> io::Result<()>,
    {
        let mut buf = Vec::new();
        f(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn timecode_srt_uses_comma_vtt_uses_dot() {
        assert_eq!(timecode(0.5, ','), "00:00:00,500");
        assert_eq!(timecode(0.5, '.'), "00:00:00.500");
        assert_eq!(timecode(3661.0, ','), "01:01:01,000");
        // rounds to nearest ms
        assert_eq!(timecode(1.2345, ','), "00:00:01,235");
        // non-finite / negative clamp to zero
        assert_eq!(timecode(-3.0, ','), "00:00:00,000");
        assert_eq!(timecode(f64::NAN, '.'), "00:00:00.000");
    }

    #[test]
    fn srt_blocks_are_numbered_and_lossless() {
        let turns = vec![turn(0, 0.5, 2.8, None), turn(1, 3.0, 4.5, Some("hello"))];
        let out = render(|w| write_srt(w, &turns));
        let expected = "\
1
00:00:00,500 --> 00:00:02,800
SPEAKER_00

2
00:00:03,000 --> 00:00:04,500
SPEAKER_01: hello

";
        assert_eq!(out, expected);
        // one block per turn (one "-->" per turn)
        assert_eq!(out.matches(" --> ").count(), turns.len());
    }

    #[test]
    fn vtt_has_header_and_dot_timecodes() {
        let turns = vec![turn(0, 0.0, 1.25, Some("hi"))];
        let out = render(|w| write_vtt(w, &turns));
        assert!(out.starts_with("WEBVTT\n\n"));
        assert!(out.contains("00:00:00.000 --> 00:00:01.250"));
        assert!(out.contains("<v SPEAKER_00>hi</v>"));
        assert_eq!(out.matches(" --> ").count(), turns.len());
    }

    #[test]
    fn vtt_without_text_uses_bare_label() {
        let turns = vec![turn(1, 0.0, 2.0, None)];
        let out = render(|w| write_vtt(w, &turns));
        assert!(out.contains("SPEAKER_01\n"));
        assert!(!out.contains("<v "));
    }

    #[test]
    fn vtt_escapes_markup_characters_in_text() {
        let turns = vec![turn(0, 0.0, 1.0, Some("<unk> & co"))];
        let out = render(|w| write_vtt(w, &turns));
        // `&` and `<` are escaped (the spec's two mandatory escapes); `>` stays raw.
        assert!(out.contains("<v SPEAKER_00>&lt;unk> &amp; co</v>"));
        assert!(!out.contains("<unk>"));
    }

    #[test]
    fn cue_text_newlines_and_arrows_stay_single_line() {
        let turns = vec![turn(0, 0.0, 1.0, Some("a\n\nb --> c"))];
        let srt = render(|w| write_srt(w, &turns));
        let vtt = render(|w| write_vtt(w, &turns));
        // one block per turn survives hostile text
        assert_eq!(srt.matches(" --> ").count(), 1);
        assert_eq!(vtt.matches(" --> ").count(), 1);
        assert!(srt.contains("SPEAKER_00: a  b -- > c"));
        assert!(vtt.contains("a  b -- > c"));
    }

    #[test]
    fn txt_is_readable_per_turn() {
        let turns = vec![turn(0, 0.5, 2.8, None), turn(1, 3.0, 4.5, Some("hello"))];
        let out = render(|w| write_txt(w, &turns));
        let expected = "\
[0.50 - 2.80] SPEAKER_00
[3.00 - 4.50] SPEAKER_01: hello
";
        assert_eq!(out, expected);
        assert_eq!(out.lines().count(), turns.len());
    }

    #[test]
    fn empty_turns_produce_minimal_output() {
        assert_eq!(render(|w| write_srt(w, &[])), "");
        assert_eq!(render(|w| write_vtt(w, &[])), "WEBVTT\n\n");
        assert_eq!(render(|w| write_txt(w, &[])), "");
    }

    #[test]
    fn timecode_clamps_positive_infinity_to_zero() {
        assert_eq!(timecode(f64::INFINITY, ','), "00:00:00,000");
        assert_eq!(timecode(f64::NEG_INFINITY, '.'), "00:00:00.000");
        // Hours beyond a day keep growing (no wraparound).
        assert_eq!(timecode(90061.0, ','), "25:01:01,000");
    }

    #[test]
    fn empty_text_falls_back_to_bare_label() {
        let turns = vec![turn(0, 0.0, 1.0, Some(""))];
        let srt = render(|w| write_srt(w, &turns));
        assert!(srt.contains("SPEAKER_00\n"), "{srt}");
        assert!(!srt.contains("SPEAKER_00: "), "{srt}");
        let vtt = render(|w| write_vtt(w, &turns));
        assert!(!vtt.contains("<v "), "{vtt}");
    }

    #[test]
    fn writers_propagate_io_errors() {
        struct FailingWriter;
        impl Write for FailingWriter {
            fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let turns = vec![turn(0, 0.0, 1.0, None)];
        assert!(write_srt(&mut FailingWriter, &turns).is_err());
        assert!(write_vtt(&mut FailingWriter, &turns).is_err());
        assert!(write_txt(&mut FailingWriter, &turns).is_err());
    }
}
