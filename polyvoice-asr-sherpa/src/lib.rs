#![cfg_attr(not(test), deny(clippy::unwrap_used))]

//! Opt-in CJK/multilingual ASR companion for polyvoice.
//!
//! Wraps SenseVoice (zh / en / ja / ko / yue) and Paraformer (zh) via
//! [`sherpa_rs`] behind the core [`polyvoice::Asr`] trait, closing the CJK gap
//! of the Parakeet companion (`polyvoice-asr`, ~25 European languages).
//!
//! **Trade-off, read before depending on this crate:** sherpa-onnx ships a
//! SECOND C++ ONNX Runtime that does not share the core crate's `ort`. This
//! crate is therefore strictly opt-in, deliberately outside the polyvoice
//! workspace, and must never appear in the core dependency graph — pick it
//! only when you need CJK. For European languages prefer `polyvoice-asr`.
//!
//! Timestamp granularity: sherpa's offline recognizers emit TOKEN start times
//! (no durations, no confidences). This crate merges BPE continuation tokens
//! into words (CJK characters stay one word each), derives each word's end
//! from the next word's start (clip end for the last one), and reports
//! `confidence = 1.0` since sherpa exposes none.

use std::path::Path;
use std::sync::Mutex;

use polyvoice::asr::{Asr, AsrError};
use polyvoice::types::{SampleRate, TimeRange, Word};
use sherpa_rs::paraformer::{ParaformerConfig, ParaformerRecognizer};
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};

/// Sherpa's offline recognizers operate on 16 kHz features.
pub const REQUIRED_SAMPLE_RATE: u32 = 16_000;

/// SenseVoice-Small backend (multilingual: zh, en, ja, ko, yue).
///
/// `transcribe` takes `&self` (the trait is object-safe) but sherpa's
/// recognizer needs `&mut`, so it lives behind a [`Mutex`].
pub struct SenseVoiceAsr {
    inner: Mutex<SenseVoiceRecognizer>,
}

impl SenseVoiceAsr {
    /// Load from `model.onnx` + `tokens.txt`. `language` is `"auto"` or one of
    /// `zh`/`en`/`ja`/`ko`/`yue`; inverse text normalization stays on.
    pub fn from_files(
        model: impl AsRef<Path>,
        tokens: impl AsRef<Path>,
        language: &str,
    ) -> Result<Self, AsrError> {
        let model = model.as_ref();
        let tokens = tokens.as_ref();
        for p in [model, tokens] {
            if !p.is_file() {
                return Err(AsrError::ModelIo {
                    path: p.to_path_buf(),
                    detail: "file not found".to_owned(),
                });
            }
        }
        let config = SenseVoiceConfig {
            model: model.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            language: language.to_owned(),
            ..SenseVoiceConfig::default()
        };
        let recognizer =
            SenseVoiceRecognizer::new(config).map_err(|e| AsrError::Backend(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(recognizer),
        })
    }
}

impl Asr for SenseVoiceAsr {
    fn transcribe(&self, audio: &[f32], sample_rate: SampleRate) -> Result<Vec<Word>, AsrError> {
        check_rate(sample_rate)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Backend("recognizer mutex poisoned".to_owned()))?;
        let result = inner.transcribe(sample_rate.get(), audio);
        Ok(words_from_tokens(
            &result.tokens,
            &result.timestamps,
            audio.len() as f32 / sample_rate.get() as f32,
        ))
    }
}

/// Paraformer backend (Mandarin Chinese).
pub struct ParaformerAsr {
    inner: Mutex<ParaformerRecognizer>,
}

impl ParaformerAsr {
    /// Load from `model.onnx` + `tokens.txt`.
    pub fn from_files(model: impl AsRef<Path>, tokens: impl AsRef<Path>) -> Result<Self, AsrError> {
        let model = model.as_ref();
        let tokens = tokens.as_ref();
        for p in [model, tokens] {
            if !p.is_file() {
                return Err(AsrError::ModelIo {
                    path: p.to_path_buf(),
                    detail: "file not found".to_owned(),
                });
            }
        }
        let config = ParaformerConfig {
            model: model.to_string_lossy().into_owned(),
            tokens: tokens.to_string_lossy().into_owned(),
            ..ParaformerConfig::default()
        };
        let recognizer =
            ParaformerRecognizer::new(config).map_err(|e| AsrError::Backend(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(recognizer),
        })
    }
}

impl Asr for ParaformerAsr {
    fn transcribe(&self, audio: &[f32], sample_rate: SampleRate) -> Result<Vec<Word>, AsrError> {
        check_rate(sample_rate)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| AsrError::Backend("recognizer mutex poisoned".to_owned()))?;
        let result = inner.transcribe(sample_rate.get(), audio);
        Ok(words_from_tokens(
            &result.tokens,
            &result.timestamps,
            audio.len() as f32 / sample_rate.get() as f32,
        ))
    }
}

fn check_rate(sample_rate: SampleRate) -> Result<(), AsrError> {
    if sample_rate.get() != REQUIRED_SAMPLE_RATE {
        return Err(AsrError::UnsupportedSampleRate {
            actual: sample_rate.get(),
        });
    }
    Ok(())
}

/// True for scripts where every character is its own word (han, kana, hangul).
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF        // hiragana + katakana
        | 0x3400..=0x4DBF      // CJK extension A
        | 0x4E00..=0x9FFF      // CJK unified ideographs
        | 0xAC00..=0xD7AF      // hangul syllables
        | 0xF900..=0xFAFF      // CJK compatibility ideographs
    )
}

/// Merge sherpa's token stream into time-stamped [`Word`]s.
///
/// Rules: tokens that look like control tags (`<|zh|>`, `<|NEUTRAL|>` …) are
/// dropped; a token starts a NEW word when it carries the BPE word-boundary
/// marker (`▁`) or a leading space, when it (or the previous token) is a CJK
/// character, or when it is the first token; otherwise it continues the
/// previous word. Each word's end is the next word's start (the clip end for
/// the last word, floored at 50 ms so `end > start` always holds). Sherpa
/// exposes no per-token confidence, so every word reports `1.0`.
fn words_from_tokens(tokens: &[String], timestamps: &[f32], clip_secs: f32) -> Vec<Word> {
    let mut words: Vec<(String, f32)> = Vec::new(); // (text, start)
    for (i, raw) in tokens.iter().enumerate() {
        let start = timestamps
            .get(i)
            .copied()
            .unwrap_or_else(|| words.last().map(|(_, s)| *s).unwrap_or(0.0));
        if raw.starts_with("<|") || raw.is_empty() {
            continue;
        }
        let trimmed = raw.trim_start_matches('▁').trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let is_boundary = raw.starts_with('▁') || raw.starts_with(' ');
        let this_cjk = trimmed.chars().next().map(is_cjk).unwrap_or(false);
        let prev_cjk = words
            .last()
            .and_then(|(w, _)| w.chars().last())
            .map(is_cjk)
            .unwrap_or(false);
        if words.is_empty() || is_boundary || this_cjk || prev_cjk {
            words.push((trimmed.to_owned(), start));
        } else if let Some((last, _)) = words.last_mut() {
            last.push_str(trimmed);
        }
    }

    let n = words.len();
    words
        .iter()
        .enumerate()
        .map(|(i, (text, start))| {
            let next_start = if i + 1 < n {
                words[i + 1].1
            } else {
                clip_secs.max(*start)
            };
            let end = next_start.max(start + 0.05);
            Word {
                word: text.clone(),
                time: TimeRange {
                    start: *start as f64,
                    end: end as f64,
                },
                confidence: 1.0,
            }
        })
        .collect()
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn bpe_continuations_merge_into_words() {
        // "▁hel" "lo" "▁world" -> "hello", "world"
        let tokens = toks(&["▁hel", "lo", "▁world"]);
        let ts = vec![0.10, 0.30, 0.62];
        let words = words_from_tokens(&tokens, &ts, 1.5);
        assert_eq!(
            words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            ["hello", "world"]
        );
        assert!((words[0].time.start - 0.10).abs() < 1e-6);
        assert!((words[0].time.end - 0.62).abs() < 1e-6, "end = next start");
        assert!((words[1].time.end - 1.5).abs() < 1e-6, "last ends at clip");
    }

    #[test]
    fn cjk_characters_stay_separate_words() {
        let tokens = toks(&["你", "好", "世", "界"]);
        let ts = vec![0.1, 0.3, 0.5, 0.7];
        let words = words_from_tokens(&tokens, &ts, 1.0);
        assert_eq!(words.len(), 4, "one word per CJK character");
        assert!(
            words
                .windows(2)
                .all(|w| w[0].time.end <= w[1].time.start + 1e-6),
            "monotonic, non-overlapping"
        );
    }

    #[test]
    fn control_tags_are_dropped_and_end_is_floored() {
        let tokens = toks(&["<|zh|>", "<|NEUTRAL|>", "▁hi"]);
        let ts = vec![0.0, 0.0, 0.9];
        // Clip shorter than the last token start: end must still exceed start.
        let words = words_from_tokens(&tokens, &ts, 0.5);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "hi");
        assert!(words[0].time.end > words[0].time.start);
    }

    #[test]
    fn mixed_cjk_and_latin_boundaries() {
        // CJK, then a BPE latin word, then CJK again.
        let tokens = toks(&["中", "▁ok", "ay", "国"]);
        let ts = vec![0.0, 0.2, 0.35, 0.6];
        let words = words_from_tokens(&tokens, &ts, 1.0);
        assert_eq!(
            words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>(),
            ["中", "okay", "国"]
        );
    }
}
