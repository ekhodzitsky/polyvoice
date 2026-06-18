//! Opt-in ASR companion for polyvoice.
//!
//! Wraps [`parakeet_rs`] (Parakeet TDT) behind the core [`polyvoice::Asr`] trait,
//! emitting native word-level timestamps for the who-said-what cascade. This crate
//! is a SEPARATE workspace member and is **never** a default feature of the core —
//! the ~600 MB Parakeet model never touches the core footprint. It shares the core
//! ONNX runtime by pinning the exact same `ort` version (enforced in CI).
//!
//! ```no_run
//! use polyvoice_asr::ParakeetAsr;
//! use polyvoice::{Asr, types::SampleRate};
//!
//! let audio: Vec<f32> = vec![0.0; 16_000]; // mono 16 kHz samples
//! let asr = ParakeetAsr::from_dir("./models/parakeet-tdt")?;
//! let sr = SampleRate::new(16_000).expect("valid rate");
//! let words = asr.transcribe(&audio, sr)?; // Vec<Word> with global timestamps
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::path::Path;
use std::sync::Mutex;

use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use polyvoice::asr::{Asr, AsrError};
use polyvoice::types::{SampleRate, TimeRange, Word};

/// Re-exported parakeet-rs execution-provider knobs, so callers can build an
/// [`ExecutionConfig`] without depending on parakeet-rs directly.
pub use parakeet_rs::{ExecutionConfig, ExecutionProvider};

/// Default chunk length (seconds). Parakeet TDT has a ~8-10 min sequence limit,
/// so long audio is split into chunks well under that ceiling.
pub const DEFAULT_CHUNK_SECS: f32 = 240.0;
/// Default overlap (seconds) between consecutive chunks. Words in the overlap are
/// de-duplicated at the seam midpoint so none are dropped or counted twice.
pub const DEFAULT_OVERLAP_SECS: f32 = 5.0;

/// Parakeet TDT ASR backend implementing the core [`Asr`] trait.
///
/// `transcribe` takes `&self` (object-safe), but the underlying parakeet session
/// needs `&mut`, so it is held behind a [`Mutex`].
pub struct ParakeetAsr {
    inner: Mutex<ParakeetTDT>,
    chunk_secs: f32,
    overlap_secs: f32,
}

impl ParakeetAsr {
    /// Load a Parakeet TDT model directory (encoder / decoder_joint / vocab) on CPU.
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        Self::from_dir_with_config(model_dir, None)
    }

    /// Load with an explicit parakeet-rs [`ExecutionConfig`] — used to forward an
    /// execution provider (CoreML / XNNPACK / NNAPI). Build the config with the
    /// re-exported [`ExecutionProvider`] / [`CoreMLComputeUnits`] types.
    pub fn from_dir_with_config(
        model_dir: impl AsRef<Path>,
        config: Option<ExecutionConfig>,
    ) -> Result<Self, AsrError> {
        let dir = model_dir.as_ref();
        let tdt = ParakeetTDT::from_pretrained(dir, config).map_err(|e| AsrError::ModelIo {
            path: dir.to_path_buf(),
            detail: e.to_string(),
        })?;
        Ok(Self {
            inner: Mutex::new(tdt),
            chunk_secs: DEFAULT_CHUNK_SECS,
            overlap_secs: DEFAULT_OVERLAP_SECS,
        })
    }

    /// Override the chunking window and overlap (both in seconds). `overlap` is
    /// clamped below `chunk` so the chunk plan always advances.
    pub fn with_chunking(mut self, chunk_secs: f32, overlap_secs: f32) -> Self {
        self.chunk_secs = chunk_secs.max(1.0);
        self.overlap_secs = overlap_secs.clamp(0.0, self.chunk_secs * 0.5);
        self
    }

    /// Transcribe a single chunk, offsetting token times into the global timeline.
    fn transcribe_chunk(
        &self,
        chunk: &[f32],
        sr: u32,
        offset_secs: f64,
    ) -> Result<Vec<Word>, AsrError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| AsrError::Backend(format!("ASR session lock poisoned: {e}")))?;
        let result = guard
            .transcribe_samples(chunk.to_vec(), sr, 1, Some(TimestampMode::Words))
            .map_err(|e| AsrError::InferenceFailed {
                detail: e.to_string(),
            })?;
        Ok(result
            .tokens
            .into_iter()
            .map(|t| Word {
                word: t.text,
                time: TimeRange {
                    start: t.start as f64 + offset_secs,
                    end: t.end as f64 + offset_secs,
                },
                // parakeet-rs TimedToken does not expose a per-word confidence;
                // timestamps are deterministic, so report full confidence.
                confidence: 1.0,
            })
            .collect())
    }
}

impl Asr for ParakeetAsr {
    fn transcribe(&self, audio: &[f32], sample_rate: SampleRate) -> Result<Vec<Word>, AsrError> {
        let sr = sample_rate.get();
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let chunk_len = ((self.chunk_secs as f64) * sr as f64) as usize;
        let overlap_len = ((self.overlap_secs as f64) * sr as f64) as usize;

        // Short audio: one shot, no stitching.
        if chunk_len == 0 || audio.len() <= chunk_len {
            return self.transcribe_chunk(audio, sr, 0.0);
        }

        // Long audio: overlapping chunks, stitched at the overlap midpoint.
        let step = chunk_len.saturating_sub(overlap_len).max(1);
        let mut acc: Vec<Word> = Vec::new();
        let mut start = 0usize;
        let mut first = true;
        loop {
            let end = (start + chunk_len).min(audio.len());
            let offset = start as f64 / sr as f64;
            let words = self.transcribe_chunk(&audio[start..end], sr, offset)?;
            if first {
                acc = words;
                first = false;
            } else {
                // Seam = midpoint of this chunk's leading overlap region.
                let seam = offset + (self.overlap_secs as f64) / 2.0;
                stitch_at(&mut acc, words, seam);
            }
            if end >= audio.len() {
                break;
            }
            start += step;
        }
        Ok(acc)
    }
}

/// Midpoint time of a word.
fn word_mid(w: &Word) -> f64 {
    (w.time.start + w.time.end) / 2.0
}

/// Stitch `next` chunk words onto `acc` at `seam` (global seconds): keep `acc`
/// words whose midpoint is before the seam and `next` words whose midpoint is at
/// or after it. Each overlap word is therefore counted exactly once (no
/// duplicates) and the timeline stays gap-free (no drops).
fn stitch_at(acc: &mut Vec<Word>, next: Vec<Word>, seam: f64) {
    acc.retain(|w| word_mid(w) < seam);
    acc.extend(next.into_iter().filter(|w| word_mid(w) >= seam));
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            word: text.to_owned(),
            time: TimeRange { start, end },
            confidence: 1.0,
        }
    }
    fn texts(ws: &[Word]) -> Vec<&str> {
        ws.iter().map(|w| w.word.as_str()).collect()
    }

    #[test]
    fn stitch_dedups_overlap_word_without_drop() {
        // prev chunk ends with "hello" straddling the seam; next chunk re-detects
        // "hello" in the overlap, then "world". The seam keeps exactly one "hello".
        let mut acc = vec![
            word("a", 0.0, 1.0),
            word("b", 1.0, 2.0),
            word("hello", 2.0, 2.6),
        ];
        let next = vec![word("hello", 2.1, 2.7), word("world", 3.0, 4.0)];
        stitch_at(&mut acc, next, 2.3);
        assert_eq!(texts(&acc), ["a", "b", "hello", "world"]);
        // The surviving "hello" is the next chunk's copy (start 2.1).
        assert!((acc[2].time.start - 2.1).abs() < 1e-9);
    }

    #[test]
    fn stitch_is_monotonic_and_gapless() {
        let mut acc = vec![
            word("a", 0.0, 1.0),
            word("b", 1.0, 2.0),
            word("c", 2.0, 3.0),
        ];
        let next = vec![
            word("c2", 2.4, 3.0),
            word("d", 3.0, 4.0),
            word("e", 4.0, 5.0),
        ];
        stitch_at(&mut acc, next, 2.5);
        // c (mid 2.5) is NOT < 2.5 -> dropped from acc; c2 (mid 2.7) kept.
        assert_eq!(texts(&acc), ["a", "b", "c2", "d", "e"]);
        for pair in acc.windows(2) {
            assert!(
                word_mid(&pair[0]) <= word_mid(&pair[1]),
                "non-monotonic stitch"
            );
        }
    }

    #[test]
    fn stitch_disjoint_chunks_concatenate() {
        // No real overlap: every acc word before seam, every next word after.
        let mut acc = vec![word("a", 0.0, 1.0), word("b", 1.0, 2.0)];
        let next = vec![word("c", 3.0, 4.0), word("d", 4.0, 5.0)];
        stitch_at(&mut acc, next, 2.5);
        assert_eq!(texts(&acc), ["a", "b", "c", "d"]);
    }

    #[test]
    fn stitch_into_empty_takes_all_next() {
        let mut acc: Vec<Word> = Vec::new();
        let next = vec![word("a", 0.0, 1.0), word("b", 1.0, 2.0)];
        stitch_at(&mut acc, next, 0.0);
        assert_eq!(texts(&acc), ["a", "b"]);
    }

    #[test]
    fn stitch_all_overlap_replaced_by_next() {
        // Both chunks fully cover [2,3); seam 2.0 keeps prev before 2.0 and all next.
        let mut acc = vec![word("a", 0.0, 1.0), word("dup", 2.0, 3.0)];
        let next = vec![word("dup", 2.1, 3.0), word("tail", 3.0, 4.0)];
        stitch_at(&mut acc, next, 2.0);
        assert_eq!(texts(&acc), ["a", "dup", "tail"]);
        assert!((acc[1].time.start - 2.1).abs() < 1e-9, "kept next's copy");
    }
}
