#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![deny(unsafe_op_in_unsafe_fn)]

//! # polyvoice
//!
//! Speaker diarization library for Rust — online (streaming) and offline
//! (file-based), ONNX-powered, and ecosystem-agnostic.
//!
//! Designed to be embedded into STT servers such as `gigastt`, `phostt`,
//! `nihostt`, `siamstt`, or any other Rust application that needs to answer
//! the question **"who spoke when?"**.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use polyvoice::{OfflineDiarizer, DiarizationConfig, DummyExtractor};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = DiarizationConfig::default();
//!     let diarizer = OfflineDiarizer::new(config);
//!     let extractor = DummyExtractor::new(256);
//!
//!     // Load 16 kHz mono f32 samples (e.g. from an audio file decoder).
//!     let samples: Vec<f32> = vec![0.0; 16000 * 10];
//!     let result = diarizer.run(&samples, &extractor)?;
//!
//!     for turn in &result.turns {
//!         println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
//!     }
//!     Ok(())
//! }
//! ```

pub mod cluster;
pub mod embedding;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod features;
pub use features::{FbankConfig, FbankExtractor};
// Deprecated: `compute_fbank` is still accessible via `polyvoice::features::compute_fbank`
// but no longer re-exported at the crate root. Use `FbankExtractor::extract` instead.
pub mod offline;
pub mod online;
pub mod overlap;
pub mod types;
pub mod utils;
pub mod vad;

#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
pub mod ecapa;

// Public re-exports for ergonomic use.
pub use cluster::SpeakerCluster;
pub use embedding::{DummyExtractor, EmbeddingExtractor, EmbeddingError};
pub use offline::OfflineDiarizer;
pub use online::OnlineDiarizer;
pub use overlap::{detect_overlaps, OverlapRegion};
pub use types::{
    Confidence, DiarizationConfig, DiarizationResult, EmbeddingDim, SampleRate, Seconds,
    Segment, SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment,
    remap_segments, remap_turns,
};
pub use vad::{segment_speech, EnergyVad, VadConfig, VoiceActivityDetector, VadError};

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingExtractor;
#[cfg(feature = "onnx")]
pub use ecapa::EcapaTdnnExtractor;
