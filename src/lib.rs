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
//! let config = DiarizationConfig::default();
//! let diarizer = OfflineDiarizer::new(config);
//! let extractor = DummyExtractor::new(256);
//!
//! // Load 16 kHz mono f32 samples (e.g. from an audio file decoder).
//! let samples: Vec<f32> = vec![0.0; 16000 * 10];
//! let result = diarizer.run(&samples, &extractor).unwrap();
//!
//! for turn in &result.turns {
//!     println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
//! }
//! ```

pub mod cluster;
pub mod embedding;
pub mod offline;
pub mod online;
pub mod overlap;
pub mod types;
pub mod utils;
pub mod vad;

#[cfg(feature = "onnx")]
pub mod onnx;

// Public re-exports for ergonomic use.
pub use cluster::SpeakerCluster;
pub use embedding::{DummyExtractor, EmbeddingExtractor, EmbeddingError};
pub use offline::OfflineDiarizer;
pub use online::OnlineDiarizer;
pub use overlap::detect_overlaps;
pub use types::*;
pub use vad::{segment_speech, EnergyVad, VoiceActivityDetector, VadError};

#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingExtractor;
