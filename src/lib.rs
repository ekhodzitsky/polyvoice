#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![deny(unsafe_op_in_unsafe_fn)]

//! # polyvoice
//!
//! Speaker diarization library for Rust — online (streaming) and offline
//! (file-based), ONNX-powered, and ecosystem-agnostic.
//!
//! Designed to be embedded into any Rust application that needs to answer
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

pub mod ahc;
pub mod cluster;
pub mod der;
pub mod embedding;
pub mod features;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod kmeans;
#[cfg(feature = "spectral")]
pub mod spectral;
pub use features::{FbankConfig, FbankExtractor};
// Deprecated: `compute_fbank` is still accessible via `polyvoice::features::compute_fbank`
// but no longer re-exported at the crate root. Use `FbankExtractor::extract` instead.
pub mod offline;
pub mod online;
pub mod overlap;
pub mod rttm;
pub mod silero_vad;
pub mod types;
pub mod utils;
pub use utils::merge_segments;
pub mod vad;
pub mod wav;

#[cfg(feature = "download")]
pub mod models;

#[cfg(feature = "segmentation")]
pub mod segmentation;

#[cfg(feature = "segmentation")]
pub use segmentation::{
    AggregationConfig, Aggregator, FrameLabel, MIN_AUDIO_SAMPLES, PowersetClass, PowersetDecoder,
    RawSegment, SegmentationError, Segmenter, WindowOutput,
};

#[cfg(all(feature = "onnx", feature = "segmentation"))]
pub use segmentation::{PowersetConfig, PowersetSegmenter};

#[cfg(feature = "embedder")]
pub mod embedder;

#[cfg(feature = "embedder")]
pub use embedder::{Embedder, EmbedderError, EmbedderPool, apply_overlap_mask};

#[cfg(all(feature = "onnx", feature = "embedder"))]
pub use embedder::{CamPlusPlusExtractor, ResNet34Adapter};

#[cfg(feature = "clusterer")]
pub mod clusterer;

#[cfg(feature = "clusterer")]
pub use clusterer::{AhcClusterer, Clusterer, ClustererError};

#[cfg(all(feature = "clusterer", feature = "spectral"))]
pub use clusterer::NmeScClusterer;

#[cfg(feature = "resegmentation")]
pub mod resegmentation;

#[cfg(feature = "resegmentation")]
pub use resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentError, ResegmentInputs, Resegmenter,
    SpeakerCentroid, compute_centroids,
};

#[cfg(all(feature = "resegmentation", feature = "segmentation"))]
pub use resegmentation::extract_overlap_time_ranges;

#[cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline;

#[cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub use pipeline::{
    ClustererKind, ConfigError, ExecutionProvider, Pipeline, PipelineBuilder,
    PipelineConfig, PipelineError,
};

#[cfg(feature = "onnx")]
pub mod ecapa;
#[cfg(feature = "onnx")]
pub mod onnx;

// Public re-exports for ergonomic use.
pub use cluster::SpeakerCluster;
pub use embedding::{DummyExtractor, EmbeddingError, EmbeddingExtractor};
#[cfg(feature = "download")]
pub use models::{ModelRegistry, ProfileModels, RegistryError};
pub use offline::OfflineDiarizer;
pub use online::OnlineDiarizer;
pub use overlap::{OverlapRegion, detect_overlaps};
pub use silero_vad::SileroVad;
pub use types::{
    ClusteringBackend, Confidence, DiarizationConfig, DiarizationResult, EmbeddingDim, Profile,
    SampleRate, Seconds, Segment, SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment,
    remap_segments, remap_turns,
};
pub use vad::{EnergyVad, VadConfig, VadError, VoiceActivityDetector, segment_speech};

#[cfg(feature = "onnx")]
#[allow(deprecated)]
pub use ecapa::EcapaTdnnExtractor;
#[cfg(feature = "onnx")]
pub use ecapa::FbankOnnxExtractor;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingExtractor;
