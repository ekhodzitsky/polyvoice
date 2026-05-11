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
//! Build a diarization pipeline using [`Pipeline`] and [`ModelRegistry`].
//! See the `pipeline` module for details.

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
pub mod online;
pub mod overlap;
pub mod streaming;
pub mod rttm;
pub mod types;
pub mod utils;
pub use utils::merge_segments;
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

pub mod pipeline;
pub use pipeline::{Pipeline, PipelineError};

pub mod vad;
pub use vad::{EnergyVad, VadConfig, VadError, VoiceActivityDetector, segment_speech};

#[cfg(feature = "onnx")]
pub mod silero_vad;
#[cfg(feature = "onnx")]
pub use silero_vad::SileroVad;

#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbeddingExtractor;

#[cfg(feature = "onnx")]
pub mod ecapa;

// Public re-exports for ergonomic use.
pub use cluster::ClusterConfig;
pub use cluster::SpeakerCluster;
pub use embedding::{DummyExtractor, EmbeddingError, EmbeddingExtractor};
#[cfg(feature = "download")]
pub use models::{ModelRegistry, ProfileModels, RegistryError};
pub use online::OnlineConfig;
#[allow(deprecated)]
pub use online::OnlineDiarizer;
pub use overlap::{OverlapRegion, detect_overlaps};
pub use types::{
    Confidence, DiarizationConfig, DiarizationResult, Profile, SampleRate, Seconds, Segment,
    SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment, remap_segments, remap_turns,
};

#[cfg(feature = "onnx")]
pub use ecapa::FbankOnnxExtractor;
