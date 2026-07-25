#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![deny(unsafe_op_in_unsafe_fn)]

//! # polyvoice
//!
//! Speaker diarization library for Rust — online (streaming) and offline
//! (file-based), ecosystem-agnostic. The ONNX path is opt-in (`features =
//! ["onnx", …]`); default features are empty so BYO-embedder consumers can
//! use `Pipeline` / `StreamingPipeline` / `EnergyVad` without linking `ort`.
//!
//! Designed to be embedded into any Rust application that needs to answer
//! the question **"who spoke when?"**.
//!
//! ## Quick start
//!
//! **ONNX path:** build a diarization pipeline using `Pipeline` / `pipeline_v2`
//! and `ModelRegistry` (features `onnx`, `download`, …).
//!
//! **Library mode (no ONNX):** `default-features = false`, implement
//! [`EmbeddingExtractor`], pair with [`EnergyVad`] — see the crate README and
//! `docs/library-mode.md`.
//!
//! ## Module organization
//!
//! polyvoice carries two parallel module families from an in-progress migration
//! to a trait-based v1.0 architecture. This is deliberate (shared math, a
//! compile-time feature guard), not accidental duplication:
//!
//! - **v1.0 trait-based (current architecture):** `embedder` (`Embedder` — the
//!   supported BYO injection trait for offline `Pipeline` and online
//!   `StreamingPipeline`), `clusterer`, `segmentation`, `resegmentation`,
//!   `silero_vad`, and `pipeline_v2` (experimental — see its README).
//! - **Legacy:** `EmbeddingExtractor` in `embedding` is soft-deprecated (still
//!   bridged to `Embedder`); `ecapa` / `onnx` ONNX wrappers remain available.
//!   `cluster` and `vad` are the legacy clustering/VAD surfaces.
//! - **Pipeline status:** `pipeline_v2` (+ VBx clusterer when PLDA is available)
//!   is the **CLI default since 0.11** after a full VoxConverse-test / AMI-test
//!   DER gate (see `docs/BENCHMARKS.md`). The library injection surface is
//!   still `Pipeline` + `StreamingPipeline` with a custom `Embedder` (no `onnx`
//!   required). CLI `--legacy` selects the offline legacy pipeline.
//! - **Shared math, reused by both families:** `ahc`, `kmeans`, `spectral`,
//!   `features`, `der`, `utils`.

pub mod ahc;
pub mod asr;
pub use asr::{Asr, AsrError};
pub mod cluster;
pub mod der;
pub mod embedding;
pub mod features;
#[cfg(feature = "ffi")]
pub mod ffi;
/// Kuhn-Munkres assignment solver. Always compiled (pure Rust, wasm32-clean):
/// shared by `der` (optimal speaker mapping) and `segmentation` (window
/// permutation alignment).
pub(crate) mod hungarian;
pub mod kmeans;
#[cfg(feature = "spectral")]
pub mod spectral;
pub use features::{FbankConfig, FbankExtractor};
pub mod format;
pub mod overlap;
pub mod rttm;
pub mod streaming;
pub mod types;
pub mod utils;
pub use utils::merge_segments;
pub mod wav;
pub mod window;

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

/// Bring-your-own speaker embedder trait (always available; pure Rust core).
/// ONNX-backed adapters still require `features = ["onnx", "embedder"]`.
pub mod embedder;

pub use embedder::{Embedder, EmbedderError, EmbedderPool, apply_overlap_mask};

#[cfg(all(feature = "onnx", feature = "embedder"))]
pub use embedder::{CamPlusPlusExtractor, ResNet34Adapter};

#[cfg(feature = "clusterer")]
pub mod clusterer;

#[cfg(feature = "clusterer")]
pub use clusterer::{AhcClusterer, Clusterer, ClustererError, MinClusterSizeClusterer};

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

#[cfg(feature = "attribution")]
pub mod attribution;
#[cfg(feature = "attribution")]
pub use attribution::{
    AttributionConfig, SpeakerEmbedding, WhoSaidWhat, WordAnchor, attribute_and_fill,
    attribute_and_fill_with_config, attribute_words, attribute_words_with_config, fill_turn_text,
    fill_turn_text_with_config, interpolate_word_timestamps, speaker_embeddings_from_segments,
    who_said_what, who_said_what_with_config,
};

pub mod pipeline;
pub use pipeline::{Pipeline, PipelineError};

#[cfg(all(
    feature = "onnx",
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline_v2;

pub mod vad;
pub use vad::{EnergyVad, VadConfig, VadError, VoiceActivityDetector, segment_speech};

#[cfg(feature = "onnx")]
pub mod silero_vad;
#[cfg(feature = "onnx")]
pub use silero_vad::SileroVad;

/// Optional pure-Rust earshot VAD. Opt-in via `--features vad-earshot`.
/// Silero remains the production default; see `benchmarks/results/earshot-vad-notes.md`.
#[cfg(feature = "vad-earshot")]
pub mod earshot_vad;
#[cfg(feature = "vad-earshot")]
pub use earshot_vad::{
    ADAPTER_TYPE as EARSHOT_ADAPTER_TYPE, EarshotVad, FRAME_SIZE as EARSHOT_FRAME_SIZE,
};

#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
#[allow(deprecated)] // re-export of legacy API; consumers still warned at use site
pub use onnx::OnnxEmbeddingExtractor;

#[cfg(feature = "onnx")]
pub mod ecapa;

/// Optional NVIDIA Streaming Sortformer v2 E2E diarizer (≤4 speakers).
/// Opt-in via `--features sortformer`. See `docs/sortformer.md`.
#[cfg(feature = "sortformer")]
pub mod sortformer;

// Public re-exports for ergonomic use.
pub use cluster::SpeakerCluster;
pub use der::{DerDecomposition, DerResult, SpeakerRecall, WderResult, compute_der, compute_wder};
// DummyExtractor is the supported test/mock embedder (also bridges to Embedder).
// EmbeddingExtractor / EmbeddingError remain soft-deprecated at the definition site.
#[allow(deprecated)]
pub use embedding::{DummyExtractor, EmbeddingError, EmbeddingExtractor};
#[cfg(feature = "download")]
pub use models::{ModelRegistry, ProfileModels, RegistryError};
pub use overlap::{OverlapRegion, detect_overlaps};
pub use types::ClusterConfig;
pub use types::{
    Confidence, DiarizationConfig, DiarizationResult, Profile, SampleRate, Seconds, Segment,
    SpeakerId, SpeakerIdRemap, SpeakerSummary, SpeakerTurn, TimeRange, Transcript, Word,
    WordAlignment, confidence_from_distance, confidence_from_similarity, exclusive_turns,
    mean_speaker_embeddings, remap_segments, remap_turns, segment_confidences_from_embeddings,
};
pub use window::{WindowBuffer, WindowIter};

#[cfg(feature = "onnx")]
#[allow(deprecated)] // re-export of legacy API; consumers still warned at use site
pub use ecapa::FbankOnnxExtractor;
