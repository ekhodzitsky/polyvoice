#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
// The generated test harness registers every #[test] fn from the crate root,
// so tests living inside a deprecated module trip `deprecated` there; allow
// it crate-wide in test builds only.
#![cfg_attr(test, allow(deprecated))]
#![deny(unsafe_op_in_unsafe_fn)]

//! # polyvoice
//!
//! Speaker diarization library for Rust — online (streaming) and offline
//! (file-based), ecosystem-agnostic. The ONNX path is opt-in (`features =
//! ["onnx", …]`); default features are empty so BYO-embedder consumers can
//! use [`pipeline::LegacyPipeline`] / `StreamingPipeline` / `EnergyVad`
//! without linking `ort`.
//!
//! Designed to be embedded into any Rust application that needs to answer
//! the question **"who spoke when?"**.
//!
//! ## Quick start
//!
//! **ONNX production path:** the crate-root `Pipeline` (re-exported from
//! `pipeline_v2`) + `ModelRegistry`, gated on features `onnx`, `download`,
//! `segmentation`, `embedder`, `clusterer`, `resegmentation` (CLI also
//! enables `vbx`). This is what CLI / FFI / Python / MCP run by default
//! since **0.11** (v2 + VBx). With the gate off there is deliberately no
//! crate-root `Pipeline` — ort-free builds use [`pipeline::LegacyPipeline`].
//!
//! **Library mode (no ONNX):** `default-features = false`, implement
//! [`Embedder`], pair with [`EnergyVad`] and [`pipeline::LegacyPipeline`] /
//! [`streaming::StreamingPipeline`] — see the crate README and
//! `docs/library-mode.md`.
//!
//! ## Module organization
//!
//! Two intentional pipeline families share math and types:
//!
//! - **ONNX production (`pipeline_v2`, crate-root `Pipeline`):** trait-wired
//!   Segmenter → Embedder → Clusterer → Resegmenter. CLI/FFI/Python/MCP
//!   default since 0.11 (VBx when PLDA is available). See
//!   `docs/PIPELINE-ARCHITECTURE.md`.
//! - **BYO / ort-free ([`pipeline::LegacyPipeline`] + `StreamingPipeline`):**
//!   inject [`Embedder`] + [`VoiceActivityDetector`]. CLI `--legacy` uses
//!   this offline path with Silero + AHC.
//! - **Shared math:** `ahc`, `kmeans`, `spectral`, `features`, `der`, `utils`.
//! - **Online centroids:** production streaming uses
//!   `streaming::ArrivalOrderSpeakerCache`; `cluster::SpeakerCluster` is
//!   deprecated (not on any production path).

pub mod ahc;
pub mod asr;
pub use asr::{Asr, AsrError};
/// Online incremental speaker centroids. Kept for the fuzz target and
/// experiments; not on any production path (offline clustering is
/// `clusterer::Clusterer`, streaming uses
/// [`streaming::ArrivalOrderSpeakerCache`]).
#[deprecated(
    since = "0.12.0",
    note = "not on the production offline or streaming path; use clusterer::Clusterer (offline) or streaming::StreamingPipeline / ArrivalOrderSpeakerCache (online)"
)]
pub mod cluster;
pub mod der;
pub mod features;
#[cfg(feature = "ffi")]
pub mod ffi;
/// Kuhn-Munkres assignment solver. Always compiled (pure Rust, wasm32-clean):
/// shared by `der` (optimal speaker mapping for DER and WDER via
/// `map_max_cooccurrence`), `segmentation` (window permutation alignment in
/// the aggregator), and `clusterer::assign` (local-to-global label mapping).
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

pub use embedder::{DummyExtractor, Embedder, EmbedderError, apply_overlap_mask};

#[cfg(all(feature = "onnx", feature = "embedder"))]
pub use embedder::{CamPlusPlusExtractor, ERes2NetV2Extractor, ResNet34Adapter};

#[cfg(feature = "clusterer")]
pub mod clusterer;

/// Deprecated alias for the pre-rename name; use [`KmeansClusterer`].
#[cfg(feature = "clusterer")]
#[allow(deprecated)]
pub use clusterer::KMeansClusterer;
#[cfg(feature = "clusterer")]
pub use clusterer::{
    AhcClusterer, AsNormClusterer, Clusterer, ClustererError, KmeansClusterer,
    MinClusterSizeClusterer,
};

#[cfg(all(feature = "clusterer", feature = "spectral"))]
pub use clusterer::NmeScClusterer;

#[cfg(all(feature = "clusterer", feature = "vbx"))]
pub use clusterer::vbx::{VbxClusterer, VbxClustererConfig};

#[cfg(feature = "resegmentation")]
pub mod resegmentation;

#[cfg(feature = "resegmentation")]
pub use resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentError, ResegmentInputs, Resegmenter,
    SpeakerCentroid, compute_centroids,
};

#[cfg(all(feature = "resegmentation", feature = "segmentation"))]
pub use resegmentation::extract_overlap_time_ranges;

/// Midpoint word→speaker labeling for STT stacks (always-on, no models).
pub mod labeling;
pub use labeling::{
    UncoveredPolicy, assign_speakers_by_midpoint, label_words, speaker_at, speaker_at_stable,
};

#[cfg(feature = "attribution")]
pub mod attribution;
#[cfg(feature = "attribution")]
pub use attribution::{
    AttributionConfig, SpeakerEmbedding, WhoSaidWhat, WordAnchor, attribute_and_fill,
    attribute_and_fill_with_config, attribute_words, attribute_words_with_config, fill_turn_text,
    fill_turn_text_with_config, interpolate_word_timestamps, speaker_embeddings_from_segments,
    who_said_what, who_said_what_with_config,
};

/// BYO / ort-free legacy pipeline (v1). The crate-root `Pipeline` is the
/// production v2 pipeline (below) when its feature gate is on; with default
/// features there is no crate-root `Pipeline` at all.
pub mod pipeline;

#[cfg(all(
    feature = "onnx",
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline_v2;

/// Production ONNX pipeline, re-exported at the crate root under the same
/// feature gate as [`pipeline_v2`]. Deliberately absent when the gate is off:
/// ort-free consumers use [`pipeline::LegacyPipeline`].
#[cfg(all(
    feature = "onnx",
    feature = "download",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub use pipeline_v2::{Pipeline, PipelineConfig, PipelineError};

/// Shared wiring helpers for the CLI-family binaries (`polyvoice`,
/// `polyvoice-bench`, `polyvoice-measure`, `polyvoice-mcp`): flag-to-config
/// translation, pipeline construction, and bench-dataset walking, so each
/// binary stays a thin wrapper. Compiled with `cli` or `mcp` — both imply the
/// full ONNX pipeline stack this module builds on.
///
/// Hidden from docs: not a supported library API (bin wiring only). Kept
/// `pub` (not `pub(crate)`) so field uses from bin targets do not trip
/// `dead_code` when building the lib alone.
#[doc(hidden)]
#[cfg(any(feature = "cli", feature = "mcp"))]
pub mod cli_common;

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
pub mod fbank_onnx;

/// Optional NVIDIA Streaming Sortformer v2 E2E diarizer (≤4 speakers).
/// Opt-in via `--features sortformer`. See `docs/sortformer.md`.
#[cfg(feature = "sortformer")]
pub mod sortformer;

// Public re-exports for ergonomic use.
pub use der::{DerDecomposition, DerResult, SpeakerRecall, WderResult, compute_der, compute_wder};
#[cfg(feature = "download")]
pub use models::{ModelRegistry, ProfileModels, RegistryError};
pub use overlap::OverlapRegion;
pub use types::ClusterConfig;
pub use types::{
    Confidence, ConfigError, DEFAULT_AHC_THRESHOLD, DiarizationConfig, DiarizationResult, Profile,
    SampleRate, Segment, SpeakerId, SpeakerIdRemap, SpeakerSummary, SpeakerTurn, TimeRange,
    Transcript, Word, WordAlignment, confidence_from_distance, confidence_from_similarity,
    exclusive_turns, mean_speaker_embeddings, remap_segments, remap_turns,
    segment_confidences_from_embeddings,
};
pub use window::{WindowBuffer, WindowIter};

#[cfg(feature = "onnx")]
pub use fbank_onnx::FbankOnnxExtractor;
