# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2025-05-05

### Fixed
- CI: macOS and Windows runners now test without `onnx` feature to avoid `ort` platform-specific binary issues.

### Changed
- Rewrote README.md with a product-focused, sales-oriented structure: badges, value proposition, comparison table, architecture diagram, and production-readiness indicators.

## [0.4.1] - 2025-05-05

### Changed
- Removed proprietary project mentions from README.

## [0.4.0] - 2025-05-05

### Breaking Changes
- `SpeakerCluster::merge()` now returns `Option<SpeakerIdRemap>` instead of invalidating `SpeakerId`s silently.
- `compute_fbank()` deprecated in favor of `FbankExtractor::extract()`.
- Removed `compute_fbank` from crate-root re-exports (still available via `polyvoice::features::compute_fbank`).

### Added
- Doc tests for all public APIs (`SampleRate`, `Confidence`, `SpeakerCluster`, `OfflineDiarizer`, `OnlineDiarizer`, `detect_overlaps`, `FbankExtractor`, `segment_speech`, etc.).
- Loom model-checking for session-pool thread safety (`tests/loom_pool.rs`).
- Fuzz targets: `fuzz_compute_fbank`, `fuzz_segment_speech`, `fuzz_detect_overlaps`, `fuzz_cluster_assign`.
- DER (Diarization Error Rate) accuracy benchmark suite (`benches/der.rs`).
- Cross-platform CI: macOS and Windows runners.
- Miri CI job for unsafe memory verification.
- cargo-semver-checks in CI.
- Nightly fuzz workflow (`.github/workflows/fuzz.yml`).
- FFI memory safety tests (`tests/ffi_memory.py`) covering lifecycle, NULL handling, and large-audio stress.
- `examples/` directory: `offline.rs`, `online.rs`, `onnx.rs`, `ffi_usage.c`.
- `docs/API.md` reference guide.
- `include/polyvoice.h` C header.
- `SpeakerIdRemap`, `remap_segments`, `remap_turns` for safe post-merge ID updates.

### Fixed
- `SpeakerCluster::merge()` now preserves `SpeakerId` validity via explicit remap table.
- `detect_overlaps()` filters zero-length and unlabeled segments to prevent phantom overlaps.

## [0.3.0] - 2025-05-05

### Added
- `FbankExtractor` — cached log-mel filterbank extractor that reuses FFT planner, Hamming window, and mel-filterbank matrices across calls.
- `VadConfig` — configurable VAD parameters (`frame_size`, `threshold`, `min_silence_ms`).
- `FbankConfig` exported publicly.
- `max_gap_secs` field in `DiarizationConfig` for configurable gap merging.
- Property-based tests for clustering, overlap detection, and fbank shape invariants.

### Fixed
- **CRITICAL** `ffi.rs`: eliminated UB in `Vec::from_raw_parts` by using `Box::into_raw` + `Box::from_raw` with slice reconstruction.
- **CRITICAL** `ffi.rs`: fixed memory leak when `CString::new` fails during turn construction.
- **CRITICAL** `utils.rs`: replaced `assert_eq!` panic in `cosine_similarity` with graceful fallback + `tracing::warn`.
- **CRITICAL** `online.rs`: fixed `align_words` logic — now stores `SpeakerId` in embedding_buffer and performs correct time-based lookup.
- **CRITICAL** `overlap.rs`: fixed phantom overlap bug when segments do not start at `t=0.0`.
- **MAJOR** `ecapa.rs` / `onnx.rs`: strict exact-match check for embedding dimension (replaces silent truncation).
- **MAJOR** `onnx.rs` / `ecapa.rs`: bounds check for ONNX model outputs before indexing.
- **MAJOR** `features.rs`: `RealFftPlanner` no longer recreated on every `compute_fbank` call when using `FbankExtractor`.

### Changed
- **BREAKING** `DiarizationConfig.sample_rate` is now `SampleRate` (newtype) instead of raw `u32`.
- **BREAKING** `EnergyVad::new` now takes `frame_size: usize`.
- **BREAKING** `segment_speech` now takes `&VadConfig` parameter.
- Bumped `ndarray` to `0.17` for `ort` 2.0.0-rc.12 compatibility.

## [0.2.0] - 2025-05-05

### Added
- `EcapaTdnnExtractor` — ONNX-based ECAPA-TDNN speaker embedding extractor with built-in log-mel filterbank preprocessing (`src/ecapa.rs`, `src/features.rs`).
- `compute_fbank` — pure-Rust 80-bin log-mel filterbank extraction (pre-emphasis, Hamming window, FFT via `realfft`, mel-filterbank, log compression).
- Real-audio benchmark suite (`benches/diarization.rs`) using Criterion: offline diarization latency and ECAPA fbank throughput on synthetic multi-speaker waveforms.
- Hoare triple doc comments on all public API functions per `cargo-kimi` guidelines.
- `// SAFETY` annotations for every `unsafe` block and `unsafe extern "C" fn` in `ffi.rs`.

### Changed
- Bumped `ndarray` to `0.17` for compatibility with `ort` 2.0.0-rc.12.

## [0.1.0] - 2025-05-05

### Added
- Initial release of `polyvoice`.
- `EmbeddingExtractor` trait with `DummyExtractor` (tests) and `OnnxEmbeddingExtractor` (ONNX Runtime, pooled sessions).
- `SpeakerCluster` with online incremental centroid updates and cosine-similarity threshold.
- `OnlineDiarizer` for real-time streaming with sliding windows.
- `OfflineDiarizer` for file-based diarization with segment merging and gap filling.
- `VoiceActivityDetector` trait with `EnergyVad` reference implementation.
- Overlap detection (`detect_overlaps`) from fine-grained segment lists.
- Word-level speaker alignment (`OnlineDiarizer::align_words`).
- Comprehensive unit and integration tests.
