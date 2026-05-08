# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (M0 — v1.0 plumbing)
- `Profile` enum (`Mobile`/`Balanced`/`Custom`) in `polyvoice::types`.
- `polyvoice::models` module: `ModelRegistry`, `Manifest`, `ModelEntry`, `ProfileEntry`, `ProfileModels`.
  Provides manifest-driven, SHA-256-verified, idempotent ONNX model downloads.
- New Cargo features: `download`, `coreml`, `nnapi`, `xnnpack`, `profile-mobile`,
  `profile-balanced`, `profile-all`, and `spectral` (now in default features). The
  `cli` feature now depends on `download` (no behavioral change for existing users).
- CLI: `polyvoice download-models --profile mobile|balanced|all`. Both `mobile`
  and `balanced` resolve to the existing v0.5.x model pair until later milestones
  ship CAM++ (M2) and INT8 versions (M5).
- CI: aarch64-unknown-linux-gnu cross-compile job and wasm32-unknown-unknown smoke
  compile job.
- `scripts/release-gate.sh` — stub release-gate script aligned with §9.10 of the
  v1.0 design.

### Changed
- `faer` is now an optional dependency gated behind the `spectral` Cargo feature
  (in `default`). Existing users see no behavior change; downstream consumers
  building with `--no-default-features` no longer pull `faer` (enables wasm32 builds).

### Added (M1 — Powerset segmentation)
- `polyvoice::segmentation` module: `Segmenter` trait, `RawSegment`, `SegmentationError`,
  `PowersetSegmenter` (ONNX-backed), `PowersetDecoder`, `PowersetClass`, `FrameLabel`,
  `Aggregator`, `WindowOutput`, `AggregationConfig`.
- New Cargo feature `segmentation` (in default features). The pure-Rust algorithmic
  core (decoder, aggregator, hungarian) is wasm32-clean; only `PowersetSegmenter`
  additionally requires `onnx`.
- In-tree Kuhn-Munkres minimum-cost assignment (~50 LOC) for sliding-window speaker
  index alignment — no external dependency added.
- New manifest entry `[models.powerset_fp32]` for sherpa-onnx-pyannote-segmentation-3-0.
  Profiles still resolve to the legacy `silero_vad` segmenter; M6 swaps them.

### Added (M2 — Embedder trait + CAM++)
- `polyvoice::embedder` module: `Embedder` trait, `EmbedderError`, `EmbedderPool`,
  `apply_overlap_mask` helper.
- `CamPlusPlusExtractor` (192-d output, gated `onnx`+`embedder`) — wraps the
  same fbank pipeline as ResNet34 with the CAM++ ONNX model.
- `ResNet34Adapter` — bridges existing `FbankOnnxExtractor` (256-d, WeSpeaker)
  to the new `Embedder` trait. Legacy `EmbeddingExtractor` trait is unchanged.
- New Cargo feature `embedder` (in default features). Pure-Rust core (trait,
  `apply_overlap_mask`, `EmbedderPool` over a generic `E: Embedder`) is
  wasm32-clean; `CamPlusPlusExtractor` and `ResNet34Adapter` additionally
  require `onnx`.
- New manifest entry `[models.cam_pp_fp32]`. Profiles still resolve to
  `wespeaker_resnet34` until M6 swaps them.

### Added (M3 — Clusterer trait + NME-SC)
- `polyvoice::clusterer` module: `Clusterer` trait, `ClustererError`,
  `AhcClusterer` (wraps legacy `agglomerative_cluster_auto`), `NmeScClusterer`
  (NME-SC with eigengap auto-K, gated `spectral`+`clusterer`).
- New Cargo feature `clusterer` (in default features). The AHC adapter is
  wasm32-clean; NME-SC additionally requires the `spectral` feature.
- Integration test on synthetic 4-cluster data (no model required) — runs in
  every PR's normal `cargo test` (not `--ignored`).

### Added (M4 — Overlap resegmenter)
- `polyvoice::resegmentation` module: `Resegmenter` trait, `ResegmentError`,
  `OverlapResegmenter` (pure-Rust post-clustering pass that attaches a second
  speaker to overlap regions via nearest-cosine cluster), `ResegmentInputs`,
  `OverlapRegionInput`, `SpeakerCentroid`, helpers `compute_centroids` and
  `extract_overlap_time_ranges` (gated `segmentation`).
- New Cargo feature `resegmentation` (in default features). Pure-Rust core,
  wasm32-clean, no `onnx` requirement.
- Integration test on synthetic two-speaker / three-speaker data + RTTM
  round-trip — runs in every PR's normal `cargo test`.
- Miri-friendly test target `tests/miri_resegmentation.rs` covering
  no-overlap, single-overlap, and centroid math paths.

### Added (M6a — Pipeline + Profile API, additive)
- `polyvoice::pipeline_v2` module: `Pipeline::builder()` returning
  `PipelineBuilder` with `.profile(Profile::Mobile|Balanced|Custom)`,
  `.with_models_from(ModelRegistry)`, `.with_segmenter/embedder/clusterer/resegmenter()`,
  and a validated `.build()`. `PipelineConfig`, `ClustererKind`,
  `ExecutionProvider`, and `ConfigError` all per spec §5.2/§5.4.
- `Pipeline::run(&samples, SampleRate)` orchestrates M1 segmenter → M2
  embedder (with `apply_overlap_mask` per primary chunk) → M3 clusterer →
  M4 resegmenter → legacy `merge_segments` → `DiarizationResult`.
- New Cargo feature `pipeline_v2` (in default features). Requires
  `onnx + segmentation + embedder + clusterer + resegmentation`; missing
  any of these triggers a `compile_error!` with an actionable message.
- Public re-exports `polyvoice::PipelineV2`, `PipelineBuilder`,
  `PipelineConfig`, `ClustererKind`, `ExecutionProvider`, `ConfigError`,
  `PipelineV2Error`. Legacy `polyvoice::Pipeline` is unchanged; M6b will
  rename `pipeline_v2 → pipeline` and remove the legacy code path.
- Synthetic integration test on Custom profile (`tests/pipeline_v2_synthetic_test.rs`,
  7 tests) + `#[ignore]` E2E test on Balanced profile via `ModelRegistry`
  (`tests/pipeline_v2_e2e_test.rs`).

### Changed (M6b — Legacy cleanup + CLI/FFI/Python migration)
- **BREAKING**: removed legacy `Pipeline::new(DiarizationConfig, VadConfig)`,
  `OfflineDiarizer`, `DiarizationConfig`, `VadConfig`, `VoiceActivityDetector`,
  `EnergyVad`, `segment_speech`, `DummyExtractor`, `OnnxEmbeddingExtractor`,
  `EcapaTdnnExtractor`, `EcapaMelOnnxExtractor`, `RawAudioOnnxExtractor`,
  `ClusteringBackend`, `EmbeddingDim`. `compute_fbank` is now private.
- Renamed `pipeline_v2 → pipeline`. The Cargo feature is `pipeline`
  (default-on, requires `download + onnx + segmentation + embedder + clusterer + resegmentation`).
  Public surface: `polyvoice::Pipeline::builder()` is the only Pipeline API.
- CLI rewritten: `polyvoice diarize <wav> --profile mobile|balanced` replaces
  the legacy threshold-based interface. New: `polyvoice models list/info`.
- `polyvoice-bench` rewritten on `Pipeline::builder()`. JSON output schema
  `polyvoice-bench-v1`.
- C FFI ABI v2 (`polyvoice_pipeline_*` family) replaces the legacy
  `polyvoice_diarizer_*` ABI. See `include/polyvoice.h`.
- Python pyo3 bindings rewritten: `polyvoice.Pipeline.mobile()` /
  `Pipeline.balanced()` / `Pipeline.run(samples, sample_rate)`.

### Added (M6b)
- `docs/MIGRATING-FROM-0.5.md`: migration guide for Rust / Python / CLI / C FFI.
- `tests/der_baseline.json`: schema for the v1.0 DER baseline. Numbers are
  deferred to an operational follow-up after M5 INT8 publish closes.
- `scripts/run-der-baseline.sh`: helper that runs `polyvoice-bench` on
  VoxConverse-test and prints the values to paste into the baseline JSON.

### Deprecated
- `polyvoice::OnlineDiarizer` — streaming redesign coming in v1.1; use
  `Pipeline` for offline.

### Added (M5 — INT8 quantization)
- New scripts: `download-voxconverse-dev.sh`, `download-voxceleb1-subset.sh`,
  `quantize_models.py` + `quantize-models.sh`, `validate_int8.py` +
  `validate-int8.sh`. Together they drive `onnxruntime.quantize_static`
  (per-channel INT8 weights, asymmetric activations, MinMax calibration on
  500 random VoxConverse-dev WAVs, seed 42), then validate FP32 → INT8 deltas
  against spec §9.4 budgets.
- Manifest pinned with `[models.powerset_int8]`, `[models.cam_pp_int8]`,
  `[models.resnet34_int8]`. Provisional sha256 / size taken from the M5
  preview run (see `docs/strategy/m5-quantization-notes.md`); will be
  regenerated when the full dev calibration sweep finishes via
  `scripts/publish-models.sh`. `[profiles.mobile]` switched to
  `powerset_int8` + `cam_pp_int8`; `[profiles.balanced]` switched to
  `powerset_int8` + `resnet34_int8`. Legacy FP32 entries retained for direct
  `ModelRegistry::ensure()` callers.
- `tests/m5_manifest_smoke_test.rs` (7 tests) enforces presence of INT8
  entries, real (non-placeholder) sha256, profile resolution, and bundle
  size budgets (Mobile ≤ 15 MB relaxed from 10 MB target — see notes,
  Balanced ≤ 35 MB).
- `scripts/release-gate.sh` checks the live manifest values via `awk`:
  Mobile bundle row passes at the 15 MB ceiling; DER thresholds remain
  PENDING (real in M6 once Pipeline wires INT8 + e2e DER).
- Engineering notes (`docs/strategy/m5-quantization-notes.md`) document
  the SincNet rank-1 weight finding (powerset compression ratio ~1.04×),
  the relaxed Mobile bundle ceiling, and the VoxCeleb1 audio mirror
  outage (EER falls back to cosine vs FP32 on VoxConverse-dev hold-out).

## [0.5.2] - 2025-05-05

### Added
- PyPI package published: `pip install polyvoice` (macOS ARM64, Linux x86_64, Windows x86_64 wheels).
- GitHub Release with prebuilt CLI binaries for Linux, macOS ARM64, Windows.

### Fixed
- Clippy 1.95.0: `unnecessary_sort_by` fixed in `src/der.rs` and `benches/der_ami.rs`.
- Miri CI: skip FFT-heavy `test_fbank_shape` under Miri; run only Miri-friendly test targets.
- Python CI: replace `maturin develop` with `maturin build --release` + `pip install` for cross-platform stability.
- Release workflow: added missing Rust toolchain to `python-wheels` job; removed slow `macos-13` runner.
- README: PyPI badge, honest Python install instructions, MSRV and Contributing links.

## [0.5.1] - 2025-05-05

### Fixed
- `FbankOnnxExtractor` re-export now correctly guarded by `#[cfg(feature = "onnx")]` (fixes `cargo publish` verification failure).
- Applied `cargo fmt` across entire codebase.
- Fixed all clippy warnings (`collapsible_if`, `needless_range_loop`, `unnecessary_map_or`).

## [0.5.0] - 2025-05-05

### Added
- DER (Diarization Error Rate) computation with collar support and optimal speaker mapping.
- RTTM parser (`src/rttm.rs`) for ground-truth evaluation.
- Agglomerative hierarchical clustering (`src/ahc.rs`) for offline re-clustering.
- Silero VAD integration (`src/silero_vad.rs`) via ONNX.
- `Pipeline` struct for end-to-end diarization (VAD → embed → cluster → turns).
- `FbankOnnxExtractor` — unified fbank + ONNX extractor.
- CLI binaries: `polyvoice` (main CLI) and `polyvoice-bench` (DER benchmark on datasets).
- WAV reader with stereo downmix (`src/wav.rs`).
- CMVN (Cepstral Mean and Variance Normalization) in fbank pipeline.
- VoxConverse test set download scripts.

## [0.4.3] - 2025-05-05

### Fixed
- `cargo fmt` applied across the entire codebase (CI `fmt` job was failing).

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
