# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- **Release builds now require a manifest signature for every profile-resolved
  model.** `ModelEntry.signature` was optional and only verified when present,
  so a tampered/forked manifest that simply dropped the signature silently
  downgraded authenticity to a self-consistent hash. Profile resolution
  (`ensure_for_profile`) now fails fast with the new
  `RegistryError::UnsignedModel` — before any network access — when a resolved
  model has no signature and the build is a release build
  (`cfg!(not(debug_assertions))`). All seven bundled models are signed, so
  existing flows are unaffected; ad-hoc single-model `ensure` stays lenient for
  dev/test. **Breaking:** `RegistryError` gained a variant (exhaustive matches
  must add an arm) — hence the version bump to 0.10.0.

### Added

- **No-collar DER is now release-gated.** The DER regression tests (legacy and
  pipeline-v2) score every run at collar 0 in addition to collar 0.25 and fail
  when the no-collar DER exceeds its committed baseline + tolerance; on the
  VoxConverse 10-file subset the no-collar gate uses a duration-weighted
  micro average (sum of error frames / sum of reference frames — the
  like-for-like convention of pyannote/speakrs headline numbers) instead of a
  mean of per-file ratios. Previously only collar-0.25 was asserted and the
  headline metric could regress silently through a release.
- No-collar baselines measured and committed for the previously null entries
  (legacy e2e smoke 14.52%, v2 e2e smoke 11.81%, VoxConverse-10 micro 27.08%),
  and the long-form overlap-excluded DER floor on AMI EN2002a is now active for
  pipeline v2 (24.80% + 3.0 tolerance) — it was a dormant placeholder gate.
- `src/der/README.md` documents the baseline/gate layout and the refresh
  procedure.
- C FFI: `polyvoice_pipeline_run_format(..., format, ...)` renders the result as
  JSON, RTTM, SRT, VTT, or TXT (new `PolyvoiceFormat` / `polyvoice_format_t`
  selector; RTTM uses the fixed file id `audio`). The existing
  `polyvoice_pipeline_run` ABI is unchanged.
- Python: typed `DiarizationResult` with `.to_json()` / `.to_rttm()` /
  `.to_srt()` / `.to_vtt()` / `.to_txt()` / `.to_dict()` projections and a
  `DiarizationResult.from_json()` constructor; `Pipeline.run_result()` returns
  it. Type stubs (`.pyi`) and a `py.typed` marker ship in the wheel.
- `examples/agent_quickstart.md` — machine-readable integration paths (CLI
  `--json`, MCP, Python typed result, FFI formats, who-said-what).

### Changed

- Legacy DER baselines tightened after the singleton-pruning improvements:
  AMI EN2002a 36.30 → 34.62 (collar 0.25) / 44.73 → 42.90 (no collar);
  VoxConverse-10 macro 17.43 → 15.82 (collar 0.25). The old VoxConverse-10
  no-collar value (25.99) was a macro-era number and is replaced by the micro
  27.08 baseline. Retired hybrid-pipeline baseline entries are marked as
  historical/ungated instead of silently keeping `null` no-collar fields.
- WebVTT output now uses the standard voice span `<v SPEAKER_NN>text</v>` for
  transcribed cues (previously `SPEAKER_NN: text`); diarization-only cues still
  render the bare `SPEAKER_NN` label.

## [0.9.0] - 2026-06-23

Accuracy release for pipeline v2. Overlap reconstruction now reuses the
segmenter's own two-speaker assignment, the VBx (PLDA + VB-HMM) clusterer is a
first-class opt-in that wins on overlap-heavy / meeting audio, and an opt-in
dense-embedding mode lowers v2 confusion on clean audio. The shipped default
(legacy pipeline) is unchanged. Source-breaking additions to `OverlapRegionInput`
and `PipelineConfig` (struct-literal constructors must add the new fields), all
flagged by `cargo-semver-checks` — hence the minor bump.

### Changed

- **Overlap reconstruction now trusts the segmenter's own speaker assignment.**
  The powerset segmenter already identifies both speakers in an overlap region
  (its file-consistent local indices are mapped to global clusters via a
  duration-weighted majority over the primary segments), so pipeline v2 emits
  both speakers directly instead of re-deriving the second one from a degraded
  mixed-voice embedding — and it now covers the *primary* speaker across the
  overlap span too (the aggregator splits the primary's run at the overlap
  boundary, so it was previously dropped there). On AMI EN2002a (~79% overlap):
  total DER 42.47% → 41.37%, overlap-region DER 74.25% → 71.27%, confusion
  18.3% → 17.3%, with fewer spurious speakers. On low-overlap VoxConverse the
  total DER is unchanged while overlap-region DER drops ~14pp. The legacy
  mixed-embedding path remains as a fallback for any overlap whose local speaker
  never appeared as a solo segment, and is skipped entirely otherwise (one fewer
  embedding inference per resolved overlap region).

### Added

- `OverlapRegionInput::secondary_speaker: Option<SpeakerId>` — when `Some`, the
  resegmenter emits that speaker directly and ignores the embedding. **Breaking**
  for code that constructs `OverlapRegionInput` with a struct literal: add
  `secondary_speaker: None` to keep the prior mixed-embedding behavior.
- `POLYVOICE_V2_DISABLE_SEG_OVERLAP` env toggle — forces the legacy
  mixed-embedding overlap path, for A/B ablation of the two strategies.
- **VBx clusterer is now selectable for pipeline v2 without env hacking.**
  `PipelineConfig.vbx_plda_dir: Option<PathBuf>` points the `vbx` clusterer at a
  precomputed PLDA directory (falls back to `POLYVOICE_VBX_PLDA_DIR`), and the
  CLI gains `polyvoice diarize --v2 --clusterer vbx --vbx-plda-dir <dir>`.
  `VbxClusterer::from_dir` is the new explicit constructor (`from_env` now
  delegates to it). VBx (PLDA + VB-HMM) beats cosine AHC on v2 — measured
  VoxConverse-30 −1.3pp, AMI −5pp, lower confusion — and makes v2+VBx the best
  option for overlap-heavy / meeting audio (AMI −7.8pp vs the legacy default).
  It is ~2–3× slower than AHC and needs the PLDA params, so it stays opt-in;
  legacy remains the default for clean conversational audio. The PLDA params are
  pyannote-derived, **CC-BY-4.0** (see `NOTICE`); shipping them bundled (model
  registry) is tracked in `docs/vbx-plda-release.md`. **Breaking:**
  `PipelineConfig` gained a field — struct-literal constructors must add
  `vbx_plda_dir: None` (or use `..PipelineConfig::default()`).
- **Dense embedding for pipeline v2** — `PipelineConfig.embed_window_secs:
  Option<f32>` and CLI `polyvoice diarize --v2 --embed-window <secs>`. `Some(w)`
  slides a `w`-second window (hop `w/2`) inside each segment so a speaker run
  yields several embeddings (like the legacy pipeline's dense windows), for
  more robust centroids and lower confusion. Pure compute, no extra weights.
  `w = 1.5` is the sweet spot (shorter over-fragments — `1.0` is worse). Measured
  on v2+AHC: VoxConverse-30 16.4% → 14.7% (confusion 12.1 → 9.7), and it beats
  legacy on overlap-heavy AMI. It does not overtake legacy on clean VoxConverse
  (13.5%) and slightly hurts the VBx clusterer on AMI, so it is opt-in (`None`
  default = one embedding per segment, unchanged). Another struct-literal field
  addition on `PipelineConfig` (add `embed_window_secs: None`).

## [0.8.0] - 2026-06-20

The "who spoke when" → "who said what" release. It adds an attribution cascade
that puts text on every turn, an MCP server, a canonical serializable result
type with a published JSON Schema, five output formats, an opt-in VBx clusterer,
and a long-form pipeline-v2 segmentation correctness fix. It is a large feature
release: the CLI command surface and output formats were reworked and the
canonical `DiarizationResult` type is new — pin exactly if you script against the
0.7 CLI or depend on the prior result shape.

### Added

- **Who-said-what attribution** (`attribution` feature, pure-Rust, wasm-clean):
  a word→speaker join mapping ASR words onto diarization turns, plus an
  end-to-end cascade (diarize → one ASR pass → join) so each turn carries text.
- **ASR companion** — `polyvoice-asr`, an opt-in workspace crate wrapping
  Parakeet TDT word-level transcription via `parakeet-rs`, sharing the core ONNX
  runtime (one `ort` pin, never two). The trait-only `Asr` interface plus
  `Word`/`Transcript` types live in the core crate; the heavy model dependency is
  isolated in the companion.
- **`polyvoice-transcribe` CLI** — who-said-what from the command line
  (diarization + transcription joined into speaker-attributed text).
- **MCP stdio server** (`polyvoice-mcp`, opt-in `mcp` feature) — a pure-Rust
  Model Context Protocol front door (rmcp + schemars) exposing a `diarize` tool
  to agents. Never in default features; it is a binary in this crate, so its
  `ort` is the same dependency as the core — no dual-runtime risk.
- **Canonical `DiarizationResult` v1** — a stable, serializable result type
  (`schema_version`, `audio`, `provenance`, `speakers[]`, `segments`, `turns`,
  `num_speakers`) designed as the agent / IPC contract.
- **Published JSON Schema** for the result
  (`schema/diarization-result-v1.json`, shipped in the crate) plus a
  `polyvoice schema` command that prints it.
- **SRT / VTT / TXT projectors** for diarization results, alongside RTTM and JSON.
- **CLI rewrite** — one-liner `polyvoice diarize <wav>`, five output formats, a
  `--json` mode, and generated shell completions.
- **VBx clusterer** (opt-in `vbx` feature) — Variational-Bayes HMM + PLDA with
  automatic speaker-count selection, in pure `ndarray` (wasm-clean: the PLDA
  diagonalization is precomputed offline, so no linear-algebra backend is needed
  at runtime). Includes a canonical forward-backward (HMM) path and a
  reproducible PLDA-precompute script with provenance and a parity check.
- **Opt-in cluster pruning** to curb speaker over-clustering: singleton-cluster
  pruning and a length-invariant duration-based prune.
- **UEM-scoped DER**: `compute_der_with_uem` + `parse_uem` (frames outside the
  scored regions are dropped from both the speaker mapping and the error counts;
  no UEM == `compute_der`), a `polyvoice-bench --uem` flag with a model-integrity
  gate (hard-fails if the on-disk embedder/VAD sha256 disagrees with the
  manifest), and `scripts/run-der-sweep.sh` for a reproducible
  VoxConverse-dev/test + AMI sweep (no-collar and 0.25 s-collar from one report).
- **Cross-engine diarization benchmark** (`benchmarks/`): a single-scorer DER
  harness (`der.py`: NIST md-eval frame model, collar + overlap + Hungarian
  mapping, miss/fa/confusion decomposition, speaker-count accuracy, bootstrap CI;
  cross-checked against `polyvoice-bench` to ~0.02 pp) with skip-if-absent runners
  for pyannote / WhisperX / sherpa-onnx / diart, plus an honest, collar-disclosed
  comparison in `docs/BENCHMARKS.md`.

### Changed

- Real VoxConverse-test + AMI DER baselines locked on the shipped FP32 models.
- README streamlined and contributing conventions added; internal tracker
  indices (roadmap task / finding / audit IDs) dropped from shipped code comments
  and docs in favor of self-contained wording.

### Fixed

- **pipeline v2 default min_cluster_size**: lowered from 12 to 1. After the
  segmentation fix above, the default min-cluster pruning dissolved every cluster
  on short clips (a 26 s clip has all clusters below 12 members) into one speaker
  — the v2 e2e DER collapsed to ~49% with 1 hypothesized speaker. v2 ships
  unpruned (pruning is net-negative for the powerset pipeline); the e2e DER is
  back to 4.34%. Callers can still set a higher `min_cluster_size` per pipeline.
- **pipeline v2 segmentation**: the sliding-window aggregator flagged a whole
  speaker run `is_overlap` when any single frame fell in an overlap class, so
  long single-speaker speech brushed by a brief overlap was excluded from the
  primary embedding set and never recovered — on conversational audio this
  dropped most speech (one 336 s file measured ~90% miss). Runs are now split at
  every overlap-status transition, so `is_overlap` is precise (a segment is
  overlap iff every frame was) and two simultaneously-active speakers emit
  time-equal overlap segments that `extract_overlap_time_ranges` pairs as
  designed; overlap-light audio is emitted identically (no regression). Pipeline
  v2 DER on VoxConverse-dev-80 drops from ~33% to 7.75% macro / 8.46% micro —
  below the legacy default (7.97% / 8.58%) — at `min_cluster_size = 1`.

## [0.7.0] - 2026-06-14

Audit-driven correctness, packaging, and CI-hardening release. Bumped to a new
minor (major in 0.x terms) because of two source-breaking API additions, both
flagged by `cargo-semver-checks`:

- `DerResult` gained four public frame-count fields (struct-literal construction
  must be updated).
- `DownloadError` gained the `InsecureScheme` and `TooLarge` variants (exhaustive
  match arms must be updated).

### Added

- **DER overlap-aware decomposition**: `compute_der_decomposition` returns a
  `DerDecomposition` (headline / single-speaker-region / overlap-region DER plus
  per-speaker recall via `SpeakerRecall`), and `compute_der_single_speaker_regions`
  exposes the overlap-excluded DER — a numeric long-form floor that discriminates
  healthy vs collapsed diarization where total DER cannot.
- **`DerResult` raw frame counts** (`total_ref_frames`, `missed_frames`,
  `false_alarm_frames`, `confusion_frames`) for correct duration-weighted
  (micro) averaging across files.
- **`polyvoice-bench`**: collar and no-collar DER at both macro and micro
  averaging, plus overlap-region DER and per-speaker recall in the per-file
  artifacts (schema `polyvoice-bench-v0.8`).
- **Model download hardening**: HTTPS-scheme enforcement and a streaming size
  cap (`DownloadError::InsecureScheme`, `DownloadError::TooLarge`, 1 GiB default).
- **CI/supply chain**: committed `Cargo.lock` with `--locked` everywhere,
  `cargo deny check advisories licenses bans sources`, an advisory
  contract-drift nudge, and a release publish gated on `cargo audit` + `cargo deny`.

### Changed

- **Optimal DER speaker mapping**: `compute_der` now uses Kuhn-Munkres
  (Hungarian) assignment instead of a greedy match, matching `pyannote.metrics`
  semantics — this lowers confusion/DER on cross-talk and fragmented files
  (reported DER figures may shift downward).
- **Unified spectral eigengap**: `spectral_cluster` and `NmeScClusterer` now
  share a single Park et al. NME-SC normalized-maximum-eigengap implementation.
- **Deterministic AHC labels**: cluster ids are assigned canonically (descending
  size, tie-break by smallest member index).
- **`merge_segments`** confidence is the arithmetic mean of the merged segments.
- **Packaging**: switched to an `include` allowlist (ship only build-required
  files), docs.rs builds the full feature surface, refreshed crates.io keywords.
- **Docs honesty**: collar-disclosed, provenance-stamped DER numbers and revised
  positioning across `README.md` and `PRODUCTION-READINESS.md`.

### Fixed

- **k-means++** degenerate-seeding guard on collinear/zero embeddings.
- **streaming**: `StreamingPipeline::turns()` accumulates emitted turns across
  `feed()`/`flush()` instead of dropping them.
- **release gate** hard-fails on missing required DER data instead of silently
  passing (`POLYVOICE_REQUIRE_DATA`).

### Deprecated

- Legacy embedding API — `EmbeddingExtractor`, `DummyExtractor`, `EmbeddingError`,
  `OnnxEmbeddingExtractor`, and `FbankOnnxExtractor` — migrate to the v1.0
  `Embedder` trait in `polyvoice::embedder`.

### Removed

- The `VERSION` file (Cargo.toml is the single source of version truth).
- Tracked `models/*.onnx.data` weight blobs (now git-ignored).

## [0.6.9] - 2026-06-13

### Fixed

- **segmentation/aggregator**: cumulative speaker permutation was double-applied
  across window boundaries in `stitch`. `window_permutation` already maps onto the
  global frame via `a_perm_so_far`, so the extra `prev[perm[...]]` composition was
  removed. Added a 3-window regression test guarding global speaker-index
  consistency.
- **pipeline**: `run_from_wav` now validates the WAV sample rate against
  `config.window.sample_rate` and returns `PipelineError::UnsupportedSampleRate`
  instead of silently discarding the file's rate.
- **embedder**: `EmbedderPool::new` now returns `Result` and rejects embedders
  with differing output dimensions via `EmbedderError::DimMismatch`. Empty-pool
  behavior is unchanged.
- **types**: `SpeakerIdRemap::from_mapping` now performs a runtime check for
  duplicate old `SpeakerId`s and returns `Option<Self>` (`None` on duplicates)
  instead of relying on `debug_assert!`.
- **clusterer**: `AhcClusterer`, `KMeansClusterer`, and `NmeScClusterer` now
  validate embedding dimension uniformity up front and return
  `ClustererError::DimMismatch` on mismatch.

## [0.6.8] - 2026-06-01

### Fixed

- **pipeline_v2 NaN-embedding collapse**: segments shorter than ~0.11s made
  WeSpeaker ResNet34's statistics-pooling emit NaN embeddings, which slipped past
  the `< 1e-8` guards in `l2_normalize`/`cosine_similarity` and poisoned AHC
  clustering (AMI `EN2002a` collapsed to 1 speaker). Sub-0.20s segments and any
  non-finite embedding are now skipped before clustering, in both the primary and
  overlap embed loops.
- `utils::l2_normalize`, `cosine_similarity`, and `cosine_similarity_f32_f64` now
  guarantee finite output for any (possibly non-finite) input — a latent bug that
  affected every caller, not just Pipeline v2.

### Added

- NaN-safety unit tests for the shared math utils and deterministic mock tests
  asserting Pipeline v2 skips short/non-finite-embedding segments.
- CI: scheduled `pipeline-v2-ami` DER regression entry. The AMI v2 baseline test
  now gates on speaker count (`>= 2`) and clustering confusion (`< 25%`) rather
  than total DER, which on ~79%-overlap AMI is miss-bound and cannot detect the
  collapse.

## [0.6.7] - 2026-05-26

### Changed

- **CI SOTA overhaul**: release gate, parallel DER matrix, cargo-deny, caching, permissions
- **Testing**: cargo-nextest with retries, insta snapshot tests, Kani formal proofs
- **Quality**: unwrap_used = "deny", cargo-hack feature powerset (120 combos), cargo-machete
- **Infrastructure**: MSRV CI job (1.85.0), cargo-mutants nightly, fuzz on PR

## [0.6.6] - 2026-05-25

### Changed

- **Python and FFI bindings migrated to Pipeline v2** (`polyvoice::pipeline_v2`).
  `Pipeline::balanced()` / `.mobile()` now build via `PipelineBuilder`.
- Python `run()` releases the GIL during inference.
- Python exception mapping improved: `ValueError` for bad sample rate, `OSError`
  for model/registry errors.

### Added

- Property test: pipeline output turns are always monotonically ordered.
- CI: Valgrind job for FFI memory safety.
- CI: Miri job restricted to focused targets (3 min instead of 6+ hours).

### Fixed

- FFI: eliminated UB on invalid C enum discriminant (`profile` now `c_int`).
- FFI: `ConfigError` variants correctly map to `InvalidArg(1)` instead of
  `Internal(99)`.
- FFI: path-traversal guard strengthened (`starts_with('/')`).
- `ecapa` test now skips under `feature = "onnx"` to avoid failure with
  `--all-features`.
- VAD property test now uses generated `min_speech_frames`.
- Mock E2E tests deduplicated and misleading names corrected.

## [0.6.5] - 2026-05-21

### Fixed

- `e2e_smoke_test.rs`: legacy v0.5 pipeline now correctly loads Silero VAD model
  (`registry.ensure("silero_vad")`) instead of passing the powerset segmenter path
  to `SileroVad::new()`.
- CI: removed broken `coad-check` job (coad-validator no longer exists).
- CI: replaced `cargo-tarpaulin` with `cargo-llvm-cov` in coverage job for stability.
- Docs: removed obsolete `coad check .` and `uvx --from 'git+...'` references from
  `AGENTS.md`, `AGENT_FLOW.md`, and `COAD_PROJECT_STANDARD.md`.

## [0.6.4] - 2026-05-21

### Added

- **K-means auto-k clusterer** (`clusterer::KMeansClusterer`) with automatic k
  selection via silhouette score. 3 trials per k, full silhouette on cached
  pairwise cosine distances. Single-speaker detection via embedding homogeneity
  check (mean pairwise distance < 0.15 → force k=1).
  - VoxConverse 10-file: **13.48%** DER (vs AHC 15.03%).
  - VoxConverse full 232-file: **14.12%** DER (vs AHC 18.77%).
- `KMeansClusterer::fast_mode()` — adaptive k_max, 1 trial, 20 iterations for
  ~10× speedup with minor quality trade-off (16.34% on 10-file).
- `KMeansClusterer::with_max_iter()` / `with_trials()` builder methods.
- CoreML Execution Provider support on macOS ARM64 (`--features coreml`).
- `examples/hybrid_benchmark.rs` — standalone benchmark with per-file timing and
  JSON checkpoint/resume.
- `hybrid_voxconverse_10_file_subset_kmeans_fast` integration test.

### Changed

- **PowersetSegmenter default hop: 0.5s → 1.0s** — 2× fewer windows, ~2× faster
  inference with slightly better DER (less segmentation noise). No API change;
  existing `PowersetConfig::default()` users automatically get the new hop.
- Full VoxConverse regression assertion tightened: 20% → **16%** DER ceiling.

## [0.6.3] - 2026-05-19

### Added

- **Hybrid pipeline** (`pipeline_v2::hybrid::HybridPipeline`) — API-only.
  Combines `PowersetSegmenter` (used purely as a VAD for speech+overlap
  detection) with legacy-style sliding-window ResNet34 embeddings and AHC
  clustering. Overcomes the 3-speaker hard limit of the Powerset model on
  long-form multi-speaker audio.
  - e2e_smoke DER: **4.43%** (vs legacy 6.62%, v2 4.79%).
  - VoxConverse 3-file average DER: **8.27%**.
- `Embedder::embed_batch` parallel implementation via `std::thread::scope`
  for `ResNet34Adapter` and `CamPlusPlusExtractor`.
- `agglomerative_cluster_max_clusters()` — fixed-threshold AHC with a hard
  ceiling on cluster count.

### Changed

- **AHC optimized from O(n³) to O(n²)** by caching the similarity matrix
  and updating only the merged centroid row/column. Massive speedup on
  inputs with >500 embeddings (e.g. long-form audio).
- `FbankOnnxExtractor` now sets `intra_op_num_threads(1)` per ONNX session
  to prevent parallel embedder threads from competing for CPU cores.
- `HybridPipeline` uses `embed_batch` and 1.5s hop windows for faster
  extraction on long recordings.
- `der_baseline.json` updated:
  - `v2_e2e_smoke`: 4.43% (improved from 4.79%).
  - Added `hybrid_e2e_smoke`: 4.43%.
  - Added `hybrid_voxconverse_3file`: 8.27%.
  - Added `hybrid_voxconverse_10file`: 16.62% (aorju is a known outlier at 52.51%).
- Zero-padding fix for partial chunks in `HybridPipeline::run` — prevents NaN
  embeddings on short trailing windows.

### Fixed

- `der_regression_test.rs` now loads `silero_vad.onnx` via explicit path
  instead of `models.segmenter_path`, which changed to `powerset_fp32.onnx`
  after the manifest update in v0.6.2.

## [0.6.2] - 2026-05-18

### Fixed

- **CLI `diarize` reverted to legacy pipeline** (v0.5). Pipeline v2
  (PowersetSegmenter → ResNet34 → AHC) achieves excellent DER on short clips
  (4.79% on e2e-smoke) but degrades on long recordings (VoxConverse ~30%,
  AMI ~80%). Legacy pipeline remains the stable default until v2 is hardened
  for long-form audio.
- `AhcClusterer` now respects `max_clusters` ceiling via
  `agglomerative_cluster_auto_max_clusters`.
- `AhcClusterer::with_threshold` added for fixed-threshold mode (legacy
  behaviour) alongside auto-threshold mode.
- `FbankOnnxExtractor` zero-pads short inputs below the fbank window length,
  preventing crashes on sub-25 ms segments from PowersetSegmenter.

### Changed

- `der_baseline.json` updated with `v2_e2e_smoke` entry (4.79% DER).

## [0.6.1] - 2026-05-18

### Summary

Pipeline v2 is now the default for CLI `diarize` command. ResNet34 + AHC
replaces broken CAM++ + NME-SC as the default embedder/clusterer pair,
fixing the DER regression and improving accuracy over the legacy pipeline.

### Changed

- `pipeline_v2` module is no longer gated behind a Cargo feature; it is
  available whenever `onnx + segmentation + embedder + clusterer + resegmentation`
  features are enabled (which the `cli` feature now includes).
- CLI `diarize` command now uses Pipeline v2 for `mobile` and `balanced`
  profiles instead of the legacy v0.5 pipeline.
- Default clusterer in Pipeline v2 changed from NME-SC to AHC.
- Default embedder in `Mobile` profile changed from CAM++ to ResNet34
  (CAM++ ONNX model produces near-identical embeddings regardless of input).
- Model manifest updated: both `mobile` and `balanced` profiles now use
  `powerset_fp32` segmenter + `wespeaker_resnet34` embedder.

### Fixed

- DER regression in Pipeline v2: achieved **4.79%** on e2e-smoke test
  (vs legacy 6.62%, vs broken v2 with CAM++ 49.91%).
- NME-SC fallback to AHC for small-n inputs (n < 8) to avoid collapse
  to a single cluster.

## [0.6.0] - 2026-05-18

### Summary

Stable release combining the v1.0 architecture milestones (M0–M6) with the proven
v0.5.2 legacy pipeline as the default execution path.

### Added

- Model registry with SHA-256 verified ONNX downloads (`polyvoice::models`).
- Powerset segmentation (`segmentation` feature), Embedder trait + CAM++/ResNet34
  adapters (`embedder` feature).
- Clusterer trait with AHC and NME-SC adapters (`clusterer` feature).
- Overlap-aware resegmentation pass (`resegmentation` feature).
- Pipeline builder API (`pipeline` feature).
- INT8 quantized model support in manifest.
- Model signing with Minisign (Ed25519), TLS hardening (`rustls`), WAV DoS guards,
  ONNX header validation.
- Comprehensive property tests and contract verification (Hoare triples) across
  all modules.

### Changed

- Default pipeline restored to v0.5.2 legacy path (DER ~13.8% on VoxConverse-test).
- New v1.0 architecture preserved as opt-in via Cargo features.

### Fixed

- See alpha release notes ([0.6.0-alpha.1] through [0.6.0-alpha.8]) for the
  complete incremental list.

## [0.6.0-alpha.7]

### Added

- Property tests for CLI argument parsing (`polyvoice`, `polyvoice-bench`).
- `debug_assert!` in `kmeans_pp` for uniform embedding dimensions precondition.

### Changed

- All 109 `TODO: precondition` stubs filled with Hoare triples across all 25 modules.
- All 6 `kind: missing` proofs in deprecated `ecapa` and `embedding` modules replaced
  with links to existing smoke/unit tests.
- CLI binaries (`src/bin/`) now have `MODULE_CONTRACT.md`, `README.md`, `TODO.md`.

### Fixed

- Rustdoc warnings from unresolved intra-doc links and bracket syntax in contracts.

## [0.6.0-alpha.6]

### Added

- COAD execution ledger (`.coad/LEDGER.md`) with completed task records.
- GitHub Actions `coad-check` CI job.
- `VERSION` file for single-source-of-truth versioning.
- Property tests for `types`, `utils`, and `kmeans`.

### Changed

- `benches/der_ami.rs` migrated to canonical `polyvoice::der::compute_der_from_rttm`.
- `AGENTS.md` updated to reference COAD as sole coordination standard.

## [0.6.0-alpha.5]

### Added

- COAD (Contract-Orchestrated Agent Development) adoption across the entire
  repository.
- `MODULE_CONTRACT.md`, `README.md`, `TODO.md` for all 25 workcells.
- Project-level COAD files: `COAD_PROJECT_STANDARD.md`, `AGENT_FLOW.md`.
- `.coad/` directory with orchestration templates for coordinated agent work
  (task, proof, handoff, review, integration, goal, write lease).
- Unit tests for `kmeans`, `wav`, and `pipeline` modules.

### Reverted (M6b rollback — Path A)

- **Default pipeline restored to v0.5.2 legacy.** The M6b new pipeline
  (`PowersetSegmenter` + INT8/FP32 embedders + AHC/NME-SC) was benchmarked
  and found non-functional (DER 52–64% on VoxConverse-test vs. legacy
  ~13.8%). The new pipeline is preserved as `polyvoice::pipeline_v1`
  (experimental, gated behind the `pipeline` feature) but is no longer the
  default.
- Restored deleted modules from v0.5.2 (`cb764ff`): `src/pipeline.rs`,
  `src/vad.rs`, `src/silero_vad.rs`, `src/onnx.rs`.
- Restored `DiarizationConfig`, `DummyExtractor`, and
  `EmbeddingExtractor::extract(samples, config)` API.
- CLI (`polyvoice`) and benchmark (`polyvoice-bench`) rewritten back to
  legacy pipeline: `SileroVad` → `FbankOnnxExtractor` → AHC.
- Manifest profiles (`mobile`, `balanced`) now resolve to proven FP32
  legacy models (`silero_vad` + `cam_pp_fp32` / `wespeaker_resnet34`).
  INT8 entries remain in manifest but are no longer used by default.
- `tests/der_baseline.json` filled with operational numbers from full
  232-file VoxConverse-test run: DER 13.83% (miss 3.82%, FA 3.68%,
  confusion 6.34%) at threshold 0.45, collar 0.25.

### Security & Hardening (2026-05-09)
- **SUPPLY-002 — Model signing (Minisign):** All official ONNX models are now
  signed with Ed25519 via `minisign`. The project public key is baked into the
  binary; per-model signatures live inline in `manifest.toml`. Downloads are
  verified with streaming Minisign verification alongside SHA-256 in the same
  64 KiB loop. `scripts/sign-models.sh` automates release signing.
- **SUPPLY-003 — TLS hardening:** `ureq` is now pinned to `rustls`, removing
  `native-tls` from the dependency tree entirely.
- **DOS-002 — WAV DoS guards:** `read_wav` rejects files > 1 GiB and headers
  declaring > 1 hour duration before reading samples.
- **DOS-003 — ONNX header validation:** `validate_onnx_header()` checks the
  first 64 bytes for ONNX magic / protobuf header before any model reaches
  `ort::commit_from_file`.
- **FFI-001/FFI-002/FFI-003:** `MAX_SAMPLES` guard, path-traversal rejection,
  and panic observability in C FFI entry points.
- **CACHE-001:** `ensure_in_cache_only` marked `#[doc(hidden)]` as test-only.
- **SERIAL-001:** RTTM parser validates `is_finite() && >= 0.0` for timestamps.
- Security audit updated: all MEDIUM and HIGH findings resolved.
  See `docs/security/audit-2026-05-08.md`.

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
