---
title: M6b — Legacy Cleanup + CLI/FFI/Python Migration Design
date: 2026-05-08
status: draft
milestone: M6b
preceding: M0, M1, M2, M3, M4, M5, M6a
following: M8 (Android + multi-platform CI), M9 (release polish + v1.0.0 GA)
authors: ekhodzitsky
---

# M6b — Legacy Cleanup + CLI/FFI/Python Migration Design

## Problem

М6a landed `polyvoice::pipeline_v2` as an additive module: new `Pipeline::builder()` API exists side-by-side with legacy `polyvoice::Pipeline::new(DiarizationConfig, VadConfig)` + `OfflineDiarizer`. CLI (`polyvoice diarize`), benchmark (`polyvoice-bench`), C FFI (`src/ffi.rs`), and Python pyo3 bindings (`python/src/lib.rs`) still consume the legacy API. Two surfaces is confusing for users and forbids migration to mobile-friendly INT8 models published in M5.

## Goal

Single milestone (~1-2 weeks), single PR with ~8–10 atomic commits — finalize the v1.0 architecture: delete legacy types, rename `pipeline_v2` → `pipeline`, rewrite CLI + bench + FFI + Python pyo3 to consume the new API, add a migration guide, ship a DER baseline schema (numbers deferred to operational follow-up). Bumps `Cargo.toml` version `0.6.0-alpha.0` → `0.6.0-alpha.3`.

End state: one canonical `polyvoice::Pipeline::builder()` API. CLI/FFI/Python all use it. v1.0.0 GA polish (CHANGELOG voice, blog post, release-gate green) becomes M9.

## Non-goals

- iOS / Windows wheels — M8.
- Android NNAPI execution provider integration — M8.
- Streaming pipeline v1.1 (`OnlineDiarizer` redesign) — separate spec.
- Real DER baseline numbers (operational follow-up after M5 INT8 publish closes; see §8 below).
- Full v1.0.0 release polish — M9.
- Backwards-compat shims for v0.5.x callers — v1.0 is intentionally breaking; see §11 migration guide.

## Approach

### Atomic-commit single PR

The deletes, renames, and rewrites must all land together: deleting `DiarizationConfig` without rewriting the CLI breaks compilation. We split the work into ~8–10 atomic commits **inside one PR**, mirroring the M5/M6a pattern:

1. Delete legacy lib types (`pipeline.rs`, `offline.rs`, `vad.rs`).
2. Rename `pipeline_v2 → pipeline` (module + Cargo feature + tests + re-exports).
3. Rewrite `src/bin/polyvoice.rs` (CLI).
4. Rewrite `src/bin/polyvoice-bench.rs` (DER bench).
5. Rewrite `src/ffi.rs` (C FFI v2 ABI).
6. Rewrite `python/src/lib.rs` (pyo3 bindings).
7. Add `docs/MIGRATING-FROM-0.5.md`.
8. Add `tests/der_baseline.json` schema + `scripts/run-der-baseline.sh`.
9. Annotate `OnlineDiarizer` `#[deprecated]`.
10. CHANGELOG + version bump + tag.

CI is permitted to fail on intermediate commits (since deletes precede their consumers) — the final PR head must be green.

### Reality-check on deletes

Before deleting a type, the implementer must `grep` for callers and ensure all are rewritten. The 15 files identified during brainstorming context check are tracked.

## Public API surface after M6b

```rust
// polyvoice::lib re-exports (final shape)
pub use pipeline::{
    ClustererKind, ConfigError, ExecutionProvider, Pipeline, PipelineBuilder,
    PipelineConfig, PipelineError,
};
pub use types::{
    Confidence, DiarizationResult, Profile, SampleRate, Seconds, Segment,
    SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment,
    remap_segments, remap_turns,
};
pub use rttm::{parse_rttm, write_rttm};
pub use der::{DerResult, compute_der};
pub use overlap::{OverlapRegion, detect_overlaps};
pub use models::{ModelRegistry, ProfileModels, RegistryError};
pub use embedder::{Embedder, EmbedderError, EmbedderPool, apply_overlap_mask};
pub use clusterer::{AhcClusterer, Clusterer, ClustererError};
pub use resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentError, ResegmentInputs,
    Resegmenter, SpeakerCentroid, compute_centroids, extract_overlap_time_ranges,
};
pub use segmentation::{
    AggregationConfig, Aggregator, FrameLabel, MIN_AUDIO_SAMPLES, PowersetClass,
    PowersetDecoder, PowersetSegmenter, RawSegment, SegmentationError, Segmenter,
    WindowOutput,
};

#[deprecated(since = "0.6.0-alpha.3", note = "streaming redesigned in v1.1; use Pipeline for offline")]
pub use online::OnlineDiarizer;
```

**Removed re-exports:**
`Pipeline as PipelineV0`, `OfflineDiarizer`, `DiarizationConfig`, `VadConfig`, `VadError`, `VoiceActivityDetector`, `EnergyVad`, `segment_speech`, `DummyExtractor`, `EmbeddingExtractor`, `EmbeddingError`, `OnnxEmbeddingExtractor`, `EcapaTdnnExtractor`, `RawAudioOnnxExtractor`, `EcapaMelOnnxExtractor`, `FbankConfig`, `FbankExtractor`, `merge_segments` (privatized), `compute_fbank` (privatized), `ClusteringBackend`, `EmbeddingDim`.

`SpeakerCluster` from `cluster.rs` remains (used internally by AHC). `cluster.rs` and `kmeans.rs` modules stay — internal helpers, not re-exported.

## Module-level changes

| Path | Action | Rationale |
|---|---|---|
| `src/pipeline.rs` (372 LOC) | **delete** | Legacy `Pipeline::new(DiarizationConfig, VadConfig)` replaced by `pipeline_v2::Pipeline::builder()`. |
| `src/offline.rs` (209 LOC) | **delete** | `OfflineDiarizer` absorbed by `Pipeline::run()`. |
| `src/vad.rs` (175 LOC) | **delete** | `VoiceActivityDetector` trait + `EnergyVad` + `VadConfig` + `segment_speech` — absorbed by `Segmenter`. `silero_vad.rs` keeps the Silero impl (used by legacy callers via direct construction; M9 may delete if unused after CLI/FFI/Python migration). |
| `src/pipeline_v2/` | **rename → `src/pipeline/`** | Drop the `_v2` suffix now that legacy is gone. |
| `src/online.rs` | modify | Add `#[deprecated]` attribute; otherwise unchanged. |
| `src/embedding.rs` | privatize / shrink | Keep `mean_vector` etc. as internal `crate::utils`-style helpers if still referenced; remove `DummyExtractor` (move to `embedding/mock.rs` `#[cfg(test)]`); remove `OnnxEmbeddingExtractor`. After cleanup, file may shrink to a thin module that re-exports test-only mocks. |
| `src/ecapa.rs` | privatize | `EcapaTdnnExtractor` (deprecated alias for FbankOnnxExtractor) + `EcapaMelOnnxExtractor` + `RawAudioOnnxExtractor` removed; `FbankOnnxExtractor` remains as M2's `ResNet34Adapter` backing. Strip the unused variants. |
| `src/onnx.rs` | check usage | `OnnxEmbeddingExtractor` removed. If no other callers, delete file. |
| `src/features.rs` | privatize `compute_fbank` | per spec §5.5; `FbankExtractor::extract` remains the public API. |
| `src/cluster.rs`, `src/kmeans.rs`, `src/ahc.rs`, `src/spectral.rs` | keep | Internal clusterer impls; not re-exported at crate root after M6b. |
| `src/types.rs` | shrink | Remove `DiarizationConfig`, `VadConfig`, `ClusteringBackend`, `EmbeddingDim`. Keep `Profile`, `SampleRate`, `SpeakerId`, `Confidence`, `TimeRange`, `Segment`, `SpeakerTurn`, `DiarizationResult`, `WordAlignment`, `SpeakerIdRemap`. |
| `src/lib.rs` | rewrite re-exports per "Public API surface after M6b" | The crate-root re-export block lists exactly the surface above. |
| `src/bin/polyvoice.rs` | rewrite | New CLI shape (see §4). |
| `src/bin/polyvoice-bench.rs` | rewrite | New bench shape (see §5). |
| `src/ffi.rs` | rewrite | ABI v2 (see §6). |
| `python/src/lib.rs` | rewrite | New pyo3 bindings (see §7). |
| `python/tests/test_smoke.py` | rewrite | Exercises `Pipeline.balanced()` etc. |
| `python/pyproject.toml` | bump | `version = "0.6.0a3"`. |
| `Cargo.toml` | bump + feature rename | `version = "0.6.0-alpha.3"`; rename feature `pipeline_v2 = ["download"]` → `pipeline = ["download"]`. |
| `include/polyvoice.h` | rewrite | C header for ABI v2. Bump major. |
| `tests/cli_smoke_test.rs` | create | `polyvoice diarize --help`, `--profile invalid` exit code. |
| `tests/ffi_smoke_test.rs` | create | ABI v2 round-trip. |
| `tests/pipeline_v2_*.rs` | rename → `tests/pipeline_*.rs` | Match the module rename. |
| `tests/der_baseline.json` | create | Schema only; numbers deferred. |
| `scripts/run-der-baseline.sh` | create | Operational helper to fill in `tests/der_baseline.json` later. |
| `docs/MIGRATING-FROM-0.5.md` | create | Migration guide per spec §11. |
| `CHANGELOG.md` | modify | Unreleased M6b section. |

Pre-existing dirty changes in working tree (`src/bin/polyvoice-bench.rs`, `src/ecapa.rs`, `src/features.rs` referencing `EcapaMelOnnxExtractor` + `RawAudioOnnxExtractor`) are **discarded** during M6b — the bench is rewritten from scratch on top of `Pipeline::builder()`, and the unused ecapa variants are deleted entirely.

## CLI rewrite (§4 from brainstorming)

`src/bin/polyvoice.rs`:

```text
polyvoice diarize <wav-path> [OPTIONS]
  --profile mobile|balanced|custom        [default: balanced]
  --custom-models <DIR>                   [required if --profile custom]
  --output <PATH>                         [default: stdout]
  --format rttm|json                      [default: rttm]
  --models-cache <DIR>                    [default: $HOME/.cache/polyvoice/models]
  --execution-provider auto|cpu|coreml|nnapi|cuda|xnnpack
                                          [default: auto]
  --resegment-overlap                     [default: true; --no-resegment-overlap to disable]
  --max-speakers <N>                      [default: 20]
  --quiet                                 [suppress progress logs]

polyvoice download-models [--profile mobile|balanced|all]
  (kept verbatim from M0)

polyvoice models list                     [new in M6b: prints available profiles + sizes]
polyvoice models info <NAME>              [prints url, sha256, calibration metadata]
```

Old flags (`--threshold`, `--max-speakers`, `--vad-*`, `--clustering-backend`, `--model-type`) are removed. Profile defaults from `Profile::default_threshold` etc. govern thresholds; users override via `--profile custom` + custom builder if needed (Custom profile from CLI uses ENV-based pipeline construction — out of scope for M6b's pure clap-driven CLI; document as «use Rust API for Custom»).

`tests/cli_smoke_test.rs`:

- `polyvoice --help` exits 0 with «diarize» listed.
- `polyvoice diarize --help` exits 0 with `--profile`, `--output`, `--format` listed.
- `polyvoice diarize foo.wav --profile invalid` exits non-zero with «invalid profile» message.
- `polyvoice models list` prints at least 2 profiles.

E2E «runs on a real WAV» test stays `#[ignore]` (requires cached ONNX bundle).

## Bench rewrite (§5)

`src/bin/polyvoice-bench.rs`:

```text
polyvoice-bench <DATASET-DIR> [OPTIONS]
  DATASET-DIR layout: { audio/*.wav, rttm/*.rttm }
  --profile mobile|balanced               [default: balanced]
  --output <PATH>                         [JSON report]
  --collar <SECS>                         [default: 0.25]
  --skip-overlap                          [default: false]
  --max-files <N>                         [smoke runs]
  --threads <N>                           [default: num_cpus]
```

Output JSON schema:

```json
{
  "schema": "polyvoice-bench-v1",
  "profile": "balanced",
  "files": 232,
  "der_collar_0_25_skip_overlap": 14.83,
  "der_no_collar": 18.12,
  "miss": 5.21,
  "false_alarm": 1.08,
  "confusion": 8.54,
  "speaker_count_exact": 0.78,
  "speaker_count_within_1": 0.91,
  "rt_factor_avg": 23.4,
  "model_versions": { ... },
  "git_sha": "...",
  "host_cpu": "...",
  "polyvoice_version": "0.6.0-alpha.3"
}
```

`scripts/run-der-baseline.sh` invokes `polyvoice-bench` and copies the relevant fields into `tests/der_baseline.json`.

## FFI rewrite (§6)

`src/ffi.rs` ABI v2:

```c
// include/polyvoice.h v2
typedef struct PolyvoicePipeline PolyvoicePipeline;

typedef enum {
    POLYVOICE_PROFILE_MOBILE = 0,
    POLYVOICE_PROFILE_BALANCED = 1,
} polyvoice_profile_t;

polyvoice_status_t polyvoice_pipeline_create(
    polyvoice_profile_t profile,
    const char* models_cache_dir,    // NULL → default ~/.cache/polyvoice/models
    PolyvoicePipeline** out_handle
);

polyvoice_status_t polyvoice_pipeline_run(
    PolyvoicePipeline* pipeline,
    const float* samples,
    size_t n_samples,
    uint32_t sample_rate,
    char** out_json,                 // caller free with polyvoice_free_string
    size_t* out_json_len
);

void polyvoice_pipeline_destroy(PolyvoicePipeline* pipeline);
void polyvoice_free_string(char* p, size_t n);
```

`Profile::Custom` not exposed via FFI (Rust-only feature). Status codes from spec §8.4. All entry points wrapped in `catch_unwind`.

`tests/ffi_smoke_test.rs` builds + tears down a Mobile pipeline (assuming cached ONNX) and runs on a synthetic 1s `[0.0; 16000]` buffer; asserts exit OK + JSON parses to empty turns.

## Python pyo3 rewrite (§7)

`python/src/lib.rs`:

```python
# Python-side surface (after pyo3 stubs):
from polyvoice import Pipeline, PipelineBuilder, Profile, DiarizationResult

# Convenience constructors
p = Pipeline.mobile(models_cache="~/.cache/polyvoice/models")
p = Pipeline.balanced()  # default cache
p = Pipeline.builder().profile(Profile.MOBILE).with_models_cache("./").build()

# Inference
result = p.run(samples_ndarray, sample_rate=16000)
# result is a dict: { "turns": [...], "segments": [...], "num_speakers": int }
```

The Custom profile is not exposed in the Python surface (would require passing Rust trait objects); document as Rust-only.

`python/tests/test_smoke.py` exercises `Pipeline.balanced()` build + `run([0.0]*16000, 16000)` → returns dict with empty `turns`.

## Migration guide (§7)

`docs/MIGRATING-FROM-0.5.md` — ~150 lines. Sections:

- Rust API: side-by-side `OfflineDiarizer + DiarizationConfig` → `Pipeline::builder().profile(...).with_models_from(...).build()` examples.
- Python API: `polyvoice.Pipeline("models/")` → `polyvoice.Pipeline.balanced("models/")`.
- CLI: `polyvoice diarize meeting.wav --threshold 0.4` → `polyvoice diarize meeting.wav --profile balanced`.
- C FFI: ABI v1 → v2 mapping table.
- Removed types and their replacements.
- Why `OnlineDiarizer` is deprecated.

## DER baseline schema (§8)

`tests/der_baseline.json` — schema with `null` placeholders. Cargo test `tests/der_baseline_test.rs` parses the JSON for schema validity (does not assert numbers). Real numbers come via operational `bash scripts/run-der-baseline.sh` after M5 INT8 publish (separate commit + PR or part of M9).

## OnlineDiarizer deprecation (§9)

```rust
// src/online.rs
#[deprecated(
    since = "0.6.0-alpha.3",
    note = "streaming pipeline redesigned in v1.1; use Pipeline for offline use"
)]
pub struct OnlineDiarizer { /* unchanged */ }
```

`#[allow(deprecated)]` on existing tests if any.

## Acceptance criteria

1. `cargo test --all-features` зелёный (all M0–M6a tests still pass; legacy-test files deleted with their owners).
2. `cargo clippy --all-targets --all-features -- -D warnings` clean.
3. `cargo fmt --check` clean.
4. `cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib` зелёный.
5. `cargo semver-checks` shows breaking change vs `b1fcc9b` (M5 squash) — expected.
6. `polyvoice diarize sample.wav --profile balanced --format rttm` runs end-to-end on cached ONNX (manual test).
7. `polyvoice-bench data/voxconverse-test --profile balanced --max-files 5 --output /tmp/bench.json` produces a valid JSON report (manual test; full DER baseline comes later).
8. `python -c "import polyvoice; p = polyvoice.Pipeline.balanced(); print(p)"` — reachable (manual test).
9. `tests/cli_smoke_test.rs`, `tests/ffi_smoke_test.rs`, `tests/der_baseline_test.rs` green in CI.
10. `docs/MIGRATING-FROM-0.5.md` exists with all 5 sections.
11. CHANGELOG appended with M6b section.
12. Tag `m6b-complete` placed.

## Tests catalogue

```text
tests/cli_smoke_test.rs (new, 4 tests)
tests/ffi_smoke_test.rs (new, 2 tests, #[ignore] for the e2e one)
tests/der_baseline_test.rs (new, 2 tests — JSON schema + tolerance bounds)

# Renamed:
tests/pipeline_synthetic_test.rs (was pipeline_v2_synthetic_test.rs)
tests/pipeline_e2e_test.rs (was pipeline_v2_e2e_test.rs)

# Untouched:
tests/m5_manifest_smoke_test.rs
tests/clusterer_test.rs
tests/resegmentation_test.rs
tests/miri_resegmentation.rs

# Deleted (legacy):
tests/pipeline_test.rs (if it existed for legacy Pipeline)
tests/offline_test.rs (if it existed)
```

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pre-existing dirty work in `src/bin/polyvoice-bench.rs` (EcapaMelOnnxExtractor / RawAudioOnnxExtractor refs) is discarded by full rewrite | high | Acknowledged trade-off. The dirty work was never committed in M0–M6a, so it has no claim to durability. The new bench, built on `Pipeline::builder()`, supersedes it. Document the discard in M6b's CHANGELOG section. |
| FFI ABI v2 breaks downstream C callers | high (intentional) | Migration guide maps old → new. Bump `polyvoice.h` major; bump Cargo `0.6.0-alpha.3`. Document in CHANGELOG. |
| Python wheels fail because `polyvoice.Pipeline.balanced()` requires INT8 download not yet uploaded to GitHub Releases | medium | M5 manifest sha256 is provisional; calling `Pipeline.balanced()` without offline-cached ONNX will hit `RegistryError::Sha256Mismatch`. Smoke tests use `#[ignore]` for any path requiring real ONNX. CI exercises only build-time correctness; runtime ONNX use is gated to manual / `#[ignore]`. |
| `polyvoice download-models` CLI subcommand needs to work on the renamed feature/types | low | The download path uses `crate::models::*`, untouched by M6b rename. Keep `download-models` subcommand verbatim. |
| Renaming `pipeline_v2 → pipeline` collides with the legacy `pipeline.rs` file (already deleted in commit 1) | none | Commit ordering: delete legacy first (commits 1–2), then rename (commit 3). After commit 1, `src/pipeline.rs` is gone and `src/pipeline/` directory creation is unambiguous. |
| Migration guide examples drift from real API after fixes | low | `cargo doc --no-deps` is in CI; doc-tests covering the migration examples ensure they compile. Add example doc-tests in `docs/MIGRATING-FROM-0.5.md` (markdown rust code blocks compiled as doc-tests). |

## Decomposition (atomic commits)

10 commits inside the single PR:

1. `feat(version): bump to 0.6.0-alpha.3`
2. `refactor(pipeline_v2): rename module to pipeline + Cargo feature pipeline_v2 → pipeline`
3. `refactor(legacy): delete src/pipeline.rs + src/offline.rs + src/vad.rs (Pipeline/OfflineDiarizer/VadConfig/EnergyVad)`
4. `refactor(legacy): privatize compute_fbank, remove EcapaTdnnExtractor + EcapaMelOnnxExtractor + RawAudioOnnxExtractor + DummyExtractor + OnnxEmbeddingExtractor`
5. `refactor(types): remove DiarizationConfig + VadConfig + ClusteringBackend + EmbeddingDim`
6. `refactor(cli): rewrite src/bin/polyvoice.rs on Pipeline::builder()`
7. `refactor(bench): rewrite src/bin/polyvoice-bench.rs on Pipeline::builder() + JSON report`
8. `refactor(ffi): rewrite src/ffi.rs to ABI v2 + update include/polyvoice.h`
9. `refactor(python): rewrite python/src/lib.rs pyo3 bindings on Pipeline.builder()`
10. `feat(m6b): MIGRATING-FROM-0.5.md + DER baseline schema + OnlineDiarizer deprecation + CHANGELOG + tag m6b-complete`

Commit ordering is intentional: delete legacy types in commits 3–5 BEFORE rewriting CLI/bench/FFI/Python which would otherwise still compile against legacy. After commit 5 CLI/FFI/Python all fail; commits 6–9 rewrite them in order. The PR head is green; intermediate CI may be red.

## References

- Roadmap: `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md`
  §3.1 (pipeline diagram), §5.4 (Pipeline + Builder), §5.5 (что удаляется), §10.1 (M6 row), §11 (migration guide draft).
- M6a spec: `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md`.
- M6a plan: `docs/superpowers/plans/2026-05-07-m6a-pipeline-v2-plan.md`.
- Existing `src/pipeline.rs` (legacy), `src/offline.rs` (legacy), `src/vad.rs` (legacy), `src/bin/polyvoice.rs`, `src/bin/polyvoice-bench.rs`, `src/ffi.rs`, `python/src/lib.rs`.

## Open questions (closed)

- ✅ Single-PR atomic-commit shape (variant 1 from brainstorming) over deprecation-chain split (variants 2/3).
- ✅ DER baseline closure: schema-only in M6b (deferred numbers — variant 1 from brainstorming).
- ✅ Pre-existing dirty bench work discarded; bench rewritten from scratch.
- ✅ `OnlineDiarizer` kept as `#[deprecated]` stub (per spec §5.6); not deleted.
- ✅ Profile::Custom not exposed via CLI / FFI / Python (Rust-only).

## Follow-ups

1. After spec approval: invoke `superpowers:writing-plans` to generate the M6b implementation plan in `docs/superpowers/plans/2026-05-08-m6b-legacy-cleanup-plan.md`. Style: 10 atomic-commit tasks, `m6b-complete` tag.
2. After M6b: operational `bash scripts/run-der-baseline.sh` populates `tests/der_baseline.json` once M5 INT8 artifacts are uploaded to GitHub Releases.
3. M8: Android NNAPI integration. M9: release polish + v1.0.0 GA tag.
