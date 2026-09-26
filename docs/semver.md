# Semver and API freeze

Development version is **0.22.0** (unreleased). This is not `1.0.0`. The GO checklist in
[`PRODUCTION-READINESS.md`](../PRODUCTION-READINESS.md) stays open until a
freeze window has held and the other 1.0 boxes are true.

This document is the advertised-surface contract. Breaking the frozen
surface without a CHANGELOG **Breaking** entry is a bug.

CI: `scripts/check-semver.sh` (job `semver-checks` in
`.github/workflows/ci.yml`) runs `cargo-semver-checks` on the Rust library,
see [the gate](#semver-gate) below. CLI help snapshots, the FFI header, and
`schema/diarization-result-v1.json` cover the other doors.

## Advertised surfaces (final RC window not started)

The frozen Rust surface is **every `pub` item that is not `#[doc(hidden)]`
and is reachable in one of the four checked feature sets** below. The
"Contract" column names the entry points consumers should build on; other
public items in those sets are checked the same way.

| Surface | Contract | Gate |
|---------|----------|------|
| Product library (`pipeline-native` + `vbx`, or `pipeline-local`) | Crate root `Pipeline`, `PipelineBuilder`, `PipelineConfig` (every field except `experimental`), `ClustererKind`, `PipelineError`; `pipeline_v2::{ConfigError, ExecutionProvider, StageTimings, MAX_AUDIO_SAMPLES}`; `ModelRegistry`, `ProfileModels`, `RegistryError`, `models::{Manifest, ManifestError}`; `clusterer::{AsNormConfig, CohortSource, DomainProfile, VOXCONVERSE, AMI}`; `VbxClusterer`, `VbxClustererConfig`; `wav::load_audio` | semver sets `product`, `local`; rustdoc |
| Result types | `types::{DiarizationResult, SpeakerTurn, Segment, SpeakerSummary, AudioMeta, Provenance, SpeakerId, TimeRange, Confidence, Profile, SampleRate}` | every semver set; JSON schema |
| BYO / no inference runtime | `pipeline::{LegacyPipeline, LegacyPipelineError}`, `streaming::{StreamingPipeline, LatencyPreset}`, `Embedder`, `EmbedderError`, `EnergyVad`, `VadConfig`, `VadError`, `VoiceActivityDetector`, `DiarizationConfig`, `ClusterConfig`, `ConfigError`, `Clusterer`, `ClustererError`, `AhcClusterer`, `VbxClusterer::from_dir` (`vbx`) with `--no-default-features` | semver sets `byo`, `byo-vbx`; `docs/library-mode.md`; job `ort-free-core` |
| CLI (`--features cli`) | Flags in `tests/snapshots/snapshot_cli_test__help_top_level.snap` | insta snapshot |
| Python | `polyvoice.Pipeline`, `polyvoice.DiarizationResult` | wheel tests |
| C FFI | [`include/polyvoice.h`](../include/polyvoice.h) ABI v3 | `ffi_smoke` |
| JSON | [`schema/diarization-result-v1.json`](../schema/diarization-result-v1.json) | `polyvoice schema`; additive fields only |

Additive changes on these surfaces are fine (new optional JSON fields, new
CLI flags, new FFI enum slots that keep old numbers). Removals, renames, and
meaning changes are breaking.

### Extensibility rules

Which Rust additions stay compatible is decided per type, not by habit:

- **Configuration structs are `#[non_exhaustive]`**: `PipelineConfig`,
  `ExperimentalConfig`, `AsNormConfig`, `VbxClustererConfig`,
  `DomainProfile`. Construct them from `Default` (or `AsNormConfig::new`,
  the domain constants) and assign fields; struct literals do not compile
  outside the crate. Adding a field is a minor change.
- **Enums are `#[non_exhaustive]`**: `ClustererKind`, `ExecutionProvider`,
  `Profile`, `CohortSource`, and every error enum on the checked surfaces
  (`PipelineError`, `pipeline_v2::ConfigError`, `RegistryError`,
  `ManifestError`, `DownloadError`, `ClustererError`, `PldaError`,
  `AsNormError`, `SegmentationError`, `ResegmentError`, `WavError`,
  `VadError`, `LegacyPipelineError`, `types::ConfigError`, `EmbedderError`).
  Match with a wildcard arm; adding a variant is a minor change.
- **Output structs are `#[non_exhaustive]`**: `DiarizationResult`,
  `SpeakerSummary`, `AudioMeta`, `Provenance`. Consumers read them and use
  `DiarizationResult::new` / `Default` to build test values, so a new JSON
  field is additive in Rust as well.
- **Value types stay exhaustive**: `SpeakerTurn`, `Segment`, `TimeRange`,
  `Word`, `WordAlignment`, `Transcript`, `SpeakerId`, `Confidence`,
  `SampleRate`. Consumers construct them, so adding a field is **breaking**.
- `PipelineConfig::experimental` (`ExperimentalConfig`) is outside the
  freeze: its fields may change in any minor release and none of them
  changes the shipped defaults.

## Out of freeze

- `PipelineConfig::experimental` and everything it switches
- `#[doc(hidden)]` items: `cli_common`, `clusterer::{plda, assign,
  short_filter}`, `models::{adapter, metadata, verify}` — still compiled,
  ignored by `cargo-semver-checks`
- tract (`cli-tract`, `pipeline-tract`, `backend-tract`) and BYO ONNX-file
  adapters on tract; `ExecutionProvider` values other than `Cpu` / `auto`
  are rejected by `PipelineBuilder::validate` on product builds
- MCP protocol/tool API (`mcp`); experimental even though its engine is native
- Silero, CAM++ / ECAPA / ERes2Net, EP-only knobs
- Domain profile `callhome` (uncalibrated placeholder)
- Kernel crate internals and bench binaries
- Experimental CLI flags already hidden (`--v2`)

The CLI rejects `--legacy`; it is not a supported runtime fallback.

## Semver gate

`scripts/check-semver.sh` compares the working tree with the **latest `v*`
tag** (override: `SEMVER_BASELINE=<rev>`) using `cargo-semver-checks`, once
per feature set:

| Set | Features |
|-----|----------|
| `product` | `pipeline-native`, `vbx` |
| `byo` | none |
| `byo-vbx` | `clusterer`, `vbx` |
| `local` | `pipeline-local` |

Every run passes `--release-type minor`, so the 0.x compatibility lints are
evaluated instead of being waived by the version bump. A detected break is
accepted only when the crate's minor version is above the baseline's **and**
`CHANGELOG.md` has a `### Breaking` section under `## [Unreleased]`;
otherwise the job fails and names what is missing. A set whose features do
not exist at the baseline is skipped: a feature introduced after the last
release has no contract until that release is tagged.

`scripts/check-semver.sh --probe` hides `EnergyVad` and the crate-root
`Pipeline` in a scratch copy and requires every compared set to report the
removal. CI runs the gate and the probe on every push.

## Release-candidate window

Require at least **two published 1.0 release candidates** and **14 consecutive
calendar days without an advertised-surface break after the final breaking
change**. Start the clock only when the API boundary is finalized, the first
qualifying candidate is published, and the non-window readiness gates have
evidence.
Record candidate revisions, publication dates, window start/end, consumer
results and resolved regressions in the release evidence. No RC window is
claimed merely because this policy exists.

An API/ABI/CLI/JSON compatibility break resets the clock and requires a new
candidate. Compatible fixes require a new candidate and rerunning affected
gates; final artifact checks must use the exact candidate being promoted.
At window end, all gates in [readiness](../PRODUCTION-READINESS.md) must pass
and no release-blocking regression may remain open. Experimental MCP/tract
and companion ASR do not expand the frozen batch-product scope.

## Bumps while 0.x

Cargo treats 0.x minor releases as potentially breaking. Before and during
the final RC window we do **not** break the advertised surface silently:

| Change | Version |
|--------|---------|
| Breaking on a frozen surface | `0.x+1.0` **and** CHANGELOG `### Breaking` |
| Additive on a frozen surface | `0.x.y+1` or the next minor, CHANGELOG `### Added` |
| Out-of-freeze only | patch or minor; say so in CHANGELOG |

`1.0.0` is the first release that treats a frozen-surface break as a **major**.
Do not ship `1.0.0` from this document alone.

## How to change a frozen symbol

1. Write the failing test or snapshot first when that surface is already tested.
2. Note the break in CHANGELOG `### Breaking` in the same PR.
3. `cargo semver-checks check-release` must report it if it is a Rust API break.
4. Do not rely on “pre-1.0, anything goes.”
