# Semver and API freeze

Development version is **0.22.0** (unreleased). This is not `1.0.0`. The GO checklist in
[`PRODUCTION-READINESS.md`](../PRODUCTION-READINESS.md) stays open until a
freeze window has held and the other 1.0 boxes are true.

This document is the advertised-surface contract. Breaking the frozen
surface without a CHANGELOG **Breaking** entry is a bug.

CI: `cargo semver-checks check-release` (job `semver-checks` in
`.github/workflows/ci.yml`) on the Rust library. CLI help snapshots, the
FFI header, and `schema/diarization-result-v1.json` cover the other doors.

## Frozen (freeze window, now)

| Surface | Contract | Gate |
|---------|----------|------|
| Product library | Crate-root `Pipeline` / `PipelineConfig` / `PipelineError` via `pipeline-native` + `vbx` | `cargo-semver-checks`; rustdoc |
| Result types | `types::{DiarizationResult, SpeakerTurn, SpeakerId, TimeRange, Profile, SampleRate}` | `cargo-semver-checks` |
| BYO / no-ort | `LegacyPipeline`, `StreamingPipeline`, `Embedder`, `EnergyVad`, `VbxClusterer::from_dir` with `--no-default-features` | `docs/library-mode.md`; job `ort-free-core` |
| CLI (`--features cli`) | Flags in `tests/snapshots/snapshot_cli_test__help_top_level.snap` | insta snapshot |
| Python | `polyvoice.Pipeline`, `polyvoice.DiarizationResult` | wheel tests |
| C FFI | [`include/polyvoice.h`](../include/polyvoice.h) ABI v3 | `ffi_smoke` |
| JSON | [`schema/diarization-result-v1.json`](../schema/diarization-result-v1.json) | `polyvoice schema`; additive fields only |

Additive changes on these surfaces are fine (new optional JSON fields, new
CLI flags, new FFI enum slots that keep old numbers). Removals, renames, and
meaning changes are breaking.

## Out of freeze

- tract (`cli-tract`), `--legacy` (hidden), BYO ONNX-file adapters on tract
- Silero, CAM++ / ECAPA / ERes2Net, EP-only knobs
- `#[doc(hidden)]` items (`cli_common`)
- Domain profile `callhome` (uncalibrated placeholder)
- Internal modules (`pipeline_v2` internals, kernels, bench binaries)
- Experimental CLI flags already hidden (`--v2`)

## Bumps while 0.x

Cargo and Cargo/Rust treat 0.x minors as allowed to break. During this freeze
window we still do **not** break the frozen surface silently:

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
