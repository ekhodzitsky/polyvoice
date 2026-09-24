# Semver and API freeze

Development version is **0.22.0** (unreleased). This is not `1.0.0`. The GO checklist in
[`PRODUCTION-READINESS.md`](../PRODUCTION-READINESS.md) stays open until a
freeze window has held and the other 1.0 boxes are true.

This document is the advertised-surface contract. Breaking the frozen
surface without a CHANGELOG **Breaking** entry is a bug.

CI: `cargo semver-checks check-release` (job `semver-checks` in
`.github/workflows/ci.yml`) on the Rust library. CLI help snapshots, the
FFI header, and `schema/diarization-result-v1.json` cover the other doors.

## Advertised surfaces (final RC window not started)

| Surface | Contract | Gate |
|---------|----------|------|
| Product library | Crate-root `Pipeline` / `PipelineConfig` / `PipelineError` via `pipeline-native` + `vbx`, or `pipeline-local` | `cargo-semver-checks`; rustdoc |
| Result types | `types::{DiarizationResult, SpeakerTurn, SpeakerId, TimeRange, Profile, SampleRate}` | `cargo-semver-checks` |
| BYO / no-ort | `LegacyPipeline`, `StreamingPipeline`, `Embedder`, `EnergyVad`, `VbxClusterer::from_dir` with `--no-default-features` (enable `vbx` for VBx) | `docs/library-mode.md`; job `ort-free-core` |
| CLI (`--features cli`) | Flags in `tests/snapshots/snapshot_cli_test__help_top_level.snap` | insta snapshot |
| Python | `polyvoice.Pipeline`, `polyvoice.DiarizationResult` | wheel tests |
| C FFI | [`include/polyvoice.h`](../include/polyvoice.h) ABI v3 | `ffi_smoke` |
| JSON | [`schema/diarization-result-v1.json`](../schema/diarization-result-v1.json) | `polyvoice schema`; additive fields only |

Additive changes on these surfaces are fine (new optional JSON fields, new
CLI flags, new FFI enum slots that keep old numbers). Removals, renames, and
meaning changes are breaking. Adding fields to exhaustive Rust structs or
variants to exhaustive enums can also break consumers; “additive” does not
automatically mean compatible.

## Out of freeze

- tract (`cli-tract`) and BYO ONNX-file adapters on tract
- MCP protocol/tool API (`mcp`); experimental even though its engine is native
- Silero, CAM++ / ECAPA / ERes2Net, EP-only knobs
- `#[doc(hidden)]` items (`cli_common`)
- Domain profile `callhome` (uncalibrated placeholder)
- Internal modules (`pipeline_v2` internals, kernels, bench binaries)
- Experimental CLI flags already hidden (`--v2`)

Public reachability still matters: calling a module internal here does not
hide it from Rust consumers. The final API audit must resolve those boundaries
and cover every advertised feature combination before the RC window starts.
The CLI rejects `--legacy`; it is not a supported runtime fallback.

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
