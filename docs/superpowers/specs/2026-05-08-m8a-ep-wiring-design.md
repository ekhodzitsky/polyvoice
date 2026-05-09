---
title: M8a — Execution Provider wiring + CI matrix expansion
date: 2026-05-08
status: draft
milestone: M8a
preceding: M0–M6b
following: M8b (Android NNAPI + cross-compile), M9 (release polish)
authors: ekhodzitsky
---

# M8a — Execution Provider wiring + CI matrix expansion

## Problem

`PipelineConfig::execution_provider` is an `ExecutionProvider` enum that selects CPU / CoreML / NNAPI / CUDA / XNNPACK. The enum exists since M6a, but no `ort::Session` actually receives the provider — every ONNX session opens with the default CPU EP. Mobile profile gains no platform acceleration despite the M5 INT8 bundle being CPU/EP-friendly. Roadmap §10.1 M8 calls for «macOS CoreML EP зелёный» and «aarch64-linux-gnu CI зелёный»; we already have the latter (M0), the former is a no-op without EP wiring.

## Goal

Single milestone (~1 week), single PR. Pass `ExecutionProvider` from `PipelineBuilder::build()` through to every ONNX `Session::builder()` in M1/M2 components. Add CI jobs that exercise CoreML on macOS aarch64 and XNNPACK on Linux aarch64. After M8a: a Mobile/Balanced pipeline built on macOS uses CoreML automatically; on Linux aarch64 uses XNNPACK; on x86 uses CPU. M8b adds Android NNAPI + cross-compile.

## Non-goals

- Android NDK / `cargo-ndk` build — M8b.
- NNAPI execution provider runtime test — M8b (requires Android emulator/device).
- CUDA EP CI — out of scope (no GPU runner).
- RT-factor measurement — M8b/M9.
- Replacing the legacy CPU-only `EmbedderPool` with EP-aware sessions — already EP-aware after this milestone.

## Approach

### One helper in `src/pipeline/ep.rs`

```rust
//! M8a — Execution-provider wiring for ort::SessionBuilder.

use crate::pipeline::config::ExecutionProvider;
use ort::session::builder::SessionBuilder;

#[derive(Debug, thiserror::Error)]
pub enum EpError {
    #[error("ONNX runtime EP registration failed: {0}")]
    Ort(#[from] ort::Error),
}

pub fn apply_execution_provider(
    builder: SessionBuilder,
    ep: ExecutionProvider,
) -> Result<SessionBuilder, EpError> {
    match ep {
        ExecutionProvider::Cpu => Ok(builder),
        #[cfg(feature = "coreml")]
        ExecutionProvider::CoreMl => Ok(builder.with_execution_providers([
            ort::execution_providers::CoreMLExecutionProvider::default().build(),
        ])?),
        #[cfg(feature = "xnnpack")]
        ExecutionProvider::XnnPack => Ok(builder.with_execution_providers([
            ort::execution_providers::XNNPACKExecutionProvider::default().build(),
        ])?),
        #[cfg(feature = "nnapi")]
        ExecutionProvider::Nnapi => Ok(builder.with_execution_providers([
            ort::execution_providers::NNAPIExecutionProvider::default().build(),
        ])?),
        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda => Ok(builder.with_execution_providers([
            ort::execution_providers::CUDAExecutionProvider::default().build(),
        ])?),
        #[cfg(not(all(
            feature = "coreml", feature = "xnnpack", feature = "nnapi", feature = "cuda"
        )))]
        _ => {
            tracing::warn!("execution provider {ep:?} requires Cargo feature; using CPU");
            Ok(builder)
        }
    }
}
```

This module is gated `#[cfg(all(feature = "pipeline", feature = "onnx"))]` and exported from `crate::pipeline::ep`.

### EP-aware constructors on existing components

Add `with_execution_provider(...)` constructors to:

1. `PowersetSegmenter::with_execution_provider(model_path, ep) -> Result<Self, SegmentationError>` — wraps existing `Session::builder().commit_from_file(path)` with `apply_execution_provider` between `builder()` and `commit_from_file`.
2. `FbankOnnxExtractor::with_execution_provider(model_path, dim, pool_size, ep)` — same pattern, applies to every session in the pool.
3. `CamPlusPlusExtractor::with_execution_provider(...)` and `ResNet34Adapter::with_execution_provider(...)` — thin wrappers that call `FbankOnnxExtractor::with_execution_provider` then construct the adapter.

The old `::new(...)` constructors stay as deprecated wrappers that pass `ExecutionProvider::Cpu`.

### Builder hookup

In `src/pipeline/builder.rs`, where `build()` constructs Mobile/Balanced ONNX components today, replace each `::new(...)` with `::with_execution_provider(..., self.config.execution_provider)`. Custom profile is unaffected (caller supplies their own components).

### Error mapping

Each component's existing error type (`SegmentationError::ModelIo`, etc.) gets a new variant or absorbs `EpError`. Simpler: `apply_execution_provider` returns `EpError::Ort(ort::Error)`, which is convertible into the component's error via the existing `ort::Error` `#[from]` chain. No new error variants needed if existing types already wrap `ort::Error`.

### CI matrix

`.github/workflows/ci.yml` gets two new jobs:

```yaml
- name: test-macos-coreml
  runs-on: macos-latest
  features: "pipeline,coreml"
  steps:
    - cargo test --features pipeline,coreml --lib pipeline -- ep_wired_through_session
    - cargo test --features pipeline,coreml --test pipeline_synthetic_test

- name: test-linux-aarch64-xnnpack
  runs-on: ubuntu-latest
  target: aarch64-unknown-linux-gnu
  features: "pipeline,xnnpack"
  uses cross or cargo --target
```

Existing `cross-aarch64-linux` job extended to include the `xnnpack` feature.

## File layout

| Path | Action |
|---|---|
| `src/pipeline/ep.rs` | create |
| `src/pipeline/mod.rs` | add `pub mod ep;` |
| `src/segmentation/powerset.rs` | add `with_execution_provider` constructor |
| `src/ecapa.rs` | add `with_execution_provider` to `FbankOnnxExtractor` |
| `src/embedder.rs` | add `with_execution_provider` to `CamPlusPlusExtractor` + `ResNet34Adapter` |
| `src/pipeline/builder.rs` | thread config.execution_provider into Mobile/Balanced build paths |
| `tests/pipeline_ep_test.rs` | new unit + integration test |
| `.github/workflows/ci.yml` | add coreml + xnnpack jobs |
| `CHANGELOG.md` | M8a section |

Total ~250 LOC Rust + ~50 LOC CI yaml + ~80 LOC doc.

## Acceptance criteria

1. `cargo build --features pipeline,coreml` clean on macOS aarch64.
2. `cargo build --features pipeline,xnnpack --target aarch64-unknown-linux-gnu` clean.
3. `cargo test --features pipeline,coreml --lib pipeline::ep` — at least one test verifies CoreML provider registers without error (gated `#[cfg(target_os = "macos")]`).
4. New `tests/pipeline_ep_test.rs` integration test — `#[ignore]` E2E test loads a Pipeline with `ExecutionProvider::CoreMl` (macOS) and asserts run() returns Ok on a synthetic 1s buffer.
5. `cargo clippy --all-targets --all-features -- -D warnings` clean.
6. `cargo fmt --check` clean.
7. `cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib` clean — wasm32 unaffected.
8. CI matrix passes the two new jobs (or skips with documented reason).
9. release-gate.sh stays exit 2 (M8b/M9 still PENDING).

## Risks

| Risk | Mitigation |
|---|---|
| `ort` crate's EP API name differs (e.g. `CoreMLExecutionProvider` vs `CoreMlExecutionProvider`) | Verify with `cargo doc --features pipeline,coreml --open` before writing the helper. Adjust to actual API. |
| EP registration succeeds at compile time but runtime panics on missing dylib (CoreML on Linux machines) | The `#[cfg(feature = ...)]` gate prevents compile if feature missing; `#[cfg(target_os = "macos")]` gate on the test prevents runtime panic. |
| `XNNPACKExecutionProvider` may need `intra_op_threads` config | Default is fine for benchmarking; can add config knob in a follow-up if needed. |
| Cross-compilation for aarch64-unknown-linux-gnu pulls XNNPACK deps that fail to build under `cross` | Document in PR description. If XNNPACK doesn't cross-compile, defer that specific job to M8b. |

## Out of scope (M8b)

- `cargo-ndk` Android build job.
- NNAPI runtime test on Android emulator.
- QEMU RT-bench.
- Mobile profile real Android testing.

## Decomposition

Single PR, ~6 atomic commits:
1. `feat(ep): add pipeline::ep helper module`
2. `feat(segmentation): add PowersetSegmenter::with_execution_provider`
3. `feat(embedder): add with_execution_provider to FbankOnnxExtractor + adapters`
4. `feat(pipeline): thread ExecutionProvider through builder.build()`
5. `test(pipeline): add tests/pipeline_ep_test.rs (unit + #[ignore] E2E)`
6. `ci: add macOS CoreML + Linux aarch64 XNNPACK matrix jobs + CHANGELOG + tag m8a-complete`

## References

- Roadmap §10.1 M8 row, §3.3 (concurrency model), §6.4 (CLI download).
- M0 plumbing (existing `coreml/nnapi/xnnpack/cuda` Cargo features).
- M6a Pipeline / `ExecutionProvider` enum.
- `ort` crate docs: <https://docs.rs/ort/latest/ort/execution_providers/>
