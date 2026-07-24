# ROADMAP_STATUS — InferenceRuntime trait (ort confinement)

## Done

| Item | Result |
|---|---|
| `InferenceRuntime` trait | Minimal surface in `src/onnx/runtime.rs`: named + ordered tensor run, input names, `InferenceTensor` (`f32`/`i64`), `Send`. Stateful VAD LSTM passes state as ordinary named I/O tensors. |
| `OrtSession` backend | Thin wrapper in `src/onnx/ort_session.rs` — **sole production module** that imports `ort::`. Preserves validate-before-build, EP wiring, `intra_threads`, warn-and-CPU-fallback. |
| `build_session_with_ep` | Now returns `OrtSession` (was `ort::session::Session`). Public pipeline / stage constructors unchanged. |
| Migrate stages | `silero_vad`, `segmentation/powerset`, `ecapa`, legacy `OnnxEmbeddingExtractor` use `InferenceRuntime` only — no `ort::`. |
| `ExecutionProvider` | Remains ort-specific config on the construction path; not part of the trait. |
| Docs | `src/onnx/MODULE_CONTRACT.md` + README: “new neural stages must not import `ort::`”. Stage contracts updated. |
| Mock unit tests | `MockRuntime` + tensor helpers under `onnx::runtime::tests`. |

## Not in scope (deferred)

| Item | Why |
|---|---|
| tract / candle / rten backends | Separate pure-Rust backend work; implement `InferenceRuntime` for tract next. |
| DER / algorithm changes | Identity-preserving wrap only; baselines must not move. |
| EP feature wiring (CoreML/CUDA/…) | Unchanged warn-and-fallback behavior. |
| Pre-existing clippy `-D warnings` noise | `collapsible_if` in cluster/der/overlap/resegmentation/pipeline_v2/segmentation aggregator; `manual_is_multiple_of` in `vad` — present without this work. |

## Notes for pure-Rust backend work

A pure-Rust ONNX backend can implement `InferenceRuntime` next:

1. Add a sibling module (e.g. `src/onnx/tract_session.rs`) behind an opt-in feature.
2. Implement `InferenceRuntime` for that session type (load path + header validation + `run` / `run_ordered`).
3. Stages already talk only to the trait / `OrtSession` concrete type at construction — introduce a factory or type parameter when wiring the alternate backend; do not re-import `ort::` into stages.
4. Keep `ExecutionProvider` ort-only; tract options stay tract-specific.
5. Parity harness: same ONNX → ort vs new backend outputs within fixed tolerances; DER on a small set must match.

## Verify commands

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice/.worktrees/task-414-runtime

# Production ort:: only under src/onnx/ort_session.rs
rg 'ort::' src --type rust

cargo test --lib --features "onnx,segmentation,embedder,clusterer,resegmentation,vbx,spectral,download"
cargo clippy --lib --features "onnx,segmentation,embedder,clusterer,resegmentation,vbx,spectral,download" -- -D warnings
```

## Results (this worktree)

- `rg 'ort::' src --type rust` (code, not docs): **only** `src/onnx/ort_session.rs`
- `cargo test --lib` (full feature set above) → **327 passed**
- clippy `-D warnings` → fails on pre-existing lints only (none introduced in runtime/session/stage migration files after local fixes)
