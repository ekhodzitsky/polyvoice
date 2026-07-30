# src/onnx

## Purpose

ONNX model validation, the pluggable `InferenceRuntime` trait, and the
default `OrtSession` backend.

## Surfaces

- `InferenceRuntime` — minimal load-agnostic session trait (named / ordered tensor run)
- `OrtSession` — default backend (`ort`); sole production `ort::` import site
- `InferenceTensor` / `NamedTensor` / `InferenceError` — runtime-agnostic I/O
- `validate_onnx_header(path)`
- `OnnxValidationError`
- `OnnxError` — typed session-construction / metadata-read failures
  (`Validation` / `SessionBuild` / `Metadata`)
- `ExecutionProvider` — **ort-specific** EP selector (re-exported by `pipeline_v2::config`)
- `build_session_with_ep(path, ep, intra_threads)` — the ONE session constructor
  (validate-before-build, EP registration, warn + CPU fallback)

## Dependencies

- `ort` — confined to `ort_session.rs`

## Invariants

- validate_onnx_header rejects non-ONNX files and accepts valid ONNX files.
- Every embedding/segmentation session is built via `build_session_with_ep`.
- New neural stages must **not** import `ort::` — use `InferenceRuntime` only.

## Verification

```bash
cargo test --lib onnx --features onnx
rg 'ort::' src --type rust   # production imports only in ort_session.rs
```

## Notes

- A pure-Rust backend can implement `InferenceRuntime` without touching stages.
