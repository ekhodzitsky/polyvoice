# src/onnx

## Purpose

ONNX model validation (header checks) and legacy OnnxEmbeddingExtractor.

## Surfaces

- `validate_onnx_header(path)`
- `OnnxValidationError`
- `ExecutionProvider` — canonical EP selector (re-exported by `pipeline_v2::config`)
- `build_session_with_ep(path, ep, intra_threads)` — the ONE ort-session
  constructor for embedding + segmentation paths (validate-before-build,
  EP registration, warn + CPU fallback for unwired providers)
- `OnnxEmbeddingExtractor` — legacy wrapper

## Dependencies

- `embedding` — EmbeddingExtractor trait
- `types` — DiarizationConfig
- `ort` — ONNX Runtime

## Invariants

- validate_onnx_header rejects non-ONNX files and accepts valid ONNX files.
- EP selection lives here: every embedding/segmentation session is built via
  build_session_with_ep, and validation always runs before ort parses a file.

## Verification

```bash
cargo test --lib onnx --features onnx
```

## Notes

- OnnxEmbeddingExtractor is legacy; prefer adapters in embedder.rs.
