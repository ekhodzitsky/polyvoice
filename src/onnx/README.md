# src/onnx

## Purpose

ONNX model validation (header checks) and legacy OnnxEmbeddingExtractor.

## Surfaces

- `validate_onnx_header(path)`
- `OnnxValidationError`
- `OnnxEmbeddingExtractor` — legacy wrapper

## Dependencies

- `embedding` — EmbeddingExtractor trait
- `types` — DiarizationConfig
- `ort` — ONNX Runtime

## Invariants

- validate_onnx_header rejects non-ONNX files and accepts valid ONNX files.

## Verification

```bash
cargo test --lib onnx --features onnx
```

## Notes

- OnnxEmbeddingExtractor is legacy; prefer adapters in embedder.rs.
