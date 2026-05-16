# src/ecapa

## Purpose

Legacy ECAPA-TDNN ONNX embedding extractor. Superseded by `embedder.rs`.
Kept for backward compatibility until M6 cleanup.

## Surfaces

- `FbankOnnxExtractor` — legacy ONNX-backed extractor

## Dependencies

- `embedding` — EmbeddingExtractor trait
- `features` — FbankExtractor, apply_cmvn
- `types` — DiarizationConfig
- `utils` — l2_normalize

## Invariants

- Outputs L2-normalized embeddings.

## Verification

```bash
cargo test --test embedder_test --features onnx
```

## Notes

- Deprecated: use `CamPlusPlusExtractor` or `ResNet34Adapter` from `embedder.rs`.
