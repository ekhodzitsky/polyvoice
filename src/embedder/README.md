# src/embedder

## Purpose

Embedder trait, overlap masking, embedder pooling, and ONNX-backed adapters
(CAM++, ResNet34, ERes2NetV2). This is the current embedding extraction
surface for polyvoice.

## Surfaces

- `Embedder` trait
- `EmbedderError`
- `apply_overlap_mask` — masks multi-speaker regions
- `CamPlusPlusExtractor` (requires `onnx` feature)
- `ResNet34Adapter` (requires `onnx` feature)
- `ERes2NetV2Extractor` (requires `onnx` feature)

All three ONNX adapters are named wrappers over one generic internal
`FbankAdapter` (shared `FbankOnnxExtractor` engine + `SessionBuild` error
mapping); they differ only in constructor conventions (fixed 256-d, explicit
`dim`, `DIM` const + `with_dim`).

`EmbedderPool<E>` is a test-only helper (compiled under `cfg(test)`), not
public API: production pipelines hold a `Box<dyn Embedder>`, and
`FbankOnnxExtractor` pools ONNX sessions internally.

## Dependencies

- `ort` — ONNX Runtime (for adapters)
- in-tree `ObjectPool` (`Mutex<Vec<_>>`) — blocking pool backing the ONNX
  session pool and the test-only embedder pool

## Invariants

- ONNX adapters output L2-normalized embeddings.
- `ObjectPool` checkout/return is safe for concurrent access.

## Verification

```bash
cargo test --lib embedder
cargo test --test embedder_test --features onnx
cargo test --test loom_pool
```

## Notes

- Pure-Rust core (trait, mask) compiles to wasm32.
- ONNX adapters require `onnx` feature.
