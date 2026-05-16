# src/embedder

## Purpose

Embedder trait, overlap masking, embedder pooling, and ONNX-backed adapters
(CAM++, ResNet34). This is the current embedding extraction surface for
polyvoice.

## Surfaces

- `Embedder` trait
- `EmbedderError`
- `EmbedderPool<E>` — lock-free pool
- `apply_overlap_mask` — masks multi-speaker regions
- `CamPlusPlusExtractor` (requires `onnx` feature)
- `ResNet34Adapter` (requires `onnx` feature)

## Dependencies

- `ort` — ONNX Runtime (for adapters)
- `crossbeam-queue` — lock-free queue (for pool)

## Invariants

- ONNX adapters output L2-normalized embeddings.
- EmbedderPool is safe for concurrent access.

## Verification

```bash
cargo test --lib embedder
cargo test --test embedder_test --features onnx
cargo test --test loom_pool
```

## Notes

- Pure-Rust core (trait, mask, pool) compiles to wasm32.
- ONNX adapters require `onnx` feature.
