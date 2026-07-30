# src/fbank_onnx — TODO

- [x] Implement `Embedder` directly (drop in-tree `EmbeddingExtractor` path).
- [x] Rename module to `fbank_onnx` (migration lease; hard rename — rustc does not lint deprecated module re-exports, so no alias was kept).
- [ ] Standalone model-load smoke test under `onnx` with a fixture ONNX.
