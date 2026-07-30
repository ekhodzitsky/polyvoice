# src/fbank_onnx

Shared **fbank + ONNX** speaker embedding engine.

Not ECAPA-specific despite the `models/ecapa_tdnn*.onnx` filenames: WeSpeaker
ResNet34, CAM++, and ERes2NetV2 all use [`FbankOnnxExtractor`]. The module
was renamed from `ecapa` (hard rename, no alias — rustc does not lint
deprecated module re-exports).

## Surface

- `FbankOnnxExtractor` — implements [`Embedder`](crate::Embedder) directly
- `FbankExtractorError` — typed construction error (`EmptyPool` /
  `SessionBuild` wrapping `onnx::OnnxError`)

Prefer architecture-specific adapters in `src/embedder` when the model family
is known (`ResNet34Adapter`, `CamPlusPlusExtractor`, `ERes2NetV2Extractor`).

## Verification

```bash
cargo test --lib fbank_onnx --features onnx
```
