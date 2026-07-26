# src/ecapa

Shared **fbank + ONNX** speaker embedding engine.

Despite the historical name, this is not ECAPA-only: WeSpeaker ResNet34,
CAM++, and ERes2NetV2 all use [`FbankOnnxExtractor`].

## Surface

- `FbankOnnxExtractor` — implements [`Embedder`](crate::Embedder) directly

Prefer architecture-specific adapters in `src/embedder` when the model family
is known (`ResNet34Adapter`, `CamPlusPlusExtractor`, `ERes2NetV2Extractor`).

## Verification

```bash
cargo test --lib ecapa --features onnx
```
