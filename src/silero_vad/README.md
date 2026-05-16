# src/silero_vad

## Purpose

ONNX-based Silero Voice Activity Detector. Implements VoiceActivityDetector
trait from vad.rs.

## Surfaces

- `SileroVad`

## Dependencies

- `vad` — VoiceActivityDetector trait
- `ort` — ONNX Runtime

## Invariants

- Requires exactly 512 samples per process call.

## Verification

```bash
cargo test --lib silero_vad --features onnx
```

## Notes

- Model downloaded via ModelRegistry (requires `download` feature).
