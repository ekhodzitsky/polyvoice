# src/segmentation

## Purpose

Speaker segmentation algorithms: powerset decoder, frame aggregation with
Hungarian assignment, and segmenter trait.

## Surfaces

- `Segmenter` trait
- `PowersetSegmenter` (requires `onnx`)
- `PowersetDecoder`
- `Aggregator`
- `FrameLabel`
- `MIN_AUDIO_SAMPLES`

## Dependencies

- `types` — Confidence, TimeRange

## Invariants

- PowersetDecoder is deterministic for identical logits.
- Hungarian assignment minimizes total cost across windows.

## Verification

```bash
cargo test --lib segmentation
cargo test --test chaos_test --features onnx,download
```

## Notes

- Pure-Rust core (decoder, aggregator, Hungarian) compiles to wasm32.
- ONNX-backed PowersetSegmenter requires `onnx` feature.
