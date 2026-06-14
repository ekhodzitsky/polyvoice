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
- One frame-time convention: `frame_index_at` uses `floor((t-start)/stride)`, which
  is the nearest-center frame (it equals `round((t-start)/stride - 0.5)` after the
  `[0, num_frames-1]` clamp), matching the run-length encoder's center placement —
  the IoU sampler and the applier agree (F03; no half-stride off-by-one).

## Verification

```bash
cargo test --lib segmentation
cargo test --test chaos_test --features onnx,download
```

## Notes

- Pure-Rust core (decoder, aggregator, Hungarian) compiles to wasm32.
- ONNX-backed PowersetSegmenter requires `onnx` feature.
