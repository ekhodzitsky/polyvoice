# src/resegmentation

## Purpose

Overlap-aware post-clustering resegmentation: refines speaker assignments
in multi-speaker overlap regions using centroids and cosine similarity.

## Surfaces

- `Resegmenter` trait
- `OverlapResegmenter`
- `compute_centroids`
- `extract_overlap_time_ranges`

## Dependencies

- `types` — SpeakerId, SpeakerTurn, TimeRange

## Invariants

- compute_centroids outputs L2-normalized vectors.

## Verification

```bash
cargo test --lib resegmentation
cargo test --test resegmenter_test --features resegmentation,onnx
```

## Notes

- Pure-Rust, wasm32-clean. Does not require ONNX.
- Operates on already-computed centroids and overlap embeddings.
