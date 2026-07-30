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
- `vad` — crate-internal scalar hysteresis core (`vad::hysteresis`:
  `HysteresisGate` + `RegionTracker`) used by calibrated binarization

## Invariants

- PowersetDecoder is deterministic for identical logits.
- Hungarian assignment minimizes total cost across windows.
- One frame-time convention: `frame_index_at` uses `floor((t-start)/stride)`, which
  is the nearest-center frame (it equals `round((t-start)/stride - 0.5)` after the
  `[0, num_frames-1]` clamp), matching the run-length encoder's center placement —
  the IoU sampler and the applier agree (no half-stride off-by-one).

## Verification

```bash
cargo test --lib segmentation
cargo test --test chaos_test --features onnx,download
```

## Notes

- Pure-Rust core (decoder, aggregator, Hungarian) compiles to wasm32.
- ONNX-backed PowersetSegmenter requires `onnx` feature.
- The softmax is computed exactly once per frame: `PowersetDecoder` returns the
  full probability vector on `FrameLabel.probs`, and the `Aggregator` averages
  and permutes those probabilities directly. The powerset class table is
  defined once (`PowersetDecoder::class_for_index`) with the checked inverse
  `PowersetClass::from_speakers` / `PowersetClass::index` (crate-visible) used
  by the permutation remap and the calibrated binarization.
- `PowersetSegmenter::segment()` validates the `PowersetConfig` window
  geometry up front (`0 < hop_secs <= window_secs`, positive sample rate, at
  least one sample per window/hop) and returns
  `SegmentationError::InvalidGeometry` on violations.
  `PowersetConfig::with_model_meta` silently keeps the current geometry when
  overlaid model metadata would be inconsistent, so bad manifests cannot
  panic the window iterator.

## Calibrated binarization (opt-in)

`AggregationConfig.binarization: Option<BinarizationConfig>` replaces the
per-frame argmax with pyannote-style calibrated binarization: each speaker's
activity probability (sum of the powerset classes containing the speaker) is
thresholded with onset/offset hysteresis, then short blips are dropped
(`min_duration_on`) and short gaps bridged (`min_duration_off`). `None`
(default) keeps the historical argmax. The smoothing runs on the shared
`vad::hysteresis` core (`HysteresisGate` + `RegionTracker` with the `Trim`
closing policy); coverage holes (`has_data == false`) hard-close a region
instead of being bridged.

Thresholds are dataset-sensitive — calibrate offline per domain with
`scripts/calibrate-binarization.sh` (grid-search over onset/offset against
no-collar DER; not in the hot path). Measured on VoxConverse-10 (v2, collar 0,
micro): argmax 28.80% → 0.6/0.4 26.24% → 0.7/0.3 26.19%; on AMI EN2002a the
VoxConverse optimum is neutral (48.40% vs 48.29%), so 0.6/0.4 is the safe
starting point and per-domain calibration is recommended.
