# src/pipeline

## Purpose

Offline diarization pipeline orchestration: segment → embed → cluster →
resegment → merge → emit DiarizationResult.

## Surfaces

- `Pipeline`
- `PipelineError`

## Dependencies

- `types` — DiarizationConfig, DiarizationResult
- `vad` — VoiceActivityDetector
- `ahc` — agglomerative clustering
- `wav` — audio input

## Invariants

- Pipeline output turns are monotonically ordered and non-overlapping
  (before overlap detection).

## Verification

```bash
cargo test --test e2e_smoke_test --features onnx,download
```

## Notes

- Pipeline does not own individual algorithms; it orchestrates them.
