# src/streaming

## Purpose

Online (streaming) diarization pipeline: windowed processing, incremental
clustering, and streaming state management.

## Surfaces

- `StreamingPipeline<V, E>`
- `StreamingError`

## Dependencies

- `types` — DiarizationConfig, SpeakerTurn, ClusterConfig
- `vad` — VoiceActivityDetector, VadConfig
- `window` — WindowBuffer
- `cluster` — SpeakerCluster
- `embedder` — Embedder

## Invariants

- Output speaker turns are monotonically ordered by time.

## Verification

```bash
cargo test --lib streaming
cargo test --test streaming_test --features onnx
```

## Notes

- Generic over VAD and Embedder types for flexibility.
