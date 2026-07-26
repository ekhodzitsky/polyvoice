# src/pipeline

Always-on **bring-your-own** offline diarization pipeline.

> Re-exported at crate root as `polyvoice::Pipeline` / `PipelineError`.
> This is the ort-free library surface and the CLI `--legacy` path.
> Production ONNX (CLI/FFI/Python/MCP default) is
> [`pipeline_v2`](../pipeline_v2/).

## Surfaces

- `Pipeline` — generic over `Embedder` + `VoiceActivityDetector` at `run`
- `PipelineError`

## Flow

```
samples ──▶ VAD (segment_speech) ──▶ WindowIter embed ──▶ AHC ──▶ merge ──▶ turns
```

Uses `AhcClusterer` (feature `clusterer`) with unlimited max clusters; falls
back to free `ahc::agglomerative_cluster` without that feature. Does **not**
use `Segmenter` / `Resegmenter`. No overlap resegmentation.

## Verification

```bash
cargo test --lib pipeline
cargo test --test e2e_smoke_test --features onnx,download
```
