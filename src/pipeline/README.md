# src/pipeline

Always-on **bring-your-own** offline diarization pipeline.

> `LegacyPipeline` / `LegacyPipelineError` (deprecated aliases `Pipeline` /
> `PipelineError` remain for downstream compatibility). This is the ort-free
> library surface and the CLI `--legacy` path; the crate-root `Pipeline`
> re-export is pipeline v2 when its feature gate is on. Production ONNX
> (CLI/FFI/Python/MCP default) is [`pipeline_v2`](../pipeline_v2/).

## Surfaces

- `LegacyPipeline` — generic over `Embedder` + `VoiceActivityDetector` at `run`
- `LegacyPipelineError`

## Flow

```
samples ──▶ config validate ──▶ VAD (segment_speech) ──▶ WindowIter embed ──▶ AHC ──▶ merge ──▶ turns
```

`run` / `run_with_clusterer` first validate `DiarizationConfig`
(`InvalidConfig` on bad window geometry or an out-of-range cosine threshold —
a zero-length window previously panicked in `WindowIter`). Uses `AhcClusterer`
(feature `clusterer`) with unlimited max clusters; inconsistent embedding
dimensions surface as `Clustering(ClustererError::DimMismatch)`. Without the
feature it runs free `ahc::agglomerative_cluster`. Does **not** use
`Segmenter` / `Resegmenter`. No overlap resegmentation.

## Verification

```bash
cargo test --lib pipeline
cargo test --test der_regression_test --features onnx,download
```
