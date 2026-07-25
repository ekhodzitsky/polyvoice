# Pipeline architecture

Who calls whom in polyvoice, and which path is the production default.

For the **development process** checklist (spec → types → verify), see
[PIPELINE.md](PIPELINE.md).

## Two families (intentional)

```
                    ┌─────────────────────────────────────┐
  no onnx / BYO     │  pipeline::Pipeline                 │
  + streaming       │  streaming::StreamingPipeline       │── Embedder + VAD
                    │  (crate-root polyvoice::Pipeline)   │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  onnx production   │  pipeline_v2::Pipeline (+ Builder)  │── Segmenter/Embedder/
  CLI/FFI/Python/MCP│  seg → embed → cluster → reseg      │   Clusterer/Resegmenter
  default since 0.11│  default clusterer: VBx             │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
  research only     │  pipeline_v2::hybrid::HybridPipeline│
                    └─────────────────────────────────────┘
```

| Consumer | Path |
|----------|------|
| CLI `polyvoice` | **v2 + VBx** default; `--legacy` → BYO offline stack |
| FFI | v2 only |
| Python | v2 (VBx when PLDA configured) |
| MCP `polyvoice-mcp` | v2 + VBx default (`clusterer=ahc` opt-out) |
| `polyvoice-bench` | **v2 + VBx** default; `--pipeline legacy` for comparison |
| Library, no features | crate-root `Pipeline` + `StreamingPipeline` only |

## Stage graphs

### BYO offline (`src/pipeline`)

```
samples → segment_speech(VAD) → WindowIter → Embedder::embed
       → ahc::agglomerative_cluster → merge_segments → DiarizationResult
```

### Production ONNX (`src/pipeline_v2`)

```
samples → Segmenter::segment
       → primary (non-overlap) segments + optional dense embed_window_secs
       → apply_overlap_mask + Embedder::embed
       → Clusterer::cluster_with_durations (AHC | NME-SC | VBx)
       → local→global map (Hungarian co-occurrence)
       → optional Resegmenter → merge → DiarizationResult
```

### Hybrid (ablation)

Powerset times as speech regions only (labels discarded) → fixed hop windows →
`Clusterer`. Prefer main v2 + `embed_window_secs` instead.

## Name collision note

Crate root re-exports **only** the BYO types:

```rust
pub use pipeline::{Pipeline, PipelineError};
```

Production ONNX types live under `polyvoice::pipeline_v2::…`. Prefer explicit
paths in application code (`pipeline_v2::Pipeline` vs crate-root `Pipeline`).

## Related docs

- [library-mode.md](library-mode.md) — ort-free surface inventory
- [BENCHMARKS.md](BENCHMARKS.md) — DER protocol and 0.11 gate numbers
- [API.md](API.md) — public API reference
- Module contracts under `src/pipeline*/MODULE_CONTRACT.md`
