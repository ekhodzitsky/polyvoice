# Pipeline architecture

Who calls whom in polyvoice, and which path is the production default.

For the **development process** checklist (spec → types → verify), see
[PIPELINE.md](PIPELINE.md).

## Two families (intentional)

```
                    ┌─────────────────────────────────────┐
  no onnx / BYO     │  pipeline::LegacyPipeline           │
  + streaming       │  streaming::StreamingPipeline       │── Embedder + VAD
                    │  (always compiled, default features)│
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  onnx production   │  pipeline_v2::Pipeline (+ Builder)  │── Segmenter/Embedder/
  CLI/FFI/Python/MCP│  seg → embed → cluster → reseg      │   Clusterer/Resegmenter
  default since 0.11│  default clusterer: VBx             │
                    │  re-exported at crate root as       │
                    │  `Pipeline` (full feature gate)     │
                    └─────────────────────────────────────┘
```

| Consumer | Path |
|----------|------|
| CLI `polyvoice` | **v2 + VBx** default; `--legacy` → BYO offline stack |
| FFI | v2 only |
| Python | **v2 + VBx** default (same as CLI); `clusterer="ahc"` opt-out |
| MCP `polyvoice-mcp` | v2 + VBx default (`clusterer=ahc` opt-out) |
| `polyvoice-bench` | **v2 + VBx** default; `--pipeline legacy` for comparison |
| Library, no features | `pipeline::LegacyPipeline` + `StreamingPipeline` only |
| Library ONNX | `features = ["pipeline-full", "vbx"]` → crate-root `Pipeline` |

**Library vs front doors:** CLI / Python / FFI / MCP **set** `ClustererKind::Vbx`.
`PipelineConfig::default()` is still **AHC** — library callers must set
`clusterer: ClustererKind::Vbx` for CLI parity (see crate-root README example).

## Config defaults (do not mix blindly)

| Knob | Legacy (`DiarizationConfig` / `ClusterConfig`) | v2 (`PipelineConfig`) |
|------|-----------------------------------------------|------------------------|
| Default clusterer (CLI/Python/FFI) | AHC when `--legacy` | **VBx** (front doors); library `default()` is **AHC** |
| AHC threshold | `DEFAULT_AHC_THRESHOLD` (0.45) | same constant when AHC is selected |
| `min_cluster_size` | **2** (prunes singletons on the dense-window path) | **1** (no prune; powerset + short clips) |

v2 leaves `min_cluster_size = 1` because pruning was net-negative on the powerset
pipeline (short clips collapsed to one speaker). Raise it only for split-heavy
files; VBx skips the min-size wrapper entirely (prior-driven speaker count).

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

## Name collision note

Crate root re-exports the **production v2** types under the full ONNX feature
gate:

```rust
pub use pipeline_v2::{Pipeline, PipelineConfig, PipelineError};
```

With default (ort-free) features there is no crate-root `Pipeline` at all — the
BYO entry point is `pipeline::LegacyPipeline`. Prefer explicit paths in
application code (`pipeline::LegacyPipeline` vs crate-root `Pipeline`).

## Related docs

- [library-mode.md](library-mode.md) — ort-free surface inventory
- [BENCHMARKS.md](BENCHMARKS.md) — DER protocol and 0.11 gate numbers
- [API.md](API.md) — public API reference
- Module docs live in rustdoc on the source modules (`pipeline`, `pipeline_v2`, …)
