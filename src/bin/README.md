# src/bin/

## Purpose

CLI and agent entry points for the polyvoice speaker diarization toolkit.

## Surfaces

- **`polyvoice`** — Main CLI:
  - bare `polyvoice <audio>` / `diarize` — production path **v2 + VBx**
  - `--legacy` — Silero VAD + sliding-window embeddings + AHC
  - `--clusterer ahc|vbx` — clusterer for the v2 path (default `vbx`)
  - `download-models`, `models`, `completions`, `schema`
- **`polyvoice-bench`** — Batch DER benchmark on `{audio,rttm}` dataset dirs.
  Default `--pipeline v2` (matches CLI); pass `--pipeline legacy` for the
  pre-0.11 path. Default `--clusterer vbx`.
- **`polyvoice-mcp`** — MCP stdio server (`polyvoice.diarize`, …). Same
  production path as CLI (pipeline v2 + VBx by default).
- **`polyvoice-measure`** — Measurement / latency utilities.

Shared wiring (flag-to-config translation, pipeline construction, dataset
walking) lives in **`src/cli_common`** (compiled with `cli` or `mcp`) so each
binary stays a thin wrapper.

## Dependencies

- `src/cli_common` — shared flag parsing, pipeline construction, dataset walking.
- `src/pipeline_v2` — production ONNX orchestration (default).
- `src/pipeline` — BYO / `--legacy` path.
- `src/models` — model registry and download.
- `src/rttm` / `src/format` — output formats.
- `src/der` — DER for bench.
- `src/wav` — audio load.

## Invariants

- Binaries are thin wrappers; shared wiring lives in `src/cli_common`,
  algorithms in `lib` modules.
- Default diarization path matches shipped CLI accuracy gate (v2 + VBx).
- `polyvoice-bench` expects `audio/*.wav` + `rttm/*.rttm`.

## Verification

```bash
cargo build --release --features cli --bin polyvoice
cargo build --release --features cli --bin polyvoice-bench
cargo build --release --features mcp --bin polyvoice-mcp
cargo test --test cli_smoke_test --features cli
```

## Notes

- ONNX models are required at runtime for the production path. Use
  `download-models` or populate `~/.cache/polyvoice/models`.
- VBx PLDA is auto-downloaded via the registry when neither
  `--vbx-plda-dir` nor `POLYVOICE_VBX_PLDA_DIR` is set (fixtures under
  `fixtures/vbx-plda/` for offline/CI).
- The `balanced` profile (WeSpeaker ResNet34, 256-d) is the default.

## Per-backend RTFx (realtime factor)

`polyvoice-bench` labels every report with `resolved_execution_provider` and,
for the v2 pipeline, a per-stage `stage_timings` breakdown per file
(segmentation / embedding / clustering / resegmentation) plus aggregate
`stage_totals`. Select a backend with
`--execution-provider auto|cpu|coreml|nnapi|cuda|xnnpack`; omitted, each
pipeline keeps its shipped default (legacy embedder: cpu, v2: auto).

```bash
BENCH_ARGS="--max-files 3" scripts/bench-backends.sh data/voxconverse-test v2 cpu coreml
```

Report schema is `polyvoice-bench-v0.10`.
