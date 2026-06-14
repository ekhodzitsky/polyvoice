# src/pipeline_v2

The **experimental** M6a v1.0 trait-based diarization pipeline.

> ⚠️ **Not the validated default.** `pipeline_v2` (and its `hybrid` path) was
> reverted from the CLI default after the **0.6.1 incident** — it degraded DER
> on long-form audio. The legacy [`pipeline`](../pipeline/) module is the
> validated default for the CLI and Python bindings. Opt in via `--v2`.

## What it is

A clean, trait-composed replacement for the legacy pipeline:

```
samples ──▶ Segmenter ──▶ Embedder ──▶ Clusterer ──▶ Resegmenter ──▶ turns
            (Powerset)    (ResNet34)   (AHC/NME-SC)  (overlap)
```

Each stage is a trait object, so backends are swappable. The `PipelineBuilder`
wires them from a `Profile` (Mobile/Balanced, via `ModelRegistry`) or from
caller-supplied `Custom` components.

## How it differs from the legacy `pipeline`

| | legacy `pipeline` | `pipeline_v2` |
|--|-------------------|---------------|
| VAD / segmentation | Silero VAD | Powerset segmentation (overlap-aware) |
| Wiring | concrete types | `Segmenter`/`Embedder`/`Clusterer`/`Resegmenter` traits |
| Overlap | detect-only | overlap masking + resegmentation |
| Status | **validated default** | **experimental (opt-in)** |

## Usage (opt-in)

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};

let registry = ModelRegistry::default()?;
let pipeline = Pipeline::builder()
    .profile(Profile::Balanced)
    .with_models_from(registry)
    .build()?;
let (samples, sr_hz) = polyvoice::wav::read_wav("meeting.wav")?;
let result = pipeline.run(&samples, SampleRate::new(sr_hz).ok_or("bad sr")?)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

CLI: `polyvoice diarize meeting.wav --v2`.

## Feature gate

The module `compile_error!`s unless **all** of `onnx + download + segmentation +
embedder + clusterer + resegmentation` are enabled — half-wired combos cannot
ship.

## Safety guards in `run`

- Sample rate must match the config (`UnsupportedSampleRate` otherwise).
- Segments shorter than `MIN_EMBED_SECS` (0.20s) are skipped — ResNet34's
  ~8× temporal downsampling makes shorter slices collapse to NaN.
- Non-finite embeddings are dropped before clustering (NaN-collapse defense).
- Output turns are always sorted by start time.

## Known gaps

`ExecutionProvider::{Nnapi, XnnPack}` are accepted but **not wired** into `ort`
yet (only CoreML is) — they fall back to CPU. See [TODO.md](TODO.md).

## Verification

```bash
cargo nextest run --lib pipeline_v2 \
  --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
cargo clippy --all-targets --all-features -- -D warnings
```
