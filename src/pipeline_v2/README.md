# src/pipeline_v2

The **production ONNX** trait-based diarization pipeline (CLI / FFI / Python /
MCP default since **0.11**).

> **Default path:** powerset segmentation → ResNet34 embeddings → **VBx**
> (VB-HMM + PLDA) clustering → overlap resegmentation. Proven on full
> VoxConverse-test + AMI-test (see `docs/BENCHMARKS.md` and
> `benchmarks/results/full-der-2026-07-25/`). Opt out on the CLI with
> `--legacy` (Silero + AHC) or `--clusterer ahc`.
>
> Ort-free / bring-your-own embedder consumers use the always-on
> [`pipeline`](../pipeline/) module + `StreamingPipeline` — not this module.

## What it is

A trait-composed production pipeline:

```
samples ──▶ Segmenter ──▶ Embedder ──▶ Clusterer ──▶ Resegmenter ──▶ turns
            (Powerset)    (ResNet34)   (AHC/NME-SC/  (overlap)
                                        VBx)
```

Each stage is a trait object, so backends are swappable. The `PipelineBuilder`
wires them from a `Profile` (Mobile/Balanced, via `ModelRegistry`) or from
caller-supplied `Custom` components.

## How it differs from the BYO `pipeline`

| | BYO `pipeline` (`LegacyPipeline`) | `pipeline_v2` |
|--|----------------------------|---------------|
| Role | Ort-free library / CLI `--legacy` | ONNX production default |
| VAD / segmentation | Injected `VoiceActivityDetector` | Powerset `Segmenter` (overlap-aware) |
| Wiring | concrete generics at `run` | `Segmenter`/`Embedder`/`Clusterer`/`Resegmenter` traits |
| Overlap | none | overlap masking + resegmentation |
| Features | none required | onnx + download + segmentation + embedder + clusterer + resegmentation |

## Usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{ClustererKind, Pipeline, PipelineConfig};
use polyvoice::types::{Profile, SampleRate};

let mut cfg = PipelineConfig {
    profile: Profile::Balanced,
    clusterer: ClustererKind::Vbx, // or Ahc { threshold: 0.45 }
    ..PipelineConfig::default()
};
let pipeline = Pipeline::builder()
    .config(cfg)
    .with_models_from(ModelRegistry::default()?)
    .build()?;
let (samples, sr_hz) = polyvoice::wav::read_wav("meeting.wav")?;
let result = pipeline.run(&samples, SampleRate::new(sr_hz).ok_or("bad sr")?)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

CLI: `polyvoice meeting.wav` (v2 + VBx). Escape hatches: `--legacy`,
`--clusterer ahc`. Hidden `--v2` is a no-op kept for old scripts.

## Feature gate

The module is only compiled when **all** of `onnx + download + segmentation +
embedder + clusterer + resegmentation` are enabled (cfg-gated in `lib.rs`,
with a `compile_error!` backstop) — half-wired combos simply exclude it. The
`cli` feature also enables `vbx` for the default clusterer.

## Safety guards in `run`

- Sample rate must match the config (`UnsupportedSampleRate` otherwise).
- Audio longer than `MAX_AUDIO_SAMPLES` (~1 hour at 16 kHz) is rejected (`AudioTooLong`).
- Segments shorter than `MIN_EMBED_SECS` (0.20s) are skipped — ResNet34's
  ~8× temporal downsampling makes shorter slices collapse to NaN.
- Non-finite embeddings are dropped before clustering (NaN-collapse defense).
- Output turns are always sorted by start time.

## Known gaps

Some `ExecutionProvider` variants fall back to CPU when not compiled in.
See [TODO.md](TODO.md).

## Verification

```bash
cargo nextest run --lib pipeline_v2 \
  --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
cargo clippy --all-targets --all-features -- -D warnings
```
