# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/ekhodzitsky/polyvoice)](https://github.com/ekhodzitsky/polyvoice/releases)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![Codecov](https://codecov.io/gh/ekhodzitsky/polyvoice/branch/master/graph/badge.svg)](https://codecov.io/gh/ekhodzitsky/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Speaker diarization for Rust — who spoke when, on CPU, without Python.**

Built for meeting-notes pipelines, voice agents, and on-prem deployments that
can't ship a PyTorch stack. One crate, four surfaces: Rust library, Python,
C FFI, and a CLI. MIT, ungated **INT8** models (~8.4 MB production pair).

![polyvoice CLI demo — real diarization run](docs/assets/demo.gif)

## Numbers

**Default stack is INT8** (`powerset_int8` + `resnet34_int8`) for every
profile (`balanced` / `mobile` / `fast`). Full-split numbers below are the
INT8 shipping path (pipeline v2 + VBx, crate 0.17+). Protocol:
[Benchmarks](docs/BENCHMARKS.md). **Linux/CPU** is the product path for
servers (powerset micro-batch N=8); Mac CoreML is a separate headline.

| Corpus | DER, forgiving (0.25 s collar) | DER, strict (collar 0) | Speed (INT8) |
|---|---:|---:|---:|
| VoxConverse-test (232 files) | **10.3 %** | **14.9 % Linux / 15.0 % CoreML** | **~82× Linux CPU / ~111× CoreML** |
| AMI-test (16 meetings) | **16.6 % Linux / 16.8 % CoreML** | **24.2 % Linux / 24.5 % CoreML** | **~95× Linux CPU / ~110–130× CoreML** |

Like-for-like (strict collar 0) VoxConverse-test **15.0 %** vs pyannote 3.1
**11.3 %** — accuracy traded for a CPU-only, MIT, **ungated** INT8 deploy.
(VoxConverse-dev FP32-era 11.4 / 7.7 % is retained in the benchmarks doc; not
re-measured on INT8 in this gate.)

## 60 seconds to first result

```bash
# 1. Get the CLI (macOS Apple Silicon here; see Install for other platforms)
curl -LO https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-macos-arm64
chmod +x polyvoice-macos-arm64

# 2. Fetch the INT8 models (~8.4 MB, MIT, no token)
./polyvoice-macos-arm64 download-models --profile balanced

# 3. Diarize
./polyvoice-macos-arm64 diarize meeting.wav --output meeting.rttm
cat meeting.rttm
```

```
SPEAKER meeting 1   0.000  12.784  <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting 1  13.005   2.530  <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting 1  15.688  10.323  <NA> <NA> SPEAKER_02 <NA> <NA>
```

A 1-hour meeting diarizes in about a minute on a laptop.

## Install

| Platform | Get it |
|---|---|
| Linux x86_64 / ARM64, macOS, Windows | [Pre-built binaries](https://github.com/ekhodzitsky/polyvoice/releases/latest) — put them on your `PATH` |
| Rust library (ONNX production) | `cargo add polyvoice --features "pipeline-full,vbx"` — crate-root `Pipeline` (v2); set `clusterer: Vbx` for CLI parity |
| Rust, no ONNX (BYO embedder) | `cargo add polyvoice --no-default-features` (extras: `clusterer,vbx`) — [library mode](docs/library-mode.md) |
| Python | `pip install polyvoice` — [python/README.md](python/README.md) |
| From source | `cargo install polyvoice --features cli` · `"cli,audio-io"` (mp3/flac/ogg, any sample rate) · `ffi` (C ABI v3) |

## Library usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::ClustererKind;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CLI / Python / FFI default is VBx. PipelineConfig::default() alone is AHC.
    let pipeline = Pipeline::builder()
        .config(PipelineConfig {
            profile: Profile::Balanced, // INT8 pair (mobile/fast are the same models)
            clusterer: ClustererKind::Vbx,
            ..PipelineConfig::default()
        })
        .with_models_from(ModelRegistry::default()?) // models auto-download
        .build()?;
    let (samples, sr) = polyvoice::wav::load_audio(std::path::Path::new("meeting.wav"))?;
    let result = pipeline.run(&samples, SampleRate::new(sr).ok_or("bad sample rate")?)?;
    for turn in &result.turns {
        println!("{}: {:.1}s - {:.1}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

Python: [python/README.md](python/README.md). Full Rust API: [docs.rs](https://docs.rs/polyvoice)
and [docs/API.md](docs/API.md).

## Why polyvoice

- **Fast on CPU.** INT8 production models (~8.4 MB); order-of **tens–hundreds×
  realtime** on a laptop CPU — no GPU. Powerset windows micro-batch (N=8)
  on non-CoreML EPs.
- **Rust-native, four surfaces.** Rust + Python + C FFI + CLI from one crate;
  no PyTorch stack. Production ONNX path uses ONNX Runtime (`ort`); the default
  feature set is empty (ort-free BYO core).
- **MIT, ungated.** No HF token, no non-commercial rider, no gated weights.
  Streaming included.
- **Honest trade-off.** Not the accuracy leader: pyannote 3.1 is ~4 DER
  points better on VoxConverse (strict collar). You trade those points for
  deployability. [Benchmarks](docs/BENCHMARKS.md) has the full protocol.

## How it works

```
audio (f32 PCM)
  → powerset neural segmentation (overlap-aware)
  → WeSpeaker ResNet34 embeddings
  → VBx clustering (AHC / K-means / NME-SC alternatives, automatic speaker count)
  → overlap resegmentation → speaker turns
```

Streaming (`streaming::StreamingPipeline`) and batch (crate-root `Pipeline`;
`pipeline::LegacyPipeline` on the ort-free BYO path), with a single-speaker
guard so quiet or single-voice audio does not hallucinate clusters.

## Status

Beta (0.x): the public API may break between minor versions — pin an exact
version in production. Deployment guidance and known gaps:
[Production readiness](PRODUCTION-READINESS.md).

## Documentation

- **[docs/README.md](docs/README.md)** — full index by audience (CLI, Rust, Python, FFI, security)
- [Benchmarks](docs/BENCHMARKS.md) — DER per corpus, speed, collar protocols, competitor context
- [API](docs/API.md) · [Pipeline architecture](docs/PIPELINE-ARCHITECTURE.md) · [Library mode (no ONNX)](docs/library-mode.md)
- [C FFI](docs/FFI.md) · [Python](python/README.md)
- [Production readiness](PRODUCTION-READINESS.md) — deployment guidance (GO / NO-GO)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

MIT

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
