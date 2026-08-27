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

**Default stack is INT8 kernels** (`powerset_int8` + `resnet34_int8`, no
`libonnxruntime`) for every profile. Protocol: [Benchmarks](docs/BENCHMARKS.md).
Darwin uses Accelerate/BNNS. Linux uses pure-Rust `rten-gemm` (OpenBLAS
optional). ONNX Runtime remains `--features cli-ort`.

| Corpus | DER, forgiving (0.25 s collar) | DER, strict (collar 0) | Speed |
|---|---:|---:|---:|
| VoxConverse-test (232) | **10.3 %** Linux ort | **14.9 %** Linux ort / **15.5 %** Darwin kernels | Darwin kernels **~130×** / Linux ort **~82×** / Linux kernels **~28×** (Vox-3 smoke) |
| AMI-test (16) | **16.6 %** Linux ort | **24.2 %** Linux ort / **25.2 %** Darwin kernels | Darwin kernels **~110×** / Linux ort **~95×** / Linux kernels **~21×** (AMI-1 smoke) |

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
| Rust library (kernels, no ort) | `cargo add polyvoice --features "pipeline-native,vbx"` — crate-root `Pipeline` (v2); default clusterer is VBx |
| Rust library (ONNX Runtime) | `cargo add polyvoice --features "pipeline-full,vbx"` |
| Rust, no models (BYO embedder) | `cargo add polyvoice --no-default-features` (extras: `clusterer,vbx`) — [library mode](docs/library-mode.md) |
| Python | `pip install polyvoice` — [python/README.md](python/README.md) |
| From source | `cargo install polyvoice --features cli` · `"cli,audio-io"` · `cli-ort` (ONNX Runtime) · `cli-tract` · `ffi` |

## Library usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // PipelineConfig::default() is VBx when the `vbx` feature is on (CLI parity).
    let pipeline = Pipeline::builder()
        .config(PipelineConfig {
            profile: Profile::Balanced, // INT8 pair (mobile/fast are the same models)
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
  no PyTorch stack. CLI / FFI / MCP default is hand-written INT8 kernels
  (`cli`, no `libonnxruntime`). ONNX Runtime is `--features cli-ort` and the
  Python wheel. The published crate default feature set is empty (ort-free BYO
  core).
- **MIT, ungated.** No HF token, no non-commercial rider, no gated weights.
  Online `StreamingPipeline` is the BYO energy-VAD path; product diarization
  is batch pipeline v2.
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
