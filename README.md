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
C FFI, and a CLI. MIT, ungated models, ~30 MB (8.4 MB with the INT8 profile).

![polyvoice CLI demo — real diarization run](docs/assets/demo.gif)

## Numbers (v0.14.0, Apple M1 Pro, one file at a time)

| Corpus | DER, forgiving (0.25 s collar) | DER, strict (collar 0) | Speed |
|---|---:|---:|---:|
| VoxConverse-test (232 files) | **10.5 %** | **15.2 %** | **53× realtime** |
| VoxConverse-dev (216 files) | **7.7 %** | 11.4 % | 56× realtime |
| AMI-test (16 meetings) | **15.7 %** | 23.4 % | **68× realtime** |

INT8 `--profile fast`: **~83× realtime** — DER at parity on VoxConverse-style
audio (+0.2 pp), ~+2 pp caveat on meeting audio; see
[Benchmarks](docs/BENCHMARKS.md). Competitors don't publish CPU RTF figures;
for orientation, WhisperX runs slower than real time (RTF > 1) on CPU.

Like-for-like (strict collar 0, overlap scored) VoxConverse-test DER is
**15.2 %** versus pyannote 3.1's **11.3 %** — a few DER points traded for a
CPU-only, MIT, **ungated** engine that needs no Python. Full protocols and
provenance: [Benchmarks](docs/BENCHMARKS.md).

## 60 seconds to first result

```bash
# 1. Get the CLI (macOS Apple Silicon here; see Install for other platforms)
curl -LO https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-macos-arm64
chmod +x polyvoice-macos-arm64

# 2. Fetch the models (~30 MB, MIT, no token)
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
| Rust library | `cargo add polyvoice --features "onnx,download"` |
| Rust, no ONNX (BYO embedder) | `cargo add polyvoice --no-default-features` (extras: `clusterer,vbx`) — [library mode](docs/library-mode.md) |
| Python | `pip install polyvoice` |
| From source | `cargo install polyvoice --features cli` · `"cli,audio-io"` (mp3/flac/ogg, any sample rate) · `ffi` (C ABI v3) |

## Library usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = Pipeline::builder()
        .config(PipelineConfig {
            profile: Profile::Balanced, // or Profile::Fast for the INT8 pair
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

Python usage and the full API live on [docs.rs](https://docs.rs/polyvoice).

## Why polyvoice

- **Fast on CPU.** 53–68× realtime on a laptop M1 Pro (~83× with the INT8
  `fast` profile) — no GPU, no batching tricks.
- **Pure Rust, four surfaces.** Rust + Python + C FFI + CLI from a single
  crate; no PyTorch, no Python runtime, `sherpa-rs` is archived.
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

- [Benchmarks](docs/BENCHMARKS.md) — DER per corpus, speed, collar protocols, competitor context
- [API](docs/API.md) · [Pipeline architecture](docs/PIPELINE-ARCHITECTURE.md) · [Library mode (no ONNX)](docs/library-mode.md)
- [Production readiness](PRODUCTION-READINESS.md) — deployment guidance (GO / NO-GO)
- [Migrating from 0.5](docs/MIGRATING-FROM-0.5.md) · [Glossary](docs/GLOSSARY.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

MIT

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
