# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Speaker diarization for Rust — who spoke when, on CPU, without Python.**

ONNX-powered, ~30 MB of models, MIT and ungated. One crate, four surfaces:
Rust library, Python, C FFI, and a CLI.

```
Speaker_0: 0.0s - 12.3s
Speaker_1: 14.1s - 28.7s
Speaker_0: 31.2s - 45.0s
```

## Numbers (v0.14.0, Apple M1 Pro, one file at a time)

| Corpus | DER (collar 0.25) | DER (collar 0, strict) | Speed |
|---|---:|---:|---:|
| VoxConverse-test (232 files) | **10.5 %** | **15.2 %** | **53× realtime** |
| VoxConverse-dev (216 files) | **7.7 %** | 11.4 % | 56× realtime |
| AMI-test (16 meetings) | **15.7 %** | 23.4 % | **68× realtime** |
| + `--profile fast` (INT8) | — | — | **~83× realtime** |

Like-for-like (collar 0, overlap scored) VoxConverse-test DER is **15.2 %**
versus pyannote 3.1's **11.3 %** — a few DER points traded for a CPU-only,
MIT, **ungated** engine that needs no Python. Protocols, decomposition, and
provenance: [Benchmarks](docs/BENCHMARKS.md).

## Install

### Pre-built CLI binary

```bash
# Linux x86_64
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-linux-x86_64
chmod +x polyvoice && sudo mv polyvoice /usr/local/bin/

# Linux ARM64
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-linux-aarch64
chmod +x polyvoice && sudo mv polyvoice /usr/local/bin/

# macOS (Apple Silicon)
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-macos-arm64
chmod +x polyvoice && sudo mv polyvoice /usr/local/bin/
```

Windows: download [`polyvoice-windows-x86_64.exe`](https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-windows-x86_64.exe) from the [Releases](https://github.com/ekhodzitsky/polyvoice/releases) page.

### Rust library

```bash
# Full ONNX path (powerset segmentation, WeSpeaker embeddings, model download)
cargo add polyvoice --features "onnx,download"

# Library mode (no ONNX): BYO embedder + pure-Rust VAD/clustering, `ort` never linked
cargo add polyvoice --no-default-features   # extras: clusterer,vbx
```

### Python

```bash
pip install polyvoice
```

### From source

```bash
cargo install polyvoice --features cli              # CLI, WAV 16 kHz mono
cargo install polyvoice --features "cli,audio-io"   # + mp3/flac/ogg/m4a/aac, any sample rate
cargo build --release --features ffi                # C FFI shared library (ABI v3)
```

## Usage

```bash
polyvoice download-models --profile balanced
polyvoice diarize meeting.wav --output meeting.rttm
# Faster INT8 models: --profile fast   |   cosine AHC: --clusterer ahc   |   old path: --legacy
```

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
  `fast` profile) — a 1-hour meeting diarizes in about a minute, no GPU.
- **Pure Rust, four surfaces.** Rust + Python + C FFI + CLI from a single
  crate; no PyTorch, no Python runtime, `sherpa-rs` is archived.
- **MIT, ungated, ~30 MB.** No HF token, no non-commercial rider, no GPU
  requirement. Streaming included.
- **Honest trade-off.** Not the accuracy leader: pyannote 3.1 is ~4 DER
  points better on VoxConverse (collar 0). You trade those points for
  deployability. [Benchmarks](docs/BENCHMARKS.md) has the full protocol.

## How it works

```
audio (f32 PCM)
  → powerset neural segmentation
  → WeSpeaker ResNet34 embeddings
  → VBx clustering (AHC / K-means / NME-SC alternatives, automatic speaker count)
  → speaker turns
```

Streaming (`streaming::StreamingPipeline`) and batch (crate-root `Pipeline`;
`pipeline::LegacyPipeline` on the ort-free BYO path), with a single-speaker
guard so quiet or single-voice audio does not hallucinate clusters.

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
