# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Speaker diarization for Rust — who spoke when, on CPU, without Python.**

Beta-quality, ONNX-powered, ~30 MB. Embeds into any Rust app, with Python, C, and
CLI bindings.

```
Speaker_0: 0.0s - 12.3s
Speaker_1: 14.1s - 28.7s
Speaker_0: 31.2s - 45.0s
```

Like-for-like (collar 0, overlap-scored) VoxConverse-test DER is **15.4%**
(v2+VBx default) vs pyannote 3.1's **11.3%** — a few DER points traded for a
CPU-only, MIT, **ungated** engine that needs no Python — see
[Benchmarks](docs/BENCHMARKS.md).

## Install

### Pre-built CLI binary

Download the latest binary for your platform from the [GitHub Releases](https://github.com/ekhodzitsky/polyvoice/releases) page:

```bash
# Linux x86_64
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-linux-x86_64
chmod +x polyvoice
sudo mv polyvoice /usr/local/bin/

# Linux ARM64
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-linux-aarch64
chmod +x polyvoice
sudo mv polyvoice /usr/local/bin/

# macOS (Apple Silicon)
curl -L -o polyvoice https://github.com/ekhodzitsky/polyvoice/releases/latest/download/polyvoice-macos-arm64
chmod +x polyvoice
sudo mv polyvoice /usr/local/bin/
```

### Docker

No pre-built image is published — build locally from the repo root (the same
image CI smoke-tests):

```bash
docker build -t polyvoice .
docker run --rm -v "$(pwd):/work" polyvoice diarize /work/meeting.wav --output /work/meeting.rttm
```

### Rust library

```bash
# Full ONNX path (Silero VAD, WeSpeaker embeddings, model download)
cargo add polyvoice --features "onnx,download"
```

#### Library mode (no ONNX)

For BYO-embedder consumers (your own speaker embeddings + pure-Rust VAD /
clustering), disable default features so `ort` is never linked:

```bash
cargo add polyvoice --no-default-features
# optional pure-Rust extras: --features clusterer,vbx
```

You get `LegacyPipeline`, `StreamingPipeline`, `EnergyVad`, the `Embedder` trait
(implement it with Candle, tract, or any other backend), and pure clustering
math — no ONNX Runtime dylib. Runnable mock: `cargo run --no-default-features
--example byo_embedder`. Surface inventory and the CI gate that keeps this path
ort-free: [docs/library-mode.md](docs/library-mode.md).

### Python

```bash
pip install polyvoice
```

### From source

```bash
# CLI (WAV 16 kHz mono input). Feature `cli` includes VBx (default clusterer).
cargo install polyvoice --features cli

# CLI + any-format audio (mp3/flac/ogg/m4a/aac, any sample rate → 16 kHz mono)
cargo install polyvoice --features "cli,audio-io"

# C FFI shared library (ABI v3). Feature `ffi` includes VBx as well.
cargo build --release --features ffi
```

## Usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::ClustererKind;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cfg = PipelineConfig {
        profile: Profile::Balanced,
        clusterer: ClustererKind::Vbx, // or Ahc { threshold: polyvoice::DEFAULT_AHC_THRESHOLD }
        ..PipelineConfig::default()
    };
    // VBx PLDA: optional cfg.vbx_plda_dir / POLYVOICE_VBX_PLDA_DIR; otherwise
    // the registry auto-downloads the six .npy files on first use.
    let pipeline = Pipeline::builder()
        .config(cfg)
        .with_models_from(ModelRegistry::default()?) // models (incl. PLDA) auto-download
        .build()?;

    // Pipeline-ready mono 16 kHz (with feature `audio-io`, also mp3/flac/… + resample).
    // Without that feature, load_audio accepts 16 kHz WAV only.
    let (samples, sr) = polyvoice::wav::load_audio(std::path::Path::new("meeting.wav"))?;
    let result = pipeline.run(&samples, SampleRate::new(sr).ok_or("bad sample rate")?)?;

    for turn in &result.turns {
        println!("{}: {:.1}s - {:.1}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

```bash
polyvoice download-models --profile balanced
# Default path: pipeline v2 + VBx (PLDA auto-downloads via the model registry;
# optional override: --vbx-plda-dir / POLYVOICE_VBX_PLDA_DIR — see docs/vbx-plda-release.md)
polyvoice diarize meeting.wav --output meeting.rttm
# Cosine AHC instead of VBx: --clusterer ahc   |   old path: --legacy
# With a build that includes `audio-io`:
# polyvoice diarize meeting.mp3 --output meeting.rttm
```

Python usage and the full API live on [docs.rs](https://docs.rs/polyvoice).

## Why polyvoice

- **Maintained, pure-Rust, streaming-capable.** The popular `sherpa-rs` bindings
  are archived; polyvoice is an actively-maintained, pure-Rust diarization path
  (ONNX via `ort`, no C++ toolkit) with first-class streaming.
- **One library, four surfaces.** Rust + Python + C FFI + CLI from a single crate.
- **CPU-first, ~30 MB, MIT.** No GPU, no Python runtime, no gated model access.

It is **not** the accuracy leader — like-for-like (collar 0, overlap-scored)
VoxConverse-test DER is **15.4%** (v2+VBx default) versus **11.3%** for
pyannote 3.1. It trades those DER points for deployability: a pure-Rust, CPU,
MIT, **ungated** engine (pyannote's weights are gated behind an HF token) with
four bindings and streaming.

## How it works

```
audio (f32 PCM)
  → VAD / Powerset segmentation
  → WeSpeaker embeddings
  → clustering (VBx default; AHC / K-means / NME-SC alternatives, automatic speaker count)
  → speaker turns
```

Streaming (`streaming::StreamingPipeline`) and batch (crate-root `Pipeline`;
`pipeline::LegacyPipeline` on the ort-free BYO path), with a single-speaker
guard so quiet or single-voice audio does not hallucinate clusters.

## Documentation

- [Library mode (no ONNX)](docs/library-mode.md) — ort-free surface for BYO embedders
- [Pipeline architecture](docs/PIPELINE-ARCHITECTURE.md) — BYO vs production ONNX paths
- [Benchmarks](docs/BENCHMARKS.md) — collar-disclosed DER numbers and provenance
- [Production readiness](PRODUCTION-READINESS.md) — deployment guidance (GO / NO-GO)
- [Migrating from 0.5](docs/MIGRATING-FROM-0.5.md) · [Glossary](docs/GLOSSARY.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

MIT

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
