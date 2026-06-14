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

~80% of pyannote's accuracy at ~10× less RAM, no GPU, no Python runtime — see
[Benchmarks](docs/BENCHMARKS.md).

## Install

```bash
cargo add polyvoice --features "onnx,download"   # Rust library
pip install polyvoice                             # Python
cargo install polyvoice --features cli            # CLI
```

## Usage

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = Pipeline::builder()
        .profile(Profile::Balanced)                  // auto speaker count
        .with_models_from(ModelRegistry::default()?) // models auto-download on first run
        .build()?;

    let (samples, sr) = polyvoice::wav::read_wav("meeting.wav")?;
    let result = pipeline.run(&samples, SampleRate::new(sr).ok_or("bad sample rate")?)?;

    for turn in &result.turns {
        println!("{}: {:.1}s - {:.1}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

```bash
polyvoice download-models --profile balanced
polyvoice diarize meeting.wav --output meeting.rttm
```

Python usage and the full API live on [docs.rs](https://docs.rs/polyvoice).

## Why polyvoice

- **Maintained, pure-Rust, streaming-capable.** The popular `sherpa-rs` bindings
  are archived; polyvoice is an actively-maintained, pure-Rust diarization path
  (ONNX via `ort`, no C++ toolkit) with first-class streaming.
- **One library, four surfaces.** Rust + Python + C FFI + CLI from a single crate.
- **CPU-first, ~30 MB, MIT.** No GPU, no Python runtime, no gated model access.

It is **not** the accuracy leader — like-for-like no-collar VoxConverse DER is
~mid-20s% versus ~11% for pyannote / speakrs. It trades a few DER points for
deployability and a maintained, multi-binding SDK.

## How it works

```
audio (f32 PCM)
  → VAD / Powerset segmentation
  → WeSpeaker embeddings
  → clustering (AHC / K-means / NME-SC, automatic speaker count)
  → speaker turns
```

Streaming (`OnlineDiarizer`) and batch (`OfflineDiarizer`), with a single-speaker
guard so quiet or single-voice audio does not hallucinate clusters.

## Documentation

- [Benchmarks](docs/BENCHMARKS.md) — collar-disclosed DER numbers and provenance
- [Production readiness](PRODUCTION-READINESS.md) — deployment guidance (GO / NO-GO)
- [Migrating from 0.5](docs/MIGRATING-FROM-0.5.md) · [Glossary](docs/GLOSSARY.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## License

MIT

---

> **Name:** this project is **polyvoice — speaker diarization for Rust**,
> unrelated to ByteDance's "PolyVoice" speech-translation research.
