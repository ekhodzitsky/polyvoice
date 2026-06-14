# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Speaker diarization for Rust — who spoke when, without Python.**

Beta-quality speaker diarization that runs on CPU and fits in ~30 MB, with
automatic K-means speaker count detection. See
[PRODUCTION-READINESS.md](PRODUCTION-READINESS.md) for deployment guidance
(GO for desktop and controlled internal use; NO-GO for public multi-tenant APIs).

```
Speaker_0: 0.0s - 12.3s
Speaker_1: 14.1s - 28.7s
Speaker_0: 31.2s - 45.0s
```

---

## At a glance

| | polyvoice | pyannote 3.1 | whisperX |
|--|-----------|--------------|----------|
| **VoxConverse DER**¹ | **13.83%** | ~12% | ~15% |
| **Model size** | **~30 MB** | ~100 MB | ~1 GB |
| **Runtime** | **CPU only** | GPU recommended | GPU required |
| **Dependencies** | **No Python / PyTorch**² | PyTorch + ONNX | PyTorch + faster-whisper |
| **Languages** | **Rust / Python / C / CLI** | Python only | Python only |
| **Streaming** | **Yes** | No | No |

~80% of pyannote's accuracy at **10× less RAM** and **no GPU**.
Runs at **~10× realtime** on CPU — 9.3× average over a VoxConverse subset ([artifact](benchmarks/results/voxconverse-test-10files-20260516.json)).

**Other Rust diarizers.** sherpa-rs (now archived), pyannote-rs, and speakrs are the closest
Rust options. None publishes a collar-matched VoxConverse DER, so this table compares only the
established Python systems; see [Why polyvoice](#why-polyvoice) for the maintained / pure-Rust /
streaming / four-binding differentiators.

¹ Legacy pipeline, VoxConverse-test (232 files), **0.25 s collar**. The 232-file no-collar
figure was not measured, but on a 10-file subset no-collar DER is **25.99%** vs 17.43% at
0.25 s collar — expect the strict number several points higher. Competitor figures use their
own conventions and are **not collar-matched** — compare only on a matched collar. All
polyvoice DER figures are sourced from [`tests/der_baseline.json`](tests/der_baseline.json);
see the [canonical table](#benchmarks) below.

² The C++ ONNX Runtime is downloaded at build time via the `ort` crate
(`download-binaries`); for hermetic builds use a static-linked / vendored ORT (see
[PRODUCTION-READINESS.md](PRODUCTION-READINESS.md) §2). No Python/PyTorch runtime.

---

## Install

```bash
# Rust
cargo add polyvoice --features "onnx,download"

# Python
pip install polyvoice

# CLI
cargo install polyvoice --features cli
```

## Quick start — Rust

> Note: the CLI and Python bindings default to the validated legacy pipeline. The
> builder below is the curated v2 API; for long-form meetings see
> [PRODUCTION-READINESS.md](PRODUCTION-READINESS.md).

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Models auto-download on first run.
    let registry = ModelRegistry::default()?;

    let pipeline = Pipeline::builder()
        .profile(Profile::Balanced) // auto-k speaker count via the Balanced profile
        .with_models_from(registry)
        .build()?;

    let (samples, sr_hz) = polyvoice::wav::read_wav("meeting.wav")?;
    let sr = SampleRate::new(sr_hz).ok_or("invalid sample rate")?;
    let result = pipeline.run(&samples, sr)?;

    for turn in &result.turns {
        println!("{}: {:.1}s - {:.1}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

## Quick start — Python

```python
import polyvoice

pipeline = polyvoice.Pipeline.balanced("models/")
result = pipeline.run(samples, sample_rate=16000)

for turn in result["turns"]:
    print(f"{turn['speaker']}: {turn['start']:.1f}s - {turn['end']:.1f}s")
```

## Quick start — CLI

```bash
# Download models once
polyvoice download-models --profile balanced

# Diarize
polyvoice diarize meeting.wav --output meeting.rttm
```

---

## Benchmarks

All figures below are sourced from [`tests/der_baseline.json`](tests/der_baseline.json)
(schema `polyvoice-der-baseline-v2`) and labeled with pipeline, dataset, file count, and
collar. **CI-gated** marks rows enforced by the release DER-regression gate.

| Pipeline | Dataset | Files | DER (0.25 s collar) | DER (no-collar) | CI-gated |
|----------|---------|-------|---------------------|-----------------|----------|
| Legacy (Silero + AHC) | VoxConverse-test | 232 | 13.83% | not measured | no |
| Legacy (Silero + AHC) | VoxConverse-test subset | 10 | 17.43% | 25.99% | yes |
| Legacy (Silero + AHC) | e2e smoke (26 s clip) | 1 | 6.62% | not measured | yes |
| Legacy (Silero + AHC) | AMI EN2002a (1 meeting) | 1 | 36.30% | 44.73% | yes |
| v2 (Powerset + ResNet34 + AHC) | e2e smoke (26 s clip) | 1 | 4.43% | not measured | yes |
| Hybrid (Powerset + ResNet34 + AHC) | e2e smoke (26 s clip) | 1 | 4.43% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | VoxConverse-test subset | 3 | 8.27% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | VoxConverse-test subset | 10 | 15.03% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | AMI EN2002a (1 meeting) | 1 | 24.95% | not measured | no |

Notes:

- **No-collar DER is materially higher** than the 0.25 s-collar figure (e.g. the 10-file
  legacy subset is 17.43% collar vs **25.99%** no-collar). Compare against other systems
  only on a matched collar.
- The previously headlined "14.12% (232-file, Hybrid + K-means)" number had no committed
  artifact and was withdrawn pending a reproducible, provenance-stamped re-run.
- AMI rows are a single meeting (EN2002a, ~79% overlap), not a multi-meeting average.

Automatic speaker count uses **silhouette-based k selection** with a **single-speaker
guard** (no 20-speaker predictions on 1-speaker files).

---

## What makes it different

- **Automatic speaker count** — K-means auto-k detects how many speakers are in
  the recording, matching well-tuned AHC without any manual threshold sweep.
- **Single-speaker guardrail** — embeddings too similar? Returns 1 speaker
  instead of hallucinating clusters.
- **Overlap-aware** — PowersetSegmenter detects overlapping speech regions;
  embeddings are masked to exclude overlaps before clustering.
- **Streaming & batch** — `OnlineDiarizer` for real-time, `OfflineDiarizer` for
  files.
- **Cross-platform** — Linux, macOS, Windows; x86_64 and aarch64.
- **Hardened** — Miri (memory safety), Loom (concurrency), cargo-fuzz (4
  targets), model signing (Minisign).

---

## Why polyvoice

- **Maintained, pure-Rust, streaming-capable.** The popular `sherpa-rs` bindings
  are now archived; polyvoice is an actively-maintained, pure-Rust diarization
  path (ONNX via `ort`, no C++ toolkit) with first-class streaming.
- **One library, four surfaces.** Rust + Python (maturin) + C FFI + CLI from a
  single crate — most Rust diarizers are Rust-only.
- **CPU-first, ~30 MB, MIT.** No GPU, no Python runtime, no gated model access.

Honest scope: polyvoice is **not** the accuracy leader — like-for-like no-collar
VoxConverse DER is ~mid-20s%, versus ~11% for pyannote community-1 / speakrs. It
trades a few DER points for deployability and a maintained, multi-binding SDK.
See [Benchmarks](#benchmarks) for the labeled, collar-disclosed numbers.

> **Brand note (open maintainer decision):** the name collides with ByteDance's
> "PolyVoice" speech-to-speech-translation research, so always refer to this
> project as **"polyvoice — speaker diarization for Rust"**. Registering a
> `polyvoice-rs` alias is an open decision; a crate *rename* would break
> downstreams and is out of scope.

---

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Audio Bytes │ --> │ Embedding       │ --> │ Speaker Cluster │ --> Turns
│ (f32 PCM)   │     │ Extractor       │     │ (AHC or K-means)│
└─────────────┘     └─────────────────┘     └─────────────────┘
       │                    │                       │
       v                    v                       v
  Powerset VAD      WeSpeaker ResNet34      Silhouette auto-k
  (10s windows,     (2s windows, 256-dim)   (pairwise cosine
   1s hop)                                  distance cache)
```

---

## License

MIT
