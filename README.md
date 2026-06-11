# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Speaker diarization for Rust — who spoke when, without Python.**

Production-ready speaker diarization that runs on CPU, fits in 30 MB, and
outperforms AHC clustering with automatic K-means speaker count detection.

```
Speaker_0: 0.0s - 12.3s
Speaker_1: 14.1s - 28.7s
Speaker_0: 31.2s - 45.0s
```

---

## At a glance

| | polyvoice | pyannote 3.1 | whisperX |
|--|-----------|--------------|----------|
| **VoxConverse DER** | **14.12%** | ~12% | ~15% |
| **Model size** | **~30 MB** | ~100 MB | ~1 GB |
| **Runtime** | **CPU only** | GPU recommended | GPU required |
| **Dependencies** | **Zero (ONNX)** | PyTorch + ONNX | PyTorch + faster-whisper |
| **Languages** | **Rust / Python / C / CLI** | Python only | Python only |
| **Streaming** | **Yes** | No | No |

~80% of pyannote's accuracy at **10× less RAM** and **no GPU**.
Runs at **~10× realtime** on CPU — 9.3× average over a VoxConverse subset ([artifact](benchmarks/results/voxconverse-test-10files-20260516.json)).

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

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::clusterer::KMeansClusterer;
use polyvoice::types::SampleRate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Models auto-download on first run
    let registry = ModelRegistry::default()?;
    let models = registry.ensure_for_profile(polyvoice::types::Profile::Balanced)?;

    let segmenter = PowersetSegmenter::new(&models.segmenter_path)?;
    let embedder = ResNet34Adapter::new(&models.embedder_path, 4)?;
    let clusterer = KMeansClusterer::new(20); // auto-k via silhouette

    let pipeline = HybridPipeline::new(
        Box::new(segmenter),
        Box::new(embedder),
        Box::new(clusterer),
    );

    let (samples, _sr) = polyvoice::wav::read_wav("meeting.wav")?;
    let result = pipeline.run(&samples, SampleRate::new(16000).unwrap())?;

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

| Pipeline | Dataset | Files | DER | Notes |
|----------|---------|-------|-----|-------|
| **Hybrid + K-means** | VoxConverse-test | 232 | **14.12%** | Auto-k, no threshold tuning |
| Hybrid + AHC | VoxConverse-test | 232 | 18.77% | Manual threshold 0.40 |
| Legacy (Silero + AHC) | VoxConverse-test | 232 | ~14% | Baseline pipeline |
| **Hybrid + K-means** | VoxConverse-test | 10 | **13.48%** | Subset |
| Hybrid + AHC | VoxConverse-test | 10 | 15.03% | Subset |
| **Hybrid + K-means** | e2e smoke | 1 | **4.43%** | 26 s clip |

K-means auto-k uses **silhouette-based k selection** with **single-speaker
detection** (no more 20-speaker predictions on 1-speaker files). It beats AHC
by **4.65% DER** on the full VoxConverse benchmark without any manual threshold
tuning.

---

## What makes it different

- **Automatic speaker count** — K-means auto-k detects how many speakers are in
  the recording. No more guessing thresholds.
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
