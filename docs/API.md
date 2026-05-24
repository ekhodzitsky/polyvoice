# polyvoice API Reference

## Overview

`polyvoice` is a speaker diarization library for Rust. It answers the question
**"who spoke when?"** given a stream or file of audio samples.

The crate exposes two pipeline layers:

| Layer | Entry point | Status | Best for |
|-------|-------------|--------|----------|
| **Legacy** (`polyvoice::Pipeline`) | `Pipeline::new(DiarizationConfig, VadConfig)` | Stable, used by CLI & Python | General use, proven DER |
| **v2 / Hybrid** (`polyvoice::pipeline_v2`) | `HybridPipeline::new(...)` or `PipelineBuilder` | Stable (v0.6.5) | Long-form multi-speaker audio, overlap detection |

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Audio Bytes │ --> │ Embedding       │ --> │ Speaker Cluster │ --> Turns
│ (f32 PCM)   │     │ Extractor       │     │ (online/offline)│
└─────────────┘     └─────────────────┘     └─────────────────┘
```

## Choosing Online vs Offline

| Mode | Use case | Latency | Accuracy |
|------|----------|---------|----------|
| **Online** (`OnlineDiarizer`) | Real-time streaming (WebSocket, microphone) | Low (chunk-by-chunk) | Lower (no future context) |
| **Offline** (`OfflineDiarizer`) | File transcription, post-processing | High (full file) | Higher (two-pass + merge) |

## Core Types

### `SpeakerId`
Opaque `u32` wrapper identifying a speaker cluster.

### `DiarizationConfig`
Central configuration struct for the legacy pipeline:
- `threshold: f32` — cosine similarity threshold for matching to existing speaker.
- `max_speakers: usize` — hard limit on concurrent speakers.
- `window_secs: f32` — analysis window size.
- `hop_secs: f32` — step between consecutive windows.
- `min_speech_secs: f32` — minimum segment duration (post-processing).
- `max_gap_secs: f32` — merge same-speaker segments with gaps ≤ this value.
- `sample_rate: SampleRate` — validated sample rate (8000–192000 Hz).

### `DiarizationResult`
```rust
pub struct DiarizationResult {
    pub segments: Vec<Segment>,
    pub turns: Vec<SpeakerTurn>,
    pub num_speakers: usize,
}
```

## Legacy Pipeline

### `Pipeline::new(config, vad_config)`
Stable entry point used by the CLI and Python bindings.

```rust
use polyvoice::{Pipeline, DiarizationConfig, VadConfig, FbankOnnxExtractor, SileroVad};
use std::path::Path;

let ext = FbankOnnxExtractor::new(Path::new("models/wespeaker_resnet34.onnx"), 256, 4)?;
let mut vad = SileroVad::new(Path::new("models/silero_vad.onnx"), 512)?;
let result = Pipeline::new(DiarizationConfig::default(), VadConfig::default())
    .run(&samples, &ext, &mut vad)?;
```

### Embedding Extractors (Legacy)

#### `DummyExtractor`
Deterministic pseudo-random extractor for testing and benchmarking.

```rust
let extractor = DummyExtractor::new(256);
```

#### `OnnxEmbeddingExtractor` (feature `onnx`)
Raw-audio ONNX model (WeSpeaker-style). Input shape: `[1, window_samples]`.

#### `EcapaTdnnExtractor` (feature `onnx`)
ECAPA-TDNN model with built-in log-mel filterbank preprocessing.
Input shape: `[1, n_frames, n_mels]`.

### Voice Activity Detection (Legacy)

#### `EnergyVad`
Simple energy-based VAD for tests and fallback scenarios.

```rust
let mut vad = EnergyVad::new(-40.0, 16000, 512);
let segments = segment_speech(&mut vad, &samples, &config, &vad_config)?;
```

#### `SileroVad` (feature `onnx`)
ONNX-based VAD used by the legacy pipeline and CLI.

## Pipeline v2 & Hybrid (API-only, v0.6.3)

> **Note**: These APIs are available in Rust, FFI, Python, and CLI. All
> interfaces use Pipeline v2 as of v0.6.5.

### `HybridPipeline`

Combines `PowersetSegmenter` (used purely as a VAD for speech+overlap detection)
with legacy-style sliding-window ResNet34 embeddings and K-means auto-k clustering.
Overcomes the 3-speaker hard limit of the Powerset model on long-form audio.

```rust
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::clusterer::KMeansClusterer;
use polyvoice::types::SampleRate;

let segmenter = PowersetSegmenter::new("models/powerset_fp32.onnx")?;
let embedder = ResNet34Adapter::new("models/wespeaker_resnet34.onnx", 4)?;
let clusterer = KMeansClusterer::new(20);

let pipeline = HybridPipeline::new(
    Box::new(segmenter),
    Box::new(embedder),
    Box::new(clusterer),
);
let sr = SampleRate::new(16000).unwrap();
let result = pipeline.run(&samples, sr)?;
```

Key parameters:
- `window_samples`: 2 seconds of audio (default).
- `hop_samples`: 1.5 seconds (default, reduced from 0.5 s to cut embeddings ~3×).
- `max_gap_secs`: 0.5 — merge same-speaker gaps.
- `min_speech_secs`: 0.25 — filter short segments.

### `PipelineBuilder` (v2)

Profile-based builder for the full v2 pipeline (segmenter → embedder → clusterer → resegmenter):

```rust
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::Pipeline;
use polyvoice::types::{Profile, SampleRate};

let registry = ModelRegistry::default()?;
let pipeline = Pipeline::builder()
    .profile(Profile::Balanced)
    .with_models_from(registry)
    .build()?;
let sr = SampleRate::new(16000).unwrap();
let result = pipeline.run(&samples, sr)?;
```

See `docs/superpowers/specs/2026-05-07-m6a-pipeline-v2-design.md` for the full
builder specification.

## Overlap Detection

```rust
let overlaps = detect_overlaps(&result.segments);
for ov in overlaps {
    println!("Overlap at {:.2}s - {:.2}s: {:?}",
             ov.time.start, ov.time.end, ov.speakers);
}
```

## FFI

Build with `--features ffi` to generate C symbols:

```bash
cargo build --features ffi
```

See `include/polyvoice.h` and `examples/ffi_usage.c` for usage.

## Performance Tuning

1. **Use `FbankExtractor`** instead of `compute_fbank` to avoid per-call FFT allocation.
2. **Increase pool size** for ONNX extractors if you have many concurrent requests.
3. **Use `HybridPipeline`** with `embed_batch` for long recordings — parallel extraction across CPU cores.
4. **Tune `threshold`** — lower values merge more aggressively; higher values split more.
5. **Tune `max_gap_secs`** — larger gaps mean fewer turns but may miss real speaker changes.
6. **K-means `max_clusters`** — set a ceiling (e.g. 20) to prevent over-clustering on noisy embeddings. K-means auto-k uses silhouette-based selection; single-speaker files are auto-detected.
