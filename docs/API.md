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
| **Online** (`StreamingPipeline`) | Real-time streaming (WebSocket, microphone) | Tunable via `LatencyPreset` | Lower (no future context) |
| **Offline** (`Pipeline` / `pipeline_v2`) | File transcription, post-processing | High (full file) | Higher (two-pass + merge) |

### Streaming latency presets

```rust
use polyvoice::streaming::{LatencyPreset, StreamingPipeline};

let mut pipeline = StreamingPipeline::with_latency_preset(
    vad, extractor, LatencyPreset::Realtime, vad_config,
)?;
```

| Preset | window | hop | cache | input-buffer budget @16 kHz |
|--------|--------|-----|-------|-----------------------------|
| `realtime` | 1.0 s | 0.5 s | 16 | ≈ 1.03 s |
| `balanced` | 1.5 s | 0.75 s | 32 | ≈ 1.53 s (default) |
| `accurate` | 2.0 s | 1.0 s | 64 | ≈ 2.28 s |

Turns may carry `stable: false` while a speaker is still provisional; once
`stable: true`, that speaker ID is immutable for the session. See
`src/streaming/README.md` and `docs/BENCHMARKS.md` (latency + RTF + DER
reported separately). CLI: `--latency-preset realtime|balanced|accurate`.

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

## Library injection pipeline (`Pipeline` / `StreamingPipeline`)

### Bring-your-own embedder (`Embedder`)

`Pipeline` and `StreamingPipeline` accept **`E: Embedder`** — the supported,
non-deprecated library injection surface. No `onnx` feature is required; an
external Candle/tract/custom encoder implements `Embedder` and pairs with
`EnergyVad` (or another `VoiceActivityDetector`).

```rust
use polyvoice::{
    DiarizationConfig, Embedder, EmbedderError, EnergyVad, Pipeline, VadConfig,
};

struct MyEmbedder;

impl Embedder for MyEmbedder {
    fn dim(&self) -> usize { 256 }
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        // Run your encoder; return an L2-normalized vector of length dim().
        let _ = audio;
        let mut v = vec![0.0f32; 256];
        v[0] = 1.0;
        Ok(v)
    }
}

let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());
let mut vad = EnergyVad::new(-40.0, 16_000, 512);
let result = pipeline.run(&samples, &MyEmbedder, &mut vad)?;
```

Shared encoders behind `Arc` are fine as long as `Embedder` is `Send + Sync`
(the trait requires it). Legacy `EmbeddingExtractor` implementors still compile
against these pipelines through an automatic bridge.

### `Pipeline::new(config, vad_config)`
Stable offline entry point. CLI/Python ONNX paths use `pipeline_v2` by default;
library consumers keep this generic surface for BYO embedders.

```rust
use polyvoice::{Pipeline, DiarizationConfig, VadConfig, DummyExtractor, EnergyVad};

let extractor = DummyExtractor::new(256);
let mut vad = EnergyVad::new(-40.0, 16_000, 512);
let result = Pipeline::new(DiarizationConfig::default(), VadConfig::default())
    .run(&samples, &extractor, &mut vad)?;
```

With feature `onnx`, the same `Pipeline::run` accepts ONNX wrappers that
implement the legacy extractor trait (bridged to `Embedder`), e.g.
`FbankOnnxExtractor` / `OnnxEmbeddingExtractor`.

### Embedders and test doubles

#### `DummyExtractor`
Deterministic pseudo-random unit vectors for tests and benchmarks. Implements
the legacy extractor trait and therefore `Embedder` via the bridge.

```rust
let extractor = DummyExtractor::new(256);
assert_eq!(polyvoice::Embedder::dim(&extractor), 256);
```

#### `OnnxEmbeddingExtractor` (feature `onnx`)
Raw-audio ONNX model (WeSpeaker-style). Input shape: `[1, window_samples]`.
Legacy; prefer `embedder::ResNet34Adapter` when using the v1.0 ONNX stack.

#### `FbankOnnxExtractor` (feature `onnx`)
WeSpeaker-style fbank → ONNX embedder (e.g. ResNet34, 256-d).

### Voice Activity Detection

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

See the `PipelineBuilder` rustdoc for the full builder API.

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

## Library mode (no ONNX)

`default = []` is intentional. With no features (or pure-Rust features such as
`clusterer` / `vbx` only), polyvoice never depends on `ort`. Use this path when
you bring your own embedder and want Energy VAD + `Pipeline` /
`StreamingPipeline` without a native ONNX Runtime dylib.

Inventory of always-on vs feature-gated pure-Rust vs `onnx`-gated APIs:
**[docs/library-mode.md](library-mode.md)**. CI job `ort-free-core` enforces the
ort-free graph on every PR.

## WebAssembly

The pure-Rust algorithmic core compiles for `wasm32-unknown-unknown` when the
ONNX-backed default features are disabled:

```bash
cargo check --target wasm32-unknown-unknown --no-default-features --lib
```

ONNX-based profiles require an execution provider that supports the target
platform. The CI job `wasm32-smoke` verifies this build on every push.

## Performance Tuning

1. **Use `FbankExtractor`** instead of `compute_fbank` to avoid per-call FFT allocation.
2. **Increase pool size** for ONNX extractors if you have many concurrent requests.
3. **Use `HybridPipeline`** with `embed_batch` for long recordings — parallel extraction across CPU cores.
4. **Tune `threshold`** — lower values merge more aggressively; higher values split more.
5. **Tune `max_gap_secs`** — larger gaps mean fewer turns but may miss real speaker changes.
6. **K-means `max_clusters`** — set a ceiling (e.g. 20) to prevent over-clustering on noisy embeddings. K-means auto-k uses silhouette-based selection; single-speaker files are auto-detected.
