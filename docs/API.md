# polyvoice API Reference

## Overview

`polyvoice` is a speaker diarization library for Rust. It answers the question
**"who spoke when?"** given a stream or file of audio samples.

The crate exposes two intentional pipeline layers (see
[PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md)):

| Layer | Entry point | Status | Best for |
|-------|-------------|--------|----------|
| **BYO / ort-free** (`polyvoice::pipeline::LegacyPipeline`) | `LegacyPipeline::new(DiarizationConfig, VadConfig)` + inject `Embedder` | Stable library surface; CLI `--legacy` | No ONNX; custom embedders; streaming sibling |
| **ONNX production** (`polyvoice::Pipeline`, re-exported from `polyvoice::pipeline_v2`) | `Pipeline::builder()` + `ModelRegistry` | **CLI/FFI/Python/MCP default since 0.11** (v2 + VBx) | Shipped accuracy path |

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
| **Offline** (`Pipeline` v2 / `LegacyPipeline`) | File transcription, post-processing | High (full file) | Higher (two-pass + merge) |

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
`docs/BENCHMARKS.md` (latency + RTF + DER reported separately) and the
`streaming` module rustdoc. CLI: `--latency-preset realtime|balanced|accurate`.

## Core Types

### `SpeakerId`
Opaque `u32` wrapper identifying a speaker cluster.

### `DiarizationConfig`
Central configuration struct for the legacy pipeline, composed of three nested
config groups plus a DoS guard:
- `cluster: ClusterConfig` — `threshold: f32` (cosine similarity threshold for
  merging clusters; `DEFAULT_AHC_THRESHOLD` = 0.45 is the shipped default),
  `max_speakers: usize` (clustering ceiling), and `min_cluster_size` /
  `min_cluster_secs` pruning controls.
- `window: WindowConfig` — `window_secs: f32` (analysis window size),
  `hop_secs: f32` (step between consecutive windows), `sample_rate: SampleRate`
  (validated, 8000–192000 Hz).
- `speech_filter: SpeechFilterConfig` — `min_speech_secs: f32` (minimum segment
  duration, post-processing), `max_gap_secs: f32` (merge same-speaker segments
  with gaps ≤ this value).
- `max_duration_secs: f32` — maximum input length (DoS guard).

`DiarizationConfig::validate()` checks the field ranges up front and returns a
typed `ConfigError` on bad input.

### `DiarizationResult`
```rust
pub struct DiarizationResult {
    pub segments: Vec<Segment>,
    pub turns: Vec<SpeakerTurn>,
    pub num_speakers: usize,
}
```

## Library injection pipeline (`LegacyPipeline` / `StreamingPipeline`)

### Bring-your-own embedder (`Embedder`)

`LegacyPipeline` and `StreamingPipeline` accept **`E: Embedder`** — the supported,
non-deprecated library injection surface. No `onnx` feature is required; an
external Candle/tract/custom encoder implements `Embedder` and pairs with
`EnergyVad` (or another `VoiceActivityDetector`).

```rust
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::{DiarizationConfig, Embedder, EmbedderError, EnergyVad, VadConfig};

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

let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
let mut vad = EnergyVad::new(-40.0, 16_000, 512);
let result = pipeline.run(&samples, &MyEmbedder, &mut vad)?;
```

Shared encoders behind `Arc` are fine as long as `Embedder` is `Send + Sync`
(the trait requires it).

### `LegacyPipeline::new(config, vad_config)`
Stable offline entry point. CLI/Python ONNX paths use `pipeline_v2` by default;
library consumers keep this generic surface for BYO embedders.

```rust
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::{DiarizationConfig, VadConfig, DummyExtractor, EnergyVad};

let extractor = DummyExtractor::new(256);
let mut vad = EnergyVad::new(-40.0, 16_000, 512);
let result = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default())
    .run(&samples, &extractor, &mut vad)?;
```

With feature `onnx`, ONNX extractors such as `FbankOnnxExtractor` (or the
architecture adapters) implement `Embedder` directly and plug into the same
`LegacyPipeline::run`.

### Embedders and test doubles

#### `DummyExtractor`
Deterministic pseudo-random unit vectors for tests and benchmarks. Implements
`Embedder` directly.

```rust
let extractor = DummyExtractor::new(256);
assert_eq!(polyvoice::Embedder::dim(&extractor), 256);
```

#### `FbankOnnxExtractor` (feature `onnx`)
WeSpeaker-style fbank → ONNX embedder (`Embedder`; e.g. ResNet34 256-d). Prefer
architecture adapters (`ResNet34Adapter`, `CamPlusPlusExtractor`) when the
model family is fixed.

### Voice Activity Detection

#### `EnergyVad`
Simple energy-based VAD for tests and fallback scenarios.

```rust
let mut vad = EnergyVad::new(-40.0, 16000, 512);
let segments = segment_speech(&mut vad, &samples, &config, &vad_config)?;
```

#### `SileroVad` (feature `onnx`)
ONNX-based VAD used by the CLI `--legacy` path and BYO pipelines when ONNX is
enabled. Production v2 path segments with powerset (no separate Silero stage).

`VadConfig::frame_geometry(sample_rate, min_speech_secs)` derives the frame
geometry (ms per frame, silence/speech duration limits in whole frames) from
the sample rate — the single derivation point, so callers do not re-implement
the conversion.

## Pipeline v2 (production ONNX)

> **Since 0.11:** CLI, FFI, Python, and MCP default to `pipeline_v2` with the
> **VBx** clusterer. Escape hatches: CLI `--legacy` / `--clusterer ahc`.

### `PipelineBuilder` (v2 — production)

Profile-based builder for the full v2 pipeline (segmenter → embedder → clusterer → resegmenter):

```rust
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::ClustererKind;
use polyvoice::types::{Profile, SampleRate};
use polyvoice::{Pipeline, PipelineConfig};

let mut cfg = PipelineConfig {
    profile: Profile::Balanced,
    clusterer: ClustererKind::Vbx,
    ..PipelineConfig::default()
};
let pipeline = Pipeline::builder()
    .config(cfg)
    .with_models_from(ModelRegistry::default()?)
    .build()?;
let sr = SampleRate::new(16000).unwrap();
let result = pipeline.run(&samples, sr)?;
```

See the `PipelineBuilder` rustdoc for the full builder API.

### VBx tuning (`VbxClustererConfig`)

With feature `vbx`, `polyvoice::VbxClustererConfig` exposes the VBx hyperparameters
(variational-inference `VbxConfig`, AHC seed threshold, embedding scale,
minimum embedding duration) for library consumers that drive
`polyvoice::VbxClusterer` directly; the v2 `Pipeline` wires the shipped
defaults itself.

### Shared CLI wiring (`cli_common`)

With features `cli` / `mcp`, `polyvoice::cli_common` holds the flag-to-config
translation, pipeline construction, and bench-dataset walking shared by the
`polyvoice` / `polyvoice-bench` / `polyvoice-measure` / `polyvoice-mcp`
binaries, so each binary stays a thin wrapper.

## Overlap Detection

```rust
use polyvoice::overlap::detect_overlaps;

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
you bring your own embedder and want Energy VAD + `LegacyPipeline` /
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

1. **Reuse `FbankExtractor`** instead of re-creating it per call — it holds the FFT planner, so per-call allocation is avoided.
2. **Increase pool size** for ONNX extractors if you have many concurrent requests.
3. **Use `embed_window_secs`** on the v2 `PipelineConfig` for long recordings — dense-window embeddings give more robust speaker centroids at the cost of more embedder calls.
4. **Tune `threshold`** — lower values merge more aggressively; higher values split more.
5. **Tune `max_gap_secs`** — larger gaps mean fewer turns but may miss real speaker changes.
6. **K-means `max_clusters`** — set a ceiling (e.g. 20) to prevent over-clustering on noisy embeddings. K-means auto-k uses silhouette-based selection; single-speaker files are auto-detected.
