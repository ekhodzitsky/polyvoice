# Library mode (no ONNX)

polyvoice’s default feature set is empty (`default = []`). That is intentional:
the **ort-free library path** is a first-class product surface for consumers that
bring their own embedder (or use pure-Rust clustering / Energy VAD only) and do
not want the ONNX Runtime (`ort`) native dylib.

```toml
# Cargo.toml
[dependencies]
polyvoice = { version = "0.12", default-features = false }
# optional pure-Rust extras:
# polyvoice = { version = "0.12", default-features = false, features = ["clusterer", "vbx"] }
```

CI enforces that `ort` never appears in the normal dependency graph for
`--no-default-features` (and for pure-Rust feature combos such as
`clusterer,vbx`). See `scripts/check-ort-free.sh` and the `ort-free-core` CI job.

## Guaranteed public surface without `onnx`

### Always-on (no features)

| Item | Module / path | Notes |
|------|---------------|-------|
| Offline pipeline | `Pipeline`, `PipelineError` | Takes BYO `Embedder` + `VoiceActivityDetector` |
| Streaming pipeline | `streaming::StreamingPipeline`, `LatencyPreset` | Same BYO `Embedder` + VAD pattern |
| Energy VAD | `EnergyVad`, `VadConfig`, `VoiceActivityDetector`, `segment_speech` | Pure-Rust energy VAD |
| Embedding (supported) | `Embedder`, `EmbedderError`, `DummyExtractor`, `EmbedderPool` | Implement `Embedder` for BYO encoders |
| Embedding (legacy bridge) | `EmbeddingExtractor`, `EmbeddingError` | Soft-deprecated; auto-bridges to `Embedder` |
| Config / result types | `DiarizationConfig`, `ClusterConfig`, `SpeakerTurn`, `DiarizationResult`, `Segment`, `SampleRate`, … | `types` |
| AHC / k-means math | `ahc`, `kmeans`, `cluster::SpeakerCluster` | Pure-Rust clustering primitives |
| DER helpers | `compute_der`, `compute_wder`, … | Evaluation |
| Window / overlap / RTTM / WAV | `window`, `overlap`, `rttm`, `wav`, `format` | I/O and post-processing |
| Fbank features | `FbankConfig`, `FbankExtractor` | Pure-Rust front-end features |
| ASR trait stubs | `Asr`, `AsrError` | Trait surface only |
| Word midpoint labeling | `labeling::{speaker_at, speaker_at_stable, assign_speakers_by_midpoint, label_words, UncoveredPolicy}` | STT product join; see module docs for offline vs streaming |
| Example / tests | `examples/byo_embedder`, `tests/byo_embedder_library` | Mock `Embedder`, offline + streaming |

### Feature-gated pure-Rust (no `ort`)

These features compile algorithmic cores without pulling `ort`. Some also have
ONNX-backed adapters that additionally need the `onnx` feature (listed under
“Requires `onnx`” below).

| Feature | Surface | Notes |
|---------|---------|-------|
| `clusterer` | `Clusterer`, `AhcClusterer`, `MinClusterSizeClusterer`, … | Trait + AHC adapters |
| `vbx` | VBx / PLDA clustering (requires `clusterer`) | Pure-Rust ndarray; PLDA weights supplied by caller |
| `spectral` | `spectral` + `NmeScClusterer` (with `clusterer`) | Pulls `faer`, not `ort` |
| `segmentation` | `PowersetDecoder`, `Aggregator`, `Segmenter` trait, … | Decoder / aggregator without ONNX segmenter |
| `embedder` | (mostly empty flag) | ONNX adapters (`CamPlusPlus`, ResNet34, …) still need `onnx` + `embedder`; the `Embedder` **trait** itself is always-on |
| `resegmentation` | `OverlapResegmenter`, `compute_centroids`, … | Post-clustering resegmentation |
| `attribution` | `who_said_what`, `attribute_words`, … | Word → speaker join (no models) |
| `vad-earshot` | `EarshotVad` | Optional pure-Rust VAD (`earshot` crate) |
| `audio-io` | multi-format decode + resample via `wav::load_audio` | `symphonia` + `rubato`; no `ort` |

### Requires `onnx`

| Surface | Notes |
|---------|-------|
| `SileroVad` | ONNX Silero VAD |
| `FbankOnnxExtractor` | ONNX fbank embedder (`Embedder`; feature `onnx`) |
| `CamPlusPlusExtractor`, `ResNet34Adapter` | Need `onnx` + `embedder` |
| `PowersetSegmenter` | Need `onnx` + `segmentation` |
| `pipeline_v2` | Full stack: `onnx` + `download` + `segmentation` + `embedder` + `clusterer` + `resegmentation` |
| `ModelRegistry` / `download` | HTTP model registry (no `ort` by itself, but production ONNX path uses it with `onnx`) |
| `cli`, `ffi`, `mcp`, `sortformer` | App / binding surfaces that include ONNX |
| EP features (`coreml`, `nnapi`, `xnnpack`), `backend-tract` | Inference backends |

## Reference consumer pattern

A production BYO-embedder consumer typically:

1. Depends on `polyvoice` with `default-features = false` (optionally `clusterer` / `vbx`).
2. Implements **`Embedder`** with an in-tree / other-runtime model (e.g. Candle WeSpeaker).
   Prefer `Embedder` over the soft-deprecated `EmbeddingExtractor` bridge.
3. Uses `EnergyVad` (or its own VAD) with `Pipeline` offline and `StreamingPipeline` online.
4. Does **not** enable `onnx`, so no `ort` native library is linked.
5. After ASR, maps words onto diarization turns (midpoint coverage — see below).

```rust,ignore
use polyvoice::{
    DummyExtractor, Embedder, EnergyVad, Pipeline, DiarizationConfig, VadConfig,
};

// Replace DummyExtractor with your Embedder impl (Candle WeSpeaker, etc.).
let extractor = DummyExtractor::new(256);
let mut vad = EnergyVad::new(-40.0, 16_000, 512);
let result = Pipeline::new(DiarizationConfig::default(), VadConfig::default())
    .run(&samples, &extractor, &mut vad)?;
```

Runnable copy of this path (mock embedder, no models, no network):

```bash
cargo run --no-default-features --example byo_embedder
cargo test --no-default-features --test byo_embedder_library
```

### What you implement

| Method | Contract |
|--------|----------|
| `Embedder::dim` | Fixed embedding length per instance |
| `Embedder::embed` | 16 kHz mono PCM → L2-normalized `Vec<f32>` of length `dim()` |

### What you get

| Item | Notes |
|------|--------|
| `Pipeline::run` | Offline turns (`SpeakerTurn`, always `stable: true`) |
| `StreamingPipeline` | Online turns; may emit `stable: false` until the speaker cache stabilizes |
| `EnergyVad` | Speech regions for diarization only (product endpointing VAD can stay separate) |
| `ClusterConfig.threshold` | Default `0.45` — share offline and streaming for comparable granularity |

### Word → speaker labels (STT)

Use always-on [`labeling`](../src/labeling/mod.rs) helpers (no `attribution` feature):

```rust,ignore
use polyvoice::{
    UncoveredPolicy, assign_speakers_by_midpoint, label_words, speaker_at_stable,
};

// Offline / file: leave uncovered words unlabeled.
let labels = assign_speakers_by_midpoint(word_times, &turns, UncoveredPolicy::None, false);

// Streaming / WS: trailing "now" falls back to the last turn.
let labels = assign_speakers_by_midpoint(word_times, &turns, UncoveredPolicy::LastTurn, false);

// Live UI: only stable covering turns (LastTurn still uses the slice tail).
let id = speaker_at_stable(&turns, t);
```

| Path | Midpoint coverage | Uncovered word |
|------|-------------------|----------------|
| Offline / file | First turn covering word midpoint | leave unlabeled (`UncoveredPolicy::None`) |
| Streaming / WS | Same | last turn (`UncoveredPolicy::LastTurn`) |

When showing speaker IDs in a live UI, prefer turns with `stable: true` (or
`speaker_at_stable` / `stable_only: true`) so labels do not jump after cache hits.
The richer max-overlap join stays behind feature `attribution`.

### Streaming latency

```rust,ignore
use polyvoice::streaming::{LatencyPreset, StreamingPipeline};

let mut pipeline = StreamingPipeline::with_latency_preset(
    vad, extractor, LatencyPreset::Realtime, vad_config,
)?;
```

| Preset | Best for |
|--------|----------|
| `Realtime` | Live STT / WebSocket |
| `Balanced` | Default trade-off |
| `Accurate` | Higher latency, more context |

### Explicit non-goals of library mode

- No bundled WeSpeaker / Silero ONNX weights on this path.
- No `pipeline_v2` (ONNX production stack) — that is the CLI/FFI default, not BYO.
- DER quality tracks **your** embedder; library mode does not claim SOTA alone.

## Related docs

- [API reference](API.md) — full type and pipeline documentation
- [README](../README.md) — install paths including library mode
- Example: `examples/byo_embedder.rs`
- CI job `ort-free-core` / `scripts/check-ort-free.sh` — regression gate
