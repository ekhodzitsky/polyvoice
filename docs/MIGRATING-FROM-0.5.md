# Migrating from polyvoice 0.5 to 1.0

`polyvoice 0.6.0` introduces the v1.0 architecture: a single
`pipeline_v2::Pipeline::builder()` API, profile-based model selection, and
INT8-quantized ONNX bundles. This is intentionally a breaking change.

> **Status box (read first):** this guide documents the **0.5 → 0.6** migration
> and is partially outdated — sections marked *archival* below describe APIs
> that have since been removed (`HybridPipeline`, `OnlineDiarizer`,
> `OnnxEmbeddingExtractor`, FFI ABI v2). As of **0.11+**, CLI, FFI, Python, and
> MCP default to **`pipeline_v2` + VBx**; the crate-root `Pipeline` is the v2
> pipeline (re-exported under the full ONNX feature gate), and the ort-free /
> BYO library surface is `pipeline::LegacyPipeline` (also the CLI `--legacy`
> escape hatch). Architecture map:
> [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md).
>
> The sections below retain historical 0.6.x narrative for context; prefer this
> status box and the README for what ships today.

## Rust API

### Before (v0.5)
```rust
use polyvoice::{OfflineDiarizer, DiarizationConfig, FbankOnnxExtractor, SileroVad, VadConfig};

let extractor = FbankOnnxExtractor::new("models/wespeaker_resnet34.onnx", 256, 4)?;
let mut vad = SileroVad::new("models/silero_vad.onnx", 512)?;
let pipeline = polyvoice::Pipeline::new(DiarizationConfig::default(), VadConfig::default());
let result = pipeline.run(&samples, &extractor, &mut vad)?;
```

### After (v1.0-alpha.3 — API-only)
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

### Hybrid Pipeline (v0.6.3 — archival, removed)

> **Archival:** `HybridPipeline` no longer exists. For dense-window embeddings
> on the production path, use `PipelineConfig::embed_window_secs` on the v2
> `Pipeline` instead. The example below is kept for historical context only.

For long-form multi-speaker audio, use the hybrid pipeline which treats
`PowersetSegmenter` as a superior VAD and resolves speakers globally via K-means auto-k:

```rust
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::clusterer::KMeansClusterer;
use polyvoice::types::SampleRate;

let segmenter = PowersetSegmenter::new("models/powerset_fp32.onnx")?;
let embedder = ResNet34Adapter::new("models/wespeaker_resnet34.onnx", 4)?;
let clusterer = KMeansClusterer::new(20);
let pipeline = HybridPipeline::new(Box::new(segmenter), Box::new(embedder), Box::new(clusterer));
let sr = SampleRate::new(16000).unwrap();
let result = pipeline.run(&samples, sr)?;
```

## Python API

### Before
```python
from polyvoice import Pipeline
p = Pipeline("models/")
result = p.run(samples)
```

### After
```python
import polyvoice
p = polyvoice.Pipeline.balanced("models/")
result = p.run(samples, sample_rate=16000)
print(result["num_speakers"], len(result["turns"]))
```

> **Update (0.11):** Python bindings use pipeline v2 (VBx when PLDA is set).

## CLI

| Before                                                     | After (0.11+)                                              |
|------------------------------------------------------------|------------------------------------------------------------|
| `polyvoice diarize meeting.wav --threshold 0.4`            | `polyvoice meeting.wav` (v2 + VBx default)                 |
| legacy Silero + AHC                                        | `polyvoice meeting.wav --legacy` or `--clusterer ahc`      |
| `polyvoice download-models --dir ./models`                 | `polyvoice download-models --profile balanced`             |

> **Update (0.11):** CLI default is pipeline v2 + VBx after a full-split DER
> gate. PLDA auto-downloads via the model registry when neither
> `--vbx-plda-dir` nor `POLYVOICE_VBX_PLDA_DIR` is set.

## C FFI

The ABI was renamed and replaced. ABI v1 (`polyvoice_diarizer_*`) is removed.
The current **ABI v3** entry points: `polyvoice_pipeline_create`,
`polyvoice_pipeline_run`, `polyvoice_pipeline_run_format`,
`polyvoice_pipeline_destroy`, `polyvoice_free_string`. See `include/polyvoice.h`
for the contract.

## Removed types and replacements

| Removed / renamed             | Replacement / note                       |
|-------------------------------|------------------------------------------|
| `OfflineDiarizer`             | `pipeline_v2::Pipeline::run` (ONNX) or `pipeline::LegacyPipeline` (BYO) |
| ONNX profile builder          | `pipeline_v2::Pipeline::builder()` + `PipelineConfig` |
| `DiarizationConfig`           | Still the BYO/`--legacy` config; ONNX uses `pipeline_v2::PipelineConfig` |
| `VadConfig` / `EnergyVad`     | Still public for BYO/streaming; ONNX v2 uses `Segmenter` |
| `DummyExtractor`              | Still public test/mock embedder (implements `Embedder`) |
| `OnnxEmbeddingExtractor`      | Removed; use `embedder::ResNet34Adapter` |
| `EcapaTdnnExtractor`, `EcapaMelOnnxExtractor`, `RawAudioOnnxExtractor` | use `embedder::CamPlusPlusExtractor` or `ResNet34Adapter` |
| `ClusteringBackend`           | `pipeline_v2::ClustererKind`             |
| `compute_fbank` (public)      | private; use `FbankExtractor::extract`   |

## OnlineDiarizer (archival, removed)

> **Archival:** `OnlineDiarizer` no longer exists. The current streaming entry
> point is `streaming::StreamingPipeline` with tunable `LatencyPreset`s (see
> [API.md](API.md)); for offline batch processing, use `Pipeline::builder()`.
