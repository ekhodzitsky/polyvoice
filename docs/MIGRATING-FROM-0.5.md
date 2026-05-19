# Migrating from polyvoice 0.5 to 1.0

`polyvoice 0.6.0` introduces the v1.0 architecture: a single
`Pipeline::builder()` API, profile-based model selection, and INT8-quantized
ONNX bundles. This is intentionally a breaking change.

> **Status as of v0.6.3**: The new `Pipeline::builder()` and `HybridPipeline`
> APIs are available in Rust but are **API-only**. The CLI (`polyvoice diarize`)
> and Python bindings continue to use the proven legacy pipeline (Silero VAD +
> ResNet34 + AHC) for stability. M6b will migrate CLI/FFI/Python to the new
> pipeline once long-form DER is fully hardened.

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

### Hybrid Pipeline (v0.6.3 — API-only)
For long-form multi-speaker audio, use the hybrid pipeline which treats
`PowersetSegmenter` as a superior VAD and resolves speakers globally via AHC:

```rust
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::clusterer::AhcClusterer;
use polyvoice::types::SampleRate;

let segmenter = PowersetSegmenter::new("models/powerset_fp32.onnx")?;
let embedder = ResNet34Adapter::new("models/wespeaker_resnet34.onnx", 4)?;
let clusterer = AhcClusterer::with_threshold(20, 0.35);
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

> Python bindings still use the legacy pipeline in v0.6.3.

## CLI

| Before                                                     | After (v0.6.3)                                       |
|------------------------------------------------------------|------------------------------------------------------|
| `polyvoice diarize meeting.wav --threshold 0.4`            | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice diarize meeting.wav --vad-threshold 0.5`        | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice download-models --dir ./models`                 | `polyvoice download-models --profile balanced`       |

> The CLI continues to run the legacy pipeline. `--profile mobile|balanced`
> resolves model paths; the actual diarization uses `SileroVad` +
> `FbankOnnxExtractor` + AHC. Hybrid pipeline CLI integration is planned for
> M6b.

## C FFI

The ABI was renamed and replaced. ABI v1 (`polyvoice_diarizer_*`) is removed.
ABI v2 entry points: `polyvoice_pipeline_create`, `polyvoice_pipeline_run`,
`polyvoice_pipeline_destroy`, `polyvoice_free_string`. See `include/polyvoice.h`
for the new contract.

## Removed types and replacements

| Removed                       | Replacement                              |
|-------------------------------|------------------------------------------|
| `Pipeline::new(cfg, vad_cfg)` | `Pipeline::builder()` (API-only)         |
| `DiarizationConfig`           | `pipeline::PipelineConfig`               |
| `VadConfig`, `EnergyVad`,    `VoiceActivityDetector` | absorbed by `Segmenter` |
| `OfflineDiarizer`             | `Pipeline::run`                          |
| `DummyExtractor`              | (test-only, no public API)               |
| `OnnxEmbeddingExtractor`      | `embedder::ResNet34Adapter`              |
| `EcapaTdnnExtractor`, `EcapaMelOnnxExtractor`, `RawAudioOnnxExtractor` | (deleted; use `embedder::CamPlusPlusExtractor` or `ResNet34Adapter`) |
| `ClusteringBackend`           | `pipeline::ClustererKind`                |
| `compute_fbank` (public)      | private; use `FbankExtractor::extract`   |

## OnlineDiarizer is deprecated

`OnlineDiarizer` remains accessible but is `#[deprecated(since = "0.6.0")]`.
The streaming pipeline is being redesigned in v1.1 with a richer latency vs.
DER tradeoff. For offline batch processing, use `Pipeline::builder()`.
