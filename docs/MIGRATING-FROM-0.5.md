# Migrating from polyvoice 0.5 to 1.0

`polyvoice 0.6.0` introduces the v1.0 architecture: a single
`Pipeline::builder()` API, profile-based model selection, and INT8-quantized
ONNX bundles. This is intentionally a breaking change.

## Rust API

### Before (v0.5)
```rust
use polyvoice::{OfflineDiarizer, DiarizationConfig, FbankOnnxExtractor, SileroVad, VadConfig};

let extractor = FbankOnnxExtractor::new("models/wespeaker_resnet34.onnx", 256, 4)?;
let mut vad = SileroVad::new("models/silero_vad.onnx", 512)?;
let pipeline = polyvoice::Pipeline::new(DiarizationConfig::default(), VadConfig::default());
let result = pipeline.run(&samples, &extractor, &mut vad)?;
```

### After (v1.0-alpha.3)
```rust
use polyvoice::{Pipeline, ModelRegistry, Profile, SampleRate};

let registry = ModelRegistry::default()?;
let pipeline = Pipeline::builder()
    .profile(Profile::Balanced)
    .with_models_from(registry)
    .build()?;
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

## CLI

| Before                                                     | After                                                |
|------------------------------------------------------------|------------------------------------------------------|
| `polyvoice diarize meeting.wav --threshold 0.4`            | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice diarize meeting.wav --vad-threshold 0.5`        | `polyvoice diarize meeting.wav --profile balanced`   |
| `polyvoice download-models --dir ./models`                 | `polyvoice download-models --profile balanced`       |

## C FFI

The ABI was renamed and replaced. ABI v1 (`polyvoice_diarizer_*`) is removed.
ABI v2 entry points: `polyvoice_pipeline_create`, `polyvoice_pipeline_run`,
`polyvoice_pipeline_destroy`, `polyvoice_free_string`. See `include/polyvoice.h`
for the new contract.

## Removed types and replacements

| Removed                       | Replacement                              |
|-------------------------------|------------------------------------------|
| `Pipeline::new(cfg, vad_cfg)` | `Pipeline::builder()`                    |
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
