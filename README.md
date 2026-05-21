# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Speaker diarization for Rust — who spoke when, without Python.
> Legacy pipeline: Silero VAD + WeSpeaker embeddings + AHC clustering.
> **New in v0.6.3**: Hybrid pipeline (Powerset VAD + ResNet34 + AHC) for long-form multi-speaker audio — API-only.
> **New in v0.6.4**: K-means auto-k clusterer (silhouette-based k selection) beats AHC by 4.65% DER on VoxConverse.

## Quick Start

```toml
[dependencies]
polyvoice = { version = "0.6", features = ["onnx"] }
```

```bash
cargo add polyvoice --features onnx
```

## Features

- **One-call pipeline** — `Pipeline::run()` wires VAD → embeddings → AHC or K-means clustering.
- **Hybrid pipeline** — `HybridPipeline` (v0.6.3, API-only) uses PowersetSegmenter as a superior VAD (overlap-aware) + global ResNet34 embedding clustering. Overcomes the 3-speaker limit of local segmentation models on long-form audio.
- **Online & offline** — `OnlineDiarizer` for streaming, `OfflineDiarizer` for batch.
- **CPU-only, ~30 MB** — ONNX Runtime, no GPU or Python runtime required.
- **Multi-language** — Rust library, Python bindings (`pip install polyvoice`), C FFI, CLI.
- **Lock-free concurrency** — `crossbeam-queue` session pool for parallel inference.
- **Parallel embedder** — `embed_batch` spreads chunks across CPU cores via `std::thread::scope`.
- **AHC O(n²)** — agglomerative clustering rewritten from cubic to quadratic; handles >500 embeddings on long recordings.
- **K-means auto-k** — silhouette-based automatic k selection with single-speaker detection. 14.12% DER on VoxConverse full (vs AHC 18.77%).
- **Hardened** — Miri (memory), Loom (concurrency), cargo-fuzz (4 targets), model signing (Minisign).

## Minimal Example (Legacy Pipeline — CLI / Python default)

```rust,no_run
use polyvoice::{Pipeline, DiarizationConfig, VadConfig, FbankOnnxExtractor, SileroVad};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ext = FbankOnnxExtractor::new(Path::new("models/wespeaker_resnet34.onnx"), 256, 4)?;
    let mut vad = SileroVad::new(Path::new("models/silero_vad.onnx"), 512)?;
    let (samples, _sr) = polyvoice::wav::read_wav(Path::new("meeting.wav"))?;
    let result = Pipeline::new(DiarizationConfig::default(), VadConfig::default())
        .run(&samples, &ext, &mut vad)?;
    for turn in &result.turns {
        println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

## Hybrid Pipeline (API-only, v0.6.3)

The hybrid pipeline is available in Rust via the `pipeline_v2::hybrid` module. It uses `PowersetSegmenter` purely for speech-region detection (including overlaps), then extracts sliding-window ResNet34 embeddings and clusters them globally with K-means auto-k (or AHC). This avoids the 3-speaker hard limit of the Powerset model.

```rust,no_run
use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::hybrid::HybridPipeline;
use polyvoice::segmentation::PowersetSegmenter;
use polyvoice::embedder::ResNet34Adapter;
use polyvoice::clusterer::KMeansClusterer;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelRegistry::default()?;
    let models = registry.ensure_for_profile(polyvoice::types::Profile::Balanced)?;

    let segmenter = PowersetSegmenter::new(&models.segmenter_path)?;
    let embedder = ResNet34Adapter::new(&models.embedder_path, 4)?;
    let clusterer = KMeansClusterer::new(20);

    let pipeline = HybridPipeline::new(
        Box::new(segmenter),
        Box::new(embedder),
        Box::new(clusterer),
    );

    let (samples, _sr) = polyvoice::wav::read_wav(Path::new("meeting.wav"))?;
    let sr = polyvoice::types::SampleRate::new(16000).unwrap();
    let result = pipeline.run(&samples, sr)?;
    for turn in &result.turns {
        println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

> **Note**: The hybrid pipeline is currently API-only. The CLI (`polyvoice diarize`) and Python bindings continue to use the legacy pipeline for stability.

## Python / C FFI

Python bindings use the **legacy** pipeline (stable default):

```python
import polyvoice
pipeline = polyvoice.Pipeline.balanced("models/")
result = pipeline.run(samples, sample_rate=16000)
for turn in result["turns"]:
    print(f"{turn['speaker']}: {turn['start']:.1f}s - {turn['end']:.1f}s")
```

```c
// cargo build --features ffi
// See include/polyvoice.h and examples/ffi_usage.c
polyvoice_pipeline_create(BALANCED, "models/", &handle);
polyvoice_pipeline_run(handle, samples, n, 16000, &json, &len);
```

## Benchmarks

| Pipeline | Dataset | DER | Speed |
|----------|---------|-----|-------|
| **Legacy** (Silero + ResNet34 + AHC) | VoxConverse (232 files) | **~14%** | 10x RT (CPU) |
| **Legacy** (Silero + ResNet34 + AHC) | AMI (16 meetings) | **~36%** | 7x RT (CPU) |
| **Hybrid** (Powerset VAD + ResNet34 + AHC) | e2e smoke (26 s clip) | **4.43%** | — |
| **Hybrid** (Powerset VAD + ResNet34 + AHC) | VoxConverse (3-file subset) | **8.27%** | — |
| **Hybrid** (Powerset VAD + ResNet34 + AHC) | VoxConverse (10-file subset) | **16.62%** | — |
| **Hybrid** (Powerset VAD + ResNet34 + **K-means**) | VoxConverse (10-file subset) | **13.48%** | — |
| **Hybrid** (Powerset VAD + ResNet34 + **K-means**) | VoxConverse (full 232 files) | **14.12%** | — |

~80% of pyannote's accuracy at 10× the speed on CPU — no GPU, no Python.

> **Note on long-form audio**: The 10-file VoxConverse subset includes one known
> outlier (`aorju`: 23 min, 12 speakers, 17% overlap → DER 52.51%). Excluding this
> file, the average DER drops to ~10.5%. The hybrid pipeline is API-only and
> optimized for typical conference/meeting recordings; extreme multi-speaker
> long-form with heavy overlap remains an active research area (VBx/PLDA).

## License

MIT
