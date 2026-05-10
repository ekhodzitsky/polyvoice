# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Speaker diarization for Rust — who spoke when, without Python.
> Silero VAD + WeSpeaker embeddings + AHC clustering in a single call.

## Quick Start

```toml
[dependencies]
polyvoice = { version = "0.6", features = ["onnx"] }
```

```bash
cargo add polyvoice --features onnx
```

## Features

- **One-call pipeline** — `Pipeline::run()` wires VAD → embeddings → AHC clustering.
- **Online & offline** — `OnlineDiarizer` for streaming, `OfflineDiarizer` for batch.
- **CPU-only, ~30 MB** — ONNX Runtime, no GPU or Python runtime required.
- **Multi-language** — Rust library, Python bindings (`pip install polyvoice`), C FFI, CLI.
- **Lock-free concurrency** — `crossbeam-queue` session pool for parallel inference.
- **Hardened** — Miri (memory), Loom (concurrency), cargo-fuzz (4 targets), model signing (Minisign).

## Minimal Example

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

## Python / C FFI

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

| Dataset | DER | Speed |
|---------|-----|-------|
| VoxConverse (232 files) | **~14%** | 10x RT (CPU) |
| AMI (16 meetings) | **~23%** | 7x RT (CPU) |

~80% of pyannote's accuracy at 10× the speed on CPU — no GPU, no Python.

## License

MIT
