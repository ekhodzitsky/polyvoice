# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![PyPI](https://img.shields.io/pypi/v/polyvoice)](https://pypi.org/project/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Speaker diarization for Rust — who spoke when, without Python.**
>
> Silero VAD + WeSpeaker embeddings + AHC clustering in a single `Pipeline::run()` call.

```
Input:  14 seconds of two-speaker audio (16 kHz mono WAV)
Output: SPEAKER_00: 0.10s -  7.60s
        SPEAKER_01: 8.10s - 14.10s
```

## Quick start

### 1. Add the dependency

```toml
[dependencies]
polyvoice = { version = "0.5", features = ["onnx"] }
```

### 2. Download models

```bash
bash scripts/download-models.sh
# Downloads WeSpeaker ResNet34 (25 MB) and Silero VAD v5 (2.2 MB) to models/
```

### 3. Run the pipeline

```rust,no_run
use polyvoice::{
    Pipeline, DiarizationConfig, VadConfig,
    FbankOnnxExtractor, SileroVad,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load models
    let extractor = FbankOnnxExtractor::new(
        Path::new("models/wespeaker_resnet34.onnx"),
        256, // embedding dim
        4,   // ONNX session pool size
    )?;
    let mut vad = SileroVad::new(Path::new("models/silero_vad.onnx"), 512)?;

    // Configure and run
    let pipeline = Pipeline::new(
        DiarizationConfig::default(),
        VadConfig::default(),
    );
    let (samples, _sr) = polyvoice::wav::read_wav(Path::new("meeting.wav"))?;
    let result = pipeline.run(&samples, &extractor, &mut vad)?;

    for turn in &result.turns {
        println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
    }
    Ok(())
}
```

## Python

```bash
pip install polyvoice
```

Or build from source:

```bash
cd python
maturin develop --release
```

```python
import polyvoice

pipeline = polyvoice.Pipeline("models/")
turns = pipeline("meeting.wav")

for turn in turns:
    print(f"{turn.speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

## CLI

```bash
cargo install polyvoice --features cli

polyvoice download-models
polyvoice diarize meeting.wav
polyvoice diarize meeting.wav --format json
polyvoice diarize meeting.wav --format rttm --max-speakers 4
```

## How it works

```
WAV / PCM audio (16 kHz mono)
       |
       v
+-------------+     +------------------+     +---------+
|  Silero VAD |---->| WeSpeaker        |---->|   AHC   |---> Speaker turns
|  (speech    |     | ResNet34         |     | cluster |
|   regions)  |     | (256-d embed.)   |     |         |
+-------------+     +------------------+     +---------+
                     fbank + CMVN           cosine similarity
                     lock-free pool         threshold merging
```

**VAD** detects speech regions, skipping silence. **WeSpeaker** extracts 256-dimensional speaker embeddings from log-mel filterbank features (80-bin, CMVN-normalized). **AHC** clusters embeddings by cosine similarity into speaker groups. The `Pipeline` wires it all together.

## Comparison with pyannote

| | polyvoice | pyannote |
|---|---|---|
| Language | Rust | Python |
| Runtime | ONNX Runtime | PyTorch |
| GIL-free | Yes | No |
| Binary size | ~30 MB (with models) | ~2 GB (torch + models) |
| Deploy | Single binary / C FFI | Python env + pip |
| Concurrent sessions | Lock-free session pool | Thread-limited |
| Streaming | `OnlineDiarizer` built-in | Third-party wrappers |

pyannote is the gold standard for accuracy. polyvoice trades some accuracy for deployment simplicity: no Python runtime, no GPU required, ~30 MB total.

## Minimum Supported Rust Version (MSRV)

1.85 (Rust 2024 edition).

## Accuracy (DER benchmarks)

Evaluated with 0.25s collar on standard diarization benchmarks:

### VoxConverse (232 files, 43.5 hours — broadcast, meetings, interviews)

| System | DER | Miss | FA | Confusion | Speed |
|--------|-----|------|-----|-----------|-------|
| **polyvoice** (AHC, t=0.4) | **16.4%** | 3.9% | 3.2% | 9.3% | **10.6x RT (CPU)** |
| pyannote 3.0 | ~11% | — | — | — | ~1x RT (GPU) |

### AMI (16 meetings, 9 hours — meeting room recordings)

| System | DER | Miss | FA | Confusion | Speed |
|--------|-----|------|-----|-----------|-------|
| **polyvoice** (AHC, t=0.4) | **27.5%** | 17.7% | 2.2% | 7.6% | 7x RT (CPU) |
| pyannote 3.0 | ~18% | — | — | — | ~1x RT (GPU) |
| Simple i-vector + AHC | ~33% | — | — | — | — |

polyvoice delivers **~80% of pyannote's accuracy at 10x the speed on CPU alone** — no GPU, no Python, ~30 MB total. The accuracy gap comes from neural end-to-end training and overlap-aware resegmentation, which polyvoice doesn't do yet.

```bash
# Reproduce benchmarks
bash scripts/download-ami-test.sh
cargo run --release --features cli --bin polyvoice-bench -- data/ami-test

bash scripts/download-voxconverse-test.sh
cargo run --release --features cli --bin polyvoice-bench -- data/voxconverse-test --threshold 0.4
```

## Features

- **Pipeline API** — `Pipeline::run()` for one-call diarization with VAD + embeddings + clustering.
- **Online & Offline** — `OnlineDiarizer` for real-time streaming, `OfflineDiarizer` for batch files.
- **ONNX-powered** — WeSpeaker and ECAPA-TDNN extractors with 80-bin log-mel fbank + CMVN.
- **Lock-free session pool** — `crossbeam-queue` backed pool for concurrent ONNX inference.
- **Silero VAD** — integrated voice activity detection with stateful LSTM context.
- **Overlap detection** — find regions where multiple speakers talk simultaneously.
- **Word alignment** — assign speaker IDs to transcript words by timestamp.
- **Python bindings** — `pip install polyvoice`, 3-line API via PyO3/maturin.
- **CLI** — `polyvoice diarize meeting.wav` with text/json/rttm output.
- **C FFI** — drop-in `.so`/`.dylib`/`.dll` for Go, Node.js, C++ callers.
- **Safety verified** — Miri (memory), Loom (concurrency), cargo-fuzz (inputs), across Linux/macOS/Windows.

## Configuration

```rust
use polyvoice::{DiarizationConfig, VadConfig, SampleRate};

let config = DiarizationConfig {
    threshold: 0.4,           // cosine similarity threshold
    max_speakers: 64,         // hard speaker limit
    window_secs: 1.5,         // analysis window
    hop_secs: 0.75,           // sliding step
    min_speech_secs: 0.25,    // discard shorter segments
    max_gap_secs: 0.5,        // merge same-speaker gaps under 500 ms
    sample_rate: SampleRate::new(16000).unwrap(),
};

let vad_config = VadConfig {
    frame_size: 512,          // Silero VAD chunk size (32 ms at 16 kHz)
    threshold: 0.5,           // speech probability threshold
    min_silence_ms: 300.0,    // minimum silence to split segments
};
```

## Streaming (real-time)

```rust,no_run
use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor};

let config = DiarizationConfig::default();
let mut diarizer = OnlineDiarizer::new(config);
let extractor = DummyExtractor::new(256);

// In your audio callback:
# let chunk = vec![0.0f32; 4800];
let segments = diarizer.feed(&chunk, &extractor).unwrap();
for seg in segments {
    println!("Speaker {:?} at {:.2}s", seg.speaker, seg.time.start);
}
```

## Verification

| Check | Tool |
|-------|------|
| Unsafe memory safety | Miri (nightly CI) |
| Concurrency correctness | Loom model-checking |
| Input fuzzing | cargo-fuzz (4 targets) |
| API stability | cargo-semver-checks |
| Cross-platform | Ubuntu, macOS, Windows CI |
| Dependency audit | cargo-audit |

## Roadmap

- [x] WeSpeaker + ECAPA-TDNN ONNX extractors
- [x] Silero VAD integration
- [x] Agglomerative hierarchical clustering (AHC)
- [x] Pipeline API (VAD + embeddings + AHC)
- [x] C FFI bindings
- [x] Miri / Loom / fuzz verification
- [x] Cross-platform CI
- [x] Python bindings (PyO3 / maturin)
- [x] CLI tool (`polyvoice diarize` / `download-models`)
- [x] DER benchmarks on AMI (27.5%) and VoxConverse (16.4%), 0.25s collar
- [ ] Spectral clustering backend
- [ ] PLDA scoring backend

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT
