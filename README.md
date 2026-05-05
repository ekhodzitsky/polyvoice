# polyvoice

[![CI](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml/badge.svg)](https://github.com/ekhodzitsky/polyvoice/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/polyvoice)](https://crates.io/crates/polyvoice)
[![Docs.rs](https://docs.rs/polyvoice/badge.svg)](https://docs.rs/polyvoice)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Speaker diarization for Rust — real-time, accurate, and production-hardened.**
>
> Turn any audio stream into a clear timeline of *who spoke when*.

## What is speaker diarization?

Speech-to-text tells you **what** was said. Speaker diarization tells you **who** said it.

```
Input:  "hello world how are you"
Output: SPEAKER_00: 0.0s - 1.2s  "hello world"
        SPEAKER_01: 1.5s - 2.8s  "how are you"
```

Without diarization, transcripts are a wall of text. With it, every word is attributed to the right person — essential for meeting minutes, call analytics, podcasts, and court recordings.

## Why polyvoice?

| You need... | `polyvoice` delivers |
|-------------|----------------------|
| **Real-time streaming** | `OnlineDiarizer` processes audio chunk-by-chunk with sub-second latency |
| **File-based batch** | `OfflineDiarizer` two-pass pipeline with gap merging and overlap detection |
| **No Python in production** | Pure Rust + ONNX Runtime. No GIL, no virtualenv, no dependency hell |
| **Concurrent inference** | Lock-free ONNX session pool — scale to many connections without `Mutex` contention |
| **Plug your own model** | `EmbeddingExtractor` trait: WeSpeaker, ECAPA-TDNN, or your custom ONNX model |
| **C FFI** | Drop-in `.so`/`.dylib`/`.dll` for Python, Go, Node.js, or C++ callers |
| **Safety guarantees** | Verified with Miri (unsafe memory), Loom (concurrency model-checking), and fuzzing |

## Quick start

```toml
[dependencies]
polyvoice = "0.4"
```

### Offline diarization (file / batch)

```rust
use polyvoice::{OfflineDiarizer, DiarizationConfig, DummyExtractor};

let config = DiarizationConfig::default();
let diarizer = OfflineDiarizer::new(config);
let extractor = DummyExtractor::new(256);

let samples: Vec<f32> = vec![0.0; 16000 * 10]; // 10 s of 16 kHz mono audio
let result = diarizer.run(&samples, &extractor).unwrap();

for turn in &result.turns {
    println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
}
```

### Real-time streaming

```rust
use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor};

let config = DiarizationConfig::default();
let mut diarizer = OnlineDiarizer::new(config);
let extractor = DummyExtractor::new(256);

while let Some(chunk) = microphone.read() {
    let segments = diarizer.feed(&chunk, &extractor).unwrap();
    for seg in segments {
        println!("Speaker {:?} from {:.2}s", seg.speaker, seg.time.start);
    }
}
```

### With an ONNX model (WeSpeaker / ECAPA-TDNN)

```toml
[dependencies]
polyvoice = { version = "0.4", features = ["onnx"] }
```

```rust
use polyvoice::{EcapaTdnnExtractor, OfflineDiarizer, DiarizationConfig};
use std::path::Path;

let config = DiarizationConfig::default();
let extractor = EcapaTdnnExtractor::new(
    Path::new("ecapa_tdnn.onnx"),
    192, // embedding dimension
    4,   // session pool size
).unwrap();

let diarizer = OfflineDiarizer::new(config);
let result = diarizer.run(&samples, &extractor).unwrap();
```

## Architecture

```
Input audio (f32 PCM)
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   VAD        │ --> │  Embedding   │ --> │   Speaker    │ --> Turns / Segments
│  (optional)  │     │  Extractor   │     │   Cluster    │
└──────────────┘     └──────────────┘     └──────────────┘
                          ONNX pool            Incremental
                          (lock-free)          cosine-sim clustering
```

## Key features

- **🎙️ Online & Offline** — stream chunks in real time or process entire files in one shot.
- **🧠 ONNX-powered** — ECAPA-TDNN and WeSpeaker extractors with built-in 80-bin log-mel filterbank.
- **⚡ Lock-free session pool** — `crossbeam-queue` backed pool eliminates `Mutex` contention under concurrent load.
- **🔌 VAD trait** — plug in Silero VAD, Energy VAD, or your own voice-activity detector.
- **🗣️ Overlap detection** — find regions where multiple speakers talk simultaneously.
- **📝 Word alignment** — assign speaker IDs to individual transcript words by timestamp.
- **🔒 Memory-safe FFI** — C ABI with Miri-verified unsafe code and Valgrind-tested Python bindings.
- **🦀 Pure Rust** — zero Python dependencies in production.

## Production readiness

This crate is hardened for production use:

| Verification | Tool |
|--------------|------|
| Unsafe memory safety | **Miri** ( nightly CI ) |
| Concurrency correctness | **Loom** model-checking |
| Input fuzzing | **cargo-fuzz** (4 targets, nightly CI) |
| API stability | **cargo-semver-checks** |
| Cross-platform | Ubuntu, macOS, Windows CI |
| Dependency audit | **cargo-audit** |

## Benchmarks

```bash
cargo bench --all-features
```

| Benchmark | Metric |
|-----------|--------|
| Offline diarization (10 s) | Latency on synthetic two-speaker audio |
| ECAPA fbank (10 s) | Log-mel throughput |
| DER (10 s) | Diarization Error Rate vs. ground truth |

## Configuration

```rust
use polyvoice::{DiarizationConfig, SampleRate};

let config = DiarizationConfig {
    threshold: 0.5,           // cosine similarity threshold
    max_speakers: 64,         // hard speaker limit
    window_secs: 1.5,         // analysis window
    hop_secs: 0.75,           // sliding step
    min_speech_secs: 0.25,    // discard shorter segments
    max_gap_secs: 0.5,        // merge same-speaker gaps under 500 ms
    sample_rate: SampleRate::new(16000).unwrap(),
};
```

## Roadmap

- [x] ECAPA-TDNN ONNX extractor
- [x] C FFI bindings
- [x] Miri / Loom / fuzz verification
- [x] Cross-platform CI
- [ ] Agglomerative re-clustering pass for offline mode
- [ ] PLDA scoring backend
- [ ] `no_std` support for embedded targets

## License

MIT
