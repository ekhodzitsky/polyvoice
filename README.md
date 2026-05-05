# polyvoice

> Speaker diarization library for Rust — online (streaming) and offline (file-based), ONNX-powered, and ecosystem-agnostic.

`polyvoice` answers the question **"who spoke when?"** in audio streams or files. It is designed to be embedded into STT servers such as [`gigastt`](https://github.com/ekhodzitsky/gigastt), [`phostt`](https://github.com/ekhodzitsky/phostt), `nihostt`, `siamstt`, or any other Rust application.

## Features

- **Online (streaming) diarization** — process audio chunk-by-chunk in real time.
- **Offline (file) diarization** — process an entire audio buffer with post-processing (segment merging, gap filling).
- **Sliding-window embeddings** — configurable window and hop sizes instead of fixed segments.
- **Session pool for ONNX models** — no `Mutex` contention under concurrent load.
- **VAD integration trait** — plug in Silero VAD, Energy VAD, or your own implementation.
- **Overlap detection** — identify regions where multiple speakers are active simultaneously.
- **Word-level speaker alignment** — assign speaker IDs to individual words using timestamps.
- **Zero Python dependencies** — pure Rust + ONNX Runtime.

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
polyvoice = { git = "https://github.com/ekhodzitsky/polyvoice" }
```

### Offline diarization

```rust
use polyvoice::{OfflineDiarizer, DiarizationConfig, DummyExtractor};

let config = DiarizationConfig::default();
let diarizer = OfflineDiarizer::new(config);
let extractor = DummyExtractor::new(256);

let samples: Vec<f32> = vec![0.0; 16000 * 10]; // 10s of 16 kHz mono audio
let result = diarizer.run(&samples, &extractor).unwrap();

for turn in &result.turns {
    println!("{}: {:.2}s - {:.2}s", turn.speaker, turn.time.start, turn.time.end);
}
```

### Online diarization

```rust
use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor};

let config = DiarizationConfig::default();
let mut diarizer = OnlineDiarizer::new(config);
let extractor = DummyExtractor::new(256);

// Feed audio chunks as they arrive (e.g. from a WebSocket stream)
let chunk = vec![0.0f32; 16000]; // 1 second
let segments = diarizer.feed(&chunk, &extractor).unwrap();
```

### With ONNX embedding extractor

Enable the `onnx` feature and use a WeSpeaker / ECAPA-TDNN ONNX model:

```toml
[dependencies]
polyvoice = { git = "https://github.com/ekhodzitsky/polyvoice", features = ["onnx"] }
```

```rust
use polyvoice::{OnnxEmbeddingExtractor, OfflineDiarizer, DiarizationConfig};
use std::path::Path;

let config = DiarizationConfig::default();
let extractor = OnnxEmbeddingExtractor::new(
    Path::new("wespeaker_resnet34.onnx"),
    256,              // embedding dimension
    24000,            // window samples (1.5s @ 16kHz)
    4,                // pool size
).unwrap();

let diarizer = OfflineDiarizer::new(config);
let result = diarizer.run(&samples, &extractor).unwrap();
```

## Architecture

```
polyvoice
├── embedding      # EmbeddingExtractor trait + ONNX pool implementation
├── cluster        # Online incremental centroid clustering
├── vad            # Voice Activity Detection trait + utilities
├── online         # StreamingDiarizer (chunk-by-chunk)
├── offline        # OfflineDiarizer (two-pass with post-processing)
├── overlap        # Overlap detection from segment lists
└── types          # Config, SpeakerId, Segment, WordAlignment, etc.
```

## Configuration

```rust
use polyvoice::DiarizationConfig;

let config = DiarizationConfig {
    threshold: 0.5,           // cosine similarity threshold
    max_speakers: 64,         // speaker limit
    window_secs: 1.5,         // analysis window
    hop_secs: 0.75,           // sliding step
    min_speech_secs: 0.25,    // discard shorter segments
    sample_rate: 16000,       // expected sample rate
};
```

## Roadmap to 1.0

- [ ] ECAPA-TDNN ONNX extractor (in addition to WeSpeaker)
- [ ] Agglomerative re-clustering pass for offline mode
- [ ] PLDA scoring backend
- [ ] `no_std` support for embedded targets
- [ ] C FFI bindings

## License

MIT
