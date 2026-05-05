# polyvoice API Reference

## Overview

`polyvoice` is a speaker diarization library for Rust. It answers the question
**"who spoke when?"** given a stream or file of audio samples.

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Audio Bytes │ --> │ Embedding       │ --> │ Speaker Cluster │ --> Turns
│ (f32 PCM)   │     │ Extractor       │     │ (online/offline)│
└─────────────┘     └─────────────────┘     └─────────────────┘
```

## Choosing Online vs Offline

| Mode | Use case | Latency | Accuracy |
|------|----------|---------|----------|
| **Online** (`OnlineDiarizer`) | Real-time streaming (WebSocket, microphone) | Low (chunk-by-chunk) | Lower (no future context) |
| **Offline** (`OfflineDiarizer`) | File transcription, post-processing | High (full file) | Higher (two-pass + merge) |

## Core Types

### `SpeakerId`
Opaque `u32` wrapper identifying a speaker cluster.

### `DiarizationConfig`
Central configuration struct:
- `threshold: f32` — cosine similarity threshold for matching to existing speaker.
- `max_speakers: usize` — hard limit on concurrent speakers.
- `window_secs: f32` — analysis window size.
- `hop_secs: f32` — step between consecutive windows.
- `min_speech_secs: f32` — minimum segment duration (post-processing).
- `max_gap_secs: f32` — merge same-speaker segments with gaps ≤ this value.
- `sample_rate: SampleRate` — validated sample rate (8000–192000 Hz).

### `DiarizationResult`
```rust
pub struct DiarizationResult {
    pub segments: Vec<Segment>,
    pub turns: Vec<SpeakerTurn>,
    pub num_speakers: usize,
}
```

## Embedding Extractors

### `DummyExtractor`
Deterministic pseudo-random extractor for testing and benchmarking.

```rust
let extractor = DummyExtractor::new(256);
```

### `OnnxEmbeddingExtractor` (feature `onnx`)
Raw-audio ONNX model (WeSpeaker-style). Input shape: `[1, window_samples]`.

### `EcapaTdnnExtractor` (feature `onnx`)
ECAPA-TDNN model with built-in log-mel filterbank preprocessing.
Input shape: `[1, n_frames, n_mels]`.

## Voice Activity Detection

### `EnergyVad`
Simple energy-based VAD for tests and fallback scenarios.

```rust
let mut vad = EnergyVad::new(-40.0, 16000, 512);
let segments = segment_speech(&mut vad, &samples, &config, &vad_config)?;
```

## Overlap Detection

```rust
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

## Performance Tuning

1. **Use `FbankExtractor`** instead of `compute_fbank` to avoid per-call FFT allocation.
2. **Increase pool size** for ONNX extractors if you have many concurrent requests.
3. **Tune `threshold`** — lower values merge more aggressively; higher values split more.
4. **Tune `max_gap_secs`** — larger gaps mean fewer turns but may miss real speaker changes.
