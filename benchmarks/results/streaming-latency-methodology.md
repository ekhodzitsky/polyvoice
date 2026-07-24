# Streaming latency measurement methodology

**Filled:** 2026-07-24  
**Artifact:** `benchmarks/results/streaming-latency-measured.json`  
**Harness:** `polyvoice-measure streaming` (release)

## Setup

| Field | Value |
|-------|--------|
| Hardware | Apple M1 Pro, 10 cores, aarch64, macOS |
| Build | `cargo build --release --features "cli,onnx,download" --bin polyvoice-measure` |
| Dataset | `data/voxconverse-test`, first **10** WAV files (lexicographic sort) |
| Audio total | ~6701 s across the 10 files |
| Chunk schedule | 3200 samples / feed (~200 ms @ 16 kHz) |
| VAD | Silero, frame 512 |
| Embedder | WeSpeaker ResNet34 (256-d) |
| DER | `compute_der`, collar 0 and 0.25, overlap scored, Hungarian |

## Metrics

- **Input-buffer latency** = `window + right_context + 512/16000` (config, not wall).
- **RTF** = sum wall-clock of (feed loop + flush) / sum audio duration.
- **DER** = macro average of per-file DER on streaming `turns()` after flush.

## Results (summary)

| Preset | latency | RTF | DER0 | DER0.25 |
|--------|---------|-----|------|---------|
| realtime | 1.032 s | 0.117 | 42.15% | 32.74% |
| balanced | 1.532 s | 0.109 | 29.99% | 20.15% |
| accurate | 2.282 s | 0.111 | 30.10% | 20.85% |

## Reproduce

```bash
cargo run --release --features "cli,onnx,download" --bin polyvoice-measure -- streaming \
  --dataset data/voxconverse-test --max-files 10 \
  --output benchmarks/results/streaming-latency-measured.json
```

## Follow-ups

- Full 232-file VoxConverse-test when budget allows.
- ≥1 h long-stream per-chunk latency series (bounded-state proof).
- Wire right-context so `accurate` can improve DER vs `balanced`.
