# Streaming latency-vs-DER methodology (stub)

This file is the placeholder for measured streaming preset numbers. Fill it
when a calibrated run is available; until then the table in
`docs/BENCHMARKS.md` keeps **TBD** cells.

## Protocol

| Item | Value |
|------|-------|
| Pipeline | `StreamingPipeline::with_latency_preset` + Balanced ONNX embedder |
| Presets | `realtime`, `balanced`, `accurate` |
| Sample rate | 16 kHz mono |
| VAD | EnergyVad or Silero, frame 512 samples (32 ms) |
| Input-buffer latency | `window_secs + right_context_secs + vad_frame_secs` (config, not RTF) |
| RTF | `sum(feed_wall_time) / audio_duration` on named hardware |
| DER | collar 0, overlap scored, Hungarian (`benchmarks/der.py`) |
| Subset | VoxConverse-test 30-file subset (or full 232 when budget allows) |
| Long-stream | ≥1 h audio or synthetic; per-chunk latency series start vs end |
| Label stability | `label_flip_rate(first_emitted, final)` |

## Expected artifact schema

```json
{
  "schema": "polyvoice-streaming-latency-v1",
  "hardware": "TBD",
  "presets": {
    "realtime": {
      "input_buffer_latency_secs": 1.032,
      "rtf_mean": null,
      "chunk_latency_ms_mean": null,
      "chunk_latency_ms_p95": null,
      "chunk_latency_ms_start_mean": null,
      "chunk_latency_ms_end_mean": null,
      "der_collar0": null,
      "label_flip_rate": null
    }
  }
}
```

## Bounded-state check

`ArrivalOrderSpeakerCache` is hard-capped (`speaker_cache_cap`). Unit tests
assert `cache.len() <= cap` under long assign loops. The ≥1 h bench must show
non-growing per-chunk latency (start window ≈ end window within noise).
