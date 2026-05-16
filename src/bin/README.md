# src/bin/

## Purpose

CLI entry points for the polyvoice speaker diarization toolkit.

## Surfaces

- **`polyvoice`** — Main CLI with subcommands:
  - `diarize <wav>` — Run diarization on a WAV file, output RTTM or JSON.
  - `download-models` — Download Mobile/Balanced ONNX model bundles.
  - `models list|info` — Inspect model registry and manifest metadata.
- **`polyvoice-bench`** — Batch DER benchmark on `{audio,rttm}` dataset directories.

## Dependencies

- `src/models` — Model registry and download.
- `src/pipeline` — Legacy v0.5 diarization pipeline.
- `src/rttm` — RTTM parsing and writing.
- `src/types` — Config, Profile, SampleRate.
- `src/vad` / `src/ecapa` — SileroVAD and FbankOnnxExtractor.
- `src/der` — DER computation for bench.
- `src/wav` — WAV file reading.

## Invariants

- CLI binaries are thin wrappers; all heavy logic lives in `lib` modules.
- `polyvoice` uses the legacy v0.5 pipeline (not the experimental M6b pipeline).
- `polyvoice-bench` expects dataset layout: `audio/*.wav` + `rttm/*.rttm`.

## Verification

```bash
cargo build --release --bin polyvoice
cargo build --release --bin polyvoice-bench
cargo test --test e2e_smoke_test
```

## Notes

- Both binaries require ONNX models at runtime. Use `download-models` or ensure
  the cache directory (`~/.cache/polyvoice/models`) is populated.
- The `balanced` profile (WeSpeaker ResNet34, 256-d) is the default.
