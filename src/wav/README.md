# src/wav

## Purpose

Audio file reading for the loading layer. WAV via `hound`; optional multi-format
decode + resampling behind the `audio-io` feature.

## Surfaces

- `read_wav(path) -> Result<(Vec<f32>, u32), WavError>` — raw WAV, any rate
- `load_audio(path) -> Result<(Vec<f32>, u32), WavError>` — mono 16 kHz for pipelines
- `TARGET_SAMPLE_RATE` — `16_000`
- `WavError`

## Dependencies

- `hound` — WAV (always)
- `rubato` + `symphonia` — only with feature `audio-io`

## Invariants

- `read_wav` returns the sample rate from the WAV header (no resampling).
- `load_audio` always returns `TARGET_SAMPLE_RATE` (16 kHz) mono on success.
- Without `audio-io`, non-WAV or non-16 kHz inputs error with a rebuild hint.
- Multi-channel is downmixed by averaging channels.

## Verification

```bash
cargo test --lib wav
cargo test --lib wav --features audio-io
cargo tree -e normal | rg "rubato|symphonia"   # empty without audio-io
```
