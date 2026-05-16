# src/wav

## Purpose

WAV file reading via hound crate. Returns floating-point samples and sample rate.

## Surfaces

- `read_wav(path) -> Result<(Vec<f32>, u32), WavError>`
- `WavError`

## Dependencies

- `hound` — WAV format parsing

## Invariants

- Returns actual sample rate from WAV header.

## Verification

```bash
cargo test --test e2e_smoke_test --features onnx,download
```

## Notes

- Converts all bit depths to f32 normalized to [-1.0, 1.0].
