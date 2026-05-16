# src/features

## Purpose

Filterbank (FBank) feature extraction and CMVN normalization for speaker
embedding pipelines.

## Surfaces

- `FbankConfig` — configuration
- `FbankExtractor` — mel-filterbank extractor
- `apply_cmvn` — cepstral mean and variance normalization

## Dependencies

- `realfft`, `rustfft` — FFT for spectrogram computation

## Invariants

- FbankExtractor outputs non-negative mel energies before CMVN.

## Verification

```bash
cargo test --lib features
```

## Notes

- Default configuration targets 16kHz audio.
