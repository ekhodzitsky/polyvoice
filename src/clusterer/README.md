# src/clusterer

## Purpose

Clusterer trait and adapter implementations: AhcClusterer and NmeScClusterer.
Provides a unified interface over agglomerative and spectral clustering
backends.

## Surfaces

- `Clusterer` trait
- `ClustererError`
- `AhcClusterer`
- `NmeScClusterer` (requires `spectral` feature)

## Dependencies

- `ahc` — agglomerative clustering algorithm
- `spectral` — spectral clustering backend (for NmeScClusterer)

## Invariants

- Output labels are contiguous integers starting from 0.

## Verification

```bash
cargo test --lib clusterer
cargo test --lib clusterer --features spectral
cargo test --test clusterer_test --features spectral
```

## Notes

- NmeScClusterer requires the `spectral` feature (pulls faer for SVD).
