# src/spectral

## Purpose

Spectral clustering using affinity matrix + Laplacian + eigendecomposition
via faer. Used by NmeScClusterer.

## Surfaces

- `spectral_cluster(embeddings, max_k) -> Vec<usize>`

## Dependencies

- `utils` — cosine_similarity
- `faer` — SVD/eigendecomposition

## Invariants

- Output labels are contiguous integers starting from 0.

## Verification

```bash
cargo test --lib spectral --features spectral
```

## Notes

- Requires `spectral` feature (pulls faer).
