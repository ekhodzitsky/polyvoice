# src/spectral

## Purpose

Spectral clustering math: k-NN cosine affinity → normalized Laplacian →
eigendecomposition via faer. The crate-private graph construction is consumed
by `NmeScClusterer` (in `src/clusterer`), which runs `kmeans_pp` on the
spectral embedding.

## Surfaces

No public items. Crate-private:

- `SpectralGraph` — graph construction + spectral embedding
- `select_k_by_normalized_eigengap` — NME-SC eigengap k selection

## Dependencies

- `utils` — pairwise_cosine_similarity_matrix (flat cosine affinity source)
- `faer` — eigendecomposition

## Verification

```bash
cargo test --lib spectral --features spectral
```

## Notes

- Requires `spectral` feature (pulls faer).
