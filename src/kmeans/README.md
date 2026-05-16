# src/kmeans

## Purpose

K-means++ clustering: smart initialization + iterative Lloyd refinement.
Used by spectral clustering backend.

## Surfaces

- `kmeans_pp(embeddings, k, max_iter) -> Vec<usize>`

## Dependencies

None (self-contained).

## Invariants

- Output labels are in 0..k.

## Verification

```bash
cargo test --lib kmeans
```

## Notes

- Uses Euclidean distance on f32 vectors.
