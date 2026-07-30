# src/kmeans

## Purpose

K-means++ clustering: smart initialization + iterative Lloyd refinement.
Used by spectral clustering backend.

## Surfaces

- `kmeans_pp(embeddings, k, max_iter) -> Vec<usize>`
- `kmeans_auto_k(embeddings, k_min, k_max, max_iter, trials) -> Vec<usize>` —
  automatic k selection via silhouette score over a precomputed pairwise
  distance matrix.

## Dependencies

- `utils` — cosine_similarity_f32_f64, pairwise_cosine_similarity_matrix,
  XorShift64Star (deterministic seeding)

## Invariants

- Output labels are in 0..k.
- Degenerate/collapsed embeddings (fewer distinct points than `k`) yield no
  duplicate centroids: the effective cluster count is capped at the number of
  distinct points.

## Verification

```bash
cargo test --lib kmeans
```

## Notes

- Uses cosine distance on f32 vectors (`crate::utils::cosine_similarity_f32_f64`).
- Degenerate-seeding guard: when the candidate-distance total is non-finite or
  `<= 0` (all remaining points sit on a chosen centroid), k-means++ stops
  seeding early and Lloyd's runs on the centroids gathered so far, instead of
  sampling duplicate/garbage centroids.
