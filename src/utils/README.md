# src/utils/

## Purpose

Math and array utilities: cosine similarity, L2 normalization, mean vector, merge segments.

## Surfaces

- `cosine_similarity` — cosine similarity between two vectors.
- `l2_normalize` — in-place L2 normalization.
- `mean_vector` — element-wise mean of a vector of vectors.
- `merge_segments` — greedy merge of overlapping segments.
- `pairwise_cosine_similarity_matrix` (crate-private) — flat row-major n×n
  cosine-similarity matrix, shared by ahc / kmeans / spectral.
- `normalized_mean_centroids` (crate-private) — per-label L2-normalized mean
  centroids over selected members.

## Dependencies

None.

## Invariants

- `cosine_similarity` returns values in `[-1.0, 1.0]` for normalized inputs.
- `l2_normalize` produces vectors with Euclidean norm `1.0` (or `0.0` for zero vectors).

## Verification

- `cargo test --lib utils`

## Notes

Property tests verify range and symmetry of cosine similarity.
