# src/ahc

## Purpose

Agglomerative hierarchical clustering (AHC) for speaker diarization.
Clusters embeddings using cosine-similarity linkage with either a fixed
threshold or automatic largest-gap threshold selection.

## Surfaces

- `agglomerative_cluster(embeddings, threshold) -> Vec<usize>`
- `agglomerative_cluster_max_clusters(embeddings, threshold, max_clusters) -> Vec<usize>`
- `agglomerative_cluster_asc(embeddings, threshold, max_clusters, stop, time_ranges) -> Vec<usize>`
- `agglomerative_cluster_auto_max_clusters(embeddings, max_clusters) -> (Vec<usize>, f32)`
- `prune_small_clusters(embeddings, labels, min_size) -> Vec<usize>`
- `prune_small_clusters_by_duration(time_ranges, embeddings, labels, min_secs) -> Vec<usize>`
- `AscStop` — cAHC-ASC established-cluster stop rule (`Off` / `MinMembers` / `MinSecs`)

## Dependencies

- `utils` — cosine_similarity, l2_normalize, normalized_mean_centroids,
  pairwise_cosine_similarity_matrix

## Invariants

- Output labels are contiguous integers from 0, **canonically ordered** by
  descending cluster size (ties broken by smallest member index) — the same
  partition yields the same ids regardless of input/merge order.
- All embeddings are L2-normalized before similarity computation.

## Verification

```bash
cargo test --lib ahc
```

## Notes

- `agglomerative_cluster_auto_max_clusters` picks the threshold at the
  largest gap in the lower half (below the median) of the sorted pairwise
  cosine similarities, clamped to `[0.2, 0.7]`. This is a heuristic, not
  guaranteed optimal. The pairwise similarity matrix is built once and
  shared between the threshold estimate and the clustering pass.
