# src/ahc

## Purpose

Agglomerative hierarchical clustering (AHC) for speaker diarization.
Clusters embeddings using cosine-similarity linkage with either a fixed
threshold or automatic elbow-based threshold selection.

## Surfaces

- `agglomerative_cluster(embeddings, threshold) -> Vec<usize>`
- `agglomerative_cluster_auto(embeddings) -> (Vec<usize>, f32)`

## Dependencies

- `utils` — cosine_similarity, l2_normalize

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

- `agglomerative_cluster_auto` uses an elbow heuristic on the linkage
  dendrogram to pick a threshold. This is a heuristic, not guaranteed optimal.
