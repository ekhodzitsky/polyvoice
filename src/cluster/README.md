# src/cluster

## Purpose

Speaker clustering data structures and helper algorithms: SpeakerCluster
with centroid tracking, label remapping, and cosine-similarity-based grouping.

## Surfaces

- `SpeakerCluster` — cluster of embeddings with centroid

## Dependencies

- `types` — ClusterConfig, SpeakerId
- `utils` — cosine_similarity, l2_normalize

## Invariants

- Cluster centroids are L2-normalized after updates.

## Verification

```bash
cargo test --lib cluster
```

## Notes

- This module owns the data structure, not the clustering algorithm itself
  (see `clusterer.rs` and `ahc.rs` for algorithms).
