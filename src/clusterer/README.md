# src/clusterer

Offline **`Clusterer`** trait and adapter implementations used by
`pipeline_v2`.

## Surfaces

- `Clusterer` / `ClustererError`
- `AhcClusterer`
- `MinClusterSizeClusterer`
- `KmeansClusterer` (deprecated alias `KMeansClusterer` kept for one cycle)
- `NmeScClusterer` (feature `spectral`)
- `VbxClusterer` / `PldaModel` (feature `vbx`)
- `assign::*` — local→global co-occurrence mapping
- `short_filter::*` — short-embedding partition / reassignment

## Not this module

- Free math: `ahc`, `kmeans`, `spectral`
- Online centroids: `cluster::SpeakerCluster` (orphaned relative to pipelines;
  streaming uses `ArrivalOrderSpeakerCache`)

## Verification

```bash
cargo test --lib clusterer --features clusterer
cargo test --lib clusterer --features "clusterer,spectral"
cargo test --lib clusterer --features vbx
```
