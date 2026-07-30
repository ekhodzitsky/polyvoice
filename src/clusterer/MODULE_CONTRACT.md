---
schema_version: 1
kind: module_contract
module: src/clusterer
level: subsystem
layer: algorithm
purpose: >
  Owns the offline Clusterer trait and adapter implementations (AHC,
  min-cluster-size wrapper, K-means, NME-SC, VBx/PLDA) plus local-to-global
  assignment helpers and short-segment filters. Does NOT own free clustering
  math (ahc, kmeans, spectral) or the online SpeakerCluster (cluster module).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/clusterer/
  context_budget:
    max_files: 12
    max_source_lines: 2500
    max_contract_lines: 220
    max_readme_lines: 120
    max_todo_lines: 80
authority:
  write_policy: single_active_write_lease
  orchestrator: polyvoice-core
  read_agents: many_allowed
  migration_lease_required:
    - cross-workcell write
    - public surface migration
surface:
  - name: Clusterer
    kind: trait
    visibility: public
    contract: >
      Offline clustering trait: embeddings (+ optional durations) -> speaker
      labels. Optional wants_raw_embeddings for PLDA backends.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
  - name: ClustererError
    kind: enum
    visibility: public
    contract: >
      Error type for clustering operations.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
  - name: AhcClusterer
    kind: struct
    visibility: public
    contract: >
      Adapter wrapping ahc::agglomerative_cluster with max clusters and threshold.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
  - name: MinClusterSizeClusterer
    kind: struct
    visibility: public
    contract: >
      Wrapper that dissolves clusters smaller than min size into nearest large
      speakers. Not applied to VBx by the pipeline builder.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
  - name: KmeansClusterer
    kind: struct
    visibility: public
    contract: >
      K-means++ / auto-k clusterer. Reachable via the crate-root re-export;
      deprecated alias `KMeansClusterer` kept for one cycle.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
  - name: NmeScClusterer
    kind: struct
    visibility: public
    contract: >
      Spectral / NME eigengap clusterer (requires `spectral` feature).
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features "clusterer,spectral"
  - name: VbxClusterer
    kind: struct
    visibility: public
    contract: >
      VBx (Variational Bayes HMM + PLDA) with automatic speaker-count
      selection. Requires the `vbx` feature; wants raw embeddings.
    proof:
      kind: unit-test
      target: src/clusterer::vbx::tests
      command: cargo test --lib clusterer::vbx --features vbx
  - name: PldaModel
    kind: struct
    visibility: public
    contract: >
      Precomputed PLDA feature transform (256-d -> 128-d) for VBx. Pure-ndarray
      runtime; diagonalization precomputed offline. Requires the `vbx` feature.
    proof:
      kind: unit-test
      target: src/clusterer::plda::tests
      command: cargo test --lib clusterer::plda --features vbx
  - name: assign helpers
    kind: module
    visibility: public
    contract: >
      build_cooccurrence, hungarian_local_to_global, majority_local_to_global
      map local powerset speaker ids to global cluster labels.
    proof:
      kind: unit-test
      target: src/clusterer::assign::tests
      command: cargo test --lib clusterer --features clusterer
  - name: short_filter helpers
    kind: module
    visibility: public
    contract: >
      partition_by_min_duration and reassign_short_by_* for short-segment
      exclusion/reassignment around AHC/VBx.
    proof:
      kind: unit-test
      target: src/clusterer::short_filter::tests
      command: cargo test --lib clusterer --features clusterer
dependencies:
  internal:
    - module: ahc
      scope: algorithm
      reason: AhcClusterer wraps free agglomerative functions.
    - module: kmeans
      scope: algorithm
      reason: KmeansClusterer and NME-SC final assignment.
    - module: spectral
      scope: algorithm
      reason: Shared eigengap helpers for NmeScClusterer.
    - module: hungarian
      scope: algorithm
      reason: Optimal local-to-global assignment.
  external: []
consumers:
  - path: src/pipeline_v2/
    uses:
      - Clusterer
      - AhcClusterer
      - NmeScClusterer
      - VbxClusterer
      - MinClusterSizeClusterer
      - assign helpers
  - path: .
    uses:
      - polyvoice_internal
invariants:
  - id: cluster-labels-contiguous
    rule: Output labels are contiguous integers starting from 0.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features clusterer
verification:
  pre_change:
    - cargo test --lib clusterer --features clusterer
  full:
    - cargo test --lib clusterer --features clusterer
    - cargo test --lib clusterer --features "clusterer,spectral"
    - cargo test --lib clusterer --features vbx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new Clusterer implementations.
    - Tuning default thresholds.
    - Thinning NME-SC over shared spectral primitives.
  forbidden_mutations:
    - Changing Clusterer trait signature without migration lease.
  escalation:
    - Changes to Clusterer trait or associated types.
---

# src/clusterer

Offline `Clusterer` trait and adapters (AHC, min-size, K-means, NME-SC, VBx).
