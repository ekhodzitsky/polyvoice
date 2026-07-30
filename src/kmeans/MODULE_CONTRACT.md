---
schema_version: 1
kind: module_contract
module: src/kmeans
level: subsystem
layer: algorithm
purpose: >
  Owns k-means++ initialization and iterative clustering. Used by spectral
  clustering and other modules. Does NOT own the Clusterer trait.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/kmeans/
  context_budget:
    max_files: 12
    max_source_lines: 1500
    max_contract_lines: 180
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
  - name: kmeans_pp
    kind: function
    visibility: public
    contract: >
      K-means++ clustering: smart initialization + iterative Lloyd refinement.
      Returns cluster labels.
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests
      command: cargo test --lib kmeans
  - name: kmeans_auto_k
    kind: function
    visibility: public
    contract: >
      Automatic k selection in [k_min, k_max] via silhouette score over a
      precomputed pairwise cosine-distance matrix, `trials` runs per k.
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests
      command: cargo test --lib kmeans
dependencies:
  internal:
    - module: utils
      scope: utility
      reason: cosine_similarity_f32_f64, pairwise_cosine_similarity_matrix, XorShift64Star.
  external: []

consumers:
  - path: src/clusterer/mod.rs
    uses:
      - kmeans_pp
      - kmeans_auto_k
invariants:
  - id: labels-contiguous
    rule: Output labels are in 0..k.
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests
      command: cargo test --lib kmeans
  - id: empty-input-empty-output
    rule: Empty embeddings input returns empty labels.
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests::empty_input_returns_empty
      command: cargo test --lib kmeans
  - id: well-separated-clusters
    rule: Well-separated clusters are correctly grouped.
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests::well_separated_clusters
      command: cargo test --lib kmeans
  - id: property-labels-range
    rule: For any valid embeddings and k, all labels are in 0..k.min(n).
    proof:
      kind: unit-test
      target: tests/property_kmeans_test.rs::labels_in_range
      command: cargo test --test property_kmeans_test
  - id: property-k-one
    rule: For k=1, all labels are 0.
    proof:
      kind: unit-test
      target: tests/property_kmeans_test.rs::k_one_all_zero
      command: cargo test --test property_kmeans_test
  - id: degenerate-seeding-no-duplicate-centroids
    rule: >
      Collapsed/homogeneous embeddings with fewer distinct points than k yield no
      duplicate centroids; the effective cluster count is capped at the number of
      distinct points (seeding stops on a non-finite or <= 0 distance total).
    proof:
      kind: unit-test
      target: src/kmeans::mod::tests::identical_embeddings_yield_single_cluster
      command: cargo test --lib kmeans
verification:
  pre_change:
    - cargo test --lib kmeans
  full:
    - cargo test --lib kmeans
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning convergence criteria.
    - Optimizing distance computations.
  forbidden_mutations:
    - Changing kmeans_pp signature without updating the consumers above.
  escalation:
    - Changes to output semantics or signature.
---

# src/kmeans

K-means++ clustering algorithm.
