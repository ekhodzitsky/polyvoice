---
schema_version: 1
kind: module_contract
module: src/ahc
level: subsystem
layer: algorithm
purpose: >
  Owns agglomerative hierarchical clustering (AHC) algorithm with automatic
  threshold selection. Does NOT own the Clusterer trait adapter (clusterer.rs)
  or spectral clustering.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/ahc/
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
  - name: agglomerative_cluster
    kind: function
    visibility: public
    contract: >
      AHC with fixed cosine-similarity threshold. Returns cluster labels.
    proof:
      kind: unit-test
      target: src/ahc::mod::tests
      command: cargo test --lib ahc
  - name: agglomerative_cluster_auto
    kind: function
    visibility: public
    contract: >
      AHC with automatic threshold selection via elbow method. Returns labels
      and selected threshold.
    proof:
      kind: unit-test
      target: src/ahc::mod::tests
      command: cargo test --lib ahc
dependencies:
  internal:
    - module: utils
      scope: utility
      reason: cosine_similarity, l2_normalize.
  external: []
consumers:
  - path: src/clusterer/mod.rs
    uses:
      - agglomerative_cluster
      - agglomerative_cluster_auto
  - path: src/pipeline/mod.rs
    uses:
      - agglomerative_cluster
invariants:
  - id: labels-contiguous
    rule: Output labels are contiguous integers from 0, canonically ordered by descending cluster size (tie-break smallest member index) — deterministic per partition, independent of input order.
    proof:
      kind: unit-test
      target: src/ahc::mod::tests::cluster_ids_are_canonical_and_shuffle_invariant
      command: cargo test --lib ahc
verification:
  pre_change:
    - cargo test --lib ahc
  full:
    - cargo test --lib ahc
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning linkage criteria.
    - Optimizing similarity computations.
  forbidden_mutations:
    - Changing function signatures without updating clusterer.rs.
  escalation:
    - Changes to AHC algorithm semantics or output format.
---

# src/ahc

Agglomerative hierarchical clustering algorithm.
