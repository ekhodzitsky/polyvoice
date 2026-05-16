---
schema_version: 1
kind: module_contract
module: src/spectral
level: subsystem
layer: algorithm
purpose: >
  Owns spectral clustering implementation using eigendecomposition via faer.
  Does NOT own the NME-SC adapter (clusterer.rs) or generic clustering traits.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/spectral/
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
  - name: spectral_cluster
    kind: function
    visibility: public
    contract: >
      Spectral clustering on embeddings: affinity matrix → Laplacian →
      eigendecomposition → k-means on eigenvectors. Returns cluster labels.
    proof:
      kind: unit-test
      target: src/spectral::mod::tests
      command: cargo test --lib spectral --features spectral
dependencies:
  internal:
    - module: utils
      scope: utility
      reason: cosine_similarity for affinity matrix.
  external:
    - name: faer
      scope: math
      reason: SVD and eigendecomposition for spectral clustering.
consumers:
  - path: src/clusterer/mod.rs
    uses:
      - spectral_cluster
invariants:
  - id: labels-contiguous
    rule: Output labels are contiguous integers starting from 0.
    proof:
      kind: unit-test
      target: src/spectral::mod::tests
      command: cargo test --lib spectral --features spectral
verification:
  pre_change:
    - cargo test --lib spectral --features spectral
  full:
    - cargo test --lib spectral --features spectral
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning affinity matrix parameters.
    - Adding normalization strategies.
  forbidden_mutations:
    - Removing spectral_cluster or changing its signature.
  escalation:
    - Changes to spectral_cluster signature or output semantics.
---

# src/spectral

Spectral clustering using eigendecomposition (faer).
