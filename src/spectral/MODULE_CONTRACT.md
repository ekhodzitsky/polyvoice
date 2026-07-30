---
schema_version: 1
kind: module_contract
module: src/spectral
level: subsystem
layer: algorithm
purpose: >
  Owns the shared spectral-graph math (k-NN cosine affinity → normalized
  Laplacian → eigenspectrum) and the NME-SC eigengap k selection behind
  NmeScClusterer. Does NOT own the NME-SC adapter (clusterer) or generic
  clustering traits.
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
  - name: SpectralGraph
    kind: struct
    visibility: crate
    contract: >
      k-NN cosine affinity → normalized Laplacian → sorted eigenspectrum,
      plus the row-normalized spectral embedding over the first k
      eigenvectors. Single graph-construction path.
    proof:
      kind: unit-test
      target: src/clusterer::mod::nme_sc_tests
      command: cargo test --lib clusterer --features "clusterer,spectral"
  - name: select_k_by_normalized_eigengap
    kind: function
    visibility: crate
    contract: >
      NME-SC normalized-maximum eigengap k selection over the ascending
      normalized-Laplacian eigenvalues. Search width is capped by
      MAX_EIGENGAP_CANDIDATES (20).
    proof:
      kind: unit-test
      target: src/spectral::mod::tests::eigengap_selects_k_on_known_sequence
      command: cargo test --lib spectral --features spectral
dependencies:
  internal:
    - module: utils
      scope: utility
      reason: pairwise_cosine_similarity_matrix for the affinity graph.
  external:
    - name: faer
      scope: math
      reason: Eigendecomposition of the normalized Laplacian.
consumers:
  - path: src/clusterer/mod.rs
    uses:
      - SpectralGraph
      - select_k_by_normalized_eigengap
verification:
  pre_change:
    - cargo test --lib spectral --features spectral
  full:
    - cargo test --lib spectral --features spectral
    - cargo test --lib clusterer --features "clusterer,spectral"
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning affinity matrix parameters.
    - Adding normalization strategies.
  forbidden_mutations:
    - Changing the crate-private surface without updating NmeScClusterer.
  escalation:
    - Changes to graph construction or eigengap semantics.
---

# src/spectral

Spectral graph math using eigendecomposition (faer), consumed by NmeScClusterer.
