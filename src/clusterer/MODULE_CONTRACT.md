---
schema_version: 1
kind: module_contract
module: src/clusterer
level: subsystem
layer: algorithm
purpose: >
  Owns the Clusterer trait and its AHC and NME-SC adapter implementations.
  Does NOT own the underlying clustering algorithms (those live in cluster,
  ahc, spectral).
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
  - name: Clusterer
    kind: trait
    visibility: public
    contract: >
      Core clustering trait: cluster embeddings → speaker labels.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer
  - name: ClustererError
    kind: enum
    visibility: public
    contract: >
      Error type for clustering operations.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer
  - name: AhcClusterer
    kind: struct
    visibility: public
    contract: >
      Adapter wrapping agglomerative clustering from cluster/ahc modules.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer
  - name: NmeScClusterer
    kind: struct
    visibility: public
    contract: >
      Adapter wrapping spectral clustering with NME eigenvalue threshold.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer --features spectral
dependencies:
  internal: []
  external: []

consumers:
  - path: .
    uses:
      - Clusterer
      - ClustererError
      - AhcClusterer
      - NmeScClusterer
      - polyvoice_internal
invariants:
  - id: cluster-labels-contiguous
    rule: Output labels are contiguous integers starting from 0.
    proof:
      kind: unit-test
      target: src/clusterer::mod::tests
      command: cargo test --lib clusterer
verification:
  pre_change:
    - cargo test --lib clusterer
  full:
    - cargo test --lib clusterer
    - cargo test --lib clusterer --features spectral
    - cargo test --test clusterer_test --features spectral
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new Clusterer implementations.
    - Tuning default thresholds.
  forbidden_mutations:
    - Changing Clusterer trait signature without migration lease.
  escalation:
    - Changes to Clusterer trait or associated types.
---

# src/clusterer

Clusterer trait and AHC / NME-SC adapter implementations.
