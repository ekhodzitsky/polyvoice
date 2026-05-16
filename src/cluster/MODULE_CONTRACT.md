---
schema_version: 1
kind: module_contract
module: src/cluster
level: subsystem
layer: algorithm
purpose: >
  Owns speaker clustering data structures and algorithms: SpeakerCluster,
  centroid computation, label remapping, and cosine-similarity-based grouping.
  Does NOT own the Clusterer trait (that lives in clusterer.rs).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/cluster/
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
  - name: SpeakerCluster
    kind: struct
    visibility: public
    contract: >
      Cluster of speaker embeddings with centroid and assigned segments.
    proof:
      kind: unit-test
      target: src/cluster::mod::tests
      command: cargo test --lib cluster
dependencies:
  internal: []
  external: []

consumers:
  - path: src/streaming/mod.rs
    uses:
      - SpeakerCluster
  - path: src/ahc/mod.rs
    uses:
      - SpeakerCluster
  - path: src/pipeline/mod.rs
    uses:
      - SpeakerCluster
  - path: tests/clusterer_test.rs
    uses:
      - SpeakerCluster
invariants:
  - id: centroid-normalized
    rule: Cluster centroids are L2-normalized after updates.
    proof:
      kind: unit-test
      target: src/cluster::mod::tests
      command: cargo test --lib cluster
verification:
  pre_change:
    - cargo test --lib cluster
  full:
    - cargo test --lib cluster
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new clustering metrics.
    - Refactoring internal data layout.
  forbidden_mutations:
    - Removing SpeakerCluster without migration lease.
  escalation:
    - Changes to public fields of SpeakerCluster.
---

# src/cluster

Speaker clustering data structures and algorithms.
