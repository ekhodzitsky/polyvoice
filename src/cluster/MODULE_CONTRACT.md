---
schema_version: 1
kind: module_contract
module: src/cluster
level: subsystem
layer: algorithm
purpose: >
  Owns the online incremental SpeakerCluster (centroid assign/merge by cosine
  threshold) and related remapping helpers used with SpeakerIdRemap. This is
  a public library utility; production offline clustering uses clusterer, and
  production streaming uses streaming::ArrivalOrderSpeakerCache — SpeakerCluster
  is not on those default paths.
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
      Online incremental speaker clusterer: assign embeddings to centroids or
      create new speakers; optional merge. Re-exported at crate root.
    proof:
      kind: unit-test
      target: src/cluster::mod::tests
      command: cargo test --lib cluster
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: ClusterConfig, SpeakerId, SpeakerIdRemap.
    - module: utils
      scope: utility
      reason: cosine_similarity, l2_normalize.
  external: []
consumers:
  - path: src/lib.rs
    uses:
      - SpeakerCluster
  - path: fuzz/fuzz_targets/fuzz_cluster_assign.rs
    uses:
      - SpeakerCluster
  - path: src/types/mod.rs
    uses:
      - SpeakerIdRemap docs reference SpeakerCluster::merge
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
    - Soft-deprecating public surface if streaming reuses a single online backend.
  forbidden_mutations:
    - Removing SpeakerCluster without migration lease.
  escalation:
    - Changes to public methods of SpeakerCluster.
---

# src/cluster

Online `SpeakerCluster` (incremental centroids). Not used by offline
`pipeline_v2` or the default streaming AOSC cache.
