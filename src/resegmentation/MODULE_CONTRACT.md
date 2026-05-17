---
schema_version: 1
kind: module_contract
module: src/resegmentation
level: subsystem
layer: algorithm
purpose: >
  Owns overlap-aware post-clustering resegmentation: centroid computation,
  overlap region extraction, and resegmenter trait. Operates on already-
  computed speaker centroids and overlap-region embeddings. Does NOT own
  embedding extraction or clustering.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/resegmentation/
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
  - name: Resegmenter
    kind: trait
    visibility: public
    contract: >
      Post-clustering resegmentation trait for refining overlap regions.
    proof:
      kind: unit-test
      target: src/resegmentation::mod::tests
      command: cargo test --lib resegmentation
  - name: OverlapResegmenter
    kind: struct
    visibility: public
    contract: >
      Default resegmenter implementation using centroids and cosine similarity.
    proof:
      kind: unit-test
      target: src/resegmentation::mod::tests
      command: cargo test --lib resegmentation
  - name: compute_centroids
    kind: function
    visibility: public
    contract: >
      Computes speaker centroids from embeddings and cluster labels.
    proof:
      kind: unit-test
      target: src/resegmentation::mod::tests
      command: cargo test --lib resegmentation
  - name: extract_overlap_time_ranges
    kind: function
    visibility: public
    contract: >
      Extracts time ranges where multiple speakers overlap.
    proof:
      kind: unit-test
      target: src/resegmentation::mod::tests
      command: cargo test --lib resegmentation
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: SpeakerId, SpeakerTurn, TimeRange.
  external: []
consumers:
  - path: .
    uses:
      - Resegmenter
      - OverlapResegmenter
      - compute_centroids
      - extract_overlap_time_ranges
      - polyvoice_internal
invariants:
  - id: centroids-normalized
    rule: compute_centroids outputs L2-normalized vectors.
    proof:
      kind: unit-test
      target: src/resegmentation::mod::tests
      command: cargo test --lib resegmentation
verification:
  pre_change:
    - cargo test --lib resegmentation
  full:
    - cargo test --lib resegmentation
    - cargo test --test resegmenter_test --features resegmentation,onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new Resegmenter strategies.
    - Improving centroid computation.
  forbidden_mutations:
    - Changing Resegmenter trait signature without migration lease.
  escalation:
    - Changes to Resegmenter trait or associated types.
---

# src/resegmentation

Overlap-aware post-clustering resegmentation.
