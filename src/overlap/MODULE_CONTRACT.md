---
schema_version: 1
kind: module_contract
module: src/overlap
level: subsystem
layer: algorithm
purpose: >
  Owns overlap detection between speaker segments. Does NOT own resegmentation
  (resegmentation.rs) or the pipeline.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/overlap/
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
  - name: detect_overlaps
    kind: function
    visibility: public
    contract: >
      Detects time ranges where multiple speakers overlap in a segment list.
    proof:
      kind: unit-test
      target: src/overlap::mod::tests
      command: cargo test --lib overlap
  - name: OverlapRegion
    kind: struct
    visibility: public
    contract: >
      Represents a single overlap region with speaker IDs and time range.
    proof:
      kind: unit-test
      target: src/overlap::mod::tests
      command: cargo test --lib overlap
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Segment, SpeakerId, TimeRange.
  external: []
consumers:
  - path: src/resegmentation/mod.rs
    uses:
      - detect_overlaps
  - path: src/lib.rs
    uses:
      - detect_overlaps
      - OverlapRegion
invariants:
  - id: overlap-time-valid
    rule: OverlapRegion time ranges have start < end and contain at least 2 speakers.
    proof:
      kind: unit-test
      target: src/overlap::mod::tests
      command: cargo test --lib overlap
verification:
  pre_change:
    - cargo test --lib overlap
  full:
    - cargo test --lib overlap
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning overlap detection thresholds.
  forbidden_mutations:
    - Changing OverlapRegion field types without migration lease.
  escalation:
    - Changes to overlap detection semantics.
---

# src/overlap

Overlap detection between speaker segments.
