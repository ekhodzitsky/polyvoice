---
schema_version: 1
kind: module_contract
module: src/window
level: subsystem
layer: algorithm
purpose: >
  Owns audio windowing utilities: fixed-size overlapping window iteration
  and window buffer management. Does NOT own feature extraction or VAD.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/window/
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
  - name: WindowIter
    kind: struct
    visibility: public
    contract: >
      Iterator over overlapping fixed-size audio windows. `new` panics on
      invalid geometry; `try_new` validates and returns WindowError.
    proof:
      kind: unit-test
      target: src/window::mod::tests
      command: cargo test --lib window
  - name: WindowBuffer
    kind: struct
    visibility: public
    contract: >
      Ring buffer for streaming audio window management. `new` panics on
      invalid geometry; `try_new` validates and returns WindowError.
    proof:
      kind: unit-test
      target: src/window::mod::tests
      command: cargo test --lib window
  - name: WindowError
    kind: enum
    visibility: public
    contract: >
      Window geometry violation reported by the fallible `try_new`
      constructors (zero window, zero hop, or hop larger than window).
    proof:
      kind: unit-test
      target: src/window::mod::tests
      command: cargo test --lib window
dependencies:
  internal: []
  external: []

consumers:
  - path: .
    uses:
      - WindowIter
      - WindowBuffer
      - polyvoice_internal
invariants:
  - id: window-size-constant
    rule: WindowIter yields windows of exactly the configured size.
    proof:
      kind: unit-test
      target: src/window::mod::tests
      command: cargo test --lib window
verification:
  pre_change:
    - cargo test --lib window
  full:
    - cargo test --lib window
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new window functions (Hann, Hamming, etc.).
    - Optimizing buffer allocations.
  forbidden_mutations:
    - Changing WindowIter/WindowBuffer core semantics without migration lease.
  escalation:
    - Changes to public API of windowing structs.
---

# src/window

Audio windowing utilities for streaming and offline processing.
