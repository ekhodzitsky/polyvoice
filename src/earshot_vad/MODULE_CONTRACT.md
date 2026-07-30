---
schema_version: 1
kind: module_contract
module: src/earshot_vad
level: leaf
layer: algorithm
purpose: >
  Optional pure-Rust earshot VAD implementing VoiceActivityDetector. Opt-in via
  feature vad-earshot. Silero remains the production/DER reference; this module
  must not become the default without a measured parity gate.
status: experimental
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/earshot_vad/
  context_budget:
    max_files: 4
    max_source_lines: 400
    max_contract_lines: 120
    max_readme_lines: 80
    max_todo_lines: 40
authority:
  write_policy: single_active_write_lease
  orchestrator: polyvoice-core
  read_agents: many_allowed
  migration_lease_required:
    - cross-workcell write
    - public surface migration
    - making earshot the production default VAD
surface:
  - name: EarshotVad
    kind: struct
    visibility: public
    contract: >
      VoiceActivityDetector over earshot 256-sample frames @ 16 kHz. Follows
      the crate-wide frame contract: process accepts only multiples of 256
      (partial chunks are rejected, never buffered) and returns one score in
      [0, 1] per frame.
    proof:
      kind: unit-test
      target: src/earshot_vad::tests
      command: cargo test --lib earshot_vad --features vad-earshot
  - name: ADAPTER_TYPE
    kind: const
    visibility: public
    contract: >
      AdapterRegistry type id string "earshot".
    proof:
      kind: unit-test
      target: src/earshot_vad::tests
      command: cargo test --lib earshot_vad --features vad-earshot
dependencies:
  internal:
    - module: vad
      scope: trait
      reason: VoiceActivityDetector + VadError.
  external:
    - crate: earshot
      reason: Pure-Rust VAD detector weights and scoring.
consumers:
  - path: src/lib.rs
    uses:
      - EarshotVad
invariants:
  - id: not-production-default
    rule: Silero remains the production VAD; earshot is opt-in only.
    proof:
      kind: doc-invariant
      target: src/lib.rs + src/earshot_vad/mod.rs
      command: rg -n "Silero remains the production" src/lib.rs src/earshot_vad/mod.rs
  - id: frame-size-256
    rule: FRAME_SIZE is 256 samples at 16 kHz.
    proof:
      kind: unit-test
      target: src/earshot_vad::tests
      command: cargo test --lib earshot_vad --features vad-earshot
verification:
  pre_change:
    - cargo test --lib earshot_vad --features vad-earshot
  full:
    - cargo test --lib earshot_vad --features vad-earshot
    - cargo clippy --all-targets --features vad-earshot -- -D warnings
agent_policy:
  allowed_mutations:
    - Improving error messages.
    - AdapterRegistry integration.
  forbidden_mutations:
    - Making earshot the default VAD without DER parity evidence.
  escalation:
    - Changing FRAME_SIZE or sample-rate contract.
---

# src/earshot_vad

Optional pure-Rust earshot VAD (`vad-earshot`). See README and benchmark notes.
