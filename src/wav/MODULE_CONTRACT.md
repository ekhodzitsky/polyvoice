---
schema_version: 1
kind: module_contract
module: src/wav
level: subsystem
layer: io
purpose: >
  Owns WAV file reading via hound crate. Returns samples and sample rate.
  Does NOT own audio playback or feature extraction.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/wav/
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
  - name: read_wav
    kind: function
    visibility: public
    contract: >
      Reads a WAV file and returns (samples as f32 Vec, sample_rate_hz).
    proof:
      kind: unit-test
      target: src/wav::mod::tests::missing_file_error
      command: cargo test --lib wav
  - name: WavError
    kind: enum
    visibility: public
    contract: >
      Error type for WAV read failures.
    proof:
      kind: integration-test
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test --features onnx,download
dependencies:
  internal: []
  external: []

consumers:
  - path: .
    uses:
      - read_wav
      - WavError
      - polyvoice_internal
invariants:
  - id: sample-rate-exposed
    rule: read_wav returns the actual sample rate from the WAV header.
    proof:
      kind: integration-test
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test --features onnx,download
verification:
  pre_change:
    - cargo check --all-features
  full:
    - cargo test --test e2e_smoke_test --features onnx,download
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding WAV format support.
    - Optimizing read buffer sizes.
  forbidden_mutations:
    - Changing read_wav return type without migration lease.
  escalation:
    - Changes to WAV read semantics or error variants.
---

# src/wav

WAV file reading utilities.
