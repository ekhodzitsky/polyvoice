---
schema_version: 1
kind: module_contract
module: src/rttm
level: subsystem
layer: io
purpose: >
  Owns RTTM (Rich Transcription Time Marked) file parsing, grouping, and
  writing. Does NOT own audio I/O (wav.rs) or DER computation (der.rs).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/rttm/
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
  - name: RttmSegment
    kind: struct
    visibility: public
    contract: >
      Single RTTM line parsed into typed fields.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: RttmError
    kind: enum
    visibility: public
    contract: >
      Error type for RTTM parse/write failures.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: parse_rttm
    kind: function
    visibility: public
    contract: >
      Parses RTTM lines from a BufRead into RttmSegment vector.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: parse_rttm_file
    kind: function
    visibility: public
    contract: >
      Convenience: parse RTTM from file path.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: group_by_file
    kind: function
    visibility: public
    contract: >
      Groups RTTM segments by file identifier.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: to_speaker_turns
    kind: function
    visibility: public
    contract: >
      Converts RttmSegment vector into polyvoice SpeakerTurns + ID mapping.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
  - name: write_rttm
    kind: function
    visibility: public
    contract: >
      Writes SpeakerTurns as RTTM lines to a Write sink.
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: SpeakerId, SpeakerTurn, TimeRange.
  external: []
consumers:
  - path: tests/der_regression_test.rs
    uses:
      - parse_rttm_file
      - group_by_file
      - to_speaker_turns
  - path: tests/der_ami_baseline_test.rs
    uses:
      - parse_rttm_file
      - group_by_file
      - to_speaker_turns
  - path: tests/e2e_smoke_test.rs
    uses:
      - parse_rttm_file
      - to_speaker_turns
      - write_rttm
invariants:
  - id: roundtrip
    rule: write_rttm ∘ parse_rttm preserves segments (within formatting).
    proof:
      kind: unit-test
      target: src/rttm::mod::tests
      command: cargo test --lib rttm
verification:
  pre_change:
    - cargo test --lib rttm
  full:
    - cargo test --lib rttm
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding RTTM variant support.
    - Tuning parse strictness.
  forbidden_mutations:
    - Changing RttmSegment field types without migration lease.
  escalation:
    - Changes to RTTM parse/write format semantics.
---

# src/rttm

RTTM file parsing, grouping, and writing.
