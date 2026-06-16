---
schema_version: 1
kind: module_contract
module: src/asr
level: leaf
layer: interface
purpose: >
  Defines the ASR (speech-to-text) trait and its error type — the stable,
  trait-only interface that the opt-in polyvoice-asr companion crate implements
  and the word->speaker join targets. Contains NO backend, ort, or model
  dependency; pure-Rust and wasm-clean.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/asr/
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
  - name: Asr
    kind: trait
    visibility: public
    contract: >
      Object-safe speech-to-text backend: transcribe(audio, sample_rate) ->
      Vec<Word> with word-level timestamps. Implemented by the opt-in
      polyvoice-asr crate; usable as Box<dyn Asr>. No backend lives here.
    proof:
      kind: unit-test
      target: src/asr::mod::tests::asr_is_object_safe
      command: cargo test --lib asr
  - name: AsrError
    kind: enum
    visibility: public
    contract: >
      thiserror enum for ASR failures: UnsupportedSampleRate, InferenceFailed,
      ModelIo, Backend.
    proof:
      kind: unit-test
      target: src/asr::mod::tests::asr_error_displays
      command: cargo test --lib asr
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Word and SampleRate are the trait input/output shapes.
  external: []
consumers:
  - path: .
    uses:
      - Asr
      - AsrError
invariants:
  - id: no-backend-deps
    rule: This module pulls no ort/model/heavy dependency; the default build stays wasm-clean and ~30 MB.
    proof:
      kind: build
      target: default + wasm32
      command: cargo build --no-default-features && cargo build --target wasm32-unknown-unknown --no-default-features --lib
  - id: object-safe
    rule: Asr is object-safe (Box<dyn Asr> compiles).
    proof:
      kind: unit-test
      target: src/asr::mod::tests::asr_is_object_safe
      command: cargo test --lib asr
verification:
  pre_change:
    - cargo test --lib asr
  full:
    - cargo test --lib asr
    - cargo build --no-default-features
    - cargo build --target wasm32-unknown-unknown --no-default-features --lib
agent_policy:
  allowed_mutations:
    - Adding trait methods with defaults.
    - Adding error variants.
    - Documentation / tests.
  forbidden_mutations:
    - Adding any backend implementation or ort/parakeet-rs dependency here (belongs in polyvoice-asr).
    - Making the trait non-object-safe.
  escalation:
    - Changing the Asr trait signature.
---

# src/asr

Trait-only ASR interface (`Asr` + `AsrError`). The actual backend (e.g. Parakeet
TDT via parakeet-rs) lives in the opt-in `polyvoice-asr` companion crate; this
module stays pure-Rust and wasm-clean so the default build carries no ASR engine.
