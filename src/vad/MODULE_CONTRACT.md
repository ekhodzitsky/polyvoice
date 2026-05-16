---
schema_version: 1
kind: module_contract
module: src/vad
level: subsystem
layer: algorithm
purpose: >
  Owns the VoiceActivityDetector trait, energy-based VAD, VAD state machine,
  and speech segmentation utilities. Does NOT own ONNX-based SileroVAD
  (that lives in silero_vad.rs).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/vad/
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
  - name: VoiceActivityDetector
    kind: trait
    visibility: public
    contract: >
      Core VAD trait: process audio samples → speech/non-speech decisions.
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
  - name: EnergyVad
    kind: struct
    visibility: public
    contract: >
      Simple energy-threshold VAD implementation.
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
  - name: VadConfig
    kind: struct
    visibility: public
    contract: >
      Configuration for VAD parameters (thresholds, frame sizes).
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
  - name: segment_speech
    kind: function
    visibility: public
    contract: >
      Segments audio into speech regions using a VAD driver.
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
  - name: VadStateMachine
    kind: struct
    visibility: public
    contract: >
      Hysteresis state machine for VAD decision smoothing.
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: DiarizationConfig for VAD parameters.
  external: []
consumers:
  - path: src/pipeline/mod.rs
    uses:
      - VoiceActivityDetector
      - VadConfig
      - segment_speech
  - path: src/streaming/mod.rs
    uses:
      - VoiceActivityDetector
      - VadConfig
      - VadEvent
      - VadStateMachine
  - path: src/silero_vad/mod.rs
    uses:
      - VoiceActivityDetector
  - path: src/ffi/mod.rs
    uses:
      - VadConfig
  - path: tests/vad_test.rs
    uses:
      - EnergyVad
invariants:
  - id: vad-monotonic
    rule: segment_speech returns non-overlapping, monotonically ordered segments.
    proof:
      kind: unit-test
      target: src/vad::mod::tests
      command: cargo test --lib vad
verification:
  pre_change:
    - cargo test --lib vad
  full:
    - cargo test --lib vad
    - cargo test --test vad_test
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning energy threshold heuristics.
    - Adding new VAD implementations.
  forbidden_mutations:
    - Changing VoiceActivityDetector trait without migration lease.
  escalation:
    - Changes to VoiceActivityDetector trait or associated types.
---

# src/vad

Voice Activity Detection trait, energy-based VAD, and speech segmentation.
