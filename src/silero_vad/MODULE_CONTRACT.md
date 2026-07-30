---
schema_version: 1
kind: module_contract
module: src/silero_vad
level: subsystem
layer: algorithm
purpose: >
  Owns the ONNX-based Silero VAD implementation. Implements
  VoiceActivityDetector trait from vad.rs. Does NOT own model download
  (models.rs) or generic VAD interfaces.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/silero_vad/
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
  - name: SileroVad
    kind: struct
    visibility: public
    contract: >
      ONNX-backed Silero VAD. Implements VoiceActivityDetector.
      Requires 512-sample frames at 16kHz. The session is a RuntimeSession
      built via onnx::build_session_with_ep (InferenceRuntime); this module
      does not import ort::. new() keeps the historical CPU default;
      with_ep() selects an execution provider explicitly. Construction
      failures are the typed SileroVadError (ZeroChunkSize / Session),
      never anyhow.
    proof:
      kind: unit-test
      target: src/silero_vad::mod::tests
      command: cargo test --lib silero_vad --features onnx
dependencies:
  internal:
    - module: vad
      scope: trait
      reason: VoiceActivityDetector trait implementation.
    - module: onnx
      scope: inference
      reason: InferenceRuntime / OrtSession / build_session_with_ep.
  external: []
consumers:
  - path: src/ffi/mod.rs
    uses:
      - SileroVad
  - path: tests/der_regression_test.rs
    uses:
      - SileroVad
invariants:
  - id: frame-size
    rule: SileroVad requires exactly 512 samples per process call.
    proof:
      kind: unit-test
      target: src/silero_vad::mod::tests
      command: cargo test --lib silero_vad --features onnx
verification:
  pre_change:
    - cargo test --lib silero_vad --features onnx
  full:
    - cargo test --lib silero_vad --features onnx
    - cargo test --test der_regression_test --features onnx,download
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Tuning internal thresholds.
    - Adding state reset methods.
  forbidden_mutations:
    - Changing required frame size without updating all consumers.
  escalation:
    - Changes to SileroVad constructor signature or frame requirements.
---

# src/silero_vad

ONNX-based Silero Voice Activity Detector.
