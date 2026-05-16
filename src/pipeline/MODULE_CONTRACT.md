---
schema_version: 1
kind: module_contract
module: src/pipeline
level: subsystem
layer: orchestration
purpose: >
  Owns the offline diarization pipeline orchestration: segment → embed →
  cluster → resegment → merge → emit DiarizationResult. Does NOT own the
  individual algorithm implementations (those live in segmentation, embedder,
  clusterer, resegmentation, vad).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/pipeline/
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
  - name: Pipeline
    kind: struct
    visibility: public
    contract: >
      Offline diarization pipeline. Holds config and VAD config.
      run() orchestrates the full diarization flow.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::pipeline_new_with_defaults
      command: cargo test --lib pipeline
  - name: PipelineError
    kind: enum
    visibility: public
    contract: >
      Error type for pipeline failures (VAD, embedding, clustering).
    proof:
      kind: integration-test
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test --features onnx,download
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: DiarizationConfig, DiarizationResult, SpeakerTurn input/output shapes.
    - module: vad
      scope: trait
      reason: VoiceActivityDriver trait for speech segmentation.
    - module: ahc
      scope: algorithm
      reason: Agglomerative clustering for speaker grouping.
    - module: wav
      scope: io
      reason: WAV file reading for pipeline input.
  external: []
consumers:
  - path: src/ffi/mod.rs
    uses:
      - Pipeline
      - PipelineError
  - path: tests/e2e_smoke_test.rs
    uses:
      - Pipeline
  - path: tests/der_regression_test.rs
    uses:
      - Pipeline
  - path: tests/der_ami_baseline_test.rs
    uses:
      - Pipeline
invariants:
  - id: pipeline-construction
    rule: Pipeline::new with default config constructs successfully.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::pipeline_new_with_defaults
      command: cargo test --lib pipeline
  - id: audio-too-long-guard
    rule: Pipeline rejects audio longer than config.max_duration_secs.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::audio_too_long_error
      command: cargo test --lib pipeline
  - id: pipeline-result-valid
    rule: Pipeline output turns are monotonically ordered and non-overlapping
      (before overlap detection).
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
    - Refactoring internal orchestration order.
    - Adding logging or telemetry.
    - Adding new pipeline stages behind existing config.
  forbidden_mutations:
    - Removing Pipeline::run() or changing its signature without updating FFI
      and all integration tests.
    - Changing the output DiarizationResult shape without consumer updates.
  escalation:
    - Adding new pipeline stages that change output semantics.
    - Changes to error variants that consumers match on.
---

# src/pipeline

Offline diarization pipeline orchestration.
