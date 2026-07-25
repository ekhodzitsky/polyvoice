---
schema_version: 1
kind: module_contract
module: src/pipeline
level: subsystem
layer: orchestration
purpose: >
  Owns the always-on offline BYO diarization pipeline: VAD speech regions ->
  sliding-window Embedder -> free AHC clustering -> merge -> DiarizationResult.
  This is the ort-free library surface and the CLI --legacy escape hatch.
  Production ONNX (CLI/FFI/Python/MCP default since 0.11) lives in pipeline_v2.
  Does NOT own algorithm implementations (vad, embedder, ahc, wav).
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
      Offline BYO diarization pipeline. Holds DiarizationConfig and VadConfig.
      run(samples, embedder, vad) orchestrates VAD -> window embed -> AHC ->
      merge. Re-exported at crate root as polyvoice::Pipeline.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::pipeline_new_with_defaults
      command: cargo test --lib pipeline
  - name: PipelineError
    kind: enum
    visibility: public
    contract: >
      Error type for pipeline failures (VAD, embedding, clustering, WAV I/O,
      sample-rate mismatch, no speech, audio too long). Distinct from
      pipeline_v2::PipelineError.
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
      reason: VoiceActivityDetector trait for speech segmentation.
    - module: embedder
      scope: trait
      reason: Embedder trait for BYO speaker vectors.
    - module: ahc
      scope: algorithm
      reason: Agglomerative clustering for speaker grouping (not Clusterer trait).
    - module: wav
      scope: io
      reason: WAV file reading for pipeline input.
  external: []
consumers:
  - path: src/bin/polyvoice.rs
    uses:
      - Pipeline
      - PipelineError
  - path: src/bin/polyvoice-bench.rs
    uses:
      - Pipeline
  - path: tests/e2e_smoke_test.rs
    uses:
      - Pipeline
  - path: tests/der_regression_test.rs
    uses:
      - Pipeline
  - path: tests/der_ami_baseline_test.rs
    uses:
      - Pipeline
  - path: docs/library-mode.md
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
  - id: wav-sample-rate-match
    rule: run_from_wav rejects WAV files whose sample rate differs from
      config.window.sample_rate.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::wav_sample_rate_mismatch_error
      command: cargo test --lib wav_sample_rate_mismatch_error
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
    - Wiring Clusterer trait instead of free AHC (DER-gated).
  forbidden_mutations:
    - Removing Pipeline::run() or changing its signature without updating
      library-mode docs and integration tests.
    - Changing the output DiarizationResult shape without consumer updates.
  escalation:
    - Adding new pipeline stages that change output semantics.
    - Changes to error variants that consumers match on.
---

# src/pipeline

Always-on offline BYO diarization pipeline (crate-root `polyvoice::Pipeline`).
Production ONNX default is [`pipeline_v2`](../pipeline_v2/).
