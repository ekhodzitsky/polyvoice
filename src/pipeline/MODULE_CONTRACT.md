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
  - name: LegacyPipeline
    kind: struct
    visibility: public
    contract: >
      Offline BYO diarization pipeline. Holds DiarizationConfig and VadConfig.
      run(samples, embedder, vad) orchestrates VAD -> window embed -> AHC ->
      merge. Deprecated alias `pipeline::Pipeline` kept for downstream
      compatibility; the crate-root `Pipeline` re-export is pipeline v2.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::pipeline_new_with_defaults
      command: cargo test --lib pipeline
  - name: LegacyPipelineError
    kind: enum
    visibility: public
    contract: >
      Error type for pipeline failures (invalid config via
      DiarizationConfig::validate, VAD, embedding, clustering, WAV I/O,
      sample-rate mismatch, no speech, audio too long). Clustering carries a
      typed clusterer::ClustererError and exists only under feature clusterer
      (no producer without it). Distinct from pipeline_v2::PipelineError (the
      crate-root `PipelineError`).
    proof:
      kind: integration-test
      target: tests/der_regression_test.rs
      command: cargo test --test der_regression_test --features onnx,download
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
      reason: Prune helpers and free AHC fallback when clusterer feature is off.
    - module: clusterer
      scope: algorithm
      reason: AhcClusterer (when feature clusterer) for speaker grouping.
    - module: wav
      scope: io
      reason: WAV file reading for pipeline input.
  external: []
consumers:
  - path: src/bin/polyvoice.rs
    uses:
      - LegacyPipeline
      - LegacyPipelineError
  - path: src/bin/polyvoice-bench.rs
    uses:
      - LegacyPipeline
  - path: tests/der_regression_test.rs
    uses:
      - LegacyPipeline
  - path: docs/library-mode.md
    uses:
      - LegacyPipeline
invariants:
  - id: pipeline-construction
    rule: LegacyPipeline::new with default config constructs successfully.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::pipeline_new_with_defaults
      command: cargo test --lib pipeline
  - id: audio-too-long-guard
    rule: LegacyPipeline rejects audio longer than config.max_duration_secs.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::audio_too_long_error
      command: cargo test --lib pipeline
  - id: config-validated-before-run
    rule: run/run_with_clusterer reject an invalid DiarizationConfig
      (zero/inverted window geometry, cosine threshold outside [-1, 1]) with
      InvalidConfig instead of panicking in the window iterator.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::run_rejects_invalid_window_geometry
      command: cargo test --lib pipeline
  - id: clustering-errors-are-typed
    rule: with feature clusterer, inconsistent embedding dimensions surface as
      Clustering(ClustererError::DimMismatch); no silent free-AHC fallback.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::run_surfaces_dim_mismatch_as_clustering_error
      command: cargo test --lib pipeline --features clusterer
  - id: wav-sample-rate-match
    rule: run_from_wav rejects WAV files whose sample rate differs from
      config.window.sample_rate.
    proof:
      kind: unit-test
      target: src/pipeline::mod::tests::wav_sample_rate_mismatch_error
      command: cargo test --lib wav_sample_rate_mismatch_error
  - id: pipeline-result-valid
    rule: LegacyPipeline output turns are monotonically ordered and
      non-overlapping (before overlap detection).
    proof:
      kind: integration-test
      target: tests/der_regression_test.rs
      command: cargo test --test der_regression_test --features onnx,download
verification:
  pre_change:
    - cargo check --all-features
  full:
    - cargo test --test der_regression_test --features onnx,download
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Refactoring internal orchestration order.
    - Adding logging or telemetry.
    - Wiring Clusterer trait instead of free AHC (DER-gated).
  forbidden_mutations:
    - Removing LegacyPipeline::run() or changing its signature without
      updating library-mode docs and integration tests.
    - Changing the output DiarizationResult shape without consumer updates.
  escalation:
    - Adding new pipeline stages that change output semantics.
    - Changes to error variants that consumers match on.
---

# src/pipeline

Always-on offline BYO diarization pipeline
(`polyvoice::pipeline::LegacyPipeline`; the crate root re-exports pipeline v2
as `Pipeline` when its feature gate is on).
Production ONNX default is [`pipeline_v2`](../pipeline_v2/).
