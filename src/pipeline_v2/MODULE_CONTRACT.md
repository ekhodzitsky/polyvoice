---
schema_version: 1
kind: module_contract
module: src/pipeline_v2
level: subsystem
layer: orchestration
purpose: >
  Owns the trait-based production ONNX diarization pipeline (Powerset
  segmentation -> ResNet34 embedder -> clusterer -> overlap resegmentation)
  and its builder. Since 0.11 this is the CLI / FFI / Python default
  (v2 + VBx after a full VoxConverse-test + AMI-test DER gate). The ort-free
  BYO surface remains the separate always-on `pipeline` module. Does NOT own
  the component implementations (segmentation, embedder, clusterer,
  resegmentation) — it only wires them.
status: stable
owners:
  - polyvoice-core
workcell:
  type: composite
  parent: ""
  children: []
  owns_paths:
    - src/pipeline_v2/
  context_budget:
    max_files: 12
    max_source_lines: 1800
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
    - changing the CLI/Python/FFI default path away from this module
surface:
  - name: Pipeline
    kind: struct
    visibility: public
    contract: >
      The wired production ONNX pipeline. `run(&samples, sr) ->
      Result<DiarizationResult, PipelineError>` rejects a sample rate
      mismatching the config, skips segments below MIN_EMBED_SECS (0.20s) and
      non-finite embeddings, clusters primary segments, optionally resegments
      overlaps, then merges and returns turns sorted by start time.
      `builder()` returns a PipelineBuilder.
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: PipelineBuilder
    kind: struct
    visibility: public
    contract: >
      Validating builder. Profile::{Mobile,Balanced} require .with_models_from()
      and reject custom components; Profile::Custom requires the three custom
      components and rejects a registry. `build()` validates then constructs.
    proof:
      kind: unit-test
      target: src/pipeline_v2::builder::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: PipelineConfig
    kind: struct
    visibility: public
    contract: >
      Pipeline configuration: profile, clusterer kind (AHC / NME-SC / VBx),
      execution provider, sample rate, speech/gap thresholds, embedder pool
      size, overlap toggle, optional dense embed window and binarization,
      optional AS-norm score normalization (as_norm) and per-domain scoring
      profile (domain — replaces the AHC threshold with the profile's
      calibrated value at build time).
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ClustererKind
    kind: enum
    visibility: public
    contract: >
      Selects the clustering backend (Ahc { threshold } | NmeSc | Vbx).
      NmeSc falls back to AHC when the `spectral` feature is absent. VBx
      requires the `vbx` feature and PLDA params (dir, env, or registry).
      KmeansClusterer remains available for Custom profile injection.
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ExecutionProvider
    kind: enum
    visibility: public
    contract: >
      Requested ONNX execution provider (Cpu | CoreMl | Nnapi | XnnPack | Cuda).
      KNOWN GAP: providers not compiled into the build log a warning and fall
      back to CPU (see TODO.md).
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: PipelineError
    kind: enum
    visibility: public
    contract: >
      Error type for build/run: UnsupportedSampleRate, AudioTooLong, Segmentation, Embedding,
      Clustering, Resegment, Config, Registry.
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests::pipeline_run_unsupported_sample_rate_returns_err
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ConfigError
    kind: enum
    visibility: public
    contract: >
      Builder validation errors (missing registry, custom component in profile,
      registry in custom profile, missing custom component, unknown model,
      model load failure with typed source).
    proof:
      kind: unit-test
      target: src/pipeline_v2::builder::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
dependencies:
  internal:
    - module: segmentation
      scope: trait
      reason: Segmenter produces overlap-aware RawSegments.
    - module: embedder
      scope: trait
      reason: Embedder + apply_overlap_mask produce speaker vectors.
    - module: clusterer
      scope: trait
      reason: Clusterer assigns speaker labels to embeddings.
    - module: resegmentation
      scope: trait
      reason: Resegmenter reassigns overlap regions to two speakers.
    - module: models
      scope: infrastructure
      reason: ModelRegistry resolves profile models for the builder.
    - module: types
      scope: data-shape
      reason: DiarizationResult, SpeakerTurn, Segment, SampleRate, Profile.
    - module: utils
      scope: utility
      reason: l2_normalize, merge_segments.
  external: []
consumers:
  - path: src/bin/polyvoice.rs
    uses:
      - Pipeline
      - PipelineBuilder
      - PipelineConfig
      - ClustererKind
  - path: src/bin/polyvoice-bench.rs
    uses:
      - Pipeline
      - PipelineConfig
      - ClustererKind
  - path: src/bin/polyvoice-mcp.rs
    uses:
      - Pipeline
      - PipelineConfig
      - ClustererKind
  - path: src/ffi/mod.rs
    uses:
      - Pipeline
  - path: python/src/lib.rs
    uses:
      - Pipeline
      - PipelineConfig
      - ClustererKind
invariants:
  - id: feature-completeness-gate
    rule: >
      The module is only compiled when ALL of onnx, download, segmentation,
      embedder, clusterer, resegmentation features are enabled (cfg-gated in
      lib.rs, with a compile_error! backstop) — half-wired feature combos
      exclude it entirely.
    proof:
      kind: compile-time
      target: src/pipeline_v2::mod (compile_error!)
      command: cargo hack check --feature-powerset --depth 2 --lib
  - id: production-default-surface
    rule: >
      pipeline_v2 is the validated ONNX production default for CLI (since 0.11,
      v2 + VBx), FFI, Python, and MCP. The always-on `pipeline` module remains
      the ort-free BYO / `--legacy` escape hatch. Changing the default path
      requires a migration lease and DER regression evidence.
    proof:
      kind: doc-invariant
      target: src/bin/polyvoice.rs (module docs) + CHANGELOG.md (0.11.0)
      command: rg -n "v2 \\+ VBx|CLI default" src/bin/polyvoice.rs CHANGELOG.md
  - id: sample-rate-guard
    rule: run() returns UnsupportedSampleRate when sr != config.sample_rate.
  - id: audio-length-cap
    rule: run() returns AudioTooLong when samples.len() > MAX_AUDIO_SAMPLES (1 h @ 16 kHz).
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests::pipeline_run_unsupported_sample_rate_returns_err
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - id: nan-and-short-segment-guard
    rule: >
      Segments shorter than MIN_EMBED_SECS (0.20s) and any non-finite embedding
      are skipped, never reaching the clusterer (NaN-collapse defense).
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests::pipeline_skips_non_finite_embeddings
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - id: turns-monotonic
    rule: Output turns are sorted by start time regardless of input segment order.
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests::pipeline_turns_are_monotonically_ordered
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
verification:
  pre_change:
    - cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  full:
    - cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
    - cargo nextest run --test pipeline_v2_integration --features "onnx,segmentation,embedder,clusterer,resegmentation,download" --run-ignored only
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Wiring additional execution providers (NNAPI/XNNPACK/CUDA) into the pipeline.
    - Adding scoring/resegmentation stages behind the existing trait seams.
    - Improving the builder validation messages.
  forbidden_mutations:
    - Changing the CLI/FFI/Python default away from this module without a
      migration lease and proven DER evidence on VoxConverse-test + AMI-test.
    - Removing the compile_error! feature-completeness gate.
    - Changing Pipeline::run's public signature without updating consumers.
  escalation:
    - Changing the production default path (requires DER regression evidence).
    - Any change to PipelineConfig/ClustererKind/ExecutionProvider public shape.
    - Changes to the MIN_EMBED_SECS / NaN-guard behavior.
---

# src/pipeline_v2

Production ONNX diarization pipeline + builder (CLI/FFI/Python/MCP default since
0.11). See [README.md](README.md) for architecture and [TODO.md](TODO.md) for
remaining hardening work. Ort-free BYO consumers use the separate
[`pipeline`](../pipeline/) module.
