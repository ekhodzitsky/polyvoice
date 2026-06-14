---
schema_version: 1
kind: module_contract
module: src/pipeline_v2
level: subsystem
layer: orchestration
purpose: >
  Owns the M6a v1.0 trait-based diarization pipeline (Powerset segmentation ->
  ResNet34 embedder -> clusterer -> overlap resegmentation) and its builder.
  This is the EXPERIMENTAL modern path: it was reverted from the CLI default
  after the 0.6.1 long-form DER regression. The validated default remains the
  legacy `pipeline` module. Does NOT own the component implementations
  (segmentation, embedder, clusterer, resegmentation) — it only wires them.
status: experimental
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
    - graduating v2 to the CLI/Python default
surface:
  - name: Pipeline
    kind: struct
    visibility: public
    contract: >
      The wired v1.0 pipeline. `run(&samples, sr) -> Result<DiarizationResult,
      PipelineError>` rejects a sample rate mismatching the config, skips
      segments below MIN_EMBED_SECS (0.20s) and non-finite embeddings, clusters
      primary segments, optionally resegments overlaps, then merges and returns
      turns sorted by start time. `builder()` returns a PipelineBuilder.
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
      Pipeline configuration: profile, clusterer kind, execution provider,
      sample rate, speech/gap thresholds, embedder pool size, overlap toggle.
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ClustererKind
    kind: enum
    visibility: public
    contract: >
      Selects the clustering backend (Ahc { threshold } | NmeSc). NmeSc falls
      back to AHC when the `spectral` feature is absent.
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ExecutionProvider
    kind: enum
    visibility: public
    contract: >
      Requested ONNX execution provider (Cpu | CoreMl | Nnapi | XnnPack).
      KNOWN GAP: only CoreML is wired into ort today; Nnapi/XnnPack are accepted
      but fall back to CPU (see TODO.md).
    proof:
      kind: unit-test
      target: src/pipeline_v2::config::tests
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: PipelineError
    kind: enum
    visibility: public
    contract: >
      Error type for build/run: UnsupportedSampleRate, Segmentation, Embedding,
      Clustering, Resegment, Config, Registry, ModelLoad.
    proof:
      kind: unit-test
      target: src/pipeline_v2::mod::tests::pipeline_run_unsupported_sample_rate_returns_err
      command: cargo nextest run --lib pipeline_v2 --features "onnx,download,segmentation,embedder,clusterer,resegmentation"
  - name: ConfigError
    kind: enum
    visibility: public
    contract: >
      Builder validation errors (missing registry, custom component in profile,
      registry in custom profile, missing custom component, unknown model).
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
  - path: .
    uses:
      - polyvoice_internal
invariants:
  - id: feature-completeness-gate
    rule: >
      The module fails to compile (compile_error!) unless ALL of onnx, download,
      segmentation, embedder, clusterer, resegmentation features are enabled —
      half-wired feature combos cannot ship.
    proof:
      kind: compile-time
      target: src/pipeline_v2::mod (compile_error!)
      command: cargo hack check --feature-powerset --depth 2 --lib
  - id: experimental-not-default
    rule: >
      pipeline_v2 is NOT the validated default. It was reverted from the CLI
      default after the 0.6.1 long-form DER regression; the legacy `pipeline`
      module is the validated default and `--v2` is opt-in / not recommended for
      long-form audio. Graduating v2 to default requires a migration lease.
    proof:
      kind: doc-invariant
      target: src/bin/polyvoice.rs (--v2 help text) + AGENTS.md (0.6.1 incident)
      command: grep -n "not recommended for long-form" src/bin/polyvoice.rs
  - id: sample-rate-guard
    rule: run() returns UnsupportedSampleRate when sr != config.sample_rate.
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
    - Wiring additional execution providers (NNAPI/XNNPACK) into the pipeline.
    - Adding scoring/resegmentation stages behind the existing trait seams.
    - Improving the builder validation messages.
  forbidden_mutations:
    - Making pipeline_v2 the CLI/Python default without a migration lease and
      proven long-form DER parity (the 0.6.1 incident must not repeat).
    - Removing the compile_error! feature-completeness gate.
    - Changing Pipeline::run's public signature without updating consumers.
  escalation:
    - Graduating v2 to default (requires DER regression evidence).
    - Any change to PipelineConfig/ClustererKind/ExecutionProvider public shape.
    - Changes to the MIN_EMBED_SECS / NaN-guard behavior.
---

# src/pipeline_v2

Experimental M6a v1.0 trait-based diarization pipeline + builder. See
[README.md](README.md) for the architecture and opt-in instructions and
[TODO.md](TODO.md) for the graduate-to-default work. **Not the validated
default** — the legacy `pipeline` module is.
