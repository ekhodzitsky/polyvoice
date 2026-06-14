---
schema_version: 1
kind: module_contract
module: src/streaming
level: subsystem
layer: algorithm
purpose: >
  Owns the online (streaming) diarization pipeline: windowed processing,
  incremental clustering, and streaming state management. Does NOT own the
  offline pipeline (pipeline.rs) or model inference.
status: pilot
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/streaming/
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
  - name: StreamingPipeline
    kind: struct
    visibility: public
    contract: >
      Online diarization pipeline that processes audio incrementally.
    proof:
      kind: unit-test
      target: src/streaming::mod::tests
      command: cargo test --lib streaming
  - name: StreamingError
    kind: enum
    visibility: public
    contract: >
      Error type for streaming pipeline failures.
    proof:
      kind: unit-test
      target: src/streaming::mod::tests
      command: cargo test --lib streaming
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: DiarizationConfig, SpeakerTurn, ClusterConfig.
    - module: vad
      scope: trait
      reason: VoiceActivityDetector for speech segmentation.
    - module: window
      scope: utility
      reason: WindowBuffer for audio framing.
    - module: cluster
      scope: algorithm
      reason: SpeakerCluster for incremental clustering.
    - module: embedder
      scope: trait
      reason: Embedder for embedding extraction.
  external: []
consumers:
  - path: .
    uses:
      - StreamingPipeline
      - StreamingError
      - polyvoice_internal
invariants:
  - id: turns-monotonic
    rule: Output speaker turns are monotonically ordered by time.
    proof:
      kind: unit-test
      target: src/streaming::mod::tests
      command: cargo test --lib streaming
  - id: turns-cumulative
    rule: >
      turns() returns the cumulative history of every turn emitted across
      feed()/flush() calls in order; flush() does not reset it.
    proof:
      kind: unit-test
      target: src/streaming::mod::tests::turns_accumulates_across_feed_and_flush
      command: cargo test --lib streaming
verification:
  pre_change:
    - cargo test --lib streaming
  full:
    - cargo test --lib streaming
    - cargo test --test streaming_test --features onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Refactoring internal state management.
    - Adding buffering strategies.
  forbidden_mutations:
    - Removing StreamingPipeline::process_chunk or changing its signature.
  escalation:
    - Changes to public API of StreamingPipeline.
---

# src/streaming

Online (streaming) diarization pipeline.
