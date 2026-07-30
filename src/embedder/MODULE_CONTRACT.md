---
schema_version: 1
kind: module_contract
module: src/embedder
level: subsystem
layer: algorithm
purpose: >
  Owns the Embedder trait, overlap masking, embedder pooling, and ONNX-backed
  adapter implementations (CAM++, ResNet34, ERes2NetV2 — named wrappers over
  one generic internal fbank+ONNX adapter). Does NOT own feature extraction
  (features.rs) or clustering.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/embedder/
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
  - name: Embedder
    kind: trait
    visibility: public
    contract: >
      Core embedder trait: extract speaker embedding from audio samples.
    proof:
      kind: unit-test
      target: src/embedder::mod::tests
      command: cargo test --lib embedder
  - name: EmbedderError
    kind: enum
    visibility: public
    contract: >
      Error type for embedder operations.
    proof:
      kind: unit-test
      target: src/embedder::mod::tests
      command: cargo test --lib embedder
  - name: EmbedderPool
    kind: struct
    visibility: private (cfg(test))
    contract: >
      Test-only blocking pool of Embedder instances using Mutex<Vec<E>>
      (utils::ObjectPool). Not part of the public API: production paths hold
      a Box<dyn Embedder> or pool ONNX sessions inside FbankOnnxExtractor.
    proof:
      kind: unit-test
      target: src/embedder::mod::tests
      command: cargo test --lib embedder
  - name: apply_overlap_mask
    kind: function
    visibility: public
    contract: >
      Masks embedding regions that overlap with multiple speakers.
    proof:
      kind: unit-test
      target: src/embedder::mod::tests
      command: cargo test --lib embedder
  - name: DummyExtractor
    kind: struct
    visibility: public
    contract: >
      Deterministic pseudo-random unit-vector embedder for tests and
      benchmarks (implements Embedder directly).
    proof:
      kind: unit-test
      target: src/embedder::mod::tests
      command: cargo test --lib embedder
  - name: CamPlusPlusExtractor
    kind: struct
    visibility: public
    contract: >
      ONNX-backed CAM++ embedding extractor.
    proof:
      kind: integration-test
      target: tests/embedder_test.rs
      command: cargo test --test embedder_test --features onnx
  - name: ResNet34Adapter
    kind: struct
    visibility: public
    contract: >
      ONNX-backed ResNet34 adapter wrapping FbankOnnxExtractor.
    proof:
      kind: integration-test
      target: tests/embedder_test.rs
      command: cargo test --test embedder_test --features onnx
  - name: ERes2NetV2Extractor
    kind: struct
    visibility: public
    contract: >
      ONNX-backed ERes2NetV2 adapter (192-d default, `with_dim` override)
      wrapping FbankOnnxExtractor.
    proof:
      kind: compile-time
      target: src/bin/polyvoice-measure.rs (constructs it); inference is the
        shared FbankOnnxExtractor engine covered by tests/embedder_test.rs
      command: cargo check --features "onnx,embedder"
dependencies:
  internal: []
  external:
    - name: ort
      scope: ml-runtime
      reason: ONNX inference for CAM++ and ResNet34 adapters.
consumers:
  - path: .
    uses:
      - Embedder
      - EmbedderError
      - apply_overlap_mask
      - DummyExtractor
      - CamPlusPlusExtractor
      - ResNet34Adapter
      - ERes2NetV2Extractor
      - ort
      - polyvoice_internal
invariants:
  - id: embedder-output-normalized
    rule: Embedder implementations must output L2-normalized embeddings
      (convention; enforced by adapters).
    proof:
      kind: integration-test
      target: tests/embedder_test.rs
      command: cargo test --test embedder_test --features onnx
  - id: pool-safe-concurrent-access
    rule: utils::ObjectPool checkout/return is safe for concurrent pop/push
      without data races (EmbedderPool and the ONNX session pool build on it).
    proof:
      kind: unit-test
      target: tests/loom_pool.rs
      command: cargo test --test loom_pool
verification:
  pre_change:
    - cargo test --lib embedder
  full:
    - cargo test --lib embedder
    - cargo test --test embedder_test --features onnx
    - cargo test --test loom_pool
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new Embedder implementations.
    - Optimizing overlap mask logic.
    - Pool sizing heuristics.
  forbidden_mutations:
    - Removing the Embedder trait without migration lease.
    - Changing Embedder::extract signature.
    - Removing L2 normalization from adapters.
  escalation:
    - Changes to Embedder trait or its associated types.
    - Adding new execution provider wiring that changes public API.
---

# src/embedder

Embedder trait, overlap masking, embedder pooling, and ONNX-backed adapters.
