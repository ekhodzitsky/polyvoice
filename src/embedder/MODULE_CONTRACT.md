---
schema_version: 1
kind: module_contract
module: src/embedder
level: subsystem
layer: algorithm
purpose: >
  Owns the Embedder trait, overlap masking, embedder pooling, and ONNX-backed
  adapter implementations (CAM++, ResNet34). Does NOT own feature extraction
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
    visibility: public
    contract: >
      Lock-free pool of Embedder instances using crossbeam-queue.
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
dependencies:
  internal: []
  external:
    - name: ort
      scope: ml-runtime
      reason: ONNX inference for CAM++ and ResNet34 adapters.
    - name: crossbeam-queue
      scope: concurrency
      reason: Lock-free queue for EmbedderPool.
consumers:
  - path: src/streaming/mod.rs
    uses:
      - Embedder
  - path: tests/embedder_test.rs
    uses:
      - Embedder
      - CamPlusPlusExtractor
      - ResNet34Adapter
  - path: tests/loom_pool.rs
    uses:
      - EmbedderPool
invariants:
  - id: embedder-output-normalized
    rule: Embedder implementations must output L2-normalized embeddings
      (convention; enforced by adapters).
    proof:
      kind: integration-test
      target: tests/embedder_test.rs
      command: cargo test --test embedder_test --features onnx
  - id: pool-safe-concurrent-access
    rule: EmbedderPool is safe for concurrent pop/push without data races.
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
