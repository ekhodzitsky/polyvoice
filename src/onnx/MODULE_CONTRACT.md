---
schema_version: 1
kind: module_contract
module: src/onnx
level: subsystem
layer: infrastructure
purpose: >
  Owns ONNX model validation (header checks) and the legacy OnnxEmbeddingExtractor
  struct. Does NOT own model download/registry (models/) or specific model
  adapters (embedder.rs, ecapa.rs).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/onnx/
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
  - name: validate_onnx_header
    kind: function
    visibility: public
    contract: >
      Validates that a file has a valid ONNX protobuf header.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - name: OnnxValidationError
    kind: struct
    visibility: public
    contract: >
      Error type for ONNX validation failures.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - name: OnnxEmbeddingExtractor
    kind: struct
    visibility: public
    contract: >
      Legacy ONNX embedding extractor wrapper.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
dependencies:
  internal:
    - module: embedding
      scope: trait
      reason: EmbeddingExtractor trait implementation.
    - module: types
      scope: data-shape
      reason: DiarizationConfig.
  external:
    - name: ort
      scope: ml-runtime
      reason: ONNX Runtime inference.
consumers:
  - path: src/lib.rs
    uses:
      - OnnxEmbeddingExtractor
invariants:
  - id: header-validation-false-positive-rate
    rule: validate_onnx_header rejects non-ONNX files and accepts valid ONNX files.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
verification:
  pre_change:
    - cargo test --lib onnx --features onnx
  full:
    - cargo test --lib onnx --features onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new validation checks.
  forbidden_mutations:
    - Removing validate_onnx_header without migration lease.
  escalation:
    - Changes to ONNX validation semantics.
---

# src/onnx

ONNX model validation and legacy embedding extractor wrapper.
