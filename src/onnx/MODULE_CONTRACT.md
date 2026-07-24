---
schema_version: 1
kind: module_contract
module: src/onnx
level: subsystem
layer: infrastructure
purpose: >
  Owns ONNX model validation (header checks), the InferenceRuntime trait,
  the OrtSession implementation, and the legacy OnnxEmbeddingExtractor.
  Does NOT own model download/registry (models/) or specific model adapters
  (embedder.rs, ecapa.rs).
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
  - name: InferenceRuntime
    kind: trait
    visibility: public
    contract: >
      Minimal pluggable inference session: named/ordered tensor run + input
      names. Neural stages depend only on this trait; they must not import ort::.
    proof:
      kind: unit-test
      target: src/onnx::runtime::tests
      command: cargo test --lib onnx --features onnx
  - name: OrtSession
    kind: struct
    visibility: public
    contract: >
      Default InferenceRuntime implementation wrapping ort. Sole production
      module allowed to import ort:: (src/onnx/ort_session.rs).
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - name: validate_onnx_header
    kind: function
    visibility: public
    contract: >
      Validates that a file has a valid ONNX protobuf header.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - name: ExecutionProvider
    kind: enum
    visibility: public
    contract: >
      Canonical ort-specific EP selector (Cpu/CoreMl/Nnapi/Cuda/XnnPack)
      with a target-aware auto(). Not part of InferenceRuntime; stages pass
      it only at construction via build_session_with_ep. pipeline_v2::config
      re-exports it.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - name: build_session_with_ep
    kind: function
    visibility: public
    contract: >
      THE single session constructor for embedding + segmentation paths.
      Validates the ONNX header BEFORE the backend parses the file, optionally
      pins intra-op threads, then registers the requested EP. Returns OrtSession.
      Unwired providers warn (tracing) and fall back to CPU.
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
      Legacy ONNX embedding extractor wrapper (pooled OrtSession).
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
      reason: Default InferenceRuntime backend (confined to ort_session.rs).
consumers:
  - path: .
    uses:
      - validate_onnx_header
      - OnnxValidationError
      - OnnxEmbeddingExtractor
      - InferenceRuntime
      - OrtSession
      - build_session_with_ep
      - ExecutionProvider
invariants:
  - id: header-validation-false-positive-rate
    rule: validate_onnx_header rejects non-ONNX files and accepts valid ONNX files.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: cargo test --lib onnx --features onnx
  - id: ort-confined-to-ort-session
    rule: >
      No production module outside src/onnx/ort_session.rs may import ort::.
      New neural stages must use InferenceRuntime / build_session_with_ep only.
    proof:
      kind: unit-test
      target: src/onnx::mod::tests
      command: rg 'ort::' src --type rust
verification:
  pre_change:
    - cargo test --lib onnx --features onnx
  full:
    - cargo test --lib onnx --features onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new validation checks.
    - Adding InferenceRuntime backends behind feature flags.
  forbidden_mutations:
    - Removing validate_onnx_header without migration lease.
    - Re-introducing ort:: imports into neural stages outside ort_session.rs.
  escalation:
    - Changes to ONNX validation semantics.
    - Changes to InferenceRuntime surface used by stages.
---

# src/onnx

ONNX validation, runtime-agnostic inference trait, and default ort backend.

## Rule for new neural stages

**Do not import `ort::`.** Load sessions via `build_session_with_ep` (or a future
backend factory) and run inference through `InferenceRuntime`. The only module
allowed to import `ort::` is `src/onnx/ort_session.rs`.
