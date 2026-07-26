---
schema_version: 1
kind: module_contract
module: src/ecapa
level: leaf
layer: algorithm
purpose: >
  Owns the shared fbank + ONNX speaker embedding engine (FbankOnnxExtractor)
  used by WeSpeaker / CAM++ / ERes2Net adapters and the CLI --legacy path.
  Implements Embedder directly. Module name is historical (not ECAPA-only).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/ecapa/
  context_budget:
    max_files: 6
    max_source_lines: 400
    max_contract_lines: 120
    max_readme_lines: 80
    max_todo_lines: 40
authority:
  write_policy: single_active_write_lease
  orchestrator: polyvoice-core
  read_agents: many_allowed
  migration_lease_required:
    - cross-workcell write
    - public surface migration
    - renaming the module away from ecapa
surface:
  - name: FbankOnnxExtractor
    kind: struct
    visibility: public
    contract: >
      Pooled fbank→ONNX Embedder. Input 16 kHz mono; output L2-normalized
      embedding of configured dim. Prefer architecture adapters in embedder
      when the model family is fixed.
    proof:
      kind: unit-test
      target: src/ecapa::tests
      command: cargo test --lib ecapa --features onnx
dependencies:
  internal:
    - module: embedder
      scope: trait
      reason: Implements Embedder / EmbedderError.
    - module: features
      scope: algorithm
      reason: FbankExtractor + CMVN.
    - module: onnx
      scope: infrastructure
      reason: RuntimeSession pool and InferenceRuntime.
  external: []
consumers:
  - path: src/embedder/mod.rs
    uses:
      - FbankOnnxExtractor
  - path: src/bin/polyvoice.rs
    uses:
      - FbankOnnxExtractor
  - path: src/bin/polyvoice-bench.rs
    uses:
      - FbankOnnxExtractor
invariants:
  - id: implements-embedder
    rule: FbankOnnxExtractor implements Embedder (not EmbeddingExtractor).
    proof:
      kind: compile-time
      target: src/ecapa/mod.rs
      command: cargo check --features onnx --lib
verification:
  pre_change:
    - cargo test --lib ecapa --features onnx
  full:
    - cargo test --lib ecapa --features onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Performance tweaks to pooling / fbank path.
    - Renaming module with migration lease.
  forbidden_mutations:
    - Removing FbankOnnxExtractor without deprecation cycle.
  escalation:
    - Changing ONNX I/O layout assumptions.
---

# src/ecapa

Shared fbank + ONNX embedder engine (`FbankOnnxExtractor` implements `Embedder`).
