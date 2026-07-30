---
schema_version: 1
kind: module_contract
module: src/fbank_onnx
level: leaf
layer: algorithm
purpose: >
  Owns the shared fbank + ONNX speaker embedding engine (FbankOnnxExtractor)
  used by WeSpeaker / CAM++ / ERes2Net adapters and the CLI --legacy path.
  Implements Embedder directly. Generic across model families (not ECAPA-only);
  renamed from `ecapa` (hard rename, no alias).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/fbank_onnx/
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
    - renaming the module away from fbank_onnx
surface:
  - name: FbankOnnxExtractor
    kind: struct
    visibility: public
    contract: >
      Pooled fbank→ONNX Embedder. Input 16 kHz mono; output L2-normalized
      embedding of configured dim. Prefer architecture adapters in embedder
      when the model family is fixed. Construction failures are the typed
      FbankExtractorError (EmptyPool vs SessionBuild), never anyhow.
    proof:
      kind: unit-test
      target: src/fbank_onnx::tests
      command: cargo test --lib fbank_onnx --features onnx
  - name: FbankExtractorError
    kind: enum
    visibility: public
    contract: >
      Typed construction error: EmptyPool (pool_size == 0) or SessionBuild
      (per-slot OnnxError source).
    proof:
      kind: compile-time
      target: src/fbank_onnx/mod.rs
      command: cargo check --features onnx --lib
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
    rule: FbankOnnxExtractor implements Embedder directly.
    proof:
      kind: compile-time
      target: src/fbank_onnx/mod.rs
      command: cargo check --features onnx --lib
verification:
  pre_change:
    - cargo test --lib fbank_onnx --features onnx
  full:
    - cargo test --lib fbank_onnx --features onnx
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

# src/fbank_onnx

Shared fbank + ONNX embedder engine (`FbankOnnxExtractor` implements `Embedder`).
