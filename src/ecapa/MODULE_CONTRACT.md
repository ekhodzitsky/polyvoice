---
schema_version: 1
kind: module_contract
module: src/ecapa
level: subsystem
layer: algorithm
purpose: >
  Owns the ECAPA-TDNN ONNX embedding extractor (legacy, superseded by
  embedder.rs CamPlusPlusExtractor). Kept for backward compatibility.
  Does NOT own feature extraction (features.rs) or model registry.
status: deprecated
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/ecapa/
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
  - name: FbankOnnxExtractor
    kind: struct
    visibility: public
    contract: >
      Legacy ONNX-backed extractor combining FBank + ECAPA-TDNN.
      Re-exported at crate root for backward compatibility.
    proof:
      kind: missing
      target: ""
      command: "Legacy surface; tests in embedder_test.rs cover equivalent behavior"
dependencies:
  internal:
    - module: embedding
      scope: trait
      reason: EmbeddingExtractor trait implementation.
    - module: features
      scope: utility
      reason: FbankExtractor, apply_cmvn.
    - module: types
      scope: data-shape
      reason: DiarizationConfig.
    - module: utils
      scope: utility
      reason: l2_normalize.
  external:
    - name: ort
      scope: ml-runtime
      reason: ONNX inference.
consumers:
  - path: src/lib.rs
    uses:
      - FbankOnnxExtractor
  - path: tests/embedder_test.rs
    uses:
      - FbankOnnxExtractor
invariants:
  - id: output-normalized
    rule: Extractor outputs L2-normalized embeddings.
    proof:
      kind: missing
      target: ""
      command: "Covered by embedder_test.rs for equivalent adapters"
verification:
  pre_change:
    - cargo check --all-features
  full:
    - cargo test --test embedder_test --features onnx
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Documentation updates.
    - Bug fixes only.
  forbidden_mutations:
    - Adding new features to this module (use embedder.rs instead).
    - Removing FbankOnnxExtractor without deprecation cycle.
  escalation:
    - Any removal of this module.
---

# src/ecapa

Legacy ECAPA-TDNN ONNX embedding extractor (deprecated; use embedder.rs).
