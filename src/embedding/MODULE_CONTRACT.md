---
schema_version: 1
kind: module_contract
module: src/embedding
level: subsystem
layer: algorithm
purpose: >
  Owns legacy embedding extraction trait and types (EmbeddingExtractor,
  DummyExtractor, EmbeddingError). Superseded by embedder.rs Embedder trait.
  Kept for backward compatibility until M6 cleanup.
status: deprecated
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/embedding/
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
  - name: EmbeddingExtractor
    kind: trait
    visibility: public
    contract: >
      Legacy embedding extraction trait. Use Embedder from embedder.rs for new code.
    proof:
      kind: smoke
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test --features cli
  - name: DummyExtractor
    kind: struct
    visibility: public
    contract: >
      Mock extractor for testing. Returns random or zero embeddings.
    proof:
      kind: smoke
      target: tests/chaos_test.rs
      command: cargo test --test chaos_test
  - name: EmbeddingError
    kind: enum
    visibility: public
    contract: >
      Legacy error type for embedding extraction.
    proof:
      kind: smoke
      target: tests/chaos_test.rs
      command: cargo test --test chaos_test
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: DiarizationConfig.
  external: []
consumers:
  - path: src/lib.rs
    uses:
      - EmbeddingExtractor
      - DummyExtractor
      - EmbeddingError
  - path: src/ecapa/mod.rs
    uses:
      - EmbeddingExtractor
      - EmbeddingError
  - path: src/onnx/mod.rs
    uses:
      - EmbeddingExtractor
      - EmbeddingError
invariants:
  - id: dummy-deterministic
    rule: DummyExtractor with fixed seed produces deterministic output.
    proof:
      kind: smoke
      target: tests/chaos_test.rs
      command: cargo test --test chaos_test
verification:
  pre_change:
    - cargo check --all-features
  full:
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Documentation updates.
    - Bug fixes only.
  forbidden_mutations:
    - Adding new features (use embedder.rs instead).
  escalation:
    - Any removal before M6 deprecation cycle.
---

# src/embedding

Legacy embedding extraction trait (deprecated; use embedder.rs).
