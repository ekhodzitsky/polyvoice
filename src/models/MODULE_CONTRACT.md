---
schema_version: 1
kind: module_contract
module: src/models
level: subsystem
layer: infrastructure
purpose: >
  Owns model registry, manifest parsing, HTTP downloads, SHA-256 verification,
  and minisig signature verification for ONNX model bundles. Does NOT own
  model inference (that lives in onnx, embedder, segmentation).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/models/
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
  - name: ModelRegistry
    kind: struct
    visibility: public
    contract: >
      Downloads, caches, and verifies ONNX model bundles by profile.
    proof:
      kind: unit-test
      target: src/models::mod::tests
      command: cargo test --lib models --features download
  - name: ProfileModels
    kind: struct
    visibility: public
    contract: >
      Paths to cached embedder, segmenter, and clusterer models for a profile.
    proof:
      kind: unit-test
      target: src/models::mod::tests
      command: cargo test --lib models --features download
  - name: RegistryError
    kind: enum
    visibility: public
    contract: >
      Errors for registry operations (download, verify, manifest parse).
    proof:
      kind: unit-test
      target: src/models::mod::tests
      command: cargo test --lib models --features download
  - name: Manifest
    kind: struct
    visibility: public
    contract: >
      Typed TOML manifest describing available model bundles.
    proof:
      kind: unit-test
      target: src/models::manifest::tests
      command: cargo test --lib models --features download
  - name: DEFAULT_MANIFEST_TOML
    kind: constant
    visibility: public
    contract: >
      Embedded default manifest string.
    proof:
      kind: unit-test
      target: src/models::mod::tests
      command: cargo test --lib models --features download
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Profile enum for model profile selection.
  external:
    - name: ureq
      scope: network
      reason: HTTP download of model bundles.
    - name: sha2
      scope: crypto
      reason: SHA-256 verification of downloaded files.
    - name: minisign-verify
      scope: crypto
      reason: Signature verification of model bundles.
    - name: dirs
      scope: filesystem
      reason: Cross-platform cache directory resolution.
    - name: toml
      scope: parsing
      reason: Manifest TOML deserialization.
consumers:
  - path: src/ffi/mod.rs
    uses:
      - ModelRegistry
  - path: src/pipeline/mod.rs
    uses:
      - ModelRegistry
      - ProfileModels
  - path: tests/der_regression_test.rs
    uses:
      - ModelRegistry
  - path: tests/der_ami_baseline_test.rs
    uses:
      - ModelRegistry
  - path: tests/m5_manifest_smoke_test.rs
    uses:
      - ModelRegistry
      - Manifest
invariants:
  - id: verify-sha256
    rule: Downloaded files must match manifest SHA-256 before use.
    proof:
      kind: unit-test
      target: src/models::verify::tests
      command: cargo test --lib models --features download
  - id: verify-signature
    rule: Downloaded files must pass minisig verification before use.
    proof:
      kind: unit-test
      target: src/models::verify::tests
      command: cargo test --lib models --features download
verification:
  pre_change:
    - cargo test --lib models --features download
  full:
    - cargo test --lib models --features download
    - cargo test --test m5_manifest_smoke_test --features download
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new model entries to manifest.
    - Improving download retry logic.
    - Adding new verification backends.
  forbidden_mutations:
    - Removing SHA-256 or signature verification.
    - Changing ModelRegistry::default() behavior without updating tests.
  escalation:
    - Changes to manifest schema version.
    - Changes to verification requirements (SHA-256, signatures).
    - Adding new external download dependencies.
---

# src/models

Model registry, manifest, download, and verification.
