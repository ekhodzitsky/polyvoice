---
schema_version: 1
kind: module_contract
module: src/models
level: subsystem
layer: infrastructure
purpose: >
  Owns model registry, manifest parsing (schema v1+v2), HTTP downloads,
  SHA-256/minisig verification, adapter selection by config string
  (AdapterRegistry), and self-describing model metadata loading. Does NOT own
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
      Typed TOML manifest describing available model bundles (schema v1 + v2).
    proof:
      kind: unit-test
      target: src/models::manifest::tests
      command: cargo test --lib models --features download
  - name: AdapterRegistry
    kind: struct
    visibility: public
    contract: >
      Selects segmentation/embedder/clusterer/scoring/VAD adapters by config
      string; public register API; unknown type returns AdapterError.
    proof:
      kind: unit-test
      target: src/models::adapter::tests
      command: cargo test --lib models --features download
  - name: ModelConfigMeta
    kind: struct
    visibility: public
    contract: >
      Self-describing model config from ONNX metadata_props with manifest/default fallback.
    proof:
      kind: unit-test
      target: src/models::metadata::tests
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
  - path: .
    uses:
      - ModelRegistry
      - ProfileModels
      - RegistryError
      - Manifest
      - DEFAULT_MANIFEST_TOML
      - ureq
      - sha2
      - minisign-verify
      - dirs
      - toml
      - polyvoice_internal
invariants:
  - id: verify-sha256
    rule: Downloaded files must match manifest SHA-256 before use.
    proof:
      kind: unit-test
      target: src/models::verify::tests
      command: cargo test --lib models --features download
  - id: verify-signature
    rule: >
      Downloaded files must pass minisig verification before use. In release
      builds (cfg!(not(debug_assertions))) every PROFILE-RESOLVED model must
      additionally carry a manifest signature — a missing signature fails fast
      with RegistryError::UnsignedModel before any network access, so a
      tampered/forked manifest cannot silently downgrade authenticity to a
      self-consistent hash. Ad-hoc single-model ensure() stays lenient for
      dev/test. (Verification-requirement change recorded per escalation
      policy; strengthens, never weakens, verification.)
    proof:
      kind: unit-test
      target: src/models::mod::tests::strict_profile_resolution_rejects_unsigned_model
      command: cargo test --lib models --features download
  - id: download-https-only
    rule: >
      Model downloads are fetched only over https:// — a non-https URL is
      rejected with DownloadError::InsecureScheme before any network access.
      Cache hits (which transmit nothing) are exempt.
    proof:
      kind: unit-test
      target: src/models::download::tests::rejects_non_https_url
      command: cargo test --lib models --features download
  - id: download-size-capped
    rule: >
      A streamed download exceeding its byte cap aborts, deletes the .partial,
      and returns DownloadError::TooLarge (default ceiling 1 GiB, well above any
      shipped model).
    proof:
      kind: unit-test
      target: src/models::download::tests::aborts_when_stream_exceeds_cap
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
