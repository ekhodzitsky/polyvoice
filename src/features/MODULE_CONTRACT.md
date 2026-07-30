---
schema_version: 1
kind: module_contract
module: src/features
level: subsystem
layer: algorithm
purpose: >
  Owns filterbank (FBank) feature extraction, configuration, and CMVN
  normalization. Does NOT own embedding extraction or windowing.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/features/
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
  - name: FbankConfig
    kind: struct
    visibility: public
    contract: >
      Configuration for filterbank extraction (sample rate, n_mels, etc.).
      `validate()` reports field violations as FbankError::InvalidConfig.
    proof:
      kind: unit-test
      target: src/features::mod::tests
      command: cargo test --lib features
  - name: FbankExtractor
    kind: struct
    visibility: public
    contract: >
      Extracts mel-filterbank features from audio samples. `try_new`
      validates the config (FbankError::InvalidConfig); `new` is a panicking
      convenience over it.
    proof:
      kind: unit-test
      target: src/features::mod::tests
      command: cargo test --lib features
  - name: apply_cmvn
    kind: function
    visibility: public
    contract: >
      Applies Cepstral Mean and Variance Normalization to feature frames.
    proof:
      kind: unit-test
      target: src/features::mod::tests
      command: cargo test --lib features
dependencies:
  internal: []
  external: []
consumers:
  - path: .
    uses:
      - FbankConfig
      - FbankExtractor
      - apply_cmvn
      - polyvoice_internal
invariants:
  - id: fbank-non-negative
    rule: FbankExtractor outputs non-negative mel energies (before CMVN).
    proof:
      kind: unit-test
      target: src/features::mod::tests
      command: cargo test --lib features
verification:
  pre_change:
    - cargo test --lib features
  full:
    - cargo test --lib features
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new feature extraction variants.
    - Tuning default FbankConfig values.
  forbidden_mutations:
    - Changing FbankExtractor output dimension without consumer updates.
  escalation:
    - Changes to public FbankConfig fields that affect output shape.
---

# src/features

Filterbank feature extraction and CMVN normalization.
