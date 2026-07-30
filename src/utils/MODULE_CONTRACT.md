---
schema_version: 1
kind: module_contract
module: src/utils
level: subsystem
layer: utility
purpose: >
  Owns shared mathematical and segment utilities: cosine similarity,
  L2 normalization, mean vector, pairwise similarity matrices, mean
  centroids, segment merging. Does NOT own any algorithm specific to a
  single module.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/utils/
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
  - name: cosine_similarity
    kind: function
    visibility: public
    contract: >
      Computes cosine similarity between two f32 slices. Returns value in [-1, 1].
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - name: l2_normalize
    kind: function
    visibility: public
    contract: >
      In-place L2 normalization of a vector.
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - name: mean_vector
    kind: function
    visibility: public
    contract: >
      Computes element-wise mean of a vector of vectors.
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - name: pairwise_cosine_similarity_matrix
    kind: function
    visibility: crate
    contract: >
      Full symmetric pairwise cosine-similarity matrix as a flat row-major
      n*n buffer (f32, diagonal exactly 1.0). Single construction path shared
      by ahc, kmeans, and spectral; consumers chunk or cast as needed.
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - name: normalized_mean_centroids
    kind: function
    visibility: crate
    contract: >
      Per-slot L2-normalized mean centroid over selected members
      (embedding index + label pairs). Empty slots stay zero vectors.
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - name: merge_segments
    kind: function
    visibility: public
    contract: >
      Merges adjacent segments with same speaker if gap <= max_gap_secs. Merged
      confidence is the arithmetic mean of present (Some) confidences across the
      run (order-independent; None values are ignored).
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Segment type.
  external: []
consumers:
  - path: .
    uses:
      - cosine_similarity
      - l2_normalize
      - mean_vector
      - merge_segments
      - polyvoice_internal
invariants:
  - id: cosine-range
    rule: cosine_similarity returns value in [-1.0, 1.0].
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - id: l2-unit
    rule: After l2_normalize, vector norm is 1.0 (or 0.0 for zero vector).
    proof:
      kind: unit-test
      target: src/utils::mod::tests
      command: cargo test --lib utils
  - id: property-cosine-range
    rule: cosine_similarity of any two f32 vectors is in [-1.0, 1.0].
    proof:
      kind: unit-test
      target: tests/property_utils_test.rs::cosine_similarity_range
      command: cargo test --test property_utils_test
  - id: property-cosine-identical
    rule: cosine_similarity of identical non-zero vectors is 1.0.
    proof:
      kind: unit-test
      target: tests/property_utils_test.rs::cosine_similarity_identical_is_one
      command: cargo test --test property_utils_test
  - id: property-l2-unit
    rule: l2_normalize of any non-zero vector produces a unit vector.
    proof:
      kind: unit-test
      target: tests/property_utils_test.rs::l2_normalize_produces_unit_vector
      command: cargo test --test property_utils_test
verification:
  pre_change:
    - cargo test --lib utils
  full:
    - cargo test --lib utils
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new utility functions.
    - Performance optimizations.
  forbidden_mutations:
    - Changing existing function signatures without migration lease.
  escalation:
    - Changes to merge_segments semantics.
---

# src/utils

Shared mathematical and segment utilities.
