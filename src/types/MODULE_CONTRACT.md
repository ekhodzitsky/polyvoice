---
schema_version: 1
kind: module_contract
module: src/types
level: subsystem
layer: core
purpose: >
  Owns all core data types, configuration structs, and newtypes for polyvoice.
  Does NOT own algorithms, I/O, or model inference.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/types/
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
  - name: SpeakerId / SpeakerIdRemap
    kind: struct
    visibility: public
    contract: Opaque newtype for speaker identifiers + in-place remap utilities.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - name: DiarizationConfig / ClusterConfig / WindowConfig / SpeechFilterConfig
    kind: struct
    visibility: public
    contract: Top-level and sub-configuration for diarization pipeline parameters.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - name: Profile
    kind: enum
    visibility: public
    contract: Pre-defined model profiles (Mobile, Balanced) with bundled parameters.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - name: SpeakerTurn / Segment / TimeRange / WordAlignment / DiarizationResult
    kind: struct
    visibility: public
    contract: Time-annotated speaker segments, alignments, and pipeline output.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - name: SampleRate / Confidence / Seconds
    kind: struct
    visibility: public
    contract: Newtypes enforcing valid ranges (sample rate 8–192kHz, confidence 0–1).
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - name: remap_segments / remap_turns
    kind: function
    visibility: public
    contract: In-place speaker ID remapping utilities.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
dependencies:
  internal: []
  external: []
consumers:
  - path: src/cluster/mod.rs
    uses: [ClusterConfig, SpeakerId]
  - path: src/clusterer/mod.rs
    uses: [SpeakerId]
  - path: src/pipeline/mod.rs
    uses: [DiarizationConfig]
  - path: src/vad/mod.rs
    uses: [DiarizationConfig]
  - path: src/streaming/mod.rs
    uses: [DiarizationConfig, SpeakerTurn]
  - path: src/overlap/mod.rs
    uses: [Segment, SpeakerId]
  - path: src/rttm/mod.rs
    uses: [SpeakerId, SpeakerTurn, TimeRange]
  - path: src/utils/mod.rs
    uses: [Segment]
  - path: src/ecapa/mod.rs
    uses: [DiarizationConfig]
  - path: src/onnx/mod.rs
    uses: [DiarizationConfig]
  - path: src/segmentation/mod.rs
    uses: [Confidence, TimeRange]
  - path: src/resegmentation/mod.rs
    uses: [SpeakerId, SpeakerTurn, TimeRange]
  - path: src/models/mod.rs
    uses: [Profile]
invariants:
  - id: sample-rate-range
    rule: SampleRate constructor rejects values outside 8000..=192000.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - id: confidence-range
    rule: Confidence constructor rejects values outside 0.0..=1.0.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - id: time-range-valid
    rule: TimeRange.start <= TimeRange.end.
    proof:
      kind: unit-test
      target: src/types::mod::tests
      command: cargo test --lib types
  - id: property-sample-rate
    rule: SampleRate constructor accepts 8000..=192000 and rejects all others.
    proof:
      kind: unit-test
      target: tests/property_types_test.rs::sample_rate_invariant
      command: cargo test --test property_types_test
  - id: property-confidence
    rule: Confidence constructor accepts [0.0, 1.0] and rejects all others.
    proof:
      kind: unit-test
      target: tests/property_types_test.rs::confidence_invariant
      command: cargo test --test property_types_test
  - id: property-time-range
    rule: TimeRange duration equals end - start for valid ranges.
    proof:
      kind: unit-test
      target: tests/property_types_test.rs::time_range_duration_non_negative
      command: cargo test --test property_types_test
verification:
  pre_change:
    - cargo test --lib types
  full:
    - cargo test --lib types
    - cargo test --test property_types_test
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new configuration fields with defaults.
    - Adding new newtypes with validated constructors.
    - Documentation improvements.
  forbidden_mutations:
    - Removing or renaming public fields on widely-used structs without migration lease.
    - Widening newtype invariants.
  escalation:
    - Any change to Profile variants or their semantic meaning.
    - Any removal of public types or fields.
    - Changes to newtype invariant bounds.
---

# src/types

Core data types and configuration shapes for polyvoice.
