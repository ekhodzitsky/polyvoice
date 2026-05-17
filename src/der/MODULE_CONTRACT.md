---
schema_version: 1
kind: module_contract
module: src/der
level: subsystem
layer: evaluation
purpose: >
  Owns Diarization Error Rate (DER) computation.
  Computes frame-based DER with forgiveness collar and greedy optimal speaker
  mapping. Does NOT own pipeline diarization, model inference, or RTTM parsing
  (consumers in tests use rttm and pipeline modules to produce inputs).
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/der/
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
  - name: compute_der
    kind: function
    visibility: public
    contract: >
      Computes DER between reference and hypothesis speaker turns using 10ms
      frames, forgiveness collar, and greedy 1-to-1 speaker mapping. Returns
      DerResult with decomposed miss, false alarm, and confusion rates.
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
  - name: DerResult
    kind: struct
    visibility: public
    contract: >
      DER evaluation result with der, miss_rate, false_alarm_rate,
      confusion_rate, and total_speech. Display impl formats as human-readable
      percentages.
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
  - name: compute_der_from_rttm
    kind: function
    visibility: public
    contract: >
      Convenience wrapper that maps string speaker labels from RTTM-like tuples
      to numeric SpeakerTurns, then delegates to compute_der.
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: SpeakerTurn, TimeRange, SpeakerId are the input/output data shapes.
  external: []
consumers:
  - path: .
    uses:
      - compute_der
      - DerResult
      - compute_der_from_rttm
      - polyvoice_internal
invariants:
  - id: der-range
    rule: der field of DerResult is always in [0.0, 1.0].
    proof:
      kind: unit-test
      target: tests/property_der_test.rs::der_range_is_0_to_1
      command: cargo test --test property_der_test
  - id: identical-zero
    rule: Identical reference and hypothesis produce DER == 0.
    proof:
      kind: unit-test
      target: tests/property_der_test.rs::der_identical_ref_hyp_is_zero
      command: cargo test --test property_der_test
  - id: empty-ref-zero
    rule: Empty reference always produces DER == 0 regardless of hypothesis.
    proof:
      kind: unit-test
      target: src/der::mod::tests::empty_reference
      command: cargo test --lib der
  - id: component-sum
    rule: miss_rate + false_alarm_rate + confusion_rate == der (within f64 rounding).
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
  - id: collar-reduces-der
    rule: For boundary errors, collar > 0 produces DER <= collar == 0 DER.
    proof:
      kind: unit-test
      target: src/der::mod::tests::collar_reduces_error
      command: cargo test --lib der
verification:
  pre_change:
    - cargo test --lib der
    - cargo test --test property_der_test
  full:
    - cargo test --lib der
    - cargo test --test property_der_test
    - cargo test --test der_baseline_test
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Refactoring internal helper functions (build_collar_mask, build_speaker_frames, greedy_speaker_mapping).
    - Adding new unit tests or property tests.
    - Improving numerical stability of frame counting.
    - Adding documentation and invariant comments.
  forbidden_mutations:
    - Changing the compute_der signature without updating all consumers.
    - Removing DerResult fields (breaks Display and consumers).
    - Changing the 10ms frame resolution without updating collar semantics.
    - Replacing greedy mapping with a different algorithm without benchmarking against existing baselines.
  escalation:
    - Any change to compute_der or compute_der_from_rttm signatures.
    - Any change to DerResult fields or their semantic meaning.
    - Collar semantics changes (boundary handling, frame resolution).
    - Speaker mapping algorithm changes.
    - Changes that would require updating der_baseline.json or regression thresholds.
---

# src/der

Frame-based Diarization Error Rate computation with forgiveness collar and
greedy optimal speaker mapping. This is the single source of DER truth for
polyvoice.
