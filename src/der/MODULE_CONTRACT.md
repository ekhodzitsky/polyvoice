---
schema_version: 1
kind: module_contract
module: src/der
level: subsystem
layer: evaluation
purpose: >
  Owns Diarization Error Rate (DER) computation.
  Computes frame-based DER with forgiveness collar and optimal (Hungarian)
  speaker mapping. Does NOT own pipeline diarization, model inference, or RTTM parsing
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
      frames, forgiveness collar, and optimal Hungarian (Kuhn-Munkres) 1-to-1 speaker mapping. Returns
      DerResult with decomposed miss, false alarm, and confusion rates.
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
  - name: DerResult
    kind: struct
    visibility: public
    contract: >
      DER evaluation result: ratios (der, miss_rate, false_alarm_rate,
      confusion_rate, total_speech) plus raw 10ms-frame counts (total_ref_frames,
      missed_frames, false_alarm_frames, confusion_frames) for duration-weighted
      micro-averaging across files. Approximate DER — the boundary collar is
      excluded from BOTH numerator and denominator and there is no UEM support,
      so it is not bit-identical to pyannote.metrics. Display impl formats the
      ratios as human-readable percentages.
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
  - name: compute_der_single_speaker_regions
    kind: function
    visibility: public
    contract: >
      Overlap-excluded DER: same frame machinery, collar and optimal Hungarian
      mapping as compute_der, but additionally ignores reference frames with >= 2
      concurrent speakers (from both the mapping and the counts). It is the numeric
      long-form quality floor that discriminates healthy vs collapsed diarization on
      high-overlap audio, where total DER cannot. MUST NOT be conflated with the
      headline (overlap-inclusive) DER.
    proof:
      kind: unit-test
      target: src/der::mod::tests
      command: cargo test --lib der
  - name: compute_der_decomposition
    kind: function
    visibility: public
    contract: >
      Overlap-aware DER decomposition (returns DerDecomposition): headline DER plus
      single-speaker-region DER, overlap-region DER (>= 2 ref speakers), and
      per-speaker recall (Vec<SpeakerRecall>). Reuses compute_der's frame machinery,
      collar and Hungarian mapping. Intended for bench artifacts and the AMI gate so
      accuracy targets are interpretable; the headline path stays on compute_der.
    proof:
      kind: unit-test
      target: src/der::mod::tests::decomposition_splits_overlap_and_recall
      command: cargo test --lib der
  - name: compute_der_with_uem
    kind: function
    visibility: public
    contract: >
      DER restricted to UEM scored regions: same machinery as compute_der but frames
      whose center is outside every scored TimeRange are excluded from BOTH the
      mapping and the counts (on top of the collar). Empty scope scores nothing;
      no UEM == compute_der (byte-identical). Used by the DER harness.
    proof:
      kind: unit-test
      target: src/der::mod::tests::uem_ignores_error_outside_scope
      command: cargo test --lib der
  - name: parse_uem
    kind: function
    visibility: public
    contract: >
      Parse a .uem file body into per-file scored regions (HashMap<String,
      Vec<TimeRange>>). Skips comments/blank/malformed lines. Pure-Rust, wasm-clean
      (callers read the file). Feeds compute_der_with_uem.
    proof:
      kind: unit-test
      target: src/der::mod::tests::parse_uem_reads_regions_and_skips_junk
      command: cargo test --lib der
  - name: DerDecomposition
    kind: struct
    visibility: public
    contract: >
      Bundles total / single_speaker / overlap DerResult plus per_speaker_recall
      (Vec<SpeakerRecall>, sorted by reference speaker id). Not Copy (owns a Vec).
    proof:
      kind: unit-test
      target: src/der::mod::tests::decomposition_splits_overlap_and_recall
      command: cargo test --lib der
  - name: SpeakerRecall
    kind: struct
    visibility: public
    contract: >
      Per-reference-speaker recall: {speaker, ref_frames, recalled_frames, recall}
      where recall = recalled_frames / ref_frames in [0, 1].
    proof:
      kind: unit-test
      target: src/der::mod::tests::decomposition_splits_overlap_and_recall
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
  - id: uem-no-op-when-absent
    rule: compute_der_with_uem over a scope covering the whole file equals compute_der; out-of-scope frames drop from both mapping and counts.
    proof:
      kind: unit-test
      target: src/der::mod::tests::uem_full_scope_matches_no_uem
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
    - Refactoring internal helper functions (build_collar_mask, build_speaker_frames, optimal_speaker_mapping).
    - Adding new unit tests or property tests.
    - Improving numerical stability of frame counting.
    - Adding documentation and invariant comments.
  forbidden_mutations:
    - Changing the compute_der signature without updating all consumers.
    - Removing DerResult fields (breaks Display and consumers).
    - Changing the 10ms frame resolution without updating collar semantics.
    - Replacing the Hungarian mapping with a different algorithm without benchmarking against existing baselines.
  escalation:
    - Any change to compute_der or compute_der_from_rttm signatures.
    - Any change to DerResult fields or their semantic meaning.
    - Collar semantics changes (boundary handling, frame resolution).
    - Speaker mapping algorithm changes.
    - Changes that would require updating der_baseline.json or regression thresholds.
---

# src/der

Frame-based Diarization Error Rate computation with forgiveness collar, optimal
(Hungarian) speaker mapping, overlap-aware decomposition, and UEM scoping. This is
the single source of DER truth for polyvoice.

The reproducible DER harness (`scripts/run-der-sweep.sh` + `polyvoice-bench`)
is the only sanctioned producer of DER numbers: it reports both no-collar and
0.25 s-collar on the exact shipped FP32 artifact (sha256-gated) across
VoxConverse-dev/test + AMI, honoring UEM via `compute_der_with_uem`/`parse_uem`.
The same harness calibrates the fixed AHC threshold and the VBx
Fa/Fb/Ploop parameters — never tune those without re-running it on a
dev split.
