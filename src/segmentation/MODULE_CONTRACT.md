---
schema_version: 1
kind: module_contract
module: src/segmentation
level: subsystem
layer: algorithm
purpose: >
  Owns speaker segmentation algorithms: powerset segmentation, frame
  decoding, aggregation with Hungarian assignment, and windowed segmenter
  trait. Does NOT own embedding extraction, clustering, or VAD.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/segmentation/
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
  - name: Segmenter
    kind: trait
    visibility: public
    contract: >
      Core segmentation trait: process audio window → speaker labels.
    proof:
      kind: unit-test
      target: src/segmentation::mod::tests
      command: cargo test --lib segmentation
  - name: PowersetSegmenter
    kind: struct
    visibility: public
    contract: >
      ONNX-backed powerset speaker segmenter (sherpa-onnx-pyannote).
      `segment()` validates PowersetConfig window geometry first and reports
      violations as SegmentationError::InvalidGeometry instead of panicking.
    proof:
      kind: unit-test
      target: src/segmentation::mod::tests
      command: cargo test --lib segmentation
  - name: PowersetDecoder
    kind: struct
    visibility: public
    contract: >
      Decodes powerset class logits → FrameLabel: argmax class (speaker_set,
      is_overlap), max-softmax confidence, and the full 7-class softmax vector.
      The class table's inverse lives on PowersetClass (index / from_speakers,
      crate-visible), shared with the calibrated binarization path.
    proof:
      kind: unit-test
      target: src/segmentation::decoder::tests
      command: cargo test --lib segmentation
  - name: Aggregator
    kind: struct
    visibility: public
    contract: >
      Aggregates frame-level predictions into segments using Hungarian
      assignment for speaker identity tracking across windows.
    proof:
      kind: unit-test
      target: src/segmentation::aggregator::tests
      command: cargo test --lib segmentation
  - name: FrameLabel
    kind: struct
    visibility: public
    contract: >
      Per-frame speaker set and overlap flag (argmax class), its max-softmax
      confidence, and the full 7-class softmax vector (`probs`) so the
      aggregator reuses the decoded probabilities instead of recomputing them.
    proof:
      kind: unit-test
      target: src/segmentation::mod::tests
      command: cargo test --lib segmentation
  - name: MIN_AUDIO_SAMPLES
    kind: constant
    visibility: public
    contract: >
      Minimum audio samples required for segmentation (1600 = 0.1s at 16kHz).
    proof:
      kind: unit-test
      target: src/segmentation::mod::tests
      command: cargo test --lib segmentation
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Confidence, TimeRange for segment timestamps.
    - module: vad
      scope: utility
      reason: >
        binarize.rs smooths speaker activity on the crate-internal
        vad::hysteresis core (HysteresisGate + RegionTracker, Trim policy).
    - module: onnx
      scope: inference
      reason: >
        PowersetSegmenter uses InferenceRuntime / OrtSession via
        build_session_with_ep; does not import ort::.
  external: []
consumers:
  - path: .
    uses:
      - Segmenter
      - PowersetSegmenter
      - PowersetDecoder
      - Aggregator
      - FrameLabel
      - MIN_AUDIO_SAMPLES
      - polyvoice_internal
invariants:
  - id: decoder-deterministic
    rule: PowersetDecoder with identical logits always produces identical output.
    proof:
      kind: unit-test
      target: src/segmentation::decoder::tests
      command: cargo test --lib segmentation
  - id: hungarian-optimal
    rule: Aggregator Hungarian assignment minimizes total cost for speaker
      tracking across windows.
    proof:
      kind: unit-test
      target: src/segmentation::aggregator::tests
      command: cargo test --lib segmentation
  - id: frame-time-nearest-center
    rule: >
      The aggregator uses ONE frame-time convention. frame_index_at maps time->frame
      via floor((t-start)/stride), which is the nearest-CENTER frame: it equals
      round((t-start)/stride - 0.5) for every in-span t after the [0, num_frames-1]
      clamp. The run-length encoder places frame f by its center start+(f+0.5)*stride,
      so the permutation sampler and the applier agree (no half-stride off-by-one).
    proof:
      kind: unit-test
      target: src/segmentation::aggregator::tests::frame_index_floor_equals_nearest_center
      command: cargo test --lib segmentation
verification:
  pre_change:
    - cargo test --lib segmentation
  full:
    - cargo test --lib segmentation
    - cargo test --test chaos_test --features onnx,download
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Refactoring decoder internals.
    - Adding aggregation strategies.
    - Performance optimizations in frame processing.
  forbidden_mutations:
    - Changing Segmenter trait without updating all implementors.
    - Changing MIN_AUDIO_SAMPLES without checking downstream consumers.
  escalation:
    - Changes to Segmenter trait signature.
    - Changes to powerset class semantics.
---

# src/segmentation

Speaker segmentation algorithms: powerset decoder, aggregator, Hungarian
assignment, and segmenter trait.
