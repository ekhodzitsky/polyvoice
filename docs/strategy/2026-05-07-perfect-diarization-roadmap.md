# Perfect Diarization Roadmap

Date: 2026-05-07

## Requirements Summary

polyvoice should become the Rust-native diarization library that can credibly beat or match pyannote-class systems while staying easier to deploy. The work should be driven by benchmark evidence, not subjective audio demos.

Primary requirements:

- Reproducible benchmark harness across datasets and competitors.
- Accuracy core with neural segmentation, overlap-aware decoding, PLDA scoring, and VBx-style resegmentation.
- Strong speaker-count estimation with explicit diagnostics.
- First-class overlap and word-level output.
- Offline and streaming pipelines with documented latency/accuracy tradeoffs.
- Rust-first deployment: CLI, crate API, Python wheel, C FFI, optional backend acceleration.

## Acceptance Criteria

- Every public DER table is generated from committed benchmark artifacts.
- Current baseline metrics are preserved and comparable after each major change.
- `polyvoice-bench` can run smoke, standard, and full benchmark suites.
- At least one public benchmark shows polyvoice matching a pyannote-community-class result.
- Overlap-aware output is represented in API, JSON, and RTTM-compatible exports.
- Speaker-count quality is reported separately from DER.
- The default pipeline remains usable without GPU or Python.
- All new public functions document preconditions and guarantees.
- No new dependency is added without a benchmark or implementation reason recorded in the relevant PR/commit.

## Phase 0: Baseline Freeze

Duration: 2-4 days

Goal: Stop guessing. Make current behavior measurable and reproducible.

Tasks:

- Add a `docs/strategy/baselines/` result location or `benchmarks/results/` if the repo prefers code-owned benchmark artifacts.
- Run current `polyvoice-bench` on available AMI/VoxConverse subsets.
- Capture model hashes, crate version, git SHA, host CPU, command line, and timing.
- Replace old duplicate DER logic in `benches/der_ami.rs` with `src/der.rs` or delete the obsolete bench if it no longer represents the real runner.
- Add a benchmark README that explains exact dataset layout.

Acceptance:

- A new contributor can reproduce the current README DER claims from commands.
- No benchmark code has a private DER implementation that diverges from `src/der.rs`.

Verification:

- `cargo test der`
- `cargo run --release --features cli --bin polyvoice-bench -- <small-dataset> --verbose true`
- `git diff --check`

## Phase 1: Competitor Runner Harness

Duration: 1-2 weeks

Goal: Compare against competitors under one evaluation policy.

Tasks:

- Add a runner interface for external systems:
  - `polyvoice`
  - `speakrs`
  - `pyannote`
  - `diarize`
  - optional `sherpa-onnx`
  - optional `parakeet-rs` Sortformer
- Normalize output to RTTM or internal `SpeakerTurn`/multi-speaker frames.
- Emit per-file and aggregate JSON.
- Report DER, JER, miss, false alarm, confusion, speaker-count exact, speaker-count +/-1, overlap F1 when reference overlap is available, and RTF.

Acceptance:

- Competitor comparisons can be run locally without editing code.
- Missing optional competitors are reported as skipped, not failed.
- README benchmark tables are generated from JSON.

Risk:

- Some competitor models require accounts or licenses. Mitigate by making runners optional and recording availability.

## Phase 2: Speaker Count and Clustering Hardening

Duration: 2-3 weeks

Goal: Improve the current architecture before adding heavier models.

Tasks:

- Benchmark existing AHC, Auto AHC, k-means, and spectral across datasets.
- Add speaker-count report buckets: 1, 2, 3-4, 5-7, 8+ speakers.
- Make `min_embeddings_per_speaker` behavior explicit and tested.
- Add count diagnostics to benchmark output.
- Add calibrated threshold search on development sets, but keep test sets sealed.
- Add score-normalized cosine or simple calibration if it improves dev and does not regress test.
- Gate temporal smoothing behind benchmark evidence.

Acceptance:

- Current pipeline has a reliable best-known config per dataset class.
- Speaker count has a clear failure profile.
- Any smoothing/refinement default is justified by result JSON.

Verification:

- `cargo test --all-targets`
- Benchmark matrix on available subsets.

## Phase 3: Result Schema V2 and Overlap Foundation

Duration: 1-2 weeks

Goal: Prepare the API before the neural pipeline lands.

Tasks:

- Add `DiarizationResultV2` with:
  - compatibility turns
  - overlap/multi-speaker turns
  - optional frame scores
  - confidences
  - speaker-count diagnostics
  - model/config metadata
- Add conversion from V2 to existing `DiarizationResult`.
- Update JSON output to represent multiple speakers per interval.
- Keep existing RTTM output compatible.
- Add tests for overlapping RTTM-style intervals.

Acceptance:

- Existing users are not broken.
- New overlap-aware systems have a place to write correct output.

Risk:

- Public API churn. Mitigate by adding V2 rather than replacing V1 immediately.

## Phase 4: Neural Segmentation Backend

Duration: 4-6 weeks

Goal: Add the missing pyannote-class structural component.

Tasks:

- Define `SegmentationModel` and `SegmentationOutput`.
- Keep Silero VAD as a lightweight segmentation implementation.
- Add ONNX segmentation backend after model license and tensor schema are verified.
- Implement overlap-add aggregation.
- Implement powerset decoding.
- Implement calibrated binarization with onset/offset thresholds.
- Add synthetic tests for overlap, silence, speaker-change, and short-turn edge cases.

Acceptance:

- Pipeline can infer overlap from audio model posteriors.
- Neural segmentation improves miss/FA balance against Silero-only segmentation on development data.
- Model metadata records license, source, and hash.

Verification:

- Unit tests for powerset decode and aggregation.
- Integration tests with a tiny fixture or mocked segmentation tensors.
- Benchmark dev-set result JSON.

## Phase 5: PLDA Scoring and VBx Resegmentation

Duration: 4-6 weeks

Goal: Reach the architecture class used by strongest competitors.

Tasks:

- Add `ScoringBackend` abstraction: cosine, calibrated cosine, PLDA.
- Implement PLDA model loading with metadata and dimensionality checks.
- Implement VBx/HMM-style resegmentation as an optional post-clustering step.
- Add cannot-link constraints from local segmentation output.
- Add benchmark gates before enabling defaults.

Acceptance:

- PLDA + VBx improves DER on at least two benchmark families.
- Regressions are visible by DER decomposition, not hidden in one aggregate.
- The old AHC path remains available.

Verification:

- Unit tests for PLDA dimensionality, scoring, and error cases.
- Synthetic HMM/resegmentation tests.
- Full benchmark subset.

## Phase 6: Model Zoo and Backend Acceleration

Duration: 3-5 weeks

Goal: Make model choice and acceleration a product feature.

Tasks:

- Add model registry with source URL, hash, license, sample rate, tensor schema, and benchmark tags.
- Add commands:
  - `polyvoice models list`
  - `polyvoice models info`
  - `polyvoice models download`
  - `polyvoice models verify`
- Evaluate WeSpeaker variants, ECAPA, TitaNet, WavLM/SSL-derived embeddings, and Sortformer if export is practical.
- Add optional ORT execution-provider features where they materially help:
  - CUDA
  - CoreML
  - OpenVINO
  - DirectML
  - WebGPU

Acceptance:

- Model loading errors are actionable.
- Benchmarks always include model identity and execution provider.
- CPU remains the default reliable path.

## Phase 7: Streaming V2

Duration: 4-6 weeks

Goal: Beat diart/parakeet-rs-style streaming ergonomics from Rust.

Tasks:

- Add rolling local segmentation buffer.
- Add latency modes: 500ms, 1s, 2s, 5s.
- Add provisional/final segment semantics.
- Add stateful speaker cache.
- Add cannot-link constraints for simultaneous local speakers.
- Add streaming benchmark mode with latency/DER tradeoff.

Acceptance:

- Streaming mode has documented DER vs latency curves.
- API users can tell whether a segment is final or provisional.
- Speaker IDs are stable enough for live transcript display.

## Phase 8: Word-Level Speaker Assignment

Duration: 1-2 weeks

Goal: Make diarization useful in transcription products.

Tasks:

- Add word assignment using overlap-aware posterior integration.
- Support ambiguous speaker labels when confidence is low or overlap is high.
- Add CLI merge command for word-timestamp JSON + diarization JSON.
- Add tests around speaker-change boundaries and overlap.

Acceptance:

- Word assignment no longer relies only on a midpoint heuristic.
- Downstream apps can inspect confidence and ambiguity.

## Phase 9: Productization and Launch Readiness

Duration: 2-4 weeks

Goal: Make the accuracy work easy to adopt.

Tasks:

- Update README from generated benchmark artifacts.
- Add "from pyannote", "from speakrs", and "from diarize" migration notes.
- Publish model cards and license notes.
- Ensure Python bindings expose V2 result fields.
- Ensure C FFI remains safe and versioned.
- Prepare small demo dataset and a one-command demo.
- Add issue templates for benchmark regressions and model export bugs.

Acceptance:

- New users can install, download models, run diarization, and understand accuracy tradeoffs in under five minutes.
- Benchmark claims are auditable.
- Release notes clearly separate accuracy, speed, and API changes.

## Immediate Issue Backlog

1. Unify all DER calculations on `src/der.rs`.
2. Add benchmark result JSON schema.
3. Add model hash/version capture to `polyvoice-bench`.
4. Add speaker-count metrics to benchmark output.
5. Add dataset manifest format.
6. Add `DiarizationResultV2` draft behind a non-breaking API.
7. Add optional competitor runner scripts.
8. Add segmentation trait with Silero adapter.
9. Add powerset decode unit tests using synthetic logits.
10. Add model registry metadata for current WeSpeaker and Silero files.

## Recommended Execution Lanes

Solo lane:

- Phase 0 and Phase 1 can be done by one owner because they touch metrics and scripts and need consistency.

Parallel lanes after Phase 1:

- Evaluation and benchmark artifacts.
- Result schema and output formats.
- Segmentation and powerset decoding.
- Scoring/PLDA/VBx research implementation.
- Streaming V2.
- Documentation and packaging.

Quality gates:

- Every algorithmic change must include before/after benchmark JSON.
- Any default change must improve at least one target dataset without unacceptable regression on another.
- Any new public API must have Rust docs and tests.
- Any unsafe/FFI change must run existing FFI memory tests.

## Stop Conditions

Stop or redesign if:

- The best available neural segmentation model cannot be distributed or referenced with acceptable license clarity.
- Benchmark results cannot reproduce published claims within a documented tolerance.
- A new dependency materially increases deployment complexity without a measurable accuracy or ergonomics win.
- The V2 result schema cannot represent overlap without breaking compatibility.

## First Concrete Milestone

Milestone name: `benchmark-truth`

Scope:

- No new diarization algorithm yet.
- Make current claims reproducible.
- Add competitor-aware benchmark structure.
- Produce baseline JSON for current polyvoice on at least one local subset.

Definition of done:

- `polyvoice-bench` emits result JSON.
- DER implementation is unified.
- README numbers can be regenerated from artifacts.
- A new document lists exact commands for current AMI/VoxConverse evaluation.
