# TODO — src/der

## Current

- [x] Add Hoare triple pre/postconditions to `compute_der`.
- [x] Add property test: DER returns values in \[0, 1\].
- [x] Verify `compute_der_from_rttm` has sufficient test coverage.

## Phase 0 (Baseline Freeze) — done

- [x] `benches/der_ami.rs` already calls `polyvoice::der::compute_der_from_rttm` — no duplicate DER impl remains.
- [x] `polyvoice-bench` emits `polyvoice-bench-v0.6` JSON with git SHA, model hashes, per-file results, speaker-count diagnostics, and no-collar DER.
- [x] Full benchmark on AMI single file committed to `benchmarks/results/ami-test-single-20260516.json`.
- [x] 10-file VoxConverse subset benchmark committed to `benchmarks/results/voxconverse-test-10files-20260516.json`.
- [x] `tests/der_baseline.json` updated with reproducibility metadata (schema v2, git SHA, model versions).
- [x] `benchmarks/README.md` documents dataset layout, commands, and quality gates.

## Next

- [ ] Add dedicated `criterion` micro-benchmark for `compute_der` on synthetic data.
- [ ] Switch from greedy to Hungarian optimal mapping (only if benchmarking shows significant accuracy improvement).
- [ ] Support overlap-aware DER variants (Jaccard-style multi-speaker metrics).
