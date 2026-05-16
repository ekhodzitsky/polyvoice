# TODO — src/der

## Current

- [ ] Add Hoare triple pre/postconditions to `compute_der` (stub comments exist).
- [ ] Add property test: DER is symmetric under optimal speaker mapping.
- [ ] Verify `compute_der_from_rttm` has sufficient test coverage.

## Next

- [x] Replace duplicate DER implementation in `benches/der_ami.rs` with calls to
      `polyvoice::der::compute_der_from_rttm`.
- [ ] Add benchmark for `compute_der` itself (micro-benchmark on synthetic data).

## Known Gaps

- `benches/der_ami.rs` maintains its own simplified 100ms-resolution DER impl.
  This diverges from `src/der` and is a maintenance risk. Tracked in
  `docs/strategy/2026-05-07-perfect-diarization-roadmap.md`.
- No dedicated `criterion` benchmark for DER computation in isolation.

## Deferred

- [ ] Switch from greedy to Hungarian optimal mapping (only if benchmarking
      shows significant accuracy improvement).
- [ ] Support overlap-aware DER variants (Jaccard-style multi-speaker metrics).
