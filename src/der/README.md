# src/der

## Purpose

Computes Diarization Error Rate (DER) between reference and hypothesis speaker
annotations. This is the single source of DER truth for the polyvoice project.

Uses frame-based evaluation at 10ms resolution with a configurable forgiveness
collar around reference boundaries. Speaker IDs are mapped optimally via
Hungarian (Kuhn-Munkres) 1-to-1 matching on co-occurrence counts.

## Surfaces

- `compute_der(reference, hypothesis, collar) -> DerResult`
- `DerResult` — struct with `der`, `miss_rate`, `false_alarm_rate`, `confusion_rate`, `total_speech`
- `compute_der_from_rttm(reference, hypothesis, collar) -> DerResult` — convenience wrapper for string-labeled references

## Dependencies

- `types` — `SpeakerTurn`, `TimeRange`, `SpeakerId`

## Invariants

- DER is always in [0.0, 1.0]
- Identical ref/hyp → DER == 0
- Empty reference → DER == 0
- miss + false_alarm + confusion == der (within f64 rounding)
- Collar reduces or preserves DER for boundary errors

## Verification

```bash
# Fast pre-change check
cargo test --lib der
cargo test --test property_der_test

# Full verification
cargo test --lib der
cargo test --test property_der_test
cargo test --test der_baseline_test
cargo clippy --all-targets --all-features -- -D warnings
```

## Notes

- The `benches/der_ami.rs` benchmark currently duplicates a simplified DER
  implementation instead of using this module. See TODO.md.
- Standard collar value is 0.25s (per NIST evaluation protocol).
- **Approximate DER:** the boundary collar excludes frames within `collar` of any
  reference boundary from BOTH numerator and denominator, so the DER is not
  bit-identical to `pyannote.metrics` — quote it with the collar used. UEM
  scoping is available via `compute_der_with_uem` / `parse_uem`. `DerResult`
  exposes raw 10ms-frame counts (`total_ref_frames`, `missed_frames`,
  `false_alarm_frames`, `confusion_frames`) for duration-weighted
  micro-averaging across files.

## Regression baselines and gates

`tests/der_baseline.json` is the committed source of truth for expected DER.
The regression tests (`tests/der_regression_test.rs` — legacy pipeline;
`tests/cli_der_regression_test.rs` — pipeline v2 via the CLI) are `#[ignore]`d
(they need real audio + the ONNX cache) and run through
`scripts/release-check.sh`, which `release.yml` executes as the publish gate
with `POLYVOICE_REQUIRE_DATA=1` — under that variable, missing data is a hard
failure, so a cache miss can never green-light a release. (An earlier plan
called this switch `POLYVOICE_DER_EVAL`; `POLYVOICE_REQUIRE_DATA=1` +
`--run-ignored only` is the shipped equivalent.)

Two metrics are gated per dataset, from one pipeline run scored at two collars:

- `der_collar_0_25` — the historical collar-0.25 gate (macro on multi-file sets).
- `der_no_collar` — the headline like-for-like metric (micro / frame-weighted on
  multi-file sets, the convention pyannote/speakrs headline numbers use). A JSON
  `null` keeps this gate inactive for that dataset and the test prints the
  measured value to record.

On high-overlap audio (AMI) total DER is miss-bound at any collar, so the v2 AMI
test gates on speaker count, confusion, and the overlap-excluded floor instead.

To refresh a baseline after a legitimate accuracy change: run the test with
`--run-ignored only --no-capture`, take the printed values, update the JSON
field plus its `_status`/`_filled_by` provenance — never widen `tolerance` to
make a regression pass.
- Speaker mapping uses the shared Kuhn-Munkres solver (`crate::hungarian`),
  giving the globally optimal label assignment (matching pyannote.metrics).
  Greedy 1-to-1 mapping was replaced with optimal Hungarian mapping because it over-counted
  confusion on cross-talk/fragmented files.
