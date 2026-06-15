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
  reference boundary from BOTH numerator and denominator, and there is no UEM
  support, so the DER is not bit-identical to `pyannote.metrics` — quote it with
  the collar used. `DerResult` exposes raw 10ms-frame counts
  (`total_ref_frames`, `missed_frames`, `false_alarm_frames`, `confusion_frames`)
  for duration-weighted micro-averaging across files.
- Speaker mapping uses the shared Kuhn-Munkres solver (`crate::hungarian`),
  giving the globally optimal label assignment (matching pyannote.metrics).
  Greedy 1-to-1 mapping was replaced with optimal Hungarian mapping because it over-counted
  confusion on cross-talk/fragmented files.
