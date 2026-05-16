# src/vad

## Purpose

Voice Activity Detection trait, energy-based VAD, VAD state machine, and
speech segmentation utilities.

## Surfaces

- `VoiceActivityDetector` trait
- `EnergyVad`
- `VadConfig`
- `segment_speech`
- `VadStateMachine`

## Dependencies

- `types` — DiarizationConfig

## Invariants

- segment_speech returns non-overlapping, monotonically ordered segments.

## Verification

```bash
cargo test --lib vad
cargo test --test vad_test
```

## Notes

- SileroVad (ONNX-based) lives in silero_vad.rs and implements the same trait.
