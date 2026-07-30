# src/vad

## Purpose

Voice Activity Detection trait, energy-based VAD, VAD state machine, and
speech segmentation utilities.

## Surfaces

- `VoiceActivityDetector` trait
- `EnergyVad` — `new` (panicking convenience), `try_new` (rejects
  `frame_size == 0` with `VadError::InvalidChunkSize`)
- `VadConfig` — `frame_geometry(sample_rate, min_speech_secs)` is the single
  derivation of `ms_per_frame` / `min_silence_frames` / `min_speech_frames`
  (used by both `segment_speech` and the streaming pipeline); rejects
  `frame_size == 0`
- `VadFrameGeometry`
- `segment_speech` — frame durations derive from the detector's own
  `sample_rate()`; `DiarizationConfig` supplies only `min_speech_secs`
- `VadStateMachine` — events alternate `SpeechStart`/`SpeechEnd`;
  `meets_min_speech_duration` is the single point for the short-region
  suppression rule applied by `segment_speech` and streaming

## Dependencies

- `types` — DiarizationConfig

## Invariants

- segment_speech returns non-overlapping, monotonically ordered segments.

## Verification

```bash
cargo test --lib vad
cargo test --test chaos_test
```

## Notes

- SileroVad (ONNX-based) lives in silero_vad.rs and implements the same trait.
- `vad::hysteresis` (crate-internal) holds the scalar hysteresis core —
  `HysteresisGate` (onset/offset) and `RegionTracker` (min-off hangover with
  `Keep`/`Trim` closing policies + the min-on `keeps` filter) — shared by
  `VadStateMachine` and by powerset binarization in
  `segmentation::binarize`.
