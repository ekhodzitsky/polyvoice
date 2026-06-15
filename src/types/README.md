# src/types

## Purpose

Core data types, configuration structs, and newtypes for polyvoice. This is
the type foundation that all other modules build on.

## Surfaces

- `SpeakerId`, `SpeakerIdRemap`
- `DiarizationConfig`, `ClusterConfig`, `WindowConfig`, `SpeechFilterConfig`
- `Profile`
- `SampleRate`, `Confidence`, `Seconds`
- `TimeRange`, `Segment`, `SpeakerTurn`, `WordAlignment`
- `DiarizationResult` (canonical v1: + `schema_version`, `audio`, `provenance`, `speakers[]`), `AudioMeta`, `Provenance`, `SpeakerSummary`
- `remap_segments`, `remap_turns`

## Dependencies

None (self-contained).

## Invariants

- SampleRate ∈ [8000, 192000].
- Confidence ∈ [0.0, 1.0].
- TimeRange.start ≤ TimeRange.end.

## Verification

```bash
cargo test --lib types
```

## Notes

- Newtypes use validated constructors to make invalid states unrepresentable.
