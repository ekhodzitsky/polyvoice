# src/rttm

## Purpose

RTTM (Rich Transcription Time Marked) file parsing, grouping, and writing.

## Surfaces

- `RttmSegment`
- `RttmError`
- `parse_rttm`, `parse_rttm_file`
- `group_by_file`
- `to_speaker_turns`
- `write_rttm`

## Dependencies

- `types` — SpeakerId, SpeakerTurn, TimeRange

## Invariants

- write_rttm ∘ parse_rttm preserves segments (within formatting).

## Verification

```bash
cargo test --lib rttm
```

## Notes

- Standard RTTM format used by NIST and pyannote.
