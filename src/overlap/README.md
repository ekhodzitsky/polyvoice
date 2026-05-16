# src/overlap

## Purpose

Overlap detection between speaker segments.

## Surfaces

- `detect_overlaps(segments) -> Vec<OverlapRegion>`
- `OverlapRegion`

## Dependencies

- `types` — Segment, SpeakerId, TimeRange

## Invariants

- OverlapRegion time ranges have start < end and contain at least 2 speakers.

## Verification

```bash
cargo test --lib overlap
```

## Notes

- Used by resegmentation pipeline stage to identify multi-speaker regions.
