# src/window

## Purpose

Audio windowing utilities: fixed-size overlapping window iteration and
ring buffer management.

## Surfaces

- `WindowIter`
- `WindowBuffer`

## Dependencies

None (self-contained).

## Invariants

- WindowIter yields windows of exactly the configured size.

## Verification

```bash
cargo test --lib window
```

## Notes

- WindowBuffer is used by the streaming pipeline for ring-buffer management.
