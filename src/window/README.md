# src/window

## Purpose

Audio windowing utilities: fixed-size overlapping window iteration and
ring buffer management.

## Surfaces

- `WindowIter` — `new` (panicking convenience), `try_new` (fallible)
- `WindowBuffer` — `new` (panicking convenience), `try_new` (fallible)
- `WindowError` — geometry violations reported by `try_new`

## Dependencies

None (self-contained).

## Invariants

- WindowIter yields windows of exactly the configured size.
- Geometry is validated at construction: `win > 0`, `hop > 0`, `hop <= win`.
  `try_new` returns `WindowError`; `new` panics (documented convenience).

## Verification

```bash
cargo test --lib window
```

## Notes

- WindowBuffer is used by the streaming pipeline for ring-buffer management.
