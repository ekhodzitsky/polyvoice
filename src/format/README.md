# src/format

## Purpose

Subtitle / plain-text projections of a diarization result (SRT, WebVTT, TXT),
projected from the canonical `DiarizationResult` turns. Pure-Rust, wasm-clean.

## Surfaces

- `write_srt`, `write_vtt`, `write_txt` — write speaker turns to an `impl Write`

## Dependencies

- `types` — `SpeakerTurn`, `SpeakerId`, `TimeRange`

## Invariants

- One turn → one block (lossless); timecodes rounded to the millisecond.
- SRT uses a comma ms separator (`HH:MM:SS,mmm`); WebVTT uses a dot (`HH:MM:SS.mmm`).
- No filesystem I/O inside the writers (wasm-clean).

## Verification

```bash
cargo test --lib format
cargo build --target wasm32-unknown-unknown --no-default-features --lib
```

## Notes

- RTTM lives in `src/rttm`; JSON is serde on `DiarizationResult`.
- WebVTT `<v SPEAKER_NN>` voice tags and CLI wiring are separate, later tasks.
