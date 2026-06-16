# src/attribution

## Purpose

Word→speaker attribution join: map raw ASR words onto diarization speaker turns,
producing `WordAlignment` per word. Pure-Rust, wasm-clean, opt-in (`attribution`
feature). No ASR/models/`ort`/IO — only interval arithmetic.

## Surfaces

- `attribute_words(&[Word], &[SpeakerTurn]) -> Vec<WordAlignment>`

## Dependencies

- `types` — `Word`, `SpeakerTurn`, `WordAlignment`, `TimeRange`, `SpeakerId`

## Invariants

- Output preserves input word order and length; every word (with turns present) gets a speaker.
- Max temporal overlap wins; no-overlap → nearest turn by gap; ties → smaller `SpeakerId`, then earlier turn.
- Straddling words get the dominant speaker with confidence scaled by coverage share.
- No model/`ort`/IO dependency; default build unchanged (~30 MB), wasm-clean.

## Verification

```bash
cargo test --features attribution --lib attribution
cargo build --target wasm32-unknown-unknown --features attribution
```

## Notes

- The ASR backend (`polyvoice-asr`) producing `Word`s is a separate opt-in crate.
