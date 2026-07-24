# src/attribution

## Purpose

Word→speaker attribution and the who-said-what cascade: map raw ASR words onto
diarization speaker turns, then assemble per-turn transcripts. Pure-Rust,
wasm-clean, opt-in (`attribution` feature). No models/`ort`/IO — only interval
arithmetic; ASR enters as `&dyn Asr` (the core trait), never a model pull.

## Surfaces

- `attribute_words(&[Word], &[SpeakerTurn]) -> Vec<WordAlignment>` — per-word
  speaker tagging (same order/length as input). Default config.
- `attribute_words_with_config(..., &AttributionConfig)` — same, with options.
- `interpolate_word_timestamps(&[Word]) -> (Vec<Word>, Vec<bool>)` — nearest-
  neighbor fill for missing/zero-duration times (never drops words).
- `fill_turn_text` / `fill_turn_text_with_config` — assemble `SpeakerTurn.text`
  from attributed words; anchor is start/mid/end (default mid).
- `attribute_and_fill` / `attribute_and_fill_with_config` — join + text-fill.
- `who_said_what` / `who_said_what_with_config` — one ASR pass, then join.
- `AttributionConfig { word_anchor, sentence_smoothing, smoothing_threshold,
  interpolate_timestamps }`, `WordAnchor { Start, Mid, End }`,
  `WhoSaidWhat { words, turns }`.

## Join algorithm

Tagging is **max temporal overlap** (nearest turn by gap when no overlap),
implemented as an **O(W+T) two-pointer sweep** over time-sorted words and turns.
Tie-break: smaller `SpeakerId`, then earlier original turn index. Behavior is
bit-identical to the historical per-word full scan (guarded by a proptest).

Optional post-steps (config):

1. **Timestamp interpolation** (default on) — missing/zero-duration word times
   get nearest-neighbor fill (`prev.end` / `next.start`, edge-clamped). Marked
   on `WordAlignment.interpolated`.
2. **Sentence smoothing** (default **off**) — if a speaker change lands inside
   a sentence (split on trailing `.?!`) and one speaker holds `> threshold`
   (default 0.5) of the sentence's words, relabel the sentence to that speaker.
3. **Word anchor** (default mid) — which point of the word must lie in a turn
   for `fill_turn_text` membership. Does **not** change max-overlap tagging.

## Cascade order & rationale

Diarize **first** (caller supplies `turns`), then run ASR as a **single pass**
over the full audio, then join. The cascade is **diarizer-agnostic**.

- **Per-segment ASR is NOT supported** — loses cross-boundary context and multiplies cost.
- **Overlap limitation:** overlapped speech → single **dominant** speaker only.
- **Smoothing default off** until measured on a real who-said-what cascade;
  opt-in avoids harming genuine mid-sentence speaker changes (interruptions).

## Dependencies

- `types` — `Word`, `SpeakerTurn`, `WordAlignment`, `TimeRange`, `SpeakerId`, `SampleRate`
- `asr` — `Asr` trait + `AsrError` (backend is a separate opt-in crate)

## Invariants

- Output length and order match input words; with turns present every word gets a speaker.
- Max temporal overlap wins; no-overlap → nearest by gap; ties → smaller id, earlier turn.
- Straddling words: dominant speaker, confidence scaled by coverage share.
- Interpolation never drops words (`|out| == |in|`).
- No model/`ort`/IO dependency; default build unchanged; wasm-clean.

## Verification

```bash
cargo test --features attribution --lib attribution
cargo build --target wasm32-unknown-unknown --features attribution
```
