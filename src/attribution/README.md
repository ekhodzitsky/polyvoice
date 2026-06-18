# src/attribution

## Purpose

Word→speaker attribution and the who-said-what cascade: map raw ASR words onto
diarization speaker turns, then assemble per-turn transcripts. Pure-Rust,
wasm-clean, opt-in (`attribution` feature). No models/`ort`/IO — only interval
arithmetic; ASR enters as `&dyn Asr` (the core trait), never a model pull.

## Surfaces

- `attribute_words(&[Word], &[SpeakerTurn]) -> Vec<WordAlignment>` — per-word
  speaker tagging (same order/length as input).
- `fill_turn_text(&[SpeakerTurn], &[WordAlignment]) -> Vec<SpeakerTurn>` —
  assemble `SpeakerTurn.text` from attributed words in time order.
- `attribute_and_fill(&[Word], &[SpeakerTurn]) -> WhoSaidWhat` — join + text-fill
  in one (pure, no ASR).
- `who_said_what(&[SpeakerTurn], &dyn Asr, &[f32], SampleRate) -> Result<WhoSaidWhat, AsrError>`
  — the cascade: **one** ASR pass over the whole audio, then join to the
  already-computed turns.
- `WhoSaidWhat { words: Vec<WordAlignment>, turns: Vec<SpeakerTurn> }`.

## Cascade order & rationale

Diarize **first** (caller supplies `turns`), then run ASR as a **single pass**
over the full audio, then join. The cascade is **diarizer-agnostic** — pass turns
from any pipeline (the validated legacy default, or v2). There is intentionally
no pipeline-bound `run_with_transcription`: the free function composes with
whichever diarizer the caller chose.

- **Per-segment ASR is NOT supported** — transcribing each turn separately loses
  cross-boundary language context and multiplies cost.
- **Overlap limitation:** words inside overlapped speech are attributed to the
  single **dominant** speaker only (max temporal overlap). This is a known loss
  on overlap-heavy audio, documented rather than masked.

## Dependencies

- `types` — `Word`, `SpeakerTurn`, `WordAlignment`, `TimeRange`, `SpeakerId`, `SampleRate`
- `asr` — `Asr` trait + `AsrError` (the backend, e.g. `polyvoice-asr`, is a separate opt-in crate)

## Invariants

- `attribute_words` preserves input word order and length; every word (with turns present) gets a speaker.
- Max temporal overlap wins; no-overlap → nearest turn by gap; ties → smaller `SpeakerId`, then earlier turn.
- Straddling words get the dominant speaker with confidence scaled by coverage share.
- `fill_turn_text` places a word in a turn when it was attributed to that turn's speaker and its midpoint lies in the turn's span; turns with no words keep `text: None`.
- No model/`ort`/IO dependency; default build unchanged (~30 MB), wasm-clean.

## Verification

```bash
cargo test --features attribution --lib attribution
cargo build --target wasm32-unknown-unknown --features attribution
```
