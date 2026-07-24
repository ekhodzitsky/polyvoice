# src/attribution — TODO

- [x] Helper to fill `SpeakerTurn.text` from attributed words — `fill_turn_text`.
- [x] Wire into the cascaded diarize→ASR→join pipeline — `who_said_what` (diarizer-agnostic, one ASR pass).
- [x] O(W+T) two-pointer sweep join (bit-identical max-overlap vs historical scan).
- [x] Nearest-neighbor timestamp interpolation + `WordAlignment.interpolated`.
- [x] Opt-in sentence-level speaker smoothing (`.?!` splitter, configurable threshold).
- [x] Configurable word anchor start/mid/end for turn-text placement (default mid).
- [ ] Measure sentence smoothing on a real who-said-what cascade before flipping the default on.
- [ ] Consider overlap-region handling beyond dominant-speaker (multi-talker words).
- [ ] Optional ergonomic `Pipeline::run_with_transcription` once a pipeline is the
      validated default (today the free `who_said_what` composes with any diarizer;
      legacy is the shipped default, v2 needs segmentation work first).
