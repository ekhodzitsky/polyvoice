# src/attribution — TODO

- [x] Helper to fill `SpeakerTurn.text` from attributed words — `fill_turn_text`.
- [x] Wire into the cascaded diarize→ASR→join pipeline — `who_said_what` (diarizer-agnostic, one ASR pass).
- [x] O(W+T) two-pointer sweep join (bit-identical max-overlap vs historical scan).
- [x] Nearest-neighbor timestamp interpolation + `WordAlignment.interpolated`.
- [x] Opt-in sentence-level speaker smoothing (`.?!` splitter, configurable threshold).
- [x] Configurable word anchor start/mid/end for turn-text placement (default mid).
- [ ] Measure sentence smoothing on a real who-said-what cascade before flipping the default on.
- [ ] Consider overlap-region handling beyond dominant-speaker (multi-talker words).
- [ ] Optional ergonomic `Pipeline::run_with_transcription` on the production
      path (today the free `who_said_what` composes with any diarizer; v2 is the
      shipped ONNX default since 0.11).
