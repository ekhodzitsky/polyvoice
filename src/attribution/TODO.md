# src/attribution — TODO

- [x] Helper to fill `SpeakerTurn.text` from attributed words — `fill_turn_text`.
- [x] Wire into the cascaded diarize→ASR→join pipeline — `who_said_what` (diarizer-agnostic, one ASR pass).
- [ ] Consider overlap-region handling beyond dominant-speaker (multi-talker words).
- [ ] Optional ergonomic `Pipeline::run_with_transcription` once a pipeline is the
      validated default (today the free `who_said_what` composes with any diarizer;
      legacy is the shipped default, v2 needs segmentation work first).
