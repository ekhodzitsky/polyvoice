---
schema_version: 1
kind: module_contract
module: src/wav
level: subsystem
layer: io
purpose: >
  Owns audio file loading for the pipeline. WAV via hound; optional multi-format
  decode (symphonia) and resampling to 16 kHz mono (rubato) behind feature
  `audio-io`. Does NOT own audio playback or feature extraction.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/wav/
  context_budget:
    max_files: 12
    max_source_lines: 1500
    max_contract_lines: 180
    max_readme_lines: 120
    max_todo_lines: 80
authority:
  write_policy: single_active_write_lease
  orchestrator: polyvoice-core
  read_agents: many_allowed
  migration_lease_required:
    - cross-workcell write
    - public surface migration
surface:
  - name: read_wav
    kind: function
    visibility: public
    contract: >
      Reads a WAV file and returns (samples as f32 Vec, sample_rate_hz).
      Does not resample.
    proof:
      kind: unit-test
      target: src/wav::mod::tests::missing_file_error
      command: cargo test --lib wav
  - name: load_audio
    kind: function
    visibility: public
    contract: >
      Returns mono f32 at TARGET_SAMPLE_RATE (16 kHz). Without audio-io:
      16 kHz WAV only. With audio-io: multi-format decode + resample.
    proof:
      kind: unit-test
      target: src/wav::mod::tests::load_audio_accepts_16k_wav
      command: cargo test --lib wav
  - name: WavError
    kind: enum
    visibility: public
    contract: >
      Error type for audio load failures (WAV, decode, resample, feature hints).
    proof:
      kind: unit-test
      target: src/wav::mod::tests::wav_error_display
      command: cargo test --lib wav
dependencies:
  internal: []
  external: []

consumers:
  - path: .
    uses:
      - read_wav
      - load_audio
      - WavError
      - polyvoice_internal
invariants:
  - id: sample-rate-exposed
    rule: read_wav returns the actual sample rate from the WAV header.
    proof:
      kind: integration-test
      target: tests/test_wav.rs
      command: cargo test --test test_wav
  - id: load-audio-target-rate
    rule: load_audio success always returns TARGET_SAMPLE_RATE (16000).
    proof:
      kind: unit-test
      target: src/wav::mod::tests::load_audio_accepts_16k_wav
      command: cargo test --lib wav
verification:
  pre_change:
    - cargo check --features audio-io
  full:
    - cargo test --lib wav
    - cargo test --lib wav --features audio-io
    - cargo test --test test_wav --features audio-io
    - cargo tree -e normal | rg "rubato|symphonia"  # empty without audio-io
agent_policy:
  allowed_mutations:
    - Adding WAV format support.
    - Optimizing read buffer sizes.
    - Extending audio-io decode/resample paths.
  forbidden_mutations:
    - Changing read_wav return type without migration lease.
    - Pulling rubato/symphonia into default features.
  escalation:
    - Changes to WAV read semantics or error variants.
---

# src/wav

WAV file reading utilities.
