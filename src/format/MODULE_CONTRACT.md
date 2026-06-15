---
schema_version: 1
kind: module_contract
module: src/format
level: leaf
layer: io-projection
purpose: >
  Projects a diarization result into subtitle/text formats (SRT, WebVTT, plain
  text). Pure-Rust, wasm-clean, write-only (impl Write); one turn -> one block,
  lossless. Does NOT own RTTM (src/rttm) or JSON (serde) projection, parsing, or
  any filesystem I/O.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/format/
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
  - name: write_srt / write_vtt / write_txt
    kind: function
    visibility: public
    contract: >
      Write speaker turns to an impl Write as SubRip (SRT, comma ms), WebVTT (dot
      ms, with a WEBVTT header), or readable plain text
      ([start - end] SPEAKER_NN: text). One turn -> one block (lossless);
      timecodes are rounded to the millisecond; the cue carries SpeakerTurn.text
      when present.
    proof:
      kind: unit-test
      target: src/format::mod::tests
      command: cargo test --lib format
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: SpeakerTurn / SpeakerId / TimeRange are the projected data.
  external: []
consumers:
  - path: .
    uses:
      - write_srt
      - write_vtt
      - write_txt
invariants:
  - id: lossless-per-turn
    rule: Each input turn produces exactly one output block (one cue for SRT/VTT, one line for TXT).
    proof:
      kind: unit-test
      target: src/format::mod::tests::srt_blocks_are_numbered_and_lossless
      command: cargo test --lib format
  - id: srt-comma-vtt-dot
    rule: SRT timecodes use a comma millisecond separator; WebVTT uses a dot.
    proof:
      kind: unit-test
      target: src/format::mod::tests::timecode_srt_uses_comma_vtt_uses_dot
      command: cargo test --lib format
verification:
  pre_change:
    - cargo test --lib format
  full:
    - cargo test --lib format
    - cargo build --target wasm32-unknown-unknown --no-default-features --lib
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new output formats or writer helpers.
    - Adding unit tests.
    - Documentation.
  forbidden_mutations:
    - Adding filesystem I/O inside writers (must stay impl Write / wasm-clean).
    - Changing SRT/VTT timecode separators (breaks subtitle validity).
  escalation:
    - Changing a writer signature.
    - Changing timecode format or rounding.
---

# src/format

Subtitle / plain-text projections of a diarization result: SRT, WebVTT, TXT.
Pure-Rust, wasm-clean, write-only — one speaker turn projects to exactly one
block. RTTM lives in `src/rttm`; JSON is serde on `DiarizationResult`.
