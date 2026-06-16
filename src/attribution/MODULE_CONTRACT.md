---
schema_version: 1
kind: module_contract
module: src/attribution
level: leaf
layer: algorithm
purpose: >
  Word->speaker attribution join: maps raw ASR words (Word) onto diarization
  speaker turns (SpeakerTurn), producing WordAlignment per word. Pure-Rust,
  wasm-clean, behind the opt-in `attribution` feature. Does NOT run ASR, load
  models, touch ort, or do I/O — only interval arithmetic on TimeRange/SpeakerId.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/attribution/
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
  - name: attribute_words
    kind: function
    visibility: public
    contract: >
      attribute_words(&[Word], &[SpeakerTurn]) -> Vec<WordAlignment>, same order
      and length as the input words. Each word -> the max-overlap turn's speaker
      (confidence scaled by coverage share, so straddling words drop); no-overlap
      word -> nearest turn by gap; empty turns -> speaker None. Ties break to the
      smaller SpeakerId then the earlier turn.
    proof:
      kind: unit-test
      target: src/attribution::mod::tests
      command: cargo test --features attribution --lib attribution
dependencies:
  internal:
    - module: types
      scope: data-shape
      reason: Word / SpeakerTurn / WordAlignment / TimeRange / SpeakerId.
  external: []
consumers:
  - path: .
    uses:
      - attribute_words
invariants:
  - id: order-length-preserved
    rule: Output has the same length and order as the input words; every word with any turn present gets Some(SpeakerId).
    proof:
      kind: unit-test
      target: src/attribution::mod::tests::order_and_length_preserved_incl_before_after
      command: cargo test --features attribution --lib attribution
  - id: straddle-lowers-confidence
    rule: A word straddling a turn boundary is attributed to the dominant-overlap speaker with a reduced confidence.
    proof:
      kind: unit-test
      target: src/attribution::mod::tests::straddling_word_picks_dominant_and_lowers_confidence
      command: cargo test --features attribution --lib attribution
  - id: no-heavy-deps
    rule: Module is pure-Rust and wasm-clean; no ort/model/IO dependency; default build unchanged.
    proof:
      kind: build
      target: wasm32
      command: cargo build --target wasm32-unknown-unknown --features attribution
verification:
  pre_change:
    - cargo test --features attribution --lib attribution
  full:
    - cargo test --features attribution --lib attribution
    - cargo build --target wasm32-unknown-unknown --features attribution
agent_policy:
  allowed_mutations:
    - Tuning attribution rules / confidence scaling.
    - Adding unit tests.
    - Documentation.
  forbidden_mutations:
    - Adding ASR/model/ort/IO dependencies (attribution stays pure interval math).
    - Enabling the `attribution` feature by default.
  escalation:
    - Changing the attribute_words signature.
    - Changing the attribution tie-break or confidence semantics.
---

# src/attribution

Word→speaker attribution join. Pure-Rust, wasm-clean, opt-in (`attribution`
feature): `attribute_words` maps ASR words onto diarization turns by interval
overlap (reusing the same overlap math as `der`). No models, no ort, no I/O — the
ASR backend lives in the opt-in `polyvoice-asr` crate.
