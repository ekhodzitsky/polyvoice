---
schema_version: 1
kind: module_contract
module: src/bin
level: subsystem
layer: cli
purpose: >
  CLI binaries: polyvoice (main diarization toolkit) and polyvoice-bench
  (DER benchmark runner). Thin wrappers over the library; no business logic.
status: stable
owners:
  - polyvoice-cli
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/bin/
  context_budget:
    max_files: 6
    max_source_lines: 700
    max_contract_lines: 180
    max_readme_lines: 120
    max_todo_lines: 80
authority:
  write_policy: single_active_write_lease
  orchestrator: polyvoice-cli
  read_agents: many_allowed
  migration_lease_required:
    - cross-workcell write
    - public surface migration
surface:
  - name: polyvoice
    kind: binary
    visibility: public
    contract: >
      Speaker diarization CLI. Commands: diarize, download-models, models.
      Uses legacy v0.5 pipeline (SileroVad -> FbankOnnxExtractor -> AHC).
    proof:
      kind: smoke
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test
  - name: polyvoice-bench
    kind: binary
    visibility: public
    contract: >
      DER benchmark on {audio,rttm} dataset directories.
      Computes DER with configurable collar and produces JSON report.
    proof:
      kind: smoke
      target: tests/e2e_smoke_test.rs
      command: cargo test --test e2e_smoke_test
dependencies:
  internal: []
  external: []
consumers:
  - path: .
    uses:
      - polyvoice
      - polyvoice-bench
      - polyvoice_internal
invariants:
  - id: thin-wrapper
    rule: CLI binaries contain no business logic; all algorithms live in lib modules.
    proof:
      kind: static-check
      target: src/bin/
      command: grep -r "impl\|fn main" src/bin/ | wc -l
  - id: legacy-pipeline
    rule: polyvoice uses the v0.5 legacy pipeline, not the experimental M6b pipeline.
    proof:
      kind: static-check
      target: src/bin/polyvoice.rs
      command: grep -c "FbankOnnxExtractor\|SileroVad" src/bin/polyvoice.rs
  - id: bench-layout
    rule: polyvoice-bench expects dataset layout audio/*.wav + rttm/*.rttm.
    proof:
      kind: static-check
      target: src/bin/polyvoice-bench.rs
      command: grep -c "audio_dir\|rttm_dir" src/bin/polyvoice-bench.rs
verification:
  pre_change:
    - cargo build --bin polyvoice
    - cargo build --bin polyvoice-bench
    - cargo test --test e2e_smoke_test
  full:
    - cargo build --bin polyvoice
    - cargo build --bin polyvoice-bench
    - cargo test --test e2e_smoke_test
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new CLI flags or subcommands.
    - Documentation improvements.
  forbidden_mutations:
    - Adding business logic or algorithms.
    - Changing default pipeline without migration lease.
  escalation:
    - Any change to CLI argument structure breaking existing scripts.
risks:
  - description: CLI panics if ONNX models are missing; handled by ModelRegistry errors.
    severity: minor
    mitigation: ModelRegistry::ensure_for_profile returns descriptive errors.
    status: accepted
gaps:
  - description: No property tests for CLI argument parsing.
    severity: info
    status: open
---
