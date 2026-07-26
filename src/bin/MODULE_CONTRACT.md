---
schema_version: 1
kind: module_contract
module: src/bin
level: subsystem
layer: cli
purpose: >
  CLI and agent-facing binaries: polyvoice (main diarization toolkit),
  polyvoice-bench (DER harness), polyvoice-measure, and polyvoice-mcp (MCP
  stdio server). Thin wrappers over the library; no business logic.
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
    max_files: 8
    max_source_lines: 3200
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
      Speaker diarization CLI. Default path since 0.11 is pipeline v2 + VBx
      (powerset -> ResNet34 -> VB-HMM/PLDA). Escape hatches: --legacy
      (Silero + AHC), --clusterer ahc. Commands: implicit diarize, download-models,
      models, completions, schema.
    proof:
      kind: smoke
      target: tests/cli_smoke_test.rs
      command: cargo test --test cli_smoke_test --features cli
  - name: polyvoice-bench
    kind: binary
    visibility: public
    contract: >
      DER benchmark on {audio,rttm} dataset directories. Default pipeline is v2
      (matches shipped CLI); --pipeline legacy remains for comparison. Emits
      JSON reports (schema polyvoice-bench-v0.10).
    proof:
      kind: smoke
      target: tests/cli_smoke_test.rs
      command: cargo test --test cli_smoke_test --features cli
  - name: polyvoice-mcp
    kind: binary
    visibility: public
    contract: >
      MCP stdio server exposing polyvoice.diarize (and ASR stubs). Uses the
      same production path as CLI (pipeline_v2, default clusterer vbx).
    proof:
      kind: unit-test
      target: src/bin/polyvoice-mcp.rs tests
      command: cargo test --bin polyvoice-mcp --features mcp
dependencies:
  internal:
    - module: pipeline_v2
      scope: orchestration
      reason: Production ONNX default for polyvoice / mcp / bench defaults.
    - module: pipeline
      scope: orchestration
      reason: --legacy and --pipeline legacy escape hatch.
    - module: models
      scope: infrastructure
      reason: ModelRegistry downloads and profile resolution.
  external: []
consumers:
  - path: .
    uses:
      - polyvoice
      - polyvoice-bench
      - polyvoice-mcp
invariants:
  - id: thin-wrapper
    rule: CLI binaries contain no business logic; all algorithms live in lib modules.
    proof:
      kind: static-check
      target: src/bin/
      command: grep -r "impl\|fn main" src/bin/ | wc -l
  - id: production-default-v2
    rule: >
      polyvoice and polyvoice-mcp default to pipeline_v2 (not the legacy
      Silero+AHC path). polyvoice-bench defaults to --pipeline v2.
    proof:
      kind: static-check
      target: src/bin/polyvoice.rs
      command: rg -n "default_value = \"vbx\"|pipeline_v2" src/bin/polyvoice.rs src/bin/polyvoice-mcp.rs src/bin/polyvoice-bench.rs
  - id: bench-layout
    rule: polyvoice-bench expects dataset layout audio/*.wav + rttm/*.rttm.
    proof:
      kind: static-check
      target: src/bin/polyvoice-bench.rs
      command: grep -c "audio_dir\|rttm_dir" src/bin/polyvoice-bench.rs
verification:
  pre_change:
    - cargo build --bin polyvoice --features cli
    - cargo build --bin polyvoice-bench --features cli
    - cargo test --test cli_smoke_test --features cli
  full:
    - cargo build --bin polyvoice --features cli
    - cargo build --bin polyvoice-bench --features cli
    - cargo build --bin polyvoice-mcp --features mcp
    - cargo test --test cli_smoke_test --features cli
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new CLI flags or subcommands.
    - Documentation improvements.
    - Aligning binary defaults with the library production path.
  forbidden_mutations:
    - Adding business logic or algorithms.
    - Changing the production default pipeline without a migration lease and
      DER evidence.
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
