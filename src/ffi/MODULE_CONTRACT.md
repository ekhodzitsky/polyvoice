---
schema_version: 1
kind: module_contract
module: src/ffi
level: subsystem
layer: bindings
purpose: >
  Owns C FFI bindings for polyvoice (ABI v3). Exposes pipeline, model registry,
  and VAD configuration to C callers. Does NOT own the Rust pipeline internals.
status: stable
owners:
  - polyvoice-core
workcell:
  type: leaf
  parent: ""
  children: []
  owns_paths:
    - src/ffi/
    - include/polyvoice.h
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
  - name: polyvoice_pipeline_create
    kind: function
    visibility: public
    contract: >
      C-visible constructor for Pipeline wrapper. Caller owns returned handle.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - name: polyvoice_pipeline_run
    kind: function
    visibility: public
    contract: >
      C-visible run function. Accepts raw audio buffer and sample rate; returns JSON.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - name: polyvoice_pipeline_run_format
    kind: function
    visibility: public
    contract: >
      C-visible run function with output-format selector (PolyvoiceFormat:
      JSON/RTTM/SRT/VTT/TXT). Same contract as polyvoice_pipeline_run otherwise;
      RTTM uses the fixed file id "audio"; unknown formats return InvalidArg.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - name: PolyvoiceFormat
    kind: enum
    visibility: public
    contract: >
      #[repr(C)] output-format selector for polyvoice_pipeline_run_format.
    proof:
      kind: static-check
      target: src/ffi/mod.rs
      command: cargo check --features ffi
  - name: polyvoice_pipeline_destroy
    kind: function
    visibility: public
    contract: >
      C-visible destructor. Must be called exactly once for every created pipeline.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - name: polyvoice_free_string
    kind: function
    visibility: public
    contract: >
      Frees strings returned by the run functions. Null-safe.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - name: PolyvoicePipeline
    kind: struct
    visibility: public
    contract: >
      Opaque C handle wrapping a Rust Pipeline.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
dependencies:
  internal:
    - module: pipeline
      scope: orchestration
      reason: Pipeline for diarization.
    - module: models
      scope: infrastructure
      reason: ModelRegistry for model downloads.
    - module: types
      scope: data-shape
      reason: DiarizationConfig, Profile.
    - module: vad
      scope: algorithm
      reason: VadConfig.
  external: []
consumers:
  - path: examples/ffi_usage.c
    uses:
      - polyvoice_pipeline_create
      - polyvoice_pipeline_run
      - polyvoice_pipeline_destroy
  - path: tests/ffi_smoke_test.rs
    uses:
      - PolyvoicePipeline
      - polyvoice_pipeline_run_format
invariants:
  - id: memory-safety
    rule: Every create has exactly one free; no use-after-free.
    proof:
      kind: integration-test
      target: tests/ffi_smoke_test.rs
      command: cargo test --test ffi_smoke_test --features ffi
  - id: abi-stable
    rule: C-visible structs use #[repr(C)] and fixed-size types.
    proof:
      kind: static-check
      target: src/ffi/mod.rs
      command: cargo check --features ffi
verification:
  pre_change:
    - cargo check --features ffi
  full:
    - cargo test --test ffi_smoke_test --features ffi
    - cargo clippy --all-targets --all-features -- -D warnings
agent_policy:
  allowed_mutations:
    - Adding new FFI wrappers for existing Rust surfaces.
    - Documentation of SAFETY invariants.
  forbidden_mutations:
    - Changing existing C function signatures (breaks ABI).
    - Removing #[repr(C)] from public structs.
  escalation:
    - Any change to C-visible function signatures or struct layouts.
---

# src/ffi

C FFI bindings for polyvoice (ABI v2).
