# COAD Project Standard — polyvoice

This repository follows the COAD (Contract-Orchestrated Agent Development)
standard.

COAD repository: https://github.com/ekhodzitsky/coad

## Target Outcome

An agent should be able to enter this repository and understand the relevant
work context in two minutes:

- what module owns the behavior;
- what surfaces are public or internal;
- what can be changed safely;
- what must not be changed silently;
- which consumers depend on the module;
- which invariants matter;
- which proof commands establish correctness;
- what evidence is needed for handoff.

## Repository Shape

```text
AGENTS.md
COAD_PROJECT_STANDARD.md
AGENT_FLOW.md
src/
  der/
    MODULE_CONTRACT.md
    README.md
    TODO.md
    mod.rs
  segmentation/
    MODULE_CONTRACT.md
    README.md
    TODO.md
    mod.rs
    ...
  models/
    MODULE_CONTRACT.md
    README.md
    TODO.md
    mod.rs
    ...
  pipeline/
    MODULE_CONTRACT.md
    README.md
    TODO.md
    mod.rs
  types/
    MODULE_CONTRACT.md
    README.md
    TODO.md
    mod.rs
  ...
tests/
docs/
```

## Workcell As Agent Workspace

Every meaningful module is a workcell: an agent workspace sized for fast
orientation, local ownership, and proof-backed edits.

Required local context per workcell:
- `MODULE_CONTRACT.md` — boundary, surfaces, consumers, invariants, verification;
- `README.md` — purpose, surfaces, dependencies, invariants, verification;
- `TODO.md` — current gaps, planned work, known debt;
- focused tests or proof commands for the module's critical promises.

## Workcell Size Standard

Advisory budgets:
- source files in normal edit scope: 12;
- source lines in normal edit scope: 1500;
- `MODULE_CONTRACT.md`: 180 lines;
- `README.md`: 120 lines;
- `TODO.md`: 80 lines;
- public or cross-workcell surfaces: 8;
- invariants: 12;
- active write agents: 1.

## Adoption Level

Current: **Level 4** — Ledger-audited orchestration.

- `AGENTS.md` with COAD guidance: yes
- All meaningful modules have `MODULE_CONTRACT.md`, `README.md`, `TODO.md`: yes
- `cargo test`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo doc --no-deps` pass: yes
- Critical surfaces have concrete proof commands: yes
- All non-deprecated modules have passing unit/property/integration tests: yes
- Write lease and migration lease discipline demonstrated: yes
- Task/proof/handoff/review/integration contracts used in practice: yes
- Execution ledger tracks completed work with evidence: yes
- CI enforces tests and clippy on every PR: yes

### What Each Level Means Here

**Level 0** — Agent-ready module context: every module has a contract, README, and TODO.

**Level 1** — Proven module invariants: every critical surface has a passing proof
command (unit test, property test, integration test, or benchmark).

**Level 2** — Workcell authority discipline: explicit write leases for leaf workcells,
migration leases for cross-workcell changes, parent-orchestrator escalation.

**Level 3** — Task handoff discipline: non-trivial changes use task contracts,
proof contracts, handoff records, and review records.

**Level 4** — Ledger-audited orchestration: completed work is recorded in the
execution ledger with reproducible evidence, and CI enforces repository compliance.

## Multi-Agent Standard

Default concurrency rule:
```text
one leaf workcell -> one active write agent
```

Read-only agents may investigate, review, or verify in parallel.
Cross-workcell changes require a migration lease approved by the nearest common
orchestrator.

## Change Standard

Every non-trivial change must declare:
- affected module;
- read scope;
- write scope;
- invariants touched or explicitly not touched;
- consumers affected or explicitly not affected;
- proof commands required before handoff;
- contract or local context updates required by the change.

## Validator

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
```
