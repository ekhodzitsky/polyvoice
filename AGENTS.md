<!-- From: /Users/ekhodzitsky/Documents/personal/polyvoice/AGENTS.md -->
# Guidelines for Correct Rust with polyvoice

> Version: 2.0.0 | COAD-native correctness guidelines

## Meta Principle

Before applying any rule, ask: **what problem does this solve?**

A newtype, a Hoare triple, or a refactor is justified only if it prevents a
concrete bug, clarifies an invariant, or removes a footgun. If the answer is
"it looks better" or "the score goes up" — revert. Decoration is not
engineering.

## The 5 Rules

### 1. Types Prove Invariants

Use the type system to make invalid states unrepresentable.

```rust
// BAD: any u32, impossible sample rates possible
fn process(samples: u32) -> u32 { ... }

// GOOD: invariant encoded in type
struct SampleRate(u32); // constructor enforces 8000..=192000
struct Confidence(f32); // constructor enforces 0.0..=1.0
```

Patterns: [docs/FORMALISM.md](docs/FORMALISM.md) §1–2 (Newtype, Phantom types, Typestate)

### 2. Functions Have Contracts

Every public function documents what it requires and guarantees.

```rust
/// { !items.is_empty() }
/// fn average(items: &[f64]) -> f64
/// { ret == sum(items) / items.len() }
pub fn average(items: &[f64]) -> f64 {
    debug_assert!(!items.is_empty(), "precondition: non-empty slice");
    items.iter().sum::<f64>() / items.len() as f64
}
```

Patterns: [docs/FORMALISM.md](docs/FORMALISM.md) §1 (Hoare triples, debug_assert!)

### 3. No unwrap/expect/panic Without Proof

Handle all cases. Use `Result`/`Option`. `unwrap()` is allowed only in tests or
with compile-time proof.

```rust
// BAD
let speaker = cluster.assign(&embedding).0;

// GOOD
let (speaker, confidence) = cluster.assign(&embedding);
```

Enforced by: clippy `unwrap_used = "deny"` + code review

### 4. Verify with Property Tests

Test universal properties, not just examples.

```rust
proptest! {
    #[test]
    fn cosine_similarity_range(a in vec(-1.0f32..=1.0, 256), b in vec(-1.0f32..=1.0, 256)) {
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim >= -1.0 && sim <= 1.0);
    }
}
```

Patterns: [docs/FORMALISM.md](docs/FORMALISM.md) §4 (proptest, fuzzing)

### 5. Check Contracts Automatically

Run verification before every commit:

```bash
# Full verification
cargo nextest run --profile ci --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
cargo deny check advisories licenses
```

---

## Release Checklist (mandatory before `cargo publish`)

The 0.6.1 incident (pipeline v2 shipped as CLI default, degraded DER on
long-form audio) must never repeat. Follow this checklist verbatim:

1. **Run the release gate script** — it fails fast if anything is broken:
   ```bash
   bash scripts/release-check.sh
   ```
2. **DER regression tests must pass** — they compare against
   `tests/der_baseline.json` with tolerance:
   - Legacy pipeline e2e_smoke: DER ≤ 7.62% (baseline 6.62% + 1.0%)
   - Legacy pipeline VoxConverse 10-file: DER ≤ 18.43% (baseline 17.43% + 1.0%)
   - Legacy pipeline AMI single: DER ≤ 37.30% (baseline 36.30% + 1.0%)
   - Pipeline v2 e2e_smoke: DER ≤ 5.79% (baseline 4.79% + 1.0%)
3. **If DER improved** — update `tests/der_baseline.json` with the new numbers
   and lower tolerance if appropriate. Never silence a regression test.
4. **If the default pipeline changed** — the release gate script will catch
   DER drift. Do not bypass it.
5. **Version bump** — update `Cargo.toml`, `python/Cargo.toml`,
   `python/pyproject.toml`, `tests/cli_smoke_test.rs`, and `CHANGELOG.md`.
6. **Tag format** — `vX.Y.Z` (e.g. `v0.6.2`).
7. **CI must be green** on the tag before crates.io publish. The
   `e2e-der-regression` job runs automatically for `refs/tags/v*`.

---

## COAD Agent Development Coordination

This repository uses COAD (Contract-Orchestrated Agent Development) for
agent work coordination.

COAD repository: https://github.com/ekhodzitsky/coad

Before editing, identify the relevant workcell and read its
`MODULE_CONTRACT.md`, `README.md`, and `TODO.md`.

Before claiming completion:

```bash
cargo nextest run --profile ci --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
cargo deny check advisories licenses
cargo hack check --feature-powerset --depth 2 --lib --bins
cargo machete
```

One leaf workcell may have only one active write agent. Read-only agents may
investigate, review, or verify in parallel. Composite workcell agents orchestrate
child work but do not directly edit child implementation files.

Keep at least one `MODULE_CONTRACT.md` for the module being changed. The module
is a workcell: one bounded agent workspace with ownership, surfaces, consumers,
invariants, verification, and write authority. The module directory must include
`README.md` and `TODO.md` for future agents.

## Reference (Optional)

For deeper theory and patterns, see:

- [docs/FORMALISM.md](docs/FORMALISM.md) — Concrete tools: PhantomData, Typestate, proptest, Miri, Kani
- [docs/GLOSSARY.md](docs/GLOSSARY.md) — Vocabulary: Invariant, Precondition, Monoid, Functor
- [docs/PIPELINE.md](docs/PIPELINE.md) — Development process: Spec → Type Design → Implement → Verify
- [docs/SEVERITY.md](docs/SEVERITY.md) — Issue classification: CRITICAL/MAJOR/MINOR/INFO

---

## Final Formula

```
Correctness = Types (prevent bugs) + Contracts (document intent)
            + Properties (verify universally) + Automation (enforce rules)
```
