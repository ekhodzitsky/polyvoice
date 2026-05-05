# Guidelines for Correct Rust with polyvoice

> Version: 1.0.0 | Based on https://github.com/ekhodzitsky/kimi-guidelines

## Meta Principle

Before applying any rule, ask: **what problem does this solve?**

A newtype, a Hoare triple, or a refactor is justified only if it prevents a concrete bug, clarifies an invariant, or removes a footgun. If the answer is "it looks better" or "the score goes up" — revert. Decoration is not engineering.

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

Handle all cases. Use `Result`/`Option`. `unwrap()` is allowed only in tests or with compile-time proof.

```rust
// BAD
let speaker = cluster.assign(&embedding).0;

// GOOD
let (speaker, confidence) = cluster.assign(&embedding);
```

Enforced by: `cargo kimi check` + clippy `unwrap_used = "deny"`

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

Run mechanized verification before every commit:

```bash
# Check contracts, unwrap/expect/panic, unsafe SAFETY comments
cargo kimi check

# Full verification
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
```

---

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
