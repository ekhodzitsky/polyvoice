# PIPELINE.md — Development Process for polyvoice

## 1. Specification

Before writing code, define:
- **What** — the functionality (e.g., "sliding-window overlap detection")
- **Invariants** — conditions that must always hold (e.g., "num_speakers <= max_speakers")
- **Contracts** — pre/postconditions for public functions

## 2. Type Design

Design types to make invalid states unrepresentable:
1. Identify values with constraints (sample rates, thresholds, counts)
2. Create newtypes with validated constructors
3. Use `Option<T>` for truly optional values, avoid sentinel values

## 3. Implementation

1. Write Hoare triples as doc comments
2. Add `debug_assert!` for preconditions
3. Propagate errors with `Result`, never `unwrap` in production
4. Mark `unsafe` blocks with `// SAFETY:` comments

## 4. Verification

### Required checks before every commit:
```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps
cargo fmt --check
```

### Optional but recommended:
```bash
cargo +nightly miri test       # If unsafe code changed
cargo audit                    # Check for vulnerabilities
```

## 5. Review Checklist

- [ ] Types encode invariants
- [ ] Public functions have Hoare triples
- [ ] No `unwrap`/`expect` in production code
- [ ] Unsafe blocks have SAFETY comments
- [ ] Tests cover edge cases and properties
- [ ] Documentation is complete (`cargo doc` has no warnings)
