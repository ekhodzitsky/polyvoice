# FORMALISM.md — Concrete Verification Tools for polyvoice

## §1 Hoare Triples & Contracts

Every public function must document its contract as a Hoare triple:

```rust
/// { pre: !samples.is_empty() }
/// fn extract(samples: &[f32]) -> Result<Vec<f32>, Error>
/// { post: ret.is_ok() => ret.as_ref().unwrap().len() == EMBEDDING_DIM }
```

Use `debug_assert!` for cheap runtime precondition checks.

## §2 Newtypes & Typestate

Encode invariants in types to prevent invalid states at compile time.

### Newtype pattern
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence(f32);

impl Confidence {
    pub fn new(v: f32) -> Option<Self> {
        (0.0..=1.0).contains(&v).then_some(Self(v))
    }
}
```

### Typestate pattern
```rust
// BAD: bool flag checked at runtime
struct Diarizer { enabled: bool, state: State }

// GOOD: two types, invalid states unrepresentable
struct InactiveDiarizer;
struct ActiveDiarizer { state: State }
```

## §3 PhantomData & Lifetime Encoding

Use `PhantomData` to encode unused generic parameters in structs.

## §4 Property-Based Testing

Use `proptest` to verify universal properties:

```toml
[dev-dependencies]
proptest = "1"
```

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn cosine_similarity_range(a in vec(-1.0f32..=1.0, 256), b in vec(-1.0f32..=1.0, 256)) {
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim >= -1.0 && sim <= 1.0);
    }
}
```

## §5 Miri (Undefined Behavior Detection)

Run Miri on unsafe code:
```bash
cargo +nightly miri test
```

## §6 Kani (Model Checking)

For critical unsafe code, use Kani to prove absence of panics:
```bash
cargo +nightly kani --function my_unsafe_fn
```
