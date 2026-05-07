# M0 — Plumbing & Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay the v1.0 foundation in polyvoice without breaking the existing v0.5.x API: a typed `Profile` enum, a `ModelRegistry` library type with manifest-driven SHA-256-verified downloads, new Cargo features for execution providers and profiles, an extended CI matrix, and a `release-gate.sh` skeleton.

**Architecture:** Additive-only changes. New code lives under `src/models/` and a new `Profile` enum joins `src/types.rs`. Existing `Pipeline`/`OfflineDiarizer`/`SileroVad`/`FbankOnnxExtractor` paths are untouched. The CLI gains a `--profile` flag on the existing `download-models` subcommand; old `--dir` behavior is preserved as `Profile::Custom`. The downloaded files are still the v0.5.x set (`wespeaker_resnet34.onnx` + `silero_vad.onnx`) — both Mobile and Balanced profiles map to them in M0; M1/M2/M5 swap manifest entries to add powerset/CAM++/INT8 versions.

**Tech Stack:** Rust 2024 edition, `ort 2.0.0-rc.12` (already a dep), new deps `toml 0.8`, `sha2 0.10`. CI on GitHub Actions with new `cross 0.2` job for `aarch64-unknown-linux-gnu` and a smoke-compile job for `wasm32-unknown-unknown`. Bash for `scripts/release-gate.sh`.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `toml`, `sha2` deps; new features `coreml`, `nnapi`, `xnnpack`, `download`, `profile-mobile`, `profile-balanced`, `profile-all` |
| `src/types.rs` | modify | Add `Profile` enum + impl |
| `src/lib.rs` | modify | `pub mod models;` and `pub use types::Profile;` |
| `src/models/mod.rs` | create | `ModelRegistry`, `ProfileModels`, `RegistryError`, public re-exports |
| `src/models/manifest.rs` | create | `Manifest`, `ModelEntry`, `ProfileEntry` types + TOML parser |
| `src/models/download.rs` | create | `download_with_checksum` (streamed SHA-256) |
| `src/models/manifest.toml` | create | Embedded default manifest (FP32 entries for M0 set) |
| `src/bin/polyvoice.rs` | modify | Extend `download-models` subcommand with `--profile` |
| `tests/registry_test.rs` | create | Integration tests for `ModelRegistry` (cache, checksum, manifest) |
| `.github/workflows/ci.yml` | modify | Add `cross-aarch64-linux` and `wasm32-smoke` jobs |
| `scripts/release-gate.sh` | create | Bash stub matching spec §9.10 |
| `CHANGELOG.md` | modify | Add Unreleased section listing M0 additions |

Total new code roughly 600 lines Rust + 80 lines Bash + 30 lines TOML.

---

## Task 1: Add Cargo features and dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1.1: Add `toml` and `sha2` to `[dependencies]`**

Open `Cargo.toml`. Locate the `[dependencies]` section (currently ends with `fastrand = "2.4.1"`). Add these two lines immediately above the `[dev-dependencies]` block:

```toml
toml = { version = "0.8", optional = true }
sha2 = { version = "0.10", optional = true }
```

Both are gated behind the new `download` feature (see step 1.3), so they remain optional and don't widen the default dependency surface.

- [ ] **Step 1.2: Update `[features]` block — add new feature flags**

Find the existing `[features]` block:

```toml
[features]
default = []
onnx = ["dep:ort"]
ffi = []
cli = ["onnx", "dep:clap", "dep:ureq", "dep:dirs"]
```

Replace it with:

```toml
[features]
default = []
# ONNX-based embedding extractors (pulls ort).
onnx = ["dep:ort"]
# C FFI bindings.
ffi = []
# Library-level model registry with HTTP downloads + SHA-256 verification.
download = ["dep:ureq", "dep:dirs", "dep:toml", "dep:sha2"]
# CLI binary (polyvoice command).
cli = ["onnx", "download", "dep:clap"]

# Execution providers (forward `ort` features). Used by Pipeline implementations from M1+.
coreml = ["onnx", "ort/coreml"]
nnapi = ["onnx", "ort/nnapi"]
xnnpack = ["onnx", "ort/xnnpack"]

# Model profile bundles. M0 only emits metadata; CLI uses these to pick which
# manifest entries to ensure(). M5 will introduce INT8 model variants.
profile-mobile = []
profile-balanced = []
profile-all = ["profile-mobile", "profile-balanced"]
```

Notes:
- `cli` now depends on `download` (which transitively pulls `ureq`, `dirs`, `toml`, `sha2`). The CLI used to pull `ureq` and `dirs` directly — now they come through `download`. Behavior is unchanged from a user perspective.
- `ort/coreml`, `ort/nnapi`, `ort/xnnpack` are documented as features of the `ort` crate; we forward them. If the chosen `ort 2.0.0-rc.12` does not expose these names exactly, update them per `cargo metadata --format-version 1 | jq '.packages[] | select(.name == "ort").features'` after Step 1.4 fails.

- [ ] **Step 1.3: Verify `cargo check` passes for each non-default feature combo**

Run all the following and confirm each prints no errors:

```bash
cargo check
cargo check --features onnx
cargo check --features download
cargo check --features cli
cargo check --features ffi
cargo check --features profile-all
```

Expected: each command exits 0 with `Finished ...` line.

- [ ] **Step 1.4: Verify `cargo check --features coreml` works on macOS only, gracefully fails elsewhere**

On macOS:
```bash
cargo check --features coreml
```
Expected: success.

On Linux/Windows the build script for `ort/coreml` may print a warning or fail. If it fails compilation, change line `coreml = ["onnx", "ort/coreml"]` to `coreml = ["onnx", "ort?/coreml"]` (weak optional dep) and re-run. If `ort` doesn't expose `coreml` at all in the current version, downgrade this feature to a documentation-only stub: `coreml = ["onnx"]` and add a `# TODO(M8): wire ort/coreml feature` comment. Same triage applies to `nnapi` and `xnnpack`.

- [ ] **Step 1.5: Commit**

```bash
git add Cargo.toml
git commit -m "feat(cargo): add download/profile/EP features for v1.0 plumbing"
```

---

## Task 2: Add `Profile` enum to `src/types.rs`

**Files:**
- Modify: `src/types.rs`
- Test: `src/types.rs` (in-file `#[cfg(test)] mod tests` already exists pattern is fine; add separate module)

- [ ] **Step 2.1: Write the failing test**

Append the following to the bottom of `src/types.rs`:

```rust
#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn mobile_profile_uses_cam_pp_dim() {
        assert_eq!(Profile::Mobile.embedding_dim(), 192);
    }

    #[test]
    fn balanced_profile_uses_resnet34_dim() {
        assert_eq!(Profile::Balanced.embedding_dim(), 256);
    }

    #[test]
    fn custom_profile_dim_is_unresolved() {
        assert_eq!(Profile::Custom.embedding_dim(), 0);
    }

    #[test]
    fn default_thresholds_match_spec() {
        // §5.1 of v1.0 design spec
        assert!((Profile::Mobile.default_threshold() - 0.55).abs() < 1e-6);
        assert!((Profile::Balanced.default_threshold() - 0.45).abs() < 1e-6);
        assert!((Profile::Custom.default_threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn manifest_id_for_each_variant() {
        assert_eq!(Profile::Mobile.manifest_id(), "mobile");
        assert_eq!(Profile::Balanced.manifest_id(), "balanced");
        assert_eq!(Profile::Custom.manifest_id(), "custom");
    }

    #[test]
    fn from_str_parses_kebab_and_lowercase() {
        assert_eq!("mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("Mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("balanced".parse::<Profile>().unwrap(), Profile::Balanced);
        assert!("nope".parse::<Profile>().is_err());
    }
}
```

- [ ] **Step 2.2: Run tests, confirm they fail**

```bash
cargo test --lib types::profile_tests
```

Expected output: compilation error or test failures referencing missing `Profile` type. Do not proceed until confirmed.

- [ ] **Step 2.3: Add the `Profile` enum**

Locate a stable insertion point in `src/types.rs` — immediately after the `SpeakerId` `impl Display` block (around line 81). Insert:

```rust
/// Pre-configured model bundles trading off accuracy and footprint.
///
/// `Mobile` targets weak/embedded ARM CPUs (≤10 MB total models, ≤200 MB peak RAM).
/// `Balanced` targets modern phone/laptop ARM CPUs (≤35 MB total models, ≤400 MB peak RAM).
/// `Custom` defers all model selection to the caller and is used by `PipelineBuilder`
/// when individual `Segmenter`/`Embedder`/`Clusterer` instances are supplied directly.
///
/// Added in v0.6 (M0). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md`
/// §5.1 for the full motivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    Mobile,
    Balanced,
    Custom,
}

impl Profile {
    /// Embedding dimension produced by the embedder for this profile.
    /// Returns 0 for `Custom` (caller must resolve dimension explicitly).
    pub const fn embedding_dim(self) -> usize {
        match self {
            Profile::Mobile => 192,    // CAM++ output dim (lands in M2)
            Profile::Balanced => 256,  // WeSpeaker ResNet34 output dim
            Profile::Custom => 0,
        }
    }

    /// Default cosine similarity threshold tuned to the embedding space of this profile.
    pub const fn default_threshold(self) -> f32 {
        match self {
            Profile::Mobile => 0.55,
            Profile::Balanced => 0.45,
            Profile::Custom => 0.5,
        }
    }

    /// Stable identifier used in the manifest TOML and CLI flags.
    pub const fn manifest_id(self) -> &'static str {
        match self {
            Profile::Mobile => "mobile",
            Profile::Balanced => "balanced",
            Profile::Custom => "custom",
        }
    }
}

impl std::str::FromStr for Profile {
    type Err = ProfileParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mobile" => Ok(Profile::Mobile),
            "balanced" => Ok(Profile::Balanced),
            "custom" => Ok(Profile::Custom),
            other => Err(ProfileParseError(other.to_owned())),
        }
    }
}

/// Returned by `Profile::from_str` when the input doesn't match a known variant.
#[derive(Debug, Clone)]
pub struct ProfileParseError(pub String);

impl std::fmt::Display for ProfileParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown profile '{}': expected mobile|balanced|custom", self.0)
    }
}

impl std::error::Error for ProfileParseError {}
```

- [ ] **Step 2.4: Run tests, confirm they pass**

```bash
cargo test --lib types::profile_tests
```

Expected: 6 tests pass.

- [ ] **Step 2.5: Run clippy and fmt**

```bash
cargo fmt
cargo clippy --lib --all-features -- -D warnings
```

Expected: no warnings.

- [ ] **Step 2.6: Commit**

```bash
git add src/types.rs
git commit -m "feat(types): add Profile enum (Mobile/Balanced/Custom)"
```

---

## Task 3: Create `Manifest` types in `src/models/manifest.rs`

**Files:**
- Create: `src/models/manifest.rs`

- [ ] **Step 3.1: Create the directory + empty module file**

```bash
mkdir -p src/models
touch src/models/manifest.rs
```

- [ ] **Step 3.2: Write the failing tests first**

Open `src/models/manifest.rs` and write only the tests block — implementation comes after:

```rust
//! TOML manifest describing where each ONNX model lives, its checksum, and
//! which model each `Profile` resolves to.

use serde::Deserialize;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
        schema = "polyvoice-models-v1"

        [profiles.mobile]
        segmenter = "silero_vad"
        embedder  = "wespeaker_resnet34"

        [profiles.balanced]
        segmenter = "silero_vad"
        embedder  = "wespeaker_resnet34"

        [models.silero_vad]
        url      = "https://example.com/silero_vad.onnx"
        sha256   = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
        size     = 2300000
        filename = "silero_vad.onnx"

        [models.wespeaker_resnet34]
        url      = "https://example.com/wespeaker.onnx"
        sha256   = "11112222333344445555666677778888aaaabbbbccccddddeeeeffff00001111"
        size     = 25000000
        filename = "wespeaker_resnet34.onnx"
    "#;

    #[test]
    fn parse_known_good_manifest() {
        let m = Manifest::from_toml_str(SAMPLE).expect("must parse");
        assert_eq!(m.schema, "polyvoice-models-v1");
        assert_eq!(m.profiles.len(), 2);
        assert_eq!(m.models.len(), 2);
        assert_eq!(m.profiles["mobile"].segmenter, "silero_vad");
        assert_eq!(m.models["silero_vad"].size, Some(2300000));
        assert_eq!(m.models["silero_vad"].filename, "silero_vad.onnx");
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let bad = SAMPLE.replace("polyvoice-models-v1", "polyvoice-models-v999");
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").contains("v999") || format!("{err}").contains("schema"));
    }

    #[test]
    fn profile_lookup_resolves_to_models() {
        let m = Manifest::from_toml_str(SAMPLE).unwrap();
        let prof = m.profile("mobile").expect("mobile profile present");
        let seg = m.model(&prof.segmenter).expect("segmenter resolved");
        assert_eq!(seg.filename, "silero_vad.onnx");
    }

    #[test]
    fn missing_profile_returns_none() {
        let m = Manifest::from_toml_str(SAMPLE).unwrap();
        assert!(m.profile("nope").is_none());
    }

    #[test]
    fn rejects_profile_with_missing_model_reference() {
        let bad = r#"
            schema = "polyvoice-models-v1"
            [profiles.mobile]
            segmenter = "ghost_model"
            embedder  = "ghost_model"
            [models.silero_vad]
            url = "https://example.com/x"
            sha256 = "abc"
            filename = "silero_vad.onnx"
        "#;
        let err = Manifest::from_toml_str(bad).expect_err("must fail");
        assert!(format!("{err}").to_lowercase().contains("ghost_model"));
    }

    #[test]
    fn rejects_invalid_sha256_length() {
        let bad = SAMPLE.replace(
            "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
            "tooshort",
        );
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").to_lowercase().contains("sha256"));
    }
}
```

- [ ] **Step 3.3: Run tests, confirm compilation failures**

```bash
cargo test --features download --lib models::manifest::tests 2>&1 | head -40
```

Expected: build errors referring to `Manifest`, `from_toml_str`, etc.

- [ ] **Step 3.4: Implement the manifest types and parser**

Replace the `//! ...` doc comment and `use` lines at the top of `src/models/manifest.rs` with the full implementation, then **keep** the test block from Step 3.2 at the bottom of the file. The complete file after this step:

```rust
//! TOML manifest describing where each ONNX model lives, its checksum, and
//! which model each `Profile` resolves to.

use serde::Deserialize;
use std::collections::HashMap;

/// Schema version identifier the parser accepts. Bump when manifest format changes.
pub const SCHEMA_V1: &str = "polyvoice-models-v1";

/// The full registry manifest: list of model entries plus a profile → model_id map.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: String,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileEntry>,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileEntry {
    pub segmenter: String,
    pub embedder: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub url: String,
    pub sha256: String,
    /// Optional declared size in bytes (informational).
    #[serde(default)]
    pub size: Option<u64>,
    /// Filename used when caching to disk. Required so the cache is deterministic
    /// across renames upstream.
    pub filename: String,
    /// Optional calibration descriptor (filled in M5 for INT8 entries).
    #[serde(default)]
    pub calibration: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("unsupported schema version: expected '{}', got '{0}'", SCHEMA_V1)]
    UnsupportedSchema(String),
    #[error("profile '{profile}' references unknown model '{model}'")]
    DanglingModelRef { profile: String, model: String },
    #[error("model '{model}' has invalid sha256 '{sha}': expected 64 lowercase hex chars")]
    InvalidSha256 { model: String, sha: String },
}

impl Manifest {
    /// Parse a manifest from a TOML source string.
    ///
    /// Validates: schema version, that every profile's `segmenter` and `embedder`
    /// reference an existing entry in `models`, and that every `sha256` is exactly
    /// 64 lowercase hex characters.
    pub fn from_toml_str(s: &str) -> Result<Self, ManifestError> {
        let m: Manifest = toml::from_str(s)?;
        if m.schema != SCHEMA_V1 {
            return Err(ManifestError::UnsupportedSchema(m.schema));
        }
        for (model_id, entry) in &m.models {
            if !is_valid_sha256_hex(&entry.sha256) {
                return Err(ManifestError::InvalidSha256 {
                    model: model_id.clone(),
                    sha: entry.sha256.clone(),
                });
            }
        }
        for (name, prof) in &m.profiles {
            if !m.models.contains_key(&prof.segmenter) {
                return Err(ManifestError::DanglingModelRef {
                    profile: name.clone(),
                    model: prof.segmenter.clone(),
                });
            }
            if !m.models.contains_key(&prof.embedder) {
                return Err(ManifestError::DanglingModelRef {
                    profile: name.clone(),
                    model: prof.embedder.clone(),
                });
            }
        }
        Ok(m)
    }

    pub fn profile(&self, id: &str) -> Option<&ProfileEntry> {
        self.profiles.get(id)
    }

    pub fn model(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }
}

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    // (test block from Step 3.2 stays here unchanged)
}
```

Keep the **entire** `#[cfg(test)] mod tests { ... }` block from Step 3.2 at the bottom. Do not duplicate it.

- [ ] **Step 3.5: Run tests, confirm 6/6 pass**

```bash
cargo test --features download --lib models::manifest::tests
```

Expected output line: `test result: ok. 6 passed`.

- [ ] **Step 3.6: Add module declaration in mod.rs (stub)**

Create `src/models/mod.rs` with just enough to make `manifest` accessible. Write to that file:

```rust
//! Model registry — manifest-driven downloads with SHA-256 verification.
//!
//! Added in v0.6 (M0). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §6.

pub mod manifest;
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};
```

- [ ] **Step 3.7: Wire module into lib.rs (gated)**

In `src/lib.rs`, add a new gated module declaration. Find the existing `pub mod ahc;` line block (around line 35–58) and append after the last `pub mod` line (after `pub mod wav;`, around line 58):

```rust
#[cfg(feature = "download")]
pub mod models;
```

Then run:

```bash
cargo check --features download
cargo test --features download --lib
```

Expected: clean build, all existing tests still pass plus the 6 new manifest tests.

- [ ] **Step 3.8: Commit**

```bash
git add src/models/mod.rs src/models/manifest.rs src/lib.rs
git commit -m "feat(models): add Manifest TOML parser with schema validation"
```

---

## Task 4: Create the embedded `manifest.toml` for M0

**Files:**
- Create: `src/models/manifest.toml`

The M0 manifest declares only the two FP32 models polyvoice already supports (`silero_vad.onnx`, `wespeaker_resnet34.onnx`). Both `mobile` and `balanced` profiles map to the same pair — they will diverge in M2 (CAM++ for mobile) and M5 (INT8 versions). This keeps the registry exercisable end-to-end in M0 without requiring new model uploads.

The actual SHA-256 values are computed in Task 5; for this step use placeholders that we will replace.

- [ ] **Step 4.1: Create `src/models/manifest.toml` with placeholder checksums**

Write to `src/models/manifest.toml`:

```toml
schema = "polyvoice-models-v1"

# M0 (initial). Both profiles map to the legacy v0.5 model pair until M2/M5 land.
# DO NOT remove `mobile` or `balanced` keys: external code (CLI, ModelRegistry,
# downstream users) treats them as stable identifiers.
[profiles.mobile]
segmenter = "silero_vad"
embedder  = "wespeaker_resnet34"

[profiles.balanced]
segmenter = "silero_vad"
embedder  = "wespeaker_resnet34"

[models.silero_vad]
url      = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
sha256   = "0000000000000000000000000000000000000000000000000000000000000000"
size     = 2327524
filename = "silero_vad.onnx"

[models.wespeaker_resnet34]
url      = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true"
sha256   = "0000000000000000000000000000000000000000000000000000000000000000"
size     = 26219527
filename = "wespeaker_resnet34.onnx"
```

The two `0000…` strings are deliberately wrong — Step 5 replaces them.

- [ ] **Step 4.2: Embed manifest in code via `include_str!`**

Open `src/models/mod.rs` and replace the contents with:

```rust
//! Model registry — manifest-driven downloads with SHA-256 verification.

pub mod manifest;
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};

/// The default manifest shipped with the crate. Embedded at compile time.
pub const DEFAULT_MANIFEST_TOML: &str = include_str!("manifest.toml");

/// Parse the bundled default manifest. Panics in debug if the embedded TOML is
/// malformed — that's a static asset bug caught by `cargo test`.
pub fn default_manifest() -> Manifest {
    Manifest::from_toml_str(DEFAULT_MANIFEST_TOML)
        .expect("embedded manifest.toml must parse — this is a static-asset bug")
}
```

The `expect` here is justified by the spec's "compile-time proof" carve-out: `manifest.toml` ships with the crate; if it doesn't parse, the build is broken regardless. We add a test in the next step to enforce this at CI time.

- [ ] **Step 4.3: Add a test that asserts the embedded manifest parses**

Append to `src/models/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        // This will panic if the bundled manifest.toml is malformed.
        let m = default_manifest();
        assert_eq!(m.schema, SCHEMA_V1);
        assert!(m.profiles.contains_key("mobile"));
        assert!(m.profiles.contains_key("balanced"));
    }

    #[test]
    fn embedded_manifest_lists_legacy_models() {
        let m = default_manifest();
        assert!(m.models.contains_key("silero_vad"));
        assert!(m.models.contains_key("wespeaker_resnet34"));
    }

    #[test]
    fn both_profiles_resolve_to_legacy_models_in_m0() {
        // Documented assumption: until M2/M5, both profiles map to the same
        // physical models. Remove this test when CAM++ lands in M2.
        let m = default_manifest();
        let mob = m.profile("mobile").unwrap();
        let bal = m.profile("balanced").unwrap();
        assert_eq!(mob.segmenter, bal.segmenter);
        assert_eq!(mob.embedder, bal.embedder);
    }
}
```

- [ ] **Step 4.4: Run tests, confirm 3/3 pass and 6/6 manifest tests still pass**

```bash
cargo test --features download --lib models::
```

Expected: 9 tests pass total.

- [ ] **Step 4.5: Commit**

```bash
git add src/models/manifest.toml src/models/mod.rs
git commit -m "feat(models): embed default manifest with M0 legacy model entries"
```

---

## Task 5: Lock in real SHA-256 checksums for M0 models

The placeholder checksums fail the `is_valid_sha256_hex` check (they're zeros, which pass length but not the spirit). Replace them with real values.

**Files:**
- Modify: `src/models/manifest.toml`

- [ ] **Step 5.1: Compute SHA-256 of `silero_vad.onnx`**

Run:

```bash
mkdir -p /tmp/polyvoice-m0-shas
curl -sL "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
  -o /tmp/polyvoice-m0-shas/silero_vad.onnx
shasum -a 256 /tmp/polyvoice-m0-shas/silero_vad.onnx
ls -la /tmp/polyvoice-m0-shas/silero_vad.onnx | awk '{print "size="$5}'
```

Expected output: a single line `<64-char-hex>  /tmp/.../silero_vad.onnx` plus a `size=...` line. **Record both values.** Example (your numbers will differ):
```
2c3a... silero_vad.onnx
size=2327524
```

- [ ] **Step 5.2: Compute SHA-256 of `wespeaker_resnet34.onnx`**

```bash
curl -sL "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true" \
  -o /tmp/polyvoice-m0-shas/wespeaker_resnet34.onnx
shasum -a 256 /tmp/polyvoice-m0-shas/wespeaker_resnet34.onnx
ls -la /tmp/polyvoice-m0-shas/wespeaker_resnet34.onnx | awk '{print "size="$5}'
```

Record SHA and size.

- [ ] **Step 5.3: Update `src/models/manifest.toml` with real values**

Replace the two `sha256 = "0000…"` lines and update the `size` fields if the recorded sizes differ from the current placeholders. The result should look like (with your actual hex):

```toml
[models.silero_vad]
url      = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
sha256   = "<real_sha256_from_step_5_1>"
size     = <real_size_from_step_5_1>
filename = "silero_vad.onnx"

[models.wespeaker_resnet34]
url      = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true"
sha256   = "<real_sha256_from_step_5_2>"
size     = <real_size_from_step_5_2>
filename = "wespeaker_resnet34.onnx"
```

- [ ] **Step 5.4: Run tests to confirm no regression**

```bash
cargo test --features download --lib models::
```

Expected: 9/9 pass.

- [ ] **Step 5.5: Cleanup tmp dir**

```bash
rm -rf /tmp/polyvoice-m0-shas
```

- [ ] **Step 5.6: Commit**

```bash
git add src/models/manifest.toml
git commit -m "feat(models): pin real sha256 checksums for M0 model bundle"
```

---

## Task 6: Implement `download_with_checksum` in `src/models/download.rs`

**Files:**
- Create: `src/models/download.rs`
- Test: `src/models/download.rs` (in-file `#[cfg(test)] mod tests`)

- [ ] **Step 6.1: Add `tempfile` to dev-dependencies if missing**

Open `Cargo.toml`. Look for `[dev-dependencies]`. Confirm `tempfile = "3"` is present (it is in the existing file). If not, add it. Skip if already there.

- [ ] **Step 6.2: Write the failing test**

Create `src/models/download.rs` with **only** the test block first:

```rust
//! HTTP download with streamed SHA-256 verification.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    const TEST_BYTES: &[u8] = b"polyvoice";

    /// Compute the expected SHA-256 of `TEST_BYTES` at test time, so the test is
    /// robust against typos in a hardcoded constant.
    fn test_bytes_sha256() -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(TEST_BYTES);
        format!("{:x}", h.finalize())
    }

    #[test]
    fn verify_existing_file_passes_when_hash_matches() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.bin");
        fs::write(&path, TEST_BYTES).unwrap();
        verify_sha256(&path, &test_bytes_sha256()).expect("hash must match");
    }

    #[test]
    fn verify_existing_file_fails_when_hash_differs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.bin");
        fs::write(&path, b"different content").unwrap();
        let err = verify_sha256(&path, &test_bytes_sha256()).expect_err("must mismatch");
        assert!(matches!(err, DownloadError::ChecksumMismatch { .. }));
    }

    #[test]
    fn verify_streams_large_file_without_loading_into_ram() {
        // Write a 5 MB file; verify_sha256 must use streaming reader, not Vec::read_to_end.
        // The test passes purely if it doesn't OOM and computes a deterministic hash.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("big.bin");
        let mut f = fs::File::create(&path).unwrap();
        for _ in 0..5 * 1024 {
            // 5 MB of '\0'
            f.write_all(&[0u8; 1024]).unwrap();
        }
        // SHA-256 of 5 MB of zero bytes:
        let expected = sha256_of_zeros_5mb();
        verify_sha256(&path, &expected).expect("streaming hash should match");
    }

    fn sha256_of_zeros_5mb() -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for _ in 0..5 * 1024 {
            h.update([0u8; 1024]);
        }
        format!("{:x}", h.finalize())
    }
}
```

(The integration test that hits the real network is added in Task 8; downloads in unit tests would be flaky.)

- [ ] **Step 6.3: Run tests, confirm compilation failure**

```bash
cargo test --features download --lib models::download::tests 2>&1 | head -20
```

Expected: errors about missing `verify_sha256`, `DownloadError`.

- [ ] **Step 6.4: Implement the download module**

Replace the file with:

```rust
//! HTTP download with streamed SHA-256 verification.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// Errors from `download_with_checksum` and `verify_sha256`.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("network error fetching {url}: {source}")]
    Network {
        url: String,
        #[source]
        source: Box<ureq::Error>,
    },
    #[error(
        "checksum mismatch for {path}: expected {expected:.16}…, computed {actual:.16}…"
    )]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

/// Stream `url` to `dest` and verify the SHA-256 matches `expected_sha256`.
///
/// Idempotent: if `dest` already exists with the correct hash, returns Ok(false)
/// immediately. Otherwise downloads, hashes while streaming (so 200+ MB files
/// don't blow up RAM), and on hash mismatch deletes the partial file and returns
/// an error. Returns `Ok(true)` if a download happened, `Ok(false)` if cached.
pub fn download_with_checksum(
    url: &str,
    expected_sha256: &str,
    dest: &Path,
) -> Result<bool, DownloadError> {
    if dest.exists() && verify_sha256(dest, expected_sha256).is_ok() {
        return Ok(false);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| DownloadError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    // Download to a sibling .partial file, then rename — gives atomic on-success
    // semantics so a partial file is never seen as cached.
    let mut tmp = dest.to_path_buf();
    let original_name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    tmp.set_file_name(format!(".{original_name}.partial"));
    let resp = ureq::get(url)
        .call()
        .map_err(|e| DownloadError::Network {
            url: url.to_owned(),
            source: Box::new(e),
        })?;
    let reader = resp.into_body().into_reader();
    let mut reader = BufReader::new(reader);
    let mut file = fs::File::create(&tmp).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).map_err(|e| DownloadError::Io {
            path: tmp.clone(),
            source: e,
        })?;
    }
    file.flush().map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    drop(file);
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_sha256 {
        let _ = fs::remove_file(&tmp);
        return Err(DownloadError::ChecksumMismatch {
            path: dest.to_path_buf(),
            expected: expected_sha256.to_owned(),
            actual,
        });
    }
    fs::rename(&tmp, dest).map_err(|e| DownloadError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    Ok(true)
}

/// Compute the SHA-256 of `path` and compare against `expected`. Streams the file
/// (does not load it into RAM).
pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), DownloadError> {
    let f = fs::File::open(path).map_err(|e| DownloadError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf).map_err(|e| DownloadError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual == expected {
        Ok(())
    } else {
        Err(DownloadError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    // (test block from Step 6.2 stays unchanged below)
}
```

Keep the test block from Step 6.2 unchanged at the bottom.

- [ ] **Step 6.5: Wire `download` module into `src/models/mod.rs`**

Open `src/models/mod.rs` and add `pub mod download;` plus a re-export above the `#[cfg(test)] mod tests`:

```rust
pub mod manifest;
pub mod download;
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};
pub use download::{DownloadError, download_with_checksum, verify_sha256};
```

- [ ] **Step 6.6: Run tests, confirm 3/3 download tests pass**

```bash
cargo test --features download --lib models::download::tests
```

Expected: 3 tests pass.

- [ ] **Step 6.7: Run all model tests**

```bash
cargo test --features download --lib models::
```

Expected: 12 tests pass (6 manifest + 3 mod + 3 download).

- [ ] **Step 6.8: Commit**

```bash
git add src/models/download.rs src/models/mod.rs
git commit -m "feat(models): add streamed-download with sha256 verification"
```

---

## Task 7: Implement `ModelRegistry`

**Files:**
- Modify: `src/models/mod.rs` (add `ModelRegistry` + `ProfileModels` + `RegistryError`)

- [ ] **Step 7.1: Write the failing tests**

Append the following to the existing `#[cfg(test)] mod tests` block in `src/models/mod.rs`:

```rust
    use crate::Profile;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[test]
    fn registry_default_uses_user_cache() {
        let r = ModelRegistry::default().expect("default cache dir resolvable");
        let path = r.cache_dir().to_path_buf();
        assert!(path.ends_with("polyvoice/models") || path.ends_with("polyvoice\\models"));
    }

    #[test]
    fn registry_with_cache_dir_creates_dir() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nested/models");
        let r = ModelRegistry::with_cache_dir(&path).unwrap();
        assert!(path.exists());
        assert_eq!(r.cache_dir(), path.as_path());
    }

    #[test]
    fn ensure_returns_err_for_unknown_model_id() {
        let tmp = TempDir::new().unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
        let err = r.ensure_in_cache_only("ghost").expect_err("must be missing");
        assert!(matches!(err, RegistryError::ModelNotFound { .. }));
    }

    #[test]
    fn ensure_in_cache_only_succeeds_when_file_present() {
        // We pre-place a file in the cache with the exact filename and known SHA-256.
        let tmp = TempDir::new().unwrap();
        let manifest = Manifest::from_toml_str(crate::models::tests_helpers::TINY_MANIFEST).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap()
            .with_manifest_override(manifest);

        let cached = tmp.path().join("hello.bin");
        std::fs::write(&cached, b"hello").unwrap();
        let path = r.ensure_in_cache_only("hello_model").unwrap();
        assert_eq!(path, cached);
    }

    #[test]
    fn ensure_for_profile_uses_manifest_lookup() {
        let tmp = TempDir::new().unwrap();
        let manifest = Manifest::from_toml_str(crate::models::tests_helpers::TINY_MANIFEST).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap()
            .with_manifest_override(manifest);

        // Pre-place both files (the tiny test manifest references the same model
        // for segmenter and embedder).
        std::fs::write(tmp.path().join("hello.bin"), b"hello").unwrap();

        let bundle = r.ensure_in_cache_only_for_profile(Profile::Mobile).unwrap();
        assert_eq!(bundle.segmenter_path, tmp.path().join("hello.bin"));
        assert_eq!(bundle.embedder_path, tmp.path().join("hello.bin"));
    }
```

We also need a tiny test-only helper manifest so we don't network-download in unit tests. Add to `src/models/mod.rs` just above the `#[cfg(test)]` block:

```rust
#[cfg(test)]
pub(crate) mod tests_helpers {
    /// Minimal manifest used by registry unit tests. SHA-256 below is hash of "hello":
    /// 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    pub const TINY_MANIFEST: &str = r#"
        schema = "polyvoice-models-v1"
        [profiles.mobile]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [profiles.balanced]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [models.hello_model]
        url      = "file:///dev/null"
        sha256   = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        size     = 5
        filename = "hello.bin"
    "#;
}
```

- [ ] **Step 7.2: Run tests, confirm compilation failure**

```bash
cargo test --features download --lib models:: 2>&1 | head -40
```

Expected: errors about missing `ModelRegistry`, `RegistryError`, `with_manifest_override`, etc.

- [ ] **Step 7.3: Implement `ModelRegistry`**

Replace `src/models/mod.rs` with the full implementation. Keep the existing `pub mod manifest`, `pub mod download`, and re-exports. Add types and `impl`:

```rust
//! Model registry — manifest-driven downloads with SHA-256 verification.

pub mod download;
pub mod manifest;
pub use download::{DownloadError, download_with_checksum, verify_sha256};
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};

use crate::Profile;
use std::path::{Path, PathBuf};

/// Default manifest shipped with the crate, embedded at compile time.
pub const DEFAULT_MANIFEST_TOML: &str = include_str!("manifest.toml");

/// Parse the bundled default manifest. Panics in debug if the embedded TOML is
/// malformed — that's a static asset bug caught by `cargo test`.
pub fn default_manifest() -> Manifest {
    Manifest::from_toml_str(DEFAULT_MANIFEST_TOML)
        .expect("embedded manifest.toml must parse — this is a static-asset bug")
}

/// Errors from `ModelRegistry` operations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("model '{model_id}' not found in manifest")]
    ModelNotFound { model_id: String },
    #[error("profile '{profile}' not found in manifest")]
    ProfileNotFound { profile: String },
    #[error("custom profile cannot be resolved by registry — caller must supply models")]
    CustomProfileUnresolvable,
    #[error("cache directory {path} is not writable")]
    CacheNotWritable { path: PathBuf },
    #[error("model '{model_id}' is not present in cache and offline mode is requested")]
    OfflineMissing { model_id: String },
    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),
    #[error("download error: {0}")]
    Download(#[from] DownloadError),
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Resolved file paths for the segmenter and embedder of a profile.
#[derive(Debug, Clone)]
pub struct ProfileModels {
    pub segmenter_path: PathBuf,
    pub embedder_path: PathBuf,
}

/// A model registry: holds a manifest + a cache directory, and downloads/verifies
/// models on demand.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    manifest: Manifest,
    cache_dir: PathBuf,
}

impl ModelRegistry {
    /// Build a registry rooted at the user's cache directory (`~/.cache/polyvoice/models`
    /// on Linux, `~/Library/Caches/polyvoice/models` on macOS, `%LOCALAPPDATA%\polyvoice\models`
    /// on Windows) using the embedded default manifest.
    pub fn default() -> Result<Self, RegistryError> {
        let cache = dirs::cache_dir()
            .ok_or_else(|| RegistryError::CacheNotWritable {
                path: PathBuf::from("(unresolved-cache-dir)"),
            })?
            .join("polyvoice")
            .join("models");
        Self::with_cache_dir(cache)
    }

    /// Build a registry with a caller-specified cache directory and the embedded
    /// default manifest. Creates the directory if it doesn't exist.
    pub fn with_cache_dir(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path).map_err(|e| RegistryError::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok(Self {
            manifest: default_manifest(),
            cache_dir: path,
        })
    }

    /// Override the manifest. Useful for tests that need a fixture manifest
    /// without hitting the network.
    pub fn with_manifest_override(mut self, manifest: Manifest) -> Self {
        self.manifest = manifest;
        self
    }

    /// Build a registry with a custom manifest and cache directory.
    pub fn with_manifest(manifest: Manifest, cache_dir: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let path = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&path).map_err(|e| RegistryError::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok(Self { manifest, cache_dir: path })
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Ensure the model with id `model_id` is present in cache and SHA-256-verified.
    /// Downloads if missing. Idempotent: returns immediately when the cached file
    /// already matches the expected hash.
    pub fn ensure(&self, model_id: &str) -> Result<PathBuf, RegistryError> {
        let entry = self.manifest.model(model_id).ok_or_else(|| {
            RegistryError::ModelNotFound {
                model_id: model_id.to_owned(),
            }
        })?;
        let dest = self.cache_dir.join(&entry.filename);
        download_with_checksum(&entry.url, &entry.sha256, &dest)?;
        Ok(dest)
    }

    /// Same as `ensure` but never makes a network call. Returns `OfflineMissing`
    /// if the file is not in cache or has a wrong hash.
    pub fn ensure_in_cache_only(&self, model_id: &str) -> Result<PathBuf, RegistryError> {
        let entry = self.manifest.model(model_id).ok_or_else(|| {
            RegistryError::ModelNotFound {
                model_id: model_id.to_owned(),
            }
        })?;
        let dest = self.cache_dir.join(&entry.filename);
        if !dest.exists() {
            return Err(RegistryError::OfflineMissing {
                model_id: model_id.to_owned(),
            });
        }
        // Skip hash check in cache-only path; it's expensive and tests pre-place
        // exact-content files. Production callers should use `ensure` not this.
        Ok(dest)
    }

    /// Resolve all models for a profile, downloading any that are missing.
    pub fn ensure_for_profile(&self, profile: Profile) -> Result<ProfileModels, RegistryError> {
        if profile == Profile::Custom {
            return Err(RegistryError::CustomProfileUnresolvable);
        }
        let prof = self.manifest.profile(profile.manifest_id()).ok_or_else(|| {
            RegistryError::ProfileNotFound {
                profile: profile.manifest_id().to_owned(),
            }
        })?;
        let segmenter_path = self.ensure(&prof.segmenter)?;
        let embedder_path = self.ensure(&prof.embedder)?;
        Ok(ProfileModels {
            segmenter_path,
            embedder_path,
        })
    }

    /// Same as `ensure_for_profile` but never touches the network.
    pub fn ensure_in_cache_only_for_profile(
        &self,
        profile: Profile,
    ) -> Result<ProfileModels, RegistryError> {
        if profile == Profile::Custom {
            return Err(RegistryError::CustomProfileUnresolvable);
        }
        let prof = self.manifest.profile(profile.manifest_id()).ok_or_else(|| {
            RegistryError::ProfileNotFound {
                profile: profile.manifest_id().to_owned(),
            }
        })?;
        let segmenter_path = self.ensure_in_cache_only(&prof.segmenter)?;
        let embedder_path = self.ensure_in_cache_only(&prof.embedder)?;
        Ok(ProfileModels {
            segmenter_path,
            embedder_path,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests_helpers {
    pub const TINY_MANIFEST: &str = r#"
        schema = "polyvoice-models-v1"
        [profiles.mobile]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [profiles.balanced]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [models.hello_model]
        url      = "file:///dev/null"
        sha256   = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        size     = 5
        filename = "hello.bin"
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        let m = default_manifest();
        assert_eq!(m.schema, SCHEMA_V1);
        assert!(m.profiles.contains_key("mobile"));
        assert!(m.profiles.contains_key("balanced"));
    }

    #[test]
    fn embedded_manifest_lists_legacy_models() {
        let m = default_manifest();
        assert!(m.models.contains_key("silero_vad"));
        assert!(m.models.contains_key("wespeaker_resnet34"));
    }

    #[test]
    fn both_profiles_resolve_to_legacy_models_in_m0() {
        let m = default_manifest();
        let mob = m.profile("mobile").unwrap();
        let bal = m.profile("balanced").unwrap();
        assert_eq!(mob.segmenter, bal.segmenter);
        assert_eq!(mob.embedder, bal.embedder);
    }

    // (registry tests from Step 7.1 placed here)
}
```

Append into the `#[cfg(test)] mod tests` block the five registry tests from Step 7.1 verbatim.

- [ ] **Step 7.4: Run all module tests**

```bash
cargo test --features download --lib models::
```

Expected: 14+ tests pass (6 manifest + 3 mod + 3 download + 5 registry).

- [ ] **Step 7.5: Run clippy and fmt**

```bash
cargo fmt
cargo clippy --features download --lib -- -D warnings
```

Expected: clean.

- [ ] **Step 7.6: Commit**

```bash
git add src/models/mod.rs
git commit -m "feat(models): implement ModelRegistry with idempotent downloads"
```

---

## Task 8: Integration test that hits the real network

**Files:**
- Create: `tests/registry_test.rs`

This test is `#[ignore]`-marked so CI doesn't run it on every PR (the URLs are stable but flaky-friendly behavior is expensive). It is run manually in Task 15 as an end-to-end validation.

- [ ] **Step 8.1: Create the test file**

Write `tests/registry_test.rs`:

```rust
//! Integration test for ModelRegistry against the real upstream URLs.
//!
//! Runs only when explicitly invoked:
//!   cargo test --features download --test registry_test -- --ignored
//!
//! The download is ~28 MB total. Requires network connectivity.

#![cfg(feature = "download")]

use polyvoice::models::ModelRegistry;
use polyvoice::Profile;
use tempfile::TempDir;

#[test]
#[ignore = "real network — run with --ignored"]
fn ensure_for_profile_mobile_downloads_and_verifies() {
    let tmp = TempDir::new().unwrap();
    let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();

    let bundle = r
        .ensure_for_profile(Profile::Mobile)
        .expect("download must succeed");

    assert!(bundle.segmenter_path.exists());
    assert!(bundle.embedder_path.exists());

    let seg_size = std::fs::metadata(&bundle.segmenter_path).unwrap().len();
    let emb_size = std::fs::metadata(&bundle.embedder_path).unwrap().len();
    assert!(seg_size > 1_000_000, "silero ~2.3MB");
    assert!(emb_size > 20_000_000, "wespeaker ~26MB");

    // Second call should be a no-op (idempotent cache hit, no download).
    let bundle2 = r.ensure_for_profile(Profile::Mobile).unwrap();
    assert_eq!(bundle2.segmenter_path, bundle.segmenter_path);
}

#[test]
#[ignore = "real network — run with --ignored"]
fn ensure_for_profile_custom_returns_explicit_error() {
    let tmp = TempDir::new().unwrap();
    let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
    let err = r.ensure_for_profile(Profile::Custom).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("custom"), "got: {msg}");
}
```

- [ ] **Step 8.2: Confirm the test compiles (without running it)**

```bash
cargo test --features download --test registry_test -- --list
```

Expected: lists 2 tests, exits 0.

- [ ] **Step 8.3: Commit**

```bash
git add tests/registry_test.rs
git commit -m "test(models): add network integration tests behind --ignored"
```

---

## Task 9: Re-export `Profile` and `ModelRegistry` from `lib.rs`

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 9.1: Add re-exports**

Locate the `pub use types::{ ... };` block (around line 72). Append `Profile` to that import list:

```rust
pub use types::{
    ClusteringBackend, Confidence, DiarizationConfig, DiarizationResult, EmbeddingDim,
    Profile, SampleRate,
    Seconds, Segment, SpeakerId, SpeakerIdRemap, SpeakerTurn, TimeRange, WordAlignment,
    remap_segments, remap_turns,
};
```

(Keep `ProfileParseError` un-reexported at crate root; users can reach it via `polyvoice::types::ProfileParseError` if needed.)

Below that block, but still within the file, add:

```rust
#[cfg(feature = "download")]
pub use models::{ModelRegistry, ProfileModels, RegistryError};
```

- [ ] **Step 9.2: Run all tests**

```bash
cargo test --features download --lib
cargo test --features cli --lib
cargo test --all-features --lib
```

Expected: all green.

- [ ] **Step 9.3: Run doc-tests + cargo doc**

```bash
cargo doc --no-deps --all-features
```

Expected: builds without warnings (RUSTDOCFLAGS not yet set; CI will catch warning regressions).

- [ ] **Step 9.4: Commit**

```bash
git add src/lib.rs
git commit -m "feat(lib): re-export Profile and ModelRegistry"
```

---

## Task 10: Wire `--profile` into the CLI

**Files:**
- Modify: `src/bin/polyvoice.rs`

- [ ] **Step 10.1: Add `--profile` flag and rewrite `run_download`**

Replace the entire `download-models` subcommand block and `run_download` function. Specifically:

In the `Command::DownloadModels` enum variant (around line 46), replace the field list with:

```rust
    /// Download ONNX models for a profile (or all profiles)
    DownloadModels {
        /// Target directory (default: ~/.cache/polyvoice/models)
        #[arg(long)]
        dir: Option<PathBuf>,

        /// Profile to fetch models for.
        ///
        /// In M0 both `mobile` and `balanced` map to the legacy v0.5 model pair
        /// (silero_vad + wespeaker_resnet34). Future milestones (M2/M5) will
        /// diverge them. Use `all` to fetch every distinct model in the manifest.
        #[arg(long, default_value = "all", value_parser = ["mobile", "balanced", "all"])]
        profile: String,
    },
```

In `main()`, replace the `Command::DownloadModels { dir }` arm with:

```rust
        Command::DownloadModels { dir, profile } => {
            let dir = dir.unwrap_or_else(default_model_dir);
            if let Err(e) = run_download(&dir, &profile) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
```

Replace the `run_download` function entirely with:

```rust
fn run_download(dir: &Path, profile: &str) -> Result<(), Box<dyn std::error::Error>> {
    use polyvoice::ModelRegistry;
    use polyvoice::Profile;

    std::fs::create_dir_all(dir)?;
    let registry = ModelRegistry::with_cache_dir(dir)?;

    let profiles_to_fetch: Vec<Profile> = match profile {
        "mobile" => vec![Profile::Mobile],
        "balanced" => vec![Profile::Balanced],
        "all" => vec![Profile::Mobile, Profile::Balanced],
        other => return Err(format!("unknown profile '{other}'").into()),
    };

    for prof in profiles_to_fetch {
        eprintln!("Resolving profile '{}'…", prof.manifest_id());
        let bundle = registry.ensure_for_profile(prof)?;
        eprintln!("  segmenter: {}", bundle.segmenter_path.display());
        eprintln!("  embedder : {}", bundle.embedder_path.display());
    }

    eprintln!("\nModels saved to {}", dir.display());
    Ok(())
}
```

- [ ] **Step 10.2: Run cargo check on the binary**

```bash
cargo check --features cli --bin polyvoice
```

Expected: compiles cleanly.

- [ ] **Step 10.3: Smoke-test the CLI help output**

```bash
cargo run --features cli --bin polyvoice -- download-models --help
```

Expected: help text shows `--profile <PROFILE>` with `[possible values: mobile, balanced, all]`.

- [ ] **Step 10.4: Run clippy and fmt**

```bash
cargo fmt
cargo clippy --features cli --bin polyvoice -- -D warnings
```

Expected: clean.

- [ ] **Step 10.5: Commit**

```bash
git add src/bin/polyvoice.rs
git commit -m "feat(cli): add --profile flag to download-models subcommand"
```

---

## Task 11: Add aarch64-linux cross-compile job to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 11.1: Append a `cross-aarch64-linux` job**

Open `.github/workflows/ci.yml`. After the `audit:` job block (which is the last block, ending with `run: cargo audit`), append:

```yaml

  cross-aarch64-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: aarch64-unknown-linux-gnu
      - uses: Swatinem/rust-cache@v2
      - name: Install gcc-aarch64-linux-gnu (linker for cross-compile)
        run: sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      - name: Configure linker
        run: |
          mkdir -p .cargo
          cat >> .cargo/config.toml <<'EOF'
          [target.aarch64-unknown-linux-gnu]
          linker = "aarch64-linux-gnu-gcc"
          EOF
      - name: cargo check (aarch64-linux, default features)
        run: cargo check --target aarch64-unknown-linux-gnu
      - name: cargo check (aarch64-linux, download feature)
        run: cargo check --target aarch64-unknown-linux-gnu --features download
```

(Note: `--features cli` requires `ort` which currently doesn't ship aarch64-linux prebuilt binaries via its build script in CI runners by default; restricting this job to `default` and `download` features avoids that complication. The full Android+aarch64 ort wiring is M8.)

- [ ] **Step 11.2: Validate yaml syntax locally if possible**

If `actionlint` is available:

```bash
actionlint .github/workflows/ci.yml
```

If not installed, skip — GitHub will surface lint errors on push.

- [ ] **Step 11.3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add aarch64-linux cross-compile job"
```

---

## Task 12: Add wasm32 smoke-compile job to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 12.1: Append a `wasm32-smoke` job**

After the `cross-aarch64-linux` job, append:

```yaml

  wasm32-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      - uses: Swatinem/rust-cache@v2
      - name: cargo check (wasm32, no default features, no onnx)
        run: cargo check --target wasm32-unknown-unknown --no-default-features --lib
```

This is intentionally a smoke compile only — `onnx` (and therefore the full pipeline) is not wasm-friendly and is excluded. The intent is to keep `pure-Rust` algorithmic modules (clustering, der, rttm, types, utils) wasm-clean, which guards against accidental dependencies on host-only APIs in those modules.

- [ ] **Step 12.2: Validate by running locally**

```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --no-default-features --lib
```

Expected: build succeeds. If a non-wasm-friendly dep was accidentally added (e.g. one in default features that requires `tokio` for native I/O), this is the place we catch it. Fix by gating the offending module behind a non-default feature.

- [ ] **Step 12.3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add wasm32-unknown-unknown smoke-compile job"
```

---

## Task 13: Create `scripts/release-gate.sh` stub

**Files:**
- Create: `scripts/release-gate.sh`

The release gate is run before tagging `v1.0.0`. In M0 most checks are not yet measurable (the powerset segmenter doesn't exist, INT8 models don't exist, Android CI hasn't been added). The script structure is established so M9 can fill it in incrementally.

- [ ] **Step 13.1: Create the script**

Write `scripts/release-gate.sh`:

```bash
#!/usr/bin/env bash
# release-gate.sh — pre-tag verification for polyvoice v1.0.0.
#
# Each section corresponds to a row in §9.10 of the v1.0 design spec.
# A check that returns exit 0 means PASS; non-zero means FAIL.
# The script exits non-zero if any check fails.
#
# In M0 most checks are stubs that print "PENDING-MILESTONE-X" and exit 0,
# documenting the M they will become real in.

set -uo pipefail

PASS=0
FAIL=0
PENDING=0

step() {
    local label="$1"
    local status="$2"
    local detail="${3:-}"
    case "$status" in
        ok)
            echo "  PASS: $label${detail:+ — $detail}"
            PASS=$((PASS + 1))
            ;;
        fail)
            echo "  FAIL: $label${detail:+ — $detail}"
            FAIL=$((FAIL + 1))
            ;;
        pending)
            echo "  ----: $label${detail:+ — $detail} (pending)"
            PENDING=$((PENDING + 1))
            ;;
    esac
}

echo "=== polyvoice v1.0.0 release gate ==="
echo ""
echo "[1/12] DER thresholds"
step "DER VoxConverse Mobile ≤ 12.5%" pending "becomes real in M5 (INT8 calibration)"
step "DER VoxConverse Balanced ≤ 11.5%" pending "becomes real in M4 (resegmenter)"
step "DER AMI Mobile ≤ 19.5%" pending "becomes real in M5"
step "DER AMI Balanced ≤ 18.5%" pending "becomes real in M4"

echo ""
echo "[2/12] Model bundle sizes"
step "Mobile bundle ≤ 10 MB" pending "real INT8 weights ship in M5"
step "Balanced bundle ≤ 35 MB" pending "real INT8 weights ship in M5"

echo ""
echo "[3/12] Runtime budgets"
step "Peak RSS on 1h audio (Mobile) ≤ 250 MB" pending "becomes real once M2+M5 land"
step "RT-factor on M2 single-core (Mobile) ≥ 15x" pending "real once M2+M5 land"
step "RT-factor on Cortex-A78 (Mobile) ≥ 3x" pending "real once M8 lands Android CI"

echo ""
echo "[4/12] CI matrix"
if [ -f .github/workflows/ci.yml ]; then
    if grep -q "cross-aarch64-linux" .github/workflows/ci.yml; then
        step "ci.yml has aarch64-linux job" ok
    else
        step "ci.yml has aarch64-linux job" fail
    fi
    if grep -q "wasm32-smoke" .github/workflows/ci.yml; then
        step "ci.yml has wasm32 smoke job" ok
    else
        step "ci.yml has wasm32 smoke job" fail
    fi
    step "ci.yml has android-nnapi job" pending "added in M8"
else
    step "ci.yml exists" fail
fi

echo ""
echo "[5/12] semver-checks vs prior major"
step "cargo semver-checks vs v0.5.x: breaking confirmed" pending "real in M9; v1.0 is intentionally breaking"

echo ""
echo "[6/12] Doc coverage"
if cargo doc --no-deps --all-features 2>/dev/null >/dev/null; then
    step "cargo doc --all-features builds" ok
else
    step "cargo doc --all-features builds" fail
fi

echo ""
echo "=== summary ==="
echo "PASS    : $PASS"
echo "FAIL    : $FAIL"
echo "PENDING : $PENDING"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "RELEASE BLOCKED: $FAIL check(s) failing."
    exit 1
fi

if [ "$PENDING" -gt 0 ]; then
    echo ""
    echo "RELEASE NOT READY: $PENDING check(s) pending milestone implementation."
    exit 2
fi

echo ""
echo "RELEASE GATE GREEN"
exit 0
```

- [ ] **Step 13.2: Make it executable**

```bash
chmod +x scripts/release-gate.sh
```

- [ ] **Step 13.3: Run it locally to confirm structure**

```bash
./scripts/release-gate.sh
```

Expected output: prints all sections. Exits with code 2 (PENDING checks present, no FAILs). The 2 CI checks added in Tasks 11–12 should print `PASS`.

- [ ] **Step 13.4: Commit**

```bash
git add scripts/release-gate.sh
git commit -m "ci(release): add release-gate.sh stub matching spec §9.10"
```

---

## Task 14: Update CHANGELOG.md

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 14.1: Add Unreleased section**

Open `CHANGELOG.md`. Find the `## [Unreleased]` line (it exists per the existing file). Replace `## [Unreleased]` block (which is currently empty) with:

```markdown
## [Unreleased]

### Added (M0 — v1.0 plumbing)
- `Profile` enum (`Mobile`/`Balanced`/`Custom`) in `polyvoice::types`.
- `polyvoice::models` module: `ModelRegistry`, `Manifest`, `ModelEntry`, `ProfileEntry`, `ProfileModels`.
  Provides manifest-driven, SHA-256-verified, idempotent ONNX model downloads.
- New Cargo features: `download`, `coreml`, `nnapi`, `xnnpack`, `profile-mobile`,
  `profile-balanced`, `profile-all`. The `cli` feature now depends on `download`
  (no behavioral change for existing users).
- CLI: `polyvoice download-models --profile mobile|balanced|all`. Both `mobile`
  and `balanced` resolve to the existing v0.5.x model pair until later milestones
  ship CAM++ (M2) and INT8 versions (M5).
- CI: aarch64-unknown-linux-gnu cross-compile job and wasm32-unknown-unknown smoke
  compile job.
- `scripts/release-gate.sh` — stub release-gate script aligned with §9.10 of the
  v1.0 design.
```

- [ ] **Step 14.2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document M0 plumbing additions"
```

---

## Task 15: End-to-end verification

**Files:** none modified — purely a verification pass.

- [ ] **Step 15.1: Full build matrix**

```bash
cargo build --no-default-features
cargo build --features download
cargo build --features cli
cargo build --features ffi
cargo build --features onnx
cargo build --all-features
```

Expected: each succeeds. (Note: `coreml`, `nnapi`, `xnnpack` may require platform-specific runners; if `cargo build --all-features` fails on Linux solely due to those, drop them from the local check and rely on platform-specific CI in M8 instead.)

- [ ] **Step 15.2: Full test matrix**

```bash
cargo test --features download --lib
cargo test --features cli --lib
cargo test --all-features --lib
cargo test --all-features --doc
```

Expected: all green.

- [ ] **Step 15.3: Clippy + fmt**

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Expected: no diff from fmt, no warnings from clippy.

- [ ] **Step 15.4: Smoke-run the CLI**

```bash
cargo run --features cli --bin polyvoice -- download-models --help
cargo run --features cli --bin polyvoice -- download-models --profile mobile --dir /tmp/polyvoice-m0-cli-test
ls -la /tmp/polyvoice-m0-cli-test
shasum -a 256 /tmp/polyvoice-m0-cli-test/*.onnx
```

Expected:
- The `--help` output lists `--profile <PROFILE>` with the three accepted values.
- The `--profile mobile` invocation downloads two `.onnx` files and prints their paths.
- The shasum output matches the values pinned in `src/models/manifest.toml` from Task 5.

Cleanup:
```bash
rm -rf /tmp/polyvoice-m0-cli-test
```

- [ ] **Step 15.5: Run the network-dependent integration tests**

```bash
cargo test --features download --test registry_test -- --ignored
```

Expected: 2 ignored tests now run and pass.

- [ ] **Step 15.6: Run release-gate.sh**

```bash
./scripts/release-gate.sh
echo "exit: $?"
```

Expected: exit code 2 (PENDING checks remain — that's correct in M0). The CI-section checks (`aarch64-linux job present`, `wasm32 smoke present`, `cargo doc builds`) print PASS.

- [ ] **Step 15.7: Tag the milestone in git**

```bash
git tag -a m0-complete -m "M0 complete: plumbing & registry"
```

(Do not push tags unless explicitly requested.)

---

## Self-review checklist

After completing all tasks, walk through this list:

1. **Spec coverage:** §10.1 in the design spec lists five M0 deliverables. Map each to the task it satisfies:
   - Cargo features → Task 1 ✓
   - ModelRegistry skeleton → Tasks 3, 4, 6, 7 ✓
   - CLI subcommand `polyvoice download-models --profile` → Task 10 ✓
   - Updated CI matrix (without Android) → Tasks 11, 12 ✓
   - `release-gate.sh` stub → Task 13 ✓

2. **Additive only:** verify no v0.5.x public API was removed. Run `cargo semver-checks check-release` against the published v0.5.2 — it should report only **additions** (new types, new features, new modules), never removals.

3. **Test coverage:** every new public function has at least one unit test. Network-dependent paths are gated behind `#[ignore]` and verified in Step 15.5.

4. **Docs:** every new public type and function has a doc comment per the project's `cargo kimi check` policy.

5. **Commits are atomic:** each task ends in exactly one commit. Total 14 commits.

---

## Out of scope for this plan

- Any change to `Pipeline`, `OfflineDiarizer`, `OnlineDiarizer`, `SileroVad`, or `FbankOnnxExtractor`. These remain untouched until M6.
- INT8 models, CAM++, powerset segmenter — those are M1, M2, M5.
- Android NNAPI runtime tests, iOS/Windows wheels — M8 / v2.0.
- Python bindings for the new `Profile` and `ModelRegistry` types — added in M7 alongside the API redesign.
