# Model Signing Infrastructure Plan

> **Date:** 2026-05-08  
> **Context:** Security audit finding **SUPPLY-002** — SHA-256 provides integrity but not authenticity for downloaded ONNX models.  
> **Goal:** Design a cryptographic signing layer that proves every model artifact was issued by the polyvoice maintainers, without breaking existing download flows.  
> **Constraint:** Do not implement; research and plan only.

---

## 1. Executive Summary

The polyvoice `ModelRegistry` downloads pretrained ONNX artifacts from GitHub and HuggingFace.  Today each model is verified only by a SHA-256 hash embedded in `manifest.toml`.  SHA-256 guarantees that the bytes received match the bytes expected, but it does **not** guarantee *who* expected those bytes.  If the CI machine that generates the manifest is compromised, an attacker can ship a malicious model whose hash matches the manifest, and every user will silently load it into ONNX Runtime.

This plan evaluates five signing tools and recommends **Minisign** (Ed25519, dead-simple CLI, pure-Rust zero-dependency verifier).  The recommended integration stores a project-level public key inside the binary, stores per-model signatures inline in `manifest.toml`, and verifies them with streaming Minisign verification during download — adding ~200 bytes per model and no extra network round-trips.

---

## 2. Threat Model

### 2.1 Adversaries

| Actor | Capability |
|---|---|
| **Compromised build/CI host** | Can regenerate `manifest.toml` with a malicious model and a matching SHA-256 hash. |
| **Compromised CDN / upstream repo** | Can replace the `.onnx` file at the download URL (HTTPS alone does not stop this if the upstream account is hijacked). |
| **Malicious insider with manifest write access** | Can edit `manifest.toml` directly to point at a back-doored model. |

### 2.2 What we are protecting against

- **Substitution attacks:** An attacker replaces a legitimate model with a malicious one *and* updates the manifest hash to match.
- **Downgrade attacks:** An attacker forces a user to load an older, vulnerable model version.

### 2.3 What we are *not* protecting against

- **Client-side compromise:** If an attacker can patch the polyvoice binary or overwrite the baked-in public key, the signature check is bypassed entirely.
- **Signing-key theft:** If the offline secret key is stolen, the attacker can sign arbitrary models.  Mitigation is key-rotation policy (see §6), not cryptography.
- **Runtime exploitation of ONNX Runtime itself:** A *legitimate* model may still trigger bugs in `ort`.  Signing does not sandbox execution.

### 2.4 Assumptions

- The public key is distributed inside the compiled binary (via `include_str!`).  Users trust the binary they download from crates.io / GitHub Releases.
- Models are immutable release artifacts.  Any update is a new URL + new hash + new signature.

---

## 3. Tool Comparison

| Tool | Key Gen & Rotation | Rust Verification Crate | Signature Size | Infra Compatibility | Complexity | Verdict |
|---|---|---|---|---|---|---|
| **Minisign** | `minisign -G` (one command, password-protected by default) | **`minisign-verify`** — zero dependencies, streaming API, pure Rust | ~200 bytes (text) | **Excellent** — signature fits inline in `manifest.toml`; no extra downloads | Low | ✅ **Recommended** |
| PGP / GnuPG | Complex (web-of-trust, subkeys, expiry, keyservers) | `sequoia-openpgp` or `pgp` — heavy dep tree, large API surface | 1–5 KB (armored) | Poor — armored blocks are bulky and hard to embed cleanly | High | ❌ Overkill |
| Cosign | `cosign generate-key-pair` | `sigstore-go` — not a simple crate; pulls in large protobuf / OCI stack | Large JSON bundle (KBs) | Poor — designed for OCI registries and container images | High | ❌ Overkill for standalone files |
| Sigstore (keyless) | OIDC-based, no long-term keys | `sigstore-go` | Large (certificate + transparency-log proof) | Poor — requires online Rekor/Fulcio access for verification | Very High | ❌ Too complex for embedded offline verification |
| OpenSSL | `openssl genrsa` + X.509 cert management | `openssl` crate or `ring` — feasible but low-level | 1–2 KB | Moderate | High | ❌ Key management burden outweighs benefit |

### 3.1 Why Minisign wins for polyvoice

1. **Minimal footprint:** A Minisign signature is ~200 bytes of ASCII.  For six models that is ~1.2 KB — negligible compared to a 28 MB manifest-free download.
2. **Zero-dependency Rust verification:** `minisign-verify` (v0.2.5) has *no* external dependencies and exposes a `StreamVerifier` that processes files in chunks.  This matches our existing streamed SHA-256 design (64 KB buffers).
3. **No extra network round-trips:** Because the signature is baked into `manifest.toml`, verification happens from bytes we already have.  Cosign / Sigstore would require fetching additional sidecar artifacts or hitting a transparency log.
4. **Easy key rotation:** A new key pair is one CLI flag.  Rotation is a simple re-sign + manifest update + release bump.
5. **Human-verifiable:** A user with the `minisign` CLI (or `rsign2`, the pure-Rust reimplementation) can manually verify any cached `.onnx` file using the public key published in the repo.

---

## 4. Recommended Solution: Minisign

### 4.1 Key Generation Commands

Use the official `minisign` CLI (available via Homebrew, Scoop, chocolatey, or `cargo install rsign2` for a pure-Rust alternative).

```bash
# 1. Generate a project-level key pair.
#    -p  public key file (distribute in repo)
#    -s  secret key file (keep offline / in CI secrets)
#    -W  disable password protection (useful for CI signing bots;
#        omit -W for interactive/offline keys)
minisign -G \
  -p models/signing.pub \
  -s ~/.minisign/polyvoice.key \
  -W

# The base64 public key string is also printed to stdout.
# Example output:
# RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3
```

> **Security note:** The secret key should never be committed.  Add `*.key`, `minisign.key`, and `polyvoice.key` to `.gitignore` immediately.

### 4.2 Signing a Model File

```bash
# Sign a single model (pre-hashed by default; streaming-friendly)
minisign -S \
  -s ~/.minisign/polyvoice.key \
  -m models/cam_pp_int8.onnx \
  -x models/cam_pp_int8.onnx.minisig \
  -t "polyvoice v0.6.0-alpha.3 | cam_pp_int8.onnx"

# The resulting .minisig file is a small text block (~200 bytes).
# It can be viewed with cat:
cat models/cam_pp_int8.onnx.minisig
```

### 4.3 Rust Verification Snippet

Below is the exact pattern we would embed in `src/models/download.rs`.  It uses `minisign-verify` v0.2.5 and streams the file in 64 KB chunks — identical to the current SHA-256 loop so that a 200 MB model never lands in RAM.

```rust
use minisign_verify::{PublicKey, Signature, StreamVerifier};
use std::io::{self, BufReader, Read};
use std::path::Path;

/// The base64 public key baked into the binary at compile time.
const SIGNING_PUBKEY_BASE64: &str = include_str!("signing.pub.base64");

/// Errors from the verification step.
#[derive(Debug, thiserror::Error)]
pub enum SignatureError {
    #[error("invalid public key: {0}")]
    BadPublicKey(String),
    #[error("invalid signature text: {0}")]
    BadSignature(String),
    #[error("signature verification failed: {0}")]
    VerificationFailed(String),
}

/// Verify `path` against a Minisign signature string (the raw `.minisig` text).
///
/// Streams the file in 64 KB chunks; does not load the whole model into memory.
pub fn verify_minisign(path: &Path, sig_text: &str) -> Result<(), SignatureError> {
    let public_key = PublicKey::from_base64(SIGNING_PUBKEY_BASE64)
        .map_err(|e| SignatureError::BadPublicKey(format!("{e:?}")))?;

    let signature = Signature::decode(sig_text)
        .map_err(|e| SignatureError::BadSignature(format!("{e:?}")))?;

    let mut verifier = public_key
        .verify_stream(&signature)
        .map_err(|e| SignatureError::VerificationFailed(format!("{e:?}")))?;

    let file = std::fs::File::open(path).map_err(|e| {
        SignatureError::VerificationFailed(format!("io open: {e}"))
    })?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 64 * 1024];

    loop {
        let n = reader.read(&mut buf).map_err(|e| {
            SignatureError::VerificationFailed(format!("io read: {e}"))
        })?;
        if n == 0 {
            break;
        }
        verifier.update(&buf[..n]);
    }

    verifier
        .finalize()
        .map_err(|e| SignatureError::VerificationFailed(format!("{e:?}")))?;

    Ok(())
}
```

**Cargo dependency to add:**

```toml
[dependencies]
minisign-verify = "0.2.5"
```

> `minisign-verify` is a **zero-dependency** crate (verified by `cargo tree`).  It pulls in no C libraries, no OpenSSL, no `std` networking — ideal for cross-compilation targets such as `wasm32-unknown-unknown` and `aarch64-linux-android`.

---

## 5. Integration Design

### 5.1 Public Key Storage

| Location | Purpose |
|---|---|
| `models/signing.pub` | Human-readable reference; the full Minisign public key file (comment + base64). Committed to git. |
| `src/models/signing.pub.base64` | **Compile-time asset** — contains *only* the base64 string (no comment).  Embedded via `include_str!("signing.pub.base64")`.  This avoids parsing logic at runtime and keeps the binary self-contained. |

Why two files?  `minisign-verify` only accepts the raw base64 string via `PublicKey::from_base64`.  Keeping the full `.pub` file in `models/` lets users verify models manually with the CLI, while the stripped `.base64` file is the minimal compile-time input.

### 5.2 Signature Storage

Minisign signatures are small ASCII text files.  Two storage strategies were considered:

| Strategy | Pros | Cons |
|---|---|---|
| **Sidecar `.minisig` files** (download alongside `.onnx`) | Standard Minisign workflow | Extra HTTP request per model; CDN may not serve them; complicates `file://` test URLs |
| **Inline in `manifest.toml`** | No extra download; single source of truth; easy to audit diffs | Slightly larger manifest (~1.2 KB for six models) |

**Decision:** Inline in `manifest.toml` using a TOML multiline literal string (`'''`).  The signature text is preserved exactly (newlines are literal), and `Signature::decode` can consume it directly.

Example manifest entry after integration:

```toml
[models.cam_pp_int8]
url       = "https://github.com/ekhodzitsky/polyvoice/releases/download/v0.6.0-alpha.2/cam_pp_int8.onnx"
sha256    = "cca48a4b36c1b46e48432b1eb1461dd69f9cf113cf506f3f660de808c93b9a85"
size      = 8803007
filename  = "cam_pp_int8.onnx"
calibration = "voxconverse_dev_500_samples_seed_42"
signature = '''
untrusted comment: signature from polyvoice release key
RUQf6LRCGA9i559r3g7V1qNyJDApGip8MfqcadIgT9CuhV3EMhHoN1mGTkUidF/z7SrlQgXdy8ofjb7bNJJylDOocrCo8KLzZwo=
trusted comment: polyvoice v0.6.0-alpha.3 | cam_pp_int8.onnx
wLMDjy9FLAuxZ3q4NlEvkgtyhrr0gtTu6KC4KBJdITbbOeAi1zBIYo0v4iTgt8jJpIidRJnp94ABQkJAgAooBQ==
'''
```

### 5.3 Manifest Schema Update

`src/models/manifest.rs` gains one optional field:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub url: String,
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
    pub filename: String,
    #[serde(default)]
    pub calibration: Option<String>,
    /// Minisign signature (raw .minisig text) — optional during transition period.
    #[serde(default)]
    pub signature: Option<String>,
}
```

Because `serde` ignores unknown fields by default and `#[serde(default)]` supplies `None` when the key is absent, this change is **backwards compatible**:
- Old binaries parsing a new manifest ignore `signature`.
- New binaries parsing an old manifest see `None` and fall back to SHA-256 only.

Validation (optional but recommended): `Manifest::from_toml_str` can attempt `Signature::decode` on any present signature.  If decoding fails, the manifest is rejected at parse time, catching typos before any network activity.

### 5.4 Download & Verification Flow

The current `download_with_checksum` in `src/models/download.rs` streams the response to disk while computing SHA-256.  The updated flow adds **signature verification in the same streaming loop**, avoiding a second disk read.

```text
1. Check cache hit (SHA-256 matches existing file)
   ├── If hit → verify signature on cached file (if present) → return Ok(false)
   └── If miss → continue
2. Open temp `.partial` file
3. Initialize SHA-256 hasher
4. If manifest has signature:
      a. Parse public key (constant)
      b. Parse signature text from manifest
      c. Create StreamVerifier
5. Stream HTTP response in 64 KB chunks:
      a. Write chunk to temp file
      b. hasher.update(chunk)
      c. if verifier exists → verifier.update(chunk)
6. Finalize SHA-256:
      a. If mismatch → delete temp, return ChecksumMismatch
7. Finalize signature:
      a. If manifest had signature and finalize fails → delete temp, return SignatureInvalid
8. Atomic rename temp → dest
9. Return Ok(true)
```

**API contract:**

```rust
/// Existing function preserved for backwards compatibility.
pub fn download_with_checksum(
    url: &str,
    expected_sha256: &str,
    dest: &Path,
) -> Result<bool, DownloadError> {
    download_with_checksum_and_signature(url, expected_sha256, None, dest)
}

/// New function with optional Minisign signature.
pub fn download_with_checksum_and_signature(
    url: &str,
    expected_sha256: &str,
    signature: Option<&str>,
    dest: &Path,
) -> Result<bool, DownloadError> {
    // ... implementation ...
}
```

`ModelRegistry::ensure` is updated to pass `entry.signature.as_deref()` into the new function.

---

## 6. Key Generation & Rotation Policy

### 6.1 Initial key creation

1. A project maintainer generates the key pair on an air-gapped machine or a secure CI secret manager.
2. The public key (`models/signing.pub`) is committed to git.
3. The base64 string is extracted into `src/models/signing.pub.base64` and committed.
4. The secret key is uploaded to the project's **GitHub Actions secrets** as `MINISIGN_SECRET_KEY` (or equivalent for other CI platforms).

### 6.2 Rotation schedule

| Trigger | Action |
|---|---|
| **Time-based** | Rotate every 12 months, aligned with a minor release. |
| **Compromise** | Immediate rotation; publish a security advisory; revoke old public key trust in the next release. |
| **Maintainer change** | Rotate when the sole key holder leaves the project. |

### 6.3 Rotation procedure

1. Generate new key pair (`minisign -G -f ...`).
2. Re-sign **all** model files in `manifest.toml`.
3. Update `src/models/signing.pub.base64` with the new public key.
4. Update `models/signing.pub` for human reference.
5. Update `manifest.toml` with new signatures.
6. Bump manifest schema or add a `pubkey_id` field (optional) to distinguish key epochs.
7. Tag release; document rotation in `CHANGELOG.md` and `docs/security/`.

### 6.4 Grace period (future enhancement)

To avoid breaking users who lag behind releases, the binary could eventually support a **list of trusted public keys** (current + previous).  This is not required for the initial implementation but should be architected with a slice of `PublicKey` rather than a single constant, so the extension is a one-line change later.

---

## 7. Integration Steps (Implementation Roadmap)

| Step | File(s) | Work |
|---|---|---|
| 1 | `Cargo.toml` | Add `minisign-verify = "0.2.5"` under the `download` feature dependency set. |
| 2 | `src/models/signing.pub.base64`, `models/signing.pub` | Add public key assets. |
| 3 | `src/models/manifest.rs` | Add `signature: Option<String>` to `ModelEntry`; add optional `Signature::decode` validation in `from_toml_str`. |
| 4 | `src/models/download.rs` | Add `SignatureInvalid` to `DownloadError`; implement `download_with_checksum_and_signature`; stream both SHA-256 and Minisign in the same 64 KB loop. |
| 5 | `src/models/mod.rs` | Update `ModelRegistry::ensure` to pass `entry.signature.as_deref()` into the downloader. |
| 6 | `src/models/manifest.toml` | Add `signature = '''…'''` entries for every model (populated by the release script in step 8). |
| 7 | `tests/` | Add unit tests for:<br>• valid signature → pass<br>• tampered signature → `SignatureInvalid`<br>• missing signature → fallback to SHA-256 (no panic) |
| 8 | `scripts/sign-models.sh` (new) | Automated release script that:<br>• downloads/signs each model with the CI secret key<br>• injects the `.minisig` text into `manifest.toml`<br>• runs `cargo test --test m5_manifest_smoke_test` to validate |
| 9 | `.github/workflows/ci.yml` | Add a job that runs `scripts/sign-models.sh --dry-run` on PRs to ensure manifest signatures are well-formed. |
| 10 | `.gitignore` | Add `*.key`, `minisign.key`, `polyvoice.key` to prevent secret key leaks. |
| 11 | `docs/security/` | Publish the public key fingerprint and rotation policy for downstream auditors. |

---

## 8. Backwards Compatibility

### 8.1 Manifest format

- `signature` is an **optional** field.  `serde` ignores unknown fields and defaults missing options to `None`.
- Old polyvoice binaries (≤ v0.6.0-alpha.3) parsing a newer manifest will simply ignore the field.
- New binaries parsing an old manifest will see `signature: None` and fall back to SHA-256 only (with an optional `tracing::warn!`).

### 8.2 Public API

- `download_with_checksum` is **preserved as a thin wrapper** calling the new `download_with_checksum_and_signature(url, sha256, None, dest)`.  No downstream crate that calls the free function will break.
- `ModelRegistry::ensure` gains no new public methods; the signature is sourced transparently from the manifest.

### 8.3 Transition to mandatory signatures

- **Phase 1 (v0.6.x):** Signature is optional; custom manifests without signatures continue to work.
- **Phase 2 (v1.0):** Bump manifest schema to `polyvoice-models-v2` and make `signature` required for official model entries.  Custom / third-party manifests can still omit it by using schema v1, which the parser continues to accept.

---

## 9. Estimated Effort

| Task | Time |
|---|---|
| Add dependency + public key assets | 30 min |
| Manifest schema + parser validation | 30 min |
| Download refactor + streaming Minisign verification | 2–3 h |
| Unit & integration tests (including failure modes) | 1–2 h |
| Release script (`sign-models.sh`) | 1–2 h |
| CI update + security docs | 1 h |
| **Total** | **~1 developer day** |

Risk level: **Low**.  `minisign-verify` is a mature, zero-dependency crate.  The change is additive and does not alter existing SHA-256 behaviour when signatures are absent.

---

## 10. Open Questions & Future Work

1. **Multi-key support:** Should the binary accept a *list* of public keys so that an old release can still verify models signed with a rotated key?  (Recommended for v1.0.)
2. **Manifest-level signing:** In addition to per-model signatures, should we sign `manifest.toml` itself to prevent tampering with URLs or profile mappings?  (Lower priority — tampering with the manifest requires commit access, which is already protected by GitHub branch protection.)
3. **Sigstore / keyless signing:** For CI transparency, we could *also* publish a Sigstore bundle as a sidecar for users who want Fulcio/Rekor attestation.  This is complementary, not a replacement for Minisign offline verification.
4. **WASM target:** `minisign-verify` is pure Rust and should compile to `wasm32-unknown-unknown`.  We should verify this in the existing `wasm32-smoke` CI job.

---

## Appendix A: Crates.io Search Results

```text
$ cargo search minisign
minisign        = "0.9.1"   # Full sign/verify library (pure Rust)
minisign-verify = "0.2.5"   # Zero-dependency verifier only (recommended)
rsign2          = "0.6.6"   # CLI tool in Rust (alternative to C minisign)
```

For polyvoice we only need **verification** at runtime, so `minisign-verify` is the smallest possible dependency.  Signing happens in CI or on a maintainer workstation using the `minisign` or `rsign2` CLI.
