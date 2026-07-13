# ONNX Runtime native binary: provenance and trust model

Status: current as of ort 2.0.0-rc.12 (verified 2026-07-13 against the
`ort-sys` sources vendored by that exact crates.io release). Supersedes the
"unpinned native binary" concern in `audit-2026-05-08.md` for this ort
version.

## What `download-binaries` actually does in rc.12

The earlier audit assumed the feature fetches an **unpinned** native ONNX
Runtime at build time. For 2.0.0-rc.12 that is not the case — the fetch is
hash-pinned end to end:

- `ort-sys/build/download/dist.txt` is a static table compiled into the build
  script: `(feature set, target) → (https URL, SHA-256)`. It ships inside the
  `ort-sys` crates.io package, which itself is pinned by our committed
  `Cargo.lock` (crate checksum verified by cargo).
- The downloaded archive is verified **while streaming**
  (`build/download/verify.rs`, `VerifyReader`) and the build errors out on any
  mismatch (`build/main.rs`: "hash of the file downloaded ... does not
  match"). The extract path treats the file as untrusted until the hash
  matches.
- The verified binary is cached under `<cache>/dfbin/<target>/<sha256>` — the
  directory name IS the pin.

Pins for our release matrix (ONNX Runtime `ms@1.24.2`, from `dist.txt`):

| target | sha256 |
|---|---|
| `aarch64-apple-darwin` | `612739f75438dc0a075461e1fb454226b4a1eb175e60a7271ba966bbbb972cd4` |
| `x86_64-unknown-linux-gnu` (default set) | see `dist.txt` row for the non-`cu*` feature set |
| `x86_64-pc-windows-msvc` (default set) | see `dist.txt` row for the non-`cu*` feature set |

Chain of trust: committed `Cargo.lock` → `ort-sys` crate checksum
(crates.io) → embedded `dist.txt` SHA-256 → native binary. A tampered CDN or
proxy cannot substitute a binary without failing the build.

## Residual trust assumptions

1. **pyke as builder**: the binaries are pyke's builds of Microsoft's ONNX
   Runtime; we trust their build process (same category of trust as any
   prebuilt toolchain component). Building ORT from source per release would
   remove this and is not currently justified.
2. **Availability**: `cdn.pyke.io` must be reachable on a cold build. CI now
   caches the verified binary (`ort.pyke.io` cache directory, keyed by
   `Cargo.lock`) in the release workflow, so publishes do not depend on the
   CDN being up and any cold fetch is visible as a cache miss in the log.
3. **Release-candidate track**: ort is still `2.0.0-rc.12`. Graduating to
   stable `ort 2.0` is already a hard precondition of the `1.0.0` milestone
   (see the versioning policy); until then we track rc bumps deliberately —
   each bump changes `dist.txt` and therefore re-pins the native binary,
   reviewed like any dependency update.

## What to re-check on any ort upgrade

- `dist.txt` still pins our three release targets with SHA-256 over https.
- The build script still hard-fails on hash mismatch (no downgrade to a
  warning).
- `scripts/check-ort-version.sh` still holds polyvoice and polyvoice-asr to
  one ort version (single native runtime in every artifact).
