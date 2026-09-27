# Release artifact qualification

The tag release workflow calls `release-artifacts.yml`. Publication to crates.io,
GitHub Releases and PyPI depends on successful artifact consumer checks. A
source-tree test or successful compilation alone does not qualify a release.

## Matrix

| Surface | Linux x86_64 | Linux ARM64 | macOS ARM64 | Windows x86_64 |
|---------|--------------|-------------|-------------|----------------|
| CLI binary | Ubuntu 24.04 | Ubuntu 24.04 | macOS 14 | Windows Server 2022 / MSVC |
| C library + header archive | Same host | Same host | Same host | Same host; includes import library |
| Python wheel | CPython 3.12 | CPython 3.12 | CPython 3.12 | CPython 3.12 |
| Packaged Rust consumers | BYO, BYO+VBx, native, local | Same | Same | Same |

These are the tested hosts, not a promise of compatibility with every older OS.
Linux is GNU/glibc, with Rust kernels and no system OpenBLAS. Wheels must pass
maturin's PyPI tag/compliance check; the filename records the actual platform
and CPython ABI. No abi3 or blanket Python >=3.9 wheel promise is made by this
matrix. Source builds on other Python versions are separate from wheel support.
macOS Intel, Linux musl, Windows ARM64 and full native wasm are not qualified.

The manual Python Wheels workflow also runs installed-wheel inference before
uploading its artifacts. It does not publish. Its manylinux build environment
may produce a different tag; its report identifies the exact tested wheel.

## What the gate executes

`scripts/smoke-release-artifacts.py` uses only Python's standard library plus
the platform C compiler, Cargo and native loader inspection tools. It:

1. Copies the checked-in 26-second `fuzfh.wav` real-speech fixture into a fresh
   temporary directory. No source-tree working directory is used for inference.
2. Downloads the INT8 pair and six PLDA assets from the embedded manifest,
   verifies SHA-256, and supplies a dedicated model directory to consumers.
   Native model loaders also enforce their normal signature verification.
   The source model cache is never consulted. An optional `--assets-dir` can
   supply separately fetched assets locally; hashes are still mandatory.
3. Runs the copied release CLI; installs the exact wheel into a fresh venv
   with `--no-index --no-deps`; compiles a C caller against the extracted
   library and header; builds external Rust consumers from the extracted
   `.crate` with empty defaults, BYO+VBx, native+VBx and local-only features.
4. Runs real native inference, requires nonempty bounded speaker turns and
   the expected crate version, and checks the C invalid-sample-rate status.
   Missing data or failed inference is an error, never a skip.
5. Inspects ELF imports plus `ldd` on Linux, Mach-O imports on macOS, and PE
   imports on Windows. Runtime execution catches loader failures as well.
   Linux permits libc/libm/libgcc and the standard threading/dl/rt/util loader
   libraries; Darwin permits OS libraries/frameworks (including Accelerate);
   Windows permits the named OS/VC runtime imports listed in the harness.
   The wheel may additionally import its Python runtime. ONNX Runtime,
   OpenBLAS, Torch, TensorFlow, MKL and OpenMP imports are rejected.
6. Writes a success report only after every requested surface succeeds.
   Reports include artifact SHA-256, revision/worktree state, OS, interpreter,
   model/fixture hashes, import lists, Rust kernel resolution and results.

The CLI file, wheel and C archive uploaded after the check are the same files
that were tested. Rust publication re-packages the same clean checkout with
`--locked` and byte-compares the resulting `.crate` to the tested archive
before invoking `cargo publish --locked`. Reports are retained with release
artifacts. This is a functional packaging gate, not a DER/performance gate.

## Published dependency prerequisite

The changed native kernels are prepared as **0.1.3**; core requires at least
that version. The already published 0.1.2 lacks `system-openblas` and cannot
satisfy the current core package, even though workspace builds succeed.

PR/manual checks package both crates and use **unpacked kernel archives** as
an explicit prepublication patch. They do not use the kernel source directory
and do not claim to verify crates.io availability. Reports say
`staged archives (not publication-ready)`.

Tag releases set `require_registry: true`. They package core alone, omit the
staging patch, and require the native/local consumer's kernel dependency to
have a registry source. Missing published dependencies block **all** product
publishers. The kernel release must therefore be reviewed and published
separately before tagging the core release. Nothing in the smoke workflow
publishes kernels automatically or silently weakens this prerequisite.

To check that prerequisite without publishing, dispatch Release artifacts
with `require_registry=true`. To test staged packages, leave it false.
Locally, package both crates and run the harness with their archive paths:

```bash
cargo package --locked --no-verify -p polyvoice-kernels -p polyvoice
python scripts/smoke-release-artifacts.py \
  --crate target/package/polyvoice-0.22.0.crate \
  --staged-kernel target/package/polyvoice-kernels-0.1.3.crate \
  --report /tmp/polyvoice-package-report.json
```

Omit `--staged-kernel` to require registry dependencies. Use `--cli`, `--ffi`
and `--wheel` to qualify their actual release files. Tests for corrupted or
missing assets, invalid inference output, loader failures and unsafe archives:

```bash
python scripts/test-release-artifacts.py
```
