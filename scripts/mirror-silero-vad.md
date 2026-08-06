# Mirroring Silero VAD ONNX to a release asset

## Problem

`models.silero_vad` in `src/models/manifest.toml` downloads from upstream,
**commit-pinned** (not floating `master`):

```
https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx
```

Floating `master` would break fresh installs when upstream replaces the file
under the same path (SHA-256 fails closed). The weights we ship against are
**v6.2-generation** Silero VAD at commit `bfdc019` (hash verified).

Pinned hash (source of truth — do not invent a new one without re-signing):

```
sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
```

## Goal

Host an immutable copy under **polyvoice GitHub Releases** (or another
first-party HTTPS origin we control), point the manifest `url` at that asset,
and keep the upstream master URL only as a documented fallback for operators
who re-pin deliberately.

This procedure is intentionally manual: publishing a release asset needs
repository credentials and a verified hash match. Do **not** put a guessed
mirror URL in the manifest.

## Publish procedure (human)

1. Obtain the current model file (from cache, `scripts/download-models.sh`, or
   a known-good tree):

   ```bash
   # Example: verify a local file matches the pin
   shasum -a 256 path/to/silero_vad.onnx
   # must print: 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
   ```

2. Create or reuse a GitHub Release (prefer a dedicated model/assets tag such
   as `models-silero-vad-v6` or the next polyvoice release) and upload
   `silero_vad.onnx` as a release asset.

3. Confirm the asset URL form:

   ```
   https://github.com/ekhodzitsky/polyvoice/releases/download/<tag>/silero_vad.onnx
   ```

4. Re-download the asset and verify SHA-256 again before changing the manifest.

5. Update `src/models/manifest.toml`:

   ```toml
   [models.silero_vad]
   url      = "https://github.com/ekhodzitsky/polyvoice/releases/download/<tag>/silero_vad.onnx"
   sha256   = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
   # filename / size / signature unchanged unless the file bytes change
   ```

6. Leave a comment in the manifest pointing at the old upstream URL for
   recovery, e.g.:

   ```toml
   # Upstream fallback (manual): snakers4/silero-vad master path — only use if
   # the release asset is unavailable AND sha256 still matches the pin above.
   ```

7. If bytes ever change: re-sign with the release key (separate signing flow),
   update `sha256` / `size` / `signature`, and re-run model smoke tests.

8. Optionally update `scripts/download-models.sh` `SILERO_URL` to the same
   mirror so local developer downloads match the registry.

## What not to do

- Do not invent or commit a mirror URL that has not been published and hash-
  checked.
- Do not loosen SHA-256 verification to “fix” master drift.
- Do not change signing keys here (covered by the model re-sign release flow).

## Status

Upstream URL is **commit-pinned** (`bfdc019`) with the pinned hash. First-party
release-asset mirror is **not yet published**; this document is the runbook for the
human publish step.
