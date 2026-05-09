#!/usr/bin/env bash
set -euo pipefail

# Sign all models listed in manifest.toml and inject signatures back into manifest.toml.
#
# Usage: ./scripts/sign-models.sh [--dry-run]
#
# Requirements: minisign CLI, Python 3.
# Secret key path: from env MINISIGN_SECRET_KEY or default models/signing.key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/src/models/manifest.toml"
MODELS_DIR="${PROJECT_ROOT}/models"
SECRET_KEY="${MINISIGN_SECRET_KEY:-${PROJECT_ROOT}/models/signing.key}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be modified."
fi

if ! command -v minisign &>/dev/null; then
    echo "ERROR: minisign CLI not found in PATH" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "false" && ! -f "$SECRET_KEY" ]]; then
    echo "ERROR: Secret key not found at $SECRET_KEY" >&2
    echo "Set MINISIGN_SECRET_KEY or ensure models/signing.key exists." >&2
    exit 1
fi

# Parse manifest.toml to get model entries and their filenames.
python3 - "$MANIFEST" "$MODELS_DIR" "$SECRET_KEY" "$DRY_RUN" << 'PYEOF'
import sys
import re
import pathlib
import subprocess

manifest_path = pathlib.Path(sys.argv[1])
models_dir = pathlib.Path(sys.argv[2])
secret_key = pathlib.Path(sys.argv[3])
dry_run = sys.argv[4].lower() == "true"

manifest_text = manifest_path.read_text()
lines = manifest_text.splitlines()

# Track section positions: list of (start_index, model_id, filename, has_signature)
sections = []
current = None

for i, line in enumerate(lines):
    m = re.match(r'^\[models\.(\w+)\]', line)
    if m:
        if current is not None:
            sections.append(current)
        current = {"start": i, "id": m.group(1), "filename": None, "has_sig": False, "end": None}
    if current is not None:
        if line.startswith("filename"):
            _, val = line.split("=", 1)
            current["filename"] = val.strip().strip('"').strip("'")
        if line.startswith("signature"):
            current["has_sig"] = True
        # Detect end of section: next section header or EOF
        if m and i > current["start"]:
            current["end"] = i
            sections.append(current)
            current = {"start": i, "id": m.group(1), "filename": None, "has_sig": False, "end": None}

if current is not None:
    current["end"] = len(lines)
    sections.append(current)

changed = False
offset = 0  # line shift due to insertions

for sec in sections:
    filename = sec.get("filename")
    if not filename:
        continue
    if sec.get("has_sig"):
        print(f"[SKIP] {sec['id']}: already has signature")
        continue

    local_path = models_dir / filename
    if not local_path.exists():
        local_path = models_dir / "int8" / filename
    if not local_path.exists():
        print(f"[SKIP] {sec['id']}: {filename} not found locally")
        continue

    sig_path = local_path.with_suffix(local_path.suffix + ".minisig")

    if dry_run:
        print(f"[DRY RUN] Would sign {sec['id']} ({local_path})")
        continue

    if not sig_path.exists():
        print(f"[SIGN] {sec['id']} -> {sig_path}")
        subprocess.run(
            [
                "minisign", "-S", "-s", str(secret_key),
                "-m", str(local_path), "-x", str(sig_path),
                "-t", f"polyvoice model signing | {filename}",
            ],
            check=True,
        )
    else:
        print(f"[CACHE] {sec['id']}: signature already exists")

    sig_text = sig_path.read_text().strip()
    sig_lines = [f"signature = '''"] + sig_text.splitlines() + ["'''"]

    insert_at = sec["end"] + offset
    for j, sl in enumerate(sig_lines):
        lines.insert(insert_at + j, sl)
    offset += len(sig_lines)
    changed = True

if changed and not dry_run:
    manifest_path.write_text("\n".join(lines) + "\n")
    print(f"[WRITE] Updated {manifest_path}")
    print("[TEST] Running m5_manifest_smoke_test...")
    subprocess.run(
        ["cargo", "test", "--test", "m5_manifest_smoke_test", "--features", "download"],
        cwd=str(manifest_path.parent.parent.parent),
        check=True,
    )
    print("[OK] Manifest smoke test passed.")
else:
    print("[OK] No changes needed.")
PYEOF
