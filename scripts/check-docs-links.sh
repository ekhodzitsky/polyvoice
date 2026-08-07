#!/usr/bin/env bash
# Check relative markdown links inside the repo (no network).
# Usage: bash scripts/check-docs-links.sh
# Exit 1 if any target is missing.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

python3 - <<'PY'
import re, sys
from pathlib import Path

root = Path(".").resolve()
globs = ["*.md", "docs/**/*.md", "python/*.md", "examples/*.md"]
files: list[Path] = []
for g in globs:
    files.extend(root.glob(g))
# de-dupe
files = sorted({f.resolve() for f in files if f.is_file()})

link_re = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
broken: list[tuple[str, str, str]] = []
ok = 0
for md in files:
    if "target" in md.parts:
        continue
    text = md.read_text(encoding="utf-8", errors="replace")
    for m in link_re.finditer(text):
        url = m.group(2).strip()
        if url.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = url.split("#", 1)[0].split("?", 1)[0]
        if not path_part:
            continue
        target = (md.parent / path_part).resolve()
        if not target.exists():
            broken.append((str(md.relative_to(root)), url, m.group(1)[:60]))
        else:
            ok += 1

print(f"docs-links: {ok} ok, {len(broken)} broken")
for path, url, label in broken:
    print(f"  BROKEN  {path}: ({url}) {label!r}")
sys.exit(1 if broken else 0)
PY
