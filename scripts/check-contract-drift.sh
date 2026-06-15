#!/usr/bin/env bash
# Advisory, NON-BLOCKING contract-drift nudge.
#
# Warns when a module's `src/<m>/mod.rs` changes in a diff without its sibling
# `src/<m>/MODULE_CONTRACT.md` being updated in the same diff — a hint that the
# contract may have drifted from the code. It is intentionally a *nudge*: it
# always exits 0 and never fails CI.
#
# It does NOT enforce the COAD `context_budget` (max_source_lines etc.) — that
# field is advisory by design and its apparent breaches are an artifact of
# counting inline `#[cfg(test)]` modules.
#
# Usage:  scripts/check-contract-drift.sh [diff-base]   (default: origin/master)
# Opt-out: put [skip-contract-drift] in the latest commit message.
set -uo pipefail

base="${1:-origin/master}"

if git log -1 --pretty=%B 2>/dev/null | grep -q '\[skip-contract-drift\]'; then
  echo "contract-drift: skipped ([skip-contract-drift] in commit message)"
  exit 0
fi

# Diff the working/HEAD tree against the base; tolerate a missing base ref.
changed="$(git diff --name-only "${base}...HEAD" 2>/dev/null \
  || git diff --name-only "${base}" 2>/dev/null \
  || git diff --name-only HEAD~1 2>/dev/null)"

drift=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    src/*/mod.rs)
      contract="${f%/mod.rs}/MODULE_CONTRACT.md"
      # Only nudge for modules that actually carry a contract.
      [ -f "$contract" ] || continue
      if ! printf '%s\n' "$changed" | grep -qxF "$contract"; then
        echo "::warning::contract-drift: ${f} changed but ${contract} was not updated"
        drift=1
      fi
      ;;
  esac
done <<EOF
$changed
EOF

if [ "$drift" -eq 0 ]; then
  echo "contract-drift: no drift detected"
fi
exit 0
