#!/usr/bin/env bash
# Bump version across all project files.
# Usage: bash scripts/bump-version.sh 0.6.9
set -euo pipefail

VERSION="${1:?Usage: bump-version.sh <new-version>}"

cd "$(dirname "$0")/.."

echo "Bumping to v${VERSION}..."

# 1. Main Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" Cargo.toml
rm Cargo.toml.bak

# 2. python/Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" python/Cargo.toml
rm python/Cargo.toml.bak

# 3. python/pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" python/pyproject.toml
rm python/pyproject.toml.bak

# 4. tests/cli_smoke_test.rs
sed -i.bak "s/polyvoice [0-9]\+\.[0-9]\+\.[0-9]\+/polyvoice ${VERSION}/" tests/cli_smoke_test.rs
rm tests/cli_smoke_test.rs.bak

# 5. CHANGELOG.md — prepend new section if not present
if ! grep -q "## \[${VERSION}\]" CHANGELOG.md; then
    DATE=$(date +%Y-%m-%d)
    awk -v ver="${VERSION}" -v date="${DATE}" '
    BEGIN { printed=0 }
    /^## \[Unreleased\]/ {
        print
        print ""
        print "## [" ver "] - " date
        print ""
        print "### Changed"
        print ""
        printed=1
        next
    }
    { print }
    ' CHANGELOG.md > CHANGELOG.md.tmp
    mv CHANGELOG.md.tmp CHANGELOG.md
fi

echo "Done. Review the changes, then commit and tag:"
echo "  git add -A && git commit -m \"chore(release): bump version to ${VERSION}\""
echo "  git tag v${VERSION}"
