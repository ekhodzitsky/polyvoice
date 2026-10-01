#!/usr/bin/env bash
# GitHub flattens asset paths; bundle same-named platform reports together.
set -euo pipefail
artifacts="${1:?Usage: prepare-release-assets.sh <downloaded-artifact-directory>}"
tar -czf "$artifacts/release-quality-evidence.tar.gz" -C "$artifacts" release-quality-evidence
rm -r -- "$artifacts/release-quality-evidence"
