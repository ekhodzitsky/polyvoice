#!/usr/bin/env bash
# Semver gate for the advertised Rust surfaces (docs/semver.md).
#
# Compares the working tree against the latest release tag (override with
# SEMVER_BASELINE=<git rev>) using cargo-semver-checks, once per advertised
# feature set, always assuming a *minor* release so the 0.x compatibility
# lints run instead of being waived by the version bump. A detected break is
# accepted only when the crate's minor version is above the baseline's and
# CHANGELOG.md has a "### Breaking" section under "## [Unreleased]".
#
#   scripts/check-semver.sh          # gate (CI)
#   scripts/check-semver.sh --probe  # prove the gate sees a deliberate break
#
# Feature sets whose features do not exist at the baseline are skipped with a
# notice: a feature introduced after the last release has no contract yet.
#
# --probe uses the working tree itself as the baseline and a scratch copy with
# two frozen entry points hidden as the candidate, so the only difference the
# tool can see is the deliberate removal; pending release breaks do not
# contaminate the result.
set -euo pipefail

cd "$(dirname "$0")/.."

# name|space-separated features ("" = no features: the BYO surface)
SURFACES=(
    "product|pipeline-native vbx"
    "byo|"
    "byo-vbx|clusterer vbx"
    "local|pipeline-local"
)

mode="gate"
case "${1:-}" in
    "") ;;
    --probe) mode="probe" ;;
    *) echo "usage: $0 [--probe]" >&2; exit 2 ;;
esac

command -v cargo-semver-checks >/dev/null || {
    echo "cargo-semver-checks is not installed (cargo install cargo-semver-checks --locked)" >&2
    exit 2
}

tmp="$(mktemp -d)"
cleanup() {
    git worktree remove --force "$tmp/baseline" >/dev/null 2>&1 || true
    rm -rf "$tmp"
}
trap cleanup EXIT

current="."
if [ "$mode" = "gate" ]; then
    baseline="${SEMVER_BASELINE:-$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)}"
    if [ -z "$baseline" ]; then
        echo "no v* tag is reachable; fetch tags or set SEMVER_BASELINE=<rev>" >&2
        exit 2
    fi
    git worktree add --detach "$tmp/baseline" "$baseline" >/dev/null 2>&1
    baseline_root="$tmp/baseline"
else
    baseline="working tree"
    baseline_root="$PWD"
    # Copy the tracked + untracked (non-ignored) tree and hide two frozen
    # entry points. Hiding never breaks compilation, so a failure below can
    # only mean the gate did not notice the removal.
    mkdir -p "$tmp/probe"
    git ls-files -z --cached --others --exclude-standard \
        | tar --null -T - -cf - | tar -xf - -C "$tmp/probe"
    sed -i \
        -e 's|^pub mod vad;|#[doc(hidden)]\npub mod vad;|' \
        -e 's|^pub use vad::{|#[doc(hidden)]\npub use vad::{|' \
        -e 's|^pub mod pipeline_v2;|#[doc(hidden)]\npub mod pipeline_v2;|' \
        -e 's|^pub use pipeline_v2::{|#[doc(hidden)]\npub use pipeline_v2::{|' \
        "$tmp/probe/src/lib.rs"
    current="$tmp/probe"
    echo "probe: EnergyVad (BYO) and the crate-root Pipeline (product) hidden"
fi
baseline_version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$baseline_root/Cargo.toml" | head -1)"
echo "baseline: $baseline (polyvoice $baseline_version)"

baseline_has_feature() {
    grep -Eq "^$1[[:space:]]*=" "$baseline_root/Cargo.toml"
}

ran=0
broken=()
for entry in "${SURFACES[@]}"; do
    name="${entry%%|*}"
    features="${entry#*|}"
    args=(--only-explicit-features)
    skip=""
    for f in $features; do
        if baseline_has_feature "$f"; then
            args+=(--features "$f")
        else
            skip="$f"
        fi
    done
    if [ -n "$skip" ]; then
        echo "--- $name: skipped (feature '$skip' does not exist at $baseline)"
        continue
    fi
    echo "--- $name: features [${features:-none}]"
    ran=$((ran + 1))
    log="$tmp/$name.log"
    if cargo semver-checks check-release \
        --manifest-path "$current/Cargo.toml" \
        --baseline-root "$baseline_root" \
        --release-type minor \
        "${args[@]}" >"$log" 2>&1; then
        grep -E "Checked|Summary" "$log" || true
    elif grep -q "requires new major version" "$log"; then
        cat "$log"
        broken+=("$name")
    else
        echo "cargo-semver-checks failed to run for $name:" >&2
        cat "$log" >&2
        exit 2
    fi
done

if [ "$ran" -eq 0 ]; then
    echo "no surface could be compared against $baseline" >&2
    exit 2
fi

if [ "$mode" = "probe" ]; then
    if [ "${#broken[@]}" -eq "$ran" ]; then
        echo "probe detected on every compared surface ($ran/$ran)"
        exit 0
    fi
    echo "probe missed: ${#broken[@]}/$ran surfaces reported the break" >&2
    exit 1
fi

if [ "${#broken[@]}" -eq 0 ]; then
    echo "no advertised-surface break against $baseline"
    exit 0
fi

echo "advertised-surface break on: ${broken[*]}"
current_version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)"
IFS=. read -r cur_major cur_minor _ <<<"$current_version"
IFS=. read -r base_major base_minor _ <<<"$baseline_version"
bumped=0
if [ "$cur_major" -gt "$base_major" ] || { [ "$cur_major" -eq "$base_major" ] && [ "$cur_minor" -gt "$base_minor" ]; }; then
    bumped=1
fi
declared=0
if awk '/^## \[Unreleased\]/{f=1; next} /^## \[/{f=0} f && /^### Breaking/{found=1} END{exit !found}' CHANGELOG.md; then
    declared=1
fi
if [ "$bumped" -eq 1 ] && [ "$declared" -eq 1 ]; then
    echo "break is declared: version $baseline_version -> $current_version and CHANGELOG Unreleased has '### Breaking'"
    exit 0
fi
[ "$bumped" -eq 1 ] || echo "  missing: bump the minor version (docs/semver.md: 0.x+1.0 for a frozen-surface break); have $current_version vs baseline $baseline_version" >&2
[ "$declared" -eq 1 ] || echo "  missing: a '### Breaking' section under '## [Unreleased]' in CHANGELOG.md" >&2
exit 1
