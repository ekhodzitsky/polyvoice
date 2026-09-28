#!/usr/bin/env bash
# Experimental local-model Apple artifact: no external C/C++ compilation,
# no Accelerate/BNNS imports. Shared CI is not the locked performance host.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo 'Requires Darwin arm64; Linux cannot certify Apple linkage.' >&2
  exit 1
fi
out="${OUT:-$PWD/bench-results/darwin-rust-audit}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
# A fresh target directory prevents native objects from a previous build from
# bypassing the disabled C/C++ compiler check.
if [[ -e "$out/target" ]]; then
  echo "Use a fresh output directory: $out/target already exists" >&2
  exit 1
fi
export CARGO_TARGET_DIR="$out/target"
features=pipeline-local,experimental-darwin-rust
cargo tree --locked --no-default-features --features "$features" \
  -e normal,build --prefix none > "$out/dependencies.txt"
if grep -Eq '^(ring|ureq|rustls|ort|ort-sys|tract-onnx) v' "$out/dependencies.txt"; then
  echo 'Unexpected downloader/native runtime in the local-model graph' >&2
  exit 1
fi
CC=/usr/bin/false CXX=/usr/bin/false cargo build --locked --release \
  --no-default-features --features "$features" --example local_native \
  > "$out/build.log" 2>&1
binary="$CARGO_TARGET_DIR/release/examples/local_native"
otool -L "$binary" > "$out/linkage.txt"
nm -u "$binary" > "$out/undefined-symbols.txt"
if grep -Eiq 'Accelerate|BNNS|cblas|pv_bnns|libblas|libopenblas' \
    "$out/linkage.txt" "$out/undefined-symbols.txt"; then
  echo 'Experimental binary imports a native math backend' >&2
  exit 1
fi
# Reuse the artifact gate's hash-checked model bootstrap; inference below is
# strictly local and contains no downloader.
python3 - "$out" "$binary" <<'PY'
import importlib.util, json, pathlib, platform, subprocess, sys
out, binary = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('assets', 'scripts/smoke-release-artifacts.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
hashes = module.assets(out / 'models')
cmd = [str(binary), str(out / 'models'), str(out / 'models'), 'tests/data/e2e-smoke/audio/fuzfh.wav']
output = subprocess.check_output(cmd, text=True)
rows = [line.split('\t') for line in output.splitlines()]
if not rows or not all(len(row) == 3 and 0 <= float(row[0]) < float(row[1]) for row in rows):
    raise ValueError('invalid or empty local inference output')
(out / 'turns.tsv').write_text(output)
report = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'platform': platform.platform(), 'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
          'model_hashes': hashes, 'binary_sha256': module.sha(binary), 'command': cmd,
          'features': ['pipeline-local', 'experimental-darwin-rust'], 'turns': len(rows),
          'status': 'build, linkage and local inference passed; locked performance unmeasured'}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
