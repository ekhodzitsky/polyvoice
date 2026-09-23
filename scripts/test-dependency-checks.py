"""Regression checks for dependency gates; no downloads or lockfile writes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


class DependencyChecks(unittest.TestCase):
    def run_gate(self, script, failure):
        with tempfile.TemporaryDirectory() as directory:
            cargo = Path(directory) / "cargo"
            cargo.write_text("""#!/usr/bin/env python3
import os, sys
args = ' '.join(sys.argv[1:])
failure = os.environ['CARGO_TEST_FAILURE']
if (failure == 'all'
    or failure == 'python' and 'python/Cargo.toml' in args
    or failure == 'asr' and '-p polyvoice-asr' in args):
    print('simulated Cargo resolution failure', file=sys.stderr)
    sys.exit(101)
if '--locked' not in sys.argv:
    print('missing --locked', file=sys.stderr)
    sys.exit(102)
print('polyvoice v0.21.0')
if 'vad-earshot' in args:
    print('earshot v1.2.0')
if any(flag in args for flag in ['backend-tract', 'pipeline-tract', 'cli-tract']):
    print('tract-onnx v0.23.4')
if any(flag in args for flag in ['native', '--features cli', '--features ffi']):
    print('polyvoice-kernels v0.1.2')
if failure == 'leak':
    print('ort v2.0.0-rc.12')
""")
            cargo.chmod(0o755)
            return subprocess.run(
                ["bash", str(ROOT / "scripts" / script)],
                cwd=ROOT,
                env={**os.environ, "PATH": directory + os.pathsep + os.environ["PATH"],
                     "CARGO_TEST_FAILURE": failure},
                capture_output=True, text=True,
            )

    def test_resolution_failure_is_not_dependency_absence(self):
        for script in ["check-ort-free.sh", "check-zero-deps.sh",
                       "check-standalone-lockfiles.sh", "check-ort-version.sh"]:
            with self.subTest(script=script):
                result = self.run_gate(script, "all")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("simulated Cargo resolution failure", result.stderr)
                self.assertNotIn("OK:", result.stdout)

    def test_stale_python_lockfile_fails_visibly(self):
        for script in ["check-ort-free.sh", "check-zero-deps.sh",
                       "check-standalone-lockfiles.sh"]:
            with self.subTest(script=script):
                result = self.run_gate(script, "python")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("simulated Cargo resolution failure", result.stderr)

    def test_asr_resolution_failure_fails_visibly(self):
        result = self.run_gate("check-zero-deps.sh", "asr")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated Cargo resolution failure", result.stderr)

    def test_forbidden_dependency_is_rejected(self):
        result = self.run_gate("check-ort-free.sh", "leak")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAIL: ort leaked", result.stdout)


if __name__ == "__main__":
    unittest.main()
