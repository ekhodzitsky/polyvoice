#!/usr/bin/env python3
"""Exercise version edits without resolving dependencies or touching this tree."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest


class VersionBumpTests(unittest.TestCase):
    def test_stable_to_candidate(self):
        self.bump('0.22.0', '1.0.0-rc.1')

    def test_candidate_to_candidate(self):
        self.bump('1.0.0-rc.1', '1.0.0-rc.2')

    def bump(self, old, new):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['scripts', 'python', 'tests', 'bin']:
                (root / name).mkdir()
            shutil.copy(Path(__file__).with_name('bump-version.sh'), root / 'scripts')
            for name in ['Cargo.toml', 'python/Cargo.toml', 'python/pyproject.toml']:
                (root / name).write_text(f'version = "{old}"\n')
            (root / 'tests/cli_smoke_test.rs').write_text(f'"polyvoice {old}"\n')
            (root / 'tests/der_baseline.json').write_text(json.dumps({'crate_version': old}))
            (root / 'CHANGELOG.md').write_text('## [Unreleased]\n\n### Fixed\n\n- Example.\n')
            cargo = root / 'bin/cargo'
            cargo.write_text('#!/bin/sh\nexit 0\n')
            cargo.chmod(0o755)
            env = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
            subprocess.run(['bash', str(root / 'scripts/bump-version.sh'), new],
                           env=env, check=True, capture_output=True, text=True)
            for name in ['Cargo.toml', 'python/Cargo.toml', 'python/pyproject.toml']:
                self.assertEqual(tomllib.loads((root / name).read_text())['version'], new)
            self.assertEqual((root / 'tests/cli_smoke_test.rs').read_text(), f'"polyvoice {new}"\n')
            self.assertEqual(json.loads((root / 'tests/der_baseline.json').read_text())['crate_version'], new)
            self.assertIn(f'## [{new}]', (root / 'CHANGELOG.md').read_text())


if __name__ == '__main__':
    unittest.main()
