#!/usr/bin/env python3
"""Keep platform reports intact without duplicate GitHub asset basenames."""
import pathlib
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name('prepare-release-assets.sh')


class ReleaseAssetsTest(unittest.TestCase):
    def test_platform_reports_survive_bundling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for platform in ('Linux-x86_64', 'Darwin-arm64'):
                report = root / 'release-quality-evidence' / platform / 'evidence.json'
                report.parent.mkdir(parents=True)
                report.write_text(platform)
            product = root / 'native-linux' / 'polyvoice-linux-x86_64'
            product.parent.mkdir()
            product.write_bytes(b'qualified binary')
            subprocess.run(['bash', str(SCRIPT), str(root)], check=True)
            files = [p for p in root.rglob('*') if p.is_file()]
            self.assertEqual(len(files), len({p.name for p in files}))
            self.assertEqual(product.read_bytes(), b'qualified binary')
            with tarfile.open(root / 'release-quality-evidence.tar.gz') as bundle:
                for platform in ('Linux-x86_64', 'Darwin-arm64'):
                    with bundle.extractfile(f'release-quality-evidence/{platform}/evidence.json') as report:
                        self.assertEqual(report.read().decode(), platform)


if __name__ == '__main__':
    unittest.main()
