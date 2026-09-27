#!/usr/bin/env python3
"""Negative checks for the release artifact gate (stdlib only)."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).with_name("smoke-release-artifacts.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class GateTests(unittest.TestCase):
    def test_real_audio_cannot_pass_with_empty_or_invalid_results(self):
        for result in ({}, {"num_speakers": 0, "turns": []},
                       {"num_speakers": 1, "turns": [{"speaker": 0, "time": {"start": 0, "end": float('nan')}}]},
                       {"num_speakers": 1, "turns": [{"speaker": 1, "time": {"start": 0, "end": 1}}]},
                       {"num_speakers": 1, "turns": [{"speaker": 0, "time": {"start": 0, "end": 100}}]}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                smoke.validate_result(result, 26)

    def test_native_dependency_allowlist_is_platform_specific(self):
        for name, system in [("libopenblas.so.0", "Linux"), ("libonnxruntime.so", "Linux"),
                             ("libssl.so.3", "Linux"), ("onnxruntime.dll", "Windows"),
                             ("python312.dll", "Windows"), ("@rpath/libunexpected.dylib", "Darwin")]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                smoke.check_library(name, system)
        smoke.check_library("libc.so.6", "Linux")
        smoke.check_library("python312.dll", "Windows", wheel=True)
        smoke.check_library("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate", "Darwin")

    def test_unresolved_loader_import_fails(self):
        with patch.object(smoke.platform, "system", return_value="Linux"), \
             patch.object(smoke, "run", side_effect=['(NEEDED) [libc.so.6]', 'libc.so.6 => not found']):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                smoke.native_imports(Path('/tmp/fake'), {})

    def test_corrupted_asset_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / 'source'
            source.mkdir()
            (source / 'powerset_int8.onnx').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                smoke.assets(root / 'models', source)

    def test_missing_asset_cannot_skip(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                smoke.assets(Path(root) / 'models', Path(root) / 'missing')

    def test_failed_check_removes_stale_success_report(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            report = root / 'report.json'
            report.write_text('{"status":"passed"}')
            args = SimpleNamespace(report=report, cli=root / 'missing-cli',
                                   ffi=None, wheel=None, crate=None, staged_kernel=None)
            with self.assertRaises(FileNotFoundError):
                smoke.smoke(args)
            self.assertFalse(report.exists())

    def test_crate_cannot_escape_consumer_directory(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / 'bad.crate'
            with tarfile.open(archive, 'w:gz') as tar:
                info = tarfile.TarInfo('../../escaped')
                info.size = 3
                tar.addfile(info, io.BytesIO(b'bad'))
            with self.assertRaises(tarfile.FilterError):
                smoke.unpack_crate(archive, root / 'out')
            self.assertFalse((root / 'escaped').exists())


if __name__ == '__main__':
    unittest.main()
