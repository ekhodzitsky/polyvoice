#!/usr/bin/env python3
"""Negative checks for the release artifact gate (stdlib only)."""
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("smoke", Path(__file__).with_name("smoke-release-artifacts.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class GateTests(unittest.TestCase):
    def test_offline_cannot_pass_on_unsupported_platform(self):
        with patch.object(smoke.platform, 'system', return_value='unsupported'), \
             self.assertRaisesRegex(ValueError, 'network isolation'):
            smoke.offline_prefix(Path('.'), {}, {})

    def test_rejection_requires_nonzero_exit_and_expected_stderr(self):
        for code, stderr in [(0, 'audio too long'), (1, 'unrelated failure')]:
            with patch.object(smoke.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=code, stdout='', stderr=stderr)), self.assertRaises(ValueError):
                smoke.run(['audio too long'], '.', reject='audio too long')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux seccomp launcher')
    def test_offline_child_denies_network_without_affecting_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {}
            prefix = smoke.offline_prefix(Path(directory), os.environ.copy(), report)
            self.assertIn('IPv4/IPv6 TCP/UDP denied', report['network_isolation']['probe'])
            with self.assertRaisesRegex(RuntimeError, 'Operation not permitted'):
                smoke.run([*prefix, sys.executable, '-c', 'import socket; socket.socket()'], directory)
            with socket.socket():
                pass

    def test_silence_accepts_only_an_empty_result(self):
        self.assertEqual(smoke.validate_result({'num_speakers': 0, 'turns': []}, 1, speech=False),
                         {'num_speakers': 0, 'turns': []})
        with self.assertRaises(ValueError):
            smoke.validate_result({'num_speakers': 1, 'turns': []}, 1, speech=False)

    def test_hour_fixture_cannot_pass_with_only_its_beginning_processed(self):
        result = {'num_speakers': 1, 'turns': [{'speaker': 0, 'time': {'start': 1, 'end': 2}}]}
        with self.assertRaisesRegex(ValueError, 'end of'):
            smoke.validate_result(result, 3600, require_tail=True)

    def test_generated_fixtures_have_real_duration_and_end_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = smoke.scenario_fixtures(Path(directory), b'\x01\x00' * 16000)
            hour = cases['hour']
            import wave
            with wave.open(str(hour['wav'])) as audio:
                self.assertEqual(audio.getnframes(), 16000 * 3600)
                audio.setpos(16000 * 3600 - 1)
                self.assertEqual(audio.readframes(1), b'\x01\x00')
            self.assertEqual(cases['too-long']['pcm'].stat().st_size, (16000 * 3600 + 1) * 4)

    def test_windows_tool_uses_compiler_environment_path(self):
        env = {"PATH": "C:/MSVC/bin"}
        with patch.object(smoke.os, 'name', 'nt'), \
             patch.object(smoke.shutil, 'which', return_value='C:/MSVC/bin/dumpbin.exe') as which, \
             patch.object(smoke.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='ok')) as child:
            self.assertEqual(smoke.run(['dumpbin', '/DEPENDENTS', 'consumer.exe'], '.', env), 'ok')
            which.assert_called_once_with('dumpbin', path=env['PATH'])
            self.assertEqual(child.call_args.args[0][0], 'C:/MSVC/bin/dumpbin.exe')

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
        smoke.check_library("combase.dll", "Windows")
        smoke.check_library("python312.dll", "Windows", wheel=True)
        smoke.check_library("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate", "Darwin")

    def test_unresolved_loader_import_fails(self):
        with patch.object(smoke.platform, "system", return_value="Linux"), \
             patch.object(smoke, "run", side_effect=['(NEEDED) [libc.so.6]', 'libc.so.6 => not found']):
            with self.assertRaisesRegex(ValueError, "unresolved"):
                smoke.native_imports(Path('/tmp/fake'), {})

    def test_macho_install_name_is_not_an_import(self):
        path = Path('/tmp/_polyvoice.cpython-312-darwin.so')
        identity = '@rpath/polyvoice._polyvoice.cpython-312-darwin.so'
        listing = f'{path}:\n\t{identity} (compatibility version 0.0.0)\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n'
        with patch.object(smoke.platform, 'system', return_value='Darwin'), \
             patch.object(smoke, 'run', side_effect=[listing, f'{path}:\n{identity}\n']):
            self.assertEqual(smoke.native_imports(path, {}, wheel=True), ['/usr/lib/libSystem.B.dylib'])

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
