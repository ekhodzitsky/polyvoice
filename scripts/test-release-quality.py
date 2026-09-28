#!/usr/bin/env python3
"""Offline fail-closed checks for release measurement evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import zipfile

spec = importlib.util.spec_from_file_location('quality', Path(__file__).with_name('release-quality.py'))
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class EvidenceTests(unittest.TestCase):
    def test_exact_coverage_rejects_partial_duplicate_and_substitution(self):
        quality.coverage(['a', 'b'], ['a', 'b'])
        for actual in (['a'], ['a', 'a'], ['a', 'c'], ['a', 'b', 'c']):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                quality.coverage(actual, ['a', 'b'])

    def test_floors_reject_nonfinite_and_regression(self):
        for value in (float('nan'), float('inf'), -1, True, '7', None, 7.12):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quality.maximum(value, 7.11)
        quality.maximum(7.11, 7.11)

    def test_scoreboard_requires_every_locked_characteristic(self):
        values = {'der_no_collar_micro': 7.11, 'der_no_collar_macro': 7.39,
                  'rt_factor_avg': 117, 'model_bytes': 8414314, 'rss_mib': 556}
        quality.scoreboard(values)
        for key, value in values.items():
            bad = copy.deepcopy(values)
            bad[key] = value - 0.01 if key == 'rt_factor_avg' else value + 0.01
            with self.subTest(key=key), self.assertRaises(ValueError):
                quality.scoreboard(bad)
        for key in values:
            bad = copy.deepcopy(values)
            bad[key] = float('nan')
            with self.subTest(key=key), self.assertRaises(ValueError):
                quality.scoreboard(bad)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = 'a' * 40
        for target in quality.PLATFORMS:
            directory = self.root / target
            directory.mkdir()
            linux = target.startswith('Linux')
            host = {'system': 'Linux' if linux else 'Darwin', 'machine': 'x86_64' if linux else 'arm64',
                    'cpu': 'Test CPU' if linux else 'Apple M1 Pro', 'platform': target,
                    'cpu_count': 10, 'hostname': 'test-host', 'python': '3.11', 'rustc': 'rustc fixture',
                    'isolated': not linux, 'runner_environment': 'self-hosted'}
            evidence = {'schema': 1, 'status': 'passed', 'clean_tree': True, 'revision': self.revision,
                        'protocol': quality.PROTOCOL, 'build_command': quality.BUILD, 'build_environment': {},
                        'manifest_sha256': quality.sha(quality.MANIFEST), 'binary_sha256': 'b' * 64,
                        'model_hashes': {k: v['sha256'] for k, v in quality.registry_models().items()},
                        'host': host, 'measured_at': '2026-09-28T00:00:00+00:00', 'model_bytes': 8414314, 'commands': {}, 'reports': {}}
            names = ['voxconverse-test', 'ami-test'] + ([] if linux else ['native-vox3'])
            for name in names:
                # Deliberately synthetic passing evidence; never publish test fixtures as measurements.
                ids = [e['id'] for e in quality.read(quality.MANIFEST)[name]]
                cmd = ['/test/polyvoice-bench', '/test/' + name, *quality.BENCH_ARGS,
                       '--output', '/test/' + name + '.json']
                report = {'git_sha': self.revision, 'host_os': 'linux' if linux else 'macos',
                          'host_arch': 'x86_64' if linux else 'aarch64', 'profile': 'balanced',
                          'resolved_execution_provider': 'Cpu', 'collar_secs': 0, 'skip_overlap': False,
                          'command_line': ' '.join(cmd), 'files_processed': len(ids), 'files_skipped': 0,
                          'model_hashes': [{'model_id': k, 'sha256': evidence['model_hashes'][k]}
                                           for k in ('powerset_int8', 'resnet34_int8')],
                          'der_no_collar_micro': 7, 'der_no_collar_macro': 7, 'rt_factor_avg': 117,
                          'per_file': [{'filename': mid, 'der_no_collar': 7} for mid in ids]}
                self.write(directory / (name + '.json'), report)
                evidence['commands'][name] = cmd
                evidence['reports'][name + '.json'] = quality.sha(directory / (name + '.json'))
            if linux:
                source = quality.ROOT / 'benchmarks/results/notsofar-eval-native-2026-09-27'
                out = directory / 'notsofar'
                out.mkdir()
                with zipfile.ZipFile(source / 'hypotheses.zip') as archive:
                    archive.extractall(out)
                report = quality.read(source / 'verification.json')
                report['revision'] = self.revision
                self.write(out / 'report.json', report)
                evidence['reports']['notsofar/report.json'] = quality.sha(out / 'report.json')
            else:
                (directory / 'native-vox3.log').write_text('  583008256 maximum resident set size\n')
                evidence['reports']['native-vox3.log'] = quality.sha(directory / 'native-vox3.log')
            self.write(directory / 'evidence.json', evidence)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def verify(self):
        quality.verify_bundle(self.root, self.revision)

    def test_complete_bundle_passes(self):
        self.verify()

    def test_manifest_matches_both_historical_full_splits(self):
        manifest = quality.read(quality.MANIFEST)
        for folder in ['linux-cpu-native-der-2026-09-13-vbx-ahc', 'darwin-native-der-2026-09-22']:
            for name, count in [('voxconverse-test', 232), ('ami-test', 16)]:
                ids = [row['id'] for row in manifest[name]]
                self.assertEqual(len(ids), count)
                historical = quality.read(quality.ROOT / 'benchmarks/results' / folder / (name + '.json'))
                quality.coverage([row['filename'] for row in historical['per_file']], ids)

    def test_missing_platform_cannot_pass(self):
        (self.root / 'Darwin-arm64/evidence.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_stale_dirty_wrong_host_and_models_fail(self):
        path = self.root / 'Darwin-arm64/evidence.json'
        original = quality.read(path)
        mutations = [lambda e: e.update(revision='c' * 40), lambda e: e.update(clean_tree=False),
                     lambda e: e.update(status='failed'), lambda e: e.update(model_bytes=0),
                     lambda e: e.update(measured_at='invalid'),
                     lambda e: e['commands'].pop('native-vox3'),
                     lambda e: e['host'].update(cpu=''), lambda e: e.update(manifest_sha256='d' * 64),
                     lambda e: e['model_hashes'].pop(next(k for k in e['model_hashes'] if k.startswith('vbx_plda_'))),
                     lambda e: e['host'].update(isolated=False),
                     lambda e: e['host'].update(runner_environment='github-hosted'),
                     lambda e: e['host'].update(cpu='Apple M2 Pro'),
                     lambda e: e['host'].update(machine='x86_64'),
                     lambda e: e.update(build_environment={'RUSTFLAGS': '-C target-cpu=native'}),
                     lambda e: e.update(build_command=['cargo', 'build']),
                     lambda e: e['protocol'].update(skip_overlap=True)]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            self.write(path, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify()
        self.write(path, original)

    def test_changed_report_or_hypothesis_fails(self):
        path = self.root / 'Darwin-arm64/voxconverse-test.json'
        original = path.read_text()
        path.write_text(original + ' ')
        with self.assertRaises(ValueError):
            self.verify()
        path.write_text(original)
        path = next((self.root / 'Linux-x86_64/notsofar').glob('MTG_*.json'))
        path.write_text('{}')
        with self.assertRaises(ValueError):
            self.verify()

    def test_even_rehashed_incomplete_regressing_or_mismatched_reports_fail(self):
        directory = self.root / 'Linux-x86_64'
        path = directory / 'voxconverse-test.json'
        original = quality.read(path)
        evidence = quality.read(directory / 'evidence.json')
        mutations = [lambda r: r.update(git_sha='c' * 40), lambda r: r.update(files_skipped=1),
                     lambda r: r['per_file'].pop(),
                     lambda r: r['per_file'][0].update(filename=r['per_file'][1]['filename']),
                     lambda r: r.update(der_no_collar_micro=100),
                     lambda r: r.update(der_no_collar_macro=float('nan')),
                     lambda r: r.update(collar_secs=0.25), lambda r: r.update(skip_overlap=True),
                     lambda r: r.update(command_line=r['command_line'] + ' --threshold 0.4')]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            self.write(path, changed)
            evidence['reports']['voxconverse-test.json'] = quality.sha(path)
            self.write(directory / 'evidence.json', evidence)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify()

    def test_historical_notsofar_revision_fails_even_with_fresh_envelope(self):
        directory = self.root / 'Linux-x86_64'
        path = directory / 'notsofar/report.json'
        report = quality.read(path)
        report['revision'] = 'c' * 40
        self.write(path, report)
        evidence = quality.read(directory / 'evidence.json')
        evidence['reports']['notsofar/report.json'] = quality.sha(path)
        self.write(directory / 'evidence.json', evidence)
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_or_corrupt_input_fails_before_build(self):
        with self.assertRaises(ValueError):
            quality.validate_inputs('native-vox3', self.root)
        (self.root / 'audio').mkdir()
        for name in ['euqef', 'fuzfh', 'msbyq']:
            (self.root / 'audio' / (name + '.wav')).write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            quality.validate_inputs('native-vox3', self.root)

    def test_rss_requires_one_valid_measurement(self):
        for log in ['', 'nan maximum resident set size', '0 maximum resident set size',
                    '12 maximum resident set size\n13 maximum resident set size']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                quality.rss_mib(log)


if __name__ == '__main__':
    unittest.main()
