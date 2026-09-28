#!/usr/bin/env python3
"""Collect or verify revision-bound native release measurements (Python stdlib)."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'benchmarks/manifests/release-corpora.json'
PLATFORMS = ('Linux-x86_64', 'Darwin-arm64')
BUILD = ['cargo', 'build', '--locked', '--release', '--features', 'cli',
         '--bin', 'polyvoice-bench', '--bin', 'polyvoice']
BENCH_ARGS = ['--profile', 'balanced', '--pipeline', 'v2', '--clusterer', 'vbx',
              '--execution-provider', 'cpu', '--collar', '0', '--jobs', '1']
PROTOCOL = {'features': 'cli', 'profile': 'balanced', 'pipeline': 'v2', 'clusterer': 'vbx',
            'execution_provider': 'cpu', 'collar': 0, 'skip_overlap': False, 'jobs': 1}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f'invalid measurement: {value!r}')
    return value


def maximum(value, limit):
    require(number(value) <= number(limit), f'{value} exceeds {limit}')


def coverage(actual, expected):
    require(len(actual) == len(expected) and set(actual) == set(expected),
            'missing, duplicate or unexpected corpus files')


def scoreboard(values):
    floors = read(ROOT / 'tests/native_scoreboard.json')
    for value, limit in [('der_no_collar_micro', 'der_no_collar_micro_max'),
                         ('der_no_collar_macro', 'der_no_collar_macro_max'),
                         ('model_bytes', 'model_bytes_max'), ('rss_mib', 'rss_mib_max')]:
        maximum(values[value], floors[limit])
    require(number(values['rt_factor_avg']) >= floors['rt_factor_min'], 'RTFx below locked floor')


def registry_models():
    registry = tomllib.loads((ROOT / 'src/models/manifest.toml').read_text())
    require(registry['profiles']['balanced'] == {
        'segmenter': 'powerset_int8', 'embedder': 'resnet34_int8'}, 'balanced profile changed')
    models = registry['models']
    return {key: models[key] for key in ['powerset_int8', 'resnet34_int8'] +
            sorted(k for k in models if k.startswith('vbx_plda_'))}


def notsofar():
    spec = importlib.util.spec_from_file_location('notsofar', ROOT / 'scripts/notsofar-eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(args, env=None):
    return subprocess.check_output(args, cwd=ROOT, env=env, text=True).strip()


def clean_revision():
    require(not command(['git', 'status', '--porcelain', '--untracked-files=normal']),
            'measurement requires a clean committed tree and ignored output directory')
    return command(['git', 'rev-parse', 'HEAD'])


def validate_inputs(name, folder):
    entries = read(MANIFEST)[name]
    coverage([p.stem for p in (folder / 'audio').glob('*.wav')], [e['id'] for e in entries])
    for entry in entries:
        for kind, suffix in [('audio', 'wav'), ('rttm', 'rttm')]:
            path = folder / kind / (entry['id'] + '.' + suffix)
            require(path.stat().st_size == entry[kind]['size'] and sha(path) == entry[kind]['sha256'],
                    f'corpus checksum mismatch: {path}')


def check_bench(report, name, revision, target):
    require(report['git_sha'] == revision, 'bench source revision mismatch')
    system, machine = target.split('-')
    require(report['host_os'] == ('macos' if system == 'Darwin' else 'linux') and report['host_arch'] ==
            ('aarch64' if machine == 'arm64' else machine), 'bench platform mismatch')
    require(report['profile'] == 'balanced' and report['resolved_execution_provider'] == 'Cpu'
            and report['collar_secs'] == 0 and report['skip_overlap'] is False, 'bench protocol mismatch')
    # The collector retains argv separately; the bench also records the invoked command.
    require(' '.join(BENCH_ARGS) in report['command_line'], 'bench command mismatch')
    ids = [e['id'] for e in read(MANIFEST)[name]]
    coverage([f['filename'] for f in report['per_file']], ids)
    require(report['files_processed'] == len(ids) and report['files_skipped'] == 0,
            'partial bench result')
    expected = registry_models()
    hashes = report['model_hashes']
    require(len(hashes) == 2 and {m['model_id']: m['sha256'] for m in hashes} ==
            {k: expected[k]['sha256'] for k in ('powerset_int8', 'resnet34_int8')}, 'bench model mismatch')
    for row in report['per_file']:
        number(row['der_no_collar'])
    for metric in ('der_no_collar_micro', 'der_no_collar_macro', 'rt_factor_avg'):
        number(report[metric])
    macro = sum(f['der_no_collar'] for f in report['per_file']) / len(ids)
    require(math.isclose(macro, report['der_no_collar_macro'], abs_tol=1e-8), 'macro inconsistent with files')
    if name != 'native-vox3':
        key = name.replace('-', '_') + '_linux_cpu_native'
        baseline = read(ROOT / 'tests/der_baseline.json')[key]
        if target == 'Darwin-arm64':
            baseline = {**baseline, **read(ROOT / 'benchmarks/results/darwin-native-der-2026-09-22' / (name + '.json'))}
        for metric in ('der_no_collar_micro', 'der_no_collar_macro'):
            maximum(report[metric], baseline[metric] + baseline['tolerance'])


def rss_mib(log):
    matches = re.findall(r'^\s*(\d+)\s+maximum resident set size\s*$', log, re.MULTILINE)
    require(len(matches) == 1 and int(matches[0]) > 0, 'missing/ambiguous Darwin peak RSS')
    return int(matches[0]) / 1024 ** 2


def validate_host(host, target):
    require(host['system'] + '-' + host['machine'] == target, 'host platform mismatch')
    for key in ('cpu', 'platform', 'rustc', 'python', 'hostname'):
        require(isinstance(host[key], str) and host[key].strip(), f'missing host {key}')
    require(type(host['cpu_count']) is int and host['cpu_count'] > 0, 'invalid CPU count')
    if target == 'Darwin-arm64':
        require(host['isolated'] is True and host['cpu'] == 'Apple M1 Pro',
                'scoreboard requires the isolated Apple M1 Pro host')
        require(host['runner_environment'] in ('local', 'self-hosted'), 'shared CI cannot certify scoreboard')


def verify_platform(directory, revision, target):
    evidence = read(directory / 'evidence.json')
    require(evidence['schema'] == 1 and evidence['status'] == 'passed', 'missing successful evidence')
    require(evidence['revision'] == revision and evidence['clean_tree'] is True, 'stale/dirty evidence')
    require(evidence['protocol'] == PROTOCOL and evidence['build_command'] == BUILD, 'build/protocol mismatch')
    require(evidence['build_environment'] == {}, 'nondefault build flags')
    require(evidence['manifest_sha256'] == sha(MANIFEST), 'corpus manifest mismatch')
    require(evidence['model_hashes'] == {k: v['sha256'] for k, v in registry_models().items()}, 'model/PLDA mismatch')
    require(re.fullmatch('[0-9a-f]{64}', evidence['binary_sha256']) is not None, 'missing binary hash')
    validate_host(evidence['host'], target)
    require(datetime.fromisoformat(evidence['measured_at']).tzinfo is not None, 'missing measurement timezone')
    names = ['voxconverse-test', 'ami-test'] + (['native-vox3'] if target == 'Darwin-arm64' else [])
    expected_files = {name + '.json' for name in names}
    expected_files.add('native-vox3.log' if target == 'Darwin-arm64' else 'notsofar/report.json')
    require(set(evidence['reports']) == expected_files, 'incomplete evidence bundle')
    for name, digest in evidence['reports'].items():
        require(sha(directory / name) == digest, f'changed report: {name}')
    require(set(evidence['commands']) == set(names), 'missing reproduction commands')
    for name in names:
        cmd = evidence['commands'][name]
        require(isinstance(cmd, list) and len(cmd) == len(BENCH_ARGS) + 4
                and cmd[2:-2] == BENCH_ARGS and cmd[-2] == '--output', 'unexpected bench argv')
        report = read(directory / (name + '.json'))
        require(report['command_line'] == ' '.join(cmd), 'report/collector command mismatch')
        check_bench(report, name, revision, target)
    if target == 'Darwin-arm64':
        values = read(directory / 'native-vox3.json')
        values.update(model_bytes=evidence['model_bytes'], rss_mib=rss_mib((directory / 'native-vox3.log').read_text()))
        scoreboard(values)
    else:
        module = notsofar()
        report = read(directory / 'notsofar/report.json')
        require(report['revision'] == revision and report['status'] == 'passed', 'stale/failed NOTSOFAR')
        require(report['host']['platform'].startswith('Linux-') and report['host']['machine'] == 'x86_64',
                'NOTSOFAR platform mismatch')
        require(report['build_environment'] == {} and report['build_command'] ==
                ['cargo', 'build', '--locked', '--release', '--features', 'cli', '--bin', 'polyvoice'],
                'NOTSOFAR build mismatch')
        module.check_report(report, read(module.BASELINE)['notsofar_eval_native'], read(module.MANIFEST))
        for row in report['per_file']:
            require(sha(directory / 'notsofar' / (row['name'] + '.json')) == row['hypothesis_sha256'],
                    'NOTSOFAR hypothesis mismatch')
    return evidence


def collect(args):
    revision = clean_revision()
    target = platform.system() + '-' + platform.machine()
    require(target in PLATFORMS, 'measurement host must be Linux x86_64 or Darwin arm64')
    host = {'system': platform.system(), 'machine': platform.machine(), 'platform': platform.platform(),
            'cpu_count': os.cpu_count(), 'hostname': platform.node(), 'python': platform.python_version(),
            'rustc': command(['rustc', '-Vv']), 'isolated': args.isolated_host,
            'runner_environment': os.environ.get('RUNNER_ENVIRONMENT', 'local')}
    host['cpu'] = command(['sysctl', '-n', 'machdep.cpu.brand_string']) if target.startswith('Darwin') else next(
        (s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), '')
    validate_host(host, target)
    env = {k: v for k, v in os.environ.items() if not k.startswith('POLYVOICE_')}
    require(not any(env.get(k) for k in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS')),
            'unset custom Rust build flags for release measurements')
    env['CARGO_TARGET_DIR'] = str(ROOT / 'target')
    # Match ModelRegistry::default on the two qualified platforms.
    cache = (Path.home() / 'Library/Caches' if target.startswith('Darwin') else
             Path(env.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))) / 'polyvoice/models'
    models = registry_models()
    for key, entry in models.items():
        require(sha(cache / entry['filename']) == entry['sha256'], f'cached model/PLDA mismatch: {key}')
    env['POLYVOICE_VBX_PLDA_DIR'] = str(cache)
    names = ['voxconverse-test', 'ami-test'] + (['native-vox3'] if target.startswith('Darwin') else [])
    datasets = {name: args.data_root / name for name in names}
    datasets['native-vox3'] = ROOT / 'tests/data/native-vox3'
    for name in names:
        validate_inputs(name, datasets[name])
    if target.startswith('Linux'):
        module = notsofar()
        module.validate_dataset(read(module.MANIFEST), args.data_root / 'notsofar-eval')
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run(BUILD, cwd=ROOT, env=env, check=True)
    bench = ROOT / 'target/release/polyvoice-bench'
    evidence = {'schema': 1, 'revision': revision, 'clean_tree': True, 'protocol': PROTOCOL,
                'host': host, 'measured_at': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha(MANIFEST), 'binary_sha256': sha(bench),
                'build_command': BUILD, 'build_environment': {},
                'model_hashes': {k: sha(cache / v['filename']) for k, v in models.items()},
                'model_bytes': sum((cache / models[k]['filename']).stat().st_size for k in ('powerset_int8', 'resnet34_int8')),
                'reports': {}, 'commands': {}}
    for name in names:
        cmd = [str(bench), str(datasets[name]), *BENCH_ARGS, '--output', str(args.output / (name + '.json'))]
        evidence['commands'][name] = cmd
        with (args.output / (name + '.log')).open('w') as log:
            subprocess.run((['/usr/bin/time', '-l'] if name == 'native-vox3' else []) + cmd,
                           cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
        check_bench(read(args.output / (name + '.json')), name, revision, target)
        evidence['reports'][name + '.json'] = sha(args.output / (name + '.json'))
    if target.startswith('Darwin'):
        evidence['reports']['native-vox3.log'] = sha(args.output / 'native-vox3.log')
    else:
        subprocess.run([sys.executable, str(ROOT / 'scripts/notsofar-eval.py'), 'run', '--models', str(cache),
                        '--data', str(args.data_root / 'notsofar-eval'), '--output', str(args.output / 'notsofar')],
                       cwd=ROOT, env=env, check=True)
        evidence['reports']['notsofar/report.json'] = sha(args.output / 'notsofar/report.json')
    require(clean_revision() == revision, 'source changed during measurement')
    for name in names:
        validate_inputs(name, datasets[name])
    for key, entry in models.items():
        require(sha(cache / entry['filename']) == evidence['model_hashes'][key], 'model changed during measurement')
    evidence['status'] = 'passed'
    # Publish success only after all checks, including resource floors, pass.
    temporary = args.output / 'evidence.json'
    temporary.write_text(json.dumps(evidence, indent=2) + '\n')
    try:
        verify_platform(args.output, revision, target)
    except Exception:
        temporary.unlink()
        raise
    print(f'PASS {target} {revision}: {args.output}')


def verify_bundle(directory, revision):
    require(re.fullmatch('[0-9a-f]{40}', revision) is not None, 'expected full commit SHA')
    for target in PLATFORMS:
        verify_platform(directory / target, revision, target)
    print(f'PASS full release quality evidence: {revision}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    collect_parser = commands.add_parser('collect')
    collect_parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    collect_parser.add_argument('--output', type=Path, required=True)
    collect_parser.add_argument('--isolated-host', action='store_true', help='attest exclusive idle scoreboard host use')
    verify = commands.add_parser('verify')
    verify.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'collect':
        args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
        collect(args)
    else:
        verify_bundle(args.evidence, clean_revision())


if __name__ == '__main__':
    main()
