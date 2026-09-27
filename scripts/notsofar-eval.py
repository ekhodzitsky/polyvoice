#!/usr/bin/env python3
"""Download and evaluate the frozen NOTSOFAR held-out split (stdlib only)."""
import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import shutil
import sys
import time
import tomllib
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import der

MANIFEST = ROOT / "benchmarks/manifests/notsofar-eval.json"
BASELINE = ROOT / "benchmarks/manifests/notsofar-eval-baseline.json"


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assets(manifest, directory):
    yield manifest['license'], directory / 'LICENSE.txt'
    for item in manifest['files']:
        yield item['audio'], directory / 'audio' / (item['id'] + '.wav')
        yield item['reference'], directory / 'gt' / (item['id'] + '.json')


def verify(entry, path):
    if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
        raise ValueError(f'checksum/size mismatch: {path}')


def download(manifest, directory):
    base = manifest['repository'] + '/resolve/' + manifest['revision'] + '/'

    def fetch(pair):
        entry, path = pair
        if path.exists():
            verify(entry, path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            with urllib.request.urlopen(base + entry['path'], timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out)
            verify(entry, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'downloaded {path.name}', flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, assets(manifest, directory)))


def validate_dataset(manifest, directory):
    ids = [f['id'] for f in manifest['files']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate meeting list')
    for entry, path in assets(manifest, directory):
        verify(entry, path)
    for folder, suffix in [('audio', '.wav'), ('gt', '.json')]:
        if {p.stem for p in (directory / folder).glob('*' + suffix)} != set(ids):
            raise ValueError(f'incomplete or extra {folder} coverage')


def reference_turns(path, duration):
    turns = []
    for segment in json.loads(path.read_text()):
        start, end = float(segment['start_time']), float(segment['end_time'])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= duration + 0.1):
            raise ValueError(f'invalid reference time in {path}')
        # Match the existing NOTSOFAR converter, without rounding through RTTM.
        if end - start >= 0.001:
            turns.append((start, end, str(segment['speaker_id'])))
    if not turns:
        raise ValueError(f'empty reference: {path}')
    return turns


def hypothesis_turns(result, duration):
    turns = []
    for turn in result['turns']:
        start, end = turn['time']['start'], turn['time']['end']
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.1):
            raise ValueError('invalid hypothesis time')
        speaker = turn['speaker']
        if type(speaker) is not int or not 0 <= speaker < result['num_speakers']:
            raise ValueError('invalid hypothesis speaker')
        turns.append((start, end, str(speaker)))
    # Empty output is scored as missed speech, never omitted from the aggregate.
    return turns


def check_report(report, baseline, manifest):
    expected = {f['id'] for f in manifest['files']}
    actual = [f['name'] for f in report['per_file']]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('incomplete or duplicate scored coverage')
    for key in ['manifest_sha256', 'scorer_sha256', 'protocol', 'model_hashes']:
        if report[key] != baseline[key]:
            raise ValueError(f'baseline protocol mismatch: {key}')
    tolerance = manifest['regression_tolerance_pp']
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid regression tolerance')
    for key in ['der_micro', 'der_macro']:
        value, reference = report[key], baseline[key]
        if not (math.isfinite(value) and math.isfinite(reference) and value >= 0 and reference >= 0):
            raise ValueError(f'invalid {key}')
        if value > reference + tolerance:
            raise ValueError(f'{key} {value:.6f}% exceeds {reference + tolerance:.6f}%')


def command(args, env=None):
    return subprocess.check_output([str(x) for x in args], cwd=ROOT, env=env, text=True).strip()


def measure(manifest, args):
    validate_dataset(manifest, args.data)
    if command(['git', 'status', '--porcelain', '--untracked-files=normal']):
        raise ValueError('commit source changes before measuring; use an ignored output directory')
    models = tomllib.loads((ROOT / 'src/models/manifest.toml').read_text())['models']
    model_ids = ['powerset_int8', 'resnet34_int8'] + sorted(k for k in models if k.startswith('vbx_plda_'))
    hashes = {}
    for key in model_ids:
        path = args.models / models[key]['filename']
        hashes[key] = sha(path)
        if hashes[key] != models[key]['sha256']:
            raise ValueError(f'model checksum mismatch: {key}')
    env = {k: v for k, v in os.environ.items() if not k.startswith('POLYVOICE_')}
    env['CARGO_TARGET_DIR'] = str(ROOT / 'target')
    build = ['cargo', 'build', '--locked', '--release', '--features', 'cli', '--bin', 'polyvoice']
    subprocess.run(build, cwd=ROOT, env=env, check=True)
    binary = ROOT / 'target/release' / ('polyvoice.exe' if os.name == 'nt' else 'polyvoice')
    args.output.mkdir(parents=True, exist_ok=False)
    version = command([binary, '--version'], env)
    report = {'schema_version': 1, 'revision': command(['git', 'rev-parse', 'HEAD']),
              'version': version, 'manifest_sha256': sha(MANIFEST), 'scorer_sha256': sha(ROOT / 'benchmarks/der.py'),
              'binary_sha256': sha(binary), 'model_hashes': hashes, 'protocol': manifest['protocol'],
              'host': {'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(),
                       'cpu': command(['sysctl', '-n', 'machdep.cpu.brand_string']) if sys.platform == 'darwin' else platform.processor(),
                       'python': platform.python_version(), 'rustc': command(['rustc', '-Vv'])},
              'command': [sys.executable, *sys.argv], 'build_command': build,
              'build_environment': {k: env[k] for k in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS'] if k in env},
              'per_file': []}
    if sys.platform.startswith('linux'):
        report['host']['cpu'] = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), '')
    scores = der.DatasetScore(collar=0, skip_overlap=False)
    started = time.monotonic()
    for item in manifest['files']:
        mid = item['id']
        wav = args.data / 'audio' / (mid + '.wav')
        with wave.open(str(wav)) as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
                raise ValueError(f'expected mono PCM16 16kHz: {wav}')
            duration = stream.getnframes() / 16000
        reference = reference_turns(args.data / 'gt' / (mid + '.json'), duration)
        cmd = [str(binary), 'diarize', str(wav), '--models-cache', str(args.models), '--vbx-plda-dir', str(args.models), '--json']
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(f'{mid} inference failed: {result.stderr}')
        output = json.loads(result.stdout)
        if output['provenance']['version'] != version.removeprefix('polyvoice '):
            raise ValueError('inference version mismatch')
        (args.output / (mid + '.json')).write_text(result.stdout)
        score = der.score_file(reference, hypothesis_turns(output, duration), collar=0, skip_overlap=False)
        if score.scored_ref <= 0:
            raise ValueError(f'no scored reference speech: {mid}')
        score.name = mid
        scores.files.append(score)
        report['per_file'].append({**dataclasses.asdict(score), 'der': score.der, 'audio_seconds': duration,
                                   'command': cmd, 'hypothesis_sha256': sha(args.output / (mid + '.json'))})
        print(f'{len(scores.files)}/{len(manifest["files"])} {mid}: DER {score.der:.3f}%', flush=True)
    report.update(der_micro=scores.der_micro, der_macro=scores.der_macro,
                  components_micro=scores.decomposition_micro(), speaker_count=scores.speaker_count_accuracy(),
                  elapsed_seconds=time.monotonic() - started, audio_seconds=sum(f['audio_seconds'] for f in report['per_file']))
    report['status'] = 'measured (baseline establishment only)'
    if not args.record_only:
        check_report(report, json.loads(BASELINE.read_text()), manifest)
        report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'der_micro', 'der_macro', 'components_micro', 'speaker_count']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'run'])
    parser.add_argument('--data', type=lambda p: Path(p).resolve(), default=ROOT / 'data/notsofar-eval')
    parser.add_argument('--models', type=lambda p: Path(p).resolve(), default=Path.home() / '.cache/polyvoice/models')
    parser.add_argument('--output', type=lambda p: Path(p).resolve(), default=ROOT / 'bench-results/notsofar-eval')
    parser.add_argument('--record-only', action='store_true', help='establish a baseline; never reports a passing gate')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    protocol = {'collar_seconds': 0.0, 'score_overlap': True, 'frame_seconds': 0.01}
    if len(manifest['files']) != 129 or any(manifest['protocol'].get(k) != v for k, v in protocol.items()):
        raise ValueError('unexpected frozen corpus protocol')
    if args.action == 'download':
        download(manifest, args.data)
        validate_dataset(manifest, args.data)
    else:
        measure(manifest, args)


if __name__ == '__main__':
    main()
