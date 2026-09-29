"""Run from a clean polyvoice checkout; build first, then measure on an idle AC-powered M1 Pro."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

root = Path.cwd()
out = root / 'bench-results/darwin-embedding-profile/unprofiled'
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('quality', root / 'scripts/release-quality.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
env = {'HOME': str(Path.home()), 'USER': os.environ['USER'],
       'PATH': str(Path.home() / '.cargo/bin') + ':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'}
cache = Path.home() / 'Library/Caches/polyvoice/models'
env['POLYVOICE_VBX_PLDA_DIR'] = str(cache)
revision = q.clean_revision()
q.validate_inputs('native-vox3', root / 'tests/data/native-vox3')
models = q.registry_models()
hashes = {k: q.sha(cache / v['filename']) for k, v in models.items()}
q.require(hashes == {k: v['sha256'] for k, v in models.items()}, 'cached model mismatch')
model_bytes = sum((cache / models[k]['filename']).stat().st_size for k in ('powerset_int8', 'resnet34_int8'))
def command(args):
    return subprocess.check_output(args, env=env, text=True).strip()
def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')
def state():
    return {k: command(v) for k, v in {
        'power': ['pmset', '-g', 'batt'], 'thermal': ['pmset', '-g', 'therm'],
        'uptime': ['uptime'], 'process_cpu': ['ps', '-Ao', 'pid,pcpu,comm']}.items()}
if sys.argv[1] == 'build':
    q.require(not (out / 'provenance.json').exists(), 'existing evidence; use a fresh output directory')
    provenance = {'revision': revision, 'clean_tree': True, 'model_hashes': hashes, 'model_bytes': model_bytes,
                  'manifest_sha256': q.sha(q.MANIFEST), 'environment': env,
                  'rustc': command(['rustc', '-Vv']), 'host': command(['uname', '-a']),
                  'hardware': command(['sysctl', 'machdep.cpu.brand_string', 'hw.ncpu', 'hw.memsize']),
                  'power_settings': command(['pmset', '-g', 'custom']), 'builds': {}}
    for backend, features in [('product', 'cli'), ('rust', 'cli,experimental-darwin-rust')]:
        target = out / (backend + '-target')
        build_env = {**env, 'CARGO_TARGET_DIR': str(target)}
        cmd = ['cargo', 'build', '--locked', '--release', '--no-default-features', '--features', features,
               '--bin', 'polyvoice-bench']
        print('Building', backend, flush=True)
        with (out / (backend + '-build.log')).open('w') as log:
            subprocess.run(cmd, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = target / 'release/polyvoice-bench'
        provenance['builds'][backend] = {'command': cmd, 'target': str(target), 'binary_sha256': q.sha(binary),
                                        'linkage': command(['otool', '-L', str(binary)])}
        print('Built', backend, flush=True)
    save('provenance.json', provenance)
else:
    p = q.read(out / 'provenance.json')
    q.require(p['revision'] == revision and p['model_hashes'] == hashes, 'provenance changed')
    q.require('AC Power' in command(['pmset', '-g', 'batt']), 'AC power required')
    for backend in ('product', 'rust'):
        q.require(q.sha(out / (backend + '-target/release/polyvoice-bench')) == p['builds'][backend]['binary_sha256'], 'binary changed')
    # One retained warm-up per backend, then five paired runs with alternating order.
    schedule = [('product', 'warmup'), ('rust', 'warmup')]
    for index in range(1, 6):
        schedule += [(b, str(index)) for b in (('product', 'rust') if index % 2 else ('rust', 'product'))]
    results = []
    for backend, label in schedule:
        name = backend + '-' + label
        q.require(not (out / (name + '.json')).exists(), 'refusing to overwrite run')
        before = state()
        q.require('AC Power' in before['power'], 'AC power required')
        cmd = [str(out / (backend + '-target/release/polyvoice-bench')), 'tests/data/native-vox3',
               *q.BENCH_ARGS, '--output', str(out / (name + '.json'))]
        started = datetime.now(timezone.utc).isoformat()
        with (out / (name + '.log')).open('w') as log:
            subprocess.run(['/usr/bin/time', '-l', *cmd], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = q.read(out / (name + '.json'))
        q.check_bench(report, 'native-vox3', revision, 'Darwin-arm64')
        values = {k: report[k] for k in ('der_no_collar_micro', 'der_no_collar_macro', 'rt_factor_avg')}
        values.update(model_bytes=model_bytes, rss_mib=q.rss_mib((out / (name + '.log')).read_text()))
        try:
            q.scoreboard(values)
            verdict = 'all floors pass'
        except ValueError as error:
            verdict = str(error)
        result = {'name': name, 'started_at': started, 'command': cmd, 'before': before, 'after': state(),
                  'metrics': values, 'floor_verdict': verdict}
        results.append(result)
        save('measurements.json', results)
        print(name, json.dumps(values), verdict, flush=True)
        time.sleep(10)
    q.require(q.clean_revision() == revision, 'source changed during measurement')
    q.require({k: q.sha(cache / v['filename']) for k, v in models.items()} == hashes, 'models changed during measurement')
