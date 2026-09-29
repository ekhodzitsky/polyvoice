"""Compare fixed release binaries; run from the clean candidate checkout on M1 Pro."""
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
out = root / 'bench-results/fused-packing/comparison'
out.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location('quality', root / 'scripts/release-quality.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
old = Path.home() / 'src/polyvoice-darwin-rust-comparison'
env = {'HOME': str(Path.home()), 'USER': os.environ['USER'],
       'PATH': str(Path.home() / '.cargo/bin') + ':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
       'POLYVOICE_VBX_PLDA_DIR': str(Path.home() / 'Library/Caches/polyvoice/models')}
cache = Path(env['POLYVOICE_VBX_PLDA_DIR'])
def cmd(args, cwd=root):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()
def save(name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')
def state():
    return {'power': cmd(['pmset','-g','batt']), 'thermal': cmd(['pmset','-g','therm']),
            'cpu': '\n'.join(s for s in cmd(['top','-l','2','-s','1','-n','0']).splitlines() if s.startswith('CPU usage:'))}
models = q.registry_models()
hashes = {k:q.sha(cache / v['filename']) for k,v in models.items()}
assert hashes == {k:v['sha256'] for k,v in models.items()}
model_bytes = sum((cache / models[k]['filename']).stat().st_size for k in ['powerset_int8','resnet34_int8'])
variants = {
    'product': (old, old / 'bench-results/darwin-rust-comparison/product-target/release/polyvoice-bench', 'cli'),
    'before': (old, old / 'bench-results/darwin-rust-comparison/rust-target/release/polyvoice-bench', 'cli,experimental-darwin-rust'),
    'after': (root, root / 'bench-results/fused-packing/target/release/polyvoice-bench', 'cli,experimental-darwin-rust'),
}
p = {'date': datetime.now(timezone.utc).isoformat(), 'host':cmd(['uname','-a']),
     'hardware':cmd(['sysctl','machdep.cpu.brand_string','hw.ncpu','hw.memsize']),
     'rustc':cmd(['rustc','-Vv']), 'power_settings':cmd(['pmset','-g','custom']),
     'model_hashes': hashes, 'model_bytes':model_bytes, 'manifest_sha256':q.sha(q.MANIFEST), 'variants':{},
     'environment': env, 'initial_state':state(), 'runs':[]}
for label,(cwd,binary,features) in variants.items():
    assert cmd(['git','status','--porcelain'],cwd)==''
    q.validate_inputs('native-vox3',cwd/'tests/data/native-vox3')
    p['variants'][label]={'revision':cmd(['git','rev-parse','HEAD'],cwd),'binary':str(binary),
        'binary_sha256':q.sha(binary),'cwd':str(cwd),'features':features,
        'build_command':['cargo','build','--locked','--release','--no-default-features','--features',features,'--bin','polyvoice-bench']}
save('provenance.json',p)
schedule=[(v,'warmup') for v in variants]
# Every ordering of the three variants occurs once; no best-of selection.
for index, order in enumerate(itertools.permutations(variants),1):
    schedule.extend((v,str(index)) for v in order)
for label,index in schedule:
    name=label+'-'+index
    cwd,binary,_=variants[label]
    before=state()
    assert 'AC Power' in before['power']
    args=[str(binary),'tests/data/native-vox3',*q.BENCH_ARGS,'--output',str(out/(name+'.json'))]
    start=datetime.now(timezone.utc).isoformat()
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(['/usr/bin/time','-l',*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=q.read(out/(name+'.json'))
    q.check_bench(report,'native-vox3',p['variants'][label]['revision'],'Darwin-arm64')
    values={k:report[k] for k in ['der_no_collar_micro','der_no_collar_macro','rt_factor_avg']}
    values.update(model_bytes=model_bytes,rss_mib=q.rss_mib((out/(name+'.log')).read_text()))
    try: q.scoreboard(values); verdict='all floors pass'
    except ValueError as e: verdict=str(e)
    p['runs'].append({'name':name,'started_at':start,'command':args,'before':before,'metrics':values,'floor_verdict':verdict})
    save('provenance.json',p)
    print(name,json.dumps(values),verdict,flush=True)
    time.sleep(5)
for label,(cwd,binary,_) in variants.items():
    assert cmd(['git','status','--porcelain'],cwd)==''
    assert cmd(['git','rev-parse','HEAD'],cwd)==p['variants'][label]['revision']
    assert q.sha(binary)==p['variants'][label]['binary_sha256']
assert {k:q.sha(cache/v['filename']) for k,v in models.items()}==hashes
p['final_state']=state()
save('provenance.json',p)
