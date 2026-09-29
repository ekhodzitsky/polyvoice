"""Summarize exported xctrace time-profile samples; requires installed GNU c++filt."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

folder = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
reports = []
for path in sorted(folder.glob('*-samples-*.xml')):
    root = ET.parse(path).getroot()
    ids = {e.get('id'): e for e in root.iter() if e.get('id')}
    def resolve(element):
        return ids[element.get('ref')] if element.get('ref') else element
    symbols = sorted({e.get('name') for e in root.iter('frame') if e.get('name')})
    decoded = subprocess.check_output(['c++filt', '-s', 'rust'], input='\n'.join(symbols) + '\n', text=True).splitlines()
    assert len(decoded) == len(symbols)
    names = dict(zip(symbols, [re.sub(r'\[[0-9a-f]+\]', '', s) for s in decoded]))
    for frame in root.iter('frame'):
        name = frame.get('name', '')
        binary = frame.find('binary')
        if name.startswith('0x') and binary is not None:
            binary = resolve(binary)
            offset = int(name, 16) - int(binary.get('load-addr'), 16)
            names[name] = '<unresolved ' + binary.get('name') + '+' + hex(offset) + '>'
    leaves, inclusive, embed_leaves, embed_inclusive, stacks = (Counter() for _ in range(5))
    rows = missing = unknown_leaf = 0
    total_weight = missing_weight = bnns_weight = bnns_async_weight = 0
    weights = Counter()
    for row in root.iter('row'):
        rows += 1
        assert resolve(row.find('thread-state')).get('fmt') == 'Running'
        weight = int(resolve(row.find('weight')).text)
        weights[weight] += 1
        total_weight += weight
        bt = row.find('backtrace')
        if bt is None:
            missing += 1
            missing_weight += weight
            continue
        frames = tuple(names[resolve(e).get('name')] for e in resolve(bt))
        assert frames
        stacks[frames] += weight
        leaves[frames[0]] += weight
        unknown_leaf += frames[0].startswith(('0x', '<unresolved '))
        for name in set(frames):
            inclusive[name] += weight
        is_embed = any('ResNet34Native' in name and '>::embed' in name for name in frames)
        bnns = any(resolve(resolve(e).find('binary')).get('name') == 'libBNNS.dylib'
                   for e in resolve(bt) if resolve(e).find('binary') is not None)
        if bnns:
            bnns_weight += weight
            if not is_embed:
                bnns_async_weight += weight
        if is_embed:
            embed_leaves[frames[0]] += weight
            for name in set(frames):
                embed_inclusive[name] += weight
    def table(counter, limit=None):
        return [{'symbol': s, 'sample_weight_ms': w / 1e6} for s, w in counter.most_common(limit)]
    stem = path.stem.replace('-samples-', '-')
    # Normalized stacks omit process/device identifiers and local binary paths.
    payload = json.dumps([{'frames_leaf_first': list(s), 'weight_ns': w} for s, w in stacks.items()], separators=(',', ':')).encode()
    packed = gzip.compress(payload, mtime=0)
    (output / (stem + '-stacks.json.gz')).write_bytes(packed)
    report = {'profile': stem, 'raw_xml_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'stacks_sha256': hashlib.sha256(packed).hexdigest(), 'rows': rows, 'missing_stack_rows': missing,
              'unknown_leaf_rows': unknown_leaf, 'sample_weights_ns': dict(weights),
              'total_sample_weight_ms': total_weight / 1e6, 'missing_stack_weight_ms': missing_weight / 1e6,
              'embedding_sample_weight_ms': sum(embed_leaves.values()) / 1e6,
              'bnns_sample_weight_ms': bnns_weight / 1e6, 'bnns_without_embedding_root_ms': bnns_async_weight / 1e6,
              'top_leaves': table(leaves, 25), 'embedding_leaves': table(embed_leaves),
              'embedding_inclusive': table(Counter({k:v for k,v in embed_inclusive.items() if any(t in k for t in ['conv_i8', 'resnet34', 'fbank', 'bnns', 'BNNS', 'gemm', 'pthread', 'intra::'])}))}
    reports.append(report)
    print(stem, rows, 'samples;', sum(embed_leaves.values())/1e6, 'ms embedding')
    print(table(embed_leaves, 6))
(output / 'profiles-summary.json').write_text(json.dumps(reports, indent=2) + '\n')
