"""Installed-wheel consumer; inputs are raw little-endian f32, not source models."""
import array
import json
from pathlib import Path
import sys
import polyvoice

models, pcm, scenario = sys.argv[1:]
samples = array.array('f')
with Path(pcm).open('rb') as stream:
    samples.fromfile(stream, Path(pcm).stat().st_size // 4)
if sys.byteorder != 'little':
    samples.byteswap()
pipeline = polyvoice.Pipeline.balanced(models, vbx_plda_dir=models)
try:
    result = pipeline.run_result(samples, 8000 if scenario == 'invalid-rate' else 16000)
except (ValueError, OSError) as error:
    expected = {'invalid-rate': 'unsupported sample rate', 'too-long': 'audio too long', 'empty': 'audio too short', 'short': 'audio too short'}
    error_type = OSError if scenario in ('empty', 'short') else ValueError
    if not isinstance(error, error_type) or scenario not in expected or expected[scenario] not in str(error):
        raise
    print(json.dumps({'error': scenario}))
else:
    if scenario in ('invalid-rate', 'too-long', 'empty', 'short'):
        raise AssertionError('invalid input unexpectedly accepted')
    print(result.to_json())
