# Agent quickstart

The fastest paths to machine-readable "who spoke when" (and "who said what")
from polyvoice, for AI agents and scripted integrations. Every command below
keeps stdout clean: the result is the only thing printed there; progress and
logs go to stderr.

Docs index: [`docs/README.md`](../docs/README.md) · schema:
[`schema/diarization-result-v1.json`](../schema/diarization-result-v1.json) ·
C embedding: [`docs/FFI.md`](../docs/FFI.md).

## 1. CLI: one-shot JSON

```sh
cargo install polyvoice --features cli   # or grab a release binary
polyvoice download-models --profile balanced
polyvoice diarize meeting.wav --json > result.json
```

`--json` implies `--format json --quiet`: stdout carries exactly one canonical
`diarization-result-v1` document. Other projections: `--format rttm|srt|vtt|txt`.

The JSON contract is versioned and printable:

```sh
polyvoice schema > diarization-result-v1.schema.json
```

## 2. MCP server (Claude Code, other MCP clients)

```json
{
  "mcpServers": {
    "polyvoice": { "command": "polyvoice-mcp" }
  }
}
```

Build with `cargo install polyvoice --features mcp`. The server exposes
diarization as tools over stdio; results follow the same v1 schema.

## 3. Python: typed result with projections

```sh
pip install polyvoice
```

```python
import polyvoice

pipeline = polyvoice.Pipeline.balanced()
result = pipeline.run_result(samples, 16000)   # list[float] mono 16 kHz

result.num_speakers          # int
result.turns                 # [{"speaker": 0, "start": 0.5, "end": 2.8, "text": None}, ...]
result.to_json()             # canonical diarization-result-v1 document
result.to_rttm("meeting1")   # RTTM with your file id
result.to_srt(); result.to_vtt(); result.to_txt()

# Re-hydrate a saved result without re-running the pipeline:
saved = polyvoice.DiarizationResult.from_json(open("result.json").read())
```

`pipeline.run()` (plain dict) remains available and unchanged.

## 4. C FFI: pick your output format

```c
#include "polyvoice.h"

char *out = NULL; size_t out_len = 0;
int rc = polyvoice_pipeline_run_format(pipe, samples, n_samples, 16000,
                                       POLYVOICE_FORMAT_RTTM, &out, &out_len);
/* POLYVOICE_FORMAT_JSON / _RTTM / _SRT / _VTT / _TXT */
polyvoice_free_string(out, out_len);
```

See `examples/ffi_usage.c` for the create/run/destroy lifecycle.

## 5. Who said what (diarization + transcription)

`polyvoice-asr` is not on crates.io yet — install from source:

```sh
git clone https://github.com/ekhodzitsky/polyvoice
cargo install --path polyvoice/polyvoice-asr --features cli
polyvoice-transcribe meeting.wav --asr-model /path/to/parakeet --format json
```

JSON output carries speaker turns with `text` plus a per-word array of
`{"word", "time": {"start", "end"}, "speaker", "confidence"}` objects.
`--format srt|vtt|txt` produce subtitle/transcript projections of the same
result.
