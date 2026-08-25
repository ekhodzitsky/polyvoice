# Contributing conventions (humans & agents)

## No internal task references in shipped artifacts

Never put task numbers, issue indices, audit finding IDs, or any internal
tracker reference (e.g. `task 300`, `F09`, `B-1`, `#142`) into anything that
ships or is read outside the tracker:

- source comments and doc-comments
- commit messages
- pull-request titles or descriptions
- shipped data/config (`tests/der_baseline.json` status strings, manifests, etc.)
- user-facing docs and the README

A future reader has no access to the local tracker and these references are
noise to them. Describe the *what* and *why* in plain terms instead.

- Bad:  `// Calibrated for task 310.`
- Good: `// Pruning singleton clusters cuts over-clustering without hurting DER.`
- Bad:  `fix: address F09 across modules`
- Good: `fix: validate input ranges before model download`

The local `roadmap/` tracker is the *only* place task numbers belong. Real
external identifiers that aren't internal indices are fine where relevant
(e.g. CVE / RUSTSEC IDs, a published security-advisory ID, an upstream issue URL).

## Commit / PR trailers

Do not add AI-attribution trailers or footers anywhere: no `Co-Authored-By`
lines for AI tools, no "Generated with …" footers in commit messages, PR
descriptions, comments, or docs. Write commit messages and PR text as plain
engineering prose.

## Locked native scoreboard floors

`cli-native` on the Vox-3 protocol (euqef / fuzfh / msbyq, collar 0, v2+VBx,
balanced, `powerset_int8` + `resnet34_int8`) has locked floors in
`tests/native_scoreboard.json`. **No characteristic may get worse** than
these, including when adding a faster kernel path:

| Characteristic | Floor | Direction |
|---|---|---|
| DER₀ micro | 7.11 % | never higher |
| DER₀ macro | 7.39 % | never higher |
| Real-time factor | 117× | never lower |
| On-disk INT8 pair | 8 414 314 bytes | never larger |
| Peak process RSS | 556 MiB | never higher |

The RSS floor is below live ort INT8 CPU on the same protocol (~580–585 MiB).
A change that is faster or more accurate but uses more than 556 MiB peak RSS
is not acceptable: keep the win and cut memory.

Product default is `cli` = kernels (`pipeline-native`), no `libonnxruntime`.
ONNX Runtime is opt-in (`cli-ort` / `onnx`). Darwin native holds the
scoreboard floors above. Linux native holds AMI DER within the ort
ceiling; RTF there is ~28× on a Vox-3 smoke (still below the old ort
band of ~82× / ~95×) and is not a reason to pull `ort` back into `cli`.
