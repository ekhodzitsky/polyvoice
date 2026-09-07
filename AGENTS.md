agent-dev-kb: 0.8.0

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
scoreboard floors above. Linux x86_64 native, full-split 2026-09-07
(Ryzen AI 9 HX 370): VoxConverse-test DER₀ 15.40 % / RTFx ~110×,
AMI DER₀ 25.50 % / RTFx ~113×, Vox-3 smoke RTFx ~95×. Same-host ort is
still faster on raw RTF (Vox-3 ~104×, AMI ~145× — the older ~82×/~95×
band was weaker hardware); kernels win on RSS (543 vs 578 MiB), size and
zero dylib. INT8 conv defaults on x86_64 with AVX-512 VNNI (exact integer
math, same as the aarch64 SDOT path); CPUs without it keep FP32.

## Backlog.md

This repo uses Backlog.md (not GitHub Issues, not a second spec tool).
Run `backlog instructions overview` before work. Mutate via the CLI
(`--plain` / `--json`). One task = one session = one PR.
Docs → `doc create`; conclusions → `decision create`; work →
`task create`. Labels: `audit`, `research`.

## Coding principles

Karpathy-inspired. Caution over speed; skip ceremony on one-liners.
The queue is Backlog.md. No second spec or task framework.

1. **Think before coding.** State assumptions. Show interpretations
   when the request is ambiguous. Push back if something simpler
   exists. Stop and ask when unclear.
2. **Simplicity first.** No extra features, single-use abstractions,
   or speculative config. If 200 lines could be 50, rewrite.
3. **Surgical changes.** Touch only the requested lines. Match the
   surrounding style. Mention unrelated dead code; do not delete it.
   Remove only your own unused imports.
4. **Goal-driven.** Tests for invalid input, bug repro, and
   before-and-after. Every step has a verify. "Make it work" is not
   a criterion.
5. **Always.** Run this repo's checks before claiming done. No new
   dependency without a why and its transitive cost. No secrets, no
   tracker ids. Comments and commits follow the repo language, else
   English.

### Rust

No `unwrap` / `expect` on production paths. Honor clippy (often
`-D warnings`). Write the failing test first when that area is
already tested.

### Minimal code
<!-- Ladder adapted from ponytail (MIT), github.com/DietrichGebert/ponytail -->
Stop at the first rung that holds:
1. Needed at all? No: skip it (YAGNI).
2. Already in this codebase? Reuse, do not rewrite.
3. Stdlib does it? Use it.
4. Native platform feature? Use it.
5. Installed dependency? Use it.
6. One line? One line.
7. Only then: the minimum that works.
Never cut trust-boundary validation, data-loss handling, security,
or accessibility.
