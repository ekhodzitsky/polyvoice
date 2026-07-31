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
