# Apple Rust kernel comparison

Source: `0ed55f75df9fe2ba504c5e4c5ab1877de4e90f39` (clean).
Host: Apple M1 Pro, 10 cores, 16 GiB, Darwin 25.1.0, Rust 1.98.0.
Protocol: committed Vox-3 (`euqef`, `fuzfh`, `msbyq`), balanced, v2 + VBx,
CPU, collar zero, overlap scored, one file worker. Both variants use the
same registry INT8 model pair and six PLDA files.

`provenance.json` records source, toolchain, build commands, distinct target
directories, binary/model/PLDA hashes, model bytes and actual Mach-O linkage.
The product imports Accelerate; the Rust variant imports only libSystem and
libiconv. The benchmark CLI includes downloader dependencies; this is not
the separate downloader-free build audit.

`product-*.json` and `rust-*.json` are unmodified benchmark reports. Matching
logs retain `/usr/bin/time -l` output, including maximum resident set size
in bytes. `measurements.json` records commands, timestamps, power/thermal
state and individual floor verdicts. Raw process inventories are retained
locally and omitted from this public summary. Build logs are also retained
locally. One warm-up per backend was followed by three paired runs:
product/rust, rust/product, product/rust, with ten seconds between processes.

| Run | DER micro % | DER macro % | RTFx | RSS MiB | All floors |
|---|---:|---:|---:|---:|---|
| Product warm-up | 7.108844 | 7.388944 | 157.313 | 441.844 | pass |
| Rust warm-up | 7.006803 | 7.329480 | 112.682 | 450.422 | fail: speed |
| Product 1 | 7.108844 | 7.388944 | 161.108 | 444.328 | pass |
| Rust 1 | 7.006803 | 7.329480 | 115.756 | 448.750 | fail: speed |
| Rust 2 | 7.006803 | 7.329480 | 118.864 | 443.766 | pass |
| Product 2 | 7.108844 | 7.388944 | 154.523 | 440.188 | pass |
| Product 3 | 7.108844 | 7.388944 | 163.693 | 439.672 | pass |
| Rust 3 | 7.006803 | 7.329480 | 115.722 | 450.516 | fail: speed |

Every run used the same 8,414,314-byte model pair. Validation checked exact
three-file coverage, no skipped files, source revision, CPU/protocol,
model hashes, macro consistency and finite metrics with the existing
`scripts/release-quality.py` helpers. All five floors were evaluated for
each run separately; passing values from different runs were not combined.

AC power and low-power mode off were confirmed; macOS reported no thermal
or performance warnings. The user closed foreground workloads, but some
background agent/system activity remained (pre-series CPU idle samples
were approximately 83–87%). This is a same-host research comparison, not
a fully quiescent-host certification. Do not infer that a roughly 1% speed
shortfall is intrinsic to the backend. The Rust variant remains experimental
because this series does not consistently satisfy the unchanged floors.

Median stage times suggest profiling embeddings next: Rust embedding
0.638 s versus product 0.337 s; segmentation 0.267 s versus 0.307 s.
Do not infer a specific kernel cause from these coarse measurements or
generalize the three-file DER difference to held-out corpora.

To reproduce, use a clean checkout of the source revision above on the
same Mac with the verified models in the default cache. From that checkout,
run the retained `collect.py` by absolute path with Python 3.11 or newer:
first `python3 /path/to/collect.py build`, then, after the host settles on
AC power, `caffeinate -i python3 /path/to/collect.py measure`. It writes to a
fresh `bench-results/darwin-rust-comparison` directory and refuses to
overwrite existing evidence. The collector records process inventories;
omit those inventories when publishing its measurement summary.

See the [experiment documentation](../../../docs/darwin-rust-experiment.md)
for feature semantics, dependency accounting and promotion requirements.
These historical reports do not qualify later source revisions or replace
the product release evidence gate.
