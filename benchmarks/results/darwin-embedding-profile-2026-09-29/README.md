# Apple Rust embedding CPU profile

The repeated CPU profiles locate the experimental Rust embedding cost in
stride-one INT8 convolution: `s1_scan_row_zip` contains 86.0–86.6% of sampled
embedding CPU weight. Its SDOT kernel/lane operations account for 58.3–61.1%
of embedding leaf samples, and memory copies account for 11.1–12.9%.
Keep the backend experimental. The smallest next experiment is to reduce
intermediate KN-to-zip copies while preserving the current layout and exact
integer results, before changing kernel arithmetic or thread defaults.

## Source and measurement conditions

Clean source: `0ed55f75df9fe2ba504c5e4c5ab1877de4e90f39`.
M1 Pro, 10 cores, 16 GiB, Darwin 25.1.0, Rust 1.98.0.
Use the same hash-verified models, six PLDA files and pinned Vox-3 inputs
as the [earlier comparison](../darwin-rust-comparison-2026-09-29/README.md).
The later master commits only add documentation/results, so the existing
release binaries were reused by hash instead of rebuilding them.

Separate profiling targets used release optimization, LTO and one codegen
unit, with only `CARGO_PROFILE_RELEASE_DEBUG=1` and
`CARGO_PROFILE_RELEASE_STRIP=false` overridden. Xcode Time Profiler
26.0 (17C52) captured three independent full-pipeline process launches per
backend, with 1 ms running-thread CPU samples and no sleeping-thread samples.
Each launch completed normally before its 15-second limit. Symbolized builds
and profiler overhead are **not** used to judge the speed/RSS floors.

AC power, low-power mode off and no thermal warnings were recorded. The
machine was not quiescent: pre-series CPU idle samples were 61.51%, 72.70%
and 67.31%. A trace-export transfer also overlapped the beginning of the
unprofiled series. No compilation or recording overlapped those runs.
Retain this interference as a limitation; the data does not certify a
quiet-host maximum or make a small speed shortfall intrinsic to the kernels.

## CPU attribution

| Rust profile | CPU rows | Missing stacks | Embedding CPU weight, ms | SDOT kernel + lanes, ms | Copy leaf weight, ms |
|---|---:|---:|---:|---:|---:|
| 1 | 3,956 | 11 | 1,804 | 1,103 | 200 |
| 2 | 4,052 | 10 | 1,912 | 1,141 | 219 |
| 3 | 3,883 | 9 | 1,800 | 1,050 | 233 |

Weights sum running samples across CPU cores. They are not elapsed time.
The embedding denominator includes stacks with a `ResNet34Native::embed*`
ancestor; it excludes model construction. Inclusive percentages overlap
their child functions and must not be added to leaf percentages.

The dominant observed stack is:

```text
ResNet34Native::embed_batch
  embed_prepared -> ResNet34::embed_fbank -> resnet34::run_layers
    conv_i8::conv3x3_s1_rows -> s1_scan -> s1_scan_row_zip
      kernel_4x16_zip_store / sdot_lane
```

Copy samples show `s1_scan_row_zip -> copy_nonoverlapping<i8> ->
_platform_memmove`; these are concrete input-panel costs, not an inference
from aggregate stage time. Other observed work includes gather/packing and
`tensor::neon_add_relu_quantize` (56–57 ms leaf weight per Rust profile).
Inlining spreads attribution between helpers and their parents, so standalone
helper totals are not a complete accounting of all packing/requantization.
No thread/mutex leaf samples occurred inside the identified embedding stacks.
Because waiting threads were excluded, that does not prove scheduling or
waiting costs are zero and does not justify changing thread counts.

`sdot-disassembly.txt` confirms a register-resident 16-accumulator SDOT inner
loop. Accumulators are stored to the stack after that loop for the output
epilogue; this is not evidence of per-iteration accumulator spills. The
kernel includes output conversion/requantization, so its inclusive cost
cannot all be called arithmetic throughput. Hardware counter/cache-miss
measurements were not collected.

The product profiles contain 3,502 / 3,473 / 3,519 CPU rows and 9 / 10 / 7
missing stacks. Embedding-parent CPU weights are 837 / 796 / 806 ms.
BNNS frames account for 854 / 779 / 818 ms across all stacks; of these,
223 / 204 / 225 ms occur without an embedding parent (dispatch workers).
Private BNNS routines lack symbols, so unresolved frames are retained as
library-relative offsets. Do not compare the embedding-parent totals as if
they included every asynchronous product worker. Rust has one unresolved
leaf row in each full-process profile; its kernel symbols are usable.

## Independent unprofiled release results

One retained warm-up per backend precedes five paired measurements with
alternating order and ten seconds between processes. The original stripped
release binaries are unchanged; their hashes are in `provenance.json`.

| Run | Product RTFx | Product RSS MiB | Rust RTFx | Rust RSS MiB |
|---|---:|---:|---:|---:|
| Warm-up | 166.624 | 456.094 | 106.118 | 427.203 |
| 1 | 168.787 | 453.844 | 121.900 | 442.984 |
| 2 | 157.242 | 454.109 | 116.778 | 443.359 |
| 3 | 163.720 | 464.891 | 116.191 | 448.281 |
| 4 | 162.771 | 456.547 | 111.994 | 442.250 |
| 5 | 163.062 | 437.797 | 122.882 | 454.125 |

DER micro/macro is constant within each backend: product
7.108843537% / 7.388943724%; Rust 7.006802721% / 7.329480110%.
Every run uses the same 8,414,314-byte INT8 pair. The product passes all five
limits in every run. Rust passes accuracy, size and memory throughout, but
misses 117× in three of five measured runs and its warm-up. No metrics from
different runs are combined, and no experimental report qualifies a release.

## Retained evidence and reproduction

- `provenance.json`: source/toolchain, hashes, build flags, linkage, profiler
  commands, run order and limitations.
- `profiles-summary.json`: per-profile coverage, leaf/inclusive attribution,
  unresolved symbols and raw XML/normalized stack hashes.
- `*-stacks.json.gz`: complete normalized sampled stacks with weights;
  device IDs, process IDs and local binary paths are omitted.
- `*-profiled-*.json` and `*-trace-*.log`: instrumented pipeline reports and
  recording logs, retained for coverage/quality checks only.
- `unprofiled/`: all twelve original benchmark reports and resource logs,
  the measurement summary and collector. Process inventories stay local.

Native `.trace` bundles and original exported XML remain on the Mac under
`~/src/polyvoice-darwin-rust-comparison/bench-results/darwin-embedding-profile`;
the XML is also retained in ignored local `bench-results/darwin-embedding-profile`.
Normalized stacks preserve every nonempty exported backtrace and sample
weight. Missing stacks are reported separately.

To reproduce, use the clean source and verified cache on the same Mac. Build
each backend into the targets recorded in `provenance.json`, adding the two
profiling overrides above to its recorded release build command. Run the
record/export command templates for each backend and repetition. The templates
run from the checkout with a clean environment, the usual Rust/system PATH
and `POLYVOICE_VBX_PLDA_DIR` pointing to the verified model cache.
Run `python3 summarize.py /path/to/exported-xml /path/to/summary` on Linux
with the installed GNU `c++filt -s rust`; this uses Python stdlib and adds no
project dependency. Keep raw traces local when sharing normalized results.

For independent release repetition, `unprofiled/collect.py build` creates
fresh uninstrumented targets; after the host settles,
`caffeinate -i python3 /path/to/unprofiled/collect.py measure` runs the series.
This session instead reused the earlier verified release targets through
symlinks in its ignored output directory and copied their provenance. The
collector refuses to overwrite previous measurements.

## Next bounded experiment

Investigate whether `gather_kn_from_rows` plus `pack_kn_zip16` can avoid an
intermediate copy in the existing stride-one path. The profiles justify
examining that path; they do not prove a particular copy is redundant or that
fusion will improve cache behavior. Preserve tail/padding semantics, panel
layout and bit-exact integer output. Use existing numerical tests and
representative convolution shapes, then verify all five unchanged floors and
held-out quality before considering promotion. No optimization, threading
change, dependency addition or new default is part of this profiling result.
