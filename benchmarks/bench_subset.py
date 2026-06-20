#!/usr/bin/env python3
"""Reproducible sharded-subset DER runner for polyvoice's own pipelines.

Measures `polyvoice-bench` on a DETERMINISTIC subset of a split (seeded slice,
gigastt-style), sharded across cores for speed, and merges the per-file results
into macro DER + bootstrap CI + decomposition + speaker-count accuracy.

Why macro: sharding splits the corpus across processes, and the bench JSON
exposes per-file DER but not per-file scored-reference seconds, so the
frame-weighted micro average cannot be reconstituted exactly across shards. Macro
(mean of per-file DER) IS exact under sharding, and is reported with a 95%
bootstrap CI over the per-file values.

    python bench_subset.py --split ../data/voxconverse-test --n 60 --seed 42 \
        --shards 5 --label v2ahc_test --out /tmp/v2ahc_test.json \
        -- --pipeline v2 --clusterer ahc --min-cluster-size 1 --collar 0.25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCH = os.path.join(os.path.dirname(HERE), "target", "release", "polyvoice-bench")


def pick_subset(split: str, n: int, seed: int) -> list[str]:
    wavs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(split, "audio", "*.wav"))}
    rttms = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(split, "rttm", "*.rttm"))}
    ids = sorted(wavs & rttms)
    if n and 0 < n < len(ids):
        # Deterministic LCG shuffle (seed) → first n. No numpy dependency.
        rng = seed & ((1 << 64) - 1)
        order = []
        pool = list(ids)
        while pool:
            rng = (rng * 6364136223846793005 + 1) & ((1 << 64) - 1)
            order.append(pool.pop((rng >> 32) % len(pool)))
        ids = sorted(order[:n])
    return ids


def make_shards(split: str, ids: list[str], k: int, root: str) -> list[str]:
    dirs = []
    for i in range(k):
        shard_ids = ids[i::k]
        if not shard_ids:
            continue
        d = os.path.join(root, f"shard_{i}")
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)
        os.makedirs(os.path.join(d, "rttm"), exist_ok=True)
        for fid in shard_ids:
            for sub, ext in (("audio", "wav"), ("rttm", "rttm")):
                src = os.path.abspath(os.path.join(split, sub, f"{fid}.{ext}"))
                dst = os.path.join(d, sub, f"{fid}.{ext}")
                if os.path.exists(src) and not os.path.exists(dst):
                    os.symlink(src, dst)
        dirs.append(d)
    return dirs


def bootstrap_ci(values: list[float], iterations: int = 1000) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng, mask = 123456789, (1 << 64) - 1
    means = []
    for _ in range(iterations):
        s = 0.0
        for _ in range(n):
            rng = (rng * 6364136223846793005 + 1) & mask
            s += values[(rng >> 32) % n]
        means.append(s / n)
    means.sort()
    return (means[(iterations * 25) // 1000], means[(iterations * 975) // 1000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=0, help="subset size (0 = all files)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default=DEFAULT_BENCH)
    ap.add_argument("bench_args", nargs=argparse.REMAINDER, help="-- <args passed to polyvoice-bench>")
    args = ap.parse_args()
    bench_args = args.bench_args[1:] if args.bench_args and args.bench_args[0] == "--" else args.bench_args

    ids = pick_subset(args.split, args.n, args.seed)
    print(f"[{args.label}] {len(ids)} files (n={args.n or 'all'}, seed={args.seed}), {args.shards} shards")
    work = tempfile.mkdtemp(prefix=f"pvsub_{args.label}_")
    shard_dirs = make_shards(args.split, ids, args.shards, work)

    procs = []
    for i, d in enumerate(shard_dirs):
        out_i = os.path.join(work, f"shard_{i}.json")
        cmd = [args.bench, d, "--profile", "balanced", *bench_args, "--output", out_i]
        procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), out_i))
    per_file = []
    for p, out_i in procs:
        p.wait()
        if p.returncode == 0 and os.path.isfile(out_i):
            per_file.extend(json.load(open(out_i)).get("per_file", []))
        else:
            print(f"   ! shard failed (rc={p.returncode})")

    n = len(per_file)
    if n == 0:
        print("no results"); sys.exit(1)
    mean = lambda key: sum(f[key] for f in per_file) / n
    der_c = [f["der_collar"] for f in per_file]
    der_n = [f["der_no_collar"] for f in per_file]
    sc = {"exact": 0, "plus_minus_1": 0, "off_by_2_or_more": 0, "files": n}
    for f in per_file:
        d = abs(f.get("hyp_speakers", 0) - f.get("ref_speakers", 0))
        sc["exact" if d == 0 else "plus_minus_1" if d == 1 else "off_by_2_or_more"] += 1
    clo, chi = bootstrap_ci(der_c)
    nlo, nhi = bootstrap_ci(der_n)
    result = {
        "label": args.label, "n_files": n, "subset_n": args.n, "seed": args.seed,
        "bench_args": bench_args,
        "der_collar_macro": round(sum(der_c) / n, 2), "der_collar_ci95": [round(clo, 2), round(chi, 2)],
        "der_no_collar_macro": round(sum(der_n) / n, 2), "der_no_collar_ci95": [round(nlo, 2), round(nhi, 2)],
        "miss": round(mean("miss_rate"), 2), "false_alarm": round(mean("false_alarm_rate"), 2),
        "confusion": round(mean("confusion_rate"), 2),
        "speaker_count": sc,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[{args.label}] collar macro={result['der_collar_macro']}% CI{result['der_collar_ci95']}  "
          f"no-collar macro={result['der_no_collar_macro']}% CI{result['der_no_collar_ci95']}  "
          f"miss={result['miss']} fa={result['false_alarm']} conf={result['confusion']}  spk={sc}")


if __name__ == "__main__":
    main()
