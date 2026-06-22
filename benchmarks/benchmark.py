#!/usr/bin/env python3
"""Cross-engine diarization benchmark driver.

Runs each available engine over a dataset, writes per-file hypothesis RTTMs
(cached), and scores every engine through the single :mod:`der` scorer at both
the 0.25 s collar and no-collar — so all rows are like-for-like. Competitors that
are not installed are skipped (their published numbers are cited in the docs).

    python benchmark.py --dataset voxconverse_test --runners all
    python benchmark.py --dataset voxconverse_dev --runners polyvoice --collar 0.25
    python benchmark.py --dataset ami_test --runners polyvoice,pyannote --no-cache

Results land in ``results/<dataset>__<timestamp>.json`` and per-engine hypothesis
RTTMs under ``results_full/<dataset>/<engine>/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import wave

import der
from runners import (DiartRunner, PolyvoiceRunner, PyannoteRunner,
                     SherpaOnnxRunner, WhisperXRunner)

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFESTS = os.path.join(HERE, "manifests")
RESULTS = os.path.join(HERE, "results")
HYP_ROOT = os.path.join(HERE, "results_full")


def build_runners() -> dict:
    return {
        "polyvoice": PolyvoiceRunner(variant="legacy"),
        "polyvoice-v2": PolyvoiceRunner(variant="v2"),
        "pyannote": PyannoteRunner(),
        "whisperx": WhisperXRunner(),
        "sherpa-onnx": SherpaOnnxRunner(),
        "diart": DiartRunner(),
    }


def audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def load_manifest(name: str) -> dict:
    path = os.path.join(MANIFESTS, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_engine(key: str, runner, manifest: dict, max_files: int, no_cache: bool) -> dict | None:
    if not runner.is_available():
        print(f"[skip] {runner.name} — not available")
        return None
    repo = os.path.dirname(HERE)

    def _resolve(p: str) -> str:
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.join(repo, p)

    audio_root = _resolve(manifest["audio_root"])
    rttm_root = _resolve(manifest["rttm_root"])
    ids = manifest["ids"][:max_files] if max_files else manifest["ids"]
    hyp_dir = os.path.join(HYP_ROOT, manifest["dataset"], key)
    os.makedirs(hyp_dir, exist_ok=True)

    proc_total = audio_total = 0.0
    failures = 0
    print(f"[run] {runner.name}: {len(ids)} files")
    for i, fid in enumerate(ids):
        wav = os.path.join(audio_root, f"{fid}.wav")
        hyp_path = os.path.join(hyp_dir, f"{fid}.rttm")
        if os.path.isfile(hyp_path) and not no_cache:
            audio_total += audio_duration(wav)
            continue
        try:
            turns, elapsed = runner.diarize(wav)
            from runners.base import write_rttm
            write_rttm(turns, fid, hyp_path)
            proc_total += elapsed
            audio_total += audio_duration(wav)
        except Exception as e:
            failures += 1
            print(f"   ! {fid}: {e}")
            from runners.base import write_rttm
            write_rttm([], fid, hyp_path)  # empty hyp ⇒ counts as full miss
        if (i + 1) % 25 == 0:
            print(f"   {i + 1}/{len(ids)}")

    collar = manifest.get("_collar", 0.25)
    scored = der.score_dataset(rttm_root, hyp_dir, collar=collar)
    nocollar = der.score_dataset(rttm_root, hyp_dir, collar=0.0)
    lo, hi = scored.bootstrap_ci()
    nlo, nhi = nocollar.bootstrap_ci()
    rtf = (proc_total / audio_total) if audio_total > 0 else None
    return {
        "name": runner.name,
        "license": runner.license,
        "files": len(scored.files),
        "failures": failures,
        "der_collar_micro": round(scored.der_micro, 2),
        "der_collar_macro": round(scored.der_macro, 2),
        "der_collar_ci95": [round(lo, 2), round(hi, 2)],
        "der_nocollar_micro": round(nocollar.der_micro, 2),
        "der_nocollar_macro": round(nocollar.der_macro, 2),
        "der_nocollar_ci95": [round(nlo, 2), round(nhi, 2)],
        "decomposition_collar": {k: round(v, 2) for k, v in scored.decomposition_micro().items()},
        "speaker_count": scored.speaker_count_accuracy(),
        "rtf": round(rtf, 4) if rtf is not None else None,
        "realtime_factor": round(1.0 / rtf, 1) if rtf else None,
        "collar": collar,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="voxconverse_test")
    ap.add_argument("--runners", default="polyvoice", help='comma list or "all"')
    ap.add_argument("--collar", type=float, default=0.25, help="primary collar (no-collar always also reported)")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.dataset)
    manifest["_collar"] = args.collar
    all_runners = build_runners()
    keys = list(all_runners) if args.runners == "all" else [k.strip() for k in args.runners.split(",")]

    results = []
    for key in keys:
        if key not in all_runners:
            print(f"[warn] unknown runner {key!r}; known: {', '.join(all_runners)}")
            continue
        r = run_engine(key, all_runners[key], manifest, args.max_files, args.no_cache)
        if r:
            results.append(r)

    report = {
        "schema": "polyvoice-diarization-benchmark-v1",
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
                 "python": platform.python_version()},
        "dataset": {k: manifest.get(k) for k in ("dataset", "source", "license", "files")},
        "primary_collar": args.collar,
        "runners": results,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = args.output or os.path.join(RESULTS, f"{args.dataset}__latest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {args.dataset} (collar {args.collar}) ===")
    print(f"{'engine':<16}{'DER%':>8}{'no-collar':>11}{'miss':>7}{'fa':>7}{'conf':>7}{'spk-exact':>11}{'RTF':>8}")
    for r in results:
        d = r["decomposition_collar"]
        sc = r["speaker_count"]
        print(f"{r['name']:<16}{r['der_collar_micro']:>8}{r['der_nocollar_micro']:>11}"
              f"{d['miss']:>7}{d['false_alarm']:>7}{d['confusion']:>7}"
              f"{sc['exact']}/{sc['files']:<6}{str(r['rtf']):>8}")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
