#!/usr/bin/env python3
"""Score hyp RTTM dirs against a reference rttm/ with der.py."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import der


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="directory of reference .rttm")
    ap.add_argument("--hyp", required=True, nargs="+", help="hypothesis dirs (name=last path component)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = []
    for hyp in args.hyp:
        hyp = os.path.abspath(hyp)
        name = os.path.basename(hyp.rstrip("/"))
        n = len([f for f in os.listdir(hyp) if f.endswith(".rttm")])
        c0 = der.score_dataset(args.ref, hyp, collar=0.0)
        c25 = der.score_dataset(args.ref, hyp, collar=0.25)
        d0 = c0.decomposition_micro()
        d25 = c25.decomposition_micro()
        row = {
            "name": name,
            "files": len(c0.files),
            "rttm_on_disk": n,
            "der_nocollar_micro": round(c0.der_micro, 2),
            "der_nocollar_macro": round(c0.der_macro, 2),
            "der_collar_micro": round(c25.der_micro, 2),
            "der_collar_macro": round(c25.der_macro, 2),
            "decomp_nocollar": {k: round(v, 2) for k, v in d0.items()},
            "decomp_collar": {k: round(v, 2) for k, v in d25.items()},
            "speaker_count": c0.speaker_count_accuracy(),
        }
        rows.append(row)
        print(
            f"{name:<28} n={row['files']:<4} DER0={row['der_nocollar_micro']:6.2f} "
            f"DER0.25={row['der_collar_micro']:6.2f} "
            f"miss={d0['miss']:.2f} fa={d0['false_alarm']:.2f} conf={d0['confusion']:.2f} "
            f"spk={row['speaker_count']}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"runners": rows}, indent=2))
        print(f"→ {args.output}")


if __name__ == "__main__":
    main()
