#!/usr/bin/env python3
"""Merge polyvoice-bench JSON shard reports into one duration-weighted micro report."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def wavg(rows: list[dict], key: str, wkey: str = "audio_duration_secs") -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        w = float(r.get(wkey) or 0.0)
        if w <= 0:
            continue
        num += float(r[key]) * w
        den += w
    return (num / den) if den else 0.0


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: merge_shard_reports.py OUT.json shard1.json [shard2.json ...]",
            file=sys.stderr,
        )
        return 2
    out = Path(sys.argv[1])
    shards = [Path(p) for p in sys.argv[2:]]
    per_file: list[dict] = []
    base: dict | None = None
    skipped = 0
    for p in shards:
        data = json.loads(p.read_text())
        if base is None:
            base = data
        per_file.extend(data.get("per_file") or [])
        skipped += int(data.get("files_skipped") or 0)

    # de-dupe by filename (last wins)
    by_name: dict[str, dict] = {}
    for row in per_file:
        by_name[row["filename"]] = row
    rows = sorted(by_name.values(), key=lambda r: r["filename"])

    assert base is not None
    merged = dict(base)
    merged["files_processed"] = len(rows)
    merged["files_skipped"] = skipped
    merged["per_file"] = rows
    merged["der_no_collar_micro"] = wavg(rows, "der_no_collar")
    merged["der_collar_micro"] = wavg(rows, "der_collar")
    merged["der_no_collar_macro"] = (
        sum(float(r["der_no_collar"]) for r in rows) / len(rows) if rows else 0.0
    )
    merged["der_collar_macro"] = (
        sum(float(r["der_collar"]) for r in rows) / len(rows) if rows else 0.0
    )
    merged["miss"] = wavg(rows, "miss_rate")
    merged["false_alarm"] = wavg(rows, "false_alarm_rate")
    merged["confusion"] = wavg(rows, "confusion_rate")
    merged["rt_factor_avg"] = (
        sum(float(r.get("rt_factor") or 0.0) for r in rows) / len(rows) if rows else 0.0
    )
    # mark merge provenance
    merged["command_line"] = (merged.get("command_line") or "") + " | merged_shards"
    out.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(
        f"merged {len(rows)} files -> {out} "
        f"no_collar_micro={merged['der_no_collar_micro']:.4f} "
        f"collar_micro={merged['der_collar_micro']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
