#!/usr/bin/env python3
"""Convert NOTSOFAR-1 gt_transcription.json files to RTTM, one per meeting.

Input layout (produced by download-notsofar.sh):
    <data-dir>/gt/MTG_xxxxx.json   — list of {"speaker_id", "start_time",
                                     "end_time", ...}
Output:
    <data-dir>/rttm/MTG_xxxxx.rttm

Adjacent segments from the same speaker are kept as-is (GT granularity);
segments shorter than 1 ms or with end <= start are dropped.
"""
import json
import sys
from pathlib import Path


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/notsofar-dev")
    gt_dir = data_dir / "gt"
    rttm_dir = data_dir / "rttm"
    rttm_dir.mkdir(parents=True, exist_ok=True)

    for gt_path in sorted(gt_dir.glob("*.json")):
        segments = json.loads(gt_path.read_text())
        lines = []
        for seg in segments:
            start = float(seg["start_time"])
            end = float(seg["end_time"])
            dur = end - start
            if dur < 0.001:
                continue
            speaker = str(seg["speaker_id"]).replace(" ", "_")
            lines.append(
                f"SPEAKER {gt_path.stem} 1 {start:.3f} {dur:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>"
            )
        (rttm_dir / f"{gt_path.stem}.rttm").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
