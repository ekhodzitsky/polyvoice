#!/usr/bin/env python3
"""Generate committed dataset manifests from the local `data/` splits.

A manifest lists the file ids present in BOTH ``audio/`` and ``rttm/`` for a
split, plus provenance. Paths are stored relative to the repo root so the
committed manifest is host-independent; the audio itself is NOT redistributed
(download it with the scripts in ``scripts/``). Re-run after refreshing a split:

    python benchmarks/make_manifests.py
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "manifests")

SPLITS = {
    "voxconverse_dev": {
        "dir": "data/voxconverse-dev",
        "source": "https://github.com/joonson/voxconverse",
        "license": "VoxConverse annotations CC-BY-4.0; audio from YouTube (not redistributed)",
    },
    "voxconverse_test": {
        "dir": "data/voxconverse-test",
        "source": "https://github.com/joonson/voxconverse",
        "license": "VoxConverse annotations CC-BY-4.0; audio from YouTube (not redistributed)",
    },
    "ami_test": {
        "dir": "data/ami-test",
        "source": "https://groups.inf.ed.ac.uk/ami/corpus/",
        "license": "AMI Meeting Corpus CC-BY-4.0",
    },
}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, spec in SPLITS.items():
        base = os.path.join(REPO, spec["dir"])
        audio_dir = os.path.join(base, "audio")
        rttm_dir = os.path.join(base, "rttm")
        if not os.path.isdir(audio_dir):
            print(f"[skip] {name}: {audio_dir} missing")
            continue
        wavs = {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(audio_dir, "*.wav"))}
        rttms = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(rttm_dir, "*.rttm"))}
        ids = sorted(wavs & rttms)
        manifest = {
            "dataset": name,
            "audio_root": os.path.join(spec["dir"], "audio"),
            "rttm_root": os.path.join(spec["dir"], "rttm"),
            "source": spec["source"],
            "license": spec["license"],
            "files": len(ids),
            "ids": ids,
        }
        path = os.path.join(OUT, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[ok] {name}: {len(ids)} files (audio∩rttm) → {path}")


if __name__ == "__main__":
    main()
