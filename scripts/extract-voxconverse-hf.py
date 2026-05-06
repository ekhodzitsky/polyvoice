#!/usr/bin/env python3.11
"""Extract VoxConverse test set audio from HuggingFace to WAV files (streaming mode)."""

import os
import sys
import soundfile as sf
from datasets import load_dataset

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/voxconverse-test/audio"
os.makedirs(OUT_DIR, exist_ok=True)

print("Streaming VoxConverse test set from HuggingFace...")
ds = load_dataset("diarizers-community/voxconverse", split="test", streaming=True)

count = 0
for i, row in enumerate(ds):
    audio = row["audio"]
    name = os.path.splitext(os.path.basename(audio["path"]))[0] if audio.get("path") else f"test_{i:04d}"
    wav_path = os.path.join(OUT_DIR, f"{name}.wav")
    if os.path.exists(wav_path):
        print(f"  [skip] {name}")
        count += 1
        continue
    sf.write(wav_path, audio["array"], audio["sampling_rate"])
    dur = len(audio["array"]) / audio["sampling_rate"]
    print(f"  [{i+1}] {name} ({dur:.1f}s, sr={audio['sampling_rate']})")
    count += 1

print(f"\nDone: {count} files in {OUT_DIR}/")
