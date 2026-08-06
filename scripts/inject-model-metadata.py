#!/usr/bin/env python3
"""Inject polyvoice metadata_props into shipped ONNX models.

Self-describing models (sherpa-onnx pattern): each ONNX carries its runtime
geometry, adapter type, license, and provenance in `metadata_props` so the
loader can read config from the model instead of hard-coded stage defaults.

Usage:
    python3 scripts/inject-model-metadata.py [--models-dir models] [--dry-run]
    python3 scripts/inject-model-metadata.py --model models/silero_vad.onnx \\
        --props sample_rate=16000 adapter_type=silero version=1.0

After injection the ONNX bytes change, so you MUST:
  1. Recompute sha256 in src/models/manifest.toml
  2. Re-sign with minisign (scripts/sign-models.sh)

Release-key re-signing is owned by the model-signing release workflow — this
script never invents signatures. Local/dev keys are fine for iteration.

Requires: `pip install onnx` (already used by quantize/validate scripts).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import onnx
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: the `onnx` Python package is required "
        f"(pip install onnx). Import failed: {exc}\n"
    )
    sys.exit(2)


# Default props keyed by filename stem (without .onnx). Values mirror the
# schema-v2 fields in src/models/manifest.toml so binary and manifest agree.
DEFAULT_PROPS: Dict[str, Dict[str, str]] = {
    "silero_vad": {
        "model_type": "vad",
        "adapter_type": "silero",
        "version": "1.0",
        "sample_rate": "16000",
        "license": "MIT",
        "license_url": "https://github.com/snakers4/silero-vad/blob/bfdc0193023f121ea5b3cc7b176dbed570a68a59/LICENSE",
        "provenance": "snakers4/silero-vad upstream ONNX @ bfdc019 (v6.2 weights)",
    },
    "wespeaker_resnet34": {
        "model_type": "embedder",
        "adapter_type": "wespeaker-resnet34",
        "version": "1.0",
        "sample_rate": "16000",
        "embedding_dim": "256",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34",
        "provenance": "Wespeaker/wespeaker-voxceleb-resnet34 Hugging Face",
    },
    "powerset_fp32": {
        "model_type": "segmentation",
        "adapter_type": "powerset-v1",
        "version": "3.0",
        "sample_rate": "16000",
        "window_secs": "10.0",
        "hop_secs": "1.0",
        "num_speakers": "3",
        "license": "MIT",
        "license_url": "https://github.com/k2-fsa/sherpa-onnx",
        "provenance": "sherpa-onnx-pyannote-segmentation-3-0 (pyannote/segmentation-3.0)",
    },
    "cam_pp_fp32": {
        "model_type": "embedder",
        "adapter_type": "cam++",
        "version": "1.0",
        "sample_rate": "16000",
        "embedding_dim": "512",
        "license": "Apache-2.0",
        "license_url": "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus",
        "provenance": "Wespeaker/wespeaker-voxceleb-campplus Hugging Face",
    },
    "powerset_int8": {
        "model_type": "segmentation",
        "adapter_type": "powerset-v1",
        "version": "3.0-int8",
        "sample_rate": "16000",
        "window_secs": "10.0",
        "hop_secs": "1.0",
        "num_speakers": "3",
        "license": "MIT",
        "provenance": "INT8 quant of sherpa-onnx-pyannote-segmentation-3-0",
    },
    "cam_pp_int8": {
        "model_type": "embedder",
        "adapter_type": "cam++",
        "version": "1.0-int8",
        "sample_rate": "16000",
        "embedding_dim": "512",
        "license": "Apache-2.0",
        "provenance": "INT8 quant of Wespeaker CAM++",
    },
    "resnet34_int8": {
        "model_type": "embedder",
        "adapter_type": "wespeaker-resnet34",
        "version": "1.0-int8",
        "sample_rate": "16000",
        "embedding_dim": "256",
        "license": "Apache-2.0",
        "provenance": "INT8 quant of Wespeaker ResNet34",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_metadata_props(model: "onnx.ModelProto", props: Dict[str, str]) -> None:
    """Upsert key/value pairs into model.metadata_props."""
    existing = {p.key: i for i, p in enumerate(model.metadata_props)}
    for key, value in props.items():
        if key in existing:
            model.metadata_props[existing[key]].value = value
        else:
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value


def inject_one(path: Path, props: Dict[str, str], dry_run: bool) -> Optional[str]:
    if not path.is_file():
        print(f"  skip (missing): {path}")
        return None
    model = onnx.load(str(path))
    before = {p.key: p.value for p in model.metadata_props}
    set_metadata_props(model, props)
    after = {p.key: p.value for p in model.metadata_props}
    changed = before != after
    if dry_run:
        print(f"  dry-run: {path.name} props={props} changed={changed}")
        return None
    if not changed:
        print(f"  unchanged: {path.name}")
        return sha256_file(path)
    onnx.save(model, str(path))
    digest = sha256_file(path)
    print(f"  wrote: {path.name} sha256={digest}")
    print("  NOTE: re-sign with scripts/sign-models.sh and update manifest.toml sha256")
    return digest


def parse_props(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"prop must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing *.onnx (default: models/)",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Single ONNX path (overrides --models-dir batch mode)",
    )
    ap.add_argument(
        "--props",
        nargs="*",
        default=[],
        help="Override props as key=value (with --model)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes without modifying files",
    )
    args = ap.parse_args()

    if args.model is not None:
        stem = args.model.stem
        props = dict(DEFAULT_PROPS.get(stem, {}))
        props.update(parse_props(args.props))
        if not props:
            sys.stderr.write(
                f"ERROR: no props for {stem}; pass --props key=value ...\n"
            )
            return 1
        print(f"injecting into {args.model}")
        inject_one(args.model, props, args.dry_run)
        return 0

    models_dir: Path = args.models_dir
    if not models_dir.is_dir():
        sys.stderr.write(f"ERROR: models dir not found: {models_dir}\n")
        return 1

    print(f"injecting metadata_props under {models_dir} (dry_run={args.dry_run})")
    # Walk top-level and int8/ subdirectory.
    candidates = list(models_dir.glob("*.onnx")) + list(models_dir.glob("int8/*.onnx"))
    if not candidates:
        print(
            "  no .onnx files present — nothing to inject.\n"
            "  (This worktree often ships only .minisig sidecars; download models "
            "first, then re-run. Manifest schema-v2 fields already carry the same "
            "metadata as a fallback.)"
        )
        return 0

    for path in sorted(candidates):
        stem = path.stem
        props = DEFAULT_PROPS.get(stem)
        if props is None:
            print(f"  skip (no default props for stem {stem!r}): {path}")
            continue
        inject_one(path, props, args.dry_run)

    print(
        "\nFollow-up (do NOT invent signatures):\n"
        "  1. Update sha256 values in src/models/manifest.toml for rewritten files\n"
        "  2. Re-sign with: bash scripts/sign-models.sh\n"
        "  3. Release-key re-signing is a separate release workflow step\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
