#!/usr/bin/env python3
"""Precompute the diagonalized PLDA parameters consumed by the VBx clusterer.

The VBx backend (`src/clusterer/{plda,vbx}.rs`) scores in a diagonalized PLDA
space. Computing that diagonalization is a one-time generalized symmetric
eigenproblem; doing it offline here keeps the Rust runtime pure-`ndarray`
(no eigendecomposition, no BLAS/LAPACK, wasm32-clean).

Provenance / license
--------------------
The raw PLDA parametrization (Kaldi `mu`/`tr`/`psi` + the LDA transform and the
two centering means) is derived from the `pyannote/speaker-diarization-community-1`
pipeline, which is licensed **CC-BY-4.0** (redistribution of derived parameters
is permitted with attribution to pyannote). It is matched to the
`pyannote/wespeaker-voxceleb-resnet34-LM` embedder — the same WeSpeaker ResNet34
(256-d) polyvoice uses. The raw `.npy` files are mirrored at the Apache-2.0
`avencera/speakrs` project's HF model repo `avencera/speakrs-models`
(`plda_{mean1,mean2,lda,mu,tr,psi}.npy`).

When shipping, attribute pyannote (CC-BY-4.0) for the PLDA parameters in the
model manifest / NOTICE.

Inputs  (in --in-dir): plda_mean1.npy plda_mean2.npy plda_lda.npy plda_mu.npy
                       plda_tr.npy plda_psi.npy
Outputs (in --out-dir): plda_transform.npy  plda_phi_computed.npy
        (plus copies of mean1/mean2/lda/mu — the full set the Rust loader reads)

Optional parity validation (--fixtures-dir with the speakrs pipeline fixtures
pipeline_train_embeddings.npy / pipeline_plda_phi.npy / pipeline_plda_features.npy)
confirms the transform reproduces the reference to ~1e-3.

Usage:
    python3 scripts/build-vbx-plda.py --in-dir RAW --out-dir OUT [--fixtures-dir FIX]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def build(in_dir: Path, out_dir: Path):
    mean1 = np.load(in_dir / "plda_mean1.npy").astype(np.float64)  # (256,)
    mean2 = np.load(in_dir / "plda_mean2.npy").astype(np.float64)  # (128,)
    lda = np.load(in_dir / "plda_lda.npy").astype(np.float64)      # (256,128)
    mu = np.load(in_dir / "plda_mu.npy").astype(np.float64)        # (128,)
    tr = np.load(in_dir / "plda_tr.npy").astype(np.float64)        # (128,128) raw transform
    psi = np.load(in_dir / "plda_psi.npy").astype(np.float64)      # (128,)

    # Two-covariance PLDA -> diagonalized space (see BUTSpeechFIT/VBx).
    precision = np.linalg.inv(tr.T @ tr)
    tr_over_psi = tr.T / psi[None, :]
    between = np.linalg.inv(tr_over_psi @ tr)

    # Generalized symmetric-definite eigenproblem between @ v = lam * precision @ v,
    # via Cholesky reduction of the SPD `precision` (no scipy needed).
    chol = np.linalg.cholesky(precision)
    chol_inv = np.linalg.inv(chol)
    reduced = chol_inv @ between @ chol_inv.T
    reduced = 0.5 * (reduced + reduced.T)
    eigvals, eigvecs = np.linalg.eigh(reduced)        # ascending
    gen_vecs = chol_inv.T @ eigvecs

    dim = lda.shape[1]                                # 128
    order = np.argsort(eigvals)[::-1][:dim]           # top `dim`, descending
    phi = eigvals[order]
    transform = gen_vecs[:, order].T                  # (dim, 128): row d = eigenvector

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "plda_transform.npy", transform)
    np.save(out_dir / "plda_phi_computed.npy", phi)
    for name in ("plda_mean1", "plda_mean2", "plda_lda", "plda_mu"):
        shutil.copyfile(in_dir / f"{name}.npy", out_dir / f"{name}.npy")
    return mean1, mean2, lda, mu, transform, phi


def validate(out_dir: Path, fixtures_dir: Path) -> bool:
    mean1 = np.load(out_dir / "plda_mean1.npy").astype(np.float64)
    mean2 = np.load(out_dir / "plda_mean2.npy").astype(np.float64)
    lda = np.load(out_dir / "plda_lda.npy").astype(np.float64)
    mu = np.load(out_dir / "plda_mu.npy").astype(np.float64)
    transform = np.load(out_dir / "plda_transform.npy").astype(np.float64)
    phi = np.load(out_dir / "plda_phi_computed.npy").astype(np.float64)

    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    def transform_emb(emb):
        xv = l2(emb - mean1) * np.sqrt(lda.shape[0])
        xv = l2(xv @ lda - mean2) * np.sqrt(lda.shape[1])
        return (xv - mu) @ transform.T

    emb = np.load(fixtures_dir / "pipeline_train_embeddings.npy").astype(np.float64)
    exp_phi = np.load(fixtures_dir / "pipeline_plda_phi.npy").astype(np.float64)
    exp_feat = np.load(fixtures_dir / "pipeline_plda_features.npy").astype(np.float64)

    feat = transform_emb(emb)
    phi_err = float(np.max(np.abs(phi - exp_phi)))
    sign = np.sign(np.sum(feat * exp_feat, axis=0))
    sign[sign == 0] = 1.0
    feat_err = float(np.max(np.abs(feat * sign[None, :] - exp_feat)))
    print(f"parity: phi_err={phi_err:.2e}  feat_err={feat_err:.2e}")
    return phi_err < 1e-3 and feat_err < 1e-2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path, help="dir with raw plda_*.npy")
    ap.add_argument("--out-dir", required=True, type=Path, help="dir to write precomputed params")
    ap.add_argument("--fixtures-dir", type=Path, default=None, help="optional speakrs fixtures for parity")
    args = ap.parse_args()

    build(args.in_dir, args.out_dir)
    print(f"wrote precomputed PLDA params to {args.out_dir}")
    if args.fixtures_dir is not None:
        ok = validate(args.out_dir, args.fixtures_dir)
        print("VALIDATION:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
