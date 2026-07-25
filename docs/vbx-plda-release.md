# Shipping the VBx PLDA weights

The `vbx` clusterer (PLDA + VB-HMM) is the most accurate v2 path for
overlap-heavy / meeting audio. Measured (collar 0.25, macro DER, same code path):

| set | legacy | v2 + AHC | v2 + VBx |
|-----|--------|----------|----------|
| VoxConverse-30 | 13.5% | 16.4% | 15.1% |
| AMI-4 (overlap) | 35.5% | 32.8% | **27.8%** |

VBx beats cosine AHC everywhere (VoxConverse −1.3pp, AMI −5pp, lower confusion)
and makes v2+VBx the best option for meetings (AMI −7.8pp vs legacy). On clean
conversational audio legacy is still best. VBx is ~2–3× slower than AHC.

## What ships

Six precomputed PLDA `.npy` files (the `PldaModel::from_dir` set), ~265 KB total:

| file | sha256 | bytes |
|------|--------|-------|
| plda_transform.npy   | `90261469714415743f4b8a86ee6b89466db858bde3c5944367cccfb7abd34f14` | 131200 |
| plda_phi_computed.npy | `6ef7cf2f5a23a45b66f440f9a996a4cf5c047b369829af695d50ef18aa0a35e3` | 1152 |
| plda_mean1.npy       | `e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f` | 2176 |
| plda_mean2.npy       | `6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf` | 640 |
| plda_lda.npy         | `e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b` | 131200 |
| plda_mu.npy          | `d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6` | 1152 |

License: **CC-BY-4.0**, attribution to pyannote (see `NOTICE`). Matched to the
`wespeaker_resnet34` embedder polyvoice already ships.

Manifest model ids (not profile-resolved):

| model id | filename |
|----------|----------|
| `vbx_plda_transform` | `plda_transform.npy` |
| `vbx_plda_phi_computed` | `plda_phi_computed.npy` |
| `vbx_plda_mean1` | `plda_mean1.npy` |
| `vbx_plda_mean2` | `plda_mean2.npy` |
| `vbx_plda_lda` | `plda_lda.npy` |
| `vbx_plda_mu` | `plda_mu.npy` |

Host URLs point at the commit-pinned fixtures under
`https://raw.githubusercontent.com/ekhodzitsky/polyvoice/0623ae0b6773db7731f0fb3e75d8950347be94db/fixtures/vbx-plda/`.
Integrity is **SHA-256 only** until minisign signatures are added (same optional
model pattern as `sortformer_v2` / `eres2netv2`). Profile resolution never
pulls these entries, so release builds do not require signatures for them.

## Rebuilding the weights (reproducible)

```sh
# 1. raw inputs (Apache-2.0 mirror of pyannote community-1 params)
mkdir -p data/vbx-plda-raw
for f in plda_mean1 plda_mean2 plda_lda plda_mu plda_tr plda_psi; do
  curl -fsSL "https://huggingface.co/avencera/speakrs-models/resolve/main/$f.npy" \
    -o "data/vbx-plda-raw/$f.npy"
done
# 2. precompute the diagonalized PLDA (numpy only)
python3 scripts/build-vbx-plda.py --in-dir data/vbx-plda-raw --out-dir data/vbx-plda
```

## Using it

Default CLI path (`--clusterer vbx`) resolves PLDA in this order:

1. `--vbx-plda-dir` if set
2. `POLYVOICE_VBX_PLDA_DIR` if set
3. Model registry auto-download (`ModelRegistry::ensure_vbx_plda_dir` →
   `VbxClusterer::from_registry`) into the user cache

```sh
# Zero setup (downloads ~265 KB once into the models cache):
cargo run --features cli --bin polyvoice -- diarize meeting.wav

# Explicit local dir (CI / offline):
cargo run --features cli --bin polyvoice -- diarize meeting.wav \
  --vbx-plda-dir fixtures/vbx-plda
# or: POLYVOICE_VBX_PLDA_DIR=fixtures/vbx-plda ...
```

Checked-in fixtures under `fixtures/vbx-plda/` keep the release gate and DER
regression tests offline-capable without hitting the network.

Hyperparameters default to the dev-calibrated optimum (fa=0.3, loop_prob=0.9,
ahc_threshold=0.5, emb_scale=4.88); override with `POLYVOICE_VBX_{FA,FB,LOOP_PROB,
AHC_THRESHOLD,EMB_SCALE}`.

## Remaining release steps (need the release signing key)

Registry wiring and SHA-256 integrity are in place. A release engineer should
still:

1. **Optionally re-host** the six `.npy` on a GitHub Release asset or HF repo
   (the commit-pinned raw.githubusercontent.com host is stable via the SHA-256
   gate; a release asset is nicer for bandwidth/CDN).
2. **Sign** each file with the minisign release secret (same flow as the ONNX
   models) and fill the `signature = '''...'''` fields on the six
   `[models.vbx_plda_*]` entries in `src/models/manifest.toml`.
3. **Do not** add PLDA ids to `[profiles.*]` — they stay ad-hoc `ensure()` /
   `from_registry` only so missing signatures never fail profile resolution.

Until signatures land, VBx keeps working with SHA-256 verification alone.
