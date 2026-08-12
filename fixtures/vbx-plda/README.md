# VBx PLDA fixtures (CC-BY-4.0)

Precomputed diagonalized PLDA parameters used by the default `vbx` clusterer
(`PldaModel::from_dir`). ~265 KB total. Attribution to pyannote — see repo
`NOTICE` and `docs/vbx-plda-release.md`.

| file | sha256 |
|------|--------|
| plda_transform.npy | `90261469714415743f4b8a86ee6b89466db858bde3c5944367cccfb7abd34f14` |
| plda_phi_computed.npy | `6ef7cf2f5a23a45b66f440f9a996a4cf5c047b369829af695d50ef18aa0a35e3` |
| plda_mean1.npy | `e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f` |
| plda_mean2.npy | `6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf` |
| plda_lda.npy | `e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b` |
| plda_mu.npy | `d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6` |

These fixtures are checked into the repo so the release gate and CLI DER
regression tests can exercise the **default** v2+VBx path without a network
host for the weights. Rebuild with `scripts/build-vbx-plda.py` if the
upstream params change; do not hand-edit the `.npy` files.

Registry downloads (`ModelRegistry` when `POLYVOICE_VBX_PLDA_DIR` is unset)
verify **SHA-256 + minisign** against the embedded manifest. Companion
`.minisig` files next to each fixture match the project model signing key
(`models/signing.pub`).
