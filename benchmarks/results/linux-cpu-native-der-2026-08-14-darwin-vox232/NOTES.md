# Native v2+VBx INT8 — Darwin VoxConverse-test 232

Host: Apple M1 Pro, Darwin arm64, `cli-native` (no ort). Not Linux.
Full official Vox split; AMI in this run was 1 file only (AMI-16 is in
the sibling `...-darwin-smoke` folder).

| Split | files | DER₀ micro | DER₀ macro | DER @0.25 | RTFx |
|---|---:|---:|---:|---:|---:|
| VoxConverse-test | 232 | **15.47 %** | 15.46 % | 10.81 % | 130× |

Ceiling = ort Linux row `voxconverse_test_linux_cpu`: 14.94 + 1.0 = **15.94 → PASS**.
Ort Linux: 14.94 % / 10.27 % / ~82×.

Native is +0.53 pp DER₀ vs ort on this split (within the product
tolerance). RTFx is Apple (BNNS / BNNSGraph), not a Linux number.

Linux aarch64 + OpenBLAS still needs `DOCKER=1 bash scripts/linux-cpu-native-der-gate.sh` when the engine is up. Product `cli` stays ort until that run holds DER and RTF.
