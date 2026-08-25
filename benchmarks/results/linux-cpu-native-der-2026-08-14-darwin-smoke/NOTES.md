# Native v2+VBx INT8 — Darwin smoke (AMI-16 + Vox-10)

Host: Apple M1 Pro, Darwin arm64, `cli-native` (no ort). Not the Linux
product machine. Protocol matches `scripts/linux-cpu-native-der-gate.sh`
smoke (`MAX_VOX=10`, full AMI-16).

| Split | files | DER₀ micro | DER @0.25 | RTFx | vs Linux-ort ceiling |
|---|---:|---:|---:|---:|---|
| Vox-10 | 10 | 15.90 % | 9.29 % | 122× | DER not asserted (not 232) |
| AMI-test | 16 | **25.19 %** | 17.56 % | 109× | ≤ 24.19 + 1.5 = **25.69 → PASS** |

Ort Linux AMI-16: 24.19 % / 16.60 % / ~95×.

JSON: `summary.json`, `ami-test.json`, `voxconverse-test.json`.
