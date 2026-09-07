---
id: TASK-1
title: Close native-vs-ort RTF gap on Linux x86_64
status: To Do
assignee: []
created_date: '2026-09-07 18:42'
labels: []
dependencies: []
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Same-host (Ryzen AI 9 HX 370): native Vox-3 95.4x vs ort 104.5x (0.91x), AMI-16 113.4x vs 145.4x (0.78x). Levers: AVX-512 vectorization of LSTM gates (lstm.rs), multithreaded GEMM, powerset window/embedder pool tuning. Constraints: DER floors (Vox-3 7.11/7.39, AMI 25.5+1.5, Vox-232 15.4+1.0), RSS <=556 MiB, no ort in default cli.
<!-- SECTION:DESCRIPTION:END -->
