# Phase: 5 — Robustness Evaluation

## Objective
Determine how robust the Teacher (`ConvNeXt-Tiny`), Phase 3 Student Baseline (`MobileNetV3-Small`), and Phase 4 KD models (Response-KD, Feature-KD, Combined-KD) are under realistic visual distribution shifts (brightness, contrast, blur, noise, jpeg compression, resolution, occlusion) and real-world outdoor domain shift on PlantDoc, verifying whether knowledge distillation improves robustness beyond clean PlantVillage performance.

## Status
PASSED / FROZEN (2026-09-06)

## Completed Work
- Implemented reproducible synthetic corruption benchmark in `src/agrirobust/robustness/corruptions.py` covering 7 corruption families at 5 frozen severities with per-sample deterministic SHA-256 hash seeding (`{sample_id}_{corruption}_{severity}_{seed}`).
- Verified corruption parameter freeze and deterministic behaviour via unit tests in `tests/test_phase_05_robustness.py` (all 31 project tests passing).
- Executed full robustness evaluation suite at full test set size (8,129 images per condition) across all 5 models (35 conditions = 175 full evaluation passes) with per-condition intermediate checkpointing.
- Computed Baseline-Normalized Relative Corruption Error (RCE) and mean RCE (mRCE) relative to the Student Baseline:
  - **Response-KD Champion**: **`mRCE = 79.87%`** (a 20.13% systematic error reduction across all corruptions relative to the uncompressed baseline).
  - **Teacher**: `mRCE = 95.81%`
  - **Feature-KD**: `mRCE = 96.12%`
  - **Combined-KD**: `mRCE = 81.75%`
- Discovered massive distillation cushions against high-frequency noise and extreme dynamic range shifts:
  - **Gaussian Noise ($\sigma=0.06$)**: Student Baseline collapsed from 0.9962 to 0.4673 F1 (50.17% Acc), while Response-KD maintained **0.7706 F1 (80.10% Acc)** (+29.93% absolute accuracy advantage).
  - **Severe Defocus Blur ($\sigma=7.0$)**: Student Baseline suffered catastrophic breakdown (4.18% Acc), while distilled models retained 24.36% (Response-KD) to 29.68% (Combined-KD) Acc.
  - **Brightness / Contrast blowout**: Response-KD maintained 92.79% and 92.41% accuracy at severity 5, conferring +13.15% and +18.42% accuracy advantages over the baseline.
- Evaluated natural domain shift on 8,883 out-of-domain PlantDoc field crops:
  - Teacher: 21.05% Acc / 0.2062 Macro F1
  - Student Baseline: 11.62% Acc / 0.1298 Macro F1
  - Response-KD: **18.39% Acc / 0.1791 Macro F1** (retaining 86.86% of the Teacher's performance).
- Generated master metrics artifact `experiments/runs/P05_robustness_evaluation/metrics.json` and compiled comprehensive Phase 5 report in `reports/phase_05_robustness.md`.

## Next Phase
Phase 6 — Uncertainty Calibration & Selective Abstention

