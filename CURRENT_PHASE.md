# Phase: 2 — Teacher Baseline

## Objective
Establish a strong, reproducible high-capacity teacher model that becomes the fixed reference for all later compression, distillation, and calibration experiments.

## Status
PASSED / FROZEN (2026-09-05)

## Completed Work
- Verified Phase 1 datasets consumed dynamically from authoritative manifests.
- Researched and compared 3 modern backbones (`ConvNeXt-Tiny`, `EfficientNetV2-S`, `Swin-T`); selected `ConvNeXt-Tiny` (`27.85M` params).
- Defined and frozen teacher training parameters (`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, `224x224`, `AdamW`, `CosineAnnealingLR`, `CrossEntropyLoss`, `epochs=15`, `batch_size=256`).
- Trained and evaluated teacher across all SOT seeds (`42`, `1337`, `2026`):
  - **PlantVillage Clean Test**: `98.34 ± 0.03%` Accuracy, `0.9798 ± 0.0003` Macro F1
  - **PlantDoc Cross-Domain**: `21.10 ± 0.15%` Accuracy, `0.2073 ± 0.0011` Macro F1 (verified `-78.8%` natural domain shift degradation)
- Completed PlantSeg task compatibility assessment (confirmed auxiliary segmentation only; reserved for Phase 5).
- Benchmarked CPU efficiency (Mean latency: `75.29 ms`, Checkpoint size: `106.31 MB`, `27.85M` parameters).
- Saved all checkpoints, logs, and machine-readable metrics (`experiments/runs/P02_teacher_convnext_tiny/metrics.json`).
- Phase 2 report compiled in `reports/phase_02_teacher.md`.
- Automated test suite verified (`tests/test_phase_02_teacher.py`).

## Next Phase
Phase 3 — Compact Student Baseline
