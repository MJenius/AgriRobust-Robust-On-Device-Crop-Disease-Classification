# Phase: 3 — Compact Student Baseline

## Objective
Establish the raw, uncompressed, non-distilled compact student baseline trained directly on the canonical 38-class PlantVillage dataset under identical training protocol, and quantify the performance/efficiency gap against the frozen teacher.

## Status
PASSED / FROZEN (2026-09-05)

## Completed Work
- Compared 3 mobile vision architectures (`MobileNetV3-Small`, `MobileNetV3-Large`, `ShuffleNetV2-1.0`); selected `MobileNetV3-Small` (`1.56M` params).
- Validated that `MobileNetV3-Small` satisfies SOT Target A (uses `5.59% <= 10.0%` of teacher parameters) and Target B (checkpoint size is `6.08 MB <= 15.0 MB`).
- Defined and trained the student end-to-end on the PlantVillage 38-class training split (`15 epochs`, `AdamW`, `CosineAnnealingLR`, `CrossEntropyLoss`) without distillation, quantization, or pruning.
- Benchmarked CPU efficiency:
  - **Batch-1 Latency**: `18.41 ms` (a `4.09x` speedup over Teacher's `75.29 ms`).
  - **Throughput**: `54.31 FPS` (vs `13.28 FPS` for Teacher).
- Evaluated on PlantVillage Clean Test:
  - **Accuracy**: `99.75%`
  - **Macro F1**: `0.9962`
- Evaluated on PlantDoc Cross-Domain Test:
  - **Accuracy**: `11.62%`
  - **Macro F1**: `0.1298` (vs Teacher's `0.2073`, a `-7.75%` absolute / `-37.39%` relative gap).
- Verified that all automated tests pass (`22/22` tests passing via `uv run pytest`).
- Published Phase 3 Report in `reports/phase_03_student.md` and saved metrics in `experiments/runs/P03_student_mobilenetv3_small/metrics.json`.

## Next Phase
Phase 4 — Knowledge Distillation
