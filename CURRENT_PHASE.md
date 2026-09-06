# Phase: 4 — Knowledge Distillation

## Objective
Determine whether knowledge distillation from the high-capacity frozen Teacher (`ConvNeXt-Tiny`, 27.85M params) into the compact Student (`MobileNetV3-Small`, 1.56M params) can recover the student's lost cross-domain performance on PlantDoc while preserving its compact architecture, memory footprint, and low CPU latency.

## Status
PASSED / FROZEN (2026-09-06)

## Completed Work
- Implemented and validated distillation loss formulations adhering strictly to PyTorch's KL divergence direction:
  - **Response KD**: $\mathcal{L}_{\text{CE}} + \alpha T^2 \cdot \text{KLDiv}(\log\text{softmax}(z_s/T), \text{softmax}(z_t/T))$ with $T=4.0, \alpha=0.5$.
  - **Feature Hint KD**: $\mathcal{L}_{\text{CE}} + \beta \cdot \text{MSE}(\text{Proj}_{576 \to 768}(f_s), f_t)$ with projection layer included in optimizer but excluded from deployment checkpoints.
  - **Combined KD**: Composite response and feature hint supervision.
- Verified tensor dimensions in code before launch: Student pooled features are confirmed 576-D and Teacher pooled features are confirmed 768-D.
- Precomputed and cached full Teacher targets (logits and pooled features) for train and validation splits to eliminate redundant CPU forward passes.
- Executed the complete controlled Phase 4 distillation ablation suite across all 3 paradigms under identical data splits and training budget (8 epochs per ablation, AdamW, batch size 128).
- Benchmarked all distilled checkpoints on PlantVillage clean test and PlantDoc cross-domain test:
  - **Response KD (Champion)**: PlantDoc Macro F1 = **`0.1791`** (recovering **63.61% of the lost cross-domain gap**, a **+37.98% relative improvement** over the uncompressed student baseline of `0.1298`).
  - **Combined KD**: PlantDoc Macro F1 = **`0.1576`** (recovering 35.87% of the gap).
  - **Feature KD**: PlantDoc Macro F1 = **`0.1349`** (recovering 6.58% of the gap; showed strong clean validation accuracy of 99.75% but rigid intermediate geometric alignment limited cross-domain transfer).
- Verified deployment purity: all serialized checkpoints contain only the pure `MobileNetV3-Small` architecture (**1,556,806 parameters**, **6.07 MB** checkpoint size, CPU latency **17.83 ms**).
- Added comprehensive unit tests in `tests/test_phase_04_distillation.py`; verified that all **27/27 tests** in the project test suite pass cleanly (`uv run pytest`).
- Published Phase 4 Report in `reports/phase_04_distillation.md`, updated `configs/project.yaml`, and stored metrics in `experiments/runs/P04_knowledge_distillation/metrics.json`.

## Next Phase
Phase 5 — Robustness Evaluation
