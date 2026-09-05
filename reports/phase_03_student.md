# AgriRobust Phase 3 — Compact Student Baseline Report

**Phase**: Phase 3 — Compact Student Baseline  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-05  
**Primary Student Checkpoint**: `experiments/checkpoints/P03_student_mobilenetv3_small_plantvillage_s42.pt`  
**Checkpoint SHA-256**: `6076d8a2b0f4d5d562b2ce380df91ceb2561b1105b13eeab08a60fe93cce2eee`  

---

## 1. Executive Summary

Phase 3 establishes the empirical **Compact Student Baseline** (`MobileNetV3-Small`) trained end-to-end with standard cross-entropy supervision on the canonical 38-class PlantVillage training split without knowledge distillation, quantization, pruning, or robustness tuning.

The central scientific question of Phase 3 is:
> *"How much performance do we lose by moving from the teacher to a genuinely mobile-sized model before applying any compression or distillation techniques?"*

Key Empirical Findings:
1. **Target A Met (Parameter Ratio)**: The student has **1,556,806 parameters**, representing **5.59%** of the Teacher's parameters (`27.85M`), comfortably within SOT Target A (`<= 10.0%`).
2. **Target B Met (Checkpoint Size)**: The serialized deployment checkpoint is **6.08 MB**, comfortably satisfying SOT Target B (`<= 15.0 MB`).
3. **Controlled Clean Performance**: On the controlled clean PlantVillage test set, the student fine-tuned end-to-end reaches **99.75% Accuracy** and **0.9962 Macro F1** (surpassing the linear-probed teacher head by +1.64% F1 due to end-to-end parameter adaptation).
4. **Severe In-The-Wild Gap (Domain Shift)**: Under natural domain shift on PlantDoc leaf crops, the student collapses to **11.62% Accuracy** and **0.1298 Macro F1** (down from the Teacher's `0.2073 Macro F1`).
   - The uncompressed student retains only **62.61% of the teacher's out-of-domain Macro F1**.
   - This **-7.75% absolute / -37.39% relative out-of-domain gap** establishes the primary target for Phase 4 Knowledge Distillation.

---

## 2. Candidate Student Architectures & Selection

Three mobile vision backbones were evaluated against the SOT deployment targets:

| Candidate Architecture | Pretrained Weights | Params (38-class) | Ratio of Teacher (27.85M) | Target A Feasibility (<=10%) | CPU Latency (batch-1) | Selection Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`MobileNetV3-Small`** *(Selected)* | `MobileNet_V3_Small_Weights.IMAGENET1K_V1` | **1,556,806** | **5.59%** | **PASSED** (<10%) | **18.41 ms** | Hardware-aware NAS architecture combining hard-swish, depthwise separable convolutions, and squeeze-and-excitation. Satisfies both Target A and Target B with 4x CPU latency reduction. |
| **`MobileNetV3-Large`** | `MobileNet_V3_Large_Weights.IMAGENET1K_V1` | 4,240,982 | 15.23% | FAILED (>10%) | ~45.0 ms | Exceeds Target A threshold (15.23% > 10.0%). |
| **`ShuffleNetV2-1.0`** | `ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1` | 1,299,800 | 4.67% | PASSED (<10%) | ~28.0 ms | Channel shuffle design; however, split/shuffle operations produce fragmented memory buffers on target mobile runtimes compared to linear inverted bottlenecks. |

**Selection Decision**: `MobileNetV3-Small` was formally selected and frozen as the student architecture.

---

## 3. Training Protocol & Specification

The student baseline was trained with standard end-to-end fine-tuning without teacher guidance:

- **Model**: `AgriStudentMobileNetV3` (`torchvision.models.mobilenet_v3_small`)
- **Pretrained Weights**: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`
- **Classifier Head**: `Linear(in_features=1024, out_features=38)`
- **Input Resolution**: `[3, 224, 224]`
- **Loss Function**: Standard `CrossEntropyLoss` (0.0 teacher weight, no KD)
- **Optimizer**: `AdamW (lr=0.001, weight_decay=1e-4)`
- **Scheduler**: `CosineAnnealingLR (T_max=15, eta_min=1e-5)`
- **Batch Size**: 128 (298 optimization steps/epoch)
- **Epochs**: 15 epochs
- **Device**: CPU (16-thread benchmark host)
- **Seed**: 42 (Primary)

> [!NOTE]
> **Training Protocol Note**: Phase 3 measures student performance under a controlled common training budget (15 epochs, AdamW) rather than exhaustive hyperparameter tuning.

---

## 4. Empirical Evaluation & Teacher vs. Student Gap Analysis

### 4.1 Comparative Performance Summary

| Metric | Teacher Baseline (`ConvNeXt-Tiny`) | Student Baseline (`MobileNetV3-Small`) | Gap / Delta | Performance Retention |
| :--- | :---: | :---: | :---: | :---: |
| **PlantVillage Clean Accuracy** | 98.34% | **99.75%** | +1.41% | 101.43% |
| **PlantVillage Clean Macro F1** | 0.9798 | **0.9962** | +0.0164 | **101.67%** |
| **PlantVillage Clean Bal. Accuracy** | 0.9779 | **0.9962** | +0.0183 | 101.87% |
| **PlantDoc Cross-Domain Accuracy** | **21.10%** | 11.62% | -9.48% | 55.07% |
| **PlantDoc Cross-Domain Macro F1** | **0.2073** | 0.1298 | **-0.0775** | **62.61%** |
| **PlantDoc Cross-Domain Bal. Accuracy**| **0.2143** | 0.1384 | -0.0759 | 64.58% |

### 4.2 Efficiency & Footprint Tradeoff

| Metric | Teacher (`ConvNeXt-Tiny`) | Student (`MobileNetV3-Small`) | Reduction / Speedup | SOT Target Compliance |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | 27,849,350 | **1,556,806** | **-94.41%** (17.9x fewer) | **PASSED Target A** (5.59% <= 10%) |
| **Checkpoint File Size** | 106.31 MB | **6.08 MB** | **-94.28%** (17.5x smaller) | **PASSED Target B** (6.08 MB <= 15 MB) |
| **CPU Batch-1 Latency (Mean)** | 75.29 ms | **18.41 ms** | **4.09x faster** | Substantial CPU speedup |
| **CPU Batch-1 Latency (P95)** | 76.40 ms | **19.40 ms** | **3.94x faster** | Consistent tail latency |
| **Throughput (Batch-1)** | 13.28 FPS | **54.31 FPS** | **+309%** | Real-time interactive rate |

---

## 5. Key Research Findings

1. **Controlled vs. In-The-Wild Divergence**:
   The student model achieves near-perfect discrimination on laboratory images (`0.9962 F1`), but suffers an catastrophic **86.97% relative performance drop** when evaluated on in-the-wild crops (`0.1298 F1`).
2. **The "Capacity Collapse" Under Natural Shift**:
   While the Teacher also degraded out-of-domain (`0.2073 F1`), the Student lost an additional **37.39% of the Teacher's remaining generalization ability**. The compact model lacks the parameter capacity to maintain invariant visual features without regularization or distillation.
3. **The Distillation Target**:
   Phase 4 (Knowledge Distillation) now has a clear, quantified benchmark: transfer the Teacher's out-of-domain knowledge to bring the Student's cross-domain Macro F1 closer to `0.2073` while retaining its `1.56M` parameter footprint and `18.4 ms` latency.

---

## 6. Phase Boundary Affirmation

"No knowledge distillation (logits, features, or attention transfer), quantization, pruning, corruption robustness training, uncertainty estimation, or Android implementation were performed during Phase 3."

---

## 7. Gate Exit Determination: PASSED

Phase 3 is complete, validated, and frozen. The baseline gap is quantified and ready for Phase 4.
