# AgriRobust Phase 2 — Teacher Baseline Report

**Phase**: Phase 2 — Teacher Baseline  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-05  
**Primary Checkpoint**: `experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt`  
**Checkpoint SHA-256**: `7b80b48404531597f098571e56d53f03676305b57a7938ff27a4389e8fa1d1af`  

---

## 1. Executive Summary

Phase 2 established a frozen, reproducible, high-capacity reference **Teacher Model** to anchor all subsequent student compression, knowledge distillation (Phase 4), and calibration research.

In strict accordance with `PROJECT_SOT.md` (Sections 6, 9, 10, 12, 13, 21):
1. Verified Phase 1 frozen data manifests were consumed dynamically without hardcoding dataset statistics or altering splits.
2. Evaluated candidate pretrained backbones (`ConvNeXt-Tiny`, `EfficientNetV2-S`, `Swin-T`) and formally selected **`ConvNeXt-Tiny`** as the high-capacity teacher baseline.
3. Fine-tuned the teacher on the canonical 38-class PlantVillage training split across the three SOT evaluation seeds (`42`, `1337`, `2026`).
4. Evaluated performance separately on:
   - **Clean-domain**: PlantVillage held-out test split (8,129 images)
   - **Cross-domain**: PlantDoc held-out leaf crops (8,883 crops across 29 shared classes)
5. Assessed PlantSeg compatibility and verified that pixel segmentation masks are structurally incompatible with whole-image disease classification; PlantSeg is preserved exclusively for auxiliary lesion localization in Phase 5.
6. Benchmarked parameter count, checkpoint disk size, and CPU inference latency.

---

## 2. Model Backbone Selection & Comparison

Three modern pretrained vision architectures were compared for the high-capacity Teacher role:

| Candidate Backbone | Pretrained Weights / Source | Total Params | Top-1 ImNet (Ref) | Architectural Profile & Agricultural Suitability |
| :--- | :--- | :---: | :---: | :--- |
| **`ConvNeXt-Tiny`** *(Selected)* | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | **27.85M** | 82.5% | Pure convolutional modern design with 7x7 depthwise convolutions and inverted bottlenecks. Outstanding spatial inductive bias for fine-grained leaf texture lesions; standard Tensor representation ideal for feature distillation. |
| **`EfficientNetV2-S`** | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | 21.46M | 84.2% | Fused-MBConv stages optimized for training speed. Slightly fewer parameters, but depthwise inverted blocks produce more heterogeneous intermediate features for distillation. |
| **`Swin-T`** | `Swin_T_Weights.IMAGENET1K_V1` | 28.29M | 81.3% | Hierarchical shifted-window transformer. Strong long-range attention, but higher memory footprint during forward pass on CPU and sensitive to variable aspect ratio crops. |

**Selection Decision**: **`ConvNeXt-Tiny`** was selected based on its superior inductive bias on micro-lesion visual patterns, clean feature maps for later student distillation, and rock-solid CPU inference stability.

---

## 3. Frozen Teacher Training Specification

All training hyperparameters are frozen as follows:

- **Backbone**: `torchvision.models.convnext_tiny`
- **Pretrained Weights**: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- **Input Resolution**: `[3, 224, 224]`
- **Classification Head**: `LayerNorm2d(768) -> Flatten -> Dropout(0.2) -> Linear(768, 38)`
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Learning Rate Schedule**: `CosineAnnealingLR(T_max=15, eta_min=1e-5)`
- **Epochs**: 15 epochs
- **Batch Size**: 256
- **Device**: CPU (16-thread benchmark environment)
- **Seeds**: `42` (primary), `1337`, `2026`

---

## 4. Empirical Evaluation Results

### 4.1 Summary Across SOT Seeds (Mean ± Std)

| Evaluation Benchmark | Evaluated Set & Classes | Accuracy (%) | Macro F1 | Balanced Accuracy | Status / Shift |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PlantVillage Clean Test** | Held-out 15% split (8,129 images, 38 classes) | **98.34 ± 0.03%** | **0.9798 ± 0.0003** | **0.9779 ± 0.0003** | **Fixed Reference Baseline** |
| **PlantDoc Cross-Domain** | Held-out crops (8,883 crops, 29 shared classes) | **21.10 ± 0.15%** | **0.2073 ± 0.0011** | **0.2143 ± 0.0011** | **Severe Domain Shift (-78.8% rel F1)** |

> [!NOTE]
> **Domain Shift Analysis**: The teacher achieves near-perfect clean accuracy (98.34%), but drops to 21.10% accuracy / 0.2073 Macro F1 under natural in-the-wild shift on PlantDoc. This stark empirical degradation directly verifies **SOT Section 7.1 & 9**: *PlantVillage-only performance is not evidence of real-world robustness*. This drop serves as the quantitative ceiling that subsequent compression, distillation, and calibration phases will study.

### 4.2 Per-Seed Run Details

| Seed | Checkpoint Path | PV Clean Acc | PV Clean Macro F1 | PlantDoc Cross Acc | PlantDoc Cross Macro F1 | Relative F1 Drop |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **42** | `experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt` | 98.31% | 0.9794 | 21.05% | 0.2062 | 78.95% |
| **1337** | `experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s1337.pt` | 98.35% | 0.9799 | 21.27% | 0.2085 | 78.72% |
| **2026** | `experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s2026.pt` | 98.35% | 0.9800 | 20.98% | 0.2071 | 78.87% |

---

## 5. Model Efficiency & Latency Benchmark

Benchmarked on the host 16-thread CPU at batch size 1 across 50 iterations following 10 warmup iterations:

- **Total Parameters**: **27,849,350** (~27.85M)
- **Trainable Parameters**: **27,849,350**
- **Checkpoint File Size**: **106.31 MB**
- **Inference Latency (Mean)**: **75.29 ms**
- **Inference Latency (P50)**: **75.37 ms**
- **Inference Latency (P95)**: **76.40 ms**
- **Throughput**: **13.28 FPS**

---

## 6. Secondary Benchmark Evaluation: PlantSeg Compatibility Assessment

In accordance with instruction item 7 and `PROJECT_SOT.md`:
- **Sample Count**: 11,458 image/mask pairs across 115 disease symptom categories.
- **Modality**: Pixel-level segmentation polygons.
- **Assessment**: Incompatible with the 38-class whole-image classification head. Forcing segmentation masks into image-level labels would violate the canonical taxonomy. PlantSeg is officially retained for Phase 5 auxiliary lesion localization and segmentation robustness.

---

## 7. Machine-Readable Artifacts & Code Manifest

- **Configuration**: [`configs/project.yaml`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/configs/project.yaml) updated with frozen teacher parameters.
- **Model Definition**: [`src/agrirobust/models/teacher.py`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/src/agrirobust/models/teacher.py)
- **Data Loaders**: [`src/agrirobust/data/dataset.py`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/src/agrirobust/data/dataset.py) & [`src/agrirobust/data/transforms.py`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/src/agrirobust/data/transforms.py)
- **Evaluation Engine**: [`src/agrirobust/evaluation/evaluator.py`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/src/agrirobust/evaluation/evaluator.py)
- **All-Seed Metrics**: [`experiments/runs/P02_teacher_convnext_tiny/metrics.json`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/runs/P02_teacher_convnext_tiny/metrics.json)
- **Primary Checkpoint**: [`experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt)
- **Automated Tests**: [`tests/test_phase_02_teacher.py`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/tests/test_phase_02_teacher.py) (4/4 tests passed; 19/19 full suite passing).

---

## 8. Phase Boundary Affirmation

"No student model training, knowledge distillation, quantization, pruning, uncertainty methods, or Android mobile implementation were performed during Phase 2."

---

## 9. Gate Exit Determination: PASSED

Phase 2 exit criteria are fully satisfied. The reference Teacher model is empirically established and frozen for Phase 3 (Compact Student Baseline).
