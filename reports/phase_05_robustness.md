# AgriRobust Phase 5 — Robustness Evaluation Report

**Phase**: Phase 5 — Robustness Evaluation  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-06  
**Primary Artifact**: [`experiments/runs/P05_robustness_evaluation/metrics.json`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/runs/P05_robustness_evaluation/metrics.json)  

---

## 1. Executive Summary

Phase 5 investigates how the high-capacity Teacher ([`ConvNeXt-Tiny`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt)), the uncompressed [`Student Baseline`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P03_student_mobilenetv3_small_plantvillage_s42.pt), and the distilled models (Response-KD, Feature-KD, Combined-KD) behave under controlled synthetic corruptions and real-world natural domain shift.

### The Scientific Core Question
> *"Does knowledge distillation make the compact model more robust to distribution shift and image degradation, or does its benefit remain specific to the PlantDoc domain?"*

### Primary Scientific Findings

1. **Distillation Yields Significant Overall Synthetic Robustness**:
   - The **Response-KD Champion** achieved a **mean Relative Corruption Error (mRCE) of 79.87%** (where Student Baseline = 100.0% by definition).
   - A score below 100% represents a systematic reduction in corruption error rate across the suite; Response-KD reduces average corruption error rate by **20.13% relative to the baseline student**.
   - Teacher mRCE is **95.81%** and Feature-KD is **96.12%**. Combined-KD achieved **81.75% mRCE**.
2. **Massive Resilience to Photometric Shifts and High-Frequency Noise**:
   - Under **Gaussian Noise** ($\sigma=0.06$, severity 2), the Student Baseline collapsed from 99.62% clean F1 down to **0.4673 F1** (50.17% Acc, a -53.1% collapse). In contrast, Response-KD maintained **0.7706 F1** (80.10% Acc), providing a **+29.93% absolute accuracy advantage**.
   - Under **Brightness** (severity 5, $2.0\times$ exposure), the Student Baseline dropped to 79.64% Acc (0.7648 F1), whereas Response-KD held at **92.79% Acc** (0.8955 F1), a **+13.15% accuracy advantage**.
   - Under **Contrast** (severity 5, $0.25\times$ contrast), the Student Baseline collapsed to 73.99% Acc (0.6936 F1), whereas Response-KD preserved **92.41% Acc** (0.8905 F1), a **+18.42% accuracy advantage**.
3. **Severe Defocus Blur Collapses Baseline, While Distillation Cushions Retention**:
   - Under extreme Defocus Blur ($\sigma=7.0$, severity 5), the Student Baseline suffered near-total catastrophic failure (**4.18% Acc**, 0.0145 F1).
   - In stark contrast, Response-KD preserved **24.36% Acc**, Feature-KD preserved **25.56% Acc**, and Combined-KD preserved **29.68% Acc** (Teacher retained 39.97% Acc).
4. **Resolution Degradation**:
   - MobileNetV3 architectures displayed better scale invariance than ConvNeXt-Tiny under severe downsampling. At $56\times 56$ (16-fold pixel reduction), the Teacher dropped to 70.51% Acc (0.6255 F1), while Student Baseline retained 76.23% Acc and Feature-KD retained **79.96% Acc** (0.7800 F1).
5. **Natural Domain Shift (PlantDoc Field Crops)**:
   - On 8,883 out-of-domain uncurated field crops, the Teacher scored 21.05% Acc / 0.2062 Macro F1.
   - The uncompressed Student Baseline dropped to 11.62% Acc / 0.1298 Macro F1 (-86.97% drop vs clean).
   - **Response-KD achieved 18.39% Acc / 0.1791 Macro F1**, preserving 86.86% of the Teacher's Macro F1 and beating the Student Baseline by **+6.77% absolute accuracy (+0.0493 absolute Macro F1)**.

---

## 2. Experimental Setup & Methodology

### 2.1 Model Registry Evaluated

All 5 models evaluated on bit-for-bit identical input tensors at full test set size (no downsampling):

| Model Key | Architecture | Source Checkpoint | Checkpoint SHA-256 | Parameters | Size (MB) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `teacher` | ConvNeXt-Tiny | `P02_teacher_convnext_tiny_plantvillage_s42.pt` | `08544c74...` | 27.85M | 106.31 MB |
| `student_baseline` | MobileNetV3-Small | `P03_student_mobilenetv3_small_plantvillage_s42.pt` | `aa87f2ff...` | 1.56M | 6.07 MB |
| `response_kd` | MobileNetV3-Small | `P04_student_kd_response_plantvillage_s42.pt` | `2b935203...` | 1.56M | 6.07 MB |
| `feature_kd` | MobileNetV3-Small | `P04_student_kd_feature_plantvillage_s42.pt` | `f41bc3ec...` | 1.56M | 6.07 MB |
| `combined_kd` | MobileNetV3-Small | `P04_student_kd_combined_plantvillage_s42.pt` | `5bf8a40a...` | 1.56M | 6.07 MB |

### 2.2 Frozen Corruption Matrix (35 Conditions)

Evaluated on 8,129 images from the canonical PlantVillage test split. For noise and cutout, per-sample deterministic random state was enforced using SHA-256 tokens (`{sample_id}_{corruption}_{severity}_{seed}`):

1. **Brightness**: Factor $\in [1.15, 1.30, 1.50, 1.75, 2.00]$
2. **Contrast**: Factor $\in [0.85, 0.70, 0.55, 0.40, 0.25]$
3. **Defocus Blur**: Gaussian blur kernel radius $\sigma \in [1.0, 2.0, 3.5, 5.0, 7.0]$
4. **Gaussian Noise**: Additive $\mathcal{N}(0, \sigma^2)$ with $\sigma \in [0.03, 0.06, 0.10, 0.15, 0.22]$
5. **JPEG Compression**: Quality factor $\in [80, 60, 40, 25, 12]$
6. **Resolution Degradation**: Downsample $\to$ Bilinear Upsample to $[176, 144, 112, 80, 56]$
7. **Occlusion / Cutout**: Black rectangular cutout masking $[5\%, 10\%, 18\%, 28\%, 40\%]$ area

### 2.3 Evaluation Metrics

- **Error Rate**: $E_{m, c, s} = 1 - \text{Accuracy}_{m, c, s}$
- **Relative Corruption Error (RCE)** for family $c$:
  $$\text{RCE}_c = \frac{\sum_{s=1}^5 (1 - \text{Acc}_{m, c, s})}{\sum_{s=1}^5 (1 - \text{Acc}_{\text{student\_base}, c, s})} \times 100$$
- **Mean Relative Corruption Error (mRCE)**:
  $$\text{mRCE} = \frac{1}{7} \sum_{c=1}^7 \text{RCE}_c$$
  *(Student Baseline $= 100.0\%$. Values $< 100\%$ denote lower corruption error than baseline).*

---

## 3. Comprehensive Results

### 3.1 Relative Corruption Error (RCE) Summary

| Corruption Family | Teacher | Student Baseline | Response-KD (Champion) | Feature-KD | Combined-KD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Brightness** | **56.22%** | 100.00% | **44.09%** | 116.96% | 44.36% |
| **Contrast** | **39.89%** | 100.00% | **40.94%** | 97.99% | **35.86%** |
| **Defocus Blur** | 82.11% | 100.00% | 96.93% | **81.21%** | 92.03% |
| **Gaussian Noise** | **49.36%** | 100.00% | **69.72%** | 98.94% | 71.50% |
| **JPEG Compression** | 172.82% | 100.00% | 73.28% | 65.96% | **64.45%** |
| **Resolution** | 135.39% | 100.00% | 106.66% | **85.72%** | 113.16% |
| **Occlusion** | 134.91% | 100.00% | 127.47% | 126.03% | 150.90% |
| **Mean RCE (mRCE)** | **95.81%** | **100.00%** | **79.87%** | **96.12%** | **81.75%** |

### 3.2 Detailed Severity-Response Trajectories

#### 1. Brightness Degradation (Accuracy / Macro F1)
| Severity | Parameter | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean | 1.00 | 98.31% / 0.9794 | 99.75% / 0.9962 | 99.30% / 0.9884 | 99.69% / 0.9955 | 99.29% / 0.9885 |
| 1 | 1.15 | 98.36% / 0.9798 | 99.75% / 0.9958 | 99.31% / 0.9889 | 99.64% / 0.9948 | 99.30% / 0.9886 |
| 2 | 1.30 | 98.15% / 0.9778 | 99.67% / 0.9948 | 99.24% / 0.9879 | 99.51% / 0.9933 | 99.30% / 0.9884 |
| 3 | 1.50 | 97.65% / 0.9730 | 98.98% / 0.9862 | 98.82% / 0.9835 | 98.29% / 0.9789 | 98.92% / 0.9846 |
| 4 | 1.75 | 95.97% / 0.9507 | 92.77% / 0.9167 | 96.97% / 0.9547 | 91.12% / 0.8969 | 97.07% / 0.9584 |
| 5 | 2.00 | 93.46% / 0.9166 | 79.64% / 0.7648 | **92.79% / 0.8955** | 77.30% / 0.7452 | 92.46% / 0.8949 |

#### 2. Contrast Degradation (Accuracy / Macro F1)
| Severity | Parameter | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.85 | 98.22% / 0.9781 | 99.69% / 0.9958 | 99.27% / 0.9884 | 99.57% / 0.9933 | 99.18% / 0.9869 |
| 2 | 0.70 | 97.81% / 0.9730 | 99.52% / 0.9934 | 99.00% / 0.9844 | 99.31% / 0.9900 | 98.91% / 0.9841 |
| 3 | 0.55 | 97.18% / 0.9656 | 98.67% / 0.9813 | 98.44% / 0.9784 | 98.54% / 0.9781 | 98.54% / 0.9802 |
| 4 | 0.40 | 96.58% / 0.9559 | 93.86% / 0.9211 | 96.85% / 0.9567 | 94.48% / 0.9094 | 97.23% / 0.9636 |
| 5 | 0.25 | 96.54% / 0.9549 | 73.99% / 0.6936 | **92.41% / 0.8905** | 74.52% / 0.6403 | 93.85% / 0.9140 |

#### 3. Defocus Blur Degradation (Accuracy / Macro F1)
| Severity | Parameter ($\sigma$) | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0 | 91.62% / 0.9155 | 95.79% / 0.9500 | 92.31% / 0.9152 | 95.69% / 0.9524 | 92.45% / 0.9223 |
| 2 | 2.0 | 74.86% / 0.6904 | 78.07% / 0.7396 | 78.82% / 0.7472 | 78.57% / 0.7509 | 77.11% / 0.7435 |
| 3 | 3.5 | 57.10% / 0.4121 | 62.04% / 0.4988 | 50.92% / 0.3930 | 67.40% / 0.5867 | 53.66% / 0.4370 |
| 4 | 5.0 | 48.96% / 0.3021 | 31.59% / 0.1738 | 32.27% / 0.1749 | 47.36% / 0.2975 | 36.97% / 0.2109 |
| 5 | 7.0 | 39.97% / 0.2080 | 4.18% / 0.0145 | **24.36% / 0.0675** | **25.56% / 0.1138** | **29.68% / 0.0866** |

#### 4. Gaussian Noise Degradation (Accuracy / Macro F1)
| Severity | Parameter ($\sigma$) | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.03 | 95.04% / 0.9409 | 89.68% / 0.8821 | **96.89% / 0.9615** | 91.04% / 0.9018 | 96.47% / 0.9597 |
| 2 | 0.06 | 89.16% / 0.8634 | 50.17% / 0.4673 | **80.10% / 0.7706** | 55.80% / 0.4863 | 78.82% / 0.7576 |
| 3 | 0.10 | 73.02% / 0.6418 | 22.12% / 0.1832 | **56.37% / 0.4796** | 20.27% / 0.1839 | 54.20% / 0.4390 |
| 4 | 0.15 | 52.27% / 0.4108 | 10.23% / 0.0895 | **30.41% / 0.2247** | 9.63% / 0.0618 | 28.40% / 0.1779 |
| 5 | 0.22 | 31.25% / 0.2085 | 5.13% / 0.0209 | **11.28% / 0.0740** | 4.02% / 0.0082 | 11.39% / 0.0763 |

#### 5. JPEG Compression Degradation (Accuracy / Macro F1)
| Severity | Parameter (Q) | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 80 | 97.15% / 0.9612 | 99.59% / 0.9939 | 99.13% / 0.9859 | 99.57% / 0.9923 | 99.19% / 0.9868 |
| 2 | 60 | 94.78% / 0.9309 | 99.32% / 0.9906 | 98.81% / 0.9818 | 99.53% / 0.9927 | 98.72% / 0.9827 |
| 3 | 40 | 92.18% / 0.8985 | 98.71% / 0.9822 | 98.25% / 0.9760 | 99.02% / 0.9880 | 98.30% / 0.9776 |
| 4 | 25 | 87.05% / 0.8260 | 95.58% / 0.9484 | 95.67% / 0.9486 | 97.10% / 0.9659 | 96.21% / 0.9566 |
| 5 | 12 | 76.39% / 0.6791 | 76.45% / 0.7478 | **85.90% / 0.8625** | 84.76% / 0.8166 | 88.02% / 0.8796 |

#### 6. Resolution Degradation (Accuracy / Macro F1)
| Severity | Parameter | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 176 | 94.50% / 0.9443 | 98.60% / 0.9801 | 96.36% / 0.9573 | 98.66% / 0.9840 | 95.92% / 0.9556 |
| 2 | 144 | 91.82% / 0.9197 | 97.08% / 0.9638 | 93.80% / 0.9316 | 96.97% / 0.9659 | 93.78% / 0.9353 |
| 3 | 112 | 85.90% / 0.8433 | 87.00% / 0.8589 | 87.72% / 0.8644 | 88.04% / 0.8602 | 86.86% / 0.8680 |
| 4 | 80 | 80.65% / 0.7691 | 84.50% / 0.8286 | 86.06% / 0.8414 | 87.86% / 0.8674 | 84.66% / 0.8393 |
| 5 | 56 | 70.51% / 0.6255 | 76.23% / 0.7207 | 75.70% / 0.6978 | **79.96% / 0.7800** | 74.74% / 0.7021 |

#### 7. Occlusion / Cutout Degradation (Accuracy / Macro F1)
| Severity | Area Masked | Teacher | Student Baseline | Response-KD | Feature-KD | Combined-KD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5% | 98.22% / 0.9783 | 99.66% / 0.9952 | 99.14% / 0.9861 | 99.56% / 0.9932 | 99.14% / 0.9865 |
| 2 | 10% | 97.45% / 0.9679 | 99.41% / 0.9901 | 98.75% / 0.9802 | 99.21% / 0.9885 | 98.72% / 0.9793 |
| 3 | 18% | 95.41% / 0.9436 | 98.55% / 0.9796 | 97.47% / 0.9626 | 98.23% / 0.9753 | 97.16% / 0.9578 |
| 4 | 28% | 91.25% / 0.8919 | 94.65% / 0.9304 | 92.50% / 0.8889 | 92.93% / 0.9160 | 91.11% / 0.8720 |
| 5 | 40% | 83.93% / 0.7984 | 82.72% / 0.7977 | 80.26% / 0.7501 | 78.55% / 0.7732 | 76.13% / 0.6903 |

### 3.3 Natural Domain Shift Benchmark (PlantDoc Field Crops)

Evaluated on 8,883 uncurated mobile phone captures under outdoor field conditions:

| Model | Accuracy | Macro F1 | Balanced Accuracy | Absolute F1 Drop vs Clean | Relative F1 Drop vs Clean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Teacher** (`ConvNeXt-Tiny`) | 21.05% | **0.2062** | 21.36% | +0.7732 | -78.95% |
| **Student Baseline** (`MobileNetV3-Small`) | 11.62% | 0.1298 | 13.84% | +0.8664 | -86.97% |
| **Response-KD** (Champion) | **18.39%** | **0.1791** | **18.65%** | **+0.8093** | **-81.88%** |
| **Feature-KD** | 13.23% | 0.1349 | 14.80% | +0.8606 | -86.44% |
| **Combined-KD** | 16.72% | 0.1576 | 16.97% | +0.8309 | -84.05% |

---

## 4. Scientific Discussion & Architectural Insights

1. **Multi-Benchmark Robustness Improvements**:
   - Response KD improves performance across both the synthetic corruption suite (**mRCE 79.87%**, a 20.13% error reduction relative to baseline) and the PlantDoc natural-domain shift (**PlantDoc Macro F1 0.1791** vs baseline 0.1298).
   - This provides empirical evidence that its robustness benefit is not limited to a single benchmark, showing that KD enforces representations that better tolerate sensor noise, overexposure, low dynamic range, and severe compression. However, this does not imply that robustness universally generalizes to arbitrary unobserved natural distribution shifts.
2. **Why Soft Targets Buffer High-Frequency Perturbations**:
   - The Student Baseline trained with hard one-hot Cross-Entropy overfits to sharp, crisp laboratory leaf textures. Under additive Gaussian noise or Defocus Blur, these high-frequency patterns are scrambled, driving accuracy down to near chance (5% and 4%).
   - Distillation trains the student on smoothed class distribution vectors. This regularizes the gradient updates and discourages the network from relying exclusively on brittle high-frequency features.
3. **Feature Hint Distillation Disparity**:
   - Feature-KD showed outstanding resolution invariance (mRCE 85.72%), retaining 80.0% accuracy even at $56\times 56$. However, it failed to confer robustness against noise (mRCE 98.94%) or natural domain shift (F1 0.1349).
   - Intermediate feature matching forces geometric alignment between disparate architectural backbones (inverted residual MobileNet vs ConvNeXt block), which appears to over-constrain the student's capacity.

---

## 5. Phase 5 Verification & Freeze Gate

- [x] Evaluated all 35 synthetic conditions at full test set size (8,129 images) across all 5 models without downsampling.
- [x] Deterministic SHA-256 seeding guaranteed bit-for-bit identical input images across all model evaluations.
- [x] Evaluated natural cross-domain shift on 8,883 PlantDoc field crops.
- [x] Baseline-Normalized Relative Corruption Error (RCE) and mRCE properly calculated and stored.
- [x] Intermediate checkpoints persisted throughout execution and master `metrics.json` verified.
- [x] All 31 project test suite unit tests pass (`uv run pytest`).

**Phase 5 is formally complete and FROZEN.**
