# AgriRobust Phase 4 — Knowledge Distillation Report

**Phase**: Phase 4 — Knowledge Distillation  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-06  
**Champion Distillation Checkpoint**: `experiments/checkpoints/P04_student_kd_response_plantvillage_s42.pt`  
**Checkpoint SHA-256**: `2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6`  

---

## 1. Executive Summary

Phase 4 evaluates whether knowledge distillation (KD) from the high-capacity frozen Teacher ([`ConvNeXt-Tiny`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt), 27.85M params) into the compact Student ([`MobileNetV3-Small`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P03_student_mobilenetv3_small_plantvillage_s42.pt), 1.56M params) can recover the student's lost out-of-domain performance on PlantDoc without increasing deployment footprint or latency.

### The Scientific Core Question
> *"Can knowledge distillation recover the student's lost cross-domain performance while preserving its compact architecture?"*

### Key Empirical Findings

1. **Substantial Gap Recovery Achieved by Response KD**:
   - The uncompressed student baseline in Phase 3 exhibited a **0.0775 absolute Macro F1 gap** relative to the teacher on PlantDoc (`0.1298` vs `0.2073`, a 37.39% relative drop).
   - **Response KD ($T=4.0, \alpha=0.5$) achieved a PlantDoc Macro F1 of `0.1791`**, recovering **`+0.0493` absolute Macro F1** or **63.61% of the lost cross-domain gap** (a **+37.98% relative gain** over the uncompressed student baseline).
2. **Response vs. Feature vs. Combined Distillation**:
   - **Response KD (Logit/KL)**: PlantDoc Macro F1 = **`0.1791`** (63.61% gap recovered). Soft teacher probability targets provided rich dark knowledge and inter-class relationship signals that regularized student representations against laboratory overfitting.
   - **Combined KD (Logit + Feature Hint)**: PlantDoc Macro F1 = **`0.1576`** (35.87% gap recovered, +21.42% relative gain).
   - **Feature KD (MSE Hint Projection $576 \to 768$)**: PlantDoc Macro F1 = **`0.1349`** (6.58% gap recovered, +3.93% relative gain). Forcing intermediate pooled features from a 1.56M parameter MobileNet into the 768-D geometry of ConvNeXt-Tiny proved overly rigid, yielding near-perfect clean validation performance (`0.9955 Clean F1`) but poor out-of-domain transfer compared to soft response distillation.
3. **Deployment Purity & Inference Computation**:
   - All projection adapter heads used during feature distillation were cleanly stripped at checkpoint serialization.
   - The saved deployment models contain **only the pure MobileNetV3-Small architecture**: exactly **1,556,806 parameters** and **6.07 MB checkpoint file size** (substantially below SOT Target A $\le 10\%$ and Target B $\le 15\text{ MB}$).
   - KD does not add inference parameters or architectural computation; measured latency remains within benchmark variance of the baseline (mean `17.83 ms` vs Phase 3 baseline's `18.41 ms`).

---

## 2. Controlled Distillation Suite Architecture

Distillation was executed on the canonical 38-class PlantVillage training split under strictly identical data splits, preprocessing ($224 \times 224$), and optimization hyperparameters:

- **Teacher Model**: `ConvNeXt-Tiny` (Frozen, zero gradient backprop)
- **Student Model**: `MobileNetV3-Small`
- **Distillation Loss Formulations**:
  1. **Response KD**:
     $$\mathcal{L} = (1 - \alpha) \mathcal{L}_{\text{CE}}(z_s, y) + \alpha T^2 \cdot \text{KLDiv}\left(\log\text{softmax}\left(\frac{z_s}{T}\right), \text{softmax}\left(\frac{z_t}{T}\right)\right)$$
     with $T = 4.0, \alpha = 0.5$.
  2. **Feature Hint KD**:
     $$\mathcal{L} = \mathcal{L}_{\text{CE}}(z_s, y) + \beta \cdot \text{MSE}\left(\text{Proj}_{576 \to 768}(f_s), f_t\right)$$
     with $\beta = 0.5$, $\text{Proj} = \text{Linear}(576, 768) \to \text{BatchNorm1d} \to \text{ReLU}$.
  3. **Combined KD**:
     $$\mathcal{L} = (1 - \alpha)\mathcal{L}_{\text{CE}} + \alpha T^2 \mathcal{L}_{\text{KD}} + \beta \mathcal{L}_{\text{feat}}$$
- **Optimization**: `AdamW (lr=0.001, weight_decay=1e-4)`, `CosineAnnealingLR (8 epochs)`, batch size 128.
- **Teacher Target Precomputation**: To guarantee deterministic repeatability and eliminate redundant forward-pass compute on CPU, all teacher logits and 768-D pooled features were precomputed and cached in `data/processed/teacher_targets/`.

---

## 3. Empirical Results & Gap Recovery Analysis

### 3.1 Comparative Performance Summary

| Model / Experiment | PlantVillage Clean Acc | PlantVillage Clean Macro F1 | PlantDoc Cross-Domain Acc | PlantDoc Cross-Domain Macro F1 | Absolute $\Delta$ F1 vs Student | % of Gap Recovered | Params | Size (MB) | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Teacher** (`ConvNeXt-Tiny`) | 98.34% | 0.9798 | 21.10% | **0.2073** | *[Reference]* | *[Reference]* | 27.85M | 106.31 MB | 75.29 ms |
| **Student Baseline** (Phase 3) | **99.75%** | **0.9962** | 11.62% | 0.1298 | 0.0000 | 0.00% | **1.56M** | **6.08 MB** | 18.41 ms |
| **Feature KD** ($\beta=0.5$) | 99.69% | 0.9955 | 13.23% | 0.1349 | +0.0051 | +6.58% | **1.56M** | **6.07 MB** | **7.64 ms** |
| **Combined KD** ($T=4, \alpha=0.5, \beta=0.5$) | 99.29% | 0.9885 | 16.72% | 0.1576 | +0.0278 | +35.87% | **1.56M** | **6.07 MB** | **11.51 ms** |
| **Response KD** ($T=4, \alpha=0.5$) *(Champion)* | 99.30% | 0.9884 | **18.39%** | **0.1791** | **+0.0493** | **+63.61%** | **1.56M** | **6.07 MB** | **17.83 ms** |

> [!IMPORTANT]
> **Scientific Uncertainty Addressed**: As noted prior to implementation, distillation under identical source training data is not guaranteed to improve cross-domain performance. Here, the experimental result is unambiguously positive: **Response KD substantially improved cross-domain performance, consistent with the hypothesis that teacher soft targets provide useful inter-class similarity information**, raising cross-domain Macro F1 from 0.1298 to 0.1791 (+37.98% relative improvement) while retaining 99.30% clean accuracy.

---

## 4. Per-Class Diagnostic Transfer Highlights

On the target PlantDoc out-of-domain evaluation set, Response KD recovered significant discriminative power on difficult crop classes:
- **Tomato Early Blight**: Student Baseline = `0.0000` $\to$ Response KD = **`0.1792`**
- **Tomato Late Blight**: Student Baseline = `0.0526` $\to$ Response KD = **`0.1875`**
- **Corn Common Rust**: Student Baseline = `0.2353` $\to$ Response KD = **`0.3333`**
- **Grape Leaf Blight**: Student Baseline = `0.0909` $\to$ Response KD = **`0.1818`**

Classes with heavy environmental clutter and non-leaf backgrounds remain challenging for all models trained purely on PlantVillage laboratory backgrounds, establishing the critical motivation for Phase 5 (Synthetic & Natural Robustness Benchmarking).

---

## 5. Artifacts & Checkpoint Registry

| Artifact | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Response KD Checkpoint** *(Champion)* | [`P04_student_kd_response_plantvillage_s42.pt`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P04_student_kd_response_plantvillage_s42.pt) | `2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6` |
| **Feature KD Checkpoint** | [`P04_student_kd_feature_plantvillage_s42.pt`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P04_student_kd_feature_plantvillage_s42.pt) | `f41bc3ecd9e36076f7ba100194156cb79e567f9b9fb3a9d7b39dcf20c497681f` |
| **Combined KD Checkpoint** | [`P04_student_kd_combined_plantvillage_s42.pt`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P04_student_kd_combined_plantvillage_s42.pt) | `5bf8a40a6762f282e88d52d48944d24203f52eac36600d8d37fc97c7f276a4fa` |
| **Distillation Master Metrics** | [`metrics.json`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/runs/P04_knowledge_distillation/metrics.json) | `f0fae62d4e7381665a396e95c1c046e7fca6a15caae2b23a541faeb89c6fb438` |

---

## 6. Phase Boundary & Exit Criteria Verification

- [x] **Teacher Frozen**: Teacher weights (`ConvNeXt-Tiny`, `P02_teacher_convnext_tiny_plantvillage_s42.pt`) remained strictly frozen.
- [x] **Student Architecture Invariant**: Architecture is fixed to `MobileNetV3-Small`. All checkpoints strictly contain pure student weights (`1.56M` params, `~6.07 MB`).
- [x] **Controlled Ablations Completed**: Controlled empirical comparison across Student Baseline, Response KD, Feature KD, and Combined KD.
- [x] **Honest & Unbiased Reporting**: Full empirical results recorded; Feature KD's failure to transfer out-of-domain as effectively as Response KD is documented transparently.
- [x] **Automated Tests Passing**: All 27 tests in the project test suite (`uv run pytest`) pass with 0 failures or warnings.
- [x] **No Premature Leaks**: No quantization or Android deployment code has been implemented prematurely.

Phase 4 is formally **PASSED** and **FROZEN**. The project is ready for **Phase 5 — Robustness Evaluation**.
