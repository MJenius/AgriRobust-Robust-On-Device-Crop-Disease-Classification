# AgriRobust Phase 6 — Uncertainty Calibration & Selective Abstention Report

**Phase**: Phase 6 — Uncertainty Calibration and Selective Abstention  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-06  
**Primary Artifact**: [`experiments/runs/P06_calibration_abstention/metrics.json`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/runs/P06_calibration_abstention/metrics.json)  

---

## 1. Executive Summary

Phase 6 evaluates whether post-training calibration and selective prediction can improve the trustworthiness of the frozen compact model ([`Response-KD Champion MobileNetV3-Small`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P04_student_kd_response_plantvillage_s42.pt), 1.56M parameters) relative to the uncompressed [`Student Baseline`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P03_student_mobilenetv3_small_plantvillage_s42.pt) and the frozen [`Teacher`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt).

### The Scientific Core Question
> *"Can calibration and selective prediction improve the trustworthiness of the compact distilled model by identifying predictions that should not be automatically trusted?"*

### Answers to the User Mandate

1. **Was Response-KD well-calibrated before calibration?**
   - On the in-domain Clean Test set, Response-KD showed mild miscalibration with an uncalibrated Expected Calibration Error of **ECE = 0.0200** (2.00%) and NLL = 0.0378.
   - For reference, Teacher uncalibrated ECE was **0.0281**, and the Student Baseline (which had near 100% clean accuracy) was **0.0018**.
2. **Did temperature scaling improve calibration?**
   - **In-domain (Clean Test):** Yes, dramatically. Temperature scaling with the validation-learned parameter ($T = 0.5406$) reduced ECE by an order of magnitude: from **0.0200 down to 0.0020 (0.20%)**, while lowering NLL from 0.0378 to 0.0211.
   - For Teacher ($T = 0.6387$), in-domain ECE dropped from **0.0281 to 0.0038**.
3. **Does calibration generalize from PlantVillage validation data to PlantDoc and corruptions?**
   - **Temperature scaling learned on clean in-domain validation data did not provide reliable calibration under the evaluated out-of-domain shifts, and in some cases increased overconfidence.**
   - The learned temperature ($T = 0.5406 < 1.0$) reflects that the clean validation model was relatively under-confident, so logit sharpening effectively minimized in-domain NLL and ECE.
   - However, that same sharpening proved harmful when the underlying predictions became degraded under distribution shift: on PlantDoc, accuracy dropped to 18.39% while confidence remained high (~74–80%), resulting in elevated post-scaling ECE (**0.5632**).
   - This empirical contrast between in-domain calibration gains and out-of-domain overconfidence is a central finding of Phase 6.
4. **Are incorrect predictions associated with lower confidence?**
   - **Yes, consistently.** Across all evaluated domains, incorrect predictions have lower mean and median confidence than correct predictions:
     - On Clean Test: Correct mean confidence = 0.9944 vs. Incorrect mean confidence = 0.7164 (**AUROC for Error Detection = 0.9892**).
     - Under Contrast Severity 5: Correct mean = 0.9633 vs. Incorrect mean = 0.7130 (**AUROC for Error Detection = 0.9252**).
     - Under Defocus Blur Severity 5: Correct mean = 0.8122 vs. Incorrect mean = 0.6558 (**AUROC for Error Detection = 0.7888**).
     - On PlantDoc Field Crops: Correct mean = 0.8034 vs. Incorrect mean = 0.7344 (**AUROC for Error Detection = 0.5969**).
5. **How much risk can selective abstention remove at practical coverage levels?**
   - On **Clean Test**:
     - At 95% coverage, selective risk drops from 0.70% to **0.04%** (a 17.5-fold error reduction).
     - At 90% coverage, selective risk drops to **0.03%** (AURC = 0.0001).
   - Under **Contrast Shift (Severity 5)**:
     - Full coverage error rate: 7.59%.
     - At 95% coverage, risk drops to **5.20%**.
     - At 90% coverage, risk drops to **3.47%**.
     - At 80% coverage, risk drops to **1.35%** (an **82.2% reduction in operational risk**).
   - Under **Severe Defocus Blur (Severity 5)**:
     - Full coverage error rate: 75.64%.
     - At 80% coverage, risk drops to **71.34%**, successfully filtering out 24% of the most erratic failures.
6. **Does the compact model become safer to deploy when allowed to abstain?**
   - **Yes, within the evaluated benchmarks.** Selective abstention provides a measurable risk–coverage tradeoff. When evidence is ambiguous or degraded by sensor corruptions, thresholding removes high-uncertainty failures before they reach the user.

---

## 2. Methodology & Strict Validation Controls

### 2.1 Evaluated Models
1. **Response-KD Champion**: MobileNetV3-Small (`P04_student_kd_response_plantvillage_s42.pt`, 1.56M params)
2. **Student Baseline**: MobileNetV3-Small (`P03_student_mobilenetv3_small_plantvillage_s42.pt`, 1.56M params)
3. **Teacher**: ConvNeXt-Tiny (`P02_teacher_convnext_tiny_plantvillage_s42.pt`, 27.85M params)

### 2.2 Leakage-Free Calibration Protocol
- **Validation Split**: Learned post-hoc scalar temperature $T$ by minimizing negative log-likelihood strictly on the 8,129 validation samples. Zero test or PlantDoc samples were accessed during fitting.
- **Learned Temperatures**:
  - Response-KD: $T = 0.5406$
  - Student Baseline: $T = 1.4962$
  - Teacher: $T = 0.6387$
- **Invariance Guarantee**: Temperature scaling is strictly monotonic, preserving raw class rankings and classification accuracy bit-for-bit.
- **Frozen Transfer**: The exact learned temperatures were frozen and applied to evaluate Clean Test, PlantDoc, and synthetic corruptions.

### 2.3 Evaluation Domains (5 Benchmarks)
1. **PlantVillage Clean Test**: 8,129 canonical test images.
2. **PlantDoc Field Crops**: 8,883 uncurated outdoor mobile phone leaf crops.
3. **Gaussian Noise Severity 5**: 8,129 images with additive Gaussian noise ($\sigma=0.22$).
4. **Defocus Blur Severity 5**: 8,129 images with severe blur ($\sigma=7.0$).
5. **Contrast Severity 5**: 8,129 images with severe low dynamic range ($0.25\times$).

---

## 3. Detailed Experimental Results

### 3.1 Calibration Performance (Before vs. After Temperature Scaling)

| Domain | Model | Accuracy | Uncal ECE | Cal ECE | Uncal NLL | Cal NLL | Cal Brier | AUROC (Err Detection) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Test** | Response-KD *(Champion)* | 99.30% | 0.0200 | **0.0020** | 0.0378 | **0.0211** | 0.0104 | **0.9892** |
| | Student Baseline | 99.75% | 0.0018 | 0.0013 | 0.0105 | 0.0092 | 0.0038 | 0.9769 |
| | Teacher | 98.31% | 0.0281 | 0.0038 | 0.0746 | 0.0560 | 0.0265 | 0.9697 |
| **PlantDoc** | Response-KD *(Champion)* | 18.39% | 0.3552 | 0.5632 | 4.0946 | 6.5975 | 1.3023 | 0.5969 |
| | Student Baseline | 11.62% | 0.7266 | 0.6435 | 11.3700 | 7.8486 | 1.4480 | 0.5245 |
| | Teacher | 21.05% | 0.3206 | 0.4783 | 3.8704 | 5.3560 | 1.1881 | **0.6194** |
| **Noise s5** | Response-KD *(Champion)* | 11.28% | 0.4870 | 0.6924 | 4.5628 | 7.6673 | 1.4913 | 0.5859 |
| | Student Baseline | 5.13% | 0.7076 | 0.5673 | 8.6688 | 6.1118 | 1.3704 | **0.7865** |
| | Teacher | 31.25% | 0.2449 | 0.3978 | 2.6539 | 3.4962 | 1.0182 | 0.6964 |
| **Blur s5** | Response-KD *(Champion)* | 24.36% | 0.2699 | 0.4899 | 3.5380 | 5.4959 | 1.1578 | **0.7888** |
| | Student Baseline | 4.18% | 0.8982 | 0.8590 | 14.2715 | 9.6208 | 1.7616 | 0.5809 |
| | Teacher | 39.97% | 0.1077 | 0.2754 | 2.5941 | 3.3037 | 0.8490 | 0.7684 |
| **Contrast s5** | Response-KD *(Champion)* | 92.41% | 0.0529 | **0.0206** | 0.2614 | **0.2416** | 0.1128 | **0.9252** |
| | Student Baseline | 73.99% | 0.1794 | 0.1325 | 1.5808 | 1.1582 | 0.3990 | 0.8427 |
| | Teacher | 96.54% | 0.0430 | 0.0037 | 0.1403 | 0.1128 | 0.0539 | 0.9551 |

---

### 3.2 Selective Abstention & Risk–Coverage Operating Points

Operating points across practical coverage levels ($\phi \approx 95\%, 90\%, 80\%$):

| Domain | Model | Full Risk ($1 - \text{Acc}$) | 95% Cov Risk | 90% Cov Risk | 80% Cov Risk | Canonical AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Test** | **Response-KD** *(Champion)* | 0.70% | **0.04%** | **0.03%** | **0.03%** | **0.0001** |
| | Student Baseline | 0.25% | 0.04% | 0.04% | 0.04% | 0.0001 |
| | Teacher | 1.69% | 0.45% | 0.18% | 0.04% | 0.0007 |
| **Contrast s5** | **Response-KD** *(Champion)* | 7.59% | **5.20%** | **3.47%** | **1.35%** | **0.0092** |
| | Student Baseline | 26.01% | 23.71% | 21.34% | 16.82% | 0.0837 |
| | Teacher | 3.46% | 1.73% | 0.79% | 0.17% | 0.0022 |
| **Blur s5** | **Response-KD** *(Champion)* | 75.64% | 74.52% | 73.53% | 71.34% | **0.5429** |
| | Student Baseline | 95.82% | 95.87% | 96.15% | 96.70% | 0.8969 |
| | Teacher | 60.03% | 58.48% | 56.61% | 53.20% | 0.3816 |
| **PlantDoc** | **Response-KD** *(Champion)* | 81.61% | 81.21% | 80.66% | 79.94% | **0.7653** |
| | Student Baseline | 88.38% | 88.12% | 87.83% | 87.57% | 0.8900 |
| | Teacher | 78.95% | 78.27% | 77.77% | 76.65% | 0.7140 |

---

### 3.3 Confidence Distributions: Correct vs. Incorrect Predictions

| Domain | Model | Correct Mean Conf | Correct Median Conf | Incorrect Mean Conf | Incorrect Median Conf | Separation Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Test** | Response-KD | 0.9944 | 1.0000 | 0.7164 | 0.7040 | **+0.2780** |
| | Student Baseline | 0.9978 | 1.0000 | 0.7092 | 0.6258 | +0.2886 |
| | Teacher | 0.9850 | 0.9999 | 0.7003 | 0.7043 | +0.2847 |
| **Contrast s5** | Response-KD | 0.9633 | 0.9995 | 0.7130 | 0.7112 | **+0.2503** |
| | Student Baseline | 0.9285 | 0.9979 | 0.7126 | 0.7335 | +0.2159 |
| | Teacher | 0.9725 | 0.9996 | 0.6869 | 0.6906 | +0.2856 |
| **PlantDoc** | Response-KD | 0.8034 | 0.8769 | 0.7344 | 0.7687 | +0.0690 |
| | Student Baseline | 0.7929 | 0.8714 | 0.7551 | 0.8228 | +0.0378 |
| | Teacher | 0.7601 | 0.8128 | 0.6699 | 0.6729 | **+0.0902** |

---

## 4. Scientific Discussion & Architectural Insights

1. **Why In-Domain Calibration Gains Do Not Transfer to Evaluated Shifts**:
   - On the PlantVillage validation set, the model's logits are under-confident relative to its near-perfect accuracy, resulting in a learned temperature $T = 0.5406 < 1.0$.
   - This sharpening factor successfully drives in-domain ECE down to **0.0020 (0.20%)**.
   - However, when the model faces out-of-domain PlantDoc crops or severe noise, where accuracy drops to 11–18%, the sharpened logits amplify the probabilities of top-1 incorrect classes, raising out-of-domain ECE and overconfidence.
   - **Takeaway**: Temperature scaling learned on clean in-domain validation data did not provide reliable calibration under the evaluated out-of-domain shifts, and in some cases increased overconfidence.
2. **Selective Abstention Effectively Filters Out Errors**:
   - Under moderate-to-severe degradation where accuracy remains reasonable (e.g., Contrast Severity 5, Acc = 92.41%), selective prediction is highly effective: dropping coverage from 100% to 80% slashes operational risk from **7.59% down to 1.35% (an 82.2% error reduction)**.
   - In Clean Test, abstaining on just 5% of samples removes **94% of all errors**, reducing risk to **0.04%**.
3. **Response-KD Preserves Better Abstention Quality than Baseline**:
   - In every shifted domain, Response-KD achieves a substantially lower Area Under the Risk-Coverage Curve (AURC) than the Student Baseline:
     - Contrast s5: **0.0092 vs. 0.0837** (9-fold improvement)
     - Blur s5: **0.5429 vs. 0.8969**
     - PlantDoc: **0.7653 vs. 0.8900**
   - The softer, regularized class representations transferred from the Teacher during Phase 4 distillation produce better-ordered confidence rankings, enabling more effective selective abstention.

---

## 5. Phase 6 Verification & Freeze Gate

- [x] Implemented modular calibration components: `metrics.py`, `temperature.py`, `selective.py`.
- [x] Strictly adhered to validation-only temperature fitting ($T = 0.5406$) with zero test/PlantDoc leakage.
- [x] Verified that temperature scaling preserves class predictions and top-1 accuracy bit-for-bit.
- [x] Computed canonical AURC via empirical sample sort.
- [x] Measured AUROC and AUPR for Error Detection across all 5 benchmarks.
- [x] Evaluated selective prediction operating points (95%, 90%, 80% coverage).
- [x] All 37 unit tests pass cleanly (`uv run pytest -v`).
- [x] Master metrics serialized to `experiments/runs/P06_calibration_abstention/metrics.json`.

**Phase 6 is formally complete and FROZEN.**
