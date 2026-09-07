# AgriRobust Phase 9 — Final Validation, Reproducibility & Project Release Report

**Phase**: Phase 9 — Final Validation, Reproducibility & Project Release  
**Status**: COMPLETED / FROZEN  
**Date**: 2026-09-07  
**Artifact Directory**: `experiments/runs/P09_final_synthesis/`  
**Consolidated Metrics**: [`summary.json`](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/experiments/runs/P09_final_synthesis/summary.json)  

---

## 1. Executive Summary

Phase 9 represents the final formal phase of the **AgriRobust** research project. This phase consolidates all empirical measurements across the full lifecycle—from Phase 0 project establishment through teacher baseline, student baseline, knowledge distillation, robustness evaluation, confidence calibration & selective prediction, compression, and Android mobile deployment.

### Key Objectives Accomplished in Phase 9:
1. **Removal of Overclaims**: Clarified across all reports and documentation that confidence calibration and selective abstention are heuristic safety layers, not universal guarantees against out-of-domain failures.
2. **Explicit Documentation of Failure Modes**: Transparently documented the **99.7% high-confidence false-acceptance failure** observed during physical camera screen-recapture testing on `03_plantdoc_potato_late_blight.jpg`.
3. **Preservation of Qualitative Evidence**: Formally framed the 6 physical Android smartphone tests as qualitative real-world case studies demonstrating both diagnostic consistency on clean leaves, effective abstention on degraded crops, and vulnerability to optical shifts.
4. **End-to-End Metric Synthesis**: Constructed the master cross-phase comparison table contrasting the **Teacher (`ConvNeXt-Tiny`)**, **Student Baseline (`MobileNetV3-Small`)**, **Response-KD Student**, and the **Deployed INT8 Champion (`model_int8_dynamic.pt`)**.
5. **Full Checkpoint & Asset Reproducibility**: Audited and cryptographically locked all primary model checkpoints, deployment containers, and configuration files via immutable SHA-256 digests.
6. **Final Test Suite Execution**: Validated that all automated regression and unit tests pass with zero errors.

---

## 2. Master Cross-Phase Empirical Comparison

The table below synthesizes the primary measured metrics across all developmental phases:

| Dimension / Metric | Teacher (`ConvNeXt-Tiny`) | Student Baseline (`MobileNetV3-Small`) | Response-KD Student *(Champion)* | Deployed INT8 Champion (`model_int8_dynamic.pt`) | Target / Compliance Requirement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 27.85M | 1.56M | 1.56M | **1.56M** | $\le 10\%$ of teacher (**5.59%**, PASSED) |
| **Storage Footprint** | 106.31 MB (FP32) | 6.08 MB (FP32) | 6.07 MB (FP32) | **4.56 MB** (Dynamic INT8) | $\le 15$ MB container (**4.56 MB**, PASSED) |
| **Storage Compression vs FP32** | — | — | — | **28.12% reduction** | Measurable footprint reduction |
| **Clean Test Accuracy (PV)** | 98.31% | **99.75%** | 99.30% | **99.31%** | Retain $\ge 90\%$ of clean accuracy |
| **Clean Test Macro F1** | 0.9794 | **0.9962** | 0.9884 | **0.9887** | Retain discriminative balance |
| **Cross-Domain Acc (PlantDoc)** | **21.05%** | 11.62% | 18.39% | **18.41%** | Measure real-world domain gap |
| **Cross-Domain Macro F1** | **0.2062** | 0.1298 | 0.1791 | **0.1791** | Distillation recovers **+37.98%** rel F1 |
| **Domain Gap Recovered by KD** | — | 0.00% | **+63.61% of gap** | **+63.61% of gap** | Recovers bulk of teacher gap |
| **Mean Relative Corruption Error** | 95.81% | 100.00% (Ref) | **79.87%** | **79.91%** | Lower is better (20.1% lower error) |
| **Contrast s5 Accuracy** | 96.54% | 73.99% | 92.41% | **92.43%** | Major student recovery (+18.4%) |
| **Gaussian Noise s5 Accuracy** | **31.25%** | 5.13% | 11.28% | **11.26%** | Teacher capacity advantage remains |
| **Defocus Blur s5 Accuracy** | **39.97%** | 4.18% | 24.36% | **24.31%** | KD improves blur resilience 5.8x |
| **Calibrated ECE (Clean Test)** | 0.0038 | 0.0013 | 0.0020 | **0.0016** | ECE $\le 0.05$ (0.16%, PASSED) |
| **Calibrated NLL (Clean Test)** | 0.0560 | 0.0092 | 0.0211 | **0.0211** | Negative log likelihood |
| **Host CPU Latency (Single Img)** | 75.29 ms | 18.41 ms | 4.73 ms | **5.25 ms** (Model) / **9.18 ms** (E2E) | < 100 ms budget (Host CPU) |
| **Physical Phone Latency (A14)** | — | — | — | **45.40 ms mean** (~22 FPS) | < 100 ms on budget ARM silicon |
| **Prediction Parity (Mobile vs Py)** | — | — | — | **100.00% Top-1 / Top-5 Agreement** | Numerical container fidelity |
| **Abstention Decision Parity** | — | — | — | **100.00% Decision Agreement** | Exact operational parity |

---

## 3. High-Confidence Failure Analysis & Qualitative Evidence

### 3.1 The 99.7% False-Acceptance Failure Mode
During on-device physical testing on a Samsung Galaxy A14 photographing test leaves displayed on a monitor:
- **Test Image**: `data/examples/03_plantdoc_potato_late_blight.jpg` (a natural field lesion crop of potato late blight).
- **Direct Digital Tensor Execution**:
  - Predicted Class: `Bell pepper — Healthy`
  - Confidence: **55.9%**
  - Decision: **`ABSTAINED`** (below $\tau = 0.8143$).
- **Physical Screen Recapture**:
  - Predicted Class: `Corn — Gray leaf spot`
  - Confidence: **`99.7%`**
  - Decision: **`ACCEPTED (HIGH-CONFIDENCE ERROR)`**

### 3.2 Scientific Implications:
1. **Calibration Is Local to Distribution**: Temperature scaling ($T = 0.5406$) calibrated strictly on in-domain validation data effectively prevents overconfidence on familiar leaf distributions, reducing Clean ECE from 2.0% to 0.16%. However, it does not guarantee conservative confidence under compounded out-of-distribution shifts (physical Moiré frequency aliasing + screen backlight glare + unseen leaf texture).
2. **Abstention as Decision Support, Not Safety Shield**: Selective prediction with $\tau = 0.8143$ filters out modest uncertainty, successfully abstaining on `01` under camera noise (56.3%) and `06` (47.4%). However, extreme domain shift can push an erroneous class logit far into the tail, producing near-certainty ($99.7\%$).
3. **Operational Recommendation**: Commercial or clinical agricultural deployment must combine confidence thresholding with **out-of-distribution (OOD) density estimators** or **conformal prediction sets** before relying on automated action.

---

## 4. Reproducibility & Cryptographic Checkpoint Registry

All released artifacts have been verified and sealed with SHA-256 hashes:

| Artifact Name | Relative Path | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Teacher Checkpoint (s42)** | `experiments/checkpoints/P02_teacher_convnext_tiny_plantvillage_s42.pt` | 106.31 MB | `7b80b48404531597f098571e56d53f03676305b57a7938ff27a4389e8fa1d1af` |
| **Student Baseline Checkpoint** | `experiments/checkpoints/P03_student_mobilenetv3_small_plantvillage_s42.pt` | 6.08 MB | `be00445d431c4f5264b3df3633dbe4aa5dd5184852077e923e3cb12c32cf4da9` |
| **Response-KD Champion** | `experiments/checkpoints/P04_student_kd_response_plantvillage_s42.pt` | 6.07 MB | `2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6` |
| **Mobile TorchScript Container** | `android/app/src/main/assets/model_int8_dynamic.pt` | 4.56 MB | `500ea8b7ed942f16ae59da60b5cab262aebb46f279b2d936bc8f7edf45617200` |
| **Deployment Labels (38 cl.)** | `android/app/src/main/assets/labels.json` | 1.48 KB | `88863aa98ea38a6a68297b7672ea46fb6599b109e46a782a2ba7ca605bf68673` |
| **Deployment Metadata** | `android/app/src/main/assets/deployment_metadata.json` | 1.46 KB | `c213459c5d0705a6234e402bbfbdfd92f5ca37617651a02ae305374625b5b03c` |

---

## 5. Phase 9 Completion & Repository Freeze Checklist

- [x] Corrected overclaims in `README.md`, `data/examples/README.md`, and all phase reports.
- [x] Transparently documented the 99.7% false-acceptance high-confidence failure mode.
- [x] Retained the 6 real-device smartphone tests as qualitative case studies.
- [x] Master synthesis table generated comparing Teacher, Student Baseline, Response KD, and Deployed INT8 model.
- [x] Machine-readable summary generated at `experiments/runs/P09_final_synthesis/summary.json`.
- [x] All cryptographic hashes verified against frozen source checkpoints.
- [x] Automated test suite passing 100% cleanly.
- [x] Repository state frozen and marked complete.

**AgriRobust is hereby formally completed, fully documented, and frozen.**
