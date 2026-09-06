# Phase: 6 — Uncertainty Calibration and Selective Abstention

## Objective
Determine whether the frozen Response-KD Champion (`MobileNetV3-Small`, 1.56M parameters) can provide reliable confidence estimates and safely abstain on uncertain predictions under distribution shift, evaluating Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Brier score, Error-Detection AUROC/AUPR, and empirical-sort canonical AURC across clean and shifted benchmarks.

## Status
PASSED / FROZEN (2026-09-06)

## Completed Work
- Implemented modular calibration infrastructure:
  - `src/agrirobust/calibration/metrics.py`: Standard 15-bin ECE, multi-class Brier score, NLL, and Error-Detection AUROC/AUPR using confidence to separate failures from correct predictions.
  - `src/agrirobust/calibration/temperature.py`: Post-hoc temperature scaling fitted via L-BFGS-B maximum likelihood strictly on the canonical validation split.
  - `src/agrirobust/calibration/selective.py`: Confidence-thresholded selective prediction, canonical empirical-sort AURC, and fixed operating-point extraction (95%, 90%, 80% coverage).
- Evaluated 3 models across 5 target domains (Clean Test, PlantDoc Field Crops, Gaussian Noise s5, Defocus Blur s5, Contrast s5):
  - **In-Domain Calibration**: On Clean Test, Response-KD ECE reduced by an order of magnitude: **`0.0200 -> 0.0020 (0.20%)`** with NLL dropping from 0.0378 to 0.0211.
  - **Out-of-Domain Calibration Divergence**: Temperature scaling learned on clean in-domain validation data did not provide reliable calibration under the evaluated out-of-domain shifts, and in some cases increased overconfidence. On PlantDoc, accuracy dropped to 18.39% while confidence remained ~74–80% (calibrated ECE = 0.5632), demonstrating that in-domain logit sharpening ($T=0.5406$) can amplify overconfidence when predictions degrade under shift.
  - **Error Detection Quality**: Incorrect predictions consistently exhibited lower confidence than correct predictions across all domains (Clean Test Error-AUROC = **`0.9892`**, Contrast s5 Error-AUROC = **`0.9252`**, Blur s5 Error-AUROC = **`0.7888`**).
  - **Selective Risk Reduction**: Under Contrast Severity 5, abstaining on the lowest 20% confident samples dropped operational risk from **`7.59% down to 1.35%`** (an 82.2% risk reduction, with AURC = **`0.0092`** vs Student Baseline's 0.0837).
  - On Clean Test, abstaining on just 5% of samples reduced operational risk from 0.70% to **`0.04%`** (AURC = **`0.0001`**).
- Verified zero data leakage: Temperature scaling ($T = 0.5406$) and validation threshold $\tau_{\text{val}}$ were optimized strictly on validation data.
- Built comprehensive unit test suite in `tests/test_phase_06_calibration.py` (all 37 unit tests pass cleanly).
- Published detailed Phase 6 report in `reports/phase_06_calibration.md` and saved master metrics in `experiments/runs/P06_calibration_abstention/metrics.json`.

## Next Phase
Phase 7 — Deployment-Aware Compression (Quantization & Pruning)

