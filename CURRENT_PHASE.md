# Phase: 8 — Android Deployment and On-Device Validation

## Objective
Convert the frozen Phase 7 deployment champion (`P07_int8_dynamic`, 4.30 MB state dict, 1.56M parameter `MobileNetV3-Small` response-distilled student) into a reproducible mobile inference artifact and standalone Android application. Evaluate numerical parity, confidence calibration transfer ($T = 0.5406$), selective abstention decision parity ($\tau = 0.8143$), preprocessing parity, and host-side mobile container execution latency.

## Status
PASSED / FROZEN (2026-09-07)

## Completed Work
- **Modular Deployment Infrastructure**:
  - `src/agrirobust/deployment/export.py`: Exported frozen model to mobile TorchScript container, packaged canonical `labels.json` (38 classes) and `deployment_metadata.json`.
  - `src/agrirobust/deployment/preprocessing.py`: Formalized mathematical preprocessing contract (`Resize((224, 224))` direct bilinear scaling, ImageNet mean/std normalization) with verified 0.000000 maximum difference against Android bitmap normalization.
  - `src/agrirobust/deployment/validate_export.py`: Evaluated numerical and prediction parity across Clean Test and stress shift conditions.
  - `src/agrirobust/deployment/mobile_benchmark.py`: Host-side mobile runtime container benchmarker separating model-only latency from end-to-end (preprocessing + inference + calibration + decision) latency.
- **Standalone Android Application**:
  - `android/build.gradle`, `android/app/build.gradle`: Gradle build configuration targeting Android SDK 34 with PyTorch Mobile Lite (`org.pytorch:pytorch_android_lite:1.13.1`).
  - `android/app/src/main/AndroidManifest.xml`: Standard application manifest with camera and gallery permissions.
  - `android/app/src/main/java/com/agrirobust/classifier/AgriClassifier.kt`: On-device inference engine implementing ImageNet tensor normalization, model forward pass, frozen temperature scaling ($T=0.5406$), and selective abstention decision logic ($\tau=0.8143$).
  - `android/app/src/main/java/com/agrirobust/classifier/MainActivity.kt`: Diagnostic UI displaying leaf image picker, classification result, confidence score, and clear visual "Accepted" vs "Abstained / Uncertain" status badge.
  - `android/app/src/main/assets/`: Synchronized deployment assets (`model_int8_dynamic.pt` at 4.56 MB, fallback `model_fp32.pt` at 6.35 MB, `labels.json`, and `deployment_metadata.json`).
- **Key Empirical Results**:
  - **Prediction Parity**: **`100.00% Top-1 agreement`** and **`100.00% Top-5 agreement`** between Python reference and the mobile TorchScript container on 1,000 Clean Test images and 500 Contrast s5 stress images.
  - **Abstention Parity**: **`100.00% decision agreement`** on whether to accept or abstain.
  - **Storage Compression**: `model_int8_dynamic.pt` occupies **`4.56 MB`**, achieving a **`28.12% storage reduction / compression`** over the FP32 mobile container (6.35 MB).
  - **Host Mobile-Runtime Benchmarks**:
    - Model-Only Latency: **`5.25 ms`** (**`190.5 FPS`**).
    - End-to-End Latency: **`9.18 ms`** (**`108.9 FPS`**; preprocessing overhead: ~3.93 ms).
  - **Physical On-Device Hardware Validation (Samsung Galaxy SM-A146B / Exynos 1330 / Android 15)**:
    - Successfully built and installed `app-debug.apk` directly to physical device via ADB.
    - True physical phone inference latency over 50 iterations: **`45.40 ms mean`**, **`44.00 ms median (P50)`**, **`49.00 ms (P90)`**, **`53.00 ms (P95)`**.
    - End-to-end mobile inference (~55 ms) easily beats the target project budget (< 100 ms) by **2.2×**.
    - Verified on-device temperature scaling ($T=0.5406$) and selective abstention ($\tau=0.8143$) functionality.
- **Verification**:
  - Source checkpoint SHA-256 confirmed immutable: `2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6`.
  - All 49 project tests passed cleanly in `uv run pytest -v`.
  - Published master report in `reports/phase_08_android_deployment.md` and exported metrics in `experiments/runs/P08_android_deployment/metrics.json`.

## Next Phase
All Phases 0 through 8 Complete — Final Project Review & Deployment Ready
