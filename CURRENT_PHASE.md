# Phase: 7 — Deployment-Aware Compression (Quantization & Pruning)

## Objective
Determine how much model size and inference efficiency can be gained through post-training deployment compression (dynamic quantization, eager static PTQ compatibility probe, and unstructured pruning) on the frozen Phase 4 Response-KD Champion (`MobileNetV3-Small`, 1.56M parameters, 6.07 MB FP32 baseline) before meaningful degradation appears in clean accuracy, distribution-shift robustness, and selective prediction reliability.

## Status
PASSED / FROZEN (2026-09-07)

## Completed Work
- **Modular Compression Suite Implemented**:
  - `src/agrirobust/compression/quantization.py`: Dynamic INT8 quantization wrapper, eager static PTQ environment compatibility probe, and model serialization/deserialization.
  - `src/agrirobust/compression/pruning.py`: Deterministic L1 unstructured pruning with permanent mask removal and layer-wise/global sparsity calculation.
  - `src/agrirobust/compression/benchmark.py`: Standardized single-image CPU latency/throughput benchmarker and model footprint measurement.
- **Experimental Pipeline Executed**:
  - `experiments/scripts/run_compression_suite.py`: Full evaluation of 5 candidates across 5 benchmark domains:
    1. PlantVillage Clean Test (8,129 images)
    2. PlantDoc Field Crops Cross-Domain (8,883 crops)
    3. Gaussian Noise Severity 5
    4. Defocus Blur Severity 5
    5. Contrast Severity 5
  - Evaluated with frozen Phase 6 calibration parameters ($T_{\text{cal}} = 0.5406$, $\tau_{\text{val}} = 0.8143$).
- **Key Scientific Findings**:
  - **Dynamic INT8 Quantization**: Quantizing classifier linear layers to `qint8` reduced serialized model size by **`29.08%`** (from **`6.07 MB`** down to **`4.30 MB`**, a **`1.41x`** compression ratio).
  - **Zero Metric Loss**: Dynamic INT8 retained **`100.03%`** of clean Macro F1 (`0.9887` vs `0.9884`), **`100.07%`** of PlantDoc cross-domain Macro F1 (`0.1368` vs `0.1367`), and matched FP32 across all stress corruptions (Contrast s5 Macro F1: `0.8904` vs `0.8905`).
  - **Calibration Preservation**: Transferred frozen validation temperature ($T=0.5406$) achieved an outstanding calibrated ECE of **`0.0016`** (vs FP32 `0.0020`), with identical canonical AURC (**`0.0001`**).
  - **Static Quantization Environment Probe**: Probed eager static quantization on PyTorch CPU `onednn`; transparently documented backend limitation (`quantized::conv2d.new` unavailable on eager CPU) as `NOT SUPPORTED / NOT EXECUTED`.
  - **Pruning Reality**: Pruning at 20%, 40%, and 60% sparsity achieved nominal sparsity only. Due to dense CPU BLAS kernels, latency did not improve (7.09–8.58 ms) and serialized file sizes remained unchanged (6.065 MB). At 60% sparsity, severe noise accuracy collapsed (11.28% -> 8.49%).
- **Phase 8 Candidate Selected**:
  - Champion: `P07_int8_dynamic`
  - Checkpoint: `experiments/runs/P07_deployment_compression/artifacts/model_int8_dynamic.pt` (4.30 MB)
- **Verification**:
  - FP32 source checkpoint verified immutable (`SHA-256: 2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6`).
  - All 43 project tests passed cleanly in `uv run pytest -v`.
  - Master metrics exported to `experiments/runs/P07_deployment_compression/metrics.json`.
  - Comprehensive scientific report published in `reports/phase_07_compression.md`.

## Next Phase
Phase 8 — On-Device Deployment and Mobile Runtime Validation
