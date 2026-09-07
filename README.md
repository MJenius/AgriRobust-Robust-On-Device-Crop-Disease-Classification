# AgriRobust

**Robust On-Device Crop Disease Diagnosis**  
*Efficient Computer Vision with Distillation, Calibration, and Abstention*

---

## Overview

AgriRobust is a research and engineering project focused on developing a compact agricultural computer-vision model that approaches the performance of a high-capacity teacher model while remaining sufficiently small, memory-efficient, and fast for fully offline smartphone inference.

The project is fundamentally a model-efficiency, calibration, and robustness study under real-world distribution shift, rather than a generic crop-disease classification application.

## Central Research Question

> **Can a high-performing agricultural vision model be compressed into a substantially smaller model that:**
> 1. retains most of the teacher's real-world diagnostic performance,
> 2. remains robust under distribution shift,
> 3. produces calibrated confidence / uncertainty estimates,
> 4. supports selective abstention when evidence is insufficient, and
> 5. runs fully offline on a smartphone?

## High-Level Pipeline

```text
Camera image
    │
    ▼
Image Preprocessing
    │
    ▼
Compact Vision Model (MobileNet family)
    │
    ▼
Disease Prediction & Uncertainty Estimation
    │
    ▼
Decision Rule: High Confidence Prediction OR Selective Abstention
```

## Project Status: Completed & Frozen (Phases 0–9)

**All 9 project phases have been successfully executed, evaluated, cryptographically verified, and frozen.**

- Refer to [CURRENT_PHASE.md](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/CURRENT_PHASE.md) for phase history and signoffs.  
- The authoritative project contract and specification is maintained in [PROJECT_SOT.md](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/PROJECT_SOT.md).
- Detailed findings, benchmarking, and failure analysis are published in [reports/phase_09_final_synthesis.md](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/reports/phase_09_final_synthesis.md).

---

## Executive Results Summary

| Metric | Teacher (`ConvNeXt-Tiny`) | Student Baseline (`MobileNetV3-Small`) | Response-KD Student *(Champion)* | Deployed INT8 Champion (`model_int8_dynamic.pt`) |
| :--- | :---: | :---: | :---: | :---: |
| **Parameters** | 27.85M | 1.56M (-94.4%) | 1.56M | **1.56M** (5.59% of teacher) |
| **Model Size** | 106.31 MB | 6.08 MB | 6.07 MB | **4.56 MB** (28.1% container compression) |
| **Clean Test Accuracy** | 98.31% | **99.75%** | 99.30% | **99.31%** |
| **PlantDoc Cross-Domain Acc** | **21.05%** | 11.62% | 18.39% | **18.41%** (Recovers 63.6% of teacher gap) |
| **Contrast s5 Stress Acc** | 96.54% | 73.99% | 92.41% | **92.43%** (+18.4% over baseline) |
| **Calibrated Clean ECE** | 0.0038 | 0.0013 | 0.0020 | **0.0016** ($T = 0.5406$) |
| **Host CPU Latency** | 75.29 ms | 18.41 ms | 4.73 ms | **5.25 ms** (Model) / **9.18 ms** (End-to-End) |
| **Physical Phone Latency** | — | — | — | **45.40 ms mean** (Samsung Galaxy A14 / Exynos 1330) |

> [!WARNING]
> **Safety Notice & Out-of-Domain Failure Mode**:
> Temperature scaling ($T=0.5406$) and selective abstention ($\tau=0.8143$) effectively filter uncertainty on familiar distributions, reducing clean ECE to 0.16%. However, empirical testing on the physical smartphone (`data/examples/03_plantdoc_potato_late_blight.jpg`) demonstrated an accepted false diagnosis (`Corn — Gray leaf spot` at **99.7% confidence**) under combined optical Moiré artifacts and out-of-domain shift. **Selective abstention must not be treated as a foolproof safety guarantee in agricultural deployments.**

---

## Repository Structure

```text
agrirobust/
├── PROJECT_SOT.md       # Authoritative Single Source of Truth
├── CURRENT_PHASE.md     # Phase tracking and completion history (Phases 0–9)
├── README.md            # Project overview, synthesis, and instructions
├── pyproject.toml       # Python package configuration and pinned dependencies
├── configs/             # Experiment, dataset, and metric configurations
│   ├── project.yaml
│   ├── datasets.yaml
│   └── metrics.yaml
├── data/
│   ├── raw/             # Unprocessed raw dataset archives (gitignored)
│   ├── processed/       # Canonicalized datasets (PlantVillage 38 classes, PlantDoc)
│   ├── manifests/       # Split and integrity metadata files
│   └── examples/        # 6 qualitative smartphone camera evaluation samples & failure log
├── src/agrirobust/      # Core package
│   ├── data/            # Canonical dataset loaders and transforms
│   ├── models/          # Teacher (ConvNeXt-Tiny) & Student (MobileNetV3-Small)
│   ├── training/        # Supervised training loop
│   ├── distillation/    # Response, feature, and combined knowledge distillation
│   ├── robustness/      # 7 synthetic corruption families (5 severities each)
│   ├── uncertainty/     # Temperature calibration and selective prediction
│   ├── compression/     # Pruning and dynamic INT8 quantization
│   ├── evaluation/      # Metrics, ECE, NLL, AURC, and reporting engines
│   └── deployment/      # TorchScript export, preprocessing parity, and benchmarks
├── experiments/         # Run outputs, logs, and checkpoints
│   ├── checkpoints/     # Cryptographically verified .pt weights
│   └── runs/            # Master metric logs (P02 through P09)
├── reports/             # Comprehensive phase research reports (Phases 00 through 09)
├── android/             # Standalone Android application (PyTorch Mobile Lite 1.13.1)
└── tests/               # Automated unit, regression, and release test suite (53 tests)
```

## Environment Setup

### Prerequisites
- Python 3.11 (`>=3.11, <3.12`)
- Package manager: `uv` (recommended) or standard `venv` + `pip`

### Installation with `uv`

```bash
# 1. Create a virtual environment with Python 3.11
uv venv .venv --python 3.11

# 2. Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install the project in editable mode with development dependencies
uv pip install -e ".[dev]"
```

## Running Verification Tests

To execute the complete regression and release test suite:

```bash
uv run pytest -v
```
