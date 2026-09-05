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

## Current Project Phase

**Current Phase: Phase 0 — Project Contract and Environment**

Refer to [CURRENT_PHASE.md](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/CURRENT_PHASE.md) for allowed activities and phase boundaries.  
The authoritative project specification is preserved in [PROJECT_SOT.md](file:///c:/Users/mjeni/OneDrive/Desktop/Own%20Projects/AgriRobust%20-%20Small%20LM%20Plants%20Disease%20Classification/Project_SOT.md).

> **Note**: In Phase 0, no model training, knowledge distillation, robustness benchmarking, compression, or Android app implementation is performed.

## Repository Structure

```text
agrirobust/
├── Project_SOT.md       # Authoritative Single Source of Truth
├── CURRENT_PHASE.md     # Phase tracking and phase gates
├── README.md            # Project overview and instructions
├── pyproject.toml       # Python package configuration and pinned dependencies
├── configs/             # Experiment, dataset, and metric configurations
│   ├── project.yaml
│   ├── datasets.yaml
│   └── metrics.yaml
├── data/
│   ├── raw/             # Unprocessed raw dataset archives (not committed)
│   ├── processed/       # Canonicalized and verified datasets
│   └── manifests/       # Split and integrity metadata files
├── src/                 # Source package: agrirobust
│   ├── agrirobust/
│   │   ├── __init__.py
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   ├── distillation/
│   │   ├── robustness/
│   │   ├── uncertainty/
│   │   ├── compression/
│   │   ├── evaluation/
│   │   └── deployment/
├── experiments/         # Experiment run outputs, logs, and configs
│   └── README.md
├── reports/             # Phase reports, decision logs, and Pareto analyses
│   ├── README.md
│   └── phase_00_decisions.md
├── tests/               # Unit and regression test suite
└── android/             # Future on-device Android deployment application
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

## Running Phase 0 Verification Tests

To verify environment integrity, configuration validity, SOT alignment, and directory structure:

```bash
pytest tests/ -v
```
