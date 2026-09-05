# AgriRobust Phase 0 Decision Log & Assumption Registry

**Date**: 2026-09-05  
**Phase**: Phase 0 — Project Contract and Environment  
**Status**: PASSED / FROZEN  

---

## 1. Executive Summary

This document records all architectural decisions, environment specifications, verified truths, acknowledged assumptions, and deferred decisions established during Phase 0 of the AgriRobust project.

In strict compliance with `PROJECT_SOT.md` (Sections 21, 23, 24, 25), this log separates measured facts and verified data from hypotheses and planned implementations.

---

## 2. Status Taxonomies

- **DECIDED**: Concrete technical decisions adopted for the project.
- **VERIFIED**: Assertions validated empirically on the active system or against authoritative sources.
- **ASSUMED**: Working assumptions that must be empirically tested before later phase completion.
- **UNKNOWN**: Unverified attributes explicitly flagged for investigation in later phases.
- **DEFERRED**: Decisions explicitly allocated to subsequent phase gates.

---

## 3. Environment & Tooling Decisions

| Item | Status | Details & Rationale |
| :--- | :---: | :--- |
| **Python Version** | VERIFIED / DECIDED | Python 3.11.5 (`>=3.11, <3.12`) is selected and active. Python 3.14 was present on the host system, but Python 3.11 ensures rock-solid compatibility with PyTorch/torchvision binaries. |
| **Package Manager** | VERIFIED / DECIDED | `uv` (v0.10.9) is used as the ultra-fast, reproducible virtual environment and package resolution tool. |
| **PyTorch & TorchVision** | VERIFIED / DECIDED | Installed `torch==2.14.0+cpu` and `torchvision==0.29.0+cpu`. Confirmed operational via CPU execution tests. |
| **CUDA Availability** | VERIFIED | CUDA is currently `False` (CPU-only execution environment). No GPU acceleration is claimed or assumed for local runs. If remote training is used in Phase 2, CUDA capability will be validated then. |
| **Linting & Testing** | VERIFIED / DECIDED | `pytest>=8.0.0` and `ruff>=0.16.0` are configured in `pyproject.toml` for automated testing and PEP-compliant code standards. |
| **Dependency Scope** | DECIDED | No extraneous LLM, RAG, agent, or Android libraries were added. Environment is strictly constrained to core vision, evaluation, and configuration packages. |

---

## 4. Repository Structure Decisions

| Item | Status | Details & Rationale |
| :--- | :---: | :--- |
| **Package Layout** | DECIDED | Adopted a standard `src/agrirobust/` package layout containing modular subpackages (`data`, `models`, `training`, `distillation`, `robustness`, `uncertainty`, `compression`, `evaluation`, `deployment`). |
| **Configuration Structure** | DECIDED | Created `configs/` housing `project.yaml`, `datasets.yaml`, and `metrics.yaml` to ensure all parameters and schemas are decoupled from code. |
| **Data Directory Policy** | DECIDED | `data/raw/`, `data/processed/`, and `data/manifests/` are established. Raw datasets will not be committed to Git. Manifests containing checksums and splits will be version-controlled. |
| **Experiments & Logging** | DECIDED | Defined schema in `experiments/` where every experiment run stores `config.yaml`, `run_metadata.json`, and `metrics.json`. |

---

## 5. Dataset Registry & Provenance Audit

| Dataset | Provenance Status | Verified Information | Unresolved / Unknown Items (Deferred to Phase 1) |
| :--- | :---: | :--- | :--- |
| **PlantVillage** | VERIFIED Source | Source URL: `https://github.com/spMohanty/PlantVillage-Dataset`. Role: Controlled training / baseline. | Exact image count, exact class count, directory structure, and split manifest are UNKNOWN; to be audited in Phase 1. |
| **PlantDoc** | VERIFIED Source | Source URL: `https://github.com/pratikkayal/PlantDoc-Dataset`. Role: Cross-domain in-the-wild evaluation. | Exact image count, bounding-box vs crop format, exact class taxonomy are UNKNOWN; to be audited in Phase 1. |
| **PlantSeg** | UNKNOWN Source | Role: In-the-wild disease localization and segmentation. | Official repository URL, license, modality format, and mask annotations are UNKNOWN; to be investigated in Phase 1. |
| **AgroBench** | UNKNOWN Source | Role: Optional broader agricultural evaluation. | Benchmark access, task structure, and evaluation protocol are UNKNOWN; to be investigated in Phase 1. |
| **Field-Collected** | VERIFIED Source | Role: Real-world smartphone evaluation asset. Source: Internal project collection. | To be captured and annotated in Phase 9 on physical Android devices. |

---

## 6. Dataset Class / Label Policy & Unresolved Decisions

- **DECIDED**: Do not perform synthetic or arbitrary label mapping across datasets during Phase 0.
- **UNKNOWN**: Cross-dataset label alignment between PlantVillage (controlled leaf) and PlantDoc (field plants with background) is unknown and may involve disjoint or partially overlapping classes.
- **DEFERRED (Phase 1)**:
  1. Define canonical label representation (`<crop>___<condition>`).
  2. Map dataset-specific class names to canonical classes.
  3. Determine policy for unshared classes (classes present in PlantVillage but absent in PlantDoc, and vice versa).
  4. Define handling of "healthy" control classes across datasets.
  5. Check for data leakage, duplicates, or corrupted files.

---

## 7. Model & Experiment Decisions (Deferred to Phase 2+)

- **DEFERRED (Phase 2)**: Teacher architecture selection (e.g., ConvNeXt, EfficientNet, Swin), input resolution, optimizer, learning rate schedule, and pretraining weights.
- **DEFERRED (Phase 3)**: Specific MobileNet-family student architecture variant (MobileNetV3-Small vs Large, width multiplier).
- **DEFERRED (Phase 4)**: Knowledge distillation loss formulations (response-based KD vs feature distillation).
- **DEFERRED (Phase 5)**: Specific corruption perturbation severity parameters.
- **DEFERRED (Phase 6)**: Uncertainty quantification method (temperature scaling vs selective prediction classifier).
- **DEFERRED (Phase 7-8)**: Quantization scheme (PTQ INT8 vs QAT) and ExecuTorch / TFLite runtime selection for Android.

---

## 8. SOT Inconsistencies or Ambiguities

- **Ambiguity**: SOT Section 7 lists `PlantSeg` and `AgroBench` as planned datasets, but external repository URLs are not specified in the SOT.
- **Resolution**: Both datasets are explicitly registered in `configs/datasets.yaml` with status `UNKNOWN — VERIFY IN PHASE 1`. No fake URLs or image counts were invented.
- **Filename normalization**: The repository initially contained `Project_SOT.md`. In Windows systems, filenames are case-insensitive, but cross-platform Git conventions prefer exact matching. A symlink or documentation note links `PROJECT_SOT.md` to `Project_SOT.md`.

---

## 9. Phase Boundary Compliance Affirmation

No model training, dataset training pipeline, distillation, robustness experiments, compression, or Android implementation were performed during Phase 0.
