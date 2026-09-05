# AgriRobust Phase 1 — Dataset Foundation Report

**Phase**: Phase 1 — Dataset Foundation  
**Status**: PASSED / FROZEN  
**Date**: 2026-09-05  

---

## 1. Executive Summary

Phase 1 established the empirical dataset foundation for AgriRobust. In strict adherence to `PROJECT_SOT.md` (Sections 7, 8, 13, 21, 23, 24) and user directives, the core research backbone was frozen:
1. **PlantVillage** → Controlled-domain baseline & training dataset
2. **PlantDoc** → In-the-wild cross-domain classification benchmark (evaluated as leaf crops)
3. **PlantSeg** → In-the-wild disease segmentation and localization benchmark
4. **AgroBench** → Optional broader agricultural VLM sanity check
5. **Field-Collected** → Reserved for real-world smartphone test evaluation (Phase 9)

All datasets have been acquired, checksummed, audited for corruptions and duplicates, harmonized into a canonical 38-class `<crop>___<condition>` taxonomy, and frozen into deterministic manifests under `data/manifests/`.

---

## 2. Acquired Datasets & Verified Statistics

| Dataset | Modality / Task | Verified Source & License | Total Acquired Samples | Valid Evaluated Samples | Verified Classes | Corruptions / Duplicates |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **PlantVillage** | RGB Image / Multi-Class Disease Classification | `https://github.com/spMohanty/PlantVillage-Dataset` (CC BY-SA) | 54,305 images | 54,305 images | 38 classes (14 crops, 12 healthy, 26 disease) | 0 corrupted, 21 internal duplicates |
| **PlantDoc** | RGB Image + Bounding Box / Cropped Leaf Classification | `https://github.com/pratikkayal/PlantDoc-Dataset` & `PlantDoc-Object-Detection-Dataset` (CC BY 4.0) | 2,581 raw images (8,921 boxes) | 8,883 leaf crops (filtered min_size >= 20px) | 29 classes (all 29 shareable with PlantVillage) | 0 unreadable images |
| **PlantSeg** | RGB Image + Mask / Semantic Segmentation | `https://zenodo.org/records/14935094` via `https://github.com/tqwei05/PlantSeg` (CC BY 4.0) | 11,458 image/mask pairs | 11,458 samples | 115 disease segmentation categories | 0 unreadable archive files |
| **AgroBench** | Multimodal Image-Text / VLM Benchmark | `https://huggingface.co/datasets/Project-AgML/AgroBench` (CC BY-NC 4.0) | 4,342 question/image pairs | Optional external benchmark | 203 crops / 682 sub-classes | N/A (held as external sanity check) |

---

## 3. Canonical Label Taxonomy & Harmonization

We established the canonical representation format: `<crop>___<condition_or_healthy>`.

- **Total Canonical Classes**: 38
- **Shared Cross-Domain Classes (PlantVillage & PlantDoc)**: 29
- **PlantVillage-Only Classes (Clean Evaluation Only)**: 9
  - `apple___black_rot`
  - `cherry___powdery_mildew`
  - `corn___healthy`
  - `grape___esca_black_measles`
  - `grape___leaf_blight`
  - `orange___citrus_greening`
  - `peach___bacterial_spot`
  - `strawberry___leaf_scorch`
  - `tomato___target_spot`
- **PlantDoc-Only Classes**: 0 (100% of PlantDoc classes map directly into canonical classes).

---

## 4. Split Policy & Manifests

### PlantVillage Split Policy (`data/manifests/plantvillage_manifest.json`)
- **Deterministic Partitioning**: Stratified per class with `random_seed=42`.
- **Training**: 70% (38,047 images)
- **Validation**: 15% (8,129 images)
- **Test (Clean-Domain)**: 15% (8,129 images)

### PlantDoc Cross-Domain Test Manifest (`data/manifests/plantdoc_crops_manifest.json`)
- **Split Role**: 100% held-out test evaluation.
- **Crop Pipeline**: Bounding box coordinates extracted with minimum box dimension of 20 pixels.
- **Total Valid Crops**: 8,883 crops across 29 classes.

### PlantSeg Segmentation Manifest (`data/manifests/plantseg_manifest.json`)
- Preserved official benchmark splits: 8,020 train, 1,146 val, 2,292 test (11,458 total).

---

## 5. Integrity & Leakage Verification

1. **Corruption Audit**: Zero corrupted or truncated image files detected across PlantVillage, PlantDoc, and PlantSeg.
2. **Duplicate Detection**: SHA-256 hash checks identified 21 exact internal duplicates within PlantVillage (cataloged in the manifest without silent deletion).
3. **Leakage Elimination**:
   - PlantVillage train/val/test splits are strictly disjoint partitions.
   - PlantDoc is 100% held out exclusively for cross-domain evaluation (zero overlap with training).
   - PlantDoc task was locked to leaf crops, eliminating background detection confounding.

---

## 6. Phase Boundary Compliance Affirmation

"No model training, dataset training pipeline, distillation, robustness experiments, compression, or Android implementation were performed."
