# Phase: 1 — Dataset Foundation

## Objective
Establish a reproducible, verified, and audited dataset foundation for the AgriRobust research spine.

## Allowed Work
- dataset acquisition and verification
- dataset integrity audits (hash duplicate checks, corruption detection)
- canonical label taxonomy specification
- reproducible manifest generation under `data/manifests/`
- dataset loading and test verification
- Phase 1 reporting and documentation

## Not Allowed
- model training (teacher or student)
- knowledge distillation
- robustness evaluation
- quantization / pruning
- uncertainty methods / abstention tuning
- Android mobile implementation

## Exit Criteria
- PlantVillage, PlantDoc, and PlantSeg sources verified and acquired
- Canonical `<crop>___<condition>` taxonomy established across 38 classes
- PlantDoc leaf crop extraction pipeline and manifests generated (8,883 crops across 29 shared classes)
- PlantVillage train/val/test splits deterministically frozen with seed 42
- Dataset manifests, integrity checks, and test suite passing with zero regressions
- Phase 1 report documented in `reports/phase_01_datasets.md`

## Next Phase
Phase 2 — Teacher Baseline
