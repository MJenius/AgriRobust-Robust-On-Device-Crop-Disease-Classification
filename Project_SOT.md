# AgriRobust — Single Source of Truth

## 1. Project Identity

Project name: AgriRobust

Working title:
Robust On-Device Crop Disease Diagnosis

Core objective:

Develop a compact agricultural computer-vision model that approaches the performance of a high-capacity teacher model while being sufficiently small and efficient for fully local smartphone inference.

The project is fundamentally a model-efficiency and robustness study, not a generic crop-disease application.

---

## 2. Central Research Question

Can a high-performing agricultural vision model be compressed into a substantially smaller model that:

1. retains most of the teacher's real-world diagnostic performance,
2. remains robust under distribution shift,
3. produces calibrated confidence / uncertainty estimates,
4. supports selective abstention when evidence is insufficient, and
5. runs fully offline on a smartphone?

All implementation decisions must serve this question.

---

## 3. Primary Research Hypothesis

A compact vision architecture trained with knowledge distillation, deployment-aware compression, and robustness/calibration techniques can retain most of the teacher model's field-domain performance while reducing model size, memory usage, and inference latency sufficiently for smartphone deployment.

This is a hypothesis, not an expected result.

Do not fabricate or assume successful compression.

---

## 4. Project Scope

### In scope

- agricultural image classification
- optional lesion segmentation as an auxiliary task
- high-capacity teacher model
- compact student model
- knowledge distillation
- quantization
- optional structured pruning
- distribution-shift evaluation
- image corruption robustness
- calibration
- uncertainty estimation
- selective prediction / abstention
- Android on-device inference
- reproducible benchmarking

### Out of scope unless explicitly approved

- crop treatment recommendation
- fertilizer recommendation
- IoT hardware
- weather APIs
- cloud inference
- generic RAG
- LLM-based agricultural chatbot
- multi-agent systems
- autonomous farming
- medical/biological claims beyond the image classification task
- claims about real-world agronomic treatment efficacy

Do not add scope merely to make the application appear more sophisticated.

---

## 5. Core System

Conceptual pipeline:

Camera image
    ->
Image preprocessing
    ->
Compact vision model
    ->
Disease prediction
    ->
Confidence / uncertainty estimation
    ->
Prediction OR abstention

All final inference must be possible without network access.

---

## 6. Model Roles

### Teacher

Purpose:
High-capacity reference model.

Requirements:
- strong transfer performance
- reproducible training
- evaluated on all primary benchmarks
- used as the performance reference for student compression

The teacher is NOT required to run on the phone.

### Student

Purpose:
Deployment-oriented model.

Requirements:
- compact parameter count
- low memory usage
- low inference latency
- deployable on Android
- evaluated against the same test sets as the teacher

A MobileNet-family model is the initial preferred student family.

The specific architecture must be recorded in experiment configuration.

---

## 7. Dataset Policy

Primary datasets:

### PlantVillage

Purpose:
Controlled-domain training / baseline dataset.

Do not treat PlantVillage-only performance as evidence of real-world robustness.

### PlantDoc

Purpose:
In-the-wild cross-dataset evaluation.

### PlantSeg

Purpose:
In-the-wild disease localization and robustness evaluation.

### AgroBench

Purpose:
Optional broader agricultural evaluation / multimodal benchmark compatibility.

### Field-collected dataset

Purpose:
Real-world smartphone evaluation.

This dataset must be clearly separated from training data and documented.

---

## 8. Dataset Integrity Rules

Every dataset must have:

- source URL
- license/access information
- download procedure
- dataset version/date where available
- class mapping
- preprocessing specification
- split definition
- duplicate policy
- leakage checks

Never silently modify a benchmark.

If images are removed, transformed, relabeled, or excluded, record the reason.

---

## 9. Evaluation Principle

The project must distinguish:

### Clean-domain performance

Performance on data similar to the training distribution.

### Cross-domain performance

Performance on a different dataset/domain.

### Corruption robustness

Performance under controlled image degradation.

### Field performance

Performance on newly collected smartphone imagery.

Do not combine these into one aggregate score.

---

## 10. Primary Metrics

Classification:

- Macro F1
- Balanced accuracy
- Per-class F1
- Accuracy where appropriate

Calibration:

- Expected Calibration Error (ECE)
- Reliability diagram
- Brier score where appropriate

Selective prediction:

- coverage
- accuracy at coverage
- risk-coverage curve

Efficiency:

- parameter count
- model file size
- peak RAM
- inference latency
- throughput
- preprocessing latency where relevant

Deployment:

- end-to-end phone latency
- on-device memory
- offline operation

Segmentation, if included:

- Dice
- IoU

---

## 11. Primary Comparisons

At minimum, the final study must contain:

A. Teacher

B. Student trained normally

C. Student + knowledge distillation

D. Student + quantization

E. Final compressed / calibrated student

Additional ablations are encouraged only when they answer a clear research question.

---

## 12. Experimental Discipline

Every experiment must specify:

- experiment ID
- code/config version
- dataset version
- random seed(s)
- model architecture
- training hyperparameters
- augmentation
- checkpoint
- evaluation datasets
- evaluation metrics

Never compare models trained under materially different conditions without documenting the difference.

Do not report the best seed while hiding other runs.

---

## 13. Robustness Evaluation

The robustness suite should include, where practical:

- brightness variation
- contrast variation
- blur
- sensor/image noise
- JPEG compression
- occlusion
- background clutter
- resolution degradation

Natural domain shifts should include:

- PlantVillage -> PlantDoc
- PlantVillage -> PlantSeg where task-compatible
- public datasets -> field-collected imagery

For every shift report:

clean performance
shifted performance
absolute degradation
relative degradation

---

## 14. Uncertainty / Abstention

The final system must not be forced to classify every image.

Permitted outputs:

1. disease prediction with confidence
2. uncertain / abstain

The abstention mechanism must be evaluated quantitatively.

Do not claim that confidence is reliable without calibration evaluation.

---

## 15. Compression Pipeline

Initial planned progression:

FP32 baseline
    ->
FP16 evaluation
    ->
INT8 quantization
    ->
optional structured pruning

Each compression step must be evaluated independently.

Do not assume that a compression technique improves latency on every device.

Measure on the target hardware.

---

## 16. Smartphone Deployment

Target:

Android smartphone.

Preferred deployment path:

PyTorch / ExecuTorch or another justified on-device runtime.

Inference must work without:

- network access
- server APIs
- cloud GPU
- remote model calls

The app should capture an image and perform inference locally.

---

## 17. Phone Benchmarking

For every device benchmark record:

- manufacturer/model
- chipset where available
- Android version
- runtime version
- model version
- input resolution
- preprocessing latency
- inference latency
- total latency
- peak memory if measurable

Do not compare latency numbers across devices without identifying the devices.

---

## 18. Success Criteria

These are target thresholds, not guaranteed results.

Target A:
Student uses <=10% of teacher parameters.

Target B:
Student model size <=15 MB.

Target C:
Student retains >=95% of teacher field-domain macro-F1.

Target D:
Final model runs fully offline on Android.

Target E:
End-to-end phone latency is below 100 ms where hardware permits.

Target F:
Calibration + abstention measurably reduces high-confidence errors.

The actual final result may miss these targets.

If a target is missed, report the result honestly and analyze why.

---

## 19. Central Research Figure

The final project should produce an accuracy-efficiency Pareto analysis.

Primary axes should include some combination of:

- field-domain Macro F1
- model size
- latency
- RAM

The final narrative should emphasize the tradeoff rather than a single accuracy number.

---

## 20. Definition of "Success"

The project is successful if it produces a defensible answer to:

"How much agricultural vision-model performance can be retained while reducing the model to true smartphone scale, and how does that compression affect robustness and uncertainty under real-world distribution shift?"

A strong implementation does NOT require beating every existing model.

It requires a rigorous, reproducible and honest experimental answer.

---

## 21. Phase Gate Rule

Do not begin the next major phase until the current phase has:

1. a working implementation,
2. reproducible output,
3. a verification report,
4. documented limitations.

Each phase is independently reviewable.

---

## 22. Phase Definitions

Phase 0:
Project contract and environment.

Phase 1:
Dataset foundation.

Phase 2:
Teacher baseline.

Phase 3:
Compact student baseline.

Phase 4:
Knowledge distillation.

Phase 5:
Robustness evaluation.

Phase 6:
Uncertainty and abstention.

Phase 7:
Compression and deployment optimization.

Phase 8:
Android on-device inference.

Phase 9:
Field evaluation.

Phase 10:
Final scientific analysis.

---

## 23. Engineering Rules

Prefer simple implementations over unnecessary abstraction.

Do not add dependencies without documenting why they are required.

Do not silently replace libraries or architectures.

Do not change datasets without updating this document.

Do not change evaluation metrics without updating this document.

Do not introduce new project objectives without updating this document.

All changes that affect experiment validity must be documented.

---

## 24. Anti-Hallucination Rules for Coding Agents

Before implementing anything:

1. Read PROJECT_SOT.md.
2. Identify the current phase.
3. Identify the phase exit criteria.
4. Inspect the existing repository before creating new code.
5. Reuse existing abstractions where appropriate.
6. Do not invent dataset files, classes, labels, APIs, metrics, or benchmark results.
7. If a required fact is unknown, explicitly mark it as unknown.
8. Never fabricate results.
9. Never claim an experiment was run unless its output exists.
10. Never claim a model is state-of-the-art without a documented source.
11. Never infer benchmark numbers from memory.
12. Separate measured values from targets and hypotheses.

---

## 25. Source-of-Truth Priority

When information conflicts, use this order:

1. Actual experiment outputs
2. Repository code/configuration
3. This PROJECT_SOT.md
4. Dataset documentation
5. Cited external papers/documentation
6. General model knowledge

Never override measured experiment results with expected values.

---

## 26. Reporting Language

Use:

"measured" for observed experimental values.

"target" for desired thresholds.

"hypothesis" for proposed explanations.

"expected" only when explicitly justified.

"baseline" for comparison systems.

Never turn a target into a result.

Never turn a hypothesis into a conclusion.

Never turn a benchmark limitation into evidence of model capability.

---

## 27. Current Status

Current phase:
Phase 0

Completed:
Project definition.

Next objective:
Freeze environment, repository structure, datasets, metrics, and experiment configuration.

Do not implement advanced model features before Phase 0 is complete.