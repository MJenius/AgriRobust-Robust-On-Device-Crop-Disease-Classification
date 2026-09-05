# AgriRobust Experiments Directory

This directory stores outputs, checkpoints, metrics, and logs from experiments executed across project phases.

## Directory Structure

```text
experiments/
├── checkpoints/       # Model weights (.pt, quantized artifacts)
├── logs/              # Training and evaluation logs
└── runs/              # Per-run artifact directories
```

## Run Directory Convention

Every run creates a folder named according to the project experiment naming pattern:

`P<phase:02d>_<role>_<architecture>_<dataset>_<variant>_s<seed>`

Each run directory contains:
- `config.yaml`: Frozen snapshot of the exact experiment configuration
- `run_metadata.json`: System info, commit hash, timestamp, seed, hyperparameters
- `metrics.json`: Final evaluated metrics
- `training.log`: Text execution log
- `checkpoints/`: Best and final model weights

> **Phase 0 Status**: No experiments have been executed yet. Model training and evaluation are strictly prohibited during Phase 0.
