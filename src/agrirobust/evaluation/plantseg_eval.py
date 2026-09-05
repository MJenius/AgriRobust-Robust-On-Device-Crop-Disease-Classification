"""Evaluation compatibility checker for auxiliary benchmarks such as PlantSeg."""

import json
from pathlib import Path
from typing import Any, Dict

from agrirobust.config import get_project_root


def assess_plantseg_compatibility() -> Dict[str, Any]:
    """Inspect PlantSeg manifest and verify task alignment.

    PlantSeg provides 11,458 semantic segmentation masks for disease localization.
    As established in SOT Section 4 and Phase 2 directives, forcing pixel segmentation masks
    into whole-image disease classification labels is structurally incompatible.
    PlantSeg is strictly preserved for auxiliary lesion segmentation / localization in Phase 5.
    """
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantseg_manifest.json"

    if not manifest_path.is_file():
        return {
            "status": "NOT_FOUND",
            "compatible_with_classification_head": False,
            "reason": "Manifest missing",
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "status": "ASSESSED",
        "dataset": "PlantSeg",
        "total_samples": data.get("total_samples", 0),
        "modality": "image_and_mask",
        "task_compatibility": "AUXILIARY_SEGMENTATION_ONLY",
        "compatible_with_classification_head": False,
        "rationale": "PlantSeg evaluates 115 pixel-level disease symptom categories with polygon masks. Forcing these masks into 38 whole-image classification classes would corrupt the taxonomy. Reserved for Phase 5 robustness/localization.",
    }
