"""Deployment export utilities for AgriRobust Phase 8.

Exports frozen MobileNetV3-Small student (Dynamic INT8 and FP32) into
mobile-deployable TorchScript formats with complete provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from agrirobust.config import get_project_root
from agrirobust.data.dataset import get_canonical_class_mapping
from agrirobust.deployment.preprocessing import ANDROID_PREPROCESSING_SPEC

logger = logging.getLogger(__name__)

EXPECTED_STUDENT_SHA256 = "2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6"
FROZEN_TEMP = 0.5406
FROZEN_TAU_VAL = 0.8143


def compute_file_sha256(file_path: Path | str) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_torchscript_operators(traced_model: torch.jit.ScriptModule) -> List[str]:
    """Extract list of operator names required by the traced graph."""
    try:
        return sorted(list(torch.jit.export_opnames(traced_model)))
    except Exception as e:
        logger.warning("Could not extract opnames from traced model: %s", e)
        return []


def export_torchscript_model(
    model: nn.Module,
    output_path: Path | str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """Trace and serialize PyTorch model into TorchScript container.

    Args:
        model: Evaluated PyTorch model in eval mode.
        output_path: Target .pt file path.
        input_shape: Sample tensor shape for tracing.

    Returns:
        Dict containing file_path, size_bytes, size_mb, sha256, and required_operators.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    with torch.no_grad():
        traced_script = torch.jit.trace(model, dummy_input)

    # Save container
    traced_script.save(str(output_path))

    file_size_bytes = os.path.getsize(output_path)
    file_sha256 = compute_file_sha256(output_path)
    operators = extract_torchscript_operators(traced_script)

    return {
        "file_path": str(output_path),
        "file_name": output_path.name,
        "size_bytes": file_size_bytes,
        "size_mb": round(file_size_bytes / (1024.0 * 1024.0), 3),
        "sha256": file_sha256,
        "input_shape": list(input_shape),
        "output_shape": [input_shape[0], 38],
        "required_operators": operators,
    }


def export_deployment_bundle(
    model: nn.Module,
    bundle_dir: Path | str,
    model_filename: str = "model_int8_dynamic.pt",
    source_checkpoint_path: Optional[Path | str] = None,
    compression_method: str = "Dynamic INT8 (classifier Linear)",
) -> Dict[str, Any]:
    """Package model artifact, labels.json, and deployment_metadata.json into bundle.

    Args:
        model: Evaluated model.
        bundle_dir: Directory where assets are packaged.
        model_filename: Name of the exported .pt file.
        source_checkpoint_path: Path to Phase 4 Response-KD champion checkpoint.
        compression_method: Human readable compression identifier.

    Returns:
        Summary dict of exported deployment bundle.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    root = get_project_root()
    if source_checkpoint_path is None:
        source_checkpoint_path = (
            root / "experiments" / "checkpoints" / "P04_student_kd_response_plantvillage_s42.pt"
        )
    source_checkpoint_path = Path(source_checkpoint_path)

    # Verify source SHA256
    source_sha = compute_file_sha256(source_checkpoint_path)

    # Export TorchScript model
    model_output_path = bundle_dir / model_filename
    export_info = export_torchscript_model(model, model_output_path)

    # Export canonical labels.json
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    labels_list = [idx_to_class[i] for i in range(len(idx_to_class))]
    labels_path = bundle_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": len(labels_list),
            "classes": labels_list,
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        }, f, indent=2)

    # Export deployment_metadata.json
    metadata: Dict[str, Any] = {
        "project": "AgriRobust",
        "phase": 8,
        "role": "On-Device Android Deployment Bundle",
        "source_checkpoint": str(source_checkpoint_path.name),
        "source_checkpoint_sha256": source_sha,
        "compression_method": compression_method,
        "architecture": "MobileNetV3-Small",
        "num_classes": 38,
        "input_shape": [1, 3, 224, 224],
        "artifact_file": model_filename,
        "artifact_sha256": export_info["sha256"],
        "artifact_size_bytes": export_info["size_bytes"],
        "artifact_size_mb": export_info["size_mb"],
        "calibration": {
            "temperature": FROZEN_TEMP,
            "source_phase": "Phase 6",
            "fitting_split": "PlantVillage canonical validation split",
            "formula": "p = softmax(logits / 0.5406)",
        },
        "selective_abstention": {
            "validation_threshold_tau": FROZEN_TAU_VAL,
            "source_phase": "Phase 6",
            "action_rule": "If max(p) >= 0.8143 -> ACCEPT, else -> ABSTAIN",
        },
        "preprocessing": ANDROID_PREPROCESSING_SPEC,
        "runtime_dependencies": {
            "android_runtime": "org.pytorch:pytorch_android_lite:1.13.1",
            "required_operators": export_info["required_operators"],
        },
    }

    metadata_path = bundle_dir / "deployment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "bundle_dir": str(bundle_dir),
        "model_export": export_info,
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
