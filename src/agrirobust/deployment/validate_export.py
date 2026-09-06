"""Export validation and numerical parity verification for AgriRobust Phase 8.

Compares predictions, probabilities, and confidence between the Python PyTorch reference
and the exported mobile deployment artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

FROZEN_TEMP = 0.5406
FROZEN_TAU_VAL = 0.8143


def compute_calibrated_probabilities(logits: np.ndarray, temp: float = FROZEN_TEMP) -> np.ndarray:
    """Apply stable softmax with temperature scaling."""
    scaled = logits / temp
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def evaluate_export_parity(
    reference_model: nn.Module,
    deployed_artifact_path: Path | str,
    dataloader: DataLoader,
    device: str = "cpu",
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """Evaluate prediction and probability parity between Python reference and TorchScript deployment artifact.

    Args:
        reference_model: Source PyTorch model in eval mode.
        deployed_artifact_path: Path to exported .pt / TorchScript file.
        dataloader: Test or validation DataLoader.
        device: CPU device.
        max_samples: Maximum samples to evaluate for parity check.

    Returns:
        Dict with top-1 agreement, top-5 agreement, max prob diff, mean prob diff, abstention agreement.
    """
    deployed_path = Path(deployed_artifact_path)
    if not deployed_path.exists():
        raise FileNotFoundError(f"Deployment artifact not found: {deployed_path}")

    reference_model.eval()
    reference_model.to(device)

    # Load deployment TorchScript module
    ts_model = torch.jit.load(str(deployed_path), map_location=device)
    ts_model.eval()

    ref_logits_list = []
    dep_logits_list = []
    targets_list = []
    collected_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            targets = batch[1]

            out_ref = reference_model(images)
            out_dep = ts_model(images)

            ref_logits_list.append(out_ref.cpu().numpy())
            dep_logits_list.append(out_dep.cpu().numpy())
            targets_list.append(targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets))

            collected_samples += len(targets)
            if collected_samples >= max_samples:
                break

    ref_logits = np.concatenate(ref_logits_list, axis=0)[:max_samples]
    dep_logits = np.concatenate(dep_logits_list, axis=0)[:max_samples]
    targets = np.concatenate(targets_list, axis=0)[:max_samples]

    n_eval = len(targets)

    # Predictions
    ref_preds = np.argmax(ref_logits, axis=1)
    dep_preds = np.argmax(dep_logits, axis=1)
    top1_agreement = float(np.mean(ref_preds == dep_preds))

    # Top-5 agreement
    ref_top5 = np.argsort(ref_logits, axis=1)[:, -5:]
    dep_top5 = np.argsort(dep_logits, axis=1)[:, -5:]
    top5_matches = [len(set(ref_top5[i]).intersection(set(dep_top5[i]))) == 5 for i in range(n_eval)]
    top5_agreement = float(np.mean(top5_matches))

    # Calibrated probabilities (T = 0.5406)
    ref_probs = compute_calibrated_probabilities(ref_logits, temp=FROZEN_TEMP)
    dep_probs = compute_calibrated_probabilities(dep_logits, temp=FROZEN_TEMP)

    abs_prob_diff = np.abs(ref_probs - dep_probs)
    max_prob_diff = float(np.max(abs_prob_diff))
    mean_prob_diff = float(np.mean(abs_prob_diff))

    # Confidence and selective abstention agreement
    ref_confs = np.max(ref_probs, axis=1)
    dep_confs = np.max(dep_probs, axis=1)

    max_conf_diff = float(np.max(np.abs(ref_confs - dep_confs)))
    mean_conf_diff = float(np.mean(np.abs(ref_confs - dep_confs)))

    ref_accepted = (ref_confs >= FROZEN_TAU_VAL)
    dep_accepted = (dep_confs >= FROZEN_TAU_VAL)
    abstention_decision_agreement = float(np.mean(ref_accepted == dep_accepted))

    return {
        "evaluated_samples": n_eval,
        "top1_agreement_pct": round(top1_agreement * 100.0, 2),
        "top5_agreement_pct": round(top5_agreement * 100.0, 2),
        "max_probability_difference": round(max_prob_diff, 6),
        "mean_probability_difference": round(mean_prob_diff, 6),
        "max_confidence_difference": round(max_conf_diff, 6),
        "mean_confidence_difference": round(mean_conf_diff, 6),
        "abstention_agreement_pct": round(abstention_decision_agreement * 100.0, 2),
        "frozen_calibration_temperature": FROZEN_TEMP,
        "frozen_decision_threshold": FROZEN_TAU_VAL,
        "parity_verdict": "PERFECT_AGREEMENT" if top1_agreement >= 0.999 else "HIGH_AGREEMENT" if top1_agreement >= 0.98 else "DIVERGENT",
    }
