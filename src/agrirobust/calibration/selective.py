"""Selective prediction and abstention analysis for AgriRobust.

Implements:
- Confidence-based decision rule: predict if max(p) >= tau, else abstain.
- Canonical AURC (Area Under the Risk-Coverage Curve) computed via empirical sample sort.
- Threshold grid evaluation (coverage, selective risk, selective accuracy, selective Macro F1).
- Operating-point extraction (e.g. 95%, 90%, 80% coverage and validation-selected tau_val).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


def compute_canonical_aurc(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the canonical Area Under the Risk-Coverage Curve (AURC) via empirical sample sort.

    Samples are sorted in descending order of confidence. As each sample is incorporated,
    coverage increases from 1/N to 1.0, and selective risk (error rate of selected samples)
    is tracked dynamically.

    Args:
        confidences: Array of shape (N,) containing prediction confidence scores in [0, 1].
        predictions: Array of shape (N,) containing predicted class indices.
        targets: Array of shape (N,) containing true class indices.

    Returns:
        Tuple of (aurc_score, cumulative_coverages, cumulative_risks)
    """
    n = len(confidences)
    if n == 0:
        return 0.0, np.array([]), np.array([])

    # Sort descending by confidence
    sort_idx = np.argsort(-confidences)
    sorted_preds = predictions[sort_idx]
    sorted_targets = targets[sort_idx]

    # Binary error indicator: 1 if incorrect, 0 if correct
    errors = (sorted_preds != sorted_targets).astype(np.float64)

    # Cumulative errors and cumulative coverage
    cum_errors = np.cumsum(errors)
    counts = np.arange(1, n + 1, dtype=np.float64)
    cum_coverage = counts / n
    cum_risk = cum_errors / counts

    # Trapezoidal integration over coverage [1/N, 1.0]
    # Anchor at coverage = 0 with risk = cum_risk[0]
    cov_points = np.concatenate(([0.0], cum_coverage))
    risk_points = np.concatenate(([cum_risk[0]], cum_risk))

    aurc = float(np.trapezoid(risk_points, cov_points))
    return round(aurc, 4), cum_coverage, cum_risk


def evaluate_selective_grid(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Evaluate coverage, selective risk, selective accuracy, and selective Macro F1 across a threshold grid.

    Note on Selective Macro F1:
    When higher thresholds remove most or all examples from certain minority classes,
    unweighted class averaging can exhibit instability or zero-division.
    Selective accuracy and risk are the primary optimization metrics.

    Args:
        confidences: Array of shape (N,) containing prediction confidence scores in [0, 1].
        predictions: Array of shape (N,) containing predicted class indices.
        targets: Array of shape (N,) containing true class indices.
        thresholds: Array of thresholds to evaluate (defaults to 101 points from 0.0 to 1.0).

    Returns:
        List of dicts per threshold containing coverage, risk, accuracy, and selective macro f1.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    n_samples = len(confidences)
    grid_results = []

    for tau in thresholds:
        tau_val = float(tau)
        selected_mask = (confidences >= tau_val)
        selected_count = int(np.sum(selected_mask))
        coverage = selected_count / n_samples

        if selected_count > 0:
            sel_preds = predictions[selected_mask]
            sel_targets = targets[selected_mask]

            correct_count = int(np.sum(sel_preds == sel_targets))
            sel_acc = correct_count / selected_count
            sel_risk = 1.0 - sel_acc

            # Selective Macro F1: evaluate over classes present in retained targets
            # Zero division handled safely
            try:
                sel_f1 = float(f1_score(sel_targets, sel_preds, average="macro", zero_division=0))
            except Exception:
                sel_f1 = None
        else:
            sel_acc = None
            sel_risk = None
            sel_f1 = None

        grid_results.append({
            "threshold": round(tau_val, 4),
            "retained_count": selected_count,
            "coverage": round(float(coverage), 4),
            "selective_accuracy": round(float(sel_acc), 4) if sel_acc is not None else None,
            "selective_risk": round(float(sel_risk), 4) if sel_risk is not None else None,
            "selective_macro_f1": round(float(sel_f1), 4) if sel_f1 is not None else None,
        })

    return grid_results


def find_operating_point_by_coverage(
    grid_results: List[Dict[str, Any]],
    target_coverage: float,
) -> Dict[str, Any]:
    """Find the grid point closest to the specified target coverage."""
    valid_points = [p for p in grid_results if p["selective_risk"] is not None]
    if not valid_points:
        return {}
    best = min(valid_points, key=lambda p: abs(p["coverage"] - target_coverage))
    return dict(best)


def evaluate_at_frozen_threshold(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    tau: float,
) -> Dict[str, Any]:
    """Evaluate selective prediction at an exact frozen confidence threshold."""
    n_samples = len(confidences)
    selected_mask = (confidences >= tau)
    selected_count = int(np.sum(selected_mask))
    coverage = selected_count / n_samples if n_samples > 0 else 0.0

    if selected_count > 0:
        sel_preds = predictions[selected_mask]
        sel_targets = targets[selected_mask]
        correct = int(np.sum(sel_preds == sel_targets))
        acc = correct / selected_count
        risk = 1.0 - acc
        sel_f1 = float(f1_score(sel_targets, sel_preds, average="macro", zero_division=0))
    else:
        acc = None
        risk = None
        sel_f1 = None

    return {
        "threshold": round(tau, 4),
        "retained_count": selected_count,
        "total_count": n_samples,
        "coverage": round(coverage, 4),
        "selective_accuracy": round(acc, 4) if acc is not None else None,
        "selective_risk": round(risk, 4) if risk is not None else None,
        "selective_macro_f1": round(sel_f1, 4) if sel_f1 is not None else None,
    }


def select_validation_threshold(
    val_confidences: np.ndarray,
    val_predictions: np.ndarray,
    val_targets: np.ndarray,
    max_acceptable_risk: float = 0.02,
) -> float:
    """Select a single threshold tau on validation data achieving target risk <= max_acceptable_risk.

    The returned threshold is frozen and transferred unchanged to all test evaluation domains.
    """
    grid = evaluate_selective_grid(val_confidences, val_predictions, val_targets)
    # Search for lowest threshold achieving risk <= max_acceptable_risk to maximize coverage
    eligible = [p for p in grid if p["selective_risk"] is not None and p["selective_risk"] <= max_acceptable_risk]
    if eligible:
        # Maximize coverage among eligible
        best = max(eligible, key=lambda p: p["coverage"])
        return float(best["threshold"])
    # Default to 0.90 if no point meets the strict target
    return 0.90
