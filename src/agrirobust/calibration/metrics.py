"""Uncertainty calibration metrics for AgriRobust.

Implements:
- Expected Calibration Error (ECE) with fixed-bin strategy (M=15 default).
- Negative Log-Likelihood (NLL).
- Multi-class Brier Score.
- Reliability diagram statistics (confidence vs accuracy per bin).
- Confidence distributions partitioned by correct vs incorrect predictions.
- Failure/Error Detection metrics: AUROC and AUPR using confidence to distinguish errors.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, Any]:
    """Compute comprehensive calibration metrics from predicted probabilities and true labels.

    Args:
        probabilities: Array of shape (N, K) containing predicted class probabilities.
        targets: Array of shape (N,) containing true integer class indices in [0, K-1].
        num_bins: Number of equally spaced bins across [0, 1] for ECE calculation.

    Returns:
        Dictionary containing:
        - expected_calibration_error: ECE float in [0, 1]
        - negative_log_likelihood: Mean cross-entropy loss
        - brier_score: Multi-class Brier score
        - accuracy: Overall top-1 classification accuracy
        - mean_confidence: Mean top-1 confidence
        - reliability_diagram: Bin-level statistics (acc, conf, count)
        - error_detection: AUROC and AUPR using confidence to detect incorrect predictions
        - confidence_stats: Summary statistics for correct vs incorrect predictions
    """
    if len(probabilities) != len(targets):
        raise ValueError(f"Lengths mismatch: {len(probabilities)} vs {len(targets)}")

    n_samples, n_classes = probabilities.shape
    if n_samples == 0:
        raise ValueError("Cannot compute calibration metrics on empty arrays.")

    # Top-1 predictions and confidences
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correct_mask = (predictions == targets)
    accuracy = float(np.mean(correct_mask))

    # 1. Expected Calibration Error (ECE) with fixed bins
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])  # 0 to num_bins - 1

    ece = 0.0
    reliability_bins: List[Dict[str, Any]] = []

    for b in range(num_bins):
        in_bin = (bin_indices == b)
        bin_count = int(np.sum(in_bin))
        bin_lower = float(bin_edges[b])
        bin_upper = float(bin_edges[b + 1])

        if bin_count > 0:
            bin_acc = float(np.mean(correct_mask[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        reliability_bins.append({
            "bin": b,
            "range": [round(bin_lower, 3), round(bin_upper, 3)],
            "count": bin_count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
        })

    # 2. Negative Log-Likelihood (NLL)
    # Clip probabilities to prevent log(0)
    eps = 1e-15
    probs_clipped = np.clip(probabilities, eps, 1.0 - eps)
    # Pick the probability assigned to the true class
    true_class_probs = probs_clipped[np.arange(n_samples), targets]
    nll = float(-np.mean(np.log(true_class_probs)))

    # 3. Multi-class Brier Score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    one_hot_targets = np.zeros_like(probabilities)
    one_hot_targets[np.arange(n_samples), targets] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot_targets) ** 2, axis=1)))

    # 4. Error Detection via Confidence (AUROC & AUPR)
    # Positive class for failure detection: 1 = Incorrect prediction, 0 = Correct
    error_labels = (~correct_mask).astype(int)
    uncertainty_scores = 1.0 - confidences  # higher uncertainty -> predicted error

    num_errors = int(np.sum(error_labels))
    num_correct = n_samples - num_errors

    if num_errors > 0 and num_correct > 0:
        auroc_error = float(roc_auc_score(error_labels, uncertainty_scores))
        aupr_error = float(average_precision_score(error_labels, uncertainty_scores))
        # AUROC for predicting success via confidence
        auroc_success = float(roc_auc_score(correct_mask.astype(int), confidences))
    else:
        # Edge case: all correct or all wrong
        auroc_error = float("nan")
        aupr_error = float("nan")
        auroc_success = float("nan")

    # 5. Confidence Distribution Summary for Correct vs Incorrect
    correct_confs = confidences[correct_mask]
    incorrect_confs = confidences[~correct_mask]

    def _stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
        if len(arr) == 0:
            return {"count": 0, "mean": None, "median": None, "std": None, "q25": None, "q75": None}
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
        }

    return {
        "expected_calibration_error": round(float(ece), 4),
        "negative_log_likelihood": round(nll, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(accuracy, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4),
        "num_samples": n_samples,
        "num_classes": n_classes,
        "num_bins": num_bins,
        "reliability_diagram": reliability_bins,
        "error_detection": {
            "auroc_error": round(auroc_error, 4) if not np.isnan(auroc_error) else None,
            "aupr_error": round(aupr_error, 4) if not np.isnan(aupr_error) else None,
            "auroc_success": round(auroc_success, 4) if not np.isnan(auroc_success) else None,
            "num_errors": num_errors,
            "num_correct": num_correct,
        },
        "confidence_distributions": {
            "correct": _stats(correct_confs),
            "incorrect": _stats(incorrect_confs),
        },
    }
