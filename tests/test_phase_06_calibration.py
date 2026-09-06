"""Unit tests for AgriRobust Phase 6 Calibration and Selective Abstention."""

import numpy as np
import pytest
import torch

from agrirobust.calibration.metrics import compute_calibration_metrics
from agrirobust.calibration.selective import (
    compute_canonical_aurc,
    evaluate_at_frozen_threshold,
    evaluate_selective_grid,
    find_operating_point_by_coverage,
    select_validation_threshold,
)
from agrirobust.calibration.temperature import TemperatureScaler


def test_compute_calibration_metrics_perfect():
    """Test calibration metrics on a synthetic perfectly calibrated distribution."""
    # 10 samples, 100% confidence, 100% accurate
    probs = np.zeros((10, 3), dtype=np.float32)
    probs[:, 0] = 1.0
    targets = np.zeros(10, dtype=np.int64)

    metrics = compute_calibration_metrics(probs, targets, num_bins=10)
    assert metrics["expected_calibration_error"] == 0.0
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == 0.0
    assert metrics["error_detection"]["num_errors"] == 0


def test_compute_calibration_metrics_overconfident():
    """Test calibration metrics on an overconfident model (100% confidence, 50% accurate)."""
    probs = np.zeros((10, 2), dtype=np.float32)
    probs[:, 0] = 1.0  # always predicts class 0 with 1.0 confidence
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)  # 5 correct, 5 wrong

    metrics = compute_calibration_metrics(probs, targets, num_bins=10)
    # Accuracy is 0.5, confidence is 1.0, so ECE should be |0.5 - 1.0| = 0.5
    assert abs(metrics["expected_calibration_error"] - 0.5) < 1e-4
    assert metrics["accuracy"] == 0.5
    assert metrics["error_detection"]["num_errors"] == 5


def test_temperature_scaler_optimization_and_invariance():
    """Verify that temperature scaling minimizes NLL and preserves class rankings and accuracy."""
    np.random.seed(42)
    n_samples = 200
    n_classes = 5

    # Simulate overconfident logits
    logits = np.random.randn(n_samples, n_classes) * 5.0
    # Simulate ground truth with some alignment to largest logit
    targets = np.argmax(logits, axis=1)
    # Randomly corrupt 30% of targets to induce miscalibration
    flip_mask = np.random.rand(n_samples) < 0.3
    targets[flip_mask] = (targets[flip_mask] + 1) % n_classes

    initial_preds = np.argmax(logits, axis=1)
    initial_acc = np.mean(initial_preds == targets)

    scaler = TemperatureScaler()
    scaler.fit(logits, targets)

    assert scaler.is_fitted
    assert scaler.temperature > 0.0
    assert scaler.fit_loss_after < scaler.fit_loss_before  # NLL decreased

    # Predict scaled probabilities
    cal_probs = scaler.predict_probabilities(logits)
    cal_preds = np.argmax(cal_probs, axis=1)
    cal_acc = np.mean(cal_preds == targets)

    # Monotonic scaling MUST preserve top-1 predictions and top-1 accuracy bit-for-bit
    assert np.array_equal(initial_preds, cal_preds)
    assert initial_acc == cal_acc


def test_temperature_scaler_leakage_safeguard():
    """Verify temperature scaler can be fit on validation data and applied to unseen test logits."""
    val_logits = np.random.randn(50, 4)
    val_targets = np.random.randint(0, 4, size=50)

    test_logits = np.random.randn(30, 4)

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_targets)
    frozen_t = scaler.temperature

    # Apply to test set
    test_probs = scaler.predict_probabilities(test_logits)
    assert test_probs.shape == (30, 4)
    assert np.allclose(test_probs.sum(axis=1), 1.0, atol=1e-5)
    # Check that temperature remained unchanged
    assert scaler.temperature == frozen_t


def test_canonical_aurc_calculation():
    """Verify canonical AURC on a toy dataset."""
    # 4 samples: confidences [0.9, 0.8, 0.7, 0.6]
    # Predictions: [0, 1, 2, 3]
    # Targets: [0, 1, 9, 9] -> 2 correct, 2 errors
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    predictions = np.array([0, 1, 2, 3])
    targets = np.array([0, 1, 9, 9])

    aurc, cov, risk = compute_canonical_aurc(confidences, predictions, targets)
    assert 0.0 <= aurc <= 1.0
    assert len(cov) == 4
    assert len(risk) == 4
    # Highest confidence samples are correct -> initial risk is 0.0
    assert risk[0] == 0.0
    assert risk[1] == 0.0
    # Then errors come in
    assert risk[2] > 0.0
    assert risk[3] == 0.5


def test_selective_grid_and_operating_points():
    """Verify selective grid evaluation and validation threshold selection."""
    confidences = np.linspace(0.5, 0.99, 100)
    # Predictions: correct when confidence >= 0.8
    targets = np.zeros(100, dtype=int)
    predictions = np.zeros(100, dtype=int)
    predictions[confidences < 0.8] = 1  # errors at lower confidence

    grid = evaluate_selective_grid(confidences, predictions, targets)
    assert len(grid) == 101

    op_90 = find_operating_point_by_coverage(grid, 0.90)
    assert "coverage" in op_90
    assert abs(op_90["coverage"] - 0.90) < 0.05

    # Select threshold achieving <= 5% risk on validation
    tau_val = select_validation_threshold(confidences, predictions, targets, max_acceptable_risk=0.05)
    assert tau_val >= 0.78  # should abstain on the low-confidence errors near ~0.80 boundary

    eval_frozen = evaluate_at_frozen_threshold(confidences, predictions, targets, tau_val)
    assert eval_frozen["selective_risk"] <= 0.05
