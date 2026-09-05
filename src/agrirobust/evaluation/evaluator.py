"""Evaluation harness and metric computation engine for AgriRobust."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def compute_model_efficiency(
    model: nn.Module,
    checkpoint_path: Optional[Path] = None,
    image_size: int = 224,
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Compute parameter count, file size, and batch-1 inference latency."""
    model.eval()
    model.to(device)

    # 1. Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Checkpoint file size
    file_size_mb = 0.0
    if checkpoint_path is not None and checkpoint_path.is_file():
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    # 3. Batch-1 latency measurement
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark loop
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_file_size_mb": round(file_size_mb, 3),
        "latency_mean_ms": round(mean_latency, 3),
        "latency_p50_ms": round(p50_latency, 3),
        "latency_p95_ms": round(p95_latency, 3),
        "throughput_fps": round(fps, 2),
    }


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    idx_to_class: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Evaluate classifier predictions and compute primary metrics.

    Returns:
      - accuracy
      - macro_f1
      - balanced_accuracy
      - per_class_f1 (dict)
      - confusion_matrix (list of lists)
      - evaluated_samples
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Primary metrics
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Per-class F1
    unique_present_labels = sorted(list(set(y_true).union(set(y_pred))))
    per_class_f1_raw = f1_score(y_true, y_pred, average=None, labels=unique_present_labels, zero_division=0)
    per_class_f1_dict = {}
    for idx_val, f1_val in zip(unique_present_labels, per_class_f1_raw):
        cls_name = idx_to_class.get(idx_val, f"class_{idx_val}") if idx_to_class else str(idx_val)
        per_class_f1_dict[cls_name] = round(float(f1_val), 4)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_present_labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "total_evaluated_samples": len(y_true),
        "per_class_f1": per_class_f1_dict,
        "confusion_matrix": cm,
        "evaluated_classes_count": len(unique_present_labels),
    }
