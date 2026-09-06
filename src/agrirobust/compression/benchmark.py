"""Deployment benchmarking utilities for AgriRobust compression.

Measures CPU latency, throughput, model footprint, and compression ratios.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def benchmark_cpu_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    warmup_runs: int = 30,
    timed_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure single-instance CPU latency and throughput.

    Args:
        model: Evaluated PyTorch model.
        input_shape: Tensor shape (default: [1, 3, 224, 224]).
        warmup_runs: Number of un-timed warm-up forward passes.
        timed_runs: Number of timed forward passes.
        device: Device to benchmark on (must be 'cpu' for Phase 7).

    Returns:
        Dict containing mean, median, P95, std latency in milliseconds, and FPS throughput.
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_shape, device=device)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Timed phase
    latencies_ms = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    std_lat = float(np.std(latencies_arr))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 4),
        "median_latency_ms": round(median_lat, 4),
        "p95_latency_ms": round(p95_lat, 4),
        "std_latency_ms": round(std_lat, 4),
        "throughput_fps": round(fps, 2),
        "timed_runs": timed_runs,
        "warmup_runs": warmup_runs,
    }


def measure_footprint(
    file_path: Path | str,
    baseline_bytes: Optional[int] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Measure serialized file size, parameter counts, and compression ratio.

    Args:
        file_path: Path to serialized artifact.
        baseline_bytes: File size of FP32 reference for compression ratio.
        model: Optional model instance to count total parameters.

    Returns:
        Dictionary with size_bytes, size_mb, compression_ratio, parameter_count.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024.0 * 1024.0)

    compression_ratio = 1.0
    if baseline_bytes is not None and size_bytes > 0:
        compression_ratio = float(baseline_bytes / size_bytes)

    total_params = None
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())

    return {
        "file_path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "compression_ratio": round(compression_ratio, 3),
        "total_parameters": total_params,
    }
