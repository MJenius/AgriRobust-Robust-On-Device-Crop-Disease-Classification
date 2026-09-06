"""Mobile runtime benchmarking utilities for AgriRobust Phase 8.

Measures host-side mobile runtime execution (model-only latency vs end-to-end latency)
and checks Android device / emulator connectivity via adb.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def probe_android_environment() -> Dict[str, Any]:
    """Probe Android SDK, ADB, connected physical devices, and running AVD emulators."""
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidate_sdk = Path(local_app_data) / "Android" / "Sdk"
    sdk_found = candidate_sdk.exists()

    adb_path = candidate_sdk / "platform-tools" / "adb.exe" if sdk_found else None
    if adb_path and not adb_path.exists():
        adb_path = shutil.which("adb")
    else:
        adb_path = str(adb_path) if adb_path and adb_path.exists() else shutil.which("adb")

    connected_devices: List[str] = []
    avds_available: List[str] = []

    if adb_path:
        try:
            res = subprocess.run([str(adb_path), "devices"], capture_output=True, text=True, timeout=5)
            lines = res.stdout.strip().split("\n")
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] == "device":
                    connected_devices.append(parts[0])
        except Exception as e:
            logger.warning("Error querying adb devices: %s", e)

    emulator_path = candidate_sdk / "emulator" / "emulator.exe" if sdk_found else None
    if emulator_path and emulator_path.exists():
        try:
            res = subprocess.run([str(emulator_path), "-list-avds"], capture_output=True, text=True, timeout=5)
            avds_available = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        except Exception as e:
            logger.warning("Error querying emulator AVDs: %s", e)

    return {
        "android_sdk_found": sdk_found,
        "android_sdk_path": str(candidate_sdk) if sdk_found else None,
        "adb_found": bool(adb_path),
        "adb_path": str(adb_path) if adb_path else None,
        "connected_devices": connected_devices,
        "num_connected_devices": len(connected_devices),
        "available_avds": avds_available,
        "execution_target": "HOST_CONTAINER" if len(connected_devices) == 0 else f"ANDROID_DEVICE ({connected_devices[0]})",
        "status_note": (
            "Host-side mobile-runtime parity benchmark (No physical Android device or active AVD connected)"
            if len(connected_devices) == 0
            else f"Connected to Android hardware target {connected_devices[0]}"
        ),
    }


def benchmark_mobile_runtime(
    torchscript_path: Path | str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    warmup_runs: int = 30,
    timed_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Measure model-only latency and end-to-end inference latency on host mobile container.

    Args:
        torchscript_path: Path to exported TorchScript model (.pt).
        input_shape: Input tensor dimension.
        warmup_runs: Number of unmeasured forward passes.
        timed_runs: Number of timed executions.
        device: 'cpu'.

    Returns:
        Dict with model-only latency stats, end-to-end latency stats, and FPS.
    """
    ts_path = Path(torchscript_path)
    model = torch.jit.load(str(ts_path), map_location=device)
    model.eval()

    dummy_tensor = torch.randn(*input_shape, device=device)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

    # 1. Model-Only Inference Benchmarking
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_tensor)

    model_latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(dummy_tensor)
            t1 = time.perf_counter()
            model_latencies.append((t1 - t0) * 1000.0)

    m_arr = np.array(model_latencies)
    mean_model = float(np.mean(m_arr))
    median_model = float(np.median(m_arr))
    p95_model = float(np.percentile(m_arr, 95))
    fps_model = float(1000.0 / mean_model) if mean_model > 0 else 0.0

    # 2. End-to-End Latency Benchmarking (Preprocessing + Model Inference + Temperature Scaling + Threshold Decision)
    def run_end_to_end(pil_img: Image.Image) -> Dict[str, Any]:
        # Preprocessing: resize + center crop + ToTensor + Normalize
        img_resized = F.resize(pil_img, 224, interpolation=F.InterpolationMode.BILINEAR)
        img_cropped = F.center_crop(img_resized, (224, 224))
        t_in = F.to_tensor(img_cropped)
        t_norm = F.normalize(t_in, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)

        # Inference
        logits = model(t_norm)

        # Temperature Scaling & Softmax
        scaled_logits = logits / 0.5406
        probs = torch.softmax(scaled_logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        accepted = conf.item() >= 0.8143
        return {"class_idx": pred.item(), "conf": conf.item(), "accepted": accepted}

    # Warmup E2E
    for _ in range(10):
        _ = run_end_to_end(dummy_image)

    e2e_latencies = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = run_end_to_end(dummy_image)
        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000.0)

    e_arr = np.array(e2e_latencies)
    mean_e2e = float(np.mean(e_arr))
    median_e2e = float(np.median(e_arr))
    p95_e2e = float(np.percentile(e_arr, 95))
    fps_e2e = float(1000.0 / mean_e2e) if mean_e2e > 0 else 0.0

    env_info = probe_android_environment()

    return {
        "benchmark_type": "HOST_CONTAINER_PARITY (Mobile Runtime Container on Host CPU)",
        "model_file": str(ts_path.name),
        "model_only": {
            "mean_latency_ms": round(mean_model, 3),
            "median_latency_ms": round(median_model, 3),
            "p95_latency_ms": round(p95_model, 3),
            "throughput_fps": round(fps_model, 2),
        },
        "end_to_end": {
            "mean_latency_ms": round(mean_e2e, 3),
            "median_latency_ms": round(median_e2e, 3),
            "p95_latency_ms": round(p95_e2e, 3),
            "throughput_fps": round(fps_e2e, 2),
            "preprocessing_overhead_ms": round(mean_e2e - mean_model, 3),
        },
        "timed_runs": timed_runs,
        "warmup_runs": warmup_runs,
        "environment_probe": env_info,
    }
