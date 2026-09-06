"""Tests for AgriRobust Phase 8 Android Deployment & Validation.

Verifies:
1. Frozen source checkpoint hash integrity and provenance.
2. Preprocessing contract parity between PyTorch and simulated Android pipeline.
3. Deployment artifact export (TorchScript container, labels.json, deployment_metadata.json).
4. Operator compatibility inspection for PyTorch Android runtime.
5. Numerical prediction parity, probability agreement, and calibration consistency.
6. Selective abstention parity between Python reference and deployment artifact.
7. Mobile runtime container benchmarking.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from agrirobust.config import get_project_root
from agrirobust.deployment.export import (
    compute_file_sha256,
    export_deployment_bundle,
    export_torchscript_model,
)
from agrirobust.deployment.mobile_benchmark import (
    benchmark_mobile_runtime,
    probe_android_environment,
)
from agrirobust.deployment.preprocessing import (
    ANDROID_PREPROCESSING_SPEC,
    preprocess_image_for_deployment,
    verify_preprocessing_parity,
)
from agrirobust.deployment.validate_export import evaluate_export_parity
from agrirobust.models.student import build_student_model

EXPECTED_SHA256 = "2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6"
FROZEN_TEMP = 0.5406
FROZEN_TAU_VAL = 0.8143


def test_frozen_source_checkpoint_sha256():
    """Verify Phase 4 Response-KD champion checkpoint exists and is strictly immutable."""
    root = get_project_root()
    ckpt_path = root / "experiments" / "checkpoints" / "P04_student_kd_response_plantvillage_s42.pt"
    assert ckpt_path.exists(), "Source checkpoint not found!"
    actual_hash = compute_file_sha256(ckpt_path)
    assert actual_hash == EXPECTED_SHA256, f"Source checkpoint hash mismatch! Expected {EXPECTED_SHA256}, got {actual_hash}"


def test_preprocessing_parity_with_android_pipeline():
    """Verify preprocessing parity between torchvision transforms and Android Bitmap normalization."""
    # Create test synthetic RGB image
    np.random.seed(42)
    sample_arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    sample_img = Image.fromarray(sample_arr)

    parity = verify_preprocessing_parity(sample_img, tolerance=1e-2)
    assert parity["parity_pass"] is True
    assert parity["max_absolute_difference"] < 1e-2
    assert parity["mean_absolute_difference"] < 1e-3


def test_torchscript_export_and_operator_extraction(tmp_path):
    """Verify TorchScript export traces MobileNetV3-Small into valid container with documented operators."""
    model = build_student_model(num_classes=38, pretrained=False)
    model.eval()

    export_path = tmp_path / "model_test.pt"
    export_info = export_torchscript_model(model, export_path)

    assert export_path.exists()
    assert export_info["size_mb"] > 0.0
    assert export_info["output_shape"] == [1, 38]
    assert len(export_info["required_operators"]) > 0
    # MobileNetV3 utilizes aten::_convolution and aten::linear
    assert any("convolution" in op or "conv" in op for op in export_info["required_operators"])
    assert any("linear" in op for op in export_info["required_operators"])


def test_deployment_bundle_packaging(tmp_path):
    """Verify export_deployment_bundle packages model, labels.json, and deployment_metadata.json."""
    model = build_student_model(num_classes=38, pretrained=False)
    model.eval()

    bundle_info = export_deployment_bundle(model, bundle_dir=tmp_path, model_filename="test_model.pt")

    assert (tmp_path / "test_model.pt").exists()
    assert (tmp_path / "labels.json").exists()
    assert (tmp_path / "deployment_metadata.json").exists()

    meta = bundle_info["metadata"]
    assert meta["num_classes"] == 38
    assert meta["calibration"]["temperature"] == FROZEN_TEMP
    assert meta["selective_abstention"]["validation_threshold_tau"] == FROZEN_TAU_VAL
    assert meta["source_checkpoint_sha256"] == EXPECTED_SHA256


def test_prediction_parity_and_calibration_transfer(tmp_path):
    """Verify numerical parity and calibration consistency between Python model and exported container."""
    model = build_student_model(num_classes=38, pretrained=False)
    model.eval()

    export_path = tmp_path / "model_parity.pt"
    export_torchscript_model(model, export_path)

    # Synthetic batch of 20 images
    dummy_x = torch.randn(20, 3, 224, 224)
    dummy_y = torch.randint(0, 38, (20,))
    dummy_loader = [(dummy_x, dummy_y)]

    parity_results = evaluate_export_parity(
        model,
        export_path,
        dataloader=dummy_loader,
        device="cpu",
        max_samples=20,
    )

    assert parity_results["top1_agreement_pct"] == 100.0
    assert parity_results["top5_agreement_pct"] == 100.0
    assert parity_results["abstention_agreement_pct"] == 100.0
    assert parity_results["max_probability_difference"] < 1e-4
    assert parity_results["parity_verdict"] == "PERFECT_AGREEMENT"


def test_mobile_runtime_benchmark_and_environment_probe(tmp_path):
    """Verify mobile runtime benchmarker records host container stats and environment probe."""
    model = build_student_model(num_classes=38, pretrained=False)
    model.eval()

    export_path = tmp_path / "model_bench.pt"
    export_torchscript_model(model, export_path)

    bench_res = benchmark_mobile_runtime(export_path, warmup_runs=5, timed_runs=10)

    assert "model_only" in bench_res
    assert "end_to_end" in bench_res
    assert "environment_probe" in bench_res
    assert bench_res["model_only"]["mean_latency_ms"] > 0.0
    assert bench_res["end_to_end"]["mean_latency_ms"] > bench_res["model_only"]["mean_latency_ms"]
    assert "HOST_CONTAINER_PARITY" in bench_res["benchmark_type"]
