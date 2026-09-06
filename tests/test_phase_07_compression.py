"""Tests for Phase 7 Deployment-Aware Compression.

Validates:
- FP32 reference checkpoint integrity and immutable state.
- Dynamic INT8 quantization execution, serialization, and output shapes.
- Static post-training quantization environment compatibility probe.
- Pruning sparsity calculation and permanent mask removal.
- Latency and footprint benchmark measurements.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from agrirobust.compression.benchmark import benchmark_cpu_latency, measure_footprint
from agrirobust.compression.pruning import apply_unstructured_pruning, calculate_sparsity
from agrirobust.compression.quantization import (
    apply_dynamic_quantization,
    load_quantized_model,
    probe_static_quantization,
    save_quantized_model,
)
from agrirobust.models.student import AgriStudentMobileNetV3


@pytest.fixture
def dummy_mobilenet():
    """Create lightweight MobileNetV3-Small test instance."""
    return AgriStudentMobileNetV3(num_classes=38, pretrained=False)


def test_fp32_checkpoint_hash_integrity():
    """Verify frozen Phase 4 Response-KD champion checkpoint exists and hash is verified."""
    ckpt_path = Path("experiments/checkpoints/P04_student_kd_response_plantvillage_s42.pt")
    assert ckpt_path.exists(), "Phase 4 Response-KD champion checkpoint must exist."

    # Verify SHA-256 matches Phase 4 frozen hash
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    assert (
        actual_hash == "2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6"
    ), "Phase 4 Response-KD champion checkpoint has been modified or corrupted!"


def test_dynamic_quantization_forward_and_serialization(dummy_mobilenet, tmp_path):
    """Verify dynamic INT8 quantization on nn.Linear produces valid logits and serializes correctly."""
    quantized = apply_dynamic_quantization(dummy_mobilenet, qconfig_spec={nn.Linear})

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = quantized(dummy_input)
    assert logits.shape == (2, 38)
    assert not torch.isnan(logits).any()

    # Test serialization & reloading
    save_file = tmp_path / "model_dynamic_test.pt"
    save_quantized_model(quantized, save_file)
    assert save_file.exists()

    reloaded = load_quantized_model(save_file)
    logits_reloaded = reloaded(dummy_input)
    assert logits_reloaded.shape == (2, 38)
    assert torch.allclose(logits, logits_reloaded, atol=1e-4)


def test_probe_static_quantization_report(dummy_mobilenet):
    """Verify probe_static_quantization records PyTorch version and engine status without crashing."""
    report = probe_static_quantization(dummy_mobilenet)
    assert "pytorch_version" in report
    assert "supported_engines" in report
    assert "status" in report
    assert isinstance(report["supported"], bool)


def test_unstructured_pruning_and_sparsity(dummy_mobilenet):
    """Verify L1 pruning removes target fraction of weights and sparsity calculation reflects it."""
    pruned_model = apply_unstructured_pruning(
        dummy_mobilenet,
        amount=0.40,
        target_layer_types=(nn.Linear,),
        make_permanent=True,
    )

    sparsity_dict = calculate_sparsity(pruned_model)
    assert "global_sparsity" in sparsity_dict
    assert sparsity_dict["global_sparsity"] > 0.0

    # Verify classifier weights have approximately 40% zeros
    for name, param in pruned_model.named_parameters():
        if "classifier" in name and "weight" in name and param.dim() > 1:
            zeros_fraction = (param == 0).sum().item() / param.numel()
            assert abs(zeros_fraction - 0.40) < 0.02


def test_benchmark_cpu_latency(dummy_mobilenet):
    """Verify CPU latency benchmarker returns correct schema and positive values."""
    metrics = benchmark_cpu_latency(
        dummy_mobilenet,
        input_shape=(1, 3, 224, 224),
        warmup_runs=5,
        timed_runs=10,
        device="cpu",
    )
    assert "mean_latency_ms" in metrics
    assert "p95_latency_ms" in metrics
    assert "throughput_fps" in metrics
    assert metrics["mean_latency_ms"] > 0.0
    assert metrics["throughput_fps"] > 0.0


def test_measure_footprint(tmp_path, dummy_mobilenet):
    """Verify footprint measurement calculates file size and compression ratio correctly."""
    dummy_file = tmp_path / "dummy.pt"
    torch.save(dummy_mobilenet.state_dict(), dummy_file)

    os_size = dummy_file.stat().st_size
    footprint = measure_footprint(dummy_file, baseline_bytes=os_size)
    assert footprint["size_bytes"] == os_size
    assert footprint["compression_ratio"] == 1.0
