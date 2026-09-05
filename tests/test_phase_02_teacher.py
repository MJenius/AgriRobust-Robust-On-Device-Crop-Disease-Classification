"""Test suite for AgriRobust Phase 2 Teacher Baseline verification."""

import json
from pathlib import Path

import torch

from agrirobust.config import get_project_root
from agrirobust.data.dataset import get_canonical_class_mapping
from agrirobust.evaluation.plantseg_eval import assess_plantseg_compatibility
from agrirobust.models.teacher import build_teacher_model


def test_teacher_architecture():
    model = build_teacher_model(num_classes=38, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, feats = model(x, return_features=True)

    assert logits.shape == (2, 38)
    assert feats.shape == (2, 768)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 27849350


def test_frozen_teacher_checkpoints_exist():
    root = get_project_root()
    ckpts_dir = root / "experiments" / "checkpoints"

    for seed in [42, 1337, 2026]:
        ckpt_path = ckpts_dir / f"P02_teacher_convnext_tiny_plantvillage_s{seed}.pt"
        assert ckpt_path.is_file(), f"Missing checkpoint for seed {seed}"

        data = torch.load(ckpt_path, map_location="cpu")
        assert data["architecture"] == "convnext_tiny"
        assert data["num_classes"] == 38
        assert data["seed"] == seed
        assert "state_dict" in data
        assert "classifier.weight" in data["state_dict"]


def test_metrics_json_integrity():
    root = get_project_root()
    metrics_path = root / "experiments" / "runs" / "P02_teacher_convnext_tiny" / "metrics.json"
    assert metrics_path.is_file()

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "COMPLETED"
    assert data["architecture"] == "convnext_tiny"

    # Clean domain test should achieve >95% Macro F1
    clean_stats = data["summary_statistics"]["clean_domain_plantvillage"]
    assert clean_stats["macro_f1_mean"] > 0.95
    assert clean_stats["accuracy_mean"] > 0.95

    # Cross domain test should document distribution shift degradation
    cross_stats = data["summary_statistics"]["cross_domain_plantdoc"]
    assert cross_stats["macro_f1_mean"] < 0.50  # Verifying natural domain shift is captured

    # Efficiency benchmark presence
    eff = data["efficiency"]
    assert eff["total_parameters"] == 27849350
    assert eff["checkpoint_file_size_mb"] > 100.0
    assert eff["latency_mean_ms"] > 0.0


def test_plantseg_compatibility_assessment():
    res = assess_plantseg_compatibility()
    assert res["status"] == "ASSESSED"
    assert res["compatible_with_classification_head"] is False
    assert res["task_compatibility"] == "AUXILIARY_SEGMENTATION_ONLY"
