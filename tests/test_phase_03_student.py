"""Phase 3 Compact Student Baseline test suite."""

import json
from pathlib import Path

import torch

from agrirobust.config import get_project_root
from agrirobust.models.student import build_student_model


def test_student_architecture_profile():
    model = build_student_model(num_classes=38, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits, feats = model(x, return_features=True)

    assert logits.shape == (2, 38)
    assert feats.shape == (2, 576)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 1556806

    # Target A validation: Student uses <= 10% of teacher parameters (27,849,350)
    teacher_params = 27849350
    ratio = total_params / teacher_params
    assert ratio <= 0.10, f"Ratio {ratio:.4f} violates Target A (<=10%)"
    assert round(ratio * 100, 2) == 5.59


def test_student_checkpoint_integrity():
    root = get_project_root()
    ckpt_path = root / "experiments" / "checkpoints" / "P03_student_mobilenetv3_small_plantvillage_s42.pt"
    assert ckpt_path.is_file(), f"Missing student deployment checkpoint: {ckpt_path}"

    # Target B validation: Deployment checkpoint <= 15 MB
    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    assert file_size_mb <= 15.0, f"File size {file_size_mb:.2f} MB exceeds 15 MB Target B"
    assert round(file_size_mb, 2) <= 6.50

    state_dict = torch.load(ckpt_path, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert "classifier.3.weight" in state_dict
    assert state_dict["classifier.3.weight"].shape == (38, 1024)


def test_metrics_json_and_gap_analysis():
    root = get_project_root()
    metrics_path = root / "experiments" / "runs" / "P03_student_mobilenetv3_small" / "metrics.json"
    assert metrics_path.is_file()

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "COMPLETED"
    assert data["student_architecture"] == "mobilenet_v3_small"
    assert data["training_mode"] == "end_to_end_fine_tuning_without_distillation"

    # Clean domain test
    clean_stats = data["summary_statistics"]["clean_domain_plantvillage"]
    assert clean_stats["macro_f1_mean"] > 0.95
    assert clean_stats["accuracy_mean"] > 0.95

    # Cross domain degradation check
    cross_stats = data["summary_statistics"]["cross_domain_plantdoc"]
    assert cross_stats["macro_f1_mean"] < 0.25

    # Gap analysis check
    gap = data["teacher_vs_student_gap_analysis"]
    assert gap["efficiency_tradeoff"]["meets_target_a_param_bound"] is True
    assert gap["efficiency_tradeoff"]["meets_target_b_size_bound"] is True
    assert gap["efficiency_tradeoff"]["cpu_speedup_factor"] > 3.0
    assert gap["cross_domain"]["absolute_f1_gap"] > 0.0  # Teacher outperforms student cross-domain
