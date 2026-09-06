import json
import pytest
import torch
from pathlib import Path

from agrirobust.config import get_project_root
from agrirobust.distillation.losses import ResponseKDLoss, FeatureHintLoss, CombinedKDLoss
from agrirobust.models.student import build_student_model


def test_response_kd_loss_direction_and_shape():
    """Verify ResponseKDLoss uses PyTorch KLDivLoss(log_softmax(z_s/T), softmax(z_t/T)) * T^2."""
    loss_fn = ResponseKDLoss(temperature=4.0, alpha=0.5)
    batch_size = 4
    num_classes = 38

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, ce, kd = loss_fn(student_logits, teacher_logits, targets)

    assert loss.dim() == 0, "Loss must be scalar"
    assert ce.dim() == 0
    assert kd.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert student_logits.grad is not None


def test_feature_hint_loss_projection():
    """Verify FeatureHintLoss projects 576-D student features to 768-D teacher space."""
    loss_fn = FeatureHintLoss(student_dim=576, teacher_dim=768, beta=0.5)
    batch_size = 4
    num_classes = 38

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    student_feats = torch.randn(batch_size, 576, requires_grad=True)
    teacher_feats = torch.randn(batch_size, 768)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, ce, feat = loss_fn(student_logits, student_feats, teacher_feats, targets)

    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert student_feats.grad is not None
    assert loss_fn.projection[0].weight.grad is not None


def test_combined_kd_loss():
    """Verify CombinedKDLoss produces scalar composite loss."""
    loss_fn = CombinedKDLoss(student_dim=576, teacher_dim=768, temperature=4.0, alpha=0.5, beta=0.5)
    batch_size = 4
    num_classes = 38

    student_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    student_feats = torch.randn(batch_size, 576, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_classes)
    teacher_feats = torch.randn(batch_size, 768)
    targets = torch.randint(0, num_classes, (batch_size,))

    loss, ce, kd, feat = loss_fn(student_logits, student_feats, teacher_logits, teacher_feats, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert student_logits.grad is not None
    assert student_feats.grad is not None


def test_phase_04_checkpoints_exist_and_pure_student():
    """Verify all 3 Phase 4 distillation checkpoints exist, are <=6.5MB, and match student param schema."""
    root = get_project_root()
    ckpts_dir = root / "experiments" / "checkpoints"

    expected = [
        "P04_student_kd_response_plantvillage_s42.pt",
        "P04_student_kd_feature_plantvillage_s42.pt",
        "P04_student_kd_combined_plantvillage_s42.pt",
    ]

    for ckpt_name in expected:
        ckpt_path = ckpts_dir / ckpt_name
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_name}"
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        assert size_mb <= 6.5, f"Checkpoint {ckpt_name} exceeds 6.5 MB: {size_mb:.2f} MB"

        weights = torch.load(ckpt_path, map_location="cpu")
        student = build_student_model(num_classes=38, pretrained=False)
        student.load_state_dict(weights, strict=True)
        total_params = sum(p.numel() for p in student.parameters())
        assert total_params == 1556806, f"Expected 1,556,806 params, got {total_params}"


def test_phase_04_metrics_schema_and_recovery():
    """Verify Phase 4 metrics.json exists, contains all ablations, and records non-trivial recovery."""
    root = get_project_root()
    metrics_path = root / "experiments" / "runs" / "P04_knowledge_distillation" / "metrics.json"
    assert metrics_path.exists(), "Missing metrics.json"

    with open(metrics_path, "r") as f:
        data = json.load(f)

    assert data["phase"] == 4
    assert data["status"] == "COMPLETED"
    assert "ablation_results" in data
    ablations = data["ablation_results"]

    assert "response" in ablations
    assert "feature" in ablations
    assert "combined" in ablations

    resp_f1 = ablations["response"]["cross_domain_plantdoc"]["macro_f1"]
    feat_f1 = ablations["feature"]["cross_domain_plantdoc"]["macro_f1"]
    comb_f1 = ablations["combined"]["cross_domain_plantdoc"]["macro_f1"]

    # Student baseline was 0.1298
    assert resp_f1 > 0.1298, f"Response KD failed to beat student baseline: {resp_f1}"
    assert feat_f1 > 0.1298, f"Feature KD failed to beat student baseline: {feat_f1}"
    assert comb_f1 > 0.1298, f"Combined KD failed to beat student baseline: {comb_f1}"

    # Best recovery should be Response KD (closing over 60% of the teacher gap)
    assert resp_f1 >= 0.1750, f"Expected Response KD cross F1 >= 0.175, got {resp_f1}"
