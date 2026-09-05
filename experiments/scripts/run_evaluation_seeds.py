"""Multi-seed evaluation runner for SOT evaluation seeds (42, 1337, 2026)."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from agrirobust.config import get_project_root
from agrirobust.data.dataset import get_canonical_class_mapping
from agrirobust.evaluation.evaluator import compute_model_efficiency
from agrirobust.evaluation.plantseg_eval import assess_plantseg_compatibility
from agrirobust.models.teacher import build_teacher_model
import sys
sys.path.insert(0, str(get_project_root()))
from experiments.scripts.train_teacher import train_teacher_head_on_features


def evaluate_head_on_features(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    idx_to_class: Dict[int, str] = None,
    restrict_to_target_classes: bool = False,
) -> Dict[str, Any]:
    head.eval()
    with torch.no_grad():
        logits = head(features)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    y_true = targets.numpy()
    y_pred = preds.numpy()

    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    if restrict_to_target_classes:
        target_labels = sorted(list(set(y_true)))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=target_labels, zero_division=0))
        present_labels = target_labels
    else:
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        present_labels = sorted(list(set(y_true).union(set(y_pred))))

    per_class_f1_raw = f1_score(y_true, y_pred, average=None, labels=present_labels, zero_division=0)
    per_class_f1 = {
        idx_to_class.get(lbl, str(lbl)): round(float(f1), 4)
        for lbl, f1 in zip(present_labels, per_class_f1_raw)
    }
    cm = confusion_matrix(y_true, y_pred, labels=present_labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "total_samples": len(y_true),
        "evaluated_classes_count": len(present_labels),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def run_all_seeds_evaluation(seeds: List[int] = [42, 1337, 2026]) -> Dict[str, Any]:
    root = get_project_root()
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # Load pre-extracted feature tensors
    feat_dir = root / "data" / "processed" / "features"
    train_data = torch.load(feat_dir / "plantvillage_train_convnext_tiny_features.pt")
    val_data = torch.load(feat_dir / "plantvillage_val_convnext_tiny_features.pt")
    test_pv = torch.load(feat_dir / "plantvillage_test_convnext_tiny_features.pt")
    test_pd = torch.load(feat_dir / "plantdoc_crops_convnext_tiny_features.pt")

    runs = []
    pv_clean_accs, pv_clean_f1s, pv_clean_bals = [], [], []
    pd_cross_accs, pd_cross_f1s, pd_cross_bals = [], [], []

    for seed in seeds:
        print(f"\n--- Running Teacher Head Training & Eval for Seed {seed} ---")
        head, summary = train_teacher_head_on_features(
            train_features=train_data["features"],
            train_labels=train_data["labels"],
            val_features=val_data["features"],
            val_labels=val_data["labels"],
            num_classes=num_classes,
            epochs=15,
            seed=seed,
            device="cpu",
        )

        # 1. Clean Test (PlantVillage)
        pv_res = evaluate_head_on_features(
            head, test_pv["features"], test_pv["labels"], idx_to_class=idx_to_class
        )
        pv_clean_accs.append(pv_res["accuracy"])
        pv_clean_f1s.append(pv_res["macro_f1"])
        pv_clean_bals.append(pv_res["balanced_accuracy"])

        # 2. Cross-Domain Test (PlantDoc - 29 shared classes)
        pd_res = evaluate_head_on_features(
            head, test_pd["features"], test_pd["labels"], idx_to_class=idx_to_class, restrict_to_target_classes=True
        )
        pd_cross_accs.append(pd_res["accuracy"])
        pd_cross_f1s.append(pd_res["macro_f1"])
        pd_cross_bals.append(pd_res["balanced_accuracy"])

        # Save checkpoint for each seed
        ckpt_dir = root / "experiments" / "checkpoints"
        ckpt_path = ckpt_dir / f"P02_teacher_convnext_tiny_plantvillage_s{seed}.pt"

        # Build full teacher model for saving
        teacher = build_teacher_model(num_classes=num_classes, pretrained=False)
        # We know features are untouched, so we save classifier weights
        teacher.classifier.weight.data.copy_(head[1].weight.data)
        teacher.classifier.bias.data.copy_(head[1].bias.data)

        torch.save(
            {
                "experiment_id": f"P02_teacher_convnext_tiny_plantvillage_s{seed}",
                "architecture": "convnext_tiny",
                "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
                "num_classes": num_classes,
                "seed": seed,
                "state_dict": teacher.state_dict(),
                "class_to_idx": class_to_idx,
            },
            ckpt_path,
        )

        # Checksum
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192 * 16):
                h.update(chunk)
        ckpt_sha256 = h.hexdigest()

        runs.append({
            "seed": seed,
            "checkpoint_path": str(ckpt_path.relative_to(root).as_posix()),
            "checkpoint_sha256": ckpt_sha256,
            "plantvillage_clean_test": {
                "accuracy": pv_res["accuracy"],
                "macro_f1": pv_res["macro_f1"],
                "balanced_accuracy": pv_res["balanced_accuracy"],
            },
            "plantdoc_cross_domain_test": {
                "accuracy": pd_res["accuracy"],
                "macro_f1": pd_res["macro_f1"],
                "balanced_accuracy": pd_res["balanced_accuracy"],
            },
            "domain_shift_degradation": {
                "absolute_f1_drop": round(pv_res["macro_f1"] - pd_res["macro_f1"], 4),
                "relative_f1_drop_pct": round(((pv_res["macro_f1"] - pd_res["macro_f1"]) / pv_res["macro_f1"]) * 100, 2),
            },
            "per_class_f1_clean": pv_res["per_class_f1"],
            "per_class_f1_cross": pd_res["per_class_f1"],
            "confusion_matrix_clean": pv_res["confusion_matrix"],
            "confusion_matrix_cross": pd_res["confusion_matrix"],
        })

    # Summary statistics across seeds
    summary_stats = {
        "seeds": seeds,
        "clean_domain_plantvillage": {
            "accuracy_mean": round(float(np.mean(pv_clean_accs)), 4),
            "accuracy_std": round(float(np.std(pv_clean_accs)), 4),
            "macro_f1_mean": round(float(np.mean(pv_clean_f1s)), 4),
            "macro_f1_std": round(float(np.std(pv_clean_f1s)), 4),
            "balanced_accuracy_mean": round(float(np.mean(pv_clean_bals)), 4),
            "balanced_accuracy_std": round(float(np.std(pv_clean_bals)), 4),
        },
        "cross_domain_plantdoc": {
            "accuracy_mean": round(float(np.mean(pd_cross_accs)), 4),
            "accuracy_std": round(float(np.std(pd_cross_accs)), 4),
            "macro_f1_mean": round(float(np.mean(pd_cross_f1s)), 4),
            "macro_f1_std": round(float(np.std(pd_cross_f1s)), 4),
            "balanced_accuracy_mean": round(float(np.mean(pd_cross_bals)), 4),
            "balanced_accuracy_std": round(float(np.std(pd_cross_bals)), 4),
        },
    }

    # Primary Teacher Efficiency (measured on seed 42 checkpoint)
    ref_ckpt = root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt"
    ref_teacher = build_teacher_model(num_classes=num_classes, pretrained=False)
    ref_teacher.load_state_dict(torch.load(ref_ckpt)["state_dict"])
    efficiency = compute_model_efficiency(ref_teacher, ref_ckpt, num_runs=50)

    # PlantSeg secondary assessment
    plantseg_assessment = assess_plantseg_compatibility()

    full_output = {
        "experiment_series": "P02_teacher_convnext_tiny_plantvillage",
        "phase": 2,
        "status": "COMPLETED",
        "architecture": "convnext_tiny",
        "backbone_weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "num_classes": num_classes,
        "input_resolution": [3, 224, 224],
        "optimizer": "AdamW (lr=0.001, weight_decay=1e-4, CosineAnnealingLR to 1e-5)",
        "epochs": 15,
        "efficiency": efficiency,
        "summary_statistics": summary_stats,
        "runs": runs,
        "plantseg_assessment": plantseg_assessment,
    }

    # Save master metrics.json
    out_dir = root / "experiments" / "runs" / "P02_teacher_convnext_tiny"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_file = out_dir / "metrics.json"
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nAll-seed evaluation complete! Master metrics written to {out_metrics_file}")
    return full_output


if __name__ == "__main__":
    run_all_seeds_evaluation()
