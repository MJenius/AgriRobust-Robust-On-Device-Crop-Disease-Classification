"""Comprehensive evaluation harness and ablation comparator for Phase 4 Knowledge Distillation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from agrirobust.config import get_project_root
from agrirobust.data.dataset import (
    build_dataloaders,
    build_plantdoc_dataloader,
    get_canonical_class_mapping,
)
from agrirobust.evaluation.evaluator import compute_model_efficiency
from agrirobust.models.student import build_student_model


def evaluate_student_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    idx_to_class: Dict[int, str] = None,
    restrict_to_target_classes: bool = False,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_confidences = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    elapsed = time.time() - t0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

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
        "evaluated_classes": len(present_labels),
        "elapsed_seconds": round(elapsed, 2),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def evaluate_all_distillation_variants() -> Dict[str, Any]:
    root = get_project_root()
    device = "cpu"
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # Load Teacher and Baseline Student reference metrics
    p2_path = root / "experiments" / "runs" / "P02_teacher_convnext_tiny" / "metrics.json"
    with open(p2_path, "r", encoding="utf-8") as f:
        teacher_metrics = json.load(f)

    p3_path = root / "experiments" / "runs" / "P03_student_mobilenetv3_small" / "metrics.json"
    with open(p3_path, "r", encoding="utf-8") as f:
        baseline_student_metrics = json.load(f)

    teacher_clean_f1 = teacher_metrics["summary_statistics"]["clean_domain_plantvillage"]["macro_f1_mean"]
    teacher_cross_f1 = teacher_metrics["summary_statistics"]["cross_domain_plantdoc"]["macro_f1_mean"]

    student_base_clean_f1 = baseline_student_metrics["summary_statistics"]["clean_domain_plantvillage"]["macro_f1_mean"]
    student_base_cross_f1 = baseline_student_metrics["summary_statistics"]["cross_domain_plantdoc"]["macro_f1_mean"]

    # Dataloaders for testing
    loaders = build_dataloaders(batch_size=128, image_size=224, num_workers=0)
    test_clean_loader = loaders["test"]
    plantdoc_loader = build_plantdoc_dataloader(batch_size=128, image_size=224, num_workers=0)

    ckpts_dir = root / "experiments" / "checkpoints"
    modes = ["response", "feature", "combined"]
    ablation_results = {}

    for mode in modes:
        ckpt_name = f"P04_student_kd_{mode}_plantvillage_s42.pt"
        ckpt_path = ckpts_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"Checkpoint for mode '{mode}' not found at {ckpt_path}, skipping...", flush=True)
            continue

        print(f"\n--- Evaluating Distillation Variant: {mode.upper()} ---", flush=True)
        student = build_student_model(num_classes=num_classes, pretrained=False)
        student.load_state_dict(torch.load(ckpt_path, map_location=device))
        student.eval()

        # SHA-256
        h = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192 * 16):
                h.update(chunk)
        ckpt_sha256 = h.hexdigest()

        # 1. Clean Test
        clean_res = evaluate_student_model(student, test_clean_loader, device=device, idx_to_class=idx_to_class)

        # 2. Cross-Domain Test (PlantDoc)
        cross_res = evaluate_student_model(
            student, plantdoc_loader, device=device, idx_to_class=idx_to_class, restrict_to_target_classes=True
        )

        # 3. Efficiency
        eff = compute_model_efficiency(student, ckpt_path, num_runs=50)

        # 4. Deltas against Raw Student Baseline
        delta_clean_f1 = round(clean_res["macro_f1"] - student_base_clean_f1, 4)
        delta_cross_f1 = round(cross_res["macro_f1"] - student_base_cross_f1, 4)
        pct_cross_gain = (
            round((delta_cross_f1 / student_base_cross_f1) * 100, 2) if student_base_cross_f1 > 0 else 0.0
        )

        # 5. Gap to Teacher recovered
        total_gap = teacher_cross_f1 - student_base_cross_f1  # 0.0775
        gap_recovered_pct = round((delta_cross_f1 / total_gap) * 100, 2) if total_gap > 0 else 0.0

        ablation_results[mode] = {
            "mode": mode,
            "checkpoint_path": str(ckpt_path.relative_to(root).as_posix()),
            "checkpoint_sha256": ckpt_sha256,
            "clean_domain_plantvillage": clean_res,
            "cross_domain_plantdoc": cross_res,
            "efficiency": eff,
            "comparison_against_baseline_student": {
                "baseline_cross_f1": student_base_cross_f1,
                "distilled_cross_f1": cross_res["macro_f1"],
                "delta_cross_macro_f1": delta_cross_f1,
                "relative_cross_improvement_pct": pct_cross_gain,
                "gap_recovered_pct_of_teacher": gap_recovered_pct,
            },
        }

    # Master output
    output_data = {
        "phase": 4,
        "experiment_series": "P04_knowledge_distillation",
        "teacher_reference": {
            "architecture": teacher_metrics["architecture"],
            "clean_macro_f1": teacher_clean_f1,
            "plantdoc_cross_macro_f1": teacher_cross_f1,
            "parameters": teacher_metrics["efficiency"]["total_parameters"],
            "checkpoint_size_mb": teacher_metrics["efficiency"]["checkpoint_file_size_mb"],
            "latency_mean_ms": teacher_metrics["efficiency"]["latency_mean_ms"],
        },
        "student_baseline_reference": {
            "architecture": baseline_student_metrics["student_architecture"],
            "clean_macro_f1": student_base_clean_f1,
            "plantdoc_cross_macro_f1": student_base_cross_f1,
            "parameters": baseline_student_metrics["efficiency"]["total_parameters"],
            "checkpoint_size_mb": baseline_student_metrics["efficiency"]["checkpoint_file_size_mb"],
            "latency_mean_ms": baseline_student_metrics["efficiency"]["latency_mean_ms"],
        },
        "distillation_ablation_results": ablation_results,
    }

    out_dir = root / "experiments" / "runs" / "P04_knowledge_distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved Phase 4 Knowledge Distillation Metrics -> {out_file}", flush=True)
    return output_data


if __name__ == "__main__":
    evaluate_all_distillation_variants()
