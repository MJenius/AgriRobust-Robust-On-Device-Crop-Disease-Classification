"""Multi-seed evaluation and gap analysis harness for Phase 3 Compact Student Baseline."""

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


def evaluate_student_on_dataloader(
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


def run_student_evaluation_and_gap_analysis(seeds: List[int] = [42, 1337, 2026]) -> Dict[str, Any]:
    root = get_project_root()
    device = "cpu"
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # Load frozen teacher metrics for comparison
    teacher_metrics_path = root / "experiments" / "runs" / "P02_teacher_convnext_tiny" / "metrics.json"
    with open(teacher_metrics_path, "r", encoding="utf-8") as f:
        teacher_data = json.load(f)

    teacher_summary = teacher_data["summary_statistics"]
    teacher_eff = teacher_data["efficiency"]

    # Dataloaders for test
    loaders = build_dataloaders(batch_size=128, image_size=224, num_workers=0)
    test_clean_loader = loaders["test"]
    plantdoc_loader = build_plantdoc_dataloader(batch_size=128, image_size=224, num_workers=0)

    runs = []
    pv_clean_accs, pv_clean_f1s, pv_clean_bals = [], [], []
    pd_cross_accs, pd_cross_f1s, pd_cross_bals = [], [], []

    ckpts_dir = root / "experiments" / "checkpoints"

    for seed in seeds:
        print(f"\n--- Evaluating Student Checkpoint (Seed {seed}) ---")
        inf_ckpt_path = ckpts_dir / f"P03_student_mobilenetv3_small_plantvillage_s{seed}.pt"
        if not inf_ckpt_path.exists():
            print(f"Checkpoint not found for seed {seed}, skipping...")
            continue

        student = build_student_model(num_classes=num_classes, pretrained=False)
        student.load_state_dict(torch.load(inf_ckpt_path, map_location=device))
        student.eval()

        # SHA-256
        h = hashlib.sha256()
        with open(inf_ckpt_path, "rb") as f:
            while chunk := f.read(8192 * 16):
                h.update(chunk)
        ckpt_sha256 = h.hexdigest()

        # 1. Clean Test
        pv_res = evaluate_student_on_dataloader(
            student, test_clean_loader, device=device, idx_to_class=idx_to_class
        )
        pv_clean_accs.append(pv_res["accuracy"])
        pv_clean_f1s.append(pv_res["macro_f1"])
        pv_clean_bals.append(pv_res["balanced_accuracy"])

        # 2. Cross-Domain Test (PlantDoc)
        pd_res = evaluate_student_on_dataloader(
            student, plantdoc_loader, device=device, idx_to_class=idx_to_class, restrict_to_target_classes=True
        )
        pd_cross_accs.append(pd_res["accuracy"])
        pd_cross_f1s.append(pd_res["macro_f1"])
        pd_cross_bals.append(pd_res["balanced_accuracy"])

        runs.append({
            "seed": seed,
            "deployment_checkpoint_path": str(inf_ckpt_path.relative_to(root).as_posix()),
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

    # Summary Statistics
    summary_stats = {
        "seeds_evaluated": [r["seed"] for r in runs],
        "clean_domain_plantvillage": {
            "accuracy_mean": round(float(np.mean(pv_clean_accs)), 4) if pv_clean_accs else 0.0,
            "accuracy_std": round(float(np.std(pv_clean_accs)), 4) if pv_clean_accs else 0.0,
            "macro_f1_mean": round(float(np.mean(pv_clean_f1s)), 4) if pv_clean_f1s else 0.0,
            "macro_f1_std": round(float(np.std(pv_clean_f1s)), 4) if pv_clean_f1s else 0.0,
            "balanced_accuracy_mean": round(float(np.mean(pv_clean_bals)), 4) if pv_clean_bals else 0.0,
            "balanced_accuracy_std": round(float(np.std(pv_clean_bals)), 4) if pv_clean_bals else 0.0,
        },
        "cross_domain_plantdoc": {
            "accuracy_mean": round(float(np.mean(pd_cross_accs)), 4) if pd_cross_accs else 0.0,
            "accuracy_std": round(float(np.std(pd_cross_accs)), 4) if pd_cross_accs else 0.0,
            "macro_f1_mean": round(float(np.mean(pd_cross_f1s)), 4) if pd_cross_f1s else 0.0,
            "macro_f1_std": round(float(np.std(pd_cross_f1s)), 4) if pd_cross_f1s else 0.0,
            "balanced_accuracy_mean": round(float(np.mean(pd_cross_bals)), 4) if pd_cross_bals else 0.0,
            "balanced_accuracy_std": round(float(np.std(pd_cross_bals)), 4) if pd_cross_bals else 0.0,
        },
    }

    # Efficiency Benchmark (measured on Primary Seed 42 checkpoint)
    ref_ckpt = ckpts_dir / "P03_student_mobilenetv3_small_plantvillage_s42.pt"
    ref_student = build_student_model(num_classes=num_classes, pretrained=False)
    if ref_ckpt.exists():
        ref_student.load_state_dict(torch.load(ref_ckpt, map_location=device))
    efficiency = compute_model_efficiency(ref_student, ref_ckpt, num_runs=50)

    # Calculate Teacher vs Student Gap
    teacher_clean_f1 = teacher_summary["clean_domain_plantvillage"]["macro_f1_mean"]
    student_clean_f1 = summary_stats["clean_domain_plantvillage"]["macro_f1_mean"]
    clean_f1_gap = round(teacher_clean_f1 - student_clean_f1, 4)
    clean_retention_pct = round((student_clean_f1 / teacher_clean_f1) * 100, 2) if teacher_clean_f1 > 0 else 0.0

    teacher_cross_f1 = teacher_summary["cross_domain_plantdoc"]["macro_f1_mean"]
    student_cross_f1 = summary_stats["cross_domain_plantdoc"]["macro_f1_mean"]
    cross_f1_gap = round(teacher_cross_f1 - student_cross_f1, 4)
    cross_retention_pct = round((student_cross_f1 / teacher_cross_f1) * 100, 2) if teacher_cross_f1 > 0 else 0.0

    param_ratio = round((efficiency["total_parameters"] / teacher_eff["total_parameters"]) * 100, 2)
    size_ratio = round((efficiency["checkpoint_file_size_mb"] / teacher_eff["checkpoint_file_size_mb"]) * 100, 2)
    speedup = round(teacher_eff["latency_mean_ms"] / efficiency["latency_mean_ms"], 2) if efficiency["latency_mean_ms"] > 0 else 0.0

    gap_analysis = {
        "teacher_architecture": teacher_data["architecture"],
        "student_architecture": "mobilenet_v3_small",
        "clean_domain": {
            "teacher_macro_f1": teacher_clean_f1,
            "student_macro_f1": student_clean_f1,
            "absolute_f1_gap": clean_f1_gap,
            "performance_retention_pct": clean_retention_pct,
        },
        "cross_domain": {
            "teacher_macro_f1": teacher_cross_f1,
            "student_macro_f1": student_cross_f1,
            "absolute_f1_gap": cross_f1_gap,
            "performance_retention_pct": cross_retention_pct,
        },
        "efficiency_tradeoff": {
            "teacher_parameters": teacher_eff["total_parameters"],
            "student_parameters": efficiency["total_parameters"],
            "parameter_ratio_pct": param_ratio,
            "meets_target_a_param_bound": efficiency["total_parameters"] <= 0.10 * teacher_eff["total_parameters"],
            "teacher_file_size_mb": teacher_eff["checkpoint_file_size_mb"],
            "student_file_size_mb": efficiency["checkpoint_file_size_mb"],
            "meets_target_b_size_bound": efficiency["checkpoint_file_size_mb"] <= 15.0,
            "teacher_cpu_latency_ms": teacher_eff["latency_mean_ms"],
            "student_cpu_latency_ms": efficiency["latency_mean_ms"],
            "cpu_speedup_factor": speedup,
        },
    }

    full_output = {
        "experiment_series": "P03_student_mobilenetv3_small_plantvillage",
        "phase": 3,
        "status": "COMPLETED",
        "student_architecture": "mobilenet_v3_small",
        "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "training_mode": "end_to_end_fine_tuning_without_distillation",
        "training_protocol": {
            "optimizer": "AdamW (lr=0.001, weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR (T_max=15, eta_min=1e-5)",
            "batch_size": 128,
            "epochs": 15,
            "loss": "CrossEntropyLoss",
        },
        "efficiency": efficiency,
        "summary_statistics": summary_stats,
        "runs": runs,
        "teacher_vs_student_gap_analysis": gap_analysis,
    }

    out_dir = root / "experiments" / "runs" / "P03_student_mobilenetv3_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nSaved Student Metrics and Gap Analysis -> {out_file}")
    return full_output


if __name__ == "__main__":
    run_student_evaluation_and_gap_analysis()
