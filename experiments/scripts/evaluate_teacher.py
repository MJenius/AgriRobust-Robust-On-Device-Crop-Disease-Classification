"""Comprehensive evaluation harness for Phase 2 Teacher Baseline."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from agrirobust.config import get_project_root
from agrirobust.data.dataset import (
    build_dataloaders,
    build_plantdoc_dataloader,
    get_canonical_class_mapping,
)
from agrirobust.evaluation.evaluator import compute_model_efficiency
from agrirobust.evaluation.plantseg_eval import assess_plantseg_compatibility
from agrirobust.models.teacher import build_teacher_model


def evaluate_model_on_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    idx_to_class: Dict[int, str] = None,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_confidences = []

    print(f"Evaluating {len(dataloader.dataset)} samples...")
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

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    elapsed = time.time() - t0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    present_labels = sorted(list(set(y_true).union(set(y_pred))))
    per_class_f1_raw = f1_score(y_true, y_pred, average=None, labels=present_labels, zero_division=0)
    per_class_f1_dict = {}
    for idx_val, f1_val in zip(present_labels, per_class_f1_raw):
        cls_name = idx_to_class.get(idx_val, f"class_{idx_val}") if idx_to_class else str(idx_val)
        per_class_f1_dict[cls_name] = round(float(f1_val), 4)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "evaluated_samples": len(y_true),
        "evaluated_classes": len(present_labels),
        "elapsed_seconds": round(elapsed, 2),
        "per_class_f1": per_class_f1_dict,
        "confusion_matrix": cm,
    }


def run_full_teacher_evaluation(seed: int = 42) -> Dict[str, Any]:
    root = get_project_root()
    device = "cpu"
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # Load Checkpoint
    ckpt_path = root / "experiments" / "checkpoints" / f"P02_teacher_convnext_tiny_plantvillage_s{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    teacher = build_teacher_model(num_classes=num_classes, pretrained=False)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    teacher.to(device)

    print("1. Computing efficiency and latency metrics...")
    efficiency = compute_model_efficiency(
        model=teacher,
        checkpoint_path=ckpt_path,
        image_size=224,
        device=device,
        num_warmup=10,
        num_runs=50,
    )

    print("2. Evaluating Clean Test: PlantVillage held-out test split...")
    loaders = build_dataloaders(batch_size=64, image_size=224, num_workers=0)
    clean_metrics = evaluate_model_on_dataloader(teacher, loaders["test"], device=device, idx_to_class=idx_to_class)

    print("3. Evaluating Cross-Domain Test: PlantDoc leaf crops...")
    pd_loader = build_plantdoc_dataloader(batch_size=64, image_size=224, num_workers=0)
    cross_metrics = evaluate_model_on_dataloader(teacher, pd_loader, device=device, idx_to_class=idx_to_class)

    print("4. Assessing PlantSeg secondary evaluation compatibility...")
    plantseg_assessment = assess_plantseg_compatibility()

    # Calculate domain shift degradation
    clean_f1 = clean_metrics["macro_f1"]
    cross_f1 = cross_metrics["macro_f1"]
    abs_drop = round(clean_f1 - cross_f1, 4)
    rel_drop = round((abs_drop / clean_f1) * 100, 2) if clean_f1 > 0 else 0.0

    domain_shift_analysis = {
        "clean_macro_f1": clean_f1,
        "cross_domain_macro_f1": cross_f1,
        "absolute_f1_degradation": abs_drop,
        "relative_f1_degradation_pct": rel_drop,
        "clean_accuracy": clean_metrics["accuracy"],
        "cross_domain_accuracy": cross_metrics["accuracy"],
    }

    results = {
        "experiment_id": f"P02_teacher_convnext_tiny_plantvillage_s{seed}",
        "architecture": "convnext_tiny",
        "pretrained_weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "checkpoint_path": str(ckpt_path.relative_to(root).as_posix()),
        "checkpoint_sha256": ckpt.get("sha256", ""),
        "seed": seed,
        "efficiency": efficiency,
        "clean_domain_test_plantvillage": clean_metrics,
        "cross_domain_test_plantdoc": cross_metrics,
        "domain_shift_degradation": domain_shift_analysis,
        "plantseg_assessment": plantseg_assessment,
    }

    # Save metrics.json
    runs_dir = root / "experiments" / "runs" / f"P02_teacher_convnext_tiny_plantvillage_s{seed}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_file = runs_dir / "metrics.json"
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved full evaluation metrics -> {out_metrics_file}")
    return results


if __name__ == "__main__":
    run_full_teacher_evaluation(seed=42)
