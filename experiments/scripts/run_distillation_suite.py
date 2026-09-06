"""Self-contained end-to-end distillation pipeline running all ablations (Response, Feature, Combined).

Features:
- Live unbuffered terminal printing (flush=True) with timestamps and progress percentages.
- Automatic intermediate checkpointing after every epoch to prevent any data loss.
- Evaluates on clean PlantVillage test set and PlantDoc cross-domain test set.
- Generates master metrics.json and gap recovery table.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from agrirobust.config import get_project_root
from agrirobust.data.dataset import (
    build_dataloaders,
    build_plantdoc_dataloader,
    get_canonical_class_mapping,
)
from agrirobust.data.transforms import get_eval_transforms, get_train_transforms
from agrirobust.distillation.dataset import DistillationDataset
from agrirobust.distillation.losses import CombinedKDLoss, FeatureHintLoss, ResponseKDLoss
from agrirobust.evaluation.evaluator import compute_model_efficiency
from agrirobust.models.student import build_student_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_student(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    idx_to_class: Dict[int, str] = None,
    restrict_to_target_classes: bool = False,
) -> Dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    t0 = time.time()
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

    elapsed = time.time() - t0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    if restrict_to_target_classes:
        labels = sorted(list(set(y_true)))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    else:
        labels = sorted(list(set(y_true).union(set(y_pred))))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1_raw = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = {
        idx_to_class.get(lbl, str(lbl)): round(float(f1), 4) for lbl, f1 in zip(labels, per_class_f1_raw)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "total_samples": len(y_true),
        "elapsed_seconds": round(elapsed, 2),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }


def train_and_evaluate_distillation_variant(
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_clean_loader: DataLoader,
    test_cross_loader: DataLoader,
    idx_to_class: Dict[int, str],
    num_classes: int = 38,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    root = get_project_root()
    ckpts_dir = root / "experiments" / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    student = build_student_model(num_classes=num_classes, pretrained=True).to(device)

    if mode == "response":
        criterion = ResponseKDLoss(temperature=temperature, alpha=alpha).to(device)
        proj_head = None
        opt_params = list(student.parameters())
    elif mode == "feature":
        criterion = FeatureHintLoss(student_dim=576, teacher_dim=768, beta=beta).to(device)
        proj_head = criterion.projection.to(device)
        opt_params = list(student.parameters()) + list(proj_head.parameters())
    elif mode == "combined":
        criterion = CombinedKDLoss(
            student_dim=576, teacher_dim=768, temperature=temperature, alpha=alpha, beta=beta
        ).to(device)
        proj_head = criterion.feature_kd.projection.to(device)
        opt_params = list(student.parameters()) + list(proj_head.parameters())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    optimizer = AdamW(opt_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_student_weights = None
    deployment_ckpt = ckpts_dir / f"P04_student_kd_{mode}_plantvillage_s{seed}.pt"

    print(f"\n=======================================================", flush=True)
    print(f"DISTILLATION EXPERIMENT: Mode = {mode.upper()}", flush=True)
    print(f"Parameters: T={temperature}, alpha={alpha}, beta={beta}, epochs={epochs}", flush=True)
    print(f"Output Checkpoint: {deployment_ckpt}", flush=True)
    print(f"=======================================================", flush=True)

    n_batches = len(train_loader)
    for epoch in range(epochs):
        student.train()
        if proj_head is not None:
            proj_head.train()

        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, t_logits, t_feats, targets) in enumerate(train_loader):
            imgs, t_logits, t_feats, targets = (
                imgs.to(device),
                t_logits.to(device),
                t_feats.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()

            if mode == "response":
                s_logits = student(imgs)
                loss, ce, kd = criterion(s_logits, t_logits, targets)
            elif mode == "feature":
                s_logits, s_feats = student(imgs, return_features=True)
                loss, ce, feat_loss = criterion(s_logits, s_feats, t_feats, targets)
            elif mode == "combined":
                s_logits, s_feats = student(imgs, return_features=True)
                loss, ce, kd, feat_loss = criterion(s_logits, s_feats, t_logits, t_feats, targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
                step_elapsed = time.time() - t0
                pct = (batch_idx + 1) / n_batches * 100.0
                print(
                    f"[{mode.upper()}] Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{n_batches}] ({pct:.1f}%) "
                    f"Loss: {loss.item():.4f} | Elapsed: {step_elapsed:.1f}s",
                    flush=True,
                )

        scheduler.step()
        epoch_loss = total_loss / n_batches

        # Validation pass
        val_res = evaluate_student(student, val_loader, device=device, idx_to_class=idx_to_class)
        print(
            f"--> [{mode.upper()}] Epoch [{epoch+1}/{epochs}] Done | "
            f"Train Loss: {epoch_loss:.4f} | Val Acc: {val_res['accuracy']:.4%} | Val F1: {val_res['macro_f1']:.4f}",
            flush=True,
        )

        # Checkpoint selection based on Val Macro F1
        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            # Important: save ONLY pure student state_dict for deployment! Projection head is discarded.
            best_student_weights = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            torch.save(best_student_weights, deployment_ckpt)
            print(f"  [CHECKPOINT] New best {mode} model saved (Val F1: {best_val_f1:.4f}) -> {deployment_ckpt}", flush=True)

    # Load best checkpoint for rigorous final evaluation
    student.load_state_dict(torch.load(deployment_ckpt, map_location=device))

    h = hashlib.sha256()
    with open(deployment_ckpt, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    ckpt_sha = h.hexdigest()

    # 1. Clean test evaluation
    clean_eval = evaluate_student(student, test_clean_loader, device=device, idx_to_class=idx_to_class)

    # 2. Cross-domain PlantDoc evaluation
    cross_eval = evaluate_student(
        student, test_cross_loader, device=device, idx_to_class=idx_to_class, restrict_to_target_classes=True
    )

    # 3. Efficiency measurement
    eff = compute_model_efficiency(student, deployment_ckpt, num_runs=50)

    return {
        "mode": mode,
        "checkpoint_path": str(deployment_ckpt.relative_to(root).as_posix()),
        "checkpoint_sha256": ckpt_sha,
        "best_val_macro_f1": best_val_f1,
        "clean_domain_plantvillage": clean_eval,
        "cross_domain_plantdoc": cross_eval,
        "efficiency": eff,
    }


def run_phase_04_ablation_suite(epochs: int = 10, batch_size: int = 128, seed: int = 42):
    root = get_project_root()
    device = "cpu"
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # 1. Dataloaders
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    train_targets_file = root / "data" / "processed" / "teacher_targets" / "plantvillage_train_teacher_targets.pt"

    train_ds = DistillationDataset(
        manifest_path=manifest_path,
        targets_file=train_targets_file,
        split="train",
        transform=get_train_transforms(224),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    loaders = build_dataloaders(batch_size=batch_size, image_size=224, num_workers=0, seed=seed)
    val_loader = loaders["val"]
    test_clean_loader = loaders["test"]
    test_cross_loader = build_plantdoc_dataloader(batch_size=batch_size, image_size=224, num_workers=0)

    # Load references
    p2_path = root / "experiments" / "runs" / "P02_teacher_convnext_tiny" / "metrics.json"
    with open(p2_path, "r", encoding="utf-8") as f:
        teacher_data = json.load(f)

    p3_path = root / "experiments" / "runs" / "P03_student_mobilenetv3_small" / "metrics.json"
    with open(p3_path, "r", encoding="utf-8") as f:
        student_base_data = json.load(f)

    t_clean_f1 = teacher_data["summary_statistics"]["clean_domain_plantvillage"]["macro_f1_mean"]
    t_cross_f1 = teacher_data["summary_statistics"]["cross_domain_plantdoc"]["macro_f1_mean"]

    s_clean_f1 = student_base_data["summary_statistics"]["clean_domain_plantvillage"]["macro_f1_mean"]
    s_cross_f1 = student_base_data["summary_statistics"]["cross_domain_plantdoc"]["macro_f1_mean"]
    total_teacher_gap = round(t_cross_f1 - s_cross_f1, 4)  # 0.0775

    ablation_results = {}
    for mode in ["response", "feature", "combined"]:
        res = train_and_evaluate_distillation_variant(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_clean_loader=test_clean_loader,
            test_cross_loader=test_cross_loader,
            idx_to_class=idx_to_class,
            num_classes=num_classes,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-4,
            temperature=4.0,
            alpha=0.5,
            beta=0.5,
            device=device,
            seed=seed,
        )

        dist_cross_f1 = res["cross_domain_plantdoc"]["macro_f1"]
        delta_f1 = round(dist_cross_f1 - s_cross_f1, 4)
        pct_rel_recovery = round((delta_f1 / total_teacher_gap) * 100, 2) if total_teacher_gap > 0 else 0.0

        res["gap_recovery"] = {
            "student_baseline_cross_f1": s_cross_f1,
            "distilled_cross_f1": dist_cross_f1,
            "teacher_cross_f1": t_cross_f1,
            "absolute_f1_change": delta_f1,
            "pct_teacher_gap_recovered": pct_rel_recovery,
        }
        ablation_results[mode] = res

    # Master output
    master_metrics = {
        "phase": 4,
        "status": "COMPLETED",
        "experiment_series": "P04_knowledge_distillation",
        "student_architecture": "mobilenet_v3_small",
        "teacher_architecture": "convnext_tiny",
        "training_budget": f"{epochs} epochs per variant, AdamW (lr=0.001, weight_decay=1e-4), batch_size={batch_size}",
        "teacher_cross_macro_f1": t_cross_f1,
        "student_baseline_cross_macro_f1": s_cross_f1,
        "uncompressed_cross_domain_gap": total_teacher_gap,
        "ablation_results": ablation_results,
    }

    out_dir = root / "experiments" / "runs" / "P04_knowledge_distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)

    print(f"\n=======================================================", flush=True)
    print(f"PHASE 4 ABLATION SUITE COMPLETE!", flush=True)
    print(f"Master metrics written to: {out_file}", flush=True)
    print(f"=======================================================", flush=True)
    return master_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_phase_04_ablation_suite(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
