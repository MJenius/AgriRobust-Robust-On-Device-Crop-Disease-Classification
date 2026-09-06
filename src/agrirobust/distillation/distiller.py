"""Distillation training pipeline supporting logit, feature hint, and combined distillation with live progress output."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from agrirobust.config import get_project_root
from agrirobust.data.dataset import build_dataloaders, get_canonical_class_mapping
from agrirobust.data.transforms import get_eval_transforms, get_train_transforms
from agrirobust.distillation.dataset import DistillationDataset
from agrirobust.distillation.losses import CombinedKDLoss, FeatureHintLoss, ResponseKDLoss
from agrirobust.evaluation.evaluator import evaluate_classifier
from agrirobust.models.student import build_student_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_distillation(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str = "response",
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(seed)
    student.to(device)

    # Instantiate loss
    if mode == "response":
        criterion = ResponseKDLoss(temperature=temperature, alpha=alpha).to(device)
        proj_head = None
        params_to_opt = list(student.parameters())
    elif mode == "feature":
        criterion = FeatureHintLoss(student_dim=576, teacher_dim=768, beta=beta).to(device)
        proj_head = criterion.projection.to(device)
        params_to_opt = list(student.parameters()) + list(proj_head.parameters())
    elif mode == "combined":
        criterion = CombinedKDLoss(
            student_dim=576, teacher_dim=768, temperature=temperature, alpha=alpha, beta=beta
        ).to(device)
        proj_head = criterion.feature_kd.projection.to(device)
        params_to_opt = list(student.parameters()) + list(proj_head.parameters())
    else:
        raise ValueError(f"Unknown distillation mode: {mode}")

    optimizer = AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_macro_f1 = 0.0
    best_student_state = None
    history = []

    print(f"\n==================================================", flush=True)
    print(f"Starting Distillation: Mode='{mode}', T={temperature}, alpha={alpha}, beta={beta}, Seed={seed}", flush=True)
    print(f"==================================================", flush=True)

    n_batches = len(train_loader)
    for epoch in range(epochs):
        student.train()
        if proj_head is not None:
            proj_head.train()

        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, t_logits, t_feats, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            t_logits = t_logits.to(device)
            t_feats = t_feats.to(device)
            targets = targets.to(device)

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

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t0
                pct = (batch_idx + 1) / n_batches * 100.0
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{n_batches}] ({pct:.1f}%) "
                    f"Loss: {loss.item():.4f} | Batch Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        val_metrics = evaluate_classifier(student, val_loader, device=device)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_bal = val_metrics["balanced_accuracy"]

        print(
            f"--> Epoch [{epoch+1}/{epochs}] Summary | Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4%} | Val Macro F1: {val_f1:.4f} | Val Bal Acc: {val_bal:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "val_balanced_accuracy": val_bal,
        })

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            # Note: Save ONLY the unchanged student state_dict for deployment!
            best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_student_state)
    return student, {"best_val_macro_f1": best_val_macro_f1, "history": history}


def run_single_distillation_experiment(
    mode: str = "response",
    temperature: float = 4.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 128,
):
    root = get_project_root()
    device = "cpu"
    set_seed(seed)

    class_to_idx, _ = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # 1. Dataset setup
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    train_targets_file = root / "data" / "processed" / "teacher_targets" / "plantvillage_train_teacher_targets.pt"

    train_ds = DistillationDataset(
        manifest_path=manifest_path,
        targets_file=train_targets_file,
        split="train",
        transform=get_train_transforms(224),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Validation DataLoader
    val_loaders = build_dataloaders(batch_size=batch_size, image_size=224, num_workers=0, seed=seed)
    val_loader = val_loaders["val"]

    # 2. Student model
    student = build_student_model(num_classes=num_classes, pretrained=True)

    # 3. Train with distillation
    student, summary = train_distillation(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    # 4. Save Deployment Checkpoint (strictly mobile architecture, projection discarded)
    ckpts_dir = root / "experiments" / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    exp_id = f"P04_student_kd_{mode}_plantvillage_s{seed}"
    ckpt_path = ckpts_dir / f"{exp_id}.pt"
    torch.save(student.state_dict(), ckpt_path)

    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    ckpt_sha256 = h.hexdigest()

    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"\n[DONE] Saved Distilled Student Checkpoint -> {ckpt_path}", flush=True)
    print(f"Deployment Checkpoint Size: {file_size_mb:.3f} MB (SHA-256: {ckpt_sha256})", flush=True)

    return student, ckpt_path, ckpt_sha256, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="response", choices=["response", "feature", "combined"])
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_single_distillation_experiment(
        mode=args.mode,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
