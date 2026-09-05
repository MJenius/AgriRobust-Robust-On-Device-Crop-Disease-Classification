"""End-to-end training and fine-tuning pipeline for AgriRobust Phase 3 Compact Student Baseline."""

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
from agrirobust.evaluation.evaluator import evaluate_classifier
from agrirobust.models.student import build_student_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_student_end_to_end(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Perform full end-to-end fine-tuning of the student model without distillation or compression."""
    set_seed(seed)
    student.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_macro_f1 = 0.0
    best_state = None
    history = []

    print(f"--> Starting End-to-End Student Fine-Tuning for {epochs} epochs (seed {seed})...")
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        t0 = time.time()

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = student(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{n_batches}] "
                    f"Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s"
                )

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Validation evaluation
        val_metrics = evaluate_classifier(student, val_loader, device=device)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        val_bal = val_metrics["balanced_accuracy"]

        print(
            f"Epoch [{epoch+1}/{epochs}] Summary -> Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4%} | Val Macro F1: {val_f1:.4f} | Val Bal Acc: {val_bal:.4f}"
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
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    student.load_state_dict(best_state)
    return student, {"best_val_macro_f1": best_val_macro_f1, "history": history}


def run_student_pipeline(seed: int = 42, epochs: int = 15, batch_size: int = 128):
    root = get_project_root()
    device = "cpu"
    set_seed(seed)

    print("==================================================")
    print(f"AgriRobust Phase 3 — Student Baseline Pipeline (Seed {seed})")
    print("==================================================")

    class_to_idx, _ = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # 1. Build Student model
    student = build_student_model(num_classes=num_classes, pretrained=True)

    # 2. Build DataLoaders
    loaders = build_dataloaders(batch_size=batch_size, image_size=224, num_workers=0, seed=seed)

    # 3. End-to-end fine-tuning
    student, train_summary = train_student_end_to_end(
        student=student,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        seed=seed,
        device=device,
    )

    # 4. Save Checkpoints
    ckpts_dir = root / "experiments" / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Training checkpoint (includes metadata)
    train_ckpt_path = ckpts_dir / f"P03_student_mobilenetv3_small_plantvillage_s{seed}_full.pt"
    torch.save(
        {
            "experiment_id": f"P03_student_mobilenetv3_small_plantvillage_s{seed}",
            "architecture": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            "num_classes": num_classes,
            "seed": seed,
            "state_dict": student.state_dict(),
            "class_to_idx": class_to_idx,
            "train_summary": train_summary,
        },
        train_ckpt_path,
    )

    # Pure inference / deployment checkpoint (state_dict only for lean file size benchmark)
    inf_ckpt_path = ckpts_dir / f"P03_student_mobilenetv3_small_plantvillage_s{seed}.pt"
    torch.save(student.state_dict(), inf_ckpt_path)

    # Checksum of deployment checkpoint
    h = hashlib.sha256()
    with open(inf_ckpt_path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    ckpt_sha256 = h.hexdigest()

    file_size_mb = inf_ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Saved Student Checkpoint -> {inf_ckpt_path}")
    print(f"Deployment Checkpoint Size: {file_size_mb:.3f} MB (SHA-256: {ckpt_sha256})")

    return student, inf_ckpt_path, ckpt_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    run_student_pipeline(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size)
