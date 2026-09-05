"""Training and fine-tuning pipeline for AgriRobust Phase 2 Teacher Baseline."""

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
from agrirobust.models.teacher import build_teacher_model
from agrirobust.training.feature_caching import CachedFeatureDataset, extract_and_cache_features


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_teacher_head_on_features(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int = 38,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 15,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(seed)
    train_loader = DataLoader(
        CachedFeatureDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CachedFeatureDataset(val_features, val_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    in_features = train_features.shape[1]
    head = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None
    history = []

    print(f"--> Training Teacher classification head for {epochs} epochs (seed {seed})...")
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        head.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats = feats.to(device)
                logits = head(feats)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(lbls.numpy().tolist())

        val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4%}")

        history.append({"epoch": epoch + 1, "train_loss": round(avg_loss, 4), "val_acc": round(float(val_acc), 4)})
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_state = {k: v.cpu() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head, {"best_val_acc": best_val_acc, "history": history}


def run_teacher_pipeline(seed: int = 42, epochs: int = 15):
    root = get_project_root()
    device = "cpu"
    set_seed(seed)

    print(f"==================================================")
    print(f"AgriRobust Phase 2 — Teacher Pipeline (Seed {seed})")
    print(f"==================================================")

    # 1. Instantiate Teacher
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    num_classes = len(class_to_idx)
    teacher = build_teacher_model(num_classes=num_classes, pretrained=True)

    # 2. Extract & Cache Features
    train_feats, train_labels = extract_and_cache_features(teacher, split="train", device=device)
    val_feats, val_labels = extract_and_cache_features(teacher, split="val", device=device)
    test_feats, test_labels = extract_and_cache_features(teacher, split="test", device=device)

    # 3. Train Head
    trained_head, training_res = train_teacher_head_on_features(
        train_features=train_feats,
        train_labels=train_labels,
        val_features=val_feats,
        val_labels=val_labels,
        num_classes=num_classes,
        epochs=epochs,
        seed=seed,
        device=device,
    )

    # 4. Integrate Head into Teacher Model
    teacher.classifier.weight.data.copy_(trained_head[1].weight.data)
    teacher.classifier.bias.data.copy_(trained_head[1].bias.data)

    # 5. Save Checkpoint
    ckpts_dir = root / "experiments" / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpts_dir / f"P02_teacher_convnext_tiny_plantvillage_s{seed}.pt"

    torch.save(
        {
            "experiment_id": f"P02_teacher_convnext_tiny_plantvillage_s{seed}",
            "architecture": "convnext_tiny",
            "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "num_classes": num_classes,
            "seed": seed,
            "state_dict": teacher.state_dict(),
            "class_to_idx": class_to_idx,
            "training_summary": training_res,
        },
        ckpt_path,
    )

    # Checksum
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    ckpt_sha256 = h.hexdigest()
    print(f"Saved Teacher Checkpoint -> {ckpt_path} (SHA-256: {ckpt_sha256})")

    return teacher, ckpt_path, ckpt_sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    run_teacher_pipeline(seed=args.seed, epochs=args.epochs)
