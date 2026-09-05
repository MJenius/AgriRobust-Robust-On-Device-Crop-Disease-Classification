"""Deterministic trainer for AgriRobust Phase 2 Teacher Baseline."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from agrirobust.evaluation.evaluator import evaluate_classifier


def get_checkpoint_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


class AgriTrainer:
    """Trainer managing fine-tuning, validation checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_dir: Path,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs
        self.seed = seed

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        self.best_val_macro_f1 = -1.0
        self.best_checkpoint_path = self.experiment_dir / "teacher_best.pt"

    def train_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        t0 = time.time()
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch_idx + 1}/{self.num_epochs}] "
                    f"Step [{batch_idx + 1}/{n_batches}] "
                    f"Loss: {loss.item():.4f} "
                    f"Elapsed: {elapsed:.1f}s"
                )

        self.scheduler.step()
        return total_loss / max(1, n_batches)

    def fit(self) -> Dict[str, Any]:
        history = []
        print(f"Starting training: {self.num_epochs} epochs on device: {self.device}")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = evaluate_classifier(self.model, self.val_loader, device=self.device)

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["macro_f1"]
            val_bal_acc = val_metrics["balanced_accuracy"]

            print(
                f"Epoch {epoch + 1} Summary -> Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f} | Val Bal Acc: {val_bal_acc:.4f}"
            )

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "val_balanced_accuracy": val_bal_acc,
            }
            history.append(epoch_record)

            # Checkpoint selection on Val Macro F1
            if val_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_f1
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_macro_f1": val_f1,
                        "val_accuracy": val_acc,
                        "seed": self.seed,
                    },
                    self.best_checkpoint_path,
                )
                print(f"--> Saved new best teacher checkpoint (Val Macro F1: {val_f1:.4f}) to {self.best_checkpoint_path}")

        # Compute checkpoint SHA-256
        ckpt_hash = get_checkpoint_sha256(self.best_checkpoint_path) if self.best_checkpoint_path.exists() else ""

        return {
            "best_val_macro_f1": round(self.best_val_macro_f1, 4),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "checkpoint_sha256": ckpt_hash,
            "history": history,
        }
