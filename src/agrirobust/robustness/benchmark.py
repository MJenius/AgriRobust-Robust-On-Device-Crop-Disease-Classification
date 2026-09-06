"""Dataset and evaluation harness for robustness benchmarks across corruptions and domain shifts."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from agrirobust.config import get_project_root
from agrirobust.data.dataset import get_canonical_class_mapping
from agrirobust.robustness.corruptions import CORRUPTION_PARAMS, apply_corruption


class CorruptedDataset(Dataset):
    """Dataset applying a specific corruption family and severity level to raw images before normalization."""

    def __init__(
        self,
        manifest_path: Path,
        split: str = "test",
        corruption_name: str = "clean",
        severity: int = 1,
        image_size: int = 224,
        class_to_idx: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        self.root = get_project_root()
        self.corruption_name = corruption_name
        self.severity = severity
        self.image_size = image_size
        self.seed = seed

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        records = manifest_data.get("records", [])
        if split is not None:
            self.records = [r for r in records if r.get("split") == split]
        else:
            self.records = records

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(list(set(r["canonical_label"] for r in records)))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Standard post-corruption normalization
        self.to_tensor_and_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img_rel = rec.get("image_path")
        sample_id = rec.get("filename", f"sample_{idx}")
        img_path = self.root / img_rel

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            # First resize to standard input canvas
            img = img.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)

            # Apply deterministic corruption
            if self.corruption_name != "clean":
                img = apply_corruption(
                    img,
                    corruption_name=self.corruption_name,
                    severity=self.severity,
                    sample_id=sample_id,
                    seed=self.seed,
                )

            # Convert to tensor and apply standard ImageNet normalization
            tensor = self.to_tensor_and_norm(img)

        label_name = rec["canonical_label"]
        label_idx = self.class_to_idx[label_name]
        return tensor, label_idx, label_name


def evaluate_model_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    idx_to_class: Optional[Dict[int, str]] = None,
    restrict_to_target_classes: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on a dataloader and compute classification metrics."""
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

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
        (idx_to_class.get(lbl, str(lbl)) if idx_to_class else str(lbl)): round(float(f1), 4)
        for lbl, f1 in zip(labels, per_class_f1_raw)
    }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "total_samples": len(y_true),
        "per_class_f1": per_class_f1,
    }
