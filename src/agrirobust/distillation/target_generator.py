"""Offline Teacher Target Generator for Fast, Highly Reproducible Distillation.

Precomputes and caches the exact frozen teacher logits and 768-D pooled features
for PlantVillage train and val splits. This eliminates the duplicate forward pass
through the 27.85M parameter ConvNeXt-Tiny teacher during student training, reducing
per-epoch distillation time by more than 50% while guaranteeing mathematical equivalence.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from agrirobust.config import get_project_root
from agrirobust.data.dataset import ManifestImageDataset, get_canonical_class_mapping
from agrirobust.data.transforms import get_eval_transforms
from agrirobust.models.teacher import build_teacher_model


def precompute_teacher_targets(
    split: str = "train",
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute frozen teacher logits and features for a split, saving them to disk."""
    root = get_project_root()
    cache_dir = root / "data" / "processed" / "teacher_targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"plantvillage_{split}_teacher_targets.pt"

    if cache_file.exists():
        print(f"Loading cached teacher targets from {cache_file}...", flush=True)
        data = torch.load(cache_file)
        return data["teacher_logits"], data["teacher_features"], data["targets"]

    print(f"Generating teacher targets for split '{split}' with frozen ConvNeXt-Tiny...", flush=True)
    class_to_idx, _ = get_canonical_class_mapping()
    num_classes = len(class_to_idx)

    # 1. Load frozen Teacher Checkpoint
    ckpt_path = root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt"
    teacher = build_teacher_model(num_classes=num_classes, pretrained=False)
    teacher.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    teacher.eval()
    teacher.to(device)

    # 2. Dataset
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        split=split,
        transform=get_eval_transforms(224),
        class_to_idx=class_to_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits = []
    all_features = []
    all_targets = []

    total_batches = len(dataloader)
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            logits, feats = teacher(inputs, return_features=True)
            all_logits.append(logits.cpu())
            all_features.append(feats.cpu())
            all_targets.append(targets)

            if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                print(f"  Target generation [{i+1}/{total_batches}] batches...", flush=True)

    logits_tensor = torch.cat(all_logits, dim=0)
    features_tensor = torch.cat(all_features, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    torch.save(
        {
            "teacher_logits": logits_tensor,
            "teacher_features": features_tensor,
            "targets": targets_tensor,
        },
        cache_file,
    )
    print(f"Saved teacher targets -> {cache_file} ({logits_tensor.shape})", flush=True)
    return logits_tensor, features_tensor, targets_tensor


if __name__ == "__main__":
    precompute_teacher_targets("train")
    precompute_teacher_targets("val")
