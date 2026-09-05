"""PlantDoc feature extraction and evaluation pipeline."""

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agrirobust.config import get_project_root
from agrirobust.data.dataset import (
    build_plantdoc_dataloader,
    get_canonical_class_mapping,
)
from agrirobust.models.teacher import build_teacher_model


def extract_plantdoc_features(
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    root = get_project_root()
    cache_dir = root / "data" / "processed" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "plantdoc_crops_convnext_tiny_features.pt"

    if cache_file.exists():
        print(f"Loading cached PlantDoc features from {cache_file}...")
        data = torch.load(cache_file)
        return data["features"], data["labels"], data["class_names"]

    print("Extracting PlantDoc crop features using ConvNeXt-Tiny...")
    class_to_idx, _ = get_canonical_class_mapping()
    dataloader = build_plantdoc_dataloader(batch_size=batch_size, image_size=224, num_workers=0)

    teacher = build_teacher_model(num_classes=len(class_to_idx), pretrained=False)
    ckpt_path = root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    teacher.to(device)

    all_feats = []
    all_lbls = []
    all_names = []

    total_batches = len(dataloader)
    t0 = time.time()
    with torch.no_grad():
        for i, (inputs, targets, names) in enumerate(dataloader):
            inputs = inputs.to(device)
            feats = teacher.extract_features(inputs)
            all_feats.append(feats.cpu())
            all_lbls.append(targets)
            all_names.extend(names)
            if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - t0
                print(f"Batch [{i+1}/{total_batches}] | Elapsed: {elapsed:.1f}s")

    features_tensor = torch.cat(all_feats, dim=0)
    labels_tensor = torch.cat(all_lbls, dim=0)

    torch.save(
        {
            "features": features_tensor,
            "labels": labels_tensor,
            "class_names": all_names,
        },
        cache_file,
    )
    print(f"Saved PlantDoc features -> {cache_file} ({features_tensor.shape})")
    return features_tensor, labels_tensor, all_names


if __name__ == "__main__":
    extract_plantdoc_features()
