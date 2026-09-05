"""Feature extractor / caching utility for fast linear-probe + fine-tuning of Teacher."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agrirobust.config import get_project_root
from agrirobust.data.dataset import ManifestImageDataset, get_canonical_class_mapping
from agrirobust.data.transforms import get_eval_transforms
from agrirobust.models.teacher import build_teacher_model


class CachedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def extract_and_cache_features(
    teacher: nn.Module,
    split: str,
    batch_size: int = 128,
    device: str = "cpu",
    cache_dir: Path = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    class_to_idx, _ = get_canonical_class_mapping()

    if cache_dir is None:
        cache_dir = root / "data" / "processed" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"plantvillage_{split}_convnext_tiny_features.pt"
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}...")
        data = torch.load(cache_file)
        return data["features"], data["labels"]

    print(f"Extracting features for PlantVillage split '{split}' using ConvNeXt-Tiny...")
    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        split=split,
        transform=get_eval_transforms(224),
        class_to_idx=class_to_idx,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    teacher.eval()
    teacher.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            feats = teacher.extract_features(inputs)
            all_features.append(feats.cpu())
            all_labels.append(targets)

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    torch.save({"features": features_tensor, "labels": labels_tensor}, cache_file)
    print(f"Saved {features_tensor.shape[0]} feature vectors (dim {features_tensor.shape[1]}) to {cache_file}")
    return features_tensor, labels_tensor
