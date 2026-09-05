"""Dataset and DataLoader loaders for AgriRobust using verified JSON manifests."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from agrirobust.config import get_project_root
from agrirobust.data.transforms import get_eval_transforms, get_train_transforms


class ManifestImageDataset(Dataset):
    """Dataset reading directly from verified JSON manifests."""

    def __init__(
        self,
        manifest_path: Path,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = get_project_root()
        self.manifest_path = manifest_path
        self.transform = transform

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        records = manifest_data.get("records", [])
        if split is not None:
            self.records = [r for r in records if r.get("split") == split]
        else:
            self.records = records

        # Derive or set class mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(list(set(r["canonical_label"] for r in records)))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img_rel = rec.get("image_path")
        img_path = self.root / img_rel

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)

        label_name = rec["canonical_label"]
        label_idx = self.class_to_idx.get(label_name, -1)
        return tensor, label_idx, label_name


class PlantDocCropsDataset(Dataset):
    """Dataset reading PlantDoc leaf crops on-the-fly using bounding boxes and manifest."""

    def __init__(
        self,
        manifest_path: Path,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = get_project_root()
        self.manifest_path = manifest_path
        self.transform = transform

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.records = manifest_data.get("records", [])

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_classes = sorted(list(set(r["canonical_label"] for r in self.records)))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self._image_cache = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img_rel = rec["source_image_path"]
        img_path = self.root / img_rel
        bbox = rec["bbox"]  # [xmin, ymin, xmax, ymax]

        # Read & crop
        with Image.open(img_path) as full_img:
            full_img = full_img.convert("RGB")
            crop_img = full_img.crop(bbox)

            if self.transform is not None:
                tensor = self.transform(crop_img)
            else:
                tensor = transforms.ToTensor()(crop_img)

        label_name = rec["canonical_label"]
        label_idx = self.class_to_idx.get(label_name, -1)
        return tensor, label_idx, label_name


def get_canonical_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load canonical class taxonomy dynamically from authoritative manifest."""
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    unique_classes = sorted(list(set(r["canonical_label"] for r in data["records"])))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_dataloaders(
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Construct deterministic DataLoaders for train, val, and clean test."""
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    class_to_idx, _ = get_canonical_class_mapping()

    train_ds = ManifestImageDataset(
        manifest_path=manifest_path,
        split="train",
        transform=get_train_transforms(image_size),
        class_to_idx=class_to_idx,
    )
    val_ds = ManifestImageDataset(
        manifest_path=manifest_path,
        split="val",
        transform=get_eval_transforms(image_size),
        class_to_idx=class_to_idx,
    )
    test_ds = ManifestImageDataset(
        manifest_path=manifest_path,
        split="test",
        transform=get_eval_transforms(image_size),
        class_to_idx=class_to_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }


def build_plantdoc_dataloader(
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 0,
) -> DataLoader:
    """Construct DataLoader for PlantDoc held-out cross-domain leaf crops."""
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantdoc_crops_manifest.json"
    class_to_idx, _ = get_canonical_class_mapping()

    dataset = PlantDocCropsDataset(
        manifest_path=manifest_path,
        transform=get_eval_transforms(image_size),
        class_to_idx=class_to_idx,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
