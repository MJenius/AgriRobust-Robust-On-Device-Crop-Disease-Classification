"""Teacher-Student Distillation Dataset supporting paired training inputs and cached teacher targets."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from agrirobust.config import get_project_root
from agrirobust.data.dataset import get_canonical_class_mapping


class DistillationDataset(Dataset):
    """Dataset pairing raw training images with precomputed teacher logits and features."""

    def __init__(
        self,
        manifest_path: Path,
        targets_file: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = get_project_root()
        self.transform = transform

        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = [r for r in data["records"] if r.get("split") == split]

        class_to_idx, _ = get_canonical_class_mapping()
        self.class_to_idx = class_to_idx

        # Load teacher targets
        t_data = torch.load(targets_file)
        self.teacher_logits = t_data["teacher_logits"]
        self.teacher_features = t_data["teacher_features"]
        assert len(self.records) == len(self.teacher_logits)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        img_p = self.root / rec["image_path"]

        with Image.open(img_p) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img_t = self.transform(img)
            else:
                img_t = transforms.ToTensor()(img)

        label_idx = self.class_to_idx[rec["canonical_label"]]
        t_logits = self.teacher_logits[idx]
        t_feat = self.teacher_features[idx]
        return img_t, t_logits, t_feat, label_idx
