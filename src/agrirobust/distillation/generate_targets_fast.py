"""Build offline teacher targets directly from cached features in seconds."""

from pathlib import Path
import torch
from agrirobust.config import get_project_root


def build_teacher_targets_from_features():
    root = get_project_root()
    feat_dir = root / "data" / "processed" / "features"
    out_dir = root / "data" / "processed" / "teacher_targets"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w = ckpt["state_dict"]["classifier.weight"]
    b = ckpt["state_dict"]["classifier.bias"]

    for split in ["train", "val"]:
        feat_path = feat_dir / f"plantvillage_{split}_convnext_tiny_features.pt"
        data = torch.load(feat_path, map_location="cpu")
        feats = data["features"]
        targets = data["labels"]

        # Forward pass through linear head: z = feats @ W^T + b
        logits = feats @ w.T + b

        out_file = out_dir / f"plantvillage_{split}_teacher_targets.pt"
        torch.save(
            {
                "teacher_logits": logits,
                "teacher_features": feats,
                "targets": targets,
            },
            out_file,
        )
        print(f"Generated {split} teacher targets -> {out_file} (logits: {logits.shape}, feats: {feats.shape})")


if __name__ == "__main__":
    build_teacher_targets_from_features()
