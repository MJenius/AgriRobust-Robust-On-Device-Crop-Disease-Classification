"""Android preprocessing specification and parity verification for AgriRobust Phase 8.

Ensures exact mathematical match between PyTorch torchvision eval transforms
and mobile image preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# Authoritative preprocessing contract for Android deployment
ANDROID_PREPROCESSING_SPEC: Dict[str, Any] = {
    "target_width": 224,
    "target_height": 224,
    "resize_mode": "direct_resize_224x224",
    "color_space": "RGB",
    "pixel_scaling": 1.0 / 255.0,
    "tensor_layout": "CHW",
    "channel_order": "RGB",
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "interpolation": "BILINEAR",
}


def preprocess_image_for_deployment(
    image: Union[Image.Image, np.ndarray],
    target_size: int = 224,
) -> torch.Tensor:
    """Deterministic preprocessing matching torchvision get_eval_transforms:

    1. Convert image to RGB.
    2. Resize directly to (target_size, target_size) using bilinear interpolation.
    3. Convert to float32 tensor scaled to [0.0, 1.0].
    4. Normalize with ImageNet mean and std.

    Args:
        image: PIL Image or RGB numpy array (H, W, 3).
        target_size: Output spatial resolution (default: 224).

    Returns:
        Tensor of shape [1, 3, target_size, target_size].
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize directly to (target_size, target_size) matching get_eval_transforms
    image_resized = F.resize(image, (target_size, target_size), interpolation=F.InterpolationMode.BILINEAR)

    # Convert to float tensor [0, 1]
    tensor = F.to_tensor(image_resized)

    # Normalize with canonical ImageNet stats
    mean = ANDROID_PREPROCESSING_SPEC["normalization"]["mean"]
    std = ANDROID_PREPROCESSING_SPEC["normalization"]["std"]
    tensor_norm = F.normalize(tensor, mean=mean, std=std)

    # Add batch dimension [1, 3, 224, 224]
    return tensor_norm.unsqueeze(0)


def simulate_android_bitmap_preprocessing(
    image: Image.Image,
    target_size: int = 224,
) -> np.ndarray:
    """Simulate Android Bitmap to FloatBuffer normalization:

    Matches Android:
    Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
    TensorImageUtils.bitmapToFloat32Tensor(resized, mean, std);
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Android Bitmap.createScaledBitmap bilinear resize
    resized = image.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # [224, 224, 3]

    mean = np.array(ANDROID_PREPROCESSING_SPEC["normalization"]["mean"], dtype=np.float32)
    std = np.array(ANDROID_PREPROCESSING_SPEC["normalization"]["std"], dtype=np.float32)

    norm_chw = np.transpose((arr - mean) / std, (2, 0, 1))  # [3, 224, 224]
    return np.expand_dims(norm_chw, axis=0)  # [1, 3, 224, 224]


def verify_preprocessing_parity(
    image: Image.Image,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Verify numerical parity between Python torchvision transforms and Android Bitmap pipeline.

    Returns:
        Dictionary with max_absolute_difference, mean_absolute_difference, and parity_pass.
    """
    py_tensor = preprocess_image_for_deployment(image).cpu().numpy()
    android_sim = simulate_android_bitmap_preprocessing(image)

    max_diff = float(np.max(np.abs(py_tensor - android_sim)))
    mean_diff = float(np.mean(np.abs(py_tensor - android_sim)))

    return {
        "max_absolute_difference": round(max_diff, 6),
        "mean_absolute_difference": round(mean_diff, 6),
        "tolerance": tolerance,
        "parity_pass": bool(max_diff <= tolerance),
    }
