"""Deterministic and reproducible image corruptions for AgriRobust Phase 5.

Implements 7 perturbation families across 5 severity levels:
1. brightness: Scaling pixel intensities (sunlight variation)
2. contrast: Compressing/expanding dynamic range (haze / washed-out sensors)
3. defocus_blur: Gaussian optical blur (camera misfocus)
4. gaussian_noise: Additive Gaussian sensor noise (low-light / high-ISO sensors)
5. jpeg_compression: DCT compression artifacts (low-bandwidth upload / storage savings)
6. resolution: Downsampling and upsampling (low-end smartphone sensors)
7. occlusion: Deterministic rectangular cutout patches (foliage / hand occlusion)

All operations are applied to PIL images or float tensors in [0, 1] prior to ImageNet normalization.
"""

import hashlib
import io
import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter


# ==============================================================================
# Frozen Numerical Corruption Parameters Specification
# ==============================================================================
CORRUPTION_PARAMS: Dict[str, Dict[int, Any]] = {
    "brightness": {
        1: 1.15,
        2: 1.30,
        3: 1.50,
        4: 1.75,
        5: 2.00,
    },
    "contrast": {
        1: 0.85,
        2: 0.70,
        3: 0.55,
        4: 0.40,
        5: 0.25,
    },
    "defocus_blur": {
        1: 1.0,
        2: 2.0,
        3: 3.5,
        4: 5.0,
        5: 7.0,
    },
    "gaussian_noise": {
        1: 0.03,
        2: 0.06,
        3: 0.10,
        4: 0.15,
        5: 0.22,
    },
    "jpeg_compression": {
        1: 80,
        2: 60,
        3: 40,
        4: 25,
        5: 12,
    },
    "resolution": {
        1: (176, 176),
        2: (144, 144),
        3: (112, 112),
        4: (80, 80),
        5: (56, 56),
    },
    "occlusion": {
        1: 0.05,
        2: 0.10,
        3: 0.18,
        4: 0.28,
        5: 0.40,
    },
}


def apply_brightness(img: Image.Image, severity: int) -> Image.Image:
    """Adjust image brightness."""
    factor = CORRUPTION_PARAMS["brightness"][severity]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, severity: int) -> Image.Image:
    """Adjust image contrast."""
    factor = CORRUPTION_PARAMS["contrast"][severity]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_defocus_blur(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian optical blur simulating camera misfocus."""
    radius = CORRUPTION_PARAMS["defocus_blur"][severity]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img: Image.Image, severity: int, sample_id: str = "", seed: int = 42) -> Image.Image:
    """Add zero-mean Gaussian noise deterministically to simulate camera sensor noise."""
    sigma = CORRUPTION_PARAMS["gaussian_noise"][severity]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Deterministic noise seeding per (sample_id, severity, seed)
    token = f"{sample_id}_gn_{severity}_{seed}".encode("utf-8")
    noise_seed = int(hashlib.sha256(token).hexdigest()[:8], 16)
    rng = np.random.RandomState(noise_seed)

    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    corrupted = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((corrupted * 255.0).astype(np.uint8))


def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    """Simulate JPEG compression artifacts via in-memory buffer."""
    quality = CORRUPTION_PARAMS["jpeg_compression"][severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_resolution(img: Image.Image, severity: int) -> Image.Image:
    """Simulate low-resolution mobile sensors by downsampling and upsampling back to original size."""
    w, h = img.size
    down_size = CORRUPTION_PARAMS["resolution"][severity]
    down = img.resize(down_size, resample=Image.Resampling.BILINEAR)
    return down.resize((w, h), resample=Image.Resampling.BILINEAR)


def apply_occlusion(img: Image.Image, severity: int, sample_id: str = "", seed: int = 42) -> Image.Image:
    """Apply deterministic rectangular cutout simulating partial occlusion (foliage/hands)."""
    area_ratio = CORRUPTION_PARAMS["occlusion"][severity]
    w, h = img.size
    total_area = w * h
    target_area = total_area * area_ratio

    # Aspect ratio between 0.5 and 2.0 deterministically derived
    token = f"{sample_id}_occ_{severity}_{seed}".encode("utf-8")
    h_hex = hashlib.sha256(token).hexdigest()
    rng_aspect = (int(h_hex[:4], 16) / 65535.0) * 1.5 + 0.5  # [0.5, 2.0]
    
    cut_w = int(math.sqrt(target_area * rng_aspect))
    cut_h = int(math.sqrt(target_area / rng_aspect))

    cut_w = max(1, min(cut_w, w - 1))
    cut_h = max(1, min(cut_h, h - 1))

    # Deterministic top-left coordinates
    max_x = max(0, w - cut_w)
    max_y = max(0, h - cut_h)
    x = int((int(h_hex[4:8], 16) / 65535.0) * max_x)
    y = int((int(h_hex[8:12], 16) / 65535.0) * max_y)

    arr = np.array(img).copy()
    # Fill with neutral gray (128, 128, 128)
    arr[y : y + cut_h, x : x + cut_w, :] = 128
    return Image.fromarray(arr)


def apply_corruption(
    img: Image.Image,
    corruption_name: str,
    severity: int,
    sample_id: str = "",
    seed: int = 42,
) -> Image.Image:
    """Apply specified corruption and severity level deterministically."""
    if corruption_name == "clean":
        return img
    elif corruption_name == "brightness":
        return apply_brightness(img, severity)
    elif corruption_name == "contrast":
        return apply_contrast(img, severity)
    elif corruption_name == "defocus_blur":
        return apply_defocus_blur(img, severity)
    elif corruption_name == "gaussian_noise":
        return apply_gaussian_noise(img, severity, sample_id=sample_id, seed=seed)
    elif corruption_name == "jpeg_compression":
        return apply_jpeg_compression(img, severity)
    elif corruption_name == "resolution":
        return apply_resolution(img, severity)
    elif corruption_name == "occlusion":
        return apply_occlusion(img, severity, sample_id=sample_id, seed=seed)
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
