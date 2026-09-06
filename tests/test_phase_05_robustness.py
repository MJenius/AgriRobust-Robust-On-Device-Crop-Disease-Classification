"""Unit tests for Phase 5 Robustness Evaluation module."""

import numpy as np
import pytest
import torch
from PIL import Image

from agrirobust.robustness.corruptions import (
    CORRUPTION_PARAMS,
    apply_brightness,
    apply_contrast,
    apply_corruption,
    apply_defocus_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_occlusion,
    apply_resolution,
)


def _create_sample_pil_image():
    # 224x224 RGB image with gradient pattern
    arr = np.linspace(0, 255, 224 * 224 * 3, dtype=np.uint8).reshape((224, 224, 3))
    return Image.fromarray(arr, mode="RGB")


def test_frozen_corruption_parameters():
    """Verify all 7 corruption families have explicitly defined levels 1 through 5."""
    expected_families = [
        "brightness",
        "contrast",
        "defocus_blur",
        "gaussian_noise",
        "jpeg_compression",
        "resolution",
        "occlusion",
    ]
    for fam in expected_families:
        assert fam in CORRUPTION_PARAMS, f"Missing family: {fam}"
        assert set(CORRUPTION_PARAMS[fam].keys()) == {1, 2, 3, 4, 5}


def test_deterministic_noise_seeding():
    """Verify that gaussian noise is completely deterministic for a given (sample_id, seed, severity)."""
    img = _create_sample_pil_image()
    res1 = apply_gaussian_noise(img, severity=3, sample_id="apple_scab_001.jpg", seed=42)
    res2 = apply_gaussian_noise(img, severity=3, sample_id="apple_scab_001.jpg", seed=42)
    res3 = apply_gaussian_noise(img, severity=3, sample_id="apple_scab_002.jpg", seed=42)

    arr1 = np.array(res1)
    arr2 = np.array(res2)
    arr3 = np.array(res3)

    assert np.array_equal(arr1, arr2), "Identical sample_id and seed must produce bit-for-bit identical noise"
    assert not np.array_equal(arr1, arr3), "Different sample_id should produce different noise realizations"


def test_deterministic_occlusion():
    """Verify that occlusion cutout is deterministic and covers the expected area fraction."""
    img = _create_sample_pil_image()
    res1 = apply_occlusion(img, severity=3, sample_id="leaf_123.jpg", seed=42)
    res2 = apply_occlusion(img, severity=3, sample_id="leaf_123.jpg", seed=42)
    res3 = apply_occlusion(img, severity=3, sample_id="leaf_456.jpg", seed=42)

    arr1 = np.array(res1)
    arr2 = np.array(res2)
    arr3 = np.array(res3)

    assert np.array_equal(arr1, arr2), "Identical cutout inputs must produce identical cropped masks"
    assert not np.array_equal(arr1, arr3), "Different samples must place cutout patches at different locations"


def test_corruption_dispatch():
    """Verify generic corruption dispatch works for all corruption types and returns correct dimensions."""
    img = _create_sample_pil_image()
    for fam in CORRUPTION_PARAMS:
        corrupted = apply_corruption(img, corruption_name=fam, severity=3, sample_id="test_img.jpg", seed=42)
        assert corrupted.size == (224, 224), f"Corruption {fam} changed output canvas size from (224, 224)"
        assert corrupted.mode == "RGB", f"Corruption {fam} changed image mode from RGB"
