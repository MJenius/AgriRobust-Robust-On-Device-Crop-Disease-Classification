"""
Phase 9 — Final Validation, Reproducibility and Repository Freeze Tests.

This test module verifies:
1. Integrity of all frozen model checkpoints and cryptographic SHA-256 hashes.
2. Integrity and file size requirements of the deployed Android mobile assets.
3. Immutability of calibration constants (T=0.5406, tau=0.8143).
4. Machine-readable summary completeness in experiments/runs/P09_final_synthesis/summary.json.
5. Presence and honest framing of the 99.7% failure case in documentation.
"""

import hashlib
import json
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


EXPECTED_HASHES = {
    "teacher_s42": {
        "path": REPO_ROOT / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt",
        "sha256": "7b80b48404531597f098571e56d53f03676305b57a7938ff27a4389e8fa1d1af",
    },
    "student_baseline_s42": {
        "path": REPO_ROOT / "experiments" / "checkpoints" / "P03_student_mobilenetv3_small_plantvillage_s42.pt",
        "sha256": "6076d8a2b0f4d5d562b2ce380df91ceb2561b1105b13eeab08a60fe93cce2eee",
    },
    "student_response_kd_s42": {
        "path": REPO_ROOT / "experiments" / "checkpoints" / "P04_student_kd_response_plantvillage_s42.pt",
        "sha256": "2b935203522c1a58ce94963b251ae80b9a6669a2c1f1c74658a48e36195eb8c6",
    },
    "android_asset_model_int8": {
        "path": REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "model_int8_dynamic.pt",
        "sha256": "500ea8b7ed942f16ae59da60b5cab262aebb46f279b2d936bc8f7edf45617200",
    },
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_checkpoint_reproducibility_and_hashes():
    """Verify that all core training checkpoints exist and match their frozen SHA-256 hashes."""
    for model_name, info in EXPECTED_HASHES.items():
        assert info["path"].exists(), f"Missing checkpoint artifact: {info['path']}"
        actual_sha = compute_sha256(info["path"])
        assert (
            actual_sha == info["sha256"]
        ), f"Checksum mismatch for {model_name}. Expected {info['sha256']}, got {actual_sha}"


def test_android_deployment_assets_integrity():
    """Verify mobile deployment assets and metadata contracts."""
    assets_dir = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"
    labels_file = assets_dir / "labels.json"
    metadata_file = assets_dir / "deployment_metadata.json"
    model_file = assets_dir / "model_int8_dynamic.pt"

    assert labels_file.exists()
    assert metadata_file.exists()
    assert model_file.exists()

    # Verify model file size is <= 15 MB budget (and ~4.56 MB)
    size_mb = model_file.stat().st_size / (1024 * 1024)
    assert size_mb < 15.0, f"Model size {size_mb} MB exceeds 15 MB budget"
    assert 4.0 < size_mb < 5.0, f"Expected ~4.56 MB container, got {size_mb:.2f} MB"

    # Verify 38 classes
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
    assert len(labels_data["classes"]) == 38
    assert labels_data["num_classes"] == 38

    # Verify calibration parameters in metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["calibration"]["temperature"] == 0.5406
    assert meta["selective_abstention"]["validation_threshold_tau"] == 0.8143
    assert meta["calibration"]["source_phase"] == "Phase 6"


def test_phase_09_synthesis_summary_completeness():
    """Verify that the Phase 9 empirical summary contains all 4 comparison architectures."""
    summary_file = REPO_ROOT / "experiments" / "runs" / "P09_final_synthesis" / "summary.json"
    assert summary_file.exists()

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    empirical = summary["empirical_summary"]
    assert "teacher_convnext_tiny" in empirical
    assert "student_baseline" in empirical
    assert "student_response_kd" in empirical
    assert "deployed_int8_champion" in empirical

    # Verify key metrics exist
    teacher = empirical["teacher_convnext_tiny"]
    assert teacher["parameters"] == 27849350
    assert teacher["clean_test_acc"] == pytest.approx(0.9831, abs=0.001)

    kd = empirical["student_response_kd"]
    assert kd["clean_test_acc"] == pytest.approx(0.9930, abs=0.001)
    assert kd["plantdoc_cross_acc"] == pytest.approx(0.1839, abs=0.001)

    deployed = empirical["deployed_int8_champion"]
    assert deployed["parity_top1_agreement"] == 1.0
    assert deployed["parity_decision_agreement"] == 1.0
    assert deployed["latency_physical_phone_a14_ms"] == 45.40


def test_honest_failure_mode_documented():
    """Verify that data/examples/README.md and reports explicitly document the 99.7% failure mode."""
    examples_readme = REPO_ROOT / "data" / "examples" / "README.md"
    assert examples_readme.exists()

    content = examples_readme.read_text(encoding="utf-8")
    assert "03_plantdoc_potato_late_blight.jpg" in content
    assert "99.7%" in content
    assert "HIGH-CONFIDENCE FAILURE" in content or "high-confidence" in content.lower()
    assert "universal safety guarantee" in content
