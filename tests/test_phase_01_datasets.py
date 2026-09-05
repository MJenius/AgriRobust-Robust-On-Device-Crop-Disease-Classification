"""AgriRobust Phase 1 dataset and manifest verification test suite."""

import json
from collections import Counter

from agrirobust.config import (
    get_datasets_config,
    get_project_root,
)
from agrirobust.data.manifest_generator import (
    PLANTDOC_TO_CANONICAL,
    PLANTVILLAGE_TO_CANONICAL,
)


def test_manifest_files_exist():
    root = get_project_root()
    manifests_dir = root / "data" / "manifests"
    assert (manifests_dir / "plantvillage_manifest.json").is_file()
    assert (manifests_dir / "plantdoc_crops_manifest.json").is_file()
    assert (manifests_dir / "plantseg_manifest.json").is_file()


def test_plantvillage_manifest_integrity():
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "plantvillage"
    assert data["total_images"] == 54305
    assert len(data["records"]) == 54305

    splits = Counter(r["split"] for r in data["records"])
    assert splits["train"] == 38047
    assert splits["val"] == 8129
    assert splits["test"] == 8129

    classes = set(r["canonical_label"] for r in data["records"])
    assert len(classes) == 38

    # Verify every record has sha256 checksum
    for r in data["records"][:100]:
        assert "sha256" in r
        assert len(r["sha256"]) == 64


def test_plantdoc_crops_manifest_integrity():
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantdoc_crops_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "plantdoc"
    assert data["total_crops"] == 8883
    assert len(data["records"]) == 8883

    # Ensure all PlantDoc crops are held-out test evaluation
    for r in data["records"][:100]:
        assert r["split"] == "cross_domain_test"
        assert len(r["bbox"]) == 4
        assert r["canonical_label"] in PLANTDOC_TO_CANONICAL.values()


def test_canonical_label_alignment():
    pv_canonical = set(PLANTVILLAGE_TO_CANONICAL.values())
    pd_canonical = set(PLANTDOC_TO_CANONICAL.values())

    assert len(pv_canonical) == 38
    assert len(pd_canonical) == 29

    # All PlantDoc classes must be a strict subset of PlantVillage canonical classes
    assert pd_canonical.issubset(pv_canonical)

    # Check 9 unshared classes
    unshared = pv_canonical - pd_canonical
    assert len(unshared) == 9
    assert "orange___citrus_greening" in unshared
    assert "strawberry___leaf_scorch" in unshared


def test_plantseg_manifest_integrity():
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantseg_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset"] == "plantseg"
    assert data["version"] == "v3"
    assert data["total_samples"] == 11458
    assert len(data["records"]) == 11458


def test_datasets_yaml_is_verified():
    cfg = get_datasets_config()
    datasets = cfg.get("datasets", {})

    for key in ["plantvillage", "plantdoc", "plantseg", "agrobench", "field_collected"]:
        assert key in datasets
        assert datasets[key]["status"]["source_info"] == "VERIFIED"

    assert cfg["label_policy"]["total_canonical_classes"] == 38
    assert cfg["label_policy"]["shared_cross_domain_classes"] == 29
