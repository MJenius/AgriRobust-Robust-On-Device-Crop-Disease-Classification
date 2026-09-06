"""AgriRobust Phase 0 contract verification test suite."""

from agrirobust.config import (
    get_datasets_config,
    get_metrics_config,
    get_project_config,
    get_project_root,
)


def test_project_root_exists():
    root = get_project_root()
    assert root.is_dir()
    assert (root / "PROJECT_SOT.md").is_file()


def test_required_directories_exist():
    root = get_project_root()
    required_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/manifests",
        "src/agrirobust",
        "src/agrirobust/data",
        "src/agrirobust/models",
        "src/agrirobust/training",
        "src/agrirobust/distillation",
        "src/agrirobust/robustness",
        "src/agrirobust/uncertainty",
        "src/agrirobust/compression",
        "src/agrirobust/evaluation",
        "src/agrirobust/deployment",
        "experiments",
        "reports",
        "tests",
        "android",
    ]
    for rel_path in required_dirs:
        target_dir = root / rel_path
        assert target_dir.is_dir(), f"Expected directory does not exist: {rel_path}"


def test_required_root_files_exist():
    root = get_project_root()
    required_files = [
        "pyproject.toml",
        "CURRENT_PHASE.md",
        "README.md",
        "configs/project.yaml",
        "configs/datasets.yaml",
        "configs/metrics.yaml",
        "experiments/README.md",
        "reports/README.md",
        "reports/phase_00_decisions.md",
        "android/README.md",
    ]
    for rel_path in required_files:
        target_file = root / rel_path
        assert target_file.is_file(), f"Expected file does not exist: {rel_path}"


def test_current_phase_contract():
    root = get_project_root()
    phase_file = root / "CURRENT_PHASE.md"
    assert phase_file.is_file()
    content = phase_file.read_text(encoding="utf-8")

    assert "# Phase:" in content
    assert "## Objective" in content
    assert "Next Phase" in content


def test_yaml_configurations_parse_validly():
    project_cfg = get_project_config()
    datasets_cfg = get_datasets_config()
    metrics_cfg = get_metrics_config()

    assert isinstance(project_cfg, dict)
    assert isinstance(datasets_cfg, dict)
    assert isinstance(metrics_cfg, dict)


def test_dataset_registry_contains_sot_datasets():
    datasets_cfg = get_datasets_config()
    datasets = datasets_cfg.get("datasets", {})

    expected_sot_datasets = {
        "plantvillage",
        "plantdoc",
        "plantseg",
        "agrobench",
        "field_collected",
    }

    for ds_key in expected_sot_datasets:
        assert ds_key in datasets, f"Dataset '{ds_key}' missing from datasets.yaml registry"
        ds_entry = datasets[ds_key]
        assert "name" in ds_entry
        assert "purpose" in ds_entry
        assert "role" in ds_entry
        assert "task" in ds_entry
        assert "expected_modality" in ds_entry
        assert "status" in ds_entry
        assert "source_info" in ds_entry["status"]


def test_metrics_registry_contains_sot_metrics():
    metrics_cfg = get_metrics_config()
    metric_groups = metrics_cfg.get("metric_groups", {})

    expected_groups = [
        "classification",
        "calibration",
        "selective_prediction",
        "efficiency",
        "deployment",
        "segmentation",
    ]
    for group in expected_groups:
        assert group in metric_groups, f"Metric group '{group}' missing from metrics.yaml"

    cls_metrics = metric_groups["classification"]
    for m in ["macro_f1", "balanced_accuracy", "accuracy", "per_class_f1"]:
        assert m in cls_metrics, f"Classification metric '{m}' missing"
        assert "definition" in cls_metrics[m]
        assert "higher_is_better" in cls_metrics[m]

    cal_metrics = metric_groups["calibration"]
    assert "ece" in cal_metrics
    assert "brier_score" in cal_metrics

    sel_metrics = metric_groups["selective_prediction"]
    assert "coverage" in sel_metrics
    assert "risk_at_coverage" in sel_metrics

    eff_metrics = metric_groups["efficiency"]
    for m in [
        "parameter_count",
        "model_file_size",
        "peak_ram",
        "inference_latency",
        "throughput",
    ]:
        assert m in eff_metrics, f"Efficiency metric '{m}' missing"

    dep_metrics = metric_groups["deployment"]
    for m in [
        "preprocessing_latency",
        "end_to_end_phone_latency",
        "on_device_memory",
        "offline_operation",
    ]:
        assert m in dep_metrics, f"Deployment metric '{m}' missing"

    seg_metrics = metric_groups["segmentation"]
    assert "dice" in seg_metrics
    assert "iou" in seg_metrics


def test_project_reproducibility_policy():
    cfg = get_project_config()
    repro = cfg.get("reproducibility", {})
    assert repro.get("primary_seed") == 42
    assert isinstance(repro.get("evaluation_seeds"), list)
    assert len(repro.get("evaluation_seeds")) >= 3
    assert repro.get("deterministic_torch") is True


def test_no_premature_phase_implementations():
    """Verify phase governance and deployment artifacts: in Phase 8, Android deployment project is authorized and valid."""
    root = get_project_root()
    # In Phase 8, Android project is fully authorized and required
    android_builds = list((root / "android").glob("**/*.gradle*"))
    assert len(android_builds) > 0, "Expected Phase 8 Android Gradle configuration files to exist"
