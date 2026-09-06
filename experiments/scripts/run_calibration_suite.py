"""Runner script for AgriRobust Phase 6: Uncertainty Calibration and Selective Abstention.

Models evaluated:
1. Primary: Phase 4 Response-KD Champion (MobileNetV3-Small)
2. Reference: Phase 3 Student Baseline (MobileNetV3-Small)
3. Reference: Phase 2 Teacher (ConvNeXt-Tiny)

Evaluation Domains:
1. PlantVillage Clean Test (8,129 images)
2. PlantDoc Field Crops Cross-Domain (8,883 crops)
3. Gaussian Noise Severity 5 (sigma=0.22)
4. Defocus Blur Severity 5 (sigma=7.0)
5. Contrast Severity 5 (factor=0.25)

Protocols:
- Temperature scaling is learned exclusively on PlantVillage validation split.
- Learned temperature is frozen and applied to test and shifted domains.
- A single validation confidence threshold tau_val is selected on validation data (target risk <= 2%)
  and transferred unchanged to all test conditions.
- Canonical AURC is computed via empirical sample sort.
- Master metrics exported to experiments/runs/P06_calibration_abstention/metrics.json.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agrirobust.calibration.metrics import compute_calibration_metrics
from agrirobust.calibration.selective import (
    compute_canonical_aurc,
    evaluate_at_frozen_threshold,
    evaluate_selective_grid,
    find_operating_point_by_coverage,
    select_validation_threshold,
)
from agrirobust.calibration.temperature import TemperatureScaler
from agrirobust.config import get_project_root
from agrirobust.data.dataset import (
    ManifestImageDataset,
    build_plantdoc_dataloader,
    get_canonical_class_mapping,
)
from agrirobust.data.transforms import get_eval_transforms
from agrirobust.models.student import build_student_model
from agrirobust.models.teacher import build_teacher_model
from agrirobust.robustness.benchmark import CorruptedDataset


def load_model(model_name: str, ckpt_path: Path, device: str = "cpu") -> nn.Module:
    """Load model architecture and weights deterministically."""
    if "teacher" in model_name:
        model = build_teacher_model(num_classes=38, pretrained=True)
        ckpt_obj = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
        model.classifier.weight.data.copy_(state_dict["classifier.weight"])
        model.classifier.bias.data.copy_(state_dict["classifier.bias"])
        if "norm.weight" in state_dict and "norm.weight" in model.norm.state_dict():
            model.norm.weight.data.copy_(state_dict["norm.weight"])
            model.norm.bias.data.copy_(state_dict["norm.bias"])
    else:
        model = build_student_model(num_classes=38, pretrained=False)
        ckpt_obj = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)
    return model


def extract_logits_and_targets(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference to collect all unnormalized logits and ground truth targets."""
    model.eval()
    model.to(device)
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            targets = batch[1]
            logits = model(images)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets))

    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return logits_arr, targets_arr


def run_phase_06_calibration_suite(
    device: str = "cpu",
    batch_size: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    root = get_project_root()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    eval_transform = get_eval_transforms(image_size=224)

    # Models to evaluate
    models_config = {
        "response_kd": {
            "name": "Response-KD MobileNetV3-Small (Champion)",
            "path": root / "experiments" / "checkpoints" / "P04_student_kd_response_plantvillage_s42.pt",
            "is_primary": True,
        },
        "student_baseline": {
            "name": "MobileNetV3-Small Baseline",
            "path": root / "experiments" / "checkpoints" / "P03_student_mobilenetv3_small_plantvillage_s42.pt",
            "is_primary": False,
        },
        "teacher": {
            "name": "ConvNeXt-Tiny Teacher",
            "path": root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt",
            "is_primary": False,
        },
    }

    print("=======================================================")
    print("PHASE 6: UNCERTAINTY CALIBRATION & SELECTIVE ABSTENTION")
    print("Loading evaluation models...")
    loaded_models = {}
    for m_key, cfg in models_config.items():
        print(f"  Loading [{m_key}]: {cfg['name']}")
        loaded_models[m_key] = load_model(m_key, cfg["path"], device=device)
    print("All models loaded successfully.\n")

    # 1. Validation Split Data (STRICT: Calibration fitting only)
    print("Building PlantVillage Validation DataLoader (for post-hoc temperature fitting)...")
    val_ds = ManifestImageDataset(
        manifest_path=manifest_path,
        split="val",
        transform=eval_transform,
        class_to_idx=class_to_idx,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Validation samples: {len(val_ds)}")

    # Extract validation logits and fit temperature scalers
    temp_scalers: Dict[str, TemperatureScaler] = {}
    val_metrics: Dict[str, Any] = {}
    frozen_thresholds: Dict[str, float] = {}

    print("\n-------------------------------------------------------")
    print("STEP 1: Validation Temperature Scaling & Threshold Selection")
    print("-------------------------------------------------------")

    for m_key, model in loaded_models.items():
        t0 = time.time()
        val_logits, val_targets = extract_logits_and_targets(model, val_loader, device=device)
        elapsed = time.time() - t0

        # Uncalibrated validation metrics
        val_probs_uncal = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
        val_probs_uncal /= np.sum(val_probs_uncal, axis=1, keepdims=True)
        uncal_val_m = compute_calibration_metrics(val_probs_uncal, val_targets)

        # Fit temperature scaler
        scaler = TemperatureScaler().fit(val_logits, val_targets)
        temp_scalers[m_key] = scaler

        # Calibrated validation metrics
        val_probs_cal = scaler.predict_probabilities(val_logits)
        cal_val_m = compute_calibration_metrics(val_probs_cal, val_targets)

        # Select validation operating threshold for target risk <= 2% (or closest available)
        val_confs = np.max(val_probs_cal, axis=1)
        val_preds = np.argmax(val_probs_cal, axis=1)
        tau_val = select_validation_threshold(val_confs, val_preds, val_targets, max_acceptable_risk=0.02)
        frozen_thresholds[m_key] = tau_val

        val_metrics[m_key] = {
            "temperature": scaler.temperature,
            "fit_time_seconds": round(elapsed, 2),
            "uncalibrated": uncal_val_m,
            "calibrated": cal_val_m,
            "selected_tau_val": tau_val,
        }

        print(f"  [{m_key:<17}] Learned T={scaler.temperature:.4f} (Fit time: {elapsed:.1f}s)")
        print(f"      ECE: {uncal_val_m['expected_calibration_error']:.4f} -> {cal_val_m['expected_calibration_error']:.4f} | "
              f"NLL: {uncal_val_m['negative_log_likelihood']:.4f} -> {cal_val_m['negative_log_likelihood']:.4f} | "
              f"Selected tau_val: {tau_val:.4f}")

    # 2. Evaluation Datasets Definition
    # Clean Test + PlantDoc + 3 Priority Phase 5 Shifts
    eval_domains = {
        "clean_test": {
            "name": "PlantVillage Clean Test",
            "loader": DataLoader(
                ManifestImageDataset(manifest_path, split="test", transform=eval_transform, class_to_idx=class_to_idx),
                batch_size=batch_size, shuffle=False, num_workers=0
            ),
        },
        "plantdoc_cross_domain": {
            "name": "PlantDoc Field Crops (Natural Domain Shift)",
            "loader": build_plantdoc_dataloader(batch_size=batch_size, image_size=224, num_workers=0),
        },
        "gaussian_noise_sev5": {
            "name": "Gaussian Noise Severity 5 (sigma=0.22)",
            "loader": DataLoader(
                CorruptedDataset(manifest_path, split="test", corruption_name="gaussian_noise", severity=5, class_to_idx=class_to_idx, seed=seed),
                batch_size=batch_size, shuffle=False, num_workers=0
            ),
        },
        "defocus_blur_sev5": {
            "name": "Defocus Blur Severity 5 (sigma=7.0)",
            "loader": DataLoader(
                CorruptedDataset(manifest_path, split="test", corruption_name="defocus_blur", severity=5, class_to_idx=class_to_idx, seed=seed),
                batch_size=batch_size, shuffle=False, num_workers=0
            ),
        },
        "contrast_sev5": {
            "name": "Contrast Severity 5 (factor=0.25 - Champion KD Robustness)",
            "loader": DataLoader(
                CorruptedDataset(manifest_path, split="test", corruption_name="contrast", severity=5, class_to_idx=class_to_idx, seed=seed),
                batch_size=batch_size, shuffle=False, num_workers=0
            ),
        },
    }

    # Master results dict
    results: Dict[str, Any] = {
        "phase": 6,
        "status": "COMPLETED",
        "validation_tuning": val_metrics,
        "domains": {},
    }

    print("\n-------------------------------------------------------")
    print("STEP 2: Evaluation Across Clean & Shift Domains")
    print("-------------------------------------------------------")

    for dom_key, dom_info in eval_domains.items():
        dom_name = dom_info["name"]
        dom_loader = dom_info["loader"]
        print(f"\n--> Evaluating Domain: {dom_name.upper()} ({len(dom_loader.dataset)} samples)")
        results["domains"][dom_key] = {
            "name": dom_name,
            "sample_count": len(dom_loader.dataset),
            "models": {},
        }

        for m_key, model in loaded_models.items():
            t0 = time.time()
            logits, targets = extract_logits_and_targets(model, dom_loader, device=device)
            inf_time = time.time() - t0

            # Raw uncalibrated probabilities
            max_l = np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits - max_l)
            probs_uncal = exp_l / np.sum(exp_l, axis=1, keepdims=True)

            # Calibrated probabilities using frozen validation temperature
            scaler = temp_scalers[m_key]
            probs_cal = scaler.predict_probabilities(logits)

            # Compute calibration metrics
            cal_metrics_uncal = compute_calibration_metrics(probs_uncal, targets)
            cal_metrics_cal = compute_calibration_metrics(probs_cal, targets)

            # Selective prediction analysis using calibrated probabilities
            cal_confs = np.max(probs_cal, axis=1)
            cal_preds = np.argmax(probs_cal, axis=1)

            # Canonical AURC via empirical sample sort
            aurc, cum_cov, cum_risk = compute_canonical_aurc(cal_confs, cal_preds, targets)

            # Dense threshold grid
            sel_grid = evaluate_selective_grid(cal_confs, cal_preds, targets)

            # Fixed coverage operating points
            op_95 = find_operating_point_by_coverage(sel_grid, 0.95)
            op_90 = find_operating_point_by_coverage(sel_grid, 0.90)
            op_80 = find_operating_point_by_coverage(sel_grid, 0.80)

            # Frozen validation threshold evaluation
            tau_val = frozen_thresholds[m_key]
            op_frozen_val = evaluate_at_frozen_threshold(cal_confs, cal_preds, targets, tau_val)

            results["domains"][dom_key]["models"][m_key] = {
                "inference_time_seconds": round(inf_time, 2),
                "uncalibrated": cal_metrics_uncal,
                "calibrated": cal_metrics_cal,
                "temperature_applied": scaler.temperature,
                "selective_prediction": {
                    "canonical_aurc": aurc,
                    "operating_points": {
                        "coverage_95_pct": op_95,
                        "coverage_90_pct": op_90,
                        "coverage_80_pct": op_80,
                        "frozen_val_threshold": op_frozen_val,
                    },
                    "threshold_grid_summary": [
                        p for p in sel_grid if p["threshold"] in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
                    ],
                },
            }

            print(f"      {m_key:<17}: Acc={cal_metrics_cal['accuracy']:.2%} | "
                  f"ECE: {cal_metrics_uncal['expected_calibration_error']:.4f}->{cal_metrics_cal['expected_calibration_error']:.4f} | "
                  f"AURC={aurc:.4f} | "
                  f"Err-AUROC={cal_metrics_cal['error_detection']['auroc_error']} ({inf_time:.1f}s)")
            if op_frozen_val.get("coverage") is not None:
                print(f"        [Frozen Val tau={tau_val:.2f}]: Cov={op_frozen_val['coverage']:.1%} | Risk={op_frozen_val['selective_risk']:.2%}")

    # Output master metrics.json
    out_dir = root / "experiments" / "runs" / "P06_calibration_abstention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)

    print("\n=======================================================")
    print(f"PHASE 6 CALIBRATION & ABSTENTION COMPLETE! Saved to:\n  {out_path}")
    print("=======================================================")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgriRobust Phase 6 Calibration Suite")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_phase_06_calibration_suite(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
