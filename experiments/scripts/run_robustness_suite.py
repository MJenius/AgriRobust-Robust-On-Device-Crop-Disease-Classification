"""Comprehensive execution script for AgriRobust Phase 5 Robustness Evaluation.

Evaluates 3 primary models (and 2 reference models):
1. Teacher: ConvNeXt-Tiny (P02)
2. Student Baseline: MobileNetV3-Small (P03)
3. Response-KD Student: MobileNetV3-Small (P04 Champion)
(Reference: Feature-KD & Combined-KD)

Under:
- 7 synthetic corruption families x 5 frozen severity levels on PlantVillage test set (8,129 images)
- Natural domain shift on PlantDoc leaf crops (8,883 crops)
- Generates Relative Corruption Error (RCE) and mean RCE (mRCE)
- Produces master metrics.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from agrirobust.config import get_project_root
from agrirobust.data.dataset import build_plantdoc_dataloader, get_canonical_class_mapping
from agrirobust.models.student import build_student_model
from agrirobust.models.teacher import build_teacher_model
from agrirobust.robustness.benchmark import CorruptedDataset, evaluate_model_on_dataset
from agrirobust.robustness.corruptions import CORRUPTION_PARAMS


def load_model(model_name: str, ckpt_path: Path, device: str = "cpu") -> nn.Module:
    """Load model architecture and weights deterministically."""
    if "teacher" in model_name:
        # Phase 2 trained the classifier head on top of frozen ImageNet-pretrained ConvNeXt features.
        model = build_teacher_model(num_classes=38, pretrained=True)
        ckpt_obj = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj
        # Copy classifier weights
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


def run_phase_05_robustness_suite(
    device: str = "cpu",
    batch_size: int = 128,
    seed: int = 42,
    run_reference_ablations: bool = True,
) -> Dict[str, Any]:
    root = get_project_root()
    class_to_idx, idx_to_class = get_canonical_class_mapping()
    manifest_path = root / "data" / "manifests" / "plantvillage_manifest.json"

    # Define model checkpoints
    models_config = {
        "teacher": {
            "name": "ConvNeXt-Tiny Teacher",
            "role": "teacher",
            "checkpoint": root / "experiments" / "checkpoints" / "P02_teacher_convnext_tiny_plantvillage_s42.pt",
        },
        "student_baseline": {
            "name": "MobileNetV3-Small Baseline",
            "role": "student_baseline",
            "checkpoint": root / "experiments" / "checkpoints" / "P03_student_mobilenetv3_small_plantvillage_s42.pt",
        },
        "response_kd": {
            "name": "Response-KD MobileNetV3-Small",
            "role": "distilled_student_champion",
            "checkpoint": root / "experiments" / "checkpoints" / "P04_student_kd_response_plantvillage_s42.pt",
        },
    }

    if run_reference_ablations:
        models_config["feature_kd"] = {
            "name": "Feature-KD MobileNetV3-Small",
            "role": "distilled_student_feature",
            "checkpoint": root / "experiments" / "checkpoints" / "P04_student_kd_feature_plantvillage_s42.pt",
        }
        models_config["combined_kd"] = {
            "name": "Combined-KD MobileNetV3-Small",
            "role": "distilled_student_combined",
            "checkpoint": root / "experiments" / "checkpoints" / "P04_student_kd_combined_plantvillage_s42.pt",
        }

    # Load all models into memory once
    print("\n=======================================================", flush=True)
    print("PHASE 5: ROBUSTNESS EVALUATION SUITE", flush=True)
    print("Loading models into memory...", flush=True)
    loaded_models = {}
    for key, cfg in models_config.items():
        print(f"  Loading [{key}]: {cfg['name']} from {cfg['checkpoint'].name}", flush=True)
        loaded_models[key] = load_model(key, cfg["checkpoint"], device=device)
    print("All models loaded successfully.\n", flush=True)

    corruption_families = [
        "brightness",
        "contrast",
        "defocus_blur",
        "gaussian_noise",
        "jpeg_compression",
        "resolution",
        "occlusion",
    ]

    results: Dict[str, Any] = {m_key: {"clean": {}, "corruptions": {}, "plantdoc_cross_domain": {}} for m_key in models_config}

    # 1. Clean Benchmark Pass (Clean Test Set: 8,129 images)
    print("-------------------------------------------------------", flush=True)
    print("STEP 1: Clean Domain Evaluation (PlantVillage Test Split)", flush=True)
    print("-------------------------------------------------------", flush=True)

    for m_key, model in loaded_models.items():
        t0 = time.time()
        clean_ds = CorruptedDataset(
            manifest_path=manifest_path,
            split="test",
            corruption_name="clean",
            class_to_idx=class_to_idx,
            seed=seed,
        )
        clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        res = evaluate_model_on_dataset(model, clean_loader, device=device, idx_to_class=idx_to_class)
        elapsed = time.time() - t0
        res["elapsed_s"] = round(elapsed, 2)
        results[m_key]["clean"] = res
        print(f"  [{m_key.upper()} Clean] Acc: {res['accuracy']:.4%} | Macro F1: {res['macro_f1']:.4f} | Bal Acc: {res['balanced_accuracy']:.4f} ({elapsed:.1f}s)", flush=True)

    # 2. Synthetic Corruption Benchmarks (7 families x 5 severities = 35 conditions)
    print("\n-------------------------------------------------------", flush=True)
    print("STEP 2: Synthetic Corruption Benchmark (7 families x 5 severities)", flush=True)
    print("-------------------------------------------------------", flush=True)

    total_conditions = len(corruption_families) * 5
    condition_idx = 0

    # Check for existing checkpoint to allow seamless resumption if needed
    intermediate_out = root / "experiments" / "runs" / "P05_robustness_evaluation" / "metrics_checkpoint.json"
    if intermediate_out.exists():
        try:
            with open(intermediate_out, "r", encoding="utf-8") as f_inc:
                saved_ckpt = json.load(f_inc)
                if "results" in saved_ckpt:
                    results = saved_ckpt["results"]
                    print(f"  [Resume] Loaded prior progress from {intermediate_out.name} (last: {saved_ckpt.get('last_completed_condition')})", flush=True)
        except Exception as e:
            print(f"  [Resume Warning] Could not load prior checkpoint: {e}", flush=True)

    for c_name in corruption_families:
        for m_key in models_config:
            if c_name not in results[m_key]["corruptions"]:
                results[m_key]["corruptions"][c_name] = {}

        for sev in range(1, 6):
            condition_idx += 1
            # Skip if already evaluated in prior run
            if all(str(sev) in results[m_key]["corruptions"].get(c_name, {}) or sev in results[m_key]["corruptions"].get(c_name, {}) for m_key in models_config):
                print(f"  [Skipping already computed] Condition {c_name} sev {sev}", flush=True)
                continue
            t_cond_start = time.time()
            param_val = CORRUPTION_PARAMS[c_name][sev]
            print(f"\n--> [{condition_idx}/{total_conditions}] Corruption: {c_name.upper()} | Severity: {sev}/5 (Param: {param_val})", flush=True)

            corr_ds = CorruptedDataset(
                manifest_path=manifest_path,
                split="test",
                corruption_name=c_name,
                severity=sev,
                class_to_idx=class_to_idx,
                seed=seed,
            )
            corr_loader = DataLoader(corr_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            # Accumulate predictions model-by-model in a single streaming pass per model
            # Note: seed is identical, filenames are identical, so deterministic corruptions are 100% identical across models.
            for m_key, model in loaded_models.items():
                m_t0 = time.time()
                res = evaluate_model_on_dataset(model, corr_loader, device=device, idx_to_class=idx_to_class)
                m_elapsed = time.time() - m_t0

                clean_f1 = results[m_key]["clean"]["macro_f1"]
                macro_f1 = res["macro_f1"]
                acc = res["accuracy"]
                abs_drop = round(clean_f1 - macro_f1, 4)
                rel_drop_pct = round(((clean_f1 - macro_f1) / clean_f1) * 100.0, 2) if clean_f1 > 0 else 0.0

                results[m_key]["corruptions"][c_name][sev] = {
                    "severity": sev,
                    "param": str(param_val),
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "balanced_accuracy": res["balanced_accuracy"],
                    "error_rate": round(1.0 - acc, 4),
                    "absolute_f1_degradation": abs_drop,
                    "relative_f1_degradation_pct": rel_drop_pct,
                }
                print(f"      {m_key:<17}: Acc={acc:.2%} | F1={macro_f1:.4f} | AbsDrop={abs_drop:+.4f} | RelDrop={rel_drop_pct:.1f}% ({m_elapsed:.1f}s)", flush=True)

            cond_time = time.time() - t_cond_start
            print(f"    Condition completed in {cond_time:.1f}s", flush=True)

            # Checkpoint intermediate metrics immediately so no progress is lost on interruption
            intermediate_out = root / "experiments" / "runs" / "P05_robustness_evaluation" / "metrics_checkpoint.json"
            intermediate_out.parent.mkdir(parents=True, exist_ok=True)
            with open(intermediate_out, "w", encoding="utf-8") as f_inc:
                json.dump({
                    "phase": 5,
                    "status": "RUNNING",
                    "last_completed_condition": f"{c_name}_sev{sev}",
                    "condition_index": condition_idx,
                    "total_conditions": total_conditions,
                    "evaluated_models": {k: cfg["name"] for k, cfg in models_config.items()},
                    "results": results,
                }, f_inc, indent=2)

    # 3. Compute Relative Corruption Error (RCE) and mean RCE (mRCE) normalized to Student Baseline
    print("\n-------------------------------------------------------", flush=True)
    print("STEP 3: Computing Relative Corruption Error (RCE)", flush=True)
    print("-------------------------------------------------------", flush=True)

    for m_key in models_config:
        rce_per_family = {}
        for c_name in corruption_families:
            model_err_sum = sum(results[m_key]["corruptions"][c_name][s]["error_rate"] for s in range(1, 6))
            base_err_sum = sum(results["student_baseline"]["corruptions"][c_name][s]["error_rate"] for s in range(1, 6))
            rce_c = (model_err_sum / base_err_sum) * 100.0 if base_err_sum > 0 else 100.0
            rce_per_family[c_name] = round(rce_c, 2)

        mrce = float(np.mean(list(rce_per_family.values())))
        results[m_key]["relative_corruption_error"] = {
            "rce_by_family_pct": rce_per_family,
            "mean_rce_pct": round(mrce, 2),
            "normalization_reference": "student_baseline",
            "formula": "RCE_c = sum_{s=1..5}(1 - Acc_{m,c,s}) / sum_{s=1..5}(1 - Acc_{student_base,c,s}) * 100",
        }
        print(f"  Model: {m_key:<17} | mRCE: {mrce:.2f}% (Baseline = 100.0%)", flush=True)

    # 4. Natural Domain Shift (PlantDoc Leaf Crops)
    print("\n-------------------------------------------------------", flush=True)
    print("STEP 4: Natural Domain Shift Evaluation (PlantDoc Crops)", flush=True)
    print("-------------------------------------------------------", flush=True)
    doc_loader = build_plantdoc_dataloader(batch_size=batch_size, num_workers=0)
    cached_doc = []
    for x, y, _ in doc_loader:
        cached_doc.append((x.to(device), y.numpy()))

    for m_key, model in loaded_models.items():
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in cached_doc:
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.tolist())

        y_true, y_pred = np.array(all_targets), np.array(all_preds)
        acc = float(np.mean(y_true == y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        labels = sorted(list(set(y_true)))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

        clean_f1 = results[m_key]["clean"]["macro_f1"]
        abs_drop = round(clean_f1 - macro_f1, 4)
        rel_drop_pct = round(((clean_f1 - macro_f1) / clean_f1) * 100.0, 2)

        results[m_key]["plantdoc_cross_domain"] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "absolute_f1_drop": abs_drop,
            "relative_f1_drop_pct": rel_drop_pct,
            "total_samples": len(y_true),
        }
        print(f"  [{m_key.upper()} PlantDoc] Acc: {acc:.2%} | F1: {macro_f1:.4f} | Drop vs Clean: {abs_drop:+.4f} (-{rel_drop_pct:.1f}%)", flush=True)

    # 5. Save master metrics.json
    out_dir = root / "experiments" / "runs" / "P05_robustness_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"

    master_payload = {
        "phase": 5,
        "status": "COMPLETED",
        "experiment_series": "P05_robustness_evaluation",
        "corruption_families": corruption_families,
        "frozen_corruption_parameters": CORRUPTION_PARAMS,
        "evaluated_models": {k: cfg["name"] for k, cfg in models_config.items()},
        "results": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    print("\n=======================================================", flush=True)
    print(f"PHASE 5 ROBUSTNESS SUITE COMPLETE! Master metrics saved to: {out_file}", flush=True)
    print("=======================================================", flush=True)
    return master_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_phase_05_robustness_suite(
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )
