"""Dataset processing, manifest generation, and integrity validation for AgriRobust Phase 1."""

import csv
import hashlib
import json
import random
import zipfile
from pathlib import Path

from PIL import Image

from agrirobust.config import get_project_root

# Canonical class taxonomy definition: standard <crop>___<condition>
PLANTVILLAGE_TO_CANONICAL = {
    "Apple___Apple_scab": "apple___apple_scab",
    "Apple___Black_rot": "apple___black_rot",
    "Apple___Cedar_apple_rust": "apple___cedar_apple_rust",
    "Apple___healthy": "apple___healthy",
    "Blueberry___healthy": "blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry___powdery_mildew",
    "Cherry_(including_sour)___healthy": "cherry___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn___gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "corn___common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "corn___northern_leaf_blight",
    "Corn_(maize)___healthy": "corn___healthy",
    "Grape___Black_rot": "grape___black_rot",
    "Grape___Esca_(Black_Measles)": "grape___esca_black_measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape___leaf_blight",
    "Grape___healthy": "grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "orange___citrus_greening",
    "Peach___Bacterial_spot": "peach___bacterial_spot",
    "Peach___healthy": "peach___healthy",
    "Pepper,_bell___Bacterial_spot": "bell_pepper___bacterial_spot",
    "Pepper,_bell___healthy": "bell_pepper___healthy",
    "Potato___Early_blight": "potato___early_blight",
    "Potato___Late_blight": "potato___late_blight",
    "Potato___healthy": "potato___healthy",
    "Raspberry___healthy": "raspberry___healthy",
    "Soybean___healthy": "soybean___healthy",
    "Squash___Powdery_mildew": "squash___powdery_mildew",
    "Strawberry___Leaf_scorch": "strawberry___leaf_scorch",
    "Strawberry___healthy": "strawberry___healthy",
    "Tomato___Bacterial_spot": "tomato___bacterial_spot",
    "Tomato___Early_blight": "tomato___early_blight",
    "Tomato___Late_blight": "tomato___late_blight",
    "Tomato___Leaf_Mold": "tomato___leaf_mold",
    "Tomato___Septoria_leaf_spot": "tomato___septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "tomato___spider_mites_two_spotted",
    "Tomato___Target_Spot": "tomato___target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato___yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "tomato___mosaic_virus",
    "Tomato___healthy": "tomato___healthy",
}

# Mapping from PlantDoc detection classes to canonical classes
PLANTDOC_TO_CANONICAL = {
    "Apple Scab Leaf": "apple___apple_scab",
    "Apple leaf": "apple___healthy",
    "Apple rust leaf": "apple___cedar_apple_rust",
    "Bell_pepper leaf": "bell_pepper___healthy",
    "Bell_pepper leaf spot": "bell_pepper___bacterial_spot",
    "Blueberry leaf": "blueberry___healthy",
    "Cherry leaf": "cherry___healthy",
    "Corn Gray leaf spot": "corn___gray_leaf_spot",
    "Corn leaf blight": "corn___northern_leaf_blight",
    "Corn rust leaf": "corn___common_rust",
    "Peach leaf": "peach___healthy",
    "Potato leaf": "potato___healthy",
    "Potato leaf early blight": "potato___early_blight",
    "Potato leaf late blight": "potato___late_blight",
    "Raspberry leaf": "raspberry___healthy",
    "Soyabean leaf": "soybean___healthy",
    "Squash Powdery mildew leaf": "squash___powdery_mildew",
    "Strawberry leaf": "strawberry___healthy",
    "Tomato Early blight leaf": "tomato___early_blight",
    "Tomato Septoria leaf spot": "tomato___septoria_leaf_spot",
    "Tomato leaf": "tomato___healthy",
    "Tomato leaf bacterial spot": "tomato___bacterial_spot",
    "Tomato leaf late blight": "tomato___late_blight",
    "Tomato leaf mosaic virus": "tomato___mosaic_virus",
    "Tomato leaf yellow virus": "tomato___yellow_leaf_curl_virus",
    "Tomato mold leaf": "tomato___leaf_mold",
    "Tomato two spotted spider mites leaf": "tomato___spider_mites_two_spotted",
    "grape leaf": "grape___healthy",
    "grape leaf black rot": "grape___black_rot",
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 16):
            h.update(chunk)
    return h.hexdigest()


def compute_crop_sha256(img: Image.Image) -> str:
    """Compute deterministic SHA-256 hash of raw crop pixel bytes."""
    return hashlib.sha256(img.tobytes()).hexdigest()


def generate_plantvillage_manifest(
    seed: int = 42, val_ratio: float = 0.15, test_ratio: float = 0.15
):
    root = get_project_root()
    pv_dir = root / "data" / "raw" / "plantvillage_repo" / "raw" / "color"
    manifests_dir = root / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records = []
    class_dirs = sorted([d for d in pv_dir.iterdir() if d.is_dir()])
    rng = random.Random(seed)

    for c_dir in class_dirs:
        raw_class_name = c_dir.name
        canonical_class = PLANTVILLAGE_TO_CANONICAL[raw_class_name]
        is_healthy = canonical_class.endswith("___healthy")
        crop = canonical_class.split("___")[0]

        img_paths = sorted(list(c_dir.glob("*.*")))
        indices = list(range(len(img_paths)))
        rng.shuffle(indices)

        n_total = len(img_paths)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        for idx_pos, orig_idx in enumerate(indices):
            img_p = img_paths[orig_idx]
            if idx_pos < n_train:
                split = "train"
            elif idx_pos < n_train + n_val:
                split = "val"
            else:
                split = "test"

            rel_path = img_p.relative_to(root).as_posix()
            records.append({
                "dataset": "plantvillage",
                "image_path": rel_path,
                "raw_label": raw_class_name,
                "canonical_label": canonical_class,
                "crop": crop,
                "is_healthy": is_healthy,
                "split": split,
                "sha256": compute_sha256(img_p),
            })

    out_file = manifests_dir / "plantvillage_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "plantvillage",
                "version": "1.0.0",
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "total_images": len(records),
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"Generated PlantVillage manifest: {len(records)} records -> {out_file}")
    return records


def generate_plantdoc_crop_manifest(min_crop_size: int = 20):
    root = get_project_root()
    pd_od_dir = root / "data" / "raw" / "plantdoc_od"
    manifests_dir = root / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records = []
    img_cache = {}

    for split in ["train", "test"]:
        csv_file = pd_od_dir / f"{split}_labels.csv"
        if not csv_file.exists():
            continue
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                raw_filename = row["filename"]
                raw_cls = row["class"]
                if raw_cls not in PLANTDOC_TO_CANONICAL:
                    continue

                canonical_cls = PLANTDOC_TO_CANONICAL[raw_cls]
                is_healthy = canonical_cls.endswith("___healthy")
                crop = canonical_cls.split("___")[0]

                xmin = int(float(row["xmin"]))
                ymin = int(float(row["ymin"]))
                xmax = int(float(row["xmax"]))
                ymax = int(float(row["ymax"]))

                box_w = max(0, xmax - xmin)
                box_h = max(0, ymax - ymin)

                if box_w < min_crop_size or box_h < min_crop_size:
                    continue

                folder = "TRAIN" if split == "train" else "TEST"
                img_path = pd_od_dir / folder / raw_filename
                if not img_path.exists():
                    safe_name = raw_filename.replace("?", "_").replace(":", "_").replace("*", "_")
                    img_path = pd_od_dir / folder / safe_name

                if not img_path.exists():
                    continue

                if img_path not in img_cache:
                    img_cache[img_path] = Image.open(img_path)

                try:
                    full_im = img_cache[img_path]
                    cropped_im = full_im.crop((xmin, ymin, xmax, ymax))
                    crop_hash = compute_crop_sha256(cropped_im)
                except Exception:
                    crop_hash = ""

                rel_path = img_path.relative_to(root).as_posix()
                records.append({
                    "dataset": "plantdoc",
                    "source_image_path": rel_path,
                    "crop_id": f"pd_{split}_{row_idx:05d}",
                    "bbox": [xmin, ymin, xmax, ymax],
                    "raw_label": raw_cls,
                    "canonical_label": canonical_cls,
                    "crop": crop,
                    "is_healthy": is_healthy,
                    "crop_sha256": crop_hash,
                    "split": "cross_domain_test",
                    "original_od_split": split,
                })

    out_file = manifests_dir / "plantdoc_crops_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "plantdoc",
                "version": "1.0.0",
                "task": "cropped_plant_leaf_disease_classification",
                "min_crop_size": min_crop_size,
                "total_crops": len(records),
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"Generated PlantDoc crops manifest: {len(records)} crops -> {out_file}")
    return records


def generate_plantseg_manifest():
    root = get_project_root()
    manifests_dir = root / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    zip_path = root / "data" / "raw" / "plantseg_zenodo" / "plantsegv3.zip"

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        image_entries = [
            n
            for n in names
            if n.startswith("plantsegv3/images/") and n.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    records = []
    for entry in image_entries:
        parts = entry.split("/")
        split = parts[2] if len(parts) > 2 else "unknown"
        filename = parts[-1]
        records.append({
            "dataset": "plantseg",
            "archive_path": entry,
            "filename": filename,
            "split": split,
            "modality": "image_and_mask",
            "role": "segmentation_and_robustness_evaluation",
        })

    out_file = manifests_dir / "plantseg_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "plantseg",
                "version": "v3",
                "source": "zenodo_14935094",
                "license": "CC-BY-4.0",
                "total_samples": len(records),
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"Generated PlantSeg manifest: {len(records)} records -> {out_file}")
    return records


if __name__ == "__main__":
    generate_plantvillage_manifest()
    generate_plantdoc_crop_manifest()
    generate_plantseg_manifest()
