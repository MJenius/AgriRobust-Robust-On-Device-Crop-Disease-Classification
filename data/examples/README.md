# Dataset Testing Examples & Empirical Validation Log

This directory contains representative plant leaf images from the project datasets.

---

## 1. Digital File Evaluation vs Live Physical Screen Capture

When testing an on-device computer vision model, there are two distinct evaluation modes:
1. **Direct Digital Input (Gallery Mode)**: Feeding the exact digital image pixels directly into the $224 \times 224$ tensor pipeline.
2. **Physical Camera Recapture (Screen-to-Camera Capture)**: Photographing a computer monitor with the phone camera introduces real physical domain perturbations:
   - **Moiré interference patterns** (visible raster lines on the screen).
   - **Screen backlight bloom and glare**.
   - **RGB subpixel sampling artifacts and parallax distortion**.

AgriRobust was designed with **temperature calibration ($T = 0.5406$)** and **selective abstention ($\tau = 0.8143 = 81.4\%$)** to help mitigate overconfidence and flag uncertain inputs. However, empirical testing on out-of-domain and optically shifted inputs shows that **abstention does not provide a universal safety guarantee against overconfidence under severe domain shift**.

---

## 2. Empirical Test Results Log (Samsung Galaxy SM-A146B)

The table below documents both the exact digital tensor predictions and the live physical camera recapture results obtained on the physical Android device:

| Image File | Test Subject & Source | Direct Digital File (Gallery) | Live Camera Recapture (Screen Photo) | Observed Decision & Behavior |
|---|---|---|---|---|
| **`01_plantvillage_tomato_early_blight.jpg`** | Tomato Early Blight *(PlantVillage)* | `Tomato — Early blight`<br>**99.6%** (ACCEPTED) | `Tomato — Early blight`<br>**56.3%** (<span style="color:red">**ABSTAINED**</span>) | Correct disease predicted; physical Moiré artifact lowered confidence below 81.4%, activating abstention. |
| **`02_plantdoc_corn___common_rust.jpg`** | Corn Leaf *(PlantDoc field shift)* | `Corn — Healthy`<br>**99.9%** (ACCEPTED) | `Corn — Healthy`<br>**83.9%** (<span style="color:green">**ACCEPTED**</span>) | Top-1 class agreed in both modes (`corn - healthy`); 83.9% confidence exceeded threshold. Both misdiagnosed common rust as healthy corn due to field shift. |
| **`03_plantdoc_potato_late_blight.jpg`** | Potato Late Blight crop *(PlantDoc)* | `Bell pepper — Healthy`<br>**55.9%** (<span style="color:red">**ABSTAINED**</span>) | `Corn — Gray leaf spot`<br>**99.7%** (<span style="color:orange">**HIGH-CONFIDENCE FAILURE**</span>) | **Critical Failure Case**: While digital mode abstained (55.9%), physical camera recapture under Moiré grid lines produced an erroneous `Corn — Gray leaf spot` prediction at **99.7% confidence**, bypassing the abstention threshold. |
| **`04_plantvillage_corn_healthy.jpg`** | Healthy Corn Leaf *(PlantVillage)* | `Corn — Healthy`<br>**100.0%** (ACCEPTED) | `Corn — Healthy`<br>**100.0%** (<span style="color:green">**ACCEPTED**</span>) | 100.0% agreement and confidence across both digital file and live screen photo capture. |
| **`05_plantvillage_apple_scab.jpg`** | Apple Scab *(PlantVillage)* | `Apple — Apple scab`<br>**100.0%** (ACCEPTED) | `Apple — Apple scab`<br>**83.3%** (<span style="color:green">**ACCEPTED**</span>) | Correct disease predicted in both modes (`apple - apple scab`); high-confidence acceptance retained above threshold. |
| **`06_plantdoc_apple_cedar_rust.jpg`** | Apple Cedar Rust crop *(PlantDoc)* | `Grape — Leaf blight`<br>**57.0%** (<span style="color:red">**ABSTAINED**</span>) | `Corn — Gray leaf spot`<br>**47.4%** (<span style="color:red">**ABSTAINED**</span>) | Natural field lesion shift + camera noise yielded low confidence (47.4%), triggering selective abstention. |

---

## 3. Key Scientific Takeaways & Qualitative Evidence

These six real-device tests serve as qualitative case studies of on-device model behavior:

1. **Diagnosis Consistency on In-Distribution Clean Leaves**:
   - For **Apple Scab** (`05`), the model correctly diagnosed `apple - apple scab` in both digital and camera modes.
   - For **Healthy Corn** (`04`), the model showed 100.0% confidence in both modes.
   - For **Tomato Early Blight** (`01`), the model predicted the correct disease class in both modes (`tomato - early blight`).

2. **Qualitative Evidence of Selective Abstention**:
   - On difficult or corrupted crops (`01` via camera, and `06`), confidence dropped below $81.4\%$, and the on-device system abstained (`ABSTAINED (Low Confidence)`) rather than accepting a low-evidence prediction.

3. **Explicit Documentation of High-Confidence Failure Mode**:
   - On `03_plantdoc_potato_late_blight.jpg`, screen recapture produced an erroneous `Corn — Gray leaf spot` prediction at **99.7% confidence** (<span style="color:orange">**HIGH-CONFIDENCE ACCEPTED ERROR**</span>).
   - This empirically demonstrates that **temperature scaling calibrated on clean validation data ($T = 0.5406$) does not prevent severe overconfidence under unexpected optical shifts (Moiré interference + screen glare) on out-of-domain plant leaves**.
   - Selective abstention is a valuable uncertainty filter for mild-to-moderate degradation, but must **not** be treated as a foolproof guarantee against false high-confidence predictions in real-world agricultural deployments.
