# Hackaholics---AISEHACK-Theme-1-Phase-2
# Flood Detection — AISEHack 2026 Phase 2

> **ANRF - AISEHack - Phase 2 - Theme 1 - Flood Detection (IBM)**  
> Team Hackaholics

---

## Problem

3-class semantic segmentation on multi-spectral + SAR satellite imagery over **West Bengal, India**.

| Class | Label |
|-------|-------|
| 0 | No Flood |
| 1 | Flood |
| 2 | Water Body |

The core challenge: **distinguishing active flood inundation from pre-existing permanent water bodies** at pixel level.

---

## Dataset

| | Count | Format |
|---|---|---|
| Train images | ~59 | 512×512 · 6-band TIF |
| Val images | ~10 | 512×512 · 6-band TIF |
| Test images | 19 | 512×512 · 6-band TIF |

**Bands:** HH (SAR) · HV (SAR) · Green · Red · NIR · SWIR  
**Class split:** No Flood 65.6% · Flood 13.5% · Water Body 20.9%

---

## Notebooks

| # | Notebook | Approach | Score |
|---|----------|----------|-------|
| 1 | [notebook76d105b1fd (2).ipynb](notebook76d105b1fd%20(2).ipynb) | 3-model ensemble (UNet++ EffB4 + DLV3+ R101 + UNet++ R50), model soup, Dice-Focal loss, TTA, threshold sweep | 0.1795 |
<!-- Add new notebooks here -->
<!-- | 2 | [notebook_name.ipynb](notebook_name.ipynb) | Description | 0.XX | -->

---

## Approaches Explored

### Models
(for each notebook this might vary)
- UNet++ (EfficientNet-B4, ResNet-50)
- DeepLabV3+ (ResNet-101)
- _More experiments ongoing..._

### Techniques
- **Model Soup** — top-K checkpoint weight averaging
- **Dice-Focal Loss** — macro dice + focal CE with class weighting
- **Flood Oversampling** — WeightedRandomSampler (3× for flood-heavy patches)
- **Test-Time Augmentation** — multi-flip ensemble at inference
- **Threshold Calibration** — per-class probability threshold sweep on validation
- **Full 512×512 Training** — no crops, preserves spatial context

### Submission
- RLE encoding of **flood-only** pixels (class 1)
- Column-major order, 1-indexed
- Empty mask → `0 0`

---

## How to Run

```bash
# Requirements
pip install segmentation_models_pytorch albumentations rasterio opencv-python-headless torch
```

1. Open any notebook on [Kaggle](https://www.kaggle.com/competitions/anrfaisehack-theme-1-phase2)
2. Attach the competition dataset
3. Enable **GPU (T4)**
4. Run all cells → `submission.csv` is generated in output

---

## References

- [Competition Page](https://www.kaggle.com/competitions/anrfaisehack-theme-1-phase2)
- [AISEHack Helper Code](https://github.com/AISEHack/AISEHack_Edition1_2026)
- [IBM Prithvi Foundation Model](https://huggingface.co/ibm-nasa-geospatial)
- [TerraTorch](https://github.com/terrastackai/terratorch)
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
