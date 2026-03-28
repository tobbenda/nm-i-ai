# NM i AI 2026 -- NorgesGruppen Product Detection

Grocery shelf product detection and classification for the [NM i AI 2026](https://www.norskindustri.no/nm-i-ai/) competition (NorgesGruppen challenge).

**Final score: 0.9143**

## Problem

Detect and classify grocery products on store shelf images. The scoring is 70% detection mAP + 30% classification mAP, evaluated in a sandboxed environment (L4 GPU, 300s total, offline).

- 248 labeled shelf images with ~99 products each (COCO format)
- 357 product categories with 7 reference angles per product
- Strict submission constraints: ultralytics 8.1.0, no internet, 500MB zip limit

## Solution

**3-model WBF ensemble** of YOLOv8x models, all fine-tuned on the full dataset and exported as FP16 ONNX:

| Model | Training imgsz | Inference imgsz | Weight |
|-------|---------------|-----------------|--------|
| model_a_1536 | 1536 | 1536 | 1.0 |
| model_b_1536 | 1536 (different seed/LR) | 1536 | 0.8 |
| model_a_1280 | 1536 | 1280 | 0.6 |

The models run independently on each image, then their predictions are fused with [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) (IoU=0.5). A horizontal flip TTA pass on the first model is added as a 4th input to WBF (weight=0.5).

Key inference settings: `conf=0.01`, `NMS=0.5`, FP16 throughout. Time budget with 15s reserve writes partial results if running long.

### Pipeline

```
train.py          COCO→YOLO conversion + YOLOv8x fine-tuning (on A100 VMs)
    ↓
build_submission.sh   ONNX export + FP16 quantization + validation checks
    ↓
run.py            3-model WBF ensemble + flip TTA → COCO predictions JSON
```

## What worked

| Change | Impact |
|--------|--------|
| conf 0.1 → 0.01 | +1.7% det_mAP (free) |
| imgsz 1280 → 1536 | +0.4% det_mAP |
| ONNX export | +0.5% det_mAP (graph optimization) |
| Multi-model WBF ensemble | +2.9% cls_mAP, competition score 0.885 → 0.914 |
| FP16 quantization | Half the model size, no accuracy loss |
| Fulldata training (no val holdout) | Needed for final submission models |

## What didn't work

| Approach | Result | Why |
|----------|--------|-----|
| RT-DETR-l | 0.000 mAP | Transformer needs far more than 248 images |
| Synthetic data augmentation | 0.525 mAP (vs 0.682 baseline) | Naive cutout pasting looks too artificial |
| SAHI tiling | -3.1% det_mAP | Too many false positives from tile boundaries |
| DINOv2 crop classification | -1.8% cls_mAP | Reference images too different from shelf crops |
| Heavy augmentation | -1.2% det_mAP | mixup=0.3 + copy_paste=0.3 hurts on small dataset |
| Lower learning rate | -1.4% det_mAP | Default AdamW lr was already optimal |
| SGD optimizer | -1.1% det_mAP | AdamW consistently better |
| YOLOv8l (smaller model) | -1.2% det_mAP | YOLOv8x worth the extra compute |
| imgsz=2048 in ensemble | -0.08% competition score | Marginal, and 26s slower |

## Repo structure

```
run.py                  Inference pipeline (3-model WBF + flip TTA)
train.py                COCO→YOLO conversion + fine-tuning script
build_submission.sh     ONNX export, FP16 quantization, submission packaging
evaluate.py             Local evaluation with HTML report
experiments/            Per-experiment training configs and results
experiments.md          Full experiment log with scores
scripts/                Utility scripts (augmentation, gallery builder, etc.)
docs/                   Work doc, results, competition docs
data/                   Training images, annotations, product references
```
