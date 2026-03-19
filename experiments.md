# Experiment Log

Focus: **Detection mAP@0.5** (70% of competition score)

## Baseline

| Experiment | det_mAP | cls_mAP | Final | Notes |
|---|---|---|---|---|
| v2-baseline (YOLOv8x, imgsz=1280, conf=0.1) | 0.8834 | 0.8041 | 0.8596 | ultralytics==8.1.0, eval_quick.py on val split |

## Phase 1: Inference-time experiments (no retraining)

All use v2-baseline best.pt weights.

| # | Change | det_mAP | cls_mAP | Final | Δ det | Notes |
|---|---|---|---|---|---|---|
| E1 | conf=0.05 | 0.8907 | 0.8108 | 0.8667 | +0.007 | 8061 preds |
| E2 | conf=0.2 | 0.8680 | 0.7929 | 0.8455 | -0.015 | 6602 preds, too aggressive |
| E3 | conf=0.01 | 0.9003 | 0.8191 | 0.8760 | +0.017 | 10880 preds |
| E4 | conf=0.01 + TTA | **0.9019** | **0.8307** | **0.8805** | **+0.019** | 12874 preds, best so far |
| E5 | conf=0.01 + NMS=0.5 | 0.9010 | 0.8194 | 0.8765 | +0.018 | |
| E6 | conf=0.01 + NMS=0.9 | 0.8833 | 0.8045 | 0.8597 | +0.000 | too many overlapping boxes |
| E7 | conf=0.01 + imgsz=1536 | 0.8957 | 0.8160 | 0.8718 | +0.012 | |
| E8 | conf=0.05 + TTA | 0.8970 | 0.8262 | 0.8758 | +0.014 | |
| E9 | conf=0.01 + TTA + imgsz=1536 | 0.9013 | 0.8293 | 0.8797 | +0.018 | |
| E10 | last.pt + conf=0.01 | 0.9017 | 0.8242 | 0.8785 | +0.018 | last.pt ≈ best.pt |
| E11 | conf=0.001 | 0.9017 | 0.8211 | 0.8775 | +0.018 | 14990 preds, diminishing returns |

**Key findings:**
- Lower conf threshold helps a lot (0.1→0.01 = +1.7% det_mAP)
- TTA adds another ~0.2% on top
- NMS=0.5 slightly better than default 0.7, NMS=0.9 hurts
- imgsz=1536 at inference doesn't help much over 1280
- conf=0.001 no better than 0.01 — floor reached

## Phase 1b: Training experiments (running on A100)

| # | Change | det_mAP | cls_mAP | Final | Δ det | Notes |
|---|---|---|---|---|---|---|
| T1 | YOLOv8l (smaller model) | | | | | running... |
| T2 | YOLOv8x imgsz=1536 train | | | | | queued |
| T3 | Heavy augmentation (mixup=0.3, cp=0.3, 150ep) | | | | | queued |
| T4 | Lower LR (0.0005, 150ep) | | | | | queued |
| T5 | SGD optimizer (lr=0.01, 150ep) | | | | | queued |

## Experiment Details

### Baseline: v2-baseline
- Model: YOLOv8x pretrained on COCO, fine-tuned on competition data
- ultralytics==8.1.0 (matches sandbox)
- imgsz=1280, batch=4, epochs=100, patience=15
- AdamW lr=0.001, warmup=5, weight_decay=0.0005
- Augmentation: mosaic=1.0, mixup=0.1, copy_paste=0.1
- 80/20 train/val split (198/50 images)
- Local eval det_mAP@0.5 = 0.8834

### Best inference config so far
- conf=0.01, TTA=True, imgsz=1280, NMS=0.7
- det_mAP=0.9019, final=0.8805
