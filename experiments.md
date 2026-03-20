# Experiment Log

Focus: **Detection mAP@0.5** (70% of competition score)

## Baseline (ONNX)

| Experiment | det_mAP | cls_mAP | Final | Notes |
|---|---|---|---|---|
| yolo_imgsz1536 (ONNX, conf=0.01, NMS=0.5) | **0.9107** | 0.8252 | 0.8850 | Current best submission |

## Phase 1: Inference tuning (no retraining)

| # | Change | det_mAP | Δ det | Notes |
|---|---|---|---|---|
| E3 | conf=0.01 | 0.9003 | +0.017 | sweet spot |
| E4 | conf=0.01 + TTA | 0.9019 | +0.019 | best with baseline weights |
| T2d | conf=0.01 + NMS=0.5 (imgsz1536 model) | 0.9055 | +0.005 | best .pt config |
| **ONNX** | **ONNX export + conf=0.01 + NMS=0.5** | **0.9107** | **+0.010** | **ONNX graph optimization helped** |

## Phase 2: Training experiments

| # | Experiment | det_mAP | Δ det | Verdict |
|---|---|---|---|---|
| T1 | YOLOv8l | 0.8987 | -0.012 | smaller model worse |
| **T2** | **YOLOv8x imgsz=1536** | **0.9047** | — | **best training config** |
| T3 | Heavy aug (mixup=0.3, cp=0.3) | 0.8926 | -0.012 | hurts |
| T4 | Lower LR (0.0005) | 0.8912 | -0.014 | hurts |
| T5 | SGD (lr=0.01) | 0.8937 | -0.011 | AdamW better |
| T6 | Synthetic product ref images | 0.525 | -0.380 | naive cutout pasting is too artificial |
| T7 | RT-DETR-l | 0.000 | -0.905 | transformer needs more data |
| T8 | yolo_tuned (cls=1.0, multi_scale, no cp) | ~0.653 | -0.035 | worse — killed early |
| T9 | yolo_fulldata (all 248 imgs, 50 epochs) | N/A | N/A | done, can't eval locally (trained on all data) |

## Phase 3: Inference architecture changes

| # | Approach | det_mAP | Δ det | Verdict |
|---|---|---|---|---|
| S1 | SAHI tiling (1024px, 20% overlap) | 0.8798 | -0.031 | worse — too many FPs from tile boundaries |
| **S2** | **WBF ensemble (1536+1280)** | **0.9082** | **-0.003** | **det slightly worse but cls +1.6%, final +0.3%** |
| S3 | DINOv2 crop classification | 0.9107 | 0.000 | cls WORSE at all thresholds — ref images too different from shelf crops |

## Current Best Configs
1. **Best detection:** ONNX + conf=0.01 + NMS=0.5 → det_mAP=0.9107, final=0.8850
2. **Best final score:** WBF (1536+1280) → det_mAP=0.9082, cls_mAP=0.8413, **final=0.8881**

## Key Learnings
1. Lower conf threshold (0.1→0.01) = +1.7% det_mAP (free)
2. Training at imgsz=1536 > 1280 (+0.4%)
3. ONNX export adds another +0.5% via graph optimization
4. TTA helps baseline but hurts 1536 model
5. NMS=0.5 > 0.7 for dense products
6. Heavy aug / lower LR / SGD / YOLOv8l all worse
7. RT-DETR needs much more data — dead end
8. Naive synthetic augmentation hurts badly
9. SAHI tiling hurts — model at 1536 already handles resolution well
