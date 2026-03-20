# Experiment Log

Focus: **Detection mAP@0.5** (70% of competition score)

## Baseline

| Experiment | det_mAP | cls_mAP | Final | Notes |
|---|---|---|---|---|
| v2-baseline (YOLOv8x, imgsz=1280, conf=0.01) | 0.9003 | 0.8191 | 0.8760 | ultralytics==8.1.0, eval_quick.py on val split |

## Phase 1a: Inference-time experiments (v2-baseline weights)

| # | Change | det_mAP | Δ det | Notes |
|---|---|---|---|---|
| E1 | conf=0.05 | 0.8907 | +0.007 | |
| E2 | conf=0.2 | 0.8680 | -0.015 | too aggressive filter |
| E3 | conf=0.01 | 0.9003 | +0.017 | sweet spot |
| E4 | conf=0.01 + TTA | 0.9019 | +0.019 | best with baseline weights |
| E5 | conf=0.01 + NMS=0.5 | 0.9010 | +0.018 | |
| E6 | conf=0.01 + NMS=0.9 | 0.8833 | +0.000 | too many overlaps |
| E7 | conf=0.01 + imgsz=1536 | 0.8957 | +0.012 | |
| E8 | conf=0.05 + TTA | 0.8970 | +0.014 | |
| E9 | conf=0.01 + TTA + imgsz=1536 | 0.9013 | +0.018 | |
| E10 | last.pt + conf=0.01 | 0.9017 | +0.018 | |
| E11 | conf=0.001 | 0.9017 | +0.018 | diminishing returns |

## Phase 1b: Training experiments (on A100)

All evaluated with conf=0.01, default NMS.

| # | Change | det_mAP | Δ det | Verdict |
|---|---|---|---|---|
| T1 | YOLOv8l | 0.8987 | -0.002 | worse — bigger model wins |
| **T2** | **YOLOv8x imgsz=1536** | **0.9047** | **+0.004** | **best training config** |
| T3 | Heavy aug (mixup=0.3, cp=0.3) | 0.8926 | -0.008 | too much aug hurts |
| T4 | Lower LR (0.0005) | 0.8912 | -0.009 | default LR better |
| T5 | SGD (lr=0.01) | 0.8937 | -0.007 | AdamW better |

## Phase 1c: Inference tuning on T2 (imgsz=1536 model)

| # | Config | det_mAP | Δ det | Notes |
|---|---|---|---|---|
| T2a | conf=0.01 (default) | 0.9047 | — | |
| T2b | conf=0.01 + TTA | 0.8937 | -0.011 | TTA hurts this model! |
| T2c | conf=0.01 + TTA + imgsz=1536 | 0.9002 | -0.005 | still worse |
| **T2d** | **conf=0.01 + NMS=0.5** | **0.9055** | **+0.001** | **overall best** |
| T2e | conf=0.005 | 0.9051 | +0.000 | no gain |

## Current Best Config
- **Model:** YOLOv8x trained at imgsz=1536 (`runs/exp_imgsz1536/weights/best.pt`)
- **Inference:** conf=0.01, imgsz=1536, iou(NMS)=0.5, no TTA
- **det_mAP = 0.9055, final = 0.8843**

## Key Learnings
1. **Lower confidence threshold** is essentially free — 0.1→0.01 gives +1.7% det_mAP
2. **Training at higher resolution** (1536) helps more than any augmentation or optimizer change
3. **TTA helps baseline model but hurts the 1536 model** — the model already sees enough detail
4. **NMS=0.5** slightly better than 0.7 for dense shelf products (overlapping boxes)
5. **Heavy augmentation, lower LR, SGD all hurt** — baseline hyperparams were near-optimal
6. **YOLOv8x > YOLOv8l** — despite only 248 images, the larger model generalizes better
