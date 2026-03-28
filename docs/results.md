# Results

## Competition Scores

| Submission | Config | Score | Time |
|---|---|---|---|
| **Best** | **A@1536+B@1536+A@1280 fulldata FP16 WBF** | **0.9143** | **102.9s** |
| 2048 variant | A@2048+A@1536+B@1536 fulldata FP16 WBF | 0.9135 | 129.3s |

## Final Best Config
- **3-way multi-model WBF:** model_a_1536 + model_b_1536 + model_a_1280 (fulldata, FP16 ONNX)
- **Competition score: 0.9143**

## Experiment Results

### Phase 1: Inference tuning (no retraining)

| # | Change | det_mAP | Notes |
|---|---|---|---|
| E3 | conf=0.01 | 0.9003 | sweet spot |
| E4 | conf=0.01 + TTA | 0.9019 | best with baseline weights |
| T2d | conf=0.01 + NMS=0.5 (imgsz1536 model) | 0.9055 | best .pt config |
| **ONNX** | **ONNX export + conf=0.01 + NMS=0.5** | **0.9107** | **ONNX graph optimization helped** |

### Phase 2: Training experiments

| # | Experiment | det_mAP | Verdict |
|---|---|---|---|
| **T2** | **YOLOv8x imgsz=1536** | **0.9047** | **best training config** |
| T1 | YOLOv8l | 0.8987 | smaller model worse |
| T3 | Heavy aug (mixup=0.3, cp=0.3) | 0.8926 | hurts |
| T4 | Lower LR (0.0005) | 0.8912 | hurts |
| T5 | SGD (lr=0.01) | 0.8937 | AdamW better |
| T6 | Synthetic product ref images | 0.525 | naive cutout pasting too artificial |
| T7 | RT-DETR-l | 0.000 | transformer needs more data |
| T8 | yolo_tuned (cls=1.0, multi_scale, no cp) | ~0.653 | worse — killed early |
| T9 | yolo_fulldata (all 248 imgs, 50 epochs) | N/A | trained on all data, can't eval locally |

### Phase 3: Inference architecture

| # | Approach | det_mAP | Verdict |
|---|---|---|---|
| S1 | SAHI tiling (1024px, 20% overlap) | 0.8798 | too many FPs from tile boundaries |
| **S2** | **WBF ensemble (1536+1280)** | **0.9082** | **det slightly worse but cls +1.6%, final +0.3%** |
| S3 | DINOv2 crop classification | 0.9107 | cls worse at all thresholds |
| **S4** | **3-way multi-model WBF (A@1536+B@1536+A@1280)** | **0.9085** | **competition=0.9143 — BEST** |
| S5 | 3-way WBF (A@2048+A@1536+B@1536) | — | competition=0.9135 — slightly worse |

## Key Learnings
1. Lower conf threshold (0.1→0.01) = +1.7% det_mAP (free)
2. Training at imgsz=1536 > 1280 (+0.4%)
3. ONNX export adds +0.5% via graph optimization
4. TTA helps baseline but hurts 1536 model
5. NMS=0.5 > 0.7 for dense products
6. Heavy aug / lower LR / SGD / YOLOv8l all worse
7. RT-DETR needs much more data — dead end
8. Naive synthetic augmentation hurts badly
9. SAHI tiling hurts — model at 1536 already handles resolution well
10. Multi-model WBF ensemble is the winning strategy for dense object detection
