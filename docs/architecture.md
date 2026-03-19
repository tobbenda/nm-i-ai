# NorgesGruppen Object Detection — Architecture

## Minimal Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING (Local)                      │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ COCO Dataset  │───▶│  Fine-tune   │───▶│  best.pt   │ │
│  │ 254 images    │    │  YOLOv8x     │    │  weights   │ │
│  │ 22.3k annots  │    │  nc=357      │    │            │ │
│  │ annotations   │    │  ultralytics │    │  (<420 MB) │ │
│  │  .json        │    │  ==8.1.0     │    │            │ │
│  └──────────────┘    └──────────────┘    └─────┬──────┘ │
└────────────────────────────────────────────────┼────────┘
                                                 │
                                                 ▼
                                        ┌────────────────┐
                                        │ submission.zip  │
                                        │ ├── run.py      │
                                        │ └── best.pt     │
                                        └───────┬────────┘
                                                │
                                             upload
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────┐
│               SANDBOX (Their Server)                     │
│               L4 GPU · 300s · No network                 │
│                                                          │
│  /data/images/              run.py                       │
│  ├── img_00001.jpg    ┌──────────────┐                   │
│  ├── img_00002.jpg───▶│ Load best.pt │                   │
│  ├── ...              │ For each img:│                   │
│  └── img_XXXXX.jpg    │  ├─ detect   │                   │
│                       │  ├─ classify │                   │
│                       │  └─ collect  │                   │
│                       └──────┬───────┘                   │
│                              │                           │
│                              ▼                           │
│                  /output/predictions.json                 │
│                  [{"image_id": 42,                        │
│                    "category_id": 7,                      │
│                    "bbox": [x,y,w,h],                    │
│                    "score": 0.92}, ...]                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  SCORING        │
                  │                 │
                  │  0.7 × det_mAP  │
                  │ +0.3 × cls_mAP  │
                  │ ────────────── │
                  │  = final score  │
                  └─────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `train.py` | Fine-tune YOLOv8x on COCO dataset, pin ultralytics==8.1.0 |
| `run.py` | Load model, inference, output predictions.json |
| `evaluate.py` | Local mAP evaluation on a val split |

## Constraints

- Sandbox: L4 GPU, 4 vCPU, 8 GB RAM, 300s timeout, no network
- Zip: max 420 MB, max 3 weight files, max 10 .py files
- No `import os` — use `pathlib`
- Pin `ultralytics==8.1.0`
- 3 submissions/day

## Scoring

```
Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
```

- Detection: box IoU ≥ 0.5, category ignored
- Classification: box IoU ≥ 0.5 AND correct category_id
- 357 categories (0–356)
