# Results

## Scores

| # | Model | Det mAP (70%) | Cls mAP (30%) | Final Score | Notes |
| - | ----- | ------------- | ------------- | ----------- | ----- |
| 1 | YOLOv8x pretrained COCO, imgsz=1280, conf=0.1 | ? | ? | ? | **baseline** — no fine-tuning, wrong categories |

## Tuning sweeps

- **imgsz**: 640 → 1280 (19 → 48 detections on 3 test images)
- **model size**: YOLOv8n → YOLOv8x (2.5x more detections)
- **conf threshold**: default 0.25 → 0.1 (more detections, noisier)

## Notes

- Detection-only (all category_id=0) can score max 0.70
- Fine-tuning with nc=357 is needed for classification (remaining 0.30)
- 3 submissions/day — validate locally before uploading
