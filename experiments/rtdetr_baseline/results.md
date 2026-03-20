# rtdetr_baseline

RT-DETR-l (transformer-based detector) on competition data.

## Config
- Model: RT-DETR-l (pretrained COCO)
- imgsz: 1280, batch: 4
- AdamW lr=0.0001, warmup=5, weight_decay=0.0001
- patience=20, epochs=100
- Trained with ultralytics==8.4.24 (RT-DETR buggy in 8.1.0)
- Will export to ONNX for sandbox submission

## Results
- Training on a100-vm-2 (us-central1-f)
- Status: **running**

## Notes
- RT-DETR has no NMS — set prediction, may handle dense shelves better
- ultralytics 8.1.0 has a stride bug for RT-DETR, needed 8.4.24 for training
- Inference via ONNX so sandbox version doesn't matter
