# yolo_imgsz1536

YOLOv8x fine-tuned at imgsz=1536. Current best model.

## Config
- Model: YOLOv8x (pretrained COCO)
- imgsz: 1536, batch: 2
- AdamW lr=0.001, warmup=5, weight_decay=0.0005
- mosaic=1.0, mixup=0.1, copy_paste=0.1
- patience=15, epochs=100

## Results
- det_mAP@0.5: **0.9047** (eval_quick.py, conf=0.01)
- det_mAP@0.5 + NMS=0.5: **0.9055**
- cls_mAP@0.5: 0.8340
- Final: 0.8835
- Early stopped at ~epoch 57

## Notes
- Best training config from Phase 1 breadth search
- ONNX export gave even better results (0.9107) due to graph optimization
