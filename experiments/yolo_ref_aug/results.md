# yolo_ref_aug

YOLOv8x with synthetic data from product reference images.

## Config
- Model: YOLOv8x (pretrained COCO)
- imgsz: 1536, batch: 2
- Same hyperparams as yolo_imgsz1536
- Dataset: original 198 train + 2724 synthetic = 2885 images
- Synthetic: product cutouts pasted onto shelf crops, rare categories oversampled

## Results
- Training on a100-vm (us-central1-a)
- Status: **running** (~6 min/epoch)
