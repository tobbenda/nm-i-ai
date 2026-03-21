"""
YOLOv8x fulldata at imgsz=2048. Images are 2000-3200px — less downscaling.
batch=1 on A100.
"""
import json
import shutil
from pathlib import Path
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

NAME = "yolo_fulldata_2048"
ANNOTATIONS = "data/train/annotations.json"
IMAGES = "data/train/images"
DATASET_DIR = f"data/yolo_{NAME}"


def coco_to_yolo():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    out = Path(DATASET_DIR)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    all_images = sorted(coco["images"], key=lambda x: x["id"])
    val_imgs = all_images[-5:]
    images_path = Path(IMAGES)
    for split_name, split_imgs in [("train", all_images), ("val", val_imgs)]:
        for img_info in split_imgs:
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]
            src = images_path / fname
            if src.exists():
                shutil.copy2(src, out / "images" / split_name / fname)
            label_file = out / "labels" / split_name / (Path(fname).stem + ".txt")
            lines = []
            for ann in anns_by_img.get(img_info["id"], []):
                bx, by, bw, bh = ann["bbox"]
                cx, cy = (bx + bw / 2) / w, (by + bh / 2) / h
                lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")
            label_file.write_text("\n".join(lines) + "\n" if lines else "")
    nc = len(coco["categories"])
    names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {nc}\nnames: {names}\n"
    )
    return str(yaml_path)


if __name__ == "__main__":
    yaml_path = coco_to_yolo()
    model = YOLO("yolov8x.pt")
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=2048,
        batch=1,
        device=0,
        patience=50,
        save=True,
        project="runs",
        name=NAME,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        close_mosaic=15,
    )
    print(f"\nDone! Weights: runs/{NAME}/weights/best.pt")
