"""
Run a training experiment with configurable parameters.
Usage: python train_experiment.py --name exp_name --model yolov8x.pt [--imgsz 1280] ...
"""
import argparse
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


def coco_to_yolo(annotations_path, images_dir, output_dir, val_split=0.2):
    with open(annotations_path) as f:
        coco = json.load(f)

    out = Path(output_dir)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    all_images = sorted(coco["images"], key=lambda x: x["id"])
    split_idx = int(len(all_images) * (1 - val_split))
    splits = [("train", all_images[:split_idx]), ("val", all_images[split_idx:])]

    images_path = Path(images_dir)
    for split_name, split_imgs in splits:
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
    print(f"Dataset: {len(splits[0][1])} train, {len(splits[1][1])} val, {nc} classes")
    return str(yaml_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="yolov8x.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.1)
    p.add_argument("--copy-paste", type=float, default=0.1)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--freeze", type=int, default=0, help="Freeze first N layers")
    args = p.parse_args()

    dataset_dir = f"data/yolo_{args.name}"
    yaml_path = coco_to_yolo("data/train/annotations.json", "data/train/images",
                              dataset_dir, args.val_split)

    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        patience=args.patience,
        save=True,
        project="runs",
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
    )
    print(f"\nDone! Weights: runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
