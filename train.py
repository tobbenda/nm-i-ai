"""
Fine-tune YOLOv8x on NorgesGruppen product detection dataset.
Converts COCO annotations to YOLO format, sets up dataset YAML, and trains.
"""
import json
import shutil
from pathlib import Path
from ultralytics import YOLO


def coco_to_yolo(annotations_path: str, images_dir: str, output_dir: str):
    """Convert COCO format annotations to YOLO format."""
    with open(annotations_path) as f:
        coco = json.load(f)

    # Build image lookup
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Create output dirs
    out = Path(output_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Split: use last 20% as val
    all_images = sorted(coco["images"], key=lambda x: x["id"])
    split_idx = int(len(all_images) * 0.8)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    images_path = Path(images_dir)

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img_info in split_imgs:
            img_id = img_info["id"]
            w, h = img_info["width"], img_info["height"]
            fname = img_info["file_name"]

            # Copy image
            src = images_path / fname
            if src.exists():
                shutil.copy2(src, out / "images" / split_name / fname)

            # Write YOLO labels
            label_file = out / "labels" / split_name / (Path(fname).stem + ".txt")
            lines = []
            for ann in anns_by_img.get(img_id, []):
                # COCO bbox: [x, y, width, height] (top-left)
                bx, by, bw, bh = ann["bbox"]
                # Convert to YOLO: center_x, center_y, width, height (normalized)
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cat_id = ann["category_id"]
                lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_file.write_text("\n".join(lines) + "\n" if lines else "")

    # Write dataset YAML
    nc = len(coco["categories"])
    names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    print(f"Converted: {len(train_imgs)} train, {len(val_imgs)} val images")
    print(f"Categories: {nc}")
    print(f"Dataset YAML: {yaml_path}")
    return str(yaml_path)


def main():
    # Paths (on the VM)
    annotations = "data/train/annotations.json"
    images = "data/train/images"
    output = "data/yolo_dataset"

    print("Converting COCO to YOLO format...")
    yaml_path = coco_to_yolo(annotations, images, output)

    print("\nStarting training...")
    model = YOLO("yolov8x.pt")
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=1280,
        batch=4,
        device=0,
        patience=15,
        save=True,
        project="runs",
        name="a100_finetune",
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=5,
    )

    print("\nTraining complete! Best weights at: runs/a100_finetune/weights/best.pt")


if __name__ == "__main__":
    main()
