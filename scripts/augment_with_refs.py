"""
Augment training data using product reference images.
Strategy: paste reference product cutouts onto shelf image crops to create
additional training examples, improving classification on rare categories.

This creates a YOLO-format dataset combining original + synthetic images.
"""
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)


def load_mappings(annotations_path, metadata_path):
    """Load category and product code mappings."""
    with open(annotations_path) as f:
        coco = json.load(f)
    with open(metadata_path) as f:
        meta = json.load(f)

    cat_by_name = {c["name"].upper().strip(): c["id"] for c in coco["categories"]}
    code_to_catid = {}
    for p in meta["products"]:
        name = p.get("product_name", "").upper().strip()
        code = p.get("product_code", "")
        if name in cat_by_name and p.get("has_images"):
            code_to_catid[code] = cat_by_name[name]

    return coco, code_to_catid


def get_annotation_counts(coco):
    """Count annotations per category to find rare ones."""
    counts = defaultdict(int)
    for ann in coco["annotations"]:
        counts[ann["category_id"]] += 1
    return counts


def create_synthetic_image(shelf_img, ref_img, target_size=(80, 120)):
    """Paste a reference product image onto a random crop of a shelf image."""
    # Random crop from shelf as background
    sw, sh = shelf_img.size
    crop_w, crop_h = min(400, sw), min(400, sh)
    cx = random.randint(0, max(0, sw - crop_w))
    cy = random.randint(0, max(0, sh - crop_h))
    bg = shelf_img.crop((cx, cy, cx + crop_w, cy + crop_h))

    # Resize reference image with random variation
    w_var = random.uniform(0.7, 1.3)
    h_var = random.uniform(0.7, 1.3)
    rw = max(20, int(target_size[0] * w_var))
    rh = max(20, int(target_size[1] * h_var))
    ref_resized = ref_img.resize((rw, rh), Image.BILINEAR)

    # Random augmentations on reference
    if random.random() < 0.3:
        ref_resized = ref_resized.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness/contrast
    arr = np.array(ref_resized, dtype=np.float32)
    brightness = random.uniform(0.7, 1.3)
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    ref_resized = Image.fromarray(arr)

    # Paste at random position (ensure it fits)
    max_x = max(0, crop_w - rw)
    max_y = max(0, crop_h - rh)
    px = random.randint(0, max_x) if max_x > 0 else 0
    py = random.randint(0, max_y) if max_y > 0 else 0

    bg.paste(ref_resized, (px, py))

    # Return image and YOLO-format label
    # YOLO format: class cx cy w h (normalized)
    cx_norm = (px + rw / 2) / crop_w
    cy_norm = (py + rh / 2) / crop_h
    w_norm = rw / crop_w
    h_norm = rh / crop_h

    return bg, (cx_norm, cy_norm, w_norm, h_norm)


def main():
    annotations_path = "data/train/annotations.json"
    metadata_path = "data/metadata.json"
    images_dir = Path("data/train/images")
    ref_dir = Path("data")
    output_dir = Path("data/yolo_augmented")

    # Load mappings
    coco, code_to_catid = load_mappings(annotations_path, metadata_path)
    ann_counts = get_annotation_counts(coco)

    print(f"Mapped {len(code_to_catid)} product codes to categories")

    # Find rare categories (fewer than median annotations)
    counts_list = sorted(ann_counts.values())
    median_count = counts_list[len(counts_list) // 2] if counts_list else 50
    print(f"Median annotations per category: {median_count}")

    # Determine how many synthetic images to generate per product
    # More for rare categories, fewer for common ones
    target_per_cat = max(median_count, 30)

    # Load shelf images for backgrounds
    shelf_images = sorted(images_dir.glob("*.jpg"))[:20]  # Use subset for backgrounds
    shelf_imgs = [Image.open(p).convert("RGB") for p in shelf_images]
    print(f"Loaded {len(shelf_imgs)} shelf images for backgrounds")

    # First, copy the original YOLO dataset
    src_dataset = Path("data/yolo_exp_imgsz1536")
    if src_dataset.exists():
        print(f"Copying base dataset from {src_dataset}...")
        for split in ["train", "val"]:
            for subdir in ["images", "labels"]:
                src = src_dataset / subdir / split
                dst = output_dir / subdir / split
                dst.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    for f in src.iterdir():
                        shutil.copy2(f, dst / f.name)
        # Copy dataset.yaml
        shutil.copy2(src_dataset / "dataset.yaml", output_dir / "dataset.yaml")
    else:
        print("ERROR: Base dataset not found, run train_experiment.py first")
        return

    # Generate synthetic training images
    synthetic_count = 0
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"

    for product_code, cat_id in code_to_catid.items():
        current_count = ann_counts.get(cat_id, 0)
        needed = max(0, target_per_cat - current_count)
        # Cap at reasonable number
        needed = min(needed, 20)
        if needed == 0:
            continue

        # Load reference images for this product
        prod_dir = ref_dir / product_code
        if not prod_dir.exists():
            continue
        ref_paths = list(prod_dir.glob("*.jpg"))
        if not ref_paths:
            continue

        for i in range(needed):
            ref_path = random.choice(ref_paths)
            ref_img = Image.open(ref_path).convert("RGB")
            bg_img = random.choice(shelf_imgs)

            # Vary target size based on typical product sizes
            target_w = random.randint(50, 120)
            target_h = random.randint(60, 160)

            synth_img, (cx, cy, w, h) = create_synthetic_image(
                bg_img, ref_img, target_size=(target_w, target_h)
            )

            # Save
            fname = f"synth_{product_code}_{i:03d}"
            synth_img.save(train_img_dir / f"{fname}.jpg", quality=90)

            label_line = f"{cat_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            (train_lbl_dir / f"{fname}.txt").write_text(label_line)
            synthetic_count += 1

    # Update dataset.yaml with correct path
    yaml_path = output_dir / "dataset.yaml"
    nc = len(coco["categories"])
    names = {c["id"]: c["name"] for c in coco["categories"]}
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    total_train = len(list(train_img_dir.glob("*.jpg")))
    print(f"\nGenerated {synthetic_count} synthetic images")
    print(f"Total training images: {total_train}")
    print(f"Dataset ready at: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
