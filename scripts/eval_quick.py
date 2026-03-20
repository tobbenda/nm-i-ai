"""
Quick local evaluation — outputs detection mAP@0.5 on the val split.
Usage: python eval_quick.py --weights best.pt [--imgsz 1280] [--conf 0.1]
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO


def compute_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    xi, yi = max(ax, bx), max(ay, by)
    xa, ya = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xa or yi >= ya:
        return 0.0
    inter = (xa - xi) * (ya - yi)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compute_ap(tp_list, fp_list, n_gt):
    if n_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]))


def compute_map(pred_by_image, gt_by_image, check_category=False):
    scored_preds = []
    for img_id, preds in pred_by_image.items():
        for p in preds:
            scored_preds.append((p["score"], img_id, p))
    scored_preds.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(gts) for gts in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    gt_matched = {img_id: [False] * len(gts) for img_id, gts in gt_by_image.items()}
    tp_list, fp_list = [], []

    for score, img_id, pred in scored_preds:
        gts = gt_by_image.get(img_id, [])
        best_iou, best_idx = 0, -1
        for j, gt in enumerate(gts):
            if gt_matched[img_id][j]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            cat_ok = (not check_category) or pred["category_id"] == gt["category_id"]
            if iou >= 0.5 and cat_ok and iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            gt_matched[img_id][best_idx] = True
            tp_list.append(1)
            fp_list.append(0)
        else:
            tp_list.append(0)
            fp_list.append(1)

    return compute_ap(tp_list, fp_list, total_gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--annotations", default="data/train/annotations.json")
    parser.add_argument("--images", default="data/train/images")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of images to use as val (from end)")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--iou-nms", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()

    # Load annotations
    with open(args.annotations) as f:
        coco = json.load(f)

    # Split val images (same split as training)
    all_images = sorted(coco["images"], key=lambda x: x["id"])
    split_idx = int(len(all_images) * (1 - args.val_split))
    val_images = all_images[split_idx:]
    val_ids = {img["id"] for img in val_images}

    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] in val_ids:
            gt_by_image[ann["image_id"]].append(ann)

    # Run inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    pred_by_image = defaultdict(list)
    images_dir = Path(args.images)

    print(f"Evaluating {args.weights} on {len(val_images)} val images...")
    print(f"  imgsz={args.imgsz}, conf={args.conf}, iou_nms={args.iou_nms}, tta={args.tta}")

    for img_info in val_images:
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        image_id = img_info["id"]
        results = model(
            str(img_path), device=device, verbose=False,
            conf=args.conf, imgsz=args.imgsz, iou=args.iou_nms,
            augment=args.tta,
        )
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                pred_by_image[image_id].append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(r.boxes.conf[i].item()),
                })

    # Compute metrics
    det_map = compute_map(dict(pred_by_image), dict(gt_by_image), check_category=False)
    cls_map = compute_map(dict(pred_by_image), dict(gt_by_image), check_category=True)
    final = 0.7 * det_map + 0.3 * cls_map

    total_preds = sum(len(p) for p in pred_by_image.values())
    total_gt = sum(len(g) for g in gt_by_image.values())

    print(f"\n{'='*50}")
    print(f"  Detection mAP@0.5:       {det_map:.4f}")
    print(f"  Classification mAP@0.5:  {cls_map:.4f}")
    print(f"  Final Score:             {final:.4f}")
    print(f"  Predictions: {total_preds}  |  GT: {total_gt}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
