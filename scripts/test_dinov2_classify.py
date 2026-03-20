"""
Test DINOv2 reclassification: YOLO detects, DINOv2 classifies crops.
Uses pre-existing predictions and evaluates on val set.
"""
import json
import numpy as np
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def compute_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi, yi = max(ax, bx), max(ay, by)
    xa, ya = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi >= xa or yi >= ya:
        return 0.0
    inter = (xa - xi) * (ya - yi)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compute_map(pbi, gbi, cc=False):
    scored = [(p["score"], i, p) for i, ps in pbi.items() for p in ps]
    scored.sort(key=lambda x: x[0], reverse=True)
    tg = sum(len(g) for g in gbi.values())
    if tg == 0:
        return 0.0
    gm = {i: [False] * len(g) for i, g in gbi.items()}
    tp, fp = [], []
    for s, i, p in scored:
        gs = gbi.get(i, [])
        bi, bj = 0, -1
        for j, g in enumerate(gs):
            if gm[i][j]:
                continue
            iou = compute_iou(p["bbox"], g["bbox"])
            co = (not cc) or p["category_id"] == g["category_id"]
            if iou >= 0.5 and co and iou > bi:
                bi = iou
                bj = j
        if bj >= 0:
            gm[i][bj] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tc = np.cumsum(tp)
    fc = np.cumsum(fp)
    r = tc / tg
    pr = tc / (tc + fc)
    r = np.concatenate([[0], r, [1]])
    pr = np.concatenate([[1], pr, [0]])
    for i in range(len(pr) - 2, -1, -1):
        pr[i] = max(pr[i], pr[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * pr[idx + 1]))


def main():
    # Load annotations
    with open("data/train/annotations.json") as f:
        coco = json.load(f)
    all_images = sorted(coco["images"], key=lambda x: x["id"])
    val_imgs = all_images[int(len(all_images) * 0.8) :]
    val_ids = {img["id"] for img in val_imgs}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] in val_ids:
            gt_by_image[ann["image_id"]].append(ann)

    # Load YOLO predictions
    with open("/tmp/onnx_preds.json") as f:
        all_preds = json.load(f)
    val_preds_by_img = defaultdict(list)
    for p in all_preds:
        if p["image_id"] in val_ids:
            val_preds_by_img[p["image_id"]].append(p)

    # Load gallery
    gallery = np.load("gallery.npz")
    gallery_embs = gallery["embeddings"]
    gallery_catids = gallery["catids"]

    # Load DINOv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
    )
    dino = dino.eval().to(device)
    dino_config = resolve_data_config(dino.pretrained_cfg)
    dino_transform = create_transform(**dino_config)
    print(f"DINOv2 loaded on {device}")

    # Test different similarity thresholds
    for sim_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        t0 = time.time()
        reclassified = defaultdict(list)
        changed = 0
        total = 0

        for img_info in val_imgs:
            img_path = Path("data/train/images") / img_info["file_name"]
            if not img_path.exists():
                continue
            img_id = img_info["id"]
            preds = val_preds_by_img.get(img_id, [])
            if not preds:
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Batch process crops
            crops = []
            valid_preds = []
            for pred in preds:
                bx, by, bw, bh = pred["bbox"]
                pad = 5
                x1 = max(0, int(bx - pad))
                y1 = max(0, int(by - pad))
                x2 = min(w, int(bx + bw + pad))
                y2 = min(h, int(by + bh + pad))
                if x2 <= x1 or y2 <= y1:
                    reclassified[img_id].append(pred)
                    continue
                crop = img.crop((x1, y1, x2, y2))
                crops.append(dino_transform(crop))
                valid_preds.append(pred)

            if not crops:
                continue

            # Batch embed
            batch = torch.stack(crops).to(device)
            with torch.no_grad():
                embs = dino(batch).cpu().numpy()

            # L2 normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.clip(norms, 1e-8, None)

            # Match against gallery
            sims = embs @ gallery_embs.T  # (N, 326)
            best_idxs = sims.argmax(axis=1)
            best_sims = sims[np.arange(len(sims)), best_idxs]

            for i, pred in enumerate(valid_preds):
                total += 1
                if best_sims[i] > sim_thresh:
                    cat_id = int(gallery_catids[best_idxs[i]])
                    if cat_id != pred["category_id"]:
                        changed += 1
                else:
                    cat_id = pred["category_id"]

                reclassified[img_id].append(
                    {
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": pred["bbox"],
                        "score": pred["score"],
                    }
                )

        elapsed = time.time() - t0
        det = compute_map(dict(reclassified), dict(gt_by_image), False)
        cls = compute_map(dict(reclassified), dict(gt_by_image), True)
        det_old = compute_map(dict(val_preds_by_img), dict(gt_by_image), False)
        cls_old = compute_map(dict(val_preds_by_img), dict(gt_by_image), True)

        print(
            f"thresh={sim_thresh:.1f}: det={det:.4f} cls={cls:.4f} "
            f"final={0.7*det+0.3*cls:.4f} "
            f"(Δcls={cls-cls_old:+.4f} Δfinal={0.7*det+0.3*cls-(0.7*det_old+0.3*cls_old):+.4f}) "
            f"changed={changed}/{total} time={elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
