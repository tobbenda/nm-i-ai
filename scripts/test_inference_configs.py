"""
Test various inference configurations on val set:
- WBF iou_thr tuning
- Soft-NMS
- Horizontal flip TTA as additional WBF input
- Adding fulldata_1280 model as 4th ensemble member
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

# Load annotations
with open("data/train/annotations.json") as f:
    coco = json.load(f)
all_images = sorted(coco["images"], key=lambda x: x["id"])
val_imgs = all_images[int(len(all_images) * 0.8):]
val_ids = {img["id"] for img in val_imgs}
gt_by_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["image_id"] in val_ids:
        gt_by_image[ann["image_id"]].append(ann)


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


prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_model(path):
    return ort.InferenceSession(path, providers=prov)


def preprocess(img_arr, imgsz, dtype=np.float16):
    h, w = img_arr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    img = Image.fromarray(img_arr).resize((nw, nh), Image.BILINEAR)
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    padded[py:py + nh, px:px + nw] = np.array(img)
    arr = padded.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(dtype)
    return arr, scale, px, py


def decode(output, scale, pad_x, pad_y, img_w, img_h, conf=0.01):
    preds = np.float32(output[0].transpose(1, 0))
    bx = preds[:, :4]
    sc = preds[:, 4:]
    ms = sc.max(axis=1)
    cids = sc.argmax(axis=1)
    mask = ms > conf
    bx, ms, cids = bx[mask], ms[mask], cids[mask]
    if len(bx) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    x1 = np.clip((bx[:, 0] - bx[:, 2] / 2 - pad_x) / scale / img_w, 0, 1)
    y1 = np.clip((bx[:, 1] - bx[:, 3] / 2 - pad_y) / scale / img_h, 0, 1)
    x2 = np.clip((bx[:, 0] + bx[:, 2] / 2 - pad_x) / scale / img_w, 0, 1)
    y2 = np.clip((bx[:, 1] + bx[:, 3] / 2 - pad_y) / scale / img_h, 0, 1)
    return np.stack([x1, y1, x2, y2], axis=1), ms, cids


def run_model(session, img_arr, imgsz, conf=0.01, flip=False):
    h, w = img_arr.shape[:2]
    if flip:
        img_arr = img_arr[:, ::-1, :].copy()
    inp, scale, px, py = preprocess(img_arr, imgsz)
    name = session.get_inputs()[0].name
    out = session.run(None, {name: inp})
    b, s, c = decode(out[0], scale, px, py, w, h, conf)
    if flip and len(b) > 0:
        # Mirror boxes back: x1_new = 1 - x2_old, x2_new = 1 - x1_old
        b_flipped = b.copy()
        b_flipped[:, 0] = 1 - b[:, 2]
        b_flipped[:, 2] = 1 - b[:, 0]
        b = b_flipped
    return b, s, c


def run_ensemble(models_configs, img_arr, weights, iou_thr=0.5):
    h, w = img_arr.shape[:2]
    bl, sl, ll = [], [], []
    for session, imgsz, flip in models_configs:
        b, s, c = run_model(session, img_arr, imgsz, 0.01, flip)
        bl.append(b.tolist() if len(b) > 0 else [])
        sl.append(s.tolist() if len(s) > 0 else [])
        ll.append(c.tolist() if len(c) > 0 else [])
    bf, sf, lf = weighted_boxes_fusion(
        bl, sl, ll, weights=weights, iou_thr=iou_thr, skip_box_thr=0.001
    )
    results = []
    for i in range(len(bf)):
        x1, y1, x2, y2 = bf[i]
        results.append({
            "category_id": int(lf[i]),
            "bbox": [round(x1 * w, 1), round(y1 * h, 1),
                     round((x2 - x1) * w, 1), round((y2 - y1) * h, 1)],
            "score": round(float(sf[i]), 3),
        })
    return results


# Load models
print("Loading models...")
sA = load_model("model_a_1536.onnx")
sB = load_model("model_b_1536.onnx")
sA1280 = load_model("model_a_1280.onnx")

# Also try fulldata_1280 if available
try:
    sC = load_model("runs/yolo_fulldata_1280/weights/best.onnx")
    has_model_c = True
    print("Also loaded fulldata_1280 model")
except:
    has_model_c = False
    print("fulldata_1280 model not available")

print("Running experiments...")

configs = {
    # Current best: A@1536 + B@1536 + A@1280
    "baseline (iou=0.5)": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False)],
        [1.0, 0.8, 0.6], 0.5
    ),
    # WBF iou_thr tuning
    "iou=0.4": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False)],
        [1.0, 0.8, 0.6], 0.4
    ),
    "iou=0.6": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False)],
        [1.0, 0.8, 0.6], 0.6
    ),
    # Weight tuning
    "equal weights": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False)],
        [1.0, 1.0, 1.0], 0.5
    ),
    "A heavy": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False)],
        [1.0, 0.5, 0.3], 0.5
    ),
    # Flip TTA as 4th input
    "3-model + flip A": (
        [(sA, 1536, False), (sB, 1536, False), (sA1280, 1280, False), (sA, 1536, True)],
        [1.0, 0.8, 0.6, 0.5], 0.5
    ),
    # 2-model only (for comparison)
    "A@1536 + A@1280 only": (
        [(sA, 1536, False), (sA1280, 1280, False)],
        [1.0, 0.8], 0.5
    ),
}

for label, (models, weights, iou_thr) in configs.items():
    pred = defaultdict(list)
    for img_info in val_imgs:
        img_path = Path("data/train/images") / img_info["file_name"]
        if not img_path.exists():
            continue
        img_id = img_info["id"]
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        dets = run_ensemble(models, img_arr, weights, iou_thr)
        for d in dets:
            d["image_id"] = img_id
            pred[img_id].append(d)

    det = compute_map(dict(pred), dict(gt_by_image), False)
    cls = compute_map(dict(pred), dict(gt_by_image), True)
    print(f"{label:30s}: det={det:.4f} cls={cls:.4f} final={0.7*det+0.3*cls:.4f}")
