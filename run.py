import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

TIMEOUT_S = 300
TIME_RESERVE_S = 15


def preprocess(img_arr, imgsz=1536):
    """Letterbox resize + normalize for YOLO ONNX."""
    orig_h, orig_w = img_arr.shape[:2]
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    img = Image.fromarray(img_arr).resize((new_w, new_h), Image.BILINEAR)
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(img)

    arr = padded.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float16)
    return arr, scale, pad_x, pad_y


def decode(output, scale, pad_x, pad_y, img_w, img_h, conf=0.01):
    """Decode ONNX output to normalized [0,1] boxes, scores, class_ids."""
    preds = np.float32(output[0].transpose(1, 0))
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > conf
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_x) / scale
    y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_y) / scale
    x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2 - pad_x) / scale
    y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2 - pad_y) / scale

    boxes_norm = np.stack([
        np.clip(x1 / img_w, 0, 1),
        np.clip(y1 / img_h, 0, 1),
        np.clip(x2 / img_w, 0, 1),
        np.clip(y2 / img_h, 0, 1),
    ], axis=1)

    return boxes_norm, max_scores, class_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    base = Path(__file__).parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load 3 models: A@1536, B@1536, A@1280
    model_configs = [
        (base / "model_a_1536.onnx", 1536),
        (base / "model_b_1536.onnx", 1536),
        (base / "model_a_1280.onnx", 1280),
    ]
    models = []
    for path, imgsz in model_configs:
        try:
            session = ort.InferenceSession(str(path), providers=providers)
            models.append((session, imgsz))
            print(f"Loaded {path.name}")
        except Exception as e:
            print(f"WARNING: Failed to load {path.name}: {e}")
    if not models:
        print("ERROR: No models loaded, writing empty predictions")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    weights = [1.0, 0.8, 0.6][:len(models)]
    predictions = []
    deadline = t_start + TIMEOUT_S - TIME_RESERVE_S

    image_paths = sorted(
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_idx, img_path in enumerate(image_paths):
        if time.time() > deadline:
            print(f"WARNING: Time budget exceeded after {img_idx}/{len(image_paths)} images, writing partial results")
            break

        try:
            image_id = int(img_path.stem.split("_")[-1])
            img_arr = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_arr.shape[:2]
            img_flipped = img_arr[:, ::-1, :].copy()

            boxes_list, scores_list, labels_list = [], [], []

            # Normal passes
            for session, imgsz in models:
                inp_name = session.get_inputs()[0].name
                inp, scale, pad_x, pad_y = preprocess(img_arr, imgsz)
                out = session.run(None, {inp_name: inp})
                b, s, c = decode(out[0], scale, pad_x, pad_y, w, h, 0.01)
                boxes_list.append(b.tolist() if len(b) > 0 else [])
                scores_list.append(s.tolist() if len(s) > 0 else [])
                labels_list.append(c.tolist() if len(c) > 0 else [])

            # Horizontal flip TTA on first model
            flip_session, flip_imgsz = models[0]
            inp_name = flip_session.get_inputs()[0].name
            inp, scale, pad_x, pad_y = preprocess(img_flipped, flip_imgsz)
            out = flip_session.run(None, {inp_name: inp})
            b_flip, s_flip, c_flip = decode(out[0], scale, pad_x, pad_y, w, h, 0.01)
            if len(b_flip) > 0:
                b_mirrored = b_flip.copy()
                b_mirrored[:, 0] = 1 - b_flip[:, 2]
                b_mirrored[:, 2] = 1 - b_flip[:, 0]
                b_flip = b_mirrored
            boxes_list.append(b_flip.tolist() if len(b_flip) > 0 else [])
            scores_list.append(s_flip.tolist() if len(s_flip) > 0 else [])
            labels_list.append(c_flip.tolist() if len(c_flip) > 0 else [])

            all_weights = weights + [0.5]

            boxes_f, scores_f, labels_f = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=all_weights,
                iou_thr=0.5,
                skip_box_thr=0.001,
            )

            for i in range(len(boxes_f)):
                x1, y1, x2, y2 = boxes_f[i]
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(labels_f[i]),
                    "bbox": [round(x1 * w, 1), round(y1 * h, 1),
                             round((x2 - x1) * w, 1), round((y2 - y1) * h, 1)],
                    "score": round(float(scores_f[i]), 3),
                })
        except Exception as e:
            print(f"WARNING: Failed on {img_path.name}: {e}")
            continue

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    elapsed = time.time() - t_start
    print(f"Wrote {len(predictions)} predictions for {len(image_paths)} images in {elapsed:.1f}s to {args.output}")


if __name__ == "__main__":
    main()
