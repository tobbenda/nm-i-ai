import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


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

    base = Path(__file__).parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load 3 models: A@2048, A@1536, B@1536
    models = [
        (ort.InferenceSession(str(base / "model_a_2048.onnx"), providers=providers), 2048),
        (ort.InferenceSession(str(base / "model_a_1536.onnx"), providers=providers), 1536),
        (ort.InferenceSession(str(base / "model_b_1536.onnx"), providers=providers), 1536),
    ]
    weights = [1.0, 1.0, 0.8]

    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_arr.shape[:2]

        boxes_list, scores_list, labels_list = [], [], []
        for session, imgsz in models:
            inp_name = session.get_inputs()[0].name
            inp, scale, pad_x, pad_y = preprocess(img_arr, imgsz)
            out = session.run(None, {inp_name: inp})
            b, s, c = decode(out[0], scale, pad_x, pad_y, w, h, 0.01)
            boxes_list.append(b.tolist() if len(b) > 0 else [])
            scores_list.append(s.tolist() if len(s) > 0 else [])
            labels_list.append(c.tolist() if len(c) > 0 else [])

        boxes_f, scores_f, labels_f = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
