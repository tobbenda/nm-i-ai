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
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr, scale, pad_x, pad_y


def decode(output, scale, pad_x, pad_y, img_w, img_h, conf=0.01):
    """Decode ONNX output to normalized [0,1] boxes, scores, class_ids."""
    preds = output[0].transpose(1, 0)
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

    # To x1,y1,x2,y2 in original image coords
    x1 = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2 - pad_x) / scale
    y1 = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2 - pad_y) / scale
    x2 = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2 - pad_x) / scale
    y2 = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2 - pad_y) / scale

    # Normalize to [0,1] for WBF
    boxes_norm = np.stack([
        np.clip(x1 / img_w, 0, 1),
        np.clip(y1 / img_h, 0, 1),
        np.clip(x2 / img_w, 0, 1),
        np.clip(y2 / img_h, 0, 1),
    ], axis=1)

    return boxes_norm, max_scores, class_ids


def run_at_scale(session, input_name, img_arr, imgsz, conf=0.01):
    """Run detection at a given image size."""
    h, w = img_arr.shape[:2]
    inp, scale, pad_x, pad_y = preprocess(img_arr, imgsz)
    output = session.run(None, {input_name: inp})
    return decode(output[0], scale, pad_x, pad_y, w, h, conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = str(Path(__file__).parent / "best.onnx")
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_arr.shape[:2]

        # Run at two scales
        b1, s1, c1 = run_at_scale(session, input_name, img_arr, 1536, 0.01)
        b2, s2, c2 = run_at_scale(session, input_name, img_arr, 1280, 0.01)

        boxes_list = [b1.tolist() if len(b1) > 0 else [],
                      b2.tolist() if len(b2) > 0 else []]
        scores_list = [s1.tolist() if len(s1) > 0 else [],
                       s2.tolist() if len(s2) > 0 else []]
        labels_list = [c1.tolist() if len(c1) > 0 else [],
                       c2.tolist() if len(c2) > 0 else []]

        boxes_f, scores_f, labels_f = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[1.0, 0.8],
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
