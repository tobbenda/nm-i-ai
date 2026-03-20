import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression on [x1, y1, x2, y2] boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def preprocess(img_path, imgsz=1536):
    """Load and preprocess image for YOLO ONNX model."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Letterbox resize
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to imgsz x imgsz
    padded = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded.paste(img_resized, (pad_x, pad_y))

    # To numpy BCHW float32 [0, 1]
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    return arr, scale, pad_x, pad_y, orig_w, orig_h


def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h,
                conf_threshold=0.01, iou_threshold=0.5):
    """Post-process YOLO ONNX output to COCO-format predictions."""
    # output shape: (1, 360, N) -> transpose to (N, 360)
    preds = output[0].transpose(1, 0)  # (N, 360)

    # Split: first 4 = bbox (cx, cy, w, h), rest = class scores
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    # Get best class per box
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # Convert cx,cy,w,h to x1,y1,x2,y2
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Undo letterbox: remove padding, then unscale
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_x) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_y) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_x) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_y) / scale

    # Clip to image bounds
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # Class-agnostic NMS for detection focus
    keep = nms(boxes_xyxy, max_scores, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]

    # Convert to COCO format [x, y, w, h]
    results = []
    for i in range(len(boxes_xyxy)):
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        results.append({
            "category_id": int(class_ids[i]),
            "bbox": [round(float(bx1), 1), round(float(by1), 1),
                     round(float(bx2 - bx1), 1), round(float(by2 - by1), 1)],
            "score": round(float(max_scores[i]), 3),
        })
    return results


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

        inp, scale, pad_x, pad_y, orig_w, orig_h = preprocess(str(img_path), imgsz=1536)
        outputs = session.run(None, {input_name: inp})

        dets = postprocess(
            outputs[0], scale, pad_x, pad_y, orig_w, orig_h,
            conf_threshold=0.01, iou_threshold=0.5,
        )
        for d in dets:
            d["image_id"] = image_id
            predictions.append(d)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
