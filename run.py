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


def preprocess_array(img_arr, imgsz=1536):
    """Preprocess a numpy RGB array (H,W,3) for YOLO ONNX."""
    orig_h, orig_w = img_arr.shape[:2]
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    img = Image.fromarray(img_arr)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(img_resized)

    arr = padded.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr, scale, pad_x, pad_y


def decode_raw(output, scale, pad_x, pad_y, conf_threshold=0.01):
    """Decode ONNX output to boxes (x1,y1,x2,y2 in model coords), scores, class_ids."""
    preds = output[0].transpose(1, 0)
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = max_scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Undo letterbox
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, max_scores, class_ids


def detect_full(session, input_name, img_arr, imgsz=1536, conf=0.01):
    """Run detection on the full image."""
    inp, scale, pad_x, pad_y = preprocess_array(img_arr, imgsz)
    outputs = session.run(None, {input_name: inp})
    boxes, scores, class_ids = decode_raw(outputs[0], scale, pad_x, pad_y, conf)

    orig_h, orig_w = img_arr.shape[:2]
    if len(boxes) > 0:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    return boxes, scores, class_ids


def detect_tiled(session, input_name, img_arr, tile_size=1024,
                 overlap=0.2, imgsz=1536, conf=0.01):
    """Run detection on overlapping tiles, offset coords back to full image."""
    orig_h, orig_w = img_arr.shape[:2]
    stride = int(tile_size * (1 - overlap))

    all_boxes, all_scores, all_class_ids = [], [], []

    for y0 in range(0, orig_h, stride):
        for x0 in range(0, orig_w, stride):
            x1 = min(x0, orig_w - tile_size) if x0 + tile_size > orig_w else x0
            y1 = min(y0, orig_h - tile_size) if y0 + tile_size > orig_h else y0
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x1 + tile_size, orig_w)
            y2 = min(y1 + tile_size, orig_h)

            tile = img_arr[y1:y2, x1:x2]
            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue

            inp, scale, pad_x, pad_y = preprocess_array(tile, imgsz)
            outputs = session.run(None, {input_name: inp})
            boxes, scores, class_ids = decode_raw(
                outputs[0], scale, pad_x, pad_y, conf
            )

            if len(boxes) > 0:
                # Offset to full image coordinates
                boxes[:, 0] = np.clip(boxes[:, 0] + x1, 0, orig_w)
                boxes[:, 1] = np.clip(boxes[:, 1] + y1, 0, orig_h)
                boxes[:, 2] = np.clip(boxes[:, 2] + x1, 0, orig_w)
                boxes[:, 3] = np.clip(boxes[:, 3] + y1, 0, orig_h)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)

    if all_boxes:
        return (np.concatenate(all_boxes),
                np.concatenate(all_scores),
                np.concatenate(all_class_ids))
    return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)


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

        # Full-image pass
        boxes_full, scores_full, cls_full = detect_full(
            session, input_name, img_arr, imgsz=1536, conf=0.01
        )

        # Tiled pass
        boxes_tile, scores_tile, cls_tile = detect_tiled(
            session, input_name, img_arr,
            tile_size=1024, overlap=0.2, imgsz=1536, conf=0.01
        )

        # Merge all detections
        if len(boxes_full) > 0 and len(boxes_tile) > 0:
            all_boxes = np.concatenate([boxes_full, boxes_tile])
            all_scores = np.concatenate([scores_full, scores_tile])
            all_cls = np.concatenate([cls_full, cls_tile])
        elif len(boxes_full) > 0:
            all_boxes, all_scores, all_cls = boxes_full, scores_full, cls_full
        elif len(boxes_tile) > 0:
            all_boxes, all_scores, all_cls = boxes_tile, scores_tile, cls_tile
        else:
            continue

        # Final NMS to merge overlapping detections
        keep = nms(all_boxes, all_scores, iou_threshold=0.5)
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_cls = all_cls[keep]

        for i in range(len(all_boxes)):
            bx1, by1, bx2, by2 = all_boxes[i]
            predictions.append({
                "image_id": image_id,
                "category_id": int(all_cls[i]),
                "bbox": [round(float(bx1), 1), round(float(by1), 1),
                         round(float(bx2 - bx1), 1), round(float(by2 - by1), 1)],
                "score": round(float(all_scores[i]), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
