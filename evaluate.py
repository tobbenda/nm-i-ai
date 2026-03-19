"""
Local evaluation + HTML error report for NorgesGruppen Object Detection.

Usage:
    python evaluate.py --predictions output/predictions.json \
                       --annotations data/coco_dataset/train/annotations.json \
                       --images data/coco_dataset/train/images \
                       --output report.html

Produces an HTML report with:
  1. mAP scores (detection + classification, matching competition formula)
  2. Visual overlays (GT, TP, FP, missed)
  3. Error categorization
  4. Per-image scorecard (worst → best)
  5. Size/position heatmap
  6. Confidence distribution (TP vs FP)
"""

import argparse
import json
import base64
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_iou(box_a, box_b):
    """Compute IoU between two COCO-format boxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    xi = max(ax, bx)
    yi = max(ay, by)
    xa = min(ax + aw, bx + bw)
    ya = min(ay + ah, by + bh)
    if xi >= xa or yi >= ya:
        return 0.0
    inter = (xa - xi) * (ya - yi)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_thresh=0.5, check_category=False):
    """Match predictions to ground truths. Returns lists of TP/FP/missed."""
    preds_sorted = sorted(preds, key=lambda p: p["score"], reverse=True)
    gt_matched = [False] * len(gts)
    tps, fps = [], []

    for pred in preds_sorted:
        best_iou = 0
        best_idx = -1
        for j, gt in enumerate(gts):
            if gt_matched[j]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            cat_ok = (not check_category) or pred["category_id"] == gt["category_id"]
            if iou >= iou_thresh and cat_ok and iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            gt_matched[best_idx] = True
            tps.append({**pred, "_iou": best_iou, "_gt": gts[best_idx]})
        else:
            fps.append(pred)

    missed = [gts[j] for j in range(len(gts)) if not gt_matched[j]]
    return tps, fps, missed


def compute_ap(tps_count, fps_count, n_gt):
    """Compute Average Precision from cumulative TP/FP counts."""
    if n_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tps_count)
    fp_cum = np.cumsum(fps_count)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    # Append sentinel values
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def compute_map(all_preds, all_gts, check_category=False):
    """Compute mAP@0.5 across all images."""
    # Sort all predictions by score
    scored_preds = []
    for img_id, preds in all_preds.items():
        for p in preds:
            scored_preds.append((p["score"], img_id, p))
    scored_preds.sort(key=lambda x: x[0], reverse=True)

    total_gt = sum(len(gts) for gts in all_gts.values())
    if total_gt == 0:
        return 0.0

    gt_matched = {img_id: [False] * len(gts) for img_id, gts in all_gts.items()}
    tp_list, fp_list = [], []

    for score, img_id, pred in scored_preds:
        gts = all_gts.get(img_id, [])
        best_iou = 0
        best_idx = -1
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


def draw_overlay(img_path, tps, fps, missed, max_width=800):
    """Draw boxes on image and return base64-encoded JPEG."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Missed GT boxes (yellow, dashed-ish with double lines)
    for gt in missed:
        x, y, w, h = gt["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="#FFD700", width=2)
        draw.rectangle([x + 3, y + 3, x + w - 3, y + h - 3], outline="#FFD700", width=1)

    # TP boxes (blue)
    for tp in tps:
        x, y, w, h = tp["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="#00AAFF", width=3)

    # FP boxes (red)
    for fp in fps:
        x, y, w, h = fp["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="#FF3333", width=2)

    # Resize for report
    ratio = max_width / img.width
    img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def generate_html(results, output_path):
    """Generate full HTML report."""
    det_map = results["detection_map"]
    cls_map = results["classification_map"]
    final_score = results["final_score"]
    per_image = results["per_image"]
    all_tps = results["all_tps"]
    all_fps = results["all_fps"]
    global_stats = results["global_stats"]

    # --- Confidence histogram data ---
    tp_confs = [t["score"] for t in all_tps]
    fp_confs = [f["score"] for f in all_fps]

    def histogram_svg(tp_confs, fp_confs, width=600, height=200):
        bins = np.linspace(0, 1, 21)
        tp_hist, _ = np.histogram(tp_confs, bins) if tp_confs else (np.zeros(20), bins)
        fp_hist, _ = np.histogram(fp_confs, bins) if fp_confs else (np.zeros(20), bins)
        max_val = max(tp_hist.max(), fp_hist.max(), 1)
        bar_w = width / 20 / 2 - 1

        bars = ""
        for i in range(20):
            x_tp = i * (width / 20) + 1
            x_fp = x_tp + bar_w + 1
            h_tp = tp_hist[i] / max_val * (height - 30)
            h_fp = fp_hist[i] / max_val * (height - 30)
            bars += f'<rect x="{x_tp:.0f}" y="{height - 30 - h_tp:.0f}" width="{bar_w:.0f}" height="{h_tp:.0f}" fill="#00AAFF" opacity="0.7"/>'
            bars += f'<rect x="{x_fp:.0f}" y="{height - 30 - h_fp:.0f}" width="{bar_w:.0f}" height="{h_fp:.0f}" fill="#FF3333" opacity="0.7"/>'
            if i % 5 == 0:
                bars += f'<text x="{x_tp:.0f}" y="{height - 5}" font-size="11" fill="#888">{bins[i]:.1f}</text>'
        bars += f'<text x="{width - 20}" y="{height - 5}" font-size="11" fill="#888">1.0</text>'

        return f'''<svg width="{width}" height="{height}" style="background:#1a1a2e;border-radius:8px;padding:10px">
            {bars}
            <rect x="10" y="5" width="12" height="12" fill="#00AAFF" opacity="0.7"/>
            <text x="26" y="15" font-size="12" fill="#ccc">TP ({len(tp_confs)})</text>
            <rect x="100" y="5" width="12" height="12" fill="#FF3333" opacity="0.7"/>
            <text x="116" y="15" font-size="12" fill="#ccc">FP ({len(fp_confs)})</text>
        </svg>'''

    # --- Size heatmap (missed vs detected by box area) ---
    def size_analysis_svg(all_tps, all_missed, width=600, height=200):
        tp_areas = [t["bbox"][2] * t["bbox"][3] for t in all_tps]
        missed_areas = [m["bbox"][2] * m["bbox"][3] for m in all_missed]
        if not tp_areas and not missed_areas:
            return "<p>No data</p>"
        all_areas = tp_areas + missed_areas
        max_area = max(all_areas) if all_areas else 1
        bins = np.linspace(0, max_area, 11)
        tp_hist, _ = np.histogram(tp_areas, bins) if tp_areas else (np.zeros(10), bins)
        ms_hist, _ = np.histogram(missed_areas, bins) if missed_areas else (np.zeros(10), bins)
        max_val = max(tp_hist.max(), ms_hist.max(), 1)
        bar_w = width / 10 / 2 - 1

        bars = ""
        for i in range(10):
            x_tp = i * (width / 10) + 1
            x_ms = x_tp + bar_w + 1
            h_tp = tp_hist[i] / max_val * (height - 30)
            h_ms = ms_hist[i] / max_val * (height - 30)
            bars += f'<rect x="{x_tp:.0f}" y="{height - 30 - h_tp:.0f}" width="{bar_w:.0f}" height="{h_tp:.0f}" fill="#00AAFF" opacity="0.7"/>'
            bars += f'<rect x="{x_ms:.0f}" y="{height - 30 - h_ms:.0f}" width="{bar_w:.0f}" height="{h_ms:.0f}" fill="#FFD700" opacity="0.7"/>'
            if i % 2 == 0:
                bars += f'<text x="{x_tp:.0f}" y="{height - 5}" font-size="10" fill="#888">{bins[i]:.0f}</text>'

        return f'''<svg width="{width}" height="{height}" style="background:#1a1a2e;border-radius:8px;padding:10px">
            {bars}
            <rect x="10" y="5" width="12" height="12" fill="#00AAFF" opacity="0.7"/>
            <text x="26" y="15" font-size="12" fill="#ccc">Detected ({len(tp_areas)})</text>
            <rect x="170" y="5" width="12" height="12" fill="#FFD700" opacity="0.7"/>
            <text x="186" y="15" font-size="12" fill="#ccc">Missed ({len(missed_areas)})</text>
        </svg>'''

    # --- Position heatmap ---
    def position_heatmap_svg(all_missed, img_w=2000, img_h=1500, width=400, height=300):
        if not all_missed:
            return "<p>No missed detections</p>"
        grid_cols, grid_rows = 10, 8
        grid = np.zeros((grid_rows, grid_cols))
        for m in all_missed:
            cx = m["bbox"][0] + m["bbox"][2] / 2
            cy = m["bbox"][1] + m["bbox"][3] / 2
            col = min(int(cx / img_w * grid_cols), grid_cols - 1)
            row = min(int(cy / img_h * grid_rows), grid_rows - 1)
            grid[row][col] += 1

        max_val = grid.max() if grid.max() > 0 else 1
        cell_w = width / grid_cols
        cell_h = height / grid_rows
        rects = ""
        for r in range(grid_rows):
            for c in range(grid_cols):
                intensity = grid[r][c] / max_val
                red = int(255 * intensity)
                rects += f'<rect x="{c * cell_w:.0f}" y="{r * cell_h:.0f}" width="{cell_w:.0f}" height="{cell_h:.0f}" fill="rgb({red},0,0)" stroke="#333" stroke-width="1"/>'
                if grid[r][c] > 0:
                    rects += f'<text x="{c * cell_w + cell_w / 2:.0f}" y="{r * cell_h + cell_h / 2 + 4:.0f}" font-size="11" fill="white" text-anchor="middle">{int(grid[r][c])}</text>'

        return f'''<svg width="{width}" height="{height}" style="background:#1a1a2e;border-radius:8px">
            {rects}
        </svg>'''

    all_missed = results["all_missed"]

    # --- Build per-image rows ---
    per_image_sorted = sorted(per_image, key=lambda x: x["det_recall"])
    image_rows = ""
    for img in per_image_sorted:
        recall_pct = img["det_recall"] * 100
        color = "#FF3333" if recall_pct < 20 else "#FFD700" if recall_pct < 50 else "#00AAFF"
        image_rows += f'''
        <div class="image-card">
            <div class="image-header">
                <span class="image-name">img_{img["image_id"]:05d}.jpg</span>
                <span class="recall-badge" style="background:{color}">{recall_pct:.0f}% recall</span>
                <span class="stats">TP:{img["tp"]} FP:{img["fp"]} Missed:{img["missed"]} GT:{img["n_gt"]}</span>
            </div>
            <img src="data:image/jpeg;base64,{img["overlay_b64"]}" style="width:100%;border-radius:4px"/>
        </div>'''

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Detection Evaluation Report</title>
<style>
    body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
    h2 {{ color: #79c0ff; margin-top: 40px; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
    .score-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; text-align: center; }}
    .score-value {{ font-size: 48px; font-weight: 700; margin: 10px 0; }}
    .score-label {{ font-size: 14px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
    .stat-value {{ font-size: 28px; font-weight: 600; }}
    .stat-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
    .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
    .chart-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
    .image-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin: 12px 0; padding: 12px; }}
    .image-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
    .image-name {{ font-weight: 600; font-family: monospace; }}
    .recall-badge {{ padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; color: #0d1117; }}
    .stats {{ color: #8b949e; font-size: 12px; font-family: monospace; }}
    .legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 13px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-dot {{ width: 14px; height: 14px; border-radius: 3px; }}
</style></head>
<body><div class="container">

<h1>Detection Evaluation Report</h1>

<div class="score-grid">
    <div class="score-card">
        <div class="score-label">Detection mAP (70%)</div>
        <div class="score-value" style="color:#00AAFF">{det_map:.3f}</div>
    </div>
    <div class="score-card">
        <div class="score-label">Classification mAP (30%)</div>
        <div class="score-value" style="color:#FFD700">{cls_map:.3f}</div>
    </div>
    <div class="score-card">
        <div class="score-label">Final Score</div>
        <div class="score-value" style="color:{"#00FF88" if final_score > 0.5 else "#FF3333"}">{final_score:.3f}</div>
        <div class="score-label">0.7 × det + 0.3 × cls</div>
    </div>
</div>

<h2>Error Summary</h2>
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value" style="color:#00AAFF">{global_stats["tp"]}</div>
        <div class="stat-label">True Positives</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" style="color:#FF3333">{global_stats["fp"]}</div>
        <div class="stat-label">False Positives</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" style="color:#FFD700">{global_stats["missed"]}</div>
        <div class="stat-label">Missed (FN)</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" style="color:#8b949e">{global_stats["total_gt"]}</div>
        <div class="stat-label">Total GT boxes</div>
    </div>
</div>

<h2>Confidence Distribution (TP vs FP)</h2>
<div class="chart-card">
    {histogram_svg(tp_confs, fp_confs)}
</div>

<h2>Analysis by Size & Position</h2>
<div class="chart-row">
    <div class="chart-card">
        <h3 style="margin-top:0;color:#79c0ff">Box Area: Detected vs Missed</h3>
        {size_analysis_svg(all_tps, all_missed)}
    </div>
    <div class="chart-card">
        <h3 style="margin-top:0;color:#79c0ff">Missed Detections Heatmap (position)</h3>
        {position_heatmap_svg(all_missed)}
    </div>
</div>

<h2>Per-Image Scorecard (worst → best)</h2>
<div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#00AAFF"></div> True Positive</div>
    <div class="legend-item"><div class="legend-dot" style="background:#FF3333"></div> False Positive</div>
    <div class="legend-item"><div class="legend-dot" style="background:#FFD700"></div> Missed (double border)</div>
</div>
{image_rows}

</div></body></html>'''

    Path(output_path).write_text(html)
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Local evaluation + HTML report")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--annotations", required=True, help="Path to COCO annotations.json")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", default="report.html", help="Output HTML path")
    parser.add_argument("--max-images", type=int, default=20, help="Max images in report (for speed)")
    args = parser.parse_args()

    # Load data
    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.annotations) as f:
        coco = json.load(f)

    # Index by image_id
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)

    image_info = {img["id"]: img for img in coco["images"]}

    # Compute mAP
    det_map = compute_map(pred_by_image, dict(gt_by_image), check_category=False)
    cls_map = compute_map(pred_by_image, dict(gt_by_image), check_category=True)
    final_score = 0.7 * det_map + 0.3 * cls_map

    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Final Score:             {final_score:.4f}")

    # Per-image analysis
    all_image_ids = sorted(set(list(gt_by_image.keys()) + list(pred_by_image.keys())))
    per_image = []
    all_tps, all_fps, all_missed = [], [], []

    images_dir = Path(args.images)
    for img_id in all_image_ids:
        gts = gt_by_image.get(img_id, [])
        preds = pred_by_image.get(img_id, [])
        tps, fps, missed = match_predictions(preds, gts, iou_thresh=0.5)

        all_tps.extend(tps)
        all_fps.extend(fps)
        all_missed.extend(missed)

        det_recall = len(tps) / len(gts) if gts else 1.0

        info = image_info.get(img_id, {})
        fname = info.get("file_name", f"img_{img_id:05d}.jpg")
        img_path = images_dir / fname

        overlay_b64 = ""
        if img_path.exists() and len(per_image) < args.max_images:
            overlay_b64 = draw_overlay(img_path, tps, fps, missed)

        per_image.append({
            "image_id": img_id,
            "tp": len(tps),
            "fp": len(fps),
            "missed": len(missed),
            "n_gt": len(gts),
            "det_recall": det_recall,
            "overlay_b64": overlay_b64,
        })

    global_stats = {
        "tp": len(all_tps),
        "fp": len(all_fps),
        "missed": len(all_missed),
        "total_gt": len(all_tps) + len(all_missed),
    }

    results = {
        "detection_map": det_map,
        "classification_map": cls_map,
        "final_score": final_score,
        "per_image": [img for img in per_image if img["overlay_b64"]],
        "all_tps": all_tps,
        "all_fps": all_fps,
        "all_missed": all_missed,
        "global_stats": global_stats,
    }

    generate_html(results, args.output)


if __name__ == "__main__":
    main()
