from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.damage.detector import Detection


BoxXYXY = tuple[float, float, float, float]


def _normalize_key(key: str) -> str:
    key = str(key)
    # Keep only the final path segment and strip extension
    key = key.replace("\\", "/").split("/")[-1]
    if "." in key:
        key = key.rsplit(".", 1)[0]
    return key


def _to_xyxy(bbox: Any) -> BoxXYXY:
    if isinstance(bbox, dict):
        bbox = bbox.get("bbox") or bbox.get("box") or bbox.get("xyxy")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox: {bbox}")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # If this looks like xywh (common COCO loader path uses xywh), caller should convert.
    return (x1, y1, x2, y2)


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> BoxXYXY:
    return (float(x), float(y), float(x + w), float(y + h))


def iou_xyxy(a: BoxXYXY, b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def load_damage_annotations(path: Path) -> dict[str, list[BoxXYXY]]:
    """Load annotations.

    Supports two formats:

    1) Simple mapping:
       {
         "frame_000001": [[x1,y1,x2,y2], ...],
         "cam0_frame_000010": [{"bbox": [x1,y1,x2,y2]}, ...]
       }

    2) COCO-ish:
       {
         "images": [{"id": 1, "file_name": "frame_000001.png"}, ...],
         "annotations": [{"image_id": 1, "bbox": [x,y,w,h], "category_id": 1}, ...]
       }

    All keys are normalized by stripping folders and extensions.
    """

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    out: dict[str, list[BoxXYXY]] = {}

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        # COCO-ish
        img_id_to_name: dict[int, str] = {}
        for im in data.get("images", []):
            try:
                img_id_to_name[int(im["id"])]= _normalize_key(str(im.get("file_name") or im.get("name") or im.get("path") or im["id"]))
            except Exception:
                continue

        for ann in data.get("annotations", []):
            try:
                img_id = int(ann["image_id"])
                name = img_id_to_name.get(img_id)
                if not name:
                    continue
                bbox = ann.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                x, y, w, h = [float(v) for v in bbox]
                box = _xywh_to_xyxy(x, y, w, h)
                out.setdefault(name, []).append(box)
            except Exception:
                continue
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            nk = _normalize_key(str(k))
            boxes: list[BoxXYXY] = []
            if v is None:
                out[nk] = []
                continue
            if isinstance(v, dict) and "bboxes" in v:
                v = v["bboxes"]
            if isinstance(v, (list, tuple)):
                for item in v:
                    try:
                        boxes.append(_to_xyxy(item))
                    except Exception:
                        continue
            out[nk] = boxes
        return out

    raise ValueError("Unsupported annotation JSON format")


def _group_preds_by_image(preds: Iterable[tuple[str, Detection]]) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for k, d in preds:
        out.setdefault(_normalize_key(k), []).append(d)
    return out


@dataclass
class DetectionSummary:
    ap: float
    recall: float
    precision: float
    n_gt: int
    n_pred: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ap": float(self.ap),
            "recall": float(self.recall),
            "precision": float(self.precision),
            "n_gt": int(self.n_gt),
            "n_pred": int(self.n_pred),
        }


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    # Precision envelope
    mpre = np.maximum.accumulate(precision[::-1])[::-1]
    # Integrate over recall steps
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre2 = np.concatenate([[0.0], mpre, [0.0]])
    # Where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre2[idx + 1])
    return float(ap)


def evaluate_detections(
    preds: dict[str, list[Detection]],
    gts: dict[str, list[BoxXYXY]],
    *,
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> DetectionSummary:
    """Compute single-class AP@IoU and recall/precision at score_thr.

    - AP is computed globally across all images by sorting detections by score.
    - Recall/precision are computed at a fixed score threshold.
    """

    # Normalize keys
    preds_n: dict[str, list[Detection]] = { _normalize_key(k): list(v) for k, v in preds.items() }
    gts_n: dict[str, list[BoxXYXY]] = { _normalize_key(k): list(v) for k, v in gts.items() }

    # Restrict to images that have GT
    pred_items: list[tuple[str, Detection]] = []
    total_gt = 0
    for k, gt_boxes in gts_n.items():
        total_gt += len(gt_boxes)
        for d in preds_n.get(k, []):
            pred_items.append((k, d))

    if total_gt == 0:
        return DetectionSummary(ap=0.0, recall=0.0, precision=0.0, n_gt=0, n_pred=0)

    # Global AP
    pred_items.sort(key=lambda kd: float(kd[1].score), reverse=True)
    matched: dict[str, list[bool]] = {k: [False] * len(gts_n[k]) for k in gts_n.keys()}

    tps: list[float] = []
    fps: list[float] = []

    for k, det in pred_items:
        gt_boxes = gts_n.get(k, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched[k][j]:
                continue
            i = iou_xyxy(det.bbox_xyxy, gt)
            if i > best_iou:
                best_iou = i
                best_j = j
        if best_iou >= float(iou_thr) and best_j >= 0:
            matched[k][best_j] = True
            tps.append(1.0)
            fps.append(0.0)
        else:
            tps.append(0.0)
            fps.append(1.0)

    tp = np.cumsum(np.asarray(tps, dtype=np.float32))
    fp = np.cumsum(np.asarray(fps, dtype=np.float32))
    recall_curve = tp / float(total_gt)
    precision_curve = tp / np.maximum(tp + fp, 1e-12)
    ap = _ap_from_pr(recall_curve, precision_curve)

    # Precision/recall at fixed threshold
    matched_thr: dict[str, list[bool]] = {k: [False] * len(gts_n[k]) for k in gts_n.keys()}
    n_pred_thr = 0
    n_tp_thr = 0

    for k, det in pred_items:
        if float(det.score) < float(score_thr):
            continue
        n_pred_thr += 1
        gt_boxes = gts_n.get(k, [])
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched_thr[k][j]:
                continue
            i = iou_xyxy(det.bbox_xyxy, gt)
            if i > best_iou:
                best_iou = i
                best_j = j
        if best_iou >= float(iou_thr) and best_j >= 0:
            matched_thr[k][best_j] = True
            n_tp_thr += 1

    recall = float(n_tp_thr) / float(max(1, total_gt))
    precision = float(n_tp_thr) / float(max(1, n_pred_thr))

    return DetectionSummary(ap=ap, recall=recall, precision=precision, n_gt=int(total_gt), n_pred=int(n_pred_thr))
