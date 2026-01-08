from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single detection in pixel coordinates."""

    # (x1, y1, x2, y2) in absolute pixels
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    category_id: int = 1


class DamageDetector(Protocol):
    def detect(self, image_rgb01_hwc: np.ndarray) -> list[Detection]:
        """Detect damage on an RGB image in [0,1], HWC float32."""


@dataclass(frozen=True)
class PlaceholderDamageDetector:
    """A lightweight, heuristic detector.

    This is NOT a real model. It exists to validate wiring + metrics, and to provide
    a drop-in place to integrate a real detector later.

    Heuristic:
      - convert to grayscale
      - run edge magnitude via simple Sobel
      - threshold + connected components (via OpenCV if available)
      - output component bounding boxes as 'damage' candidates
    """

    edge_threshold: float = 0.25
    min_area: int = 80
    max_boxes: int = 50

    def detect(self, image_rgb01_hwc: np.ndarray) -> list[Detection]:
        img = np.asarray(image_rgb01_hwc)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected HWC RGB image")

        h, w, _ = img.shape
        gray = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.float32)

        # Sobel magnitude (pure NumPy)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        if mag.size:
            mag = mag / (mag.max() + 1e-6)

        mask = mag >= float(self.edge_threshold)

        # Prefer OpenCV connected components if available (faster, better).
        boxes: list[tuple[int, int, int, int, float]] = []
        try:
            import cv2  # type: ignore

            mask_u8 = (mask.astype(np.uint8) * 255)
            num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
            # stats: [label, x, y, w, h, area]
            for i in range(1, int(num)):
                x, y, bw, bh, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
                if int(area) < int(self.min_area):
                    continue
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + bw), int(y + bh)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                score = float(area) / float(h * w)
                boxes.append((x1, y1, x2, y2, score))
        except Exception:
            # Fallback: no CC; emit a single bbox for any edges.
            ys, xs = np.where(mask)
            if xs.size:
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                area = float((x2 - x1) * (y2 - y1))
                score = float(area) / float(h * w)
                boxes.append((x1, y1, x2, y2, score))

        # Sort by score and cap.
        boxes.sort(key=lambda b: b[4], reverse=True)
        boxes = boxes[: int(self.max_boxes)]

        dets: list[Detection] = []
        for x1, y1, x2, y2, score in boxes:
            dets.append(Detection(bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)), score=float(score)))
        return dets


def build_damage_detector(name: str, **kwargs) -> DamageDetector | None:
    name = str(name).lower().strip()
    if name in ("none", "off", "disabled"):
        return None
    if name in ("placeholder", "heuristic"):
        return PlaceholderDamageDetector(**kwargs)
    raise ValueError(f"Unknown damage detector: {name}")
