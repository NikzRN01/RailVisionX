from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.quality.blur_score import mean_gradient_magnitude, variance_of_laplacian


BlurMetric = Literal["lap_var", "grad_mean", "combo"]


@dataclass(frozen=True)
class BlurGateConfig:
    """Configuration for blur severity gating.

    Important: the underlying metrics are *sharpness* proxies.
    Higher score => sharper image.

    Routing logic:
      - if score >= sharp_threshold: consider frame sharp and skip deblur
      - if score <= blur_threshold: consider frame blurry and run deblur
      - otherwise: "mild" / uncertain zone; default is to run deblur
    """

    metric: BlurMetric = "combo"
    sharp_threshold: float = 0.020
    blur_threshold: float = 0.008


@dataclass(frozen=True)
class BlurScore:
    lap_var: float
    grad_mean: float
    score: float


def compute_blur_score(img: np.ndarray, *, metric: BlurMetric = "combo") -> BlurScore:
    """Compute blur/sharpness proxy scores for an RGB image.

    Args:
      img: RGB array, uint8 (0..255) or float (0..1 or 0..255).
      metric: which single score to expose as BlurScore.score.

    Returns:
      BlurScore with lap_var, grad_mean, and combined score.
    """

    lap = float(variance_of_laplacian(img))
    grad = float(mean_gradient_magnitude(img))

    if metric == "lap_var":
        s = lap
    elif metric == "grad_mean":
        s = grad
    elif metric == "combo":
        # Heuristic combo: grad captures edges robustly; lap adds sensitivity.
        s = lap + 10.0 * grad
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return BlurScore(lap_var=lap, grad_mean=grad, score=float(s))


def should_skip_deblur(score: float, *, sharp_threshold: float) -> bool:
    return float(score) >= float(sharp_threshold)


def should_run_deblur(score: float, *, blur_threshold: float) -> bool:
    return float(score) <= float(blur_threshold)


def severity(score: float, *, cfg: BlurGateConfig) -> str:
    s = float(score)
    if s >= float(cfg.sharp_threshold):
        return "sharp"
    if s <= float(cfg.blur_threshold):
        return "blurred"
    return "mild"
