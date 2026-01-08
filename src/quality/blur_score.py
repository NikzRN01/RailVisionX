from __future__ import annotations

import numpy as np


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        # Assume RGB in [0,1] or [0,255]
        rgb = img[..., :3].astype(np.float32)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

    gray = gray.astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def variance_of_laplacian(image: np.ndarray) -> float:
    """Simple sharpness metric: variance of Laplacian response.

    Higher values generally mean sharper/more edges.
    """
    gray = _to_gray_float(image)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Convolution (valid-ish with padding)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    lap = (
        k[0, 0] * padded[:-2, :-2]
        + k[0, 1] * padded[:-2, 1:-1]
        + k[0, 2] * padded[:-2, 2:]
        + k[1, 0] * padded[1:-1, :-2]
        + k[1, 1] * padded[1:-1, 1:-1]
        + k[1, 2] * padded[1:-1, 2:]
        + k[2, 0] * padded[2:, :-2]
        + k[2, 1] * padded[2:, 1:-1]
        + k[2, 2] * padded[2:, 2:]
    )
    return float(np.var(lap))


def mean_gradient_magnitude(image: np.ndarray) -> float:
    gray = _to_gray_float(image)
    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))
