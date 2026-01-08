from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _to_gray_float(img: NDArray[np.generic]) -> NDArray[np.float32]:
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


def mean_intensity(image: NDArray[np.generic]) -> float:
    """Mean grayscale intensity in [0,1].

    Accepts either grayscale HxW or RGB HxWxC (C>=3).
    """
    gray = _to_gray_float(image)
    return float(np.mean(gray))
