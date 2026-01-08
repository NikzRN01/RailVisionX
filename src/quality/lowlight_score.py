import numpy as np


def mean_intensity(gray: np.ndarray) -> float:
    return float(np.mean(gray))
