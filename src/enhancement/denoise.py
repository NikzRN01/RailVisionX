from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayAny: TypeAlias = NDArray[np.generic]
RGBUInt8: TypeAlias = NDArray[np.uint8]
RGBFloat32: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class DenoiseConfig:
	"""Config for denoising.

	Default behavior (`method="best"`) estimates noise strength and chooses:
	  - light noise  -> bilateral filter (preserves edges)
	  - heavier noise -> fast NLMeans (stronger smoothing)
	"""

	# Noise estimation
	mild_noise_sigma: float = 0.02
	strong_noise_sigma: float = 0.05

	# Bilateral
	bilateral_d: int = 7
	bilateral_sigma_color: float = 35.0
	bilateral_sigma_space: float = 7.0

	# NLMeans
	nlm_h: float = 8.0
	nlm_h_color: float = 8.0
	nlm_template_window_size: int = 7
	nlm_search_window_size: int = 21

	# Median (optional)
	median_ksize: int = 3


DenoiseMethod = Literal["best", "bilateral", "nlmeans", "median"]


def _as_float01_rgb(img: ArrayAny) -> tuple[RGBFloat32, bool]:
	if img.ndim != 3 or img.shape[2] != 3:
		raise ValueError(f"Expected RGB image (H,W,3), got shape={img.shape}")

	if img.dtype == np.uint8:
		return (img.astype(np.float32) / 255.0), True

	arr = np.asarray(img, dtype=np.float32)
	if float(arr.max(initial=0.0)) > 1.5:
		arr = arr / 255.0
	return np.clip(arr, 0.0, 1.0).astype(np.float32), False


def _to_uint8_rgb(img01: ArrayAny) -> RGBUInt8:
	arr = np.asarray(img01, dtype=np.float32)
	arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
	return (arr * 255.0 + 0.5).astype(np.uint8)


def estimate_noise_sigma(img: ArrayAny) -> float:
	"""Rough noise estimate in [0,1] based on high-frequency energy.

	This is intentionally simple and fast; it works well enough to decide between
	bilateral vs NLMeans for typical webcam/video frames.
	"""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("Noise estimation requires OpenCV. Install 'opencv-python'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)
	gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
	gray_f = gray.astype(np.float32) / 255.0
	# Laplacian magnitude correlates with edges + noise; blur first to suppress edges.
	blur = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
	lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
	lap32 = np.asarray(lap, dtype=np.float32)
	sigma = float(np.sqrt(np.mean(lap32 * lap32)))
	return sigma


def denoise_bilateral(img: ArrayAny, *, cfg: DenoiseConfig = DenoiseConfig()) -> RGBFloat32:
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("Bilateral denoise requires OpenCV. Install 'opencv-python'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)
	bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
	out = cv2.bilateralFilter(
		bgr,
		d=int(cfg.bilateral_d),
		sigmaColor=float(cfg.bilateral_sigma_color),
		sigmaSpace=float(cfg.bilateral_sigma_space),
	)
	rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
	return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def denoise_nlmeans(img: ArrayAny, *, cfg: DenoiseConfig = DenoiseConfig()) -> RGBFloat32:
	"""Fast NLMeans color denoising. Stronger, can soften details."""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("NLMeans denoise requires OpenCV. Install 'opencv-python'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)
	bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
	out = cv2.fastNlMeansDenoisingColored(
		bgr,
		None,
		h=float(cfg.nlm_h),
		hColor=float(cfg.nlm_h_color),
		templateWindowSize=int(cfg.nlm_template_window_size),
		searchWindowSize=int(cfg.nlm_search_window_size),
	)
	rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
	return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def denoise_median(img: ArrayAny, *, cfg: DenoiseConfig = DenoiseConfig()) -> RGBFloat32:
	"""Median filter (best for salt-and-pepper noise)."""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("Median denoise requires OpenCV. Install 'opencv-python'.") from e

	k = int(cfg.median_ksize)
	if k <= 1 or (k % 2) == 0:
		k = 3

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)
	bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
	out = cv2.medianBlur(bgr, k)
	rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
	return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def denoise(
	img: ArrayAny,
	*,
	method: DenoiseMethod = "best",
	cfg: DenoiseConfig = DenoiseConfig(),
) -> ArrayAny:
	"""Denoise an RGB image.

	Input:
	  - `img`: RGB uint8 (0..255) or float (0..1 or 0..255)

	Output:
	  - uint8 input -> uint8 output
	  - float input -> float32 output in [0,1]
	"""
	img01, was_u8 = _as_float01_rgb(img)

	chosen = method
	if chosen == "best":
		sigma = estimate_noise_sigma(img01)
		if sigma < float(cfg.mild_noise_sigma):
			chosen = "bilateral"
		elif sigma < float(cfg.strong_noise_sigma):
			chosen = "bilateral"
		else:
			chosen = "nlmeans"

	if chosen == "bilateral":
		out01 = denoise_bilateral(img01, cfg=cfg)
	elif chosen == "nlmeans":
		out01 = denoise_nlmeans(img01, cfg=cfg)
	elif chosen == "median":
		out01 = denoise_median(img01, cfg=cfg)
	else:
		raise ValueError(f"Unknown denoise method: {method}")

	if was_u8:
		return _to_uint8_rgb(out01)
	return out01.astype(np.float32)

