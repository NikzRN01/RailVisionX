from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayAny: TypeAlias = NDArray[np.generic]
RGBUInt8: TypeAlias = NDArray[np.uint8]
RGBFloat32: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class LowLightEnhanceConfig:
	"""Configuration for low-light enhancement.

	The default pipeline is: auto-gamma -> CLAHE -> unsharp mask.
	"""

	# Auto-gamma
	target_mean: float = 0.50
	gamma_min: float = 0.50
	gamma_max: float = 2.50

	# CLAHE
	clahe_clip_limit: float = 2.0
	clahe_tile_grid_size: Tuple[int, int] = (8, 8)

	# Unsharp mask
	unsharp_amount: float = 0.60
	unsharp_sigma: float = 1.0
	unsharp_threshold: int = 0


def _as_float01_rgb(img: ArrayAny) -> tuple[RGBFloat32, bool]:
	"""Return float32 RGB in [0,1] and whether input was uint8."""
	if img.ndim != 3 or img.shape[2] != 3:
		raise ValueError(f"Expected RGB image (H,W,3), got shape={img.shape}")

	if img.dtype == np.uint8:
		return (img.astype(np.float32) / 255.0), True

	arr = img.astype(np.float32)
	# If it's likely already in 0..255, normalize.
	if arr.max(initial=0.0) > 1.5:
		arr = arr / 255.0
	return np.clip(arr, 0.0, 1.0).astype(np.float32), False


def _to_uint8_rgb(img01: ArrayAny) -> RGBUInt8:
	arr: RGBFloat32 = np.asarray(img01, dtype=np.float32)
	arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
	return (arr * 255.0 + 0.5).astype(np.uint8)


def auto_gamma(
	img: ArrayAny,
	*,
	target_mean: float = 0.5,
	gamma_min: float = 0.5,
	gamma_max: float = 2.5,
) -> RGBFloat32:
	"""Automatic gamma correction based on mean luminance.

	Works on RGB images. Returns float32 RGB in [0,1].
	"""
	img01, _ = _as_float01_rgb(img)
	# Luma approximation (Rec. 709)
	y = 0.2126 * img01[..., 0] + 0.7152 * img01[..., 1] + 0.0722 * img01[..., 2]
	mean_y = float(np.mean(y))
	eps = 1e-6
	target_mean = float(np.clip(target_mean, 0.05, 0.95))

	# Solve: mean_y ** gamma = target_mean  -> gamma = log(target)/log(mean)
	gamma = np.log(target_mean + eps) / np.log(mean_y + eps)
	gamma = float(np.clip(gamma, gamma_min, gamma_max))

	out = np.power(np.clip(img01, 0.0, 1.0), gamma, dtype=np.float32)
	return out.astype(np.float32)


def clahe_rgb(
	img: ArrayAny,
	*,
	clip_limit: float = 2.0,
	tile_grid_size: tuple[int, int] = (8, 8),
) -> RGBFloat32:
	"""Apply CLAHE to the luminance channel (LAB L) of an RGB image.

	Returns float32 RGB in [0,1]. Requires OpenCV.
	"""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("CLAHE requires OpenCV. Install 'opencv-python' or 'opencv-python-headless'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)

	# OpenCV expects BGR
	bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)

	clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])))
	l2 = clahe.apply(l)

	lab2 = cv2.merge([l2, a, b])
	bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
	rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
	return (rgb2.astype(np.float32) / 255.0).astype(np.float32)


def unsharp_mask_rgb(
	img: ArrayAny,
	*,
	amount: float = 0.6,
	sigma: float = 1.0,
	threshold: int = 0,
) -> RGBFloat32:
	"""Unsharp mask for RGB images.

	Returns float32 RGB in [0,1]. Requires OpenCV.
	"""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("Unsharp masking requires OpenCV. Install 'opencv-python' or 'opencv-python-headless'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)
	blurred = cv2.GaussianBlur(img8, ksize=(0, 0), sigmaX=float(sigma))

	# (1+amount)*img - amount*blur
	sharp = cv2.addWeighted(img8, 1.0 + float(amount), blurred, -float(amount), 0)

	if int(threshold) > 0:
		low_contrast = np.abs(img8.astype(np.int16) - blurred.astype(np.int16)) < int(threshold)
		sharp = np.where(low_contrast, img8, sharp)

	return (sharp.astype(np.float32) / 255.0).astype(np.float32)


def unsharp_mask_luma_rgb(
	img: ArrayAny,
	*,
	amount: float = 0.6,
	sigma: float = 1.0,
	threshold: int = 0,
) -> RGBFloat32:
	"""Unsharp mask applied only to luminance (LAB L) channel.

	This tends to preserve color fidelity and reduces grainy artifacts compared
	to sharpening all RGB channels.

	Returns float32 RGB in [0,1]. Requires OpenCV.
	"""
	try:
		import cv2  # type: ignore
	except ImportError as e:
		raise RuntimeError("Luma unsharp masking requires OpenCV. Install 'opencv-python' or 'opencv-python-headless'.") from e

	img01, _ = _as_float01_rgb(img)
	img8 = _to_uint8_rgb(img01)

	# Convert RGB->LAB via OpenCV BGR
	bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
	lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)

	blurred = cv2.GaussianBlur(l, ksize=(0, 0), sigmaX=float(sigma))
	sharp_l = cv2.addWeighted(l, 1.0 + float(amount), blurred, -float(amount), 0)

	if int(threshold) > 0:
		low_contrast = np.abs(l.astype(np.int16) - blurred.astype(np.int16)) < int(threshold)
		sharp_l = np.where(low_contrast, l, sharp_l).astype(np.uint8)

	lab2 = cv2.merge([sharp_l, a, b])
	bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
	rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
	return (rgb2.astype(np.float32) / 255.0).astype(np.float32)


EnhanceMethod = Literal[
	"best",
	"plates",
	"plates-strong",
	"auto-gamma",
	"clahe",
	"unsharp",
	"auto-gamma+clahe",
	"auto-gamma+clahe+unsharp",
]


def enhance_lowlight(
	img: ArrayAny,
	*,
	method: EnhanceMethod = "best",
	cfg: LowLightEnhanceConfig = LowLightEnhanceConfig(),
) -> ArrayAny:
	"""Enhance a low-light RGB image.

	Input:
	  - `img`: RGB image as uint8 (0..255) or float (0..1 or 0..255)

	Output:
	  - Same dtype style as input (uint8 stays uint8; float returns float32 in [0,1]).
	"""
	img01, was_u8 = _as_float01_rgb(img)

	pipeline = method
	if pipeline == "best":
		pipeline = "auto-gamma+clahe+unsharp"
	if pipeline == "plates":
		# Plate/vehicle-friendly preset:
		# - still improves luminance and local contrast
		# - avoids the harsher sharpening that can create halos/cartoon-ish edges
		plates_cfg = LowLightEnhanceConfig(
			target_mean=0.45,
			gamma_min=0.70,
			gamma_max=1.80,
			clahe_clip_limit=1.60,
			clahe_tile_grid_size=(8, 8),
			unsharp_amount=0.30,
			unsharp_sigma=1.00,
			unsharp_threshold=2,
		)
		cfg = plates_cfg
		pipeline = "auto-gamma+clahe+unsharp"
	if pipeline == "plates-strong":
		# More aggressive preset for readability:
		# - stronger local contrast
		# - stronger sharpening (may introduce halos/noise)
		plates_cfg = LowLightEnhanceConfig(
			target_mean=0.50,
			gamma_min=0.60,
			gamma_max=2.00,
			clahe_clip_limit=2.60,
			clahe_tile_grid_size=(8, 8),
			unsharp_amount=0.85,
			unsharp_sigma=0.90,
			unsharp_threshold=0,
		)
		cfg = plates_cfg
		pipeline = "auto-gamma+clahe+unsharp"

	out: RGBFloat32 = img01
	if pipeline in ("auto-gamma", "auto-gamma+clahe", "auto-gamma+clahe+unsharp"):
		out = auto_gamma(out, target_mean=cfg.target_mean, gamma_min=cfg.gamma_min, gamma_max=cfg.gamma_max)
	if pipeline in ("clahe", "auto-gamma+clahe", "auto-gamma+clahe+unsharp"):
		out = clahe_rgb(out, clip_limit=cfg.clahe_clip_limit, tile_grid_size=cfg.clahe_tile_grid_size)
	if pipeline in ("unsharp", "auto-gamma+clahe+unsharp"):
		# For plate-focused presets, sharpen luminance only to reduce artifacts.
		if method in ("plates", "plates-strong"):
			out = unsharp_mask_luma_rgb(out, amount=cfg.unsharp_amount, sigma=cfg.unsharp_sigma, threshold=cfg.unsharp_threshold)
		else:
			out = unsharp_mask_rgb(out, amount=cfg.unsharp_amount, sigma=cfg.unsharp_sigma, threshold=cfg.unsharp_threshold)

	if was_u8:
		return _to_uint8_rgb(out)
	return out.astype(np.float32)

