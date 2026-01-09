from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Sample:
	key: str
	blurred_path: Path
	sharp_path: Path


def _repo_root() -> Path:
	# src/dataset/dataloader.py -> repo root
	return Path(__file__).resolve().parents[2]


def _read_image_rgb(path: Path) -> np.ndarray:
	"""Read an image as RGB uint8 numpy array.

	Tries OpenCV first (often faster / more robust for large PNGs), then Pillow.
	"""
	try:
		import cv2  # type: ignore

		img = cv2.imread(str(path), cv2.IMREAD_COLOR)
		if img is None:
			raise FileNotFoundError(f"Failed to read image: {path}")
		# OpenCV loads BGR
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	except ImportError as e:
		# Fall back to Pillow below.
		cv2 = None  # type: ignore[assignment]

	try:
		from PIL import Image  # type: ignore

		with Image.open(path) as im:
			im = im.convert("RGB")
			return np.array(im)
	except ImportError as e:
		raise RuntimeError(
			"No image backend available. Install 'opencv-python' or 'pillow' to load PNGs."
		) from e


def _to_float01(img: np.ndarray) -> np.ndarray:
	if img.dtype == np.uint8:
		return img.astype(np.float32) / 255.0
	return img.astype(np.float32)


def _to_chw(img: np.ndarray) -> np.ndarray:
	# HWC -> CHW
	if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
		raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
	return np.transpose(img, (2, 0, 1))


def _resize_hwc_rgb01(img01: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
	"""Resize HWC RGB float32 image in [0,1].

	Uses OpenCV if available, else Pillow. Returns float32 in [0,1].
	"""
	h, w = int(size_hw[0]), int(size_hw[1])
	if img01.shape[0] == h and img01.shape[1] == w:
		return img01.astype(np.float32)

	try:
		import cv2  # type: ignore

		img8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
		bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
		bgr2 = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
		rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
		return rgb2.astype(np.float32) / 255.0
	except Exception:
		pass

	try:
		from PIL import Image  # type: ignore

		img8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
		im = Image.fromarray(img8, mode="RGB")
		resampling = getattr(Image, "Resampling", Image)
		im = im.resize((w, h), resample=resampling.BILINEAR)
		return np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
	except Exception:
		# If no backend is available, return as-is.
		return img01.astype(np.float32)


def _apply_motion_blur_rgb01(
	img01: np.ndarray,
	*,
	rng: np.random.Generator,
	max_len: int,
) -> np.ndarray:
	"""Apply a synthetic linear motion blur to RGB float image in [0,1].

	This helps the model generalize to moving vehicles/cameras.
	"""
	max_len = max(3, int(max_len))
	# Odd kernel sizes behave better.
	k = int(rng.integers(3, max_len + 1))
	if k % 2 == 0:
		k += 1
	angle = float(rng.uniform(0.0, np.pi))

	# Create a line kernel (k x k) rotated by angle.
	kernel = np.zeros((k, k), dtype=np.float32)
	c = k // 2
	# Draw a line through the center
	for t in range(k):
		x = c + (t - c) * np.cos(angle)
		y = c + (t - c) * np.sin(angle)
		xi = int(np.clip(round(x), 0, k - 1))
		yi = int(np.clip(round(y), 0, k - 1))
		kernel[yi, xi] = 1.0

	s = float(kernel.sum())
	if s <= 0.0:
		return img01
	kernel /= s

	try:
		import cv2  # type: ignore

		img8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
		bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
		b = cv2.filter2D(bgr, -1, kernel)
		rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
		return rgb.astype(np.float32) / 255.0
	except Exception:
		# If OpenCV isn't available, skip augmentation.
		return img01


def discover_split_pairs(split_dir: Path) -> list[Sample]:
	"""Discover paired blurred/sharp images in a split directory.

	Expected layout:
	  split_dir/
		blurred/*.png
		sharp/*.png

	Pairs are matched by filename stem (e.g. 123.png -> key '123').
	"""
	blurred_dir = split_dir / "blurred"
	sharp_dir = split_dir / "sharp"
	if not blurred_dir.exists() or not sharp_dir.exists():
		raise FileNotFoundError(
			f"Expected '{blurred_dir}' and '{sharp_dir}' to exist. "
			"Run src/dataset/make_splits.py first."
		)

	# Avoid per-file stat() calls (p.is_file()) which can be very slow on /mnt/<drive>
	# in WSL. Our expected layout is flat folders of PNGs.
	blurred = {p.stem: p for p in blurred_dir.glob("*.png")}
	sharp = {p.stem: p for p in sharp_dir.glob("*.png")}
	keys = sorted(set(blurred).intersection(sharp))
	return [Sample(key=k, blurred_path=blurred[k], sharp_path=sharp[k]) for k in keys]


class PairedDeblurDataset:
	"""Dataset that yields (blurred, sharp) pairs from `data/split/...`.

	By default returns numpy arrays:
	  - `blurred`: float32 HWC RGB in [0,1]
	  - `sharp`:   float32 HWC RGB in [0,1]

	If `as_torch=True`, returns torch tensors:
	  - float32 CHW in [0,1]
	"""

	def __init__(
		self,
		split: SplitName,
		data_dir: Path | str | None = None,
		image_size: tuple[int, int] | None = None,
		augment_motion_blur: bool = False,
		motion_blur_prob: float = 0.0,
		motion_blur_max_len: int = 15,
		seed: int = 42,
		transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
		as_torch: bool = False,
	) -> None:
		self.split: SplitName = split
		self.data_dir = Path(data_dir) if data_dir is not None else _repo_root() / "data" / "split"
		self.split_dir = self.data_dir / split
		self.samples = discover_split_pairs(self.split_dir)
		self.image_size = image_size
		self.augment_motion_blur = bool(augment_motion_blur)
		self.motion_blur_prob = float(motion_blur_prob)
		self.motion_blur_max_len = int(motion_blur_max_len)
		self.rng = np.random.default_rng(int(seed))
		self.transform = transform
		self.as_torch = as_torch

		if not self.samples:
			raise RuntimeError(
				f"No paired PNGs found in '{self.split_dir}'. "
				"Expected blurred/*.png and sharp/*.png with matching names."
			)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		s = self.samples[idx]
		blurred = _to_float01(_read_image_rgb(s.blurred_path))
		sharp = _to_float01(_read_image_rgb(s.sharp_path))

		if self.image_size is not None:
			blurred = _resize_hwc_rgb01(blurred, self.image_size)
			sharp = _resize_hwc_rgb01(sharp, self.image_size)

		if self.split == "train" and self.augment_motion_blur and self.motion_blur_prob > 0.0:
			if float(self.rng.random()) < float(np.clip(self.motion_blur_prob, 0.0, 1.0)):
				blurred = _apply_motion_blur_rgb01(blurred, rng=self.rng, max_len=self.motion_blur_max_len)

		item: dict[str, Any] = {
			"key": s.key,
			"blurred": blurred,
			"sharp": sharp,
			"blurred_path": str(s.blurred_path),
			"sharp_path": str(s.sharp_path),
		}

		if self.transform is not None:
			item = self.transform(item)

		if self.as_torch:
			try:
				import torch

				item["blurred"] = torch.from_numpy(_to_chw(item["blurred"])).float()
				item["sharp"] = torch.from_numpy(_to_chw(item["sharp"])).float()
			except ImportError as e:  # pragma: no cover
				raise RuntimeError(
					"as_torch=True requires PyTorch installed. Install 'torch' or set as_torch=False."
				) from e

		return item


def build_torch_dataloader(
	dataset: PairedDeblurDataset,
	batch_size: int = 8,
	shuffle: bool | None = None,
	num_workers: int = 0,
	pin_memory: bool = False,
) -> Any:
	"""Create a torch DataLoader if torch is installed."""
	try:
		from torch.utils.data import DataLoader
	except ImportError as e:  # pragma: no cover
		raise RuntimeError("PyTorch is required to build a DataLoader.") from e

	if shuffle is None:
		shuffle = dataset.split == "train"

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

