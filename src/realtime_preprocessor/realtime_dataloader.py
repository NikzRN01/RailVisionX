from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

SplitName = Literal["train", "val", "test"]

RGBUInt8: TypeAlias = NDArray[np.uint8]
ArrayAny: TypeAlias = NDArray[np.generic]
RGBFloat32: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class FrameSample:
    key: str
    path: Optional[Path] = None
    source: Optional[str] = None


def _repo_root() -> Path:
    # src/realtime_preprocessor/realtime_dataloader.py -> repo root
    return Path(__file__).resolve().parents[2]


def _default_realtime_split_dir() -> Path:
    """Return the default realtime split directory.

    Note: the repo has used multiple spellings over time; we support all.
    """
    root = _repo_root() / "realtime_data"
    if (root / "spilts").exists():
        return root / "spilts"
    if (root / "split").exists():
        return root / "split"
    return root / "spilt"


def _read_image_rgb(path: Path) -> RGBUInt8:
    """Read an image as RGB uint8 numpy array.

    Tries Pillow first (lighter dependency), then OpenCV.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except ImportError:
        pass

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        # OpenCV loads BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError as e:
        raise RuntimeError(
            "No image backend available. Install 'pillow' or 'opencv-python' to load images."
        ) from e


def _to_float01(img: ArrayAny) -> RGBFloat32:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_chw(img: ArrayAny) -> ArrayAny:
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
    return np.transpose(img, (2, 0, 1))


def _resize_hwc_rgb01(img01: RGBFloat32, size_hw: tuple[int, int]) -> RGBFloat32:
        """Resize HWC RGB float32 image in [0,1].

        Prefers Pillow when available; falls back to OpenCV.
        Uses interpolation that preserves detail:
            - downscale: AREA
            - upscale: CUBIC
        """
    h, w = int(size_hw[0]), int(size_hw[1])
    try:
        from PIL import Image  # type: ignore

        img8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        im = Image.fromarray(img8, mode="RGB")
        # Pillow 10+ uses Image.Resampling.BILINEAR; older versions use Image.BILINEAR.
        resampling = getattr(Image, "Resampling", Image)
        im = im.resize((w, h), resample=resampling.BILINEAR)
        return np.array(im, dtype=np.uint8).astype(np.float32) / 255.0
    except ImportError:
        pass

    try:
        import cv2  # type: ignore

        ih, iw = int(img01.shape[0]), int(img01.shape[1])
        if ih == h and iw == w:
            return img01

        interp = cv2.INTER_AREA if (h < ih or w < iw) else cv2.INTER_CUBIC
        out = cv2.resize(img01, (w, h), interpolation=interp)
        return out.astype(np.float32)
    except ImportError:
        # If no backend is available, return as-is.
        return img01


def _discover_images(dir_path: Path, exts: tuple[str, ...] = ("png", "jpg", "jpeg")) -> list[FrameSample]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths: list[Path] = []
    for ext in exts:
        paths.extend(dir_path.glob(f"*.{ext}"))
        paths.extend(dir_path.glob(f"*.{ext.upper()}"))

    paths = sorted(set(paths))
    return [FrameSample(key=p.stem, path=p, source=str(dir_path)) for p in paths]


def discover_realtime_frame_folder(
    split_dir: Path | None = None,
    split: SplitName | None = None,
    kind: Literal["blurred", "sharp"] = "blurred",
) -> list[FrameSample]:
    """Discover frames under realtime_data for inference.

    Supported layouts:
      - realtime_data/spilt/<kind>/*.png
      - realtime_data/spilt/<split>/<kind>/*.png

    Args:
      split_dir: Base realtime split directory (defaults to realtime_data/spilt or realtime_data/split).
      split: Optional train/val/test subfolder.
      kind: "blurred" or "sharp".
    """
    base = split_dir if split_dir is not None else _default_realtime_split_dir()
    target = base / split / kind if split is not None else base / kind
    return _discover_images(target)


def discover_realtime_paired_frames(
    split_dir: Path | None = None,
    split: SplitName | None = None,
) -> list[tuple[FrameSample, FrameSample]]:
    """Discover paired blurred/sharp frames under realtime_data for training/eval.

    Expected layout:
      - realtime_data/spilt/<split>/blurred/*.png
      - realtime_data/spilt/<split>/sharp/*.png
    Or without split:
      - realtime_data/spilt/blurred/*.png
      - realtime_data/spilt/sharp/*.png

    Pairs are matched by filename stem.
    """
    base = split_dir if split_dir is not None else _default_realtime_split_dir()
    blurred_dir = base / split / "blurred" if split is not None else base / "blurred"
    sharp_dir = base / split / "sharp" if split is not None else base / "sharp"

    blurred = {p.stem: p for p in blurred_dir.glob("*.png")}
    sharp = {p.stem: p for p in sharp_dir.glob("*.png")}
    keys = sorted(set(blurred).intersection(sharp))

    pairs: list[tuple[FrameSample, FrameSample]] = []
    for k in keys:
        pairs.append(
            (
                FrameSample(key=k, path=blurred[k], source=str(blurred_dir)),
                FrameSample(key=k, path=sharp[k], source=str(sharp_dir)),
            )
        )
    return pairs


class RealtimeStream:
    """Realtime frame generator from a webcam or video file.

    This yields frames suitable for inference (no ground truth).

    Requirements:
      - OpenCV (`opencv-python` or `opencv-python-headless`) for VideoCapture.

    Yields dict:
      - key: str
      - frame: float32 HWC RGB in [0,1]
      - source: str
    """

    def __init__(
        self,
        source: int | str = 0,
        stride: int = 1,
        max_frames: int | None = None,
        image_size: tuple[int, int] | None = None,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        as_torch: bool = False,
    ) -> None:
        self.source = source
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.image_size = image_size
        self.transform = transform
        self.as_torch = as_torch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            import cv2  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "RealtimeStream requires OpenCV. Install 'opencv-python' to use webcam/video streaming."
            ) from e

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        count = 0
        emitted = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                count += 1
                if (count - 1) % self.stride != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame01 = _to_float01(frame_rgb)

                if self.image_size is not None:
                    frame01 = _resize_hwc_rgb01(frame01, self.image_size)

                item: dict[str, Any] = {
                    "key": f"frame_{count:06d}",
                    "frame": frame01,
                    "source": str(self.source),
                }

                if self.transform is not None:
                    item = self.transform(item)

                if self.as_torch:
                    try:
                        import torch  # type: ignore

                        item["frame"] = torch.from_numpy(_to_chw(item["frame"])).float()
                    except ImportError as e:
                        raise RuntimeError("as_torch=True requires PyTorch installed.") from e

                yield item
                emitted += 1

                if self.max_frames is not None and emitted >= int(self.max_frames):
                    break
        finally:
            cap.release()


class RealtimeFolderFramesDataset:
    """Dataset for inference frames stored on disk (non-realtime, but used as realtime-like input).

    Typical use: read frames from `realtime_data/spilt/...` for inference.

    Returns dict:
      - key
      - frame (float32 HWC [0,1] or torch CHW)
      - path
    """

    def __init__(
        self,
        root_dir: Path | str,
        image_size: tuple[int, int] | None = None,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        as_torch: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.samples = _discover_images(self.root_dir)
        self.image_size = image_size
        self.transform = transform
        self.as_torch = as_torch

        if not self.samples:
            raise RuntimeError(f"No images found in: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        if s.path is None:
            raise RuntimeError("Sample has no path")

        frame01 = _to_float01(_read_image_rgb(s.path))
        if self.image_size is not None:
            frame01 = _resize_hwc_rgb01(frame01, self.image_size)

        item: dict[str, Any] = {"key": s.key, "frame": frame01, "path": str(s.path)}

        if self.transform is not None:
            item = self.transform(item)

        if self.as_torch:
            try:
                import torch  # type: ignore

                item["frame"] = torch.from_numpy(_to_chw(item["frame"])).float()
            except ImportError as e:
                raise RuntimeError("as_torch=True requires PyTorch installed.") from e

        return item


class RealtimePairedFramesDataset:
    """Paired blurred/sharp dataset under realtime_data/spilt for training.

    This mirrors `PairedDeblurDataset`, but reads from the realtime_data tree.

    Returns dict:
      - key
      - blurred
      - sharp
    """

    def __init__(
        self,
        split: SplitName | None = None,
        split_dir: Path | str | None = None,
        image_size: tuple[int, int] | None = (256, 256),
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        as_torch: bool = False,
    ) -> None:
        self.split = split
        self.split_dir = Path(split_dir) if split_dir is not None else _default_realtime_split_dir()
        self.pairs = discover_realtime_paired_frames(self.split_dir, split=split)
        self.image_size = image_size
        self.transform = transform
        self.as_torch = as_torch

        if not self.pairs:
            raise RuntimeError(
                f"No paired frames found under '{self.split_dir}' (split={split}). "
                "Expected blurred/*.png and sharp/*.png with matching names."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        b, s = self.pairs[idx]
        if b.path is None or s.path is None:
            raise RuntimeError("Pair has missing paths")

        blurred01 = _to_float01(_read_image_rgb(b.path))
        sharp01 = _to_float01(_read_image_rgb(s.path))

        if self.image_size is not None:
            blurred01 = _resize_hwc_rgb01(blurred01, self.image_size)
            sharp01 = _resize_hwc_rgb01(sharp01, self.image_size)

        item: dict[str, Any] = {
            "key": b.key,
            "blurred": blurred01,
            "sharp": sharp01,
            "blurred_path": str(b.path),
            "sharp_path": str(s.path),
        }

        if self.transform is not None:
            item = self.transform(item)

        if self.as_torch:
            try:
                import torch  # type: ignore

                item["blurred"] = torch.from_numpy(_to_chw(item["blurred"])).float()
                item["sharp"] = torch.from_numpy(_to_chw(item["sharp"])).float()
            except ImportError as e:
                raise RuntimeError("as_torch=True requires PyTorch installed.") from e

        return item


class RandomSplitInferenceDataset:
    """Load existing images from `data/split/...` for inference/testing.

    This is *not* realtime streaming; it's the "inference on random images" input the user asked for.

    Example:
      RandomSplitInferenceDataset(split="test", kind="blurred") reads:
        data/split/test/blurred/*.png

    Returns dict:
      - key
      - image
      - path
    """

    def __init__(
        self,
        split: SplitName = "test",
        kind: Literal["blurred", "sharp"] = "blurred",
        data_dir: Path | str | None = None,
        image_size: tuple[int, int] | None = (256, 256),
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        as_torch: bool = False,
    ) -> None:
        self.split = split
        self.kind = kind
        base = Path(data_dir) if data_dir is not None else _repo_root() / "data" / "split"
        self.root_dir = base / split / kind
        self.samples = _discover_images(self.root_dir)
        self.image_size = image_size
        self.transform = transform
        self.as_torch = as_torch

        if not self.samples:
            raise RuntimeError(f"No images found in: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        if s.path is None:
            raise RuntimeError("Sample has no path")

        img01 = _to_float01(_read_image_rgb(s.path))
        if self.image_size is not None:
            img01 = _resize_hwc_rgb01(img01, self.image_size)

        item: dict[str, Any] = {"key": s.key, "image": img01, "path": str(s.path)}

        if self.transform is not None:
            item = self.transform(item)

        if self.as_torch:
            try:
                import torch  # type: ignore

                item["image"] = torch.from_numpy(_to_chw(item["image"])).float()
            except ImportError as e:
                raise RuntimeError("as_torch=True requires PyTorch installed.") from e

        return item


def build_torch_dataloader(
    dataset: Any,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Any:
    """Create a torch DataLoader if torch is installed."""
    try:
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build a DataLoader.") from e

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
