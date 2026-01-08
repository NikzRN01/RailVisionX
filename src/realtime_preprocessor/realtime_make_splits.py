from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Pair:
    key: str
    blurred: Path
    sharp: Path


def _repo_root() -> Path:
    # src/realtime_preprocessor/realtime_make_splits.py -> repo root
    return Path(__file__).resolve().parents[2]


def _iter_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _index_by_stem(folder: Path) -> dict[str, Path]:
    items = sorted(_iter_images(folder), key=lambda p: str(p).lower())
    index: dict[str, Path] = {}
    for p in items:
        index[p.stem] = p
    return index


def discover_pairs(raw_dir: Path) -> list[Pair]:
    blurred_dir = raw_dir / "blurred"
    sharp_dir = raw_dir / "sharp"
    if not blurred_dir.exists() or not sharp_dir.exists():
        raise FileNotFoundError(f"Expected '{blurred_dir}' and '{sharp_dir}' to exist.")

    blurred = _index_by_stem(blurred_dir)
    sharp = _index_by_stem(sharp_dir)
    common_keys = sorted(set(blurred).intersection(sharp))
    return [Pair(key=k, blurred=blurred[k], sharp=sharp[k]) for k in common_keys]


def discover_unpaired(raw_dir: Path, kind: Literal["blurred", "frames"] = "blurred") -> list[Path]:
    # Prefer realtime_data/raw/blurred; fallback to realtime_data/raw/frames
    folder = raw_dir / kind
    if not folder.exists() and kind == "blurred":
        folder = raw_dir / "frames"

    imgs = sorted(_iter_images(folder), key=lambda p: str(p).lower())
    return imgs


def _as_png_name(src: Path) -> str:
    return f"{src.stem}.png"


def _copy_as_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {src}")
        ok = cv2.imwrite(str(dst), img)
        if not ok:
            raise ValueError(f"Failed to write image: {dst}")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Cannot convert '{src.name}' to PNG. Install OpenCV (opencv-python) or provide PNG inputs.\n{e}"
        )


def _split_counts(n: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> tuple[int, int, int]:
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio <= 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("Invalid ratios: require train_ratio>0, val_ratio>=0, and train_ratio+val_ratio<1")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _clean_out_dir(out_dir: Path) -> None:
    for name in ("train", "val", "test"):
        target = out_dir / name
        if target.exists():
            shutil.rmtree(target)


def write_paired_split(pairs: list[Pair], out_dir: Path, split_name: SplitName) -> None:
    blurred_out = out_dir / split_name / "blurred"
    sharp_out = out_dir / split_name / "sharp"
    for pair in pairs:
        _copy_as_png(pair.blurred, blurred_out / _as_png_name(pair.blurred))
        _copy_as_png(pair.sharp, sharp_out / _as_png_name(pair.sharp))


def write_unpaired_split(images: list[Path], out_dir: Path, split_name: SplitName) -> None:
    blurred_out = out_dir / split_name / "blurred"
    for p in images:
        _copy_as_png(p, blurred_out / _as_png_name(p))


def extract_frames_to_raw(
    source: int | str,
    raw_dir: Path,
    *,
    out_kind: Literal["blurred", "frames"] = "blurred",
    stride: int = 1,
    max_frames: int | None = None,
    prefix: str = "frame",
) -> int:
    """Extract frames from a webcam/video into realtime_data/raw.

    Notes:
      - This only creates *unpaired* frames (no sharp ground truth).
      - Output goes into raw/<out_kind>/ as PNG.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("Frame extraction requires OpenCV. Install 'opencv-python'.") from e

    out_dir = raw_dir / out_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    stride = max(1, int(stride))
    count = 0
    emitted = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            count += 1
            if (count - 1) % stride != 0:
                continue

            out_path = out_dir / f"{prefix}_{count:06d}.png"
            ok2 = cv2.imwrite(str(out_path), frame)
            if not ok2:
                raise RuntimeError(f"Failed to write frame: {out_path}")

            emitted += 1
            if max_frames is not None and emitted >= int(max_frames):
                break
    finally:
        cap.release()

    return emitted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime preprocessing: extract frames into realtime_data/raw and/or split realtime raw images into "
            "realtime_data/spilts/{train,val,test}/... (80/10/10).\n\n"
            "Paired mode expects realtime_data/raw/{blurred,sharp}.\n"
            "Unpaired mode expects realtime_data/raw/blurred or realtime_data/raw/frames."
        )
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract frames from camera/video into realtime_data/raw")
    p_extract.add_argument("--raw-dir", type=Path, default=_repo_root() / "realtime_data" / "raw")
    p_extract.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0) or video filepath",
    )
    p_extract.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    p_extract.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    p_extract.add_argument("--prefix", type=str, default="frame")
    p_extract.add_argument(
        "--out-kind",
        choices=["blurred", "frames"],
        default="blurred",
        help="Subfolder under raw/ to write frames into",
    )

    p_split = sub.add_parser("split", help="Split realtime_data/raw into realtime_data/spilts")
    p_split.add_argument("--raw-dir", type=Path, default=_repo_root() / "realtime_data" / "raw")
    p_split.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "realtime_data" / "spilts",
        help="Output directory for train/val/test",
    )
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--clean", action="store_true")
    p_split.add_argument(
        "--mode",
        choices=["auto", "paired", "unpaired"],
        default="auto",
        help="auto: paired if raw/blurred+raw/sharp exist else unpaired",
    )
    p_split.add_argument("--limit", type=int, default=0, help="0 = no limit")

    args = parser.parse_args()

    if args.cmd == "extract":
        src: str = args.source
        source: int | str
        if src.isdigit():
            source = int(src)
        else:
            source = src

        max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)
        n = extract_frames_to_raw(
            source,
            Path(args.raw_dir),
            out_kind=args.out_kind,
            stride=int(args.stride),
            max_frames=max_frames,
            prefix=str(args.prefix),
        )
        print(f"Extracted {n} frames into: {Path(args.raw_dir) / args.out_kind}")
        return

    if args.cmd == "split":
        raw_dir = Path(args.raw_dir)
        out_dir = Path(args.out_dir)

        mode: str = args.mode
        if mode == "auto":
            if (raw_dir / "blurred").exists() and (raw_dir / "sharp").exists():
                mode = "paired"
            else:
                mode = "unpaired"

        if args.clean:
            _clean_out_dir(out_dir)

        rng = random.Random(int(args.seed))

        if mode == "paired":
            pairs = discover_pairs(raw_dir)
            if args.limit and int(args.limit) > 0:
                pairs = pairs[: int(args.limit)]

            if not pairs:
                raise SystemExit(
                    f"No paired images found under '{raw_dir}'. Expected matching filenames in blurred/ and sharp/."
                )

            rng.shuffle(pairs)

            n_train, n_val, n_test = _split_counts(len(pairs), train_ratio=0.7, val_ratio=0.1)
            train_pairs = pairs[:n_train]
            val_pairs = pairs[n_train : n_train + n_val]
            test_pairs = pairs[n_train + n_val :]

            write_paired_split(train_pairs, out_dir, "train")
            write_paired_split(val_pairs, out_dir, "val")
            write_paired_split(test_pairs, out_dir, "test")

            print(
                f"Wrote paired splits to: {out_dir}\n"
                f"Pairs total: {len(pairs)}\n"
                f"train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}\n"
                "Each split contains blurred/ and sharp/ PNGs."
            )
            return

        # unpaired
        images = discover_unpaired(raw_dir, kind="blurred")
        if args.limit and int(args.limit) > 0:
            images = images[: int(args.limit)]

        if not images:
            raise SystemExit(
                f"No images found under '{raw_dir}/blurred' or '{raw_dir}/frames'. "
                "Put images there first (or run the 'extract' subcommand)."
            )

        rng.shuffle(images)

        n_train, n_val, n_test = _split_counts(len(images), train_ratio=0.7, val_ratio=0.1)
        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        write_unpaired_split(train_imgs, out_dir, "train")
        write_unpaired_split(val_imgs, out_dir, "val")
        write_unpaired_split(test_imgs, out_dir, "test")

        print(
            f"Wrote unpaired splits to: {out_dir}\n"
            f"Images total: {len(images)}\n"
            f"train: {len(train_imgs)}  val: {len(val_imgs)}  test: {len(test_imgs)}\n"
            "Each split contains blurred/ PNGs only (for inference)."
        )
        return


if __name__ == "__main__":
    main()
