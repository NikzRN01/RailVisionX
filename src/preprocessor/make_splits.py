from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Pair:
    key: str
    blurred: Path
    sharp: Path


def _repo_root() -> Path:
    # src/dataset/make_splits.py -> repo root
    return Path(__file__).resolve().parents[2]


def _iter_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _index_by_stem(folder: Path) -> dict[str, Path]:
    # If duplicates exist, prefer deterministic order (sorted path string)
    items = sorted(_iter_images(folder), key=lambda p: str(p).lower())
    index: dict[str, Path] = {}
    for p in items:
        index[p.stem] = p
    return index


def discover_pairs(raw_dir: Path) -> list[Pair]:
    blurred_dir = raw_dir / "blurred"
    sharp_dir = raw_dir / "sharp"
    if not blurred_dir.exists() or not sharp_dir.exists():
        raise FileNotFoundError(
            f"Expected '{blurred_dir}' and '{sharp_dir}' to exist."
        )

    blurred = _index_by_stem(blurred_dir)
    sharp = _index_by_stem(sharp_dir)
    common_keys = sorted(set(blurred).intersection(sharp))
    return [Pair(key=k, blurred=blurred[k], sharp=sharp[k]) for k in common_keys]


def _as_png_name(src: Path) -> str:
    # Always write PNG files to the split directories.
    return f"{src.stem}.png"


def _copy_as_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Most datasets here are already .png; copy directly.
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dst)
        return

    # If non-png images exist, try converting via OpenCV if available.
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


def write_split(
    pairs: list[Pair],
    out_dir: Path,
    split_name: str,
) -> None:
    blurred_out = out_dir / split_name / "blurred"
    sharp_out = out_dir / split_name / "sharp"

    for pair in pairs:
        _copy_as_png(pair.blurred, blurred_out / _as_png_name(pair.blurred))
        _copy_as_png(pair.sharp, sharp_out / _as_png_name(pair.sharp))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess and split paired images from data/raw/{blurred,sharp} into "
            "data/split/{train,val,test}/{blurred,sharp} with an 80/10/10 split."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_repo_root() / "data" / "raw",
        help="Path to data/raw (expects blurred/ and sharp/ subfolders)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "split",
        help="Output directory (creates train/val/test subfolders)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing files in out-dir/{train,val,test} before writing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for debugging (0 = no limit)",
    )
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir

    pairs = discover_pairs(raw_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    if not pairs:
        raise SystemExit(
            f"No paired images found under '{raw_dir}'. Expected matching filenames in blurred/ and sharp/."
        )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    if args.clean:
        for name in ("train", "val", "test"):
            target = out_dir / name
            if target.exists():
                shutil.rmtree(target)

    write_split(train_pairs, out_dir, "train")
    write_split(val_pairs, out_dir, "val")
    write_split(test_pairs, out_dir, "test")

    print(
        f"Wrote splits to: {out_dir}\n"
        f"Pairs total: {n}\n"
        f"train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}\n"
        f"Each split contains blurred/ and sharp/ PNGs."
    )


if __name__ == "__main__":
    main()
