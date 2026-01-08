from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


# Allow running this file directly: `python apps/train_realtime.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_out_dirs(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    frames_dir = out_dir / "frames"
    original_dir = frames_dir / "original"
    restored_dir = frames_dir / "restored"
    gt_dir = frames_dir / "groundtruth"
    result_dir = out_dir / "result"

    original_dir.mkdir(parents=True, exist_ok=True)
    restored_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return original_dir, restored_dir, gt_dir, result_dir


def _save_rgb01_png(img01, path: Path) -> None:
    import numpy as np

    arr = img01
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    img8 = (arr * 255.0 + 0.5).astype(np.uint8)

    try:
        from PIL import Image

        Image.fromarray(img8).save(path)
    except Exception:
        # Fallback to OpenCV if PIL isn't available
        import cv2  # type: ignore

        img_bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img_bgr)


def _pick_device(device_flag: str):
    import torch

    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preprocess_tensor_nchw01(x_in, *, mode: str, lowlight_method: str, denoise_method: str):
    """Apply optional preprocessing to NCHW float tensor in [0,1].

    Uses CPU OpenCV/Pillow ops via src/enhancement utilities, so it is slower.
    """
    if mode == "none":
        return x_in

    import numpy as np
    import torch

    from src.enhancement.denoise import denoise
    from src.enhancement.lowlight_enhance import enhance_lowlight

    device = x_in.device
    x_cpu = x_in.detach().to("cpu")
    x_nhwc = x_cpu.permute(0, 2, 3, 1).contiguous().numpy()

    out_list: list[torch.Tensor] = []
    for i in range(x_nhwc.shape[0]):
        img = x_nhwc[i]
        if mode in ("lowlight", "both"):
            img = enhance_lowlight(img, method=lowlight_method)
        if mode in ("denoise", "both"):
            img = denoise(img, method=denoise_method)

        img = np.asarray(img, dtype=np.float32)
        if img.max(initial=0.0) > 1.5:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        chw = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        out_list.append(chw)

    x_out = torch.stack(out_list, dim=0)
    return x_out.to(device, non_blocking=True)


def _plot_history(history: dict[str, list[float]], out_path: Path, epochs: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(1, epochs + 1))

    # Accuracy-only graph (requested)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, history.get("accuracy", []), label="train")
    plt.plot(xs, history.get("val_accuracy", []), label="val")
    plt.title("Pixel Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "accuracy_curve_torch.png")
    plt.close()

    # Combined curves (loss/mae/psnr/ssim)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(xs, history.get("loss", []), label="train")
    plt.plot(xs, history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(xs, history.get("mae", []), label="train")
    plt.plot(xs, history.get("val_mae", []), label="val")
    plt.title("MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(xs, history.get("psnr", []), label="train")
    plt.plot(xs, history.get("val_psnr", []), label="val")
    plt.title("PSNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(xs, history.get("ssim", []), label="train")
    plt.plot(xs, history.get("val_ssim", []), label="val")
    plt.title("SSIM")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path / "training_curves_torch.png")
    plt.close()


def train(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RealtimePairedFramesDataset, build_torch_dataloader
    from src.quality.blur_score import mean_gradient_magnitude, variance_of_laplacian
    from src.quality.lowlight_score import mean_intensity

    out_dir = Path(args.out_dir)
    _, _, _, result_dir = _ensure_out_dirs(out_dir)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = _pick_device(args.device)
    if args.require_gpu and device.type != "cuda":
        raise SystemExit("--require-gpu was set but CUDA is not available")

    effective_batch = int(args.batch_size)
    micro_batch = int(args.micro_batch_size) if args.micro_batch_size is not None else effective_batch
    if device.type == "cuda" and args.micro_batch_size is None:
        micro_batch = min(effective_batch, 8)

    if micro_batch <= 0 or effective_batch <= 0:
        raise SystemExit("batch sizes must be > 0")
    if micro_batch > effective_batch:
        raise SystemExit("--micro-batch-size cannot be larger than --batch-size")
    if effective_batch % micro_batch != 0:
        raise SystemExit("--batch-size must be divisible by --micro-batch-size")
    accum_steps = max(1, effective_batch // micro_batch)

    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    lowlight_method = str(args.lowlight_method)
    denoise_method = str(args.denoise_method)

    train_ds = RealtimePairedFramesDataset(
        split="train",
        split_dir=args.realtime_split_dir,
        image_size=(h, w),
        as_torch=True,
    )
    val_ds = RealtimePairedFramesDataset(
        split="val",
        split_dir=args.realtime_split_dir,
        image_size=(h, w),
        as_torch=True,
    )

    train_loader = build_torch_dataloader(
        train_ds,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = build_torch_dataloader(
        val_ds,
        batch_size=micro_batch,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    weights = None if str(args.weights).lower() in ("none", "null") else str(args.weights)
    model = build_deblur_mobilenetv2_torch(weights=weights, backbone_trainable=bool(args.backbone_trainable))
    model.to(device)

    def charbonnier(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return torch.mean(torch.sqrt((y_true - y_pred) ** 2 + eps * eps))

    try:
        from pytorch_msssim import ssim as msssim_ssim  # type: ignore

        def ssim01(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            return msssim_ssim(y_pred, y_true, data_range=1.0, size_average=True)

    except Exception:

        def ssim01(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0, device=y_true.device)

    def loss_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        l1 = torch.mean(torch.abs(y_true - y_pred))
        ssim_loss = 1.0 - ssim01(y_true, y_pred)
        return 0.8 * l1 + 0.2 * ssim_loss + 0.1 * charbonnier(y_true, y_pred)

    def psnr_db(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((y_true - y_pred) ** 2)
        mse = torch.clamp(mse, min=1e-12)
        return 10.0 * torch.log10(1.0 / mse)

    def pixel_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        ok = (torch.abs(y_true - y_pred) < threshold).float()
        return ok.mean()

    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    history: dict[str, list[float]] = {
        "loss": [],
        "mae": [],
        "psnr": [],
        "ssim": [],
        "accuracy": [],
        "val_loss": [],
        "val_mae": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_accuracy": [],
    }

    if bool(args.quality_metrics):
        history.update(
            {
                "lap_var_in": [],
                "lap_var_out": [],
                "lap_var_delta": [],
                "grad_mean_in": [],
                "grad_mean_out": [],
                "grad_mean_delta": [],
                "intensity_in": [],
                "intensity_out": [],
                "intensity_delta": [],
                "val_lap_var_in": [],
                "val_lap_var_out": [],
                "val_lap_var_delta": [],
                "val_grad_mean_in": [],
                "val_grad_mean_out": [],
                "val_grad_mean_delta": [],
                "val_intensity_in": [],
                "val_intensity_out": [],
                "val_intensity_delta": [],
            }
        )

    def run_epoch(loader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        total_loss = total_mae = total_psnr = total_ssim = total_acc = 0.0
        n = 0

        q_enabled = bool(args.quality_metrics)
        q_max_batches = max(0, int(args.quality_max_batches))
        q_lap_in = q_lap_out = q_lap_delta = 0.0
        q_grad_in = q_grad_out = q_grad_delta = 0.0
        q_int_in = q_int_out = q_int_delta = 0.0
        q_count = 0

        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            x = batch["blurred"].to(device, non_blocking=True)
            y = batch["sharp"].to(device, non_blocking=True)

            if x.shape[-2:] != (h, w):
                x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            if y.shape[-2:] != (h, w):
                y = torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

            x_for_metrics = x

            if preprocess_mode != "none":
                with torch.no_grad():
                    x = _preprocess_tensor_nchw01(
                        x,
                        mode=preprocess_mode,
                        lowlight_method=lowlight_method,
                        denoise_method=denoise_method,
                    )

            with torch.set_grad_enabled(train_mode):
                pred = model(x)
                loss = loss_fn(y, pred)

                if train_mode:
                    (loss / float(accum_steps)).backward()
                    if (step + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optim.step()
                        optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                mae = torch.mean(torch.abs(y - pred))
                psnr = psnr_db(y, pred)
                ssim_val = ssim01(y, pred)
                acc = pixel_accuracy(y, pred)

                if q_enabled and (q_max_batches <= 0 or step < q_max_batches):
                    bsz = int(x_for_metrics.shape[0])
                    k = min(bsz, 2)
                    x_np = x_for_metrics[:k].detach().cpu().numpy()
                    p_np = pred[:k].detach().cpu().numpy()
                    for j in range(k):
                        blurred_hwc = np.transpose(x_np[j], (1, 2, 0))
                        restored_hwc = np.transpose(p_np[j], (1, 2, 0))

                        lap_in = variance_of_laplacian(blurred_hwc)
                        lap_out = variance_of_laplacian(restored_hwc)
                        grad_in = mean_gradient_magnitude(blurred_hwc)
                        grad_out = mean_gradient_magnitude(restored_hwc)
                        inten_in = mean_intensity(blurred_hwc)
                        inten_out = mean_intensity(restored_hwc)

                        q_lap_in += lap_in
                        q_lap_out += lap_out
                        q_lap_delta += (lap_out - lap_in)
                        q_grad_in += grad_in
                        q_grad_out += grad_out
                        q_grad_delta += (grad_out - grad_in)
                        q_int_in += inten_in
                        q_int_out += inten_out
                        q_int_delta += (inten_out - inten_in)
                        q_count += 1

            bsz = int(x.shape[0])
            total_loss += float(loss.detach().cpu()) * bsz
            total_mae += float(mae.detach().cpu()) * bsz
            total_psnr += float(psnr.detach().cpu()) * bsz
            total_ssim += float(ssim_val.detach().cpu()) * bsz
            total_acc += float(acc.detach().cpu()) * bsz
            n += bsz

        if train_mode and (len(loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / max(1, n),
            "mae": total_mae / max(1, n),
            "psnr": total_psnr / max(1, n),
            "ssim": total_ssim / max(1, n),
            "accuracy": total_acc / max(1, n),
            **(
                {
                    "lap_var_in": q_lap_in / max(1, q_count),
                    "lap_var_out": q_lap_out / max(1, q_count),
                    "lap_var_delta": q_lap_delta / max(1, q_count),
                    "grad_mean_in": q_grad_in / max(1, q_count),
                    "grad_mean_out": q_grad_out / max(1, q_count),
                    "grad_mean_delta": q_grad_delta / max(1, q_count),
                    "intensity_in": q_int_in / max(1, q_count),
                    "intensity_out": q_int_out / max(1, q_count),
                    "intensity_delta": q_int_delta / max(1, q_count),
                }
                if q_enabled
                else {}
            ),
        }

    epochs = int(args.epochs)
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(train_loader, train_mode=True)
        val_m = run_epoch(val_loader, train_mode=False)

        print(
            f"Epoch {epoch}/{epochs} "
            f"loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} mae={train_m['mae']:.4f} psnr={train_m['psnr']:.2f} ssim={train_m['ssim']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} val_mae={val_m['mae']:.4f} val_psnr={val_m['psnr']:.2f} val_ssim={val_m['ssim']:.4f}"
        )

        history["loss"].append(train_m["loss"])
        history["mae"].append(train_m["mae"])
        history["psnr"].append(train_m["psnr"])
        history["ssim"].append(train_m["ssim"])
        history["accuracy"].append(train_m["accuracy"])
        history["val_loss"].append(val_m["loss"])
        history["val_mae"].append(val_m["mae"])
        history["val_psnr"].append(val_m["psnr"])
        history["val_ssim"].append(val_m["ssim"])
        history["val_accuracy"].append(val_m["accuracy"])

        if bool(args.quality_metrics):
            history["lap_var_in"].append(train_m["lap_var_in"])
            history["lap_var_out"].append(train_m["lap_var_out"])
            history["lap_var_delta"].append(train_m["lap_var_delta"])
            history["grad_mean_in"].append(train_m["grad_mean_in"])
            history["grad_mean_out"].append(train_m["grad_mean_out"])
            history["grad_mean_delta"].append(train_m["grad_mean_delta"])
            history["intensity_in"].append(train_m["intensity_in"])
            history["intensity_out"].append(train_m["intensity_out"])
            history["intensity_delta"].append(train_m["intensity_delta"])

            history["val_lap_var_in"].append(val_m["lap_var_in"])
            history["val_lap_var_out"].append(val_m["lap_var_out"])
            history["val_lap_var_delta"].append(val_m["lap_var_delta"])
            history["val_grad_mean_in"].append(val_m["grad_mean_in"])
            history["val_grad_mean_out"].append(val_m["grad_mean_out"])
            history["val_grad_mean_delta"].append(val_m["grad_mean_delta"])
            history["val_intensity_in"].append(val_m["intensity_in"])
            history["val_intensity_out"].append(val_m["intensity_out"])
            history["val_intensity_delta"].append(val_m["intensity_delta"])

        (result_dir / "history_torch_realtime.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(history).to_csv(result_dir / "history_torch_realtime.csv", index=False)
    except Exception:
        pass

    _plot_history(history, result_dir, epochs=epochs)
    print(f"Saved realtime training outputs to: {out_dir}")


def infer_random(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import csv

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RandomSplitInferenceDataset

    out_dir = Path(args.out_dir)
    original_dir, restored_dir, gt_dir, result_dir = _ensure_out_dirs(out_dir)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    lowlight_method = str(args.lowlight_method)
    denoise_method = str(args.denoise_method)

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = RandomSplitInferenceDataset(split=args.split, kind="blurred", image_size=(h, w), as_torch=True)
    rng = random.Random(int(args.seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: int(args.num_images)]

    # No-ground-truth restoration metrics (blurred -> restored)
    from src.analytics.metrics import restoration_metrics

    metrics_rows: list[dict[str, object]] = []

    for i, idx in enumerate(idxs, start=1):
        item = ds[idx]
        key = item["key"]
        x = item["image"].unsqueeze(0)
        if preprocess_mode != "none":
            x = _preprocess_tensor_nchw01(
                x,
                mode=preprocess_mode,
                lowlight_method=lowlight_method,
                denoise_method=denoise_method,
            )
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).detach().cpu()

        # Compute simple restoration score from input/output only
        blurred_hwc = item["image"].detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = pred.detach().cpu().numpy().transpose(1, 2, 0)
        m = restoration_metrics(blurred_hwc, restored_hwc)
        metrics_rows.append({"key": key, **m.as_dict()})

        _save_rgb01_png(item["image"], original_dir / f"{key}.png")
        _save_rgb01_png(pred, restored_dir / f"{key}.png")

        # Optional groundtruth from data/split/<split>/sharp/<key>.png
        gt_path = REPO_ROOT / "data" / "split" / str(args.split) / "sharp" / f"{key}.png"
        if gt_path.exists():
            try:
                from PIL import Image

                with Image.open(gt_path) as im:
                    im = im.convert("RGB")
                    arr = np.asarray(im).astype(np.float32) / 255.0
                _save_rgb01_png(arr, gt_dir / f"{key}.png")
            except Exception:
                pass

        if i % 10 == 0 or i == len(idxs):
            print(f"Saved {i}/{len(idxs)} outputs")

    if metrics_rows:
        out_csv = result_dir / "restoration_metrics_random.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"Saved inference outputs to: {out_dir}")


def infer_stream(args: argparse.Namespace) -> None:
    import torch
    import csv

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RealtimeStream

    out_dir = Path(args.out_dir)
    original_dir, restored_dir, _, result_dir = _ensure_out_dirs(out_dir)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    lowlight_method = str(args.lowlight_method)
    denoise_method = str(args.denoise_method)

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    src: str = args.source
    source: int | str = int(src) if src.isdigit() else src

    stream = RealtimeStream(
        source=source,
        stride=int(args.stride),
        max_frames=None if int(args.max_frames) <= 0 else int(args.max_frames),
        image_size=(h, w),
        as_torch=True,
    )

    from src.analytics.metrics import restoration_metrics

    metrics_rows: list[dict[str, object]] = []

    for i, item in enumerate(stream, start=1):
        key = item["key"]
        x = item["frame"].unsqueeze(0)
        if preprocess_mode != "none":
            x = _preprocess_tensor_nchw01(
                x,
                mode=preprocess_mode,
                lowlight_method=lowlight_method,
                denoise_method=denoise_method,
            )
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).detach().cpu()

        blurred_hwc = item["frame"].detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = pred.detach().cpu().numpy().transpose(1, 2, 0)
        m = restoration_metrics(blurred_hwc, restored_hwc)
        metrics_rows.append({"key": key, **m.as_dict()})

        _save_rgb01_png(item["frame"], original_dir / f"{key}.png")
        _save_rgb01_png(pred, restored_dir / f"{key}.png")

        if i % 10 == 0:
            print(f"Saved {i} frames")

    if metrics_rows:
        out_csv = result_dir / "restoration_metrics_stream.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"Saved stream inference outputs to: {out_dir}")


def infer_frames(args: argparse.Namespace) -> None:
    import torch
    import csv

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RealtimeFolderFramesDataset, build_torch_dataloader

    out_dir = Path(args.out_dir)
    original_dir, restored_dir, _, result_dir = _ensure_out_dirs(out_dir)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    lowlight_method = str(args.lowlight_method)
    denoise_method = str(args.denoise_method)

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ds = RealtimeFolderFramesDataset(root_dir=args.frames_dir, image_size=(h, w), as_torch=True)
    loader = build_torch_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)

    from src.analytics.metrics import restoration_metrics

    metrics_rows: list[dict[str, object]] = []

    for i, batch in enumerate(loader, start=1):
        key = batch["key"][0]
        x = batch["frame"]
        if preprocess_mode != "none":
            x = _preprocess_tensor_nchw01(
                x,
                mode=preprocess_mode,
                lowlight_method=lowlight_method,
                denoise_method=denoise_method,
            )
        x = x.to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).detach().cpu()

        blurred_hwc = batch["frame"].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = pred.detach().cpu().numpy().transpose(1, 2, 0)
        m = restoration_metrics(blurred_hwc, restored_hwc)
        metrics_rows.append({"key": str(key), **m.as_dict()})

        _save_rgb01_png(batch["frame"].squeeze(0), original_dir / f"{key}.png")
        _save_rgb01_png(pred, restored_dir / f"{key}.png")

        if int(args.max_images) > 0 and i >= int(args.max_images):
            break

        if i % 10 == 0:
            print(f"Saved {i} images")

    if metrics_rows:
        out_csv = result_dir / "restoration_metrics_frames.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    print(f"Saved folder inference outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime training + inference.\n"
            "- train: trains on paired realtime_data/spilts/{train,val}/...\n"
            "- infer-stream: runs webcam/video inference\n"
            "- infer-frames: runs inference on a folder of frames\n"
            "- infer-random: runs inference on random images from data/split"
        )
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train on paired realtime images")
    p_train.add_argument("--realtime-split-dir", type=Path, default=None, help="Defaults to realtime_data/spilts")
    p_train.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--micro-batch-size", type=int, default=None)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_train.add_argument("--require-gpu", action="store_true")
    p_train.add_argument("--weights", type=str, default="imagenet")
    p_train.add_argument("--backbone-trainable", action="store_true")
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_train.add_argument(
        "--preprocess",
        choices=("none", "lowlight", "denoise", "both"),
        default="none",
        help="Optional preprocessing applied to INPUT frames before the model (runs on CPU; slower).",
    )
    p_train.add_argument("--lowlight-method", type=str, default="best")
    p_train.add_argument("--denoise-method", type=str, default="best")
    p_train.add_argument(
        "--quality-metrics",
        action="store_true",
        help=(
            "Compute simple no-ground-truth quality metrics between INPUT (blurred) and OUTPUT (restored). "
            "This adds CPU overhead (uses NumPy). Metrics are saved into history_torch_realtime.json/csv."
        ),
    )
    p_train.add_argument(
        "--quality-max-batches",
        type=int,
        default=5,
        help="When --quality-metrics is enabled, only evaluate on the first N batches per epoch (default: 5).",
    )
    p_train.set_defaults(func=train)

    p_stream = sub.add_parser("infer-stream", help="Inference on webcam/video")
    p_stream.add_argument("--source", type=str, default="0", help="Camera index (0) or video path")
    p_stream.add_argument("--stride", type=int, default=1)
    p_stream.add_argument("--max-frames", type=int, default=200)
    p_stream.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_stream.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_stream.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_stream.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_stream.add_argument("--preprocess", choices=("none", "lowlight", "denoise", "both"), default="none")
    p_stream.add_argument("--lowlight-method", type=str, default="best")
    p_stream.add_argument("--denoise-method", type=str, default="best")
    p_stream.set_defaults(func=infer_stream)

    p_frames = sub.add_parser("infer-frames", help="Inference on frames folder")
    p_frames.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("realtime_data") / "raw" / "blurred",
        help="Folder containing images (png/jpg) to restore",
    )
    p_frames.add_argument("--max-images", type=int, default=200)
    p_frames.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_frames.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_frames.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_frames.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_frames.add_argument("--preprocess", choices=("none", "lowlight", "denoise", "both"), default="none")
    p_frames.add_argument("--lowlight-method", type=str, default="best")
    p_frames.add_argument("--denoise-method", type=str, default="best")
    p_frames.set_defaults(func=infer_frames)

    p_rand = sub.add_parser("infer-random", help="Inference on random images from data/split")
    p_rand.add_argument("--split", choices=("train", "val", "test"), default="test")
    p_rand.add_argument("--num-images", type=int, default=20)
    p_rand.add_argument("--seed", type=int, default=42)
    p_rand.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_rand.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_rand.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_rand.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_rand.add_argument("--preprocess", choices=("none", "lowlight", "denoise", "both"), default="none")
    p_rand.add_argument("--lowlight-method", type=str, default="best")
    p_rand.add_argument("--denoise-method", type=str, default="best")
    p_rand.set_defaults(func=infer_random)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
