from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


# Allow running this file directly: `python apps/train_deblur_torch.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch MobileNetV2-based deblur model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "split")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Actual per-step batch size. If set smaller than --batch-size, gradients are accumulated "
            "to keep effective batch size == --batch-size. Default on CUDA: 8."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--weights", type=str, default="imagenet", help="'imagenet' or 'none'")
    parser.add_argument("--backbone-trainable", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs") / "analytics")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("outputs") / "result",
        help="Also write plots/history here (requested outputs folder).",
    )
    parser.add_argument(
        "--preprocess",
        choices=("none", "lowlight"),
        default="none",
        help=(
            "Optional preprocessing applied to the INPUT (blurred) images before the model. "
            "Note: this runs on CPU using OpenCV/Pillow and will slow training."
        ),
    )
    parser.add_argument(
        "--lowlight-method",
        type=str,
        default="best",
        help="Low-light method passed to enhance_lowlight(): best|auto-gamma|clahe|unsharp|auto-gamma+clahe|auto-gamma+clahe+unsharp",
    )
    parser.add_argument(
        "--quality-metrics",
        action="store_true",
        help=(
            "Compute simple no-ground-truth quality metrics between INPUT (blurred) and OUTPUT (restored). "
            "This adds CPU overhead (uses NumPy). Metrics are saved into history_torch.json/csv."
        ),
    )
    parser.add_argument(
        "--quality-max-batches",
        type=int,
        default=5,
        help="When --quality-metrics is enabled, only evaluate on the first N batches per epoch (default: 5).",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.preprocessor.dataloader import PairedDeblurDataset
    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.enhancement.lowlight_enhance import enhance_lowlight
    from src.quality.blur_score import mean_gradient_magnitude, variance_of_laplacian
    from src.quality.lowlight_score import mean_intensity

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def pick_device() -> torch.device:
        if args.device == "cpu":
            return torch.device("cpu")
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
            return torch.device("cuda")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = pick_device()
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

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if accum_steps > 1:
        print(f"Using gradient accumulation: micro_batch={micro_batch}, effective_batch={effective_batch}, steps={accum_steps}")

    h, w = int(args.image_size[0]), int(args.image_size[1])

    def preprocess_batch(x_in: torch.Tensor) -> torch.Tensor:
        """Apply optional preprocessing to a NCHW float tensor in [0,1]."""
        mode = str(args.preprocess)
        if mode == "none":
            return x_in
        if mode != "lowlight":
            raise ValueError("Unsupported preprocess mode. Use: none|lowlight")

        # Do CPU preprocessing (OpenCV/Pillow). Inputs do not need gradients.
        x_cpu = x_in.detach().to("cpu")
        x_nhwc = x_cpu.permute(0, 2, 3, 1).contiguous().numpy()

        out_list: list[torch.Tensor] = []
        for i in range(x_nhwc.shape[0]):
            img = x_nhwc[i]
            img = enhance_lowlight(img, method=str(args.lowlight_method))

            img = np.asarray(img, dtype=np.float32)
            if img.max(initial=0.0) > 1.5:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            chw = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            out_list.append(chw)

        x_out = torch.stack(out_list, dim=0)
        return x_out.to(device, non_blocking=True)

    train_ds = PairedDeblurDataset(split="train", data_dir=args.data_dir, as_torch=True)
    val_ds = PairedDeblurDataset(split="val", data_dir=args.data_dir, as_torch=True)

    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=micro_batch, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    weights = None if args.weights.lower() in ("none", "null") else args.weights
    model = build_deblur_mobilenetv2_torch(weights=weights, backbone_trainable=args.backbone_trainable)
    model.to(device)

    # Loss: weighted L1 + (1-SSIM) + Charbonnier
    def charbonnier(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return torch.mean(torch.sqrt((y_true - y_pred) ** 2 + eps * eps))

    # SSIM metric/loss helper (optional dependency)
    try:
        from pytorch_msssim import ssim as msssim_ssim  # type: ignore

        def ssim01(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            return msssim_ssim(y_pred, y_true, data_range=1.0, size_average=True)

    except Exception:
        msssim_ssim = None

        def ssim01(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
            # Fallback: return a dummy value so training still runs.
            # (Metric will be noisy; install pytorch-msssim for real SSIM.)
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

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

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

    if args.quality_metrics:
        # Metrics comparing INPUT (blurred) -> OUTPUT (restored)
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

    def run_epoch(loader: DataLoader, train: bool) -> dict[str, float]:
        model.train(train)
        total_loss = total_mae = total_psnr = total_ssim = total_acc = 0.0
        n = 0

        # Optional quality metrics (INPUT vs OUTPUT) on a subset for speed
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

            # Resize safety (if dataset ever changes)
            if x.shape[-2:] != (h, w):
                x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            if y.shape[-2:] != (h, w):
                y = torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

            # Keep a copy of the *original* input used for quality metrics.
            x_for_metrics = x

            if args.preprocess != "none":
                with torch.no_grad():
                    x = preprocess_batch(x)

            with torch.set_grad_enabled(train):
                pred = model(x)
                loss = loss_fn(y, pred)

                if train:
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
                    # Compute on at most 2 samples per batch to limit CPU cost.
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

        # flush tail optimizer step if steps not divisible (should be divisible by construction)
        if train and (len(loader) % accum_steps) != 0:
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

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(train_loader, train=True)
        val_m = run_epoch(val_loader, train=False)

        print(
            f"Epoch {epoch}/{args.epochs} "
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

        if args.quality_metrics:
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

        # write history as we go (helps if interrupted)
        (args.plot_dir / "history_torch.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.result_dir / "history_torch.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Also write a CSV for convenience
    import pandas as pd

    df = pd.DataFrame(history)
    df.to_csv(args.plot_dir / "history_torch.csv", index=False)
    df.to_csv(args.result_dir / "history_torch.csv", index=False)

    # Plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, args.epochs + 1))

    def save_line(x, y1, y2, title: str, ylab: str, path: Path) -> None:
        plt.figure(figsize=(7, 4))
        plt.plot(x, y1, label="train")
        plt.plot(x, y2, label="val")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylab)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    save_line(
        epochs,
        history["accuracy"],
        history["val_accuracy"],
        "Pixel Accuracy",
        "accuracy",
        args.plot_dir / "accuracy_curve_torch.png",
    )
    save_line(
        epochs,
        history["accuracy"],
        history["val_accuracy"],
        "Pixel Accuracy",
        "accuracy",
        args.result_dir / "accuracy_curve_torch.png",
    )

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["mae"], label="train")
    plt.plot(epochs, history["val_mae"], label="val")
    plt.title("MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["psnr"], label="train")
    plt.plot(epochs, history["val_psnr"], label="val")
    plt.title("PSNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["ssim"], label="train")
    plt.plot(epochs, history["val_ssim"], label="val")
    plt.title("SSIM")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.plot_dir / "training_curves_torch.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["mae"], label="train")
    plt.plot(epochs, history["val_mae"], label="val")
    plt.title("MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["psnr"], label="train")
    plt.plot(epochs, history["val_psnr"], label="val")
    plt.title("PSNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["ssim"], label="train")
    plt.plot(epochs, history["val_ssim"], label="val")
    plt.title("SSIM")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.result_dir / "training_curves_torch.png")
    plt.close()

    print(f"Saved plots/history to: {args.plot_dir}")
    print(f"Also saved plots/history to: {args.result_dir}")
    print("Done.")


if __name__ == "__main__":
    # Avoid fork-related issues in WSL DataLoader
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
