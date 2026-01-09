from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import threading
import queue
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


def _torch_load_checkpoint(path: Path, *, map_location):
    """Load a torch checkpoint robustly across torch versions.

    Prefers weights_only=True (safer) when supported. Some of our checkpoints
    include metadata containing pathlib Path objects; in that case we allowlist
    WindowsPath/PosixPath for weights_only loading. As a last resort (trusted
    local checkpoints), falls back to weights_only=False.
    """

    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch versions don't support weights_only.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Try allowlisting pathlib Path types (common in our saved metadata).
        try:
            from pathlib import PosixPath, WindowsPath

            try:
                from torch.serialization import add_safe_globals  # type: ignore

                add_safe_globals([WindowsPath, PosixPath])
            except Exception:
                pass
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            # Last resort: trusted local checkpoints.
            return torch.load(path, map_location=map_location, weights_only=False)


def _preprocess_tensor_nchw01(x_in, *, mode: str, lowlight_method: str):
    """Apply optional preprocessing to NCHW float tensor in [0,1].

    Uses CPU OpenCV/Pillow ops via src/enhancement utilities, so it is slower.
    """
    if mode == "none":
        return x_in
    if mode != "lowlight":
        raise ValueError("Unsupported preprocess mode. Use: none|lowlight")

    import numpy as np
    import torch

    from src.enhancement.lowlight_enhance import enhance_lowlight

    device = x_in.device
    x_cpu = x_in.detach().to("cpu")
    x_nhwc = x_cpu.permute(0, 2, 3, 1).contiguous().numpy()

    out_list: list[torch.Tensor] = []
    for i in range(x_nhwc.shape[0]):
        img = x_nhwc[i]
        img = enhance_lowlight(img, method=lowlight_method)

        img = np.asarray(img, dtype=np.float32)
        if img.max(initial=0.0) > 1.5:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        chw = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        out_list.append(chw)

    x_out = torch.stack(out_list, dim=0)
    return x_out.to(device, non_blocking=True)


def _postprocess_rgb01(
    img01,
    *,
    mode: str,
    lowlight_method: str,
    auto_lowlight_threshold: float,
):
    """Apply optional postprocessing to RGB HWC image in [0,1].

    This runs on CPU via src/enhancement utilities.
    """
    if mode == "none":
        return img01
    if mode not in ("lowlight", "auto"):
        raise ValueError("Unsupported postprocess mode. Use: none|lowlight|auto")

    import numpy as np

    from src.enhancement.lowlight_enhance import enhance_lowlight
    from src.quality.lowlight_score import mean_intensity

    arr = np.asarray(img01, dtype=np.float32)
    if arr.max(initial=0.0) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    if mode == "auto":
        thr = float(np.clip(float(auto_lowlight_threshold), 0.0, 1.0))
        intensity = float(mean_intensity(arr))
        if intensity >= thr:
            return arr

    out = enhance_lowlight(arr, method=lowlight_method)
    out = np.asarray(out, dtype=np.float32)
    if out.max(initial=0.0) > 1.5:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


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


def calibrate_blur(args: argparse.Namespace) -> None:
    import numpy as np

    from src.realtime.blur_gate import compute_blur_score
    from src.realtime_preprocessor.realtime_dataloader import _discover_images, _read_image_rgb

    sharp_dir = Path(args.sharp_dir)
    blurred_dir = Path(args.blurred_dir)
    metric = str(args.metric)
    sharp_q = float(args.sharp_quantile)
    blur_q = float(args.blur_quantile)

    if not (0.0 < sharp_q < 1.0):
        raise SystemExit("--sharp-quantile must be in (0,1)")
    if not (0.0 < blur_q < 1.0):
        raise SystemExit("--blur-quantile must be in (0,1)")

    sharp_samples = _discover_images(sharp_dir)
    blur_samples = _discover_images(blurred_dir)
    if not sharp_samples:
        raise SystemExit(f"No images found in sharp-dir: {sharp_dir}")
    if not blur_samples:
        raise SystemExit(f"No images found in blurred-dir: {blurred_dir}")

    def scores_for(samples) -> list[float]:
        out: list[float] = []
        for s in samples:
            if s.path is None:
                continue
            img = _read_image_rgb(s.path)
            bs = compute_blur_score(img, metric=metric)
            out.append(float(bs.score))
        return out

    sharp_scores = scores_for(sharp_samples)
    blur_scores = scores_for(blur_samples)

    if len(sharp_scores) < 5 or len(blur_scores) < 5:
        print("Warning: very few samples; thresholds may be unstable")

    sharp_scores_np = np.asarray(sharp_scores, dtype=np.float32)
    blur_scores_np = np.asarray(blur_scores, dtype=np.float32)

    # Recommended thresholds:
    # - sharp_threshold: conservative lower bound of sharp scores
    # - blur_threshold: conservative upper bound of blurred scores
    sharp_threshold = float(np.quantile(sharp_scores_np, sharp_q))
    blur_threshold = float(np.quantile(blur_scores_np, blur_q))

    out = {
        "metric": metric,
        "recommended": {
            "sharp_threshold": sharp_threshold,
            "blur_threshold": blur_threshold,
        },
        "sharp": {
            "n": int(sharp_scores_np.size),
            "min": float(np.min(sharp_scores_np)),
            "p10": float(np.quantile(sharp_scores_np, 0.10)),
            "p50": float(np.quantile(sharp_scores_np, 0.50)),
            "p90": float(np.quantile(sharp_scores_np, 0.90)),
            "max": float(np.max(sharp_scores_np)),
        },
        "blurred": {
            "n": int(blur_scores_np.size),
            "min": float(np.min(blur_scores_np)),
            "p10": float(np.quantile(blur_scores_np, 0.10)),
            "p50": float(np.quantile(blur_scores_np, 0.50)),
            "p90": float(np.quantile(blur_scores_np, 0.90)),
            "max": float(np.max(blur_scores_np)),
        },
        "notes": [
            "Scores are sharpness proxies: higher => sharper.",
            "Enable routing with: --blur-gate --sharp-threshold <value>",
        ],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote: {out_path}")


def infer_multistream(args: argparse.Namespace) -> None:
    """Inference on multiple camera/video sources with per-source queues.

    This is a pragmatic prototype for 3 fixed cameras:
      - one capture thread per source (OpenCV VideoCapture)
      - one inference loop consuming frames round-robin
      - telemetry includes queue depths, FPS, and avg latency
    """

    import numpy as np
    import torch
    import csv

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime.blur_gate import BlurGateConfig, compute_blur_score, severity, should_skip_deblur
    from src.realtime.telemetry import FpsWindow, TimingStats, torch_cuda_memory, try_get_gpu_utilization
    from src.damage.detector import build_damage_detector, Detection
    from src.analytics.detection_metrics import load_damage_annotations, evaluate_detections

    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("infer-multistream requires OpenCV (opencv-python)") from e

    sources = list(args.sources)
    if not sources:
        raise SystemExit("Provide at least one --sources value")

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    result_dir = out_dir / "result"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    postprocess_mode = str(getattr(args, "postprocess", "none"))
    auto_lowlight_threshold = float(getattr(args, "auto_lowlight_threshold", 0.35))
    lowlight_method = str(args.lowlight_method)

    gate = BlurGateConfig(
        metric=str(args.blur_metric),
        sharp_threshold=float(args.sharp_threshold),
        blur_threshold=float(args.blur_threshold),
    )
    enable_gate = bool(args.blur_gate)
    report_every = max(1, int(args.report_every))
    fps_window_s = float(args.fps_window_s)
    max_frames = int(args.max_frames)
    max_frames = 0 if max_frames < 0 else max_frames

    batch_size = max(1, int(args.batch_size))
    max_wait_ms = max(0.0, float(args.max_wait_ms))
    use_cuda = device.type == "cuda"
    use_fp16 = bool(args.fp16) and use_cuda
    use_pin = bool(args.pin_memory) and use_cuda

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = _torch_load_checkpoint(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # One queue per camera
    qmax = int(args.queue_size)
    queues: list[queue.Queue] = [queue.Queue(maxsize=qmax) for _ in sources]
    stop = threading.Event()

    # Stats per camera
    cam_stats: list[TimingStats] = [TimingStats() for _ in sources]
    cam_fps: list[FpsWindow] = [FpsWindow(window_s=fps_window_s) for _ in sources]
    cam_in_fps: list[FpsWindow] = [FpsWindow(window_s=fps_window_s) for _ in sources]

    def parse_source(s: str) -> int | str:
        return int(s) if s.isdigit() else s

    def capture_loop(cam_idx: int, src: str) -> None:
        cap = cv2.VideoCapture(parse_source(src))
        if not cap.isOpened():
            print(f"Failed to open source[{cam_idx}]: {src}")
            stop.set()
            return
        frame_count = 0
        try:
            while not stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_count += 1
                cam_in_fps[cam_idx].tick()

                # Convert to RGB float32 [0,1]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame01 = frame_rgb.astype(np.float32) / 255.0

                # Resize to model size
                if frame01.shape[0] != h or frame01.shape[1] != w:
                    frame01 = cv2.resize(frame01, (w, h), interpolation=cv2.INTER_LINEAR)

                item = {
                    "cam": cam_idx,
                    "key": f"cam{cam_idx}_frame_{frame_count:06d}",
                    "t_capture": time.perf_counter(),
                    "frame_hwc": frame01,
                }
                try:
                    queues[cam_idx].put(item, timeout=0.2)
                except queue.Full:
                    # Drop frames under load; this is intentional for realtime.
                    continue
        finally:
            cap.release()

    threads: list[threading.Thread] = []
    for i, s in enumerate(sources):
        t = threading.Thread(target=capture_loop, args=(i, s), daemon=True)
        t.start()
        threads.append(t)

    from src.analytics.metrics import restoration_metrics

    metrics_rows: list[dict[str, object]] = []
    processed_total = 0
    rr = 0

    detector = build_damage_detector(str(getattr(args, "damage_detector", "none")))
    ann_path = getattr(args, "damage_annotations", None)
    gts = load_damage_annotations(Path(ann_path)) if ann_path else None
    dmg_iou = float(getattr(args, "damage_iou_thr", 0.5))
    dmg_score = float(getattr(args, "damage_score_thr", 0.25))
    preds_orig: dict[str, list[Detection]] = {}
    preds_rest: dict[str, list[Detection]] = {}

    tta = bool(getattr(args, "tta", False))

    def _forward(x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if tta:
                        y0 = model(x)
                        x1 = torch.flip(x, dims=[3])
                        y1 = model(x1)
                        y1 = torch.flip(y1, dims=[3])
                        y = (y0 + y1) * 0.5
                    else:
                        y = model(x)
            else:
                if tta:
                    y0 = model(x)
                    x1 = torch.flip(x, dims=[3])
                    y1 = model(x1)
                    y1 = torch.flip(y1, dims=[3])
                    y = (y0 + y1) * 0.5
                else:
                    y = model(x)
        return y

    def _infer_batch(frames_hwc: list[np.ndarray]) -> np.ndarray:
        x_cpu = torch.from_numpy(np.stack([f.transpose(2, 0, 1) for f in frames_hwc], axis=0)).float().contiguous()
        if preprocess_mode != "none":
            x_cpu = _preprocess_tensor_nchw01(
                x_cpu,
                mode=preprocess_mode,
                lowlight_method=lowlight_method,
            )
        if use_pin:
            x_cpu = x_cpu.pin_memory()
        x = x_cpu.to(device, non_blocking=True) if use_cuda else x_cpu
        y = _forward(x)
        return y.detach().float().cpu().numpy()

    try:
        batch: list[dict[str, object]] = []
        batch_started = time.perf_counter()

        while not stop.is_set():
            if max_frames and processed_total >= max_frames:
                break

            cam_idx = rr % len(sources)
            rr += 1

            try:
                item = queues[cam_idx].get(timeout=0.05)
                batch.append(item)
            except queue.Empty:
                pass

            now = time.perf_counter()
            ready = len(batch) >= batch_size
            if not ready and max_wait_ms > 0.0:
                ready = (now - batch_started) * 1000.0 >= max_wait_ms
            if not ready:
                if not batch and all(q.empty() for q in queues):
                    time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            frames_hwc: list[np.ndarray] = [it["frame_hwc"] for it in batch]  # type: ignore[index]
            cams: list[int] = [int(it["cam"]) for it in batch]  # type: ignore[index]
            keys: list[str] = [str(it["key"]) for it in batch]  # type: ignore[index]
            t_caps: list[float] = [float(it["t_capture"]) for it in batch]  # type: ignore[index]

            blur_meta: list[dict[str, object]] = []
            run_mask: list[bool] = []
            for frame in frames_hwc:
                bs = compute_blur_score(frame, metric=gate.metric)
                sev = severity(bs.score, cfg=gate)
                skip = bool(enable_gate and should_skip_deblur(bs.score, sharp_threshold=gate.sharp_threshold))
                run_mask.append(not skip)
                blur_meta.append(
                    {
                        "severity": sev,
                        "blur_score": float(bs.score),
                        "blur_lap_var": float(bs.lap_var),
                        "blur_grad_mean": float(bs.grad_mean),
                        "skipped_deblur": bool(skip),
                    }
                )

            preds_hwc: list[np.ndarray] = [f for f in frames_hwc]
            to_run = [idx for idx, run in enumerate(run_mask) if run]
            t_inf = 0.0

            if to_run:
                run_frames = [frames_hwc[idx] for idx in to_run]
                t_inf0 = time.perf_counter()
                y = _infer_batch(run_frames)  # NCHW
                t_inf = time.perf_counter() - t_inf0
                for j, idx in enumerate(to_run):
                    preds_hwc[idx] = np.clip(y[j].transpose(1, 2, 0), 0.0, 1.0)

            for cam_i, key, frame, pred, meta, tcap in zip(cams, keys, frames_hwc, preds_hwc, blur_meta, t_caps):
                if bool(meta["skipped_deblur"]):
                    cam_stats[cam_i].skipped += 1

                pred = _postprocess_rgb01(
                    pred,
                    mode=postprocess_mode,
                    lowlight_method=lowlight_method,
                    auto_lowlight_threshold=auto_lowlight_threshold,
                )

                m = restoration_metrics(frame, pred)
                cam_dir = frames_dir / f"cam{cam_i}"
                (cam_dir / "original").mkdir(parents=True, exist_ok=True)
                (cam_dir / "restored").mkdir(parents=True, exist_ok=True)
                _save_rgb01_png(frame, cam_dir / "original" / f"{key}.png")
                _save_rgb01_png(pred, cam_dir / "restored" / f"{key}.png")

                if detector is not None:
                    try:
                        preds_orig.setdefault(key, []).extend(detector.detect(frame))
                        preds_rest.setdefault(key, []).extend(detector.detect(pred))
                    except Exception:
                        pass

                end_to_end = time.perf_counter() - float(tcap)
                metrics_rows.append(
                    {
                        "key": key,
                        "cam": int(cam_i),
                        "latency_end_to_end_ms": float(end_to_end * 1000.0),
                        **meta,
                        **m.as_dict(),
                    }
                )

                st = cam_stats[cam_i]
                st.frames += 1
                st.t_total_s += (time.perf_counter() - t0)
                st.t_infer_s += float(t_inf) / max(1, len(to_run)) if to_run else 0.0
                cam_fps[cam_i].tick()
                processed_total += 1

            if processed_total % report_every == 0:
                msg: dict[str, object] = {
                    "processed": int(processed_total),
                    "queues": [int(q.qsize()) for q in queues],
                    "in_fps": [f.fps() for f in cam_in_fps],
                    "proc_fps": [f.fps() for f in cam_fps],
                    "avg_total_ms": [1000.0 * (s.t_total_s / max(1, s.frames)) for s in cam_stats],
                    "skipped": [int(s.skipped) for s in cam_stats],
                    "batch_size": int(batch_size),
                }
                gpu = try_get_gpu_utilization()
                if gpu is not None:
                    msg.update(gpu)
                tc = torch_cuda_memory()
                if tc is not None:
                    msg.update(tc)
                print(json.dumps(msg))

            batch.clear()
            batch_started = time.perf_counter()

    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)

    if metrics_rows:
        out_csv = result_dir / "restoration_metrics_multistream.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    if detector is not None and gts is not None:
        s0 = evaluate_detections(preds_orig, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        s1 = evaluate_detections(preds_rest, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        summary = {
            "iou_thr": float(dmg_iou),
            "score_thr": float(dmg_score),
            "original": s0.as_dict(),
            "restored": s1.as_dict(),
            "delta": {
                "ap": float(s1.ap - s0.ap),
                "recall": float(s1.recall - s0.recall),
                "precision": float(s1.precision - s0.precision),
            },
        }
        out_json = result_dir / "damage_metrics_multistream.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"damage": summary}))

    print(f"Saved multistream outputs to: {out_dir}")


def train(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import shutil
    from datetime import datetime

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.quality.blur_score import mean_gradient_magnitude, variance_of_laplacian
    from src.quality.lowlight_score import mean_intensity

    out_dir = Path(args.out_dir)
    _, _, _, result_dir = _ensure_out_dirs(out_dir)

    ckpt_best_path = Path(getattr(args, "checkpoint_out", Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt"))
    ckpt_last_path = Path(getattr(args, "checkpoint_last", Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_last.pt"))
    ckpt_best_path.parent.mkdir(parents=True, exist_ok=True)

    # Safety: snapshot current checkpoints so we can rollback if needed.
    try:
        to_backup: list[Path] = []
        if ckpt_best_path.exists():
            to_backup.append(ckpt_best_path)
        if ckpt_last_path.exists():
            to_backup.append(ckpt_last_path)
        if to_backup:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rb_dir = ckpt_best_path.parent / f"rollback_{ts}"
            rb_dir.mkdir(parents=True, exist_ok=True)
            for p in to_backup:
                shutil.copy2(p, rb_dir / p.name)
            print(f"Rollback snapshot saved to: {rb_dir}")
    except Exception:
        pass

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

    aug_motion_blur = bool(getattr(args, "augment_motion_blur", False))
    motion_blur_prob = float(getattr(args, "motion_blur_prob", 0.0))
    motion_blur_max_len = int(getattr(args, "motion_blur_max_len", 15))

    train_dataset = str(getattr(args, "train_dataset", "realtime"))

    if train_dataset == "data":
        from src.preprocessor.dataloader import PairedDeblurDataset
        from src.preprocessor.dataloader import build_torch_dataloader as _build_dl

        data_dir = getattr(args, "data_dir", None)
        train_ds = PairedDeblurDataset(
            split="train",
            data_dir=data_dir,
            image_size=(h, w),
            augment_motion_blur=aug_motion_blur,
            motion_blur_prob=motion_blur_prob,
            motion_blur_max_len=motion_blur_max_len,
            seed=int(args.seed),
            as_torch=True,
        )
        val_ds = PairedDeblurDataset(
            split="val",
            data_dir=data_dir,
            image_size=(h, w),
            augment_motion_blur=False,
            motion_blur_prob=0.0,
            motion_blur_max_len=motion_blur_max_len,
            seed=int(args.seed),
            as_torch=True,
        )

        train_loader = _build_dl(
            train_ds,
            batch_size=micro_batch,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        val_loader = _build_dl(
            val_ds,
            batch_size=micro_batch,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
    else:
        from src.realtime_preprocessor.realtime_dataloader import build_torch_dataloader as _build_dl

        if train_dataset == "realtime-synthetic":
            from src.realtime_preprocessor.realtime_dataloader import RealtimeSyntheticPairsDataset

            train_ds = RealtimeSyntheticPairsDataset(
                split="train",
                split_dir=args.realtime_split_dir,
                image_size=(h, w),
                seed=int(args.seed),
                augment_motion_blur=aug_motion_blur,
                motion_blur_prob=motion_blur_prob,
                motion_blur_max_len=motion_blur_max_len,
                as_torch=True,
            )
            val_ds = RealtimeSyntheticPairsDataset(
                split="val",
                split_dir=args.realtime_split_dir,
                image_size=(h, w),
                seed=int(args.seed) + 1337,
                augment_motion_blur=aug_motion_blur,
                motion_blur_prob=max(0.0, motion_blur_prob * 0.5),
                motion_blur_max_len=motion_blur_max_len,
                as_torch=True,
            )
        else:
            from src.realtime_preprocessor.realtime_dataloader import RealtimePairedFramesDataset

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

        train_loader = _build_dl(
            train_ds,
            batch_size=micro_batch,
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        val_loader = _build_dl(
            val_ds,
            batch_size=micro_batch,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    weights = None if str(args.weights).lower() in ("none", "null") else str(args.weights)
    model = build_deblur_mobilenetv2_torch(weights=weights, backbone_trainable=bool(args.backbone_trainable))
    model.to(device)

    # Optional: initialize from an existing checkpoint (quick fine-tune).
    init_ckpt = getattr(args, "init_checkpoint", None)
    if init_ckpt is not None:
        p = Path(init_ckpt)
        if p.exists():
            st = _torch_load_checkpoint(p, map_location=device)
            if isinstance(st, dict) and "state_dict" in st:
                st = st["state_dict"]
            model.load_state_dict(st)
            print(f"Initialized weights from: {p}")

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
    best_val_loss: float | None = None
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

        # Always save a "last" checkpoint, and update "best" on improved val_loss.
        try:
            payload = {
                "state_dict": model.state_dict(),
                "epoch": int(epoch),
                "train": train_m,
                "val": val_m,
                "args": vars(args),
            }
            torch.save(payload, ckpt_last_path)

            v = float(val_m["loss"])
            if best_val_loss is None or v < best_val_loss:
                best_val_loss = v
                torch.save(payload, ckpt_best_path)
        except Exception:
            # Training should not fail just because checkpoint writing failed.
            pass

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
    postprocess_mode = str(getattr(args, "postprocess", "none"))
    auto_lowlight_threshold = float(getattr(args, "auto_lowlight_threshold", 0.35))
    lowlight_method = str(args.lowlight_method)

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = _torch_load_checkpoint(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"
    use_fp16 = bool(getattr(args, "fp16", False)) and use_cuda
    tta = bool(getattr(args, "tta", False))

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
            )
        x = x.to(device)
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if tta:
                        y0 = model(x)
                        y1 = model(torch.flip(x, dims=[3]))
                        y1 = torch.flip(y1, dims=[3])
                        y = (y0 + y1) * 0.5
                    else:
                        y = model(x)
            else:
                if tta:
                    y0 = model(x)
                    y1 = model(torch.flip(x, dims=[3]))
                    y1 = torch.flip(y1, dims=[3])
                    y = (y0 + y1) * 0.5
                else:
                    y = model(x)
            pred = y.squeeze(0).detach().float().cpu()

        # Compute simple restoration score from input/output only
        blurred_hwc = item["image"].detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = pred.detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = _postprocess_rgb01(
            restored_hwc,
            mode=postprocess_mode,
            lowlight_method=lowlight_method,
            auto_lowlight_threshold=auto_lowlight_threshold,
        )
        m = restoration_metrics(blurred_hwc, restored_hwc)
        metrics_rows.append({"key": key, **m.as_dict()})

        _save_rgb01_png(item["image"], original_dir / f"{key}.png")
        _save_rgb01_png(restored_hwc, restored_dir / f"{key}.png")

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
    import numpy as np

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RealtimeStream
    from src.realtime.blur_gate import BlurGateConfig, compute_blur_score, severity, should_skip_deblur
    from src.realtime.telemetry import FpsWindow, TimingStats, torch_cuda_memory, try_get_gpu_utilization
    from src.damage.detector import build_damage_detector, Detection
    from src.analytics.detection_metrics import load_damage_annotations, evaluate_detections

    out_dir = Path(args.out_dir)
    original_dir, restored_dir, _, result_dir = _ensure_out_dirs(out_dir)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    postprocess_mode = str(getattr(args, "postprocess", "none"))
    auto_lowlight_threshold = float(getattr(args, "auto_lowlight_threshold", 0.35))
    lowlight_method = str(args.lowlight_method)

    gate = BlurGateConfig(
        metric=str(args.blur_metric),
        sharp_threshold=float(args.sharp_threshold),
        blur_threshold=float(args.blur_threshold),
    )
    enable_gate = bool(args.blur_gate)
    report_every = max(1, int(args.report_every))
    fpsw = FpsWindow(window_s=float(args.fps_window_s))
    stats = TimingStats()

    select_best_of = max(1, int(getattr(args, "select_best_of", 1)))

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = _torch_load_checkpoint(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"
    use_fp16 = bool(args.fp16) and use_cuda
    use_pin = bool(args.pin_memory) and use_cuda
    batch_size = max(1, int(args.batch_size))
    max_wait_ms = max(0.0, float(args.max_wait_ms))
    tta = bool(getattr(args, "tta", False))

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

    detector = build_damage_detector(str(getattr(args, "damage_detector", "none")))
    ann_path = getattr(args, "damage_annotations", None)
    gts = load_damage_annotations(Path(ann_path)) if ann_path else None
    dmg_iou = float(getattr(args, "damage_iou_thr", 0.5))
    dmg_score = float(getattr(args, "damage_score_thr", 0.25))
    preds_orig: dict[str, list[Detection]] = {}
    preds_rest: dict[str, list[Detection]] = {}

    def _run_model(x_cpu: torch.Tensor) -> torch.Tensor:
        """Run model on CPU batch NCHW in [0,1]. Returns CPU NCHW."""
        if preprocess_mode != "none":
            x_cpu = _preprocess_tensor_nchw01(
                x_cpu,
                mode=preprocess_mode,
                lowlight_method=lowlight_method,
            )

        if use_pin:
            x_cpu = x_cpu.pin_memory()

        x = x_cpu.to(device, non_blocking=True) if use_cuda else x_cpu
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if tta:
                        y0 = model(x)
                        x1 = torch.flip(x, dims=[3])
                        y1 = model(x1)
                        y1 = torch.flip(y1, dims=[3])
                        y = (y0 + y1) * 0.5
                    else:
                        y = model(x)
            else:
                if tta:
                    y0 = model(x)
                    x1 = torch.flip(x, dims=[3])
                    y1 = model(x1)
                    y1 = torch.flip(y1, dims=[3])
                    y = (y0 + y1) * 0.5
                else:
                    y = model(x)
        return y.detach().float().cpu()

    batch: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    frame_idx = 0

    stream_it = iter(stream)
    while True:
        try:
            item0 = next(stream_it)
        except StopIteration:
            break

        # Optional: select the sharpest of the next N frames (by blur score).
        # This helps with motion blur in videos by avoiding the worst-smeared frames.
        if select_best_of > 1:
            candidates: list[dict[str, object]] = [item0]
            for _ in range(select_best_of - 1):
                try:
                    candidates.append(next(stream_it))
                except StopIteration:
                    break

            best_item = candidates[0]
            best_score = None
            for cand in candidates:
                frame_chw = cand["frame"]
                if not hasattr(frame_chw, "detach"):
                    continue
                frame_chw_t = frame_chw.detach().cpu().contiguous()
                blurred_hwc = frame_chw_t.numpy().transpose(1, 2, 0)
                bs = compute_blur_score(blurred_hwc, metric=gate.metric)
                score = float(bs.score)
                if best_score is None or score > best_score:
                    best_item = cand
                    best_score = score
            item = best_item
        else:
            item = item0

        frame_idx += 1
        batch.append(item)

        now = time.perf_counter()
        ready = len(batch) >= batch_size
        if not ready and max_wait_ms > 0.0:
            ready = (now - batch_started) * 1000.0 >= max_wait_ms
        if not ready:
            continue

        t0 = time.perf_counter()
        keys: list[str] = []
        frames_chw: list[torch.Tensor] = []
        blurred_hwcs: list[np.ndarray] = []
        meta: list[dict[str, object]] = []
        run_mask: list[bool] = []

        for b in batch:
            key = str(b["key"])
            frame_chw = b["frame"]
            if not hasattr(frame_chw, "detach"):
                raise RuntimeError("Expected torch frames from RealtimeStream(as_torch=True)")

            frame_chw_t = frame_chw.detach().cpu().contiguous()
            blurred_hwc = frame_chw_t.numpy().transpose(1, 2, 0)

            bs = compute_blur_score(blurred_hwc, metric=gate.metric)
            sev = severity(bs.score, cfg=gate)
            skip = bool(enable_gate and should_skip_deblur(bs.score, sharp_threshold=gate.sharp_threshold))

            keys.append(key)
            frames_chw.append(frame_chw_t)
            blurred_hwcs.append(blurred_hwc)
            meta.append(
                {
                    "severity": sev,
                    "blur_score": float(bs.score),
                    "blur_lap_var": float(bs.lap_var),
                    "blur_grad_mean": float(bs.grad_mean),
                    "skipped_deblur": bool(skip),
                    "select_best_of": int(select_best_of),
                }
            )
            run_mask.append(not skip)
            if skip:
                stats.skipped += 1

        # Inference for subset; others pass through.
        preds: list[torch.Tensor] = [f for f in frames_chw]
        to_run = [idx for idx, run in enumerate(run_mask) if run]

        t_pre = 0.0
        t_inf = 0.0
        if to_run:
            x_cpu = torch.stack([frames_chw[idx] for idx in to_run], dim=0).contiguous()
            t_pre0 = time.perf_counter()
            if preprocess_mode != "none":
                x_cpu = _preprocess_tensor_nchw01(
                    x_cpu,
                    mode=preprocess_mode,
                    lowlight_method=lowlight_method,
                )
            t_pre = time.perf_counter() - t_pre0

            if use_pin:
                x_cpu = x_cpu.pin_memory()

            x = x_cpu.to(device, non_blocking=True) if use_cuda else x_cpu
            t_inf0 = time.perf_counter()
            with torch.inference_mode():
                if use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        y = model(x)
                else:
                    y = model(x)
            y_cpu = y.detach().float().cpu()
            t_inf = time.perf_counter() - t_inf0

            for j, idx in enumerate(to_run):
                preds[idx] = y_cpu[j]

        t_save0 = time.perf_counter()
        for key, frame_chw_t, pred_chw, blurred_hwc, m0 in zip(keys, frames_chw, preds, blurred_hwcs, meta):
            restored_hwc = pred_chw.detach().cpu().numpy().transpose(1, 2, 0)
            restored_hwc = _postprocess_rgb01(
                restored_hwc,
                mode=postprocess_mode,
                lowlight_method=lowlight_method,
                auto_lowlight_threshold=auto_lowlight_threshold,
            )
            m = restoration_metrics(blurred_hwc, restored_hwc)
            metrics_rows.append({"key": key, **m0, **m.as_dict()})
            _save_rgb01_png(frame_chw_t, original_dir / f"{key}.png")
            _save_rgb01_png(restored_hwc, restored_dir / f"{key}.png")

            if detector is not None:
                try:
                    preds_orig.setdefault(key, []).extend(detector.detect(blurred_hwc))
                    preds_rest.setdefault(key, []).extend(detector.detect(restored_hwc))
                except Exception:
                    pass
        t_save = time.perf_counter() - t_save0

        t_total = time.perf_counter() - t0
        stats.frames += len(batch)
        stats.t_total_s += float(t_total)
        stats.t_preprocess_s += float(t_pre)
        stats.t_infer_s += float(t_inf)
        stats.t_save_s += float(t_save)
        for _ in batch:
            fpsw.tick()

        if frame_idx % report_every == 0:
            avg_total_ms = 1000.0 * (stats.t_total_s / max(1, stats.frames))
            ran = max(1, stats.frames - stats.skipped)
            avg_inf_ms = 1000.0 * (stats.t_infer_s / ran)
            msg = {
                "frames": int(stats.frames),
                "fps_window": fpsw.fps(),
                "avg_total_ms": avg_total_ms,
                "avg_infer_ms": avg_inf_ms,
                "skipped": int(stats.skipped),
                "batch_size": int(batch_size),
            }
            gpu = try_get_gpu_utilization()
            if gpu is not None:
                msg.update(gpu)
            tc = torch_cuda_memory()
            if tc is not None:
                msg.update(tc)
            print(json.dumps(msg))

        batch.clear()
        batch_started = time.perf_counter()

    if metrics_rows:
        out_csv = result_dir / "restoration_metrics_stream.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    if detector is not None and gts is not None:
        s0 = evaluate_detections(preds_orig, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        s1 = evaluate_detections(preds_rest, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        summary = {
            "iou_thr": float(dmg_iou),
            "score_thr": float(dmg_score),
            "original": s0.as_dict(),
            "restored": s1.as_dict(),
            "delta": {
                "ap": float(s1.ap - s0.ap),
                "recall": float(s1.recall - s0.recall),
                "precision": float(s1.precision - s0.precision),
            },
        }
        out_json = result_dir / "damage_metrics_stream.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"damage": summary}))

    print(f"Saved stream inference outputs to: {out_dir}")


def infer_frames(args: argparse.Namespace) -> None:
    import torch
    import csv
    import numpy as np

    from src.enhancement.deblur_net_torch import build_deblur_mobilenetv2_torch
    from src.realtime_preprocessor.realtime_dataloader import RealtimeFolderFramesDataset, build_torch_dataloader
    from src.realtime.blur_gate import BlurGateConfig, compute_blur_score, severity, should_skip_deblur
    from src.realtime.telemetry import FpsWindow, TimingStats, torch_cuda_memory, try_get_gpu_utilization
    from src.damage.detector import build_damage_detector, Detection
    from src.analytics.detection_metrics import load_damage_annotations, evaluate_detections

    out_dir = Path(args.out_dir)
    original_dir, restored_dir, _, result_dir = _ensure_out_dirs(out_dir)

    device = _pick_device(args.device)
    h, w = int(args.image_size[0]), int(args.image_size[1])
    preprocess_mode = str(args.preprocess)
    postprocess_mode = str(getattr(args, "postprocess", "none"))
    auto_lowlight_threshold = float(getattr(args, "auto_lowlight_threshold", 0.35))
    lowlight_method = str(args.lowlight_method)

    gate = BlurGateConfig(
        metric=str(args.blur_metric),
        sharp_threshold=float(args.sharp_threshold),
        blur_threshold=float(args.blur_threshold),
    )
    enable_gate = bool(args.blur_gate)
    report_every = max(1, int(args.report_every))
    fpsw = FpsWindow(window_s=float(args.fps_window_s))
    stats = TimingStats()

    model = build_deblur_mobilenetv2_torch(weights=None, backbone_trainable=False)
    ckpt = Path(args.checkpoint)
    state = _torch_load_checkpoint(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"
    use_fp16 = bool(getattr(args, "fp16", False)) and use_cuda
    tta = bool(getattr(args, "tta", False))

    ds = RealtimeFolderFramesDataset(root_dir=args.frames_dir, image_size=(h, w), as_torch=True)
    loader = build_torch_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)

    from src.analytics.metrics import restoration_metrics

    metrics_rows: list[dict[str, object]] = []

    detector = build_damage_detector(str(getattr(args, "damage_detector", "none")))
    ann_path = getattr(args, "damage_annotations", None)
    gts = load_damage_annotations(Path(ann_path)) if ann_path else None
    dmg_iou = float(getattr(args, "damage_iou_thr", 0.5))
    dmg_score = float(getattr(args, "damage_score_thr", 0.25))
    preds_orig: dict[str, list[Detection]] = {}
    preds_rest: dict[str, list[Detection]] = {}

    for i, batch in enumerate(loader, start=1):
        t0 = time.perf_counter()
        key = batch["key"][0]
        frame_chw = batch["frame"].squeeze(0)
        blurred_hwc = frame_chw.detach().cpu().numpy().transpose(1, 2, 0)

        bs = compute_blur_score(blurred_hwc, metric=gate.metric)
        sev = severity(bs.score, cfg=gate)
        skip = bool(enable_gate and should_skip_deblur(bs.score, sharp_threshold=gate.sharp_threshold))

        t_pre = 0.0
        t_inf = 0.0
        if skip:
            pred = frame_chw.detach().cpu()
            stats.skipped += 1
        else:
            x = frame_chw.unsqueeze(0)
            if preprocess_mode != "none":
                t_pre0 = time.perf_counter()
                x = _preprocess_tensor_nchw01(
                    x,
                    mode=preprocess_mode,
                    lowlight_method=lowlight_method,
                )
                t_pre = time.perf_counter() - t_pre0
            x = x.to(device)

            t_inf0 = time.perf_counter()
            with torch.inference_mode():
                if use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if tta:
                            y0 = model(x)
                            y1 = model(torch.flip(x, dims=[3]))
                            y1 = torch.flip(y1, dims=[3])
                            y = (y0 + y1) * 0.5
                        else:
                            y = model(x)
                else:
                    if tta:
                        y0 = model(x)
                        y1 = model(torch.flip(x, dims=[3]))
                        y1 = torch.flip(y1, dims=[3])
                        y = (y0 + y1) * 0.5
                    else:
                        y = model(x)
                pred = y.squeeze(0).detach().float().cpu()
            t_inf = time.perf_counter() - t_inf0

        restored_hwc = pred.detach().cpu().numpy().transpose(1, 2, 0)
        restored_hwc = _postprocess_rgb01(
            restored_hwc,
            mode=postprocess_mode,
            lowlight_method=lowlight_method,
            auto_lowlight_threshold=auto_lowlight_threshold,
        )
        m = restoration_metrics(blurred_hwc, restored_hwc)
        metrics_rows.append(
            {
                "key": str(key),
                "severity": sev,
                "blur_score": float(bs.score),
                "blur_lap_var": float(bs.lap_var),
                "blur_grad_mean": float(bs.grad_mean),
                "skipped_deblur": bool(skip),
                **m.as_dict(),
            }
        )

        if detector is not None:
            try:
                preds_orig.setdefault(str(key), []).extend(detector.detect(blurred_hwc))
                preds_rest.setdefault(str(key), []).extend(detector.detect(restored_hwc))
            except Exception:
                pass

        t_save0 = time.perf_counter()
        _save_rgb01_png(frame_chw, original_dir / f"{key}.png")
        _save_rgb01_png(restored_hwc, restored_dir / f"{key}.png")
        t_save = time.perf_counter() - t_save0

        t_total = time.perf_counter() - t0
        stats.frames += 1
        stats.t_total_s += t_total
        stats.t_preprocess_s += float(t_pre)
        stats.t_infer_s += float(t_inf)
        stats.t_save_s += float(t_save)
        fpsw.tick()

        if i % report_every == 0:
            avg_total_ms = 1000.0 * (stats.t_total_s / max(1, stats.frames))
            avg_inf_ms = 1000.0 * (stats.t_infer_s / max(1, stats.frames - stats.skipped)) if (stats.frames - stats.skipped) > 0 else 0.0
            msg = {
                "frames": int(stats.frames),
                "fps_window": fpsw.fps(),
                "avg_total_ms": avg_total_ms,
                "avg_infer_ms": avg_inf_ms,
                "skipped": int(stats.skipped),
            }
            gpu = try_get_gpu_utilization()
            if gpu is not None:
                msg.update(gpu)
            tc = torch_cuda_memory()
            if tc is not None:
                msg.update(tc)
            print(json.dumps(msg))

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

    if detector is not None and gts is not None:
        s0 = evaluate_detections(preds_orig, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        s1 = evaluate_detections(preds_rest, gts, iou_thr=dmg_iou, score_thr=dmg_score)
        summary = {
            "iou_thr": float(dmg_iou),
            "score_thr": float(dmg_score),
            "original": s0.as_dict(),
            "restored": s1.as_dict(),
            "delta": {
                "ap": float(s1.ap - s0.ap),
                "recall": float(s1.recall - s0.recall),
                "precision": float(s1.precision - s0.precision),
            },
        }
        out_json = result_dir / "damage_metrics_frames.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"damage": summary}))

    print(f"Saved folder inference outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime training + inference.\n"
            "- train: trains on paired realtime_data/spilts/{train,val}/...\n"
            "- infer-stream: runs webcam/video inference\n"
            "- infer-frames: runs inference on a folder of frames\n"
            "- infer-random: runs inference on random images from data/split\n"
            "- prep-extract: extract video frames into realtime_data/raw\n"
            "- prep-split: split realtime_data/raw into realtime_data/spilts"
        )
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    def prep_extract(args: argparse.Namespace) -> None:
        from src.realtime_preprocessor.realtime_make_splits import extract_to_raw

        src: str = str(args.source)
        if src.isdigit():
            source: int | str = int(src)
        else:
            raw_dir = Path(args.raw_dir)
            rel = Path(src)
            if rel.is_absolute():
                raise SystemExit("--source must be a filename under --raw-dir (absolute paths are not allowed)")

            candidate = (raw_dir / rel).resolve()
            raw_res = raw_dir.resolve()
            try:
                ok = candidate.is_relative_to(raw_res)
            except AttributeError:
                ok = str(candidate).lower().startswith(str(raw_res).lower())

            if not ok:
                raise SystemExit("--source must point inside --raw-dir")
            if not candidate.exists():
                raise SystemExit(f"Video not found under --raw-dir: {rel}")
            source = str(candidate)
        max_frames = None if int(args.max_frames) <= 0 else int(args.max_frames)
        n = extract_to_raw(
            source=source,
            raw_dir=Path(args.raw_dir),
            out_kind=str(args.out_kind),
            stride=int(args.stride),
            max_frames=max_frames,
            prefix=str(args.prefix),
        )
        print(f"Extracted {n} frames into: {Path(args.raw_dir) / args.out_kind}")

    def prep_split(args: argparse.Namespace) -> None:
        from src.realtime_preprocessor.realtime_make_splits import split_raw_to_spilts

        counts = split_raw_to_spilts(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            mode=str(args.mode),
            seed=int(args.seed),
            clean=bool(args.clean),
            limit=int(args.limit),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
        )
        print(
            f"Wrote splits to: {Path(args.out_dir)}\n"
            f"Total: {counts['total']}  train: {counts['train']}  val: {counts['val']}  test: {counts['test']}"
        )

    p_train = sub.add_parser("train", help="Train on paired realtime images")
    p_train.add_argument("--realtime-split-dir", type=Path, default=None, help="Defaults to realtime_data/spilts")
    p_train.add_argument(
        "--train-dataset",
        choices=("realtime", "realtime-synthetic", "data"),
        default="realtime",
        help=(
            "Train dataset source: realtime (paired realtime_data/spilts), "
            "realtime-synthetic (unpaired realtime_data/spilts + synthetic blur), "
            "or data (data/split)."
        ),
    )
    p_train.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "split",
        help="Used when --train-dataset=data. Points to the split root containing train/val/test folders.",
    )
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
        choices=("none", "lowlight"),
        default="none",
        help="Optional preprocessing applied to INPUT frames before the model (runs on CPU; slower).",
    )
    p_train.add_argument(
        "--augment-motion-blur",
        action="store_true",
        help="Augment training inputs with synthetic motion blur (helps moving vehicles/plates).",
    )
    p_train.add_argument(
        "--motion-blur-prob",
        type=float,
        default=0.25,
        help="When --augment-motion-blur is enabled, probability of applying motion blur per training sample.",
    )
    p_train.add_argument(
        "--motion-blur-max-len",
        type=int,
        default=15,
        help="When --augment-motion-blur is enabled, maximum motion blur kernel length.",
    )
    _ll_methods = (
        "best",
        "plates",
        "plates-strong",
        "auto-gamma",
        "clahe",
        "unsharp",
        "auto-gamma+clahe",
        "auto-gamma+clahe+unsharp",
    )
    p_train.add_argument(
        "--lowlight-method",
        choices=_ll_methods,
        default="best",
        help="Lowlight enhancement pipeline used by --preprocess/--postprocess.",
    )
    p_train.add_argument(
        "--quality-metrics",
        action="store_true",
        help=(
            "Compute simple no-ground-truth quality metrics between INPUT (blurred) and OUTPUT (restored). "
            "This adds CPU overhead (uses NumPy). Metrics are saved into history_torch_realtime.json/csv."
        ),
    )
    p_train.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
        help="Where to save the best checkpoint (by lowest val_loss).",
    )
    p_train.add_argument(
        "--checkpoint-last",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_last.pt",
        help="Where to save the last checkpoint (overwritten each epoch).",
    )

    p_train.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize weights from (quick fine-tune).",
    )
    p_train.add_argument(
        "--quality-max-batches",
        type=int,
        default=5,
        help="When --quality-metrics is enabled, only evaluate on the first N batches per epoch (default: 5).",
    )
    p_train.set_defaults(func=train)

    p_prep_extract = sub.add_parser("prep-extract", help="Extract frames from camera/video into realtime_data/raw")
    p_prep_extract.add_argument("--raw-dir", type=Path, default=Path("realtime_data") / "raw")
    p_prep_extract.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0) or VIDEO FILENAME under --raw-dir (e.g. myvideo.mp4)",
    )
    p_prep_extract.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    p_prep_extract.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    p_prep_extract.add_argument("--prefix", type=str, default="frame")
    p_prep_extract.add_argument(
        "--out-kind",
        choices=["frames", "blurred"],
        default="frames",
        help="Subfolder under raw/ to write frames into",
    )
    p_prep_extract.set_defaults(func=prep_extract)

    p_prep_split = sub.add_parser("prep-split", help="Split realtime_data/raw into realtime_data/spilts")
    p_prep_split.add_argument("--raw-dir", type=Path, default=Path("realtime_data") / "raw")
    p_prep_split.add_argument(
        "--out-dir",
        type=Path,
        default=Path("realtime_data") / "spilts",
        help="Output directory for train/val/test",
    )
    p_prep_split.add_argument("--seed", type=int, default=42)
    p_prep_split.add_argument("--clean", action="store_true")
    p_prep_split.add_argument(
        "--mode",
        choices=["auto", "paired", "unpaired"],
        default="auto",
        help="auto: paired if raw/blurred+raw/sharp exist else unpaired",
    )
    p_prep_split.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p_prep_split.add_argument("--train-ratio", type=float, default=0.7)
    p_prep_split.add_argument("--val-ratio", type=float, default=0.1)
    p_prep_split.set_defaults(func=prep_split)

    p_stream = sub.add_parser("infer-stream", help="Inference on webcam/video")
    p_stream.add_argument("--source", type=str, default="0", help="Camera index (0) or video path")
    p_stream.add_argument("--stride", type=int, default=1)
    p_stream.add_argument(
        "--select-best-of",
        type=int,
        default=1,
        help=(
            "For videos: read N frames and process only the sharpest one (by blur score). "
            "Improves plate readability under motion blur, but reduces temporal resolution."
        ),
    )
    p_stream.add_argument("--max-frames", type=int, default=200)
    p_stream.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_stream.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_stream.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_stream.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_stream.add_argument("--preprocess", choices=("none", "lowlight"), default="none")
    p_stream.add_argument(
        "--postprocess",
        choices=("none", "lowlight", "auto"),
        default="none",
        help=(
            "Optional postprocessing applied to RESTORED frames after the model (runs on CPU; slower). "
            "Use 'auto' to only enhance frames detected as low-light."
        ),
    )
    p_stream.add_argument(
        "--auto-lowlight-threshold",
        type=float,
        default=0.35,
        help="When --postprocess=auto, enhance only if mean intensity < threshold (0..1).",
    )
    p_stream.add_argument(
        "--lowlight-method",
        choices=_ll_methods,
        default="best",
        help="Lowlight enhancement pipeline used by --preprocess/--postprocess.",
    )
    p_stream.add_argument("--blur-gate", action="store_true", help="Skip deblur when frame is already sharp")
    p_stream.add_argument("--blur-metric", choices=("lap_var", "grad_mean", "combo"), default="combo")
    p_stream.add_argument("--sharp-threshold", type=float, default=0.020, help="Skip deblur if score >= this")
    p_stream.add_argument("--blur-threshold", type=float, default=0.008, help="Severe blur if score <= this")
    p_stream.add_argument("--report-every", type=int, default=30, help="Print telemetry JSON every N frames")
    p_stream.add_argument("--fps-window-s", type=float, default=2.0, help="Sliding window size for FPS")
    p_stream.add_argument("--batch-size", type=int, default=4, help="Micro-batch size for inference")
    p_stream.add_argument("--max-wait-ms", type=float, default=30.0, help="Max wait to fill a batch (0 disables)")
    p_stream.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory + non-blocking H2D")
    p_stream.add_argument("--fp16", action="store_true", help="Use CUDA autocast fp16 for faster inference")
    p_stream.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation (hflip ensembling). Improves quality slightly but ~2x slower.",
    )
    p_stream.add_argument("--damage-detector", type=str, default="none", help="Damage detector: none|placeholder")
    p_stream.add_argument("--damage-annotations", type=Path, default=None, help="Path to GT annotation JSON")
    p_stream.add_argument("--damage-iou-thr", type=float, default=0.5, help="IoU threshold (e.g. 0.5)")
    p_stream.add_argument("--damage-score-thr", type=float, default=0.25, help="Score threshold for recall/precision")
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
    p_frames.add_argument("--preprocess", choices=("none", "lowlight"), default="none")
    p_frames.add_argument(
        "--postprocess",
        choices=("none", "lowlight", "auto"),
        default="none",
        help=(
            "Optional postprocessing applied to RESTORED frames after the model (runs on CPU; slower). "
            "Use 'auto' to only enhance frames detected as low-light."
        ),
    )
    p_frames.add_argument(
        "--auto-lowlight-threshold",
        type=float,
        default=0.35,
        help="When --postprocess=auto, enhance only if mean intensity < threshold (0..1).",
    )
    p_frames.add_argument(
        "--lowlight-method",
        choices=_ll_methods,
        default="best",
        help="Lowlight enhancement pipeline used by --preprocess/--postprocess.",
    )
    p_frames.add_argument("--blur-gate", action="store_true", help="Skip deblur when frame is already sharp")
    p_frames.add_argument("--blur-metric", choices=("lap_var", "grad_mean", "combo"), default="combo")
    p_frames.add_argument("--sharp-threshold", type=float, default=0.020, help="Skip deblur if score >= this")
    p_frames.add_argument("--blur-threshold", type=float, default=0.008, help="Severe blur if score <= this")
    p_frames.add_argument("--report-every", type=int, default=30, help="Print telemetry JSON every N frames")
    p_frames.add_argument("--fps-window-s", type=float, default=2.0, help="Sliding window size for FPS")
    p_frames.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation (hflip ensembling). Improves quality slightly but ~2x slower.",
    )
    p_frames.add_argument("--damage-detector", type=str, default="none", help="Damage detector: none|placeholder")
    p_frames.add_argument("--damage-annotations", type=Path, default=None, help="Path to GT annotation JSON")
    p_frames.add_argument("--damage-iou-thr", type=float, default=0.5, help="IoU threshold (e.g. 0.5)")
    p_frames.add_argument("--damage-score-thr", type=float, default=0.25, help="Score threshold for recall/precision")
    p_frames.set_defaults(func=infer_frames)

    p_cal = sub.add_parser("calibrate-blur", help="Calibrate blur thresholds from sample folders")
    p_cal.add_argument(
        "--sharp-dir",
        type=Path,
        required=True,
        help="Folder of known-sharp frames (day, or curated sharp night frames)",
    )
    p_cal.add_argument(
        "--blurred-dir",
        type=Path,
        required=True,
        help="Folder of known-blurry frames (night motion blur, or curated blur)",
    )
    p_cal.add_argument("--metric", choices=("lap_var", "grad_mean", "combo"), default="combo")
    p_cal.add_argument("--sharp-quantile", type=float, default=0.10, help="Lower quantile of sharp scores")
    p_cal.add_argument("--blur-quantile", type=float, default=0.90, help="Upper quantile of blurred scores")
    p_cal.add_argument(
        "--out-json",
        type=Path,
        default=Path("outputs") / "realtime" / "result" / "blur_thresholds.json",
        help="Write recommended thresholds here",
    )
    p_cal.set_defaults(func=calibrate_blur)

    p_ms = sub.add_parser("infer-multistream", help="Inference on multiple camera/video sources")
    p_ms.add_argument(
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="One or more camera indices or video paths (e.g. 0 1 2 or rtsp://...)",
    )
    p_ms.add_argument("--max-frames", type=int, default=300, help="Stop after N processed frames (0=run forever)")
    p_ms.add_argument("--queue-size", type=int, default=8, help="Per-camera queue size (frames)")
    p_ms.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_ms.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_ms.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_ms.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_ms.add_argument("--preprocess", choices=("none", "lowlight"), default="none")
    p_ms.add_argument(
        "--postprocess",
        choices=("none", "lowlight", "auto"),
        default="none",
        help=(
            "Optional postprocessing applied to RESTORED frames after the model (runs on CPU; slower). "
            "Use 'auto' to only enhance frames detected as low-light."
        ),
    )
    p_ms.add_argument(
        "--auto-lowlight-threshold",
        type=float,
        default=0.35,
        help="When --postprocess=auto, enhance only if mean intensity < threshold (0..1).",
    )
    p_ms.add_argument(
        "--lowlight-method",
        choices=_ll_methods,
        default="best",
        help="Lowlight enhancement pipeline used by --preprocess/--postprocess.",
    )
    p_ms.add_argument("--blur-gate", action="store_true", help="Skip deblur when frame is already sharp")
    p_ms.add_argument("--blur-metric", choices=("lap_var", "grad_mean", "combo"), default="combo")
    p_ms.add_argument("--sharp-threshold", type=float, default=0.020)
    p_ms.add_argument("--blur-threshold", type=float, default=0.008)
    p_ms.add_argument("--report-every", type=int, default=60)
    p_ms.add_argument("--fps-window-s", type=float, default=2.0)
    p_ms.add_argument("--batch-size", type=int, default=6, help="Micro-batch size for inference")
    p_ms.add_argument("--max-wait-ms", type=float, default=30.0, help="Max wait to fill a batch (0 disables)")
    p_ms.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory + non-blocking H2D")
    p_ms.add_argument("--fp16", action="store_true", help="Use CUDA autocast fp16 for faster inference")
    p_ms.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation (hflip ensembling). Improves quality slightly but ~2x slower.",
    )
    p_ms.add_argument("--damage-detector", type=str, default="none", help="Damage detector: none|placeholder")
    p_ms.add_argument("--damage-annotations", type=Path, default=None, help="Path to GT annotation JSON")
    p_ms.add_argument("--damage-iou-thr", type=float, default=0.5, help="IoU threshold (e.g. 0.5)")
    p_ms.add_argument("--damage-score-thr", type=float, default=0.25, help="Score threshold for recall/precision")
    p_ms.set_defaults(func=infer_multistream)

    p_rand = sub.add_parser("infer-random", help="Inference on random images from data/split")
    p_rand.add_argument("--split", choices=("train", "val", "test"), default="test")
    p_rand.add_argument("--num-images", type=int, default=20)
    p_rand.add_argument("--seed", type=int, default=42)
    p_rand.add_argument("--image-size", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    p_rand.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p_rand.add_argument(
        "--fp16",
        action="store_true",
        help="Use CUDA autocast fp16 for faster inference",
    )
    p_rand.add_argument(
        "--tta",
        action="store_true",
        help="Test-time augmentation (hflip ensembling). Improves quality slightly but ~2x slower.",
    )
    p_rand.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / "checkpoints" / "deblur_mobilenetv2_unet_torch_best.pt",
    )
    p_rand.add_argument("--out-dir", type=Path, default=Path("outputs") / "realtime")
    p_rand.add_argument("--preprocess", choices=("none", "lowlight"), default="none")
    p_rand.add_argument(
        "--postprocess",
        choices=("none", "lowlight", "auto"),
        default="none",
        help=(
            "Optional postprocessing applied to RESTORED frames after the model (runs on CPU; slower). "
            "Use 'auto' to only enhance frames detected as low-light."
        ),
    )
    p_rand.add_argument(
        "--auto-lowlight-threshold",
        type=float,
        default=0.35,
        help="When --postprocess=auto, enhance only if mean intensity < threshold (0..1).",
    )
    p_rand.add_argument(
        "--lowlight-method",
        choices=_ll_methods,
        default="best",
        help="Lowlight enhancement pipeline used by --preprocess/--postprocess.",
    )
    p_rand.set_defaults(func=infer_random)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
