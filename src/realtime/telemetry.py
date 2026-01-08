from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class TimingStats:
    frames: int = 0
    skipped: int = 0
    t_total_s: float = 0.0
    t_preprocess_s: float = 0.0
    t_infer_s: float = 0.0
    t_save_s: float = 0.0


class FpsWindow:
    """Simple sliding-window FPS counter."""

    def __init__(self, *, window_s: float = 2.0) -> None:
        self.window_s = float(window_s)
        self._ts: Deque[float] = deque()

    def tick(self) -> None:
        now = time.perf_counter()
        self._ts.append(now)
        cutoff = now - self.window_s
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        dt = self._ts[-1] - self._ts[0]
        if dt <= 1e-9:
            return 0.0
        return float((len(self._ts) - 1) / dt)


def try_get_gpu_utilization() -> dict[str, float] | None:
    """Best-effort GPU telemetry.

    - If NVML is available (nvidia-ml-py3 / pynvml), returns utilization + memory.
    - Otherwise returns None.

    Notes:
      - Jetson support via NVML can vary by JetPack version.
    """

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out = {
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_util_percent": float(util.memory),
            "gpu_mem_used_mb": float(mem.used) / (1024.0 * 1024.0),
            "gpu_mem_total_mb": float(mem.total) / (1024.0 * 1024.0),
        }
        return out
    except Exception:
        return None


def torch_cuda_memory() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "torch_cuda_mem_alloc_mb": float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0),
            "torch_cuda_mem_reserved_mb": float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0),
        }
    except Exception:
        return None
