"""Convert the training history files written by apps/train.py and apps/train_realtime.py
into TensorBoard event files so that `tensorboard --logdir outputs/tensorboard` shows the
training curves.

This is a one-off helper: the training scripts currently save history as CSV/JSON +
matplotlib PNGs (not TF events). This script replays those histories through
torch.utils.tensorboard.SummaryWriter, producing standard tfevents files per run.

Usage:
    python scripts/export_tensorboard.py [--out-dir outputs/tensorboard]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The pairing of history files to TensorBoard runs (one scalar-group per run).
DEFAULTS = {
    "train": REPO_ROOT / "outputs" / "result" / "history_torch.csv",
    "realtime": REPO_ROOT / "outputs" / "realtime" / "result" / "history_torch_realtime.json",
}


def load_history(path: Path) -> dict[str, list[float]]:
    """Load a history file (CSV or JSON) into {metric_name: [values...]}."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"No rows found in {path}")
        return {key: [float(r[key]) for r in rows] for key in rows[0]}
    # JSON (may contain trailing garbage after the main object, be lenient)
    raw = path.read_text(encoding="utf-8", errors="replace")
    end = raw.rfind("}")
    if end == -1:
        raise ValueError(f"No JSON object found in {path}")
    data = json.loads(raw[: end + 1])
    return {key: [float(v) for v in values] for key, values in data.items()}


def write_run(run_name: str, history: dict[str, list[float]], out_dir: Path) -> Path:
    """Write one tfevents file under out_dir/<run_name>/ and return that directory."""
    from torch.utils.tensorboard import SummaryWriter

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    n_steps = max(len(values) for values in history.values())
    for step in range(n_steps):
        for tag, values in history.items():
            if step < len(values):
                writer.add_scalar(tag, values[step], step + 1)  # 1-based epoch
    writer.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "tensorboard")
    args = parser.parse_args()

    written = []
    for run_name, hist_path in DEFAULTS.items():
        if not hist_path.exists():
            print(f"[skip] {run_name}: no history at {hist_path}")
            continue
        history = load_history(hist_path)
        run_dir = write_run(run_name, history, args.out_dir)
        n_epochs = max(len(v) for v in history.values())
        tags = ", ".join(sorted(history))
        print(f"[ok]   {run_name}: {n_epochs} epochs, {len(history)} metrics -> {run_dir}")
        print(f"      tags: {tags}")
        written.append(run_dir)

    if not written:
        print("Nothing to export. Train first with apps/train.py or apps/train_realtime.py.")
        sys.exit(1)

    print(f"\nAll event files are under: {args.out_dir.resolve()}")
    print(f"Launch with: tensorboard --logdir {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()