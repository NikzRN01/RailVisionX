#!/usr/bin/env bash
set -euo pipefail

# Run this inside Ubuntu WSL, from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh   # (creates .venv-wsl)
#   bash scripts/wsl_run_train_torch_gpu.sh

cd "$(dirname "$0")/.."

if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "[wsl_train_gpu] This script must be run inside WSL." >&2
  echo "[wsl_train_gpu] On Windows, use .venv instead." >&2
  exit 1
fi

if [[ ! -d ".venv-wsl" ]]; then
  echo "[wsl_train_gpu] Missing .venv-wsl. Create it first:" >&2
  echo "  bash scripts/wsl_setup_ubuntu_tf_gpu.sh" >&2
  exit 1
fi

source .venv-wsl/bin/activate

python - <<'PY'
import sys
if sys.version_info >= (3, 13):
    print('[wsl_train_gpu][warn] Python 3.13+ detected. If installs fail, consider using Ubuntu WSL with Python 3.11/3.12.')
PY

# Ensure torch deps exist in this venv
python -m pip install -r requirements-wsl.txt

python apps/train.py \
  --epochs 10 \
  --batch-size 32 \
  --micro-batch-size 8 \
  --weights imagenet \
  --device cuda \
  --require-gpu

echo "Done. Check outputs/analytics/accuracy_curve_torch.png"