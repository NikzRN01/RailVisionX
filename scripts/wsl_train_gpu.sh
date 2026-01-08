#!/usr/bin/env bash
set -euo pipefail

# Run this inside Ubuntu WSL, from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh   # (creates .venv-wsl)
#   bash scripts/wsl_run_train_torch_gpu.sh

cd "$(dirname "$0")/.."

source .venv-wsl/bin/activate

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