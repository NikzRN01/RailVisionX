#!/usr/bin/env bash
set -euo pipefail

# Run this inside Ubuntu WSL, from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh   # (creates .venv-wsl)
#   bash scripts/wsl_train_realtime_gpu.sh

cd "$(dirname "$0")/.."

source .venv-wsl/bin/activate

# Ensure torch deps exist in this venv
python -m pip install -r requirements-wsl.txt

# You can override these via environment variables:
#   REALTIME_SPLIT_DIR=/mnt/a/aidtm/RailVisionX/realtime_data/spilts
#   EPOCHS=10 BATCH_SIZE=32 MICRO_BATCH_SIZE=8
REALTIME_SPLIT_DIR=${REALTIME_SPLIT_DIR:-realtime_data/spilts}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

python apps/train_realtime.py train \
	--realtime-split-dir "${REALTIME_SPLIT_DIR}" \
	--epochs "${EPOCHS}" \
	--batch-size "${BATCH_SIZE}" \
	--micro-batch-size "${MICRO_BATCH_SIZE}" \
	--weights imagenet \
	--device cuda \
	--require-gpu

echo "Done. Check outputs/realtime/result/training_curves_torch.png"
