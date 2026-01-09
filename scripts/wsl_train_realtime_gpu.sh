#!/usr/bin/env bash
set -euo pipefail

# Run this inside Ubuntu WSL, from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh   # (creates .venv-wsl)
#   bash scripts/wsl_train_realtime_gpu.sh

cd "$(dirname "$0")/.."

if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
	echo "[wsl_train_realtime_gpu] This script must be run inside WSL." >&2
	echo "[wsl_train_realtime_gpu] On Windows, use .venv instead." >&2
	exit 1
fi

if [[ ! -d ".venv-wsl" ]]; then
	echo "[wsl_train_realtime_gpu] Missing .venv-wsl. Create it first:" >&2
	echo "  bash scripts/wsl_setup_ubuntu_gpu.sh" >&2
	exit 1
fi

source .venv-wsl/bin/activate

python - <<'PY'
import sys
if sys.version_info >= (3, 13):
    print('[wsl_train_realtime_gpu][warn] Python 3.13+ detected. If installs fail, consider using Ubuntu WSL with Python 3.11/3.12.')
PY

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
