#!/usr/bin/env bash
set -euo pipefail

# Run this inside Ubuntu WSL, from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

cd "$(dirname "$0")/.."

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

python -m pip install -U pip
pip install -r requirements-wsl.txt

python - <<'PY'
import tensorflow as tf
print('TF', tf.__version__)
print('Build is_cuda_build:', tf.sysconfig.get_build_info().get('is_cuda_build'))
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY
