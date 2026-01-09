#!/usr/bin/env bash
set -euo pipefail

# Run this inside WSL (Ubuntu recommended), from the repo root:
#   cd /mnt/a/aidtm/RailVisionX
#   bash scripts/wsl_setup_ubuntu_tf_gpu.sh

if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
	echo "[wsl_setup] This script is intended to run inside WSL." >&2
	echo "[wsl_setup] On Windows PowerShell, use .venv instead." >&2
	exit 1
fi

sudo apt-get update

sudo apt-get install -y python3 python3-venv python3-pip

cd "$(dirname "$0")/.."

echo "[wsl_setup] Using python3 for .venv-wsl" >&2

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

python -m pip install -U pip
pip install -r requirements-wsl.txt

python -V

python - <<'PY'
import sys
print('Python version:', sys.version)
if sys.version_info >= (3, 13):
	print('[wsl_setup][warn] Python 3.13+ detected. Some packages may not have wheels yet; if installs fail, use an Ubuntu WSL with Python 3.11/3.12.')
PY

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
	print('cuda device:', torch.cuda.get_device_name(0))
PY

echo "[wsl_setup] Done. You can now run: bash scripts/wsl_run_dashboard.sh" >&2
