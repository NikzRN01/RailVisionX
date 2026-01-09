#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
	echo "[wsl_run_dashboard] This script must be run inside WSL." >&2
	echo "[wsl_run_dashboard] On Windows PowerShell, use: .\\.venv\\Scripts\\Activate.ps1" >&2
	exit 1
fi

if [[ ! -d ".venv-wsl" ]]; then
	echo "[wsl_run_dashboard] Missing .venv-wsl. Create it first:" >&2
	echo "  bash scripts/wsl_setup_ubuntu_tf_gpu.sh" >&2
	exit 1
fi

source .venv-wsl/bin/activate

# Warning: some packages may lag on very new Python versions.
python - <<'PY'
import sys
if sys.version_info >= (3, 13):
    print('[wsl_run_dashboard][warn] Python 3.13+ detected. If installs fail, consider using Ubuntu WSL with Python 3.11/3.12.')
PY

# Base deps (opencv headless, torch, etc.)
python -m pip install -r requirements-wsl.txt

# Dashboard deps
# - streamlit: UI
# - scikit-image: psnr/ssim metrics used by src/dashboard/pages/realtime.py
python -m pip install -U streamlit scikit-image

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8501}
ENTRY=${ENTRY:-apps/dashboard.py}

echo "[wsl_run_dashboard] Starting Streamlit dashboard"
echo "  Entry: ${ENTRY}"
echo "  URL:   http://localhost:${PORT}"

python -m streamlit run "${ENTRY}" \
	--server.address "${HOST}" \
	--server.port "${PORT}" \
	--server.headless true
