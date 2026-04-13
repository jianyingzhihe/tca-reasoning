#!/usr/bin/env bash
set -euo pipefail

# Create venv and install dependencies needed for circuit-tracer CLI.
# Usage:
#   bash scripts/server/setup_env.sh [.venv]

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup_env] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[setup_env] Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# Install local package first (editable), then runtime deps.
python -m pip install -e . --no-deps
python -m pip install \
  "torch>=2.0.0" \
  "transformers>=4.57.0,<5.0.0" \
  "transformer-lens==2.18.0" \
  "huggingface_hub>=0.26.0" \
  "pydantic>=2.0.0" \
  "safetensors>=0.5.0" \
  "tokenizers>=0.21.0" \
  "tqdm>=4.60.0" \
  "einops>=0.8.0" \
  "numpy>=1.24.0,<2.0.0" \
  "pyyaml>=5.1" \
  "requests>=2.28.0" \
  "pillow>=10.0.0"

echo "[setup_env] Done. Activate with: source ${VENV_DIR}/bin/activate"

