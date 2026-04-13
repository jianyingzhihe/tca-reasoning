#!/usr/bin/env bash
# One-command server session bootstrap.
# Usage:
#   source scripts/server/dev.sh
#   source scripts/server/dev.sh /path/to/.env /path/to/.venv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-${ROOT_DIR}/.env}"
VENV_DIR="${2:-${ROOT_DIR}/.venv}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[dev] Missing env file: ${ENV_FILE}" >&2
  echo "[dev] Create it first: cp ${ROOT_DIR}/.env.example ${ROOT_DIR}/.env" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[dev] venv not found at ${VENV_DIR}, creating now..."
  bash "${ROOT_DIR}/scripts/server/setup_env.sh" "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/server/load_env.sh" "${ENV_FILE}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "[dev] Warning: HF_TOKEN/HUGGINGFACE_HUB_TOKEN is empty." >&2
fi

echo "[dev] Ready."
echo "[dev] Python: $(which python)"
echo "[dev] circuit-tracer: $(which circuit-tracer)"
echo "[dev] HF_HOME=${HF_HOME:-unset}"

