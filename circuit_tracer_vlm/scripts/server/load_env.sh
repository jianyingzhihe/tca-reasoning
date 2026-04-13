#!/usr/bin/env bash
set -euo pipefail

# Load KEY=VALUE lines from .env into current shell.
# Usage: source scripts/server/load_env.sh [path_to_env]

ENV_FILE="${1:-.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[load_env] .env file not found: ${ENV_FILE}" >&2
  return 1 2>/dev/null || exit 1
fi

while IFS='=' read -r key value; do
  [[ -z "${key}" ]] && continue
  [[ "${key}" =~ ^[[:space:]]*# ]] && continue

  key="$(echo "${key}" | xargs)"
  value="${value:-}"
  value="$(echo "${value}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"

  export "${key}=${value}"
done < "${ENV_FILE}"

# Backward compatibility with old key name HF=...
if [[ -n "${HF:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HF}"
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Set defaults if not provided
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"
echo "[load_env] Loaded env from ${ENV_FILE}"

