#!/usr/bin/env bash
set -euo pipefail

# Run full smoke flow with one command:
#   bash scripts/server/smoke_one_command.sh [image_path] [output_pt]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_PATH="${1:-${ROOT_DIR}/demos/img/gemma/213.png}"
OUTPUT_PT="${2:-${ROOT_DIR}/outputs/gemma_demo_213.pt}"

# shellcheck source=/dev/null
source "${ROOT_DIR}/scripts/server/dev.sh"

python "${ROOT_DIR}/scripts/server/check_hf_access.py"
bash "${ROOT_DIR}/scripts/server/run_gemma_smoke.sh" "${IMAGE_PATH}" "${OUTPUT_PT}"

