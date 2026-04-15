#!/usr/bin/env bash
set -euo pipefail

# Overnight runner for full OKVQA val A/B/C prompt evaluation.
# Uses the stable sequential pipeline:
#   scripts/server/run_okvqa_prompt_abc_eval_simple.sh
#
# Usage:
#   bash scripts/server/run_okvqa_prompt_abc_overnight.sh \
#     research/work/sample_manifest_okvqa_val_full.csv
#
# Optional env vars:
#   RUN_TAG=okvqa_val_full_abc_night1
#   DEVICE=cuda
#   MODEL=google/gemma-3-4b-it
#   TRANSCODER_SET=tianhux2/gemma3-4b-it-plt
#   MAX_NEW_TOKENS=8
#   LOG_EVERY=10
#   CORRECT_RULE=vqa_0.3
#   OUT_ROOT=outputs/phase_ab/eval_abc
#   OKVQA_LIMIT=0   # 0 means full split

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_MANIFEST="${1:-research/work/sample_manifest_okvqa_val_full.csv}"
RUN_TAG="${RUN_TAG:-overnight_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/eval_abc}"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-}"
TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
LOG_EVERY="${LOG_EVERY:-10}"
CORRECT_RULE="${CORRECT_RULE:-vqa_0.3}"
OKVQA_LIMIT="${OKVQA_LIMIT:-0}"

LOG_DIR="${OUT_ROOT}/logs/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
NIGHT_LOG="${LOG_DIR}/night_abc.log"

echo "[night] start: $(date '+%F %T')"
echo "[night] run_tag=${RUN_TAG}"
echo "[night] base_manifest=${BASE_MANIFEST}"
echo "[night] device=${DEVICE} max_new_tokens=${MAX_NEW_TOKENS} log_every=${LOG_EVERY}"
echo "[night] out_root=${OUT_ROOT}"
echo "[night] full_split_if_okvqa_limit_0=${OKVQA_LIMIT}"
echo "[night] log_file=${NIGHT_LOG}"

export RUN_TAG OUT_ROOT DEVICE MODEL TRANSCODER_SET MAX_NEW_TOKENS LOG_EVERY CORRECT_RULE OKVQA_LIMIT

bash scripts/server/run_okvqa_prompt_abc_eval_simple.sh "${BASE_MANIFEST}" 2>&1 | tee "${NIGHT_LOG}"

echo "[night] end: $(date '+%F %T')"
echo "[night] done. main log: ${NIGHT_LOG}"

