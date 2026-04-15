#!/usr/bin/env bash
set -euo pipefail

# One-command runner:
# 1) build prompt A/B/C manifests from one base manifest
# 2) run A and B eval in parallel (2 processes on one GPU, if memory allows)
# 3) run C eval after A/B finish
#
# Why this matches your requirement:
# - each run_batch_eval process loads model once and reuses it for all rows
# - no per-sample model reload
#
# Usage:
#   bash scripts/server/run_okvqa_prompt_abc_eval.sh \
#     research/work/sample_manifest_okvqa_val1000.csv
#
# Optional env vars:
#   RUN_TAG=exp1
#   OUT_ROOT=outputs/phase_ab/eval_abc
#   DEVICE=cuda
#   MODEL=google/gemma-3-4b-it
#   TRANSCODER_SET=tianhux2/gemma3-4b-it-plt
#   MAX_NEW_TOKENS=16
#   LOG_EVERY=20
#   OKVQA_ANN=~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json
#   CORRECT_RULE=vqa_0.3
#   EXAMPLE_INSTRUCTION="What color is the bus?"
#   EXAMPLE_RESPONSE="The answer is yellow."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

BASE_MANIFEST="${1:-research/work/sample_manifest_okvqa_val1000.csv}"
BASE_MANIFEST="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${BASE_MANIFEST}")"
if [[ ! -f "${BASE_MANIFEST}" ]]; then
  echo "[err] base manifest not found: ${BASE_MANIFEST}" >&2
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/eval_abc}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="research/work/abc_${RUN_TAG}"
LOG_DIR="${OUT_ROOT}/logs/${RUN_TAG}"
EVAL_DIR="${OUT_ROOT}/eval/${RUN_TAG}"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-}"
TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
LOG_EVERY="${LOG_EVERY:-20}"
OKVQA_ANN="${OKVQA_ANN:-$HOME/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json}"
CORRECT_RULE="${CORRECT_RULE:-vqa_0.3}"
OKVQA_ANN="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OKVQA_ANN}")"
if [[ ! -f "${OKVQA_ANN}" ]]; then
  echo "[warn] annotations not found: ${OKVQA_ANN}" >&2
  echo "[warn] set OKVQA_ANN to official annotations for VQA soft score." >&2
  if [[ "${CORRECT_RULE}" == vqa_* ]]; then
    CORRECT_RULE="strict_gold"
    echo "[warn] fallback CORRECT_RULE=strict_gold" >&2
  fi
fi

EXAMPLE_INSTRUCTION="${EXAMPLE_INSTRUCTION:-What color is the bus?}"
EXAMPLE_RESPONSE="${EXAMPLE_RESPONSE:-The answer is yellow.}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${EVAL_DIR}"

MANIFEST_A="${WORK_DIR}/manifest_promptA.csv"
MANIFEST_B="${WORK_DIR}/manifest_promptB.csv"
MANIFEST_C="${WORK_DIR}/manifest_promptC_oneshot.csv"

PROMPT_A_TEMPLATE="{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>."
PROMPT_B_TEMPLATE="{question} Reply exactly in the format: The answer is <short answer>."
PROMPT_C_TEMPLATE=$'Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response.\n### Instruction:\n{instruction}\n### Reference Answer:\n{original_response}\n### Response:'

CSV_A="${EVAL_DIR}/promptA_eval.csv"
CSV_B="${EVAL_DIR}/promptB_eval.csv"
CSV_C="${EVAL_DIR}/promptC_eval.csv"

LOG_A="${LOG_DIR}/promptA_eval.log"
LOG_B="${LOG_DIR}/promptB_eval.log"
LOG_C="${LOG_DIR}/promptC_eval.log"

echo "[stage] build A/B/C manifests"
[[ -f "scripts/research/build_prompt_ab_manifests.py" ]] || {
  echo "[err] missing scripts/research/build_prompt_ab_manifests.py" >&2
  exit 1
}
[[ -f "scripts/research/run_batch_eval.py" ]] || {
  echo "[err] missing scripts/research/run_batch_eval.py" >&2
  exit 1
}

python scripts/research/build_prompt_ab_manifests.py \
  --base-manifest "${BASE_MANIFEST}" \
  --out-manifest-a "${MANIFEST_A}" \
  --out-manifest-b "${MANIFEST_B}" \
  --out-manifest-c "${MANIFEST_C}" \
  --prompt-a-template "${PROMPT_A_TEMPLATE}" \
  --prompt-b-template "${PROMPT_B_TEMPLATE}" \
  --prompt-c-template "${PROMPT_C_TEMPLATE}" \
  --example-instruction "${EXAMPLE_INSTRUCTION}" \
  --example-response "${EXAMPLE_RESPONSE}" \
  --copy-gold-from-notes

run_eval() {
  local manifest="$1"
  local out_csv="$2"

  local cmd=(
    python scripts/research/run_batch_eval.py
    --manifest "${manifest}"
    --output-csv "${out_csv}"
    --correct-rule "${CORRECT_RULE}"
    --device "${DEVICE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --log-every "${LOG_EVERY}"
  )
  if [[ -f "${OKVQA_ANN}" ]]; then
    cmd+=(--annotations-json "${OKVQA_ANN}")
  fi
  if [[ -n "${MODEL}" ]]; then
    cmd+=(--model "${MODEL}")
  else
    cmd+=(--transcoder-set "${TRANSCODER_SET}")
  fi
  "${cmd[@]}"
}

echo "[stage] run A and B in parallel"
( run_eval "${MANIFEST_A}" "${CSV_A}" ) >"${LOG_A}" 2>&1 &
PID_A=$!
( run_eval "${MANIFEST_B}" "${CSV_B}" ) >"${LOG_B}" 2>&1 &
PID_B=$!

wait "${PID_A}"
wait "${PID_B}"
echo "[ok] A/B finished"

echo "[stage] run C"
( run_eval "${MANIFEST_C}" "${CSV_C}" ) >"${LOG_C}" 2>&1
echo "[ok] C finished"

echo "[stage] summary"
python - <<'PY' "${CSV_A}" "${CSV_B}" "${CSV_C}"
import csv, os, sys
for p in sys.argv[1:]:
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n = len(rows)
    ok = sum(1 for r in rows if (r.get("correct") or "") == "1")
    print(f"{os.path.basename(p)} rows={n} correct={ok} acc={ok/max(n,1):.4f}")
PY

echo "[done] outputs:"
echo "  manifests: ${WORK_DIR}"
echo "  eval csv:  ${EVAL_DIR}"
echo "  logs:      ${LOG_DIR}"
