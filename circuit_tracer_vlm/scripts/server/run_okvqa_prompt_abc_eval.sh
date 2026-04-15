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
#   OKVQA_ROOT=~/tca-reasoning/data/okvqa
#   OKVQA_LIMIT=1000  # 0 means full split
#   EXAMPLE_INSTRUCTION="What color is the bus?"
#   EXAMPLE_RESPONSE="The answer is yellow."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

csv_done_rows() {
  local csv_path="$1"
  python - <<'PY' "${csv_path}"
import csv, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0)
    raise SystemExit(0)
with open(p, "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

manifest_rows() {
  local manifest="$1"
  python - <<'PY' "${manifest}"
import csv, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0)
    raise SystemExit(0)
with open(p, "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

render_bar() {
  local done="$1"
  local total="$2"
  local width="${3:-20}"
  if [[ "${total}" -le 0 ]]; then
    printf "[%${width}s]" "" | tr ' ' '.'
    return
  fi
  if [[ "${done}" -gt "${total}" ]]; then
    done="${total}"
  fi
  local filled=$((done * width / total))
  local empty=$((width - filled))
  printf "["
  printf "%${filled}s" "" | tr ' ' '#'
  printf "%${empty}s" "" | tr ' ' '.'
  printf "]"
}

progress_pct() {
  local done="$1"
  local total="$2"
  if [[ "${total}" -le 0 ]]; then
    echo 0
  else
    echo $((done * 100 / total))
  fi
}

monitor_parallel_progress() {
  local pid_a="$1"
  local pid_b="$2"
  local csv_a="$3"
  local csv_b="$4"
  local total_a="$5"
  local total_b="$6"

  local start_ts
  start_ts="$(date +%s)"

  while kill -0 "${pid_a}" 2>/dev/null || kill -0 "${pid_b}" 2>/dev/null; do
    local da db pa pb elapsed
    da="$(csv_done_rows "${csv_a}")"
    db="$(csv_done_rows "${csv_b}")"
    pa="$(progress_pct "${da}" "${total_a}")"
    pb="$(progress_pct "${db}" "${total_b}")"
    elapsed=$(( $(date +%s) - start_ts ))

    printf "\r[progress][AB] A %5d/%-5d %3d%% %s | B %5d/%-5d %3d%% %s | elapsed=%4ds" \
      "${da}" "${total_a}" "${pa}" "$(render_bar "${da}" "${total_a}")" \
      "${db}" "${total_b}" "${pb}" "$(render_bar "${db}" "${total_b}")" \
      "${elapsed}"
    sleep 5
  done
  local da db pa pb elapsed
  da="$(csv_done_rows "${csv_a}")"
  db="$(csv_done_rows "${csv_b}")"
  pa="$(progress_pct "${da}" "${total_a}")"
  pb="$(progress_pct "${db}" "${total_b}")"
  elapsed=$(( $(date +%s) - start_ts ))
  printf "\r[progress][AB] A %5d/%-5d %3d%% %s | B %5d/%-5d %3d%% %s | elapsed=%4ds\n" \
    "${da}" "${total_a}" "${pa}" "$(render_bar "${da}" "${total_a}")" \
    "${db}" "${total_b}" "${pb}" "$(render_bar "${db}" "${total_b}")" \
    "${elapsed}"
}

monitor_single_progress() {
  local pid="$1"
  local csv_path="$2"
  local total="$3"

  local start_ts
  start_ts="$(date +%s)"

  while kill -0 "${pid}" 2>/dev/null; do
    local d p elapsed
    d="$(csv_done_rows "${csv_path}")"
    p="$(progress_pct "${d}" "${total}")"
    elapsed=$(( $(date +%s) - start_ts ))
    printf "\r[progress][C ] C %5d/%-5d %3d%% %s | elapsed=%4ds" \
      "${d}" "${total}" "${p}" "$(render_bar "${d}" "${total}")" "${elapsed}"
    sleep 5
  done
  local d p elapsed
  d="$(csv_done_rows "${csv_path}")"
  p="$(progress_pct "${d}" "${total}")"
  elapsed=$(( $(date +%s) - start_ts ))
  printf "\r[progress][C ] C %5d/%-5d %3d%% %s | elapsed=%4ds\n" \
    "${d}" "${total}" "${p}" "$(render_bar "${d}" "${total}")" "${elapsed}"
}

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

BASE_MANIFEST="${1:-research/work/sample_manifest_okvqa_val1000.csv}"
BASE_MANIFEST="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${BASE_MANIFEST}")"

OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/eval_abc}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WORK_DIR="research/work/abc_${RUN_TAG}"
LOG_DIR="${OUT_ROOT}/logs/${RUN_TAG}"
EVAL_DIR="${OUT_ROOT}/eval/${RUN_TAG}"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-}"
TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
LOG_EVERY="${LOG_EVERY:-1}"
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

OKVQA_ROOT="${OKVQA_ROOT:-$HOME/tca-reasoning/data/okvqa}"
OKVQA_LIMIT="${OKVQA_LIMIT:-1000}"
OKVQA_SEED="${OKVQA_SEED:-42}"
OKVQA_SPLIT="${OKVQA_SPLIT:-val}"
OKVQA_QUESTIONS="${OKVQA_QUESTIONS:-$OKVQA_ROOT/questions/OpenEnded_mscoco_val2014_questions.json}"
OKVQA_ANNOTATIONS="${OKVQA_ANNOTATIONS:-$OKVQA_ROOT/annotations/mscoco_val2014_annotations.json}"
OKVQA_IMAGE_ROOT="${OKVQA_IMAGE_ROOT:-$OKVQA_ROOT/images}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${EVAL_DIR}"
mkdir -p "$(dirname "${BASE_MANIFEST}")"

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
[[ -f "scripts/research/okvqa_to_manifest.py" ]] || {
  echo "[err] missing scripts/research/okvqa_to_manifest.py" >&2
  exit 1
}

MANIFEST_LINES=0
if [[ -f "${BASE_MANIFEST}" ]]; then
  MANIFEST_LINES="$(wc -l < "${BASE_MANIFEST}")"
fi
if [[ ! -f "${BASE_MANIFEST}" || "${MANIFEST_LINES}" -le 1 ]]; then
  echo "[stage] base manifest missing/empty -> auto-build from OKVQA"
  python scripts/research/okvqa_to_manifest.py \
    --questions "${OKVQA_QUESTIONS}" \
    --annotations "${OKVQA_ANNOTATIONS}" \
    --image-root "${OKVQA_IMAGE_ROOT}" \
    --split "${OKVQA_SPLIT}" \
    --output "${BASE_MANIFEST}" \
    --limit "${OKVQA_LIMIT}" \
    --seed "${OKVQA_SEED}" \
    --id-prefix okvqa
fi

MANIFEST_LINES="$(wc -l < "${BASE_MANIFEST}")"
if [[ "${MANIFEST_LINES}" -le 1 ]]; then
  echo "[err] base manifest still empty: ${BASE_MANIFEST}" >&2
  exit 1
fi
echo "[ok] base manifest: ${BASE_MANIFEST} rows=$((MANIFEST_LINES-1))"

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

TOTAL_A="$(manifest_rows "${MANIFEST_A}")"
TOTAL_B="$(manifest_rows "${MANIFEST_B}")"
TOTAL_C="$(manifest_rows "${MANIFEST_C}")"
echo "[ok] prompt manifests rows: A=${TOTAL_A} B=${TOTAL_B} C=${TOTAL_C}"

run_eval() {
  local manifest="$1"
  local out_csv="$2"

  rm -f "${out_csv}"

  local cmd=(
    python -u scripts/research/run_batch_eval.py
    --manifest "${manifest}"
    --output-csv "${out_csv}"
    --correct-rule "${CORRECT_RULE}"
    --device "${DEVICE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --log-every "${LOG_EVERY}"
    --no-resume
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
rm -f "${LOG_A}" "${LOG_B}" "${LOG_C}"
( run_eval "${MANIFEST_A}" "${CSV_A}" ) >"${LOG_A}" 2>&1 &
PID_A=$!
( run_eval "${MANIFEST_B}" "${CSV_B}" ) >"${LOG_B}" 2>&1 &
PID_B=$!

monitor_parallel_progress "${PID_A}" "${PID_B}" "${CSV_A}" "${CSV_B}" "${TOTAL_A}" "${TOTAL_B}"
wait "${PID_A}"
wait "${PID_B}"
echo "[ok] A/B finished"

echo "[stage] run C"
( run_eval "${MANIFEST_C}" "${CSV_C}" ) >"${LOG_C}" 2>&1 &
PID_C=$!
monitor_single_progress "${PID_C}" "${CSV_C}" "${TOTAL_C}"
wait "${PID_C}"
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
