#!/usr/bin/env bash
set -euo pipefail

# Simple and robust ABC eval pipeline:
# - explicit .env/.venv
# - auto-build base manifest if missing/empty
# - build A/B/C manifests
# - run A then B then C in foreground with live per-sample progress from run_batch_eval.py
# - remove old CSV before each run (fresh run)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

count_csv_rows() {
  local path="$1"
  python - <<'PY' "${path}"
import csv, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0)
    raise SystemExit(0)
with open(p, "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

BASE_MANIFEST="${1:-research/work/sample_manifest_okvqa_val1000.csv}"
BASE_MANIFEST="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${BASE_MANIFEST}")"

RUN_TAG="${RUN_TAG:-abc_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/eval_abc}"
WORK_DIR="research/work/abc_${RUN_TAG}"
EVAL_DIR="${OUT_ROOT}/eval/${RUN_TAG}"
LOG_DIR="${OUT_ROOT}/logs/${RUN_TAG}"
mkdir -p "${WORK_DIR}" "${EVAL_DIR}" "${LOG_DIR}" "$(dirname "${BASE_MANIFEST}")"

DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-}"
TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
LOG_EVERY="${LOG_EVERY:-1}"
CORRECT_RULE="${CORRECT_RULE:-vqa_0.3}"

OKVQA_ROOT="${OKVQA_ROOT:-$HOME/tca-reasoning/data/okvqa}"
OKVQA_LIMIT="${OKVQA_LIMIT:-1000}"   # 0 means full split
OKVQA_SEED="${OKVQA_SEED:-42}"
OKVQA_SPLIT="${OKVQA_SPLIT:-val}"
OKVQA_QUESTIONS="${OKVQA_QUESTIONS:-$OKVQA_ROOT/questions/OpenEnded_mscoco_val2014_questions.json}"
OKVQA_ANNOTATIONS="${OKVQA_ANNOTATIONS:-$OKVQA_ROOT/annotations/mscoco_val2014_annotations.json}"
OKVQA_IMAGE_ROOT="${OKVQA_IMAGE_ROOT:-$OKVQA_ROOT/images}"
OKVQA_ANN="${OKVQA_ANN:-$OKVQA_ANNOTATIONS}"

EXAMPLE_INSTRUCTION="${EXAMPLE_INSTRUCTION:-What color is the bus?}"
EXAMPLE_RESPONSE="${EXAMPLE_RESPONSE:-The answer is yellow.}"

PROMPT_A_TEMPLATE="{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>."
PROMPT_B_TEMPLATE="{question} Reply exactly in the format: The answer is <short answer>."
PROMPT_C_TEMPLATE=$'Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response.\n### Instruction:\n{instruction}\n### Reference Answer:\n{original_response}\n### Response:'

MANIFEST_A="${WORK_DIR}/manifest_promptA.csv"
MANIFEST_B="${WORK_DIR}/manifest_promptB.csv"
MANIFEST_C="${WORK_DIR}/manifest_promptC_oneshot.csv"

CSV_A="${EVAL_DIR}/promptA_eval.csv"
CSV_B="${EVAL_DIR}/promptB_eval.csv"
CSV_C="${EVAL_DIR}/promptC_eval.csv"

LOG_A="${LOG_DIR}/promptA_eval.log"
LOG_B="${LOG_DIR}/promptB_eval.log"
LOG_C="${LOG_DIR}/promptC_eval.log"

echo "[stage] ensure base manifest"
if [[ ! -f "${BASE_MANIFEST}" || "$(count_csv_rows "${BASE_MANIFEST}")" -le 0 ]]; then
  echo "[info] base manifest missing/empty -> build from OKVQA"
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
echo "[ok] base manifest rows=$(count_csv_rows "${BASE_MANIFEST}")"

echo "[stage] build A/B/C manifests"
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
echo "[ok] prompt rows: A=$(count_csv_rows "${MANIFEST_A}") B=$(count_csv_rows "${MANIFEST_B}") C=$(count_csv_rows "${MANIFEST_C}")"

run_one() {
  local name="$1"
  local manifest="$2"
  local out_csv="$3"
  local out_log="$4"
  local n_rows
  n_rows="$(count_csv_rows "${manifest}")"
  echo "[stage] run ${name} rows=${n_rows}"
  rm -f "${out_csv}" "${out_log}"
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
  if [[ -n "${MODEL}" ]]; then
    cmd+=(--model "${MODEL}")
  else
    cmd+=(--transcoder-set "${TRANSCODER_SET}")
  fi
  if [[ -f "${OKVQA_ANN}" ]]; then
    cmd+=(--annotations-json "${OKVQA_ANN}")
  fi
  "${cmd[@]}" 2>&1 | tee "${out_log}"
}

run_one "A" "${MANIFEST_A}" "${CSV_A}" "${LOG_A}"
run_one "B" "${MANIFEST_B}" "${CSV_B}" "${LOG_B}"
run_one "C" "${MANIFEST_C}" "${CSV_C}" "${LOG_C}"

echo "[stage] summary"
python - <<'PY' "${CSV_A}" "${CSV_B}" "${CSV_C}"
import csv, os, sys
for p in sys.argv[1:]:
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    n = len(rows)
    ok = sum(1 for r in rows if (r.get("correct") or "") == "1")
    err = sum(1 for r in rows if (r.get("error_message") or "").strip())
    print(f"{os.path.basename(p)} rows={n} correct={ok} acc={ok/max(n,1):.4f} errors={err}")
PY

echo "[done] manifests=${WORK_DIR}"
echo "[done] eval_csv=${EVAL_DIR}"
echo "[done] logs=${LOG_DIR}"
