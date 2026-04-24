#!/usr/bin/env bash
set -euo pipefail

# Full pipeline for answer-token-aligned A/B circuit comparison.
# Inputs:
#   - Prompt A eval csv
#   - Prompt B eval csv
#   - Bucket csv with at least: sample_id,bucket,image_path,a_input,b_input
#
# Example:
# RUN_TAG=ab_answer_aligned_4x60 \
# EVAL_CSV_A=research/work/promptA_eval_final.csv \
# EVAL_CSV_B=research/work/promptB_eval.csv \
# BUCKET_SOURCE_CSV=research/work/ab_buckets_by_hit_from_latest.csv \
# PER_BUCKET=60 \
# CLEAN_PT_AFTER=1 \
# bash scripts/server/run_ab_answer_aligned_trace_full.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.venv/bin/activate"
fi
if [[ -f "${ROOT_DIR}/scripts/server/load_env.sh" && -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/scripts/server/load_env.sh" "${ROOT_DIR}/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
export TCA_PYTHON="${PYTHON_BIN}"

RUN_TAG="${RUN_TAG:-ab_answer_aligned_4x60_$(date +%Y%m%d_%H%M%S)}"
EVAL_CSV_A="${EVAL_CSV_A:-research/work/promptA_eval_final.csv}"
EVAL_CSV_B="${EVAL_CSV_B:-research/work/promptB_eval.csv}"
BUCKET_SOURCE_CSV="${BUCKET_SOURCE_CSV:-research/work/ab_buckets_by_hit_from_latest.csv}"
WORK_DIR="${WORK_DIR:-research/work/ab_answer_aligned/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/ab_answer_aligned/${RUN_TAG}}"
BUCKETS="${BUCKETS:-A0_B1,A1_B1,A0_B0,A1_B0}"
PER_BUCKET="${PER_BUCKET:-60}"

TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
DTYPE="${DTYPE:-bfloat16}"
MAX_FEATURE_NODES="${MAX_FEATURE_NODES:-112}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFLOAD="${OFFLOAD:-cpu}"
TOPK="${TOPK:-16}"
ANSWER_SOURCE="${ANSWER_SOURCE:-predicted}"
ANSWER_ATTR_EXEC_MODE="${ANSWER_ATTR_EXEC_MODE:-subprocess}"
ANSWER_ATTR_VERBOSE_ATTRIBUTION="${ANSWER_ATTR_VERBOSE_ATTRIBUTION:-1}"
STOP_ON_ATTR_ERROR="${STOP_ON_ATTR_ERROR:-0}"

TOPK_PER_NODE="${TOPK_PER_NODE:-3}"
BEAM_PER_DEPTH="${BEAM_PER_DEPTH:-96}"
COVERAGE="${COVERAGE:-0.98}"
MAX_DEPTH="${MAX_DEPTH:-40}"
MIN_ABS_WEIGHT="${MIN_ABS_WEIGHT:-0.0}"
LOG_EVERY="${LOG_EVERY:-10}"
CLEAN_PT_AFTER="${CLEAN_PT_AFTER:-0}"

mkdir -p "${WORK_DIR}" "${OUT_ROOT}"

SELECTED_CSV="${WORK_DIR}/selected_4bucket.csv"
PT_DIR_A="${OUT_ROOT}/pt_a"
PT_DIR_B="${OUT_ROOT}/pt_b"
META_A="${OUT_ROOT}/answer_aligned_meta_a.csv"
META_B="${OUT_ROOT}/answer_aligned_meta_b.csv"
ATTEMPT_LOG_DIR_A="${OUT_ROOT}/logs_a"
ATTEMPT_LOG_DIR_B="${OUT_ROOT}/logs_b"
ATTR_EXTRA_ARGS=()
if [[ "${ANSWER_ATTR_VERBOSE_ATTRIBUTION}" == "1" ]]; then
  ATTR_EXTRA_ARGS+=(--verbose-attribution)
fi
if [[ "${STOP_ON_ATTR_ERROR}" == "1" ]]; then
  ATTR_EXTRA_ARGS+=(--stop-on-error)
fi

echo "[stage] sample ${PER_BUCKET} per bucket from ${BUCKET_SOURCE_CSV}"
"${PYTHON_BIN}" - <<'PY' "${BUCKET_SOURCE_CSV}" "${SELECTED_CSV}" "${BUCKETS}" "${PER_BUCKET}"
import csv, sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
buckets = [x.strip() for x in sys.argv[3].split(",") if x.strip()]
per_bucket = int(sys.argv[4])

rows = list(csv.DictReader(open(src, "r", encoding="utf-8", newline="")))
if not rows:
    raise ValueError(f"empty source: {src}")

need_cols = ["sample_id", "bucket", "image_path", "a_input", "b_input"]
for c in need_cols:
    if c not in rows[0]:
        raise ValueError(f"missing column `{c}` in {src}")

by_bucket = defaultdict(list)
for r in rows:
    b = (r.get("bucket") or "").strip()
    sid = (r.get("sample_id") or "").strip()
    if b and sid:
        by_bucket[b].append(r)

selected = []
for b in buckets:
    part = sorted(by_bucket.get(b, []), key=lambda x: x["sample_id"])
    selected.extend(part[:max(0, per_bucket)])

if not selected:
    raise ValueError("no selected rows")

dst.parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
    w.writeheader()
    w.writerows(selected)
print(f"[ok] selected rows={len(selected)} -> {dst}")
PY

echo "[stage] answer-aligned attribution A"
"${PYTHON_BIN}" scripts/research/run_batch_answer_aligned_attribute.py \
  --eval-csv "${EVAL_CSV_A}" \
  --selected-csv "${SELECTED_CSV}" \
  --output-dir "${PT_DIR_A}" \
  --metadata-csv "${META_A}" \
  --transcoder-set "${TRANSCODER_SET}" \
  --dtype "${DTYPE}" \
  --max-feature-nodes "${MAX_FEATURE_NODES}" \
  --batch-size "${BATCH_SIZE}" \
  --offload "${OFFLOAD}" \
  --topk "${TOPK}" \
  --answer-source "${ANSWER_SOURCE}" \
  --exec-mode "${ANSWER_ATTR_EXEC_MODE}" \
  --attempt-log-dir "${ATTEMPT_LOG_DIR_A}" \
  "${ATTR_EXTRA_ARGS[@]}"

echo "[stage] answer-aligned attribution B"
"${PYTHON_BIN}" scripts/research/run_batch_answer_aligned_attribute.py \
  --eval-csv "${EVAL_CSV_B}" \
  --selected-csv "${SELECTED_CSV}" \
  --output-dir "${PT_DIR_B}" \
  --metadata-csv "${META_B}" \
  --transcoder-set "${TRANSCODER_SET}" \
  --dtype "${DTYPE}" \
  --max-feature-nodes "${MAX_FEATURE_NODES}" \
  --batch-size "${BATCH_SIZE}" \
  --offload "${OFFLOAD}" \
  --topk "${TOPK}" \
  --answer-source "${ANSWER_SOURCE}" \
  --exec-mode "${ANSWER_ATTR_EXEC_MODE}" \
  --attempt-log-dir "${ATTEMPT_LOG_DIR_B}" \
  "${ATTR_EXTRA_ARGS[@]}"

echo "[stage] controlled trace compare"
"${PYTHON_BIN}" scripts/research/trace_compare_ab_controlled.py \
  --pt-dir-a "${PT_DIR_A}" \
  --pt-dir-b "${PT_DIR_B}" \
  --bucket-csv "${SELECTED_CSV}" \
  --out-dir "${OUT_ROOT}/${RUN_TAG}" \
  --target-logit-rank 0 \
  --topk-per-node "${TOPK_PER_NODE}" \
  --beam-per-depth "${BEAM_PER_DEPTH}" \
  --coverage "${COVERAGE}" \
  --max-depth "${MAX_DEPTH}" \
  --min-abs-weight "${MIN_ABS_WEIGHT}" \
  --log-every "${LOG_EVERY}"

echo "[done] run_tag=${RUN_TAG}"
echo "[done] selected=${SELECTED_CSV}"
echo "[done] meta_a=${META_A}"
echo "[done] meta_b=${META_B}"
echo "[done] pt_a=${PT_DIR_A}"
echo "[done] pt_b=${PT_DIR_B}"
echo "[done] compare_out=${OUT_ROOT}/${RUN_TAG}"

if [[ "${CLEAN_PT_AFTER}" == "1" ]]; then
  echo "[cleanup] removing pt dirs to save disk..."
  rm -rf "${PT_DIR_A}" "${PT_DIR_B}"
  echo "[cleanup] done"
fi
