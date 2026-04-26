#!/usr/bin/env bash
set -euo pipefail

# Stage 1 overnight run:
# - random sample from each A/B behavior bucket
# - answer-token-aligned attribution with 1024 feature nodes
# - compare after all attribution jobs finish
# - generate consolidated Stage 1 analysis tables
#
# Recommended safe run:
#   TMPDIR=/root/autodl-tmp/tmp \
#   bash scripts/server/run_stage1_random_answer_aligned_overnight.sh
#
# More aggressive run:
#   STAGE1_PER_BUCKET=30 TMPDIR=/root/autodl-tmp/tmp \
#   bash scripts/server/run_stage1_random_answer_aligned_overnight.sh

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

STAGE1_RUN_TAG_BASE="${STAGE1_RUN_TAG_BASE:-stage1_random20_1024_$(date +%Y%m%d_%H%M%S)}"
STAGE1_BUCKETS="${STAGE1_BUCKETS:-A0_B1,A1_B0,A1_B1,A0_B0}"
STAGE1_PER_BUCKET="${STAGE1_PER_BUCKET:-20}"
STAGE1_MAX_FEATURE_NODES="${STAGE1_MAX_FEATURE_NODES:-1024}"
STAGE1_SELECTION_SEED="${STAGE1_SELECTION_SEED:-42501}"
STAGE1_ANSWER_SOURCE="${STAGE1_ANSWER_SOURCE:-predicted}"
STAGE1_OUTPUTS_ROOT="${STAGE1_OUTPUTS_ROOT:-outputs/phase_ab/ab_answer_aligned}"
STAGE1_WORK_ROOT="${STAGE1_WORK_ROOT:-research/work/ab_answer_aligned}"
STAGE1_TMPDIR="${STAGE1_TMPDIR:-${TMPDIR:-/root/autodl-tmp/tmp}}"
STAGE1_OFFLOAD="${STAGE1_OFFLOAD:-disk}"
STAGE1_TOPK_PER_NODE="${STAGE1_TOPK_PER_NODE:-3}"
STAGE1_BEAM_PER_DEPTH="${STAGE1_BEAM_PER_DEPTH:-96}"
STAGE1_COVERAGE="${STAGE1_COVERAGE:-0.98}"
STAGE1_MAX_DEPTH="${STAGE1_MAX_DEPTH:-40}"
STAGE1_MIN_ABS_WEIGHT="${STAGE1_MIN_ABS_WEIGHT:-0.0}"
STAGE1_LOG_EVERY="${STAGE1_LOG_EVERY:-10}"
STAGE1_TOP_N="${STAGE1_TOP_N:-20}"
STAGE1_DISK_BUDGET_GB="${STAGE1_DISK_BUDGET_GB:-260}"
STAGE1_DISK_RESERVE_GB="${STAGE1_DISK_RESERVE_GB:-30}"
STAGE1_ALLOW_OVER_BUDGET="${STAGE1_ALLOW_OVER_BUDGET:-0}"
STAGE1_CLEAN_PT_AFTER_SUMMARY="${STAGE1_CLEAN_PT_AFTER_SUMMARY:-0}"

# Empirical estimate from the 2026-04-25 run:
# 4 buckets x 20 samples x 1024 feature nodes produced about 82.7 GB.
# Dense graph storage grows closer to quadratic than linear as the feature-node
# cap increases, so use a conservative square scaling for larger node budgets.
STAGE1_EST_PAIR_GB_AT_1024="${STAGE1_EST_PAIR_GB_AT_1024:-1.1}"

mkdir -p "${STAGE1_TMPDIR}" "${STAGE1_OUTPUTS_ROOT}"
export TMPDIR="${STAGE1_TMPDIR}"

QUEUE_LOG_DIR="${STAGE1_OUTPUTS_ROOT}/${STAGE1_RUN_TAG_BASE}_queue_logs"
STATUS_TSV="${QUEUE_LOG_DIR}/bucket_status.tsv"
SUMMARY_DIR="${STAGE1_OUTPUTS_ROOT}/${STAGE1_RUN_TAG_BASE}_stage1_summary"

echo "[stage1] run_tag_base=${STAGE1_RUN_TAG_BASE}"
echo "[stage1] buckets=${STAGE1_BUCKETS}"
echo "[stage1] per_bucket=${STAGE1_PER_BUCKET}"
echo "[stage1] max_feature_nodes=${STAGE1_MAX_FEATURE_NODES}"
echo "[stage1] selection_seed=${STAGE1_SELECTION_SEED}"
echo "[stage1] answer_source=${STAGE1_ANSWER_SOURCE}"
echo "[stage1] offload=${STAGE1_OFFLOAD}"
echo "[stage1] tmpdir=${TMPDIR}"
echo "[stage1] disk_budget_gb=${STAGE1_DISK_BUDGET_GB}"
echo "[stage1] disk_reserve_gb=${STAGE1_DISK_RESERVE_GB}"

STAGE1_BUCKET_COUNT="$("${PYTHON_BIN}" - <<'PY' "${STAGE1_BUCKETS}"
import sys
print(len([x.strip() for x in sys.argv[1].split(",") if x.strip()]))
PY
)"
STAGE1_ESTIMATED_GB="$("${PYTHON_BIN}" - <<'PY' "${STAGE1_BUCKET_COUNT}" "${STAGE1_PER_BUCKET}" "${STAGE1_MAX_FEATURE_NODES}" "${STAGE1_EST_PAIR_GB_AT_1024}"
import sys
buckets = int(sys.argv[1])
per_bucket = int(sys.argv[2])
nodes = float(sys.argv[3])
base = float(sys.argv[4])
estimate = buckets * per_bucket * base * (nodes / 1024.0) ** 2
print(f"{estimate:.1f}")
PY
)"
STAGE1_AVAILABLE_GB="$(df -Pk "${STAGE1_OUTPUTS_ROOT}" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')"
echo "[stage1] estimated_pt_size_gb=${STAGE1_ESTIMATED_GB}"
echo "[stage1] available_output_disk_gb=${STAGE1_AVAILABLE_GB}"

"${PYTHON_BIN}" - <<'PY' \
  "${STAGE1_ESTIMATED_GB}" \
  "${STAGE1_DISK_BUDGET_GB}" \
  "${STAGE1_AVAILABLE_GB}" \
  "${STAGE1_DISK_RESERVE_GB}" \
  "${STAGE1_ALLOW_OVER_BUDGET}"
import sys

estimated = float(sys.argv[1])
budget = float(sys.argv[2])
available = float(sys.argv[3])
reserve = float(sys.argv[4])
allow = sys.argv[5] == "1"

messages = []
if estimated > budget:
    messages.append(
        f"estimated graph size {estimated:.1f}GB exceeds STAGE1_DISK_BUDGET_GB={budget:.1f}GB"
    )
if available > 0 and estimated > max(0.0, available - reserve):
    messages.append(
        f"estimated graph size {estimated:.1f}GB leaves less than {reserve:.1f}GB free "
        f"on output disk (available={available:.1f}GB)"
    )

if messages and not allow:
    for msg in messages:
        print(f"[stage1:disk-guard] {msg}", file=sys.stderr)
    print(
        "[stage1:disk-guard] reduce STAGE1_PER_BUCKET or STAGE1_MAX_FEATURE_NODES, "
        "or set STAGE1_ALLOW_OVER_BUDGET=1 to override.",
        file=sys.stderr,
    )
    raise SystemExit(3)

if messages:
    for msg in messages:
        print(f"[stage1:disk-warning] {msg}", file=sys.stderr)
PY

QUEUE_RUN_TAG_BASE="${STAGE1_RUN_TAG_BASE}" \
QUEUE_BUCKETS="${STAGE1_BUCKETS}" \
QUEUE_PER_BUCKET="${STAGE1_PER_BUCKET}" \
QUEUE_MAX_SELECTED_ROWS="${STAGE1_PER_BUCKET}" \
QUEUE_MAX_FEATURE_NODES="${STAGE1_MAX_FEATURE_NODES}" \
QUEUE_RETRY_FEATURE_NODES="" \
QUEUE_SKIP_COMPARE="1" \
QUEUE_CONTINUE_ON_ERROR="1" \
QUEUE_OFFLOAD="${STAGE1_OFFLOAD}" \
QUEUE_TMPDIR="${TMPDIR}" \
QUEUE_SELECTION_MODE="random" \
QUEUE_SELECTION_SEED="${STAGE1_SELECTION_SEED}" \
QUEUE_ANSWER_SOURCE="${STAGE1_ANSWER_SOURCE}" \
QUEUE_LOG_DIR="${QUEUE_LOG_DIR}" \
bash scripts/server/run_ab_answer_aligned_trace_bucket_queue.sh

if [[ ! -f "${STATUS_TSV}" ]]; then
  echo "[stage1:error] missing status file: ${STATUS_TSV}" >&2
  exit 2
fi

echo "[stage1] attribution queue finished; starting compare for successful buckets"
tail -n +2 "${STATUS_TSV}" | while IFS=$'\t' read -r bucket run_tag status exit_code log_path; do
  if [[ -z "${bucket:-}" ]]; then
    continue
  fi
  if [[ "${status}" != "ok" ]]; then
    echo "[stage1:skip] bucket=${bucket} run_tag=${run_tag} status=${status} exit_code=${exit_code}"
    continue
  fi

  out_root="${STAGE1_OUTPUTS_ROOT}/${run_tag}"
  selected_csv="${STAGE1_WORK_ROOT}/${run_tag}/selected_4bucket.csv"
  pt_dir_a="${out_root}/pt_a"
  pt_dir_b="${out_root}/pt_b"
  compare_out="${out_root}/${run_tag}"
  compare_log="${QUEUE_LOG_DIR}/${run_tag}.compare.log"

  echo "[stage1:compare] bucket=${bucket} run_tag=${run_tag}"
  "${PYTHON_BIN}" scripts/research/trace_compare_ab_controlled.py \
    --pt-dir-a "${pt_dir_a}" \
    --pt-dir-b "${pt_dir_b}" \
    --bucket-csv "${selected_csv}" \
    --out-dir "${compare_out}" \
    --target-logit-rank 0 \
    --topk-per-node "${STAGE1_TOPK_PER_NODE}" \
    --beam-per-depth "${STAGE1_BEAM_PER_DEPTH}" \
    --coverage "${STAGE1_COVERAGE}" \
    --max-depth "${STAGE1_MAX_DEPTH}" \
    --min-abs-weight "${STAGE1_MIN_ABS_WEIGHT}" \
    --log-every "${STAGE1_LOG_EVERY}" \
    2>&1 | tee "${compare_log}"
done

echo "[stage1] validating queue outputs"
"${PYTHON_BIN}" scripts/research/validate_answer_aligned_queue.py \
  --run-tag-base "${STAGE1_RUN_TAG_BASE}" \
  --buckets "${STAGE1_BUCKETS}" \
  --outputs-root "${STAGE1_OUTPUTS_ROOT}" \
  --work-root "${STAGE1_WORK_ROOT}" \
  2>&1 | tee "${QUEUE_LOG_DIR}/${STAGE1_RUN_TAG_BASE}.validate.log"

echo "[stage1] compact queue summary"
"${PYTHON_BIN}" scripts/research/show_answer_aligned_queue_results.py \
  --run-tag-base "${STAGE1_RUN_TAG_BASE}" \
  --buckets "${STAGE1_BUCKETS}" \
  --outputs-root "${STAGE1_OUTPUTS_ROOT}" \
  --work-root "${STAGE1_WORK_ROOT}" \
  2>&1 | tee "${QUEUE_LOG_DIR}/${STAGE1_RUN_TAG_BASE}.show.log"

echo "[stage1] writing consolidated analysis tables"
"${PYTHON_BIN}" scripts/research/summarize_answer_aligned_compare.py \
  --compare-root "${STAGE1_OUTPUTS_ROOT}" \
  --path-contains "${STAGE1_RUN_TAG_BASE}" \
  --out-dir "${SUMMARY_DIR}" \
  --top-n "${STAGE1_TOP_N}" \
  2>&1 | tee "${QUEUE_LOG_DIR}/${STAGE1_RUN_TAG_BASE}.summarize.log"

if [[ "${STAGE1_CLEAN_PT_AFTER_SUMMARY}" == "1" ]]; then
  echo "[stage1:cleanup] removing pt_a/pt_b dirs after summary"
  tail -n +2 "${STATUS_TSV}" | while IFS=$'\t' read -r bucket run_tag status exit_code log_path; do
    if [[ "${status}" != "ok" ]]; then
      continue
    fi
    rm -rf "${STAGE1_OUTPUTS_ROOT}/${run_tag}/pt_a" "${STAGE1_OUTPUTS_ROOT}/${run_tag}/pt_b"
  done
  echo "[stage1:cleanup] done"
fi

echo "[stage1:done] run_tag_base=${STAGE1_RUN_TAG_BASE}"
echo "[stage1:done] status_tsv=${STATUS_TSV}"
echo "[stage1:done] summary_dir=${SUMMARY_DIR}"
