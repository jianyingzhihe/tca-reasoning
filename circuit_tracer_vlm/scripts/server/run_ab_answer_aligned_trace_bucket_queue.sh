#!/usr/bin/env bash
set -uo pipefail

# Queue overnight answer-token-aligned tracing jobs bucket-by-bucket.
# Default behavior:
# - 4 buckets run sequentially
# - 20 samples per bucket
# - 1024 feature nodes per graph
# - skip compare to maximize overnight throughput
#
# Example:
#   TMPDIR=/root/autodl-tmp/tmp \
#   bash scripts/server/run_ab_answer_aligned_trace_bucket_queue.sh

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

QUEUE_RUN_TAG_BASE="${QUEUE_RUN_TAG_BASE:-ab_queue20_1024_$(date +%Y%m%d_%H%M%S)}"
QUEUE_BUCKETS="${QUEUE_BUCKETS:-A0_B1,A1_B1,A0_B0,A1_B0}"
QUEUE_PER_BUCKET="${QUEUE_PER_BUCKET:-20}"
QUEUE_MAX_SELECTED_ROWS="${QUEUE_MAX_SELECTED_ROWS:-20}"
QUEUE_MAX_FEATURE_NODES="${QUEUE_MAX_FEATURE_NODES:-1024}"
QUEUE_RETRY_FEATURE_NODES="${QUEUE_RETRY_FEATURE_NODES:-}"
QUEUE_SKIP_COMPARE="${QUEUE_SKIP_COMPARE:-1}"
QUEUE_CONTINUE_ON_ERROR="${QUEUE_CONTINUE_ON_ERROR:-1}"
QUEUE_OFFLOAD="${QUEUE_OFFLOAD:-disk}"
QUEUE_TMPDIR="${QUEUE_TMPDIR:-${TMPDIR:-/root/autodl-tmp/tmp}}"
QUEUE_SELECTION_MODE="${QUEUE_SELECTION_MODE:-first}"
QUEUE_SELECTION_SEED="${QUEUE_SELECTION_SEED:-42501}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${ROOT_DIR}/outputs/phase_ab/ab_answer_aligned/${QUEUE_RUN_TAG_BASE}_queue_logs}"

mkdir -p "${QUEUE_TMPDIR}" "${QUEUE_LOG_DIR}"
export TMPDIR="${QUEUE_TMPDIR}"

IFS=',' read -r -a BUCKET_ARRAY <<< "${QUEUE_BUCKETS}"
STATUS_TSV="${QUEUE_LOG_DIR}/bucket_status.tsv"
printf "bucket\trun_tag\tstatus\texit_code\tlog_path\n" > "${STATUS_TSV}"

echo "[queue] run_tag_base=${QUEUE_RUN_TAG_BASE}"
echo "[queue] buckets=${QUEUE_BUCKETS}"
echo "[queue] per_bucket=${QUEUE_PER_BUCKET}"
echo "[queue] max_selected_rows=${QUEUE_MAX_SELECTED_ROWS}"
echo "[queue] max_feature_nodes=${QUEUE_MAX_FEATURE_NODES}"
echo "[queue] retry_feature_nodes=${QUEUE_RETRY_FEATURE_NODES:-<empty>}"
echo "[queue] skip_compare=${QUEUE_SKIP_COMPARE}"
echo "[queue] offload=${QUEUE_OFFLOAD}"
echo "[queue] tmpdir=${TMPDIR}"
echo "[queue] selection_mode=${QUEUE_SELECTION_MODE}"
echo "[queue] selection_seed=${QUEUE_SELECTION_SEED}"
echo "[queue] log_dir=${QUEUE_LOG_DIR}"

for bucket in "${BUCKET_ARRAY[@]}"; do
  bucket="$(echo "${bucket}" | xargs)"
  if [[ -z "${bucket}" ]]; then
    continue
  fi

  run_tag="${QUEUE_RUN_TAG_BASE}_${bucket}"
  log_path="${QUEUE_LOG_DIR}/${run_tag}.log"
  echo "[queue] starting bucket=${bucket} run_tag=${run_tag}"

  (
    set -euo pipefail
    RUN_TAG="${run_tag}" \
    BUCKETS="${bucket}" \
    PER_BUCKET="${QUEUE_PER_BUCKET}" \
    MAX_SELECTED_ROWS="${QUEUE_MAX_SELECTED_ROWS}" \
    MAX_FEATURE_NODES="${QUEUE_MAX_FEATURE_NODES}" \
    RETRY_FEATURE_NODES="${QUEUE_RETRY_FEATURE_NODES}" \
    SELECTION_MODE="${QUEUE_SELECTION_MODE}" \
    SELECTION_SEED="${QUEUE_SELECTION_SEED}" \
    ANSWER_ATTR_EXEC_MODE="${ANSWER_ATTR_EXEC_MODE:-subprocess}" \
    ANSWER_ATTR_VERBOSE_ATTRIBUTION="${ANSWER_ATTR_VERBOSE_ATTRIBUTION:-1}" \
    OFFLOAD="${QUEUE_OFFLOAD}" \
    TMPDIR="${TMPDIR}" \
    SKIP_COMPARE="${QUEUE_SKIP_COMPARE}" \
    bash scripts/server/run_ab_answer_aligned_trace_full.sh
  ) 2>&1 | tee "${log_path}"

  rc=${PIPESTATUS[0]}
  if [[ ${rc} -eq 0 ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "${bucket}" "${run_tag}" "ok" "${rc}" "${log_path}" >> "${STATUS_TSV}"
    echo "[queue] finished bucket=${bucket} status=ok"
  else
    printf "%s\t%s\t%s\t%s\t%s\n" "${bucket}" "${run_tag}" "failed" "${rc}" "${log_path}" >> "${STATUS_TSV}"
    echo "[queue] finished bucket=${bucket} status=failed exit_code=${rc}"
    if [[ "${QUEUE_CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "[queue] stopping because QUEUE_CONTINUE_ON_ERROR=${QUEUE_CONTINUE_ON_ERROR}"
      exit "${rc}"
    fi
  fi
done

echo "[queue] all buckets finished"
echo "[queue] status_tsv=${STATUS_TSV}"
