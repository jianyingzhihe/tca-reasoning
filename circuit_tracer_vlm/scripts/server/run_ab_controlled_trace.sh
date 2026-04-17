#!/usr/bin/env bash
set -euo pipefail

# Run controlled A/B circuit backtrace comparison:
# - Sample 4 buckets (default 60 each)
# - Controlled reverse trace: topk-per-node + layer beam + coverage
# - Export detailed nodes/edges and sample/bucket metrics
#
# Usage:
#   bash scripts/server/run_ab_controlled_trace.sh
#
# Optional env vars:
#   RUN_TAG=ab_trace_240
#   PT_DIR_A=outputs/phase_ab/ab_circuit_240/pt_a
#   PT_DIR_B=outputs/phase_ab/ab_circuit_240/pt_b
#   BUCKET_CSV=research/work/ab_circuit_shortlist_240.csv
#   OUT_ROOT=outputs/phase_ab/ab_trace_controlled
#   BUCKETS=A0_B1,A1_B1,A0_B0,A1_B0
#   PER_BUCKET=60
#   TARGET_LOGIT_RANK=0
#   TOPK_PER_NODE=3
#   BEAM_PER_DEPTH=64
#   COVERAGE=0.95
#   MAX_DEPTH=40
#   MIN_ABS_WEIGHT=0.0
#   LOG_EVERY=10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

RUN_TAG="${RUN_TAG:-ab_trace_controlled_$(date +%Y%m%d_%H%M%S)}"
PT_DIR_A="${PT_DIR_A:-outputs/phase_ab/ab_circuit_240/pt_a}"
PT_DIR_B="${PT_DIR_B:-outputs/phase_ab/ab_circuit_240/pt_b}"
BUCKET_CSV="${BUCKET_CSV:-research/work/ab_circuit_shortlist_240.csv}"
OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/ab_trace_controlled}"

BUCKETS="${BUCKETS:-A0_B1,A1_B1,A0_B0,A1_B0}"
PER_BUCKET="${PER_BUCKET:-60}"
TARGET_LOGIT_RANK="${TARGET_LOGIT_RANK:-0}"
TOPK_PER_NODE="${TOPK_PER_NODE:-3}"
BEAM_PER_DEPTH="${BEAM_PER_DEPTH:-64}"
COVERAGE="${COVERAGE:-0.95}"
MAX_DEPTH="${MAX_DEPTH:-40}"
MIN_ABS_WEIGHT="${MIN_ABS_WEIGHT:-0.0}"
LOG_EVERY="${LOG_EVERY:-10}"

OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

if [[ ! -d "${PT_DIR_A}" ]]; then
  echo "[err] PT_DIR_A not found: ${PT_DIR_A}" >&2
  exit 1
fi
if [[ ! -d "${PT_DIR_B}" ]]; then
  echo "[err] PT_DIR_B not found: ${PT_DIR_B}" >&2
  exit 1
fi
if [[ ! -f "${BUCKET_CSV}" ]]; then
  echo "[err] BUCKET_CSV not found: ${BUCKET_CSV}" >&2
  exit 1
fi

echo "[run] controlled trace compare"
echo "[cfg] PT_DIR_A=${PT_DIR_A}"
echo "[cfg] PT_DIR_B=${PT_DIR_B}"
echo "[cfg] BUCKET_CSV=${BUCKET_CSV}"
echo "[cfg] OUT_DIR=${OUT_DIR}"
echo "[cfg] BUCKETS=${BUCKETS}"
echo "[cfg] PER_BUCKET=${PER_BUCKET}"
echo "[cfg] TOPK_PER_NODE=${TOPK_PER_NODE} BEAM_PER_DEPTH=${BEAM_PER_DEPTH} COVERAGE=${COVERAGE}"
echo "[cfg] MAX_DEPTH=${MAX_DEPTH} MIN_ABS_WEIGHT=${MIN_ABS_WEIGHT}"

python scripts/research/trace_compare_ab_controlled.py \
  --pt-dir-a "${PT_DIR_A}" \
  --pt-dir-b "${PT_DIR_B}" \
  --bucket-csv "${BUCKET_CSV}" \
  --out-dir "${OUT_DIR}" \
  --buckets "${BUCKETS}" \
  --per-bucket "${PER_BUCKET}" \
  --target-logit-rank "${TARGET_LOGIT_RANK}" \
  --topk-per-node "${TOPK_PER_NODE}" \
  --beam-per-depth "${BEAM_PER_DEPTH}" \
  --coverage "${COVERAGE}" \
  --max-depth "${MAX_DEPTH}" \
  --min-abs-weight "${MIN_ABS_WEIGHT}" \
  --log-every "${LOG_EVERY}"

echo "[done] outputs:"
echo "  ${OUT_DIR}/selected_samples.csv"
echo "  ${OUT_DIR}/sample_compare_controlled.csv"
echo "  ${OUT_DIR}/bucket_summary_controlled.csv"
echo "  ${OUT_DIR}/nodes_detailed_controlled.csv"
echo "  ${OUT_DIR}/edges_detailed_controlled.csv"
