#!/usr/bin/env bash
set -euo pipefail

# Run a lightweight intervention smoke test over one completed Stage 1 run.
#
# This script does not try to prove causality yet. It simply picks top
# answer-aligned feature nodes from the compare output and zeroes them one by
# one, measuring target-logit / target-probability changes.
#
# Example:
#   RUN_TAG_BASE=stage1_random20_1024_seed42501_20260425_235607 \
#   GENERIC_NODES_CSV=outputs/phase_ab/ab_answer_aligned/stage1_random20_1024_seed42501_20260425_235607_stage1_summary_v2/stage1_generic_nodes.csv \
#   bash scripts/server/run_stage1_intervention_smoke.sh

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

RUN_TAG_BASE="${RUN_TAG_BASE:?set RUN_TAG_BASE to a completed Stage 1 run tag base}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/phase_ab/ab_answer_aligned}"
RUNS_GLOB="${RUNS_GLOB:-A0_B1,A1_B0,A1_B1,A0_B0}"
GENERIC_NODES_CSV="${GENERIC_NODES_CSV:-}"
TOP_FEATURES_PER_SAMPLE="${TOP_FEATURES_PER_SAMPLE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
SAMPLE_RANK_METRIC="${SAMPLE_RANK_METRIC:-edge_overlap_jaccard}"
SAMPLE_RANK_DESC="${SAMPLE_RANK_DESC:-0}"
TRANS_SET="${TRANS_SET:-tianhux2/gemma3-4b-it-plt}"
SAMPLE_IDS_DIR="${SAMPLE_IDS_DIR:-}"
REQUIRE_SAME_TARGET="${REQUIRE_SAME_TARGET:-0}"

IFS=',' read -r -a BUCKET_ARRAY <<< "${RUNS_GLOB}"
produced_csvs=()
for bucket in "${BUCKET_ARRAY[@]}"; do
  bucket="$(echo "${bucket}" | xargs)"
  if [[ -z "${bucket}" ]]; then
    continue
  fi

  run_tag="${RUN_TAG_BASE}_${bucket}"
  run_root="${OUTPUTS_ROOT}/${run_tag}"
  compare_dir="${run_root}/${run_tag}"
  out_csv="${run_root}/intervention_smoke_${bucket}.csv"

  if [[ ! -d "${compare_dir}" ]]; then
    echo "[skip] missing compare_dir=${compare_dir}"
    continue
  fi

  extra_args=()
  if [[ -n "${GENERIC_NODES_CSV}" ]]; then
    extra_args+=(--generic-nodes-csv "${GENERIC_NODES_CSV}")
  fi
  if [[ -n "${SAMPLE_IDS_DIR}" && -f "${SAMPLE_IDS_DIR}/${bucket}.csv" ]]; then
    extra_args+=(--sample-ids-csv "${SAMPLE_IDS_DIR}/${bucket}.csv")
  fi
  if [[ "${SAMPLE_RANK_DESC}" == "1" ]]; then
    extra_args+=(--sample-rank-desc)
  fi
  if [[ "${REQUIRE_SAME_TARGET}" == "1" ]]; then
    extra_args+=(--require-same-target)
  fi

  echo "[run] bucket=${bucket} run_root=${run_root}"
  "${PYTHON_BIN}" scripts/research/run_answer_aligned_intervention_smoke.py \
    --run-root "${run_root}" \
    --compare-dir "${compare_dir}" \
    --bucket "${bucket}" \
    --run both \
    --transcoder-set "${TRANS_SET}" \
    --sample-rank-metric "${SAMPLE_RANK_METRIC}" \
    --max-samples "${MAX_SAMPLES}" \
    --top-features-per-sample "${TOP_FEATURES_PER_SAMPLE}" \
    --out-csv "${out_csv}" \
    "${extra_args[@]}"
  produced_csvs+=("${out_csv}")
done

if [[ "${#produced_csvs[@]}" -gt 0 ]]; then
  summary_dir="${OUTPUTS_ROOT}/${RUN_TAG_BASE}_intervention_smoke_summary"
  echo "[summary] out_dir=${summary_dir}"
  "${PYTHON_BIN}" scripts/research/summarize_intervention_smoke.py \
    --inputs "${produced_csvs[@]}" \
    --out-dir "${summary_dir}"
fi
