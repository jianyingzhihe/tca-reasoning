#!/usr/bin/env bash
set -euo pipefail

# Full pipeline:
# 1) sample N per bucket from AB bucket csv
# 2) build prompt-A / prompt-B manifests
# 3) run batch attribution for A and B
# 4) run controlled trace compare export
#
# Usage:
#   bash scripts/server/run_ab_controlled_trace_full.sh
#
# Required input CSV schema (BUCKET_SOURCE_CSV):
#   sample_id,bucket,image_path,a_input,b_input
#
# Optional env vars:
#   RUN_TAG=ab_full_4x60
#   BUCKET_SOURCE_CSV=research/work/ab_buckets_by_hit_from_latest.csv
#   WORK_DIR=research/work/ab_controlled_trace
#   OUT_ROOT=outputs/phase_ab/ab_controlled_trace
#   BUCKETS=A0_B1,A1_B1,A0_B0,A1_B0
#   PER_BUCKET=60
#   TRANSCODER_SET=tianhux2/gemma3-4b-it-plt
#   DTYPE=bfloat16
#   MAX_FEATURE_NODES=96
#   MAX_N_LOGITS=1
#   BATCH_SIZE=1
#   OFFLOAD=cpu
#   TOPK=16
#   TOPK_PER_NODE=3
#   BEAM_PER_DEPTH=64
#   COVERAGE=0.95
#   MAX_DEPTH=40
#   MIN_ABS_WEIGHT=0.0
#   LOG_EVERY=10
#   CLEAN_PT_AFTER=0  # 1 to delete pt_a/pt_b after trace export

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

RUN_TAG="${RUN_TAG:-ab_full_4x60_$(date +%Y%m%d_%H%M%S)}"
BUCKET_SOURCE_CSV="${BUCKET_SOURCE_CSV:-research/work/ab_buckets_by_hit_from_latest.csv}"
WORK_DIR="${WORK_DIR:-research/work/ab_controlled_trace/${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-outputs/phase_ab/ab_controlled_trace/${RUN_TAG}}"
BUCKETS="${BUCKETS:-A0_B1,A1_B1,A0_B0,A1_B0}"
PER_BUCKET="${PER_BUCKET:-60}"

TRANSCODER_SET="${TRANSCODER_SET:-tianhux2/gemma3-4b-it-plt}"
DTYPE="${DTYPE:-bfloat16}"
MAX_FEATURE_NODES="${MAX_FEATURE_NODES:-96}"
MAX_N_LOGITS="${MAX_N_LOGITS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OFFLOAD="${OFFLOAD:-cpu}"
TOPK="${TOPK:-16}"

TOPK_PER_NODE="${TOPK_PER_NODE:-3}"
BEAM_PER_DEPTH="${BEAM_PER_DEPTH:-64}"
COVERAGE="${COVERAGE:-0.95}"
MAX_DEPTH="${MAX_DEPTH:-40}"
MIN_ABS_WEIGHT="${MIN_ABS_WEIGHT:-0.0}"
LOG_EVERY="${LOG_EVERY:-10}"
CLEAN_PT_AFTER="${CLEAN_PT_AFTER:-0}"

if [[ ! -f "${BUCKET_SOURCE_CSV}" ]]; then
  echo "[err] BUCKET_SOURCE_CSV not found: ${BUCKET_SOURCE_CSV}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}" "${OUT_ROOT}"

SELECTED_CSV="${WORK_DIR}/selected_4bucket.csv"
MANIFEST_A="${WORK_DIR}/manifest_A.csv"
MANIFEST_B="${WORK_DIR}/manifest_B.csv"
PT_DIR_A="${OUT_ROOT}/pt_a"
PT_DIR_B="${OUT_ROOT}/pt_b"
TRACE_OUT_DIR="${OUT_ROOT}/trace_compare"

echo "[stage] sample ${PER_BUCKET} per bucket from ${BUCKET_SOURCE_CSV}"
python - <<'PY' "${BUCKET_SOURCE_CSV}" "${SELECTED_CSV}" "${BUCKETS}" "${PER_BUCKET}"
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

echo "[stage] build A/B manifests"
python - <<'PY' "${SELECTED_CSV}" "${MANIFEST_A}" "${MANIFEST_B}"
import csv, sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
ma = Path(sys.argv[2]).expanduser().resolve()
mb = Path(sys.argv[3]).expanduser().resolve()
rows = list(csv.DictReader(open(src, "r", encoding="utf-8", newline="")))
if not rows:
    raise ValueError("empty selected csv")

fields = ["sample_id", "image_path", "question", "split", "trace_source", "question_type", "notes"]
ma.parent.mkdir(parents=True, exist_ok=True)

with open(ma, "w", encoding="utf-8", newline="") as fa, open(mb, "w", encoding="utf-8", newline="") as fb:
    wa, wb = csv.DictWriter(fa, fieldnames=fields), csv.DictWriter(fb, fieldnames=fields)
    wa.writeheader()
    wb.writeheader()
    for r in rows:
        sid = (r.get("sample_id") or "").strip()
        if not sid:
            continue
        bucket = (r.get("bucket") or "").strip()
        image = (r.get("image_path") or "").strip()
        qa = (r.get("a_input") or "").strip()
        qb = (r.get("b_input") or "").strip()
        notes = f"bucket={bucket}"
        wa.writerow(
            {
                "sample_id": sid,
                "image_path": image,
                "question": qa,
                "split": "val",
                "trace_source": "ab_controlled",
                "question_type": "okvqa",
                "notes": notes,
            }
        )
        wb.writerow(
            {
                "sample_id": sid,
                "image_path": image,
                "question": qb,
                "split": "val",
                "trace_source": "ab_controlled",
                "question_type": "okvqa",
                "notes": notes,
            }
        )
print(f"[ok] manifest A -> {ma}")
print(f"[ok] manifest B -> {mb}")
PY

echo "[stage] run attribution A"
python scripts/research/run_batch_attribute.py \
  --manifest "${MANIFEST_A}" \
  --output-dir "${PT_DIR_A}" \
  --transcoder-set "${TRANSCODER_SET}" \
  --dtype "${DTYPE}" \
  --max-feature-nodes "${MAX_FEATURE_NODES}" \
  --max-n-logits "${MAX_N_LOGITS}" \
  --batch-size "${BATCH_SIZE}" \
  --offload "${OFFLOAD}" \
  --topk "${TOPK}"

echo "[stage] run attribution B"
python scripts/research/run_batch_attribute.py \
  --manifest "${MANIFEST_B}" \
  --output-dir "${PT_DIR_B}" \
  --transcoder-set "${TRANSCODER_SET}" \
  --dtype "${DTYPE}" \
  --max-feature-nodes "${MAX_FEATURE_NODES}" \
  --max-n-logits "${MAX_N_LOGITS}" \
  --batch-size "${BATCH_SIZE}" \
  --offload "${OFFLOAD}" \
  --topk "${TOPK}"

echo "[stage] controlled trace compare"
RUN_TAG="${RUN_TAG}" \
PT_DIR_A="${PT_DIR_A}" \
PT_DIR_B="${PT_DIR_B}" \
BUCKET_CSV="${SELECTED_CSV}" \
OUT_ROOT="${OUT_ROOT}" \
BUCKETS="${BUCKETS}" \
PER_BUCKET="${PER_BUCKET}" \
TOPK_PER_NODE="${TOPK_PER_NODE}" \
BEAM_PER_DEPTH="${BEAM_PER_DEPTH}" \
COVERAGE="${COVERAGE}" \
MAX_DEPTH="${MAX_DEPTH}" \
MIN_ABS_WEIGHT="${MIN_ABS_WEIGHT}" \
LOG_EVERY="${LOG_EVERY}" \
bash scripts/server/run_ab_controlled_trace.sh

echo "[done] run_tag=${RUN_TAG}"
echo "[done] selected=${SELECTED_CSV}"
echo "[done] pt_a=${PT_DIR_A}"
echo "[done] pt_b=${PT_DIR_B}"
echo "[done] trace_out=${OUT_ROOT}/${RUN_TAG}"

if [[ "${CLEAN_PT_AFTER}" == "1" ]]; then
  echo "[cleanup] removing pt dirs to save disk..."
  rm -rf "${PT_DIR_A}" "${PT_DIR_B}"
  echo "[cleanup] done"
fi
