#!/usr/bin/env bash
set -euo pipefail

# Robust single-prompt pool runner.
# - Splits one manifest into N shards
# - Runs N workers in parallel (one model process per worker)
# - Monitors progress by UNIQUE sample_id (never >100%)
# - Exits cleanly when workers finish
# - Validates completion and merges shard CSVs
#
# Example:
#   bash scripts/server/run_okvqa_single_prompt_pool_eval.sh \
#     --name promptA \
#     --base-manifest research/work/sample_manifest_okvqa_val_full.csv \
#     --prompt-template "{question} Think step by step from visual evidence, then reply exactly in the format: The answer is <short answer>." \
#     --workers 4 \
#     --max-new-tokens 200

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/run_okvqa_single_prompt_pool_eval.sh [options]

Required:
  --prompt-template TEMPLATE

Optional:
  --name NAME                       Prompt name tag (default: pool)
  --base-manifest PATH              Base manifest (default: research/work/sample_manifest_okvqa_val1000.csv)
  --run-tag TAG                     Run tag (default: pool_<name>_YYYYmmdd_HHMMSS)
  --out-root DIR                    Output root (default: outputs/phase_ab/eval_pool)
  --workers N                       Parallel workers (default: 4)
  --device DEVICE                   cuda/cpu (default: cuda)
  --model MODEL_ID                  e.g. google/gemma-3-4b-it
  --transcoder-set REPO_ID          default: tianhux2/gemma3-4b-it-plt
  --max-new-tokens N                default: 200
  --log-every N                     default: 1
  --correct-rule RULE               strict_gold|majority|vqa_0.3|vqa_0.6|vqa_1.0 (default: vqa_0.3)
  --okvqa-ann PATH                  default: ~/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json
  --force-fresh 0|1                 delete existing run dir first (default: 1)
  --okvqa-root DIR                  default: ~/tca-reasoning/data/okvqa
  --okvqa-limit N                   default: 1000; 0 means full split
  --okvqa-split SPLIT               val|train (default: val)
  --poll-seconds N                  progress refresh period (default: 5)
EOF
}

PROMPT_NAME="pool"
PROMPT_TEMPLATE=""
BASE_MANIFEST="research/work/sample_manifest_okvqa_val1000.csv"
OUT_ROOT="outputs/phase_ab/eval_pool"
WORKERS=4
DEVICE="cuda"
MODEL=""
TRANSCODER_SET="tianhux2/gemma3-4b-it-plt"
MAX_NEW_TOKENS=200
LOG_EVERY=1
CORRECT_RULE="vqa_0.3"
OKVQA_ANN="$HOME/tca-reasoning/data/okvqa/annotations/mscoco_val2014_annotations.json"
FORCE_FRESH=1
RUN_TAG=""
POLL_SECONDS=5

OKVQA_ROOT="$HOME/tca-reasoning/data/okvqa"
OKVQA_LIMIT=1000
OKVQA_SPLIT="val"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) PROMPT_NAME="$2"; shift 2 ;;
    --prompt-template) PROMPT_TEMPLATE="$2"; shift 2 ;;
    --base-manifest) BASE_MANIFEST="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --transcoder-set) TRANSCODER_SET="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --correct-rule) CORRECT_RULE="$2"; shift 2 ;;
    --okvqa-ann) OKVQA_ANN="$2"; shift 2 ;;
    --force-fresh) FORCE_FRESH="$2"; shift 2 ;;
    --okvqa-root) OKVQA_ROOT="$2"; shift 2 ;;
    --okvqa-limit) OKVQA_LIMIT="$2"; shift 2 ;;
    --okvqa-split) OKVQA_SPLIT="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[err] unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${PROMPT_TEMPLATE}" ]]; then
  echo "[err] --prompt-template is required" >&2
  usage
  exit 1
fi

# shellcheck source=/dev/null
source scripts/server/dev.sh "${ROOT_DIR}/.env" "${ROOT_DIR}/.venv"

BASE_MANIFEST="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${BASE_MANIFEST}")"
OKVQA_ANN="$(python -c 'import os,sys;print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OKVQA_ANN}")"
if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG="${PROMPT_NAME}_pool${WORKERS}_$(date +%Y%m%d_%H%M%S)"
fi

RUN_DIR="${OUT_ROOT}/${RUN_TAG}"
SPLIT_DIR="${RUN_DIR}/splits"
LOG_DIR="${RUN_DIR}/logs"

if [[ "${FORCE_FRESH}" == "1" ]]; then
  rm -rf "${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}" "${SPLIT_DIR}" "${LOG_DIR}" "$(dirname "${BASE_MANIFEST}")"

OKVQA_QUESTIONS="${OKVQA_ROOT}/questions/OpenEnded_mscoco_val2014_questions.json"
OKVQA_ANNOTATIONS="${OKVQA_ROOT}/annotations/mscoco_val2014_annotations.json"
OKVQA_IMAGE_ROOT="${OKVQA_ROOT}/images"

echo "[stage] ensure base manifest"
if [[ ! -f "${BASE_MANIFEST}" ]]; then
  python scripts/research/okvqa_to_manifest.py \
    --questions "${OKVQA_QUESTIONS}" \
    --annotations "${OKVQA_ANNOTATIONS}" \
    --image-root "${OKVQA_IMAGE_ROOT}" \
    --split "${OKVQA_SPLIT}" \
    --output "${BASE_MANIFEST}" \
    --limit "${OKVQA_LIMIT}" \
    --seed 42 \
    --id-prefix okvqa
fi

echo "[stage] build single-prompt manifest"
PROMPT_MANIFEST="${RUN_DIR}/manifest_${PROMPT_NAME}.csv"
python - <<'PY' "${BASE_MANIFEST}" "${PROMPT_MANIFEST}" "${PROMPT_TEMPLATE}" "${PROMPT_NAME}"
import csv
import sys
from pathlib import Path

in_path = Path(sys.argv[1]).expanduser().resolve()
out_path = Path(sys.argv[2]).expanduser().resolve()
template = sys.argv[3]
name = sys.argv[4]

def parse_gold(notes: str) -> str:
    if not notes:
        return ""
    for part in notes.split(";"):
        p = part.strip()
        if p.startswith("answer="):
            return p.split("=", 1)[1].strip()
    return ""

def fmt(template: str, question: str) -> str:
    q = (question or "").strip()
    vars = {
        "question": q,
        "instruction": q,
    }
    has_ph = any(("{" + k + "}") in template for k in vars)
    return template.format(**vars).strip() if has_ph else f"{template.strip()} {q}".strip()

rows = list(csv.DictReader(open(in_path, "r", encoding="utf-8", newline="")))
if not rows:
    raise ValueError(f"empty manifest: {in_path}")

base_fields = list(rows[0].keys())
extra_fields = ["orig_question", "prompt_variant", "pair_id", "prompt_template", "gold_answer"]
fieldnames = base_fields + [x for x in extra_fields if x not in base_fields]

out_rows = []
for r in rows:
    sid = (r.get("sample_id") or "").strip()
    q = (r.get("question") or "").strip()
    if not sid or not q:
        continue
    x = dict(r)
    x["orig_question"] = q
    x["question"] = fmt(template, q)
    x["prompt_variant"] = name
    x["pair_id"] = sid
    x["prompt_template"] = template
    x["gold_answer"] = (r.get("gold_answer") or "").strip() or parse_gold((r.get("notes") or "").strip())
    out_rows.append(x)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)
print(f"[ok] prompt manifest rows={len(out_rows)} -> {out_path}")
PY

echo "[stage] split manifest into ${WORKERS} parts"
python - <<'PY' "${PROMPT_MANIFEST}" "${SPLIT_DIR}" "${WORKERS}"
import csv
import sys
from pathlib import Path

in_path = Path(sys.argv[1]).expanduser().resolve()
out_dir = Path(sys.argv[2]).expanduser().resolve()
workers = int(sys.argv[3])
rows = list(csv.DictReader(open(in_path, "r", encoding="utf-8", newline="")))
if not rows:
    raise ValueError(f"empty prompt manifest: {in_path}")

fieldnames = list(rows[0].keys())
buckets = [[] for _ in range(workers)]
for i, r in enumerate(rows):
    buckets[i % workers].append(r)

out_dir.mkdir(parents=True, exist_ok=True)
for i, bucket in enumerate(buckets):
    p = out_dir / f"part_{i}.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(bucket)
    print(f"[ok] part_{i}: {len(bucket)} rows -> {p}")
PY

pool_status_line() {
  python - <<'PY' "${RUN_DIR}" "${WORKERS}"
import csv
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).expanduser().resolve()
workers = int(sys.argv[2])
total_done = 0
total_target = 0
parts = []
for i in range(workers):
    split_csv = run_dir / "splits" / f"part_{i}.csv"
    eval_csv = run_dir / f"part_{i}_eval.csv"
    target = 0
    if split_csv.exists():
        with split_csv.open("r", encoding="utf-8", newline="") as f:
            target = sum(1 for _ in csv.DictReader(f))
    uniq = set()
    raw_rows = 0
    if eval_csv.exists():
        with eval_csv.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                raw_rows += 1
                sid = (r.get("sample_id") or "").strip()
                if sid:
                    uniq.add(sid)
    done = min(len(uniq), target)
    dup = max(0, raw_rows - len(uniq))
    total_done += done
    total_target += target
    parts.append((done, target, dup))

tokens = [str(total_done), str(total_target)]
for done, target, dup in parts:
    tokens.extend([str(done), str(target), str(dup)])
print(" ".join(tokens))
PY
}

PIDS=()
cleanup_children() {
  for p in "${PIDS[@]:-}"; do
    kill "${p}" 2>/dev/null || true
  done
}
trap cleanup_children INT TERM

echo "[stage] launch ${WORKERS} workers"
for ((i=0; i<WORKERS; i++)); do
  PART_CSV="${SPLIT_DIR}/part_${i}.csv"
  OUT_CSV="${RUN_DIR}/part_${i}_eval.csv"
  OUT_LOG="${LOG_DIR}/part_${i}.log"
  rm -f "${OUT_CSV}" "${OUT_LOG}"
  CMD=(
    python -u scripts/research/run_batch_eval.py
    --manifest "${PART_CSV}"
    --output-csv "${OUT_CSV}"
    --correct-rule "${CORRECT_RULE}"
    --device "${DEVICE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --log-every "${LOG_EVERY}"
    --no-resume
  )
  if [[ -f "${OKVQA_ANN}" ]]; then
    CMD+=(--annotations-json "${OKVQA_ANN}")
  fi
  if [[ -n "${MODEL}" ]]; then
    CMD+=(--model "${MODEL}")
  else
    CMD+=(--transcoder-set "${TRANSCODER_SET}")
  fi

  (
    "${CMD[@]}" 2>&1 | sed -u "s/^/[W${i}] /"
  ) | tee "${OUT_LOG}" &
  PIDS+=("$!")
done

echo "[stage] monitor pool progress"
declare -a ANNOUNCED_DONE
for ((i=0; i<WORKERS; i++)); do ANNOUNCED_DONE[$i]=0; done

while true; do
  read -r -a TOK < <(pool_status_line)
  TOTAL_DONE="${TOK[0]}"
  TOTAL_TARGET="${TOK[1]}"
  if [[ "${TOTAL_TARGET}" -le 0 ]]; then TOTAL_TARGET=1; fi
  PCT=$((100 * TOTAL_DONE / TOTAL_TARGET))

  MSG="[POOL] ${TOTAL_DONE}/${TOTAL_TARGET} (${PCT}.0%) |"
  idx=2
  for ((i=0; i<WORKERS; i++)); do
    d="${TOK[$idx]}"; idx=$((idx+1))
    t="${TOK[$idx]}"; idx=$((idx+1))
    dup="${TOK[$idx]}"; idx=$((idx+1))
    MSG="${MSG} W${i}:${d}/${t}"
    if [[ "${dup}" -gt 0 ]]; then
      MSG="${MSG} dup=${dup}"
    fi
  done
  echo "${MSG}"

  for ((i=0; i<WORKERS; i++)); do
    part_done_idx=$((2 + i*3))
    part_target_idx=$((3 + i*3))
    d="${TOK[$part_done_idx]}"
    t="${TOK[$part_target_idx]}"
    if [[ "${t}" -gt 0 && "${d}" -ge "${t}" && "${ANNOUNCED_DONE[$i]}" -eq 0 ]]; then
      echo "[done][W${i}] ${d}/${t}"
      ANNOUNCED_DONE[$i]=1
    fi
  done

  alive=0
  for p in "${PIDS[@]}"; do
    if kill -0 "${p}" 2>/dev/null; then
      alive=$((alive+1))
    fi
  done
  if [[ "${alive}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "${p}"; then
    FAIL=1
  fi
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[err] one or more workers failed; see ${LOG_DIR}/part_*.log" >&2
  exit 2
fi

echo "[stage] final completeness check"
read -r -a TOK < <(pool_status_line)
TOTAL_DONE="${TOK[0]}"
TOTAL_TARGET="${TOK[1]}"
if [[ "${TOTAL_DONE}" -ne "${TOTAL_TARGET}" ]]; then
  echo "[err] incomplete run: ${TOTAL_DONE}/${TOTAL_TARGET}" >&2
  idx=2
  for ((i=0; i<WORKERS; i++)); do
    d="${TOK[$idx]}"; idx=$((idx+1))
    t="${TOK[$idx]}"; idx=$((idx+1))
    dup="${TOK[$idx]}"; idx=$((idx+1))
    echo "[err] W${i}: ${d}/${t} dup=${dup}" >&2
  done
  exit 3
fi

echo "[stage] merge part csv files"
MERGED_CSV="${RUN_DIR}/${PROMPT_NAME}_eval.csv"
python - <<'PY' "${RUN_DIR}" "${WORKERS}" "${MERGED_CSV}"
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).expanduser().resolve()
workers = int(sys.argv[2])
out_csv = Path(sys.argv[3]).expanduser().resolve()

all_rows = []
fieldnames = None
for i in range(workers):
    p = run_dir / f"part_{i}_eval.csv"
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = list(r.fieldnames or [])
        for row in r:
            all_rows.append(row)

if fieldnames is None:
    raise ValueError("no part eval csv found")

seen = set()
merged = []
for row in all_rows:
    sid = (row.get("sample_id") or "").strip()
    key = sid if sid else f"__row_{len(merged)}"
    if key in seen:
        continue
    seen.add(key)
    merged.append(row)

def sort_key(r):
    sid = (r.get("sample_id") or "").strip()
    return sid

merged.sort(key=sort_key)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(merged)

ok = sum(1 for r in merged if (r.get("correct") or "") == "1")
err = sum(1 for r in merged if (r.get("error_message") or "").strip())
print(f"[ok] merged rows={len(merged)} correct={ok} acc={ok/max(len(merged),1):.4f} errors={err}")
print(f"[ok] merged csv -> {out_csv}")
PY

echo "[done] run_dir=${RUN_DIR}"
echo "[done] merged_csv=${MERGED_CSV}"
echo "[done] logs_dir=${LOG_DIR}"
