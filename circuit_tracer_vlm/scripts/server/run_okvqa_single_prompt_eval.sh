#!/usr/bin/env bash
set -euo pipefail

# Single-prompt, single-run, foreground eval.
# No parallelism. Real-time progress is printed by run_batch_eval.py.
#
# Example:
#   bash scripts/server/run_okvqa_single_prompt_eval.sh \
#     --name A \
#     --base-manifest research/work/sample_manifest_okvqa_val_full.csv \
#     --prompt-template "{question} Reply exactly in the format: The answer is <short answer>."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/run_okvqa_single_prompt_eval.sh [options]

Required:
  --prompt-template TEMPLATE

Optional:
  --name NAME                         Prompt name tag (default: single)
  --base-manifest PATH                Base manifest (default: research/work/sample_manifest_okvqa_val1000.csv)
  --run-tag TAG                       Run tag (default: single_YYYYmmdd_HHMMSS)
  --out-root DIR                      Output root (default: outputs/phase_ab/eval_single)
  --device DEVICE                     cuda/cpu (default: cuda)
  --model MODEL_ID                    e.g. google/gemma-3-4b-it
  --transcoder-set REPO_ID            default: tianhux2/gemma3-4b-it-plt
  --max-new-tokens N                  default: 16
  --log-every N                       default: 20
  --correct-rule RULE                 strict_gold|majority|vqa_0.3|vqa_0.6|vqa_1.0 (default: vqa_0.3)
  --force-fresh 0|1                   delete old csv/log before run (default: 1)
  --example-instruction TEXT          default: What color is the bus?
  --example-response TEXT             default: The answer is yellow.
  --okvqa-root DIR                    default: ~/tca-reasoning/data/okvqa
  --okvqa-limit N                     default: 1000; 0 means full split
  --okvqa-split SPLIT                 val|train (default: val)
EOF
}

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

PROMPT_NAME="single"
PROMPT_TEMPLATE=""
BASE_MANIFEST="research/work/sample_manifest_okvqa_val1000.csv"
RUN_TAG="single_$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/phase_ab/eval_single"
DEVICE="cuda"
MODEL=""
TRANSCODER_SET="tianhux2/gemma3-4b-it-plt"
MAX_NEW_TOKENS="16"
LOG_EVERY="20"
CORRECT_RULE="vqa_0.3"
FORCE_FRESH="1"
EXAMPLE_INSTRUCTION="What color is the bus?"
EXAMPLE_RESPONSE="The answer is yellow."
OKVQA_ROOT="$HOME/tca-reasoning/data/okvqa"
OKVQA_LIMIT="1000"
OKVQA_SPLIT="val"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) PROMPT_NAME="$2"; shift 2 ;;
    --prompt-template) PROMPT_TEMPLATE="$2"; shift 2 ;;
    --base-manifest) BASE_MANIFEST="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --transcoder-set) TRANSCODER_SET="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --log-every) LOG_EVERY="$2"; shift 2 ;;
    --correct-rule) CORRECT_RULE="$2"; shift 2 ;;
    --force-fresh) FORCE_FRESH="$2"; shift 2 ;;
    --example-instruction) EXAMPLE_INSTRUCTION="$2"; shift 2 ;;
    --example-response) EXAMPLE_RESPONSE="$2"; shift 2 ;;
    --okvqa-root) OKVQA_ROOT="$2"; shift 2 ;;
    --okvqa-limit) OKVQA_LIMIT="$2"; shift 2 ;;
    --okvqa-split) OKVQA_SPLIT="$2"; shift 2 ;;
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
WORK_DIR="research/work/single_${RUN_TAG}"
EVAL_DIR="${OUT_ROOT}/eval/${RUN_TAG}"
LOG_DIR="${OUT_ROOT}/logs/${RUN_TAG}"
mkdir -p "${WORK_DIR}" "${EVAL_DIR}" "${LOG_DIR}" "$(dirname "${BASE_MANIFEST}")"

OKVQA_QUESTIONS="$OKVQA_ROOT/questions/OpenEnded_mscoco_val2014_questions.json"
OKVQA_ANNOTATIONS="$OKVQA_ROOT/annotations/mscoco_val2014_annotations.json"
OKVQA_IMAGE_ROOT="$OKVQA_ROOT/images"
OKVQA_ANN="$OKVQA_ANNOTATIONS"

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
    --seed 42 \
    --id-prefix okvqa
fi
echo "[ok] base manifest rows=$(count_csv_rows "${BASE_MANIFEST}")"

OUT_MANIFEST="${WORK_DIR}/manifest_${PROMPT_NAME}.csv"
OUT_CSV="${EVAL_DIR}/${PROMPT_NAME}_eval.csv"
OUT_LOG="${LOG_DIR}/${PROMPT_NAME}_eval.log"

echo "[stage] build single manifest for prompt=${PROMPT_NAME}"
python - <<'PY' "${BASE_MANIFEST}" "${OUT_MANIFEST}" "${PROMPT_TEMPLATE}" "${PROMPT_NAME}" "${EXAMPLE_INSTRUCTION}" "${EXAMPLE_RESPONSE}"
import csv
import sys
from pathlib import Path

in_path = Path(sys.argv[1]).expanduser().resolve()
out_path = Path(sys.argv[2]).expanduser().resolve()
template = sys.argv[3]
name = sys.argv[4]
example_instruction = sys.argv[5]
example_response = sys.argv[6]

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
        "example_instruction": example_instruction,
        "example_response": example_response,
        "original_response": example_response,
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
    x["gold_answer"] = parse_gold((r.get("notes") or "").strip())
    out_rows.append(x)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(out_rows)

print(f"[ok] wrote single manifest rows={len(out_rows)} -> {out_path}")
PY

echo "[stage] run eval prompt=${PROMPT_NAME} rows=$(count_csv_rows "${OUT_MANIFEST}")"
if [[ "${FORCE_FRESH}" == "1" ]]; then
  rm -f "${OUT_CSV}" "${OUT_LOG}"
fi

CMD=(
  python scripts/research/run_batch_eval.py
  --manifest "${OUT_MANIFEST}"
  --output-csv "${OUT_CSV}"
  --correct-rule "${CORRECT_RULE}"
  --device "${DEVICE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --log-every "${LOG_EVERY}"
)
if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
else
  CMD+=(--transcoder-set "${TRANSCODER_SET}")
fi
if [[ -f "${OKVQA_ANN}" ]]; then
  CMD+=(--annotations-json "${OKVQA_ANN}")
fi

"${CMD[@]}" 2>&1 | tee "${OUT_LOG}"

echo "[stage] summary"
python - <<'PY' "${OUT_CSV}"
import csv, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p, encoding="utf-8")))
n = len(rows)
ok = sum(1 for r in rows if (r.get("correct") or "") == "1")
err = sum(1 for r in rows if (r.get("error_message") or "").strip())
print(f"rows={n} correct={ok} acc={ok/max(n,1):.4f} errors={err}")
PY

echo "[done] manifest=${OUT_MANIFEST}"
echo "[done] eval_csv=${OUT_CSV}"
echo "[done] log=${OUT_LOG}"

