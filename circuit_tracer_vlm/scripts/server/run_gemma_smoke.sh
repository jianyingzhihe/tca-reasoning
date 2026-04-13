#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run for Gemma3 + official transcoder set.
# Usage:
#   source scripts/server/load_env.sh .env
#   source .venv/bin/activate
#   bash scripts/server/run_gemma_smoke.sh [image_path] [output_pt]

IMAGE_PATH="${1:-demos/img/gemma/213.png}"
OUTPUT_PT="${2:-./outputs/gemma_demo_213.pt}"

mkdir -p "$(dirname "${OUTPUT_PT}")"

circuit-tracer attribute \
  --prompt "<start_of_image> What is in the image?" \
  --transcoder_set "tianhux2/gemma3-4b-it-plt" \
  --image "${IMAGE_PATH}" \
  --graph_output_path "${OUTPUT_PT}" \
  --batch_size 1 \
  --max_n_logits 2 \
  --max_feature_nodes 128 \
  --dtype float16 \
  --offload cpu

echo "[run_gemma_smoke] Saved graph to ${OUTPUT_PT}"

