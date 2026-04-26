#!/usr/bin/env bash
set -euo pipefail

# Stage 1 target-token control run.
#
# This wrapper runs the same random-bucket answer-aligned pipeline, but changes
# attribution target from each prompt's predicted answer token to the shared
# gold/canonical answer token stored in the eval CSV.
#
# Recommended run:
#   TMPDIR=/root/autodl-tmp/tmp \
#   bash scripts/server/run_stage1_gold_target_control_overnight.sh
#
# This is the next experiment after the predicted-target random20 run because it
# directly tests whether the A/B circuit differences remain when both prompts
# target the same answer token.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAGE1_RUN_TAG_BASE="${STAGE1_RUN_TAG_BASE:-stage1_goldtarget20_1024_seed42501_$(date +%Y%m%d_%H%M%S)}" \
STAGE1_PER_BUCKET="${STAGE1_PER_BUCKET:-20}" \
STAGE1_MAX_FEATURE_NODES="${STAGE1_MAX_FEATURE_NODES:-1024}" \
STAGE1_SELECTION_SEED="${STAGE1_SELECTION_SEED:-42501}" \
STAGE1_ANSWER_SOURCE="gold" \
STAGE1_DISK_BUDGET_GB="${STAGE1_DISK_BUDGET_GB:-260}" \
STAGE1_DISK_RESERVE_GB="${STAGE1_DISK_RESERVE_GB:-30}" \
STAGE1_CLEAN_PT_AFTER_SUMMARY="${STAGE1_CLEAN_PT_AFTER_SUMMARY:-0}" \
TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}" \
bash scripts/server/run_stage1_random_answer_aligned_overnight.sh
