# Stage 1 Next Tests

## 1. Target-Mode Comparison

After the gold-target run finishes, compare it to the predicted-target run:

```bash
python scripts/research/compare_stage1_target_modes.py \
  --predicted-summary-dir outputs/phase_ab/ab_answer_aligned/stage1_random20_1024_seed42501_20260425_235607_stage1_summary_v2 \
  --gold-summary-dir outputs/phase_ab/ab_answer_aligned/<gold_run_tag>_stage1_summary \
  --out-dir outputs/phase_ab/ab_answer_aligned/<gold_run_tag>_vs_predicted_comparison
```

Key outputs:

- `target_mode_metric_comparison.csv`
- `target_mode_same_target_comparison.csv`
- `target_mode_filtered_overlap_comparison.csv`
- `target_mode_comparison.md`

## 2. Intervention Smoke

Once a Stage 1 run has completed and compare CSVs exist, run lightweight feature-zeroing interventions:

```bash
RUN_TAG_BASE=stage1_random20_1024_seed42501_20260425_235607 \
GENERIC_NODES_CSV=outputs/phase_ab/ab_answer_aligned/stage1_random20_1024_seed42501_20260425_235607_stage1_summary_v2/stage1_generic_nodes.csv \
TOP_FEATURES_PER_SAMPLE=2 \
MAX_SAMPLES=4 \
bash scripts/server/run_stage1_intervention_smoke.sh
```

The wrapper now also writes an aggregated summary directory:

- `outputs/phase_ab/ab_answer_aligned/${RUN_TAG_BASE}_intervention_smoke_summary/`
- `intervention_smoke_summary.csv`
- `intervention_smoke_strongest_rows.csv`
- `intervention_smoke_summary.md`

What this does:

- ranks samples by low overlap by default;
- picks top answer-aligned feature nodes per sample/run;
- skips generic scaffold features if `GENERIC_NODES_CSV` is provided;
- zeroes those features one by one;
- measures target logit / target probability change.

This is a smoke test, not the final causal claim. It helps answer:

- are the traced feature nodes actually connected to answer-token logit changes;
- do some candidate nodes produce larger target-logit drops than others;
- which samples are worth promoting into a more expensive patching experiment.

If you already have per-bucket smoke CSVs and just want to rebuild the summary:

```bash
python scripts/research/summarize_intervention_smoke.py \
  --inputs \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A0_B0/intervention_smoke_A0_B0.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A0_B1/intervention_smoke_A0_B1.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A1_B0/intervention_smoke_A1_B0.csv \
    outputs/phase_ab/ab_answer_aligned/<run_tag>_A1_B1/intervention_smoke_A1_B1.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/<run_tag>_intervention_smoke_summary
```

## 3. Focused Intervention Planning

After the gold-target run, build a focused shortlist of same-target, low-overlap, feature-down /
token-up samples and export exact per-bucket sample lists for the smoke run:

```bash
python scripts/research/plan_stage1_interventions.py \
  --run-tag-base stage1_goldtarget20_1024_seed42501_<timestamp> \
  --summary-dir outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary \
  --generic-nodes-csv outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary/stage1_generic_nodes.csv \
  --out-dir outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_intervention_plan \
  --same-target-only \
  --max-samples-per-bucket 2 \
  --top-features-per-run 2
```

This writes:

- `candidate_samples.csv`
- `candidate_features.csv`
- `bucket_sample_ids/<bucket>.csv`
- `intervention_plan.md`

Then run the smoke test only on those selected samples:

```bash
RUN_TAG_BASE=stage1_goldtarget20_1024_seed42501_<timestamp> \
SAMPLE_IDS_DIR=outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_intervention_plan/bucket_sample_ids \
GENERIC_NODES_CSV=outputs/phase_ab/ab_answer_aligned/stage1_goldtarget20_1024_seed42501_<timestamp>_stage1_summary/stage1_generic_nodes.csv \
REQUIRE_SAME_TARGET=1 \
TOP_FEATURES_PER_SAMPLE=2 \
MAX_SAMPLES=999 \
bash scripts/server/run_stage1_intervention_smoke.sh
```

Notes:

- when `SAMPLE_IDS_DIR` is provided, `MAX_SAMPLES` should be set high enough that it does not
  silently truncate the per-bucket CSV lists;
- the planner now filters out samples whose `answer_aligned_meta_a.csv` or
  `answer_aligned_meta_b.csv` rows are missing `target_token_id`, so the resulting
  shortlist should not waste overnight slots on known-invalid samples.

## 4. Current Status (2026-05-05)

Completed stages so far:

1. predicted-target random20 summary:
   - stable feature-down / token-up composition shift
   - only `31/80` same-target cases
2. gold-target control summary:
   - same-target rate improved to `73/80`
   - overlap confound reduced substantially
   - feature-down / token-up shift survived control
3. focused + overnight intervention smoke:
   - pipeline runs end to end after multimodal batch-forwarding fix
   - `88` valid single-feature interventions completed in the overnight run
   - `64.8%` of interventions produced `delta_target_logit < 0`
   - overall mean `delta_target_logit` stayed slightly positive (`+0.073`)

Interpretation:

- traced feature nodes are not all equivalent;
- many high-path-mass feature nodes do causally support the answer token;
- a substantial minority appear suppressive or competitive, so path mass alone is not a signed
  causal score.

## 5. Recommended Next Step

The next best experiment is not another large Stage 1 attribution sweep.

The next best step is a cleaner same-target intervention pass using the repaired planner, followed
by prioritizing the strongest negative-`delta_target_logit` nodes for more targeted follow-up.

That should answer two narrower questions:

1. how often do top traced nodes act as positive supports versus suppressors;
2. whether the strongest support nodes cluster differently across `A0_B1` / `A1_B0` versus
   agreement buckets.
