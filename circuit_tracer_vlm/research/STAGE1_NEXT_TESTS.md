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
