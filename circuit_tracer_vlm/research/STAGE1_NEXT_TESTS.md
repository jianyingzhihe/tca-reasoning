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

## 6. Refocused Roadmap

Based on the current results, the Stage 1 roadmap should now narrow from
"show that A/B graphs differ" to "explain what signed causal mechanism produces the
difference."

### Phase A: Clean Signed-Intervention Pass

Goal:

- estimate the balance of supportive vs suppressive traced features under clean
  same-target conditions

Required settings:

- same-target only
- generic-feature filtered
- meta-complete samples only
- moderate breadth rather than another full Stage 1 sweep

Recommended scale:

- `8-12` samples per bucket
- `4` features per sample per run
- `run=both`

Main outputs to track:

- fraction with `delta_target_logit < 0`
- strongest negative-`delta_target_logit` nodes
- per-bucket supportive vs suppressive mix

### Phase B: Required Controls

Before making stronger mechanistic claims, add these controls:

1. random matched-node control:
   - sample features matched on layer / position and rough scale, then compare their
     ablation effect against traced features
2. non-target token control:
   - test whether ablations are target-specific or just produce broad logit movement
3. prompt paraphrase control:
   - use several Prompt A and Prompt B paraphrases to separate prompt style from
     template-specific artifacts
4. prompt length / formatting control:
   - distinguish style effects from answer-format or prompt-length effects
5. image-side control:
   - no-image, mismatched-image, or degraded-image variants to test whether candidate
     support nodes depend on visual evidence

### Phase C: From Zeroing To Patching

Single-feature zeroing is a good smoke test, but it is not yet a full circuit claim.

The next causal upgrade should include:

- A->B activation patching
- B->A activation patching
- corrupt-and-restore style tests
- multi-node ablations
- small-circuit sufficiency tests

Goal:

- move from "this node matters" to "this small set of nodes can reproduce or explain
  the A/B behavior difference"

### Phase D: Feature Semantics

The current analysis is strong at the graph/composition level, but still weak at the
semantic level.

We should assign human-interpretable labels to strong support and suppressor features
using:

- top activating examples
- token-position analysis
- image occlusion / corruption responses
- question perturbations
- limited LLM-assisted labeling followed by manual checks

The target taxonomy should at least distinguish:

- visual evidence features
- text / question-type features
- answer-prior features
- prompt-format / instruction-following features
- suppressor / competitor features

## 7. Updated Main Question

At the beginning of Stage 1, the key question was:

- do Prompt A and Prompt B produce different answer-target circuits?

That is now mostly established.

The updated question is:

- which traced nodes are supportive vs suppressive,
- whether those signed node groups differ across buckets,
- and whether they correspond to visual evidence, answer priors, or prompt-format effects.

## 8. Multimodal Refocus

To avoid making this look like a text-only circuit workflow copied onto a VLM,
the next phase should explicitly center a multimodal question:

- does prompt style change **modality routing** inside the VLM?

That means moving from:

- prompt A/B changes attribution graphs

to:

- prompt A/B changes the causal allocation of answer generation across
  visual evidence, language prior, and cross-modal binding routes.

### Core Reframing

The main Stage 1 follow-up should now be framed as:

- **Prompt-Induced Modality Routing in Vision-Language Models**

Concretely:

- under the same image, question, and answer target,
- does Prompt A increase or decrease reliance on visual evidence nodes,
- language-prior nodes,
- or image-question binding nodes?

### Modality-Counterfactual Conditions

The next intervention stage should include more than clean-image ablations.

For a small, carefully selected set of samples, add:

1. clean image + original question
2. no-image / blank-image + original question
3. wrong-image + original question
4. object-masked image + original question

Optional later extension:

5. image-question mismatch or object-swapped question

These conditions should be used for both tracing and intervention analysis.

### Node Taxonomy We Need

We should no longer stop at feature/token/error categories.

Strong candidate nodes should be classified into:

- visual-evidence features
- text-question features
- answer-prior features
- cross-modal binding features
- prompt-format / instruction-following features
- suppressor / competitor features

### Multimodal-Specific Metrics

The current `delta_target_logit` metric remains useful, but it should be paired with
modality-sensitive metrics such as:

1. visual dependence score:
   - target-logit drop from clean image to no-image / wrong-image / masked-image
2. visual restoration effect:
   - effect of restoring clean visual nodes under a corrupted-image condition
3. language prior leakage:
   - target strength under no-image or mismatched-image settings
4. cross-modal binding sensitivity:
   - difference between matched and mismatched image-question conditions

### Bucket-Level Multimodal Questions

The bucket split is especially useful once the project is framed as modality routing:

- `A1_B1`:
  - both prompts succeed; do they rely on different visual/prior/binding routes?
- `A1_B0`:
  - does Prompt A succeed by strengthening visual evidence or by suppressing a wrong prior?
- `A0_B1`:
  - does Prompt B succeed by preserving a more direct visual route?
- `A0_B0`:
  - when both fail, do they collapse toward answer-prior or format-driven routes?

### Immediate Experimental Implication

The next small, focused study should ideally be:

- same-target only
- selected buckets: `A1_B1`, `A1_B0`, `A0_B1`
- modality counterfactual conditions
- signed interventions on top support and suppressor candidates

This is the cleanest way to make the project specifically about VLM mechanism,
not just prompt-conditioned graph overlap.
